use std::{error::Error, time::Instant};

use clap::{command, Parser};
use itertools::Itertools;
use markov_music::neural2::{assert_shape, reshape, write_csv, Frames, NetworkParams, NeuralNet};
use rand::Rng;
use serde::{Deserialize, Serialize};
use tch::{nn, Device, IndexOp, Tensor};

mod util;

const ARG_FILE: &str = "args.json";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
/// A WAV file generator powered by markov chain.
struct Args {
    #[clap(flatten)]
    other_args: OtherArgs,
    #[clap(flatten)]
    network_params: NetworkParams,
}

#[derive(Parser, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct OtherArgs {
    /// Path to input file.
    #[arg(short, long = "in")]
    in_path: String,
    /// Name of output WAV files
    #[arg(short, long = "out", default_value = "epoch")]
    out_path: String,
    /// Length, in seconds, of audio to generate.
    #[arg(long, default_value_t = 60)]
    length: usize,
    /// How often to generate audio
    #[arg(long, default_value_t = 10)]
    generate_every: usize,
    /// How often to checkpoint to a file
    #[arg(long, default_value_t = 100)]
    checkpoint_every: usize,
    /// Enables debug mode
    #[arg(long, default_value_t = 0)]
    debug: usize,
}

impl OtherArgs {
    fn try_from_file(path: &str) -> Result<OtherArgs, Box<dyn Error>> {
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }

    fn write_to_file(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let json = serde_json::ser::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

struct Audio {
    audio: Tensor,
    min: f32,
    max: f32,
    rounded_min: i64,
    len: usize,
    sample_rate: usize,
    quantization: usize,
}

impl Audio {
    fn new(input: &[i16], sample_rate: usize, quantization: usize) -> Audio {
        let len = input.len();
        let max = input.iter().cloned().max().unwrap() as f32;
        let min = input.iter().cloned().min().unwrap() as f32;
        let scale = (max - min) / ((quantization - 1) as f32);
        let audio = input
            .iter()
            .cloned()
            .map(|x| {
                let sample = x as f32 / scale;
                sample.floor() as i64
            })
            .collect_vec();

        let rounded_min = *audio.iter().min().unwrap();
        let audio = audio.iter().cloned().map(|x| x - rounded_min).collect_vec();
        assert!(audio.iter().all(|x| 0 <= *x && *x < quantization as i64));

        let audio = Tensor::of_slice(&audio);
        Audio {
            audio,
            min,
            max,
            rounded_min,
            len,
            sample_rate,
            quantization,
        }
    }

    fn unnormalize(&self, audio: &[i64]) -> Vec<f32> {
        let scale = (self.max - self.min) / ((self.quantization - 1) as f32);
        audio
            .iter()
            .cloned()
            .map(|x| {
                let sample = (x + self.rounded_min) as f32;
                sample * scale
            })
            .collect_vec()
    }

    fn batch(&self, batch_size: usize, num_frames: usize, frame_size: usize) -> (Frames, Frames) {
        let seq_len = num_frames * frame_size;
        let (input, targets): (Vec<_>, Vec<_>) = (0..batch_size)
            .map(|_| {
                let i = rand::thread_rng().gen_range(0..(self.len - seq_len - frame_size)) as i64;
                let input = self.audio.i(i..i + seq_len as i64);
                let targets = self
                    .audio
                    .i((i + frame_size as i64)..(i + frame_size as i64) + seq_len as i64);

                assert_shape(&[seq_len], &input);
                assert_shape(&[seq_len], &targets);
                (input, targets)
            })
            .unzip();

        let input = Tensor::stack(&input, 0);
        let targets = Tensor::stack(&targets, 0);

        assert_shape(&[batch_size, seq_len], &input);
        assert_shape(&[batch_size, seq_len], &targets);

        let input = reshape(&[batch_size, num_frames, frame_size], &input);
        let targets = reshape(&[batch_size, num_frames, frame_size], &targets);

        assert_shape(&[batch_size, num_frames, frame_size], &input);
        assert_shape(&[batch_size, num_frames, frame_size], &targets);

        (
            Frames::new(input, batch_size, num_frames, frame_size),
            Frames::new(targets, batch_size, num_frames, frame_size),
        )
    }

    fn write_to_file(&self, name: &str, audio: &[i64]) {
        let samples = self.unnormalize(audio);
        let samples = samples.iter().map(|x| *x as i16).collect_vec();

        util::write_wav(name, self.sample_rate, &samples, None).unwrap();
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let (mut args, params) = (args.other_args, args.network_params);
    args.write_to_file(ARG_FILE)?;

    match args.debug {
        1 => std::env::set_var("RUST_BACKTRACE", "1"),
        2 => std::env::set_var("RUST_BACKTRACE", "full"),
        _ => (),
    }

    let (signal, _, sample_rate) = util::read_file(&args.in_path)?;
    let signal = Audio::new(&signal, sample_rate, params.quantization);

    println!("Input samples: {}", signal.len);

    let device = Device::cuda_if_available();
    println!("Training neural net on {:?}", device);
    println!("== Parameters ==\n{:#?}", params);

    let vs = nn::VarStore::new(device);

    let mut network = NeuralNet::new(&vs, params);
    let mut losses = vec![];

    if args.debug != 0 {
        signal.write_to_file("ground_truth.wav", &Vec::<i64>::from(&signal.audio));
    }

    for epoch_i in 0.. {
        let now = Instant::now();
        let (frames, targets) =
            signal.batch(params.batch_size, params.num_frames, params.frame_size);

        let loss = network.backward(&frames, &targets, args.debug != 0);

        println!(
            "Epoch {}, loss = {:.8} (time = {:?})",
            epoch_i,
            loss,
            now.elapsed(),
        );
        losses.push(loss);
        if args.generate_every != 0 && epoch_i != 0 && epoch_i % args.generate_every == 0 {
            write_csv(&format!("{}losses.csv", args.out_path), &[losses.clone()]);
            generate(
                &args.out_path,
                epoch_i,
                sample_rate * args.length,
                &network,
                &signal,
            );
        }

        match OtherArgs::try_from_file(ARG_FILE) {
            Ok(new_args) => {
                if new_args != args {
                    println!("Updated args!");
                    args = new_args;
                }
            }
            Err(err) => println!("Couldn't update args: {:?}", err),
        }
    }
    Ok(())
}

fn generate(name: &str, epoch_i: usize, length: usize, network: &NeuralNet, signal: &Audio) {
    let now = Instant::now();
    let mut state = network.zeros(1);
    let mut frame = signal.batch(1, 1, network.params.frame_size).0.samples();
    let mut samples = Vec::with_capacity(length);
    samples.extend(frame.iter());
    println!("Generating {} samples...", length);

    let mut i = 0;
    while samples.len() < length {
        let (next_frame, next_state) = network.forward(frame, &state, false);
        state = next_state;
        samples.extend(next_frame.iter());
        frame = next_frame;

        if i % 500 == 0 {
            println!(
                "Generating... ({:?} / {} samples ({:.2}%), epoch {})",
                samples.len(),
                length,
                100.0 * (samples.len() as f32 / length as f32),
                epoch_i
            )
        }
        i += 1;
    }

    signal.write_to_file(&format!("{}{}.wav", name, epoch_i), &samples);
    println!("Generated! time = {:?}", now.elapsed());
}
