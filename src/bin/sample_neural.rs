use std::{error::Error, time::Instant};

use clap::{command, Parser};
use itertools::Itertools;
use markov_music::neural2::{assert_shape, reshape, write_csv, Frames, NetworkParams, NeuralNet};
use rand::Rng;
use serde::{Deserialize, Serialize};
use tch::{nn, Device, IndexOp, Tensor};

mod util;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
/// A WAV file generator powered by markov chain.
struct Args {
    /// Path to input files.
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
    /// The learn rate of the network.
    #[arg(long, default_value_t = 0.001)]
    learn_rate: f64,
    /// The batch size for the network.
    #[arg(long, default_value_t = 128)]
    batch_size: usize,
    /// The size of each frame, in samples.
    #[arg(long, default_value_t = 16)]
    frame_size: usize,
    /// The number of frames to use during training.
    #[arg(long, default_value_t = 64)]
    num_frames: usize,
    /// The size of the hidden layers.
    #[arg(long, default_value_t = 1024)]
    hidden_size: usize,
    /// The number of RNN layers to use.
    #[arg(long, default_value_t = 5)]
    rnn_layers: usize,
    /// The size of the embedding
    #[arg(long, default_value_t = 256)]
    embed_size: usize,
    /// The number of quantization levels to use.
    #[arg(long, default_value_t = 256)]
    quantization: usize,
    /// The file to read parameters from
    #[arg(long, default_value_t = String::from("args.json"))]
    args_file: String,
}

impl Args {
    fn network_params(&self) -> NetworkParams {
        NetworkParams {
            learn_rate: self.learn_rate,
            batch_size: self.batch_size,
            frame_size: self.frame_size,
            num_frames: self.num_frames,
            hidden_size: self.hidden_size,
            rnn_layers: self.rnn_layers,
            embed_size: self.embed_size,
            quantization: self.quantization,
        }
    }
    fn get_updatable(&self) -> UpdatableArgs {
        UpdatableArgs {
            out_path: self.out_path.clone(),
            length: self.length,
            generate_every: self.generate_every,
            checkpoint_every: self.checkpoint_every,
            debug: self.debug,
            learn_rate: EqF64(self.learn_rate),
        }
    }
    fn set_updatable(&mut self, updatable: &UpdatableArgs) {
        self.out_path = updatable.out_path.clone();
        self.length = updatable.length;
        self.generate_every = updatable.generate_every;
        self.checkpoint_every = updatable.checkpoint_every;
        self.debug = updatable.debug;
        self.learn_rate = updatable.learn_rate.0;
    }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
struct EqF64(f64);
impl PartialEq for EqF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0).is_eq()
    }
}

impl Eq for EqF64 {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct UpdatableArgs {
    out_path: String,
    length: usize,
    generate_every: usize,
    checkpoint_every: usize,
    debug: usize,
    learn_rate: EqF64,
}

impl UpdatableArgs {
    fn try_from_file(path: &str) -> Result<UpdatableArgs, Box<dyn Error>> {
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
    let mut args = Args::parse();
    let params = args.network_params();
    args.get_updatable().write_to_file(&args.args_file)?;

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
    println!("== Arguments ==\n{:#?}", args);
    println!("===============");

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

        match UpdatableArgs::try_from_file(&args.args_file) {
            Ok(new_args) => {
                if new_args != args.get_updatable() {
                    println!("Updated args!\n{:#?}", new_args);
                    args.set_updatable(&new_args);
                    network.set_learn_rate(new_args.learn_rate.0);
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
