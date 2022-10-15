use std::error::Error;

use clap::{command, Parser};
use itertools::Itertools;
use markov_music::neural2::{
    assert_shape, reshape, NeuralNet, BATCH_SIZE, FRAME_SIZE, NUM_FRAMES, QUANTIZATION,
};
use rand::Rng;
use serde::{Deserialize, Serialize};
use tch::{nn, Device, IndexOp, Tensor};

mod util;

macro_rules! print_tensor {
    ($var:ident) => {
        let header = format!("=== {}: (shape = {:?}) ===", stringify!($var), $var.size());
        println!("{}", header);
        $var.print();
        println!("{:=<1$}", "", header.len());
    };
}

#[derive(Parser, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[command(author, version, about, long_about = None)]
/// A WAV file generator powered by markov chain.
struct Args {
    /// Path to input MP3 file.
    #[arg(short, long = "in")]
    in_path: String,
    /// Name of output WAV file. (will always have suffix of "i.wav", where i is the ith epoch)
    #[arg(short, long = "out", default_value = "epoch")]
    out_path: String,
    /// Length, in seconds, of audio to generate.
    #[arg(long, default_value_t = 60)]
    length: usize,
    #[arg(long, default_value_t = 10)]
    generate_every: usize,
    /// If passed, allows settings to be updated each epoch via a text file
    #[arg(long)]
    settings_file: Option<String>,
}

impl Args {
    fn try_from_file(path: &str) -> Result<Args, Box<dyn Error>> {
        let mut args: Args = serde_json::from_slice(&std::fs::read(path)?)?;
        args.settings_file = Some(path.to_string());
        Ok(args)
    }
}

struct Audio {
    audio: Tensor,
    // audio_onehot: Tensor,
    min: f32,
    max: f32,
    rounded_min: i64,
    len: usize,
    sample_rate: usize,
}

impl Audio {
    fn new(input: &[i16], sample_rate: usize) -> Audio {
        let len = input.len();
        let max = input.iter().cloned().max().unwrap() as f32;
        let min = input.iter().cloned().min().unwrap() as f32;
        let quantization = (QUANTIZATION - 1) as f32;
        let scale = (max - min) / quantization;
        let audio = input
            .iter()
            .cloned()
            .map(|x| {
                let sample = x as f32 / scale;
                sample.round() as i64
            })
            .collect_vec();

        let rounded_min = *audio.iter().min().unwrap();
        let audio = audio.iter().cloned().map(|x| x - rounded_min).collect_vec();
        assert!(audio.iter().all(|x| 0 <= *x && *x < QUANTIZATION as i64));

        let audio = Tensor::of_slice(&audio);
        // let audio_onehot = audio.onehot(QUANTIZATION as i64);
        Audio {
            audio,
            // audio_onehot,
            min,
            max,
            rounded_min,
            len,
            sample_rate,
        }
    }

    fn unnormalize(&self, audio: &[i64]) -> Vec<f32> {
        let quantization = (QUANTIZATION - 1) as f32;
        let scale = (self.max - self.min) / quantization;
        audio
            .iter()
            .cloned()
            .map(|x| {
                let sample = (x + self.rounded_min) as f32;
                sample * scale
            })
            .collect_vec()
    }

    fn batch(&self, batch_size: usize, num_frames: usize, frame_size: usize) -> (Tensor, Tensor) {
        let seq_len = num_frames * frame_size;
        let (input, targets): (Vec<_>, Vec<_>) = (0..batch_size)
            .map(|_| {
                let i = rand::thread_rng().gen_range(0..(self.len - seq_len - 1)) as i64;
                let input = self.audio.i(i..i + seq_len as i64);
                let targets = self.audio.i((i + 1)..(i + 1) + seq_len as i64);

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

        (input, targets)
    }

    fn debug_batch(batch_size: usize, num_frames: usize, frame_size: usize) -> (Tensor, Tensor) {
        let total_elements = batch_size * num_frames * frame_size;
        let input = (0..)
            .take(total_elements)
            .map(|i| (i % QUANTIZATION) as i64)
            .collect::<Vec<_>>();
        let targets = (0..)
            .take(total_elements)
            .map(|i| ((i + 1) % QUANTIZATION) as i64)
            .collect::<Vec<_>>();

        let input = Tensor::of_slice(&input);
        let targets = Tensor::of_slice(&targets);

        let input = reshape(&[batch_size, num_frames, frame_size], &input);
        let targets = reshape(&[batch_size, num_frames, frame_size], &targets);

        (input, targets)
    }

    fn write_to_file(&self, name: &str, audio: &[i64]) {
        let samples = self.unnormalize(audio);
        let samples = samples.iter().map(|x| *x as i16).collect_vec();
        let samples = samples.iter().map(|x| *x as i16).collect_vec();

        let mut partials = vec![vec![]; FRAME_SIZE];

        for chunk in &samples.clone().into_iter().chunks(FRAME_SIZE) {
            for (i, sample) in chunk.enumerate() {
                partials[i].push(sample);
            }
        }

        let samples_alt = partials.into_iter().flatten().collect_vec();

        util::write_wav(name, self.sample_rate, &samples, None).unwrap();
        util::write_wav(
            format!("alt_{}", name),
            self.sample_rate,
            &samples_alt,
            None,
        )
        .unwrap();
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    std::env::set_var("RUST_BACKTRACE", "1");

    let mut args = Args::parse();

    let (signal, _, sample_rate) = util::read_file(&args.in_path)?;
    let signal = Audio::new(&signal, sample_rate);

    println!("Input samples: {}", signal.len);

    let length = sample_rate * args.length;

    let device = Device::cuda_if_available();
    println!("Training neural net on {:?}", device);
    let vs = nn::VarStore::new(device);

    let mut network = NeuralNet::new(&vs, device);
    for epoch_i in 0..201 {
        // let (frames, targets) = signal.batch(BATCH_SIZE, NUM_FRAMES, FRAME_SIZE);
        let (frames, targets) = Audio::debug_batch(BATCH_SIZE, NUM_FRAMES, FRAME_SIZE);

        let loss = network.backward(&frames, &targets);

        if epoch_i != 0 && epoch_i % args.generate_every == 0 {
            println!("Epoch {}, loss = {:.5}", epoch_i, loss);
            generate(&args.out_path, epoch_i, length, &network, &signal);
        }

        if let Some(path) = &args.settings_file {
            match Args::try_from_file(path) {
                Ok(new_args) => {
                    if new_args != args {
                        println!("Updated args!");
                        args = new_args;
                    }
                }
                Err(err) => println!("Couldn't update args: {:?}", err),
            }
        }
    }
    Ok(())
}

fn generate(name: &str, epoch_i: usize, length: usize, network: &NeuralNet, signal: &Audio) {
    let (mut frame, mut state) = network.zeros(1);
    frame = Audio::debug_batch(1, 1, FRAME_SIZE).0;
    let mut samples = Vec::with_capacity(length);
    println!("Generating {} samples...", length);
    while samples.len() < length {
        let (next_samples, next_state) = network.forward(&frame, &state, samples.len() == 0);
        state = next_state;
        samples.extend(Vec::<i64>::from(&next_samples));
        frame = reshape(&[1, 1, FRAME_SIZE], &next_samples);

        if samples.len() % (length / 10).max(10_000) == 0 {
            println!("Generating... ({:?} / {})", samples.len(), length)
        }
    }

    signal.write_to_file(&format!("{}{}.wav", name, epoch_i), &samples);
}
