use std::error::Error;

use clap::{command, Parser};
use itertools::Itertools;
use markov_music::neural2::{NeuralNet, SEQ_LEN};
use rand::Rng;
use tch::{nn, Device};

mod util;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
/// A WAV file generator powered by markov chain.
struct Args {
    /// Path to input MP3 file.
    #[arg(short, long = "in")]
    in_path: String,
    /// Path to output WAV file.
    #[arg(short, long = "out", default_value = "out.wav")]
    out_path: String,
    #[arg(long, default_value_t = 10)]
    epochs: usize,
    #[arg(long, default_value_t = 1000)]
    batch_size: usize,
    /// Length, in seconds, of audio to generate.
    #[arg(long, default_value_t = 60)]
    length: usize,
}

struct NormalizedAudio {
    audio: Vec<f32>,
    min: f32,
    max: f32,
}

impl NormalizedAudio {
    fn new(input: &[i16]) -> NormalizedAudio {
        let max = input.iter().cloned().max().unwrap() as f32;
        let min = input.iter().cloned().min().unwrap() as f32;
        let audio = input
            .iter()
            .map(|x| (*x as f32 - min) / (max - min))
            .collect_vec();
        NormalizedAudio { audio, min, max }
    }

    fn len(&self) -> usize {
        self.audio.len()
    }

    fn unnormalize(&self, audio: &[f32]) -> Vec<f32> {
        audio
            .iter()
            .map(|x| x * (self.max - self.min) + self.min)
            .collect_vec()
    }

    fn random_example(&self) -> [f32; SEQ_LEN] {
        let i = rand::thread_rng().gen_range(0..(self.audio.len() - (SEQ_LEN + 1)));
        into_window(&self.audio[i..i + SEQ_LEN])
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    std::env::set_var("RUST_BACKTRACE", "1");

    let args = Args::parse();

    let (signal, _, sample_rate) = util::read_file(&args.in_path)?;
    let signal = NormalizedAudio::new(&signal);
    println!("Input samples: {}", signal.len());

    let length = sample_rate * args.length;

    let device = Device::cuda_if_available();
    println!("Training neural net on {:?}", device);
    let vs = nn::VarStore::new(device);

    let mut network = NeuralNet::new(&vs, device);
    for epoch_i in 0..args.epochs {
        let mut batch_loss = 0.0;
        for _batch_i in 0..args.batch_size {
            let input = signal.random_example();
            let loss = network.train(input);
            batch_loss += loss;
            // println!("Batch {}/{}", _batch_i, args.batch_size);
        }
        println!(
            "Epoch {}/{}, Average loss = {:.5}",
            epoch_i,
            args.epochs,
            batch_loss / args.batch_size as f32
        );

        if epoch_i != 0 && epoch_i % 10 == 0 {
            generate(epoch_i, sample_rate, length, &network, &signal);
        }
    }
    Ok(())
}

fn generate(
    epoch: usize,
    sample_rate: usize,
    length: usize,
    network: &NeuralNet,
    signal: &NormalizedAudio,
) {
    let mut input = 0.0;
    let mut samples = Vec::with_capacity(length);
    let mut state = network.zero_state();
    println!("Generating {} samples...", length);
    while samples.len() < length {
        let (next, new_state) = network.compute(input, state);
        state = new_state;
        samples.push(next.clone());
        input = next;
    }

    let samples = signal.unnormalize(&samples);
    let samples = samples.iter().map(|x| *x as i16).collect_vec();

    util::write_wav(format!("epoch{}.wav", epoch), sample_rate, &samples, None).unwrap();
}

fn into_window<const WINDOW_SIZE: usize>(x: &[f32]) -> [f32; WINDOW_SIZE] {
    assert!(
        x.len() == WINDOW_SIZE,
        "Expected slice of len {}, got {}",
        WINDOW_SIZE,
        x.len()
    );
    x.try_into().unwrap()
}
