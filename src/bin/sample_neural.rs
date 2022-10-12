use std::error::Error;

use clap::{command, Parser};
use itertools::Itertools;
use markov_music::neural2::{NeuralNet, IN_WINDOW_SIZE, OUT_WINDOW_SIZE};
use rand::Rng;

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

    fn unnormalize(&self, audio: &[f32]) -> Vec<f32> {
        audio
            .iter()
            .map(|x| x * (self.max - self.min) + self.min)
            .collect_vec()
    }

    fn random_example(&self) -> ([f32; IN_WINDOW_SIZE], [f32; OUT_WINDOW_SIZE]) {
        let i = rand::thread_rng()
            .gen_range(0..(self.audio.len() - (IN_WINDOW_SIZE + OUT_WINDOW_SIZE + 1)));
        (
            into_window(&self.audio[i..i + IN_WINDOW_SIZE]),
            into_window(&self.audio[i + IN_WINDOW_SIZE..i + IN_WINDOW_SIZE + OUT_WINDOW_SIZE]),
        )
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    std::env::set_var("RUST_BACKTRACE", "1");

    let args = Args::parse();

    let (channel, _, sample_rate) = util::read_file(&args.in_path)?;

    let samples = NormalizedAudio::new(&channel);

    println!("Input samples: {}", samples.audio.len());

    let length = args.length * sample_rate;

    println!("Training neural net...");
    let mut network = NeuralNet::new();
    for epoch_i in 0.. {
        let mut batch_loss = 0.0;
        for _ in 0..args.batch_size {
            let (input, output) = samples.random_example();
            let loss = network.train(input, output);
            batch_loss += loss;
        }
        println!(
            "Epoch {}/{}, Average loss = {:.5}",
            epoch_i,
            args.epochs,
            batch_loss / args.batch_size as f32
        );

        if epoch_i % 10 == 0 {
            generate(epoch_i, sample_rate, length, &network, &samples);
        }
    }
    Ok(())
}

fn generate(
    epoch: usize,
    sample_rate: usize,
    length: usize,
    network: &NeuralNet,
    samples: &NormalizedAudio,
) {
    let mut window = samples.random_example().0.to_vec();
    let mut out_samples = window.clone();
    while out_samples.len() < length {
        let next = network.compute(into_window(&window));
        out_samples.extend(next.clone());
        window = [&window[OUT_WINDOW_SIZE..], &next].concat();
    }

    let samples = samples.unnormalize(&out_samples);
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
