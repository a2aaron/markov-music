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

struct Audio {
    audio: Vec<i64>,
    min: f32,
    max: f32,
    rounded_min: i64,
}

impl Audio {
    fn new(input: &[i16]) -> Audio {
        let max = input.iter().cloned().max().unwrap() as f32;
        let min = input.iter().cloned().min().unwrap() as f32;
        let scale = (max - min) / 255.0;
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
        assert!(
            *audio.iter().min().unwrap() >= 0i64,
            "{}",
            audio.iter().min().unwrap()
        );
        assert!(
            *audio.iter().max().unwrap() < 256i64,
            "{}",
            audio.iter().max().unwrap()
        );
        assert_eq!((0..256).len(), 256);
        Audio {
            audio,
            min,
            max,
            rounded_min,
        }
    }

    fn len(&self) -> usize {
        self.audio.len()
    }

    fn unnormalize(&self, audio: &[i64]) -> Vec<f32> {
        let scale = (self.max - self.min) / 255.0;
        audio
            .iter()
            .cloned()
            .map(|x| {
                let sample = (x + self.rounded_min) as f32;
                sample * scale
            })
            .collect_vec()
    }

    fn random_example(&self) -> &[i64] {
        let i = rand::thread_rng().gen_range(0..(self.audio.len() - (SEQ_LEN + 1)));
        &self.audio[i..i + SEQ_LEN]
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    std::env::set_var("RUST_BACKTRACE", "1");

    let args = Args::parse();

    let (signal, _, sample_rate) = util::read_file(&args.in_path)?;
    let signal = Audio::new(&signal);
    println!("Input samples: {}", signal.len());

    let length = sample_rate * args.length;

    let device = Device::cuda_if_available();
    println!("Training neural net on {:?}", device);
    let vs = nn::VarStore::new(device);

    let quantization = 256;
    let mut network = NeuralNet::new(&vs, device, quantization);
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

        if epoch_i % 10 == 0 {
            generate(epoch_i, sample_rate, length, &network, &signal);
        }
    }
    Ok(())
}

fn generate(epoch: usize, sample_rate: usize, length: usize, network: &NeuralNet, signal: &Audio) {
    let mut input = 0;
    let mut samples = Vec::with_capacity(length);
    let mut state = network.zero_state();
    println!("Generating {} samples...", length);
    while samples.len() < length {
        let (next, new_state) = network.compute(input, state);
        state = new_state;
        samples.push(next.clone());
        input = next;

        if samples.len() % 10_000 == 0 {
            println!("Generating... ({:?} / {})", samples.len(), length)
        }
    }

    let samples = signal.unnormalize(&samples);
    let samples = samples.iter().map(|x| *x as i16).collect_vec();

    util::write_wav(format!("epoch{}.wav", epoch), sample_rate, &samples, None).unwrap();
}
