use std::error::Error;

use clap::{command, Parser};
use itertools::Itertools;
use markov_music::neural2::{NeuralNet, BATCH_SIZE, QUANTIZATION, SEQ_LEN};
use rand::Rng;
use tch::{nn, Device, IndexOp, Tensor};

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
    #[arg(long, default_value_t = 1000)]
    epochs: usize,
    /// Length, in seconds, of audio to generate.
    #[arg(long, default_value_t = 60)]
    length: usize,
    #[arg(long, default_value_t = 10)]
    generate_every: usize,
}

struct Audio {
    audio: Tensor,
    audio_onehot: Tensor,
    min: f32,
    max: f32,
    rounded_min: i64,
    len: usize,
}

impl Audio {
    fn new(input: &[i16]) -> Audio {
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
        let audio_onehot = audio.onehot(QUANTIZATION as i64);
        Audio {
            audio,
            audio_onehot,
            min,
            max,
            rounded_min,
            len,
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

    fn batch(&self, seq_len: usize, batch_size: usize) -> (Tensor, Tensor) {
        let seq_len = seq_len as i64;

        let (input_one_hots, targets): (Vec<_>, Vec<_>) = (0..batch_size)
            .map(|_| {
                let i = rand::thread_rng().gen_range(0..(self.len as i64 - (seq_len + 1)));
                let input_one_hots = self.audio_onehot.i(i..i + seq_len);
                let targets = self.audio.i((i + 1)..(i + 1) + seq_len);
                (input_one_hots, targets)
            })
            .unzip();

        let input_one_hots = Tensor::stack(&input_one_hots, 0);
        let targets = Tensor::stack(&targets, 0);

        (input_one_hots, targets)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    std::env::set_var("RUST_BACKTRACE", "1");

    let args = Args::parse();

    let (signal, _, sample_rate) = util::read_file(&args.in_path)?;
    let signal = Audio::new(&signal);
    println!("Input samples: {}", signal.len);

    let length = sample_rate * args.length;

    let device = Device::cuda_if_available();
    println!("Training neural net on {:?}", device);
    let vs = nn::VarStore::new(device);

    let mut network = NeuralNet::new(&vs, device);
    for epoch_i in 0..args.epochs {
        let (inputs_onehot, targets) = signal.batch(SEQ_LEN, BATCH_SIZE);
        let loss = network.train(inputs_onehot, targets);
        println!("Epoch {}/{}, loss = {:.5}", epoch_i, args.epochs, loss);

        if epoch_i % args.generate_every == 0 {
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

        if samples.len() % (length / 10).max(10_000) == 0 {
            println!("Generating... ({:?} / {})", samples.len(), length)
        }
    }

    let samples = signal.unnormalize(&samples);
    let samples = samples.iter().map(|x| *x as i16).collect_vec();

    util::write_wav(format!("epoch{}.wav", epoch), sample_rate, &samples, None).unwrap();
}
