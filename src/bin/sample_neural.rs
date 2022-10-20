use std::{error::Error, time::Instant};

use clap::{command, Parser};
use itertools::Itertools;
use markov_music::neural2::{DEVICE, assert_shape, reshape, write_csv, Frames, NetworkParams, NeuralNet};
use rand::Rng;
use serde::{Deserialize, Serialize};
use tch::{nn, IndexOp, Tensor};

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
    /// If provided, load a model from a checkpoint.
    #[arg(long)]
    load_model: Option<String>,
    #[arg(long)]
    load_params: Option<String>,
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
    /// If provided, do not use skip connections in the RNN layers.
    #[arg(long, default_value_t = false)]
    no_skip_connections: bool,
    /// The size of the embedding
    #[arg(long, default_value_t = 256)]
    embed_size: usize,
    /// The number of quantization levels to use.
    #[arg(long, default_value_t = 256)]
    quantization: usize,
    /// The file to read parameters from
    #[arg(long)]
    args_file: Option<String>,
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
            skip_connections: !self.no_skip_connections,
            epoch: 0,
        }
    }

    fn set_network_params(&mut self, params: NetworkParams) {
        self.learn_rate = params.learn_rate;
        self.batch_size = params.batch_size;
        self.frame_size = params.frame_size;
        self.num_frames = params.num_frames;
        self.hidden_size = params.hidden_size;
        self.rnn_layers = params.rnn_layers;
        self.embed_size = params.embed_size;
        self.quantization = params.quantization;
        self.no_skip_connections = !params.skip_connections;
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

        let audio = Tensor::of_slice(&audio).to_device(*DEVICE);
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

    fn batch(
        &self,
        batch_size: usize,
        num_frames: usize,
        frame_size: usize,
    ) -> (Frames, Frames, Frames) {
        fn index(tensor: &Tensor, start: usize, length: usize) -> Tensor {
            tensor.i(start as i64..(start + length) as i64)
        }

        let seq_len = num_frames * frame_size;

        let mut input_vec = vec![];
        let mut overlap_vec = vec![];
        let mut targets_vec = vec![];

        for _ in 0..batch_size {
            let i = rand::thread_rng().gen_range(0..(self.len - (seq_len + frame_size)));

            let input = index(&self.audio, i, seq_len);
            let overlap = index(&self.audio, i, seq_len + frame_size);
            let targets = index(&self.audio, i + frame_size, seq_len);

            assert_shape(&[seq_len], &input);
            assert_shape(&[seq_len + frame_size], &overlap);
            assert_shape(&[seq_len], &targets);

            input_vec.push(input);
            overlap_vec.push(overlap);
            targets_vec.push(targets);
        }

        let input = Tensor::stack(&input_vec, 0);
        let overlap = Tensor::stack(&overlap_vec, 0);
        let targets = Tensor::stack(&targets_vec, 0);

        assert_shape(&[batch_size, seq_len], &input);
        assert_shape(&[batch_size, seq_len + frame_size], &overlap);
        assert_shape(&[batch_size, seq_len], &targets);

        let input = reshape(&[batch_size, num_frames, frame_size], &input);
        let overlap = reshape(&[batch_size, num_frames + 1, frame_size], &overlap);
        let targets = reshape(&[batch_size, num_frames, frame_size], &targets);

        (
            Frames::new(input, batch_size, num_frames, frame_size),
            Frames::new(overlap, batch_size, num_frames + 1, frame_size),
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
    let args_file = args
        .args_file
        .clone()
        .unwrap_or(format!("{}_args.json", args.out_path));

    args.get_updatable().write_to_file(&args_file)?;

    match args.debug {
        1 => std::env::set_var("RUST_BACKTRACE", "1"),
        2 => std::env::set_var("RUST_BACKTRACE", "full"),
        _ => (),
    }

    let (signal, _, sample_rate) = util::read_file(&args.in_path)?;
    let signal = Audio::new(&signal, sample_rate, params.quantization);

    let mut epoch_i = 0;
    let (mut network, vs) = match (&args.load_model, &args.load_params) {
        (None, None) => {
            println!("Initializing new neural net");
            let vs = nn::VarStore::new(*DEVICE);
            let network = NeuralNet::new(&vs, params);
            (network, vs)
        }
        (Some(model_path), Some(params_path)) => {
            println!(
                "Initializing neural net from saved model:\nmodel: {}\nparams: {}",
                model_path, params_path
            );
            let mut vs = nn::VarStore::new(*DEVICE);
            let (network, params) = NeuralNet::from_saved(&mut vs, model_path, params_path)?;
            epoch_i = params.epoch;
            args.set_network_params(params);
            println!(
                "Successfully initialized network with the following params: {:?}",
                params
            );
            (network, vs)
        }
        _ => {
            return Err("Must provide both model and parameters file!".into());
        }
    };
    let mut losses = vec![];

    println!("Input samples: {}", signal.len);
    println!("Training neural net on {:?}", *DEVICE);
    println!("== Arguments ==\n{:#?}", args);
    println!("===============");
    {
        signal.write_to_file(
            &format!("{}_ground_truth.wav", args.out_path),
            &Vec::<i64>::from(&signal.audio),
        );

        let (inputs, overlap, targets) =
            signal.batch(args.batch_size, args.num_frames, args.frame_size);
        signal.write_to_file(
            &format!("{}_batch_example_inputs.wav", args.out_path),
            &inputs.samples(),
        );
        signal.write_to_file(
            &format!("{}_batch_example_overlap.wav", args.out_path),
            &overlap.samples(),
        );
        signal.write_to_file(
            &format!("{}_batch_example_targets.wav", args.out_path),
            &targets.samples(),
        );
    }

    loop {
        let now = Instant::now();
        let (frames, overlap, targets) =
            signal.batch(params.batch_size, params.num_frames, params.frame_size);

        let backwards_debug = network.backward(&frames, &overlap, &targets);

        println!(
            "Epoch {}, loss = {:.8} (time = {:?}, accuracy = {:.2}%)",
            epoch_i,
            backwards_debug.loss,
            now.elapsed(),
            backwards_debug.accuracy * 100.0,
        );
        losses.push(backwards_debug.loss);

        if let Err(err) = write_csv(&format!("{}_losses.csv", args.out_path), &[losses.clone()]) {
            println!(
                "Couldn't write losses file {}. Reason: {}",
                format!("{}_losses.csv", args.out_path),
                err
            );
        }

        if args.checkpoint_every != 0 && epoch_i % args.checkpoint_every == 0 {
            let path = format!("{}_epoch_{}", args.out_path, epoch_i);
            if let Err(err) = network.checkpoint(&vs, &path) {
                println!(
                    "[ERROR] Couldn't checkpoint to file {}! Reason: {}",
                    path, err
                );
            } else {
                println!("Checkpointed at epoch {} to file {}", epoch_i, path);
            };
            if args.debug != 0 {
                let now = Instant::now();
                let logits_path = format!("{}_epoch_{}_logits_sample.wav", args.out_path, epoch_i);
                let target_path = format!("{}_epoch_{}_logits_target.wav", args.out_path, epoch_i);
                let logits = backwards_debug.logits.sample();
                let targets = Vec::<i64>::from(backwards_debug.targets);
                signal.write_to_file(&logits_path, &logits);
                signal.write_to_file(&target_path, &targets);
                println!(
                    "Saved debug files {} and {} (in {:?})",
                    logits_path,
                    target_path,
                    now.elapsed()
                );
            }
        }

        if args.generate_every != 0 && epoch_i % args.generate_every == 0 {
            generate(
                &args.out_path,
                epoch_i,
                sample_rate * args.length,
                &network,
                &signal,
            );
        }

        match UpdatableArgs::try_from_file(&args_file) {
            Ok(new_args) => {
                if new_args != args.get_updatable() {
                    println!("Updated args!\n{:#?}", new_args);
                    args.set_updatable(&new_args);
                    network.set_learn_rate(new_args.learn_rate.0);
                }
            }
            Err(err) => println!("Couldn't update args: {:?}", err),
        }
        epoch_i += 1;
    }
}

fn generate(name: &str, epoch_i: usize, length: usize, network: &NeuralNet, signal: &Audio) {
    let total_time = Instant::now();
    let mut state = network.zeros(1);
    let mut frame = signal.batch(1, 1, network.params.frame_size).0.samples();
    let mut samples = Vec::with_capacity(length);
    samples.extend(frame.iter());
    println!("Generating {} samples...", length);

    let mut i = 0;
    let mut now = Instant::now();
    while samples.len() < length {
        let (next_frame, next_state) = network.forward(frame, &state);
        state = next_state;
        samples.extend(next_frame.iter());
        frame = next_frame;

        if i % 500 == 0 {
            println!(
                "Generating... ({:?} / {} samples ({:.2}%), epoch {}, took {:?})",
                samples.len(),
                length,
                100.0 * (samples.len() as f32 / length as f32),
                epoch_i,
                now.elapsed()
            );
            now = Instant::now();
        }
        i += 1;
    }

    signal.write_to_file(&format!("{}{}.wav", name, epoch_i), &samples);
    println!("Generated! total time = {:?}", total_time.elapsed());
}
