use std::error::Error;

use clap::{command, Parser, ValueEnum};
use itertools::Itertools;
use markov_music::distribution::{normalize, unnormalize, Distribution};

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
    #[arg(long, default_value_t = 3)]
    order: usize,
    /// Length, in seconds, of audio to generate.
    #[arg(long, default_value_t = 60)]
    length: usize,
    /// Which channel of the mp3 to use, (ignored if there is only one channel)
    #[arg(long, value_enum, default_value_t = Channel::Both)]
    channel: Channel,
}

#[derive(Debug, Copy, Clone, ValueEnum)]
enum Mode {
    Sample,
    Wavelet,
}

#[derive(Debug, Copy, Clone, ValueEnum)]
enum Channel {
    Left,
    Right,
    Both,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let (left, right, sample_rate) = util::read_mp3_file(&args.in_path)?;

    let channels = if let Some(right) = right {
        match args.channel {
            Channel::Left => vec![left],
            Channel::Right => vec![right],
            Channel::Both => vec![left, right],
        }
    } else {
        println!("Ignoring --channel flag because there is only one channel");
        vec![left]
    };

    let samples: Vec<Vec<i16>> = channels
        .iter()
        .map(|channel| {
            let channel = channel.iter().map(|x| *x as f32).collect_vec();

            let (channel, min, max) = normalize(&channel);

            // let length = args.length * sample_rate;
            let length = 5000;
            println!("Constructing distribution...");
            let distribution = Distribution::new(&channel, args.order);

            println!("Generating {} samples...", length);

            let mut window = channel[4000..4000 + args.order].to_vec(); // random_vector(args.order);
            let mut out_samples = window.clone();
            for i in 0..length {
                let (error, next) = distribution.next_sample(&window);
                if i % 300 == 0 {
                    println!("{} / {}, next: {}, error: {}", i, length, next, error);
                }
                out_samples.push(next);
                window.remove(0);
                window.push(next);
            }

            let out_samples = unnormalize(&out_samples, min, max);
            let out_samples = out_samples.iter().map(|x| *x as i16).collect_vec();
            out_samples
        })
        .collect();

    util::write_wav(
        &args.out_path,
        sample_rate,
        &samples[0],
        samples.get(1).map(Vec::as_ref),
    )?;
    Ok(())
}
