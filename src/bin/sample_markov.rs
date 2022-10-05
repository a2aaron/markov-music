use std::error::Error;

use clap::{command, Parser, ValueEnum};
use markov::Chain;
use markov_music::quantize::Quantizable;

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
    /// Markov chain order. Higher values means the output is less chaotic, but more deterministic.
    /// Recommended values are betwee 3 and 8, depending on the length and type of input file.
    #[arg(short = 'O', long, default_value_t = 3)]
    order: usize,
    /// Bit depth. This is sets the range of allowed values that
    /// the samples may take on. Higher values result in nicer sounding output, but are more likely
    /// to be deterministic. Often, setting the order to a lower value cancels out setting the depth
    /// to a higher value. Note that this does not actually affect the bit-depth of the output WAV
    /// file, which is always 16 bits. Recommended values are between 8 and 16.
    #[arg(short, long, default_value_t = 14)]
    depth: u32,
    /// Length, in seconds, of audio to generate.
    #[arg(short, long, default_value_t = 60)]
    length: usize,
    /// Which channel of the mp3 to use, (ignored if there is only one channel)
    #[arg(short, long, value_enum, default_value_t = Channel::Both)]
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

    let samples = channels
        .iter()
        .map(|channel| {
            println!(
                "Generating markov chain with order = {}, depth = {} (total states = 2^{})",
                args.order,
                args.depth,
                args.order * args.depth as usize
            );

            let quantization_level = 2usize.pow(args.depth);
            let samples = quantize_and_generate(
                channel,
                args.order,
                args.length * sample_rate,
                quantization_level,
            );
            samples
        })
        .collect::<Vec<Vec<_>>>();

    util::write_wav(
        &args.out_path,
        sample_rate,
        &samples[0],
        samples.get(1).map(Vec::as_ref),
    )?;
    Ok(())
}

fn quantize_and_generate(
    samples: &[i16],
    order: usize,
    length: usize,
    quantization_level: usize,
) -> Vec<i16> {
    println!("Quantizing samples... (level = {})", quantization_level);
    let samples = samples
        .iter()
        .map(|x| Quantizable::quantize(*x, i16::MIN, i16::MAX, quantization_level))
        .collect::<Vec<_>>();

    println!(
        "Training Markov chain of order {}... ({} samples)",
        order,
        samples.len()
    );
    let mut chain = Chain::of_order(order);
    chain.feed(samples);

    println!("Generating Markov chain... ({} samples)", length);
    chain
        .iter()
        .flatten()
        .map(|x| Quantizable::unquantize(x, i16::MIN, i16::MAX, quantization_level))
        .take(length)
        .collect()
}
