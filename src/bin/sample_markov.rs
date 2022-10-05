use std::error::Error;

use clap::{command, Parser, ValueEnum};
use markov_music::{
    samples::markov_samples,
    wavelet::{wavelet_transform, wavelet_untransform, Sample, WaveletType},
};

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
    /// Bit depth. Only compatible with sample mode. This is sets the range of allowed values that
    /// the samples may take on. Higher values result in nicer sounding output, but are more likely
    /// to be deterministic. Often, setting the order to a lower value cancels out setting the depth
    /// to a higher value. Note that this does not actually affect the bit-depth of the output WAV
    /// file, which is always 16 bits. Recommended values are between 8 and 16.
    #[arg(short, long, default_value_t = 14)]
    depth: u32,
    /// Length, in seconds, of audio to generate.
    #[arg(short, long, default_value_t = 60)]
    length: usize,
    /// Which channel of the mp3 to use.
    #[arg(short, long, value_enum, default_value_t = Channel::Left)]
    channel: Channel,
    /// Number of levels to use in the wavelet transform.
    #[arg(short, long, default_value_t = 6)]
    levels: usize,
    /// Wavelet type to use
    #[arg(short, long, value_enum, default_value_t = WaveletType::Haar)]
    wavelet: WaveletType,
    /// What generation mode to use. "sample" means the markov chain directly generates audio samples,
    /// while "wavelet" means the markov chain will generate wavelet coefficents.
    #[arg(short, long, value_enum, default_value_t = Mode::Sample)]
    mode: Mode,
    /// Enable debug mode.
    #[arg(long)]
    debug: bool,
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
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let (left, right, sample_rate) = util::read_mp3_file(&args.in_path)?;

    let orig_samples = match args.channel {
        Channel::Left => left,
        Channel::Right => right,
    };

    let samples = match args.mode {
        Mode::Sample => {
            println!(
                "Generating markov chain with order = {}, depth = {} (total states = 2^{})",
                args.order,
                args.depth,
                args.order * args.depth as usize
            );

            let max_range = 2usize.pow(args.depth);
            let samples = markov_samples(
                &orig_samples,
                args.order,
                max_range,
                args.length * sample_rate,
            );
            samples
        }
        Mode::Wavelet => {
            let orig_samples = orig_samples
                .iter()
                .map(|x| (*x as Sample) / i16::MAX as Sample)
                .collect();
            let (hi_passes, lowest_pass, low_passes) =
                wavelet_transform(&orig_samples, args.levels, args.wavelet);

            let samples = wavelet_untransform(&hi_passes, &lowest_pass, args.wavelet);

            if args.debug {
                println!("Layers: {}, Wavelet: {:?}", args.levels, args.wavelet);

                println!(
                    "Max error: {}",
                    orig_samples
                        .iter()
                        .zip(samples.iter())
                        .map(|(a, b)| (a - b).abs())
                        .reduce(Sample::max)
                        .unwrap(),
                );

                let error_sum = orig_samples
                    .iter()
                    .zip(samples.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<Sample>();
                println!("Sum of absolute error: {}", error_sum);
                println!(
                    "Average error per sample: {}\n",
                    error_sum / orig_samples.len() as Sample
                );

                samples
                    .iter()
                    .chain(low_passes.iter().flatten())
                    .chain(hi_passes.iter().flatten())
                    .cloned()
                    .map(|x| (x * i16::MAX as Sample / 4.0) as i16)
                    .collect()
            } else {
                samples
                    .iter()
                    .map(|x| (x * i16::MAX as Sample / 4.0) as i16)
                    .collect()
            }
        }
    };

    util::write_wav(&args.out_path, sample_rate, &samples, None)?;

    Ok(())
}
