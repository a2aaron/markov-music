use std::error::Error;

use clap::{command, Parser, ValueEnum};
use markov_music::{
    quantize::{Quantizable, QuantizedSample},
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
    #[arg(long, default_value_t = 3)]
    order: usize,
    /// The number of levels to quantize to.
    #[arg(long, default_value_t = 256)]
    quantization: u32,
    /// Length, in seconds, of audio to generate.
    #[arg(long, default_value_t = 60)]
    length: usize,
    /// Which channel of the mp3 to use.
    #[arg(value_enum, default_value_t = Channel::Left)]
    channel: Channel,
    /// Number of levels to use in the wavelet transform.
    #[arg(long, default_value_t = 6)]
    levels: usize,
    /// Wavelet type to use
    #[arg(long, value_enum, default_value_t = WaveletType::Haar)]
    wavelet: WaveletType,
    /// Enable debug mode.
    #[arg(long)]
    debug: bool,
}

#[derive(Debug, Copy, Clone, ValueEnum)]
enum Channel {
    Left,
    Right,
    Both,
}

fn quantize(signal: &[Sample], quantization_level: u32) -> (Vec<QuantizedSample>, Sample, Sample) {
    let min = signal.iter().cloned().reduce(f64::min).unwrap();
    let max = signal.iter().cloned().reduce(f64::max).unwrap();

    let quantized = signal
        .iter()
        .map(|sample| Quantizable::quantize(*sample, min, max, quantization_level))
        .collect();
    (quantized, min, max)
}

fn unquantize(
    (signal, min, max): &(Vec<QuantizedSample>, Sample, Sample),
    quantization_level: u32,
) -> Vec<Sample> {
    signal
        .iter()
        .map(|quantized| Quantizable::unquantize(*quantized, *min, *max, quantization_level))
        .collect()
}

fn unquantize_bands(
    hi_bands: &[(Vec<QuantizedSample>, f64, f64)],
    lowest_pass: &(Vec<QuantizedSample>, f64, f64),
    quantization_level: u32,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let hi_bands = hi_bands
        .iter()
        .map(|hi_pass| unquantize(&hi_pass, quantization_level))
        .collect::<Vec<_>>();
    let lowest_pass = unquantize(&lowest_pass, quantization_level);
    (hi_bands, lowest_pass)
}

fn quantize_bands(
    hi_bands: &[Vec<f64>],
    lowest_pass: &[f64],
    quantization_level: u32,
) -> (
    Vec<(Vec<QuantizedSample>, f64, f64)>,
    (Vec<QuantizedSample>, f64, f64),
) {
    let hi_bands = hi_bands
        .iter()
        .map(|hi_pass| quantize(&hi_pass, quantization_level))
        .collect();
    let lowest_pass = quantize(&lowest_pass, quantization_level);
    (hi_bands, lowest_pass)
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
            let orig_samples = channel
                .iter()
                .map(|x| (*x as Sample) / i16::MAX as Sample)
                .collect();
            let (hi_bands, lowest_pass, low_bands) =
                wavelet_transform(&orig_samples, args.levels, args.wavelet);

            let (hi_bands, lowest_pass) =
                quantize_bands(&hi_bands, &lowest_pass, args.quantization);

            let (hi_bands, lowest_pass) =
                unquantize_bands(&hi_bands, &lowest_pass, args.quantization);

            let samples = wavelet_untransform(&hi_bands, &lowest_pass, args.wavelet);

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
                    .chain(low_bands.iter().flatten())
                    .chain(hi_bands.iter().flatten())
                    .cloned()
                    .map(|x| (x * i16::MAX as Sample) as i16)
                    .collect()
            } else {
                samples
                    .iter()
                    .map(|x| (x * i16::MAX as Sample) as i16)
                    .collect::<Vec<_>>()
            }
        })
        .collect::<Vec<_>>();

    util::write_wav(
        &args.out_path,
        sample_rate,
        &samples[0],
        samples.get(1).map(Vec::as_ref),
    )?;

    Ok(())
}
