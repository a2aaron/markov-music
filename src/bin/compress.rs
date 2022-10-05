use std::{error::Error, path::PathBuf};

use clap::{error::ErrorKind, CommandFactory, Parser};
use markov_music::wavelet::{self, Sample, WaveletType};

mod util;

#[derive(Parser, Debug)]
#[command(version, about)]
/// A wavelet compressor focused on producing distortion effects.
struct Args {
    /// Path to input MP3.
    #[arg(short, long = "input")]
    in_path: PathBuf,
    /// Path to output WAV.
    #[arg(short, long = "output", default_value = "out.wav")]
    out_path: PathBuf,
    /// The number of wavelet detail levels.
    #[arg(short, long, default_value_t = 6)]
    levels: usize,
    /// Wavelet type ot use.
    #[arg(short, long, value_enum, default_value_t = WaveletType::Haar)]
    wavelet: WaveletType,
    /// The number of levels to quantize to.
    ///
    /// If a single value is specified, it is used for all wavelet levels.
    /// When multiple values are specified, they are applied for each level, starting from the
    /// lowest frequency. The last value will apply to all higher frequencies.
    ///
    /// Unusual values: 0 indicates no quantization, 1 suppresses the level.
    #[arg(short, long, required = true, num_args = 1..)]
    quantization: Vec<usize>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = Args::parse();
    // Limit the number of allowed quantization levels.
    if args.quantization.len() > args.levels + 1 {
        Args::command()
            .error(
                ErrorKind::TooManyValues,
                format!(
                    "Too many quantization values. Got {}, expected at most {}.",
                    args.quantization.len(),
                    args.levels + 1
                ),
            )
            .exit();
    }
    // Pad out the quantization with the final value.
    let remaining_levels = (args.levels + 1) - args.quantization.len();
    let fill_value = *args.quantization.last().expect("required value");
    args.quantization
        .extend((0..remaining_levels).map(|_| fill_value));

    let (mut left, mut right, sample_rate) = util::read_mp3_file(&args.in_path)?;

    compress(args.wavelet, &args.quantization, &mut left);
    compress(args.wavelet, &args.quantization, &mut right);

    util::write_wav(&args.out_path, sample_rate, &left, Some(&right))?;

    Ok(())
}

fn compress(wavelet: WaveletType, quantization: &[usize], samples: &mut [i16]) {
    let num_levels = quantization.len() - 1;
    let float_samples = samples
        .iter()
        .map(|&x| x as Sample / i16::MAX as Sample)
        .collect();
    let (mut detail, mut approx, _) = wavelet::wavelet_transform(&float_samples, num_levels, wavelet);
    quantize(quantization[0], &mut approx);
    for (&quant, detail) in quantization[1..].iter().zip(detail.iter_mut().rev()) {
        quantize(quant, detail);
    }
    let float_samples = wavelet::wavelet_untransform(&detail, &approx, wavelet);
    for (out_sample, x) in samples.iter_mut().zip(float_samples) {
        *out_sample = (x * i16::MAX as Sample) as i16;
    }
}

fn quantize(levels: usize, data: &mut [Sample]) {
    // I don't know if this is possible? But handle it anyway, for simplicity
    if data.is_empty() {
        return;
    }

    if levels == 0 {
        return;
    }

    if levels == 1 {
        data.fill(0.0);
        return;
    }

    let (min, max) = data.iter().fold((data[0], data[0]), |(min, max), x| {
        (x.min(min), x.max(max))
    });
    for sample in data {
        *sample = quantize_one(levels, min, max, *sample);
    }
}

fn quantize_one(levels: usize, min: Sample, max: Sample, sample: Sample) -> Sample {
    let scale = (max - min) / levels as Sample;
    (sample / scale).round() * scale
}
