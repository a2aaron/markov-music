use std::error::Error;

use clap::{command, Parser, ValueEnum};
use markov_music::wavelet::{
    solo_bands, wavelet_transform, wavelet_untransform, Sample, WaveletHeirarchy, WaveletToken,
    WaveletType,
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
    #[arg(long, default_value_t = 2)]
    factor: usize,
}

#[derive(Debug, Copy, Clone, ValueEnum)]
enum Channel {
    Left,
    Right,
    Both,
}

fn lerp_token(a: &WaveletToken, b: &WaveletToken, t: f64) -> WaveletToken {
    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + (b - a) * t
    }

    assert!(a.levels() == b.levels());
    let approx_sample = (a.approx_sample + b.approx_sample) / 2.0;
    let detail_samples = a
        .detail_samples
        .iter()
        .zip(b.detail_samples.iter())
        .map(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .map(|(a, b)| lerp(*a, *b, t))
                .collect()
        })
        .collect();

    WaveletToken {
        approx_sample,
        detail_samples,
    }
}

fn interpolate(tokens: &[WaveletToken], factor: usize) -> Vec<WaveletToken> {
    let mut out_tokens = vec![];
    for i in 0..tokens.len() - 1 {
        let this = &tokens[i];
        let next = &tokens[i + 1];

        out_tokens.push(this.clone());

        for j in 0..(factor - 1) {
            let t = (j + 1) as f64 / (factor as f64);
            let avg = lerp_token(this, next, t);
            out_tokens.push(avg);
        }
    }

    out_tokens.push(tokens.last().unwrap().clone());

    out_tokens
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
            let orig_samples = channel
                .iter()
                .map(|x| (*x as Sample) / i16::MAX as Sample)
                .collect();

            let wavelets = wavelet_transform(&orig_samples, args.levels, args.wavelet);
            let tokens = wavelets.tokenize();
            let tokens = interpolate(&tokens, args.factor);

            let wavelets = WaveletHeirarchy::from_tokens(&tokens, args.wavelet);

            let samples = wavelet_untransform(&wavelets);

            if args.debug {
                println!("Generating solo-band samples...");
                let additional_samples = solo_bands(&wavelets);
                samples
                    .iter()
                    .chain(additional_samples.iter().flatten())
                    .map(|x| (x * i16::MAX as Sample) as i16)
                    .collect()
            } else {
                samples
                    .iter()
                    .map(|x| (x * i16::MAX as Sample) as i16)
                    .collect()
            }
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
