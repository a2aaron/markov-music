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
}

#[derive(Debug, Copy, Clone, ValueEnum)]
enum Channel {
    Left,
    Right,
    Both,
}

fn average(a: &WaveletToken, b: &WaveletToken) -> WaveletToken {
    assert!(a.levels() == b.levels());
    let approx_sample = (a.approx_sample + b.approx_sample) / 2.0;
    let detail_samples = a
        .detail_samples
        .iter()
        .zip(b.detail_samples.iter())
        .map(|(a, b)| a.iter().zip(b.iter()).map(|(a, b)| (a + b) / 2.0).collect())
        .collect();

    WaveletToken {
        approx_sample,
        detail_samples,
    }
}

fn interpolate(tokens: &[WaveletToken]) -> Vec<WaveletToken> {
    let mut out_tokens = vec![];
    for i in 0..tokens.len() - 1 {
        let this = &tokens[i];
        let next = &tokens[i + 1];

        out_tokens.push(this.clone());
        let avg = average(this, next);
        out_tokens.push(avg);
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
            let tokens = interpolate(&tokens);

            let wavelets = WaveletHeirarchy::from_tokens(&tokens);

            let samples = wavelet_untransform(&wavelets, args.wavelet);

            if args.debug {
                println!("Generating solo-band samples...");
                let additional_samples = solo_bands(&wavelets, args.wavelet);
                samples
                    .iter()
                    .chain(additional_samples.iter().flatten())
                    .map(|x| (x * i16::MAX as Sample * 0.5) as i16)
                    .collect()
            } else {
                samples
                    .iter()
                    .map(|x| (x * i16::MAX as Sample * 0.5) as i16)
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
