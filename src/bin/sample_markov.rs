use std::{error::Error, io::Read};

use clap::{command, Parser, ValueEnum};
use markov_music::{
    samples::markov_samples,
    wavelet::{wavelet_transform, wavelet_untransform},
};
use minimp3::Decoder;
use wav::WAV_FORMAT_PCM;

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
    /// What generation mode to use. "sample" means the markov chain directly generates audio samples,
    /// while "wavelet" means the markov chain will generate wavelet coefficents.
    #[arg(short, long, value_enum, default_value_t = Mode::Sample)]
    mode: Mode,
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

fn get_mp3_data<R: Read>(
    mut decoder: Decoder<R>,
) -> Result<(Vec<i16>, Vec<i16>, usize), Box<dyn Error>> {
    let mut frames = vec![];
    loop {
        match decoder.next_frame() {
            Ok(frame) => frames.push(frame),
            Err(err) => match err {
                minimp3::Error::Io(err) => return Err(err.into()),
                minimp3::Error::InsufficientData => continue,
                minimp3::Error::SkippedData => continue,
                minimp3::Error::Eof => break,
            },
        }
    }
    let samples = frames
        .iter()
        .flat_map(|frame| frame.data.clone())
        .collect::<Vec<i16>>();
    let (left, right) = split_channels(&samples);

    let sample_rate = frames[0].sample_rate as usize;

    Ok((left, right, sample_rate))
}

fn split_channels<T: Copy>(samples: &[T]) -> (Vec<T>, Vec<T>) {
    let mut left = Vec::with_capacity(samples.len() / 2);
    let mut right = Vec::with_capacity(samples.len() / 2);
    for chunk in samples.chunks_exact(2) {
        left.push(chunk[0]);
        right.push(chunk[1]);
    }
    (left, right)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let file = std::fs::File::open(args.in_path)?;
    let decoder = Decoder::new(file);
    let (left, right, sample_rate) = get_mp3_data(decoder)?;

    let samples = match args.channel {
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
            let samples =
                markov_samples(&samples, args.order, max_range, args.length * sample_rate);
            samples
        }
        Mode::Wavelet => {
            let samples = samples.iter().map(|x| *x as f32).collect();
            let (hi_passes, lowest_pass) = wavelet_transform(&samples, 1);
            let samples = wavelet_untransform(&hi_passes, &lowest_pass);
            let samples = samples.iter().map(|x| *x as i16).collect();

            samples
        }
    };

    let wav_header = wav::header::Header::new(WAV_FORMAT_PCM, 1, sample_rate as u32, 16);
    let track = wav::BitDepth::Sixteen(samples);
    let mut out_file = std::fs::File::create(args.out_path)?;
    wav::write(wav_header, &track, &mut out_file)?;

    Ok(())
}
