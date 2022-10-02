use std::{error::Error, io::Read, path::Path};

use markov::{Chain, Chainable};
use minimp3::{Decoder as Mp3Decoder, Frame};
use wav::{header::Header as WavHeader, WAV_FORMAT_PCM};

pub fn get_mp3_data<R: Read>(mut decoder: Mp3Decoder<R>) -> Result<Vec<Frame>, Box<dyn Error>> {
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
    Ok(frames)
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

fn merge_channels<T: Copy>(left: &[T], right: &[T]) -> Vec<T> {
    let mut merge = Vec::with_capacity(left.len() + right.len());
    for (l, r) in left.iter().zip(right.iter()) {
        merge.push(*l);
        merge.push(*r);
    }
    merge
}

fn markov<T: Chainable>(samples: &[T], order: usize) -> Chain<T> {
    let mut chain = Chain::of_order(order);
    chain.feed(samples);
    chain
}

fn generate<T: Chainable>(samples: &[T], order: usize, length: usize) -> Vec<T> {
    let sample_len = samples.len();
    println!("Training Markov chain... ({} samples)", sample_len);
    let chain = markov(&samples[..sample_len], order);

    println!("Generating Markov chain... ({} samples)", length);
    chain.iter().flatten().take(length).collect::<Vec<_>>()
}

fn inv_lerp(a: f32, b: f32, x: f32) -> f32 {
    (x - a) / (b - a)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn expand_to_i16(x: usize, max_range: usize) -> i16 {
    let x = x as f32;
    let i16_max = i16::MAX as f32;
    let i16_min = i16::MIN as f32;

    let t = inv_lerp(0.0, max_range as f32, x);
    lerp(i16_min, i16_max, t) as i16
}

fn constrain_to_range(x: i16, max_range: usize) -> usize {
    let x = x as f32;
    let i16_max = i16::MAX as f32;
    let i16_min = i16::MIN as f32;

    let t = inv_lerp(i16_min, i16_max, x);
    lerp(0.0, max_range as f32, t) as usize
}

pub fn markov_mp3(
    in_path: impl AsRef<Path>,
    out_path: impl AsRef<Path>,
    order: usize,
    max_range: usize,
) -> Result<(), Box<dyn Error>> {
    println!("Opening file...");
    let file = std::fs::File::open(in_path)?;
    let decoder = Mp3Decoder::new(file);

    println!("Decoding mp3 data...");
    let frames = get_mp3_data(decoder)?;
    let sample_rate = frames[0].sample_rate as u32;

    println!("Flat map...");
    let samples = frames
        .iter()
        .flat_map(|frame| frame.data.iter().map(|x| constrain_to_range(*x, max_range)))
        .collect::<Vec<_>>();

    println!("Splitting...");
    let (left, _right) = split_channels(&samples);

    println!("Generating left channel...");
    let left = generate(&left, order, sample_rate as usize * 60);

    // println!("Generating right channel...");
    // let right = generate(&right, order);

    println!("Merging...");
    let samples = merge_channels(&left, &left);
    let samples = samples
        .iter()
        .map(|x| expand_to_i16(*x, max_range))
        .collect();

    println!("Writing wav data...");
    let wav_header = WavHeader::new(WAV_FORMAT_PCM, 2, sample_rate, 16);
    let track = wav::BitDepth::Sixteen(samples);
    let mut out_file = std::fs::File::create(out_path)?;
    wav::write(wav_header, &track, &mut out_file)?;
    Ok(())
}
