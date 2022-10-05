use minimp3::Decoder;
use std::{error::Error, io::Read, path::Path};
use wav::WAV_FORMAT_PCM;

pub fn read_mp3_file(
    path: impl AsRef<Path>,
) -> Result<(Vec<i16>, Vec<i16>, usize), Box<dyn Error>> {
    let file = std::fs::File::open(path)?;
    let decoder = Decoder::new(file);
    get_mp3_data(decoder)
}

pub fn get_mp3_data<R: Read>(
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

pub fn split_channels<T: Copy>(samples: &[T]) -> (Vec<T>, Vec<T>) {
    let mut left = Vec::with_capacity(samples.len() / 2);
    let mut right = Vec::with_capacity(samples.len() / 2);
    for chunk in samples.chunks_exact(2) {
        left.push(chunk[0]);
        right.push(chunk[1]);
    }
    (left, right)
}

pub fn write_wav(path: impl AsRef<Path>, sample_rate: usize, left: &[i16], right: Option<&[i16]>) -> Result<(), Box<dyn Error>> {
    let channel_count = if right.is_some() { 2 } else { 1 };
    let wav_header =
        wav::header::Header::new(WAV_FORMAT_PCM, channel_count, sample_rate as u32, 16);
    let samples = if let Some(right) = right {
        left.iter()
            .zip(right)
            .map(|(&a, &b)| [a, b].into_iter())
            .flatten()
            .collect()
    } else {
        left.to_vec()
    };
    let track = wav::BitDepth::Sixteen(samples);
    let mut out_file = std::fs::File::create(path)?;
    wav::write(wav_header, &track, &mut out_file)?;
    Ok(())
}
