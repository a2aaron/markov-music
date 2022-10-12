use minimp3::Decoder;
use std::{error::Error, io::Read, path::Path};
use wav::WAV_FORMAT_PCM;

pub fn read_file(
    path: impl AsRef<Path>,
) -> Result<(Vec<i16>, Option<Vec<i16>>, usize), Box<dyn Error>> {
    if let Ok(mp3_data) = read_mp3_file(&path) {
        Ok(mp3_data)
    } else {
        read_wav_file(path)
    }
}

pub fn read_wav_file(
    path: impl AsRef<Path>,
) -> Result<(Vec<i16>, Option<Vec<i16>>, usize), Box<dyn Error>> {
    let mut file = std::fs::File::open(path)?;
    let (header, bit_depth) = wav::read(&mut file)?;
    let data = match bit_depth {
        wav::BitDepth::Eight(_samples) => todo!(),
        wav::BitDepth::Sixteen(samples) => samples,
        wav::BitDepth::TwentyFour(_samples) => todo!(),
        wav::BitDepth::ThirtyTwoFloat(_samples) => todo!(),
        wav::BitDepth::Empty => todo!(),
    };
    let sample_rate = header.sampling_rate as usize;
    if header.channel_count == 1 {
        Ok((data, None, sample_rate))
    } else if header.channel_count == 2 {
        let (left, right) = split_channels(&data);
        Ok((left, Some(right), sample_rate))
    } else {
        Err(format!(
            "Expected channel count to equal 1 or 2, got {}",
            header.channel_count
        )
        .into())
    }
}

pub fn read_mp3_file(
    path: impl AsRef<Path>,
) -> Result<(Vec<i16>, Option<Vec<i16>>, usize), Box<dyn Error>> {
    let file = std::fs::File::open(path)?;
    let decoder = Decoder::new(file);
    get_mp3_data(decoder)
}

pub fn get_mp3_data<R: Read>(
    mut decoder: Decoder<R>,
) -> Result<(Vec<i16>, Option<Vec<i16>>, usize), Box<dyn Error>> {
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

    if frames.is_empty() {
        return Err("MP3 file is empty!".into());
    }

    let channels = frames[0].channels;
    let sample_rate = frames[0].sample_rate as usize;

    let all_mono = frames.iter().all(|frame| frame.channels == 1);
    let all_stereo = frames.iter().all(|frame| frame.channels == 2);
    let all_same_sample_rate = frames
        .iter()
        .all(|frame| frame.sample_rate == frames[0].sample_rate);

    if !all_mono && !all_stereo {
        println!("[Warning] MP3 file is not entirely mono or stereo! Treating MP3 file as if it only has {} channels.", channels);
    }

    if !all_same_sample_rate {
        println!("[Warning] MP3 file does not have a constant sample rate! Treating MP3 file as if it only has a sample rate of {}.", sample_rate);
    }

    let samples = frames
        .iter()
        .flat_map(|frame| frame.data.clone())
        .collect::<Vec<i16>>();
    if channels == 1 {
        Ok((samples, None, sample_rate))
    } else if channels == 2 {
        let (left, right) = split_channels(&samples);
        Ok((left, Some(right), sample_rate))
    } else {
        Err(format!("Expected MP3 to contain 1 or 2 channels, got {}", channels).into())
    }
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

pub fn write_wav(
    path: impl AsRef<Path>,
    sample_rate: usize,
    left: &[i16],
    right: Option<&[i16]>,
) -> Result<(), Box<dyn Error>> {
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
