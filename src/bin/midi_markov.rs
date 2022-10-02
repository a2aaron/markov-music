use std::error::Error;

use clap::{command, Parser};
use markov_music::{
    midi::generate_markov,
    notes::{MidiInfo, NoteDuration},
};
use midly::{Format, TrackEvent};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
/// A MIDI generator powered by markov chain.
struct Args {
    /// Path to input midi file.
    #[arg(short, long = "in")]
    in_path: String,
    /// Path to output midi file.
    #[arg(short, long = "out", default_value = "out.mid")]
    out_path: String,
    /// Markov chain order. Higher values means the output is less chaotic, but more deterministic.
    #[arg(short = 'O', long, default_value_t = 1)]
    order: usize,
    /// Number of measures to generate.
    #[arg(short, long, default_value_t = 32)]
    measures: usize,
    /// Quantization level to use. Usually, eighth or sixteenth is good. High quantization gives more accurate results, but also lower quality output.
    #[arg(short, long, value_enum, default_value_t = NoteDuration::Eighth)]
    quantization: NoteDuration,
    /// Specific track indicies to use from the midi file. If not provided, then every track will be used.
    #[arg(short = 'I', long = "indicies")]
    track_indicies: Option<Vec<usize>>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let raw = std::fs::read(args.in_path)?;
    let (mut header, tracks) = midly::parse(&raw)?;
    let tracks = tracks.collect_tracks()?;
    let midi_info = MidiInfo::new(header, &tracks);

    let tracks: Box<dyn Iterator<Item = &Vec<TrackEvent>>> =
        if let Some(indicies) = args.track_indicies {
            Box::new(tracks.iter().enumerate().filter_map(move |(i, track)| {
                if indicies.contains(&i) {
                    Some(track)
                } else {
                    None
                }
            }))
        } else {
            Box::new(tracks.iter())
        };

    // let quantization = match args.quantization {
    //     1 => NoteDuration::Whole,
    //     2 => NoteDuration::Half,
    //     4 => NoteDuration::Quarter,
    //     8 => NoteDuration::Eighth,
    //     16 => NoteDuration::Sixteenth,
    //     32 => NoteDuration::ThirtySecond,
    //     64 => NoteDuration::SixtyFourth,
    // };

    let out_tracks: Vec<_> = tracks
        .map(|track| {
            generate_markov(
                midi_info,
                track,
                args.quantization,
                args.order,
                args.measures,
            )
        })
        .collect();

    header.format = if out_tracks.len() == 1 {
        Format::SingleTrack
    } else {
        Format::Parallel
    };

    let mut outfile = std::fs::File::create(args.out_path)?;
    midly::write_std(&header, &out_tracks, &mut outfile)?;

    Ok(())
}
