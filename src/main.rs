#![feature(hash_drain_filter)]

mod notes;

use std::error::Error;

use markov::Chain;
use midly::{
    num::{u28, u4, u7},
    MetaMessage, MidiMessage, TrackEvent, TrackEventKind,
};

use notes::{MidiInfo, Note, NoteDuration, StepSequencer};

use crate::notes::{MarkovNote, QuantizedNote};

const MEGALOVANIA: &str = "Undertale_-_Megalovania.mid";
// const MEGALOVANIA: &str = "Megalovania Loop.mid";
// const MEGALOVANIA: &str = "test.mid";

fn debug_print_midi(path: &str) -> Result<(), Box<dyn Error>> {
    println!("-- FILE {} --", path);
    let raw = std::fs::read(path)?;
    let (header, tracks) = midly::parse(&raw)?;
    println!("{:?}", header);
    for (i, track) in tracks.enumerate() {
        println!("-- TRACK {} --", i);
        for event in track? {
            let event = event?;
            match event.kind {
                TrackEventKind::Midi { message, .. } => match message {
                    MidiMessage::NoteOff { .. } => continue,
                    MidiMessage::NoteOn { .. } => continue,
                    _ => (),
                },
                _ => (),
            }
            println!("delta = {:?}\t{:?}", event.delta, event.kind);
        }
        println!("-- END TRACK {} --", i);
    }
    println!("-- END FILE {} --", path);
    Ok(())
}

fn extract_meta_messages<'a>(track: &[TrackEvent<'a>]) -> (Vec<TrackEvent<'a>>, Option<u4>) {
    let mut likely_channel = None;
    let events = track
        .iter()
        .cloned()
        .filter(|event| match event.kind {
            TrackEventKind::Meta(event) => match event {
                MetaMessage::EndOfTrack => false,
                _ => true,
            },
            TrackEventKind::Midi {
                channel, message, ..
            } => {
                if let Some(existing_channel) = likely_channel {
                    println!(
                        "[Warning] Track contains multiple channels: {} {}",
                        existing_channel, channel
                    )
                }
                likely_channel = Some(channel);
                match message {
                    MidiMessage::NoteOff { .. } => false,
                    MidiMessage::NoteOn { .. } => false,
                    _ => true,
                }
            }
            _ => false,
        })
        .collect();
    (events, likely_channel)
}

fn channel(channel: impl Into<u4>) -> TrackEvent<'static> {
    TrackEvent {
        delta: 0.into(),
        kind: TrackEventKind::Meta(MetaMessage::MidiChannel(channel.into())),
    }
}

fn program_change(channel: impl Into<u4>, program: impl Into<u7>) -> TrackEvent<'static> {
    TrackEvent {
        delta: 0.into(),
        kind: TrackEventKind::Midi {
            channel: channel.into(),
            message: MidiMessage::ProgramChange {
                program: program.into(),
            },
        },
    }
}

fn end_of_track() -> TrackEvent<'static> {
    TrackEvent {
        delta: 19200.into(),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    }
}

fn note(
    delta: impl Into<u28>,
    channel: impl Into<u4>,
    key: impl Into<u7>,
    on: bool,
) -> TrackEvent<'static> {
    TrackEvent {
        delta: delta.into(),
        kind: TrackEventKind::Midi {
            channel: channel.into(),
            message: match on {
                true => MidiMessage::NoteOn {
                    key: key.into(),
                    vel: 63.into(),
                },
                false => MidiMessage::NoteOff {
                    key: key.into(),
                    vel: 63.into(),
                },
            },
        },
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let raw = std::fs::read(MEGALOVANIA)?;
    let (mut header, tracks) = midly::parse(&raw)?;
    let tracks = tracks.collect_tracks()?;

    let midi_info = MidiInfo::new(header, &tracks);

    let mut out_tracks = vec![];
    for track in tracks.iter() {
        // [&tracks[0], &tracks[1], &tracks[3]].iter() {
        let quantization = NoteDuration::Eighth;

        let (mut meta_messages, likely_channel) = extract_meta_messages(track);

        let notes = Note::from_events(&track)
            .iter()
            .map(|x| MarkovNote::from(x.quantize(midi_info, quantization)))
            .collect::<Vec<MarkovNote>>();

        let mut chain = Chain::of_order(8);
        let notes: Box<dyn Iterator<Item = MarkovNote>> = if notes.is_empty() {
            Box::new(notes.iter().cloned())
        } else {
            chain.feed(&notes);
            Box::new(chain.iter().flatten().take(256))
        };

        let notes = notes
            .enumerate()
            .map(|(i, x)| {
                let note = QuantizedNote {
                    key: x.key.into(),
                    vel: 63.into(),
                    channel: likely_channel.unwrap_or(0.into()),
                    quantization,
                    start: i as u32,
                    length: x.length,
                };
                Note::from_quantized(midi_info, note)
            })
            .collect::<Vec<Note>>();

        let mut out_track = vec![];
        out_track.append(&mut meta_messages);
        out_track.append(&mut Note::to_events(notes));
        out_track.push(end_of_track());
        // let out_track = track.clone();
        out_tracks.push(out_track);
    }

    header.format = midly::Format::Parallel;
    let mut outfile = std::fs::File::create("out.mid")?;
    midly::write_std(&header, &out_tracks, &mut outfile).unwrap();

    debug_print_midi(MEGALOVANIA)?;
    println!("----");
    debug_print_midi("out.mid")?;

    Ok(())
}
