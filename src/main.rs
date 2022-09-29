#![feature(hash_drain_filter)]

pub mod notes;

use std::error::Error;

use midly::{
    num::{u28, u4, u7},
    MetaMessage, MidiMessage, TrackEvent, TrackEventKind,
};
use notes::{MidiInfo, Note, NoteDuration, StepSequencer};

// const MEGALOVANIA: &str = "Undertale_-_Megalovania.mid";
const MEGALOVANIA: &str = "Megalovania Loop.mid";
// const MEGALOVANIA: &str = "test.mid";

fn debug_print_midi(path: &str) {
    let raw = std::fs::read(path).unwrap();
    let (header, tracks) = midly::parse(&raw).unwrap();
    println!("{:?}", header);
    for track in tracks {
        for event in track.unwrap() {
            println!("{:?}", event);
        }
    }
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

    let mut out_tracks = vec![tracks[0].clone()];
    for track in tracks.iter() {
        let notes = Note::from_events(&track);
        let notes = notes
            .iter()
            .map(|note| note.quantize(midi_info, NoteDuration::Quarter))
            .collect::<Vec<_>>();
        // let step_sequence = StepSequencer::from_notes(&notes, midi_info, NoteDuration::Sixteenth);
        // let notes = step_sequence.to_notes();
        let notes = notes
            .iter()
            .map(|note| Note::from_quantized(midi_info, *note));
        let mut out_track = vec![];
        out_track.append(&mut Note::to_events(notes));
        out_track.push(note(100, 0, 40, true));
        out_track.push(note(100, 0, 40, false));
        out_track.push(end_of_track());
        // let out_track = track.clone();
        out_tracks.push(out_track);
    }

    header.format = midly::Format::Parallel;
    let mut outfile = std::fs::File::create("out.mid")?;
    midly::write_std(&header, &out_tracks, &mut outfile).unwrap();

    debug_print_midi(MEGALOVANIA);
    println!("----");
    debug_print_midi("out.mid");

    Ok(())
}
