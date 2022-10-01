use std::{collections::BTreeMap, error::Error};

use markov::Chain;
use midly::{num::u4, MetaMessage, MidiMessage, TrackEvent, TrackEventKind};

use crate::notes::{MidiInfo, Note, NoteDuration};

use crate::notes::{MarkovNote, MarkovNotePitches};

pub fn debug_print_midi(path: &str) -> Result<(), Box<dyn Error>> {
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
                if let Some(existing_channel) = likely_channel && existing_channel != channel {
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

fn end_of_track() -> TrackEvent<'static> {
    TrackEvent {
        delta: 19200.into(),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    }
}

pub fn generate_markov(path: &str, order: usize, measures: usize) -> Result<(), Box<dyn Error>> {
    let raw = std::fs::read(path)?;
    let (mut header, tracks) = midly::parse(&raw)?;
    let tracks = tracks.collect_tracks()?;

    let midi_info = MidiInfo::new(header, &tracks);

    let mut out_tracks = vec![];
    for track in [&tracks[0], &tracks[1], &tracks[2], &tracks[3]].iter() {
        let quantization = NoteDuration::Eighth;

        let (mut meta_messages, likely_channel) = extract_meta_messages(track);

        let markov_notes = {
            let mut map = BTreeMap::new();
            Note::from_events(&track)
                .iter()
                .map(|x| {
                    let note = x.quantize(midi_info, quantization);
                    (note.start, note)
                })
                .for_each(|(start, note)| map.entry(start).or_insert_with(Vec::new).push(note));

            map.iter()
                .map(|(_, notes)| {
                    if notes.len() > 3 {
                        println!("[Warning] Dropping extra notes {}", notes.len());
                    }
                    let a = notes[0];
                    let b = notes.get(1).cloned();
                    let c = notes.get(2).cloned();
                    let length = a.length;
                    let pitches = match (b, c) {
                        (None, None) => MarkovNotePitches::one(a.key),
                        (Some(b), None) => MarkovNotePitches::two(a.key, b.key),
                        (Some(b), Some(c)) => MarkovNotePitches::three(a.key, b.key, c.key),
                        (None, Some(_)) => unreachable!(),
                    };
                    MarkovNote { pitches, length }
                })
                .collect::<Vec<_>>()
        };

        let mut chain = Chain::of_order(order);
        let markov_notes: Box<dyn Iterator<Item = MarkovNote>> = if markov_notes.is_empty() {
            Box::new(markov_notes.iter().cloned())
        } else {
            let mut length = 0;
            chain.feed(&markov_notes);
            Box::new(chain.iter().flatten().take_while(move |note| {
                length += note.length;
                length < quantization.from_beats(measures * 4) as u32
            }))
        };

        let mut notes = vec![];
        let mut start = 0;
        for note in markov_notes {
            note.to_notes(
                start,
                likely_channel.unwrap_or(0.into()),
                63.into(),
                quantization,
            )
            .into_iter()
            .for_each(|note| {
                let note = Note::from_quantized(midi_info, note);
                notes.push(note);
            });
            start += note.length;
        }

        let mut out_track = vec![];
        out_track.append(&mut meta_messages);
        out_track.append(&mut Note::to_events(notes));
        out_track.push(end_of_track());
        out_tracks.push(out_track);
    }

    header.format = midly::Format::Parallel;
    let mut outfile = std::fs::File::create("out.mid")?;
    midly::write_std(&header, &out_tracks, &mut outfile)?;
    Ok(())
}
