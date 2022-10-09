use std::{collections::BTreeMap, error::Error};

use midly::{num::u4, MetaMessage, MidiMessage, TrackEvent, TrackEventKind};

use crate::markov::Chain;
use crate::notes::{MidiInfo, Note, NoteDuration};

use crate::notes::MarkovNote;

/// Debug print a midi's messages.
#[allow(dead_code)]
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

/// Extracts all the meta messages from a track, excluding the EndOfTrack message. This will also
/// include any MidiMessages that are typically used as meta messages, such as Tempo and Program Change
/// messages. The function will also return a "likely channel", assuming the track has any MidiMessages.
/// (The likely channel is set to the channel of the last MidiMessage, with warnings occuring if a
/// track has multiple channels used within it.)
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

pub fn generate_markov<'a>(
    midi_info: MidiInfo,
    track: &[TrackEvent<'a>],
    quantization: NoteDuration,
    order: usize,
    measures: usize,
) -> Vec<TrackEvent<'a>> {
    let (mut meta_messages, likely_channel) = extract_meta_messages(&track);

    let markov_notes = {
        // Quantize notes and place them into a map whose keys are the start times.
        let mut map = BTreeMap::new();
        Note::from_events(&track)
            .iter()
            .map(|x| {
                let note = x.quantize(midi_info, quantization);
                (note.start, note)
            })
            .for_each(|(start, note)| map.entry(start).or_insert_with(Vec::new).push(note));

        // Then extract the notes from the map, grouped by the start time key, and convert them
        // to MarkovNotes. This ultimately has the effect of producing a list of MarkovNotes, ordered
        // by start time.
        map.iter()
            .map(|(_, notes)| {
                if notes.len() > 3 {
                    println!(
                        "[Warning] Dropping extra notes (total: {}, can only accomodate 3)",
                        notes.len()
                    );
                }
                let length = notes[0].length;
                let a = notes[0].key;
                let b = notes.get(1).cloned().map(|note| note.key);
                let c = notes.get(2).cloned().map(|note| note.key);
                MarkovNote::new(a, b, c, length)
            })
            .collect::<Vec<_>>()
    };

    // Do chain generation, if the chain has any notes.
    let markov_notes = if let Ok(chain) = Chain::new(&markov_notes, order) {
        let mut length = 0;
        chain
            .iter_from_start()
            .take_while(move |note| {
                length += note.length;
                length < quantization.from_beats(measures * 4) as u32
            })
            .collect()
    } else {
        markov_notes
    };

    // Finally, turn the MarkovNotes back into regular Notes
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

    // Write out the events for the track, first sticking the meta messages in, then the notes events
    // and then finally an end of track event.
    let mut out_track = vec![];
    out_track.append(&mut meta_messages);
    out_track.append(&mut Note::to_events(notes));
    out_track.push(end_of_track());
    out_track
}
