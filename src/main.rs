use std::{
    collections::{hash_map::Entry, HashMap},
    error::Error,
};

use midly::{
    num::{u28, u4, u7},
    Header, MidiMessage, Timing, TrackEvent, TrackEventKind,
};

const MEGALOVANIA: &str = "Undertale_-_Megalovania.mid";

#[derive(Debug)]
struct TimeSignature {
    numerator: u8,
    denominator: u8,
    midi_clocks_per_click: u8,
    thirty_second_notes_per_quarter_note: u8,
}

#[derive(Debug)]
struct MidiInfo {
    timing: Timing,
    tempo: Option<u32>,
    time_sig: Option<TimeSignature>,
}

impl MidiInfo {
    fn new<'a>(
        header: Header,
        tracks: impl IntoIterator<Item = impl IntoIterator<Item = &'a TrackEvent<'a>>>,
    ) -> MidiInfo {
        let meta_events = tracks.into_iter().flatten().filter_map(|x| match x.kind {
            TrackEventKind::Meta(meta_msg) => Some(meta_msg),
            _ => None,
        });

        let mut time_sig = None;
        let mut tempo = None;
        for event in meta_events {
            match event {
                midly::MetaMessage::Tempo(tempo_u24) => {
                    if let Some(tempo) = tempo {
                        println!(
                            "[Warning] Tempo already set! Old: {:?}, new: {:?}",
                            tempo_u24, tempo
                        );
                    }
                    tempo = Some(tempo_u24.as_int())
                }
                midly::MetaMessage::TimeSignature(
                    numerator,
                    denominator,
                    midi_clocks_per_click,
                    thirty_second_notes_per_quarter_note,
                ) => {
                    let new_time_sig = TimeSignature {
                        numerator,
                        denominator,
                        midi_clocks_per_click,
                        thirty_second_notes_per_quarter_note,
                    };
                    if let Some(time_sig) = time_sig {
                        println!(
                            "[Warning] Time signature already set! Old: {:?}, new: {:?}",
                            time_sig, new_time_sig
                        )
                    }

                    time_sig = Some(new_time_sig)
                }
                _ => (),
            }
        }
        MidiInfo {
            timing: header.timing,
            tempo,
            time_sig,
        }
    }

    fn to_beats(&self, ticks: u28) -> f32 {
        let ticks = ticks.as_int() as f32;
        match self.timing {
            Timing::Metrical(ticks_per_beat) => {
                let ticks_per_beat = ticks_per_beat.as_int() as f32;
                ticks / ticks_per_beat
            }
            Timing::Timecode(fps, ticks_per_frame) => {
                let fps = fps.as_f32();
                let ticks_per_frame = ticks_per_frame as f32;
                let seconds = ticks / fps / ticks_per_frame;
                if let Some(seconds_per_beat) = self.seconds_per_beat() {
                    let beats_per_second = 1.0 / seconds_per_beat;
                    seconds * beats_per_second
                } else {
                    // If the tempo wasn't provided, then we do not know how many beats it has been.
                    // Panic in this scenario.
                    panic!("Cannot determine beat offset for a Timecode midi with no tempo information!");
                }
            }
        }
    }

    fn to_ticks(&self, sixteenths: u32) -> u28 {
        let beats = sixteenths as f32 / 4.0;
        match self.timing {
            Timing::Metrical(ticks_per_beat) => {
                let ticks_per_beat = ticks_per_beat.as_int() as f32;
                u28::new((ticks_per_beat * beats) as u32)
            }
            Timing::Timecode(fps, ticks_per_frame) => {
                let fps = fps.as_f32();
                let ticks_per_frame = ticks_per_frame as f32;
                if let Some(seconds_per_beat) = self.seconds_per_beat() {
                    let seconds = seconds_per_beat * beats;
                    let ticks = seconds * fps * ticks_per_frame;
                    u28::new(ticks as u32)
                } else {
                    // If the tempo wasn't provided, then we do not know how many beats it has been.
                    // Panic in this scenario.
                    panic!("Cannot determine beat offset for a Timecode midi with no tempo information!");
                }
            }
        }
    }

    fn seconds_per_beat(&self) -> Option<f32> {
        // Note: tempo is in microseconds per beat (so a value of 1,000,000 equals 1 second per beat)
        // To convert to BPM, the conversion is 60 / (tempo / 1,000,000)
        // For example, the Megalovania MIDI has a tempo value of 260,870 microseconds per beat.
        // This equals 260,870 / 1,000,000 = 0.26087 seconds per beat
        // or 60 / 0.26087 = 229.99996 -> 230 beats per minute.
        self.tempo.map(|tempo| (tempo as f32) / 1_000_000.0)
    }

    fn quantize_to_16ths(&self, ticks: u28) -> u28 {
        let beats = self.to_beats(ticks);
        let sixteenths = beats * 4.0;
        self.to_ticks(sixteenths as u32)
    }
}

struct Note {
    key: u7,
    vel: u7,
    start: u28,
    length: u28,
    channel: u4,
}

impl Note {
    fn from_events(events: &[TrackEvent]) -> Vec<Note> {
        let mut notes = vec![];
        let mut ticks = u28::new(0);

        let mut active_notes = HashMap::<u7, Vec<(u28, u7)>>::new();

        for event in events {
            ticks = ticks + event.delta;
            match event.kind {
                TrackEventKind::Midi { message, channel } => match message {
                    midly::MidiMessage::NoteOn { key, vel } => {
                        active_notes
                            .entry(key)
                            .and_modify(|vec| vec.push((ticks, vel)))
                            .or_insert(vec![(ticks, vel)]);
                    }
                    midly::MidiMessage::NoteOff { key, vel: _ } => {
                        if let Entry::Occupied(mut entry) = active_notes.entry(key) {
                            let vec = entry.get_mut();
                            if let Some((note_on, vel)) = vec.pop() {
                                let length = ticks - note_on;
                                notes.push(Note {
                                    channel,
                                    key,
                                    vel,
                                    start: note_on,
                                    length,
                                });
                            } else {
                                println!("[Warning] Dropping NoteOff event with no corresponding NoteOn event! {:?}", event);
                            }
                        } else {
                            println!("[Warning] Dropping NoteOff event with no corresponding NoteOn event! {:?}", event);
                        }
                    }
                    _ => (),
                },
                _ => (),
            }
        }

        notes
    }

    fn to_events(notes: impl IntoIterator<Item = Note>) -> Vec<TrackEvent<'static>> {
        let mut events = vec![];
        for note in notes {
            let note_on = TrackEventKind::Midi {
                channel: note.channel,
                message: MidiMessage::NoteOn {
                    key: note.key,
                    vel: note.vel,
                },
            };
            let note_off = TrackEventKind::Midi {
                channel: note.channel,
                message: MidiMessage::NoteOff {
                    key: note.key,
                    vel: note.vel,
                },
            };
            events.push((note.start, note_on));
            events.push((note.start + note.length, note_off));
        }
        events.sort_by(|(start, _), (end, _)| start.cmp(end));

        let mut track_events = vec![];
        for (i, (time, event)) in events.iter().enumerate() {
            let delta = if i != 0 {
                let (prev_time, _) = events[i - 1];
                *time - prev_time
            } else {
                *time
            };
            track_events.push(TrackEvent {
                delta,
                kind: *event,
            });
        }
        track_events
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let raw = std::fs::read(MEGALOVANIA)?;
    let (header, tracks) = midly::parse(&raw)?;
    let tracks = tracks.collect_tracks()?;

    let midi_info = MidiInfo::new(header, &tracks);

    let mut out_tracks = vec![tracks[0].clone(), tracks[1].clone()];
    for event in &tracks[0] {
        println!("{:?}", event)
    }
    for event in &tracks[1] {
        println!("{:?}", event)
    }
    for event in tracks[2].iter().take(10) {
        println!("{:?}", event)
    }
    for track in tracks.iter().skip(2) {
        let notes = Note::from_events(&track);
        let out_track = Note::to_events(notes);

        out_tracks.push(out_track);
    }

    let mut outfile = std::fs::File::create("out.mid")?;
    midly::write_std(&header, &out_tracks, &mut outfile)?;

    let raw = std::fs::read("out.mid")?;
    let (header, tracks) = midly::parse(&raw)?;
    let tracks = tracks.collect_tracks()?;
    Ok(())
}
