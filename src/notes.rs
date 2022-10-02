use std::collections::{hash_map, HashMap};

use clap::ValueEnum;
use derive_more::{Add, AddAssign, From, Sub, SubAssign};
use midly::{
    num::{u24, u28, u4, u7},
    Header, MidiMessage, Timing, TrackEvent, TrackEventKind,
};

#[derive(Debug, Clone, Copy)]
/// A struct which records the timing information about a midi (specifically, the Tempo and Timing
/// of the midi).
pub struct MidiInfo {
    timing: Timing,
    tempo: u24,
}

impl MidiInfo {
    /// Create a new MidiInfo from the given header and set of tracks. This will attempt to find the
    /// tempo midi messages. If there are multiple, warnings are printed and only the first one is used.
    /// If no tempo is specified, the tempo defaults to 120 BPM (equal to 500000 ticks per beat)
    pub fn new<'a>(
        header: Header,
        tracks: impl IntoIterator<Item = impl IntoIterator<Item = &'a TrackEvent<'a>>>,
    ) -> MidiInfo {
        let meta_events = tracks.into_iter().flatten().filter_map(|x| match x.kind {
            TrackEventKind::Meta(meta_msg) => Some(meta_msg),
            _ => None,
        });

        let mut tempo = None;
        for event in meta_events {
            match event {
                midly::MetaMessage::Tempo(new_tempo) => {
                    if let Some(tempo) = tempo {
                        println!(
                            "[Warning] Tempo already set! Old: {:?}, new: {:?}",
                            tempo, new_tempo
                        );
                    } else {
                        tempo = Some(new_tempo)
                    }
                }
                _ => (),
            }
        }

        if tempo.is_none() {
            println!(
                "[Warning] No tempo specified, defaulting to 120 BPM (500,000 ticks per beat)"
            );
        }

        MidiInfo {
            timing: header.timing,
            tempo: tempo.unwrap_or(u24::from(500000)),
        }
    }

    /// Convert midi Ticks into beats, using the specified Timing recorded by the MidiInfo. Note that
    /// if the Midi has Timecode timing, but no Tempo value specified, then this function will panic,
    /// since it's not possible to determine the right number of beats without it.
    fn to_beats(&self, ticks: Ticks) -> f32 {
        let ticks: f32 = ticks.into();
        match self.timing {
            Timing::Metrical(ticks_per_beat) => {
                let ticks_per_beat = ticks_per_beat.as_int() as f32;
                ticks / ticks_per_beat
            }
            Timing::Timecode(fps, ticks_per_frame) => {
                let fps = fps.as_f32();
                let ticks_per_frame = ticks_per_frame as f32;
                let seconds = ticks / fps / ticks_per_frame;
                seconds / self.seconds_per_beat()
            }
        }
    }

    fn to_ticks(&self, beats: f32) -> Ticks {
        match self.timing {
            Timing::Metrical(ticks_per_beat) => {
                let ticks_per_beat = ticks_per_beat.as_int() as f32;
                Ticks::from(ticks_per_beat * beats)
            }
            Timing::Timecode(fps, ticks_per_frame) => {
                let fps = fps.as_f32();
                let ticks_per_frame = ticks_per_frame as f32;
                let seconds = self.seconds_per_beat() * beats;
                let ticks = seconds * fps * ticks_per_frame;
                Ticks::from(ticks)
            }
        }
    }

    /// Get the number of seconds per beat that this Midi file specifies. This is based off of the
    /// given tempo value.
    fn seconds_per_beat(&self) -> f32 {
        // Note: tempo is in microseconds per beat (so a value of 1,000,000 equals 1 second per beat)
        // To convert to BPM, the conversion is 60 / (tempo / 1,000,000)
        // For example, the Megalovania MIDI has a tempo value of 260,870 microseconds per beat.
        // This equals 260,870 / 1,000,000 = 0.26087 seconds per beat
        // or 60 / 0.26087 = 229.99996 -> 230 beats per minute.
        self.tempo.as_int() as f32 / 1_000_000.0
    }
}

#[derive(Debug, Clone, Copy)]
/// A midi note, represented with a start time and length. Note that midi does not really have a notion
/// of "notes"--instead there are just NoteOn and NoteOff events.
pub struct Note {
    key: u7,
    vel: u7,
    start: Ticks,
    length: Ticks,
    channel: u4,
}

impl Note {
    /// Attempt to turn a set of TrackEvents into a set of Notes. A note is created whenever a NoteOn
    /// event of a specific key is later followed by a NoteOff event of the same key. The Note will
    /// be considered to start at the NoteOn event and end at the NoteOff event. The velocity
    /// will be equal to the NoteOn event's velocity. NoteOff events without a corresponding NoteOn
    /// event are dropped, as are NoteOn events that have no corresponding NoteOff event.
    pub fn from_events(events: &[TrackEvent]) -> Vec<Note> {
        let mut notes = vec![];
        let mut ticks = Ticks::from(0);

        let mut active_notes = HashMap::<u7, Vec<(Ticks, u7)>>::new();

        for event in events {
            ticks = ticks + event.delta.into();
            match event.kind {
                TrackEventKind::Midi { message, channel } => match message {
                    midly::MidiMessage::NoteOn { key, vel } => {
                        active_notes
                            .entry(key)
                            .and_modify(|vec| vec.push((ticks, vel)))
                            .or_insert(vec![(ticks, vel)]);
                    }
                    midly::MidiMessage::NoteOff { key, vel: _ } => {
                        if let hash_map::Entry::Occupied(mut entry) = active_notes.entry(key) {
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

    pub fn to_events(notes: impl IntoIterator<Item = Note>) -> Vec<TrackEvent<'static>> {
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
                delta: delta.into(),
                kind: *event,
            });
        }
        track_events
    }

    /// Quantize a note to the given qunatization level. Both the start time and length of the note
    /// are quantized.
    pub fn quantize(&self, midi_info: MidiInfo, quantization: NoteDuration) -> QuantizedNote {
        QuantizedNote {
            key: self.key,
            vel: self.vel,
            channel: self.channel,
            quantization,
            start: self
                .start
                .to_duration_units(midi_info, quantization)
                .floor() as u32,
            length: self
                .length
                .to_duration_units(midi_info, quantization)
                .ceil() as u32,
        }
    }

    /// Convert a QuantizedNote to a Note using the given midi_info.
    pub fn from_quantized(midi_info: MidiInfo, note: QuantizedNote) -> Self {
        Self {
            key: note.key,
            vel: note.vel,
            channel: note.channel,
            start: midi_info.to_ticks(note.quantization.to_beats(note.start)),
            length: midi_info.to_ticks(note.quantization.to_beats(note.length)),
        }
    }
}

#[derive(Debug, Clone, Copy)]
/// A QuantizedNote is similar to a Note, but it's start and length values are quantized to some
/// NoteDuration, rather than ticks.
pub struct QuantizedNote {
    pub key: u7,
    pub vel: u7,
    pub channel: u4,
    pub quantization: NoteDuration,
    pub start: u32,
    pub length: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// A MarkovNote is a note which can be fed to the markov chain. A MarkovNote denotes up to three
/// simultaneous pitches, all of which have some equal, specified length. The length is assumed to
/// be in some NoteDuration.
pub struct MarkovNote {
    pitches: MarkovNotePitches,
    pub length: u32,
}

impl MarkovNote {
    pub fn new(a: u7, b: Option<u7>, c: Option<u7>, length: u32) -> MarkovNote {
        let pitches = match (b, c) {
            (None, None) => MarkovNotePitches::one(a),
            (Some(b), None) => MarkovNotePitches::two(a, b),
            (Some(b), Some(c)) => MarkovNotePitches::three(a, b, c),
            (None, Some(_)) => unreachable!(),
        };
        MarkovNote { pitches, length }
    }

    pub fn to_notes(
        &self,
        start: u32,
        channel: u4,
        vel: u7,
        quantization: NoteDuration,
    ) -> Vec<QuantizedNote> {
        let to_note = |key: u7| -> QuantizedNote {
            QuantizedNote {
                key,
                vel,
                channel,
                quantization,
                start,
                length: self.length,
            }
        };
        match self.pitches {
            MarkovNotePitches::One(a) => vec![to_note(a)],
            MarkovNotePitches::Two(a, b) => vec![to_note(a), to_note(b)],
            MarkovNotePitches::Three(a, b, c) => vec![to_note(a), to_note(b), to_note(c)],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum MarkovNotePitches {
    One(u7),
    Two(u7, u7),
    Three(u7, u7, u7),
}

impl MarkovNotePitches {
    pub fn one(a: u7) -> MarkovNotePitches {
        MarkovNotePitches::One(a)
    }

    pub fn two(a: u7, b: u7) -> MarkovNotePitches {
        let (lower, higher) = if a <= b { (a, b) } else { (b, a) };
        MarkovNotePitches::Two(lower, higher)
    }

    pub fn three(a: u7, b: u7, c: u7) -> MarkovNotePitches {
        let [min, mid, max] = {
            let mut arr = [a, b, c];
            arr.sort();
            arr
        };
        MarkovNotePitches::Three(min, mid, max)
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
/// A unit of time in some note duration.
pub enum NoteDuration {
    Whole,
    Half,
    Quarter,
    Eighth,
    Sixteenth,
    ThirtySecond,
    SixtyFourth,
}

impl NoteDuration {
    /// Convert beats to the NoteDuration unit. For example, 3 beats is equal to the following:
    /// - 0.75 Whole notes
    /// - 1.5 Half notes
    /// - 3 Quater notes
    /// - 6 Eighth notes
    /// - 12 Sixteenth notes
    /// - 24 ThirtySecond notes
    /// - 48 SixtyFourth notes
    pub fn from_beats(&self, beats: usize) -> f32 {
        beats as f32 * self.beat_factor()
    }

    /// Convert from the NoteDuration unit to beats. For example:
    /// 1 Whole note = 4.0 beats
    /// 1 Half note = 2.0 beats
    /// 1 Quater note = 1.0 beats
    /// 1 Eighth note = 0.5 beats
    /// 1 Sixteenth note = 0.25 beats
    /// 1 ThirtySecond note = 0.125 beats
    /// 1 SixtyFourth note = 0.0625 beats
    fn to_beats(&self, num_units: u32) -> f32 {
        num_units as f32 / self.beat_factor()
    }
    // Return the multiplier for converting from a beat to the given NoteDuration
    fn beat_factor(&self) -> f32 {
        match self {
            NoteDuration::Whole => 0.25,
            NoteDuration::Half => 0.5,
            NoteDuration::Quarter => 1.0,
            NoteDuration::Eighth => 2.0,
            NoteDuration::Sixteenth => 4.0,
            NoteDuration::ThirtySecond => 8.0,
            NoteDuration::SixtyFourth => 16.0,
        }
    }
}

#[derive(
    From, Add, AddAssign, Sub, SubAssign, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug,
)]
/// Wrapper struct for the Midi Tick unit.
struct Ticks(u32);
impl Ticks {
    // Convert from Ticks to NoteDuration units
    fn to_duration_units(&self, midi_info: MidiInfo, duration: NoteDuration) -> f32 {
        let beats = midi_info.to_beats(*self);
        let factor = duration.beat_factor();
        beats * factor
    }
}
impl From<f32> for Ticks {
    fn from(x: f32) -> Self {
        Self(x as u32)
    }
}
impl From<u28> for Ticks {
    fn from(x: u28) -> Self {
        Ticks(x.as_int())
    }
}
impl From<Ticks> for u28 {
    fn from(x: Ticks) -> Self {
        u28::new(x.0)
    }
}
impl From<Ticks> for f32 {
    fn from(x: Ticks) -> f32 {
        x.0 as f32
    }
}
