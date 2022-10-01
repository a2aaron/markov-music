use std::collections::{btree_map, hash_map, BTreeMap, HashMap};

use derive_more::{Add, AddAssign, From, Sub, SubAssign};
use midly::{
    num::{u24, u28, u4, u7},
    Header, MetaMessage, MidiMessage, Timing, TrackEvent, TrackEventKind,
};

#[derive(Debug, Clone, Copy)]
pub struct MidiInfo {
    timing: Timing,
    tempo: Option<u24>,
    time_sig: Option<TimeSignature>,
}

impl MidiInfo {
    pub fn new<'a>(
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
                midly::MetaMessage::Tempo(new_tempo) => {
                    if let Some(tempo) = tempo {
                        println!(
                            "[Warning] Tempo already set! Old: {:?}, new: {:?}",
                            tempo, new_tempo
                        );
                    }
                    tempo = Some(new_tempo)
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

    pub fn tempo_event(&self) -> Option<TrackEvent<'static>> {
        if let Some(tempo) = self.tempo {
            Some(TrackEvent {
                delta: 0.into(),
                kind: TrackEventKind::Meta(MetaMessage::Tempo(tempo)),
            })
        } else {
            None
        }
    }

    pub fn time_signature_event(&self) -> Option<TrackEvent<'static>> {
        if let Some(time_sig) = self.time_sig {
            let msg = time_sig.as_meta_message();
            Some(TrackEvent {
                delta: 0.into(),
                kind: TrackEventKind::Meta(msg),
            })
        } else {
            None
        }
    }

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

    fn to_ticks(&self, beats: f32) -> Ticks {
        match self.timing {
            Timing::Metrical(ticks_per_beat) => {
                let ticks_per_beat = ticks_per_beat.as_int() as f32;
                Ticks::from(ticks_per_beat * beats)
            }
            Timing::Timecode(fps, ticks_per_frame) => {
                let fps = fps.as_f32();
                let ticks_per_frame = ticks_per_frame as f32;
                if let Some(seconds_per_beat) = self.seconds_per_beat() {
                    let seconds = seconds_per_beat * beats;
                    let ticks = seconds * fps * ticks_per_frame;
                    Ticks::from(ticks)
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
        self.tempo
            .map(|tempo| (tempo.as_int() as f32) / 1_000_000.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Note {
    key: u7,
    vel: u7,
    start: Ticks,
    length: Ticks,
    channel: u4,
}

impl Note {
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
pub struct QuantizedNote {
    pub key: u7,
    pub vel: u7,
    pub channel: u4,
    pub quantization: NoteDuration,
    pub start: u32,
    pub length: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MarkovNote {
    pub pitches: MarkovNotePitches,
    pub length: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarkovNotePitches {
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

impl MarkovNote {
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

#[derive(Debug)]
pub struct StepSequencer {
    notes: BTreeMap<u32, Vec<u7>>,
    quantization: NoteDuration,
}
impl StepSequencer {
    pub fn from_notes(
        notes: &[Note],
        midi_info: MidiInfo,
        quantization: NoteDuration,
    ) -> StepSequencer {
        let notes: Vec<_> = notes
            .iter()
            .map(|note| note.quantize(midi_info, quantization))
            .collect();

        let mut map = BTreeMap::new();
        for note in notes {
            for i in note.start..note.start + note.length {
                match map.entry(i) {
                    btree_map::Entry::Vacant(entry) => {
                        entry.insert(vec![note.key]);
                    }
                    btree_map::Entry::Occupied(mut entry) => entry.get_mut().push(note.key),
                }
            }
        }

        StepSequencer {
            notes: map,
            quantization,
        }
    }

    pub fn get_notes(&self, step: u32) -> Vec<u7> {
        self.notes.get(&step).cloned().unwrap_or(vec![])
    }

    pub fn to_notes(&self) -> Vec<QuantizedNote> {
        let mut quantized_notes = vec![];
        let mut active_notes = HashMap::<u7, u32>::new();
        for (&step, notes) in self.notes.iter() {
            for &key in notes {
                if let hash_map::Entry::Vacant(entry) = active_notes.entry(key) {
                    entry.insert(step);
                } else {
                    for (_, start) in
                        active_notes.drain_filter(|&active_note, _| key != active_note)
                    {
                        let note = QuantizedNote {
                            key,
                            vel: u7::max_value(),
                            channel: u4::new(0),
                            quantization: self.quantization,
                            start,
                            length: step - start,
                        };
                        quantized_notes.push(note);
                    }
                }
            }
        }
        quantized_notes
    }
}

#[derive(Debug, Clone, Copy)]
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
    pub fn from_beats(&self, beats: usize) -> f32 {
        beats as f32 * self.beat_factor()
    }

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

#[derive(Debug, Clone, Copy)]
struct TimeSignature {
    numerator: u8,
    denominator: u8,
    midi_clocks_per_click: u8,
    thirty_second_notes_per_quarter_note: u8,
}

impl TimeSignature {
    fn as_meta_message(&self) -> MetaMessage<'static> {
        MetaMessage::TimeSignature(
            self.numerator,
            self.denominator,
            self.midi_clocks_per_click,
            self.thirty_second_notes_per_quarter_note,
        )
    }
}

#[derive(
    From, Add, AddAssign, Sub, SubAssign, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug,
)]
struct Ticks(u32);
impl Ticks {
    fn quantize_to_ticks(&self, midi_info: MidiInfo, duration: NoteDuration) -> Ticks {
        let beats = midi_info.to_beats(*self);
        let factor = duration.beat_factor();
        let beats = (beats * factor).trunc() / factor;
        midi_info.to_ticks(beats)
    }

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
