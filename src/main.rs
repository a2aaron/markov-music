pub mod notes;

use std::error::Error;

use notes::{MidiInfo, Note, NoteDuration};

const MEGALOVANIA: &str = "Undertale_-_Megalovania.mid";

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
        let notes = notes.iter().map(|note| {
            let note = note.quantize(midi_info, NoteDuration::Eighth);
            Note::from_quantized(midi_info, note)
        });
        let out_track = Note::to_events(notes);

        out_tracks.push(out_track);
    }

    let mut outfile = std::fs::File::create("out.mid")?;
    midly::write_std(&header, &out_tracks, &mut outfile)?;

    Ok(())
}
