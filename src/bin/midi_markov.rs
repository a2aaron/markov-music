use std::error::Error;

use markov_music::{midi, notes::NoteDuration};

const MEGALOVANIA: &str = "megalovania.mid";

fn main() -> Result<(), Box<dyn Error>> {
    midi::generate_markov(
        format!("inputs/{}", MEGALOVANIA),
        "outputs/out.mid",
        &[0, 1, 2, 3],
        NoteDuration::Eighth,
        1,
        32,
    )
}
