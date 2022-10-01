#![feature(hash_drain_filter)]
#![feature(let_chains)]

use std::error::Error;

use notes::NoteDuration;

mod midi;
mod notes;

const MEGALOVANIA: &str = "Undertale_-_Megalovania.mid";

fn main() -> Result<(), Box<dyn Error>> {
    midi::generate_markov(
        MEGALOVANIA,
        "out.mid",
        &[0, 1, 2, 3],
        NoteDuration::Eighth,
        1,
        32,
    )
}
