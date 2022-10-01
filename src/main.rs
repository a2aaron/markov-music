#![feature(hash_drain_filter)]
#![feature(let_chains)]

use std::error::Error;

mod midi;
mod notes;

const MEGALOVANIA: &str = "Undertale_-_Megalovania.mid";

fn main() -> Result<(), Box<dyn Error>> {
    midi::generate_markov(MEGALOVANIA, 1, 32)
}
