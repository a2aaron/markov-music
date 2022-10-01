#![feature(hash_drain_filter)]
#![feature(let_chains)]

use std::error::Error;

use notes::NoteDuration;

mod midi;
mod notes;
mod samples;

const MEGALOVANIA: &str = "Undertale_-_Megalovania.mid";

fn main() -> Result<(), Box<dyn Error>> {
    // midi::generate_markov(
    //     MEGALOVANIA,
    //     "out.mid",
    //     &[0, 1, 2, 3],
    //     NoteDuration::Eighth,
    //     1,
    //     32,
    // )

    for order in 1..=7 {
        let order = order;
        println!("Order {}", order);
        samples::markov_mp3("ghost_low.mp3", &format!("ghost_low_{}.wav", order), order)?;
    }

    Ok(())
}
