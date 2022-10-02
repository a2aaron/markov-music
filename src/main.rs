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

    for name in ["miserable", "ghost", "celeste"] {
        for order in 1..=8 {
            for power in [8, 10, 12, 14, 16] {
                let range = 2usize.pow(power);
                let order = order;
                let states = power as usize * order;
                if states > 64 {
                    continue;
                }
                println!("Order {}", order);
                samples::markov_mp3(
                    &format!("{}.mp3", name),
                    &format!(
                        "outputs/{}_order_{}_range_{}_({}_states).wav",
                        name, order, power, states
                    ),
                    order,
                    range,
                )?;
            }
        }
    }

    Ok(())
}
