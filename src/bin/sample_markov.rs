use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
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
                markov_music::samples::markov_mp3(
                    format!("inputs/{}.mp3", name),
                    format!(
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
