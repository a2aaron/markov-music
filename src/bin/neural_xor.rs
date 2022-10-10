use markov_music::neural;
use pixel_canvas::{Canvas, Color};

fn main() {
    let network = neural::FullNetwork::with_size(2, &[2, 1]);
    println!("Initial Network: {:#?}", network);

    let canvas = Canvas::new(256, 256).state((network, 0)).show_ms(true);
    canvas.render(|(network, i), image| {
        if *i % 100_000 == 0 {
            *network = neural::FullNetwork::with_size(2, &[2, 1]);
        }
        for _ in 0..1_000 {
            *i += 1;
            let oracle = [
                (vec![0.0, 0.0], vec![0.0]),
                (vec![0.0, 1.0], vec![1.0]),
                (vec![1.0, 0.0], vec![1.0]),
                (vec![1.0, 1.0], vec![0.0]),
            ];
            let (i, _) = oracle.iter().enumerate().fold(
                (0, 0.0),
                |(max_i, max_cost), (i, (inputs, outputs))| {
                    let cost = network.cost(inputs, outputs);
                    if max_cost > cost {
                        (max_i, max_cost)
                    } else {
                        (i, cost)
                    }
                },
            );
            let (inputs, outputs) = &oracle[i];
            network.backprop(1.0, &inputs, &outputs);
        }
        if *i % 100_000 == 0 {
            println!("i = {}", i);
            println!(
                "Costs: {:>6.04}  {:>6.04}  {:>6.04}  {:>6.04}",
                network.cost(&[0.0, 0.0], &[0.0]),
                network.cost(&[0.0, 1.0], &[1.0]),
                network.cost(&[1.0, 0.0], &[1.0]),
                network.cost(&[1.0, 1.0], &[0.0]),
            );
            println!(
                "Output: {:>6.04}  {:>6.04}  {:>6.04}  {:>6.04}",
                network.compute(&[0.0, 0.0])[0],
                network.compute(&[0.0, 1.0])[0],
                network.compute(&[1.0, 0.0])[0],
                network.compute(&[1.0, 1.0])[0],
            );
        }

        image.fill(Color::BLACK);
        let width = image.width() as usize;
        let height = image.height() as usize;
        for (y, row) in image.chunks_mut(width).enumerate() {
            for (x, pixel) in row.iter_mut().enumerate() {
                let x_float = ((x as f32 / width as f32) - 0.5) * 3.0;
                let y_float = ((y as f32 / height as f32) - 0.5) * 3.0;
                let output = network.compute(&[x_float, y_float]);
                *pixel = Color {
                    r: (output[0].clamp(0.0, 1.0) * 250.0) as u8, // if output[0] > 0.5 { 255 } else { 0 },
                    g: if (x_float - 1.0).abs() < 0.01 || (y_float - 1.0).abs() < 0.01 {
                        255
                    } else {
                        0
                    },
                    b: if (x_float).abs() < 0.01 || (y_float).abs() < 0.01 {
                        255
                    } else {
                        0
                    },
                }
            }
        }
    });
}
