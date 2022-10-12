use markov_music::neural2::NeuralNet;
use pixel_canvas::{Canvas, Color};
use rand::{seq::SliceRandom, thread_rng};

fn main() {
    let network = NeuralNet::new();

    let canvas = Canvas::new(256, 256).state((network, 0usize)).show_ms(true);
    canvas.render(|(network, i), image| {
        let oracle = [
            ([0.0, 0.0], 0.0),
            ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([1.0, 1.0], 0.0),
        ];

        *i += 1;
        for _ in 0..100 {
            let (inputs, outputs) = oracle.choose(&mut thread_rng()).unwrap();
            network.train(*inputs, *outputs);
        }

        let mut total_loss = 0.0;
        for (input, output) in oracle {
            let (_, loss) = network.compute_with_loss(input, output);
            total_loss += loss;
        }

        if *i % 10 == 0 {
            println!("total_loss: {}", total_loss / 4.0);
        }

        if total_loss < 0.00001 {
            println!("Reset!");
            network.reset();
        }

        image.fill(Color::BLACK);
        let width = image.width() as usize;
        let height = image.height() as usize;
        for (y, row) in image.chunks_mut(width).enumerate() {
            for (x, pixel) in row.iter_mut().enumerate() {
                let x_float = ((x as f32 / width as f32) - 0.5) * 3.0;
                let y_float = ((y as f32 / height as f32) - 0.5) * 3.0;
                let output = network.compute([x_float, y_float]);
                *pixel = Color {
                    r: {
                        let amt = (4.0 * output - 2.0).tanh() / 2.0 + 0.5;
                        (amt * 250.0) as u8
                    },
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
