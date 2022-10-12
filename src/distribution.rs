use rand::Rng;

use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::split_into_windows;

pub struct Distribution {
    // pub samples: Vec<(Vec<f32>, f32)>
    pub samples: Vec<f32>,
    pub order: usize,
}

impl Distribution {
    pub fn new(in_samples: &[f32], order: usize) -> Distribution {
        // let samples = split_into_windows(in_samples, order + 1)
        //     .filter_map(|window| {
        //         if window.iter().all(|x| (*x - window[0]).abs() < 0.01) {
        //             return None;
        //         }
        //         let main_window = window[0..order].to_vec();
        //         let next = window[order];
        //         Some((main_window, next))
        //     })
        //     .collect::<Vec<_>>();

        // println!("{} / {}", samples.len(), in_samples.len());
        if in_samples.len() < order {
            panic!("Order must be less than in samples length!");
        }

        Distribution {
            samples: in_samples.to_vec(),
            order,
        }
    }

    pub fn next_sample(&self, other_window: &[f32]) -> (f32, f32) {
        if other_window.len() != self.order {
            panic!("Expected other window to have length equal to self.order!");
        }

        let errors = (0..self.samples.len() - (self.order + 2))
            .into_par_iter()
            .filter_map(|i| -> Option<(f32, usize)> {
                let window = &self.samples[i..i + self.order];
                let error = mean_squared_error(window, other_window);
                if error < 0.01 {
                    Some((f32::exp(-error), i))
                } else {
                    None
                }
            });

        // let cross_corr = cross_correlation(&self.samples, other_window);

        let sum: f32 = errors.clone().map(|(err, _)| err).sum();
        let probabilities = errors.map(|(err, i)| (err / sum, i)).collect::<Vec<_>>();

        let err_sum = probabilities.iter().map(|(err, _)| err).sum();
        let val = rand::thread_rng().gen_range(0.0..err_sum);

        let mut sum = 0.0;

        for (error, i) in probabilities.iter() {
            if sum >= val {
                return (*error, self.samples[i + self.order]);
            }
            sum += error;
        }
        (999.0, self.samples[0])
    }
}

pub fn normalize(a: &[f32]) -> (Vec<f32>, f32, f32) {
    let min = a.iter().cloned().reduce(|acc, x| acc.min(x)).unwrap();
    let max = a.iter().cloned().reduce(|acc, x| acc.max(x)).unwrap();
    let normalized = a
        .iter()
        .map(|x| (x - min) / (max - min))
        .collect::<Vec<_>>();
    (normalized, min, max)
}

pub fn unnormalize(a: &[f32], min: f32, max: f32) -> Vec<f32> {
    a.iter().map(|x| x * (max - min) + min).collect::<Vec<_>>()
}

fn mean_squared_error(a: &[f32], b: &[f32]) -> f32 {
    a.par_iter()
        .zip_eq(b.into_par_iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / a.len() as f32
}
