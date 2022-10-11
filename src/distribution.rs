use rand::Rng;

use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::split_into_windows;

pub struct Distribution {
    pub samples: Vec<(Vec<f32>, f32)>,
    pub order: usize,
}

impl Distribution {
    pub fn new(in_samples: &[f32], order: usize) -> Distribution {
        let samples = split_into_windows(in_samples, order + 1)
            .filter_map(|window| {
                if window.iter().all(|x| (*x - window[0]).abs() < 0.01) {
                    return None;
                }
                let main_window = window[0..order].to_vec();
                let next = window[order];
                Some((main_window, next))
            })
            .collect::<Vec<_>>();

        println!("{} / {}", samples.len(), in_samples.len());

        Distribution { samples, order }
    }

    pub fn next_sample(&self, other_window: &[f32]) -> (f32, f32) {
        // let mut errors = Vec::with_capacity(self.samples.len());
        let errors = self
            .samples
            .par_iter()
            .filter_map(|(window, next)| {
                let error = mean_squared_error(window, other_window);
                if error > 0.01 {
                    None
                } else {
                    Some((-error, next))
                }
            })
            .collect::<Vec<_>>();

        // println!("{} / {}", errors.len(), self.samples.len());

        let sum: f32 = errors.par_iter().map(|(err, _)| f32::exp(*err)).sum();
        let probabilities = errors
            .par_iter()
            .map(|(err, &next)| (f32::exp(*err) / sum, next))
            .collect::<Vec<_>>();

        let err_sum = probabilities.iter().map(|(err, _)| err).sum::<f32>();
        let val = rand::thread_rng().gen_range(0.0..err_sum);

        let mut sum = 0.0;
        for (error, next) in &probabilities {
            if sum >= val {
                return (*error, *next);
            }
            sum += error;
        }
        println!("{} {}", err_sum, sum);
        probabilities[0]
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
