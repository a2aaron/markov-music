use rand::seq::SliceRandom;

use crate::wavelet::Sample;

fn chaos_select<T>(target: &[T], min_size: f32, max_size: f32) -> (usize, usize) {
    loop {
        let a = rand::random::<usize>() % target.len();
        let b = rand::random::<usize>() % target.len();
        let lower = a.min(b);
        let upper = a.max(b);
        let range = upper - lower;
        let min_range = (min_size * target.len() as f32) as usize;
        let max_range = (max_size * target.len() as f32) as usize;
        if max_range - min_range == 0 {
            return (lower, upper);
        } else if min_range <= range && range <= max_range {
            return (lower, upper);
        }
    }
}

#[allow(dead_code)]
pub fn chaos_copy<T: Copy>(target: &mut [T], layers: &[Vec<T>], min_size: f32, max_size: f32) {
    let mut i = 0;
    loop {
        let random_layer = layers.choose(&mut rand::thread_rng()).unwrap();
        let (lower, upper) = chaos_select(random_layer, min_size, max_size);
        let random_layer = &random_layer[lower..upper];
        for j in 0..random_layer.len() {
            if let Some(element) = target.get_mut(i) {
                *element = random_layer[j];
                i += 1;
            } else {
                return;
            }
        }
    }
}
#[allow(dead_code)]
pub fn chaos_reverse(target: &mut [Sample], chaos_level: usize, min_size: f32, max_size: f32) {
    for _ in 0..chaos_level {
        let (lower, upper) = chaos_select(target, min_size, max_size);
        let mut slice = target[lower..=upper].to_vec();
        slice.reverse();
        target[lower..=upper].clone_from_slice(&slice);
    }
}
#[allow(dead_code)]
pub fn chaos_zero(target: &mut [Sample], chaos_level: usize, min_size: f32, max_size: f32) {
    for _ in 0..chaos_level {
        let (lower, upper) = chaos_select(target, min_size, max_size);
        target[lower..=upper].fill(0.0);
    }
}
