use markov::{Chain, Chainable};

fn markov<T: Chainable>(samples: &[T], order: usize) -> Chain<T> {
    let mut chain = Chain::of_order(order);
    chain.feed(samples);
    chain
}

fn generate<T: Chainable>(samples: &[T], order: usize, length: usize) -> Vec<T> {
    let sample_len = samples.len();
    println!("Training Markov chain... ({} samples)", sample_len);
    let chain = markov(&samples[..sample_len], order);

    println!("Generating Markov chain... ({} samples)", length);
    chain.iter().flatten().take(length).collect::<Vec<_>>()
}

fn inv_lerp(a: f32, b: f32, x: f32) -> f32 {
    (x - a) / (b - a)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn expand_to_i16(x: usize, max_range: usize) -> i16 {
    let x = x as f32;
    let i16_max = i16::MAX as f32;
    let i16_min = i16::MIN as f32;

    let t = inv_lerp(0.0, max_range as f32, x);
    lerp(i16_min, i16_max, t) as i16
}

fn constrain_to_range(x: i16, max_range: usize) -> usize {
    let x = x as f32;
    let i16_max = i16::MAX as f32;
    let i16_min = i16::MIN as f32;

    let t = inv_lerp(i16_min, i16_max, x);
    lerp(0.0, max_range as f32, t) as usize
}

pub fn markov_samples(samples: &[i16], order: usize, max_range: usize, length: usize) -> Vec<i16> {
    let samples = samples
        .iter()
        .map(|x| constrain_to_range(*x, max_range))
        .collect::<Vec<_>>();

    let samples = generate(&samples, order, length);
    let samples = samples
        .iter()
        .map(|x| expand_to_i16(*x, max_range))
        .collect();
    samples
}
