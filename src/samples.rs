use markov::{Chain, Chainable};

use crate::quantize::Quantizable;

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

pub fn markov_samples(
    samples: &[i16],
    order: usize,
    quantization_level: usize,
    length: usize,
) -> Vec<i16> {
    let min = i16::MIN;
    let max = i16::MAX;

    let samples = samples
        .iter()
        .map(|x| Quantizable::quantize(*x, min, max, quantization_level))
        .collect::<Vec<_>>();

    let samples = generate(&samples, order, length);
    let samples = samples
        .iter()
        .map(|x| Quantizable::unquantize(*x, min, max, quantization_level))
        .collect();
    samples
}
