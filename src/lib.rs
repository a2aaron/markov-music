#![feature(hash_drain_filter)]
#![feature(let_chains)]
#![feature(iter_intersperse)]

pub mod chaos;
pub mod distribution;
pub mod markov;
pub mod midi;
pub mod neural2;
pub mod notes;
pub mod quantize;
pub mod wavelet;

fn split_into_windows<T>(data: &[T], window_size: usize) -> impl Iterator<Item = &[T]> {
    (0..(data.len() - window_size)).map(move |i| &data[i..i + window_size])
}
