use std::hash::Hash;
use std::{collections::HashMap, error::Error};

use rand::seq::IteratorRandom;
use rand::{thread_rng, Rng};

/// The definition of all types that can be used in a `Chain`.
pub trait Chainable: Eq + Hash + Clone + std::fmt::Debug {}
impl<T> Chainable for T where T: Eq + Hash + Clone + std::fmt::Debug {}

pub struct Chain<T: Chainable> {
    map: HashMap<Vec<T>, HashMap<T, usize>>,
    order: usize,
    starting_state: Vec<T>,
}

impl<T: Chainable> Chain<T> {
    pub fn new(data: &[T], order: usize) -> Result<Chain<T>, Box<dyn Error>> {
        if data.len() == 0 {
            return Err("Expected data be a non-empty slice!".into());
        }
        let mut map = HashMap::new();

        for window in split_into_windows(data, order + 1) {
            let this = window[0..window.len() - 1].to_vec();
            let next = window[window.len() - 1].clone();
            let key = map.entry(this).or_insert(HashMap::new());
            let count = key.entry(next).or_insert(0usize);
            *count = *count + 1;
        }

        let starting_state = data[0..order].to_vec();
        Ok(Chain {
            map,
            order,
            starting_state,
        })
    }

    pub fn get_order(&self) -> usize {
        self.order
    }

    fn generate_one(&self, state: &[T]) -> Option<T> {
        match self.map.get(state) {
            Some(map) => {
                let sum = map.values().sum();
                let threshold = thread_rng().gen_range(0..sum);
                let mut x = 0;
                let (out, _) = map
                    .iter()
                    .find(|(_, v)| {
                        x += *v;
                        x > threshold
                    })
                    .unwrap();
                Some(out.clone())
            }
            None => None,
        }
    }

    fn random_state(&self) -> Vec<T> {
        self.map.keys().choose(&mut thread_rng()).unwrap().clone()
    }

    pub fn iter_from_start(&self) -> InfiniteIterator<T> {
        self.iter_from(self.starting_state.clone())
    }

    pub fn iter_from_random(&self) -> InfiniteIterator<T> {
        self.iter_from(self.random_state())
    }

    fn iter_from(&self, starting_state: Vec<T>) -> InfiniteIterator<T> {
        InfiniteIterator {
            curr_state: starting_state,
            markov: self,
        }
    }

    /// Return some statistical information about the Markov chain
    /// The returned tuple is (total_states, total_choices, deterministic_states)
    pub fn get_stats(&self) -> (usize, usize, usize, usize) {
        let mut deterministic_states = 0;
        let mut total_choices = 0;
        let mut empty_states = 0;
        let total_states = self.map.len();
        for (_, next_states) in self.map.iter() {
            let choices = next_states.len();
            total_choices += choices;
            if choices == 1 {
                deterministic_states += 1;
            } else if choices == 0 {
                empty_states += 1;
            }
        }
        (
            total_states,
            total_choices,
            deterministic_states,
            empty_states,
        )
    }
}

pub struct InfiniteIterator<'a, T: Chainable> {
    curr_state: Vec<T>,
    markov: &'a Chain<T>,
}

impl<'a, T: Chainable> Iterator for InfiniteIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next_state = self.markov.generate_one(&self.curr_state);
            match next_state {
                Some(next_state) => {
                    self.curr_state.remove(0);
                    self.curr_state.push(next_state.clone());
                    return Some(next_state);
                }
                None => {
                    self.curr_state = self.markov.random_state();
                }
            }
        }
    }
}

pub fn split_into_windows<T>(data: &[T], window_size: usize) -> impl Iterator<Item = &[T]> {
    (0..(data.len() - window_size)).map(move |i| &data[i..i + window_size])
}

pub fn print_statistics<T: Chainable>(chain: &Chain<T>) {
    let (total_states, total_choices, deterministic_states, empty_states) = chain.get_stats();
    println!(
        "Order: {}, Total states: {}, deter/other states: {} / {}, average deter/other: {:.2}% / {:.2}%, average choices per state: {:.2}",
        chain.get_order(),
        total_states,
        deterministic_states,
        total_states - empty_states - deterministic_states,
        100.0 * (deterministic_states as f32 / total_states as f32),
        100.0 * ((total_states - empty_states - deterministic_states) as f32 / total_states as f32),
        (total_choices as f32 / total_states as f32),
    )
}
