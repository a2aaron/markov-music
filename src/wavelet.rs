type Signal = Vec<f32>;

fn interleave_exact<'a, T>(a: &'a [T], b: &'a [T]) -> impl Iterator<Item = &'a T> {
    assert!(a.len() == b.len());
    a.iter()
        .zip(b)
        .map(|(a, b)| std::iter::once(a).chain(std::iter::once(b)))
        .flatten()
}

fn haar_filter() -> [f32; 2] {
    [
        std::f32::consts::FRAC_1_SQRT_2,
        std::f32::consts::FRAC_1_SQRT_2,
    ]
}

fn to_hipass_filter(filter: &[f32]) -> Vec<f32> {
    filter
        .iter()
        .rev()
        .enumerate()
        .map(|(i, x)| if i % 2 == 0 { *x } else { -x })
        .collect()
}

fn invert_filter(filter: &[f32]) -> Vec<f32> {
    (0..filter.len())
        .map(|i| {
            if i % 2 == 0 {
                filter[filter.len() - i - 1]
            } else {
                filter[i]
            }
        })
        .collect()
}

fn convolve(signal: &Signal, filter: &[f32]) -> Signal {
    (0..signal.len())
        .map(|i| {
            (0..filter.len())
                .map(|j| filter[j] * signal.get(i + j).unwrap_or(&0.0))
                .sum()
        })
        .collect()
}

fn low_pass(signal: &Signal, filter: &[f32]) -> Signal {
    convolve(signal, &filter)
}

fn high_pass(signal: &Signal, filter: &[f32]) -> Signal {
    convolve(signal, &to_hipass_filter(filter))
}

fn decimate(signal: &Signal) -> Signal {
    signal
        .iter()
        .cloned()
        .enumerate()
        .filter_map(|(i, sample)| if i % 2 == 0 { Some(sample) } else { None })
        .collect()
}

pub fn wavelet_transform(signal: &Signal, num_levels: usize) -> (Vec<Signal>, Signal) {
    let mut signal = signal.clone();
    let mut hi_passes = vec![];
    let filter = haar_filter();
    for _ in 0..num_levels {
        let low_pass = decimate(&low_pass(&signal, &filter));
        let high_pass = decimate(&high_pass(&signal, &filter));
        assert!(low_pass.len() == high_pass.len());
        assert!(low_pass.len() * 2 == signal.len());

        hi_passes.push(high_pass);
        signal = low_pass;
    }
    (hi_passes, signal)
}

fn upsample(low_signal: &Signal, high_signal: &Signal) -> Signal {
    assert!(low_signal.len() == high_signal.len());
    let interleave = interleave_exact(low_signal, high_signal).cloned().collect();
    let filter = invert_filter(&haar_filter());

    let low_signal = decimate(&low_pass(&interleave, &filter));
    let high_signal = decimate(&high_pass(&interleave, &filter));

    let interleave = interleave_exact(&low_signal, &high_signal)
        .cloned()
        .collect();
    interleave
}

pub fn wavelet_untransform(hi_passes: &[Signal], lowest_pass: &Signal) -> Signal {
    let mut out_signal = lowest_pass.clone();
    for hi_pass in hi_passes.iter().rev() {
        out_signal = upsample(&out_signal, hi_pass);
    }
    out_signal
}
