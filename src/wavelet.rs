// See https://mil.ufl.edu/nechyba/www/eel6562/course_materials/t5.wavelets/intro_dwt.pdf for more info.
// http://bearcave.com/misl/misl_tech/wavelets/index.html

use clap::ValueEnum;

type Signal = Vec<Sample>;
pub type Sample = f64;

fn interleave_exact<'a, T>(a: &'a [T], b: &'a [T]) -> impl Iterator<Item = &'a T> {
    assert!(a.len() == b.len());
    a.iter()
        .zip(b)
        .map(|(a, b)| std::iter::once(a).chain(std::iter::once(b)))
        .flatten()
}
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum WaveletType {
    Haar,
    Daubechies4,
}

impl WaveletType {
    fn filter(&self) -> Vec<Sample> {
        match self {
            WaveletType::Haar => vec![
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
            ],
            WaveletType::Daubechies4 => {
                let root_3 = Sample::from(3.0).sqrt();
                let four_times_root_2 = 4.0 * std::f64::consts::SQRT_2;
                vec![
                    (1.0 + root_3) / four_times_root_2,
                    (3.0 + root_3) / four_times_root_2,
                    (3.0 - root_3) / four_times_root_2,
                    (1.0 - root_3) / four_times_root_2,
                ]
            }
        }
    }
}

fn to_hipass_filter(filter: &[Sample]) -> Vec<Sample> {
    filter
        .iter()
        .rev()
        .enumerate()
        .map(|(i, x)| if i % 2 == 0 { *x } else { -x })
        .collect()
}

// 0 1 2 3
// 2 1 0 3

// 0 1 2 3 4 5
// 4 1 2 3 0 5

// 0 1 2 3 4 5 6 7
// 6 1 4 3 2 5 0 7
fn invert_filter(filter: &[Sample]) -> Vec<Sample> {
    (0..filter.len())
        .map(|i| {
            if i % 2 == 0 {
                filter[filter.len() - i - 2]
            } else {
                filter[i]
            }
        })
        .collect()
}

fn convolve(signal: &Signal, filter: &[Sample], centering: usize) -> Signal {
    let mut out_signal = vec![Sample::NAN; signal.len()];
    for i in 0..signal.len() {
        let mut value = 0.0;
        for j in 0..filter.len() {
            let signal_i = (i + j).checked_add_signed(-(centering as isize));
            if let Some(signal_i) = signal_i {
                value += filter[j] * signal.get(signal_i).unwrap_or(&0.0);
            }
        }
        out_signal[i] = value;
    }

    assert!(out_signal.iter().all(|x| x.is_finite()));
    out_signal
}

fn low_pass(signal: &Signal, filter: &[Sample], centering: usize) -> Signal {
    convolve(signal, &filter, centering)
}

fn high_pass(signal: &Signal, filter: &[Sample], centering: usize) -> Signal {
    convolve(signal, &to_hipass_filter(filter), centering)
}

fn decimate(signal: &Signal) -> Signal {
    signal
        .iter()
        .cloned()
        .enumerate()
        .filter_map(|(i, sample)| if i % 2 == 0 { Some(sample) } else { None })
        .collect()
}

pub fn wavelet_transform(
    orig_signal: &Signal,
    num_levels: usize,
    wavelet: WaveletType,
    centering: usize,
) -> (Vec<Signal>, Signal, Vec<Signal>) {
    let mut signal = orig_signal.clone();

    let power_of_two = 2usize.pow(num_levels as u32);
    if signal.len() % power_of_two != 0 {
        let padding_length = power_of_two - (signal.len() % power_of_two);
        signal.append(&mut vec![0.0; padding_length]);
        println!(
            "Added {} zeros of padding (original input length: {}, new length: {})",
            padding_length,
            orig_signal.len(),
            signal.len()
        );
    }

    let mut hi_passes = vec![];
    let mut low_passes = vec![];
    let filter = wavelet.filter();
    for _ in 0..num_levels {
        // let centering = filter.len() / 2;

        let low_pass = decimate(&low_pass(&signal, &filter, centering));
        let high_pass = decimate(&high_pass(&signal, &filter, centering));
        assert!(low_pass.len() == high_pass.len());
        assert!(low_pass.len() * 2 == signal.len());

        hi_passes.push(high_pass);
        low_passes.push(low_pass.clone());
        signal = low_pass;
    }
    (hi_passes, signal, low_passes)
}

fn upsample(
    low_signal: &Signal,
    high_signal: &Signal,
    wavelet: WaveletType,
    centering: usize,
) -> Signal {
    assert!(low_signal.len() == high_signal.len());
    let interleave = interleave_exact(low_signal, high_signal).cloned().collect();

    let filter = invert_filter(&wavelet.filter());

    let low_signal = decimate(&low_pass(&interleave, &filter, centering));
    let high_signal = decimate(&high_pass(&interleave, &filter, centering));

    let interleave = interleave_exact(&low_signal, &high_signal)
        .cloned()
        .collect();
    interleave
}

pub fn wavelet_untransform(
    hi_passes: &[Signal],
    lowest_pass: &Signal,
    wavelet: WaveletType,
    centering: usize,
) -> Signal {
    let mut out_signal = lowest_pass.clone();
    for hi_pass in hi_passes.iter().rev() {
        out_signal = upsample(&out_signal, hi_pass, wavelet, centering);
    }
    out_signal
}
