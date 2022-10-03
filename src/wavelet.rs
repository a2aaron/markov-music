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
    #[value(alias("daub4"))]
    Daubechies4,
}

impl WaveletType {
    fn filter(&self) -> Convolution {
        let filter = match self {
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
        };
        Convolution {
            filter,
            centered_on: 0,
            edge_behavior: EdgeBehavior::WrapAround,
            stride: 2,
        }
    }
}

struct Convolution {
    filter: Vec<Sample>,
    centered_on: usize,
    edge_behavior: EdgeBehavior,
    stride: usize,
}

impl Convolution {
    fn convolve(&self, signal: &Signal) -> Signal {
        let stride_iters = signal.len() / self.stride;
        let mut out_signal = Vec::with_capacity(stride_iters);
        for i in 0..stride_iters {
            let mut value = 0.0;
            for j in 0..self.filter.len() {
                let signal_i = (i * self.stride + j) as isize - (self.centered_on as isize);
                let signal_value = if 0 <= signal_i && signal_i < signal.len() as isize {
                    signal[signal_i as usize]
                } else {
                    match self.edge_behavior {
                        EdgeBehavior::Zeros => 0.0,
                        EdgeBehavior::WrapAround => {
                            let wrapped_i = isize::rem_euclid(signal_i, signal.len() as isize);
                            signal[wrapped_i as usize]
                        }
                        EdgeBehavior::Mirrored => {
                            let len = (signal.len() - 1) as isize;
                            let bounced_i =
                                (isize::rem_euclid(signal_i - len, 2 * len) - len).abs();
                            signal[bounced_i as usize]
                        }
                    }
                };
                value += self.filter[j] * signal_value;
            }
            out_signal.push(value);
        }

        out_signal
    }

    // Transform a low pass filter into a high pass filter
    fn quadrature_mirror(&self) -> Convolution {
        let filter = self
            .filter
            .iter()
            .rev()
            .enumerate()
            .map(|(i, x)| if i % 2 == 0 { *x } else { -x })
            .collect();
        Convolution { filter, ..*self }
    }

    fn invert(&self) -> Convolution {
        let filter = (0..self.filter.len())
            .map(|i| {
                if i % 2 == 0 {
                    self.filter[self.filter.len() - i - 2]
                } else {
                    self.filter[i]
                }
            })
            .collect();
        Convolution { filter, ..*self }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum EdgeBehavior {
    Zeros,
    WrapAround,
    Mirrored,
}

fn low_pass(signal: &Signal, filter: &Convolution) -> Signal {
    filter.convolve(signal)
}

fn high_pass(signal: &Signal, filter: &Convolution) -> Signal {
    filter.quadrature_mirror().convolve(signal)
}

pub fn wavelet_transform(
    orig_signal: &Signal,
    num_levels: usize,
    wavelet: WaveletType,
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
        let low_pass = low_pass(&signal, &filter);
        let high_pass = high_pass(&signal, &filter);
        assert!(low_pass.len() == high_pass.len());
        assert!(low_pass.len() * 2 == signal.len());

        hi_passes.push(high_pass);
        low_passes.push(low_pass.clone());
        signal = low_pass;
    }
    (hi_passes, signal, low_passes)
}

fn upsample(low_signal: &Signal, high_signal: &Signal, wavelet: WaveletType) -> Signal {
    assert!(low_signal.len() == high_signal.len());
    let interleave = interleave_exact(low_signal, high_signal).cloned().collect();

    let mut filter = wavelet.filter().invert();

    // Choice of centering on 2 here for both signals is because of Numerical Recipes
    // See 13.10 Wavelet Transforms, page 705.
    // http://numerical.recipes/book/book.html
    match wavelet {
        WaveletType::Haar => (),
        WaveletType::Daubechies4 => filter.centered_on = 2,
    }

    let low_signal = low_pass(&interleave, &filter);
    let high_signal = high_pass(&interleave, &filter);

    let interleave = interleave_exact(&low_signal, &high_signal)
        .cloned()
        .collect();
    interleave
}

pub fn wavelet_untransform(
    hi_passes: &[Signal],
    lowest_pass: &Signal,
    wavelet: WaveletType,
) -> Signal {
    let mut out_signal = lowest_pass.clone();
    for hi_pass in hi_passes.iter().rev() {
        out_signal = upsample(&out_signal, hi_pass, wavelet);
    }
    out_signal
}
