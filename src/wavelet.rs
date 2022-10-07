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
    #[value(alias("daub12"))]
    Daubechies12,
    #[value(alias("daub20"))]
    Daubechies20,
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
            // Numerical constants taken from http://numerical.recipes/book/book.html
            // See 13.10 Wavelet Transforms, page 705.
            WaveletType::Daubechies12 => vec![
                0.111540743350,
                0.494623890398,
                0.751133908021,
                0.315250351709,
                -0.226264693965,
                -0.129766867567,
                0.097501605587,
                0.027522865530,
                -0.031582039318,
                0.000553842201,
                0.004777257511,
                -0.001077301085,
            ],
            WaveletType::Daubechies20 => vec![
                0.026670057901,
                0.188176800078,
                0.527201188932,
                0.688459039454,
                0.281172343661,
                -0.249846424327,
                -0.195946274377,
                0.127369340336,
                0.093057364604,
                -0.071394147166,
                -0.029457536822,
                0.033212674059,
                0.003606553567,
                -0.010733175483,
                0.001395351747,
                0.001992405295,
                -0.000685856695,
                -0.000116466855,
                0.000093588670,
                -0.000013264203,
            ],
        };
        Convolution {
            filter,
            centered_on: 0,
            edge_behavior: EdgeBehavior::WrapAround,
            stride: 2,
        }
    }

    fn inverse_filter(&self) -> Convolution {
        let filter = self.filter().filter;
        let filter = (0..filter.len())
            .map(|i| {
                if i % 2 == 0 {
                    filter[filter.len() - i - 2]
                } else {
                    filter[i]
                }
            })
            .collect();

        // Choice of centering on 2 here for Daub4 is because of Numerical Recipes
        // See 13.10 Wavelet Transforms, page 705.
        // http://numerical.recipes/book/book.html
        // In general it seems like the correct centering choice is filter length - 2?
        let centered_on = match self {
            WaveletType::Haar => 0,
            WaveletType::Daubechies4 => 2,
            WaveletType::Daubechies12 => 10,
            WaveletType::Daubechies20 => 18,
        };
        Convolution {
            filter,
            centered_on,
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

pub fn nearest_power_of_two(x: usize, power_of_two: usize) -> usize {
    let power_of_two = 2usize.pow(power_of_two as u32);
    let rounded = x - x % power_of_two;
    assert_eq!(rounded % power_of_two, 0);
    rounded
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

    let mut detail_bands = vec![];
    let mut approx_band = vec![];
    let filter = wavelet.filter();
    for _ in 0..num_levels {
        let approx = low_pass(&signal, &filter);
        let detail = high_pass(&signal, &filter);
        assert!(
            approx.len() == detail.len(),
            "Expected approx and detail bands to have same length. Got approx: {} and detail: {}",
            approx.len(),
            detail.len()
        );
        assert!(approx.len() * 2 == signal.len());

        detail_bands.push(detail);
        approx_band.push(approx.clone());
        signal = approx;
    }
    (detail_bands, signal, approx_band)
}

fn upsample(approx: &Signal, detail: &Signal, wavelet: WaveletType) -> Signal {
    assert!(
        approx.len() == detail.len(),
        "Expected approx and detail bands to have same length. Got approx: {} and detail: {}",
        approx.len(),
        detail.len()
    );
    let interleave = interleave_exact(approx, detail).cloned().collect();

    let filter = wavelet.inverse_filter();

    let approx = low_pass(&interleave, &filter);
    let detail = high_pass(&interleave, &filter);

    let interleave = interleave_exact(&approx, &detail).cloned().collect();
    interleave
}

pub fn wavelet_untransform(
    detail_bands: &[Signal],
    approx_band: &Signal,
    wavelet: WaveletType,
) -> Signal {
    let mut out_signal = approx_band.clone();
    for detail in detail_bands.iter().rev() {
        out_signal = upsample(&out_signal, detail, wavelet);
    }
    out_signal
}
