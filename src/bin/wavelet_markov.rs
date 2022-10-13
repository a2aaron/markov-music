use std::{cmp::Ordering, error::Error};

use clap::{command, Parser, ValueEnum};
use markov_music::{
    markov::{print_statistics, Chain},
    quantize::{Quantizable, QuantizedSample},
    wavelet::{
        nearest_power_of_two, solo_bands, wavelet_transform, wavelet_untransform, Sample,
        WaveletHeirarchy, WaveletType,
    },
};

mod util;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
/// A WAV file generator powered by markov chain.
struct Args {
    /// Path to input MP3 file.
    #[arg(short, long = "in")]
    in_path: String,
    /// Path to output WAV file.
    #[arg(short, long = "out", default_value = "out.wav")]
    out_path: String,
    /// Markov chain order. Higher values means the output is less chaotic, but more deterministic.
    /// Recommended values are betwee 3 and 8, depending on the length and type of input file. This
    /// is specified per band. The first value is the approximation band, then followed by the first
    /// detail band to the last detail band. (Here, "first" detail band means the shortest detail
    /// band, and hence the lowest-frequency detail band. Similarly, "last" detail band means the
    /// longest detail band, and hence the highest-frequency band). Since each successive detail
    /// band is twice as long as the previous, it is recommended that higher bands have higher order.
    /// If there are less values specified by this flag than there are bands, the last
    /// value specified is applied to all the other bands For example,
    /// "--levels 5 --order 3" is the same as "--levels 5 --order 3 3 3 3 3".
    #[arg(long, required = true, num_args = 1..)]
    order: Vec<usize>,
    /// The number of levels to quantize to. This value is applied to every channel equally.
    #[arg(long, default_value_t = 256)]
    quantization: u32,
    /// Length, in seconds, of audio to generate.
    #[arg(long, default_value_t = 60)]
    length: usize,
    /// Which channel of the mp3 to use.
    #[arg(value_enum, default_value_t = Channel::Left)]
    channel: Channel,
    /// Number of levels to use in the wavelet transform.
    #[arg(long, default_value_t = 6)]
    levels: usize,
    /// Wavelet type to use
    #[arg(long, value_enum, default_value_t = WaveletType::Haar)]
    wavelet: WaveletType,
    /// Use tokenization mode
    #[arg(long)]
    tokenize: bool,
    /// Enable debug mode.
    #[arg(long)]
    debug: bool,
}

#[derive(Debug, Copy, Clone, ValueEnum)]
enum Channel {
    Left,
    Right,
    Both,
}

#[derive(Debug)]
struct QuantizedBand {
    signal: Vec<QuantizedSample>,
    min: f64,
    max: f64,
}

impl QuantizedBand {
    fn quantize(signal: &[Sample], quantization_level: u32) -> QuantizedBand {
        let min = signal.iter().cloned().reduce(f64::min).unwrap();
        let max = signal.iter().cloned().reduce(f64::max).unwrap();

        let signal = signal
            .iter()
            .map(|sample| Quantizable::quantize(*sample, min, max, quantization_level))
            .collect();
        QuantizedBand { signal, min, max }
    }

    fn unquantize(band: &QuantizedBand, quantization_level: u32) -> Vec<Sample> {
        band.signal
            .iter()
            .map(|quantized| {
                Quantizable::unquantize(*quantized, band.min, band.max, quantization_level)
            })
            .collect()
    }

    fn with_signal(&self, signal: Vec<QuantizedSample>) -> QuantizedBand {
        QuantizedBand {
            signal,
            min: self.min,
            max: self.max,
        }
    }

    fn len(&self) -> usize {
        self.signal.len()
    }
}

#[derive(Debug)]
struct QuantizedHeirarchy {
    approx_band: QuantizedBand,
    detail_bands: Vec<QuantizedBand>,
    quantization_level: u32,
    wavelet_type: WaveletType,
}

impl QuantizedHeirarchy {
    fn quantize(wavelets: &WaveletHeirarchy, quantization_level: u32) -> QuantizedHeirarchy {
        let approx_band = QuantizedBand::quantize(&wavelets.approx_band, quantization_level);
        let detail_bands = wavelets
            .detail_bands
            .iter()
            .map(|band| QuantizedBand::quantize(&band, quantization_level))
            .collect();
        QuantizedHeirarchy::new(
            approx_band,
            detail_bands,
            quantization_level,
            wavelets.wave_type,
        )
    }

    fn unquantize(&self) -> WaveletHeirarchy {
        let approx_band = QuantizedBand::unquantize(&self.approx_band, self.quantization_level);
        let detail_bands = self
            .detail_bands
            .iter()
            .map(|band| QuantizedBand::unquantize(&band, self.quantization_level))
            .collect();
        WaveletHeirarchy::new(approx_band, detail_bands, self.wavelet_type)
    }

    fn new(
        approx_band: QuantizedBand,
        detail_bands: Vec<QuantizedBand>,
        quantization_level: u32,
        wavelet_type: WaveletType,
    ) -> QuantizedHeirarchy {
        assert!(approx_band.len() == detail_bands[0].len());
        for i in 0..(detail_bands.len() - 1) {
            assert!(detail_bands[i].len() * 2 == detail_bands[i + 1].len());
        }
        QuantizedHeirarchy {
            detail_bands,
            approx_band,
            quantization_level,
            wavelet_type,
        }
    }

    fn with_tokens(&self, tokens: &[WaveletToken]) -> QuantizedHeirarchy {
        let num_layers = self.levels();

        let (approx_signal, detail_signals) = tokens.iter().fold(
            (vec![], vec![vec![]; num_layers]),
            |(mut approx_signal, mut detail_signals), token| {
                let (approx_sample, detail_samples) = token.get_samples();
                approx_signal.push(approx_sample);

                assert!(detail_samples.len() == num_layers);
                for (detail_samples, detail_signal) in
                    detail_samples.iter().zip(detail_signals.iter_mut())
                {
                    detail_signal.extend_from_slice(detail_samples);
                }

                (approx_signal, detail_signals)
            },
        );

        let approx_band = self.approx_band.with_signal(approx_signal);
        let detail_bands = self
            .detail_bands
            .iter()
            .zip(detail_signals)
            .map(|(band, signal)| band.with_signal(signal))
            .collect();

        QuantizedHeirarchy::new(
            approx_band,
            detail_bands,
            self.quantization_level,
            self.wavelet_type,
        )
    }

    fn levels(&self) -> usize {
        self.detail_bands.len()
    }

    fn tokenize<'a>(&'a self) -> Vec<WaveletToken<'a>> {
        let mut tokens = vec![];
        for i in 0..self.approx_band.len() {
            // let approx_sample = &self.approx_band.signal[i];
            // let mut detail_samples = vec![];
            // for j in 0..self.detail_bands.len() {
            //     let window = 2usize.pow(j as u32);
            //     let lower = i * window;
            //     let upper = (i + 1) * window;
            //     detail_samples.push(&self.detail_bands[j].signal[lower..upper]);
            // }
            tokens.push(WaveletToken {
                approx_sample: i,
                wavelets: self,
                // approx_sample,
                // detail_samples,
            });
        }
        tokens
    }
}

#[derive(Debug, Clone)]
struct WaveletToken<'a> {
    approx_sample: usize,
    wavelets: &'a QuantizedHeirarchy,
}

impl<'a> WaveletToken<'a> {
    fn get_samples(&self) -> (QuantizedSample, Vec<&[QuantizedSample]>) {
        let wavelets = self.wavelets;
        let i = self.approx_sample;
        let approx_sample = wavelets.approx_band.signal[i];
        let mut detail_samples = Vec::with_capacity(wavelets.levels());
        for j in 0..wavelets.detail_bands.len() {
            let window = 2usize.pow(j as u32);
            let lower = i * window;
            let upper = (i + 1) * window;
            detail_samples.push(&wavelets.detail_bands[j].signal[lower..upper]);
        }

        (approx_sample, detail_samples)
    }

    fn cmp_list(&self, other: &WaveletToken) -> Vec<Ordering> {
        let mut list = vec![];
        let (approx_self, detail_self) = self.get_samples();
        let (approx_other, detail_other) = other.get_samples();
        assert!(detail_self.len() == detail_other.len());
        for (this, other) in detail_self.iter().zip(detail_other.iter()).rev() {
            list.push(this[0].abs().cmp(&other[0].abs()));
        }
        list.push(approx_self.abs().cmp(&approx_other.abs()));
        list
    }
}

impl PartialEq for WaveletToken<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_list(other)
            .iter()
            .all(|cmp| *cmp == Ordering::Equal)
    }
}

impl PartialOrd for WaveletToken<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let cmp = self
            .cmp_list(other)
            .iter()
            .find(|result| **result != Ordering::Equal)
            .copied()
            .unwrap_or(Ordering::Equal);
        Some(cmp)
    }
}

impl Eq for WaveletToken<'_> {}
impl Ord for WaveletToken<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl std::hash::Hash for WaveletToken<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let (approx_sample, detail_samples) = self.get_samples();
        approx_sample.hash(state);
        for detail_sample in detail_samples {
            detail_sample.hash(state);
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let (left, right, sample_rate) = util::read_file(&args.in_path)?;

    let channels = if let Some(right) = right {
        match args.channel {
            Channel::Left => vec![left],
            Channel::Right => vec![right],
            Channel::Both => vec![left, right],
        }
    } else {
        println!("Ignoring --channel flag because there is only one channel");
        vec![left]
    };

    let mut orders = args.order;

    if orders.len() == 0 {
        return Err("Not enough order values! Expected least 1 value, got zero".into());
    } else if orders.len() > args.levels + 1 {
        return Err(format!(
            "Too many order values! Expected atm most {}, got {}",
            orders.len(),
            args.levels
        )
        .into());
    } else if orders.len() < args.levels + 1 {
        while orders.len() != args.levels + 1 {
            let last_value = *orders.last().unwrap();
            orders.push(last_value);
        }
    }

    let samples: Vec<Vec<i16>> = channels
        .iter()
        .map(|channel| {
            let orig_samples = channel
                .iter()
                .map(|x| (*x as Sample) / i16::MAX as Sample)
                .collect();

            let wavelets = wavelet_transform(&orig_samples, args.levels, args.wavelet);
            let wavelets = QuantizedHeirarchy::quantize(&wavelets, args.quantization);

            let length = args.length * sample_rate;

            let wavelets = if args.tokenize {
                generate_markov_2(&wavelets, orders[0], length)
            } else {
                generate_markov(&wavelets, &orders, length)
            };
            let wavelets = wavelets.unquantize();

            let samples = wavelet_untransform(&wavelets);

            if args.debug {
                println!("Generating solo-band samples...");
                let additional_samples = solo_bands(&wavelets);
                samples
                    .iter()
                    .chain(additional_samples.iter().flatten())
                    .map(|x| (x * i16::MAX as Sample * 0.5) as i16)
                    .collect()
            } else {
                samples
                    .iter()
                    .map(|x| (x * i16::MAX as Sample * 0.5) as i16)
                    .collect()
            }
        })
        .collect();

    util::write_wav(
        &args.out_path,
        sample_rate,
        &samples[0],
        samples.get(1).map(Vec::as_ref),
    )?;

    Ok(())
}

struct MarkovHeirachy {
    approx_chain: Chain<QuantizedSample>,
    detail_chains: Vec<Chain<QuantizedSample>>,
}

impl MarkovHeirachy {
    // Train a MarkovHeirachy on the given detail and approximation bands, with the given orders.
    fn train(wavelets: &QuantizedHeirarchy, orders: &[usize]) -> Self {
        println!(
            "Training approx chain   ({} samples, order {})",
            wavelets.approx_band.signal.len(),
            orders[0]
        );
        let approx_chain = Chain::new(&wavelets.approx_band.signal, orders[0]).unwrap();

        let detail_chains: Vec<_> = wavelets
            .detail_bands
            .iter()
            .enumerate()
            .map(|(i, detail_band)| {
                let order = orders[i + 1];
                println!(
                    "Training detail chain {} ({} samples, order {})",
                    i,
                    detail_band.signal.len(),
                    order
                );
                Chain::new(&detail_band.signal, order).unwrap()
            })
            .collect();

        print_statistics(&approx_chain);
        detail_chains.iter().for_each(print_statistics);

        Self {
            approx_chain,
            detail_chains,
        }
    }

    /// Generate wavelet bands from the trained markov chains such that the reconstructed sample
    /// will be approximately `length` samples long. If `length` is not a multiple of a power of two
    /// of the number of levels, then `length` is adjusted to match (in this case, rounded down).
    fn generate(&self, length: usize) -> (Vec<Vec<QuantizedSample>>, Vec<QuantizedSample>) {
        let num_layers = self.detail_chains.len();
        let length = nearest_power_of_two(length, num_layers);

        // Note that these divisions of length are actually one more than you would expect from zero
        // indexing. This is because each layer is half of the prior signal. Hence, the first layer
        // (layer 0) is actually half the length of the input audio. Therefore, we need to add one
        // to the exponent when dividing by 2^n in order to account for this.
        let length = length / 2usize.pow(num_layers as u32);

        let detail_signals = self
            .detail_chains
            .iter()
            .enumerate()
            .map(|(i, chain)| {
                chain
                    .iter_from_start()
                    .take(length * 2usize.pow(i as u32))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let approx_signal = self
            .approx_chain
            .iter_from_start()
            .take(length)
            .collect::<Vec<_>>();

        (detail_signals, approx_signal)
    }
}

fn generate_markov(
    wavelets: &QuantizedHeirarchy,
    orders: &[usize],
    length: usize,
) -> QuantizedHeirarchy {
    let total_samples = wavelets
        .detail_bands
        .iter()
        .map(|x| x.signal.len())
        .sum::<usize>()
        + wavelets.approx_band.signal.len();
    println!(
        "Training Markov chain of orders {:?} (total samples: {})",
        orders, total_samples
    );
    let markov_heirachry = MarkovHeirachy::train(&wavelets, orders);

    println!(
        "Generating Markov chain samples... (total samples: {})",
        length
    );

    let (detail_signals, approx_signal) = markov_heirachry.generate(length);

    let detail_bands = wavelets
        .detail_bands
        .iter()
        .zip(detail_signals.into_iter())
        .map(|(band, signal)| band.with_signal(signal))
        .collect();

    let approx_band = wavelets.approx_band.with_signal(approx_signal);

    QuantizedHeirarchy::new(
        approx_band,
        detail_bands,
        wavelets.quantization_level,
        wavelets.wavelet_type,
    )
}

fn generate_markov_2(
    wavelets: &QuantizedHeirarchy,
    order: usize,
    length: usize,
) -> QuantizedHeirarchy {
    let tokens = wavelets.tokenize();
    // tokens.shuffle(&mut rand::thread_rng());
    // tokens.sort();

    println!(
        "Training Markov chain of order {:?} (total samples: {})",
        order,
        tokens.len()
    );

    let chain = Chain::new(&tokens, order).unwrap();

    print_statistics(&chain);

    let num_layers = wavelets.levels();
    let length = nearest_power_of_two(length, num_layers);

    // Note that these divisions of length are actually one more than you would expect from zero
    // indexing. This is because each layer is half of the prior signal. Hence, the first layer
    // (layer 0) is actually half the length of the input audio. Therefore, we need to add one
    // to the exponent when dividing by 2^n in order to account for this.
    let length = length / 2usize.pow(num_layers as u32);
    let tokens: Vec<_> = chain.iter_from_start().take(length).collect();
    wavelets.with_tokens(&tokens)
}
