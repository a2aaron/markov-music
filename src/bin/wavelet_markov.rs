use std::error::Error;

use clap::{command, Parser, ValueEnum};
use markov_music::{
    markov::{Chain, Chainable},
    quantize::{Quantizable, QuantizedSample},
    wavelet::{
        nearest_power_of_two, wavelet_transform, wavelet_untransform, Sample, WaveletHeirarchy,
        WaveletType,
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

struct QuantizedHeirarchy {
    approx_band: QuantizedBand,
    detail_bands: Vec<QuantizedBand>,
    quantization_level: u32,
}

impl QuantizedHeirarchy {
    fn quantize(wavelets: &WaveletHeirarchy, quantization_level: u32) -> QuantizedHeirarchy {
        let approx_band = QuantizedBand::quantize(&wavelets.approx_band, quantization_level);
        let detail_bands = wavelets
            .detail_bands
            .iter()
            .map(|band| QuantizedBand::quantize(&band, quantization_level))
            .collect();
        QuantizedHeirarchy::new(approx_band, detail_bands, quantization_level)
    }

    fn unquantize(&self) -> WaveletHeirarchy {
        let approx_band = QuantizedBand::unquantize(&self.approx_band, self.quantization_level);
        let detail_bands = self
            .detail_bands
            .iter()
            .map(|band| QuantizedBand::unquantize(&band, self.quantization_level))
            .collect();
        WaveletHeirarchy::new(approx_band, detail_bands)
    }

    fn new(
        approx_band: QuantizedBand,
        detail_bands: Vec<QuantizedBand>,
        quantization_level: u32,
    ) -> QuantizedHeirarchy {
        assert!(approx_band.len() == detail_bands[0].len());
        for i in 0..(detail_bands.len() - 1) {
            assert!(detail_bands[i].len() * 2 == detail_bands[i + 1].len());
        }
        QuantizedHeirarchy {
            detail_bands,
            approx_band,
            quantization_level,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let (left, right, sample_rate) = util::read_mp3_file(&args.in_path)?;

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
            let wavelets = generate_markov(&wavelets, &orders, args.length * sample_rate);
            let wavelets = wavelets.unquantize();

            let samples = wavelet_untransform(&wavelets, args.wavelet);

            if args.debug {
                println!("Generating solo-band samples...");
                let mut additional_samples = vec![];

                {
                    let mut wavelets = wavelets.clone();
                    wavelets.detail_bands.iter_mut().for_each(|x| x.fill(0.0));
                    println!("Generating solo-band for approx band");
                    let samples = wavelet_untransform(&wavelets, args.wavelet);
                    additional_samples.push(samples);
                }
                for i in 0..wavelets.levels() {
                    let mut wavelets = wavelets.clone();
                    wavelets.approx_band.fill(0.0);
                    for j in 0..wavelets.levels() {
                        if i != j {
                            wavelets.detail_bands[j].fill(0.0);
                        }
                    }
                    println!("Generating solo-band for detail band {}", i);
                    let samples = wavelet_untransform(&wavelets, args.wavelet);
                    additional_samples.push(samples);
                }

                samples
                    .iter()
                    .chain(additional_samples.iter().flatten())
                    .cloned()
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

fn print_statistics<T: Chainable>(chain: &Chain<T>) {
    let (total_states, total_choices, deterministic_states) = chain.get_stats();

    println!(
        "Order: {}, Total states: {}, deterministic states: {}, average determinism: {:.2}%, average choices per state: {:.2}",
        chain.get_order(),
        total_states,
        deterministic_states,
        100.0 * (deterministic_states as f32 / total_states as f32),
        (total_choices as f32 / total_states as f32),
    )
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

    QuantizedHeirarchy::new(approx_band, detail_bands, wavelets.quantization_level)
}
