use std::error::Error;

use clap::{command, Parser, ValueEnum};
use markov::Chain;
use markov_music::{
    quantize::{Quantizable, QuantizedSample},
    wavelet::{nearest_power_of_two, wavelet_transform, wavelet_untransform, Sample, WaveletType},
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
    /// is specified per band. The first value is the approximation band, then followed by the last
    /// detail band to the first detail band. (Here, "last" detail band means the shortest detail
    /// band, and hence the lowest-frequency detail band. Similarly, "first" detail band means the
    /// longest detail band, and hence the highest-frequency band). Since each successive detail
    /// band is twice as long as the previous, it is recommended that the bands follow a power-of-two
    /// heirachy. If there are less values specified by this flag than there are bands, the last
    /// value specified is applied to all the other bands, doubled for each successive detail band.
    /// For example, "--levels 5 --order 3" is the same as "--levels 5 --order 3 3 6 12 24". This
    /// process caps out at order 128 (going too high results in extremely slow generation--you can
    /// manually override this limit)
    #[arg(short, long, required = true, num_args = 1..)]
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
    fn with_signal(&self, signal: Vec<QuantizedSample>) -> QuantizedBand {
        QuantizedBand {
            signal,
            min: self.min,
            max: self.max,
        }
    }
}

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

fn unquantize_bands(
    detail_bands: &[QuantizedBand],
    approx_band: &QuantizedBand,
    quantization_level: u32,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let detail_bands = detail_bands
        .iter()
        .map(|hi_pass| unquantize(&hi_pass, quantization_level))
        .collect::<Vec<_>>();
    let approx_band = unquantize(&approx_band, quantization_level);
    (detail_bands, approx_band)
}

fn quantize_bands(
    detail_bands: &[Vec<f64>],
    approx_band: &[f64],
    quantization_level: u32,
) -> (Vec<QuantizedBand>, QuantizedBand) {
    let detail_bands = detail_bands
        .iter()
        .map(|hi_pass| quantize(&hi_pass, quantization_level))
        .collect();
    let approx_band = quantize(&approx_band, quantization_level);
    (detail_bands, approx_band)
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
            let value = if orders.len() == 1 {
                last_value
            } else {
                last_value * 2
            };
            orders.push(value.min(128));
        }
    }

    let samples: Vec<Vec<i16>> = channels
        .iter()
        .map(|channel| {
            let orig_samples = channel
                .iter()
                .map(|x| (*x as Sample) / i16::MAX as Sample)
                .collect();

            let (detail_bands, approx_band, _) =
                wavelet_transform(&orig_samples, args.levels, args.wavelet);

            let (detail_bands, approx_band) =
                quantize_bands(&detail_bands, &approx_band, args.quantization);

            let (detail_bands, approx_band) = generate_markov(
                &detail_bands,
                &approx_band,
                &orders,
                args.length * sample_rate,
            );

            let (detail_bands, approx_band) =
                unquantize_bands(&detail_bands, &approx_band, args.quantization);

            let samples = wavelet_untransform(&detail_bands, &approx_band, args.wavelet);

            if args.debug {
                println!("Layers: {}, Wavelet: {:?}", args.levels, args.wavelet);
                samples
                    .iter()
                    .chain(detail_bands.iter().flatten())
                    .chain(approx_band.iter())
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

struct MarkovHeirachy {
    approx_chain: Chain<QuantizedSample>,
    detail_chains: Vec<Chain<QuantizedSample>>,
}

impl MarkovHeirachy {
    // Train a MarkovHeirachy on the given detail and approximation bands, with the given orders.
    fn train(
        detail_bands: &[QuantizedBand],
        approx_band: &QuantizedBand,
        orders: &[usize],
    ) -> Self {
        let detail_chains = detail_bands
            .iter()
            .enumerate()
            .map(|(i, detail_band)| {
                let order = orders[orders.len() - 1 - i];
                let mut detail_chain = Chain::of_order(order);
                println!(
                    "Training detail chain {} ({} samples, order {})",
                    i,
                    detail_band.signal.len(),
                    order
                );
                detail_chain.feed(&detail_band.signal);
                detail_chain
            })
            .collect();

        println!(
            "Training approx chain   ({} samples, order {})",
            approx_band.signal.len(),
            orders[0]
        );
        let mut approx_chain = Chain::of_order(orders[0]);
        approx_chain.feed(&approx_band.signal);

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
        let approx_signal = self
            .approx_chain
            .iter()
            .flatten()
            .take(length / 2usize.pow(num_layers as u32))
            .collect::<Vec<_>>();

        let detail_signals = self
            .detail_chains
            .iter()
            .enumerate()
            .map(|(i, chain)| {
                chain
                    .iter()
                    .flatten()
                    .take(length / 2usize.pow((i + 1) as u32))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        (detail_signals, approx_signal)
    }
}

fn generate_markov(
    detail_bands: &[QuantizedBand],
    approx_band: &QuantizedBand,
    orders: &[usize],
    length: usize,
) -> (Vec<QuantizedBand>, QuantizedBand) {
    let total_samples =
        detail_bands.iter().map(|x| x.signal.len()).sum::<usize>() + approx_band.signal.len();
    println!(
        "Training Markov chain of orders {:?} (total samples: {})",
        orders, total_samples
    );
    let markov_heirachry = MarkovHeirachy::train(detail_bands, approx_band, orders);

    println!(
        "Generating Markov chain samples... (total samples: {})",
        length
    );
    let (detail_signals, approx_signal) = markov_heirachry.generate(length);

    assert!(detail_signals.len() == detail_bands.len());

    let detail_bands = detail_bands
        .iter()
        .zip(detail_signals.into_iter())
        .map(|(band, signal)| band.with_signal(signal))
        .collect();

    (detail_bands, approx_band.with_signal(approx_signal))
}
