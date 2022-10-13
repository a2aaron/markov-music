use crate::wavelet::Sample;

#[derive(Debug)]
pub struct QuantizedSignal {
    pub signal: Vec<QuantizedSample>,
    pub min: f64,
    pub max: f64,
}

impl QuantizedSignal {
    pub fn quantize(signal: &[Sample], quantization_level: u32) -> QuantizedSignal {
        let min = signal.iter().cloned().reduce(f64::min).unwrap();
        let max = signal.iter().cloned().reduce(f64::max).unwrap();

        let signal = signal
            .iter()
            .map(|sample| Quantizable::quantize(*sample, min, max, quantization_level))
            .collect();
        QuantizedSignal { signal, min, max }
    }

    pub fn unquantize(band: &QuantizedSignal, quantization_level: u32) -> Vec<Sample> {
        band.signal
            .iter()
            .map(|quantized| {
                Quantizable::unquantize(*quantized, band.min, band.max, quantization_level)
            })
            .collect()
    }

    pub fn with_signal(&self, signal: Vec<QuantizedSample>) -> QuantizedSignal {
        QuantizedSignal {
            signal,
            min: self.min,
            max: self.max,
        }
    }

    pub fn len(&self) -> usize {
        self.signal.len()
    }
}

pub type QuantizedSample = i64;

pub trait Quantizable: Sized {
    fn from_f64(x: f64) -> Self;
    fn into_f64(x: Self) -> f64;

    fn unquantize(x: QuantizedSample, min: Self, max: Self, quantization_level: u32) -> Self {
        let min = Self::into_f64(min);
        let max = Self::into_f64(max);
        let scale = (max - min) / (quantization_level as f64);
        let x = x as f64;
        Self::from_f64(x * scale)
    }

    /// Quantize `x` to a u32. The `min` and `max` parameters should be set to the
    /// largest and smallest values that `x` may take on. The `quantization_level` parameter
    /// determines the range of the output u32, and consequently, the number of values
    /// the quantization can take on. For example, a quantization_level of 256 means
    /// that the number of representable values is reduced to 256.
    fn quantize(x: Self, min: Self, max: Self, quantization_level: u32) -> QuantizedSample {
        let min = Self::into_f64(min);
        let max = Self::into_f64(max);
        let x = Self::into_f64(x);
        // Note that this transformation is deliberately done so that we remain around zero (in other
        // words, the transformation is _linear_, not affine). This is important--if the transformation
        // was affine (for example, we went from f64 to u32, instead of f64 to i32), then there is
        // a good chance that 0u32 would map to something that is not 0.0f64. This would mean that
        // 0.0f64 would be impossible to represent in the quantized signal, which is bad because
        // most of the upper bands in the wavelet transform are actually zero (or very close to it)
        // when they are mostly empty. Hence, we would be rounding lots of those bands to a nonzero
        // value, which produces really bad artifacting.

        // TODO: we could have a quantization scheme which is non-linear and zero-preserving. For
        // example, we could come up with a dictionary that maps quantized values to non-quantized
        // which always preserves zero (0 -> 0.0) and picks the other quantizations values to minimize
        // some error function.
        let scale = (max - min) / (quantization_level as f64);
        (x / scale).round() as QuantizedSample
    }
}

impl Quantizable for i16 {
    fn from_f64(x: f64) -> Self {
        x as i16
    }

    fn into_f64(x: Self) -> f64 {
        x.into()
    }
}

impl Quantizable for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }

    fn into_f64(x: Self) -> f64 {
        x
    }
}

impl Quantizable for f32 {
    fn from_f64(x: f64) -> Self {
        x as f32
    }

    fn into_f64(x: Self) -> f64 {
        x as f64
    }
}
