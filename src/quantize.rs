pub type QuantizedSample = i32;

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
