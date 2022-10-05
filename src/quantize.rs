fn inv_lerp(a: f64, b: f64, x: f64) -> f64 {
    (x - a) / (b - a)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

pub trait Quantizable: Sized {
    fn from_f64(x: f64) -> Self;
    fn into_f64(x: Self) -> f64;

    fn unquantize(x: usize, min: Self, max: Self, quantization_level: usize) -> Self {
        let min = Self::into_f64(min);
        let max = Self::into_f64(max);

        // We have the quantized value of x in the range [0, quantization_level]
        // We first map it into the [0.0, 1.0] range
        let x = inv_lerp(0.0, quantization_level as f64, x as f64);
        // Then we map from [0.0, 1.0] to [min, max].
        let x = lerp(min, max, x);

        Self::from_f64(x)
    }

    /// Quantize `x` to a usize. The `min` and `max` parameters should be set to the
    /// largest and smallest values that `x` may take on. The `quantization_level` parameter
    /// determines the range of the output usize, and consequently, the number of values
    /// the quantization can take on. For example, a quantization_level of 256 means
    /// that the number of representable values is reduced to 256.
    fn quantize(x: Self, min: Self, max: Self, quantization_level: usize) -> usize {
        let min = Self::into_f64(min);
        let max = Self::into_f64(max);
        let x = Self::into_f64(x);

        // First, map x from the [min, max] range to the [0.0, 1.0] range (which turns x into t)
        let t = inv_lerp(min, max, x);
        // Then, map from the [0.0, 1.0] range to the [0.0, quantization_level] range, and quantize
        // by casting to a usize.
        // TODO: should i round instead of cast?
        lerp(0.0, quantization_level as f64, t) as usize
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
