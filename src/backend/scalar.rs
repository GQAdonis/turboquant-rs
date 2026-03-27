//! Scalar (non-vectorized) backend implementation.
//!
//! This wraps the existing hadamard and utility functions,
//! providing the baseline implementation that SIMD/GPU backends
//! must match in correctness.

use crate::error::{Result, TurboQuantError};
use crate::hadamard::fwht_normalized_inplace;

/// Scalar compute backend — no SIMD or GPU acceleration.
///
/// This is the default backend used when no acceleration features are enabled.
/// All operations use standard Rust iterators and scalar arithmetic.
#[derive(Clone, Debug, Default)]
pub struct ScalarBackend;

impl super::Backend for ScalarBackend {
    #[inline]
    fn fwht_normalized_inplace(&self, data: &mut [f32]) {
        fwht_normalized_inplace(data);
    }

    #[inline]
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    fn validate_dimension(&self, dim: usize) -> Result<()> {
        if !dim.is_power_of_two() {
            Err(TurboQuantError::DimensionNotPowerOfTwo(dim))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;

    #[test]
    fn scalar_backend_implements_trait() {
        let backend = ScalarBackend;
        // Validate dimension check
        assert!(backend.validate_dimension(128).is_ok());
        assert!(backend.validate_dimension(7).is_err());
    }

    #[test]
    fn scalar_dot_product_correctness() {
        let backend = ScalarBackend;
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let result = backend.dot_product(&a, &b);
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert!((result - 70.0).abs() < 1e-6);
    }

    #[test]
    fn scalar_fwht_roundtrip() {
        let backend = ScalarBackend;
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = original.clone();
        backend.fwht_normalized_inplace(&mut data);
        backend.fwht_normalized_inplace(&mut data);
        for (a, b) in original.iter().zip(&data) {
            assert!((a - b).abs() < 1e-5, "FWHT roundtrip failed: {a} vs {b}");
        }
    }
}
