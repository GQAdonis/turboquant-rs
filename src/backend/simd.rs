//! SIMD-accelerated backend implementation.
//!
//! Uses AVX2 on x86_64 and NEON on aarch64 for vectorized FWHT
//! and dot product operations. Falls back to scalar for small
//! dimensions or unsupported strides.

use crate::backend::{Backend, ScalarBackend};
use crate::error::{Result, TurboQuantError};
use crate::hadamard::fwht_normalized_inplace;

/// SIMD-accelerated compute backend.
///
/// Uses AVX2 on x86_64 and NEON on aarch64 for vectorized FWHT
/// and dot product operations. Falls back to scalar for small
/// dimensions or unsupported strides.
#[derive(Clone, Debug, Default)]
pub struct SimdBackend;

impl Backend for SimdBackend {
    #[inline]
    fn fwht_normalized_inplace(&self, data: &mut [f32]) {
        // TODO(plan-02): Replace with SIMD FWHT (AVX2/NEON)
        fwht_normalized_inplace(data);
    }

    #[inline]
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        // TODO(plan-02): Replace with SIMD dot product (AVX2/NEON)
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

/// Runtime-selected backend that automatically uses the fastest
/// available implementation based on CPU feature detection.
///
/// Uses static dispatch via enum -- no dynamic dispatch overhead.
#[derive(Clone, Debug)]
pub enum RuntimeBackend {
    /// Scalar fallback (no SIMD)
    Scalar(ScalarBackend),
    /// SIMD-accelerated (AVX2 on x86_64, NEON on aarch64)
    Simd(SimdBackend),
}

impl RuntimeBackend {
    /// Detect CPU features and return the fastest available backend.
    pub fn best_available() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return RuntimeBackend::Simd(SimdBackend);
            }
        }

        #[cfg(target_arch = "x86")]
        {
            if is_x86_feature_detected!("avx2") {
                return RuntimeBackend::Simd(SimdBackend);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return RuntimeBackend::Simd(SimdBackend);
            }
        }

        RuntimeBackend::Scalar(ScalarBackend)
    }
}

impl Backend for RuntimeBackend {
    #[inline]
    fn fwht_normalized_inplace(&self, data: &mut [f32]) {
        match self {
            RuntimeBackend::Scalar(backend) => backend.fwht_normalized_inplace(data),
            RuntimeBackend::Simd(backend) => backend.fwht_normalized_inplace(data),
        }
    }

    #[inline]
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            RuntimeBackend::Scalar(backend) => backend.dot_product(a, b),
            RuntimeBackend::Simd(backend) => backend.dot_product(a, b),
        }
    }

    fn validate_dimension(&self, dim: usize) -> crate::error::Result<()> {
        match self {
            RuntimeBackend::Scalar(backend) => backend.validate_dimension(dim),
            RuntimeBackend::Simd(backend) => backend.validate_dimension(dim),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;

    #[test]
    fn simd_backend_implements_trait() {
        let backend = SimdBackend;
        assert!(backend.validate_dimension(128).is_ok());
        assert!(backend.validate_dimension(7).is_err());
    }

    #[test]
    fn simd_dot_product_correctness() {
        let backend = SimdBackend;
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let result = backend.dot_product(&a, &b);
        assert!((result - 70.0).abs() < 1e-6);
    }

    #[test]
    fn simd_fwht_roundtrip() {
        let backend = SimdBackend;
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = original.clone();
        backend.fwht_normalized_inplace(&mut data);
        backend.fwht_normalized_inplace(&mut data);
        for (a, b) in original.iter().zip(&data) {
            assert!((a - b).abs() < 1e-5, "FWHT roundtrip failed: {a} vs {b}");
        }
    }

    #[test]
    fn runtime_backend_selects_best() {
        let backend = RuntimeBackend::best_available();
        // Should always succeed regardless of which variant is selected
        assert!(backend.validate_dimension(128).is_ok());
        assert!(backend.validate_dimension(7).is_err());
    }

    #[test]
    fn runtime_backend_fwht_correctness() {
        let backend = RuntimeBackend::best_available();
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = original.clone();
        backend.fwht_normalized_inplace(&mut data);
        backend.fwht_normalized_inplace(&mut data);
        for (a, b) in original.iter().zip(&data) {
            assert!((a - b).abs() < 1e-5, "RuntimeBackend FWHT roundtrip failed");
        }
    }

    #[test]
    fn runtime_backend_dot_product() {
        let backend = RuntimeBackend::best_available();
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let result = backend.dot_product(&a, &b);
        assert!((result - 70.0).abs() < 1e-6);
    }

    #[test]
    fn simd_matches_scalar_fwht() {
        let scalar = ScalarBackend;
        let simd = SimdBackend;
        let mut data_s = vec![1.0f32, -2.0, 3.0, -1.0, 0.5, 1.5, -0.5, 2.0,
                              4.0, -3.0, 2.5, 0.0, -1.5, 3.5, -2.5, 1.0];
        let mut data_v = data_s.clone();
        scalar.fwht_normalized_inplace(&mut data_s);
        simd.fwht_normalized_inplace(&mut data_v);
        for (a, b) in data_s.iter().zip(&data_v) {
            assert!((a - b).abs() < 1e-6,
                "SIMD/scalar FWHT mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn simd_matches_scalar_dot_product() {
        let scalar = ScalarBackend;
        let simd = SimdBackend;
        let a = vec![1.0f32, -2.0, 3.0, -1.0, 0.5, 1.5, -0.5, 2.0];
        let b = vec![4.0f32, -3.0, 2.5, 0.0, -1.5, 3.5, -2.5, 1.0];
        let s_result = scalar.dot_product(&a, &b);
        let v_result = simd.dot_product(&a, &b);
        assert!((s_result - v_result).abs() < 1e-6,
            "SIMD/scalar dot mismatch: {s_result} vs {v_result}");
    }
}
