//! SIMD-accelerated backend implementation.
//!
//! Uses AVX2 on x86_64 and NEON on aarch64 for vectorized FWHT
//! and dot product operations. Falls back to scalar for small
//! dimensions or unsupported strides.

use crate::backend::{Backend, ScalarBackend};
use crate::error::{Result, TurboQuantError};
use crate::hadamard::fwht_normalized_inplace;

/// AVX2-accelerated unnormalized FWHT.
///
/// Processes 8 f32 butterflies per iteration when stride >= 8.
/// Falls back to scalar for strides 1, 2, 4 where vectorization
/// would mix incorrect butterfly pairs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fwht_inplace_avx2(data: &mut [f32]) {
    use std::arch::x86_64::*;

    debug_assert!(data.len().is_power_of_two());
    debug_assert!(data.len() >= 2);

    let n = data.len();
    let mut step = 1usize;

    while step < n {
        let mut i = 0usize;
        while i < n {
            if step >= 8 {
                // SIMD path: process 8 butterflies at a time
                let mut j = 0usize;
                while j + 8 <= step {
                    // SAFETY:
                    // 1. AVX2 available: caller checked via is_x86_feature_detected!
                    // 2. Bounds: i + j + step + 8 <= n because j + 8 <= step
                    //    and i + 2*step <= n (loop invariant)
                    // 3. Alignment: using _mm256_loadu_ps (unaligned load)
                    let a_ptr = data.as_ptr().add(i + j);
                    let b_ptr = data.as_ptr().add(i + j + step);

                    let a = _mm256_loadu_ps(a_ptr);
                    let b = _mm256_loadu_ps(b_ptr);

                    let sum = _mm256_add_ps(a, b);
                    let diff = _mm256_sub_ps(a, b);

                    _mm256_storeu_ps(data.as_mut_ptr().add(i + j), sum);
                    _mm256_storeu_ps(data.as_mut_ptr().add(i + j + step), diff);

                    j += 8;
                }
            } else {
                // Scalar fallback for small strides (1, 2, 4)
                // Vectorizing these requires complex shuffles that negate SIMD benefit
                for j in 0..step {
                    let a_val = data[i + j];
                    let b_val = data[i + j + step];
                    data[i + j] = a_val + b_val;
                    data[i + j + step] = a_val - b_val;
                }
            }
            i += 2 * step;
        }
        step <<= 1;
    }
}

/// NEON-accelerated unnormalized FWHT.
///
/// Processes 4 f32 butterflies per iteration when stride >= 4.
/// Falls back to scalar for strides 1, 2.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fwht_inplace_neon(data: &mut [f32]) {
    use std::arch::aarch64::*;

    debug_assert!(data.len().is_power_of_two());
    debug_assert!(data.len() >= 2);

    let n = data.len();
    let mut step = 1usize;

    while step < n {
        let mut i = 0usize;
        while i < n {
            if step >= 4 {
                // SIMD path: process 4 butterflies at a time
                let mut j = 0usize;
                while j + 4 <= step {
                    // SAFETY:
                    // 1. NEON available: caller checked via is_aarch64_feature_detected!
                    // 2. Bounds: i + j + step + 4 <= n because j + 4 <= step
                    //    and i + 2*step <= n (loop invariant)
                    // 3. Alignment: vld1q_f32 does not require alignment
                    let a_ptr = data.as_ptr().add(i + j);
                    let b_ptr = data.as_ptr().add(i + j + step);

                    let a = vld1q_f32(a_ptr);
                    let b = vld1q_f32(b_ptr);

                    let sum = vaddq_f32(a, b);
                    let diff = vsubq_f32(a, b);

                    vst1q_f32(data.as_mut_ptr().add(i + j), sum);
                    vst1q_f32(data.as_mut_ptr().add(i + j + step), diff);

                    j += 4;
                }
            } else {
                // Scalar fallback for strides 1, 2
                for j in 0..step {
                    let a_val = data[i + j];
                    let b_val = data[i + j + step];
                    data[i + j] = a_val + b_val;
                    data[i + j + step] = a_val - b_val;
                }
            }
            i += 2 * step;
        }
        step <<= 1;
    }
}

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
        assert!(
            data.len().is_power_of_two(),
            "FWHT requires power-of-two length, got {}",
            data.len()
        );

        // Dispatch to SIMD or scalar based on CPU features
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 availability confirmed by is_x86_feature_detected!
                unsafe { fwht_inplace_avx2(data); }
                let scale = 1.0 / (data.len() as f32).sqrt();
                data.iter_mut().for_each(|x| *x *= scale);
                return;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                // SAFETY: NEON availability confirmed by is_aarch64_feature_detected!
                unsafe { fwht_inplace_neon(data); }
                let scale = 1.0 / (data.len() as f32).sqrt();
                data.iter_mut().for_each(|x| *x *= scale);
                return;
            }
        }

        // Scalar fallback
        crate::hadamard::fwht_normalized_inplace(data);
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
