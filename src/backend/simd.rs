//! SIMD-accelerated backend implementations.
//!
//! ## Tiers (fastest first, auto-selected by `RuntimeBackend::best_available`)
//!
//! | Tier | Target | Width | Key ops |
//! |---|---|---|---|
//! | `Avx512Backend` | x86_64 + AVX-512F | 16 f32/cycle | FWHT, dot (FMA), sign-flip |
//! | `SimdBackend` (FMA) | x86_64 + AVX2+FMA | 8 f32/cycle | dot (fused mul-add) |
//! | `SimdBackend` (AVX2) | x86_64 + AVX2 | 8 f32/cycle | FWHT, dot, sign-flip |
//! | `SimdBackend` (NEON) | aarch64 | 4 f32/cycle | FWHT, dot (FMA), sign-flip |
//! | `ScalarBackend` | all | 1 f32/cycle | fallback |
//!
//! # Safety Verification
//!
//! Miri cannot interpret platform-specific SIMD intrinsics.
//! Safety is verified through:
//! 1. SAFETY documentation on every unsafe block
//! 2. Equivalence tests: SIMD output matches scalar output for all dimensions
//! 3. Roundtrip tests: FWHT applied twice recovers original vector
//! 4. Bounds checks: `debug_assert!` on all size assumptions
//! 5. Unaligned loads only: `_mm256_loadu_ps`, `_mm512_loadu_ps`, `vld1q_f32`

use crate::backend::{Backend, ScalarBackend};
use crate::error::{Result, TurboQuantError};

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

/// AVX2 sign-flip via XOR on the float sign bit.
///
/// `sign_masks[i]` is `0x80000000` when the sign is −1, else `0`.
/// Processes 8 floats per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_signs_avx2(data: &mut [f32], sign_masks: &[u32]) {
    use std::arch::x86_64::*;

    debug_assert_eq!(data.len(), sign_masks.len());
    let n = data.len();
    let mut i = 0usize;

    while i + 8 <= n {
        // SAFETY:
        // 1. AVX2 available: caller checked via is_x86_feature_detected!
        // 2. Bounds: i + 8 <= n checked in loop condition
        // 3. Alignment: using unaligned loads/stores
        let v = _mm256_loadu_ps(data.as_ptr().add(i));
        let m = _mm256_loadu_si256(sign_masks.as_ptr().add(i) as *const __m256i);
        // XOR float bits with sign mask — flips sign bit where mask = 0x80000000
        let result = _mm256_xor_ps(v, _mm256_castsi256_ps(m));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), result);
        i += 8;
    }
    // Scalar tail
    for j in i..n {
        data[j] = f32::from_bits(data[j].to_bits() ^ sign_masks[j]);
    }
}

/// AVX2 dot product — mul+add (no FMA). 8 f32/cycle.
///
/// Use `dot_product_avx2_fma` on Haswell+ for better throughput.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let mut sum_vec = _mm256_setzero_ps();
    let mut i = 0usize;

    // Process 8 elements at a time
    while i + 8 <= n {
        // SAFETY:
        // 1. AVX2 available: caller checked via is_x86_feature_detected!
        // 2. Bounds: i + 8 <= n checked in loop condition
        // 3. Alignment: using _mm256_loadu_ps (unaligned load)
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(va, vb));
        i += 8;
    }

    // Horizontal reduction: store 8 lanes to array and sum
    let mut temp = [0.0f32; 8];
    // SAFETY: temp is exactly 8 f32s; unaligned store
    _mm256_storeu_ps(temp.as_mut_ptr(), sum_vec);
    let mut result: f32 = temp.iter().sum();

    while i < n {
        result += a[i] * b[i];
        i += 1;
    }
    result
}

/// AVX2 + FMA dot product. Uses `_mm256_fmadd_ps` for fused multiply-add.
/// Reduces retirement pressure: 1 μop instead of 2. ~10-15% faster on Haswell+.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;

    while i + 8 <= n {
        // SAFETY:
        // 1. AVX2+FMA available: caller confirmed both features
        // 2. Bounds: i + 8 <= n
        // 3. Alignment: unaligned loads
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(va, vb, acc); // acc += va * vb (fused)
        i += 8;
    }

    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), acc);
    let mut result: f32 = temp.iter().sum();

    while i < n {
        result += a[i] * b[i];
        i += 1;
    }
    result
}

/// NEON sign-flip via XOR on the float sign bit. Processes 4 floats/iter.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn apply_signs_neon(data: &mut [f32], sign_masks: &[u32]) {
    use std::arch::aarch64::*;

    debug_assert_eq!(data.len(), sign_masks.len());
    let n = data.len();
    let mut i = 0usize;

    while i + 4 <= n {
        // SAFETY:
        // 1. NEON available: caller checked
        // 2. Bounds: i + 4 <= n
        // 3. vld1q_f32 / vld1q_u32 do not require alignment
        let v = vreinterpretq_u32_f32(vld1q_f32(data.as_ptr().add(i)));
        let m = vld1q_u32(sign_masks.as_ptr().add(i));
        let result = vreinterpretq_f32_u32(veorq_u32(v, m));
        vst1q_f32(data.as_mut_ptr().add(i), result);
        i += 4;
    }
    for j in i..n {
        data[j] = f32::from_bits(data[j].to_bits() ^ sign_masks[j]);
    }
}

/// NEON dot product with FMA (`vmlaq_f32`). 4 f32/cycle.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let mut sum_vec = vdupq_n_f32(0.0);
    let mut i = 0usize;

    while i + 4 <= n {
        // SAFETY:
        // 1. NEON available: caller checked via is_aarch64_feature_detected!
        // 2. Bounds: i + 4 <= n
        // 3. vld1q_f32 does not require alignment
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        sum_vec = vmlaq_f32(sum_vec, va, vb); // fused multiply-add
        i += 4;
    }

    // SAFETY: NEON available; operating on register value
    let result = vaddvq_f32(sum_vec);
    let mut tail = result;
    while i < n {
        tail += a[i] * b[i];
        i += 1;
    }
    tail
}

// ── AVX-512 kernels ──────────────────────────────────────────────────────────
// Processes 16 f32 per cycle. Requires AVX-512F (Intel Sapphire Rapids,
// AMD Zen 4, and newer).  Gated on both the feature flag and runtime detection.

/// AVX-512F unnormalized FWHT. 16 f32 butterflies per iteration when stride ≥ 16.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fwht_inplace_avx512f(data: &mut [f32]) {
    use std::arch::x86_64::*;

    debug_assert!(data.len().is_power_of_two());
    debug_assert!(data.len() >= 2);

    let n = data.len();
    let mut step = 1usize;

    while step < n {
        let mut i = 0usize;
        while i < n {
            if step >= 16 {
                let mut j = 0usize;
                while j + 16 <= step {
                    // SAFETY:
                    // 1. AVX-512F confirmed by caller
                    // 2. Bounds: j + 16 <= step, i + 2*step <= n
                    // 3. Unaligned loads
                    let a_ptr = data.as_ptr().add(i + j);
                    let b_ptr = data.as_ptr().add(i + j + step);
                    let a = _mm512_loadu_ps(a_ptr);
                    let b = _mm512_loadu_ps(b_ptr);
                    _mm512_storeu_ps(data.as_mut_ptr().add(i + j),        _mm512_add_ps(a, b));
                    _mm512_storeu_ps(data.as_mut_ptr().add(i + j + step), _mm512_sub_ps(a, b));
                    j += 16;
                }
                // Mop up remaining elements with AVX2 (step may not be multiple of 16)
                let mut j2 = (step / 16) * 16;
                while j2 + 8 <= step {
                    let a_ptr = data.as_ptr().add(i + j2);
                    let b_ptr = data.as_ptr().add(i + j2 + step);
                    let a = _mm256_loadu_ps(a_ptr);
                    let b = _mm256_loadu_ps(b_ptr);
                    _mm256_storeu_ps(data.as_mut_ptr().add(i + j2),        _mm256_add_ps(a, b));
                    _mm256_storeu_ps(data.as_mut_ptr().add(i + j2 + step), _mm256_sub_ps(a, b));
                    j2 += 8;
                }
                for j3 in j2..step {
                    let a_val = data[i + j3];
                    let b_val = data[i + j3 + step];
                    data[i + j3]        = a_val + b_val;
                    data[i + j3 + step] = a_val - b_val;
                }
            } else if step >= 8 {
                let mut j = 0usize;
                while j + 8 <= step {
                    let a_ptr = data.as_ptr().add(i + j);
                    let b_ptr = data.as_ptr().add(i + j + step);
                    let a = _mm256_loadu_ps(a_ptr);
                    let b = _mm256_loadu_ps(b_ptr);
                    _mm256_storeu_ps(data.as_mut_ptr().add(i + j),        _mm256_add_ps(a, b));
                    _mm256_storeu_ps(data.as_mut_ptr().add(i + j + step), _mm256_sub_ps(a, b));
                    j += 8;
                }
            } else {
                for j in 0..step {
                    let a_val = data[i + j];
                    let b_val = data[i + j + step];
                    data[i + j]        = a_val + b_val;
                    data[i + j + step] = a_val - b_val;
                }
            }
            i += 2 * step;
        }
        step <<= 1;
    }
}

/// AVX-512F + FMA dot product. 16 f32/cycle via `_mm512_fmadd_ps`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_product_avx512_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(a.len(), b.len());

    let n = a.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0usize;

    while i + 16 <= n {
        // SAFETY: AVX-512F confirmed; bounds checked; unaligned loads
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        acc = _mm512_fmadd_ps(va, vb, acc);
        i += 16;
    }

    // Horizontal reduction
    let result = _mm512_reduce_add_ps(acc);

    // Scalar tail
    let mut tail = result;
    while i < n {
        tail += a[i] * b[i];
        i += 1;
    }
    tail
}

/// AVX-512F sign-flip via `_mm512_xor_ps`. 16 floats/cycle.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_signs_avx512f(data: &mut [f32], sign_masks: &[u32]) {
    use std::arch::x86_64::*;

    debug_assert_eq!(data.len(), sign_masks.len());
    let n = data.len();
    let mut i = 0usize;

    while i + 16 <= n {
        // SAFETY: AVX-512F confirmed; bounds i+16<=n; unaligned
        let v = _mm512_loadu_ps(data.as_ptr().add(i));
        let m = _mm512_loadu_si512(sign_masks.as_ptr().add(i) as *const _);
        let result = _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(v), m));
        _mm512_storeu_ps(data.as_mut_ptr().add(i), result);
        i += 16;
    }
    // Mop up with AVX2
    while i + 8 <= n {
        use std::arch::x86_64::*;
        let v = _mm256_loadu_ps(data.as_ptr().add(i));
        let m = _mm256_loadu_si256(sign_masks.as_ptr().add(i) as *const __m256i);
        let result = _mm256_xor_ps(v, _mm256_castsi256_ps(m));
        _mm256_storeu_ps(data.as_mut_ptr().add(i), result);
        i += 8;
    }
    for j in i..n {
        data[j] = f32::from_bits(data[j].to_bits() ^ sign_masks[j]);
    }
}

/// AVX2 / NEON SIMD backend.
///
/// Implements FWHT, dot product (with FMA when available), and sign-flip
/// via vectorized XOR. Auto-detects the best available sub-feature at runtime.
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

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 confirmed by is_x86_feature_detected!
                unsafe { fwht_inplace_avx2(data); }
                let scale = 1.0 / (data.len() as f32).sqrt();
                data.iter_mut().for_each(|x| *x *= scale);
                return;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                // SAFETY: NEON confirmed by is_aarch64_feature_detected!
                unsafe { fwht_inplace_neon(data); }
                let scale = 1.0 / (data.len() as f32).sqrt();
                data.iter_mut().for_each(|x| *x *= scale);
                return;
            }
        }

        crate::hadamard::fwht_normalized_inplace(data);
    }

    #[inline]
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            // Prefer FMA when available (Haswell+): fused mul-add, fewer μops
            if is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2") {
                // SAFETY: both AVX2 and FMA confirmed
                return unsafe { dot_product_avx2_fma(a, b) };
            }
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 confirmed
                return unsafe { dot_product_avx2(a, b) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                // SAFETY: NEON confirmed
                return unsafe { dot_product_neon(a, b) };
            }
        }

        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    #[inline]
    fn apply_signs(&self, data: &mut [f32], sign_masks: &[u32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 confirmed
                unsafe { apply_signs_avx2(data, sign_masks); }
                return;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                // SAFETY: NEON confirmed
                unsafe { apply_signs_neon(data, sign_masks); }
                return;
            }
        }

        // Scalar fallback (also the default in the trait)
        for (x, &mask) in data.iter_mut().zip(sign_masks) {
            *x = f32::from_bits(x.to_bits() ^ mask);
        }
    }

    fn validate_dimension(&self, dim: usize) -> Result<()> {
        if !dim.is_power_of_two() {
            Err(TurboQuantError::DimensionNotPowerOfTwo(dim))
        } else {
            Ok(())
        }
    }
}

// ── Avx512Backend ────────────────────────────────────────────────────────────

/// AVX-512F backend. Processes 16 f32 per cycle.
///
/// Selected automatically on Intel Sapphire Rapids, AMD Zen 4, and newer CPUs.
/// Gracefully degrades to AVX2 for butterfly strides smaller than 16.
#[derive(Clone, Debug, Default)]
pub struct Avx512Backend;

impl Backend for Avx512Backend {
    #[inline]
    fn fwht_normalized_inplace(&self, data: &mut [f32]) {
        assert!(
            data.len().is_power_of_two(),
            "FWHT requires power-of-two length, got {}",
            data.len()
        );

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F confirmed
                unsafe { fwht_inplace_avx512f(data); }
                let scale = 1.0 / (data.len() as f32).sqrt();
                data.iter_mut().for_each(|x| *x *= scale);
                return;
            }
        }

        // Graceful fallback to AVX2 / scalar
        SimdBackend.fwht_normalized_inplace(data);
    }

    #[inline]
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F confirmed (implies FMA on all AVX-512F CPUs)
                return unsafe { dot_product_avx512_fma(a, b) };
            }
        }
        SimdBackend.dot_product(a, b)
    }

    #[inline]
    fn apply_signs(&self, data: &mut [f32], sign_masks: &[u32]) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F confirmed
                unsafe { apply_signs_avx512f(data, sign_masks); }
                return;
            }
        }
        SimdBackend.apply_signs(data, sign_masks);
    }

    fn validate_dimension(&self, dim: usize) -> Result<()> {
        if !dim.is_power_of_two() {
            Err(TurboQuantError::DimensionNotPowerOfTwo(dim))
        } else {
            Ok(())
        }
    }
}

/// Runtime-selected backend — auto-detects the fastest available tier.
///
/// Tiers (best first): `Avx512` → `Simd` (AVX2/FMA/NEON) → `Scalar`.
/// Uses static enum dispatch — zero dynamic-dispatch overhead.
#[derive(Clone, Debug)]
pub enum RuntimeBackend {
    /// Pure scalar — no SIMD.
    Scalar(ScalarBackend),
    /// AVX2 / FMA (x86_64) or NEON (aarch64).
    Simd(SimdBackend),
    /// AVX-512F — 16 f32/cycle on Intel SPR / AMD Zen 4.
    Avx512(Avx512Backend),
}

impl RuntimeBackend {
    /// Detect CPU features and return the fastest available backend.
    pub fn best_available() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return RuntimeBackend::Avx512(Avx512Backend);
            }
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

    /// Return a human-readable description of the selected tier.
    pub fn name(&self) -> &'static str {
        match self {
            RuntimeBackend::Scalar(_) => "scalar",
            RuntimeBackend::Simd(_)   => "simd (avx2/fma/neon)",
            RuntimeBackend::Avx512(_) => "avx512f",
        }
    }
}

impl Backend for RuntimeBackend {
    #[inline]
    fn fwht_normalized_inplace(&self, data: &mut [f32]) {
        match self {
            RuntimeBackend::Scalar(bk) => bk.fwht_normalized_inplace(data),
            RuntimeBackend::Simd(bk)   => bk.fwht_normalized_inplace(data),
            RuntimeBackend::Avx512(bk) => bk.fwht_normalized_inplace(data),
        }
    }

    #[inline]
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            RuntimeBackend::Scalar(bk) => bk.dot_product(a, b),
            RuntimeBackend::Simd(bk)   => bk.dot_product(a, b),
            RuntimeBackend::Avx512(bk) => bk.dot_product(a, b),
        }
    }

    #[inline]
    fn apply_signs(&self, data: &mut [f32], sign_masks: &[u32]) {
        match self {
            RuntimeBackend::Scalar(bk) => bk.apply_signs(data, sign_masks),
            RuntimeBackend::Simd(bk)   => bk.apply_signs(data, sign_masks),
            RuntimeBackend::Avx512(bk) => bk.apply_signs(data, sign_masks),
        }
    }

    fn validate_dimension(&self, dim: usize) -> crate::error::Result<()> {
        match self {
            RuntimeBackend::Scalar(bk) => bk.validate_dimension(dim),
            RuntimeBackend::Simd(bk)   => bk.validate_dimension(dim),
            RuntimeBackend::Avx512(bk) => bk.validate_dimension(dim),
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

    #[test]
    fn simd_matches_scalar_dot_product_large() {
        let scalar = ScalarBackend;
        let simd = SimdBackend;
        // Test with dim=128 (typical head dimension, 16 AVX2 iterations)
        let a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).cos()).collect();
        let s_result = scalar.dot_product(&a, &b);
        let v_result = simd.dot_product(&a, &b);
        assert!((s_result - v_result).abs() < 1e-4,
            "SIMD/scalar dot mismatch for dim=128: {s_result} vs {v_result}");
    }

    #[test]
    fn simd_matches_scalar_fwht_dim128() {
        let scalar = ScalarBackend;
        let simd = SimdBackend;
        let mut data_s: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut data_v = data_s.clone();
        scalar.fwht_normalized_inplace(&mut data_s);
        simd.fwht_normalized_inplace(&mut data_v);
        for (idx, (a, b)) in data_s.iter().zip(&data_v).enumerate() {
            assert!((a - b).abs() < 1e-4,
                "SIMD/scalar FWHT mismatch at index {idx} for dim=128: {a} vs {b}");
        }
    }

    #[test]
    fn simd_dot_product_empty_and_small() {
        let backend = SimdBackend;
        // Empty
        assert!((backend.dot_product(&[], &[]) - 0.0).abs() < 1e-10);
        // 1 element
        assert!((backend.dot_product(&[3.0], &[4.0]) - 12.0).abs() < 1e-6);
        // 7 elements (not multiple of 4 or 8)
        let a = vec![1.0f32; 7];
        let b = vec![2.0f32; 7];
        assert!((backend.dot_product(&a, &b) - 14.0).abs() < 1e-6);
    }

    #[test]
    fn simd_inner_product_accuracy() {
        // Verify that using SIMD backend in a PolarQuant-like scenario
        // maintains <2% relative error on inner products
        let scalar = ScalarBackend;
        let simd = SimdBackend;
        let dim = 128;
        // Generate random-ish unit vectors
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.31).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.47).cos()).collect();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let a_unit: Vec<f32> = a.iter().map(|x| x / norm_a).collect();
        let b_unit: Vec<f32> = b.iter().map(|x| x / norm_b).collect();

        let scalar_dot = scalar.dot_product(&a_unit, &b_unit);
        let simd_dot = simd.dot_product(&a_unit, &b_unit);
        let relative_error = ((scalar_dot - simd_dot) / scalar_dot).abs();
        assert!(relative_error < 0.02,
            "SIMD inner product error {:.4}% exceeds 2% threshold (scalar={scalar_dot}, simd={simd_dot})",
            relative_error * 100.0);
    }

    #[test]
    fn simd_fwht_all_power_of_two_dims() {
        let scalar = ScalarBackend;
        let simd = SimdBackend;
        for exp in 1..=10 {  // dims 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
            let dim = 1 << exp;
            let mut data_s: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.13).sin()).collect();
            let mut data_v = data_s.clone();
            scalar.fwht_normalized_inplace(&mut data_s);
            simd.fwht_normalized_inplace(&mut data_v);
            for (idx, (a, b)) in data_s.iter().zip(&data_v).enumerate() {
                assert!((a - b).abs() < 1e-3,
                    "SIMD/scalar FWHT mismatch at dim={dim}, index={idx}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn simd_dot_product_all_power_of_two_dims() {
        let scalar = ScalarBackend;
        let simd = SimdBackend;
        for exp in 0..=10 {  // dims 1, 2, 4, ..., 1024
            let dim = 1 << exp;
            let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.07).cos()).collect();
            let s = scalar.dot_product(&a, &b);
            let v = simd.dot_product(&a, &b);
            assert!((s - v).abs() < 1e-3 * dim as f32,
                "SIMD/scalar dot mismatch at dim={dim}: {s} vs {v}");
        }
    }

    #[test]
    fn simd_fwht_roundtrip_all_dims() {
        let simd = SimdBackend;
        for exp in 1..=10 {
            let dim = 1 << exp;
            let original: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.17).cos()).collect();
            let mut data = original.clone();
            simd.fwht_normalized_inplace(&mut data);
            simd.fwht_normalized_inplace(&mut data);
            for (idx, (a, b)) in original.iter().zip(&data).enumerate() {
                assert!((a - b).abs() < 1e-3,
                    "SIMD FWHT roundtrip failed at dim={dim}, index={idx}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn simd_fwht_norm_preservation() {
        let simd = SimdBackend;
        for exp in 1..=10 {
            let dim = 1 << exp;
            let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.11).sin()).collect();
            let norm_before: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            let mut transformed = data.clone();
            simd.fwht_normalized_inplace(&mut transformed);
            let norm_after: f32 = transformed.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm_before - norm_after).abs() < 1e-3,
                "Norm not preserved at dim={dim}: {norm_before} vs {norm_after}");
        }
    }

    // ── apply_signs tests ────────────────────────────────────────────────────

    fn make_sign_masks(signs: &[i8]) -> Vec<u32> {
        signs.iter().map(|&s| if s < 0 { 0x8000_0000u32 } else { 0 }).collect()
    }

    #[test]
    fn apply_signs_scalar_default() {
        let backend = ScalarBackend;
        let original = vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let signs: Vec<i8> = vec![1, -1, 1, -1, -1, 1, -1, 1];
        let masks = make_sign_masks(&signs);
        let mut data = original.clone();
        backend.apply_signs(&mut data, &masks);
        let manually: Vec<f32> = original.iter().zip(&signs)
            .map(|(&x, &s)| x * s as f32)
            .collect();
        for (a, b) in data.iter().zip(&manually) {
            assert!((a - b).abs() < 1e-7, "scalar apply_signs: {a} vs {b}");
        }
    }

    #[test]
    fn apply_signs_simd_matches_scalar() {
        let scalar  = ScalarBackend;
        let simd    = SimdBackend;
        let original = vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0,
                            0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5];
        let signs: Vec<i8> = vec![1,-1,1,-1,-1,1,-1,1, -1,1,-1,1,1,-1,1,-1];
        let masks = make_sign_masks(&signs);

        let mut ds = original.clone();
        let mut dv = original.clone();
        scalar.apply_signs(&mut ds, &masks);
        simd.apply_signs(&mut dv, &masks);

        for (idx, (a, b)) in ds.iter().zip(&dv).enumerate() {
            assert!((a - b).abs() < 1e-7,
                "apply_signs SIMD/scalar mismatch at {idx}: {a} vs {b}");
        }
    }

    #[test]
    fn apply_signs_simd_dim128() {
        let scalar = ScalarBackend;
        let simd   = SimdBackend;
        let original: Vec<f32> = (0..128).map(|i| (i as f32 * 0.23).sin()).collect();
        let signs: Vec<i8> = (0..128).map(|i| if i % 3 == 0 { -1i8 } else { 1i8 }).collect();
        let masks = make_sign_masks(&signs);

        let mut ds = original.clone();
        let mut dv = original.clone();
        scalar.apply_signs(&mut ds, &masks);
        simd.apply_signs(&mut dv, &masks);

        for (idx, (a, b)) in ds.iter().zip(&dv).enumerate() {
            assert!((a - b).abs() < 1e-7,
                "apply_signs dim128 SIMD/scalar mismatch at {idx}: {a} vs {b}");
        }
    }

    #[test]
    fn apply_signs_runtime_matches_scalar() {
        let scalar  = ScalarBackend;
        let runtime = RuntimeBackend::best_available();
        let original: Vec<f32> = (0..256).map(|i| (i as f32 * 0.07).cos()).collect();
        let signs: Vec<i8> = (0..256).map(|i| if i % 2 == 0 { 1i8 } else { -1i8 }).collect();
        let masks = make_sign_masks(&signs);

        let mut ds = original.clone();
        let mut dv = original.clone();
        scalar.apply_signs(&mut ds, &masks);
        runtime.apply_signs(&mut dv, &masks);

        for (idx, (a, b)) in ds.iter().zip(&dv).enumerate() {
            assert!((a - b).abs() < 1e-7,
                "apply_signs runtime/scalar mismatch at {idx}: {a} vs {b}");
        }
    }

    #[test]
    fn apply_signs_involution() {
        // Applying the same sign mask twice should recover the original vector.
        let backend = SimdBackend;
        let original: Vec<f32> = (0..128).map(|i| i as f32 * 0.1 - 6.4).collect();
        let signs: Vec<i8> = (0..128).map(|i| if i % 5 == 0 { -1i8 } else { 1i8 }).collect();
        let masks = make_sign_masks(&signs);

        let mut data = original.clone();
        backend.apply_signs(&mut data, &masks);
        backend.apply_signs(&mut data, &masks); // apply twice = identity

        for (idx, (a, b)) in original.iter().zip(&data).enumerate() {
            assert!((a - b).abs() < 1e-6,
                "apply_signs involution failed at {idx}: {a} vs {b}");
        }
    }

    // ── Avx512Backend tests ──────────────────────────────────────────────────

    #[test]
    fn avx512_backend_implements_trait() {
        let backend = Avx512Backend;
        assert!(backend.validate_dimension(128).is_ok());
        assert!(backend.validate_dimension(7).is_err());
    }

    #[test]
    fn avx512_fwht_roundtrip_all_dims() {
        let backend = Avx512Backend;
        for exp in 1..=10 {
            let dim = 1 << exp;
            let original: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.19).sin()).collect();
            let mut data = original.clone();
            backend.fwht_normalized_inplace(&mut data);
            backend.fwht_normalized_inplace(&mut data);
            for (idx, (a, b)) in original.iter().zip(&data).enumerate() {
                assert!((a - b).abs() < 1e-3,
                    "Avx512Backend FWHT roundtrip failed at dim={dim}, idx={idx}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn avx512_matches_scalar_fwht() {
        let scalar = ScalarBackend;
        let avx512 = Avx512Backend;
        for exp in 1..=10 {
            let dim = 1 << exp;
            let mut ds: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.13).cos()).collect();
            let mut dv = ds.clone();
            scalar.fwht_normalized_inplace(&mut ds);
            avx512.fwht_normalized_inplace(&mut dv);
            for (idx, (a, b)) in ds.iter().zip(&dv).enumerate() {
                assert!((a - b).abs() < 1e-3,
                    "Avx512/scalar FWHT mismatch at dim={dim}, idx={idx}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn avx512_dot_product_correctness() {
        let scalar = ScalarBackend;
        let avx512 = Avx512Backend;
        for exp in 0..=10 {
            let dim = 1 << exp;
            let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.11).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.17).cos()).collect();
            let s = scalar.dot_product(&a, &b);
            let v = avx512.dot_product(&a, &b);
            let tol = 1e-3 * (dim as f32).max(1.0);
            assert!((s - v).abs() < tol,
                "Avx512/scalar dot mismatch at dim={dim}: {s} vs {v}");
        }
    }

    #[test]
    fn avx512_apply_signs_matches_scalar() {
        let scalar = ScalarBackend;
        let avx512 = Avx512Backend;
        let original: Vec<f32> = (0..256).map(|i| (i as f32 * 0.31).sin()).collect();
        let signs: Vec<i8> = (0..256).map(|i| if i % 4 < 2 { 1i8 } else { -1i8 }).collect();
        let masks = make_sign_masks(&signs);
        let mut ds = original.clone();
        let mut dv = original.clone();
        scalar.apply_signs(&mut ds, &masks);
        avx512.apply_signs(&mut dv, &masks);
        for (idx, (a, b)) in ds.iter().zip(&dv).enumerate() {
            assert!((a - b).abs() < 1e-7,
                "Avx512/scalar apply_signs mismatch at {idx}: {a} vs {b}");
        }
    }

    // ── RuntimeBackend tier selection ────────────────────────────────────────

    #[test]
    fn runtime_backend_name_is_non_empty() {
        let name = RuntimeBackend::best_available().name();
        assert!(!name.is_empty());
    }

    #[test]
    fn runtime_backend_apply_signs_correctness() {
        let scalar  = ScalarBackend;
        let runtime = RuntimeBackend::best_available();
        let original: Vec<f32> = (0..128).map(|i| (i as f32 * 0.41).sin()).collect();
        let signs: Vec<i8> = (0..128).map(|i| if i % 3 == 0 { -1i8 } else { 1i8 }).collect();
        let masks = make_sign_masks(&signs);
        let mut ds = original.clone();
        let mut dv = original.clone();
        scalar.apply_signs(&mut ds, &masks);
        runtime.apply_signs(&mut dv, &masks);
        for (idx, (a, b)) in ds.iter().zip(&dv).enumerate() {
            assert!((a - b).abs() < 1e-7,
                "RuntimeBackend apply_signs mismatch at {idx}: {a} vs {b}");
        }
    }
}
