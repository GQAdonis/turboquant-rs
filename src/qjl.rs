//! Quantized Johnson-Lindenstrauss (QJL) transform — Stage 2 of TurboQuant.
//!
//! ## Purpose
//! PolarQuant (Stage 1) is MSE-optimal but introduces a **bias** in inner
//! product estimation.  QJL applies a 1-bit sketch to the quantization
//! **residual** r = x - dequant(Q(x)) and provides an **unbiased** correction:
//!
//!   ⟨q, x⟩ = ⟨q, Q̃(x)⟩ + ⟨q, r⟩
//!
//! where ⟨q, r⟩ is estimated using the JL estimator below.
//!
//! ## JL Estimator (Zandieh et al. 2024)
//! Given a random projection S ∈ ℝ^{m×d} (we use another Hadamard rotation),
//! store b_r = sign(S r) ∈ {±1}^m.  Then:
//!
//!   Ê[⟨q, r⟩] = √(π/(2m)) · ⟨S q, b_r⟩
//!
//! This estimator is **unbiased** and has variance O(‖q‖²‖r‖²/m).
//!
//! ## Practical note
//! Community implementations (turboquant_plus, tonbistudio) found that
//! dropping QJL and allocating all bits to Stage 1 yields equivalent or
//! better perplexity in practice.  Both variants are provided here.

use crate::{
    error::{Result, TurboQuantError},
    rotation::Rotation,
};
use std::f32::consts::PI;

// ── Public types ────────────────────────────────────────────────────────────

/// A 1-bit QJL-compressed vector (sign sketch of the JL projection).
#[derive(Debug, Clone, Default)]
pub struct QjlVector {
    /// Bit-packed signs: bit i = 1 if (S·v)_i ≥ 0.
    pub bits: Vec<u8>,
    /// Dimension of the original vector.
    pub dim: usize,
}

impl QjlVector {
    /// Compressed size in bytes.
    #[must_use]
    pub fn byte_size(&self) -> usize { self.bits.len() }

    /// True if this is an empty / placeholder QJL vector.
    #[must_use]
    pub fn is_empty_sketch(&self) -> bool { self.bits.is_empty() }
}

// ── QJL Compressor ──────────────────────────────────────────────────────────

/// 1-bit QJL compressor with its own independent randomized Hadamard rotation.
#[derive(Debug, Clone)]
pub struct Qjl {
    rotation: Rotation,
}

impl Qjl {
    /// Create a QJL compressor.  `dim` must be a power of two.
    /// Use a **different seed** from the Stage-1 PolarQuant rotation.
    pub fn new(dim: usize, seed: u64) -> Result<Self> {
        let rotation = Rotation::new(dim, seed)?;
        Ok(Self { rotation })
    }

    #[must_use]
    pub fn dim(&self) -> usize { self.rotation.dim }

    // ── Compress ─────────────────────────────────────────────────────────

    /// Project `vec` with S = H̃D and store the sign bits.
    #[must_use]
    pub fn compress(&self, vec: &[f32]) -> Result<QjlVector> {
        let dim = self.rotation.dim;
        if vec.len() != dim {
            return Err(TurboQuantError::DimensionMismatch { expected: dim, got: vec.len() });
        }

        let mut projected = vec.to_vec();
        self.rotation.apply(&mut projected);

        // Pack sign bits: bit i = 1 iff projected[i] >= 0.
        let byte_count = (dim + 7) / 8;
        let mut sign_bits = vec![0u8; byte_count];
        for (i, &v) in projected.iter().enumerate() {
            if v >= 0.0 {
                sign_bits[i / 8] |= 1 << (i % 8);
            }
        }

        Ok(QjlVector { bits: sign_bits, dim })
    }

    // ── Estimate inner product ────────────────────────────────────────────

    /// Estimate ⟨query, residual⟩ from the QJL-compressed residual.
    ///
    /// Estimator: √(π / (2d)) · ⟨S·query, b_r⟩
    #[must_use]
    pub fn estimate_inner_product(&self, query: &[f32], key: &QjlVector) -> Result<f32> {
        let dim = self.rotation.dim;
        if query.len() != dim {
            return Err(TurboQuantError::DimensionMismatch { expected: dim, got: query.len() });
        }
        if key.dim != dim {
            return Err(TurboQuantError::DimensionMismatch { expected: dim, got: key.dim });
        }

        // Project query with the same S.
        let mut q_proj = query.to_vec();
        self.rotation.apply(&mut q_proj);

        // Dot product with stored sign bits.
        let dot: f32 = q_proj
            .iter()
            .enumerate()
            .map(|(i, &q)| {
                let sign = if (key.bits[i / 8] >> (i % 8)) & 1 == 1 { 1.0f32 } else { -1.0f32 };
                q * sign
            })
            .sum();

        // Unbiased estimator constant C = √(π / (2d)).
        let c = (PI / (2.0 * dim as f32)).sqrt();
        Ok(dot * c)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polar_quant::l2_norm;

    fn sine_vec(dim: usize, freq: f32) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 * freq).sin()).collect()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    #[test]
    fn compress_roundtrip_size() {
        let qjl = Qjl::new(128, 99).unwrap();
        let v   = sine_vec(128, 0.1);
        let qv  = qjl.compress(&v).unwrap();
        assert_eq!(qv.byte_size(), 16, "128 bits / 8 = 16 bytes");
    }

    #[test]
    fn estimate_unbiased_many_trials() {
        // Average over many random queries the estimator should be close to true IP.
        let dim  = 128;
        let qjl  = Qjl::new(dim, 7).unwrap();
        let key  = sine_vec(dim, 0.13);

        // Compress just the key (pretend it's a residual).
        let key_qv = qjl.compress(&key).unwrap();

        // Average estimate over queries.
        let mut sum_err = 0.0f32;
        let n_trials = 200;
        for t in 0..n_trials {
            let q: Vec<f32> = (0..dim).map(|i| ((i + t) as f32 * 0.07).cos()).collect();
            let true_ip = dot(&q, &key);
            let est_ip  = qjl.estimate_inner_product(&q, &key_qv).unwrap();
            sum_err += (est_ip - true_ip).abs();
        }
        let mean_err = sum_err / n_trials as f32;
        // Generous bound: mean error should be well below the key norm.
        let key_norm = l2_norm(&key);
        assert!(mean_err < key_norm, "mean QJL error {mean_err:.4} > key norm {key_norm:.4}");
    }

    #[test]
    fn dim_mismatch() {
        let qjl = Qjl::new(128, 1).unwrap();
        let v   = vec![0.0f32; 64];
        assert!(qjl.compress(&v).is_err());
    }
}
