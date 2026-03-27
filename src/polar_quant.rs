//! PolarQuant — Stage 1 of TurboQuant (Algorithm 1 in the paper).
//!
//! ## Algorithm
//! **Quantize(x)**
//!   1. Compute and store ‖x‖₂.
//!   2. Normalize: x̂ = x / ‖x‖.
//!   3. Rotate:    ŷ = R x̂  (randomized Hadamard, see [`Rotation`]).
//!   4. Quantize:  for each coordinate ŷᵢ find nearest Lloyd-Max centroid.
//!   5. Bit-pack the indices.
//!
//! **Dequantize(q)**
//!   1. Unpack indices → centroid values ŷ̃.
//!   2. Rotate back:   x̃̂ = R^T ŷ̃.
//!   3. Rescale:       x̃ = ‖x‖ · x̃̂.
//!
//! **InnerProduct(query, key_qv)** — avoids full dequantization.
//!   ⟨q, x⟩ ≈ ‖x‖ · ⟨R q, dequant(R x̂)⟩  =  ‖x‖ · Σᵢ (Rq)ᵢ · centroid[idxᵢ]

use crate::{
    backend::{Backend, ScalarBackend},
    bitpack,
    codebook::Codebook,
    error::{Result, TurboQuantError},
    rotation::Rotation,
};
use std::cell::RefCell;

// ── Public data type ────────────────────────────────────────────────────────

/// A PolarQuant-compressed vector.
#[derive(Debug, Clone)]
pub struct QuantizedVector {
    /// L2 norm of the original (unquantized) vector.
    pub norm: f32,
    /// Bit-packed quantization indices.
    pub packed: Vec<u8>,
    /// Original dimension.
    pub dim: usize,
    /// Bit width used.
    pub bits: u8,
}

impl QuantizedVector {
    /// Compressed size in bytes (norm f32 + packed indices).
    #[must_use]
    pub fn byte_size(&self) -> usize {
        4 + self.packed.len()
    }

    /// Compression ratio relative to f32 storage.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        (self.dim * 4) as f32 / self.byte_size() as f32
    }
}

// ── PolarQuant ──────────────────────────────────────────────────────────────

/// Stage-1 TurboQuant quantizer: rotation + optimal scalar quantization.
#[derive(Debug)]
pub struct PolarQuant<B: Backend = ScalarBackend> {
    rotation: Rotation<B>,
    codebook: Codebook,
    backend: B,
    scratch: RefCell<Vec<f32>>,
}

impl<B: Backend> Clone for PolarQuant<B> {
    fn clone(&self) -> Self {
        let dim = self.rotation.dim;
        Self {
            rotation: self.rotation.clone(),
            codebook: self.codebook.clone(),
            backend: self.backend.clone(),
            scratch: RefCell::new(Vec::with_capacity(dim)),
        }
    }
}

impl PolarQuant<ScalarBackend> {
    /// Create a PolarQuant with the default scalar backend.
    ///
    /// - `dim`  — head dimension; **must be a power of two** (64, 128, 256, …)
    /// - `bits` — target bit-width per coordinate: 2, 3, or 4
    /// - `seed` — RNG seed for the rotation matrix
    pub fn new(dim: usize, bits: u8, seed: u64) -> Result<Self> {
        Self::new_with_backend(dim, bits, seed, ScalarBackend)
    }
}

impl<B: Backend> PolarQuant<B> {
    /// Create a PolarQuant with an explicit backend.
    pub fn new_with_backend(dim: usize, bits: u8, seed: u64, backend: B) -> Result<Self> {
        let rotation = Rotation::new_with_backend(dim, seed, backend.clone())?;
        let codebook = Codebook::new(bits, dim)?;
        let scratch = RefCell::new(Vec::with_capacity(dim));
        Ok(Self { rotation, codebook, backend, scratch })
    }

    // ── Quantize ──────────────────────────────────────────────────────────

    /// Compress a vector to a [`QuantizedVector`].
    #[must_use = "quantized vector should be stored or used"]
    pub fn quantize(&self, vec: &[f32]) -> Result<QuantizedVector> {
        let dim = self.rotation.dim;
        self.check_dim(vec.len())?;

        let norm: f32 = l2_norm(vec);

        // Normalize to unit sphere; if zero-vector, leave as-is.
        let mut rotated: Vec<f32> = if norm > f32::EPSILON {
            vec.iter().map(|&x| x / norm).collect()
        } else {
            vec![0.0f32; dim]
        };

        // Apply the randomized Hadamard rotation.
        self.rotation.apply(&mut rotated);

        // Scalar-quantize each coordinate independently.
        let indices = self.codebook.quantize_slice(&rotated);

        // Pack to compact bit representation.
        let packed = bitpack::pack(&indices, self.codebook.bits)?;

        Ok(QuantizedVector { norm, packed, dim, bits: self.codebook.bits })
    }

    // ── Dequantize ────────────────────────────────────────────────────────

    /// Reconstruct an approximate vector from a [`QuantizedVector`].
    #[must_use = "dequantized vector should be used"]
    pub fn dequantize(&self, qv: &QuantizedVector) -> Result<Vec<f32>> {
        self.check_dim(qv.dim)?;

        let indices  = bitpack::unpack(&qv.packed, qv.dim, qv.bits)?;
        let mut recon = self.codebook.dequantize_slice(&indices);

        // Invert the rotation.
        self.rotation.apply_inverse(&mut recon);

        // Rescale by the stored norm.
        recon.iter_mut().for_each(|x| *x *= qv.norm);

        Ok(recon)
    }

    // ── Inner product (fast path) ─────────────────────────────────────────

    /// Estimate ⟨query, key⟩ without full dequantization.
    ///
    /// This is the primary hot path for attention logit computation.
    /// The query is rotated once; the dot product is then computed
    /// against the codebook centroids indexed by the packed key.
    #[must_use = "inner product result should be used"]
    pub fn inner_product(&self, query: &[f32], key: &QuantizedVector) -> Result<f32> {
        self.check_dim(query.len())?;
        self.check_dim(key.dim)?;

        // Compute rotated query using scratch buffer (avoids allocation).
        // Scope the borrow_mut guard so it drops before we return.
        let dot = {
            let mut scratch = self.scratch.borrow_mut();
            scratch.clear();
            scratch.extend_from_slice(query);
            self.rotation.apply(&mut scratch);

            let indices = bitpack::unpack(&key.packed, key.dim, key.bits)?;

            scratch
                .iter()
                .zip(&indices)
                .map(|(&q, &idx)| q * self.codebook.dequantize_scalar(idx))
                .sum::<f32>()
        }; // borrow_mut guard dropped here

        // Scale by key norm (query norm does not factor in here;
        // the caller applies it via the standard softmax attention formula).
        Ok(dot * key.norm)
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    pub fn codebook(&self)  -> &Codebook { &self.codebook  }
    pub fn rotation(&self)  -> &Rotation<B> { &self.rotation  }
    pub fn dim(&self)       -> usize      { self.rotation.dim }
    pub fn bits(&self)      -> u8         { self.codebook.bits }

    // ── Private helpers ───────────────────────────────────────────────────

    #[inline]
    fn check_dim(&self, got: usize) -> Result<()> {
        let expected = self.rotation.dim;
        if got != expected {
            Err(TurboQuantError::DimensionMismatch { expected, got })
        } else {
            Ok(())
        }
    }
}

// ── Utilities ──────────────────────────────────────────────────────────────

#[inline]
pub(crate) fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pq(bits: u8) -> PolarQuant {
        PolarQuant::new(128, bits, 42).unwrap()
    }

    fn sine_vec(dim: usize, freq: f32) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 * freq).sin()).collect()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    #[test]
    fn quantize_preserves_byte_budget_3bit() {
        let pq = make_pq(3);
        let v  = sine_vec(128, 0.1);
        let qv = pq.quantize(&v).unwrap();
        // 4 bytes norm + 48 bytes indices
        assert_eq!(qv.byte_size(), 52, "expected 52 bytes for 3-bit 128-dim");
        assert!((qv.compression_ratio() - 128.0 * 4.0 / 52.0).abs() < 0.01);
    }

    #[test]
    fn cosine_similarity_3bit() {
        let pq   = make_pq(3);
        let orig = sine_vec(128, 0.1);
        let qv   = pq.quantize(&orig).unwrap();
        let recon = pq.dequantize(&qv).unwrap();

        let cosine: f32 = dot(&orig, &recon)
            / (l2_norm(&orig) * l2_norm(&recon));
        assert!(cosine > 0.98, "cosine similarity too low: {cosine:.4}");
    }

    #[test]
    fn inner_product_accuracy_3bit() {
        let pq    = make_pq(3);
        // Use vectors with a meaningful dot product (not near-orthogonal).
        let key   = sine_vec(128, 0.05);
        let query = sine_vec(128, 0.07);

        let true_ip = dot(&key, &query);
        let qv      = pq.quantize(&key).unwrap();
        let est_ip  = pq.inner_product(&query, &qv).unwrap();

        // Bound error relative to ||key||·||query|| — the theoretical maximum
        // inner product. This matches the paper's distortion measure and avoids
        // false failures when true_ip is near zero.
        let max_ip  = l2_norm(&key) * l2_norm(&query);
        let norm_err = (est_ip - true_ip).abs() / max_ip;
        assert!(norm_err < 0.05, "IP norm-relative error too large: {norm_err:.3}");
    }

    #[test]
    fn inner_product_matches_dequant() {
        let pq    = make_pq(4);
        let key   = sine_vec(128, 0.07);
        let query = sine_vec(128, 0.11);

        let qv        = pq.quantize(&key).unwrap();
        let recon     = pq.dequantize(&qv).unwrap();
        let ip_recon  = dot(&recon, &query);
        let ip_direct = pq.inner_product(&query, &qv).unwrap();

        // Should agree to floating-point precision.
        assert!((ip_recon - ip_direct).abs() < 1e-4,
            "methods disagree: {ip_recon} vs {ip_direct}");
    }

    #[test]
    fn dim_mismatch_errors() {
        let pq = make_pq(3);
        let v  = vec![0.0f32; 64];
        assert!(pq.quantize(&v).is_err());
    }

    #[test]
    fn zero_vector() {
        let pq = make_pq(3);
        let v  = vec![0.0f32; 128];
        let qv = pq.quantize(&v).unwrap();
        assert_eq!(qv.norm, 0.0);
        let recon = pq.dequantize(&qv).unwrap();
        assert!(l2_norm(&recon) < 1e-6);
    }

    #[test]
    fn inner_product_repeated_calls_no_panic() {
        let pq = make_pq(3);
        let key = sine_vec(128, 0.05);
        let query = sine_vec(128, 0.07);
        let qv = pq.quantize(&key).unwrap();

        // Call inner_product many times - should never panic from double borrow
        for _ in 0..1000 {
            let _ = pq.inner_product(&query, &qv).unwrap();
        }
    }

    #[test]
    fn quantize_after_inner_product() {
        let pq = make_pq(3);
        let key = sine_vec(128, 0.05);
        let query = sine_vec(128, 0.07);
        let qv = pq.quantize(&key).unwrap();

        // inner_product then quantize should work (no borrow conflict)
        let _ = pq.inner_product(&query, &qv).unwrap();
        let qv2 = pq.quantize(&query).unwrap();
        assert_eq!(qv2.dim, 128);
    }
}
