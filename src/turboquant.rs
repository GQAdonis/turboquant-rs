//! Full TurboQuant — combines PolarQuant (Stage 1) and QJL (Stage 2).
//!
//! Two variants are provided:
//!
//! | Variant            | Description                                   | Recommendation |
//! |--------------------|-----------------------------------------------|----------------|
//! | **MSE** (`_mse`)  | All `b` bits go to Lloyd-Max PolarQuant        | ✅ Default     |
//! | **Prod** (`_prod`) | `(b-1)`-bit PolarQuant + 1-bit QJL correction | Use when unbiased inner products matter |
//!
//! Community implementations (turboquant_plus, tonbistudio) found that the
//! MSE variant matches or beats the Prod variant on perplexity benchmarks.
//! The paper uses Prod for the theoretical unbiasedness guarantee.

use crate::{
    backend::{Backend, ScalarBackend},
    error::Result,
    polar_quant::{PolarQuant, QuantizedVector},
    qjl::{Qjl, QjlVector},
};

// ── Public compressed types ─────────────────────────────────────────────────

/// MSE-variant compressed vector (b-bit PolarQuant, no QJL).
pub type TurboVectorMse = QuantizedVector;

/// Inner-product-variant compressed vector ((b-1)-bit PolarQuant + 1-bit QJL).
#[derive(Debug, Clone)]
pub struct TurboVectorProd {
    pub polar: QuantizedVector,
    pub qjl:   QjlVector,
    pub bits:  u8,
}

impl TurboVectorProd {
    /// Total compressed size in bytes.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.polar.byte_size() + self.qjl.byte_size()
    }

    /// Compression ratio vs f32.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        (self.polar.dim * 4) as f32 / self.byte_size() as f32
    }
}

// ── TurboQuant ──────────────────────────────────────────────────────────────

/// Combined TurboQuant compressor (both MSE and Prod variants).
///
/// For most use cases, use [`TurboQuant::compress_mse`] and
/// [`TurboQuant::inner_product_mse`] — they are simpler, faster, and match the
/// Prod variant's quality in practice.
#[derive(Debug)]
pub struct TurboQuant<B: Backend = ScalarBackend> {
    // Stage-1 MSE quantizer (full b bits).
    mse: PolarQuant<B>,
    // Stage-1 Prod quantizer (b-1 bits); None when bits <= 2.
    prod_polar: Option<PolarQuant<B>>,
    // Stage-2 QJL residual compressor; None when bits <= 2.
    qjl: Option<Qjl>,
    bits: u8,
}

impl TurboQuant<ScalarBackend> {
    /// Create a [`TurboQuant`] instance with the default scalar backend.
    ///
    /// - `dim`  — vector dimension (must be a power of two)
    /// - `bits` — target bit-width: 2, 3, or 4
    /// - `seed` — base seed; the QJL rotation uses `seed.wrapping_add(1)`
    pub fn new(dim: usize, bits: u8, seed: u64) -> Result<Self> {
        Self::new_with_backend(dim, bits, seed, ScalarBackend)
    }
}

impl<B: Backend> TurboQuant<B> {
    /// Create a [`TurboQuant`] instance with an explicit backend.
    pub fn new_with_backend(dim: usize, bits: u8, seed: u64, backend: B) -> Result<Self> {
        let mse = PolarQuant::new_with_backend(dim, bits, seed, backend.clone())?;

        let (prod_polar, qjl) = if bits >= 3 {
            let pp = PolarQuant::new_with_backend(dim, bits - 1, seed.wrapping_add(0x1111), backend)?;
            let q  = Qjl::new(dim, seed.wrapping_add(0x2222))?;
            (Some(pp), Some(q))
        } else {
            // 2-bit: can't split into (1-bit PolarQuant + QJL) meaningfully.
            (None, None)
        };

        Ok(Self { mse, prod_polar, qjl, bits })
    }

    // ── MSE variant ──────────────────────────────────────────────────────

    /// Compress using the MSE variant (all bits to Lloyd-Max).
    #[must_use]
    pub fn compress_mse(&self, vec: &[f32]) -> Result<TurboVectorMse> {
        self.mse.quantize(vec)
    }

    /// Decompress an MSE-compressed vector.
    #[must_use]
    pub fn decompress_mse(&self, qv: &TurboVectorMse) -> Result<Vec<f32>> {
        self.mse.dequantize(qv)
    }

    /// Estimate ⟨query, key⟩ from an MSE-compressed key.
    #[must_use]
    pub fn inner_product_mse(&self, query: &[f32], key: &TurboVectorMse) -> Result<f32> {
        self.mse.inner_product(query, key)
    }

    // ── Prod variant ─────────────────────────────────────────────────────

    /// Compress using the Prod variant ((b-1)-bit PolarQuant + 1-bit QJL).
    ///
    /// When `bits == 2`, falls back to MSE (QJL unavailable at 1-bit PolarQuant).
    #[must_use]
    pub fn compress_prod(&self, vec: &[f32]) -> Result<TurboVectorProd> {
        match (&self.prod_polar, &self.qjl) {
            (Some(pp), Some(qjl)) => {
                // Stage 1: (b-1)-bit PolarQuant.
                let polar = pp.quantize(vec)?;

                // Compute quantization residual in the original space.
                let recon: Vec<f32> = pp.dequantize(&polar)?;
                let residual: Vec<f32> = vec.iter().zip(&recon).map(|(&v, &r)| v - r).collect();

                // Stage 2: 1-bit QJL on the residual.
                let qjl_vec = qjl.compress(&residual)?;

                Ok(TurboVectorProd { polar, qjl: qjl_vec, bits: self.bits })
            }
            // Fallback: bits == 2.
            _ => {
                let polar = self.mse.quantize(vec)?;
                let qjl   = QjlVector::default(); // empty placeholder
                Ok(TurboVectorProd { polar, qjl, bits: self.bits })
            }
        }
    }

    /// Estimate ⟨query, key⟩ from a Prod-compressed key (unbiased estimator).
    #[must_use]
    pub fn inner_product_prod(&self, query: &[f32], key: &TurboVectorProd) -> Result<f32> {
        // Stage-1 estimate.
        let stage1_ip = match &self.prod_polar {
            Some(pp) => pp.inner_product(query, &key.polar)?,
            None     => self.mse.inner_product(query, &key.polar)?,
        };

        // Stage-2 QJL correction (zero if empty sketch or no QJL).
        let qjl_correction = match &self.qjl {
            Some(q) if !key.qjl.is_empty_sketch() => {
                q.estimate_inner_product(query, &key.qjl)?
            }
            _ => 0.0,
        };

        Ok(stage1_ip + qjl_correction)
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    pub fn dim(&self)  -> usize { self.mse.dim()  }
    pub fn bits(&self) -> u8    { self.bits        }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_vec(dim: usize, freq: f32) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 * freq).sin()).collect()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    #[test]
    fn mse_inner_product_accuracy() {
        use crate::polar_quant::l2_norm;
        let tq    = TurboQuant::new(128, 3, 42).unwrap();
        // Use vectors with a meaningful dot product.
        let key   = sine_vec(128, 0.05);
        let query = sine_vec(128, 0.07);
        let true_ip = dot(&key, &query);
        let qv  = tq.compress_mse(&key).unwrap();
        let est = tq.inner_product_mse(&query, &qv).unwrap();
        let max_ip = l2_norm(&key) * l2_norm(&query);
        let err = (est - true_ip).abs() / max_ip;
        assert!(err < 0.05, "MSE IP error: {err:.3}");
    }

    #[test]
    fn prod_compresses() {
        let tq  = TurboQuant::new(128, 3, 42).unwrap();
        let key = sine_vec(128, 0.13);
        let tv  = tq.compress_prod(&key).unwrap();
        assert!(tv.byte_size() > 0);
        assert!(tv.compression_ratio() > 1.0);
    }

    #[test]
    fn fallback_for_2bit() {
        let tq  = TurboQuant::new(128, 2, 99).unwrap();
        let key = sine_vec(128, 0.05);
        let tv  = tq.compress_prod(&key).unwrap();
        assert!(tv.qjl.is_empty_sketch(), "should fall back to MSE at 2-bit");
    }
}
