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
    backend::{Backend, DefaultBackend},
    bitpack,
    codebook::Codebook,
    error::{Result, TurboQuantError},
    rotation::Rotation,
};
#[cfg(not(feature = "simd"))]
use crate::backend::ScalarBackend;
use rayon::prelude::*;
use std::sync::Mutex;

#[cfg(feature = "gpu")]
use crate::backend::gpu::{GpuBackend, GPU_BATCH_THRESHOLD};

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
///
/// The default type parameter resolves to [`DefaultBackend`], which is
/// [`ScalarBackend`] without the `simd` feature and [`RuntimeBackend`] with it.
#[derive(Debug)]
pub struct PolarQuant<B: Backend = DefaultBackend> {
    rotation: Rotation<B>,
    codebook: Codebook,
    backend: B,
    scratch: Mutex<Vec<f32>>,
}

impl<B: Backend> Clone for PolarQuant<B> {
    fn clone(&self) -> Self {
        let dim = self.rotation.dim;
        Self {
            rotation: self.rotation.clone(),
            codebook: self.codebook.clone(),
            backend: self.backend.clone(),
            scratch: Mutex::new(Vec::with_capacity(dim)),
        }
    }
}

impl PolarQuant<DefaultBackend> {
    /// Create a [`PolarQuant`] with the best available backend.
    ///
    /// Without the `simd` feature this uses [`ScalarBackend`].
    /// With `simd` it auto-selects AVX2 / FMA / AVX-512 / NEON at runtime.
    ///
    /// - `dim`  — head dimension; **must be a power of two** (64, 128, 256, …)
    /// - `bits` — target bit-width per coordinate: 2, 3, or 4
    /// - `seed` — RNG seed for the rotation matrix
    pub fn new(dim: usize, bits: u8, seed: u64) -> Result<Self> {
        #[cfg(feature = "simd")]
        { Self::new_with_backend(dim, bits, seed, crate::backend::RuntimeBackend::best_available()) }
        #[cfg(not(feature = "simd"))]
        { Self::new_with_backend(dim, bits, seed, ScalarBackend) }
    }
}

impl<B: Backend> PolarQuant<B> {
    /// Create a PolarQuant with an explicit backend.
    pub fn new_with_backend(dim: usize, bits: u8, seed: u64, backend: B) -> Result<Self> {
        let rotation = Rotation::new_with_backend(dim, seed, backend.clone())?;
        let codebook = Codebook::new(bits, dim)?;
        let scratch = Mutex::new(Vec::with_capacity(dim));
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

        // Rotate query into scratch buffer, then accumulate via backend dot product.
        // Using dequantize_slice + backend.dot_product lets the SIMD backend
        // vectorize the accumulation loop (8 or 16 f32/cycle with AVX2/AVX-512).
        let dot = {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            scratch.clear();
            scratch.extend_from_slice(query);
            self.rotation.apply(&mut scratch);

            let indices = bitpack::unpack(&key.packed, key.dim, key.bits)?;
            let centroids = self.codebook.dequantize_slice(&indices);

            self.backend.dot_product(&scratch, &centroids)
        }; // Mutex guard dropped here

        // Scale by key norm (query norm does not factor in here;
        // the caller applies it via the standard softmax attention formula).
        Ok(dot * key.norm)
    }

    // ── Batch operations ──────────────────────────────────────────────────

    /// Quantize multiple vectors in parallel.
    ///
    /// Returns `Vec<QuantizedVector>` with same length as input.
    /// Fails fast on first error (dimension mismatch, invalid input).
    ///
    /// # Performance
    /// - Batch-of-1: delegates to `quantize()` directly (zero overhead)
    /// - Batch >= 2: parallel processing via rayon work-stealing
    #[must_use = "quantized vectors should be stored"]
    pub fn batch_quantize(&self, vecs: &[Vec<f32>]) -> Result<Vec<QuantizedVector>> {
        if vecs.is_empty() {
            return Ok(vec![]);
        }
        if vecs.len() == 1 {
            return Ok(vec![self.quantize(&vecs[0])?]);
        }
        // Validate all dimensions up front before spawning threads
        for v in vecs {
            self.check_dim(v.len())?;
        }

        // Extract values needed for reconstruction inside the thread-local init
        let dim = self.dim();
        let bits = self.bits();
        let seed = self.rotation.seed;
        let backend = self.backend.clone();

        vecs.par_iter()
            .map_init(
                move || Self::new_with_backend(dim, bits, seed, backend.clone()).unwrap(),
                |pq, v| pq.quantize(v)
            )
            .collect()
    }

    /// Zero-copy variant accepting borrowed slices.
    #[must_use = "quantized vectors should be stored"]
    pub fn batch_quantize_slices(&self, vecs: &[&[f32]]) -> Result<Vec<QuantizedVector>> {
        if vecs.is_empty() {
            return Ok(vec![]);
        }
        if vecs.len() == 1 {
            return Ok(vec![self.quantize(vecs[0])?]);
        }
        for v in vecs {
            self.check_dim(v.len())?;
        }

        let dim = self.dim();
        let bits = self.bits();
        let seed = self.rotation.seed;
        let backend = self.backend.clone();

        vecs.par_iter()
            .map_init(
                move || Self::new_with_backend(dim, bits, seed, backend.clone()).unwrap(),
                |pq, v| pq.quantize(v)
            )
            .collect()
    }

    /// Compute inner products of one query against multiple quantized keys in parallel.
    ///
    /// This is the primary hot path for batch attention logit computation.
    /// The query is shared read-only; each worker gets a cloned PolarQuant
    /// with independent scratch buffers.
    ///
    /// # Performance
    /// - Batch-of-1: delegates to `inner_product()` directly (zero overhead)
    /// - Batch >= 2: parallel processing via rayon work-stealing
    #[must_use = "inner product results should be used"]
    pub fn batch_inner_product(
        &self,
        query: &[f32],
        keys: &[QuantizedVector],
    ) -> Result<Vec<f32>> {
        self.check_dim(query.len())?;
        if keys.is_empty() {
            return Ok(vec![]);
        }
        if keys.len() == 1 {
            return Ok(vec![self.inner_product(query, &keys[0])?]);
        }
        for k in keys {
            self.check_dim(k.dim)?;
        }

        let dim = self.dim();
        let bits = self.bits();
        let seed = self.rotation.seed;
        let backend = self.backend.clone();
        let query_vec: Vec<f32> = query.to_vec();

        keys.par_iter()
            .map_init(
                move || {
                    let pq = Self::new_with_backend(dim, bits, seed, backend.clone()).unwrap();
                    let q = query_vec.clone();
                    (pq, q)
                },
                |(pq, q), k| pq.inner_product(q, k)
            )
            .collect()
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    pub fn codebook(&self)  -> &Codebook { &self.codebook  }
    pub fn rotation(&self)  -> &Rotation<B> { &self.rotation  }
    pub fn backend(&self)   -> &B        { &self.backend   }
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

// ── GPU-specific batch dispatch ─────────────────────────────────────────────

#[cfg(feature = "gpu")]
impl PolarQuant<GpuBackend> {
    /// GPU batch FWHT rotation + CPU quantization for large batches.
    fn batch_quantize_gpu(&self, vecs: &[Vec<f32>]) -> Result<Vec<QuantizedVector>> {
        let dim = self.dim();
        let batch_size = vecs.len();
        let gpu = self.backend();

        // 1. Compute norms on CPU (cheap, O(n*dim))
        let norms: Vec<f32> = vecs.iter().map(|v| l2_norm(v)).collect();

        // 2. Normalize and flatten into contiguous buffer
        let mut flat: Vec<f32> = Vec::with_capacity(batch_size * dim);
        for (v, &norm) in vecs.iter().zip(&norms) {
            if norm > f32::EPSILON {
                flat.extend(v.iter().map(|&x| x / norm));
            } else {
                flat.extend(std::iter::repeat(0.0f32).take(dim));
            }
        }

        // 3. Apply random signs (D matrix) on CPU
        let signs = self.rotation().signs();
        for b in 0..batch_size {
            let start = b * dim;
            for i in 0..dim {
                flat[start + i] *= signs[i] as f32;
            }
        }

        // 4. Copy to GPU and run batch FWHT kernel
        let mut d_data = gpu.device().htod_sync_copy(&flat)
            .map_err(|e| TurboQuantError::GpuKernelFailed { reason: format!("htod copy: {}", e) })?;
        gpu.launch_fwht_batch(&mut d_data, dim, batch_size)?;

        // 5. Copy back
        let rotated = gpu.device().dtoh_sync_copy(&d_data)
            .map_err(|e| TurboQuantError::GpuKernelFailed { reason: format!("dtoh copy: {}", e) })?;

        // 6. Quantize each vector on CPU (codebook lookup + bitpack)
        let mut results = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let start = b * dim;
            let slice = &rotated[start..start + dim];
            let indices = self.codebook().quantize_slice(slice);
            let packed = bitpack::pack(&indices, self.codebook().bits)?;
            results.push(QuantizedVector {
                norm: norms[b],
                packed,
                dim,
                bits: self.codebook().bits,
            });
        }

        Ok(results)
    }

    /// GPU batch dequantize + dot product for large key sets.
    fn batch_inner_product_gpu(
        &self,
        query: &[f32],
        keys: &[QuantizedVector],
    ) -> Result<Vec<f32>> {
        let dim = self.dim();
        let batch_size = keys.len();
        let gpu = self.backend();

        // 1. Rotate query on CPU (single vector, fast)
        let mut q_rotated = query.to_vec();
        self.rotation().apply(&mut q_rotated);

        // 2. Prepare key data: flatten packed indices and norms
        let packed_bytes = crate::bitpack::packed_byte_size(dim, self.bits());
        let mut flat_packed: Vec<u8> = Vec::with_capacity(batch_size * packed_bytes);
        let mut key_norms: Vec<f32> = Vec::with_capacity(batch_size);
        for k in keys {
            flat_packed.extend_from_slice(&k.packed);
            key_norms.push(k.norm);
        }

        // 3. Upload to GPU
        let d_query = gpu.device().htod_sync_copy(&q_rotated)
            .map_err(|e| TurboQuantError::GpuKernelFailed { reason: format!("htod query: {}", e) })?;
        let d_packed = gpu.device().htod_sync_copy(&flat_packed)
            .map_err(|e| TurboQuantError::GpuKernelFailed { reason: format!("htod packed: {}", e) })?;
        let d_norms = gpu.device().htod_sync_copy(&key_norms)
            .map_err(|e| TurboQuantError::GpuKernelFailed { reason: format!("htod norms: {}", e) })?;
        let d_centroids = gpu.device().htod_sync_copy(self.codebook().centroids())
            .map_err(|e| TurboQuantError::GpuKernelFailed { reason: format!("htod centroids: {}", e) })?;

        // 4. Dequantize keys on GPU
        let mut d_keys = gpu.get_f32_buffer(batch_size * dim)?;
        gpu.launch_batch_dequantize(
            &d_packed, &d_centroids, &mut d_keys,
            dim, self.bits(), packed_bytes, batch_size,
        )?;

        // 5. Compute batch dot products on GPU
        let mut d_results = gpu.get_f32_buffer(batch_size)?;
        gpu.launch_batch_dot_product(
            &d_query, &d_keys, &d_norms, &mut d_results,
            dim, batch_size,
        )?;

        // 6. Copy results back
        let results = gpu.device().dtoh_sync_copy(&d_results)
            .map_err(|e| TurboQuantError::GpuKernelFailed { reason: format!("dtoh results: {}", e) })?;

        // 7. Return buffers to pool
        gpu.return_f32_buffer(batch_size * dim, d_keys);
        gpu.return_f32_buffer(batch_size, d_results);

        Ok(results)
    }

    /// GPU-aware batch quantize: routes to GPU for large batches, CPU for small.
    pub fn batch_quantize_dispatch(&self, vecs: &[Vec<f32>]) -> Result<Vec<QuantizedVector>> {
        if vecs.len() >= GPU_BATCH_THRESHOLD {
            self.batch_quantize_gpu(vecs)
        } else {
            self.batch_quantize(vecs)
        }
    }

    /// GPU-aware batch inner product: routes to GPU for large batches, CPU for small.
    pub fn batch_inner_product_dispatch(
        &self,
        query: &[f32],
        keys: &[QuantizedVector],
    ) -> Result<Vec<f32>> {
        if keys.len() >= GPU_BATCH_THRESHOLD {
            self.batch_inner_product_gpu(query, keys)
        } else {
            self.batch_inner_product(query, keys)
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

    #[allow(dead_code)]
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

#[cfg(test)]
mod batch_tests {
    use super::*;

    fn make_pq(bits: u8) -> PolarQuant {
        PolarQuant::new(128, bits, 42).unwrap()
    }

    fn sine_vec(dim: usize, freq: f32) -> Vec<f32> {
        (0..dim).map(|i| (i as f32 * freq).sin()).collect()
    }

    #[allow(dead_code)]
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    #[test]
    fn batch_quantize_length_correctness() {
        let pq = make_pq(3);
        let vecs: Vec<Vec<f32>> = (0..10).map(|i| sine_vec(128, 0.1 + i as f32 * 0.01)).collect();
        let result = pq.batch_quantize(&vecs).unwrap();

        assert_eq!(result.len(), 10, "batch_quantize should return 10 vectors");
        for qv in &result {
            assert_eq!(qv.dim, 128);
            assert_eq!(qv.bits, 3);
            assert_eq!(qv.byte_size(), 52); // 4 + 48 for 128-dim 3-bit
        }
    }

    #[test]
    fn batch_quantize_empty() {
        let pq = make_pq(3);
        let vecs: Vec<Vec<f32>> = vec![];
        let result = pq.batch_quantize(&vecs).unwrap();
        assert!(result.is_empty(), "empty input should return empty result");
    }

    #[test]
    fn batch_quantize_dimension_mismatch() {
        let pq = make_pq(3);
        let vecs = vec![
            sine_vec(128, 0.1),
            sine_vec(64, 0.2),  // Wrong dimension!
        ];
        let result = pq.batch_quantize(&vecs);
        assert!(result.is_err(), "should fail on dimension mismatch");
    }

    #[test]
    fn batch_quantize_slices_equivalence() {
        let pq = make_pq(3);
        let vecs: Vec<Vec<f32>> = (0..5).map(|i| sine_vec(128, 0.1 + i as f32 * 0.01)).collect();

        let result_owned = pq.batch_quantize(&vecs).unwrap();

        let slices: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let result_slices = pq.batch_quantize_slices(&slices).unwrap();

        assert_eq!(result_owned.len(), result_slices.len());
        for (owned, sliced) in result_owned.iter().zip(&result_slices) {
            assert_eq!(owned.norm, sliced.norm);
            assert_eq!(owned.packed, sliced.packed);
            assert_eq!(owned.dim, sliced.dim);
            assert_eq!(owned.bits, sliced.bits);
        }
    }

    #[test]
    fn batch_inner_product_length_correctness() {
        let pq = make_pq(3);
        let keys: Vec<Vec<f32>> = (0..20).map(|i| sine_vec(128, 0.1 + i as f32 * 0.01)).collect();
        let query = sine_vec(128, 0.05);

        let qvs: Vec<QuantizedVector> = keys.iter().map(|k| pq.quantize(k).unwrap()).collect();
        let result = pq.batch_inner_product(&query, &qvs).unwrap();

        assert_eq!(result.len(), 20, "batch_inner_product should return 20 results");
    }

    #[test]
    fn batch_inner_product_sequential_equivalence() {
        let pq = make_pq(3);
        let keys: Vec<Vec<f32>> = (0..10).map(|i| sine_vec(128, 0.1 + i as f32 * 0.01)).collect();
        let query = sine_vec(128, 0.05);

        let qvs: Vec<QuantizedVector> = keys.iter().map(|k| pq.quantize(k).unwrap()).collect();

        // Sequential approach
        let sequential: Vec<f32> = qvs.iter()
            .map(|qv| pq.inner_product(&query, qv).unwrap())
            .collect();

        // Batch approach
        let batch = pq.batch_inner_product(&query, &qvs).unwrap();

        assert_eq!(sequential.len(), batch.len());
        for (seq, bat) in sequential.iter().zip(&batch) {
            assert!((seq - bat).abs() < 1e-5, "sequential and batch differ: {seq} vs {bat}");
        }
    }

    #[test]
    fn batch_quantize_single_vector_fast_path() {
        let pq = make_pq(3);
        let vec = sine_vec(128, 0.1);

        let single = pq.quantize(&vec).unwrap();
        let batch = pq.batch_quantize(&[vec.clone()]).unwrap();

        assert_eq!(batch.len(), 1);
        assert_eq!(single.norm, batch[0].norm);
        assert_eq!(single.packed, batch[0].packed);
        assert_eq!(single.dim, batch[0].dim);
        assert_eq!(single.bits, batch[0].bits);
    }

    #[test]
    fn batch_inner_product_single_key_fast_path() {
        let pq = make_pq(3);
        let key = sine_vec(128, 0.1);
        let query = sine_vec(128, 0.05);

        let qv = pq.quantize(&key).unwrap();
        let single = pq.inner_product(&query, &qv).unwrap();
        let batch = pq.batch_inner_product(&query, &[qv]).unwrap();

        assert_eq!(batch.len(), 1);
        assert!((single - batch[0]).abs() < 1e-6);
    }

    #[test]
    fn batch_quantize_64_vectors_parallel() {
        let pq = make_pq(3);
        let vecs: Vec<Vec<f32>> = (0..64).map(|i| sine_vec(128, 0.1 + i as f32 * 0.001)).collect();

        let result = pq.batch_quantize(&vecs).unwrap();

        assert_eq!(result.len(), 64);
        for qv in &result {
            assert_eq!(qv.dim, 128);
            assert_eq!(qv.bits, 3);
        }
    }
}
