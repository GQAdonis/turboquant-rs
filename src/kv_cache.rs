//! TurboQuant-compressed KV cache for transformer attention.
//!
//! ## Usage model
//! 1. Create a [`KvCache`] per attention head (or share across heads with the
//!    same `dim`).
//! 2. Call [`KvCache::push`] for each new token position.
//! 3. At decode time call [`KvCache::attend`] with the current query vector.
//!
//! ## Memory layout
//! Each compressed entry stores:
//!   - Key:   4 bytes norm + `ceil(dim * bits / 8)` bytes indices
//!   - Value: same
//!
//! At 3-bit, d=128 this is 52 + 52 = 104 bytes per token versus 1024 bytes
//! uncompressed — a ~9.8x reduction (keys + values together).

use crate::{
    backend::{Backend, ScalarBackend},
    error::Result,
    turboquant::{TurboQuant, TurboVectorMse},
};
use rayon::prelude::*;

// ── KV entry ────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct Entry {
    key: TurboVectorMse,
    val: TurboVectorMse,
}

// ── Attention output ─────────────────────────────────────────────────────────

/// Raw attention logits and helper methods for the full attention computation.
pub struct AttentionOutput {
    /// ⟨query, key_i⟩ for each cached position i.
    pub logits: Vec<f32>,
}

impl AttentionOutput {
    /// Numerically-stable softmax over the logits.
    #[must_use]
    pub fn softmax(&self) -> Vec<f32> {
        if self.logits.is_empty() {
            return vec![];
        }
        let max = self.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = self.logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Compute the softmax-weighted sum of the provided value vectors.
    #[must_use]
    pub fn weighted_sum(&self, values: &[Vec<f32>]) -> Vec<f32> {
        debug_assert_eq!(self.logits.len(), values.len());
        if values.is_empty() {
            return vec![];
        }
        let weights = self.softmax();
        let dim     = values[0].len();
        let mut out = vec![0.0f32; dim];
        for (&w, v) in weights.iter().zip(values) {
            for (o, &vi) in out.iter_mut().zip(v.iter()) {
                *o += w * vi;
            }
        }
        out
    }
}

// ── KvCache ─────────────────────────────────────────────────────────────────

/// A single-head TurboQuant-compressed KV cache.
pub struct KvCache<B: Backend = ScalarBackend> {
    key_tq: TurboQuant<B>,
    val_tq: TurboQuant<B>,
    entries: Vec<Entry>,
    head_dim: usize,
}

impl<B: Backend> Clone for KvCache<B> {
    fn clone(&self) -> Self {
        Self {
            key_tq: self.key_tq.clone(),
            val_tq: self.val_tq.clone(),
            entries: self.entries.clone(),
            head_dim: self.head_dim,
        }
    }
}

impl KvCache<ScalarBackend> {
    /// Create a new KV cache with the default scalar backend.
    ///
    /// - `head_dim`  — attention head dimension (must be power of two)
    /// - `bits`      — compression bit-width (2, 3, or 4)
    /// - `key_seed`, `val_seed` — independent seeds; use different values to
    ///   give keys and values independent random rotations
    pub fn new(head_dim: usize, bits: u8, key_seed: u64, val_seed: u64) -> Result<Self> {
        Self::new_with_backend(head_dim, bits, key_seed, val_seed, ScalarBackend)
    }
}

impl<B: Backend> KvCache<B> {
    /// Create a new KV cache with an explicit backend.
    pub fn new_with_backend(
        head_dim: usize,
        bits: u8,
        key_seed: u64,
        val_seed: u64,
        backend: B,
    ) -> Result<Self> {
        let key_tq = TurboQuant::new_with_backend(head_dim, bits, key_seed, backend.clone())?;
        let val_tq = TurboQuant::new_with_backend(head_dim, bits, val_seed, backend)?;
        Ok(Self { key_tq, val_tq, entries: Vec::new(), head_dim })
    }

    // ── Mutation ─────────────────────────────────────────────────────────

    /// Append a new (key, value) pair to the cache.
    pub fn push(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
        let k = self.key_tq.compress_mse(key)?;
        let v = self.val_tq.compress_mse(value)?;
        self.entries.push(Entry { key: k, val: v });
        Ok(())
    }

    /// Remove all cached entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    // ── Query ────────────────────────────────────────────────────────────

    /// Compute attention logits ⟨query, key_i⟩ for all cached positions.
    #[must_use]
    pub fn attention_logits(&self, query: &[f32]) -> Result<AttentionOutput> {
        let logits = self.entries
            .iter()
            .map(|e| self.key_tq.inner_product_mse(query, &e.key))
            .collect::<Result<Vec<f32>>>()?;
        Ok(AttentionOutput { logits })
    }

    /// Decompress all stored value vectors.
    ///
    /// In a production system you would typically do this lazily or keep
    /// values in a separate partially-decompressed ring buffer, but this
    /// simple approach is correct and convenient.
    #[must_use]
    pub fn decompress_values(&self) -> Result<Vec<Vec<f32>>> {
        self.entries
            .iter()
            .map(|e| self.val_tq.decompress_mse(&e.val))
            .collect()
    }

    /// Full attention: returns softmax-weighted sum of decompressed values.
    #[must_use]
    pub fn attend(&self, query: &[f32]) -> Result<Vec<f32>> {
        let out    = self.attention_logits(query)?;
        let values = self.decompress_values()?;
        Ok(out.weighted_sum(&values))
    }

    /// Compute attention for multiple queries in parallel.
    ///
    /// Each query independently computes logits -> softmax -> weighted sum
    /// against all cached entries. Parallelism is across queries, not within
    /// a single attention computation.
    ///
    /// # Performance
    /// - Batch-of-1: delegates to `attend()` directly (zero overhead)
    /// - Batch >= 2: parallel processing via rayon work-stealing
    ///
    /// # Implementation Note
    /// Each worker thread reconstructs TurboQuant instances (to avoid Sync
    /// issues with RefCell) and clones the entries Vec. For very large caches,
    /// this cloning cost can be optimized in future phases using shared
    /// references if needed for GPU dispatch.
    #[must_use = "attention outputs should be used"]
    pub fn batch_attend(&self, queries: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if queries.is_empty() {
            return Ok(vec![]);
        }
        if queries.len() == 1 {
            return Ok(vec![self.attend(&queries[0])?]);
        }
        // Validate dimensions up front
        for q in queries {
            if q.len() != self.head_dim {
                return Err(crate::error::TurboQuantError::DimensionMismatch {
                    expected: self.head_dim,
                    got: q.len(),
                });
            }
        }

        // Extract reconstruction parameters (all are Sync/Send)
        let head_dim = self.head_dim;
        let bits = self.key_tq.bits();
        let key_seed = self.key_tq.seed();
        let val_seed = self.val_tq.seed();
        let backend = self.key_tq.backend().clone();
        let entries_clone = self.entries.clone();

        queries.par_iter()
            .map_init(
                move || {
                    // Each thread reconstructs KvCache with fresh RefCells
                    let mut cache = KvCache::new_with_backend(
                        head_dim,
                        bits,
                        key_seed,
                        val_seed,
                        backend.clone()
                    ).unwrap();
                    cache.entries = entries_clone.clone();
                    cache
                },
                |cache, q| cache.attend(q)
            )
            .collect()
    }

    /// Zero-copy variant accepting borrowed slices.
    #[must_use = "attention outputs should be used"]
    pub fn batch_attend_slices(&self, queries: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        if queries.is_empty() {
            return Ok(vec![]);
        }
        if queries.len() == 1 {
            return Ok(vec![self.attend(queries[0])?]);
        }
        for q in queries {
            if q.len() != self.head_dim {
                return Err(crate::error::TurboQuantError::DimensionMismatch {
                    expected: self.head_dim,
                    got: q.len(),
                });
            }
        }

        // Extract reconstruction parameters (all are Sync/Send)
        let head_dim = self.head_dim;
        let bits = self.key_tq.bits();
        let key_seed = self.key_tq.seed();
        let val_seed = self.val_tq.seed();
        let backend = self.key_tq.backend().clone();
        let entries_clone = self.entries.clone();

        queries.par_iter()
            .map_init(
                move || {
                    // Each thread reconstructs KvCache with fresh RefCells
                    let mut cache = KvCache::new_with_backend(
                        head_dim,
                        bits,
                        key_seed,
                        val_seed,
                        backend.clone()
                    ).unwrap();
                    cache.entries = entries_clone.clone();
                    cache
                },
                |cache, q| cache.attend(q)
            )
            .collect()
    }

    // ── Metrics ──────────────────────────────────────────────────────────

    /// Number of cached token positions.
    #[must_use]
    pub fn len(&self) -> usize { self.entries.len() }

    /// True if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Head dimension.
    #[must_use]
    pub fn head_dim(&self) -> usize { self.head_dim }

    /// Memory consumed by the compressed cache, in bytes.
    #[must_use]
    pub fn compressed_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.key.byte_size() + e.val.byte_size()).sum()
    }

    /// Memory that would be consumed if stored in fp32.
    #[must_use]
    pub fn uncompressed_bytes(&self) -> usize {
        self.entries.len() * 2 * self.head_dim * 4
    }

    /// Compression ratio (uncompressed / compressed).
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        let c = self.compressed_bytes();
        if c == 0 { return 1.0; }
        self.uncompressed_bytes() as f32 / c as f32
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache(bits: u8) -> KvCache {
        KvCache::new(128, bits, 42, 99).unwrap()
    }

    fn fill(cache: &mut KvCache, n: usize) {
        for i in 0..n {
            let k: Vec<f32> = (0..128).map(|j| ((i * 128 + j) as f32 * 0.01).sin()).collect();
            let v: Vec<f32> = (0..128).map(|j| ((i * 128 + j) as f32 * 0.01).cos()).collect();
            cache.push(&k, &v).unwrap();
        }
    }

    #[test]
    fn push_and_len() {
        let mut cache = make_cache(3);
        fill(&mut cache, 10);
        assert_eq!(cache.len(), 10);
    }

    #[test]
    fn attend_returns_correct_dim() {
        let mut cache = make_cache(3);
        fill(&mut cache, 8);
        let query: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();
        let output = cache.attend(&query).unwrap();
        assert_eq!(output.len(), 128);
    }

    #[test]
    fn compression_ratio_3bit() {
        let mut cache = make_cache(3);
        fill(&mut cache, 100);
        // At 3-bit: expect roughly 6-7x ratio (key+value together).
        let ratio = cache.compression_ratio();
        assert!(ratio > 4.0, "compression ratio too low: {ratio:.2}");
    }

    #[test]
    fn attention_softmax_sums_to_one() {
        let mut cache = make_cache(3);
        fill(&mut cache, 16);
        let query: Vec<f32> = (0..128).map(|i| (i as f32 * 0.03).sin()).collect();
        let out = cache.attention_logits(&query).unwrap();
        let weights = out.softmax();
        let total: f32 = weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "softmax doesn't sum to 1: {total}");
    }

    #[test]
    fn clear_resets_len() {
        let mut cache = make_cache(3);
        fill(&mut cache, 5);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn attend_without_entries_returns_empty() {
        let cache  = make_cache(3);
        let query  = vec![0.0f32; 128];
        let logits = cache.attention_logits(&query).unwrap();
        assert!(logits.logits.is_empty());
        let weighted = logits.weighted_sum(&[]);
        assert!(weighted.is_empty());
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;

    fn make_cache(bits: u8) -> KvCache {
        KvCache::new(128, bits, 42, 99).unwrap()
    }

    fn fill(cache: &mut KvCache, n: usize) {
        for i in 0..n {
            let k: Vec<f32> = (0..128).map(|j| ((i * 128 + j) as f32 * 0.01).sin()).collect();
            let v: Vec<f32> = (0..128).map(|j| ((i * 128 + j) as f32 * 0.01).cos()).collect();
            cache.push(&k, &v).unwrap();
        }
    }

    #[test]
    fn batch_attend_dimensions() {
        let mut cache = make_cache(3);
        fill(&mut cache, 16);

        let queries: Vec<Vec<f32>> = (0..4)
            .map(|i| (0..128).map(|j| ((i * 128 + j) as f32 * 0.05).sin()).collect())
            .collect();

        let result = cache.batch_attend(&queries).unwrap();
        assert_eq!(result.len(), 4, "should return 4 outputs for 4 queries");
        for output in &result {
            assert_eq!(output.len(), 128, "each output should be head_dim=128");
        }
    }

    #[test]
    fn batch_attend_empty() {
        let mut cache = make_cache(3);
        fill(&mut cache, 8);

        let queries: Vec<Vec<f32>> = vec![];
        let result = cache.batch_attend(&queries).unwrap();
        assert!(result.is_empty(), "empty input should return empty result");
    }

    #[test]
    fn batch_attend_single_matches_attend() {
        let mut cache = make_cache(3);
        fill(&mut cache, 10);

        let query: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();

        // Single attend
        let single = cache.attend(&query).unwrap();

        // Batch-of-1
        let batch = cache.batch_attend(&[query]).unwrap();

        assert_eq!(batch.len(), 1);
        assert_eq!(single.len(), batch[0].len());
        for (s, b) in single.iter().zip(&batch[0]) {
            assert!((s - b).abs() < 1e-5, "batch-of-1 should match single attend");
        }
    }

    #[test]
    fn batch_attend_slices_equivalence() {
        let mut cache = make_cache(3);
        fill(&mut cache, 12);

        let queries: Vec<Vec<f32>> = (0..3)
            .map(|i| (0..128).map(|j| ((i * 128 + j) as f32 * 0.04).sin()).collect())
            .collect();

        let result_owned = cache.batch_attend(&queries).unwrap();

        let slices: Vec<&[f32]> = queries.iter().map(|v| v.as_slice()).collect();
        let result_slices = cache.batch_attend_slices(&slices).unwrap();

        assert_eq!(result_owned.len(), result_slices.len());
        for (owned, sliced) in result_owned.iter().zip(&result_slices) {
            assert_eq!(owned.len(), sliced.len());
            for (o, s) in owned.iter().zip(sliced) {
                assert!((o - s).abs() < 1e-5, "slices variant should match owned");
            }
        }
    }

    #[test]
    fn batch_attend_sequential_equivalence() {
        let mut cache = make_cache(3);
        fill(&mut cache, 8);

        let queries: Vec<Vec<f32>> = (0..5)
            .map(|i| (0..128).map(|j| ((i * 128 + j) as f32 * 0.03).sin()).collect())
            .collect();

        // Sequential loop
        let sequential: Vec<Vec<f32>> = queries.iter()
            .map(|q| cache.attend(q).unwrap())
            .collect();

        // Batch
        let batch = cache.batch_attend(&queries).unwrap();

        assert_eq!(sequential.len(), batch.len());
        for (seq, bat) in sequential.iter().zip(&batch) {
            assert_eq!(seq.len(), bat.len());
            for (s, b) in seq.iter().zip(bat) {
                assert!((s - b).abs() < 1e-5, "batch should match sequential within epsilon");
            }
        }
    }

    #[test]
    fn batch_attend_dimension_mismatch() {
        let mut cache = make_cache(3);
        fill(&mut cache, 5);

        let queries = vec![
            (0..128).map(|i| (i as f32 * 0.02).sin()).collect(),
            vec![0.0f32; 64], // Wrong dimension!
        ];

        let result = cache.batch_attend(&queries);
        assert!(result.is_err(), "should fail on dimension mismatch");
    }
}
