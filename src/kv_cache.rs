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

// ── KV entry ────────────────────────────────────────────────────────────────

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
