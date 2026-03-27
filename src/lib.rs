//! # turboquant
//!
//! A pure-Rust implementation of **TurboQuant** (ICLR 2026) — near-optimal
//! KV-cache compression for transformer inference.
//!
//! ## What it does
//!
//! TurboQuant compresses the key/value cache of attention mechanisms to as
//! few as **3 bits per coordinate** with virtually no accuracy loss.  It
//! consists of two stages:
//!
//! | Stage      | Algorithm   | Purpose                                    |
//! |------------|-------------|--------------------------------------------|
//! | 1          | PolarQuant  | Rotate + Lloyd-Max scalar quantize per coord |
//! | 2          | QJL         | 1-bit residual correction for unbiased IPs  |
//!
//! Community implementations (turboquant_plus, tonbistudio) found that
//! Stage 1 alone (MSE variant) matches or beats the two-stage approach in
//! practice.  Both are provided here.
//!
//! ## Quick start
//!
//! ```rust
//! use turboquant::KvCache;
//!
//! // One cache per attention head; 128-dim head, 3-bit compression.
//! let mut cache = KvCache::new(128, 3, /*key_seed*/ 42, /*val_seed*/ 99).unwrap();
//!
//! // Append tokens as they are generated.
//! let key   = vec![0.1f32; 128];
//! let value = vec![0.2f32; 128];
//! cache.push(&key, &value).unwrap();
//!
//! // At decode time: compressed attention in one call.
//! let query  = vec![0.15f32; 128];
//! let output = cache.attend(&query).unwrap();
//! assert_eq!(output.len(), 128);
//!
//! println!("Compression ratio: {:.1}x", cache.compression_ratio());
//! ```
//!
//! ## Lower-level access
//!
//! Use [`PolarQuant`] directly for standalone vector compression, or
//! [`TurboQuant`] to access both MSE and Prod variants.

pub mod backend;
pub mod bitpack;
pub mod codebook;
pub mod error;
pub mod hadamard;
pub mod kv_cache;
pub mod polar_quant;
pub mod qjl;
pub mod rotation;
pub mod turboquant;

// ── Re-exports ──────────────────────────────────────────────────────────────

// Backend abstraction
pub use backend::{Backend, ScalarBackend};

#[cfg(feature = "simd")]
pub use backend::{SimdBackend, RuntimeBackend};

pub use error::{Result, TurboQuantError};

// Core compression types
pub use polar_quant::{PolarQuant, QuantizedVector};
pub use qjl::{Qjl, QjlVector};
pub use turboquant::{TurboQuant, TurboVectorMse, TurboVectorProd};

// KV cache
pub use kv_cache::{AttentionOutput, KvCache};

// ── Utility functions ───────────────────────────────────────────────────────

/// Dot product of two equal-length slices.
#[inline]
#[must_use]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// L2 norm of a vector.
#[inline]
#[must_use]
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Normalize a vector to unit length.  Returns zeros if `v` is the zero vector.
#[must_use]
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let n = l2_norm(v);
    if n > f32::EPSILON {
        v.iter().map(|&x| x / n).collect()
    } else {
        v.to_vec()
    }
}

/// Cosine similarity between two vectors.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    dot_product(a, b) / (l2_norm(a) * l2_norm(b)).max(f32::EPSILON)
}

