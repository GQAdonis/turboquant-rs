//! Backend abstraction for compute operations.
//!
//! The [`Backend`] trait abstracts performance-critical operations (FWHT,
//! dot product, dimension validation) so that future phases can provide
//! SIMD and GPU implementations without changing the core algorithm code.
//!
//! Phase 1 provides [`ScalarBackend`] which wraps the existing scalar
//! implementations.  Phase 2 will add `SimdBackend`.

mod scalar;

pub use scalar::ScalarBackend;

use crate::error::Result;

/// Trait for compute backend implementations.
///
/// All methods use concrete types (no generics in methods) to maintain
/// object safety for potential future runtime backend selection.
pub trait Backend: Clone + std::fmt::Debug {
    /// Apply normalized Fast Walsh-Hadamard Transform in-place.
    ///
    /// Input `data` must have power-of-two length.
    fn fwht_normalized_inplace(&self, data: &mut [f32]);

    /// Compute dot product of two equal-length f32 slices.
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32;

    /// Validate that `dim` is a power of two.
    fn validate_dimension(&self, dim: usize) -> Result<()>;
}
