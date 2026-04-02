//! Backend abstraction for compute operations.
//!
//! The [`Backend`] trait abstracts performance-critical operations (FWHT,
//! dot product, sign-flip, and dimension validation) so that SIMD and GPU
//! implementations can be swapped without changing algorithm code.
//!
//! ## Default backend selection
//!
//! | Feature flag | [`DefaultBackend`] | Runtime dispatch |
//! |---|---|---|
//! | *(none)* | [`ScalarBackend`] | none |
//! | `simd` | [`RuntimeBackend`] | AVX2 / FMA / AVX-512 / NEON |
//!
//! Use [`DefaultBackend`] as the type parameter in [`PolarQuant`] and
//! [`TurboQuant`] to automatically get the fastest available implementation.

mod scalar;

#[cfg(feature = "simd")]
mod simd;

#[cfg(feature = "gpu")]
mod gpu;

pub use scalar::ScalarBackend;

#[cfg(feature = "simd")]
pub use simd::{Avx512Backend, SimdBackend, RuntimeBackend};

#[cfg(feature = "gpu")]
pub use gpu::GpuBackend;

use crate::error::Result;

/// Trait for compute backend implementations.
///
/// All methods use concrete types (no generics in methods) to maintain
/// object safety for potential future runtime backend selection.
///
/// The `Send + Sync` bounds enable parallel batch processing via rayon.
pub trait Backend: Clone + std::fmt::Debug + Send + Sync {
    /// Apply normalized Fast Walsh-Hadamard Transform in-place.
    ///
    /// Input `data` must have power-of-two length.
    fn fwht_normalized_inplace(&self, data: &mut [f32]);

    /// Compute dot product of two equal-length f32 slices.
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32;

    /// Apply diagonal sign-flip: `data[i] ^= sign_masks[i]` (XOR on sign bit).
    ///
    /// `sign_masks[i]` is `0x80000000u32` when the sign is −1, else `0`.
    /// SIMD backends implement this as a bitwise XOR on 8 or 16 floats at a time.
    /// The default implementation is a scalar loop using bit-cast.
    #[inline]
    fn apply_signs(&self, data: &mut [f32], sign_masks: &[u32]) {
        debug_assert_eq!(data.len(), sign_masks.len());
        for (x, &mask) in data.iter_mut().zip(sign_masks) {
            *x = f32::from_bits(x.to_bits() ^ mask);
        }
    }

    /// Validate that `dim` is a power of two.
    fn validate_dimension(&self, dim: usize) -> Result<()>;
}

/// The default backend, selected at compile time.
///
/// With `features = ["simd"]` this resolves to [`RuntimeBackend`], which
/// auto-detects AVX2/FMA/AVX-512 on x86_64 and NEON on aarch64.
/// Without it, resolves to [`ScalarBackend`].
#[cfg(feature = "simd")]
pub type DefaultBackend = RuntimeBackend;

/// The default backend, selected at compile time.
///
/// Without the `simd` feature this is the pure-scalar baseline.
#[cfg(not(feature = "simd"))]
pub type DefaultBackend = ScalarBackend;
