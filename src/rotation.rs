//! Randomized Hadamard rotation  R = H̃ · D
//!
//! where:
//!   - D   = diag(s₁, …, s_d), sᵢ ∈ {+1, -1} uniformly at random
//!   - H̃   = (1/√d) · H_d  (normalized Walsh-Hadamard matrix)
//!
//! R is orthogonal: R^T R = I.
//!
//! ## Forward transform  (quantization path)
//!   y = R x = H̃ (D x)
//!   → multiply by signs, then apply normalized FWHT.
//!
//! ## Inverse transform  (dequantization path)
//!   x = R^{-1} y = R^T y = D^T H̃^T y = D H̃ y
//!   → apply normalized FWHT (H̃ = H̃^T), then multiply by signs (D = D^T = D^{-1}).
//!
//! ## Why this works for TurboQuant
//! After the rotation each coordinate of a unit vector follows a
//! Beta((d-1)/2, (d-1)/2) distribution — close to N(0, 1/d) for large d.
//! This allows independent optimal scalar quantization per coordinate.

use crate::{
    backend::{Backend, DefaultBackend},
    error::{Result, TurboQuantError},
};
#[cfg(not(feature = "simd"))]
use crate::backend::ScalarBackend;

/// A seeded, reproducible randomized Hadamard rotation for a fixed dimension.
#[derive(Debug, Clone)]
pub struct Rotation<B: Backend = DefaultBackend> {
    /// ±1 signs for the diagonal matrix D, stored as i8 for compact layout.
    signs: Vec<i8>,
    /// Precomputed XOR masks for SIMD sign-flip: `0x80000000` when sign==-1, else `0`.
    sign_masks: Vec<u32>,
    pub dim: usize,
    pub seed: u64,
    backend: B,
}

impl Rotation<DefaultBackend> {
    /// Create a new rotation with the default backend (scalar without `simd` feature,
    /// [`RuntimeBackend`][crate::backend::RuntimeBackend] with it).
    pub fn new(dim: usize, seed: u64) -> Result<Self> {
        #[cfg(feature = "simd")]
        { Self::new_with_backend(dim, seed, crate::backend::RuntimeBackend::best_available()) }
        #[cfg(not(feature = "simd"))]
        { Self::new_with_backend(dim, seed, ScalarBackend) }
    }
}

impl<B: Backend> Rotation<B> {
    /// Create a new rotation with an explicit backend.
    pub fn new_with_backend(dim: usize, seed: u64, backend: B) -> Result<Self> {
        if !dim.is_power_of_two() {
            return Err(TurboQuantError::DimensionNotPowerOfTwo(dim));
        }
        let signs = gen_signs(dim, seed);
        let sign_masks = signs.iter().map(|&s| if s < 0 { 0x8000_0000u32 } else { 0u32 }).collect();
        Ok(Self { signs, sign_masks, dim, seed, backend })
    }

    /// Apply R: x → H̃(D x)   (in-place).
    #[inline]
    pub fn apply(&self, v: &mut [f32]) {
        debug_assert_eq!(v.len(), self.dim);
        self.backend.apply_signs(v, &self.sign_masks);
        self.backend.fwht_normalized_inplace(v);
    }

    /// Apply R^{-1} = R^T: y → D(H̃ y)   (in-place).
    #[inline]
    pub fn apply_inverse(&self, v: &mut [f32]) {
        debug_assert_eq!(v.len(), self.dim);
        self.backend.fwht_normalized_inplace(v);
        self.backend.apply_signs(v, &self.sign_masks);
    }

    #[inline]
    #[must_use]
    pub fn signs(&self) -> &[i8] { &self.signs }
}

/// Splitmix64-based sign generation — bijective, fast, no external deps.
fn gen_signs(dim: usize, seed: u64) -> Vec<i8> {
    let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    (0..dim)
        .map(|_| {
            s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
            s ^= s >> 30;
            s = s.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            s ^= s >> 27;
            s = s.wrapping_mul(0x94d0_49bb_1331_11eb);
            s ^= s >> 31;
            if (s >> 63) & 1 == 0 { 1i8 } else { -1i8 }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }

    fn norm(v: &[f32]) -> f32 {
        v.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn roundtrip_dim128() {
        let rot = Rotation::new(128, 42).unwrap();
        let orig: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut v = orig.clone();
        rot.apply(&mut v);
        rot.apply_inverse(&mut v);
        for (&a, &b) in orig.iter().zip(&v) {
            assert!((a - b).abs() < 1e-5, "roundtrip mismatch {a} vs {b}");
        }
    }

    #[test]
    fn norm_preserving() {
        let rot  = Rotation::new(64, 7).unwrap();
        let orig: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let n0   = norm(&orig);
        let mut v = orig.clone();
        rot.apply(&mut v);
        assert!((norm(&v) - n0).abs() < 1e-3, "norm not preserved");
    }

    #[test]
    fn orthogonality() {
        // Rows of R are orthonormal: for any unit vector e_i, ‖R e_i‖ = 1.
        let rot  = Rotation::new(16, 99).unwrap();
        let mut e = vec![0.0f32; 16];
        e[3] = 1.0;
        rot.apply(&mut e);
        assert!((norm(&e) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn different_seeds_give_different_rotations() {
        let r1 = Rotation::new(32, 1).unwrap();
        let r2 = Rotation::new(32, 2).unwrap();
        assert_ne!(r1.signs(), r2.signs());
    }

    #[test]
    fn inner_product_preserved() {
        // ⟨R a, R b⟩ = ⟨a, b⟩
        let rot = Rotation::new(64, 13).unwrap();
        let a: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..64).map(|i| (i as f32).cos()).collect();
        let ip_orig = dot(&a, &b);
        let mut ra = a.clone(); rot.apply(&mut ra);
        let mut rb = b.clone(); rot.apply(&mut rb);
        let ip_rot = dot(&ra, &rb);
        assert!((ip_orig - ip_rot).abs() < 1e-3, "IP not preserved: {ip_orig} vs {ip_rot}");
    }
}
