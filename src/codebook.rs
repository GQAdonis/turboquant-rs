//! Precomputed Lloyd-Max codebooks.
//!
//! After a random rotation, each coordinate of a d-dimensional unit vector
//! follows a Beta((d-1)/2, (d-1)/2) distribution on [-1, 1].  For large d
//! (e.g. d = 128) this is well-approximated by N(0, 1/d).
//!
//! The Lloyd-Max algorithm solves the continuous k-means problem for this
//! distribution, yielding the MSE-optimal centroids for scalar quantization.
//! We store the N(0,1) centroids and scale them by 1/√d at construction time.
//!
//! Sources: Lloyd (1982), Max (1960), and empirical validation from the
//! turboquant-pytorch and turboquant_plus community implementations.

// ── N(0,1) Lloyd-Max centroids ─────────────────────────────────────────────
// 2-bit  → 4 centroids
const C_2BIT: [f32; 4] = [-1.510_4, -0.452_8, 0.452_8, 1.510_4];

// 3-bit  → 8 centroids
const C_3BIT: [f32; 8] = [
    -2.151_9, -1.343_9, -0.756_0, -0.245_1,
     0.245_1,  0.756_0,  1.343_9,  2.151_9,
];

// 4-bit  → 16 centroids
const C_4BIT: [f32; 16] = [
    -2.732_6, -2.069_4, -1.618_0, -1.256_2,
    -0.942_4, -0.656_8, -0.388_1, -0.128_4,
     0.128_4,  0.388_1,  0.656_8,  0.942_4,
     1.256_2,  1.618_0,  2.069_4,  2.732_6,
];
// ───────────────────────────────────────────────────────────────────────────

/// MSE-optimal scalar codebook for a given bit-width and head dimension.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Centroids scaled to N(0, 1/dim).  Always sorted ascending.
    centroids: Vec<f32>,
    /// Decision boundaries — midpoints of adjacent centroids.
    boundaries: Vec<f32>,
    /// Bit width (2, 3, or 4).
    pub bits: u8,
}

impl Codebook {
    /// Build a codebook for `bits`-bit quantization of d-dimensional vectors.
    pub fn new(bits: u8, dim: usize) -> crate::error::Result<Self> {
        use crate::error::TurboQuantError;
        let base: &[f32] = match bits {
            2 => &C_2BIT,
            3 => &C_3BIT,
            4 => &C_4BIT,
            _ => return Err(TurboQuantError::UnsupportedBitWidth { bits }),
        };
        // Scale from N(0,1) to N(0, 1/dim)
        let scale      = 1.0 / (dim as f32).sqrt();
        let centroids: Vec<f32>  = base.iter().map(|&c| c * scale).collect();
        let boundaries: Vec<f32> = centroids.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect();
        Ok(Self { centroids, boundaries, bits })
    }

    /// Number of centroids (= 2^bits).
    #[inline]
    #[must_use]
    pub fn size(&self) -> usize { self.centroids.len() }

    /// Raw centroid slice.
    #[inline]
    #[must_use]
    pub fn centroids(&self) -> &[f32] { &self.centroids }

    /// Quantize a single scalar value → index in 0..size().
    #[inline]
    #[must_use]
    pub fn quantize_scalar(&self, v: f32) -> u8 {
        // `partition_point` returns the first index where boundary[i] ≥ v.
        self.boundaries.partition_point(|&b| b < v) as u8
    }

    /// Dequantize an index → centroid value.
    #[inline]
    #[must_use]
    pub fn dequantize_scalar(&self, idx: u8) -> f32 {
        self.centroids[idx as usize % self.centroids.len()]
    }

    /// Batch quantize.
    #[must_use]
    pub fn quantize_slice(&self, values: &[f32]) -> Vec<u8> {
        values.iter().map(|&v| self.quantize_scalar(v)).collect()
    }

    /// Batch dequantize.
    #[must_use]
    pub fn dequantize_slice(&self, indices: &[u8]) -> Vec<f32> {
        indices.iter().map(|&i| self.dequantize_scalar(i)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_3bit() {
        let cb = Codebook::new(3, 128).unwrap();
        // For a centroid value, quantize→dequantize must be a no-op.
        for (i, &c) in cb.centroids.iter().enumerate() {
            let idx = cb.quantize_scalar(c);
            assert_eq!(idx as usize, i, "centroid {i} round-trips to wrong idx {idx}");
        }
    }

    #[test]
    fn sorted_centroids() {
        for bits in [2u8, 3, 4] {
            let cb = Codebook::new(bits, 128).unwrap();
            assert!(cb.centroids.windows(2).all(|w| w[0] < w[1]), "centroids not sorted");
        }
    }

    #[test]
    fn scale_inversely_with_dim() {
        let cb64  = Codebook::new(3, 64).unwrap();
        let cb128 = Codebook::new(3, 128).unwrap();
        // cb64 centroids should be larger by √(128/64) = √2
        let ratio = cb64.centroids[0] / cb128.centroids[0];
        assert!((ratio - std::f32::consts::SQRT_2).abs() < 1e-4,
            "scale ratio wrong: {ratio}");
    }
}
