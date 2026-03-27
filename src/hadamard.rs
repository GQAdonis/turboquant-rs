//! Fast Walsh-Hadamard Transform (FWHT).
//!
//! The unnormalized Hadamard matrix H_n satisfies H_n · H_n = n·I.
//! The normalized version H̃_n = H_n / √n satisfies H̃_n² = I, making it
//! its own inverse — a crucial property exploited by the rotation module.

/// In-place unnormalized FWHT.
/// Input length must be a power of two.
#[inline]
pub fn fwht_inplace(data: &mut [f32]) {
    assert!(
        data.len().is_power_of_two(),
        "FWHT requires power-of-two length, got {}",
        data.len()
    );

    let mut step = 1usize;
    while step < data.len() {
        let mut i = 0;
        while i < data.len() {
            for j in i..i + step {
                let a = data[j];
                let b = data[j + step];
                data[j]        = a + b;
                data[j + step] = a - b;
            }
            i += 2 * step;
        }
        step <<= 1;
    }
}

/// In-place **normalized** FWHT.
///
/// Applying this twice recovers the original vector (H̃² = I), so it serves
/// as both the forward rotation and its own inverse.
#[inline]
pub fn fwht_normalized_inplace(data: &mut [f32]) {
    fwht_inplace(data);
    let scale = 1.0 / (data.len() as f32).sqrt();
    data.iter_mut().for_each(|x| *x *= scale);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn roundtrip() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data  = original.clone();
        fwht_normalized_inplace(&mut data);
        fwht_normalized_inplace(&mut data);
        for (a, b) in original.iter().zip(&data) {
            assert!(close(*a, *b), "roundtrip failed: {a} vs {b}");
        }
    }

    #[test]
    fn known_values_n4() {
        // H̃₄ [1,0,0,0]^T = [1,1,1,1]^T / 2
        let mut data = vec![1.0f32, 0.0, 0.0, 0.0];
        fwht_normalized_inplace(&mut data);
        for &x in &data {
            assert!(close(x, 0.5), "expected 0.5, got {x}");
        }
    }

    #[test]
    fn orthogonality() {
        // ‖H̃ x‖ = ‖x‖
        let x = vec![1.0f32, -2.0, 3.0, -1.0, 0.5, 1.5, -0.5, 2.0];
        let norm_before: f32 = x.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let mut data = x.clone();
        fwht_normalized_inplace(&mut data);
        let norm_after: f32 = data.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(close(norm_before, norm_after), "norm not preserved: {norm_before} vs {norm_after}");
    }

    #[test]
    #[should_panic(expected = "FWHT requires power-of-two length")]
    fn rejects_non_power_of_two() {
        let mut data = vec![1.0f32; 7];
        fwht_inplace(&mut data);
    }
}
