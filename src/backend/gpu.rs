//! GPU-accelerated backend using CUDA via cudarc.
//!
//! Single-vector Backend trait methods delegate to ScalarBackend
//! to avoid GPU transfer overhead. GPU acceleration is provided
//! through batch-specific methods (added in Plan 03).

use crate::backend::ScalarBackend;
use crate::error::{Result, TurboQuantError};
use cudarc::driver::{CudaDevice, CudaStream};
use std::sync::Arc;

/// CUDA GPU compute backend.
///
/// For single-vector operations (fwht, dot_product), delegates to
/// ScalarBackend — GPU overhead is not justified for single vectors.
/// GPU acceleration comes from batch methods added in Phase 4 Plan 03.
#[derive(Clone, Debug)]
pub struct GpuBackend {
    device: Arc<CudaDevice>,
    scalar: ScalarBackend,
}

impl GpuBackend {
    /// Initialize GPU backend on CUDA device 0.
    ///
    /// Returns `TurboQuantError::GpuInitFailed` with actionable message
    /// if CUDA device unavailable.
    pub fn new() -> Result<Self> {
        Self::with_device(0)
    }

    /// Initialize GPU backend on a specific CUDA device ordinal.
    pub fn with_device(ordinal: usize) -> Result<Self> {
        let device = CudaDevice::new(ordinal).map_err(|e| {
            let reason = if e.to_string().contains("no CUDA-capable device") {
                "No NVIDIA GPU detected".to_string()
            } else if e.to_string().contains("CUDA driver version is insufficient") {
                "CUDA driver outdated — update NVIDIA drivers".to_string()
            } else if e.to_string().contains("libcuda.so") || e.to_string().contains("nvcuda.dll") {
                "CUDA runtime library not found — install CUDA Toolkit".to_string()
            } else {
                format!("CUDA device {} init failed: {}", ordinal, e)
            };
            TurboQuantError::GpuInitFailed { reason }
        })?;

        Ok(Self {
            device: Arc::new(device),
            scalar: ScalarBackend,
        })
    }

    /// Access the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }
}

impl super::Backend for GpuBackend {
    #[inline]
    fn fwht_normalized_inplace(&self, data: &mut [f32]) {
        // Single-vector: delegate to scalar (GPU overhead not justified)
        self.scalar.fwht_normalized_inplace(data);
    }

    #[inline]
    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        // Single-vector: delegate to scalar
        self.scalar.dot_product(a, b)
    }

    fn validate_dimension(&self, dim: usize) -> Result<()> {
        self.scalar.validate_dimension(dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;

    // Note: These tests require a CUDA-capable GPU.
    // Run with: cargo test --features gpu gpu_backend
    // Skip in CI without GPU: these tests will fail gracefully.

    #[test]
    fn gpu_init_error_message_is_actionable() {
        // This test verifies error message quality.
        // On machines without GPU, new() should fail with helpful message.
        // On machines with GPU, it should succeed.
        let result = GpuBackend::new();
        match result {
            Ok(backend) => {
                // GPU available — verify trait methods work
                assert!(backend.validate_dimension(128).is_ok());
                assert!(backend.validate_dimension(7).is_err());
            }
            Err(TurboQuantError::GpuInitFailed { reason }) => {
                // No GPU — verify error is actionable
                assert!(
                    reason.contains("NVIDIA") || reason.contains("CUDA") || reason.contains("init failed"),
                    "Error should mention NVIDIA or CUDA: {reason}"
                );
            }
            Err(other) => panic!("Unexpected error type: {other}"),
        }
    }

    #[test]
    fn gpu_backend_delegates_to_scalar() {
        // Only run if GPU is available
        if let Ok(backend) = GpuBackend::new() {
            // FWHT roundtrip
            let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let mut data = original.clone();
            backend.fwht_normalized_inplace(&mut data);
            backend.fwht_normalized_inplace(&mut data);
            for (a, b) in original.iter().zip(&data) {
                assert!((a - b).abs() < 1e-5, "FWHT roundtrip: {a} vs {b}");
            }

            // Dot product
            let a = vec![1.0f32, 2.0, 3.0, 4.0];
            let b = vec![5.0f32, 6.0, 7.0, 8.0];
            let result = backend.dot_product(&a, &b);
            assert!((result - 70.0).abs() < 1e-6);
        }
    }
}
