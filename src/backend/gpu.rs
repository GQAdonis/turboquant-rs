//! GPU-accelerated backend using CUDA via cudarc.
//!
//! Single-vector Backend trait methods delegate to ScalarBackend
//! to avoid GPU transfer overhead. GPU acceleration is provided
//! through batch-specific methods (added in Plan 03).

use crate::backend::ScalarBackend;
use crate::error::{Result, TurboQuantError};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// PTX modules embedded from build output
const FWHT_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/fwht.ptx"));
const ATTENTION_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/attention.ptx"));

/// Minimum batch size to route to GPU kernels.
/// Below this, CPU (scalar/SIMD + rayon) is faster due to GPU transfer overhead.
pub const GPU_BATCH_THRESHOLD: usize = 32;

/// CUDA GPU compute backend.
///
/// For single-vector operations (fwht, dot_product), delegates to
/// ScalarBackend — GPU overhead is not justified for single vectors.
/// GPU acceleration comes from batch methods added in Phase 4 Plan 03.
#[derive(Clone, Debug)]
pub struct GpuBackend {
    device: Arc<CudaDevice>,
    scalar: ScalarBackend,
    modules_loaded: Arc<Mutex<bool>>,
    /// Pool of reusable device buffers, keyed by size in elements.
    /// Each size maps to a stack of available buffers.
    f32_pool: Arc<Mutex<HashMap<usize, Vec<CudaSlice<f32>>>>>,
    u8_pool: Arc<Mutex<HashMap<usize, Vec<CudaSlice<u8>>>>>,
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
            modules_loaded: Arc::new(Mutex::new(false)),
            f32_pool: Arc::new(Mutex::new(HashMap::new())),
            u8_pool: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Access the underlying CUDA device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Ensure CUDA kernel modules are loaded (lazy, thread-safe).
    fn ensure_modules_loaded(&self) -> Result<()> {
        let mut loaded = self.modules_loaded.lock().unwrap();
        if !*loaded {
            self.device
                .load_ptx(Ptx::from_src(FWHT_PTX), "fwht_module", &["fwht_batch"])
                .map_err(|e| TurboQuantError::GpuKernelFailed {
                    reason: format!("Failed to load FWHT PTX: {}", e),
                })?;

            self.device
                .load_ptx(
                    Ptx::from_src(ATTENTION_PTX),
                    "attention_module",
                    &["batch_dot_product", "batch_dequantize"],
                )
                .map_err(|e| TurboQuantError::GpuKernelFailed {
                    reason: format!("Failed to load attention PTX: {}", e),
                })?;

            *loaded = true;
        }
        Ok(())
    }

    /// Get or allocate a device f32 buffer of `count` elements.
    pub fn get_f32_buffer(&self, count: usize) -> Result<CudaSlice<f32>> {
        let mut pool = self.f32_pool.lock().unwrap();
        if let Some(stack) = pool.get_mut(&count) {
            if let Some(buf) = stack.pop() {
                return Ok(buf);
            }
        }
        self.device.alloc_zeros::<f32>(count).map_err(|e| TurboQuantError::GpuAllocFailed {
            size: count,
            reason: e.to_string(),
        })
    }

    /// Return a device f32 buffer to the pool for reuse.
    pub fn return_f32_buffer(&self, count: usize, buffer: CudaSlice<f32>) {
        let mut pool = self.f32_pool.lock().unwrap();
        pool.entry(count).or_default().push(buffer);
    }

    /// Get or allocate a device u8 buffer of `count` elements.
    pub fn get_u8_buffer(&self, count: usize) -> Result<CudaSlice<u8>> {
        let mut pool = self.u8_pool.lock().unwrap();
        if let Some(stack) = pool.get_mut(&count) {
            if let Some(buf) = stack.pop() {
                return Ok(buf);
            }
        }
        self.device.alloc_zeros::<u8>(count).map_err(|e| TurboQuantError::GpuAllocFailed {
            size: count,
            reason: e.to_string(),
        })
    }

    /// Return a device u8 buffer to the pool for reuse.
    pub fn return_u8_buffer(&self, count: usize, buffer: CudaSlice<u8>) {
        let mut pool = self.u8_pool.lock().unwrap();
        pool.entry(count).or_default().push(buffer);
    }

    /// Clear all pooled buffers (frees GPU memory).
    pub fn clear_pools(&self) {
        self.f32_pool.lock().unwrap().clear();
        self.u8_pool.lock().unwrap().clear();
    }

    /// Launch batch FWHT kernel on GPU.
    ///
    /// `d_data` must be a device buffer of size `batch_size * dim`.
    /// Modifies data in-place on device.
    pub fn launch_fwht_batch(
        &self,
        d_data: &mut CudaSlice<f32>,
        dim: usize,
        batch_size: usize,
    ) -> Result<()> {
        self.ensure_modules_loaded()?;

        let threads_per_block = dim.min(256) as u32;
        let shared_mem_bytes = (dim * std::mem::size_of::<f32>()) as u32;
        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes,
        };

        let func = self
            .device
            .get_func("fwht_module", "fwht_batch")
            .ok_or_else(|| TurboQuantError::GpuKernelFailed {
                reason: "fwht_batch kernel not found in loaded module".to_string(),
            })?;

        // SAFETY: kernel params match fwht_batch signature
        unsafe { func.launch(cfg, (d_data, dim as i32, batch_size as i32)) }.map_err(|e| {
            TurboQuantError::GpuKernelFailed {
                reason: format!("fwht_batch launch failed: {}", e),
            }
        })
    }

    /// Launch batch dot product kernel on GPU.
    ///
    /// Computes: results[i] = dot(query, keys[i]) * norms[i]
    pub fn launch_batch_dot_product(
        &self,
        d_query: &CudaSlice<f32>,
        d_keys: &CudaSlice<f32>,
        d_norms: &CudaSlice<f32>,
        d_results: &mut CudaSlice<f32>,
        dim: usize,
        batch_size: usize,
    ) -> Result<()> {
        self.ensure_modules_loaded()?;

        let threads_per_block = dim.min(256) as u32;
        let shared_mem_bytes = (threads_per_block as usize * std::mem::size_of::<f32>()) as u32;
        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes,
        };

        let func = self
            .device
            .get_func("attention_module", "batch_dot_product")
            .ok_or_else(|| TurboQuantError::GpuKernelFailed {
                reason: "batch_dot_product kernel not found".to_string(),
            })?;

        unsafe {
            func.launch(
                cfg,
                (
                    d_query,
                    d_keys,
                    d_norms,
                    d_results,
                    dim as i32,
                    batch_size as i32,
                ),
            )
        }
        .map_err(|e| TurboQuantError::GpuKernelFailed {
            reason: format!("batch_dot_product launch failed: {}", e),
        })
    }

    /// Launch batch dequantize kernel on GPU.
    pub fn launch_batch_dequantize(
        &self,
        d_packed: &CudaSlice<u8>,
        d_centroids: &CudaSlice<f32>,
        d_output: &mut CudaSlice<f32>,
        dim: usize,
        bits: u8,
        packed_bytes_per_vec: usize,
        batch_size: usize,
    ) -> Result<()> {
        self.ensure_modules_loaded()?;

        let threads_per_block = dim.min(256) as u32;
        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        let func = self
            .device
            .get_func("attention_module", "batch_dequantize")
            .ok_or_else(|| TurboQuantError::GpuKernelFailed {
                reason: "batch_dequantize kernel not found".to_string(),
            })?;

        unsafe {
            func.launch(
                cfg,
                (
                    d_packed,
                    d_centroids,
                    d_output,
                    dim as i32,
                    bits as i32,
                    packed_bytes_per_vec as i32,
                    batch_size as i32,
                ),
            )
        }
        .map_err(|e| TurboQuantError::GpuKernelFailed {
            reason: format!("batch_dequantize launch failed: {}", e),
        })
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

    #[test]
    fn gpu_fwht_batch_matches_scalar() {
        let gpu = match GpuBackend::new() {
            Ok(g) => g,
            Err(_) => return, // Skip if no GPU
        };

        let dim = 128;
        let batch_size = 4;
        let scalar = ScalarBackend;

        // Create test vectors
        let mut cpu_data: Vec<f32> = (0..batch_size * dim)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();

        // CPU reference
        let mut cpu_ref = cpu_data.clone();
        for b in 0..batch_size {
            let start = b * dim;
            scalar.fwht_normalized_inplace(&mut cpu_ref[start..start + dim]);
        }

        // GPU
        let mut d_data = gpu
            .device
            .htod_sync_copy(&cpu_data)
            .expect("htod copy");
        gpu.launch_fwht_batch(&mut d_data, dim, batch_size)
            .expect("kernel launch");
        let gpu_result = gpu.device.dtoh_sync_copy(&d_data).expect("dtoh copy");

        // Compare
        for (i, (cpu_val, gpu_val)) in cpu_ref.iter().zip(&gpu_result).enumerate() {
            assert!(
                (cpu_val - gpu_val).abs() < 1e-4,
                "FWHT mismatch at index {i}: CPU={cpu_val}, GPU={gpu_val}"
            );
        }
    }

    #[test]
    fn gpu_memory_pool_reuse() {
        let gpu = match GpuBackend::new() {
            Ok(g) => g,
            Err(_) => return, // Skip if no GPU
        };

        // Allocate and return a buffer
        let buf = gpu.get_f32_buffer(128).expect("alloc");
        gpu.return_f32_buffer(128, buf);

        // Second allocation should reuse
        let buf2 = gpu.get_f32_buffer(128).expect("reuse");
        // Different size should allocate fresh
        let buf3 = gpu.get_f32_buffer(256).expect("alloc new size");

        gpu.return_f32_buffer(128, buf2);
        gpu.return_f32_buffer(256, buf3);
        gpu.clear_pools();
    }
}
