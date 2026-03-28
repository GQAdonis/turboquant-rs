use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum TurboQuantError {
    #[error("dimension {0} is not a power of two")]
    DimensionNotPowerOfTwo(usize),

    #[error("bit width {bits} is unsupported; must be 2, 3, or 4")]
    UnsupportedBitWidth { bits: u8 },

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("input is empty")]
    EmptyInput,

    #[error("GPU initialization failed: {reason}\n\nTo use GPU acceleration:\n1. Install NVIDIA CUDA Toolkit 11.8+ from https://developer.nvidia.com/cuda-downloads\n2. Verify: nvcc --version\n3. Update NVIDIA GPU drivers\n\nFor CPU-only: cargo build --release (omit --features gpu)")]
    GpuInitFailed { reason: String },

    #[error("GPU memory allocation failed: requested {size} floats — {reason}")]
    GpuAllocFailed { size: usize, reason: String },

    #[error("GPU kernel launch failed: {reason}")]
    GpuKernelFailed { reason: String },
}

pub type Result<T> = std::result::Result<T, TurboQuantError>;
