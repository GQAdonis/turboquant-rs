# Feature Flags

turboquant provides optional acceleration backends via Cargo feature flags. The core library has zero required dependencies beyond `thiserror` and `rayon`.

## Available Features

| Feature | Backend | Dependency | Platforms |
|---------|---------|------------|-----------|
| (default) | `ScalarBackend` | None | All |
| `simd` | `SimdBackend`, `RuntimeBackend` | None (uses `std::arch`) | x86_64 (AVX2), aarch64 (NEON) |
| `gpu` | `GpuBackend` | `cudarc` 0.12 | Linux/Windows with NVIDIA GPU + CUDA Toolkit |

## Build Commands

### Default (scalar only)

No acceleration, works everywhere:

    cargo build --release
    cargo test

### SIMD Acceleration

2-4x speedup on FWHT operations. Auto-detects CPU features at runtime:

    cargo build --release --features simd
    cargo test --features simd
    cargo bench --features simd

### GPU Acceleration

10x+ speedup for batch operations (batch size >= 32). Requires NVIDIA GPU and CUDA Toolkit:

    cargo build --release --features gpu
    cargo test --features gpu --release
    cargo bench --features gpu

### All Features

    cargo build --release --features simd,gpu
    cargo test --features simd,gpu --release

## Backend Selection

### ScalarBackend (default)

Used automatically when no feature flags are enabled. All operations use standard Rust scalar arithmetic.

    use turboquant::PolarQuant;

    let pq = PolarQuant::new(128, 3, 42)?;

### SimdBackend (requires `simd` feature)

Vectorized FWHT and dot product using AVX2 (x86_64) or NEON (aarch64):

    use turboquant::{PolarQuant, SimdBackend};

    let pq = PolarQuant::new_with_backend(128, 3, 42, SimdBackend)?;

`RuntimeBackend` auto-detects SIMD support and falls back to scalar:

    use turboquant::{PolarQuant, RuntimeBackend};

    let backend = RuntimeBackend::detect();
    let pq = PolarQuant::new_with_backend(128, 3, 42, backend)?;

### GpuBackend (requires `gpu` feature)

CUDA-accelerated batch operations. Single-vector operations delegate to scalar (GPU transfer overhead not justified):

    use turboquant::{PolarQuant, GpuBackend};

    let gpu = GpuBackend::new()?;  // Initializes CUDA device 0
    let pq = PolarQuant::new_with_backend(128, 3, 42, gpu)?;

    // Batch operations >= 32 vectors use GPU kernels
    let results = pq.batch_quantize_dispatch(&large_batch)?;

    // Single operations use scalar (no GPU overhead)
    let qv = pq.quantize(&single_vector)?;

## GPU Prerequisites

To use the `gpu` feature, you need:

1. **NVIDIA GPU** — Any CUDA-capable GPU (Volta/sm_70 or newer recommended)
2. **CUDA Toolkit 11.8+** — Install from https://developer.nvidia.com/cuda-downloads
3. **NVIDIA drivers** — Latest stable from https://www.nvidia.com/Download/index.aspx

Verify installation:

    nvcc --version    # Should show CUDA 11.8+
    nvidia-smi        # Should show GPU info

### Troubleshooting

**"GPU initialization failed: No NVIDIA GPU detected"**
- Ensure NVIDIA GPU is present and drivers installed
- On cloud VMs, verify GPU instance type (e.g., AWS p3/g4, GCP A2/G2)

**"GPU initialization failed: CUDA runtime library not found"**
- Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- Ensure nvcc is in PATH: `which nvcc`

**"GPU initialization failed: CUDA driver outdated"**
- Update NVIDIA drivers to match CUDA Toolkit version

**Build error: "nvcc not found"**
- The `gpu` feature requires nvcc to compile CUDA kernels at build time
- Install CUDA Toolkit and add to PATH

## Combining Features

Features are additive and independent:
- `simd` affects single-vector FWHT and dot product performance
- `gpu` affects batch operations (>= 32 vectors)
- Both can be enabled simultaneously for maximum performance
- Neither feature changes the public API signatures or output values
