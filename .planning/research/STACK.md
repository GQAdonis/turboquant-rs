# Stack Research: SIMD and GPU Acceleration for Rust ML

**Domain:** Performance optimization for Rust ML quantization library
**Researched:** 2026-03-27
**Confidence:** MEDIUM (based on training data + existing Rust ecosystem patterns, not verified with current documentation)

## Executive Summary

For adding SIMD and GPU acceleration to Rust ML libraries in 2025/2026, the standard approach uses:
- **std::arch** for portable SIMD (zero dependencies, stable)
- **cudarc** for CUDA GPU support (most mature Rust CUDA wrapper)
- **Feature flags** for conditional compilation (cargo standard)
- **Runtime detection** for SIMD capability checks

This stack minimizes dependencies while maximizing performance and portability.

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **std::arch** | Rust 1.70+ (stable) | SIMD intrinsics (AVX2, NEON) | Zero-dependency, stable since Rust 1.27, covers 95% of SIMD needs, compile-time + runtime checks available |
| **cudarc** | 0.12+ | CUDA GPU integration | Most mature Rust CUDA wrapper (2025), type-safe kernel launches, good ergonomics, active maintenance |
| **cargo features** | Built-in | Backend selection | Standard Rust pattern for optional dependencies, enables SIMD/GPU/CPU-only builds |

**Rationale:**
- **std::arch over portable_simd**: std::arch is stable, portable_simd still nightly-only as of early 2025
- **std::arch over simdeez/packed_simd**: No extra dependencies, stable API, better IDE support
- **cudarc over cuda-sys/cust**: Higher-level API, better type safety, active development
- **Feature flags**: Industry standard for conditional GPU/SIMD compilation (candle, burn, etc. all use this pattern)

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **raw-cpuid** | 11.0+ | Runtime CPU feature detection | Detect AVX2/FMA support at runtime, graceful fallback to scalar code |
| **cuda-runtime-sys** | 0.3+ | CUDA runtime bindings | Dependency of cudarc, manage CUDA memory/streams |
| **bindgen** | 0.69+ (build-dep) | Generate CUDA FFI bindings | If writing custom CUDA kernels beyond cudarc's abstractions |
| **libc** | 0.2+ | Platform detection | ARM NEON feature detection on Linux/macOS |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| **cargo-asm** | Inspect generated assembly | Verify SIMD codegen quality: `cargo asm --release --target-cpu native` |
| **nvidia-smi** | GPU monitoring | Check CUDA version, memory usage during development |
| **perf** (Linux) / **Instruments** (macOS) | CPU profiling | Verify SIMD paths are being taken, identify cache misses |
| **nsight-compute** | GPU profiling | Detailed CUDA kernel performance analysis |

## Installation

```toml
# Cargo.toml

[dependencies]
thiserror = "2"

# Optional: CPU feature detection for runtime dispatch
raw-cpuid = { version = "11", optional = true }

# Optional: CUDA GPU support
cudarc = { version = "0.12", optional = true, features = ["cuda-12"] }

[features]
# Default: CPU-only with runtime SIMD detection
default = ["simd-runtime"]

# SIMD variants
simd-compile-time = []  # Compile with -C target-cpu=native
simd-runtime = ["raw-cpuid"]  # Runtime detection + fallback

# GPU support
cuda = ["cudarc"]
cuda-11 = ["cudarc/cuda-11"]
cuda-12 = ["cudarc/cuda-12"]

[build-dependencies]
# Only if writing custom CUDA kernels
bindgen = { version = "0.69", optional = true }

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1

# For maximum SIMD performance (x86_64 only)
[profile.release-native]
inherits = "release"
# Users can build with: RUSTFLAGS="-C target-cpu=native" cargo build --release
```

```bash
# CPU-only build (SIMD runtime detection)
cargo build --release --features simd-runtime

# CPU-only build (native SIMD, not portable)
RUSTFLAGS="-C target-cpu=native" cargo build --release --features simd-compile-time

# GPU build (CUDA 12)
cargo build --release --features cuda

# All features
cargo build --release --features cuda,simd-runtime
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| **std::arch** | portable_simd | If nightly Rust is acceptable AND you need cross-lane operations not in std::arch |
| **std::arch** | packed_simd | Never (unmaintained since 2020) |
| **std::arch** | simdeez | If you need unified ARM+x86 SIMD with same code (trades API simplicity for portability) |
| **cudarc** | cust | If you need more control over CUDA context management |
| **cudarc** | cuda-sys | If you need raw CUDA C API access (very low-level) |
| **cudarc** | wgpu/wgsl | If cross-platform GPU (Metal/Vulkan) > raw performance (adds ~30% overhead vs CUDA) |
| **Feature flags** | cfg!(target_feature) | Only for std::arch intrinsics, not for dependencies |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **packed_simd** | Unmaintained since 2020, superseded by portable_simd | std::arch (stable) or portable_simd (nightly) |
| **faster** crate | Abandoned, doesn't work with modern Rust | std::arch |
| **arrayfire-rust** | Heavy dependency, GPU support incomplete | cudarc for CUDA, wgpu for cross-platform |
| **nvptx-* targets** | Requires nightly Rust, experimental CUDA codegen | cudarc with PTX kernels |
| **opencl-rust** | OpenCL deprecated by Apple/NVIDIA, poor tooling | CUDA (NVIDIA), Metal (Apple), Vulkan (cross-platform) |

## Architecture Patterns

### Pattern 1: Multi-Backend with Feature Flags

```rust
// lib.rs
#[cfg(feature = "cuda")]
mod gpu;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod simd_avx2;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod simd_neon;

mod scalar;  // Fallback

pub struct PolarQuant {
    rotation: Rotation,
    codebook: Codebook,

    #[cfg(feature = "cuda")]
    gpu_ctx: Option<gpu::GpuContext>,

    scratch: RefCell<Vec<f32>>,
}

impl PolarQuant {
    pub fn fwht_inplace(&self, data: &mut [f32]) {
        #[cfg(feature = "cuda")]
        if let Some(ctx) = &self.gpu_ctx {
            return ctx.fwht_gpu(data);
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            unsafe { simd_avx2::fwht_avx2(data) }
            return;
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            unsafe { simd_neon::fwht_neon(data) }
            return;
        }

        scalar::fwht_scalar(data);
    }
}
```

### Pattern 2: Runtime SIMD Detection

```rust
// runtime_dispatch.rs
use std::sync::OnceLock;

#[cfg(feature = "simd-runtime")]
use raw_cpuid::CpuId;

static HAS_AVX2: OnceLock<bool> = OnceLock::new();

pub fn detect_avx2() -> bool {
    *HAS_AVX2.get_or_init(|| {
        #[cfg(all(target_arch = "x86_64", feature = "simd-runtime"))]
        {
            CpuId::new()
                .get_feature_info()
                .map_or(false, |f| f.has_avx2())
        }
        #[cfg(not(all(target_arch = "x86_64", feature = "simd-runtime")))]
        {
            false
        }
    })
}

pub fn fwht_inplace(data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if detect_avx2() && is_x86_feature_detected!("avx2") {
        unsafe {
            return avx2::fwht_avx2(data);
        }
    }

    scalar::fwht_scalar(data);
}
```

### Pattern 3: CUDA Batch Processing

```rust
#[cfg(feature = "cuda")]
pub mod gpu {
    use cudarc::driver::*;
    use cudarc::nvrtc::Ptx;

    pub struct GpuContext {
        device: Arc<CudaDevice>,
        fwht_kernel: CudaFunction,
    }

    impl GpuContext {
        pub fn new() -> Result<Self> {
            let device = CudaDevice::new(0)?;

            // Load PTX kernel (compiled separately)
            let ptx = Ptx::from_file("kernels/fwht.ptx")?;
            let module = device.load_ptx(ptx, "fwht_module", &[])?;
            let fwht_kernel = module.get_function("fwht_butterfly")?;

            Ok(Self { device, fwht_kernel })
        }

        pub fn fwht_batch(&self, batch: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
            // Flatten batch
            let total_len: usize = batch.iter().map(|v| v.len()).sum();
            let mut flat: Vec<f32> = Vec::with_capacity(total_len);
            for vec in batch {
                flat.extend_from_slice(vec);
            }

            // Upload to GPU
            let d_input = self.device.htod_copy(&flat)?;

            // Launch kernel
            let cfg = LaunchConfig {
                grid_dim: (batch.len() as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.fwht_kernel.launch(cfg, (&d_input,))?;
            }

            // Download result
            let result = self.device.dtoh_sync_copy(&d_input)?;

            // Unflatten
            Ok(unflatten(result, batch.iter().map(|v| v.len()).collect()))
        }
    }
}
```

## Stack Patterns by Variant

**If targeting x86_64 servers (AWS, GCP, on-prem):**
- Use compile-time SIMD: `RUSTFLAGS="-C target-cpu=native"`
- Enable AVX2 unconditionally (available since Haswell 2013)
- Reason: Maximum performance, controlled environment

**If targeting ARM devices (Apple Silicon, AWS Graviton):**
- Use runtime SIMD detection with NEON fallback
- NEON is mandatory on aarch64, but check for advanced features (dot product, FP16)
- Reason: Heterogeneous ARM ecosystem

**If targeting consumer devices (desktop app, CLI tool):**
- Use runtime detection: `raw-cpuid` + `is_x86_feature_detected!`
- Provide scalar fallback
- Reason: Wide hardware support, graceful degradation

**If targeting NVIDIA GPUs:**
- Use CUDA 11.8+ for broadest compatibility (GTX 10-series through RTX 40-series)
- Batch size ≥ 32 for efficient GPU utilization
- Reason: CUDA 11.8 is widely available, 12.x still rolling out

**If targeting cross-platform GPU (AMD, Intel, Apple):**
- Use wgpu + wgsl compute shaders
- Accept 20-30% performance penalty vs CUDA
- Reason: Portability > peak performance

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| cudarc 0.12.x | CUDA 11.8, 12.0-12.5 | Use feature flags: `cudarc/cuda-11` or `cudarc/cuda-12` |
| std::arch | Rust 1.27+ | Stable, no compatibility issues |
| raw-cpuid 11.x | Rust 1.70+ | x86_64 only, use libc for ARM detection |
| criterion 0.5 | Rust 1.70+ | Benchmarking with SIMD requires `--release` |

**CUDA Compatibility:**
- CUDA 11.8: GTX 10-series (Pascal) through RTX 40-series (Ada Lovelace)
- CUDA 12.x: RTX 30-series (Ampere) and newer preferred, backwards compatible
- Driver requirement: ≥ 520.61.05 (Linux) or ≥ 527.41 (Windows) for CUDA 12

**Rust Version:**
- Minimum: 1.70 (for cudarc type system features)
- Recommended: 1.75+ (better SIMD codegen, improved type inference)
- Current: 1.94.1 ✓ (fully compatible)

## Platform-Specific Notes

### x86_64 (Intel/AMD)

**SIMD Tiers:**
1. **SSE2**: Baseline (available on all x86_64)
2. **AVX2**: Target this (2013+, 99% of deployed servers)
3. **AVX-512**: Optional (Xeon Scalable, not on consumer CPUs)

**Feature Detection:**
```rust
#[cfg(target_arch = "x86_64")]
{
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // Use AVX2 + FMA path (8x f32 per instruction)
    } else {
        // Fallback to SSE2 (4x f32 per instruction)
    }
}
```

### ARM (Apple Silicon, Graviton)

**SIMD Tiers:**
1. **NEON**: Baseline (mandatory on aarch64)
2. **NEON + FP16**: Apple M1+, Graviton3+
3. **SVE/SVE2**: Future (not widely available)

**Feature Detection:**
```rust
#[cfg(target_arch = "aarch64")]
{
    // NEON is always available on aarch64
    // Check for advanced features:
    #[cfg(target_feature = "fp16")]
    {
        // Use FP16 acceleration
    }
}
```

### CUDA (NVIDIA GPUs)

**Compute Capabilities:**
- **7.0** (Volta): V100, Tesla T4
- **7.5** (Turing): RTX 20-series, GTX 16-series
- **8.0** (Ampere): A100, RTX 30-series
- **8.6** (Ampere): RTX 30-series consumer
- **8.9** (Ada Lovelace): RTX 40-series

**Minimum for ML workloads:** 7.0 (Volta, 2017)
**Recommended target:** 7.5+ (covers 90% of active GPUs)

## Performance Considerations

### SIMD Alignment

```rust
// Align data for efficient SIMD loads/stores
#[repr(align(32))]  // AVX2 requires 32-byte alignment
pub struct AlignedBuffer {
    data: Vec<f32>,
}

// Or use slice alignment checks:
fn is_aligned(ptr: *const f32, align: usize) -> bool {
    (ptr as usize) % align == 0
}
```

### Batch Size Guidelines

| Backend | Minimum Batch | Optimal Batch | Reason |
|---------|---------------|---------------|--------|
| AVX2 | 8 | 32-128 | Amortize loop overhead, cache locality |
| NEON | 4 | 16-64 | Smaller SIMD width, cache-conscious |
| CUDA | 32 | 256-1024 | SM occupancy, hide memory latency |

### Memory Layout

- **AoS vs SoA**: SIMD prefers SoA (Structure of Arrays)
- **Padding**: Pad to SIMD width multiples (e.g., 128-dim is perfect for AVX2)
- **Prefetching**: Use `_mm_prefetch` (x86) or `__builtin_prefetch` for predictable access patterns

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simd_matches_scalar() {
        let mut data_simd = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data_scalar = data_simd.clone();

        #[cfg(target_arch = "x86_64")]
        unsafe {
            simd::fwht_avx2(&mut data_simd);
        }

        scalar::fwht_scalar(&mut data_scalar);

        for (a, b) in data_simd.iter().zip(data_scalar.iter()) {
            assert!((a - b).abs() < 1e-5, "SIMD diverged from scalar");
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn gpu_matches_cpu() {
        let ctx = GpuContext::new().unwrap();
        let input = vec![vec![1.0; 128]; 32];

        let gpu_result = ctx.fwht_batch(&input).unwrap();
        let cpu_result: Vec<_> = input.iter()
            .map(|v| { let mut v = v.clone(); fwht_inplace(&mut v); v })
            .collect();

        assert_results_close(&gpu_result, &cpu_result);
    }
}
```

## Sources

**Confidence Level: MEDIUM**

- **std::arch**: HIGH confidence (stable Rust feature, well-documented)
- **cudarc**: MEDIUM confidence (based on training data, could not verify current version/docs)
- **Feature flags pattern**: HIGH confidence (standard Rust practice, observed in candle, burn, tch-rs)
- **Performance numbers**: LOW confidence (theoretical, not benchmarked on this codebase)
- **Version numbers**: LOW confidence (based on training data from early 2025, not verified with crates.io)

**Unable to verify:**
- cudarc latest version (listed 0.12+, could be newer)
- raw-cpuid latest version (listed 11.0+, could be newer)
- CUDA 12.x specific compatibility notes

**Verification needed:**
- Check crates.io for cudarc, raw-cpuid latest versions
- Verify CUDA 11.8 vs 12.x compatibility in 2026
- Benchmark actual performance gains on target hardware

**References (from training data):**
- Rust std::arch documentation (stable since 1.27)
- cudarc crate patterns (observed in burn, dfdx)
- NVIDIA CUDA documentation (11.8, 12.x series)
- Rust embedded working group SIMD patterns
- ML framework patterns (candle, burn, tch-rs, dfdx)

---
*Stack research for: SIMD and GPU acceleration for Rust ML quantization*
*Researched: 2026-03-27*
*Note: This research is based primarily on training data due to unavailability of live documentation sources. Version numbers and specific features should be verified against current crates.io and official documentation before implementation.*
