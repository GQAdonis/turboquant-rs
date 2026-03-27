# Architecture Research: SIMD/GPU Acceleration in Rust

**Domain:** Machine Learning / Numerical Computing (LLM KV-cache compression)
**Researched:** 2026-03-27
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Public API Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ PolarQuant   │  │ TurboQuant   │  │ KvCache      │                   │
│  │ (unchanged)  │  │ (unchanged)  │  │ (unchanged)  │                   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                   │
│         │                  │                  │                           │
├─────────┴──────────────────┴──────────────────┴───────────────────────────┤
│                    Backend Abstraction Layer                              │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  Trait: Backend                                                   │    │
│  │    - fwht_inplace(&mut [f32])                                    │    │
│  │    - l2_norm(&[f32]) -> f32                                      │    │
│  │    - dot_product(&[f32], &[f32]) -> f32                          │    │
│  │    - quantize_batch(&[&[f32]]) -> Vec<QuantizedVector>           │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│         │                      │                      │                   │
├─────────┴──────────────────────┴──────────────────────┴───────────────────┤
│              Backend Implementations (feature-gated)                      │
├───────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐       ┌──────────┐       ┌──────────┐       ┌──────────┐  │
│  │  Scalar  │       │   SIMD   │       │   GPU    │       │  Async   │  │
│  │ (always) │       │ (opt-in) │       │ (opt-in) │       │ (future) │  │
│  └──────────┘       └──────────┘       └──────────┘       └──────────┘  │
│       │                   │                   │                 │         │
│    Portable         AVX2 / NEON           CUDA             cuBLAS        │
│    baseline         std::arch          cudarc/driver     + streams       │
└───────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **Public API** | User-facing interface, backward-compatible | Unchanged existing modules (polar_quant, turboquant, kv_cache) |
| **Backend Trait** | Polymorphic dispatch for hot-path ops | Zero-cost abstraction via generics + monomorphization |
| **Scalar Backend** | Portable baseline implementation | Pure Rust, no unsafe, always compiled |
| **SIMD Backend** | Architecture-specific acceleration | `std::arch::{x86_64, aarch64}`, target features, `unsafe` blocks |
| **GPU Backend** | CUDA kernel execution | `cudarc` for driver API, custom kernels via `nvrtc` or PTX |
| **Feature Flags** | Compile-time backend selection | Cargo features control which backends are compiled |

## Recommended Project Structure

```
src/
├── lib.rs                  # Re-exports, feature flag coordination
├── error.rs                # Existing error types (unchanged)
├── bitpack.rs              # Existing (unchanged)
├── codebook.rs             # Existing (unchanged)
├── rotation.rs             # Existing (unchanged)
├── polar_quant.rs          # Existing public API (unchanged)
├── turboquant.rs           # Existing (unchanged)
├── qjl.rs                  # Existing (unchanged)
├── kv_cache.rs             # Existing (unchanged)
│
├── backends/               # NEW: Backend abstraction layer
│   ├── mod.rs              # Trait definition + feature coordination
│   ├── scalar.rs           # Baseline portable implementation
│   │
│   ├── simd/               # SIMD implementations
│   │   ├── mod.rs          # Platform dispatch logic
│   │   ├── x86_64.rs       # AVX2/AVX-512 intrinsics
│   │   ├── aarch64.rs      # NEON intrinsics
│   │   └── common.rs       # Shared SIMD utilities
│   │
│   └── gpu/                # GPU implementations
│       ├── mod.rs          # GPU backend coordinator
│       ├── cuda/           # CUDA-specific code
│       │   ├── mod.rs      # Device management + kernel launch
│       │   ├── kernels.cu  # CUDA C++ kernels
│       │   └── memory.rs   # Device memory management
│       └── context.rs      # GPU context lifecycle
│
├── hadamard.rs             # MODIFIED: Delegates to Backend
└── ops/                    # NEW: Batch operations
    ├── mod.rs
    ├── batch_quantize.rs   # Batch quantization API
    └── batch_attention.rs  # Batch attention computation
```

### Structure Rationale

- **`backends/`:** Isolates all acceleration logic behind a trait. Public API remains unchanged, making SIMD/GPU purely additive.
- **`backends/scalar.rs`:** Always compiled, ensures library works everywhere. Acts as reference implementation.
- **`backends/simd/`:** Platform-specific intrinsics guarded by `#[cfg(target_arch)]` and `target_feature` checks.
- **`backends/gpu/`:** Optional GPU support via `cuda` feature flag. Completely removable if not needed.
- **`ops/`:** Batch APIs for production ML workloads. Amortizes overhead across multiple vectors.
- **Existing modules:** Unchanged, maintaining backward compatibility. Hot paths internally delegate to Backend.

## Architectural Patterns

### Pattern 1: Zero-Cost Backend Abstraction

**What:** Use trait + monomorphization for backend dispatch with zero runtime overhead.

**When to use:** When you need multiple implementations of the same operation without dynamic dispatch cost.

**Trade-offs:**
- ✅ Zero runtime cost (dispatch resolved at compile time)
- ✅ Type-safe backend selection
- ✅ Easy to add new backends
- ❌ Increased compile time (each backend instantiates all generic code)
- ❌ Larger binary if all features enabled

**Example:**
```rust
// backends/mod.rs
pub trait Backend: Send + Sync {
    fn fwht_inplace(&self, data: &mut [f32]);
    fn l2_norm(&self, v: &[f32]) -> f32;
    // ... other hot-path operations
}

// backends/scalar.rs
#[derive(Debug, Clone, Copy)]
pub struct ScalarBackend;

impl Backend for ScalarBackend {
    #[inline]
    fn fwht_inplace(&self, data: &mut [f32]) {
        // Portable butterfly implementation
        let mut step = 1;
        while step < data.len() {
            let mut i = 0;
            while i < data.len() {
                for j in i..i + step {
                    let a = data[j];
                    let b = data[j + step];
                    data[j] = a + b;
                    data[j + step] = a - b;
                }
                i += 2 * step;
            }
            step <<= 1;
        }
    }

    #[inline]
    fn l2_norm(&self, v: &[f32]) -> f32 {
        v.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }
}

// Usage in polar_quant.rs
pub struct PolarQuant<B: Backend = ScalarBackend> {
    rotation: Rotation,
    codebook: Codebook,
    backend: B,  // Zero-cost generic
}

impl<B: Backend> PolarQuant<B> {
    pub fn quantize(&self, vec: &[f32]) -> Result<QuantizedVector> {
        let norm = self.backend.l2_norm(vec);  // ← Monomorphized at compile time
        // ... rest unchanged
    }
}
```

**Why this works:**
- Compiler monomorphizes `PolarQuant<ScalarBackend>` and `PolarQuant<SimdBackend>` separately
- No vtable, no dynamic dispatch, no runtime cost
- `#[inline]` hints propagate through generic boundaries

### Pattern 2: Target Feature Detection + Dispatch

**What:** Runtime CPU feature detection with compile-time fallback.

**When to use:** When shipping pre-compiled binaries that must work on varied hardware (e.g., AVX2 vs non-AVX2 CPUs).

**Trade-offs:**
- ✅ Single binary works on all CPUs
- ✅ Automatically uses best available instructions
- ✅ Graceful fallback if features unavailable
- ❌ Runtime feature detection overhead (mitigated by caching)
- ❌ Code duplication for each target feature path

**Example:**
```rust
// backends/simd/mod.rs
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy)]
pub struct SimdBackend {
    dispatcher: Dispatcher,
}

#[derive(Debug, Clone, Copy)]
enum Dispatcher {
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Sse2,
    #[cfg(target_arch = "aarch64")]
    Neon,
    Scalar,
}

static DISPATCHER: OnceLock<Dispatcher> = OnceLock::new();

impl SimdBackend {
    pub fn new() -> Self {
        let dispatcher = *DISPATCHER.get_or_init(|| {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    return Dispatcher::Avx2;
                }
                if is_x86_feature_detected!("sse2") {
                    return Dispatcher::Sse2;
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("neon") {
                    return Dispatcher::Neon;
                }
            }
            Dispatcher::Scalar
        });

        Self { dispatcher }
    }
}

impl Backend for SimdBackend {
    #[inline]
    fn fwht_inplace(&self, data: &mut [f32]) {
        match self.dispatcher {
            #[cfg(target_arch = "x86_64")]
            Dispatcher::Avx2 => unsafe {
                // SAFETY: Feature detected at construction
                fwht_avx2(data)
            },
            #[cfg(target_arch = "aarch64")]
            Dispatcher::Neon => unsafe {
                // SAFETY: Feature detected at construction
                fwht_neon(data)
            },
            _ => scalar_fwht(data),
        }
    }
}

// backends/simd/x86_64.rs
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn fwht_avx2(data: &mut [f32]) {
    let mut step = 1;
    while step < data.len() {
        let mut i = 0;
        while i < data.len() {
            // Process 8 elements at a time with AVX2
            let mut j = i;
            while j + 8 <= i + step {
                let a = _mm256_loadu_ps(data.as_ptr().add(j));
                let b = _mm256_loadu_ps(data.as_ptr().add(j + step));
                let sum = _mm256_add_ps(a, b);
                let diff = _mm256_sub_ps(a, b);
                _mm256_storeu_ps(data.as_mut_ptr().add(j), sum);
                _mm256_storeu_ps(data.as_mut_ptr().add(j + step), diff);
                j += 8;
            }
            // Scalar tail for remainder
            while j < i + step {
                let a = data[j];
                let b = data[j + step];
                data[j] = a + b;
                data[j + step] = a - b;
                j += 1;
            }
            i += 2 * step;
        }
        step <<= 1;
    }
}
```

**Key safety:**
- `#[target_feature(enable = "...")]` guarantees the function can only be called if the feature is available
- Runtime detection via `is_x86_feature_detected!` ensures safety before dispatch
- `unsafe` is isolated to SIMD intrinsics blocks

### Pattern 3: GPU Device Context + Memory Pooling

**What:** Manage GPU device lifetime and reuse memory allocations across operations.

**When to use:** For GPU backends where kernel launch overhead and memory transfer dominate.

**Trade-offs:**
- ✅ Amortizes overhead across batch operations
- ✅ Reduces memory allocation churn
- ✅ Supports asynchronous operations
- ❌ Complex lifecycle management
- ❌ Requires careful error handling (GPU errors are async)
- ❌ Memory leaks if context not properly dropped

**Example:**
```rust
// backends/gpu/mod.rs
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream};
use std::sync::Arc;

pub struct GpuBackend {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    kernel: CudaFunction,
    scratch_pool: MemoryPool,
}

impl GpuBackend {
    pub fn new(device_id: usize) -> Result<Self> {
        let device = CudaDevice::new(device_id)?;
        let stream = device.fork_default_stream()?;

        // Compile or load PTX kernel
        let ptx = include_str!("cuda/kernels.ptx");
        let kernel = device.load_ptx(ptx.into(), "fwht_kernel", &["fwht_kernel"])?;

        let scratch_pool = MemoryPool::new(&device, 4096)?;  // 4KB page size

        Ok(Self { device, stream, kernel, scratch_pool })
    }
}

impl Backend for GpuBackend {
    fn fwht_inplace(&self, data: &mut [f32]) {
        // Allocate device memory from pool
        let mut d_data = self.scratch_pool.alloc(data.len()).unwrap();

        // Copy host → device
        d_data.copy_from_slice(data).unwrap();

        // Launch kernel
        let cfg = LaunchConfig::for_num_elems(data.len() as u32);
        unsafe {
            self.kernel
                .launch(cfg, (&mut d_data, data.len() as i32))
                .unwrap();
        }

        // Synchronize and copy device → host
        self.stream.synchronize().unwrap();
        d_data.copy_to_slice(data).unwrap();
    }
}

// backends/gpu/memory.rs
pub struct MemoryPool {
    device: Arc<CudaDevice>,
    free_buffers: Mutex<Vec<CudaSlice<f32>>>,
    page_size: usize,
}

impl MemoryPool {
    pub fn alloc(&self, elements: usize) -> Result<CudaSlice<f32>> {
        let pages = (elements + self.page_size - 1) / self.page_size;
        let size = pages * self.page_size;

        let mut free = self.free_buffers.lock().unwrap();

        // Reuse existing allocation if available
        if let Some(buf) = free.pop() {
            if buf.len() >= size {
                return Ok(buf);
            }
        }

        // Allocate new buffer
        self.device.alloc_zeros::<f32>(size)
    }

    pub fn release(&self, buffer: CudaSlice<f32>) {
        self.free_buffers.lock().unwrap().push(buffer);
    }
}
```

**Key patterns:**
- Device context held via `Arc<CudaDevice>` for shared ownership
- Memory pool reduces allocation overhead
- Stream for asynchronous operations
- Kernel pre-compiled and cached at backend construction

### Pattern 4: Feature Flag Coordination

**What:** Use Cargo features to enable/disable backends at compile time, with clear dependencies.

**When to use:** When shipping a library that supports optional acceleration backends.

**Trade-offs:**
- ✅ Users only pay (compile time + binary size) for what they use
- ✅ No dependencies on CUDA libs if GPU feature not enabled
- ✅ Easy to test individual backends in CI
- ❌ Feature combinations can be tricky (need additive features)
- ❌ Documentation must explain feature flags

**Example:**
```toml
# Cargo.toml
[features]
default = ["std"]
std = []

# Acceleration backends (all optional, additive)
simd = []
simd-avx2 = ["simd"]
simd-neon = ["simd"]
cuda = ["dep:cudarc", "dep:nvrtc"]
gpu = ["cuda"]  # Alias for CUDA, future: add ROCm here

# Batch operations (requires one backend)
batch = []

[dependencies]
thiserror = "2"

[dependencies.cudarc]
version = "0.11"
optional = true

[dependencies.nvrtc]
version = "0.11"
optional = true
```

```rust
// lib.rs
#[cfg(feature = "simd")]
pub use backends::simd::SimdBackend;

#[cfg(feature = "gpu")]
pub use backends::gpu::GpuBackend;

pub use backends::scalar::ScalarBackend;

// Convenience alias for best available backend
#[cfg(feature = "gpu")]
pub type DefaultBackend = GpuBackend;

#[cfg(all(feature = "simd", not(feature = "gpu")))]
pub type DefaultBackend = SimdBackend;

#[cfg(not(any(feature = "simd", feature = "gpu")))]
pub type DefaultBackend = ScalarBackend;

// backends/mod.rs
pub mod scalar;

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "gpu")]
pub mod gpu;

pub trait Backend: Send + Sync {
    fn fwht_inplace(&self, data: &mut [f32]);
    // ...
}
```

**Usage:**
```bash
# Scalar only (smallest binary, portable)
cargo build

# SIMD enabled
cargo build --features simd

# GPU enabled
cargo build --features gpu

# All backends (for benchmarking)
cargo build --features simd,gpu
```

### Pattern 5: Batch Operations with Backend Affinity

**What:** Process multiple vectors in a single call to amortize overhead.

**When to use:** For ML workloads where you quantize/dequantize hundreds of vectors.

**Trade-offs:**
- ✅ GPU: Amortizes kernel launch + memory transfer overhead
- ✅ SIMD: Better instruction pipelining and cache locality
- ✅ Enables async/parallel processing
- ❌ API complexity (must manage output storage)
- ❌ All vectors must have same dimension

**Example:**
```rust
// ops/batch_quantize.rs
pub struct BatchQuantizer<B: Backend> {
    pq: PolarQuant<B>,
    backend: B,
}

impl<B: Backend> BatchQuantizer<B> {
    pub fn quantize_batch(&self, vectors: &[&[f32]]) -> Result<Vec<QuantizedVector>> {
        // Validate all same dimension
        let dim = vectors[0].len();
        for v in vectors {
            if v.len() != dim {
                return Err(TurboQuantError::DimensionMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
        }

        let mut results = Vec::with_capacity(vectors.len());

        // CPU/SIMD: Process in chunks for cache locality
        #[cfg(not(feature = "gpu"))]
        {
            const CHUNK_SIZE: usize = 64;
            for chunk in vectors.chunks(CHUNK_SIZE) {
                for &vec in chunk {
                    results.push(self.pq.quantize(vec)?);
                }
            }
        }

        // GPU: Transfer all at once, process on device
        #[cfg(feature = "gpu")]
        {
            results = self.backend.quantize_batch_gpu(vectors)?;
        }

        Ok(results)
    }
}

// backends/gpu/mod.rs (GPU-specific batch)
impl GpuBackend {
    pub fn quantize_batch_gpu(&self, vectors: &[&[f32]]) -> Result<Vec<QuantizedVector>> {
        let dim = vectors[0].len();
        let count = vectors.len();

        // Flatten all vectors into single buffer
        let mut flat: Vec<f32> = Vec::with_capacity(count * dim);
        for v in vectors {
            flat.extend_from_slice(v);
        }

        // Transfer host → device (single transfer)
        let mut d_data = self.device.htod_copy(flat)?;

        // Launch batch kernel (processes all vectors in parallel)
        let cfg = LaunchConfig {
            grid_dim: (count as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.batch_kernel.launch(cfg, (&mut d_data, dim as i32))?;
        }

        // Copy results back
        self.stream.synchronize()?;
        let results = d_data.to_vec()?;

        // Parse results into QuantizedVectors
        // ... (parsing logic)
    }
}
```

## Data Flow

### Quantization Flow (Single Vector)

```
User Code
    ↓
PolarQuant::quantize(vec)
    ↓
Backend::l2_norm(vec) → norm
    ↓
Normalize: vec / norm
    ↓
Backend::fwht_inplace(rotated)  ← Hot path (SIMD/GPU accelerated)
    ↓
Codebook::quantize_slice(rotated)
    ↓
bitpack::pack(indices)
    ↓
Return QuantizedVector { norm, packed, dim, bits }
```

### Batch Quantization Flow (Multiple Vectors)

```
User Code
    ↓
BatchQuantizer::quantize_batch(&[&[f32]])
    ↓
┌─── CPU/SIMD Path ───┐     ┌─── GPU Path ───────────────────────┐
│ For each chunk:      │     │ Flatten all vectors → single buffer │
│   For each vector:   │     │    ↓                                │
│     quantize(vec)    │     │ Host → Device transfer (one copy)   │
│       ↓              │     │    ↓                                │
│     SIMD FWHT        │     │ Launch batch kernel                 │
│       ↓              │     │   (grid_dim = num_vectors)          │
│     Pack             │     │    ↓                                │
└──────────────────────┘     │ Device → Host transfer              │
                             │    ↓                                │
                             │ Parse results                       │
                             └─────────────────────────────────────┘
    ↓
Return Vec<QuantizedVector>
```

### Attention Computation Flow

```
User: KvCache::attend(query)
    ↓
For each key in cache:
    ↓
  PolarQuant::inner_product(query, key_quantized)
    ↓
  Backend::fwht_inplace(query_rotated)  ← Hot path (SIMD/GPU)
    ↓
  Backend::dot_product(query_rot, codebook_values)  ← Hot path
    ↓
  Multiply by key.norm
    ↓
Store logit
    ↓
Softmax over all logits
    ↓
Weighted sum of values → output
```

### Key Data Flows

1. **Hot path optimization:** FWHT and dot product are delegated to Backend trait, allowing SIMD/GPU acceleration without API changes.
2. **Memory reuse:** Scratch buffers in PolarQuant reduce allocations in attention loop.
3. **Batch amortization:** GPU path transfers all data once, processes in parallel, reducing overhead from O(N) to O(1) + O(N/P) where P = parallelism.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| **Prototype / Research** | Scalar backend only. Simple, portable, sufficient for correctness validation. |
| **Small models (<1B params)** | SIMD backend. 2-4x speedup with minimal complexity. AVX2/NEON sufficient. |
| **Medium models (1-10B)** | SIMD + batch APIs. Process multiple queries together for better cache utilization. |
| **Large models (>10B)** | GPU backend. Kernel launch overhead amortized by large batch sizes (>128 vectors). Memory pooling critical. |

### Scaling Priorities

1. **First bottleneck (at ~1K tokens/sequence):** Memory allocations in inner_product.
   - **Fix:** Scratch buffer reuse (Pattern 3 from ASSESSMENT.md).
   - **Gain:** 1.5-2x speedup, reduces GC pressure.

2. **Second bottleneck (at ~8K tokens):** FWHT scalar operations become dominant.
   - **Fix:** SIMD backend with AVX2/NEON.
   - **Gain:** 2-4x speedup in FWHT (projected from ASSESSMENT.md).

3. **Third bottleneck (at batch size >128):** CPU can't saturate throughput for large batches.
   - **Fix:** GPU backend with batch operations.
   - **Gain:** 10-100x throughput for large batches (depends on GPU vs CPU).

## Anti-Patterns

### Anti-Pattern 1: Dynamic Dispatch for Hot Paths

**What people do:** Use `Box<dyn Backend>` for runtime backend selection.

**Why it's wrong:**
- Vtable indirection prevents inlining (critical for SIMD)
- Adds 1-2ns per call (significant for operations taking <100ns)
- Prevents compiler optimizations across trait boundaries

**Do this instead:** Use generics with monomorphization:
```rust
// ❌ BAD: Dynamic dispatch
pub struct PolarQuant {
    backend: Box<dyn Backend>,
}

// ✅ GOOD: Static dispatch
pub struct PolarQuant<B: Backend = ScalarBackend> {
    backend: B,
}
```

### Anti-Pattern 2: Copying Data for Every GPU Operation

**What people do:** Allocate fresh GPU memory for each kernel call.

**Why it's wrong:**
- GPU memory allocation is slow (~100μs per allocation)
- Memory transfer dominates for small operations (<1ms compute)
- Defeats purpose of GPU acceleration for small workloads

**Do this instead:** Use memory pooling and batch operations:
```rust
// ❌ BAD
fn fwht_gpu(&self, data: &[f32]) -> Vec<f32> {
    let d_data = self.device.htod_copy(data).unwrap();  // Allocate every time
    // ... kernel ...
    d_data.to_vec().unwrap()
}

// ✅ GOOD
fn fwht_batch_gpu(&self, data: &[&[f32]]) -> Vec<Vec<f32>> {
    let d_data = self.scratch_pool.alloc(total_size).unwrap();  // Reuse
    // ... batch kernel ...
}
```

### Anti-Pattern 3: Feature Flag Explosion

**What people do:** Create a feature flag for every possible combination:
```toml
# ❌ BAD: Exponential combinations
[features]
simd-avx2 = []
simd-avx512 = []
gpu-cuda = []
gpu-rocm = []
simd-avx2-and-cuda = ["simd-avx2", "gpu-cuda"]
simd-avx512-and-cuda = ["simd-avx512", "gpu-cuda"]
# ... 2^N combinations
```

**Why it's wrong:**
- Unmaintainable as backends grow
- Users don't understand which to enable
- CI must test all combinations

**Do this instead:** Use additive features that compose:
```toml
# ✅ GOOD: Additive features
[features]
default = ["std"]
std = []
simd = []  # Enables all available SIMD for target
gpu = ["dep:cudarc"]  # Enables GPU support
batch = []  # Enables batch APIs
```

### Anti-Pattern 4: Ignoring Target Feature Safety

**What people do:** Use SIMD intrinsics without proper guards:
```rust
// ❌ BAD: Undefined behavior if AVX2 not available
fn fwht_avx2(data: &mut [f32]) {
    unsafe {
        let a = _mm256_loadu_ps(data.as_ptr());  // May crash!
    }
}
```

**Why it's wrong:**
- Causes illegal instruction fault on CPUs without the feature
- Silent data corruption if CPU doesn't support instructions
- No compile-time or runtime protection

**Do this instead:** Use target_feature + runtime detection:
```rust
// ✅ GOOD: Safe dispatch
#[target_feature(enable = "avx2")]
unsafe fn fwht_avx2(data: &mut [f32]) {
    let a = _mm256_loadu_ps(data.as_ptr());
    // ... safe to use AVX2 here
}

fn fwht(data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { fwht_avx2(data) };
        }
    }
    fwht_scalar(data)  // Fallback
}
```

### Anti-Pattern 5: Premature GPU Optimization

**What people do:** Start with GPU implementation before profiling.

**Why it's wrong:**
- GPU has high fixed overhead (~10μs kernel launch)
- Only beneficial for operations >100μs or large batches
- SIMD often sufficient and simpler (2-4x vs 10-100x GPU)
- Adds significant complexity (memory management, error handling)

**Do this instead:** Optimize in order:
1. Profile to find actual bottlenecks (use `cargo flamegraph`)
2. Fix allocations (scratch buffers, pre-allocation)
3. Add SIMD (2-3 days, 2-4x gain)
4. Add GPU only if:
   - Batch size >128 vectors, OR
   - Single operation >1ms compute time, OR
   - Profiling shows CPU saturated

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| **CUDA Runtime** | `cudarc::driver::CudaDevice` | Requires CUDA 11.0+ installed. Feature-gated behind `cuda` feature. |
| **GPU Memory** | `cudarc::driver::CudaSlice<T>` | Typed GPU buffers. Use memory pool to reduce allocation overhead. |
| **Kernel Compilation** | `nvrtc` for runtime PTX, or pre-compiled `.ptx` | Runtime compilation flexible but slower. Pre-compiled faster but less portable. |
| **SIMD Intrinsics** | `std::arch::{x86_64, aarch64}` | Requires nightly for some features. Use stable `is_x86_feature_detected!` for dispatch. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| **Public API ↔ Backend** | Trait method calls (monomorphized) | Zero-cost abstraction. Compiler inlines across boundary. |
| **Backend ↔ SIMD** | `unsafe` intrinsics | Isolated to simd/ modules. Safety documented via SAFETY comments. |
| **Backend ↔ GPU** | cudarc driver API | Error handling via `Result`. GPU errors can be async (use `stream.synchronize()`). |
| **Batch API ↔ Single** | Batch calls single repeatedly (CPU) or native batch (GPU) | CPU path falls back to loop. GPU path uses specialized kernels. |

## Build Order Implications

Based on component dependencies and complexity, suggested phase structure:

### Phase 1: Backend Abstraction (Foundation)
**What:** Create `Backend` trait and `ScalarBackend` implementation.
**Why first:** Establishes abstraction without changing behavior. Allows testing infrastructure before adding complexity.
**Components:**
- `backends/mod.rs` (trait definition)
- `backends/scalar.rs` (existing logic extracted)
- Modify `hadamard.rs`, `polar_quant.rs` to delegate to Backend

**Complexity:** Low (refactoring existing code)
**Risk:** Low (no new behavior, just abstraction)

### Phase 2: SIMD Backend (High Value)
**What:** Implement `SimdBackend` with AVX2 (x86_64) and NEON (ARM).
**Why second:** High impact (2-4x speedup), moderate complexity, no external dependencies.
**Components:**
- `backends/simd/mod.rs` (dispatcher)
- `backends/simd/x86_64.rs` (AVX2 intrinsics)
- `backends/simd/aarch64.rs` (NEON intrinsics)
- Feature flags: `simd`

**Complexity:** Medium (unsafe code, platform-specific)
**Risk:** Medium (must validate correctness across architectures)

### Phase 3: Batch Operations (Efficiency)
**What:** Batch quantization and batch attention APIs.
**Why third:** Prepares for GPU (requires batching to amortize overhead), useful for CPU too.
**Components:**
- `ops/batch_quantize.rs`
- `ops/batch_attention.rs`
- Feature flag: `batch`

**Complexity:** Low-Medium (API design, memory management)
**Risk:** Low (builds on existing single-vector ops)

### Phase 4: GPU Backend (Scale)
**What:** Implement `GpuBackend` with CUDA kernels.
**Why fourth:** Highest complexity, requires batching infrastructure from Phase 3.
**Components:**
- `backends/gpu/mod.rs`
- `backends/gpu/cuda/` (kernels, device management)
- `backends/gpu/memory.rs` (memory pooling)
- Feature flags: `cuda`, `gpu`
- Dependencies: `cudarc`, `nvrtc`

**Complexity:** High (GPU programming, memory management, async errors)
**Risk:** High (many failure modes, platform-specific)

### Dependency Graph

```
Phase 1 (Backend Trait)
    ↓
    ├─→ Phase 2 (SIMD)
    │       ↓
    └─→ Phase 3 (Batch APIs)
                ↓
            Phase 4 (GPU)
```

**Critical path:** Phase 1 must complete before any other phase.
**Parallelizable:** Phase 2 (SIMD) and Phase 3 (Batch) can be developed independently after Phase 1.
**Sequential:** Phase 4 (GPU) depends on Phase 3 (batching infrastructure).

## Performance Expectations

### Single Vector Operations (128-dim, 3-bit)

| Backend | FWHT (μs) | Quantize (μs) | Inner Product (μs) | Speedup vs Scalar |
|---------|-----------|---------------|---------------------|-------------------|
| Scalar | ~0.8 | ~2.5 | ~1.2 | 1x (baseline) |
| SIMD (AVX2) | ~0.2-0.3 | ~0.8-1.0 | ~0.4-0.6 | 2.5-4x |
| SIMD (NEON) | ~0.3-0.4 | ~1.0-1.2 | ~0.5-0.7 | 2-3x |
| GPU | ~50 (overhead) | N/A (too small) | N/A (too small) | 0.02x (worse!) |

**Key insight:** GPU is slower than CPU for single vectors due to launch overhead (~10-50μs).

### Batch Operations (128 vectors, 128-dim, 3-bit)

| Backend | Quantize Batch (μs) | Throughput (vectors/ms) | Speedup vs Scalar |
|---------|---------------------|-------------------------|-------------------|
| Scalar | ~320 | ~400 | 1x |
| SIMD (AVX2) | ~100 | ~1280 | 3.2x |
| GPU (CUDA) | ~150 (includes transfer) | ~850 | 2.1x |

### Large Batch (1024 vectors, 128-dim)

| Backend | Quantize Batch (ms) | Throughput (vectors/ms) | Speedup vs Scalar |
|---------|---------------------|-------------------------|-------------------|
| Scalar | ~2.6 | ~394 | 1x |
| SIMD (AVX2) | ~0.8 | ~1280 | 3.25x |
| GPU (CUDA) | ~0.4 | ~2560 | 6.5x |

**Key insight:** GPU advantage emerges at batch size >128 where transfer overhead is amortized.

## Sources

**Architecture patterns:**
- Rust std::arch documentation (official, HIGH confidence)
- cudarc crate documentation (official, HIGH confidence)
- Personal knowledge of SIMD optimization patterns (MEDIUM confidence - based on training data)
- faer, nalgebra, and burn crate architectures (MEDIUM confidence - architectural patterns)

**Backend abstraction:**
- Zero-cost abstractions from Rust Book (official, HIGH confidence)
- Trait design patterns from Rust API Guidelines (official, HIGH confidence)

**GPU patterns:**
- CUDA Best Practices Guide (NVIDIA official, HIGH confidence)
- cudarc examples and documentation (official, HIGH confidence)

**Feature flags:**
- Cargo Book on features (official, HIGH confidence)
- Rust target_feature documentation (official, HIGH confidence)

**Performance projections:**
- Based on ASSESSMENT.md findings (project-specific, HIGH confidence)
- SIMD speedup estimates from training data (MEDIUM confidence - typical for butterfly ops)
- GPU overhead characteristics from CUDA documentation (HIGH confidence)

**Confidence assessment:**
- Architecture patterns: HIGH (well-established Rust idioms)
- SIMD implementation: HIGH (std::arch is stable and documented)
- GPU implementation: MEDIUM-HIGH (cudarc is stable but GPU programming complex)
- Performance numbers: MEDIUM (projections based on assessment + typical speedups)

---
*Architecture research for: TurboQuant Rust SIMD/GPU integration*
*Researched: 2026-03-27*
