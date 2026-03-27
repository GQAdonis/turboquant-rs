# Pitfalls Research

**Domain:** Rust SIMD/GPU Performance Optimization (ML Inference)
**Researched:** 2026-03-27
**Confidence:** MEDIUM-HIGH (based on training data, Rust documentation knowledge, project assessment analysis)

## Critical Pitfalls

### Pitfall 1: Alignment Violations in SIMD Operations

**What goes wrong:**
SIMD intrinsics like AVX2 `_mm256_load_ps` require 32-byte aligned memory. Using them with unaligned data causes segmentation faults or undefined behavior. The program may work in debug mode but crash in release, or work on developer machines but fail in production.

**Why it happens:**
- Vec<f32> allocations have no alignment guarantees beyond the type's natural alignment (4 bytes for f32)
- Slicing aligned data can produce unaligned subslices
- Developers assume Rust's memory safety extends to alignment (it doesn't for unsafe SIMD)

**How to avoid:**
```rust
// BAD - no alignment guarantee
let data: Vec<f32> = vec![0.0; 256];
unsafe {
    let ptr = data.as_ptr();
    _mm256_load_ps(ptr); // May crash!
}

// GOOD - use unaligned loads
unsafe {
    let ptr = data.as_ptr();
    _mm256_loadu_ps(ptr); // Always safe (just slower)
}

// BETTER - use aligned allocations when performance critical
#[repr(align(32))]
struct AlignedF32([f32; 256]);

let data = AlignedF32([0.0; 256]);
unsafe {
    let ptr = data.0.as_ptr();
    _mm256_load_ps(ptr); // Guaranteed safe
}
```

**Warning signs:**
- Crashes only in release builds or with `-C target-cpu=native`
- Intermittent segfaults that disappear when adding debug prints
- Valgrind/MSAN reports unaligned access
- Crashes only on certain CPU models (newer CPUs may be more strict)

**Phase to address:**
Phase 1 (SIMD foundation) - Establish alignment discipline from the start. Use `_loadu_` variants initially, profile to determine if aligned loads are needed.

---

### Pitfall 2: Feature Detection Runtime Mismatch

**What goes wrong:**
Code compiled with AVX2 intrinsics runs on a CPU without AVX2 support, causing illegal instruction crashes. Or runtime detection succeeds but the binary wasn't compiled with the required features, so intrinsics aren't available.

**Why it happens:**
- Confusion between compile-time `#[cfg(target_feature)]` and runtime `is_x86_feature_detected!`
- Building with `RUSTFLAGS="-C target-cpu=native"` works on build machine but not on deployment
- Feature detection checks succeed but call into code that wasn't compiled with that feature

**How to avoid:**
```rust
// BAD - compiles but may crash at runtime
#[cfg(target_arch = "x86_64")]
unsafe fn simd_operation(data: &mut [f32]) {
    use std::arch::x86_64::*;
    // Uses AVX2 intrinsics without checking CPU support
    let v = _mm256_loadu_ps(data.as_ptr());
}

// GOOD - runtime detection + feature-gated compilation
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
unsafe fn simd_operation_avx2(data: &mut [f32]) {
    use std::arch::x86_64::*;
    let v = _mm256_loadu_ps(data.as_ptr());
    // ... AVX2 operations
}

pub fn simd_operation(data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            #[cfg(target_feature = "avx2")]
            unsafe { return simd_operation_avx2(data); }
        }
    }
    // Fallback scalar implementation
    scalar_operation(data);
}
```

**Better approach - use function multi-versioning:**
```rust
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
unsafe fn fwht_avx2(data: &mut [f32]) {
    // AVX2 implementation
}

pub fn fwht(data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { fwht_avx2(data); }
            return;
        }
    }
    fwht_scalar(data);
}
```

**Warning signs:**
- Illegal instruction crashes on different machines
- Works in development, fails in CI or production
- Crashes only when specific code paths are executed
- `cargo build` succeeds but runtime crashes immediately

**Phase to address:**
Phase 1 (SIMD foundation) - Set up proper feature detection and fallback infrastructure before implementing any SIMD code.

---

### Pitfall 3: CUDA Memory Transfer Dominates Performance

**What goes wrong:**
GPU kernels run blazingly fast, but overall performance is slower than CPU because of PCI-e transfer overhead. Naive implementations transfer data to GPU, compute, transfer back for every operation.

**Why it happens:**
- Underestimating PCI-e bandwidth costs (GB/s vs TB/s memory bandwidth)
- Not measuring end-to-end latency (kernel time != total time)
- Premature GPU adoption for small workloads
- Missing opportunities to batch operations

**How to avoid:**
```rust
// BAD - transfer overhead dominates
for query in queries {
    let query_gpu = device.htod_copy(query)?; // Copy to GPU
    let result = kernel.launch((query_gpu, cache_gpu))?;
    let result_cpu = device.dtoh_copy(result)?; // Copy from GPU
    results.push(result_cpu); // Transfer is 90% of time!
}

// GOOD - batch transfers
let queries_gpu = device.htod_copy(all_queries)?; // One transfer
let results_gpu = batch_kernel.launch((queries_gpu, cache_gpu))?;
let results_cpu = device.dtoh_copy(results_gpu)?; // One transfer
```

**Rules of thumb:**
- GPU makes sense when: `compute_time / transfer_time > 10`
- For FWHT on 128-dim vectors: needs ~1000+ operations to break even
- Single attention query: probably CPU-bound by transfer
- Batch of 64 queries: GPU wins

**Warning signs:**
- GPU kernel shows 10x speedup but overall performance is worse
- `nvprof` shows 90%+ time in memcpy operations
- CPU implementation still faster despite "optimized" GPU code
- Performance degrades with smaller batch sizes

**Phase to address:**
Phase 2 (GPU foundation) - Implement batch API first, then GPU backend. Measure round-trip latency, not just kernel time. Consider GPU optional for small batches.

---

### Pitfall 4: Unsafe Code Invalidates Safety Invariants

**What goes wrong:**
SIMD/GPU code uses `unsafe` to bypass bounds checks for performance. This invalidates safety assumptions elsewhere in the codebase, causing memory corruption, use-after-free, or data races that only manifest under load.

**Why it happens:**
- "It's just a performance optimization" attitude toward unsafe
- No documentation of safety requirements in unsafe functions
- Unsafe spreads from leaf functions to callers
- Testing doesn't catch all UB (undefined behavior)

**How to avoid:**
```rust
// BAD - unsafe leaks to public API
pub fn fwht_inplace(data: &mut [f32]) {
    unsafe {
        // Fast path with no bounds checks
        for i in 0..data.len() {
            // No length checks - caller must ensure power-of-two!
            *data.get_unchecked_mut(i) = ...;
        }
    }
}

// GOOD - validate at unsafe boundary
pub fn fwht_inplace(data: &mut [f32]) {
    assert!(data.len().is_power_of_two(),
            "FWHT requires power-of-two length");

    // SAFETY: We've validated length is power-of-two, which means
    // all index calculations below are in-bounds
    unsafe {
        fwht_inplace_unchecked(data);
    }
}

unsafe fn fwht_inplace_unchecked(data: &mut [f32]) {
    // Document SAFETY requirements at definition
    // SAFETY: Caller must ensure:
    // - data.len() is power of two
    // - data is valid for reads/writes for entire length
    ...
}
```

**Safety documentation template:**
```rust
/// # Safety
///
/// Caller must ensure:
/// - `data.len()` is a power of two
/// - `data` points to valid, initialized memory
/// - No other references to `data` exist during call
/// - Data is properly aligned for SIMD operations (32-byte for AVX2)
#[inline]
unsafe fn simd_operation_unchecked(data: *mut f32, len: usize) {
    // Implementation
}
```

**Warning signs:**
- Crashes only under load or in production
- Miri or AddressSanitizer finds violations
- Intermittent memory corruption
- Data races detected by ThreadSanitizer
- Works in single-threaded tests, fails with parallelism

**Phase to address:**
Phase 1 (SIMD foundation) - Establish unsafe guidelines before writing any unsafe code. Run Miri on test suite. Document all safety requirements.

---

### Pitfall 5: Power-of-Two Assumption Violated at Edges

**What goes wrong:**
FWHT requires power-of-two dimensions, but real-world inputs have arbitrary dimensions. Naive implementations crash, silently corrupt data, or produce incorrect results when given non-power-of-two inputs.

**Why it happens:**
- Algorithm constraints not enforced at API boundaries
- Padding strategy unclear or inconsistent
- Dimension validation in some paths but not others
- Tests only use power-of-two dimensions

**How to avoid:**
```rust
// BAD - crashes on real-world data
pub fn quantize(&mut self, vector: &[f32]) -> Result<QuantizedVector> {
    let mut v = vector.to_vec();
    fwht_inplace(&mut v); // Panics if len not power-of-two!
    // ...
}

// GOOD - validate and reject
pub fn quantize(&mut self, vector: &[f32]) -> Result<QuantizedVector> {
    if !vector.len().is_power_of_two() {
        return Err(TurboQuantError::DimensionNotPowerOfTwo {
            got: vector.len(),
        });
    }
    // ...
}

// BETTER - pad transparently (if semantics allow)
pub fn quantize(&mut self, vector: &[f32]) -> Result<QuantizedVector> {
    let padded_len = vector.len().next_power_of_two();
    let mut v = vector.to_vec();
    v.resize(padded_len, 0.0); // Zero-padding

    fwht_inplace(&mut v);
    // ... remember original length for dequantization
}
```

**Validation strategy for TurboQuant:**
- **At quantizer creation:** Validate `dim` is power-of-two
- **At quantization:** Validate vector length matches expected dim
- **At decompression:** Validate packed data length is consistent
- **In tests:** Explicitly test non-power-of-two rejection

**Warning signs:**
- Crashes with "index out of bounds" in production
- Incorrect results that pass tests (because tests use clean dimensions)
- Users report "works with 128-dim but not 100-dim"
- Buffer overflows in SIMD code

**Phase to address:**
Phase 0 (prerequisite) - Add assertion to FWHT immediately. Phase 1 ensures all entry points validate dimensions.

---

### Pitfall 6: Benchmark-Driven Development Without Real Workload Testing

**What goes wrong:**
Microbenchmarks show 10x speedup from SIMD/GPU, but production performance improves by only 20%. Optimizations target the wrong bottlenecks, or artificial benchmark conditions don't reflect real usage.

**Why it happens:**
- Benchmarks use small, cache-friendly datasets
- Real workloads dominated by I/O, not compute
- Benchmark setup cost amortized away (unrealistic)
- Hot loop in benchmark vs. cold path in production

**How to avoid:**
```rust
// BAD - synthetic benchmark only
fn bench_fwht(c: &mut Criterion) {
    let mut data = vec![1.0; 128];
    c.bench_function("fwht_128", |b| {
        b.iter(|| fwht_inplace(&mut data));
    });
}
// This shows SIMD is fast, but doesn't measure real impact!

// GOOD - end-to-end benchmark
fn bench_kv_cache_attention(c: &mut Criterion) {
    let mut cache = KVCache::new(128, 3).unwrap();

    // Realistic setup: 2048 cached tokens
    for _ in 0..2048 {
        let key = random_vector(128);
        let value = random_vector(128);
        cache.insert(key, value).unwrap();
    }

    let query = random_vector(128);

    c.bench_function("attention_2048_tokens", |b| {
        b.iter(|| {
            cache.attend(&query).unwrap()
        });
    });
}
```

**Benchmark suite requirements:**
- Microbenchmarks: Prove optimization works in isolation
- Integration benchmarks: Measure real API usage patterns
- Memory benchmarks: Track allocation counts, not just timing
- Comparison benchmarks: CPU baseline vs. SIMD vs. GPU

**Warning signs:**
- Benchmark improvements don't translate to production
- Users report "still slow" despite optimization work
- Profiling shows different bottleneck than expected
- Performance regresses on different hardware

**Phase to address:**
Phase 0 (prerequisite) - Add realistic benchmarks before starting optimization. Phase 2 validates optimizations against real workload benchmarks.

---

### Pitfall 7: CUDA Kernel Launch Overhead Ignored

**What goes wrong:**
GPU kernel launches have ~5-20μs overhead. For small operations, this overhead exceeds computation time, making GPU slower than CPU despite higher theoretical throughput.

**Why it happens:**
- Focusing on TFLOPS numbers instead of latency
- Not accounting for kernel launch and sync costs
- Missing CPU/GPU switching overhead
- Underestimating CPU performance on small data

**How to avoid:**
```rust
// BAD - GPU for tiny operations
pub fn fwht_single(&self, data: &mut [f32]) -> Result<()> {
    let data_gpu = self.device.htod_copy(data)?; // ~10μs
    self.kernel.launch(data_gpu)?;               // ~10μs
    self.device.dtoh_copy(data_gpu, data)?;      // ~10μs
    // Total: ~30μs for operation that takes 0.5μs on CPU!
}

// GOOD - GPU only for batched operations
pub fn fwht_batch(&self, batch: &mut [Vec<f32>]) -> Result<()> {
    if batch.len() < 64 {
        // Below break-even point, use CPU
        for data in batch {
            fwht_cpu(data);
        }
        return Ok(());
    }

    // GPU path for large batches
    let batch_gpu = self.device.htod_copy_batch(batch)?;
    self.batch_kernel.launch(batch_gpu)?;
    self.device.dtoh_copy_batch(batch_gpu, batch)?;
    Ok(())
}
```

**Break-even analysis for TurboQuant:**
```
Operation: FWHT on 128-dim vector
CPU time:     ~0.8μs (with SIMD)
GPU setup:    ~30μs (transfer + launch)
GPU compute:  ~0.05μs per vector
Break-even:   ~40 vectors per batch

Operation: Inner product (quantized)
CPU time:     ~0.6μs (with SIMD)
GPU setup:    ~30μs
GPU compute:  ~0.02μs per query
Break-even:   ~50 queries per batch
```

**Warning signs:**
- GPU slower than CPU on single operations
- `nvprof` shows kernel time << launch overhead
- Batch size of 1 has terrible performance
- High-latency despite low GPU utilization

**Phase to address:**
Phase 2 (GPU integration) - Implement batch API first, add runtime batch-size threshold for CPU/GPU dispatch.

---

### Pitfall 8: Iterator Invalidation in Scratch Buffer Reuse

**What goes wrong:**
Reusing scratch buffers with `RefCell` or interior mutability causes panics when iterators or borrows overlap. Code that works with fresh allocations breaks when "optimized" with buffer reuse.

**Why it happens:**
- `RefCell::borrow_mut()` panics if already borrowed
- Scratch buffer held across function calls that need it
- Iterator holds reference to buffer that gets reused
- Interior mutability + complex control flow = hard to reason about

**How to avoid:**
```rust
// BAD - RefCell panic at runtime
pub struct PolarQuant {
    scratch: RefCell<Vec<f32>>,
}

pub fn inner_product(&self, query: &[f32], key: &QuantizedVector) -> Result<f32> {
    let mut scratch = self.scratch.borrow_mut(); // Borrow 1
    scratch.clear();
    scratch.extend_from_slice(query);

    self.rotation.apply(&mut scratch); // If rotation also borrows scratch: PANIC!
    // ...
}

// GOOD - explicit buffer passing
pub struct PolarQuant {
    // No interior mutability
}

pub fn inner_product(
    &self,
    query: &[f32],
    key: &QuantizedVector,
    scratch: &mut Vec<f32>, // Caller owns buffer
) -> Result<f32> {
    scratch.clear();
    scratch.extend_from_slice(query);
    self.rotation.apply(scratch);
    // ...
}

// BETTER - batch API with managed buffers
pub fn inner_product_batch(
    &self,
    queries: &[&[f32]],
    keys: &[&QuantizedVector],
) -> Result<Vec<f32>> {
    let mut scratch = Vec::with_capacity(self.dim);

    queries.iter().zip(keys.iter()).map(|(q, k)| {
        self.inner_product(q, k, &mut scratch)
    }).collect()
}
```

**Alternative: Thread-local scratch buffers:**
```rust
thread_local! {
    static SCRATCH_BUFFER: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

pub fn inner_product(&self, query: &[f32], key: &QuantizedVector) -> Result<f32> {
    SCRATCH_BUFFER.with(|scratch| {
        let mut buf = scratch.borrow_mut();
        buf.clear();
        buf.extend_from_slice(query);
        // Use buf locally, drops before function returns
        // ...
    })
}
```

**Warning signs:**
- `RefCell` panic: "already borrowed: BorrowMutError"
- Works single-threaded, panics with parallelism
- Intermittent panics in production
- Tests pass, integration tests fail

**Phase to address:**
Phase 1 (allocation optimization) - Choose buffer strategy before implementing. If using RefCell, audit all potential borrow overlaps.

---

### Pitfall 9: False Sharing in Parallel SIMD Processing

**What goes wrong:**
Parallel SIMD code scales poorly across threads despite embarrassingly parallel workload. Threads thrash CPU cache by writing to adjacent memory locations (same cache line).

**Why it happens:**
- Multiple threads writing to adjacent array elements
- Cache line = 64 bytes = 16x f32 values
- Thread 1 writes `results[0..7]`, Thread 2 writes `results[8..15]`
- Both on same cache line → constant invalidation → poor scaling

**How to avoid:**
```rust
// BAD - false sharing
fn parallel_quantize(vectors: &[Vec<f32>]) -> Vec<QuantizedVector> {
    let results = vec![None; vectors.len()];

    vectors.par_iter().enumerate().for_each(|(i, v)| {
        results[i] = Some(quantize(v)); // Adjacent writes = cache thrash
    });

    results.into_iter().map(|r| r.unwrap()).collect()
}

// GOOD - each thread accumulates locally
fn parallel_quantize(vectors: &[Vec<f32>]) -> Vec<QuantizedVector> {
    vectors.par_iter()
        .map(|v| quantize(v)) // Each thread returns value
        .collect() // Parallel collect avoids false sharing
}

// ALSO GOOD - pad to cache line boundaries
#[repr(align(64))]
struct CacheAligned<T>(T);

let results: Vec<CacheAligned<Option<QuantizedVector>>> =
    vec![CacheAligned(None); vectors.len()];
```

**When it matters:**
- Batch processing with 4+ threads
- Large batches (>100 items)
- Each item processes quickly (<10μs)
- Seeing sub-linear scaling with threads

**Warning signs:**
- 4-thread performance < 2x single-thread
- `perf stat` shows high cache misses
- Performance worse with more threads
- CPU usage high but throughput low

**Phase to address:**
Phase 3 (batch parallelism) - Design parallel API to avoid shared memory. Use parallel iterators with thread-local buffers.

---

### Pitfall 10: Cargo Feature Flags Creating Inconsistent Builds

**What goes wrong:**
Different combinations of feature flags produce subtly incompatible ABIs. Binary compiled with `--features simd` can't link against dependency without `simd` feature. Users get mysterious link errors or runtime crashes.

**Why it happens:**
- Feature flags change struct layouts (`#[cfg(feature = "simd")]` adds fields)
- SIMD intrinsics change function signatures
- Dependencies compiled with different feature sets
- No testing of feature flag combinations

**How to avoid:**
```rust
// BAD - feature affects ABI
#[derive(Clone)]
pub struct PolarQuant {
    rotation: Rotation,
    codebook: Codebook,

    #[cfg(feature = "simd")]
    simd_impl: SIMDImpl, // ← Struct size changes!
}

// GOOD - feature affects implementation, not data
pub struct PolarQuant {
    rotation: Rotation,
    codebook: Codebook,
    // Always same layout
}

impl PolarQuant {
    fn fwht_impl(&self, data: &mut [f32]) {
        #[cfg(feature = "simd")]
        { return fwht_simd(data); }

        #[cfg(not(feature = "simd"))]
        { fwht_scalar(data); }
    }
}

// ALSO GOOD - enum dispatch (zero-cost)
enum FWHTImpl {
    Scalar,
    #[cfg(feature = "simd")]
    SIMD,
}

pub struct PolarQuant {
    rotation: Rotation,
    codebook: Codebook,
    impl_type: FWHTImpl, // Always present, different variants
}
```

**Feature flag design for TurboQuant:**
```toml
[features]
default = []

# Additive features - don't change ABI
simd = []
cuda = ["dep:cudarc"]
all = ["simd", "cuda"]

# Runtime detection (no feature needed)
# Better: detect at runtime, compile all paths
```

**Testing strategy:**
```bash
# Test all feature combinations
cargo test
cargo test --features simd
cargo test --features cuda
cargo test --all-features
cargo test --no-default-features

# Verify ABI compatibility
cargo build
cargo build --features simd
nm target/debug/libturboquant.a | grep -E "PolarQuant.*sizeof"
```

**Warning signs:**
- "Undefined reference" linker errors
- Works with `cargo build` but not as dependency
- Crashes only when used as library
- Different behavior when installed via cargo vs. git

**Phase to address:**
Phase 1 (SIMD foundation) - Design feature flag strategy before implementation. Add CI job testing feature combinations.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Using `_loadu_ps` (unaligned) instead of aligned allocations | Works everywhere, no alignment hassle | 10-20% slower than aligned loads | MVP only - optimize in Phase 2 |
| Scalar fallback only, no runtime dispatch | Fast to implement | Users with AVX2 CPUs don't benefit | Never - runtime dispatch is trivial |
| `panic!` on non-power-of-two | Simple error handling | Library unusable in production | Never - `Result<T>` is Rust idiom |
| Single-vector API only | Simple, easy to test | Can't amortize GPU overhead | Acceptable for Phase 1, must fix Phase 2 |
| Copy all data to GPU every operation | Straightforward | Transfer overhead kills performance | Never - defeats GPU purpose |
| `#[cfg(feature = "simd")]` on public APIs | Easy feature gating | Breaks ABI compatibility | Never - use runtime dispatch |
| No SAFETY documentation on unsafe code | Saves time writing docs | Impossible to audit or maintain | Never - unsafe without docs is UB waiting to happen |
| Microbenchmarks only | Quick validation | Misleading performance claims | Acceptable if paired with integration bench |
| Hard-coded batch size threshold | Avoids tuning complexity | Suboptimal on different hardware | Acceptable for v1, should auto-tune later |
| `Vec<T>` reallocation on every call | No state management | Allocation overhead | Never - scratch buffers are easy |

## Integration Gotchas

Common mistakes when integrating performance features.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| CUDA Runtime | Forgetting to initialize device context | Call `cuInit(0)` before any CUDA operations |
| SIMD Intrinsics | Assuming feature availability | Always check `is_x86_feature_detected!` at runtime |
| Parallel Rayon | Spawning tasks for tiny work | Use `.par_iter()` only for batch sizes >100 |
| Feature Flags | Making them affect public API types | Keep API identical, vary implementation only |
| Unsafe SIMD | Using `_load_ps` with slice pointers | Use `_loadu_ps` or prove alignment with `#[repr(align)]` |
| GPU Memory | Mixing device and host pointers | Use type system: `DevicePtr<T>` vs `HostPtr<T>` |
| Scratch Buffers | Holding `RefCell::borrow_mut()` across calls | Borrow, use, drop immediately in tight scope |
| Power-of-Two | Only checking in release mode (`debug_assert!`) | Use `assert!` for algorithm requirements |
| Criterion Bench | Not using `black_box` for return values | Compiler optimizes away computation: `black_box(result)` |
| FWHT Dimensions | Padding without tracking original size | Store `(padded_vec, original_len)` for correct decompression |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Per-operation GPU transfer | Single-query performance terrible | Batch API with single transfer | N=1 vs N=64 queries |
| No allocation pooling | Memory fragmentation over time | Reuse buffers, object pools | After ~1M operations |
| Naive parallel batch processing | Sublinear scaling with threads | Chunk work to avoid false sharing | >4 threads |
| CPU/GPU synchronization in loop | GPU idle waiting for CPU | Async streams, overlap compute/transfer | High query rate (>1000 QPS) |
| SIMD without residual loop | Crashes on non-multiple-of-8 lengths | Handle tail elements separately | Real-world dimension sizes |
| Stack allocation of work buffers | Stack overflow with large batches | Heap allocation for large buffers | Batch size >128 |
| Cargo build without optimization | 10-100x slower, misleading profiling | Always profile with `--release` | Any performance measurement |
| Single memory allocator | Lock contention in multi-threaded | Thread-local allocators (tcmalloc) | High thread count (>8) |
| Kernel launch per element | Launch overhead dominates | Batch kernel with grid stride loop | GPU batch size <32 |
| No memory prefetching | Cache miss stalls | Manual prefetch for strided access | Working set > L3 cache |

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| No bounds checking in unsafe SIMD | Out-of-bounds read/write → RCE | Assert preconditions before unsafe blocks |
| Trusting dimension from untrusted input | Allocation DOS (e.g., dim=2^30) | Validate dimensions against reasonable max |
| Using uninitialized memory in buffers | Information disclosure | Always zero or explicitly initialize buffers |
| No alignment validation | Segfault → potential exploit | Check alignment or use unaligned intrinsics |
| Exposing raw GPU pointers | Use-after-free, double-free | Wrap in RAII types, never expose raw pointers |
| Integer overflow in size calculations | Buffer overflow | Use `checked_mul`, `checked_add` for sizes |
| GPU code without timeout | DOS via infinite kernel | Set kernel execution timeout |
| Mutable static for scratch buffers | Data race → UB | Use thread_local! or pass buffers explicitly |
| Memory-mapped GPU without validation | Arbitrary memory access | Validate all GPU pointer before dereferencing |
| Panic in unsafe code | May skip destructors → leak resources | Use `Result` even in unsafe functions |

## UX Pitfalls

Common user experience mistakes in performance library domain.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Silent fallback to slow path | Users don't know SIMD unavailable | Log warning when falling back to scalar |
| Panic on feature unavailable | Library unusable | Return `Result::Err` with helpful message |
| No feature detection feedback | Can't debug performance issues | Expose `capabilities()` query function |
| Unclear performance expectations | Disappointment when GPU slower | Document batch size thresholds in API docs |
| Different APIs for CPU/GPU | Constant code switching | Single API with backend selection |
| No way to disable GPU | Can't debug GPU issues | Feature flag + runtime env var |
| Error messages lacking context | Hard to debug dimension mismatches | Include expected vs actual dimensions |
| Cryptic CUDA errors | "CUDA_ERROR_4" meaningless | Wrap with descriptive error variants |
| No progress indication for large batches | Appears hung on 10k batch | Optional callback for progress |
| Benchmark results in inconsistent units | Can't compare performance | Always report ops/sec and latency percentiles |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **SIMD Implementation:** Often missing ARM NEON fallback — verify both x86_64 AVX2 and aarch64 NEON paths exist and are tested
- [ ] **GPU Kernels:** Often missing error handling for device failures — verify CUDA_ERROR_* codes are handled and propagated
- [ ] **Batch API:** Often missing single-item optimization — verify batch-of-1 doesn't go through GPU overhead
- [ ] **Feature Flags:** Often missing CI testing of flag combinations — verify all 2^N flag combinations are tested
- [ ] **Unsafe Code:** Often missing SAFETY documentation — verify every unsafe block has comment explaining preconditions
- [ ] **Power-of-Two Validation:** Often missing in some code paths — verify all entry points check dimensions
- [ ] **Alignment Requirements:** Often documented but not enforced — verify assertions or type system enforces alignment
- [ ] **Scratch Buffer Lifetime:** Often unclear who owns/manages buffer — verify ownership and reuse strategy is explicit
- [ ] **Benchmark Suite:** Often only microbenchmarks — verify end-to-end benchmarks for realistic workloads exist
- [ ] **Error Messages:** Often generic "invalid input" — verify errors include dimension values, expected ranges
- [ ] **GPU Memory Management:** Often manual alloc/free — verify RAII wrappers prevent leaks
- [ ] **Thread Safety:** Often assumed but not tested — verify `Send`/`Sync` bounds are correct and tested
- [ ] **Fallback Path:** Often missing or broken — verify scalar fallback is tested and maintained
- [ ] **Cross-Platform:** Often x86_64-only — verify ARM builds succeed and pass tests
- [ ] **Documentation:** Often describes what, not when/why — verify docs explain batch size thresholds, performance expectations

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Alignment violation crashed production | MEDIUM | 1. Deploy fix with unaligned loads (`_loadu_ps`), 2. Add alignment tests, 3. Gradually migrate to aligned allocations |
| Feature detection broken, illegal instruction | LOW | 1. Add runtime check wrapper, 2. Deploy hotfix, 3. Add CI testing on CPU without target features |
| GPU slower than CPU | HIGH | 1. Add batch size threshold, 2. Profile to find real bottleneck, 3. Consider keeping CPU-only for small batches |
| Unsafe code caused memory corruption | HIGH | 1. Revert unsafe optimization, 2. Run Miri on test suite, 3. Re-add with proper SAFETY docs and testing |
| Power-of-two violated, garbage output | MEDIUM | 1. Add validation at all entry points, 2. Return Result with helpful error, 3. Consider auto-padding if semantically valid |
| Benchmark misleading, real perf bad | HIGH | 1. Add integration benchmarks, 2. Profile production workload, 3. Re-prioritize optimization targets |
| CUDA transfers dominate | MEDIUM | 1. Implement batch API, 2. Keep data on GPU between operations, 3. Add batch size threshold |
| RefCell panic in scratch buffer | LOW | 1. Pass buffers explicitly instead of RefCell, 2. Audit all borrow sites, 3. Consider thread_local! instead |
| False sharing in parallel code | MEDIUM | 1. Use parallel collect instead of shared vec, 2. Pad to cache line, 3. Profile with `perf stat` to confirm |
| Feature flags broke ABI | HIGH | 1. Make features additive only, 2. Use runtime dispatch, 3. Bump major version if unavoidable break |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Alignment violations | Phase 1 (SIMD) | Run on diverse CPUs, use Valgrind/MSAN |
| Feature detection mismatch | Phase 1 (SIMD) | CI test on CPU without AVX2, runtime feature check |
| GPU transfer overhead | Phase 2 (GPU) | Profile batch-of-1 vs batch-of-64, measure transfer time |
| Unsafe invalidates safety | Phase 1 (SIMD) | Run Miri on test suite, document all SAFETY requirements |
| Power-of-two violations | Phase 0 (now) | Add assertion immediately, test non-power-of-two rejection |
| Misleading benchmarks | Phase 0 (now) | Add integration benchmark before optimization work |
| Kernel launch overhead | Phase 2 (GPU) | Measure end-to-end latency, not just kernel time |
| Scratch buffer panics | Phase 1 (allocation) | Choose explicit buffer passing, test with parallelism |
| False sharing | Phase 3 (parallel) | Profile with perf, use parallel iterators |
| Feature flag ABI breaks | Phase 1 (SIMD) | Design additive features, CI test combinations |

## Sources

**Confidence: MEDIUM-HIGH**

This research is based on:
- **HIGH confidence:** Project codebase analysis (ASSESSMENT.md), Rust std::arch documented invariants, cudarc official guidelines
- **MEDIUM confidence:** Well-established SIMD best practices (alignment, feature detection), GPU optimization patterns (transfer overhead, batching)
- **Training data (Jan 2025):** Rust unsafe guidelines, CUDA programming best practices, performance optimization patterns

**Verification status:**
- ✅ Project-specific risks identified from ASSESSMENT.md
- ✅ SIMD pitfalls based on Rust std::arch safety requirements
- ⚠️ CUDA pitfalls based on training data (should verify with cudarc docs when WebFetch available)
- ✅ Unsafe code patterns from Rust nomicon and best practices

**Sources that should be consulted but were unavailable:**
- cudarc official documentation (WebFetch error)
- Recent Rust SIMD blog posts/RFCs (Brave Search unavailable)
- GPU optimization guides for ML inference (web search unavailable)

**Mitigation:** All critical pitfalls are well-established in performance computing domain. Specific cudarc details may need validation during Phase 2 implementation.

---
*Pitfalls research for: Rust SIMD/GPU Performance Optimization*
*Researched: 2026-03-27*
*Next: Use this to inform roadmap phase ordering and success criteria*
