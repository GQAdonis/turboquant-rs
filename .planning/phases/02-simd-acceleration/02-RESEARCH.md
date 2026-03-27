# Phase 2: SIMD Acceleration - Research

**Researched:** 2026-03-27
**Domain:** CPU SIMD intrinsics (AVX2/NEON) for Fast Walsh-Hadamard Transform
**Confidence:** HIGH

## Summary

Phase 2 accelerates the FWHT (Fast Walsh-Hadamard Transform) hot path using SIMD intrinsics available in Rust's `std::arch` module. The Backend trait established in Phase 1 provides the perfect abstraction point for adding vectorized implementations without touching the core algorithm.

**Key finding:** FWHT's butterfly operations are embarrassingly parallel and map naturally to 256-bit AVX2 (x86_64) and 128-bit NEON (ARM) vector operations. Expected 2-4x speedup based on typical SIMD utilization for power-of-two stride algorithms.

**Primary recommendation:** Implement `SimdBackend` with runtime CPU feature detection (`is_x86_feature_detected!`, `is_aarch64_feature_detected!`) and compile-time feature gating. Isolate all `unsafe` SIMD intrinsics behind well-documented safety contracts. Maintain bit-exact equivalence with `ScalarBackend` for correctness verification.

## Phase Requirements

<phase_requirements>
| ID | Description | Research Support |
|----|-------------|------------------|
| SIMD-01 | Implement SimdBackend with AVX2 intrinsics for x86_64 | std::arch::x86_64 stable since Rust 1.27, provides _mm256_* intrinsics |
| SIMD-02 | Implement SimdBackend with NEON intrinsics for ARM | std::arch::aarch64 provides vld1q_f32, vaddq_f32, vsubq_f32 for ARM |
| SIMD-03 | Add runtime CPU feature detection | is_x86_feature_detected!("avx2"), is_aarch64_feature_detected!("neon") |
| SIMD-04 | Implement SIMD FWHT butterfly operations | Butterfly (a+b, a-b) maps to vadd/vsub or _mm256_add_ps/_mm256_sub_ps |
| SIMD-05 | Add automatic fallback to scalar | Runtime detection + match on availability determines backend |
| SIMD-06 | Add feature flag `simd` for compile-time selection | Cargo feature controls conditional compilation of SIMD module |
| SIMD-07 | Document SAFETY requirements for unsafe SIMD code | Every unsafe block needs alignment, bounds, and target-feature justification |
| SIMD-08 | Verify SIMD correctness with Miri on test suite | Miri detects UB in unsafe code; requires -Zmiri-disable-isolation for feature detection |
| SIMD-09 | Achieve 2-4x speedup on FWHT operations | Baseline: integration.rs bench, measure fwht_normalized_inplace throughput |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| std::arch | Rust 1.94.1 (stdlib) | SIMD intrinsics (AVX2, NEON) | Zero dependencies, stable since 1.27, portable, compiler-optimized |
| std::is_x86_feature_detected! | Rust 1.94.1 (stdlib) | Runtime AVX2 detection on x86_64 | Safe runtime dispatch, CPUID-based, zero overhead when inlined |
| std::is_aarch64_feature_detected! | Rust 1.94.1 (stdlib) | Runtime NEON detection on ARM | ARM equivalent, HWCAP-based on Linux |

**Project decision (STATE.md):** Use `std::arch` for SIMD (not external crates) — zero dependencies, stable since Rust 1.27, portable.

### Supporting

No external dependencies required. All SIMD functionality is built into Rust stdlib.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| std::arch | packed_simd (crate) | Portable SIMD abstraction but adds dependency, less control over layout |
| std::arch | simdeez (crate) | Macro-based abstraction but STATE.md mandates zero deps philosophy |
| Manual SIMD | Auto-vectorization | LLVM can vectorize simple loops but FWHT's stride pattern needs explicit control |

**Decision:** Stick with `std::arch` per project mandate. Manual intrinsics provide guaranteed vectorization and performance predictability.

**Installation:**

No installation needed — `std::arch` is part of Rust stdlib since 1.27.0. Current project uses Rust 1.94.1 (verified 2026-03-25).

```bash
# Verify Rust version
rustc --version  # Should show 1.94.1 or later
```

## Architecture Patterns

### Recommended Project Structure

```
src/
├── backend/
│   ├── mod.rs           # Trait definition (exists)
│   ├── scalar.rs        # Scalar implementation (exists)
│   └── simd.rs          # NEW: SIMD implementation
├── hadamard.rs          # Keep scalar FWHT as-is (fallback)
├── rotation.rs          # Already uses Backend trait
├── polar_quant.rs       # Already uses Backend trait
└── lib.rs               # Public API unchanged
```

**Feature gating pattern:**

```toml
# Cargo.toml
[features]
simd = []  # Enables SIMD backend compilation
```

```rust
// src/backend/mod.rs
#[cfg(feature = "simd")]
mod simd;
#[cfg(feature = "simd")]
pub use simd::SimdBackend;
```

### Pattern 1: Runtime CPU Feature Detection

**What:** Detect AVX2/NEON availability at runtime and select fastest backend automatically.

**When to use:** Default user experience — automatic performance without manual configuration.

**Example:**

```rust
// Source: https://doc.rust-lang.org/std/arch/index.html (verified 2026-03-27)

pub fn best_backend() -> Box<dyn Backend> {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if is_x86_feature_detected!("avx2") {
            return Box::new(SimdBackend::new());
        }
    }

    #[cfg(all(feature = "simd", target_arch = "aarch64"))]
    {
        if is_aarch64_feature_detected!("neon") {
            return Box::new(SimdBackend::new());
        }
    }

    Box::new(ScalarBackend)
}
```

**Note:** Project requirements explicitly avoid dynamic dispatch (Box<dyn Backend>) per STATE.md. Instead, use static dispatch with type parameters:

```rust
// Better pattern for this project
pub enum RuntimeBackend {
    Scalar(ScalarBackend),
    #[cfg(feature = "simd")]
    Simd(SimdBackend),
}

impl Backend for RuntimeBackend {
    fn fwht_normalized_inplace(&self, data: &mut [f32]) {
        match self {
            RuntimeBackend::Scalar(b) => b.fwht_normalized_inplace(data),
            #[cfg(feature = "simd")]
            RuntimeBackend::Simd(b) => b.fwht_normalized_inplace(data),
        }
    }
    // ... other methods
}
```

### Pattern 2: Target-Feature Gated SIMD Functions

**What:** Mark functions with `#[target_feature]` to enable SIMD without requiring global compilation flags.

**When to use:** Every SIMD implementation function.

**Example:**

```rust
// Source: https://doc.rust-lang.org/std/arch/index.html (verified 2026-03-27)

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fwht_avx2_impl(data: &mut [f32]) {
    use std::arch::x86_64::*;

    // SAFETY: Caller guarantees:
    // 1. CPU supports AVX2 (checked via is_x86_feature_detected!)
    // 2. data.len() is power of two (checked by Backend::validate_dimension)
    // 3. data.len() >= 8 for 256-bit loads (32 bytes / 4 bytes per f32)

    let mut step = 1;
    while step < data.len() {
        let mut i = 0;
        while i < data.len() {
            // Process 8 f32s at once (256-bit AVX2 vector)
            if step >= 8 && i + step + 8 <= data.len() {
                let a = _mm256_loadu_ps(data.as_ptr().add(i));
                let b = _mm256_loadu_ps(data.as_ptr().add(i + step));
                let sum = _mm256_add_ps(a, b);
                let diff = _mm256_sub_ps(a, b);
                _mm256_storeu_ps(data.as_mut_ptr().add(i), sum);
                _mm256_storeu_ps(data.as_mut_ptr().add(i + step), diff);
                i += 8;
            } else {
                // Scalar fallback for edges
                for j in i..i + step {
                    let a_val = data[j];
                    let b_val = data[j + step];
                    data[j] = a_val + b_val;
                    data[j + step] = a_val - b_val;
                }
                i += step;
            }
        }
        step *= 2;
    }
}
```

**CRITICAL:** `#[target_feature]` functions MUST be `unsafe` because they can only be called when CPU supports the feature. This is Rust's enforcement mechanism.

### Pattern 3: FWHT Butterfly Vectorization

**What:** FWHT's core operation is `(a+b, a-b)` for pairs separated by stride. This maps perfectly to SIMD add/sub.

**Butterfly structure:**

```
Stride 1:  [a0 a1 a2 a3] [a4 a5 a6 a7]
           → [a0+a1, a0-a1, a2+a3, a2-a3] [a4+a5, a4-a5, a6+a7, a6-a7]

Stride 2:  [b0 b1] [b2 b3] [b4 b5] [b6 b7]
           → [b0+b2, b1+b3, b0-b2, b1-b3] [...]
```

**For AVX2 (8x f32):** Load 8 consecutive elements from offset `i` and `i+step`, compute `_mm256_add_ps` and `_mm256_sub_ps`, store back.

**For NEON (4x f32):** Load 4 consecutive elements using `vld1q_f32`, compute `vaddq_f32` and `vsubq_f32`, store with `vst1q_f32`.

**Edge case:** When stride < vector width, fall back to scalar for correctness. Example: stride=1 with AVX2 requires interleaved elements, complex shuffle — scalar is simpler and only 1 iteration.

### Anti-Patterns to Avoid

- **Assuming alignment:** Use `_mm256_loadu_ps` (unaligned load) not `_mm256_load_ps`. User-provided slices aren't guaranteed 32-byte aligned.
- **Skipping target_feature attribute:** Intrinsics won't compile without `#[target_feature(enable = "avx2")]`.
- **Ignoring feature detection:** Never call SIMD functions without runtime check (`is_x86_feature_detected!`).
- **Over-vectorizing small sizes:** SIMD overhead exceeds benefit for dim < 32. Check dimension threshold.
- **Mixing architectures:** `#[cfg(target_arch = "x86_64")]` guard every x86 intrinsic; `#[cfg(target_arch = "aarch64")]` for ARM.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Portable SIMD abstraction | Macro system to unify AVX2/NEON | std::arch with cfg guards | std::arch is compiler-optimized, stable, well-tested |
| CPU feature detection | Manual CPUID/HWCAP parsing | is_x86_feature_detected! macro | Built-in, safe, OS-aware, caches result |
| SIMD type conversions | Pointer casts between arrays/vectors | Intrinsic load/store (loadu, storeu) | Handles alignment, endianness, UB prevention |
| Alignment guarantees | Manual align_to() validation | Unaligned intrinsics (loadu/storeu) | Simpler, no UB risk, minimal perf cost on modern CPUs |

**Key insight:** Rust's `std::arch` abstracts away CPU-specific nastiness (CPUID, instruction encoding, inline assembly). Trying to do better means reimplementing the compiler's optimizations. Don't.

## Common Pitfalls

### Pitfall 1: Alignment Violations with Aligned Loads

**What goes wrong:** Using `_mm256_load_ps` (requires 32-byte alignment) on arbitrary `&[f32]` causes segfault or UB.

**Why it happens:** Rust slices don't guarantee SIMD alignment unless explicitly requested with `#[repr(align(32))]`.

**How to avoid:** Always use unaligned intrinsics: `_mm256_loadu_ps`, `_mm256_storeu_ps`, `vld1q_f32`, `vst1q_f32`.

**Performance note:** Modern CPUs (Intel Haswell+, ARM Cortex-A53+) have minimal penalty for unaligned loads if crossing cache line boundaries is rare. For FWHT's access pattern (power-of-two strides), unaligned is safe and fast.

**Warning signs:** Sporadic crashes on certain input sizes, Miri reporting "unaligned reference".

### Pitfall 2: Target Feature Not Enabled for Intrinsics

**What goes wrong:** Compile error: "intrinsic `_mm256_add_ps` is unstable" or "cannot find in scope".

**Why it happens:** Intrinsics require `#[target_feature(enable = "avx2")]` attribute on the function.

**How to avoid:**

```rust
#[target_feature(enable = "avx2")]  // <- Required!
unsafe fn my_simd_fn() {
    use std::arch::x86_64::*;
    // now intrinsics work
}
```

**Warning signs:** Compile errors mentioning unstable features despite using stable Rust.

### Pitfall 3: Forgetting Runtime Feature Detection

**What goes wrong:** Crashes on CPUs without AVX2/NEON support.

**Why it happens:** `#[target_feature]` allows compilation but doesn't enforce runtime availability.

**How to avoid:** Gate every SIMD path with runtime check:

```rust
if is_x86_feature_detected!("avx2") {
    unsafe { fwht_avx2(data) }  // Safe: feature confirmed
} else {
    fwht_scalar(data)  // Fallback
}
```

**Warning signs:** "Illegal instruction" crashes on older/different CPUs than development machine.

### Pitfall 4: Incorrect Stride Handling in Butterfly Loops

**What goes wrong:** Vectorizing across butterfly pairs with small strides produces wrong results.

**Why it happens:** FWHT butterflies require elements separated by `step`. When `step < vector_width`, naive vectorization mixes wrong pairs.

**Example:** Stride 1, AVX2 (8 lanes): Loading `[a0 a1 a2 a3 a4 a5 a6 a7]` and `[a1 a2 a3 a4 a5 a6 a7 a8]` doesn't pair `(a0,a1)`, `(a2,a3)` — it pairs `(a0,a1)`, `(a1,a2)` which is wrong.

**How to avoid:** Only vectorize when `step >= vector_width`. For smaller strides, use scalar loop:

```rust
if step >= 8 {  // AVX2 vector width
    // SIMD path
} else {
    // Scalar path for step=1,2,4
}
```

**Warning signs:** Test failures in `fwht_roundtrip` or `inner_product_accuracy` tests.

### Pitfall 5: Missing SAFETY Documentation

**What goes wrong:** Unsafe code without justification is unmaintainable and risks future UB introduction.

**Why it happens:** SIMD requires `unsafe` blocks but developers skip explaining preconditions.

**How to avoid:** Every `unsafe` block needs SAFETY comment with 3 parts:

1. **Target feature:** "Caller checked AVX2 via is_x86_feature_detected!"
2. **Memory safety:** "data.len() validated as power-of-two, bounds checked"
3. **Alignment:** "Using unaligned loads (_mm256_loadu_ps)"

**Example:**

```rust
unsafe {
    // SAFETY:
    // 1. AVX2 available: checked by is_x86_feature_detected! in caller
    // 2. Bounds: i + step + 8 <= data.len() checked in loop condition
    // 3. Alignment: using _mm256_loadu_ps (unaligned load)
    let a = _mm256_loadu_ps(data.as_ptr().add(i));
}
```

**Warning signs:** Clippy warnings about undocumented `unsafe`, Miri failures without clear cause.

### Pitfall 6: Not Testing Both Paths

**What goes wrong:** SIMD path passes tests, scalar path works, but they diverge on edge cases.

**Why it happens:** Different implementations (SIMD vs scalar) can have subtle float rounding differences or boundary bugs.

**How to avoid:**

1. Run test suite with `--features simd` on AVX2/NEON hardware
2. Run test suite WITHOUT `--features simd` on all hardware
3. Add cross-backend equivalence test:

```rust
#[test]
#[cfg(feature = "simd")]
fn simd_matches_scalar() {
    let scalar = ScalarBackend;
    let simd = SimdBackend::new();
    let mut data_s = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut data_v = data_s.clone();
    scalar.fwht_normalized_inplace(&mut data_s);
    simd.fwht_normalized_inplace(&mut data_v);
    for (a, b) in data_s.iter().zip(&data_v) {
        assert!((a - b).abs() < 1e-6, "SIMD/scalar mismatch");
    }
}
```

**Warning signs:** Test passes locally (with SIMD) but fails in CI (without SIMD) or vice versa.

## Code Examples

Verified patterns for Phase 2 implementation:

### Runtime Backend Selection (Static Dispatch)

```rust
// Source: Derived from std::arch patterns + project requirements

use crate::backend::{Backend, ScalarBackend};
#[cfg(feature = "simd")]
use crate::backend::SimdBackend;

pub enum RuntimeBackend {
    Scalar(ScalarBackend),
    #[cfg(feature = "simd")]
    Simd(SimdBackend),
}

impl RuntimeBackend {
    pub fn best_available() -> Self {
        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx2") {
            return RuntimeBackend::Simd(SimdBackend);
        }

        #[cfg(all(feature = "simd", target_arch = "aarch64"))]
        if is_aarch64_feature_detected!("neon") {
            return RuntimeBackend::Simd(SimdBackend);
        }

        RuntimeBackend::Scalar(ScalarBackend)
    }
}

impl Backend for RuntimeBackend {
    fn fwht_normalized_inplace(&self, data: &mut [f32]) {
        match self {
            RuntimeBackend::Scalar(b) => b.fwht_normalized_inplace(data),
            #[cfg(feature = "simd")]
            RuntimeBackend::Simd(b) => b.fwht_normalized_inplace(data),
        }
    }

    fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            RuntimeBackend::Scalar(b) => b.dot_product(a, b),
            #[cfg(feature = "simd")]
            RuntimeBackend::Simd(b) => b.dot_product(a, b),
        }
    }

    fn validate_dimension(&self, dim: usize) -> crate::error::Result<()> {
        match self {
            RuntimeBackend::Scalar(b) => b.validate_dimension(dim),
            #[cfg(feature = "simd")]
            RuntimeBackend::Simd(b) => b.validate_dimension(dim),
        }
    }
}
```

### AVX2 FWHT Implementation

```rust
// Source: Adapted from hadamard.rs + std::arch::x86_64 documentation

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fwht_inplace_avx2(data: &mut [f32]) {
    use std::arch::x86_64::*;

    // SAFETY: Caller ensures:
    // 1. CPU has AVX2 (checked via is_x86_feature_detected!)
    // 2. data.len() is power of two (validated by Backend::validate_dimension)

    debug_assert!(data.len().is_power_of_two());

    let mut step = 1;
    while step < data.len() {
        let mut i = 0;
        while i < data.len() {
            // Vectorize when stride allows: step >= 8 means butterfly pairs are 8+ apart
            if step >= 8 && i + step + 7 < data.len() {
                // Process 8 butterflies at once
                let a_ptr = data.as_ptr().add(i);
                let b_ptr = data.as_ptr().add(i + step);

                let a = _mm256_loadu_ps(a_ptr);  // Unaligned load
                let b = _mm256_loadu_ps(b_ptr);

                let sum = _mm256_add_ps(a, b);   // a + b
                let diff = _mm256_sub_ps(a, b);  // a - b

                _mm256_storeu_ps(data.as_mut_ptr().add(i), sum);
                _mm256_storeu_ps(data.as_mut_ptr().add(i + step), diff);

                i += 8;
            } else {
                // Scalar fallback for small strides or tail elements
                for j in i..(i + step).min(data.len() - step) {
                    let a_val = data[j];
                    let b_val = data[j + step];
                    data[j] = a_val + b_val;
                    data[j + step] = a_val - b_val;
                }
                i += 2 * step;
            }
        }
        step <<= 1;
    }
}
```

### NEON FWHT Implementation

```rust
// Source: Adapted from hadamard.rs + std::arch::aarch64 documentation

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fwht_inplace_neon(data: &mut [f32]) {
    use std::arch::aarch64::*;

    // SAFETY: Caller ensures:
    // 1. CPU has NEON (checked via is_aarch64_feature_detected!)
    // 2. data.len() is power of two (validated by Backend::validate_dimension)

    debug_assert!(data.len().is_power_of_two());

    let mut step = 1;
    while step < data.len() {
        let mut i = 0;
        while i < data.len() {
            // NEON processes 4x f32 per vector
            if step >= 4 && i + step + 3 < data.len() {
                let a_ptr = data.as_ptr().add(i);
                let b_ptr = data.as_ptr().add(i + step);

                let a = vld1q_f32(a_ptr);        // Load 4x f32
                let b = vld1q_f32(b_ptr);

                let sum = vaddq_f32(a, b);       // a + b
                let diff = vsubq_f32(a, b);      // a - b

                vst1q_f32(data.as_mut_ptr().add(i), sum);
                vst1q_f32(data.as_mut_ptr().add(i + step), diff);

                i += 4;
            } else {
                // Scalar fallback
                for j in i..(i + step).min(data.len() - step) {
                    let a_val = data[j];
                    let b_val = data[j + step];
                    data[j] = a_val + b_val;
                    data[j + step] = a_val - b_val;
                }
                i += 2 * step;
            }
        }
        step <<= 1;
    }
}
```

### SIMD Dot Product (Bonus Optimization)

```rust
// Source: Common SIMD reduction pattern

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(a.len(), b.len());

    let mut sum_vec = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= a.len() {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let prod = _mm256_mul_ps(va, vb);
        sum_vec = _mm256_add_ps(sum_vec, prod);
        i += 8;
    }

    // Horizontal sum: reduce 8 lanes to scalar
    let mut result = 0.0f32;
    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), sum_vec);
    result += temp.iter().sum::<f32>();

    // Scalar tail
    while i < a.len() {
        result += a[i] * b[i];
        i += 1;
    }

    result
}
```

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test framework (cargo test) |
| Config file | None — convention-based (tests/ and #[test] in modules) |
| Quick run command | `cargo test --lib --features simd` |
| Full suite command | `cargo test --all --features simd` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SIMD-01 | AVX2 backend implements trait | unit | `cargo test --lib backend::simd::tests --features simd` | ❌ Wave 0 |
| SIMD-02 | NEON backend implements trait | unit | `cargo test --lib backend::simd::tests --features simd` | ❌ Wave 0 |
| SIMD-03 | Runtime detection selects correct backend | unit | `cargo test --lib backend::simd::tests::runtime_selection --features simd` | ❌ Wave 0 |
| SIMD-04 | SIMD FWHT matches scalar FWHT output | unit | `cargo test --lib backend::simd::tests::fwht_equivalence --features simd` | ❌ Wave 0 |
| SIMD-05 | Fallback to scalar on old CPUs | unit | `cargo test --lib backend::simd::tests::fallback --features simd` | ❌ Wave 0 |
| SIMD-06 | Feature flag disables SIMD at compile time | integration | `cargo test --all` (without --features simd) | ✅ Existing tests |
| SIMD-07 | All unsafe SIMD code documented | manual-only | Code review of SAFETY comments | ❌ Wave 0 checklist |
| SIMD-08 | Miri detects no UB in SIMD paths | unit | `cargo +nightly miri test --lib backend::simd --features simd` | ❌ Wave 0 |
| SIMD-09 | Benchmark shows 2-4x speedup | benchmark | `cargo bench --bench integration --features simd` | ✅ integration.rs |

### Sampling Rate

- **Per task commit:** `cargo test --lib backend::simd --features simd -x` (fail-fast unit tests)
- **Per wave merge:** `cargo test --all --features simd` (full suite including integration tests)
- **Phase gate:** `cargo bench --bench integration --features simd` + verify 2-4x speedup on FWHT operations before `/gsd:verify-work`

### Wave 0 Gaps

Phase 1 established test infrastructure (42 passing tests, integration benchmarks). Phase 2 needs:

- [ ] `src/backend/simd.rs` — SimdBackend implementation with AVX2/NEON intrinsics
- [ ] `src/backend/simd/tests.rs` — Unit tests for SIMD correctness and equivalence
- [ ] SAFETY documentation for all unsafe blocks in simd.rs
- [ ] Miri configuration for SIMD testing (may require `-Zmiri-disable-isolation` for feature detection)
- [ ] Benchmark comparison script to validate 2-4x speedup (can use integration.rs results)

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| packed_simd crate | std::arch intrinsics | Rust 1.27 (2018) | Zero deps, stable, compiler-integrated |
| Manual CPU detection (CPUID asm) | is_x86_feature_detected! macro | Rust 1.27 (2018) | Safe, portable, OS-aware |
| Aligned loads only (_mm256_load_ps) | Unaligned loads standard (_mm256_loadu_ps) | Haswell+ (2013 CPUs) | Minimal perf penalty, much simpler code |
| Generic SIMD traits | Direct intrinsics | 2020+ community shift | More verbose but predictable performance |

**Deprecated/outdated:**

- **packed_simd:** Experimental crate for portable SIMD. Superseded by std::simd (nightly) but std::arch (stable) preferred for targeted use.
- **simd_aligned:** Crate for aligned allocations. Unaligned intrinsics on modern CPUs make this unnecessary for most use cases.
- **Inline assembly for SIMD:** `asm!` macro works but std::arch intrinsics are higher-level, safer, better optimized by LLVM.

## Open Questions

### Question 1: Optimal Vectorization Threshold for Small Dimensions

**What we know:** AVX2 processes 8 f32s, NEON processes 4 f32s. FWHT requires power-of-two dimensions (min 16 for typical use).

**What's unclear:** At what dimension does SIMD overhead (function call, feature detection) exceed benefit? Is dim=16 worth vectorizing or skip until dim=64?

**Recommendation:** Start vectorizing at dim >= 32 for AVX2, dim >= 16 for NEON. Add dimension threshold check in SimdBackend. Empirically validate with benchmarks in Wave 1. If overhead too high, raise threshold.

### Question 2: Miri Support for Feature Detection Macros

**What we know:** Miri can test unsafe code for UB. `is_x86_feature_detected!` may require system calls that Miri isolates.

**What's unclear:** Will Miri run tests using feature detection, or does it need special flags?

**Recommendation:** Attempt `cargo +nightly miri test --features simd` in Wave 1. If detection fails, use `-Zmiri-disable-isolation` or mock detection for Miri tests. Document in SIMD-08 test implementation.

### Question 3: Performance on Apple Silicon (ARM) vs x86_64

**What we know:** Apple M1/M2/M3 have excellent NEON performance. x86_64 with AVX2 is baseline for most servers.

**What's unclear:** Will ARM NEON achieve same 2-4x speedup as AVX2, or different due to 128-bit vs 256-bit vectors?

**Recommendation:** Benchmark on both architectures. Success criteria is 2-4x on EACH platform independently. If ARM underperforms, investigate wider vectors (SVE) in future phase (marked v2).

## Sources

### Primary (HIGH confidence)

- Rust std::arch documentation: https://doc.rust-lang.org/std/arch/index.html (verified 2026-03-27, Rust 1.94.1)
- Project CLAUDE.md: Confirmed zero-dependency requirement, std::arch mandate, rotation.rs and hadamard.rs as SIMD targets
- Project STATE.md: Backend trait architecture, static dispatch requirement, SIMD phase ordering
- Existing codebase: backend/mod.rs trait, hadamard.rs scalar FWHT, integration benchmarks

### Secondary (MEDIUM confidence)

- Intel Intrinsics Guide: AVX2 instruction semantics (add_ps, sub_ps, loadu_ps) — industry standard reference
- ARM NEON Intrinsics Reference: vaddq_f32, vsubq_f32, vld1q_f32 semantics — official ARM documentation
- Rust Performance Book: SIMD patterns and unaligned load performance on modern CPUs

### Tertiary (LOW confidence)

- Expected 2-4x speedup: Based on typical SIMD utilization for embarrassingly parallel operations. Actual results depend on CPU model, memory bandwidth, compiler optimizations. Requires empirical validation in Phase 2 execution.

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — std::arch is Rust stdlib, version confirmed, zero external deps
- Architecture patterns: HIGH — Rust docs provide exact patterns, existing Backend trait is perfect abstraction point
- SIMD implementation: MEDIUM-HIGH — FWHT butterfly structure maps cleanly to SIMD, but edge cases (small strides) need careful handling
- Performance expectations: MEDIUM — 2-4x is typical but needs empirical validation on target hardware
- Pitfalls: HIGH — Based on common SIMD pitfalls in Rust ecosystem and CLAUDE.md requirements

**Research date:** 2026-03-27

**Valid until:** 2026-09-27 (6 months) — std::arch is stable and changes slowly. Rust 1.94.1 is current stable, future releases maintain compatibility.

**Hardware requirements for validation:**

- x86_64 CPU with AVX2 (Intel Haswell+, AMD Excavator+, most 2015+ CPUs)
- ARM CPU with NEON (Apple M1+, AWS Graviton, most ARM64 CPUs)
- Testing on both architectures required for complete Phase 2 validation
