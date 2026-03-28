---
phase: 02-simd-acceleration
plan: 02
subsystem: backend/simd
tags: [simd, avx2, neon, performance, hot-path]
dependency_graph:
  requires: [02-01]
  provides: [simd-fwht, simd-dot-product]
  affects: [PolarQuant, Rotation, KvCache]
tech_stack:
  added: [std::arch::x86_64, std::arch::aarch64]
  patterns: [target_feature, runtime-detection, unsafe-intrinsics]
key_files:
  created: []
  modified:
    - src/backend/simd.rs
decisions:
  - AVX2 vectorizes at stride >= 8, scalar fallback for strides 1/2/4
  - NEON vectorizes at stride >= 4, scalar fallback for strides 1/2
  - Horizontal reduction via store-to-array for AVX2, vaddvq_f32 for NEON
  - Unaligned loads/stores (_mm256_loadu_ps, vld1q_f32) for safety
  - All unsafe blocks require 3-part SAFETY comments (target feature, bounds, alignment)
metrics:
  duration_minutes: 3
  completed_date: 2026-03-27
  tasks_completed: 2
  files_modified: 1
  commits: 2
  safety_comments: 10
---

# Phase 02 Plan 02: SIMD Intrinsics Implementation Summary

**One-liner:** AVX2 and NEON SIMD intrinsics for FWHT butterfly operations and dot product with comprehensive SAFETY documentation and equivalence tests

## Objective Achieved

Implemented actual SIMD intrinsics for FWHT butterfly operations and dot product, replacing scalar placeholders from Plan 01. Both AVX2 (x86_64) and NEON (aarch64) implementations complete with runtime dispatch and scalar fallbacks.

## Tasks Completed

### Task 1: Implement AVX2 and NEON FWHT with SAFETY documentation
**Status:** Complete
**Commit:** 6ed00cf
**Files:** src/backend/simd.rs

Added SIMD FWHT implementations:
- `fwht_inplace_avx2`: 8-wide vectorization using `_mm256_add_ps`/`_mm256_sub_ps`
- `fwht_inplace_neon`: 4-wide vectorization using `vaddq_f32`/`vsubq_f32`
- Updated `SimdBackend::fwht_normalized_inplace` with runtime CPU feature detection
- Scalar fallback for small strides where vectorization would mix incorrect butterfly pairs
- All unsafe blocks documented with 3-part SAFETY comments (target feature, bounds, alignment)

**Tests:** All existing tests pass with and without SIMD feature. SIMD output matches scalar output within f32 precision (< 1e-6 error).

### Task 2: Implement AVX2 and NEON dot product with SAFETY documentation
**Status:** Complete
**Commit:** b6c0543
**Files:** src/backend/simd.rs

Added SIMD dot product implementations:
- `dot_product_avx2`: 8-wide multiply-accumulate with horizontal reduction via store-to-array
- `dot_product_neon`: 4-wide fused multiply-add (`vmlaq_f32`) with `vaddvq_f32` reduction
- Updated `SimdBackend::dot_product` with runtime dispatch
- Scalar tail handling for non-multiple-of-vector-width lengths
- Removed unused import to eliminate clippy warning

**Tests:** Added comprehensive test suite:
- `simd_matches_scalar_dot_product_large`: dim=128 equivalence (< 1e-4 error)
- `simd_matches_scalar_fwht_dim128`: dim=128 FWHT equivalence (< 1e-4 error)
- `simd_dot_product_empty_and_small`: edge cases (empty, 1 element, 7 elements)
- `simd_inner_product_accuracy`: <2% relative error verified (PERF-01 requirement)

## Deviations from Plan

None - plan executed exactly as written.

## Key Implementation Details

### FWHT Butterfly Vectorization Strategy
- **AVX2 (x86_64):** Processes 8 butterflies per iteration when stride >= 8
  - Small strides (1, 2, 4) use scalar fallback to avoid complex shuffles
  - Unaligned loads/stores for safety (`_mm256_loadu_ps` / `_mm256_storeu_ps`)
- **NEON (aarch64):** Processes 4 butterflies per iteration when stride >= 4
  - Small strides (1, 2) use scalar fallback
  - Intrinsics don't require alignment (`vld1q_f32` / `vst1q_f32`)

### Dot Product Reduction Strategy
- **AVX2:** Accumulate 8-wide multiply-add, then horizontal reduction via store-to-array and scalar sum
- **NEON:** Accumulate 4-wide fused multiply-add (`vmlaq_f32`), reduce using `vaddvq_f32` (horizontal sum intrinsic)
- **Both:** Scalar tail loop handles remaining elements for non-multiple-of-vector-width lengths

### SAFETY Documentation Pattern
Every unsafe block documents exactly 3 items:
1. **Target feature:** How CPU support is confirmed (is_x86_feature_detected!, is_aarch64_feature_detected!)
2. **Bounds:** How memory accesses are proven in-bounds (loop conditions, debug_asserts)
3. **Alignment:** Confirmation of unaligned intrinsic usage or alignment guarantee

Total SAFETY comments: 10 (exceeds 8 minimum)

## Verification Results

All success criteria met:

- [x] SimdBackend::fwht_normalized_inplace uses AVX2/NEON intrinsics (SIMD-01, SIMD-02, SIMD-04)
- [x] SimdBackend::dot_product uses SIMD multiply-accumulate
- [x] Small strides fall back to scalar (step < 8 for AVX2, step < 4 for NEON)
- [x] All unsafe code has 3-part SAFETY comments (10 total) (SIMD-07)
- [x] SIMD output matches scalar output within f32 precision (< 1e-6 for most tests, < 1e-4 for dim=128) (TEST-02)
- [x] Inner product accuracy maintained at <2% error (PERF-01)
- [x] All existing tests still pass (TEST-01)
- [x] No TODOs remain in src/backend/simd.rs
- [x] No clippy warnings for simd.rs
- [x] Tests pass with and without SIMD feature

### Test Results
- `cargo test --lib --features simd`: 54 tests passed
- `cargo test --features simd`: Full suite passed (54 unit + 1 doctest)
- `cargo test --lib`: 42 tests passed (no regressions without SIMD)
- `cargo test`: Full suite passed without SIMD
- `cargo clippy --features simd`: No warnings in simd.rs

## Performance Implications

This plan provides the foundation for 2-4x speedup on FWHT and dot product operations (hot paths in attention computation). Actual benchmarking deferred to Plan 03 (SIMD benchmarks and analysis).

**Expected impact:**
- FWHT: 8-wide (AVX2) or 4-wide (NEON) parallelism vs scalar
- Dot product: Vectorized multiply-accumulate with efficient horizontal reduction
- Zero overhead when SIMD unavailable (runtime detection, scalar fallback)

## Requirements Satisfied

- SIMD-01: AVX2 implementation for x86_64
- SIMD-02: NEON implementation for aarch64
- SIMD-04: Runtime feature detection dispatch
- SIMD-07: Comprehensive SAFETY documentation (10 comments)
- TEST-01: No test regressions (42 tests pass without SIMD)
- TEST-02: SIMD equivalence with scalar (< 1e-4 error at dim=128)
- PERF-01: Inner product accuracy <2% maintained

## Next Steps

- Plan 02-03: SIMD benchmarks to quantify 2-4x speedup vs scalar baseline
- Plan 02-04: End-to-end integration testing with PolarQuant and KvCache

## Self-Check

### Created Files Verification
No new files created (only modifications).

### Modified Files Verification
```
FOUND: src/backend/simd.rs
```

### Commits Verification
```
FOUND: 6ed00cf (Task 1: AVX2 and NEON FWHT)
FOUND: b6c0543 (Task 2: AVX2 and NEON dot product)
```

## Self-Check: PASSED
