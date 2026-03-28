---
phase: 02-simd-acceleration
plan: 01
subsystem: backend
tags: [infrastructure, simd, feature-flag, runtime-detection]
dependencies:
  requires: [01-foundation-quick-wins]
  provides: [simd-infrastructure, runtime-backend, cpu-detection]
  affects: [backend-module, public-api]
tech_stack:
  added: [simd-feature-flag, RuntimeBackend-enum, SimdBackend-struct]
  patterns: [static-dispatch, conditional-compilation, runtime-detection]
key_files:
  created:
    - src/backend/simd.rs
  modified:
    - Cargo.toml
    - src/backend/mod.rs
    - src/lib.rs
decisions:
  - id: SIMD-01
    choice: Static dispatch via enum (not Box<dyn Backend>)
    rationale: Zero-cost abstraction, matches project requirement
  - id: SIMD-02
    choice: Scalar delegation in SimdBackend initially
    rationale: Separates infrastructure (Plan 01) from intrinsics (Plan 02)
  - id: SIMD-03
    choice: Runtime detection via is_x86_feature_detected!/is_aarch64_feature_detected!
    rationale: Standard library approach, no dependencies, production-ready
metrics:
  duration_seconds: 122
  tasks_completed: 2
  files_created: 1
  files_modified: 3
  commits: 2
  tests_added: 8
  tests_passed: 51
  completed_at: "2026-03-27T20:44:17Z"
---

# Phase 02 Plan 01: SIMD Infrastructure Summary

**One-liner:** SIMD feature flag with SimdBackend (scalar delegation) and RuntimeBackend with CPU detection (AVX2/NEON)

## Overview

Established the foundational SIMD infrastructure needed for Phase 2 acceleration work. This plan creates the scaffolding (feature flag, backend structs, runtime detection) while Plan 02 will fill in the actual SIMD intrinsics.

## Implementation Details

### Feature Flag (Cargo.toml)
- Added `[features]` section with `simd = []` entry
- Enables compile-time control of SIMD code inclusion
- Zero overhead when feature disabled

### SimdBackend (src/backend/simd.rs)
- Implements `Backend` trait with three methods: `fwht_normalized_inplace`, `dot_product`, `validate_dimension`
- Currently delegates to scalar implementations (TODO markers for Plan 02)
- Identical API to `ScalarBackend` ensuring drop-in compatibility

### RuntimeBackend (src/backend/simd.rs)
- Enum with `Scalar(ScalarBackend)` and `Simd(SimdBackend)` variants
- Static dispatch via pattern matching (no Box<dyn> overhead)
- `best_available()` function with CPU feature detection:
  - x86/x86_64: checks for AVX2 via `is_x86_feature_detected!("avx2")`
  - aarch64: checks for NEON via `std::arch::is_aarch64_feature_detected!("neon")`
  - Falls back to `Scalar` if neither available
- Implements `Backend` trait by dispatching to wrapped backend

### Conditional Compilation (src/backend/mod.rs)
- `#[cfg(feature = "simd")]` guards simd module import
- Conditional re-export of `SimdBackend` and `RuntimeBackend`
- Maintains backward compatibility when feature disabled

### Public API (src/lib.rs)
- Added conditional re-exports: `pub use backend::{SimdBackend, RuntimeBackend}`
- Only exported when `simd` feature enabled
- Existing code unaffected

## Testing

### New Tests (8 total)
All tests in `src/backend/simd.rs::tests`:
1. `simd_backend_implements_trait` - Dimension validation
2. `simd_dot_product_correctness` - Numeric correctness (1·5 + 2·6 + 3·7 + 4·8 = 70)
3. `simd_fwht_roundtrip` - FWHT involution property (H̃² = I)
4. `runtime_backend_selects_best` - CPU detection doesn't crash
5. `runtime_backend_fwht_correctness` - RuntimeBackend FWHT roundtrip
6. `runtime_backend_dot_product` - RuntimeBackend dot product correctness
7. `simd_matches_scalar_fwht` - Bit-exact agreement with ScalarBackend
8. `simd_matches_scalar_dot_product` - Bit-exact agreement with ScalarBackend

### Regression Testing
- **With `--features simd`:** 51 tests pass (43 existing + 8 new)
- **Without feature:** 43 tests pass (existing tests unaffected)
- **Clippy:** Pre-existing warnings only (double_must_use), no new warnings
- **Release build:** Compiles successfully

## Verification

All acceptance criteria met:
- ✅ Cargo.toml contains `[features]` section with `simd = []`
- ✅ src/backend/simd.rs exists with `SimdBackend` struct
- ✅ src/backend/simd.rs contains `RuntimeBackend` enum
- ✅ src/backend/simd.rs contains `best_available()` function
- ✅ Both SimdBackend and RuntimeBackend implement `Backend` trait
- ✅ CPU feature detection present (AVX2 and NEON checks)
- ✅ src/backend/mod.rs uses `#[cfg(feature = "simd")]` guards
- ✅ src/lib.rs conditionally re-exports SIMD types
- ✅ All tests pass with and without SIMD feature

## Deviations from Plan

None - plan executed exactly as written.

## Key Decisions

**Static dispatch over dynamic dispatch:** Used `enum RuntimeBackend` with pattern matching instead of `Box<dyn Backend>`. This provides zero-cost abstraction (monomorphization at compile time) and matches the project's no-overhead philosophy.

**Scalar delegation in SimdBackend:** Plan correctly separated infrastructure setup (this plan) from actual SIMD implementation (Plan 02). SimdBackend currently calls scalar functions but has clear TODO markers for replacement.

**Standard library CPU detection:** Used `is_x86_feature_detected!` and `std::arch::is_aarch64_feature_detected!` macros. These are stable, have zero dependencies, and are production-ready.

## Files Created/Modified

### Created (1)
- `src/backend/simd.rs` (217 lines) - SimdBackend, RuntimeBackend, 8 tests

### Modified (3)
- `Cargo.toml` - Added `[features]` section
- `src/backend/mod.rs` - Added conditional simd module import and re-exports
- `src/lib.rs` - Added conditional public re-exports of SIMD types

## Commits

1. **ad9a08b** - `feat(02-01): add simd feature flag and backend infrastructure`
   - Cargo.toml, src/backend/simd.rs, src/backend/mod.rs
   - Feature flag, SimdBackend, RuntimeBackend with CPU detection
   - 8 new tests, all passing

2. **9339f0a** - `feat(02-01): add SIMD type re-exports to public API`
   - src/lib.rs
   - Conditional public API exports

## Next Steps

**For Plan 02-02 (SIMD Intrinsics):**
1. Replace SimdBackend::fwht_normalized_inplace() TODO with AVX2/NEON implementation
2. Replace SimdBackend::dot_product() TODO with SIMD horizontal sum
3. Add unsafe blocks with SAFETY documentation
4. Benchmark SIMD vs scalar performance (target 2-4x speedup)
5. Test on both x86_64 and aarch64 hardware

**Dependencies:**
- Plan 02-02 requires this plan's infrastructure
- Plan 02-03 (tests and benchmarks) requires Plan 02-02's implementations

## Self-Check: PASSED

✅ Created file exists: src/backend/simd.rs
✅ Modified files exist: Cargo.toml, src/backend/mod.rs, src/lib.rs
✅ Commit ad9a08b exists in git log
✅ Commit 9339f0a exists in git log
✅ All 51 tests pass with --features simd
✅ All 43 tests pass without feature flag
