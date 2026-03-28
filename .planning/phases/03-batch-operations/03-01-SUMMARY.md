---
phase: 03-batch-operations
plan: 01
subsystem: batch-api
tags: [rayon, parallelism, performance, batch-processing]
dependencies:
  requires: [Backend trait (01-03), RefCell scratch buffers (01-04)]
  provides: [batch_quantize, batch_quantize_slices, batch_inner_product]
  affects: [PolarQuant API surface, Backend trait bounds]
tech_stack:
  added: [rayon 1.11]
  patterns: [map_init for thread-local state, batch-of-1 fast paths]
key_files:
  created: []
  modified:
    - path: Cargo.toml
      lines_added: 1
      description: Added rayon 1.11 dependency
    - path: src/polar_quant.rs
      lines_added: 116
      description: Added batch methods and comprehensive test suite
    - path: src/backend/mod.rs
      lines_added: 2
      description: Added Send + Sync bounds to Backend trait
    - path: src/rotation.rs
      lines_added: 2
      description: Added seed field for reconstruction in parallel contexts
decisions:
  - what: Use map_init pattern instead of sharing PolarQuant across threads
    why: PolarQuant contains RefCell (not Sync), so each thread needs independent instance
    alternatives_considered:
      - Shared Arc<PolarQuant> - rejected because RefCell is !Sync
      - RwLock instead of RefCell - rejected to avoid locking overhead in sequential use
    impact: Minimal - Clone is cheap (rotation signs vector, codebook constants)
  - what: Add Send + Sync bounds to Backend trait
    why: Enable rayon parallelism across all backend implementations
    alternatives_considered: []
    impact: Breaking change for hypothetical non-Send backends (none exist in codebase)
  - what: Store seed in Rotation struct
    why: Enables reconstruction of Rotation in parallel worker threads
    alternatives_considered:
      - Pass seed separately - rejected for encapsulation
      - Reconstruct from signs vector - rejected as non-trivial
    impact: 8 bytes per Rotation instance (negligible)
  - what: Batch-of-1 fast paths
    why: Zero overhead for single-vector calls, simplifies API
    alternatives_considered:
      - Always use parallel path - rejected due to rayon overhead for n=1
    impact: Improved performance for edge cases
metrics:
  duration_seconds: 348
  tasks_completed: 2
  files_modified: 4
  tests_added: 9
  commits: 3
  completed_date: "2026-03-28"
---

# Phase 03 Plan 01: Batch Operations - Rayon Parallel APIs Summary

**One-liner:** Added rayon-powered batch APIs (batch_quantize, batch_quantize_slices, batch_inner_product) with thread-local PolarQuant instances via map_init pattern, achieving zero-overhead batch-of-1 fast paths and comprehensive test coverage.

## What Was Built

### Core Functionality
- **batch_quantize**: Parallel quantization of multiple owned vectors
- **batch_quantize_slices**: Zero-copy variant accepting borrowed slices
- **batch_inner_product**: Parallel attention logit computation (1 query × N keys)

All three methods follow the same pattern:
1. Empty input → return empty vec
2. Batch-of-1 → delegate to single-vector method (zero overhead)
3. Batch ≥ 2 → up-front dimension validation, then rayon parallel dispatch

### Thread Safety Architecture
```rust
// Each rayon worker gets independent PolarQuant via map_init
vecs.par_iter()
    .map_init(
        move || Self::new_with_backend(dim, bits, seed, backend.clone()).unwrap(),
        |pq, v| pq.quantize(v)
    )
    .collect()
```

**Key insight:** PolarQuant contains `RefCell<Vec<f32>>` scratch buffer which is `!Sync`. The map_init pattern creates a fresh PolarQuant instance with independent RefCell for each rayon worker thread, avoiding Sync requirement while maintaining zero-allocation inner_product hot path.

## Tests Added

Nine comprehensive tests covering:
- Length correctness (10 vectors → 10 results)
- Empty input handling
- Dimension mismatch error propagation
- Slices variant equivalence with owned variant
- Sequential equivalence (batch matches loop of single calls)
- Batch-of-1 fast paths (same result as single call)
- Large batch (64 vectors) exercising parallel path

**Result:** 51 total tests (42 existing + 9 new), all passing.

## Deviations from Plan

### Auto-fixed Issues (Deviation Rules Applied)

**1. [Rule 2 - Missing Critical Functionality] Added seed field to Rotation**
- **Found during:** Task 1 GREEN phase - implementing map_init closures
- **Issue:** Cannot reconstruct Rotation in worker threads without seed
- **Fix:** Added `pub seed: u64` field to `Rotation<B>` struct
- **Files modified:** `src/rotation.rs` (+2 lines)
- **Commit:** d820dfc
- **Rationale:** Essential for batch API correctness. Without seed, each worker would create non-deterministic rotations, breaking quantization/dequantization symmetry.

**2. [Rule 2 - Missing Critical Functionality] Added Send + Sync bounds to Backend trait**
- **Found during:** Task 1 GREEN phase - rayon compilation errors
- **Issue:** Backend trait lacked thread safety bounds for parallel processing
- **Fix:** Changed `pub trait Backend: Clone + std::fmt::Debug` to include `+ Send + Sync`
- **Files modified:** `src/backend/mod.rs` (+2 lines)
- **Commit:** d820dfc
- **Rationale:** Required for rayon parallelism. All existing backends (ScalarBackend, SimdBackend, RuntimeBackend) are trivially Send + Sync (no interior mutability, no raw pointers).

**3. [Rule 1 - Bug] Removed unused Arc import**
- **Found during:** Task 2 - clippy validation
- **Issue:** Leftover import from exploratory implementation attempt
- **Fix:** Removed `sync::Arc` from use statement
- **Files modified:** `src/polar_quant.rs` (1 line)
- **Commit:** 95b0349
- **Rationale:** Clippy warning - unused import violates code quality standards.

## Performance Characteristics

### Batch-of-1 Fast Path
Zero overhead - delegates directly to single-vector method:
```rust
if vecs.len() == 1 {
    return Ok(vec![self.quantize(&vecs[0])?]);
}
```

### Parallel Dispatch Overhead
Up-front dimension validation prevents spawning threads for invalid inputs:
```rust
for v in vecs {
    self.check_dim(v.len())?;  // Check before par_iter
}
```

### Clone Cost Analysis
Each worker clones PolarQuant, which includes:
- Rotation (dim × i8 signs vector + seed)
- Codebook (8 × f32 centroids)
- Backend (zero-sized for ScalarBackend, small enum for RuntimeBackend)
- RefCell scratch buffer (capacity allocated, length 0)

**Total clone cost:** ~dim bytes + 64 bytes overhead. For 128-dim: ~200 bytes per worker. Negligible compared to quantization work.

## Success Criteria Verification

- [x] Three batch methods on PolarQuant: ✓ (batch_quantize, batch_quantize_slices, batch_inner_product)
- [x] All use rayon par_iter with map_init for thread-safe RefCell handling: ✓
- [x] Batch-of-1 fast paths delegate to single-vector methods: ✓
- [x] Up-front dimension validation before parallel dispatch: ✓
- [x] All 42+ existing tests pass, new batch tests pass: ✓ (51 tests total)
- [x] Rayon 1.11 in Cargo.toml dependencies: ✓

## Integration Notes

### For Plan 02 (KvCache batch_attend)
KvCache can now call `polar_quant.batch_inner_product(query, &compressed_keys)` to compute all attention logits in parallel. The batch_inner_product method handles:
- Query dimension validation
- Empty keys handling
- Per-key dimension validation
- Thread-local PolarQuant instances with independent scratch buffers

### API Stability
All three methods marked `#[must_use]` to prevent silent dropping of results. Public API surface of PolarQuant expanded by 3 methods - fully backward compatible (no breaking changes).

### Thread Safety Guarantees
- Backend trait now requires Send + Sync
- All batch methods accept `&self` (immutable reference)
- Thread-local state (RefCell scratch buffer) isolated per worker
- No shared mutable state across threads

## Files Changed

| File | +Lines | Purpose |
|------|--------|---------|
| Cargo.toml | +1 | rayon 1.11 dependency |
| src/polar_quant.rs | +116 | Batch methods + 9 tests |
| src/backend/mod.rs | +2 | Send + Sync bounds |
| src/rotation.rs | +2 | seed field for reconstruction |

**Total:** 4 files modified, 121 lines added, 1 line removed.

## Commits

| Hash | Type | Description |
|------|------|-------------|
| 54884cf | test | Add failing tests for batch APIs (TDD RED) |
| d820dfc | feat | Implement batch APIs with rayon parallelism (TDD GREEN) |
| 95b0349 | chore | Remove unused Arc import (clippy cleanup) |

## Self-Check: PASSED

**Files created:**
- None (only modifications)

**Key files modified exist:**
```
✓ /Users/gqadonis/Projects/turboquant-rs/Cargo.toml
✓ /Users/gqadonis/Projects/turboquant-rs/src/polar_quant.rs
✓ /Users/gqadonis/Projects/turboquant-rs/src/backend/mod.rs
✓ /Users/gqadonis/Projects/turboquant-rs/src/rotation.rs
```

**Commits exist:**
```
✓ 54884cf: test(03-01): add failing tests for batch APIs
✓ d820dfc: feat(03-01): implement batch APIs with rayon parallelism
✓ 95b0349: chore(03-01): remove unused Arc import
```

**Tests passing:**
```
✓ cargo test --lib: 51 passed
✓ cargo test --lib polar_quant::batch: 9 passed
✓ cargo test --doc: 1 passed
✓ cargo clippy: no errors in polar_quant.rs
```

## Next Steps

1. **Plan 02:** Implement KvCache::batch_attend using batch_inner_product
2. **Plan 03:** Add benchmarks for batch throughput vs sequential baseline
3. **Future:** Consider SIMD-accelerated batch operations once Phase 02 SIMD work completes
