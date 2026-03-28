---
phase: 03-batch-operations
plan: 02
subsystem: batch-api
tags: [parallel, rayon, kv-cache, batch-attend]
dependency_graph:
  requires: [03-01-batch-quantization]
  provides: [kv-cache-batch-api]
  affects: [transformer-inference, multi-query-attention]
tech_stack:
  added: []
  patterns: [thread-local-reconstruction, batch-of-1-fast-path]
key_files:
  created: []
  modified:
    - src/kv_cache.rs
    - src/turboquant.rs
    - src/polar_quant.rs
decisions:
  - desc: "Thread-local KvCache reconstruction pattern to avoid RefCell Sync issues"
    rationale: "KvCache contains RefCell (not Sync), so we extract parameters and reconstruct in each thread"
  - desc: "Added backend() and seed() accessors to enable reconstruction"
    rationale: "Needed to extract TurboQuant construction parameters for parallel threads"
  - desc: "Clone entries Vec per thread rather than Arc sharing"
    rationale: "Acceptable for Phase 3 scope - can optimize with shared refs in Phase 4 if GPU dispatch requires it"
metrics:
  duration_seconds: 278
  completed_date: "2026-03-28"
  tasks_completed: 1
  tests_added: 6
  tests_passing: 57
---

# Phase 03 Plan 02: Batch Attend API Summary

**One-liner:** Parallel multi-query attention via batch_attend and batch_attend_slices with thread-local KvCache reconstruction

## Overview

Completed the batch API surface for KvCache by adding `batch_attend` and `batch_attend_slices` methods. Enables parallel processing of multiple query vectors against a compressed KV cache, critical for efficient transformer inference with multiple attention heads or batch decoding.

## What Was Built

### Core Features

1. **Clone Implementations**
   - `Clone for TurboQuant<B>` - enables cache cloning for parallel workers
   - `Clone for KvCache<B>` - enables rayon parallelism
   - `Clone` derive on `Entry` struct - enables cloning compressed key-value pairs

2. **Batch Attend Methods**
   - `batch_attend(&[Vec<f32>])` - owned vectors, parallel processing via rayon
   - `batch_attend_slices(&[&[f32]])` - zero-copy borrowed slices variant
   - Batch-of-1 fast paths delegate directly to single `attend()` (zero overhead)
   - Up-front dimension validation before parallel dispatch

3. **Accessor Methods**
   - `TurboQuant::seed()` - exposes rotation seed for reconstruction
   - `TurboQuant::backend()` - exposes backend for reconstruction
   - `PolarQuant::backend()` - exposes backend for reconstruction

## Implementation Details

### Thread-Local Reconstruction Pattern

**Challenge:** KvCache contains `RefCell<Vec<f32>>` via PolarQuant scratch buffers, which is `!Sync`.

**Solution:** Extract reconstruction parameters (head_dim, bits, seeds, backend) and reconstruct fresh KvCache instances in each worker thread:

```rust
let head_dim = self.head_dim;
let bits = self.key_tq.bits();
let key_seed = self.key_tq.seed();
let val_seed = self.val_tq.seed();
let backend = self.key_tq.backend().clone();
let entries_clone = self.entries.clone();

queries.par_iter()
    .map_init(
        move || {
            let mut cache = KvCache::new_with_backend(
                head_dim, bits, key_seed, val_seed, backend.clone()
            ).unwrap();
            cache.entries = entries_clone.clone();
            cache
        },
        |cache, q| cache.attend(q)
    )
    .collect()
```

This pattern follows Phase 03-01 approach for PolarQuant batch operations.

### Performance Characteristics

- **Batch-of-1:** Delegates to single `attend()` - zero overhead
- **Batch ≥ 2:** Parallel via rayon work-stealing
- **Memory cost:** Each worker clones entries Vec (acceptable for Phase 3 scope)
- **Future optimization:** Can use Arc or shared refs in Phase 4 for GPU dispatch if needed

## Testing

Added 6 new tests in `kv_cache::batch_tests` module:

1. `batch_attend_dimensions` - verifies output shape (4 queries → 4 outputs of head_dim)
2. `batch_attend_empty` - empty input returns empty result
3. `batch_attend_single_matches_attend` - batch-of-1 equivalence with single attend
4. `batch_attend_slices_equivalence` - slices variant matches owned variant
5. `batch_attend_sequential_equivalence` - batch results match sequential loop of attend()
6. `batch_attend_dimension_mismatch` - dimension validation error handling

All tests verify correctness within f32 epsilon (< 1e-5).

## Deviations from Plan

None - plan executed exactly as written.

## Integration Points

- **Upstream:** Requires 03-01 batch quantization pattern (thread-local reconstruction)
- **Downstream:** Enables multi-head attention batching in transformer models
- **External:** Rayon for parallelism (already in dependencies from 03-01)

## Known Limitations

1. **Memory cost:** Each worker clones the entries Vec. For very large caches (>10K entries) this may be expensive.
2. **RefCell overhead:** Each thread gets fresh RefCell instances despite being independent.
3. **No SIMD:** Batch operations use scalar backend (SIMD from Phase 02 applies within each attend).

These are acceptable for Phase 3 scope. Phase 4 GPU work may motivate shared reference optimization.

## Verification

```bash
# All library tests pass
cargo test --lib
# Result: 57 passed (51 existing + 6 new)

# Batch-specific tests pass
cargo test --lib kv_cache::batch
# Result: 6 passed

# Full test suite including integration
cargo test
# Result: all pass
```

## Next Steps

- Phase 03-03: Batch KvCache operations (batch_push, batch_clear) if planned
- Phase 04: GPU acceleration for large-batch scenarios (may optimize memory sharing)

## Self-Check: PASSED

Verified all files and commits exist:

```bash
# Created files: none (all modifications)
# Modified files verified:
[✓] src/kv_cache.rs exists and contains batch_attend, batch_attend_slices
[✓] src/turboquant.rs exists and contains Clone impl
[✓] src/polar_quant.rs exists and contains backend() accessor

# Commits verified:
[✓] 26abd2a exists: feat(03-02): implement batch_attend and batch_attend_slices for KvCache
```

All deliverables present and tests passing.
