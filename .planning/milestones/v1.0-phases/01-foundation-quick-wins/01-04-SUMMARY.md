---
phase: 01-foundation-quick-wins
plan: 04
subsystem: performance-optimization
tags: [allocation-elimination, hot-path-optimization, refcell, scratch-buffer]
dependency_graph:
  requires:
    - "01-03: Backend trait abstraction enabling generic PolarQuant<B>"
  provides:
    - "Zero-allocation inner_product() via RefCell scratch buffer"
    - "PolarQuant with pre-allocated scratch buffer reuse"
  affects:
    - "Phase 2 SIMD implementations will inherit zero-allocation pattern"
    - "All attention computation now allocation-free in hot path"
tech_stack:
  added:
    - "std::cell::RefCell for interior mutability"
  patterns:
    - "Scratch buffer reuse pattern via RefCell<Vec<f32>>"
    - "Manual Clone implementation for correct scratch buffer initialization"
    - "Explicit borrow scope to ensure guard drops before return"
key_files:
  modified:
    - path: "src/polar_quant.rs"
      change: "Added RefCell<Vec<f32>> scratch field, modified inner_product to reuse buffer"
      impact: "Eliminates per-call Vec allocation in attention hot path (N allocations per query)"
      lines_changed: 44
decisions:
  - what: "Use RefCell for scratch buffer instead of Mutex"
    why: "Single-threaded hot path, RefCell has zero runtime cost vs Mutex overhead"
    alternatives: "Could use Mutex for thread safety, but inner_product is called sequentially per query"
  - what: "Remove Clone derive, add manual Clone impl"
    why: "Manual Clone needed to pre-allocate scratch buffer with correct capacity on clone"
    alternatives: "Could derive Clone but would get uninitialized scratch buffer"
  - what: "Explicit borrow scope block in inner_product"
    why: "Ensures borrow_mut guard drops before function returns, prevents runtime panic"
    alternatives: "Could use method chaining, but explicit scope is clearer and safer"
metrics:
  duration_minutes: 2
  completed_at: "2026-03-27T14:41:19Z"
  tasks_completed: 1
  files_modified: 1
  tests_added: 2
  tests_passing: 42
  commits: 1
requirements:
  - id: FOUND-07
    status: complete
    evidence: "PolarQuant::inner_product() uses RefCell scratch buffer, query.to_vec() removed from method"
---

# Phase 1 Plan 4: Scratch Buffer Allocation Elimination

**One-liner:** Eliminated per-call Vec allocation in PolarQuant::inner_product() by adding RefCell scratch buffer, targeting 1.5-2x speedup on attention hot path for 8192-token sequences.

## What Was Built

Refactored `PolarQuant::inner_product()` to reuse a pre-allocated scratch buffer instead of allocating a new `Vec<f32>` on every call. For attention computation with N tokens, this eliminates N allocations per query (8192 × 512 bytes = 4MB of throwaway allocations per attention pass at 128-dim).

### Performance Impact

**Before (01-03):**
```rust
pub fn inner_product(&self, query: &[f32], key: &QuantizedVector) -> Result<f32> {
    let mut q_rot: Vec<f32> = query.to_vec();  // ← ALLOCATION
    self.rotation.apply(&mut q_rot);
    // ... compute dot product
}
```

**After (01-04):**
```rust
pub fn inner_product(&self, query: &[f32], key: &QuantizedVector) -> Result<f32> {
    let dot = {
        let mut scratch = self.scratch.borrow_mut();  // ← REUSE
        scratch.clear();
        scratch.extend_from_slice(query);
        self.rotation.apply(&mut scratch);
        // ... compute dot product
    }; // borrow_mut guard dropped here
    Ok(dot * key.norm)
}
```

**Memory savings per attention pass:**
- 128-dim, 8192 tokens: 8192 × 512 bytes = 4 MB avoided allocations
- 256-dim, 8192 tokens: 8192 × 1024 bytes = 8 MB avoided allocations

### Implementation Details

**PolarQuant struct changes:**
```rust
#[derive(Debug)]  // Removed Clone derive
pub struct PolarQuant<B: Backend = ScalarBackend> {
    rotation: Rotation<B>,
    codebook: Codebook,
    backend: B,
    scratch: RefCell<Vec<f32>>,  // ← NEW FIELD
}
```

**Manual Clone implementation:**
```rust
impl<B: Backend> Clone for PolarQuant<B> {
    fn clone(&self) -> Self {
        let dim = self.rotation.dim;
        Self {
            rotation: self.rotation.clone(),
            codebook: self.codebook.clone(),
            backend: self.backend.clone(),
            scratch: RefCell::new(Vec::with_capacity(dim)),  // ← Pre-allocate
        }
    }
}
```

**Constructor initialization:**
```rust
pub fn new_with_backend(dim: usize, bits: u8, seed: u64, backend: B) -> Result<Self> {
    let rotation = Rotation::new_with_backend(dim, seed, backend.clone())?;
    let codebook = Codebook::new(bits, dim)?;
    let scratch = RefCell::new(Vec::with_capacity(dim));  // ← Pre-allocate
    Ok(Self { rotation, codebook, backend, scratch })
}
```

### Safety Properties

**RefCell Borrow Rules:**
1. Explicit scope block `{ ... }` ensures `borrow_mut()` guard drops before return
2. No re-entrant calls — inner_product never calls itself
3. Single-threaded hot path — no cross-thread borrow conflicts
4. Clear error if double-borrow (runtime panic with clear message)

**Testing verification:**
- `inner_product_repeated_calls_no_panic` — 1000 sequential calls succeed
- `quantize_after_inner_product` — No borrow conflict with other methods
- All existing accuracy tests pass — numerical equivalence verified

## Testing & Verification

### Test Results

**Library tests:** 42/42 passed
- 40 existing tests (unchanged, proving numerical equivalence)
- 2 new tests for scratch buffer safety

**New tests added:**
```rust
#[test]
fn inner_product_repeated_calls_no_panic() {
    let pq = make_pq(3);
    let key = sine_vec(128, 0.05);
    let query = sine_vec(128, 0.07);
    let qv = pq.quantize(&key).unwrap();

    for _ in 0..1000 {
        let _ = pq.inner_product(&query, &qv).unwrap();
    }
}

#[test]
fn quantize_after_inner_product() {
    let pq = make_pq(3);
    let key = sine_vec(128, 0.05);
    let query = sine_vec(128, 0.07);
    let qv = pq.quantize(&key).unwrap();

    let _ = pq.inner_product(&query, &qv).unwrap();
    let qv2 = pq.quantize(&query).unwrap();
    assert_eq!(qv2.dim, 128);
}
```

### Verification Commands

**Allocation removed:**
```bash
$ grep -n 'query.to_vec()' src/polar_quant.rs
# Returns: (empty) — allocation eliminated from inner_product
```

**Scratch buffer present:**
```bash
$ grep -n 'scratch.borrow_mut\|scratch.clear\|scratch.extend_from_slice' src/polar_quant.rs
163:            let mut scratch = self.scratch.borrow_mut();
164:            scratch.clear();
165:            scratch.extend_from_slice(query);
```

## Deviations from Plan

None — plan executed exactly as written.

## Performance Analysis

### Expected Speedup

**Allocation overhead eliminated:**
- Before: O(N) allocations per attention pass (N = sequence length)
- After: 0 allocations per attention pass (scratch buffer reused)
- Target: 1.5-2x speedup on inner_product hot path

**Why 1.5-2x improvement:**
1. Malloc/free overhead eliminated (system call per allocation)
2. Memory initialization overhead eliminated (memset for new Vec)
3. Better cache locality (reusing warm memory)
4. Reduced allocator contention in multi-threaded scenarios

**Benchmark validation needed:**
- Baseline: Plan 01-02 established integration benchmarks
- Next: Re-run `cargo bench --bench integration` to measure improvement
- Expected: `inner_product/batch_1000` throughput increase 1.5-2x

### Scalability Impact

**Short sequences (128 tokens):**
- Modest improvement (128 allocations saved)

**Long sequences (8192 tokens):**
- Significant improvement (8192 allocations × 512 bytes = 4 MB saved)
- Reduced GC pressure in managed runtimes (if binding to Python/JS)

**Multi-query batching:**
- Scratch buffer reused across all queries in batch
- Linear memory footprint (1 buffer per PolarQuant) vs quadratic allocations

## Accomplishments

1. Zero-allocation inner_product() implementation via RefCell scratch buffer
2. Pre-allocated scratch buffer (capacity = dim) in constructors and Clone
3. Explicit borrow scope ensuring guard drops before return (no panic risk)
4. All 42 tests pass, including 2 new safety tests
5. Numerical equivalence verified (existing accuracy tests unchanged)
6. Foundation for SIMD phase — SIMD backends will inherit zero-allocation pattern

## Task Commits

1. **Task 1: Add RefCell scratch buffer to PolarQuant and use in inner_product** - `206eb65` (feat)

## Files Modified

- `src/polar_quant.rs` - Added RefCell scratch buffer, modified inner_product, manual Clone impl

## Decisions Made

1. **RefCell over Mutex:** Single-threaded hot path requires zero-cost interior mutability
2. **Manual Clone impl:** Ensures scratch buffer pre-allocated with correct capacity on clone
3. **Explicit borrow scope:** Safety pattern for guaranteeing borrow_mut guard drops before return

## Next Phase Readiness

- Allocation-free hot path established for Phase 2 (SIMD) implementations
- SIMD backends can use same scratch buffer pattern for vectorized operations
- Baseline for measuring SIMD speedup (1.5-2x from scratch buffer + 2-4x from SIMD = 3-8x total)

## Commits

| Hash    | Message                                                                 |
| ------- | ----------------------------------------------------------------------- |
| 206eb65 | feat(01-04): add RefCell scratch buffer to PolarQuant for zero-allocation inner_product |

## Self-Check

Verifying claims made in summary:

**Modified file:**
```bash
[ -f "/Users/gqadonis/Projects/turboquant-rs/src/polar_quant.rs" ] && echo "✓ polar_quant.rs exists" || echo "✗ MISSING"
```

**Commit:**
```bash
git log --oneline --all | grep -q "206eb65" && echo "✓ Commit 206eb65 exists" || echo "✗ MISSING"
```

**Scratch buffer implementation:**
```bash
grep -q "scratch: RefCell<Vec<f32>>" src/polar_quant.rs && echo "✓ RefCell scratch field present" || echo "✗ MISSING"
grep -q "self.scratch.borrow_mut()" src/polar_quant.rs && echo "✓ borrow_mut usage present" || echo "✗ MISSING"
grep -q "scratch.clear()" src/polar_quant.rs && echo "✓ scratch.clear() present" || echo "✗ MISSING"
grep -q "scratch.extend_from_slice(query)" src/polar_quant.rs && echo "✓ extend_from_slice present" || echo "✗ MISSING"
```

**Allocation eliminated:**
```bash
grep -q "query.to_vec()" src/polar_quant.rs && echo "✗ Allocation still present" || echo "✓ query.to_vec() removed"
```

**New tests:**
```bash
grep -q "fn inner_product_repeated_calls_no_panic" src/polar_quant.rs && echo "✓ Repeated calls test present" || echo "✗ MISSING"
grep -q "fn quantize_after_inner_product" src/polar_quant.rs && echo "✓ Borrow conflict test present" || echo "✗ MISSING"
```

**Test results:**
```bash
cargo test --lib polar_quant 2>&1 | grep -q "8 passed" && echo "✓ All polar_quant tests pass" || echo "✗ FAILED"
cargo test --lib 2>&1 | grep -q "42 passed" && echo "✓ All library tests pass" || echo "✗ FAILED"
```

Running verification...

## Self-Check: PASSED

All files modified, commit exists, all claims verified.

---
*Phase: 01-foundation-quick-wins*
*Plan: 04*
*Completed: 2026-03-27*
