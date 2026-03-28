---
phase: 01-foundation-quick-wins
plan: 02
subsystem: benchmarks
tags: [benchmarks, integration, baseline, performance]
completed: 2026-03-27T14:21:42Z
duration_seconds: 160

dependency_graph:
  requires: []
  provides:
    - integration-benchmarks
    - attention-throughput-baselines
    - inner-product-throughput-baselines
  affects:
    - benches/integration.rs
    - Cargo.toml

tech_stack:
  added:
    - criterion-integration-benchmarks
  patterns:
    - realistic-attention-workload-simulation
    - batch-throughput-measurement
    - amortized-overhead-benchmarking

key_files:
  created:
    - benches/integration.rs
  modified:
    - Cargo.toml

decisions:
  - id: SEQ-LENGTHS
    summary: "Benchmark 128/512/2048/8192 sequence lengths"
    rationale: "Covers short to long contexts, validates Phase 1 success criterion #3"
    alternatives: "Could add 4096, but 2048→8192 jump sufficient for detecting O(n) vs O(n²) patterns"

  - id: BATCH-1000
    summary: "Use batch_1000 for inner product throughput"
    rationale: "Establishes baseline for scratch buffer improvement target (1.5-2x from FOUND-07)"
    alternatives: "Could use smaller batch, but 1000 fully amortizes per-call overhead"

  - id: LOGITS-ONLY
    summary: "Separate logits_only benchmark from full attend"
    rationale: "Isolates inner_product hot path from softmax/weighted_sum overhead"
    alternatives: "Could combine, but separation enables precise optimization targeting"

metrics:
  tasks_completed: 2
  tasks_total: 2
  files_modified: 2
  commits: 2
  test_status: passing
---

# Phase 01 Plan 02: Integration Benchmarks Summary

**One-liner:** Criterion integration benchmarks for attention at 128-8192 tokens, inner product throughput at 2-4 bits, establishing baselines for all future optimization work

## Execution Report

### Tasks Completed

| Task | Description | Commit | Status |
|------|-------------|--------|--------|
| 1 | Add integration benchmark target to Cargo.toml | 4ad99a4 | ✓ Complete |
| 2 | Create integration benchmarks for realistic attention workloads | 357bb14 | ✓ Complete |

### Verification Results

All verification criteria passed:

1. **Cargo.toml validation**: Integration benchmark target properly declared
2. **Compilation test**: `cargo bench --bench integration -- --test` passes successfully
3. **Existing benchmarks**: `cargo bench --bench bench -- --test` still passes
4. **Benchmark groups**: All three groups present (attention_e2e, inner_product_throughput, quantize_throughput)
5. **Sequence lengths**: All four lengths (128, 512, 2048, 8192) covered
6. **Bit widths**: All three widths (2, 3, 4) covered
7. **Line count**: benches/integration.rs has 146 lines (exceeds 40-line minimum)

### Deviations from Plan

**None** - Plan executed exactly as written. All tasks completed without modifications or auto-fixes.

## Technical Implementation

### Benchmarks Created

**attention_e2e group:**
- `attend/{seq_len}`: Full attention (logits + softmax + weighted sum) at 128/512/2048/8192 tokens
- `logits_only/{seq_len}`: Inner product hot path only at 128/512/2048/8192 tokens
- 3-bit compression, 128-dim head
- 3-second warmup, 10-second measurement for statistical stability

**inner_product_throughput group:**
- `single/{bits}bit`: Single inner product call at 2/3/4-bit (measures per-call overhead)
- `batch_1000/{bits}bit`: 1000 inner products at 2/3/4-bit (amortized overhead baseline)
- 2-second warmup, 5-second measurement
- Establishes baseline for scratch buffer optimization target

**quantize_throughput group:**
- `batch_100/{bits}bit`: Quantize 100 vectors at 2/3/4-bit
- Simulates cache fill operation for 100 tokens
- Baseline for quantization performance

### Architecture Patterns

**Benchmark design:**
- Follows existing `benches/bench.rs` patterns (sine_vec helper, black_box usage)
- Uses `fill_cache` helper to simulate realistic cache state
- Separates hot path (logits) from full computation (attend) for precise measurement
- Batch benchmarks establish amortized overhead baselines

**Integration focus:**
- Measures end-to-end attention workloads (not microbenchmarks)
- Realistic sequence lengths for transformer inference
- Provides baseline numbers for Phase 1 success criterion #3

## Impact & Next Steps

### Baselines Established

These benchmarks provide baseline numbers for measuring:
1. **FOUND-07 target**: 1.5-2x improvement from scratch buffer reuse (measured by batch_1000 benchmarks)
2. **SIMD target**: 2-4x improvement from FWHT vectorization (measured by logits_only benchmarks)
3. **Combined target**: 3-8x total improvement (measured by attend benchmarks)

### Files Changed

**Created:**
- `benches/integration.rs` (146 lines): Criterion integration benchmarks with 3 benchmark groups

**Modified:**
- `Cargo.toml`: Added `[[bench]]` target for integration benchmarks

### Next Actions

To use these benchmarks:

```bash
# Run all integration benchmarks
cargo bench --bench integration

# Run specific benchmark group
cargo bench --bench integration -- attention_e2e
cargo bench --bench integration -- inner_product_throughput

# Compare before/after optimization
cargo bench --bench integration --save-baseline before
# ... make optimization changes ...
cargo bench --bench integration --baseline before
```

## Validation

### Acceptance Criteria

- [x] `cargo bench --bench integration -- --test` compiles successfully
- [x] benches/integration.rs contains criterion_group! macro
- [x] benches/integration.rs contains bench_attention_e2e function
- [x] benches/integration.rs contains bench_inner_product_throughput function
- [x] benches/integration.rs contains bench_quantize_throughput function
- [x] benches/integration.rs benchmarks 8192 sequence length
- [x] benches/integration.rs imports KvCache and PolarQuant
- [x] Cargo.toml contains `name = "integration"` in [[bench]] section
- [x] Cargo.toml contains `path = "benches/integration.rs"`
- [x] Existing `cargo bench --bench bench` still works

### Must-Have Verification

**Truths validated:**
- ✓ User can run `cargo bench --bench integration` and see attention throughput for 128/512/2048/8192 tokens
- ✓ Benchmark results provide baseline numbers for measuring future optimization gains

**Artifacts validated:**
- ✓ benches/integration.rs provides Criterion integration benchmarks (146 lines, contains criterion_group!)
- ✓ Cargo.toml provides bench target declaration (contains `name = "integration"`)

**Key links validated:**
- ✓ benches/integration.rs → src/kv_cache.rs via `cache.attend()` pattern
- ✓ Cargo.toml → benches/integration.rs via [[bench]] target declaration

## Self-Check

### Files Verification

```bash
[ -f "benches/integration.rs" ] && echo "FOUND: benches/integration.rs" || echo "MISSING: benches/integration.rs"
```
**Result:** FOUND: benches/integration.rs

```bash
grep -q 'name = "integration"' Cargo.toml && echo "FOUND: integration target in Cargo.toml" || echo "MISSING: integration target"
```
**Result:** FOUND: integration target in Cargo.toml

### Commits Verification

```bash
git log --oneline --all | grep -q "4ad99a4" && echo "FOUND: 4ad99a4" || echo "MISSING: 4ad99a4"
```
**Result:** FOUND: 4ad99a4

```bash
git log --oneline --all | grep -q "357bb14" && echo "FOUND: 357bb14" || echo "MISSING: 357bb14"
```
**Result:** FOUND: 357bb14

## Self-Check: PASSED

All files created, all commits exist, all verification criteria met.
