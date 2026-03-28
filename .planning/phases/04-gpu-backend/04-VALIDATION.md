---
phase: 4
slug: gpu-backend
status: active
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-28
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in test (`#[test]`) + Criterion 0.5 (benchmarks) |
| **Config file** | Existing `benches/integration.rs` extended; new `benches/gpu_bench.rs` and `tests/gpu_integration.rs` created in Plan 04 |
| **Quick run command** | `cargo test --features gpu --release --lib gpu_` |
| **Full suite command** | `cargo test --features gpu --release && cargo bench --features gpu --bench gpu_bench` |
| **Estimated runtime** | Quick: ~10s, Full: ~2-5 min (includes Criterion warm-up) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --features gpu --release --lib gpu_` (~10s)
- **After every plan wave:** Run `cargo test --features gpu --release && cargo bench --features gpu --bench gpu_bench --no-fail-fast` (~2-5 min)
- **Before `/gsd:verify-work`:** Full suite must be green: `cargo test --features gpu --release && cargo bench --features gpu`
- **Max feedback latency:** 10 seconds (quick run), 300 seconds (full suite)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | GPU-07 | build | `cargo build && cargo build --features gpu` | Existing Cargo.toml | pending |
| 04-01-01 | 01 | 1 | GPU-01 | unit | `cargo test --features gpu --release --lib gpu_backend::tests::gpu_init_error_message_is_actionable` | W0: src/backend/gpu.rs (tests module) | pending |
| 04-01-01 | 01 | 1 | GPU-08 | unit | `cargo test --features gpu --release --lib gpu_backend::tests::gpu_init_error_message_is_actionable` | W0: src/backend/gpu.rs (tests module) | pending |
| 04-01-02 | 01 | 1 | GPU-02 | unit | `cargo test --features gpu --release --lib gpu_backend::tests::gpu_backend_delegates_to_scalar` | W0: src/backend/gpu.rs (tests module) | pending |
| 04-02-01 | 02 | 2 | GPU-03 | unit | `cargo test --features gpu --release --lib gpu_backend::tests::gpu_fwht_batch_matches_scalar` | W0: src/backend/gpu.rs (tests module) | pending |
| 04-02-02 | 02 | 2 | GPU-04 | integration | `cargo test --features gpu --release --test gpu_integration gpu_batch_inner_product_matches_cpu` | W0: tests/gpu_integration.rs | pending |
| 04-03-01 | 03 | 3 | GPU-05 | unit | `cargo test --features gpu --release --lib gpu_backend::tests::gpu_memory_pool_reuse` | W0: src/backend/gpu.rs (tests module) | pending |
| 04-03-02 | 03 | 3 | GPU-06 | integration | `cargo test --features gpu --release --test gpu_integration gpu_batch_dispatch_threshold` | W0: tests/gpu_integration.rs | pending |
| 04-03-02 | 03 | 3 | GPU-10 | integration | `cargo test --features gpu --release --test gpu_integration gpu_small_batch_uses_cpu_fallback` | W0: tests/gpu_integration.rs | pending |
| 04-04-01 | 04 | 4 | GPU-03 | integration | `cargo test --features gpu --release --test gpu_integration gpu_batch_quantize_matches_cpu` | W0: tests/gpu_integration.rs | pending |
| 04-04-01 | 04 | 4 | GPU-04 | integration | `cargo test --features gpu --release --test gpu_integration gpu_kvcache_attend_matches_cpu` | W0: tests/gpu_integration.rs | pending |
| 04-04-02 | 04 | 4 | GPU-09 | benchmark | `cargo bench --features gpu --bench gpu_bench -- batch_quantize_gpu_vs_cpu` | W0: benches/gpu_bench.rs | pending |
| 04-04-02 | 04 | 4 | PERF-02 | benchmark | `cargo bench --features gpu --bench integration -- combined_speedup_perf02` | W0: benches/integration.rs (updated) | pending |
| 04-05-01 | 05 | 4 | DOC-01 | file-exists | `test -f docs/FEATURE_FLAGS.md && grep -q "cargo build --features" docs/FEATURE_FLAGS.md` | W0: docs/FEATURE_FLAGS.md | pending |
| 04-05-02 | 05 | 4 | DOC-02 | file-exists | `test -f docs/PERFORMANCE_GUIDE.md && grep -q "GPU_BATCH_THRESHOLD" docs/PERFORMANCE_GUIDE.md` | W0: docs/PERFORMANCE_GUIDE.md | pending |

*Status: pending -- green -- red -- flaky*

---

## Wave 0 Requirements

The following test files and kernel sources do NOT exist yet. They are created by their respective plans during execution. Each plan's tasks include these files in their `<files>` declarations.

- [ ] `src/backend/gpu.rs` -- unit tests inline (Plan 01 Task 2 creates file with `#[cfg(test)] mod tests` block). Covers GPU-01, GPU-02, GPU-05, GPU-08.
- [ ] `src/kernels/fwht.cu` -- FWHT CUDA kernel source (Plan 02 Task 1 creates). Covers GPU-03.
- [ ] `src/kernels/attention.cu` -- Batch attention CUDA kernel source (Plan 02 Task 1 creates). Covers GPU-04.
- [ ] `build.rs` -- CUDA kernel compilation via nvcc (Plan 01 Task 1 creates stub, Plan 02 Task 1 upgrades). Covers GPU-07.
- [ ] `tests/gpu_integration.rs` -- GPU correctness integration tests (Plan 04 Task 1 creates). Covers GPU-03, GPU-04, GPU-06, GPU-10.
- [ ] `benches/gpu_bench.rs` -- GPU-specific Criterion benchmarks (Plan 04 Task 2 creates). Covers GPU-09.
- [ ] `benches/integration.rs` -- Updated with combined speedup benchmark (Plan 04 Task 2 modifies existing file). Covers PERF-02.
- [ ] `docs/FEATURE_FLAGS.md` -- Feature flag documentation (Plan 05 Task 1 creates). Covers DOC-01.
- [ ] `docs/PERFORMANCE_GUIDE.md` -- Performance guide (Plan 05 Task 2 creates). Covers DOC-02.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GPU batch >= 32 is measurably faster than CPU | GPU-09 | Benchmark results depend on hardware; Criterion reports must be human-interpreted for statistical significance | Run `cargo bench --features gpu --bench gpu_bench`, open `target/criterion/` HTML reports, confirm GPU bars shorter than CPU bars for batch sizes >= 32 |
| Combined Phase 1-4 speedup is 3-8x | PERF-02 | Absolute speedup depends on hardware and baseline; requires human judgment on whether target range is hit | Run `cargo bench --features gpu --bench integration -- combined_speedup`, compare `phase0_scalar_sequential` vs `phase4_gpu_batch` mean times, confirm ratio is 3-8x |

All other phase behaviors have automated verification via `cargo test` and `cargo bench`.

---

## Sampling Continuity Check

Wave-by-wave coverage ensures no 3 consecutive tasks lack automated verification:

| Wave | Plans | Tasks | Automated Tests |
|------|-------|-------|-----------------|
| 1 | 04-01 | 2 tasks | `cargo build`, `cargo test --lib gpu_backend` (both tasks produce testable output) |
| 2 | 04-02 | 2 tasks | `cargo build`, `cargo test --lib gpu_backend::tests::gpu_fwht_batch_matches_scalar` |
| 3 | 04-03 | 2 tasks | `cargo build`, `cargo test --lib gpu_backend::tests::gpu_memory_pool_reuse` |
| 4 | 04-04, 04-05 | 4 tasks | `cargo test --test gpu_integration`, `cargo bench --bench gpu_bench`, `test -f docs/*.md` |

No gaps: every wave has at least one automated verification command. No 3 consecutive tasks without automated verify.

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (9 files listed above, created by Plans 01-05)
- [x] No watch-mode flags (all commands terminate)
- [x] Feedback latency < 10s for quick run, < 300s for full suite
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-03-28
