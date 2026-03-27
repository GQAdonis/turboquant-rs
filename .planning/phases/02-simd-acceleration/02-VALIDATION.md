---
phase: 2
slug: simd-acceleration
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-27
updated: 2026-03-27
---

# Phase 2 -- Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | cargo test (Rust built-in) |
| **Config file** | Cargo.toml (existing) |
| **Quick run command** | `cargo test --lib --features simd` |
| **Full suite command** | `cargo test --features simd` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test --lib --features simd`
- **After every plan wave:** Run `cargo test --features simd`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Wave 0 Assessment

No separate Wave 0 plan is needed. Justification:

1. **Test framework exists:** `cargo test` is Rust's built-in test harness, already configured in Cargo.toml with dev-dependencies (criterion, rand).
2. **Test infrastructure exists:** Phase 1 established 42+ tests across all modules. The `src/backend/` module already has tests in `scalar.rs`.
3. **New tests are co-located:** Plan 02-01 creates `src/backend/simd.rs` with inline `#[cfg(test)] mod tests` containing 8 unit tests. Tests are created in the same task as the code they verify -- no separate test scaffold needed.
4. **Verify commands use existing infrastructure:** All `<automated>` verify commands use `cargo test --lib --features simd` which works as soon as the simd feature is added to Cargo.toml (Task 1 of Plan 02-01).

The existing `cargo test` infrastructure is sufficient. No MISSING Wave 0 references exist.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 02-01-T1 | 01 | 1 | SIMD-03,SIMD-05,SIMD-06 | unit | `cargo test --lib --features simd && cargo test --lib` | Created in task | pending |
| 02-01-T2 | 01 | 1 | API-01,TEST-01 | integration | `cargo test --features simd && cargo test` | Existing | pending |
| 02-02-T1 | 02 | 2 | SIMD-01,SIMD-02,SIMD-04,SIMD-07 | unit | `cargo test --lib --features simd && cargo test --lib` | Created in prior task | pending |
| 02-02-T2 | 02 | 2 | TEST-02,PERF-01 | unit | `cargo test --lib --features simd && cargo test --lib` | Created in task | pending |
| 02-03-T1 | 03 | 3 | SIMD-08 | unit | `cargo test --lib backend::simd --features simd` | Created in task | pending |
| 02-03-T2 | 03 | 3 | SIMD-09,TEST-03 | bench | `cargo bench --bench integration --features simd -- --test && cargo bench --bench integration -- --test` | Modified in task | pending |

*Status: pending -- awaiting execution*

---

## Sampling Continuity

No 3 consecutive tasks without automated verify. Every task has an `<automated>` verify command:
- 02-01-T1: `cargo test --lib --features simd` (~5s)
- 02-01-T2: `cargo test --features simd && cargo test` (~10s)
- 02-02-T1: `cargo test --lib --features simd` (~5s)
- 02-02-T2: `cargo test --lib --features simd` (~5s)
- 02-03-T1: `cargo test --lib backend::simd --features simd` (~3s)
- 02-03-T2: `cargo bench --bench integration --features simd -- --test` (~5s)

All commands execute in <15 seconds. Wave 3 tasks (02-03-T1 and 02-03-T2) are verification-heavy by design (Miri + benchmarks), but automated commands remain fast because:
- T1 runs unit tests only (not Miri itself -- Miri is attempted as a diagnostic, not a gate)
- T2 uses `-- --test` flag which compiles but does not run full benchmarks

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| AVX2 2-4x speedup | SIMD-09 | Requires x86_64 hardware | Run `cargo bench --features simd` on AVX2-capable CPU, compare FWHT benchmarks to Phase 1 baseline |
| NEON 2-4x speedup | SIMD-09 | Requires ARM hardware | Run `cargo bench --features simd` on ARM CPU with NEON, compare FWHT benchmarks to Phase 1 baseline |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (no MISSING references exist)
- [x] No watch-mode flags
- [x] Feedback latency < 10s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** validated
