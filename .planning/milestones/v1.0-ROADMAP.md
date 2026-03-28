# Roadmap: TurboQuant Rust - Production Optimization

**Project:** TurboQuant Rust
**Milestone:** v1.0 - Production Optimization
**Created:** 2026-03-27
**Granularity:** Standard
**Total Phases:** 4

## Phases

- [x] **Phase 1: Foundation & Quick Wins** - Establish backend architecture and eliminate allocation overhead (completed 2026-03-27)
- [x] **Phase 2: SIMD Acceleration** - Vectorize FWHT operations for 2-4x speedup on CPU (completed 2026-03-27)
- [x] **Phase 3: Batch Operations** - Enable multi-vector processing for throughput optimization (completed 2026-03-28)
- [x] **Phase 4: GPU Backend** - CUDA acceleration for large-batch inference workloads (completed 2026-03-28)

## Phase Details

### Phase 1: Foundation & Quick Wins
**Goal:** Establish performance foundation through backend abstraction, API safety improvements, and allocation optimization

**Depends on:** Nothing (first phase)

**Requirements:** FOUND-01, FOUND-02, FOUND-03, FOUND-04, FOUND-05, FOUND-06, FOUND-07, FOUND-08

**Success Criteria** (what must be TRUE):
  1. User can quantize vectors without triggering panics in bitpack operations (Result-based error handling)
  2. User's incorrect API usage produces compile-time warnings for ignored return values (#[must_use])
  3. User experiences 1.5-2x faster inner_product operations through scratch buffer reuse (measured with benchmarks)
  4. User can switch between backend implementations (scalar vs future SIMD/GPU) without API changes
  5. User receives clear error messages for invalid inputs (power-of-two dimension validation)

**Plans:** 4/4 plans complete

Plans:
- [ ] 01-01-PLAN.md — API safety fixes (panic to Result, #[must_use], release assertions)
- [ ] 01-02-PLAN.md — Integration benchmarks for realistic attention workloads
- [ ] 01-03-PLAN.md — Backend trait abstraction with ScalarBackend extraction
- [ ] 01-04-PLAN.md — Scratch buffer reuse in PolarQuant::inner_product()

---

### Phase 2: SIMD Acceleration
**Goal:** Accelerate FWHT operations through AVX2/NEON vectorization with runtime CPU detection

**Depends on:** Phase 1 (requires Backend trait)

**Requirements:** SIMD-01, SIMD-02, SIMD-03, SIMD-04, SIMD-05, SIMD-06, SIMD-07, SIMD-08, SIMD-09

**Success Criteria** (what must be TRUE):
  1. User experiences 2-4x faster FWHT operations on x86_64 CPUs with AVX2 (benchmarked)
  2. User experiences 2-4x faster FWHT operations on ARM CPUs with NEON (benchmarked)
  3. User's application automatically uses SIMD when available and falls back to scalar on older CPUs (runtime detection)
  4. User compiles with `--features simd` and gets vectorized implementations, or without flag and gets scalar only (compile-time selection)
  5. User sees identical quantization results between SIMD and scalar backends (within f32 precision, verified by tests)

**Plans:** 3/3 plans complete

Plans:
- [x] 02-01-PLAN.md — Feature flag, SimdBackend scaffold, RuntimeBackend with CPU detection
- [x] 02-02-PLAN.md — AVX2/NEON FWHT and dot product intrinsics with SAFETY docs
- [x] 02-03-PLAN.md — Miri verification, correctness sweep, benchmark comparison
- [ ] 02-04-PLAN.md — Integration testing and validation

---

### Phase 3: Batch Operations
**Goal:** Enable efficient multi-vector processing through batch APIs with CPU parallelization

**Depends on:** Phase 2 (benefits from SIMD, validates API before GPU)

**Requirements:** BATCH-01, BATCH-02, BATCH-03, BATCH-04, BATCH-05, BATCH-06, BATCH-07

**Success Criteria** (what must be TRUE):
  1. User can quantize multiple vectors in a single call via batch_quantize(&[Vec<f32>]) API
  2. User can compute attention scores for multiple queries via batch_inner_product and batch_attend APIs
  3. User processing single vectors through batch API experiences zero performance regression (batch-of-1 matches single-vector API)
  4. User processing 64-vector batches experiences measurable throughput improvement over 64 sequential single-vector calls (benchmarked)
  5. User's batch operations leverage CPU parallelism automatically (rayon integration, multi-core utilization)

**Plans:** 3/3 plans complete

Plans:
- [x] 03-01-PLAN.md — Rayon dependency + batch APIs on PolarQuant (batch_quantize, batch_inner_product)
- [x] 03-02-PLAN.md — Clone for TurboQuant/KvCache + batch_attend on KvCache
- [x] 03-03-PLAN.md — Batch-of-1 regression and batch-of-64 throughput benchmarks

---

### Phase 4: GPU Backend
**Goal:** Accelerate large-batch inference through CUDA kernels with intelligent CPU/GPU dispatch

**Depends on:** Phase 3 (requires batch API infrastructure)

**Requirements:** GPU-01, GPU-02, GPU-03, GPU-04, GPU-05, GPU-06, GPU-07, GPU-08, GPU-09, GPU-10, PERF-02, DOC-01, DOC-02

**Success Criteria** (what must be TRUE):
  1. User compiles with `--features gpu` and gains access to CUDA-accelerated batch operations (feature flag)
  2. User processing batches >=32 on GPU experiences faster execution than equivalent CPU batch (benchmarked end-to-end)
  3. User processing batches <32 automatically uses CPU path without manual configuration (threshold-based dispatch)
  4. User without CUDA toolkit receives clear error messages explaining GPU unavailable and how to resolve
  5. User measures 3-8x combined speedup on realistic attention hot path compared to Phase 0 baseline (integration benchmark validation)
  6. User accesses performance guide with benchmark results explaining when to use GPU vs CPU (documentation)

**Plans:** 5/5 plans complete

Plans:
- [x] 04-01-PLAN.md — Feature flag + cudarc dependency + GpuBackend scaffold + error variants
- [x] 04-02-PLAN.md — CUDA kernels (FWHT + attention) + build.rs PTX compilation
- [x] 04-03-PLAN.md — GPU memory pooling + batch dispatch integration
- [ ] 04-04-PLAN.md — GPU benchmarks + combined Phase 1-4 speedup validation
- [ ] 04-05-PLAN.md — Feature flags and performance guide documentation

---

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation & Quick Wins | 4/4 | Complete    | 2026-03-27 |
| 2. SIMD Acceleration | 3/3 | Complete    | 2026-03-27 |
| 3. Batch Operations | 3/3 | Complete    | 2026-03-28 |
| 4. GPU Backend | 5/5 | Complete    | 2026-03-28 |

**Overall:** 75% complete (3/4 phases, Phase 4 at 60%)

---

## Cross-Cutting Requirements

These requirements apply to all phases or specific checkpoints:

| Requirement | Applies To | Validation |
|-------------|------------|------------|
| TEST-01 | All phases | All 35 existing tests pass after each phase |
| TEST-02 | Phases 2, 3, 4 | New backends have correctness tests matching scalar |
| TEST-03 | Phases 2, 3, 4 | Performance regression tests prevent slowdowns |
| PERF-01 | All phases | <2% inner product error maintained (accuracy invariant) |
| PERF-02 | Phase 4 completion | 3-8x combined speedup demonstrated |
| API-01 | All phases | Existing API remains backward compatible |
| DOC-01 | Phase 4 | Feature flags and backend selection documented |
| DOC-02 | Phase 4 | Performance guide with benchmark results published |

---

## Dependencies

```
Phase 1 (Foundation)
   |
Phase 2 (SIMD) <-- Can proceed independently
   |                     |
Phase 3 (Batch) <--------+
   |
Phase 4 (GPU)
```

**Critical path:** Phase 1 must complete before any other phase (establishes Backend trait). Phase 3 must complete before Phase 4 (GPU requires batch infrastructure). Phase 2 can proceed after Phase 1 independently of Phase 3.

---

## Coverage Validation

**Total v1 requirements:** 45
**Mapped to phases:** 45
**Unmapped:** 0

| Category | Count | Phase Assignment |
|----------|-------|------------------|
| FOUND | 8 | Phase 1 |
| SIMD | 9 | Phase 2 |
| BATCH | 7 | Phase 3 |
| GPU | 10 | Phase 4 |
| Cross-cutting | 11 | All phases / Phase 4 completion |

**Coverage:** 100%

---

*Last updated: 2026-03-28*
