# TurboQuant Rust - Production Optimization

## What This Is

A pure-Rust implementation of Google Research's TurboQuant (ICLR 2026) - an LLM KV-cache compression algorithm that achieves 6-8x memory reduction with near-zero accuracy loss. Currently feature-complete with excellent test coverage (35/35 passing), but missing critical performance optimizations and production-ready features needed for real-world ML inference workloads.

## Core Value

**Achieve 3-8x performance improvement through SIMD, allocation optimization, and batch processing while maintaining the algorithm's theoretical guarantees.**

If everything else fails, the core quantization algorithm must remain mathematically correct and produce accurate compressed representations.

## Requirements

### Validated

<!-- Existing working implementation from codebase analysis -->

- ✓ PolarQuant Stage 1 compression (rotation + Lloyd-Max quantization) — existing
- ✓ QJL Stage 2 residual correction (1-bit JL transform) — existing
- ✓ KV-cache API for transformer attention — existing
- ✓ Fast inner product computation (no full decompression) — existing
- ✓ 2/3/4-bit quantization support — existing
- ✓ Comprehensive test suite (35 tests, all passing) — existing
- ✓ Zero external dependencies (except thiserror) — existing
- ✓ Proper error handling with Result types (mostly) — existing

### Active

<!-- New requirements for this milestone -->

- [ ] **PERF-01**: Replace panic! with Result in bitpack.rs public API
- [ ] **PERF-02**: Add #[must_use] attributes to prevent silent bugs
- [ ] **PERF-03**: SIMD acceleration for FWHT (AVX2 on x86_64, NEON on ARM)
- [ ] **PERF-04**: Scratch buffer reuse in inner_product hot path
- [ ] **PERF-05**: Pre-allocation in codebook operations
- [ ] **PERF-06**: Batch quantization API (process multiple vectors at once)
- [ ] **PERF-07**: Batch inner product API (multiple queries against cache)
- [ ] **PERF-08**: CUDA kernels for FWHT butterfly operations
- [ ] **PERF-09**: CUDA kernels for batch attention computation
- [ ] **PERF-10**: Feature flags for SIMD/GPU backends

### Out of Scope

- ROCm/AMD GPU support — CUDA first, AMD later if demand exists
- Distributed inference — single-GPU focus for v1
- Model-specific integrations (llama.cpp, candle) — standalone library for now
- Auto-tuning/adaptive bit-width — fixed 2/3/4-bit is sufficient
- Quantization of attention weights — KV-cache only per paper
- Support for non-power-of-two dimensions — FWHT requirement is fundamental

## Context

**Existing Implementation:**
- Clean modular architecture: `polar_quant`, `qjl`, `turboquant`, `kv_cache`, `rotation`, `hadamard`, `codebook`, `bitpack`
- Comprehensive ASSESSMENT.md identifies optimization opportunities with projected speedups
- All tests passing, good error handling (except bitpack.rs panics)
- Performance headroom: ASSESSMENT.md projects 3-8x combined speedup from optimizations

**ML Domain Context:**
- Target use case: LLM inference optimization (reduce KV-cache memory by 6-8x)
- Hot path: FWHT called 2N times per attention (N=sequence length)
- For 8192 tokens: ~2.1M FWHT operations per attention pass
- Memory bottleneck: Allocation in inner_product (4MB for 8192-token sequence)
- GPU utilization: Batch operations needed for efficient GPU use

**Research Background:**
- Google Research ICLR 2026 paper
- Two-stage: PolarQuant (MSE-optimal) + QJL (unbiased correction)
- Community validation: MSE variant often matches/beats Prod variant
- Power-of-two dimensions required (FWHT constraint)

## Constraints

- **Architecture**: Must maintain power-of-two dimension requirement (fundamental to FWHT)
- **Compatibility**: Existing API must remain backward compatible
- **Safety**: No unsafe code in public API surface (keep isolated to SIMD intrinsics)
- **Dependencies**: Minimize new dependencies (cudarc for GPU, std::arch for SIMD only)
- **Testing**: All existing tests must pass, new features need equivalent coverage
- **Performance**: Optimizations must not degrade accuracy (maintain <2% inner product error)
- **Platform**: x86_64 and ARM support for SIMD, CUDA 11.0+ for GPU

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| CUDA before ROCm | Larger market share, better tooling, can add ROCm later | — Pending |
| Feature flags for backends | Allow compile-time backend selection, reduce binary size | — Pending |
| Backward-compatible API | Add batch APIs alongside single-vector APIs | — Pending |
| Unsafe isolation | Keep unsafe code in dedicated SIMD/GPU modules with safety docs | — Pending |
| MSE variant primary | Community validation shows MSE matches Prod quality, simpler | ✓ Good (from existing impl) |
| Zero dependencies (core) | Portability and simplicity over convenience | ✓ Good (from existing impl) |

---
*Last updated: 2026-03-27 after project initialization*
