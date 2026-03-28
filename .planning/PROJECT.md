# TurboQuant Rust

## What This Is

A production-ready pure-Rust implementation of Google Research's TurboQuant (ICLR 2026) - an LLM KV-cache compression algorithm that achieves 6-8x memory reduction with near-zero accuracy loss. Features three performance backends (Scalar, SIMD, GPU) with intelligent dispatch, comprehensive test coverage (57 tests passing), and complete documentation.

## Core Value

**Achieve 3-8x performance improvement through SIMD, allocation optimization, and batch processing while maintaining the algorithm's theoretical guarantees.**

If everything else fails, the core quantization algorithm must remain mathematically correct and produce accurate compressed representations.

## Requirements

### Validated

<!-- Core algorithm (existing implementation) -->

- ✓ PolarQuant Stage 1 compression (rotation + Lloyd-Max quantization) — v1.0
- ✓ QJL Stage 2 residual correction (1-bit JL transform) — v1.0
- ✓ KV-cache API for transformer attention — v1.0
- ✓ Fast inner product computation (no full decompression) — v1.0
- ✓ 2/3/4-bit quantization support — v1.0
- ✓ Comprehensive test suite (57 tests, all passing) — v1.0
- ✓ Zero external dependencies for core (thiserror only) — v1.0
- ✓ Proper error handling with Result types — v1.0

<!-- Performance optimizations (v1.0 milestone) -->

- ✓ Backend trait abstraction for polymorphic acceleration — v1.0
- ✓ SIMD acceleration (AVX2/NEON) with runtime CPU detection — v1.0 (2-3x speedup)
- ✓ Batch processing APIs with CPU parallelization — v1.0
- ✓ GPU backend with CUDA kernels and intelligent dispatch — v1.0 (5-10x speedup for large batches)
- ✓ Feature flags for optional backends (simd, gpu) — v1.0
- ✓ Allocation optimization and scratch buffer reuse — v1.0
- ✓ Combined 3-8x speedup validated through benchmarks — v1.0

### Active

<!-- Future enhancements for v2.0 or later -->

(To be defined in next milestone planning)

### Out of Scope

- ROCm/AMD GPU support — CUDA first, AMD later if demand exists
- Distributed inference — single-GPU focus for v1
- Model-specific integrations (llama.cpp, candle) — standalone library for now
- Auto-tuning/adaptive bit-width — fixed 2/3/4-bit is sufficient
- Quantization of attention weights — KV-cache only per paper
- Support for non-power-of-two dimensions — FWHT requirement is fundamental

## Context

**v1.0 Shipped (2026-03-28):**
- Production-ready implementation with 3 backends: Scalar (baseline), SIMD (2-3x), GPU (5-10x on large batches)
- ~3,438 LOC pure Rust across 14 source files
- Tech stack: Rust 1.x, std::arch (SIMD), cudarc 0.12 (GPU), thiserror (errors), criterion (benchmarks)
- 57 tests passing (unit + integration + benchmarks)
- Complete documentation: feature flags guide, performance guide with benchmark methodology
- Validated 3-8x combined speedup target on attention hot path

**Architecture:**
- Backend trait enabling zero-cost polymorphic acceleration via static dispatch
- Optional feature flags: `simd` (AVX2/NEON), `gpu` (CUDA)
- Intelligent dispatch: GPU batch threshold at 32 vectors (PCIe transfer amortization)
- Memory pooling: Per-size buffer reuse to amortize cudaMalloc overhead
- Clean module separation: `polar_quant`, `qjl`, `turboquant`, `kv_cache`, `rotation`, `hadamard`, `codebook`, `bitpack`, `backend`

**ML Domain Context:**
- Target: LLM inference optimization (reduce KV-cache memory by 6-8x)
- Hot path: FWHT called 2N times per attention (N=sequence length)
- Performance validated: 2-3x SIMD speedup, 5-10x GPU speedup for batches ≥32
- Dimensions: 64, 128, 256 (power-of-two requirement for FWHT)

**Research Background:**
- Google Research ICLR 2026 paper
- Two-stage: PolarQuant (MSE-optimal) + QJL (unbiased correction)
- Community validation: MSE variant matches/beats Prod variant
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
| Backend trait with static dispatch | Zero-cost abstraction via monomorphization | ✓ Good - enables polymorphic acceleration without runtime overhead |
| std::arch for SIMD (not external crates) | Zero dependencies, stable since Rust 1.27, portable | ✓ Good - clean implementation, 2-3x speedup achieved |
| cudarc for CUDA integration | Most mature Rust CUDA wrapper as of 2026 | ✓ Good - safe RAII memory management, easy PTX loading |
| Phase order: Foundation → SIMD → Batch → GPU | Dependencies + risk mitigation + incremental value | ✓ Good - each phase validated before building on it |
| GPU batch threshold ≥32 | Amortizes memory transfer overhead per research | ✓ Good - validates through benchmarks, intelligent dispatch working |
| Feature flags for backends (simd, gpu) | Optional acceleration, compile-time selection | ✓ Good - zero impact on non-accelerated builds |
| MSE variant primary | Community validation shows MSE matches Prod quality, simpler | ✓ Good - implementation clean, quality validated |
| Memory pooling for GPU buffers | Amortize ~100μs cudaMalloc overhead | ✓ Good - per-size pools enable buffer reuse |
| FWHT assertion in release mode | Negligible cost vs O(n log n), prevents undefined behavior | ✓ Good - safety without meaningful performance cost |

---
*Last updated: 2026-03-28 after v1.0 milestone completion*
