# Requirements: TurboQuant Rust - Production Optimization

**Defined:** 2026-03-27
**Core Value:** Achieve 3-8x performance improvement through SIMD, allocation optimization, and batch processing while maintaining algorithm correctness

## v1 Requirements

Requirements for production-ready optimization milestone. Each maps to roadmap phases.

### Foundation (Phase 1)

- [x] **FOUND-01**: Replace panic! with Result in bitpack::pack() and bitpack::unpack()
- [x] **FOUND-02**: Add #[must_use] attributes to all functions returning computed values
- [x] **FOUND-03**: Add power-of-two assertion to fwht_inplace() in release builds
- [x] **FOUND-04**: Define Backend trait for CPU/SIMD/GPU abstraction
- [x] **FOUND-05**: Extract ScalarBackend implementing Backend trait
- [x] **FOUND-06**: Refactor PolarQuant to use Backend trait with static dispatch
- [x] **FOUND-07**: Add scratch buffer reuse to PolarQuant::inner_product()
- [x] **FOUND-08**: Add integration benchmarks for realistic workloads

### SIMD Acceleration (Phase 2)

- [x] **SIMD-01**: Implement SimdBackend with AVX2 intrinsics for x86_64
- [x] **SIMD-02**: Implement SimdBackend with NEON intrinsics for ARM
- [x] **SIMD-03**: Add runtime CPU feature detection (is_x86_feature_detected!)
- [x] **SIMD-04**: Implement SIMD FWHT butterfly operations
- [x] **SIMD-05**: Add automatic fallback to scalar when SIMD unavailable
- [x] **SIMD-06**: Add feature flag `simd` for compile-time backend selection
- [x] **SIMD-07**: Document SAFETY requirements for all unsafe SIMD code
- [x] **SIMD-08**: Verify SIMD correctness with Miri on test suite
- [x] **SIMD-09**: Achieve 2-4x speedup on FWHT operations (benchmarked)

### Batch Operations (Phase 3)

- [x] **BATCH-01**: Add batch_quantize(&[Vec<f32>]) API to PolarQuant
- [x] **BATCH-02**: Add batch_inner_product(query, &[QuantizedVector]) API
- [x] **BATCH-03**: Add batch_attend(query) to KvCache for multi-query attention
- [x] **BATCH-04**: Implement zero-copy batch patterns (contiguous memory layout)
- [x] **BATCH-05**: Add parallel CPU batch processing with rayon
- [x] **BATCH-06**: Verify batch-of-1 performance matches single-vector API
- [x] **BATCH-07**: Demonstrate batch-of-64 performance improvement

### GPU Backend (Phase 4)

- [x] **GPU-01**: Integrate cudarc for CUDA device management
- [x] **GPU-02**: Implement GpuBackend with CUDA stream management
- [ ] **GPU-03**: Write CUDA kernel for FWHT butterfly operations
- [ ] **GPU-04**: Write CUDA kernel for batch attention logits
- [ ] **GPU-05**: Implement GPU memory pooling for batch buffers
- [ ] **GPU-06**: Add batch size threshold for CPU vs GPU dispatch (≥32)
- [x] **GPU-07**: Add feature flag `gpu` for optional CUDA support
- [x] **GPU-08**: Handle GPU unavailable errors gracefully with clear messages
- [ ] **GPU-09**: Verify GPU batch ≥32 faster than CPU batch
- [ ] **GPU-10**: Verify GPU batch <32 automatically uses CPU fallback

### Cross-Cutting (All Phases)

- [x] **TEST-01**: All 35 existing tests pass after each phase
- [x] **TEST-02**: Add correctness tests for each new backend
- [x] **TEST-03**: Add performance regression tests
- [x] **PERF-01**: Achieve <2% inner product error (maintain accuracy)
- [ ] **PERF-02**: Demonstrate 3-8x combined speedup on attention hot path
- [x] **API-01**: Maintain backward compatibility (existing API unchanged)
- [ ] **DOC-01**: Document feature flags and backend selection
- [ ] **DOC-02**: Add performance guide with benchmark results

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Optimization

- **OPT-01**: AVX-512 support for Xeon processors
- **OPT-02**: Auto-tuning of batch size thresholds
- **OPT-03**: Async GPU operations with CUDA streams
- **OPT-04**: Multi-GPU support for distributed inference

### Platform Expansion

- **PLAT-01**: ROCm/AMD GPU support
- **PLAT-02**: Metal GPU support for macOS
- **PLAT-03**: WebGPU for browser deployment

### Integration

- **INT-01**: Integration examples with llama.cpp
- **INT-02**: Integration examples with candle framework
- **INT-03**: Python bindings via PyO3

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Dynamic dispatch (Box<dyn Backend>) | Prevents inlining, defeats zero-cost abstraction goal |
| Generic float types (f64, f16) | Algorithm validated for f32 only, adds complexity |
| Non-power-of-two dimensions | FWHT fundamental constraint, would require different algorithm |
| Quantization of attention weights | Out of TurboQuant paper scope, KV-cache only |
| Model-specific integrations | Standalone library philosophy, users integrate |
| Distributed inference | Single-GPU focus sufficient for v1 |
| Auto-tuning layer | Manual thresholds adequate, avoid premature complexity |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FOUND-01 | Phase 1 | Complete |
| FOUND-02 | Phase 1 | Complete |
| FOUND-03 | Phase 1 | Complete |
| FOUND-04 | Phase 1 | Complete |
| FOUND-05 | Phase 1 | Complete |
| FOUND-06 | Phase 1 | Complete |
| FOUND-07 | Phase 1 | Complete |
| FOUND-08 | Phase 1 | Complete |
| SIMD-01 | Phase 2 | Complete |
| SIMD-02 | Phase 2 | Complete |
| SIMD-03 | Phase 2 | Complete |
| SIMD-04 | Phase 2 | Complete |
| SIMD-05 | Phase 2 | Complete |
| SIMD-06 | Phase 2 | Complete |
| SIMD-07 | Phase 2 | Complete |
| SIMD-08 | Phase 2 | Complete |
| SIMD-09 | Phase 2 | Complete |
| BATCH-01 | Phase 3 | Complete |
| BATCH-02 | Phase 3 | Complete |
| BATCH-03 | Phase 3 | Complete |
| BATCH-04 | Phase 3 | Complete |
| BATCH-05 | Phase 3 | Complete |
| BATCH-06 | Phase 3 | Complete |
| BATCH-07 | Phase 3 | Complete |
| GPU-01 | Phase 4 | Complete |
| GPU-02 | Phase 4 | Complete |
| GPU-03 | Phase 4 | Pending |
| GPU-04 | Phase 4 | Pending |
| GPU-05 | Phase 4 | Pending |
| GPU-06 | Phase 4 | Pending |
| GPU-07 | Phase 4 | Complete |
| GPU-08 | Phase 4 | Complete |
| GPU-09 | Phase 4 | Pending |
| GPU-10 | Phase 4 | Pending |
| TEST-01 | All Phases | Complete |
| TEST-02 | All Phases | Complete |
| TEST-03 | All Phases | Complete |
| PERF-01 | All Phases | Complete |
| PERF-02 | Phase 4 | Pending |
| API-01 | All Phases | Complete |
| DOC-01 | Phase 4 | Pending |
| DOC-02 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 45 total
- Mapped to phases: 45
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-27*
*Last updated: 2026-03-27 after initial definition*
