# Requirements: TurboQuant Rust - Production Optimization

**Defined:** 2026-03-27
**Core Value:** Achieve 3-8x performance improvement through SIMD, allocation optimization, and batch processing while maintaining algorithm correctness

## v1 Requirements

Requirements for production-ready optimization milestone. Each maps to roadmap phases.

### Foundation (Phase 1)

- [x] **FOUND-01**: Replace panic! with Result in bitpack::pack() and bitpack::unpack()
- [x] **FOUND-02**: Add #[must_use] attributes to all functions returning computed values
- [x] **FOUND-03**: Add power-of-two assertion to fwht_inplace() in release builds
- [ ] **FOUND-04**: Define Backend trait for CPU/SIMD/GPU abstraction
- [ ] **FOUND-05**: Extract ScalarBackend implementing Backend trait
- [ ] **FOUND-06**: Refactor PolarQuant to use Backend trait with static dispatch
- [ ] **FOUND-07**: Add scratch buffer reuse to PolarQuant::inner_product()
- [x] **FOUND-08**: Add integration benchmarks for realistic workloads

### SIMD Acceleration (Phase 2)

- [ ] **SIMD-01**: Implement SimdBackend with AVX2 intrinsics for x86_64
- [ ] **SIMD-02**: Implement SimdBackend with NEON intrinsics for ARM
- [ ] **SIMD-03**: Add runtime CPU feature detection (is_x86_feature_detected!)
- [ ] **SIMD-04**: Implement SIMD FWHT butterfly operations
- [ ] **SIMD-05**: Add automatic fallback to scalar when SIMD unavailable
- [ ] **SIMD-06**: Add feature flag `simd` for compile-time backend selection
- [ ] **SIMD-07**: Document SAFETY requirements for all unsafe SIMD code
- [ ] **SIMD-08**: Verify SIMD correctness with Miri on test suite
- [ ] **SIMD-09**: Achieve 2-4x speedup on FWHT operations (benchmarked)

### Batch Operations (Phase 3)

- [ ] **BATCH-01**: Add batch_quantize(&[Vec<f32>]) API to PolarQuant
- [ ] **BATCH-02**: Add batch_inner_product(query, &[QuantizedVector]) API
- [ ] **BATCH-03**: Add batch_attend(query) to KvCache for multi-query attention
- [ ] **BATCH-04**: Implement zero-copy batch patterns (contiguous memory layout)
- [ ] **BATCH-05**: Add parallel CPU batch processing with rayon
- [ ] **BATCH-06**: Verify batch-of-1 performance matches single-vector API
- [ ] **BATCH-07**: Demonstrate batch-of-64 performance improvement

### GPU Backend (Phase 4)

- [ ] **GPU-01**: Integrate cudarc for CUDA device management
- [ ] **GPU-02**: Implement GpuBackend with CUDA stream management
- [ ] **GPU-03**: Write CUDA kernel for FWHT butterfly operations
- [ ] **GPU-04**: Write CUDA kernel for batch attention logits
- [ ] **GPU-05**: Implement GPU memory pooling for batch buffers
- [ ] **GPU-06**: Add batch size threshold for CPU vs GPU dispatch (≥32)
- [ ] **GPU-07**: Add feature flag `gpu` for optional CUDA support
- [ ] **GPU-08**: Handle GPU unavailable errors gracefully with clear messages
- [ ] **GPU-09**: Verify GPU batch ≥32 faster than CPU batch
- [ ] **GPU-10**: Verify GPU batch <32 automatically uses CPU fallback

### Cross-Cutting (All Phases)

- [ ] **TEST-01**: All 35 existing tests pass after each phase
- [ ] **TEST-02**: Add correctness tests for each new backend
- [ ] **TEST-03**: Add performance regression tests
- [ ] **PERF-01**: Achieve <2% inner product error (maintain accuracy)
- [ ] **PERF-02**: Demonstrate 3-8x combined speedup on attention hot path
- [ ] **API-01**: Maintain backward compatibility (existing API unchanged)
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
| FOUND-04 | Phase 1 | Pending |
| FOUND-05 | Phase 1 | Pending |
| FOUND-06 | Phase 1 | Pending |
| FOUND-07 | Phase 1 | Pending |
| FOUND-08 | Phase 1 | Complete |
| SIMD-01 | Phase 2 | Pending |
| SIMD-02 | Phase 2 | Pending |
| SIMD-03 | Phase 2 | Pending |
| SIMD-04 | Phase 2 | Pending |
| SIMD-05 | Phase 2 | Pending |
| SIMD-06 | Phase 2 | Pending |
| SIMD-07 | Phase 2 | Pending |
| SIMD-08 | Phase 2 | Pending |
| SIMD-09 | Phase 2 | Pending |
| BATCH-01 | Phase 3 | Pending |
| BATCH-02 | Phase 3 | Pending |
| BATCH-03 | Phase 3 | Pending |
| BATCH-04 | Phase 3 | Pending |
| BATCH-05 | Phase 3 | Pending |
| BATCH-06 | Phase 3 | Pending |
| BATCH-07 | Phase 3 | Pending |
| GPU-01 | Phase 4 | Pending |
| GPU-02 | Phase 4 | Pending |
| GPU-03 | Phase 4 | Pending |
| GPU-04 | Phase 4 | Pending |
| GPU-05 | Phase 4 | Pending |
| GPU-06 | Phase 4 | Pending |
| GPU-07 | Phase 4 | Pending |
| GPU-08 | Phase 4 | Pending |
| GPU-09 | Phase 4 | Pending |
| GPU-10 | Phase 4 | Pending |
| TEST-01 | All Phases | Pending |
| TEST-02 | All Phases | Pending |
| TEST-03 | All Phases | Pending |
| PERF-01 | All Phases | Pending |
| PERF-02 | Phase 4 | Pending |
| API-01 | All Phases | Pending |
| DOC-01 | Phase 4 | Pending |
| DOC-02 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 45 total
- Mapped to phases: 45
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-27*
*Last updated: 2026-03-27 after initial definition*
