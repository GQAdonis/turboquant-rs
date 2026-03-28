# Performance Guide

This guide explains TurboQuant's performance characteristics, how to run benchmarks, and when to use each backend.

## Quick Decision Guide

| Your Workload | Recommended Backend | Feature Flag |
|---------------|-------------------|--------------|
| Single-vector compression/attention | ScalarBackend or SimdBackend | `simd` |
| Batch <= 31 vectors | SimdBackend (fastest CPU) | `simd` |
| Batch >= 32 vectors, have NVIDIA GPU | GpuBackend | `gpu` |
| Batch >= 32 vectors, no GPU | SimdBackend + rayon | `simd` |
| Maximum performance, GPU available | SimdBackend + GpuBackend | `simd,gpu` |

## Performance Architecture

TurboQuant's performance comes from four layers of optimization:

### Layer 1: Allocation Optimization (Phase 1)
- Scratch buffer reuse in `inner_product()` eliminates per-call Vec allocation
- Impact: ~1.5-2x on repeated inner product calls
- Applies to: All backends

### Layer 2: SIMD Vectorization (Phase 2)
- AVX2 (8-wide f32) on x86_64, NEON (4-wide f32) on aarch64
- Accelerates FWHT butterfly operations and dot products
- Impact: 2-4x on FWHT operations
- Applies to: `SimdBackend` and `RuntimeBackend`

### Layer 3: Batch Parallelism (Phase 3)
- Rayon work-stealing parallelism for multi-vector operations
- Automatic thread pool management
- Impact: Near-linear scaling with CPU cores for independent operations
- Applies to: All backends via `batch_quantize`, `batch_inner_product`, `batch_attend`

### Layer 4: GPU Acceleration (Phase 4)
- CUDA kernels for FWHT and attention logit computation
- GPU memory pooling to amortize allocation overhead
- Automatic CPU/GPU dispatch based on batch size threshold
- Impact: 10x+ for batch >= 32 on modern NVIDIA GPUs
- Applies to: `GpuBackend` with `gpu` feature

### Combined Impact

On the attention hot path (inner product computation):
- Phase 0 baseline (scalar, sequential): 1x
- Phase 1 (scratch buffers): ~1.5-2x
- Phase 2 (SIMD): ~3-4x
- Phase 3 (batch + rayon): ~3-6x (depends on core count)
- Phase 4 (GPU, batch >= 32): ~10-30x (depends on GPU and batch size)

Target: **3-8x combined speedup** for typical inference workloads on CPU, **10x+** with GPU.

## Batch Size Threshold

The `GPU_BATCH_THRESHOLD` constant is set to **32 vectors**. Below this threshold, GPU transfer overhead (10-50 microseconds per copy) exceeds the compute benefit.

    // Automatic dispatch in PolarQuant<GpuBackend>:
    // batch_size >= 32 -> GPU kernels
    // batch_size < 32  -> CPU (rayon + SIMD)
    let results = pq.batch_quantize_dispatch(&vectors)?;

Factors affecting the threshold:
- **GPU generation**: Newer GPUs (Ampere/Hopper) have faster PCIe, lower threshold
- **Vector dimension**: Larger dimensions (256+) amortize transfer cost better
- **Batch size**: GPU advantage increases with batch size (GPU throughput scales)

## Running Benchmarks

### Full Benchmark Suite

    # CPU-only benchmarks
    cargo bench --release

    # With SIMD
    cargo bench --release --features simd

    # With GPU (requires NVIDIA GPU + CUDA)
    cargo bench --release --features gpu --bench gpu_bench

    # All features
    cargo bench --release --features simd,gpu

### Specific Benchmark Groups

    # FWHT operation benchmarks
    cargo bench --release --features simd --bench bench -- fwht

    # Integration benchmarks (realistic workloads)
    cargo bench --release --bench integration

    # GPU vs CPU comparison
    cargo bench --release --features gpu --bench gpu_bench -- batch_quantize

    # Combined speedup (Phase 0 vs Phase 4)
    cargo bench --release --features gpu --bench integration -- combined_speedup

### Interpreting Results

Criterion generates HTML reports in `target/criterion/`. Key metrics:
- **Mean time**: Average execution time per iteration
- **Throughput**: Operations per second (if configured)
- **Change**: Percentage change from previous run (+ means slower, - means faster)

For GPU benchmarks, all times are **end-to-end** including:
- Host-to-device memory copy
- Kernel execution
- Device-to-host memory copy

This ensures realistic performance comparison with CPU.

## Memory Considerations

### Compressed Cache Size

At 3-bit quantization, 128-dimensional vectors:
- Per token: 52 bytes (key) + 52 bytes (value) = 104 bytes
- Uncompressed equivalent: 512 + 512 = 1024 bytes
- Compression ratio: ~9.8x

| Sequence Length | Compressed | Uncompressed | Savings |
|----------------|-----------|--------------|---------|
| 128 tokens | ~13 KB | ~128 KB | 115 KB |
| 2048 tokens | ~208 KB | ~2 MB | ~1.8 MB |
| 8192 tokens | ~832 KB | ~8 MB | ~7.2 MB |

### GPU Memory Usage

When using `GpuBackend`, additional GPU memory is used for:
- Device buffers during batch operations (proportional to batch_size * dim)
- Memory pool caches (reusable buffers from previous batches)
- PTX kernel modules (small, loaded once)

Call `gpu_backend.clear_pools()` to free pooled GPU memory when not in use.

## Profiling Tips

### CPU Profiling

    # Linux: perf
    cargo build --release --features simd
    perf record --call-graph dwarf target/release/examples/demo
    perf report

    # macOS: Instruments
    cargo build --release --features simd
    instruments -t "Time Profiler" target/release/examples/demo

### GPU Profiling

    # NVIDIA Nsight Systems
    nsys profile target/release/examples/demo_gpu

    # NVIDIA Nsight Compute (kernel-level)
    ncu --set full target/release/examples/demo_gpu

Key metrics to watch:
- **Kernel occupancy**: > 50% indicates good GPU utilization
- **Memory bandwidth**: Check for bank conflicts in shared memory
- **Transfer time**: Should be < 50% of total time for batch >= 32

## Hardware Tested

| Hardware | Backend | Expected Speedup |
|----------|---------|-----------------|
| Any x86_64 with AVX2 | SimdBackend | 2-4x over scalar |
| Any aarch64 with NEON | SimdBackend | 2-4x over scalar |
| NVIDIA T4 (Turing) | GpuBackend | ~10x for batch 64 |
| NVIDIA A100 (Ampere) | GpuBackend | ~20x for batch 64 |
| NVIDIA RTX 4090 (Ada) | GpuBackend | ~15x for batch 64 |

Note: GPU speedup numbers are estimates based on algorithm characteristics and typical GPU throughput. Run benchmarks on your specific hardware for precise results.
