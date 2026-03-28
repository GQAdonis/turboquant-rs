# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**turboquant-rs** is a pure-Rust implementation of **TurboQuant** — Google Research's KV-cache compression algorithm presented at ICLR 2026. The algorithm compresses transformer attention key-value caches to as few as **3 bits per coordinate** (down from 16-bit float), achieving **6-8x memory reduction** with virtually no accuracy loss.

### What is TurboQuant?

TurboQuant addresses the memory bottleneck in large language model inference by compressing the key-value cache that stores context information. As models process longer inputs, the cache grows rapidly and consumes GPU memory that could otherwise serve more users or run larger models.

**Research Background:**
- Paper: "TurboQuant: Redefining AI Efficiency with Extreme Compression" (ICLR 2026)
- Authors: Amir Zandieh, Vahab Mirrokni (Google), collaborators from Google DeepMind, KAIST, NYU
- Builds on: PolarQuant (AISTATS 2026) and QJL (AAAI 2025)
- Application: KV-cache compression, vector search, semantic similarity at scale

### Algorithm Architecture

TurboQuant operates in two stages:

**Stage 1 - PolarQuant:** MSE-optimal vector quantization via randomized Hadamard rotation + Lloyd-Max scalar quantization
- Converts vectors from Cartesian to polar coordinates (magnitude + angles)
- Eliminates expensive per-block normalization
- Uses precomputed Lloyd-Max centroids for Beta((d-1)/2, (d-1)/2) distribution

**Stage 2 - QJL (Quantized Johnson-Lindenstrauss):** 1-bit residual correction for unbiased inner products
- Applies sign-bit sketch to quantization residual
- Provides theoretical unbiasedness guarantee
- Optional: Community implementations found Stage 1 alone often matches or beats two-stage quality

**Two Variants Provided:**
- **MSE variant** (recommended): All bits allocated to PolarQuant, simpler and faster
- **Prod variant**: (b-1)-bit PolarQuant + 1-bit QJL, theoretically unbiased inner products

## Architecture

### Module Structure

```
src/
├── lib.rs              # Public API, utility functions (dot_product, l2_norm, cosine_similarity)
├── turboquant.rs       # Main TurboQuant struct combining PolarQuant + QJL
├── polar_quant.rs      # Stage 1: Rotation + Lloyd-Max quantization
├── qjl.rs              # Stage 2: 1-bit Johnson-Lindenstrauss residual correction
├── kv_cache.rs         # Transformer KV-cache with compressed attention
├── rotation.rs         # Randomized Hadamard transform R = H̃·D
├── hadamard.rs         # Fast Walsh-Hadamard Transform (FWHT) implementation
├── codebook.rs         # Precomputed Lloyd-Max centroids for 2/3/4-bit quantization
├── bitpack.rs          # Bit-packing utilities for compact storage
└── error.rs            # Error types
```

### Data Flow

**Compression Path (Quantization):**
```
Input vector x
  → Compute ‖x‖₂ and normalize (x̂ = x/‖x‖)
  → Apply rotation (ŷ = R·x̂)
  → Scalar quantize each coordinate (Lloyd-Max)
  → Bit-pack indices
  → Store: QuantizedVector { norm, packed_indices, dim, bits }
```

**Inner Product Fast Path (Attention Logits):**
```
Query q, Compressed Key k_qv
  → Rotate query (q_rot = R·q)
  → Unpack key indices
  → Dot product in rotated space: Σᵢ (q_rot)ᵢ · centroid[idx_i]
  → Scale by key norm: result = dot · ‖k‖
```

This fast path avoids full decompression and is the critical hot path for attention computation.

### Key Design Decisions

1. **Power-of-Two Dimensions Required:** FWHT (Fast Walsh-Hadamard Transform) requires dim = 2^k. Typical transformer head dimensions (64, 128, 256) satisfy this naturally.

2. **Seeded Randomization:** All rotations are deterministic given a seed. Keys and values use different seeds for independent rotations. This ensures reproducibility and enables compression/decompression in distributed systems.

3. **Lloyd-Max Codebooks:** Precomputed optimal centroids for N(0, 1/dim) distribution. Scaled at construction time based on vector dimension — higher dimensions get tighter centroids.

4. **MSE vs Prod Trade-off:** Community validation (turboquant_plus, tonbistudio) found MSE variant (all bits to Stage 1) matches or exceeds Prod variant quality. Both provided for completeness and research purposes.

5. **No External Dependencies:** Core algorithm uses only `thiserror` for error handling. No BLAS, LAPACK, or GPU dependencies — pure Rust for portability.

## Development Commands

### Build
```bash
cargo build                    # Debug build
cargo build --release          # Optimized release build
```

### Testing
```bash
cargo test                     # Run all tests
cargo test --lib               # Run library unit tests only
cargo test --test integration  # Run integration tests
cargo test polar_quant         # Run tests containing "polar_quant"
cargo test -- --nocapture      # Show println! output from tests
```

### Run Examples
```bash
cargo run --example demo       # Comprehensive demo of all components
cargo run --example demo --release  # Run demo with optimizations
```

### Benchmarks
```bash
cargo bench                    # Run all benchmarks with Criterion
cargo bench --bench bench      # Run specific benchmark suite
```

### Documentation
```bash
cargo doc --open               # Build and open documentation in browser
cargo doc --no-deps --open     # Documentation for this crate only
```

### Code Quality
```bash
cargo clippy                   # Rust linter
cargo clippy -- -W clippy::pedantic  # Strict linting
cargo fmt                      # Format code
cargo fmt -- --check           # Check formatting without modifying
```

## Implementation Notes

### Critical Mathematical Properties

**Orthogonality:** The randomized Hadamard rotation R preserves inner products and norms:
- ⟨R·a, R·b⟩ = ⟨a, b⟩
- ‖R·x‖ = ‖x‖
- R^T·R = I (rotation is orthogonal)

This property enables the fast inner product computation without full decompression.

**Coordinate Distribution:** After rotation, each coordinate of a unit vector follows Beta((d-1)/2, (d-1)/2) ≈ N(0, 1/d) for large d. This concentrated, predictable distribution enables optimal scalar quantization without per-block calibration.

### Memory Layout

Compressed representation of a d-dimensional vector at b bits:
- Norm: 4 bytes (f32)
- Indices: ⌈d·b / 8⌉ bytes (bit-packed)
- Total: 4 + ⌈d·b / 8⌉ bytes

Example (d=128, b=3):
- Uncompressed: 128 × 4 = 512 bytes
- Compressed: 4 + ⌈128×3 / 8⌉ = 4 + 48 = 52 bytes
- Ratio: 512 / 52 ≈ 9.8x

For KV-cache (key + value pairs):
- At 3-bit, 128-dim: 104 bytes per token vs 1024 bytes uncompressed (~10x reduction)
- At 8192 token sequence: ~0.8 MB compressed vs ~8 MB uncompressed

### Testing Strategy

Tests are distributed across modules with each module testing its own invariants:

**rotation.rs:** Roundtrip consistency, norm preservation, orthogonality, inner product preservation
**codebook.rs:** Centroid roundtrip, ordering, dimension scaling
**polar_quant.rs:** Compression ratio, cosine similarity, inner product accuracy, dimension validation
**qjl.rs:** Unbiased estimation (verified over many random queries), dimension validation
**turboquant.rs:** MSE vs Prod comparison, fallback behavior at 2-bit
**kv_cache.rs:** Attention softmax correctness, weighted sum, compression metrics

### Performance Considerations

**Hot Paths:**
1. `PolarQuant::inner_product()` — Used for every attention logit computation
2. `Rotation::apply()` — Called once per query, once per key quantization
3. `fwht_normalized_inplace()` — Core of rotation, in-place O(d log d) transform

**Optimization Opportunities:**
- SIMD vectorization of FWHT (currently scalar)
- Parallel batch quantization for multiple vectors
- GPU kernels for large-scale inference (cuBLAS integration)
- Lazy decompression or ring buffer for values in attention

### Dimension Requirements

All operations require power-of-two dimensions (16, 32, 64, 128, 256, etc.) due to FWHT. Attempting to use non-power-of-two dimensions will result in `TurboQuantError::DimensionNotPowerOfTwo`.

Transformer models typically use head dimensions that satisfy this constraint naturally (most common: 64, 128).

### Bit-Width Support

Supported: 2, 3, 4 bits
- 2-bit: 4 centroids, high compression, moderate quality
- 3-bit: 8 centroids, sweet spot — 6-8x compression with <2% quality loss
- 4-bit: 16 centroids, lower compression, near-lossless quality

Higher bit-widths not supported due to diminishing returns and increased codebook maintenance.

## Common Development Tasks

### Adding a New Bit-Width (e.g., 5-bit)

1. Compute Lloyd-Max centroids for N(0,1) distribution with k=32 levels
2. Add `C_5BIT` constant to `codebook.rs`
3. Update `Codebook::new()` match statement
4. Add tests for 5-bit quantization
5. Update benchmarks to include 5-bit comparison

### Integrating with a Transformer Model

```rust
use turboquant::KvCache;

// Per attention head, usually in model initialization:
let head_dim = 128;
let bits = 3;
let mut cache = KvCache::new(head_dim, bits, key_seed, val_seed)?;

// During forward pass for each token:
cache.push(&key_vector, &value_vector)?;

// During attention computation:
let attention_output = cache.attend(&query_vector)?;
```

For multi-head attention, create one `KvCache` per head with different seeds.

### Debugging Accuracy Issues

1. Compare MSE vs Prod variants — MSE often more stable
2. Check cosine similarity of reconstructed vectors (should be >0.98)
3. Validate inner product relative error (should be <5% for 3-bit)
4. Increase bit-width (3→4) to isolate quantization error
5. Verify input vectors are not near-zero (norm < ε triggers edge cases)
6. Check seed consistency between compression and decompression

### Adding SIMD Acceleration

Target files: `hadamard.rs`, `rotation.rs`
- FWHT is embarrassingly parallel — vectorize butterfly operations
- Use `std::arch` for platform-specific SIMD (AVX2, NEON)
- Maintain scalar fallback for portability
- Add feature flag: `simd` in Cargo.toml

## Related Work & References

- Original paper: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- PolarQuant foundation (AISTATS 2026)
- QJL algorithm (AAAI 2025)
- Community implementations: turboquant_plus, tonbistudio
- Lloyd-Max quantization: Lloyd (1982), Max (1960)
- Johnson-Lindenstrauss transform: Zandieh et al. (2024)

## Project Status

This is a research implementation demonstrating the TurboQuant algorithm in pure Rust. Production deployment would benefit from:
- SIMD/GPU acceleration
- Integration with specific LLM frameworks (llama.cpp, candle, burn)
- Multi-threaded batch processing
- Profiling and optimization of hot paths
- Support for dynamic sequence lengths and cache eviction strategies
