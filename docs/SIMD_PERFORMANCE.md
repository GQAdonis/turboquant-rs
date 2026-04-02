# SIMD Performance Guide

This document describes the SIMD acceleration tiers available in `turboquant-rs`,
how to enable them, and what performance improvements to expect.

---

## Enabling SIMD

```toml
# Cargo.toml
[dependencies]
turboquant = { version = "0.1", features = ["simd"] }
```

For maximum throughput on the local build host, also set:

```toml
# .cargo/config.toml  (already present in this repo)
[build]
rustflags = ["-C", "target-cpu=native"]
```

Without `target-cpu=native`, the compiler cannot assume AVX2/FMA/AVX-512 at
compile time, but the library still detects them **at runtime** via
`is_x86_feature_detected!` and dispatches to the fastest available path.

---

## Backend Tiers

| Tier | Struct | Selected when | Width |
|---|---|---|---|
| AVX-512F | `Avx512Backend` | `avx512f` detected | 16 f32/cycle |
| AVX2 + FMA | `SimdBackend` | `avx2` + `fma` detected | 8 f32/cycle (FMA) |
| AVX2 | `SimdBackend` | `avx2` detected | 8 f32/cycle |
| NEON | `SimdBackend` | aarch64 | 4 f32/cycle (FMA via `vmlaq_f32`) |
| Scalar | `ScalarBackend` | fallback | 1 f32/cycle |

`RuntimeBackend::best_available()` walks this list top-to-bottom and returns
the first matching tier.  `TurboQuant::new()` and `PolarQuant::new()` call this
automatically when the `simd` feature is enabled.

---

## Vectorized Operations

### 1. FWHT butterfly (`fwht_normalized_inplace`)

The Fast Walsh-Hadamard Transform is the primary hot path in both
`Rotation::apply` (forward) and `Rotation::apply_inverse` (backward).

- **AVX-512**: 16 f32 butterflies per iteration for strides ≥ 16; falls back to
  AVX2 for strides 8–15 and scalar for strides 1–7.
- **AVX2**: 8 f32 butterflies per iteration for strides ≥ 8; scalar for 1–4.
- **NEON**: 4 f32 butterflies per iteration for strides ≥ 4; scalar for 1–2.

For typical head dimensions (64–256), most butterfly passes are fully
vectorized.  The scalar tail only handles the first two passes (strides 1, 2).

### 2. Sign-flip (`apply_signs`)

The diagonal matrix D multiplies each vector element by ±1 before and after
FWHT.  The naive scalar form is `x *= sign as f32`.

The SIMD path precomputes XOR masks (`0x80000000` when sign = −1, else `0`) at
construction time and uses bitwise XOR on the float sign bit:

```
_mm256_xor_ps(v_data, _mm256_castsi256_ps(v_masks))  // AVX2: 8 floats
_mm512_castsi512_ps(_mm512_xor_si512(...))             // AVX-512: 16 floats
veorq_u32(v, m) → reinterpret as f32                   // NEON: 4 floats
```

This eliminates the multiply entirely (XOR is 1 cycle latency vs 4 for f32 mul).

### 3. Dot product / inner product accumulation (`dot_product`)

Used in `PolarQuant::inner_product` (attention logit estimation):

- **AVX2**: `_mm256_mul_ps` + `_mm256_add_ps`, 8 f32/iter
- **AVX2 + FMA**: `_mm256_fmadd_ps`, 8 f32/iter, fused → ~10–15% fewer μops
- **AVX-512F**: `_mm512_fmadd_ps` + `_mm512_reduce_add_ps`, 16 f32/iter
- **NEON**: `vmlaq_f32` (FMA), 4 f32/iter

The `inner_product` hot path was also restructured in this release:

```rust
// Before: scalar loop, no backend involvement
scratch.iter().zip(&indices).map(|(&q, &idx)| q * dequantize(idx)).sum()

// After: dequantize to temp buffer, then backend-dispatched dot product
let centroids = self.codebook.dequantize_slice(&indices);
self.backend.dot_product(&scratch, &centroids)
```

---

## Expected Speedups (head_dim = 128, 3-bit)

These are projected estimates based on operation counts and instruction
throughput.  Actual numbers depend on CPU model, memory layout, and workload.

| Operation | Scalar baseline | AVX2 | AVX2+FMA | AVX-512 |
|---|---|---|---|---|
| `fwht_normalized_inplace` | 1× | ~3–4× | ~3–4× | ~5–7× |
| `apply_signs` | 1× | ~6–8× | ~6–8× | ~10–12× |
| `dot_product` (128-dim) | 1× | ~4–5× | ~5–6× | ~8–10× |
| `PolarQuant::quantize` | 1× | ~2–3× | ~2–3× | ~3–4× |
| `PolarQuant::inner_product` | 1× | ~3–4× | ~3–4× | ~5–6× |

Run the included benchmarks to get numbers for your specific hardware:

```bash
cargo bench --features simd --bench backend_bench
```

Results are written to `target/criterion/` as HTML reports.

---

## Platform Matrix

| CPU Family | AVX2 | FMA | AVX-512F | Notes |
|---|---|---|---|---|
| Intel Haswell (2013+) | ✅ | ✅ | ❌ | Most consumer CPUs since 2015 |
| Intel Skylake-X | ✅ | ✅ | ✅ (AVX-512F) | HEDT / workstation |
| Intel Ice Lake (client) | ✅ | ✅ | ✅ | 10th-gen laptop i7/i9 |
| Intel Sapphire Rapids | ✅ | ✅ | ✅ | 4th-gen Xeon, best tier |
| AMD Zen 2 (Ryzen 3000) | ✅ | ✅ | ❌ | |
| AMD Zen 4 (Ryzen 7000) | ✅ | ✅ | ✅ | Server: EPYC 9004 |
| Apple M1/M2/M3 | N/A | N/A | N/A | NEON path (4 f32/cycle FMA) |
| AWS Graviton 3 | N/A | N/A | N/A | NEON path |

---

## Integration with candle-vllm

`candle-vllm` enables the `simd` feature unconditionally:

```toml
turboquant = { path = "../turboquant-rs", features = ["simd"] }
```

The `CompressedLayerCache` creates `TurboQuant` instances via `TurboQuant::new()`,
which now resolves to `RuntimeBackend::best_available()` — the correct SIMD tier
is selected once at cache initialization and reused for every compress/decompress
call across the lifetime of the serving session.

On a 4090 host (x86_64, AVX2+FMA available), the KV-cache compression cycle is
expected to run 3–4× faster than the pre-SIMD scalar baseline, reducing the
CPU-side compression overhead from ~15% to ~4% of total inference time at 4k
context lengths.

---

## Adding a New SIMD Tier

To add a new instruction set (e.g., AVX-512 VNNI for int8 accumulation):

1. Add unsafe kernel functions in `src/backend/simd.rs` gated on the new
   `#[target_feature]`.
2. Add a new `struct NewBackend` implementing `Backend`.
3. Add a `RuntimeBackend::NewBackend(NewBackend)` variant.
4. Update `RuntimeBackend::best_available()` to detect and return it.
5. Add equivalence tests comparing output to `ScalarBackend`.
6. Document the new tier in this file.
