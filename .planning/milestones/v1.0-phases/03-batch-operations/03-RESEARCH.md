# Phase 3: Batch Operations - Research

**Researched:** 2026-03-27
**Domain:** CPU batch parallelization for vector quantization and attention operations
**Confidence:** HIGH

## Summary

Phase 3 introduces batch processing APIs that enable efficient multi-vector operations through CPU parallelization with rayon. Building on the Backend trait foundation (Phase 1) and SIMD acceleration (Phase 2), this phase focuses on ergonomic batch APIs that automatically leverage multi-core CPUs without sacrificing single-vector performance.

The research validates that rayon 1.11.0 provides mature work-stealing parallelism ideal for independent vector operations. The FWHT-dominant workload (O(d log d) per vector) has sufficient granularity to amortize parallelization overhead at batch sizes >= 16-32 vectors. Simple `&[Vec<f32>]` API design is sufficient for Phase 3 - contiguous memory layouts and strided access patterns can be deferred to Phase 4 where GPU transfer costs justify the complexity.

**Critical finding:** Batch-of-1 performance must match single-vector API exactly. This requires explicit fast-path checks before invoking rayon to avoid ~1-5μs spawn overhead on degenerate cases. Each batch API must call the corresponding single-vector method when `batch.len() == 1`.

**Primary recommendation:** Add three batch APIs to PolarQuant (`batch_quantize`, `batch_inner_product`) and KvCache (`batch_attend`), integrate rayon 1.11.0 with par_iter() for automatic parallelization, implement batch-of-1 fast paths, and benchmark throughput improvement at batch size 64 to validate success criteria.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BATCH-01 | Add batch_quantize(&[Vec<f32>]) API to PolarQuant | API design patterns - ergonomic slice-of-vectors signature with rayon par_iter |
| BATCH-02 | Add batch_inner_product(query, &[QuantizedVector]) API | Batch inner product pattern - single query against multiple keys (attention logits) |
| BATCH-03 | Add batch_attend(query) to KvCache for multi-query attention | Multi-query attention - parallel logit computation + sequential softmax/weighted_sum |
| BATCH-04 | Implement zero-copy batch patterns (contiguous memory layout) | Zero-copy research - &[&[f32]] API alternative, defer strided layouts to Phase 4 |
| BATCH-05 | Add parallel CPU batch processing with rayon | Rayon 1.11.0 integration - par_iter().map().collect() with work-stealing |
| BATCH-06 | Verify batch-of-1 performance matches single-vector API | Batch-of-1 fast path - explicit len check before rayon spawn |
| BATCH-07 | Demonstrate batch-of-64 performance improvement | Benchmark strategy - integration.rs batch throughput vs 64 sequential calls |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| rayon | 1.11.0 | CPU work-stealing parallelism | De facto standard for data parallelism in Rust, mature (since 2015), zero-cost when unused, automatic thread pool management |
| criterion | 0.5 | Batch throughput benchmarking | Already in dev-dependencies, statistical rigor for measuring speedup |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| rayon::ThreadLocal | 1.11.0 (submodule) | Thread-local scratch buffers | Optimization if allocation profiling shows bottleneck (likely unnecessary) |
| std::hint::black_box | stdlib | Prevent benchmark optimization | Ensure compiler doesn't elide batch operations in throughput tests |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| rayon | std::thread manual pool | More control but complex, rayon handles load balancing automatically |
| rayon | async/tokio | Wrong tool - batch processing is CPU-bound not I/O-bound |
| &[Vec<f32>] | &[&[f32]] (zero-copy) | Both provided - slice version is more flexible, vec version more ergonomic |
| par_iter() | par_chunks() | Chunks add complexity with no benefit - each vector is independent work unit |

**Installation:**
```bash
cargo add rayon@1.11
```

**Version verification:**
Rayon 1.11.0 verified via `cargo search rayon --limit 1` on 2026-03-27. Requires Rust 1.80+, project uses 1.94.1 (confirmed in Cargo.toml rust-version field).

## Architecture Patterns

### Recommended API Structure
```rust
// src/polar_quant.rs additions
impl<B: Backend> PolarQuant<B> {
    /// Quantize multiple vectors in parallel (batch API)
    pub fn batch_quantize(&self, vecs: &[Vec<f32>]) -> Result<Vec<QuantizedVector>>

    /// Zero-copy variant accepting slices
    pub fn batch_quantize_slices(&self, vecs: &[&[f32]]) -> Result<Vec<QuantizedVector>>

    /// Compute inner products for one query against multiple keys
    pub fn batch_inner_product(&self, query: &[f32], keys: &[QuantizedVector]) -> Result<Vec<f32>>
}

// src/kv_cache.rs additions
impl<B: Backend> KvCache<B> {
    /// Compute attention for multiple queries in parallel
    pub fn batch_attend(&self, queries: &[Vec<f32>]) -> Result<Vec<Vec<f32>>>

    /// Zero-copy variant
    pub fn batch_attend_slices(&self, queries: &[&[f32]]) -> Result<Vec<Vec<f32>>>
}
```

### Pattern 1: Batch Quantization with Rayon

**What:** Parallelize independent vector quantization operations across CPU cores

**When to use:** When processing multiple vectors of same dimension with uniform computational cost

**Example:**
```rust
// Source: rayon documentation + project-specific integration
use rayon::prelude::*;

pub fn batch_quantize(&self, vecs: &[Vec<f32>]) -> Result<Vec<QuantizedVector>> {
    // CRITICAL: Fast path for batch-of-1 to avoid rayon overhead
    if vecs.len() == 1 {
        return Ok(vec![self.quantize(&vecs[0])?]);
    }

    // Validate all dimensions up front (fail fast)
    for v in vecs {
        self.check_dim(v.len())?;
    }

    // Parallel processing with work-stealing
    vecs.par_iter()
        .map(|v| self.quantize(v))
        .collect()
}
```

**Key properties:**
- Each quantize() call is independent (no shared state beyond read-only codebook/rotation)
- Work-stealing handles load balancing automatically
- Error handling: early termination on first error via Result propagation
- Memory: each thread allocates scratch buffers independently (acceptable overhead)

### Pattern 2: Batch Inner Product (Single Query, Multiple Keys)

**What:** Compute attention logits for one query against all cached keys in parallel

**When to use:** Standard attention computation in decoder (single query token, many cached key tokens)

**Example:**
```rust
pub fn batch_inner_product(&self, query: &[f32], keys: &[QuantizedVector]) -> Result<Vec<f32>> {
    self.check_dim(query.len())?;

    // Batch-of-1 fast path
    if keys.len() == 1 {
        return Ok(vec![self.inner_product(query, &keys[0])?]);
    }

    // Validate all key dimensions
    for k in keys {
        self.check_dim(k.dim)?;
    }

    // Parallel inner products (query shared read-only)
    keys.par_iter()
        .map(|k| self.inner_product(query, k))
        .collect()
}
```

**Performance note:** Each inner_product() uses RefCell scratch buffer. With rayon parallelism, each thread gets independent scratch buffer (RefCell is not Sync, so thread-local instantiation). This is correct and efficient - no contention.

### Pattern 3: Multi-Query Attention

**What:** Process multiple independent attention queries in parallel (batch inference or parallel decoding scenarios)

**When to use:** Speculative decoding, batch inference, multi-head attention across different sequences

**Example:**
```rust
pub fn batch_attend(&self, queries: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
    // Batch-of-1 fast path
    if queries.len() == 1 {
        return Ok(vec![self.attend(&queries[0])?]);
    }

    // Validate dimensions
    for q in queries {
        self.check_dim(q.len())?;
    }

    // Parallel attention computations
    queries.par_iter()
        .map(|q| self.attend(q))
        .collect()
}
```

**Note:** Each attend() call does logits → softmax → weighted_sum sequentially. This is correct - the parallelism is across queries, not within a single attention computation.

### Anti-Patterns to Avoid

- **Premature optimization:** Don't implement thread-local scratch buffer pooling until allocation profiling proves it's a bottleneck
- **False sharing:** Don't pack results into contiguous buffer where threads write to adjacent elements (rayon's Vec collection handles this correctly)
- **Over-parallelization:** Don't parallelize batches smaller than ~8 vectors - overhead exceeds benefit
- **API inconsistency:** Don't make batch APIs return different error types or semantics than single-vector APIs

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Thread pool management | Manual std::thread spawning | rayon::ThreadPoolBuilder | Handles work-stealing, thread count tuning, panic propagation |
| Load balancing | Manual work queue + thread pool | rayon par_iter() | Automatic work-stealing beats manual scheduling for uniform tasks |
| Thread-local storage | Manual ThreadLocal<RefCell<T>> | rayon::ThreadLocal or accept per-thread allocation | Complexity vs benefit unclear until profiled |
| Parallel error handling | Custom Result aggregation | rayon collect::<Result<Vec<T>>>() | Built-in early termination on first error |

**Key insight:** Rayon's work-stealing scheduler is battle-tested and handles heterogeneous workloads (though turboquant's uniform vector dimensions make this less critical). Manual thread pool management adds 100+ lines of complex, bug-prone code with no measurable benefit.

## Common Pitfalls

### Pitfall 1: Batch-of-1 Performance Regression
**What goes wrong:** Calling rayon par_iter() on single-element slice incurs ~1-5μs spawn overhead, causing batch API to be slower than single-vector API

**Why it happens:** Rayon's work-stealing has fixed overhead to set up parallel context, even when there's only one work item

**How to avoid:**
```rust
if batch.len() == 1 {
    return Ok(vec![self.single_vector_method(&batch[0])?]);
}
```
Add explicit fast-path check at start of every batch method

**Warning signs:**
- Benchmark shows batch_quantize([single_vec]) slower than quantize(single_vec)
- Success criterion "batch-of-1 matches single-vector" fails
- Profiling shows rayon::spawn overhead in flamegraph

### Pitfall 2: Dimension Validation After Parallel Dispatch
**What goes wrong:** Launching parallel work before validating inputs means invalid dimensions trigger errors across all threads, wasting CPU cycles

**Why it happens:** Natural to write `vecs.par_iter().map(|v| self.quantize(v))` without up-front validation

**How to avoid:**
```rust
// Validate all dimensions sequentially BEFORE par_iter()
for v in vecs {
    self.check_dim(v.len())?;
}
// Now safe to parallelize
vecs.par_iter().map(|v| self.quantize(v)).collect()
```

**Warning signs:**
- Error messages appear multiple times (once per thread)
- Profiling shows wasted parallel work before error termination

### Pitfall 3: Over-Parallelization of Small Batches
**What goes wrong:** Parallelizing batches of 2-4 vectors adds overhead (thread spawn, cache coherence traffic) that exceeds benefit

**Why it happens:** Rayon makes parallelization so easy that it's tempting to apply everywhere

**How to avoid:**
- Use batch-of-1 fast path (required for BATCH-06)
- For batch sizes 2-7, consider sequential processing OR let rayon handle (work-stealing overhead is low, ~1μs per spawn)
- Don't add complex threshold logic unless benchmarks prove it's needed

**Warning signs:**
- Small batch performance doesn't scale linearly with batch size
- Sequential loop outperforms par_iter() at batch size 4-8

**Verdict:** Phase 3 scope is batch-of-1 fast path only. More sophisticated thresholds can be added in Phase 4 if GPU dispatch requires it.

### Pitfall 4: RefCell Borrow Conflicts in Parallel Context
**What goes wrong:** If PolarQuant wasn't Clone, sharing across threads would cause borrow panics when multiple threads access scratch RefCell

**Why it happens:** RefCell is not thread-safe (not Sync), but rayon moves work to different threads

**How to avoid:** Already solved by Phase 1 - PolarQuant implements Clone, so each thread gets independent instance with its own RefCell scratch buffer

**Warning signs:**
- BorrowMutError panics in batch operations
- Only one thread successfully processes vectors

**Status:** Not an issue for this project - Clone impl exists, RefCell is thread-local per design.

### Pitfall 5: Result Collection Error Handling
**What goes wrong:** Misunderstanding how `collect::<Result<Vec<T>>>()` behaves - does it stop on first error or collect all errors?

**Why it happens:** FromIterator impl for Result is subtle - stops on first Err, short-circuits remaining work

**How to avoid:** This is correct behavior for turboquant (fail fast). Document in API that batch operations stop on first error, partial results are discarded.

**Warning signs:**
- Users expect to see which specific vectors failed in batch
- Confusion about why batch succeeded with some errors

**Resolution:** Document explicitly: "Batch operations fail fast on first error. For per-vector error handling, iterate and call single-vector API."

## Code Examples

Verified patterns from Rust best practices and rayon documentation:

### Batch Quantization (Full Implementation)
```rust
// Source: rayon documentation + project patterns
use rayon::prelude::*;

impl<B: Backend> PolarQuant<B> {
    /// Quantize multiple vectors in parallel.
    ///
    /// Returns `Vec<QuantizedVector>` with same length as input.
    /// Fails fast on first error (dimension mismatch, invalid input).
    ///
    /// # Performance
    /// - Batch-of-1: identical to `quantize()` (zero overhead)
    /// - Batch >= 16: parallel processing across CPU cores
    /// - Expected speedup: ~3-6x on 8-core CPU at batch size 64
    #[must_use = "quantized vectors should be stored"]
    pub fn batch_quantize(&self, vecs: &[Vec<f32>]) -> Result<Vec<QuantizedVector>> {
        // Fast path: batch-of-1 matches single-vector performance
        if vecs.len() == 1 {
            return Ok(vec![self.quantize(&vecs[0])?]);
        }

        // Validate all dimensions up front (fail fast before spawning threads)
        for v in vecs {
            self.check_dim(v.len())?;
        }

        // Parallel quantization with automatic work-stealing
        vecs.par_iter()
            .map(|v| self.quantize(v))
            .collect()
    }

    /// Zero-copy variant accepting borrowed slices.
    #[must_use = "quantized vectors should be stored"]
    pub fn batch_quantize_slices(&self, vecs: &[&[f32]]) -> Result<Vec<QuantizedVector>> {
        if vecs.len() == 1 {
            return Ok(vec![self.quantize(vecs[0])?]);
        }

        for v in vecs {
            self.check_dim(v.len())?;
        }

        vecs.par_iter()
            .map(|v| self.quantize(v))
            .collect()
    }
}
```

### Batch Inner Product (Attention Logits)
```rust
impl<B: Backend> PolarQuant<B> {
    /// Compute inner products of one query against multiple keys.
    ///
    /// This is the primary hot path for attention logit computation.
    /// Query is shared (read-only) across all parallel computations.
    ///
    /// # Example
    /// ```rust
    /// let pq = PolarQuant::new(128, 3, 42)?;
    /// let keys: Vec<QuantizedVector> = /* compressed keys */;
    /// let query = vec![0.1f32; 128];
    /// let logits = pq.batch_inner_product(&query, &keys)?;
    /// assert_eq!(logits.len(), keys.len());
    /// ```
    #[must_use = "inner product results should be used"]
    pub fn batch_inner_product(
        &self,
        query: &[f32],
        keys: &[QuantizedVector],
    ) -> Result<Vec<f32>> {
        self.check_dim(query.len())?;

        // Fast path: batch-of-1
        if keys.len() == 1 {
            return Ok(vec![self.inner_product(query, &keys[0])?]);
        }

        // Validate all key dimensions
        for k in keys {
            self.check_dim(k.dim)?;
        }

        // Parallel inner products (query is shared, keys are independent)
        keys.par_iter()
            .map(|k| self.inner_product(query, k))
            .collect()
    }
}
```

### Batch Attend (KvCache Multi-Query)
```rust
// Source: project KvCache pattern extended with rayon
impl<B: Backend> KvCache<B> {
    /// Compute attention for multiple queries in parallel.
    ///
    /// Each query independently computes:
    /// 1. Logits: ⟨query, key_i⟩ for all cached keys
    /// 2. Softmax: normalize logits
    /// 3. Weighted sum: Σ softmax[i] · value[i]
    ///
    /// Parallelism is across queries, not within single attention.
    #[must_use = "attention outputs should be used"]
    pub fn batch_attend(&self, queries: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        // Fast path: batch-of-1
        if queries.len() == 1 {
            return Ok(vec![self.attend(&queries[0])?]);
        }

        // Validate dimensions
        for q in queries {
            self.check_dim(q.len())?;
        }

        // Parallel attention computations
        queries.par_iter()
            .map(|q| self.attend(q))
            .collect()
    }

    /// Zero-copy variant.
    #[must_use = "attention outputs should be used"]
    pub fn batch_attend_slices(&self, queries: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        if queries.len() == 1 {
            return Ok(vec![self.attend(queries[0])?]);
        }

        for q in queries {
            self.check_dim(q.len())?;
        }

        queries.par_iter()
            .map(|q| self.attend(q))
            .collect()
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual thread pool | rayon work-stealing | ~2016 (rayon 0.1) | Simplified parallelization, automatic load balancing |
| Box<dyn Backend> | Static dispatch (Phase 1) | Phase 1 complete | Enables inlining across thread boundaries, critical for batch perf |
| Sequential processing | Batch APIs with rayon | This phase | 3-6x throughput on multi-core for batch >= 32 |

**Deprecated/outdated:**
- **std::thread::spawn per vector:** Too expensive (kernel scheduling overhead), use rayon work-stealing
- **Scoped threads with crossbeam:** Rayon supersedes for data parallelism use cases
- **Manual work queue:** Rayon's work-stealing is more efficient and battle-tested

## Open Questions

1. **Thread-local scratch buffer optimization**
   - What we know: Each thread allocates scratch buffers in inner_product via RefCell
   - What's unclear: Is this a measurable bottleneck? Profile needed.
   - Recommendation: Defer optimization until profiling proves allocation is >5% of batch time

2. **Optimal rayon thread pool configuration**
   - What we know: Rayon defaults to num_cpus threads
   - What's unclear: Should turboquant override with ThreadPoolBuilder?
   - Recommendation: Use defaults. Users needing control can set RAYON_NUM_THREADS env var.

3. **Batch size threshold for parallelization**
   - What we know: Batch-of-1 must use fast path (required by BATCH-06)
   - What's unclear: Should batch sizes 2-7 go through rayon or sequential?
   - Recommendation: Let rayon handle all batch >= 2. Work-stealing overhead is minimal (~1μs), not worth complex threshold logic.

4. **Zero-copy API adoption**
   - What we know: `&[&[f32]]` API is more flexible than `&[Vec<f32>]`
   - What's unclear: Will users actually use it? API surface complexity cost.
   - Recommendation: Provide both. Vec variant delegates to slice variant internally. Minimal cost, maximum flexibility.

## Validation Architecture

> Nyquist validation enabled per .planning/config.json (workflow.nyquist_validation: true)

### Test Framework
| Property | Value |
|----------|-------|
| Framework | cargo test (built-in) + criterion 0.5.1 for benchmarks |
| Config file | benches/integration.rs (extend existing) |
| Quick run command | `cargo test --lib polar_quant::batch --release` |
| Full suite command | `cargo test --release && cargo bench --bench integration` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| BATCH-01 | batch_quantize() returns Vec<QuantizedVector> of correct length | unit | `cargo test --lib polar_quant::batch_quantize` | ❌ Wave 0 |
| BATCH-01 | batch_quantize() preserves compression ratio per vector | unit | `cargo test --lib polar_quant::batch_compression_ratio` | ❌ Wave 0 |
| BATCH-02 | batch_inner_product() returns Vec<f32> matching sequential calls | unit | `cargo test --lib polar_quant::batch_inner_product_correctness` | ❌ Wave 0 |
| BATCH-03 | batch_attend() returns correct dimensions and softmax properties | unit | `cargo test --lib kv_cache::batch_attend` | ❌ Wave 0 |
| BATCH-04 | batch_quantize_slices() zero-copy variant works correctly | unit | `cargo test --lib polar_quant::batch_slices` | ❌ Wave 0 |
| BATCH-05 | batch operations use multiple CPU cores (rayon integration) | integration | `cargo test --release batch::rayon_utilization -- --nocapture` | ❌ Wave 0 |
| BATCH-06 | batch-of-1 performance matches single-vector API (no regression) | benchmark | `cargo bench --bench integration batch_of_1_regression` | ❌ Wave 0 |
| BATCH-07 | batch-of-64 shows throughput improvement vs sequential | benchmark | `cargo bench --bench integration batch_64_throughput` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test --lib --release polar_quant::batch kv_cache::batch` (< 5 seconds)
- **Per wave merge:** `cargo test --release` (all tests, < 30 seconds)
- **Phase gate:** `cargo test --release && cargo bench --bench integration` before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/polar_quant.rs` — add batch_quantize tests (BATCH-01, BATCH-02, BATCH-04)
- [ ] `tests/kv_cache.rs` — add batch_attend tests (BATCH-03)
- [ ] `benches/integration.rs` — add batch_of_1_regression benchmark (BATCH-06)
- [ ] `benches/integration.rs` — add batch_64_throughput benchmark (BATCH-07)
- [ ] `tests/batch_parallel.rs` — NEW file verifying rayon multi-core utilization (BATCH-05)

All tests will be added in Wave 0 (infrastructure wave) before implementation waves.

## Sources

### Primary (HIGH confidence)
- **Rayon 1.11.0:** Verified via `cargo search rayon --limit 1` on 2026-03-27
  - Repository: https://github.com/rayon-rs/rayon
  - Documentation: https://docs.rs/rayon/1.11.0
  - Rust requirement: 1.80+ (project uses 1.94.1, compatible)
- **Existing codebase:** polar_quant.rs, kv_cache.rs, backend.rs
  - Backend trait with Clone + Debug bounds (supports thread-safe usage)
  - RefCell scratch buffer pattern (thread-local via Clone)
  - Existing single-vector APIs provide correctness baseline

### Secondary (MEDIUM confidence)
- **Training data (January 2025):** Rayon work-stealing patterns, par_iter() API, performance characteristics
  - Confidence: MEDIUM - rayon API stable since 2016, unlikely to change
  - Validation: Verified version 1.11.0 exists, Rust 1.80 requirement matches project

### Tertiary (LOW confidence)
- **Batch size thresholds:** "~1-5μs rayon spawn overhead" from general knowledge
  - Needs empirical validation via benchmarking in Phase 3
  - Conservative recommendation: explicit batch-of-1 fast path (required), let rayon handle rest

## Metadata

**Confidence breakdown:**
- Standard stack (rayon): HIGH - verified version, mature ecosystem, project requirements compatible
- Architecture patterns: HIGH - based on existing codebase patterns + rayon documentation
- API design: HIGH - extends existing PolarQuant/KvCache APIs consistently
- Performance expectations: MEDIUM - "3-6x throughput" based on typical parallelism gains, needs validation
- Pitfalls: HIGH - based on common rayon gotchas and project-specific RefCell patterns

**Research date:** 2026-03-27
**Valid until:** 2026-06-27 (90 days - rayon is stable, batch patterns unlikely to change)

**Methodology:**
1. Verified rayon 1.11.0 via cargo registry (current as of 2026-03-27)
2. Analyzed existing codebase (polar_quant.rs, kv_cache.rs, backend.rs) for API consistency
3. Reviewed Phase 1/2 research for architectural context (Backend trait, SIMD integration)
4. Applied rayon best practices from documentation and training data
5. Identified critical pitfalls based on RefCell thread-safety patterns and rayon spawn overhead

**Assumptions validated:**
- ✅ PolarQuant<B: Backend> implements Clone (verified in polar_quant.rs line 68)
- ✅ Backend trait requires Clone + Debug (verified in backend/mod.rs line 26)
- ✅ RefCell scratch buffer exists (verified in polar_quant.rs line 65)
- ✅ SIMD backend completed (Phase 2 complete per STATE.md)
- ✅ Criterion 0.5 in dev-dependencies (verified in Cargo.toml line 19)

**Negative claims verified:**
- No contiguous memory layout requirement found in current API (deferred to Phase 4 GPU)
- No existing batch API in codebase (grep for "batch" in src/ confirmed)
- No rayon dependency currently (Cargo.toml checked, not present)
