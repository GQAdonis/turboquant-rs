//! Integration benchmarks for realistic attention workloads.
//!
//! These benchmarks measure end-to-end performance of the attention hot path
//! at multiple sequence lengths to establish baselines for optimization work.
//!
//! Run with: `cargo bench --bench integration`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use turboquant::{KvCache, PolarQuant};

#[cfg(feature = "simd")]
use turboquant::{Backend, ScalarBackend, SimdBackend};

// ── Helpers ─────────────────────────────────────────────────────────────────

fn sine_vec(dim: usize, freq: f32) -> Vec<f32> {
    (0..dim).map(|i| (i as f32 * freq).sin()).collect()
}

fn fill_cache(cache: &mut KvCache, n: usize, dim: usize) {
    for i in 0..n {
        let freq = i as f32 * 0.01;
        cache
            .push(&sine_vec(dim, freq), &sine_vec(dim, freq + 0.5))
            .unwrap();
    }
}

// ── Attention end-to-end benchmarks ────────────────────────────────────────

fn bench_attention_e2e(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_e2e");
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    let dim = 128;
    let bits = 3;

    for seq_len in [128, 512, 2048, 8192] {
        let mut cache = KvCache::new(dim, bits, 42, 99).unwrap();
        fill_cache(&mut cache, seq_len, dim);
        let query = sine_vec(dim, 0.07);

        // Full attention (logits + softmax + weighted sum)
        group.bench_with_input(
            BenchmarkId::new("attend", seq_len),
            &seq_len,
            |b, _| b.iter(|| cache.attend(black_box(&query)).unwrap()),
        );

        // Logits only (the inner_product hot path)
        group.bench_with_input(
            BenchmarkId::new("logits_only", seq_len),
            &seq_len,
            |b, _| b.iter(|| cache.attention_logits(black_box(&query)).unwrap()),
        );
    }

    group.finish();
}

// ── Inner product throughput benchmarks ────────────────────────────────────

fn bench_inner_product_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("inner_product_throughput");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    let dim = 128;

    for bits in [2u8, 3, 4] {
        let pq = PolarQuant::new(dim, bits, 42).unwrap();
        let key_vec = sine_vec(dim, 0.1);
        let query = sine_vec(dim, 0.07);
        let qv = pq.quantize(&key_vec).unwrap();

        // Single inner product (measures per-call overhead including allocation)
        group.bench_with_input(
            BenchmarkId::new("single", format!("{bits}bit")),
            &bits,
            |b, _| b.iter(|| pq.inner_product(black_box(&query), black_box(&qv)).unwrap()),
        );

        // Batch of 1000 inner products (amortized overhead)
        let keys: Vec<_> = (0..1000)
            .map(|i| {
                let v = sine_vec(dim, 0.01 * i as f32);
                pq.quantize(&v).unwrap()
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch_1000", format!("{bits}bit")),
            &bits,
            |b, _| {
                b.iter(|| {
                    let mut sum = 0.0f32;
                    for k in &keys {
                        sum += pq.inner_product(black_box(&query), black_box(k)).unwrap();
                    }
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

// ── Quantization throughput benchmarks ─────────────────────────────────────

fn bench_quantize_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_throughput");
    group.warm_up_time(Duration::from_secs(2));

    let dim = 128;

    for bits in [2u8, 3, 4] {
        let pq = PolarQuant::new(dim, bits, 42).unwrap();

        // Quantize 100 vectors (simulates filling cache for 100 tokens)
        let vecs: Vec<Vec<f32>> = (0..100)
            .map(|i| sine_vec(dim, 0.01 * i as f32))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch_100", format!("{bits}bit")),
            &bits,
            |b, _| {
                b.iter(|| {
                    for v in &vecs {
                        black_box(pq.quantize(black_box(v)).unwrap());
                    }
                })
            },
        );
    }

    group.finish();
}

// ── Batch-of-1 regression benchmarks ───────────────────────────────────────

fn bench_batch_of_1_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_of_1_regression");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    let dim = 128;
    let bits = 3;
    let pq = PolarQuant::new(dim, bits, 42).unwrap();

    // --- Quantize: single vs batch-of-1 ---
    let vec_data = sine_vec(dim, 0.1);

    group.bench_function("quantize_single", |b| {
        b.iter(|| pq.quantize(black_box(&vec_data)).unwrap())
    });

    group.bench_function("quantize_batch_1", |b| {
        let batch = vec![vec_data.clone()];
        b.iter(|| pq.batch_quantize(black_box(&batch)).unwrap())
    });

    // --- Inner product: single vs batch-of-1 ---
    let query = sine_vec(dim, 0.07);
    let key = pq.quantize(&sine_vec(dim, 0.1)).unwrap();

    group.bench_function("inner_product_single", |b| {
        b.iter(|| pq.inner_product(black_box(&query), black_box(&key)).unwrap())
    });

    group.bench_function("inner_product_batch_1", |b| {
        let keys = vec![key.clone()];
        b.iter(|| pq.batch_inner_product(black_box(&query), black_box(&keys)).unwrap())
    });

    // --- Attend: single vs batch-of-1 ---
    let mut cache = KvCache::new(dim, bits, 42, 99).unwrap();
    fill_cache(&mut cache, 128, dim);

    group.bench_function("attend_single", |b| {
        b.iter(|| cache.attend(black_box(&query)).unwrap())
    });

    group.bench_function("attend_batch_1", |b| {
        let queries = vec![query.clone()];
        b.iter(|| cache.batch_attend(black_box(&queries)).unwrap())
    });

    group.finish();
}

// ── SIMD vs Scalar comparison benchmarks ───────────────────────────────────

#[cfg(feature = "simd")]
fn bench_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar");
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));

    let scalar = ScalarBackend;
    let simd = SimdBackend;

    // FWHT comparison across dimensions
    for dim in [16, 32, 64, 128, 256, 512] {
        let data_template: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();

        group.bench_with_input(
            BenchmarkId::new("fwht_scalar", dim),
            &dim,
            |b, _| {
                let mut data = data_template.clone();
                b.iter(|| {
                    // Reset data each iteration for fair comparison
                    data.copy_from_slice(&data_template);
                    scalar.fwht_normalized_inplace(black_box(&mut data));
                    black_box(&data);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("fwht_simd", dim),
            &dim,
            |b, _| {
                let mut data = data_template.clone();
                b.iter(|| {
                    data.copy_from_slice(&data_template);
                    simd.fwht_normalized_inplace(black_box(&mut data));
                    black_box(&data);
                })
            },
        );
    }

    // Dot product comparison across dimensions
    for dim in [16, 32, 64, 128, 256, 512] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let b_vec: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.07).cos()).collect();

        group.bench_with_input(
            BenchmarkId::new("dot_scalar", dim),
            &dim,
            |b, _| b.iter(|| scalar.dot_product(black_box(&a), black_box(&b_vec))),
        );

        group.bench_with_input(
            BenchmarkId::new("dot_simd", dim),
            &dim,
            |b, _| b.iter(|| simd.dot_product(black_box(&a), black_box(&b_vec))),
        );
    }

    group.finish();
}

#[cfg(feature = "simd")]
criterion_group!(
    benches,
    bench_attention_e2e,
    bench_inner_product_throughput,
    bench_quantize_throughput,
    bench_batch_of_1_regression,
    bench_simd_vs_scalar
);

#[cfg(not(feature = "simd"))]
criterion_group!(
    benches,
    bench_attention_e2e,
    bench_inner_product_throughput,
    bench_quantize_throughput,
    bench_batch_of_1_regression
);

criterion_main!(benches);
