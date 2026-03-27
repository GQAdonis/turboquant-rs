//! Integration benchmarks for realistic attention workloads.
//!
//! These benchmarks measure end-to-end performance of the attention hot path
//! at multiple sequence lengths to establish baselines for optimization work.
//!
//! Run with: `cargo bench --bench integration`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use turboquant::{KvCache, PolarQuant};

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

criterion_group!(
    benches,
    bench_attention_e2e,
    bench_inner_product_throughput,
    bench_quantize_throughput
);
criterion_main!(benches);
