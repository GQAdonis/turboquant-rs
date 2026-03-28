use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use turboquant::{KvCache, PolarQuant, TurboQuant};

// ── Helpers ─────────────────────────────────────────────────────────────────

fn sine_vec(dim: usize, freq: f32) -> Vec<f32> {
    (0..dim).map(|i| (i as f32 * freq).sin()).collect()
}

// ── PolarQuant benchmarks ────────────────────────────────────────────────────

fn bench_polar_quant(c: &mut Criterion) {
    let mut group = c.benchmark_group("polar_quant");

    for bits in [2u8, 3, 4] {
        for dim in [64usize, 128, 256] {
            let pq    = PolarQuant::new(dim, bits, 42).unwrap();
            let vec   = sine_vec(dim, 0.1);
            let query = sine_vec(dim, 0.07);
            let qv    = pq.quantize(&vec).unwrap();

            let id = format!("{bits}bit-{dim}dim");

            group.bench_with_input(BenchmarkId::new("quantize", &id), &id, |b, _| {
                b.iter(|| pq.quantize(black_box(&vec)).unwrap())
            });

            group.bench_with_input(BenchmarkId::new("dequantize", &id), &id, |b, _| {
                b.iter(|| pq.dequantize(black_box(&qv)).unwrap())
            });

            group.bench_with_input(BenchmarkId::new("inner_product", &id), &id, |b, _| {
                b.iter(|| pq.inner_product(black_box(&query), black_box(&qv)).unwrap())
            });
        }
    }
    group.finish();
}

// ── TurboQuant benchmarks ────────────────────────────────────────────────────

fn bench_turboquant(c: &mut Criterion) {
    let mut group = c.benchmark_group("turboquant");

    for bits in [2u8, 3, 4] {
        let tq    = TurboQuant::new(128, bits, 42).unwrap();
        let key   = sine_vec(128, 0.1);
        let query = sine_vec(128, 0.07);
        let mse   = tq.compress_mse(&key).unwrap();
        let prod  = tq.compress_prod(&key).unwrap();
        let id    = format!("{bits}bit");

        group.bench_with_input(BenchmarkId::new("compress_mse", &id), &id, |b, _| {
            b.iter(|| tq.compress_mse(black_box(&key)).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("compress_prod", &id), &id, |b, _| {
            b.iter(|| tq.compress_prod(black_box(&key)).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("ip_mse", &id), &id, |b, _| {
            b.iter(|| tq.inner_product_mse(black_box(&query), black_box(&mse)).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("ip_prod", &id), &id, |b, _| {
            b.iter(|| tq.inner_product_prod(black_box(&query), black_box(&prod)).unwrap())
        });
    }
    group.finish();
}

// ── KV Cache benchmarks ──────────────────────────────────────────────────────

fn bench_kv_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache");

    for seq_len in [128usize, 512, 2048] {
        let mut cache = KvCache::new(128, 3, 42, 99).unwrap();
        for i in 0..seq_len {
            let k: Vec<f32> = (0..128).map(|j| ((i * 128 + j) as f32 * 0.01).sin()).collect();
            let v: Vec<f32> = (0..128).map(|j| ((i * 128 + j) as f32 * 0.01).cos()).collect();
            cache.push(&k, &v).unwrap();
        }
        let query = sine_vec(128, 0.05);

        group.bench_with_input(
            BenchmarkId::new("attention_logits", seq_len),
            &seq_len,
            |b, _| b.iter(|| cache.attention_logits(black_box(&query)).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("attend", seq_len),
            &seq_len,
            |b, _| b.iter(|| cache.attend(black_box(&query)).unwrap()),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_polar_quant, bench_turboquant, bench_kv_cache);
criterion_main!(benches);
