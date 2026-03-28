//! GPU acceleration benchmarks for TurboQuant.
//!
//! Run with: cargo bench --features gpu --bench gpu_bench
//! Requires NVIDIA GPU + CUDA toolkit.

#![cfg(feature = "gpu")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use turboquant::{GpuBackend, KvCache, PolarQuant};

fn sine_vec(dim: usize, freq: f32) -> Vec<f32> {
    (0..dim).map(|i| (i as f32 * freq).sin()).collect()
}

fn gpu_vs_cpu_batch_quantize(c: &mut Criterion) {
    let gpu = match GpuBackend::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping GPU benchmarks: {}", e);
            return;
        }
    };

    let dim = 128;
    let bits = 3;
    let seed = 42u64;

    let pq_cpu = PolarQuant::new(dim, bits, seed).unwrap();
    let pq_gpu = PolarQuant::new_with_backend(dim, bits, seed, gpu).unwrap();

    let mut group = c.benchmark_group("batch_quantize_gpu_vs_cpu");

    for batch_size in [16, 32, 64, 128] {
        let vecs: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| sine_vec(dim, 0.1 + i as f32 * 0.001))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("cpu", batch_size),
            &vecs,
            |b, vecs| b.iter(|| pq_cpu.batch_quantize(vecs).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("gpu", batch_size),
            &vecs,
            |b, vecs| b.iter(|| pq_gpu.batch_quantize_dispatch(vecs).unwrap()),
        );
    }
    group.finish();
}

fn gpu_vs_cpu_batch_inner_product(c: &mut Criterion) {
    let gpu = match GpuBackend::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping GPU benchmarks: {}", e);
            return;
        }
    };

    let dim = 128;
    let bits = 3;
    let seed = 42u64;

    let pq_cpu = PolarQuant::new(dim, bits, seed).unwrap();
    let pq_gpu = PolarQuant::new_with_backend(dim, bits, seed, gpu).unwrap();

    let query = sine_vec(dim, 0.05);

    let mut group = c.benchmark_group("batch_inner_product_gpu_vs_cpu");

    for batch_size in [16, 32, 64, 128, 256] {
        let keys: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| sine_vec(dim, 0.1 + i as f32 * 0.001))
            .collect();
        let qvs: Vec<_> = keys.iter().map(|k| pq_cpu.quantize(k).unwrap()).collect();

        group.bench_with_input(
            BenchmarkId::new("cpu", batch_size),
            &qvs,
            |b, qvs| b.iter(|| pq_cpu.batch_inner_product(&query, qvs).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("gpu", batch_size),
            &qvs,
            |b, qvs| b.iter(|| pq_gpu.batch_inner_product_dispatch(&query, qvs).unwrap()),
        );
    }
    group.finish();
}

fn gpu_vs_cpu_kvcache_attend(c: &mut Criterion) {
    let gpu = match GpuBackend::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping GPU benchmarks: {}", e);
            return;
        }
    };

    let dim = 128;
    let bits = 3;

    let mut group = c.benchmark_group("kvcache_attend_gpu_vs_cpu");

    for seq_len in [32, 64, 128, 256] {
        let mut cache_cpu = KvCache::new(dim, bits, 42, 99).unwrap();
        let mut cache_gpu = KvCache::new_with_backend(dim, bits, 42, 99, gpu.clone()).unwrap();

        for i in 0..seq_len {
            let k = sine_vec(dim, 0.01 * (i as f32 + 1.0));
            let v = sine_vec(dim, 0.02 * (i as f32 + 1.0));
            cache_cpu.push(&k, &v).unwrap();
            cache_gpu.push(&k, &v).unwrap();
        }

        let query = sine_vec(dim, 0.05);

        group.bench_with_input(BenchmarkId::new("cpu", seq_len), &query, |b, q| {
            b.iter(|| cache_cpu.attend(q).unwrap())
        });

        group.bench_with_input(BenchmarkId::new("gpu", seq_len), &query, |b, q| {
            b.iter(|| cache_gpu.attend_gpu(q).unwrap())
        });
    }
    group.finish();
}

criterion_group!(
    gpu_benches,
    gpu_vs_cpu_batch_quantize,
    gpu_vs_cpu_batch_inner_product,
    gpu_vs_cpu_kvcache_attend
);
criterion_main!(gpu_benches);
