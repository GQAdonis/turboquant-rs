use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use turboquant::backend::{Backend, ScalarBackend};
use turboquant::{PolarQuant, TurboQuant};

#[cfg(feature = "simd")]
use turboquant::{Avx512Backend, RuntimeBackend, SimdBackend};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn sine_vec(dim: usize, freq: f32) -> Vec<f32> {
    (0..dim).map(|i| (i as f32 * freq).sin()).collect()
}

fn make_sign_masks(dim: usize) -> Vec<u32> {
    (0..dim)
        .map(|i| if i % 3 == 0 { 0x8000_0000u32 } else { 0 })
        .collect()
}

// ── Backend comparison: FWHT ─────────────────────────────────────────────────

fn bench_fwht_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("fwht_backend_comparison");

    for dim in [64usize, 128, 256, 512] {
        let data = sine_vec(dim, 0.13);

        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |b, _| {
            let backend = ScalarBackend;
            b.iter(|| {
                let mut d = data.clone();
                backend.fwht_normalized_inplace(black_box(&mut d));
                d
            });
        });

        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd_avx2", dim), &dim, |b, _| {
            let backend = SimdBackend;
            b.iter(|| {
                let mut d = data.clone();
                backend.fwht_normalized_inplace(black_box(&mut d));
                d
            });
        });

        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("avx512", dim), &dim, |b, _| {
            let backend = Avx512Backend;
            b.iter(|| {
                let mut d = data.clone();
                backend.fwht_normalized_inplace(black_box(&mut d));
                d
            });
        });

        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("runtime_best", dim), &dim, |b, _| {
            let backend = RuntimeBackend::best_available();
            b.iter(|| {
                let mut d = data.clone();
                backend.fwht_normalized_inplace(black_box(&mut d));
                d
            });
        });
    }
    group.finish();
}

// ── Backend comparison: dot product ──────────────────────────────────────────

fn bench_dot_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product_backend_comparison");

    for dim in [64usize, 128, 256, 512] {
        let a = sine_vec(dim, 0.11);
        let b = sine_vec(dim, 0.17);

        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            let backend = ScalarBackend;
            bench.iter(|| backend.dot_product(black_box(&a), black_box(&b)));
        });

        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd_avx2", dim), &dim, |bench, _| {
            let backend = SimdBackend;
            bench.iter(|| backend.dot_product(black_box(&a), black_box(&b)));
        });

        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("avx512", dim), &dim, |bench, _| {
            let backend = Avx512Backend;
            bench.iter(|| backend.dot_product(black_box(&a), black_box(&b)));
        });

        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("runtime_best", dim), &dim, |bench, _| {
            let backend = RuntimeBackend::best_available();
            bench.iter(|| backend.dot_product(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

// ── Backend comparison: sign-flip (apply_signs) ───────────────────────────

fn bench_apply_signs_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_signs_backend_comparison");

    for dim in [64usize, 128, 256, 512] {
        let data = sine_vec(dim, 0.23);
        let masks = make_sign_masks(dim);

        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |b, _| {
            let backend = ScalarBackend;
            b.iter(|| {
                let mut d = data.clone();
                backend.apply_signs(black_box(&mut d), black_box(&masks));
                d
            });
        });

        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("simd_avx2", dim), &dim, |b, _| {
            let backend = SimdBackend;
            b.iter(|| {
                let mut d = data.clone();
                backend.apply_signs(black_box(&mut d), black_box(&masks));
                d
            });
        });

        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("avx512", dim), &dim, |b, _| {
            let backend = Avx512Backend;
            b.iter(|| {
                let mut d = data.clone();
                backend.apply_signs(black_box(&mut d), black_box(&masks));
                d
            });
        });

        #[cfg(feature = "simd")]
        group.bench_with_input(BenchmarkId::new("runtime_best", dim), &dim, |b, _| {
            let backend = RuntimeBackend::best_available();
            b.iter(|| {
                let mut d = data.clone();
                backend.apply_signs(black_box(&mut d), black_box(&masks));
                d
            });
        });
    }
    group.finish();
}

// ── PolarQuant end-to-end: scalar vs runtime best ────────────────────────────

fn bench_polar_quant_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("polar_quant_backend_comparison");

    for (bits, dim) in [(3u8, 128usize), (4, 128), (3, 256)] {
        let key   = sine_vec(dim, 0.10);
        let query = sine_vec(dim, 0.07);

        let pq_scalar = PolarQuant::new_with_backend(dim, bits, 42, ScalarBackend).unwrap();
        let qv_scalar = pq_scalar.quantize(&key).unwrap();

        let id = format!("{bits}bit-{dim}dim");

        group.bench_with_input(BenchmarkId::new("quantize/scalar", &id), &id, |b, _| {
            b.iter(|| pq_scalar.quantize(black_box(&key)).unwrap())
        });
        group.bench_with_input(BenchmarkId::new("inner_product/scalar", &id), &id, |b, _| {
            b.iter(|| pq_scalar.inner_product(black_box(&query), black_box(&qv_scalar)).unwrap())
        });

        #[cfg(feature = "simd")]
        {
            let pq_simd = PolarQuant::new_with_backend(dim, bits, 42, RuntimeBackend::best_available()).unwrap();
            let qv_simd = pq_simd.quantize(&key).unwrap();

            group.bench_with_input(BenchmarkId::new("quantize/simd", &id), &id, |b, _| {
                b.iter(|| pq_simd.quantize(black_box(&key)).unwrap())
            });
            group.bench_with_input(BenchmarkId::new("inner_product/simd", &id), &id, |b, _| {
                b.iter(|| pq_simd.inner_product(black_box(&query), black_box(&qv_simd)).unwrap())
            });
        }
    }
    group.finish();
}

// ── TurboQuant end-to-end: scalar vs default (runtime best) ──────────────────

fn bench_turboquant_default(c: &mut Criterion) {
    let mut group = c.benchmark_group("turboquant_default_backend");

    // TurboQuant::new() now uses DefaultBackend (RuntimeBackend with simd feature)
    let tq = TurboQuant::new(128, 3, 42).unwrap();
    let key   = sine_vec(128, 0.10);
    let query = sine_vec(128, 0.07);
    let mse   = tq.compress_mse(&key).unwrap();

    group.bench_function("compress_mse/default", |b| {
        b.iter(|| tq.compress_mse(black_box(&key)).unwrap())
    });
    group.bench_function("inner_product_mse/default", |b| {
        b.iter(|| tq.inner_product_mse(black_box(&query), black_box(&mse)).unwrap())
    });

    group.finish();
}

// ── DefaultBackend type sanity ────────────────────────────────────────────────

fn bench_default_backend(c: &mut Criterion) {
    let mut group = c.benchmark_group("default_backend_fwht");

    for dim in [128usize, 256] {
        let data = sine_vec(dim, 0.19);
        // DefaultBackend = RuntimeBackend when simd feature is on
        #[cfg(feature = "simd")]
        let backend = RuntimeBackend::best_available();
        #[cfg(not(feature = "simd"))]
        let backend = ScalarBackend;
        group.bench_with_input(BenchmarkId::new("fwht", dim), &dim, |b, _| {
            b.iter(|| {
                let mut d = data.clone();
                backend.fwht_normalized_inplace(black_box(&mut d));
                d
            });
        });
    }
    group.finish();
}

criterion_group!(
    backend_benches,
    bench_fwht_backends,
    bench_dot_backends,
    bench_apply_signs_backends,
    bench_polar_quant_backends,
    bench_turboquant_default,
    bench_default_backend,
);
criterion_main!(backend_benches);
