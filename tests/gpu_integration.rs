//! GPU integration tests — verify GPU batch results match CPU batch results.
//!
//! Run with: cargo test --features gpu --release --test gpu_integration
//! These tests require an NVIDIA GPU with CUDA toolkit installed.

#![cfg(feature = "gpu")]

use turboquant::{GpuBackend, KvCache, PolarQuant, ScalarBackend};

fn gpu_or_skip() -> GpuBackend {
    match GpuBackend::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Skipping GPU test: {}", e);
            std::process::exit(0); // Skip gracefully
        }
    }
}

fn sine_vec(dim: usize, freq: f32) -> Vec<f32> {
    (0..dim).map(|i| (i as f32 * freq).sin()).collect()
}

#[test]
fn gpu_batch_quantize_matches_cpu() {
    let gpu = gpu_or_skip();
    let dim = 128;
    let bits = 3;
    let seed = 42u64;

    let pq_cpu = PolarQuant::new(dim, bits, seed).unwrap();
    let pq_gpu = PolarQuant::new_with_backend(dim, bits, seed, gpu).unwrap();

    let vecs: Vec<Vec<f32>> = (0..64)
        .map(|i| sine_vec(dim, 0.1 + i as f32 * 0.001))
        .collect();

    let cpu_result = pq_cpu.batch_quantize(&vecs).unwrap();
    let gpu_result = pq_gpu.batch_quantize_dispatch(&vecs).unwrap();

    assert_eq!(cpu_result.len(), gpu_result.len());
    for (i, (cpu_qv, gpu_qv)) in cpu_result.iter().zip(&gpu_result).enumerate() {
        assert!(
            (cpu_qv.norm - gpu_qv.norm).abs() < 1e-6,
            "Norm mismatch at index {i}: CPU={}, GPU={}",
            cpu_qv.norm,
            gpu_qv.norm
        );
        // Packed indices should match (same quantization)
        assert_eq!(
            cpu_qv.packed, gpu_qv.packed,
            "Packed indices mismatch at index {i}"
        );
    }
}

#[test]
fn gpu_batch_inner_product_matches_cpu() {
    let gpu = gpu_or_skip();
    let dim = 128;
    let bits = 3;
    let seed = 42u64;

    let pq_cpu = PolarQuant::new(dim, bits, seed).unwrap();
    let pq_gpu = PolarQuant::new_with_backend(dim, bits, seed, gpu).unwrap();

    let query = sine_vec(dim, 0.05);
    let keys: Vec<Vec<f32>> = (0..64)
        .map(|i| sine_vec(dim, 0.1 + i as f32 * 0.001))
        .collect();
    let qvs: Vec<_> = keys.iter().map(|k| pq_cpu.quantize(k).unwrap()).collect();

    let cpu_ips = pq_cpu.batch_inner_product(&query, &qvs).unwrap();
    let gpu_ips = pq_gpu.batch_inner_product_dispatch(&query, &qvs).unwrap();

    assert_eq!(cpu_ips.len(), gpu_ips.len());
    for (i, (cpu_ip, gpu_ip)) in cpu_ips.iter().zip(&gpu_ips).enumerate() {
        assert!(
            (cpu_ip - gpu_ip).abs() < 1e-3,
            "Inner product mismatch at index {i}: CPU={cpu_ip}, GPU={gpu_ip}"
        );
    }
}

#[test]
fn gpu_small_batch_uses_cpu_fallback() {
    let gpu = gpu_or_skip();
    let dim = 128;
    let bits = 3;
    let seed = 42u64;

    let pq_cpu = PolarQuant::new(dim, bits, seed).unwrap();
    let pq_gpu = PolarQuant::new_with_backend(dim, bits, seed, gpu).unwrap();

    // 16 vectors < GPU_BATCH_THRESHOLD (32), should use CPU path
    let vecs: Vec<Vec<f32>> = (0..16)
        .map(|i| sine_vec(dim, 0.1 + i as f32 * 0.01))
        .collect();

    let cpu_result = pq_cpu.batch_quantize(&vecs).unwrap();
    let gpu_result = pq_gpu.batch_quantize_dispatch(&vecs).unwrap();

    // Should be identical (both use CPU path)
    for (cpu_qv, gpu_qv) in cpu_result.iter().zip(&gpu_result) {
        assert_eq!(cpu_qv.norm, gpu_qv.norm);
        assert_eq!(cpu_qv.packed, gpu_qv.packed);
    }
}

#[test]
fn gpu_kvcache_attend_matches_cpu() {
    let gpu = gpu_or_skip();
    let dim = 128;
    let bits = 3;

    let mut cache_cpu = KvCache::new(dim, bits, 42, 99).unwrap();
    let mut cache_gpu = KvCache::new_with_backend(dim, bits, 42, 99, gpu).unwrap();

    // Fill both caches identically
    for i in 0..64 {
        let k = sine_vec(dim, 0.01 * (i as f32 + 1.0));
        let v = sine_vec(dim, 0.02 * (i as f32 + 1.0));
        cache_cpu.push(&k, &v).unwrap();
        cache_gpu.push(&k, &v).unwrap();
    }

    let query = sine_vec(dim, 0.05);
    let cpu_out = cache_cpu.attend(&query).unwrap();
    let gpu_out = cache_gpu.attend_gpu(&query).unwrap();

    assert_eq!(cpu_out.len(), gpu_out.len());
    for (i, (c, g)) in cpu_out.iter().zip(&gpu_out).enumerate() {
        assert!(
            (c - g).abs() < 1e-2,
            "Attend mismatch at dim {i}: CPU={c}, GPU={g}"
        );
    }
}

#[test]
fn gpu_batch_dispatch_threshold() {
    let gpu = gpu_or_skip();
    let dim = 128;
    let bits = 3;
    let seed = 42u64;

    let pq = PolarQuant::new_with_backend(dim, bits, seed, gpu).unwrap();

    // Test at exact threshold boundary
    let vecs_at_threshold: Vec<Vec<f32>> = (0..32)
        .map(|i| sine_vec(dim, 0.1 + i as f32 * 0.001))
        .collect();
    let vecs_below: Vec<Vec<f32>> = (0..31)
        .map(|i| sine_vec(dim, 0.1 + i as f32 * 0.001))
        .collect();

    // Both should succeed regardless of path
    let result_at = pq.batch_quantize_dispatch(&vecs_at_threshold).unwrap();
    let result_below = pq.batch_quantize_dispatch(&vecs_below).unwrap();

    assert_eq!(result_at.len(), 32);
    assert_eq!(result_below.len(), 31);
}

#[test]
fn gpu_multiple_dimensions() {
    let gpu = gpu_or_skip();
    let bits = 3;
    let seed = 42u64;

    // Test dimensions 64, 128, 256
    for dim in [64, 128, 256] {
        let pq_cpu = PolarQuant::new(dim, bits, seed).unwrap();
        let pq_gpu = PolarQuant::new_with_backend(dim, bits, seed, gpu.clone()).unwrap();

        let vecs: Vec<Vec<f32>> = (0..32)
            .map(|i| sine_vec(dim, 0.1 + i as f32 * 0.001))
            .collect();

        let cpu_result = pq_cpu.batch_quantize(&vecs).unwrap();
        let gpu_result = pq_gpu.batch_quantize_dispatch(&vecs).unwrap();

        for (j, (c, g)) in cpu_result.iter().zip(&gpu_result).enumerate() {
            assert_eq!(c.packed, g.packed, "dim={dim} vec={j}: packed mismatch");
        }
    }
}
