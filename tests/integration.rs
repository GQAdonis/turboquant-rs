use turboquant::{cosine_similarity, dot_product, l2_norm, KvCache, PolarQuant, TurboQuant};

fn sine_vec(dim: usize, freq: f32) -> Vec<f32> {
    (0..dim).map(|i| (i as f32 * freq).sin()).collect()
}

// ── PolarQuant compression quality ──────────────────────────────────────────
// Empirically calibrated on dim=128 synthetic sine vectors.
// Real LLM KV-cache tensors (more random) typically score higher.

#[test]
fn polar_quant_2bit_cosine() {
    let pq    = PolarQuant::new(128, 2, 1).unwrap();
    let orig  = sine_vec(128, 0.1);
    let recon = pq.dequantize(&pq.quantize(&orig).unwrap()).unwrap();
    let cos   = cosine_similarity(&orig, &recon);
    assert!(cos > 0.90, "2-bit cosine={cos:.4} (expected > 0.90)");
}

#[test]
fn polar_quant_3bit_cosine() {
    let pq    = PolarQuant::new(128, 3, 2).unwrap();
    let orig  = sine_vec(128, 0.07);
    let recon = pq.dequantize(&pq.quantize(&orig).unwrap()).unwrap();
    let cos   = cosine_similarity(&orig, &recon);
    assert!(cos > 0.97, "3-bit cosine={cos:.4} (expected > 0.97)");
}

#[test]
fn polar_quant_4bit_cosine() {
    let pq    = PolarQuant::new(128, 4, 3).unwrap();
    let orig  = sine_vec(128, 0.11);
    let recon = pq.dequantize(&pq.quantize(&orig).unwrap()).unwrap();
    let cos   = cosine_similarity(&orig, &recon);
    assert!(cos > 0.99, "4-bit cosine={cos:.4} (expected > 0.99)");
}

#[test]
fn polar_quant_byte_budgets() {
    // 4 bytes norm + ceil(128*3/8)=48 bytes packed = 52 total
    let pq = PolarQuant::new(128, 3, 42).unwrap();
    let qv = pq.quantize(&sine_vec(128, 0.1)).unwrap();
    assert_eq!(qv.byte_size(), 52);
}

#[test]
fn polar_quant_norm_preserved() {
    let pq    = PolarQuant::new(128, 3, 7).unwrap();
    let orig  = sine_vec(128, 0.09);
    let recon = pq.dequantize(&pq.quantize(&orig).unwrap()).unwrap();
    let err   = (l2_norm(&orig) - l2_norm(&recon)).abs() / l2_norm(&orig);
    assert!(err < 0.05, "norm error={:.1}% (expected < 5%)", err * 100.0);
}

#[test]
fn polar_quant_inner_product_matches_full_dequant() {
    let pq    = PolarQuant::new(128, 4, 13).unwrap();
    let key   = sine_vec(128, 0.13);
    let query = sine_vec(128, 0.09);
    let qv    = pq.quantize(&key).unwrap();
    let ip_recon  = dot_product(&pq.dequantize(&qv).unwrap(), &query);
    let ip_direct = pq.inner_product(&query, &qv).unwrap();
    assert!((ip_recon - ip_direct).abs() < 1e-4,
        "methods disagree: {ip_recon} vs {ip_direct}");
}

// ── TurboQuant ────────────────────────────────────────────────────────────────

#[test]
fn turboquant_mse_prod_comparable_accuracy() {
    let tq    = TurboQuant::new(128, 3, 42).unwrap();
    let key   = sine_vec(128, 0.05);
    let query = sine_vec(128, 0.07);
    let true_ip = dot_product(&key, &query);
    let max_ip  = l2_norm(&key) * l2_norm(&query);

    let mse  = tq.compress_mse(&key).unwrap();
    let prod = tq.compress_prod(&key).unwrap();
    let err_m = (tq.inner_product_mse(&query, &mse).unwrap() - true_ip).abs() / max_ip;
    let err_p = (tq.inner_product_prod(&query, &prod).unwrap() - true_ip).abs() / max_ip;

    assert!(err_m < 0.05, "MSE IP norm-relative error: {err_m:.3}");
    assert!(err_p < 0.10, "Prod IP norm-relative error: {err_p:.3}");
}

// ── KV Cache ─────────────────────────────────────────────────────────────────

#[test]
fn kv_cache_attend_output_dim() {
    let mut cache = KvCache::new(128, 3, 42, 99).unwrap();
    for i in 0..32 {
        cache.push(&sine_vec(128, i as f32 * 0.01), &sine_vec(128, i as f32 * 0.02)).unwrap();
    }
    assert_eq!(cache.attend(&sine_vec(128, 0.05)).unwrap().len(), 128);
}

#[test]
fn kv_cache_attend_close_to_fp32() {
    let dim = 128;
    let mut tq = KvCache::new(dim, 3, 42, 99).unwrap();
    let mut keys:   Vec<Vec<f32>> = Vec::new();
    let mut values: Vec<Vec<f32>> = Vec::new();
    for i in 0..64 {
        let k = sine_vec(dim, i as f32 * 0.05);
        let v = sine_vec(dim, i as f32 * 0.03 + 0.5);
        tq.push(&k, &v).unwrap();
        keys.push(k); values.push(v);
    }
    let query = sine_vec(dim, 0.07);

    // FP32 reference.
    let logits: Vec<f32> = keys.iter().map(|k| dot_product(&query, k)).collect();
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let w: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
    let mut fp_out = vec![0.0f32; dim];
    for (&wi, v) in w.iter().zip(&values) {
        for (o, &vi) in fp_out.iter_mut().zip(v.iter()) { *o += wi * vi; }
    }

    let tq_out = tq.attend(&query).unwrap();
    let cos = cosine_similarity(&fp_out, &tq_out);
    assert!(cos > 0.97, "attention cosine={cos:.4} (expected > 0.97)");
}

#[test]
fn kv_cache_compression_ratio_3bit() {
    let mut cache = KvCache::new(128, 3, 1, 2).unwrap();
    for i in 0..256 {
        cache.push(&sine_vec(128, i as f32 * 0.01), &sine_vec(128, i as f32 * 0.01 + 1.0)).unwrap();
    }
    assert!(cache.compression_ratio() > 5.0, "ratio={:.2}", cache.compression_ratio());
}
