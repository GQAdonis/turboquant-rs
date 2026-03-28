use turboquant::{cosine_similarity, dot_product, KvCache, PolarQuant, TurboQuant};

fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║          TurboQuant — Pure Rust Demo             ║");
    println!("╚══════════════════════════════════════════════════╝\n");

    let dim = 128;

    // ── 1. PolarQuant standalone ────────────────────────────────────────────
    println!("┌─ PolarQuant (dim={dim}) ─────────────────────────────┐");
    for bits in [2u8, 3, 4] {
        let pq    = PolarQuant::new(dim, bits, 42).unwrap();
        let key   = sine_vec(dim, 0.13);
        let query = sine_vec(dim, 0.09);

        let true_ip = dot_product(&key, &query);
        let qv      = pq.quantize(&key).unwrap();
        let recon   = pq.dequantize(&qv).unwrap();
        let est_ip  = pq.inner_product(&query, &qv).unwrap();
        let cos_sim = cosine_similarity(&key, &recon);

        println!(
            "  {bits}-bit | mem {:3}B → {:2}B ({:.1}x) | cos_sim {:.4} | IP err {:.2}%",
            dim * 4,
            qv.byte_size(),
            qv.compression_ratio(),
            cos_sim,
            (est_ip - true_ip).abs() / true_ip.abs() * 100.0,
        );
    }
    println!();

    // ── 2. TurboQuant MSE vs Prod ───────────────────────────────────────────
    println!("┌─ TurboQuant MSE vs Prod (3-bit, dim={dim}) ─────────┐");
    let tq      = TurboQuant::new(dim, 3, 42).unwrap();
    let key     = sine_vec(dim, 0.17);
    let query   = sine_vec(dim, 0.11);
    let true_ip = dot_product(&key, &query);

    let mse_qv  = tq.compress_mse(&key).unwrap();
    let prod_qv = tq.compress_prod(&key).unwrap();
    let mse_ip  = tq.inner_product_mse(&query, &mse_qv).unwrap();
    let prod_ip = tq.inner_product_prod(&query, &prod_qv).unwrap();

    println!("  True IP:  {true_ip:.6}");
    println!(
        "  MSE  ({:2}B):  {mse_ip:.6}  err {:.2}%",
        mse_qv.byte_size(),
        (mse_ip - true_ip).abs() / true_ip.abs() * 100.0,
    );
    println!(
        "  Prod ({:2}B):  {prod_ip:.6}  err {:.2}%",
        prod_qv.byte_size(),
        (prod_ip - true_ip).abs() / true_ip.abs() * 100.0,
    );
    println!();

    // ── 3. KV Cache ─────────────────────────────────────────────────────────
    println!("┌─ KV Cache (3-bit, dim={dim}) ──────────────────────────┐");
    for &seq_len in &[128usize, 512, 2048, 8192] {
        let mut cache = KvCache::new(dim, 3, 42, 99).unwrap();
        for i in 0..seq_len {
            let k = sine_vec_offset(dim, 0.01, i);
            let v = cos_vec_offset(dim, 0.01, i);
            cache.push(&k, &v).unwrap();
        }
        println!(
            "  seq={seq_len:5}  fp32 {:5.1} MB → compressed {:4.2} MB  ({:.1}x)",
            cache.uncompressed_bytes() as f32 / 1e6,
            cache.compressed_bytes() as f32 / 1e6,
            cache.compression_ratio(),
        );
    }
    println!();

    // ── 4. Attention quality sanity check ───────────────────────────────────
    println!("┌─ Attention quality sanity check ────────────────────┐");
    let mut cache_fp = KvCacheFp32::new(dim);
    let mut cache_tq = KvCache::new(dim, 3, 42, 99).unwrap();

    for i in 0..64 {
        let k = sine_vec_offset(dim, 0.05, i);
        let v = cos_vec_offset(dim, 0.05, i);
        cache_fp.push(k.clone(), v.clone());
        cache_tq.push(&k, &v).unwrap();
    }

    let query = sine_vec(dim, 0.07);
    let out_fp = cache_fp.attend(&query);
    let out_tq = cache_tq.attend(&query).unwrap();

    let cos = cosine_similarity(&out_fp, &out_tq);
    println!("  Attention output cosine similarity (fp32 vs 3-bit): {cos:.4}");
    println!("  (above 0.99 indicates near-lossless compression)\n");
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn sine_vec(dim: usize, freq: f32) -> Vec<f32> {
    (0..dim).map(|i| (i as f32 * freq).sin()).collect()
}

fn sine_vec_offset(dim: usize, freq: f32, offset: usize) -> Vec<f32> {
    (0..dim).map(|i| ((i + offset * dim) as f32 * freq).sin()).collect()
}

fn cos_vec_offset(dim: usize, freq: f32, offset: usize) -> Vec<f32> {
    (0..dim).map(|i| ((i + offset * dim) as f32 * freq).cos()).collect()
}

/// Uncompressed fp32 KV cache for quality comparison.
struct KvCacheFp32 {
    keys:   Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

impl KvCacheFp32 {
    fn new(_dim: usize) -> Self { Self { keys: vec![], values: vec![] } }

    fn push(&mut self, k: Vec<f32>, v: Vec<f32>) {
        self.keys.push(k);
        self.values.push(v);
    }

    fn attend(&self, query: &[f32]) -> Vec<f32> {
        let logits: Vec<f32> = self.keys.iter().map(|k| dot_product(query, k)).collect();
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let weights: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

        let dim = self.values[0].len();
        let mut out = vec![0.0f32; dim];
        for (&w, v) in weights.iter().zip(&self.values) {
            for (o, &vi) in out.iter_mut().zip(v.iter()) {
                *o += w * vi;
            }
        }
        out
    }
}
