use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    #[cfg(feature = "gpu")]
    compile_cuda_kernels();
}

#[cfg(feature = "gpu")]
fn compile_cuda_kernels() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let kernels = ["fwht", "attention"];

    for kernel in &kernels {
        let cu_path = format!("src/kernels/{}.cu", kernel);
        let ptx_path = format!("{}/{}.ptx", out_dir, kernel);

        println!("cargo:rerun-if-changed={}", cu_path);

        if !Path::new(&cu_path).exists() {
            panic!("CUDA kernel source not found: {}", cu_path);
        }

        let status = Command::new("nvcc")
            .args(&[
                "--ptx",
                "-o",
                &ptx_path,
                &cu_path,
                "-arch=sm_70", // Volta+ (covers T4, A100, consumer RTX)
                "--use_fast_math",
            ])
            .status();

        match status {
            Ok(s) if s.success() => {}
            Ok(s) => panic!("nvcc failed to compile {}.cu with exit code: {}", kernel, s),
            Err(e) => panic!(
                "Failed to run nvcc for {}.cu: {}.\n\
                 Ensure CUDA Toolkit is installed: https://developer.nvidia.com/cuda-downloads\n\
                 Verify: nvcc --version",
                kernel, e
            ),
        }
    }
}
