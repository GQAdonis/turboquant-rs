fn main() {
    // GPU kernel compilation is handled conditionally.
    // When the `gpu` feature is enabled and CUDA toolkit is available,
    // this will compile .cu kernels to PTX and embed them.
    #[cfg(feature = "gpu")]
    {
        // Phase 4 Plan 02 will add actual kernel compilation here.
        // For now, just print a cargo directive so rebuilds trigger on kernel changes.
        println!("cargo:rerun-if-changed=src/kernels/");
    }
}
