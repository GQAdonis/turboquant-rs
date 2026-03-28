// Batch Fast Walsh-Hadamard Transform for TurboQuant GPU backend.
// Each thread block processes one vector from the batch.
// Uses shared memory for fast butterfly operations.

extern "C" __global__ void fwht_batch(
    float* data,        // [batch_size * dim] flattened row-major
    const int dim,      // Must be power of two (64, 128, 256)
    const int batch_size
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    float* vec = data + batch_idx * dim;

    // Shared memory for this block's vector
    extern __shared__ float shared[];

    // Load vector into shared memory (coalesced)
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        shared[i] = vec[i];
    }
    __syncthreads();

    // Butterfly steps: step = 1, 2, 4, ..., dim/2
    for (int step = 1; step < dim; step *= 2) {
        // Each thread handles one or more butterfly pairs
        int half_group = step;
        int group_size = step * 2;

        for (int idx = threadIdx.x; idx < dim / 2; idx += blockDim.x) {
            // Map linear index to butterfly pair
            int group = idx / half_group;
            int pos = idx % half_group;
            int i = group * group_size + pos;
            int j = i + step;

            float a = shared[i];
            float b = shared[j];
            shared[i] = a + b;
            shared[j] = a - b;
        }
        __syncthreads();
    }

    // Normalize by 1/sqrt(dim)
    float norm = rsqrtf((float)dim);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        vec[i] = shared[i] * norm;
    }
}
