// Batch dot product: one query against many keys.
// Each thread block computes one dot product, then reduces.
// Output: results[i] = dot(query, keys[i*dim..(i+1)*dim]) * norms[i]

extern "C" __global__ void batch_dot_product(
    const float* query,   // [dim]
    const float* keys,    // [batch_size * dim] dequantized key vectors
    const float* norms,   // [batch_size] key norms
    float* results,       // [batch_size] output logits
    const int dim,
    const int batch_size
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* key = keys + batch_idx * dim;

    // Shared memory for partial sums within this block
    extern __shared__ float partial[];

    // Each thread accumulates partial dot product
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        sum += query[i] * key[i];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes result, scaled by key norm
    if (threadIdx.x == 0) {
        results[batch_idx] = partial[0] * norms[batch_idx];
    }
}

// Batch dequantize: convert codebook indices to float values.
// Each thread block processes one vector from the batch.
extern "C" __global__ void batch_dequantize(
    const unsigned char* packed_indices, // [batch_size * packed_bytes]
    const float* centroids,             // [num_centroids] codebook values
    float* output,                       // [batch_size * dim]
    const int dim,
    const int bits,
    const int packed_bytes_per_vec,
    const int batch_size
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const unsigned char* packed = packed_indices + batch_idx * packed_bytes_per_vec;
    float* out = output + batch_idx * dim;
    int num_centroids = 1 << bits;  // 2^bits
    unsigned char mask = (unsigned char)((1 << bits) - 1);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        // Extract index from bitpacked data
        int bit_offset = i * bits;
        int byte_idx = bit_offset / 8;
        int bit_shift = bit_offset % 8;

        unsigned int raw;
        if (bit_shift + bits <= 8) {
            raw = (packed[byte_idx] >> bit_shift) & mask;
        } else {
            // Index spans two bytes
            raw = ((packed[byte_idx] >> bit_shift) | (packed[byte_idx + 1] << (8 - bit_shift))) & mask;
        }

        out[i] = centroids[raw];
    }
}
