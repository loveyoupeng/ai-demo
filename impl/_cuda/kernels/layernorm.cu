// kernels/layernorm.cu — RMSNorm kernel (warp reduction, shared memory)
//
// RMSNorm computes: out = x / sqrt(mean(x^2, dim=-1) + eps) * gamma
// where mean is taken over the last dimension (embed_dim).
//
// Learning objectives:
// - Warp-level reduction: threads in a block sum their partial products
// - Shared memory: used for intermediate reduction storage
// - Coalesced access: all threads access contiguous memory addresses
// - Grid-stride loop: handles arbitrary input sizes
// - 2D kernel mapping: blockIdx.x encodes batch*seq positions
//
// Mathematical derivation
// =======================
// Let x be the input with shape (N, D) where N = batch_size * seq_len
// and D = embed_dim. Each row independently computes:
//   rms = sqrt(mean(x[row]^2) + eps)
//   out[row] = x[row] / rms * gamma
//
// Kernel configuration:
//   block_size = 256 threads, so D ≤ 256 (typically D=64..256).
//   Each thread processes one element. Grid size = N rows.
//   When D < 256, unused threads do nothing (safe with the if-guard).

__device__ float warp_reduce_sum(float val) {
    // Tree-based reduction within a warp (32 threads).
    // After this, lane 0 of the warp holds the full warp sum; the other
    // lanes hold partial sums (only lane 0 is consumed below).
    #pragma unroll
    for (int stride = 16; stride >= 1; stride /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, stride);
    }
    return val;
}

__device__ float block_reduce_sum(float val) {
    // Warp-level reduction; lane 0 of each warp holds its warp's sum.
    float warp_sum = warp_reduce_sum(val);

    // Publish each warp's partial sum (one slot per warp), then have warp 0
    // combine all partials and write the block total back to slot 0 so that
    // EVERY thread in EVERY warp reads the full block-wide sum. Publishing
    // only to warp 0 (via a single-warp __shfl_sync) would leave warps 1..N
    // with a zero total whenever block size > 32.
    extern __shared__ float shared_sums[];
    int warp_id = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0) {
        shared_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float total = 0.0f;
        for (int i = 0; i < num_warps; i++) {
            total += shared_sums[i];
        }
        if (threadIdx.x == 0) {
            shared_sums[0] = total;
        }
    }
    __syncthreads();
    return shared_sums[0];
}

__device__ double warp_reduce_sum_d(double val) {
    #pragma unroll
    for (int stride = 16; stride >= 1; stride /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, stride);
    }
    return val;
}

__device__ double block_reduce_sum_d(double val) {
    // Warp-level reduction; lane 0 of each warp holds its warp's sum.
    double warp_sum = warp_reduce_sum_d(val);

    // Same publish pattern as the float version: every warp must be able to
    // read the full block-wide total, not just warp 0.
    extern __shared__ double shared_sums_d[];
    int warp_id = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0) {
        shared_sums_d[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        double total = 0.0;
        for (int i = 0; i < num_warps; i++) {
            total += shared_sums_d[i];
        }
        if (threadIdx.x == 0) {
            shared_sums_d[0] = total;
        }
    }
    __syncthreads();
    return shared_sums_d[0];
}

// float32 forward kernel
__global__ void rmsnorm_forward_kernel(const float* x, const float* gamma,
                                       float* output, int rows, int cols,
                                       float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    // Each thread computes x_i^2 for one element of the row
    float val_sq = 0.0f;
    if (threadIdx.x < cols) {
        val_sq = x[row * cols + threadIdx.x] * x[row * cols + threadIdx.x];
    }

    // Block-wide reduction of sum of squares
    float sum_sq = block_reduce_sum(val_sq);

    // Compute RMS norm: rsqrt(sum_sq / cols + eps).
    // Every thread holds the correct block-wide sum of squares, so each
    // thread computes the same inverse RMS locally — no broadcast needed.
    float inv_rms = rsqrtf(sum_sq / cols + eps);

    if (inv_rms == 0.0f) return;

    // Compute output[rows][cols] = x[rows][cols] * inv_rms * gamma
    int col = threadIdx.x;
    if (col < cols) {
        float normalized = x[row * cols + col] * inv_rms;
        output[row * cols + col] = normalized * gamma[col];
    }
}

// float64 forward kernel
__global__ void rmsnorm_forward_f64_kernel(const double* x, const double* gamma,
                                           double* output, int rows, int cols,
                                           double eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    double val_sq = 0.0;
    if (threadIdx.x < cols) {
        val_sq = x[row * cols + threadIdx.x] * x[row * cols + threadIdx.x];
    }

    double sum_sq = block_reduce_sum_d(val_sq);

    // Every thread holds the correct block-wide sum of squares, so each
    // thread computes the same inverse RMS locally — no broadcast needed.
    double inv_rms = 1.0 / sqrt(sum_sq / cols + eps);

    if (inv_rms == 0.0) return;

    int col = threadIdx.x;
    if (col < cols) {
        double normalized = x[row * cols + col] * inv_rms;
        output[row * cols + col] = normalized * gamma[col];
    }
}