/**
 * Attention kernel — stable softmax + weighted sum.
 *
 * Implements scaled dot-product attention:
 *   attention(Q, K, V) = softmax(QK^T / sqrt(d)) @ V
 *
 * Algorithm
 * ---------
 * Memory access pattern:
 *   1. q @ k^T → scores [B, H, Sq, Sk] — PyTorch cuBLAS matmul
 *   2. softmax per row → attention_weights [B, H, Sq, Sk] — CUDA kernel
 *   3. attention_weights @ v → output [B, H, Sq, D] — CUDA kernel
 *
 * Stable softmax
 * --------------
 * For each row (one query position) of the attention matrix:
 *   max_val = max(scores[row, :])
 *   shifted = scores[row, :] - max_val
 *   exp_vals = exp(shifted)
 *   sum_exp = sum(exp_vals)
 *   output = exp_vals / sum_exp
 *
 * This avoids numerical overflow from exp(large values).
 *
 * Weighted sum
 * ------------
 * For each (batch, head, query_pos, head_dim):
 *   output[b][h][q][d] = sum_{k} attention[b][h][q][k] * v[b][h][k][d]
 *
 * Each thread handles one (batch, head, query_pos, head_dim) element.

 */

/* ============================================================ */
/*  stable softmax kernel (float32)                               */
/* ============================================================ */

__global__ void attention_softmax_f32(
    const float* scores,
    float* output,
    int total_rows,
    int num_keys
)
{
    int row = blockIdx.x;
    if (row >= total_rows) return;

    // Shared memory layout (the launcher allocates 514 floats):
    //   [0 .. 255]  : per-warp max partials
    //   [256]       : block max result
    //   [257..512]  : per-warp sum partials
    //   [513]       : block sum result
    extern __shared__ char shared_mem[];
    float* s_max = reinterpret_cast<float*>(shared_mem);
    float* s_block_max = reinterpret_cast<float*>(shared_mem + 256 * 4);
    float* s_sum = reinterpret_cast<float*>(shared_mem + 257 * 4);
    float* s_block_sum = reinterpret_cast<float*>(shared_mem + (257 + 256) * 4);

    // Step 1: each thread takes the max over the elements it owns in this row.
    float local_max = -1e9f;
    for (int i = threadIdx.x; i < num_keys; i += blockDim.x) {
        float val = scores[row * num_keys + i];
        if (val > local_max) {
            local_max = val;
        }
    }

    // Step 2: warp-level max reduction — lane 0 of each warp holds the warp max.
    float warp_max = local_max;
    for (int mask = 16; mask > 0; mask >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, warp_max, mask);
        if (other > warp_max) {
            warp_max = other;
        }
    }

    // Step 3: block-level max. Every warp publishes its partial; warp 0
    // combines them and publishes the result so ALL threads (not just warp 0)
    // read the true row max.
    int warp_id = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0) {
        s_max[warp_id] = warp_max;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float block_max = -1e9f;
        for (int i = 0; i < num_warps; i++) {
            if (s_max[i] > block_max) {
                block_max = s_max[i];
            }
        }
        s_block_max[0] = block_max;
    }
    __syncthreads();
    float block_max_val = s_block_max[0];

    // Step 4: each thread sums exp(value - block_max) over its elements.
    float local_exp_sum = 0.0f;
    for (int i = threadIdx.x; i < num_keys; i += blockDim.x) {
        float shifted = scores[row * num_keys + i] - block_max_val;
        local_exp_sum += expf(shifted);
    }

    // Step 5: warp-level sum reduction.
    float warp_sum = local_exp_sum;
    for (int mask = 16; mask > 0; mask >>= 1) {
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, mask);
    }

    // Step 6: block-level sum — same publish pattern as the max.
    if (threadIdx.x % 32 == 0) {
        s_sum[warp_id] = warp_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float total = 0.0f;
        for (int i = 0; i < num_warps; i++) {
            total += s_sum[i];
        }
        s_block_sum[0] = total;
    }
    __syncthreads();
    float block_sum = s_block_sum[0];

    // Step 7: normalize. exp is recomputed (it is cheap) so the kernel needs
    // no per-thread storage for the exponentials.
    for (int i = threadIdx.x; i < num_keys; i += blockDim.x) {
        float shifted = scores[row * num_keys + i] - block_max_val;
        output[row * num_keys + i] = expf(shifted) / (block_sum + 1e-9f);
    }
}

/* ============================================================ */
/*  Stable softmax kernel (float64)                               */
/* ============================================================ */

__global__ void attention_softmax_f64(
    const double* scores,
    double* output,
    int total_rows,
    int num_keys
)
{
    int row = blockIdx.x;
    if (row >= total_rows) return;

    // Shared memory layout (the launcher allocates 514 doubles):
    //   [0 .. 255]  : per-warp max partials
    //   [256]       : block max result
    //   [257..512]  : per-warp sum partials
    //   [513]       : block sum result
    extern __shared__ char shared_mem[];
    double* s_max = reinterpret_cast<double*>(shared_mem);
    double* s_block_max = reinterpret_cast<double*>(shared_mem + 256 * 8);
    double* s_sum = reinterpret_cast<double*>(shared_mem + 257 * 8);
    double* s_block_sum = reinterpret_cast<double*>(shared_mem + (257 + 256) * 8);

    // Step 1: each thread takes the max over the elements it owns in this row.
    double local_max = -1e18;
    for (int i = threadIdx.x; i < num_keys; i += blockDim.x) {
        double val = scores[row * num_keys + i];
        if (val > local_max) {
            local_max = val;
        }
    }

    // Step 2: warp-level max reduction — lane 0 of each warp holds the warp max.
    double warp_max = local_max;
    for (int mask = 16; mask > 0; mask >>= 1) {
        double other = __shfl_down_sync(0xFFFFFFFF, warp_max, mask);
        if (other > warp_max) {
            warp_max = other;
        }
    }

    // Step 3: block-level max — publish pattern shared with the float32 kernel.
    int warp_id = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0) {
        s_max[warp_id] = warp_max;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        double block_max = -1e18;
        for (int i = 0; i < num_warps; i++) {
            if (s_max[i] > block_max) {
                block_max = s_max[i];
            }
        }
        s_block_max[0] = block_max;
    }
    __syncthreads();
    double block_max_val = s_block_max[0];

    // Step 4: each thread sums exp(value - block_max) over its elements.
    double local_sum = 0.0;
    for (int i = threadIdx.x; i < num_keys; i += blockDim.x) {
        double shifted = scores[row * num_keys + i] - block_max_val;
        local_sum += exp(shifted);
    }

    // Step 5: warp-level sum reduction.
    double warp_sum = local_sum;
    for (int mask = 16; mask > 0; mask >>= 1) {
        warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, mask);
    }

    // Step 6: block-level sum — same publish pattern as the max.
    if (threadIdx.x % 32 == 0) {
        s_sum[warp_id] = warp_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        double total = 0.0;
        for (int i = 0; i < num_warps; i++) {
            total += s_sum[i];
        }
        s_block_sum[0] = total;
    }
    __syncthreads();
    double block_sum = s_block_sum[0];

    // Step 7: normalize (exp recomputed — no per-thread storage needed).
    for (int i = threadIdx.x; i < num_keys; i += blockDim.x) {
        double shifted = scores[row * num_keys + i] - block_max_val;
        output[row * num_keys + i] = exp(shifted) / (block_sum + 1e-15);
    }
}

/* ============================================================ */
/*  Weighted sum kernel (float32)                                 */
/* ============================================================ */

/**
 * Weighted sum kernel: output = attn_weights @ values.
 *
 * Each thread handles one (query_pos, value_dim) element.
 * The weighted sum is computed over all key positions.
 *
 * Parameters
 * ----------
 * attn : float*
 *     Attention weights [total_queries, num_keys] — each row sums to 1.
 * values : float*
 *     Value vectors [num_keys, head_dim].
 * output : float*
 *     Output [total_queries, head_dim] — weighted sum of values.

 * total_queries : int
 *     Total number of query positions (B * H * Sq).
 * num_keys : int
 *     Key sequence length (Sk).
 * head_dim : int
 *     Feature dimension (D).
 */
__global__ void attention_weighted_sum_f32(
    const float* attn,
    const float* values,
    float* output,
    int total_queries,
    int num_keys,
    int head_dim
)
{
    int query_idx = blockIdx.x;   // 0..total_queries-1
    int value_idx = threadIdx.x;  // 0..head_dim-1

    if (query_idx >= total_queries) return;

    float sum = 0.0f;
    for (int k = 0; k < num_keys; k++) {
        sum += attn[query_idx * num_keys + k] * values[k * head_dim + value_idx];
    }
    output[query_idx * head_dim + value_idx] = sum;
}

/* ============================================================ */
/*  Weighted sum kernel (float64)                                 */
/* ============================================================ */

__global__ void attention_weighted_sum_f64(
    const double* attn,
    const double* values,
    double* output,
    int total_queries,
    int num_keys,
    int head_dim
)
{
    int query_idx = blockIdx.x;
    int value_idx = threadIdx.x;

    if (query_idx >= total_queries) return;

    double sum = 0.0;
    for (int k = 0; k < num_keys; k++) {
        sum += attn[query_idx * num_keys + k] * values[k * head_dim + value_idx];
    }
    output[query_idx * head_dim + value_idx] = sum;
}