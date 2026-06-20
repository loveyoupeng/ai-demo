/**
 * MoE (Mixture of Experts) kernel — expert scoring.
 *
 * For each token, compute routing score for every expert:
 *   score[token, expert] = tokens[token] ⋅ expert_weights[expert]
 *
 * Each thread handles one (token, expert) pair.
 *
 * Parameters
 * ----------
 * tokens : float*
 *     Input token features [total_tokens, dim].
 * expert_weights : float*
 *     Expert weight vectors [n_experts, dim].
 * scores : float*
 *     Output routing scores [total_tokens, n_experts].
 * total_tokens : int
 *     Total token positions (B * S).
 * n_experts : int
 *     Number of experts.
 * dim : int
 *     Feature dimension.
 */
__global__ void moe_score_f32(
    const float* tokens,
    const float* expert_weights,
    float* scores,
    int total_tokens,
    int n_experts,
    int dim
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_tokens * n_experts;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int token_idx = i / n_experts;
        int expert_idx = i % n_experts;

        float dot = 0.0f;
        for (int d = 0; d < dim; d++) {
            dot += tokens[token_idx * dim + d] * expert_weights[expert_idx * dim + d];
        }
        scores[i] = dot;
    }
}

/* ============================================================ */
/*  MoE expert scoring kernel (float64)                         */
/* ============================================================ */

__global__ void moe_score_f64(
    const double* tokens,
    const double* expert_weights,
    double* scores,
    int total_tokens,
    int n_experts,
    int dim
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_tokens * n_experts;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int token_idx = i / n_experts;
        int expert_idx = i % n_experts;

        double dot = 0.0;
        for (int d = 0; d < dim; d++) {
            dot += tokens[token_idx * dim + d] * expert_weights[expert_idx * dim + d];
        }
        scores[i] = dot;
    }
}

/* ============================================================ */
/*  MoE weighted sum kernel (float32)                           */
/* ============================================================ */

/**
 * Weighted sum kernel: for each token, combine expert outputs
 * using top-k indices and softmax-processed weights.
 *
 * out[token][d] = sum_{k=0}^{top_k} weights[token][k]
 *                 * expert_outputs[token][indices[token][k]][d]
 *
 * Parameters
 * ----------
 * expert_outputs : float*
 *     Pre-computed expert outputs [total_tokens, n_experts, dim].
 * indices : int*
 *     Top-k expert indices per token [total_tokens * top_k].
 * weights : float*
 *     Softmax weights per token per top-k [total_tokens * top_k].
 * output : float*
 *     Combined MoE output [total_tokens, dim].
 * total_tokens : int
 *     Total token positions (B * S).
 * dim : int
 *     Feature dimension.
 * top_k : int
 *     Number of experts per token.
 */
__global__ void moe_weighted_sum_f32(
    const float* expert_outputs,
    const int* indices,
    const float* weights,
    float* output,
    int total_tokens,
    int dim,
    int n_experts,
    int top_k
)
{
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    int value_idx = threadIdx.x;  // 0..dim-1

    float result = 0.0f;
    for (int k = 0; k < top_k; k++) {
        int expert_id = indices[token_idx * top_k + k];
        float w = weights[token_idx * top_k + k];
        result += w * expert_outputs[(token_idx * n_experts + expert_id) * dim + value_idx];
    }
    output[token_idx * dim + value_idx] = result;
}

/* ============================================================ */
/*  MoE weighted sum kernel (float64)                           */
/* ============================================================ */

__global__ void moe_weighted_sum_f64(
    const double* expert_outputs,
    const int* indices,
    const double* weights,
    double* output,
    int total_tokens,
    int dim,
    int n_experts,
    int top_k
)
{
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    int value_idx = threadIdx.x;

    double result = 0.0;
    for (int k = 0; k < top_k; k++) {
        int expert_id = (int)(indices[token_idx * top_k + k]);
        double w = weights[token_idx * top_k + k];
        result += w * expert_outputs[(token_idx * n_experts + expert_id) * dim + value_idx];
    }
    output[token_idx * dim + value_idx] = result;
}