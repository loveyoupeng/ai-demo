/**
 * Rotary Position Embedding (RoPE) kernel — CUDA C.
 *
 * Rotates pairs of dimensions by position-dependent angles:
 *
 *   [x'_{2m}, x'_{2m+1}] = [cos(θ_m·p), -sin(θ_m·p)] [x_{2m}]
 *                        [sin(θ_m·p),  cos(θ_m·p)] [x_{2m+1}]
 *
 * where θ_m = 10000^(-2m/d) is the frequency for pair m,
 *       p     = position index.
 *
 * Memory access: each thread processes one (token, pair) element.
 * Coalesced read of x, coalesced write of x_out.
 * Broadcast of cos/sin per (token, pair) to all threads in block.
 */

/* ============================================================ */
/*  Forward: RoPE apply (float32)                                 */
/* ============================================================ */

__global__ void rope_fwd_f32(
    const float* x,              // (B*S*H, D) or (S*B*H, D) row-major token storage
    const float* cos_table,      // (max_pos, D/2)
    const float* sin_table,      // (max_pos, D/2)
    float* x_out,                // (B*S*H, D)
    int total_tokens,            // B * S * H — total number of tokens
    int S,                       // sequence length (used for position computation)
    int D,                       // head dimension
    int rope_dim                 // number of dims to rotate (must be even)
)
{
    // Flat index over all tokens
    int token = blockIdx.x * blockDim.x + threadIdx.x;

    if (token >= total_tokens) return;

    // Base index into x (row-major token storage)
    // Token layout: (batch, seq, head) → flat index = (batch * S + seq) * H + head
    int s_idx = token % S;   // sequence index: position-dependent
    int base_idx = token * D;

    // Process dimension pairs
    for (int m = 0; m < rope_dim; m += 2) {
        // Read x_{2m}, x_{2m+1} from this token
        float x0 = x[base_idx + m];      // odd dim (index 0 in pair)
        float x1 = x[base_idx + m + 1];  // even dim (index 1 in pair)

        // Load cos and sin for this position and dimension pair
        float c = cos_table[s_idx * (rope_dim / 2) + (m / 2)];
        float s = sin_table[s_idx * (rope_dim / 2) + (m / 2)];

        // Apply 2D rotation (orthogonal matrix)
        x_out[base_idx + m]     = c * x0 - s * x1;
        x_out[base_idx + m + 1] = s * x0 + c * x1;
    }

    // Copy unchanged tail dims (if rope_dim < D)
    for (int d = rope_dim; d < D; d++) {
        x_out[base_idx + d] = x[base_idx + d];
    }
}

/* ============================================================ */
/*  Forward: RoPE apply (float64)                                 */
/* ============================================================ */

__global__ void rope_fwd_f64(
    const double* x,
    const double* cos_table,
    const double* sin_table,
    double* x_out,
    int total_tokens,
    int S,
    int D,
    int rope_dim
)
{
    int token = blockIdx.x * blockDim.x + threadIdx.x;

    if (token >= total_tokens) return;

    int s_idx = token % S;
    int base_idx = token * D;

    for (int m = 0; m < rope_dim; m += 2) {
        double x0 = x[base_idx + m];
        double x1 = x[base_idx + m + 1];

        double c = cos_table[s_idx * (rope_dim / 2) + (m / 2)];
        double s = sin_table[s_idx * (rope_dim / 2) + (m / 2)];

        x_out[base_idx + m]     = c * x0 - s * x1;
        x_out[base_idx + m + 1] = s * x0 + c * x1;
    }

    for (int d = rope_dim; d < D; d++) {
        x_out[base_idx + d] = x[base_idx + d];
    }
}

/* ============================================================ */
/*  Backward: RoPE apply gradient (float32)                       */
/* ============================================================ */

__global__ void rope_bwd_f32(
    const float* dx,
    const float* cos_table,
    const float* sin_table,
    float* dx_out,
    int total_tokens,
    int S,
    int D,
    int rope_dim
)
{
    int token = blockIdx.x * blockDim.x + threadIdx.x;

    if (token >= total_tokens) return;

    int s_idx = token % S;

    for (int m = 0; m < rope_dim; m += 2) {
        float c = cos_table[s_idx * (rope_dim / 2) + (m / 2)];
        float s = sin_table[s_idx * (rope_dim / 2) + (m / 2)];

        // Gradient of rotation: x' = R*x → dx = R^T * dx_out
        // R = [[cos, -sin], [sin, cos]] → R^T = [[cos, sin], [-sin, cos]]
        float dx0 = dx[token * D + m];
        float dx1 = dx[token * D + m + 1];

        dx_out[token * D + m]     = c * dx0 + s * dx1;
        dx_out[token * D + m + 1] = -s * dx0 + c * dx1;
    }

    for (int d = rope_dim; d < D; d++) {
        dx_out[token * D + d] = dx[token * D + d];
    }
}

/* ============================================================ */
/*  Backward: RoPE apply gradient (float64)                      */
/* ============================================================ */

__global__ void rope_bwd_f64(
    const double* dx,
    const double* cos_table,
    const double* sin_table,
    double* dx_out,
    int total_tokens,
    int S,
    int D,
    int rope_dim
)
{
    int token = blockIdx.x * blockDim.x + threadIdx.x;

    if (token >= total_tokens) return;

    int s_idx = token % S;

    for (int m = 0; m < rope_dim; m += 2) {
        double c = cos_table[s_idx * (rope_dim / 2) + (m / 2)];
        double s = sin_table[s_idx * (rope_dim / 2) + (m / 2)];

        double dx0 = dx[token * D + m];
        double dx1 = dx[token * D + m + 1];

        dx_out[token * D + m]     = c * dx0 + s * dx1;
        dx_out[token * D + m + 1] = -s * dx0 + c * dx1;
    }

    for (int d = rope_dim; d < D; d++) {
        dx_out[token * D + d] = dx[token * D + d];
    }
}
