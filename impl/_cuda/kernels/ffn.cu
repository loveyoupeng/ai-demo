/**
 * SwiGLU FFN — CUDA kernels for element-wise activation.
 *
 * SwiGLU formula:
 *   output = SiLU(x @ W1) * (x @ W3) @ W2
 *
 * The matmul operations use PyTorch's cuBLAS. This file provides
 * the element-wise SiLU kernel for CUDA dispatch.
 *
 * SiLU(x) = x / (1 + exp(-x)) = x * sigmoid(x)
 *   x large positive → x (near-identity)
 *   x large negative → 0 (suppressed)
 *   x = 0 → 0
 */

/**
 * Device function: numerically stable SiLU activation.
 *
 * Uses the formula: SiLU(x) = x / (1 + exp(-x))
 * For numerical stability, we use the identity:
 *   x / (1 + exp(-x)) = sigmoid(x)
 *   or equivalently: (1 - sigmoid(-x)) * x
 *
 * The second form avoids overflow for large positive x,
 * where exp(-x) would underflow to 0 (which is fine, result = x).
 *
 * Parameters
 * ----------
 * x : float
 *     Input value.

 * Returns
 * -------
 * float
 *     SiLU(x) = sigmoid(x) * x
 */
__device__ float silu(float x) {
    // For large positive x: exp(-x) → 0, so 1/(1+0) = 1, result = x
    // For large negative x: exp(-x) → inf, so 1/infinity = 0, result → 0
    // For x = 0: 1/(1+1) = 0.5, result = 0
    return x / (1.0f + expf(-x));
}

/**
 * Forward: element-wise SiLU activation.
 *
 * Each thread processes one element of the input array.
 * Coalesced memory access: consecutive threads access consecutive elements.
 *
 * Parameters
 * ----------
 * input : const float*
 *     Input array of size `size`.
 * output : float*
 *     Output array — SiLU(input).
 * size : int
 *     Total number of elements.

 */
__global__ void swiglu_silu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop: handles any input size ≥ 1
    for (int i = idx; i < size; i += stride) {
        output[i] = silu(input[i]);
    }
}

/**
 * Forward: element-wise SiLU activation (float64).
 *
 * Float64 variant — CUDA is statically typed.
 *
 * Parameters
 * ----------
 * input : const double*
 *     Input array of size `size`.
 * output : double*
 *     Output array — SiLU(input).
 * size : int
 *     Total number of elements.
 */
__global__ void swiglu_silu_f64_kernel(const double* input, double* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < size; i += stride) {
        output[i] = input[i] / (1.0 + exp(-input[i]));
    }
}
