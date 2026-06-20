// SiLU (Swish) activation kernel — element-wise x * sigmoid(x).
//
// This is the simplest possible CUDA kernel: pure element-wise mapping
// with no cross-element communication. Designed as a warm-up to learn
// the CUDA bare-metal patterns.
//
// Algorithm
// ---------
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// For large positive x: SiLU(x) ≈ x (near-identity)
// For large negative x: SiLU(x) ≈ 0 (suppressed)
// For x = 0: SiLU(0) = 0
//
// Supports both float32 and float64 for learning how CUDA handles different
// precisions. Float64 uses `double` type and `exp` (double precision exponent).

// ============================================================================
// Float32 device functions
// ============================================================================

__device__ float _silu_forward_f32(float x) {
    // Compute x / (1 + exp(-x)) with numerically stable sigmoid
    float exp_neg_x = expf(-x);
    return x / (1.0f + exp_neg_x);
}

__device__ float _silu_backward_f32(float x) {
    // Gradient: d/dx SiLU(x) = sigmoid(x) + SiLU(x) * (1 - sigmoid(x))
    float sigma = 1.0f / (1.0f + expf(-x));
    float silu_x = x * sigma;
    return sigma + silu_x * (1.0f - sigma);
}

// ============================================================================
// Float64 device functions
// ============================================================================

__device__ double _silu_forward_f64(double x) {
    // Double precision version: use `exp` instead of `expf`
    // This preserves all 64 bits of precision through the computation
    double exp_neg_x = exp(-x);
    return x / (1.0 + exp_neg_x);
}

__device__ double _silu_backward_f64(double x) {
    // Double precision gradient
    double sigma = 1.0 / (1.0 + exp(-x));
    double silu_x = x * sigma;
    return sigma + silu_x * (1.0 - sigma);
}

// ============================================================================
// Float32 kernels
// ============================================================================

// Forward kernel (float32): output[i] = silu_forward(input[i])
__global__ void silu_forward_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = _silu_forward_f32(input[idx]);
    }
}

// Backward kernel (float32): grad_input[i] = grad_output[i] * silu_backward(input[i])
__global__ void silu_backward_kernel(const float* input, const float* grad_output, float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = grad_output[idx] * _silu_backward_f32(input[idx]);
    }
}

// ============================================================================
// Float64 kernels
// ============================================================================

// Forward kernel (float64): output[i] = silu_forward(input[i])
__global__ void silu_forward_f64_kernel(const double* input, double* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = _silu_forward_f64(input[idx]);
    }
}

// Backward kernel (float64): grad_input[i] = grad_output[i] * silu_backward(input[i])
__global__ void silu_backward_f64_kernel(const double* input, const double* grad_output, double* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = grad_output[idx] * _silu_backward_f64(input[idx]);
    }
}