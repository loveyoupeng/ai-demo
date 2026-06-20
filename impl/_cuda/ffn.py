"""SwiGLU Feed-Forward Network — CUDA SiLU kernel + PyTorch matmul.

SwiGLU ( gated SiLU activation) is the FFN activation used in modern
large language models (LLaMA, PaLM, Gemma). It replaces the standard
ReLU/gated activations with a more expressive gating mechanism.

Architecture
------------
Unlike attention or Layernorm, NOT all of SwiGLU is implemented as
CUDA kernels because the matrix multiplications use cuBLAS (highly
optimized). Instead, we use a HYBRID approach:

1. PyTorch matmul for W1, W3, W2 projections (cuBLAS)
2. CUDA C kernel for SiLU element-wise activation
3. PyTorch autograd for full backward pass

This demonstrates the practical pattern: CUDA kernels for element-wise
non-linearities, PyTorch for linear algebra.

Why not pure CUDA?
------------------
Three arguments against custom CUDA kernel for full SwiGLU:

  1. Memory-bound: Matmuls are already bandwidth-limited.
     Fusing them into one kernel doesn't save memory access
     (each matmul reads W, writes output — that's the bottleneck).

  2. cuBLAS quality: NVIDIA's cuBLAS matmul implementation is
     hand-optimized at the assembly level for every GPU architecture.
     A Python-written CUDA kernel can't match this performance.

  3. Kernel launch overhead: Three separate matmuls → three cuBLAS calls.
     Fusing them into one kernel would remove kernel launch overhead
     (microseconds) but add compute overhead (no shared memory reuse).
     The tradeoff is negative for large matrices.

When WOULD we use a CUDA kernel for FFN?
  - When ff_dim is small (< 128) and matmuls are tiny GEMMs
    (cuBLAS overhead dominates)
  - When using INT8/FP8 quantization (requires custom kernels)
  - When implementing novel activation functions not in PyTorch

For our purposes (float32/float64, large ff_dim), PyTorch matmuls
are optimal.

SwiGLU formula
--------------
For input x ∈ R^D, hidden dimension ff_dim, and learnable weights:

  gate = SiLU(x · W1)        → vector of size ff_dim
  proj = x · W3              → vector of size ff_dim
  gated = gate ⊙ proj        → element-wise multiply, size ff_dim
  output = gated · W2        → vector of size D

where:
  W1 ∈ R^{D×ff_dim}, W3 ∈ R^{D×ff_dim}, W2 ∈ R^{ff_dim×D}
  SiLU(x) = x / (1 + exp(-x))  — smooth gating between 0 and x

Element-wise non-linearity
--------------------------
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

  - For x ≫ 0 (large positive): SiLU(x) ≈ x (near-identity)
  - For x ≪ 0 (large negative): SiLU(x) → 0 (suppressed)
  - At x = 0: SiLU(0) = 0 (smooth, continuous)

Compared to ReLU:
  - SiLU is smooth everywhere (C^1 continuous) → better gradient flow
  - For negative x: SiLU has a soft decay (not hard zero cut-off)
  - For positive x: SiLU ≈ identity (gradients don't vanish)

Why three weight matrices (W1, W3, W2)?
----------------------------------------
Standard FFN:   output = ReLU(x·W1) · W2
SwiGLU FFN:     output = SiLU(x·W1) ⊙ (x·W3) · W2

The key difference: instead of one projection → activation → output,
SwiGLU has TWO projections (W1 → SiLU, W3 → identity) multiplied
element-wise, then projected to output. This gives:

  - More parameters for same hidden dim (3M vs 2M, where M=ff_dim*D)
  - Better expressivity (the gate is input-dependent, not fixed)
  - Improved gradient flow through multiple paths

Memory access pattern (CUDA SiLU kernel)
----------------------------------------
The CUDA kernel for SiLU processes each element of the gate tensor:

  1. x @ W1:   (B*S, D) @ (D, FF) → (B*S, FF)   [cuBLAS]
  2. SiLU:     (B*S, FF) → (B*S, FF)              [CUDA kernel, coalesced]
  3. x @ W3:   (B*S, D) @ (D, FF) → (B*S, FF)    [cuBLAS]
  4. Multiply: (B*S, FF) ⊙ (B*S, FF) → (B*S, FF)  [PyTorch fused]
  5. @ W2:     (B*S, FF) @ (FF, D) → (B*S, D)     [cuBLAS]

The CUDA kernel (step 2) uses:
  - 1D grid: one thread per (token, feature) element
  - Grid-stride loop: handles any batch size
  - Coalesced memory access: consecutive threads access consecutive elements
  - No shared memory: element-wise op, no cross-thread communication

Reference
---------
Shazeer, "GLU Variants Improve Transformer" (2020)
https://arxiv.org/abs/2002.05202
"""

from __future__ import annotations

import ctypes
from typing import Any

import torch
from cuda import cuda as _cuda_lib

from impl._cuda.compiler import compile_and_load, get_kernel_handle

# ---------------------------------------------------------------------------
# CUDA kernel source — loaded from companion .cu file
# ---------------------------------------------------------------------------

_KERNEL_SOURCE_PATH = __file__.rsplit("/", 1)[0] + "/kernels/ffn.cu"


def _load_kernel_source() -> str:
    """Load CUDA kernel source from file for nvrtc compilation.

    Returns
    -------
    str
        Raw CUDA C source code.
    """
    with open(_KERNEL_SOURCE_PATH) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Kernel compiler — nvrtc compile → PTX → module → kernel handles
# ---------------------------------------------------------------------------


class _FFNKernels:
    """Compile and cache SwiGLU SiLU kernels.

    Manages the nvrtc compilation pipeline for the element-wise SiLU
    kernel within SwiGLU. The matrix multiplications use PyTorch/cuBLAS;
    only the non-linearity is implemented as a CUDA kernel.

    This demonstrates the hybrid approach: manual CUDA for element-wise
    ops, framework for matrix multiplication.
    """

    _silu_kernel = None
    _silu_f64_kernel = None
    _module = None
    _ptx_data = None

    @classmethod
    def compile(cls) -> None:
        """Compile CUDA source and store kernel handles.

        Pipeline:
        1. nvrtcCreateProgram + nvrtcCompileProgram → PTX bytecode
        2. cuModuleLoadDataEx → CUDA runtime module
        3. cuModuleGetFunction → kernel function handles

        The handles are cached for the lifetime of the process.
        """
        source = _load_kernel_source()
        module, ptx_data = compile_and_load(source)
        cls._module = module
        cls._ptx_data = ptx_data

        cls._silu_kernel = get_kernel_handle(module, "swiglu_silu_kernel", ptx_data)
        cls._silu_f64_kernel = get_kernel_handle(module, "swiglu_silu_f64_kernel", ptx_data)

    @classmethod
    def get_silu_kernel(cls) -> Any:
        """Get or compile the SiLU kernel (float32).

        Returns
        -------
        Any
            Float32 SiLU kernel handle.
        """
        if cls._silu_kernel is None:
            cls.compile()
        return cls._silu_kernel

    @classmethod
    def get_silu_f64_kernel(cls) -> Any:
        """Get or compile the SiLU kernel (float64).

        Returns
        -------
        Any
            Float64 SiLU kernel handle.
        """
        if cls._silu_f64_kernel is None:
            cls.compile()
        return cls._silu_f64_kernel


# ---------------------------------------------------------------------------
# Kernel launcher
# ---------------------------------------------------------------------------


def _launch_silu_kernel(
    kernel: Any,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    size: int,
    block_size: int = 256,
) -> None:
    """Launch a SiLU kernel via the CUDA driver API.

    Parameters
    ----------
    kernel : Any
        Kernel handle from cuModuleGetFunction.
    input_tensor : torch.Tensor
        Input tensor on device — will be read.
    output_tensor : torch.Tensor
        Output tensor on device — will be written.
    size : int
        Total number of elements across all dimensions.
    block_size : int
        Number of threads per block (default 256).
    """
    grid_size = (size + block_size - 1) // block_size

    # Build kernel parameters: input_ptr, output_ptr, size
    params = [input_tensor, output_tensor, ctypes.c_int(size)]

    values, types = [], []
    for p in params:
        if isinstance(p, torch.Tensor):
            values.append(ctypes.c_void_p(p.data_ptr()))
            types.append(ctypes.c_void_p)
        else:
            values.append(p)
            types.append(ctypes.c_int)

    kernel_args = (tuple(values), tuple(types))

    status = _cuda_lib.cuLaunchKernel(
        kernel,
        grid_size,  # grid x
        1,  # grid y
        1,  # grid z
        block_size,  # block x
        1,  # block y
        1,  # block z
        0,  # shared memory — none for element-wise
        None,  # stream — let PyTorch manage concurrent execution
        kernel_args,
        0,  # extra — required to be 0 on this platform
    )
    if status[0] != _cuda_lib.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuLaunchKernel failed: {status}")


# ---------------------------------------------------------------------------
# Custom autograd function — forward with CUDA SiLU, backward with PyTorch
# ---------------------------------------------------------------------------


class _CUDASiLU(torch.autograd.Function):
    """SiLU activation forward pass via CUDA kernel, backward via PyTorch.

    Demonstrates the hybrid approach: CUDA for element-wise activation,
    PyTorch for automatic backward computation.
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> torch.Tensor:
        """Forward pass: output = SiLU(input).

        Parameters
        ----------
        ctx : Any
            Autograd context — stores input for backward.
        input : torch.Tensor
            Input tensor on device (any shape, any dtype float).

        Returns
        -------
        torch.Tensor
            Output with same shape and dtype as input.
        """
        ctx.save_for_backward(input)

        output = torch.empty_like(input)
        num_elements = input.numel()

        if input.dtype == torch.float64:
            _launch_silu_kernel(
                _FFNKernels.get_silu_f64_kernel(),
                input,
                output,
                num_elements,
            )
        else:
            _launch_silu_kernel(
                _FFNKernels.get_silu_kernel(),
                input,
                output,
                num_elements,
            )

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, torch.Tensor]:
        """Backward pass: grad_input = grad_output * d/dx(SiLU(x)).

        Parameters
        ----------
        ctx : Any
            Autograd context containing saved input tensor.
        grad_output : torch.Tensor
            Gradient from the loss.

        Returns
        -------
        tuple
            (None for positional args, grad_input for input).
        """
        (input,) = ctx.saved_tensors
        grad_output = grad_outputs[0]

        sigmoid = 1.0 / (1.0 + torch.exp(-input))
        silu_output = input * sigmoid
        grad_input = grad_output * (sigmoid + silu_output * (1.0 - sigmoid))

        return grad_input


# ---------------------------------------------------------------------------
# Public API — user-facing SwiGLU function
# ---------------------------------------------------------------------------


def swiglu_ffn(
    x: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU Feed-Forward Network with CUDA SiLU kernel.

    Computes: output = (CUDA_SiLU(xW1) ⊙ xW3) @ W2

    This is a HYBRID implementation:
    - Matrix multiplications use PyTorch (cuBLAS) — for performance
    - SiLU activation uses nvrtc-compiled CUDA kernel — for learning

    SwiGLU ( gated SiLU activation) is the FFN activation used in modern
    large language models (LLaMA, PaLM, Gemma). It replaces the standard
    ReLU/gated activations with a more expressive gating mechanism.

    Forward pass
    ------------
    Step 1: Gate path — PyTorch matmul + CUDA SiLU
      gate = x @ W1                  # (..., ff_dim) — PyTorch cuBLAS
      gate = CUDA_SiLU(gate)         # (..., ff_dim) — CUDA kernel
      # SiLU(x) = x / (1 + exp(-x)) = x * sigmoid(x)
      # For x ≥ 0: SiLU(x) ∈ [0, x]  — passes positive values, scales them
      # For x <  0: SiLU(x) ∈ (-sigmoid(0^+), 0) → 0 — suppresses negatives

    Step 2: Parallel projection — PyTorch matmul
      proj = x @ W3                  # (..., ff_dim) — PyTorch cuBLAS
      # W3 is independent of W1, providing a parallel signal

    Step 3: Element-wise gating — PyTorch fused multiply
      gated = gate * proj            # (..., ff_dim) — fused PyTorch
      # Only dimensions where BOTH gate AND proj are strong propagate.
      # This is more selective than ReLU (which only checks one path).

    Step 4: Output projection — PyTorch matmul
      out = gated @ W2               # (..., D) — PyTorch cuBLAS
      # Projects back to original dimension

    Dimensions
    ----------
    Input:       x   ~  (..., D)
    W1:         w1   ~  (D, ff_dim)       — gate projection
    W3:         w3   ~  (D, ff_dim)       — proj path
    W2:         w2   ~  (ff_dim, D)       — output projection

    Output:      out ~   (..., D)
                 gate ~   (..., ff_dim)
                 proj ~   (..., ff_dim)
                 gated ~  (..., ff_dim)

    Parameters
    ----------
    x : torch.Tensor, shape (..., D)
        Input activations. Can be any shape where the last
        dimension matches D (e.g., (B,S,D) or (B*S,D)).
    w1 : torch.Tensor, shape (D, ff_dim)
        Gate projection weights. The CUDA SiLU applied to
        xW1 forms the gating signal.
    w3 : torch.Tensor, shape (D, ff_dim)
        Parallel projection weights. xW3 is multiplied with
        CUDA_SiLU(xW1) — W3 is independent from W1.
    w2 : torch.Tensor, shape (ff_dim, D)
        Output projection weights. Projects the gated
        (..., ff_dim) back to (..., D).

    Returns
    -------
    torch.Tensor, shape (..., D)
        Gated feedforward output. Same shape as input x but
        last dimension unchanged (D in, D out).

    Notes
    -----
    Memory layout: All tensors are row-major (C-order).
    Batch dimensions (...) can vary — the function supports
    arbitrary batch shapes via PyTorch's broadcasting.

    Activation: CUDA_SiLU is implemented as a nvrtc-compiled CUDA C
    kernel. The kernel uses a grid-stride loop for arbitrary input sizes
    and coalesced memory access for maximum bandwidth.

    CUDA kernel configuration (SiLU step)
    --------------------------------------
    - Block size: 256 threads/block
    - Grid size: ceil(num_elements / 256)
    - Memory access: coalesced (contiguous tensor, consecutive threads)
    - No shared memory (element-wise operation, no cross-thread communication)
    - No warp reduction (element-wise op, no inter-thread aggregation)

    Example
    -------
    >>> import torch
    >>> x = torch.randn(2, 8, 64, device='cuda')        # (B, S, D)
    >>> w1 = torch.randn(64, 256, device='cuda')         # (D, ff_dim)
    >>> w3 = torch.randn(64, 256, device='cuda')         # (D, ff_dim)
    >>> w2 = torch.randn(256, 64, device='cuda')         # (ff_dim, D)
    >>> out = swiglu_ffn(x, w1, w3, w2)
    >>> out.shape
    torch.Size([2, 8, 64])

    Gradient flow
    -------------
    Gradients flow through FOUR paths:
      1. dL/dx ← dL/dout ← ... ← dL/dgated ← dL/dproj (via W2, W3)
      2. dL/dx ← dL/dout ← ... ← dL/dgated ← dL/dgate (via W2, W1)
      3. dL/dW2 ← gated^T @ dL/dout (outer product, sparse)
      4. dL/dW1, dL/dW3 ← gradients through CUDA_SiLU gates

    The dual-path architecture (W1→gate AND W3→proj) provides
    richer gradient flow than single-path FFNs.

    Reference
    ---------
    Shazeer, "GLU Variants Improve Transformer" (2020)
    https://arxiv.org/abs/2002.05202
    """
    # Step 1: Gate path — PyTorch matmul + CUDA SiLU
    gate = x @ w1  # (..., ff_dim) — cuBLAS
    gate = _CUDASiLU.apply(gate)  # (..., ff_dim) — CUDA kernel

    # Step 2: Parallel projection — PyTorch matmul
    proj = x @ w3  # (..., ff_dim) — cuBLAS

    # Step 3: Element-wise gating — PyTorch fused
    gated = gate * proj  # (..., ff_dim) — fused multiply

    # Step 4: Output projection — PyTorch matmul
    return gated @ w2  # (..., D) — cuBLAS
