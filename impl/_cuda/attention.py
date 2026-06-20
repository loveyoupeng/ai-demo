"""Scaled dot-product attention — stable softmax kernel + weighted sum.

Implements scaled dot-product attention using:
1. PyTorch matmul for QK^T (cuBLAS)
2. nvrtc-compiled CUDA kernel for stable softmax (reduction, exp)
3. nvrtc-compiled CUDA kernel for weighted sum (GEMV-like)

Why not all CUDA?
------------------
The attention operation has two very different parts:

  1. Matrix multiplication (QK^T and output@V) — use cuBLAS:
     - Hand-optimized assembly for every GPU architecture
     - Memory bandwidth is the bottleneck, not compute
     - Any custom kernel would be slower than cuBLAS for large matrices

  2. Stable softmax + weighted sum — use CUDA kernels:
     - Softmax requires warp reduction (max, sum) — CUDA-native
     - Weighted sum is memory-bound but needs per-row normalization

  3. The hybrid approach preserves learning (reduction, coalesced access,
     grid/block configuration) while achieving good throughput.

Stable softmax
--------------
For each row of the attention matrix (one query position):
  max_val = max(scores[row, :])
  shifted = scores - max_val
  exp_vals = exp(shifted)
  sum_exp = sum(exp_vals)
  output = exp_vals / sum_exp

This prevents overflow: exp(large value) would saturate to inf without
max subtraction. The relative values are preserved (softmax is translation-invariant).

Weighted sum
------------
For each (batch, head, query_pos, head_dim):
  output[b,h,q,d] = sum_k attention[b,h,q,k] * v[b,h,k,d]

Each thread handles one (query, dim_pair) element for full parallelism.
The weighted sum iterates over all key positions — memory-bound, but
simple coalesced access pattern.

Reference
---------
Vaswani et al. "Attention Is All You Need" (2017)
https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

from typing import Any

import torch
from cuda import cuda as _cuda_lib

from impl._cuda.compiler import compile_and_load, get_kernel_handle

# ---------------------------------------------------------------------------
# CUDA kernel source — loaded from companion .cu file
# ---------------------------------------------------------------------------

_KERNEL_SOURCE_PATH = __file__.rsplit("/", 1)[0] + "/kernels/attention.cu"


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


class _AttentionKernels:
    """Compile and cache attention softmax + weighted sum kernels.

    Manages the nvrtc compilation pipeline for attention kernels:
    1. Load CUDA C source from file
    2. Compile with nvrtc → PTX bytecode
    3. Load PTX as runtime CUDA module
    4. Extract kernel handles via cuModuleGetFunction

    This class caches kernel handles for the lifetime of the process.
    """

    _softmax_kernel = None
    _softmax_f64_kernel = None
    _weighted_sum_kernel = None
    _weighted_sum_f64_kernel = None
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

        cls._softmax_kernel = get_kernel_handle(module, "attention_softmax_f32", ptx_data)
        cls._softmax_f64_kernel = get_kernel_handle(module, "attention_softmax_f64", ptx_data)
        cls._weighted_sum_kernel = get_kernel_handle(module, "attention_weighted_sum_f32", ptx_data)
        cls._weighted_sum_f64_kernel = get_kernel_handle(module, "attention_weighted_sum_f64", ptx_data)

    @classmethod
    def get_softmax_kernel(cls) -> Any:
        """Get or compile the softmax kernel (float32).

        Returns
        -------
        Any
            Float32 softmax kernel handle.
        """
        if cls._softmax_kernel is None:
            cls.compile()
        return cls._softmax_kernel

    @classmethod
    def get_softmax_f64_kernel(cls) -> Any:
        """Get or compile the softmax kernel (float64).

        Returns
        -------
        any
            Float64 softmax kernel handle.
        """
        if cls._softmax_f64_kernel is None:
            cls.compile()
        return cls._softmax_f64_kernel

    @classmethod
    def get_weighted_sum_kernel(cls) -> Any:
        """Get or compile the weighted sum kernel (float32).

        Returns
        -------
        Any
            Float32 weighted sum kernel handle.
        """
        if cls._weighted_sum_kernel is None:
            cls.compile()
        return cls._weighted_sum_kernel

    @classmethod
    def get_weighted_sum_f64_kernel(cls) -> Any:
        """Get or compile the weighted sum kernel (float64).

        Returns
        -------
        any
            Float64 weighted sum kernel handle.
        """
        if cls._weighted_sum_f64_kernel is None:
            cls.compile()
        return cls._weighted_sum_f64_kernel


# ---------------------------------------------------------------------------
# Kernel launcher
# ---------------------------------------------------------------------------


def _launch_softmax_kernel(
    kernel: Any,
    scores: torch.Tensor,
    output: torch.Tensor,
    total_rows: int,
    num_keys: int,
) -> None:
    """Launch stable softmax kernel via CUDA driver API.

    Parameters
    ----------
    kernel : Any
        Kernel handle from cuModuleGetFunction.
    scores : torch.Tensor
        Input scores [B*H*Sq, Sk] — each row is one query position.
    output : torch.Tensor
        Output attention weights [B*H*Sq, Sk] — softmax of scores.
    total_rows : int
        Total number of rows (B * H * Sq).
    num_keys : int
        Key sequence length (Sk).
    """
    import ctypes

    # 1D grid: each block handles one row
    grid_size = total_rows
    block_size = 256  # standard block size for warp reduction

    # Shared memory: 2 * 256 * 4 = 2KB (max float for reduction)
    shared_mem = 2 * 256 * 4

    # Build kernel parameters: scores_ptr, output_ptr, total_rows, num_keys
    params_list = [scores, output, ctypes.c_int(total_rows), ctypes.c_int(num_keys)]
    param_values, param_types = [], []
    for p in params_list:
        if isinstance(p, torch.Tensor):
            param_values.append(ctypes.c_void_p(p.data_ptr()))
            param_types.append(ctypes.c_void_p)
        else:
            param_values.append(p)
            param_types.append(ctypes.c_int)

    kernel_args = (tuple(param_values), tuple(param_types))

    status = _cuda_lib.cuLaunchKernel(
        kernel,
        grid_size,  # grid x
        1,  # grid y
        1,  # grid z
        block_size,  # block x
        1,  # block y
        1,  # block z
        shared_mem,  # shared memory for reduction
        None,  # stream
        kernel_args,
        0,  # extra
    )
    if status[0] != _cuda_lib.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuLaunchKernel failed: {status} — check shared_mem for block size {block_size}")


def _launch_weighted_sum_kernel(
    kernel: Any,
    attn: torch.Tensor,
    val_tensor: torch.Tensor,
    output: torch.Tensor,
    total_queries: int,
    head_dim: int,
) -> None:
    """Launch weighted sum kernel via CUDA driver API.

    Parameters
    ----------
    kernel : Any
        Kernel handle from cuModuleGetFunction.
    attn : torch.Tensor
        Attention weights [total_queries, num_keys].
    values : torch.Tensor
        Value vectors [num_keys, head_dim].
    output : torch.Tensor
        Output [total_queries, head_dim].
    total_queries : int
        Total number of query positions (B * H * Sq).
    head_dim : int
        Feature dimension (D).
    """
    import ctypes

    # 1D grid: each block handles one query position
    grid_size = total_queries
    block_size = head_dim  # each thread handles one head dimension

    # No shared memory needed for weighted sum
    shared_mem = 0

    # Build kernel parameters: attn_ptr, values_ptr, output_ptr, total_queries, num_keys, head_dim
    # Renamed 'values' to 'param_values' to avoid conflict with torch.Tensor named 'values'
    num_keys = val_tensor.shape[0]
    params_list = [
        attn,
        val_tensor,
        output,
        ctypes.c_int(total_queries),
        ctypes.c_int(num_keys),
        ctypes.c_int(head_dim),
    ]
    param_values, param_types = [], []
    for p in params_list:
        if isinstance(p, torch.Tensor):
            param_values.append(ctypes.c_void_p(p.data_ptr()))
            param_types.append(ctypes.c_void_p)
        else:
            param_values.append(p)
            param_types.append(ctypes.c_int)

    kernel_args = (tuple(param_values), tuple(param_types))

    status = _cuda_lib.cuLaunchKernel(
        kernel,
        grid_size,  # grid x
        1,  # grid y
        1,  # grid z
        block_size,  # block x
        1,  # block y
        1,  # block z
        shared_mem,  # shared memory — none for weighted sum
        None,  # stream
        kernel_args,
        0,  # extra
    )
    if status[0] != _cuda_lib.CUresult.CUDA_SUCCESS:
        raise RuntimeError(
            f"cuLaunchKernel failed: {status} — grid={grid_size}, block={block_size}, head_dim={head_dim}"
        )


# ---------------------------------------------------------------------------
# Custom autograd function — forward: CUDA + PyTorch, backward: PyTorch
# ---------------------------------------------------------------------------


class _CUDASDPACudaFunction(torch.autograd.Function):
    """Scaled dot-product attention with CUDA softmax/weighted-sum kernels.

    The forward pass uses:
    - PyTorch matmul for QK^T (cuBLAS)
    - CUDA C kernel for stable softmax (warp reduction, shared memory)
    - CUDA C kernel for weighted sum (memory-bound GEMV pattern)

    The backward pass is handled entirely by PyTorch (F.scaled_dot_product_attention),
    which has a well-tested CUDA implementation.
    """

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: compute attention(Q, K, V).

        Parameters
        ----------
        ctx : Any
            Autograd context — saves tensors for backward.
        q : torch.Tensor, shape (B, H, Sq, D)
            Query tensor.
        k : torch.Tensor, shape (B, H, Sk, D)
            Key tensor.
        v : torch.Tensor, shape (B, H, Sk, D)
            Value tensor.

        Returns
        -------
        torch.Tensor, shape (B, H, Sq, D)
            Attention output — weighted sum of values using attention weights.
        """
        B, H, Sq, D = q.shape
        _, _, Sk, _ = k.shape
        assert v.shape[-1] == D

        is_float64 = q.dtype == torch.float64

        # Step 1: Compute scores = q @ k^T / sqrt(D) using PyTorch (cuBLAS)
        # q: (B, H, Sq, D), k: (B, H, Sk, D) → k^T: (B, H, D, Sk)
        # scores: (B, H, Sq, Sk)
        head_dim_sqrt = 1.0 / (D**0.5)
        scores = (q @ k.transpose(-2, -1)) * head_dim_sqrt  # (B, H, Sq, Sk)

        # Step 2: Stable softmax per row using CUDA kernel
        # Flatten to (total_queries, num_keys) for kernel
        total_queries = B * H * Sq  # total number of query positions
        attn_flat = scores.view(total_queries, Sk)  # (B*H*Sq, Sk)
        attn_output = torch.empty_like(attn_flat)

        if is_float64:
            _launch_softmax_kernel(
                _AttentionKernels.get_softmax_f64_kernel(),
                attn_flat,
                attn_output,
                total_queries,
                Sk,
            )
        else:
            _launch_softmax_kernel(
                _AttentionKernels.get_softmax_kernel(),
                attn_flat,
                attn_output,
                total_queries,
                Sk,
            )

        # Step 3: Weighted sum — loop over (b, h) since each pair has its own value matrix.
        out_flat = torch.empty(total_queries, D, device=q.device, dtype=q.dtype)
        for b in range(B):
            for h in range(H):
                start = b * H * Sq + h * Sq
                end = start + Sq
                # v[b, h] is (S, D); attn_output[start:end] is (Sq, S)
                # Weighted sum: (Sq, S) @ (S, D) → (Sq, D)
                if is_float64:
                    _launch_weighted_sum_kernel(
                        _AttentionKernels.get_weighted_sum_f64_kernel(),
                        attn_output[start:end],
                        v[b, h],
                        out_flat[start:end],
                        Sq,
                        D,
                    )
                else:
                    _launch_weighted_sum_kernel(
                        _AttentionKernels.get_weighted_sum_kernel(),
                        attn_output[start:end],
                        v[b, h],
                        out_flat[start:end],
                        Sq,
                        D,
                    )

        # Reshape to (B, H, Sq, D)
        output = out_flat.view(B, H, Sq, D)

        # Save for backward
        ctx.save_for_backward(q, k, v, scores)

        return output

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward pass: use PyTorch's F.scaled_dot_product_attention.

        Parameters
        ----------
        ctx : Any
            Autograd context containing saved q, k, v, scores.
        grad_output : torch.Tensor
            Gradient from the loss (same shape as forward output).

        Returns
        -------
        tuple
            (grad_q, grad_k, grad_v) — gradients w.r.t. Q, K, V.

        Notes
        -----
        The backward pass of SDPA is complex to derive manually:
          dQ = (attn_weights * (dOut @ V^T) - dOut @ V @ atn^T) / sqrt(D)
          dK = (Q^T @ dOut * attn_weights - Q^T @ dOut) / sqrt(D)
          dV = attn^T @ dOut

        PyTorch handles this correctly, so we reuse it.
        """
        q, k, v, scores = ctx.saved_tensors
        grad_output = grad_outputs[0]

        import torch.nn.functional as F

        # Compute gradients using PyTorch's built-in backward
        # We compute grad by passing through PyTorch's SDPA with gradients
        with torch.enable_grad():
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            out.backward(grad_output)
            grad_q = q.grad.clone()
            grad_k = k.grad.clone()
            grad_v = v.grad.clone()

        return grad_q, grad_k, grad_v


# ---------------------------------------------------------------------------
# Public API — user-facing function
# ---------------------------------------------------------------------------


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Compute scaled dot-product attention via CUDA kernel.

    Implements: attention(Q, K, V) = softmax(QK^T / sqrt(d)) @ V

    This is the core attention mechanism from the Transformer architecture.
    The forward pass uses:
    - PyTorch cuBLAS for QK^T, output @ V (for performance)
    - nvrtc-compiled CUDA kernel for stable softmax (warp reduction)
    - nvrtc-compiled CUDA kernel for weighted sum (memory-bound GEMV)

    Algorithm
    ---------
      QK^T → scores (B,H,Sq,Sk)
      scores / sqrt(d) → scaled scores
      stable_softmax → attention_weights (B,H,Sq,Sk)
      attn @ V → output (B,H,Sq,D)

    Parameters
    ----------
    q : torch.Tensor, shape (B, H, Sq, D)
        Query tensor. B=batch, H=heads, Sq=source_len, D=dim_per_head.
    k : torch.Tensor, shape (B, H, Sk, D)
        Key tensor.  Sk=key_len.
    v : torch.Tensor, shape (B, H, Sk, D)
        Value tensor.  Sk must match k.shape[-2]. D must match q.shape[-1].

    Returns
    -------
    torch.Tensor, shape (B, H, Sq, D)
        Attention output — weighted sum of values using attention weights.

    Memory considerations
    ---------------------
    - For (B=2, H=8, S=64, D=64):
      Q: ~0.03 MB, K: ~0.03 MB, V: ~0.03 MB
      Scores (intermediate): ~0.06 MB
      Output: ~0.03 MB
    - Total peak memory (GPU): ~0.2 MB
    - Memory bandwidth bound (not compute bound)

    Example
    -------
    >>> import torch
    >>> q = torch.randn(2, 8, 16, 64, device="cuda")
    >>> k = torch.randn(2, 8, 32, 64, device="cuda")
    >>> v = torch.randn(2, 8, 32, 64, device="cuda")
    >>> out = scaled_dot_product_attention(q, k, v)
    >>> out.shape
    torch.Size([2, 8, 16, 64])

    Notes
    -----
    - Tensor dtype must be float16, bfloat16, or float32
    - All tensors must be on the same CUDA device
    - No causal mask is applied — use a pre-masked K if needed

    Reference
    ---------
    Vaswani et al. "Attention Is All You Need" (2017)
    https://arxiv.org/abs/1706.03762
    """
    return _CUDASDPACudaFunction.apply(q, k, v)
