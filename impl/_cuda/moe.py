"""
MoE (Mixture of Experts) dispatcher — top-k routing + weighted sum.

Uses:
- PyTorch for expert outputs (linear) and top-k+softmax
- CUDA C kernel for expert scoring and weighted combination

This hybrid approach teaches:
- Expert scoring: dot-product via grid-stride loop (compute-bound)
- Weighted sum: indexed access to expert outputs (memory-bound)
"""

from __future__ import annotations

from typing import Any

import torch
from cuda import cuda as _cuda_lib

from impl._cuda.compiler import compile_and_load, get_kernel_handle

# ---------------------------------------------------------------------------
# CUDA kernel source — loaded from companion .cu file
# ---------------------------------------------------------------------------

_KERNEL_SOURCE_PATH = __file__.rsplit("/", 1)[0] + "/kernels/moe.cu"


# ---------------------------------------------------------------------------
# Kernel compiler — nvrtc compile → PTX → module → kernel handles
# ---------------------------------------------------------------------------


class _MoeKernels:
    """Compile and cache MoE kernels.

    Manages the nvrtc compilation pipeline for MoE kernels:
    1. Load CUDA C source from file
    2. Compile with nvrtc → PTX bytecode
    3. Load as CUDA module
    4. Cache and resolve kernel handles
    """

    _module: Any = None
    _ptx: bytes = b""

    @classmethod
    def _ensure_loaded(cls) -> None:
        """Ensure kernels are compiled and cached."""
        if cls._module is not None:
            return

        with open(_KERNEL_SOURCE_PATH) as f:
            source = f.read()
        cls._module, cls._ptx = compile_and_load(source)

    @classmethod
    def get_score_f32_kernel(cls) -> Any:
        """Get MoE scoring kernel for float32."""
        cls._ensure_loaded()
        return get_kernel_handle(cls._module, "moe_score_f32", cls._ptx)

    @classmethod
    def get_score_f64_kernel(cls) -> Any:
        """Get MoE scoring kernel for float64."""
        cls._ensure_loaded()
        return get_kernel_handle(cls._module, "moe_score_f64", cls._ptx)

    @classmethod
    def get_weighted_sum_f32_kernel(cls) -> Any:
        """Get weighted sum kernel for float32."""
        cls._ensure_loaded()
        return get_kernel_handle(cls._module, "moe_weighted_sum_f32", cls._ptx)

    @classmethod
    def get_weighted_sum_f64_kernel(cls) -> Any:
        """Get weighted sum kernel for float64."""
        cls._ensure_loaded()
        return get_kernel_handle(cls._module, "moe_weighted_sum_f64", cls._ptx)


def _launch_moe_score_kernel(
    kernel: Any,
    tokens: torch.Tensor,
    expert_weights: torch.Tensor,
    scores: torch.Tensor,
    total_tokens: int,
    n_experts: int,
    dim: int,
) -> None:
    """Launch MoE expert scoring kernel.

    Each thread handles one (token, expert) pair:
        score = tokens[token] ⋅ expert_weights[expert]

    Parameters
    ----------
    kernel : Any
        Kernel handle from cuModuleGetFunction.
    tokens : torch.Tensor
        Input features [total_tokens, dim].
    expert_weights : torch.Tensor
        Expert weights [n_experts, dim].
    scores : torch.Tensor
        Output routing scores [total_tokens, n_experts].
    total_tokens : int
        Total token positions (B * S).
    n_experts : int
        Number of experts.
    dim : int
        Feature dimension.
    """
    import ctypes as _ctypes  # noqa: F811

    total_pairs = total_tokens * n_experts
    block_size = min(256, total_pairs)

    kernel_params = (
        tuple(
            _ctypes.c_void_p(t.data_ptr())
            for t in (tokens, expert_weights, scores)
        )
        + (
            _ctypes.c_int(total_tokens),
            _ctypes.c_int(n_experts),
            _ctypes.c_int(dim),
        ),
        tuple(_ctypes.c_void_p for _ in range(3))
        + (
            _ctypes.c_int,
            _ctypes.c_int,
            _ctypes.c_int,
        ),
    )

    status = _cuda_lib.cuLaunchKernel(
        kernel,
        (total_pairs + block_size - 1) // block_size,
        1,
        1,
        block_size,
        1,
        1,
        0,
        None,
        kernel_params,
        0,
    )
    if status[0] != _cuda_lib.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuLaunchKernel score failed: {status}")


def _launch_moe_weighted_sum_kernel(
    expert_outputs: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    total_tokens: int,
    dim: int,
    n_experts: int,
    top_k: int,
    is_float64: bool = False,
) -> None:
    """Launch MoE weighted sum kernel.

    Parameters
    ----------
    expert_outputs : torch.Tensor
        Pre-computed expert outputs [total_tokens, n_experts, dim].
    indices : torch.Tensor
        Top-k expert indices per token [total_tokens * top_k].
    weights : torch.Tensor
        Softmax weights [total_tokens * top_k].
    output : torch.Tensor
        Combined MoE output [total_tokens, dim].
    total_tokens : int
        Total token positions.
    dim : int
        Feature dimension.
    n_experts : int
        Number of experts.
    top_k : int
        Number of experts per token.
    is_float64 : bool
        Whether to use float64.
    """
    import ctypes as _ctypes  # noqa: F811

    # Kernel uses const int* (32-bit). Ensure contiguous and int32 to prevent silent memory misalignment.
    assert indices.is_contiguous(), "indices must be contiguous for indexed kernel access"
    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)

    kernel = _MoeKernels.get_weighted_sum_f64_kernel() if is_float64 else _MoeKernels.get_weighted_sum_f32_kernel()

    block_size = min(dim, 1024)

    params = (
        tuple(_ctypes.c_void_p(t.data_ptr()) for t in (expert_outputs, indices, weights, output))
        + (
            _ctypes.c_int(total_tokens),
            _ctypes.c_int(dim),
            _ctypes.c_int(n_experts),
            _ctypes.c_int(top_k),
        )
    )
    types = tuple(_ctypes.c_void_p for _ in range(4)) + tuple(_ctypes.c_int for _ in range(4))

    status = _cuda_lib.cuLaunchKernel(
        kernel,
        total_tokens,
        1,
        1,
        block_size,
        1,
        1,
        0,
        None,
        (params, types),
        0,
    )
    if status[0] != _cuda_lib.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuLaunchKernel weighted_sum failed: {status}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def moe_forward(
    tokens: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_bias: torch.Tensor,
    routing_weights: torch.Tensor,
    top_k: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Mixture-of-Experts forward pass.

    Mixes CUDA expert scoring + PyTorch top-k/softmax and expert outputs.
    The CUDA kernel handles expert scoring (dot products) and weighted sum
    (indexed memory access), while PyTorch handles top-k routing.

    Parameters
    ----------
    tokens : torch.Tensor, shape (B, S, D)
        Input token features for batch S positions.
    expert_weights : torch.Tensor, shape (N, D, D)
        Weight matrix for each of N experts (D -> D).
    expert_bias : torch.Tensor, shape (N, D)
        Bias matrix for each expert.
    routing_weights : torch.Tensor, shape (N, D)
        Routing score weights — dot product with tokens gives expert scores.
    top_k : int
        Number of experts to activate per token (default: 2).

    Returns
    -------
    output : torch.Tensor, shape (B, S, D)
        Combined expert output — weighted sum of selected expert predictions.
    indices : torch.Tensor, shape (B, S, top_k)
        Selected expert indices per token.
    weights : torch.Tensor, shape (B, S, top_k)
        Softmax weights for selected experts.
    """
    is_float64 = tokens.dtype == torch.float64

    B, S, D = tokens.shape
    N = expert_weights.shape[0]
    assert expert_weights.shape == (N, D, D)
    assert expert_bias.shape == (N, D)
    assert routing_weights.shape == (N, D)
    assert top_k <= N

    # Step 1: Expert outputs via PyTorch (cuBLAS)
    # expert_outputs[b, s, n, d] = tokens[b, s] @ expert_weights[n] + bias[n]
    expert_outputs = torch.stack([
        torch.nn.functional.linear(tokens, expert_weights[n])
        for n in range(N)
    ], dim=2)  # (B, S, N, D)

    # Step 2: Expert scoring via CUDA kernel
    # tokens_flat: (B*S, D), scores: (B*S, N)
    # Score = tokens @ routing_weights[n] for each expert n
    tokens_flat = tokens.view(-1, D)
    total_tokens = B * S
    routing_scores = torch.empty(total_tokens, N, device=tokens.device, dtype=tokens.dtype)

    if is_float64:
        _launch_moe_score_kernel(
            _MoeKernels.get_score_f64_kernel(),
            tokens_flat,
            routing_weights,
            routing_scores,
            total_tokens,
            N,
            D,
        )
    else:
        _launch_moe_score_kernel(
            _MoeKernels.get_score_f32_kernel(),
            tokens_flat,
            routing_weights,
            routing_scores,
            total_tokens,
            N,
            D,
        )

    # Step 3: Top-k routing via PyTorch (top-k + softmax)
    routing_scores = routing_scores.view(B, S, N)
    topk_scores, topk_idx = torch.topk(routing_scores, top_k, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_scores, dim=-1)

    # Step 4: Weighted sum via CUDA kernel
    exp_out = expert_outputs.contiguous().view(total_tokens, N, D)
    idx_flat = topk_idx.contiguous().view(-1).to(torch.int32)
    w_flat = topk_weights.contiguous().view(-1)
    out_flat = torch.empty(total_tokens, D, device=tokens.device, dtype=tokens.dtype)

    _launch_moe_weighted_sum_kernel(
        exp_out,
        idx_flat,
        w_flat,
        out_flat,
        total_tokens,
        D,
        N,
        top_k,
        is_float64=is_float64,
    )

    output = out_flat.view(B, S, D)
    return output, topk_idx, topk_weights
