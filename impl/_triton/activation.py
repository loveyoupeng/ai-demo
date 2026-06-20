"""SiLU activation kernel — element-wise x * sigmoid(x).

This is the simplest possible Triton kernel: pure element-wise mapping
with no cross-element communication. Designed as a warm-up to learn
the Triton DSL patterns.

Algorithm
---------
SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

For large positive x: SiLU(x) ≈ x (near-identity)
For large negative x: SiLU(x) ≈ 0 (suppressed)
For x = 0: SiLU(0) = 0

Memory access pattern
---------------------
1D coalesced loads/stores — each thread block handles a contiguous
slice of the flattened tensor.

Numerical stability
-------------------
Uses tl.sigmoid() which is numerically stable across the full
float32 range. For float64 tests, values are passed through
without additional clipping since Triton handles precision correctly.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_kernel(
    x_ptr: tl.pointer_type,  # pyright: ignore[reportInvalidTypeForm]
    y_ptr: tl.pointer_type,  # pyright: ignore[reportInvalidTypeForm]
    n_elements: tl.int32,  # pyright: ignore[reportInvalidTypeForm]
    BLOCK_SIZE: tl.constexpr,
) -> None:  # pyright: ignore[reportInvalidTypeForm]
    """Triton kernel: y = x * sigmoid(x).

    Each program instance handles a 1D block of elements.

    Parameters
    ----------
    x_ptr : pointer
        Pointer to input tensor (float32 or float64).
    y_ptr : pointer
        Pointer to output tensor (same dtype as input).
    n_elements : int32
        Total number of elements in the tensor.
    BLOCK_SIZE : int
        Number of elements per block (constexpr).

    """
    # Create offsets for this program instance
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask: only process elements within bounds
    mask = offsets < n_elements

    # Load input values (coalesced memory access)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute SiLU: y = x * sigmoid(x)
    # tl.sigmoid is numerically stable across the full float range
    y = x * tl.sigmoid(x)

    # Store output (coalesced memory access)
    tl.store(y_ptr + offsets, y, mask=mask)


class _SiluTriton(torch.autograd.Function):
    """Custom autograd function wrapping the SiLU Triton kernel.

    Forward pass: y = x * sigmoid(x) computed by Triton kernel on GPU.
    Backward pass: dy/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                   = siLU(x) + sigmoid(x) * (1 - x * siLU(x))

    This demonstrates how to integrate Triton kernels with PyTorch's
    autograd system — the kernel is a pure forward computation, and
    gradients are computed in the backward() classmethod.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportUnknownArgumentType]
        """Forward pass: compute y = x * sigmoid(x) using Triton kernel.

        Parameters
        ----------
        ctx : Any
            Context object for storing tensors for backward pass.
        x : torch.Tensor
            Input tensor on GPU.

        Returns
        -------
        torch.Tensor
            SiLU(x) = x * sigmoid(x), same shape as input.

        Notes
        -----
        The forward pass is GPU-only for the activation computation.
        Gradients are not computed here — they use PyTorch's native
        formula in backward() because the SiLU gradient is well-known.

        """
        if x.requires_grad:
            ctx.save_for_backward(x)
        original_shape = x.shape
        x_flat = x.contiguous().view(-1)
        y_flat = torch.empty_like(x_flat)
        n_elements = x_flat.numel()
        if n_elements == 0:
            return x.view(original_shape)
        BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 4096))
        BLOCK_SIZE = max(BLOCK_SIZE, 128)
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _silu_kernel[grid](x_flat, y_flat, n_elements, BLOCK_SIZE=BLOCK_SIZE)  # pyright: ignore[reportArgumentType]
        return y_flat.view(original_shape)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> torch.Tensor:  # pyright: ignore[reportUnknownArgumentType]
        """Backward pass: compute gradient using PyTorch's SiLU formula.

        The gradient of SiLU(x) = x * sigmoid(x) is:
            d/dx SiLU(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                         = sigmoid(x) + x * sigma * (1 - sigma)

        where sigma = sigmoid(x).

        Parameters
        ----------
        ctx : Any
            Context with saved input tensor.
        grad_outputs : torch.Tensor
            Gradient from upstream (same shape as SiLU output).

        Returns
        -------
        torch.Tensor
            Gradient w.r.t. input, same shape as the saved input.

        """
        (x,) = ctx.saved_tensors
        # Compute gradient directly using PyTorch's SiLU gradient formula
        # d/dx SiLU(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        sigma = torch.sigmoid(x)
        silu_x = x * sigma
        grad_input = grad_outputs[0] * (sigma + silu_x * (1 - sigma))
        return grad_input


def silu(x: torch.Tensor) -> torch.Tensor:
    """Apply SiLU activation element-wise using a Triton GPU kernel.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor on GPU, shape (..., embed_dim) or any shape.
        dtype should be float32 or float64.

    Returns
    -------
    torch.Tensor, same shape and dtype as x
        SiLU(x) = x * sigmoid(x), computed on GPU.

    Notes
    -----
    This function handles arbitrary tensor shapes by flattening to 1D
    for the Triton kernel and reshaping back to the original shape.
    The kernel uses coalesced memory access for optimal GPU throughput.

    Autograd Integration:
    ---------------------
    The kernel computes the forward pass on GPU. Gradients are computed
    in the backward() method using PyTorch's well-known SiLU gradient formula.
    This is a simple example — for more complex kernels, gradients would
    typically be computed by a separate Triton kernel.

    Example
    -------
    >>> import torch
    >>> x = torch.randn(2, 4, 8, device='cuda', dtype=torch.float32)
    >>> y = silu(x)
    >>> y.shape
    torch.Size([2, 4, 8])

    """
    # Validate input
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    if not x.is_cuda:
        raise ValueError("Input tensor must be on GPU (CUDA)")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Expected float32 or float64, got {x.dtype}")

    return _SiluTriton.apply(x)
