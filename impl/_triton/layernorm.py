"""Layer normalization kernel — RMSNorm via Triton.

RMSNorm (Root Mean Square Layer Normalization) is used in modern LLMs
(Transformer-XL, GPT-2, LLaMA) instead of standard LayerNorm because
it removes the mean-shift operation, reducing computational cost while
maintaining comparable accuracy.

Algorithm
---------
Standard LayerNorm:  y = (x - mean(x)) /sqrt(var(x) + eps) * gamma
RMSNorm:            y = x / sqrt(mean(x^2) + eps) * gamma

RMSNorm skips the mean subtraction step, computing normalization
from the root-mean-square of activations only. This is mathematically
equivalent when activations are zero-centered, which they tend to be
after residual connections.

Mathematical derivation
-----------------------
Let x ∈ R^D be a single activation vector.

1. Mean of squares:      m = (1/D) * sum_i(x_i^2)
2. RMS:                 r = sqrt(m + eps)       # (scalar)
3. Normalize + scale:    y_i = (x_i / r) * gamma_i

Forward pass:  x (B,S,D) → y (B,S,D) with learnable gamma (D,)
Backward pass: PyTorch formula for dL/dx and dL/dgamma

Memory access pattern
---------------------
Two-pass algorithm over the feature dimension:

  Pass 1 (lines 51-58): Load x^2 per block, accumulate to scalar sum.
    - Each program handles one row (one token position)
    - Within-row, features are split into BLOCK_SIZE chunks
    - Uses tl.sum() to reduce block to scalar

  Pass 2 (lines 64-70): Normalize each element and scale by gamma.
    - Re-load x and gamma per block
    - Element-wise divide by pre-computed RMS scalar
    - Multiply by gamma, store result

Why two passes?
  Triton requires all values in a block to have the same dtype.
  The sum accumulator must be a scalar (not block-sized), so we
  collect block-wise sums in a temporary tensor and tl.sum() them.

BLOCK_SIZE selection
--------------------
- Set to next power of 2 of embed_dim for efficient memory access
- Minimum 128 to ensure GPU occupancy (enough threads per SM)
- For embed_dim=512: BLOCK_SIZE=512 (full row fits in one block)
- For embed_dim=4096: BLOCK_SIZE=4096 (no iteration needed)

Numerical stability
-------------------
- All computation happens in float32 internally (even for fp64 I/O)
- eps=1e-6 prevents division by zero for zero activations
- Mask prevents out-of-bounds access when features don't fill entire block

Example usage
-------------
>>> import torch
>>> x = torch.randn(2, 8, 64, device='cuda')
>>> gamma = torch.ones(64, device='cuda')
>>> y = rmsnorm(x, gamma)
>>> y.shape
torch.Size([2, 8, 64])
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr: tl.pointer_type,  # pyright: ignore[reportInvalidTypeForm]
    gamma_ptr: tl.pointer_type,  # pyright: ignore[reportInvalidTypeForm]
    out_ptr: tl.pointer_type,  # pyright: ignore[reportInvalidTypeForm]
    n_features: tl.int32,  # pyright: ignore[reportInvalidTypeForm]
    n_rows: tl.int32,  # pyright: ignore[reportInvalidTypeForm]
    eps: tl.float32,  # pyright: ignore[reportInvalidTypeForm]
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Triton kernel: RMSNorm — y = x / sqrt(mean(x^2) + eps) * gamma.

    Computes RMS normalization row-by-row. Each program instance handles
    one row of the (n_rows, n_features) matrix.

    Algorithm
    ---------
    This kernel uses a two-pass approach:

      Pass 1: Compute mean of x^2 per row (reduction over features).
        - Accumulate x^2 block-by-block using tl.sum()
        - Result: per-row RMS scalar

      Pass 2: Normalize and scale per row.
        - Load x and gamma per block
        - Compute y = (x / rms) * gamma
        - Store output

    Memory layout
    -------------
    Input:  (n_rows, n_features) row-major → flattened pointer stride
    Output: (n_rows, n_features) row-major → same layout
    Gamma:  (n_features,) → broadcast across rows

    Why row-major? Each token in a batch is one row; normalizing
    per-token independently means each row is an independent reduction.

    Coalesced access: Within a row, adjacent threads load adjacent
    elements — the GPU memory controller coalesces these into single
    transactions.

    Numerical considerations
    ------------------------
    - Accumulator is tl.float32 regardless of input dtype
    - eps clipped to float32 precision even for float64 inputs
    - Mask prevents reading past valid elements when BLOCK_SIZE > n_features

    Parameters
    ----------
    x_ptr : pointer
        Pointer to input tensor of shape (n_rows, n_features).
    gamma_ptr : pointer
        Pointer to scale parameter of shape (n_features,).
    out_ptr : pointer
        Pointer to output tensor of shape (n_rows, n_features).
    n_features : int
        Number of features (last dimension). Must equal len(gamma).
    n_rows : int
        Number of rows (batch * sequence length flattened).
    eps : float32
        Small constant for numerical stability. Default 1e-6.
    BLOCK_SIZE : constexpr int
        Number of features per tile. Must be a power of 2 >= 128.

    Grid configuration
    ------------------
    Block grid: (n_rows,) — one program per row.
    Within a program: BLOCK_SIZE threads per vector.

    Performance
    -----------
    - FLOPs per row: 2*D (multiply + divide + scale) + 1 (sqrt)
    - Memory bandwidth: 3 reads (x, gamma, x) + 1 write (out) per row
    - Ideal for large batch sizes where n_rows >> n_features

    Reference
    ---------
    Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
    https://arxiv.org/abs/1910.07467

    """
    # ── Row assignment ─────────────────────────────────────────
    # Each program instance handles one row (e.g., one token's
    # embedding vector). With n_rows programs and BLOCK_SIZE
    # vector width, we handle n_rows * BLOCK_SIZE elements total.
    row_idx = tl.program_id(axis=0)  # Row index

    # Feature indices within this block (e.g., [0,1,2,...,BLOCK_SIZE-1])
    cols = tl.arange(0, BLOCK_SIZE)  # [0..BLOCK_SIZE)

    # Calculate the starting offset for this row
    # Row-major layout: row i starts at i * n_features
    row_offset = row_idx * n_features

    # ── Pass 1: Compute mean(x^2) ──────────────────────────────
    # We need sqrt(1/D * sum(x^2)), so first accumulate x^2 values
    # across all feature blocks for this row.

    # The accumulator is a tensor of shape (BLOCK_SIZE,) initialized
    # to zero. Each block of features updates its slice.
    x_sq_acc: tl.tensor = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)  # (BLOCK_SIZE,)

    for _start in range(0, n_features, BLOCK_SIZE):
        # Compute feature indices for this block
        feature_idx = _start + cols  # e.g., [0,1,...,127] or [128,...,255]

        # Mask: only process valid elements within this batch row
        feature_mask = feature_idx < n_features  # (BLOCK_SIZE,) bool

        # Load x values, convert to float32 for computation
        # other=0.0 means masked (invalid) elements contribute 0 to sum
        x_block: tl.tensor = tl.load(
            x_ptr + row_offset + feature_idx,  # Row offset + feature offset
            mask=feature_mask,  # Mask for bounds checking
            other=0.0,  # Default value for masked elements
        ).to(tl.float32)  # (BLOCK_SIZE,) — element-wise float32 conversion

        # Square each element: x^2
        x_sq_acc = tl.where(
            feature_mask,            # Condition per element
            x_block * x_block,       # True: use x^2
            x_sq_acc,                # False: keep accumulated value
        )

    # Reduce the block accumulator to a single scalar sum
    x_sq_sum: float = tl.sum(x_sq_acc)  # Scalar float32
    mean_x_sq = x_sq_sum / n_features   # (float) Mean of squares

    # Compute RMS with stability epsilon
    rms = tl.sqrt(mean_x_sq + eps)  # (float) Root mean square

    # ── Pass 2: Normalize and scale ────────────────────────────
    # Now that we have the RMS scalar, divide by it and multiply
    # by the learnable gamma parameter.

    for _start in range(0, n_features, BLOCK_SIZE):
        feature_idx = _start + cols  # [0..BLOCK_SIZE) or offset block
        feature_mask = feature_idx < n_features  # (BLOCK_SIZE,) bool

        # Re-load x for this block (not cached — too large)
        x_block = tl.load(
            x_ptr + row_offset + feature_idx,
            mask=feature_mask,
            other=0.0,
        ).to(tl.float32)  # (BLOCK_SIZE,)

        # Load gamma for this block (broadcasts from (D,) to (BLOCK_SIZE,))
        gamma_block = tl.load(
            gamma_ptr + feature_idx,
            mask=feature_mask,
            other=0.0,
        ).to(tl.float32)  # (BLOCK_SIZE,)

        # Apply RMSNorm formula: y = (x / rms) * gamma
        # Step 1: Divide by RMS scalar (same for all features in this row)
        # Step 2: Scale by gamma (learned per-feature)
        out_block = (x_block / rms) * gamma_block

        # Store the normalized result
        tl.store(
            out_ptr + row_offset + feature_idx,
            out_block,
            mask=feature_mask,
        )


class _RmsNormTriton(torch.autograd.Function):
    """Custom autograd function for RMS normalization.

    Forward pass: Triton GPU kernel for the computation.
    Backward pass: PyTorch formula (well-known gradient for RMSNorm).

    Why use Triton for forward but PyTorch for backward?
    -----------------------------------------------------
    The RMSNorm gradient is well-documented and computationally cheap
    in PyTorch (element-wise ops + reductions). Writing a Triton kernel
    for it would add compilation overhead with minimal performance gain,
    since this is not the performance bottleneck in transformer training.

    We only write Triton kernels for the complex operations (attention,
    softmax, large matmuls) that benefit from coalesced memory access
    and on-chip compute.

    Backward formula derivation
    ---------------------------
    Given: y_i = x_i / r * gamma_i, where r = sqrt(1/D * sum(x_j^2) + eps)

    dL/dgamma = sum_i(y_i * dL/dy_i)  →  (D,)  (simple accumulation)

    dL/dx_i:
        Let d_i = dL/dy_i
        dr/dx_i = x_i / (r * D)
        dy_i/dx_i = 1/r - x_i * gamma_i / (r^2 * D)
        dy_i/dr = -x_i * gamma_i / (r^2)

        Summing via chain rule:
        dL/dx_i = (1/r) * gamma_i * (d_i - mean(d_j * y_j) * y_i)

    The key insight: dx can be computed from the OUTPUT y, not from x
    directly. This is why we must save x to recompute y in backward.

    Parameters
    ----------
    ctx : Context
        PyTorch autograd context, saves x and gamma for backward.
    x : torch.Tensor, shape (..., D)
        Input activations.
    gamma : torch.Tensor, shape (D,)
        Learnable scale parameter.

    Returns
    -------
    torch.Tensor, shape (..., D)
        Normalized, scaled output.

    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        gamma: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Forward pass: apply RMS normalization via Triton kernel.

        Parameters
        ----------
        ctx : Any
            Autograd context for saving tensors.
        x : torch.Tensor, shape (..., D)
            Input activations. Must be on CUDA.
        gamma : torch.Tensor, shape (D,)
            Learnable scale parameter. Must be on CUDA.
        eps : float, optional
            Epsilon for numerical stability. Default: 1e-6.

        Returns
        -------
        torch.Tensor, shape (..., D)
            RMSNorm(x, gamma, eps) = x / sqrt(mean(x^2) + eps) * gamma.

        Notes
        -----
        Tensor shapes are flattened to (n_rows, D) where n_rows = prod(
        *x.shape[:-1]), processed row-by-row by the Triton kernel, then
        reshaped back to the original shape. Internal computation uses
        float32 for numerical stability regardless of input dtype.

        """
        # Validate shapes and device placement
        if x.dim() < 2:
            raise ValueError(f"x must have at least 2 dimensions, got {x.dim()}")
        if gamma.dim() != 1:
            raise ValueError(f"gamma must be 1D, got {gamma.dim()}")
        if x.shape[-1] != gamma.shape[0]:
            raise ValueError(
                f"Last dim of x ({x.shape[-1]}) must match gamma ({gamma.shape[0]})"
            )
        if x.device.type != "cuda":
            raise ValueError("x must be on CUDA device")
        if gamma.device.type != "cuda":
            raise ValueError("gamma must be on CUDA device")
        if x.dtype != gamma.dtype:
            raise ValueError(f"x and gamma must have same dtype, got {x.dtype} and {gamma.dtype}")

        # Save inputs for backward pass (needed to recompute y)
        ctx.save_for_backward(x, gamma)
        ctx.eps = eps
        ctx.n_features = x.shape[-1]

        # Flatten last dimension to (n_rows, D) for row-major processing
        original_shape = x.shape
        D = x.shape[-1]
        n_rows = x.numel() // D  # batch * sequence length

        x_flat = x.contiguous().view(n_rows, D)  # (n_rows, D) row-major
        out_flat = torch.empty_like(x_flat)  # Pre-allocate output

        # Select BLOCK_SIZE based on feature dimension for optimal occupancy
        # Power of 2 ensures efficient memory coalescing and shared memory usage
        BLOCK_SIZE = triton.next_power_of_2(D)
        BLOCK_SIZE = max(BLOCK_SIZE, 128)  # Guarantee >= 128 for good GPU utilization

        # Launch: 1D grid with n_rows blocks, one per row/sequence position
        _rmsnorm_kernel[(n_rows,)](  # pyright: ignore[reportCallIssue]
            x_flat,
            gamma,
            out_flat,
            D,
            n_rows,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Reshape back to original dimensions
        result = out_flat.view(original_shape)

        # Cast output to match input dtype (kernel computes in float32)
        if result.dtype != x.dtype:
            result = result.to(x.dtype)

        return result

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Backward pass: compute dL/dx and dL/dgamma.

        RMSNorm backward derivation:
        ─────────────────────────
        Forward: y = x / r * gamma,  where r = sqrt(mean(x^2) + eps)

        dL/dgamma:
          = sum(output * grad_output) over batch/seq  → shape (D,)

        dL/dx:
          = (1/r) * gamma * (grad_output - mean(grad_output * y) * y)
          → shape (batch, seq, D)

        where mean is taken over the feature dimension D.

        Key insight: the gradient computation uses the FORWARD output y,
        not the input x. This means we must save x during forward to
        recompute y = x / r * gamma in the backward pass.

        Parameters
        ----------
        ctx : context
            Autograd context with saved x and gamma.
        grad_output : torch.Tensor, shape (..., D)
            Gradient from upstream layer.

        Returns
        -------
        grad_x : torch.Tensor, shape (..., D)
            Gradient w.r.t. input activations.
        grad_gamma : torch.Tensor, shape (D,)
            Gradient w.r.t. scale parameter.
        None : None
            Placeholder for eps (not learned).

        """
        x, gamma = ctx.saved_tensors
        eps = ctx.eps
        D = ctx.n_features

        # Flatten to 2D: (n_rows, D) for per-row reduction
        x_flat = x.view(-1, D)  # (n_rows, D)
        grad_out_flat = grad_output.contiguous().view(-1, D)  # (n_rows, D)

        n_rows, D_actual = x_flat.shape
        assert D_actual == D

        # ── Compute dgamma: element-wise product + sum ────────────
        # Reconstruct y from saved x: y = (x / r) * gamma
        # where r = sqrt(mean(x^2) + eps)
        mean_x_sq = torch.mean(x_flat ** 2, dim=-1, keepdim=True)  # (n_rows, 1)
        rms_x = torch.sqrt(mean_x_sq + eps)  # (n_rows, 1)
        y = (x_flat / rms_x) * gamma  # (n_rows, D) — forward output

        # dgamma = sum(y * grad_output) — accumulate per-feature
        dgamma = torch.sum(y * grad_out_flat, dim=0)  # (D,)

        # ── Compute dx: chain rule through RMSNorm ────────────────
        # mean(dy_y) = sum_j(grad_out_ij * y_ij) / D → (n_rows, 1)
        # This is the projection of grad_output onto the output direction,
        # captured for each row independently.
        mean_dy_y = torch.sum(grad_out_flat * y, dim=-1, keepdim=True) / D  # (n_rows, 1)

        # dx = (1/r) * gamma * (grad_out - mean(dy_y) * y)
        # This formula is derived from the chain rule through the
        # RMS normalization — see the class docstring for derivation.
        dx = (1.0 / rms_x) * gamma * (grad_out_flat - mean_dy_y * y)  # (n_rows, D)

        # Reshape gradients back to original input shape
        return dx.view_as(x), dgamma, None, None


def rmsnorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply RMS normalization via Triton GPU kernel.

    RMSNorm outputs y = x / sqrt(mean(x^2) + eps) * gamma, where the
    mean is taken over the last dimension (D).

    This is the normalization standard in modern LLMs (LLaMA, GPT-2,
    Transformer-XL) because it is cheaper than LayerNorm while
    maintaining comparable accuracy. Unlike LayerNorm, it does not
    subtract the mean, relying instead on the residual connections
    to keep activations centered.

    Parameters
    ----------
    x : torch.Tensor, shape (..., D)
        Input activations. Must be on CUDA. Dtype must be float32
        or float64 to match gamma.
    gamma : torch.Tensor, shape (D,)
        Learnable scale parameter. Same dtype as x.
    eps : float, optional
        Epsilon for numerical stability to prevent division by zero.
        Default: 1e-6.

    Returns
    -------
    torch.Tensor, shape (..., D)
        RMS-normalized output, then scaled by gamma. Same dtype as x.

    Notes
    -----
    Internal computation uses float32 for numerical stability even
    when inputs are float64. The kernel processes each row (token
    position) independently, normalizing over the feature dimension.

    Example
    -------
    >>> import torch
    >>> x = torch.randn(2, 8, 64, device='cuda')  # Batch, Seq, Dim
    >>> gamma = torch.ones(64, device='cuda')
    >>> y = rmsnorm(x, gamma)
    >>> y.shape
    torch.Size([2, 8, 64])
    >>> torch.std(y, dim=-1).mean().item()  # Roughly 1.0 (unit variance)
    0.999...

    Reference
    ---------
    Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
    https://arxiv.org/abs/1910.07467

    """
    return _RmsNormTriton.apply(x, gamma, eps)
