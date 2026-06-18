import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr,  # pyright: ignore[reportInvalidTypeForm]
    gamma_ptr,  # pyright: ignore[reportInvalidTypeForm]
    out_ptr,  # pyright: ignore[reportInvalidTypeForm]
    n_features,  # pyright: ignore[reportInvalidTypeForm]
    n_rows,  # pyright: ignore[reportInvalidTypeForm]
    eps,  # pyright: ignore[reportInvalidTypeForm]
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Triton kernel for RMS normalization.

    RMSNorm formula: out = x / sqrt(mean(x^2) + eps) * gamma

    Each row of the input matrix is processed by one program instance.
    The reduction over the last dimension (features) is performed in-kernel.

    Parameters
    ----------
    x_ptr : pointer
        Pointer to input tensor of shape (n_rows, n_features), stored in row-major order.
    gamma_ptr : pointer
        Pointer to gamma tensor of shape (n_features,).
    out_ptr : pointer
        Pointer to output tensor of shape (n_rows, n_features), stored in row-major order.
    n_features : int
        Number of features (last dimension size).
    n_rows : int
        Number of rows (batch_size * seq_len flattened).
    eps : float
        Small constant for numerical stability.
    BLOCK_SIZE : int
        Block size for feature dimension (compile-time constant).
    """
    # Row index — each program instance handles one row
    row_idx = tl.program_id(0)
    # Feature (column) indices
    cols = tl.arange(0, BLOCK_SIZE)

    # Compute row offset in the flat pointer
    row_offset = row_idx * n_features

    # First pass: compute mean(x^2) for this row
    # Accumulate x^2 over all feature blocks — use block-sized accumulator for Triton compatibility
    x_sq_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start in range(0, n_features, BLOCK_SIZE):
        feature_idx = start + cols
        mask = feature_idx < n_features
        x_block = tl.load(x_ptr + row_offset + feature_idx, mask=mask, other=0.0).to(tl.float32)
        x_sq_acc = tl.where(mask, x_block * x_block, x_sq_acc)

    # Sum over all features to get a single scalar
    x_sq_sum = tl.sum(x_sq_acc)

    # Compute RMS from the partial sums
    rms = tl.sqrt(x_sq_sum / n_features + eps)  # scalar float32

    # Second pass: normalize and scale
    for start in range(0, n_features, BLOCK_SIZE):
        feature_idx = start + cols
        mask = feature_idx < n_features
        x_block = tl.load(x_ptr + row_offset + feature_idx, mask=mask, other=0.0).to(tl.float32)
        gamma_block = tl.load(gamma_ptr + feature_idx, mask=mask, other=0.0).to(tl.float32)
        out_block = (x_block / rms) * gamma_block
        tl.store(out_ptr + row_offset + feature_idx, out_block, mask=mask)


class _RmsNormTriton(torch.autograd.Function):
    """Custom autograd function for RMS normalization.

    Forward pass: Triton GPU kernel.
    Backward pass: PyTorch formula (well-known gradient for RMSNorm).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        gamma: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Apply RMS normalization.

        Parameters
        ----------
        x : torch.Tensor, shape (..., D)
            Input activations.
        gamma : torch.Tensor, shape (D,)
            Learnable scale parameter.
        eps : float
            Epsilon for numerical stability.

        Returns
        -------
        torch.Tensor, shape (..., D)
            Normalized output scaled by gamma.
        """
        # Validate shapes
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

        original_shape = x.shape
        D = x.shape[-1]
        n_rows = x.numel() // D

        # Flatten to 2D: (B*S, D)
        x_flat = x.contiguous().view(n_rows, D)
        out_flat = torch.empty_like(x_flat)

        # Launch kernel with BLOCK_SIZE tuned to D
        BLOCK_SIZE = triton.next_power_of_2(D)
        BLOCK_SIZE = max(BLOCK_SIZE, 128)  # Ensure good occupancy

        _rmsnorm_kernel[
            (n_rows,)  # 1D grid — one program per row
        ](
            x_flat,
            gamma,
            out_flat,
            D,
            n_rows,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Save for backward
        ctx.save_for_backward(x, gamma)
        ctx.eps = eps
        ctx.n_features = D

        out_flat = torch.empty_like(x_flat)

        # Launch kernel with BLOCK_SIZE tuned to D
        BLOCK_SIZE = triton.next_power_of_2(D)
        BLOCK_SIZE = max(BLOCK_SIZE, 128)  # Ensure good occupancy

        _rmsnorm_kernel[
            (n_rows,)  # 1D grid — one program per row
        ](
            x_flat,
            gamma,
            out_flat,
            D,
            n_rows,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        result = out_flat.view(original_shape)
        # Ensure output dtype matches input dtype (kernel computes in float32 internally)
        if result.dtype != x.dtype:
            result = result.to(x.dtype)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """Compute gradient w.r.t. input and gamma.

        RMSNorm backward formula:
        - dgamma: sum over batch/seq of (output * grad_output)
        - dx: (1 / rms) * (gamma * (d_out - mean(d_out * output) * output))

        where mean is computed over the feature dimension.
        """
        x, gamma = ctx.saved_tensors
        eps = ctx.eps
        D = ctx.n_features

        # grad_output shape: (..., D)
        # Reconstruct batch/seq dimensions from flattened x
        x_flat = x.view(-1, D)
        grad_out_flat = grad_output.contiguous().view(-1, D)

        n_rows, D_actual = x_flat.shape
        assert D_actual == D

        # === Compute dgamma (w.r.t. gamma) ===
        # dgamma = sum_{i,j} output[i,j] * grad_output[i,j]  → shape (D,)
        out = (x_flat / torch.sqrt(torch.mean(x_flat ** 2, dim=-1, keepdim=True) + eps)) * gamma
        dgamma = torch.sum(out * grad_out_flat, dim=0)  # (D,)

        # === Compute dx (w.r.t. input x) ===
        # rms_x shape: (n_rows, 1)
        rms_x = torch.sqrt(torch.mean(x_flat ** 2, dim=-1, keepdim=True) + eps)  # (n_rows, 1)

        # Normalize output for reuse
        # y = x / rms_x * gamma  →  y / gamma = x / rms_x
        y_normed = out / gamma  # (n_rows, D) — output normalized

        # Mean of (grad_out * y) over feature dim
        # mean_dy_y = sum_j (d_out_ij * y_j) / D  → shape (n_rows, 1)
        mean_dy_y = torch.sum(grad_out_flat * y_normed, dim=-1, keepdim=True) / D  # (n_rows, 1)

        # dx = (1 / rms_x) * gamma * (d_out - mean(d_out * y) * y)
        dx = (1.0 / rms_x) * gamma * (grad_out_flat - mean_dy_y * y_normed)  # (n_rows, D)

        return dx.view_as(x), dgamma, None, None


def rmsnorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply RMS normalization via Triton GPU kernel.

    Parameters
    ----------
    x : torch.Tensor, shape (..., D)
        Input activations. Must be on CUDA.
    gamma : torch.Tensor, shape (D,)
        Learnable scale parameter. Must be on CUDA.
    eps : float
        Epsilon for numerical stability.

    Returns
    -------
    torch.Tensor, shape (..., D)
        RMS-normalized output scaled by gamma.

    Notes
    -----
    RMSNorm formula: out = x / sqrt(mean(x^2) + eps) * gamma
    where mean is taken over the last dimension (D).
    """
    return _RmsNormTriton.apply(x, gamma, eps)
