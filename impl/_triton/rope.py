import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    x_ptr,  # pyright: ignore[reportInvalidTypeForm]
    out_ptr,  # pyright: ignore[reportInvalidTypeForm]
    cos_ptr,  # pyright: ignore[reportInvalidTypeForm]
    sin_ptr,  # pyright: ignore[reportInvalidTypeForm]
    num_tokens,  # pyright: ignore[reportInvalidTypeForm]
    num_pairs,  # pyright: ignore[reportInvalidTypeForm]
    head_dim,  # pyright: ignore[reportInvalidTypeForm]
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Triton kernel for RoPE rotation.

    Each program handles one (token, pair_block) — loads 2 elements,
    applies rotation, stores 2 elements.

    Parameters
    ----------
    x_ptr : pointer
        Input tensor of shape (num_tokens, head_dim).
    out_ptr : pointer
        Output tensor of same shape.
    cos_ptr : pointer
        Cos table of shape (num_tokens, num_pairs).
    sin_ptr : pointer
        Sin table of shape (num_tokens, num_pairs).
    num_tokens : int
        Total number of tokens.
    num_pairs : int
        Number of (odd, even) pairs = head_dim // 2.
    head_dim : int
        Total head dimension.
    BLOCK_SIZE : int
        Pairs per program block.
    """
    token_idx = tl.program_id(0)
    pair_block_idx = tl.program_id(1)

    token_offset = token_idx * head_dim
    pair_range = pair_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    pair_mask = pair_range < num_pairs
    dim_range = 2 * pair_range
    inner_offset = token_offset + dim_range

    x_m = tl.load(x_ptr + inner_offset, mask=pair_mask, other=0.0).to(tl.float32)
    x_m1 = tl.load(x_ptr + inner_offset + 1, mask=pair_mask, other=0.0).to(tl.float32)

    cos_val = tl.load(cos_ptr + token_idx * num_pairs + pair_range, mask=pair_mask, other=0.0).to(tl.float32)
    sin_val = tl.load(sin_ptr + token_idx * num_pairs + pair_range, mask=pair_mask, other=0.0).to(tl.float32)

    y_m = x_m * cos_val - x_m1 * sin_val
    y_m1 = x_m * sin_val + x_m1 * cos_val

    # Accumulate outputs
    out_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    out_vals = tl.where(pair_mask, y_m, out_vals)
    tl.store(out_ptr + inner_offset, out_vals, mask=pair_mask)

    out_vals2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    out_vals2 = tl.where(pair_mask, y_m1, out_vals2)
    tl.store(out_ptr + inner_offset + 1, out_vals2, mask=pair_mask)


@triton.jit
def _rope_copy_kernel(
    x_ptr,  # pyright: ignore[reportInvalidTypeForm]
    out_ptr,  # pyright: ignore[reportInvalidTypeForm]
    num_tokens,  # pyright: ignore[reportInvalidTypeForm]
    tail_dim,  # pyright: ignore[reportInvalidTypeForm]
    stride_dim,  # pyright: ignore[reportInvalidTypeForm]
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Copy tail dimensions through unchanged (RoPE pass-through)."""
    token_idx = tl.program_id(0)
    token_offset = token_idx * stride_dim
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < tail_dim
    vals = tl.load(x_ptr + token_offset + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + token_offset + cols, vals, mask=mask)


class _RoPETriton(torch.autograd.Function):
    """Custom autograd function for RoPE.

    Forward pass: Triton GPU kernel.
    Backward pass: PyTorch (reverse rotation with swapped sin).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary positional embedding on flattened 2D tensor.

        Parameters
        ----------
        x : torch.Tensor, shape (num_tokens, D)
            Flattened Q or K tensor.
        cos : torch.Tensor, shape (num_tokens, D//2)
            Per-token cosine values.
        sin : torch.Tensor, shape (num_tokens, D//2)
            Per-token sine values.

        Returns
        -------
        torch.Tensor, shape (num_tokens, D)
            Rotated tensor.
        """
        if x.device.type != "cuda":
            raise ValueError("x must be on CUDA")
        if x.dim() != 2:
            raise ValueError(f"x must be 2D (num_tokens, D), got {x.dim()}D")

        orig_shape = x.shape
        num_tokens, D = orig_shape
        pair_dim = D // 2

        if pair_dim % 2 != 0:
            raise ValueError(f"head_dim must be divisible by 4, got {D}")

        x_flat = x.contiguous()
        out_flat = torch.empty_like(x_flat)

        BLOCK_SIZE = 32
        num_pair_blocks = triton.cdiv(pair_dim, BLOCK_SIZE)

        grid = (num_tokens, num_pair_blocks)
        _rope_kernel[grid](
            x_flat,
            out_flat,
            cos,
            sin,
            num_tokens,
            pair_dim,
            D,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(cos, sin)

        return out_flat

    @staticmethod
    def backward(ctx, grad_output):
        """Backward for RoPE: swap sin → -sin for reverse rotation."""
        cos, sin = ctx.saved_tensors
        grad = _RoPETriton.apply(grad_output, cos, -sin)
        return grad, None, None


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    rope_dim: int = 0,
) -> torch.Tensor:
    """Apply Rotary Positional Embedding via Triton GPU kernel.

    Parameters
    ----------
    x : torch.Tensor, shape (B, S, H, D)
        Q or K tensor.
    positions : torch.Tensor, shape (S,)
        Position indices for each sequence element.
    rope_dim : int, optional
        Number of head_dims to rotate. 0 = full head_dim rotation.

    Returns
    -------
    torch.Tensor, shape (B, S, H, D)
        Rotated tensor.

    Notes
    -----
    RoPE rotates each (odd, even) dimension pair by position-dependent angle:
      x_m' = x_m * cos(mθ) - x_{m+1} * sin(mθ)
      x_{m+1}' = x_m * sin(mθ) + x_{m+1} * cos(mθ)
    where θ = 10000^(-2k/d) for the k-th dimension pair.
    """
    B, S, H, D = x.shape

    # Separate unrotated dims (after rope_dim)
    if rope_dim > 0 and rope_dim < D:
        x_rot = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]
    else:
        x_rot = x
        x_pass = None

    positions = positions.to(x_rot.device, dtype=torch.int64)
    max_pos = int(positions.max()) + 1

    # Compute frequencies for the rotating portion
    cos_full, sin_full = _compute_rope_frequencies(
        x_rot.shape[-1], max_pos, x_rot.device, dtype=torch.float64
    )

    # rotating_dim = number of dims to rotate (rope_dim or D)
    rotating_dim = rope_dim if rope_dim > 0 else D
    pair_count = rotating_dim // 2

    pos_per_token = positions[torch.arange(B * S * H, device=positions.device)
                                  % (S * H) // H]  # (B*S*H,)

    # Build per-token cos/sin tables: (B*S*H, pair_count)
    cos_token = cos_full[pos_per_token, :pair_count].to(x_rot.dtype)
    sin_token = sin_full[pos_per_token, :pair_count].to(x_rot.dtype)

    # Apply RoPE using the Triton kernel
    x_flat = x_rot.flatten(0, 2)  # (B*S*H, rotating_dim)

    # For dims beyond rotating_dim, pass through unchanged
    if pair_count * 2 < rotating_dim:
        # This shouldn't happen since rotating_dim = pair_count * 2 by construction
        raise ValueError(f"rotating_dim {rotating_dim} must equal pair_count * 2 = {pair_count * 2}")

    out_flat = _RoPETriton.apply(x_flat, cos_token, sin_token)

    if x_pass is not None:
        # Concatenate rotated + pass-through
        out_flat = torch.cat([out_flat, x_pass.flatten(0, 2)], dim=-1)
    return out_flat.view(B, S, H, D)


def _compute_rope_frequencies(
    head_dim: int,
    max_position: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> tuple:
    """Compute RoPE frequencies and cos/sin tables.

    Parameters
    ----------
    head_dim : int
        Head dimension (>= 2, even).
    max_position : int
        Maximum position index.
    device : torch.device
        CUDA device.
    dtype : torch.dtype
        Output dtype.

    Returns
    -------
    cos : torch.Tensor, shape (max_position, head_dim // 2)
    sin : torch.Tensor, shape (max_position, head_dim // 2)
    """
    if head_dim < 2 or head_dim % 2 != 0:
        raise ValueError(f"head_dim must be >= 2 and even, got {head_dim}")

    pair_dim = head_dim // 2

    # freqs: (pair_dim,) = 1 / 10000^(2k / head_dim)
    freqs = 1.0 / (10000.0 ** (torch.arange(pair_dim, device=device, dtype=dtype) * 2.0 / head_dim))

    # positions: (max_position,)
    positions = torch.arange(max_position, device=device, dtype=dtype)

    # angles: (max_position, pair_dim)
    angles = positions[:, None] * freqs[None, :]

    return torch.cos(angles), torch.sin(angles)
