"""Rotary Positional Embedding (RoPE) kernel — Triton.

RoPE encodes absolute position information into each token's embedding
by applying a position-dependent rotation in 2D subspaces. This is the
positional encoding used in LLaMA, LLaVA, Qwen, and many other LLMs.

Algorithm
---------
RoPE rotates pairs of dimensions by an angle proportional to position:

  For each (odd, even) pair (x_{2m}, x_{2m+1}) at position p:

    [x'_{2m},   x'_{2m+1}] = [cos(p*θ_m), -sin(p*θ_m)] [x_{2m}]
                           [sin(p*θ_m),  cos(p*θ_m)] [x_{2m+1}]

  where θ_m = 10000^(-2m/d_model) is the frequency for dimension pair m.

Frequency schedule
------------------
θ_m = 10000^(-2m/d) for m = 0, 1, ..., d/2-1

This geometric progression creates a range of frequencies:
  θ_0   = 10000^0     ≈ 1       (slowest rotation, period ~2π)
  θ_1   = 10000^(-2/d)
  ...
  θ_{d/2-1} = 10000^(-(d-2)/d) ≈ 10000^{-1} (fastest rotation, period ~2π/10000)

Each pair (x_{2m}, x_{2m+1}) rotates at a different angular frequency,
creating a multi-scale positional signal. Larger position gaps → larger
angle differences → more distinguishable positions.

Why pair odd/even dimensions?
  The (odd, even) pairing means position 0 affects pair (d0, d1),
  position 1 affects pair (d2, d3), etc. The odd-dim encodes cos,
  the even-dim encodes sin. This creates the 2D rotation structure.

Memory access pattern
---------------------
The kernel is coalesced within each (token, pair_block) tile:
  - token_idx → rows (contiguous access within a row)
  - pair_block_idx → groups of dimension pairs processed in parallel
  - BLOCK_SIZE=32 means 32 pairs (64 dims) processed per (token, block)

Each program handles one (token, pair_block) combination:
  1. Load x_{2m}, x_{2m+1} for all m in this block
  2. Load cos_m, sin_m from position-dependent tables
  3. Apply rotation matrix
  4. Store results

Note: cos and sin tables are accessed per-token, causing row-strided
memory access. This is acceptable because:
  - The tables are small (max_positions × head_dim/2 floats)
  - They fit in cache for typical context lengths
  - The compute (rotation) is more memory-intensive than table lookup

Numerical stability
-------------------
- cos/sin computed in float64 then downcast to input dtype
- This prevents precision loss for large position indices
- Rotation matrix is orthogonal (det=1), so norms are preserved
- No overflow possible since |cos|, |sin| ≤ 1

BLOCK_SIZE selection
--------------------
- Fixed at 32: processes 32 dimension pairs (64 dimensions) per program
- For head_dim=64: one program per token (32 pairs)
- For head_dim=4096: head_dim/2/32 = 64 programs per tensor dimension
- 32 is chosen for good occupancy (32 threads × 32 warps = 1024)

Triton kernel design
--------------------
Two-kernel approach:
  1. _rope_kernel: Rotates the first head_dim dimensions
  2. _rope_copy_kernel: Passes through remaining dimensions unchanged

The separation allows different BLOCK_SIZE parameters for the rotation
(32) vs pass-through (tuned to tail_dim).

Reference
---------
Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding"
https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

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
    """Triton kernel: RoPE rotation on (num_tokens, head_dim) tensor.

    Each program instance handles one (token, pair_block) combination:
    - token_idx: which token in the batch (0 to num_tokens-1)
    - pair_block_idx: which chunk of dimension pairs (0 to num_pairs/BLOCK_SIZE-1)

    Algorithm
    ---------
    For each pair (m, m+1) in this block:
      y_m     = x_m * cos_m - x_{m+1} * sin_m
      y_{m+1} = x_m * sin_m + x_{m+1} * cos_m

    This is a standard 2D rotation matrix applied element-wise:
      [y_m,    y_{m+1}] = R(θ_m * position) [x_m, x_{m+1}]

    Where θ_m depends on the dimension pair index m, and position
    depends on the token index.

    Memory layout
    -------------
    x_ptr:   (num_tokens, head_dim) — row-major, contiguous within row
    out_ptr: (num_tokens, head_dim) — row-major, same layout
    cos_ptr: (num_tokens, num_pairs) — one frequency per token per pair
    sin_ptr: (num_tokens, num_pairs) — same shape as cos

    Why row-strided cos/sin access?
      Accessing cos_ptr[token * num_pairs + pair] means each row of
      cos/sin is contiguous (good cache behavior), but going from
      row to row jumps by a whole row stride. This is acceptable
      because the tables are small enough to fit in L2 cache.

    Parameters
    ----------
    x_ptr : pointer
        Input tensor (num_tokens, head_dim). Row-major layout.
    out_ptr : pointer
        Output tensor, same shape. Written back in-place (conceptually).
    cos_ptr : pointer
        Cosine table (num_tokens, num_pairs), precomputed per-token.
    sin_ptr : pointer
        Sine table (num_tokens, num_pairs), precomputed per-token.
    num_tokens : int
        Number of tokens (rows), typically B * S.
    num_pairs : int
        Number of dimension pairs = head_dim // 2.
    head_dim : int
        Total head dimension (number of columns).
    BLOCK_SIZE : constexpr int
        Number of dimension pairs per program. Each program handles
        BLOCK_SIZE pairs, i.e., 2*BLOCK_SIZE dimensions.

    Grid configuration
    ------------------
    Grid: (num_tokens, num_pairs/BLOCK_SIZE)
    Program (i, j): handles token i, pair range [j*BLOCK_SIZE, ...)

    Performance
    -----------
    - Each program: 2 reads from x, 2 reads from cos/sin, 2 stores
    - Total: 6 memory operations per dimension per token
    - Computation is memory-bound (rotate is just multiply-add)
    - Good for large batch sizes where memory bandwidth dominates

    Numerical notes
    ---------------
    - cos/sin values are float64 from _compute_rope_frequencies(),
      loaded as float32 for computation (acceptable precision for rotation)
    - No clipping needed: sin/cos are always in [-1, 1]
    - Mask handles variable sequence lengths (pad tokens get zeros)
    """
    # ── Program assignment ─────────────────────────────────────
    # Row (token) and column (pair block) indices identify the
    # tile of (num_tokens, head_dim) handled by this program.
    token_idx = tl.program_id(axis=0)       # Which token row [0..num_tokens)
    pair_block_idx = tl.program_id(axis=1)  # Which pair block [0..num_pairs/BLOCK_SIZE)

    # Compute offsets for this tile:
    # - token_offset: starting byte offset for this token's row
    # - pair_range: dimension pair indices in this block

    token_offset = token_idx * head_dim     # Row offset in x/output
    pair_range = pair_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # Pair indices
    pair_mask = pair_range < num_pairs      # Bounds: only valid pairs
    dim_range = 2 * pair_range              # Actual dimension indices: 2*m, 2*m+1
    inner_offset = token_offset + dim_range # Byte offset: row + per-dimension

    # Load input x values: x_m (even dims), x_{m+1} (odd dims)
    # Both are (BLOCK_SIZE, ) vectors, masked for valid pairs
    x_m = tl.load(
        x_ptr + inner_offset,              # Even: 2*m within row
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)  # (BLOCK_SIZE,) — odd dimension values
    x_m1 = tl.load(
        x_ptr + inner_offset + 1,          # Odd: 2*m+1 within row
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)  # (BLOCK_SIZE,) — even dimension values

    # Load cos/sin for this token's position and pair block
    # Cos/sin table layout: (num_tokens, num_pairs)
    # Access pattern: row-strided (jump by num_pairs between rows)
    cos_val = tl.load(
        cos_ptr + token_idx * num_pairs + pair_range,  # Row offset + column offset
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)  # (BLOCK_SIZE,) — frequencies for this position
    sin_val = tl.load(
        sin_ptr + token_idx * num_pairs + pair_range,  # Same indexing, sin table
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)  # (BLOCK_SIZE,)

    # ── Rotation computation ───────────────────────────────────
    # 2D rotation matrix applied to each (odd, even) pair:
    #
    #   [cos  -sin]   [x_m]     [x_m * cos - x_m1 * sin]
    #   [sin   cos] × [x_m1] =  [x_m * sin + x_m1 * cos]
    #
    # The key insight: position p affects the ANGLE of rotation,
    # while the pair index m determines the FREQUENCY. This creates
    # a multi-scale positional signal.

    y_m = x_m * cos_val - x_m1 * sin_val      # Rotated even dimension
    y_m1 = x_m * sin_val + x_m1 * cos_val      # Rotated odd dimension

    # ── Store results ──────────────────────────────────────────
    # Each output dimension gets its own tl.store call. We initialize
    # to zeros (masked elements get 0), then overlay only valid pairs.

    out_vals = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    out_vals = tl.where(pair_mask, y_m, out_vals)
    tl.store(out_ptr + inner_offset, out_vals, mask=pair_mask)  # Even dim

    out_vals2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    out_vals2 = tl.where(pair_mask, y_m1, out_vals2)
    tl.store(out_ptr + inner_offset + 1, out_vals2, mask=pair_mask)  # Odd dim


@triton.jit
def _rope_copy_kernel(
    x_ptr,  # pyright: ignore[reportInvalidTypeForm]
    out_ptr,  # pyright: ignore[reportInvalidTypeForm]
    num_tokens,  # pyright: ignore[reportInvalidTypeForm]
    tail_dim,  # pyright: ignore[reportInvalidTypeForm]
    stride_dim,  # pyright: ignore[reportInvalidTypeForm]
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Pass-through kernel: copy tail dimensions (not rotated) through.

    When rope_dim < head_dim, some dimensions at the end of the tensor
    are not involved in the rotation. This kernel copies them through
    unchanged, ensuring only rotating dims are touched.

    Layout: (num_tokens, stride_dim) with stride_dim ≥ tail_dim
    Access: row-strided copy of the first tail_dim columns.

    Parameters
    ----------
    x_ptr : pointer
        Input tensor (num_tokens, stride_dim).
    out_ptr : pointer
        Output tensor (num_tokens, stride_dim).
    num_tokens : int
        Number of token rows.
    tail_dim : int
        Number of unrotated dimensions at the end.
    stride_dim : int
        Total column count (may be larger than tail_dim for padding).
    BLOCK_SIZE : constexpr int
        Elements per block for the pass-through copy.
    """
    token_idx = tl.program_id(axis=0)
    token_offset = token_idx * stride_dim
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < tail_dim
    vals = tl.load(x_ptr + token_offset + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + token_offset + cols, vals, mask=mask)


class _RoPETriton(torch.autograd.Function):
    """Custom autograd function for RoPE.

    Forward pass: Triton GPU kernel (rotation matrix application).
    Backward pass: Re-run forward with swapped sin to reverse rotation.

    Why reverse rotation equals swapping sin?
    ------------------------------------------
    Forward:  [cos  -sin] [x_m] → [y_m]
              [sin   cos] [x_m1]   [y_m1]

    Inverse:  [cos   sin] [y_m] → [x_m]       (transpose of rotation)
              [-sin  cos] [y_m1]   [x_m1]

    The inverse rotation matrix is the transpose (cosine-symmetric,
    sine anti-symmetric). So we reuse the same kernel with
    sin → -sin to invert the operation.

    This trick avoids writing a separate backward kernel — the
    forward kernel IS its own inverse with a sign flip on sin.

    Parameters
    ----------
    ctx : context
        Saves cos and sin tables for backward pass.
    x : torch.Tensor, shape (num_tokens, D)
        Q or K tensor.
    cos : torch.Tensor, shape (num_tokens, D//2)
        Per-token cosine values.
    sin : torch.Tensor, shape (num_tokens, D//2)
        Per-token sine values.

    Returns
    -------
    torch.Tensor, shape (num_tokens, D)
        Rotated tensor.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RoPE rotation on a 2D tensor.

        Parameters
        ----------
        x : torch.Tensor, shape (num_tokens, D)
            Flattened Q or K tensor. Must be 2D, on CUDA.
        cos : torch.Tensor, shape (num_tokens, D//2)
            Precomputed cosine table.
        sin : torch.Tensor, shape (num_tokens, D//2)
            Precomputed sine table.

        Returns
        -------
        torch.Tensor, shape (num_tokens, D)
            Rotated tensor. Each (odd, even) pair rotated by
            position-dependent angle.

        Raises
        ------
        ValueError
            If x is not 2D, not on CUDA, or head_dim not divisible by 4.
        """
        if x.device.type != "cuda":
            raise ValueError("x must be on CUDA")
        if x.dim() != 2:
            raise ValueError(f"x must be 2D (num_tokens, D), got {x.dim()}D")

        orig_shape = x.shape  # (num_tokens, D)
        num_tokens, D = orig_shape
        pair_dim = D // 2     # Number of rotation pairs

        if pair_dim % 2 != 0:
            raise ValueError(f"head_dim must be divisible by 4, got {D}")

        x_flat = x.contiguous()
        out_flat = torch.empty_like(x_flat)

        # Fixed BLOCK_SIZE=32: processes 32 pairs (64 dims) per program
        BLOCK_SIZE = 32
        num_pair_blocks = triton.cdiv(pair_dim, BLOCK_SIZE)

        # Grid: (num_tokens, num_pair_blocks) — each token processed
        # in num_pair_blocks separate calls (one per BLOCK_SIZE chunk)
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
    def backward(
        ctx,
        *grad_outputs,  # type: ignore[override]
    ):
        """Backward pass: reverse rotation by flipping sin sign.

        Forward: y = R(θ) · x where R(θ) = [cos -sin; sin cos]
        Backward: dx = R(θ)^T · dy = R(-θ) · dy

        R(-θ) = [cos  sin; -sin cos], which is the same kernel call
        with sin → -sin. This is possible because the rotation matrix
        is orthogonal (R^T = R^{-1}).

        Parameters
        ----------
        grad_outputs : tuple[torch.Tensor]
            Gradient from upstream: (dL/dy,) of shape (num_tokens, D).

        Returns
        -------
        tuple[torch.Tensor, None, None]
            - dL/dx of same shape as input x (gradient w.r.t. Q/K)
            - None (no gradient w.r.t. cos)
            - None (no gradient w.r.t. sin)
        """
        cos, sin = ctx.saved_tensors
        # Reverse rotation: swap sin → -sin and apply forward kernel
        grad = _RoPETriton.apply(grad_outputs[0], cos, -sin)
        return grad, None, None


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    rope_dim: int = 0,
) -> torch.Tensor:
    """Apply Rotary Positional Embedding via Triton GPU kernel.

    RoPE encodes absolute position information into each query/key vector
    by applying a position-dependent rotation in 2D subspaces. Tokens at
    different positions rotate by different angles, creating a rich
    positional signal that attention can exploit.

    Algorithm
    ---------
    For each (odd, even) dimension pair at position p:

      [x'_m,     x'_{m+1}] = [cos(p·θ_m), -sin(p·θ_m)] [x_m]
                            [sin(p·θ_m),  cos(p·θ_m)] [x_{m+1}]

    where θ_m = 10000^(-2m/d) is the frequency for pair m, and p
    is the token position. The frequency decreases geometrically across
    pairs, creating multi-scale positional encoding.

    Parameters
    ----------
    x : torch.Tensor, shape (B, S, H, D)
        Q or K tensor. B=batch, S=sequence, H=heads, D=dim_per_head.
    positions : torch.Tensor, shape (S,)
        Position indices for each sequence element (0 to S-1).
    rope_dim : int, optional
        Number of head dimensions to rotate. 0 = rotate all D dims.
        Must be divisible by 2.

    Returns
    -------
    torch.Tensor, shape (B, S, H, D)
        Rotated tensor, same shape as input.

    Notes
    -----
    - The kernel processes dimensions in (odd, even) pairs: (0,1), (2,3), ...
    - Each pair rotates at a different frequency determined by the pair index
    - Position p controls the rotation angle for that pair
    - The rotation preserves vector norms (orthogonal transformation)
    - Non-rotated dimensions (if rope_dim < D) pass through unchanged

    Example
    -------
    >>> import torch
    >>> x = torch.randn(2, 8, 4, 16, device='cuda')  # (B, S, H, D)
    >>> positions = torch.arange(8, device='cuda')
    >>> y = apply_rope(x, positions)
    >>> y.shape
    torch.Size([2, 8, 4, 16])
    >>> torch.allclose(y.norm(dim=-1), x.norm(dim=-1))  # Norm preserved
    True

    Reference
    ---------
    Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864
    """
    B, S, H, D = x.shape

    # Separate rotated dims from unrotated tail dims
    if rope_dim > 0 and rope_dim < D:
        x_rot = x[..., :rope_dim]      # Dimensions to rotate
        x_pass = x[..., rope_dim:]      # Dimensions to pass through
    else:
        x_rot = x
        x_pass = None

    positions = positions.to(x_rot.device, dtype=torch.int64)
    max_pos = int(positions.max()) + 1

    # Compute frequencies for the rotating portion
    cos_full, sin_full = _compute_rope_frequencies(
        x_rot.shape[-1], max_pos, x_rot.device, dtype=torch.float64
    )

    # rotating_dim = number of dims to rotate (rope_dim or full D)
    rotating_dim = rope_dim if rope_dim > 0 else D
    pair_count = rotating_dim // 2

    # Generate per-token position indices: each token gets its position
    # Extract position for each (batch, seq, head) combination
    pos_per_token = positions[torch.arange(B * S * H, device=positions.device)
                                  % (S * H) // H]  # (B*S*H,)

    # Build per-token cos/sin tables: (B*S*H, pair_count)
    # Each token's row gets the frequencies at its position
    cos_token = cos_full[pos_per_token, :pair_count].to(x_rot.dtype)
    sin_token = sin_full[pos_per_token, :pair_count].to(x_rot.dtype)

    # Apply RoPE using the Triton kernel
    x_flat = x_rot.flatten(0, 2)  # (B*S*H, rotating_dim)
    out_flat = _RoPETriton.apply(x_flat, cos_token, sin_token)

    if x_pass is not None:
        out_flat = torch.cat([out_flat, x_pass.flatten(0, 2)], dim=-1)
    return out_flat.view(B, S, H, D)


def _compute_rope_frequencies(
    head_dim: int,
    max_position: int,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE frequencies and cos/sin tables.

    Generates the geometric frequency schedule and precomputes cos/sin
    values for all positions. This is done ONCE per model, not per-forward.

    Parameters
    ----------
    head_dim : int
        Head dimension (>= 2, even).
    max_position : int
        Maximum position index (context length).
    device : torch.device
        CUDA device.
    dtype : torch.dtype
        Output dtype (default float64 for precision).

    Returns
    -------
    cos : torch.Tensor, shape (max_position, head_dim // 2)
        Per-position cosine values.
    sin : torch.Tensor, shape (max_position, head_dim // 2)
        Per-position sine values.

    Notes
    -----
    Frequency formula: θ_m = 10000^(-2m / head_dim) for m = 0..D/2-1

    The 10000 base is empirically chosen — it creates a range of
    frequencies from period 2π (slowest) to period 2π/10000 (fastest),
    allowing the model to detect both local and distant position relations.
    """
    if head_dim < 2 or head_dim % 2 != 0:
        raise ValueError(f"head_dim must be >= 2 and even, got {head_dim}")

    pair_dim = head_dim // 2

    # freqs: (pair_dim,) = 1 / 10000^(2k / head_dim)
    freqs = 1.0 / (10000.0 ** (torch.arange(pair_dim, device=device, dtype=dtype) * 2.0 / head_dim))

    # positions: (max_position,)
    positions = torch.arange(max_position, device=device, dtype=dtype)

    # angles: (max_position, pair_dim)
    # Each row is the angle in radians for each dimension pair at that position
    angles = positions[:, None] * freqs[None, :]

    return torch.cos(angles), torch.sin(angles)
