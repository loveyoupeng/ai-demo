"""FlashAttention-style online-softmax attention kernel — Triton.

The existing Triton attention kernel (``impl/_triton/attn.py::_attn_fwd_kernel``)
computes attention in two passes over the key/value dimension:

    1. Load ALL of K and V into registers, compute the full (BLOCK_M, Sk)
       score matrix.
    2. Softmax the full score matrix.
    3. Multiply by V.

This materializes the (BLOCK_M, Sk) score matrix in shared memory, so peak
memory scales with the *full* key length Sk. For long sequences this is the
bottleneck: a (2048, 2048) score tile is ~16 MB per head in float32.

The FlashAttention insight (Dao et al., 2022) is that softmax can be computed
*online* — in a single pass over the keys — without ever materializing the
full score matrix. The key identity:

    softmax([a; b]) = [exp(a - m) * e^(m_a - m); exp(b - m) * e^(m_b - m)]
    where m = logaddexp(m_a, m_b) is the log-sum-exp of the two blocks.

Concretely, for each key block we:
    1. Compute the block's scores: s = q @ k_block^T / sqrt(D)  (BLOCK_M, BLOCK_N)
    2. Update the running max: m_new = max(m_old, max(s, axis=1))
    3. Rescale the running sum: l_new = l_old * exp(m_old - m_new) + sum(exp(s - m_new))
    4. Rescale the running accumulator: acc = acc * exp(m_old - m_new) + exp(s - m_new) @ v_block
    5. After all blocks: out = acc / l

Peak memory is now O(BLOCK_M * D + BLOCK_N * D) — independent of Sk. The
tradeoff: we do one extra pass over K/V per query tile (to recompute the
scores for the rescale), which is a compute/memory-bandwidth trade that
favors the online version for long sequences (memory-bound) and is roughly
neutral for short ones.

Reference
---------
Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention"
(2022). https://arxiv.org/abs/2205.14135
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Minimum dimension for tl.dot — ensures we don't hit CUDA compute < 70 limits.
_MIN_KERNEL_DIM = 16


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Compute scaled dot-product attention via the Triton online-softmax kernel.

    Same math as ``scaled_dot_product_attention`` in ``impl/_triton/attn.py``,
    but the forward pass uses the FlashAttention online-softmax algorithm:
    a single pass over the keys, no materialized (BLOCK_M, Sk) score matrix.

    Parameters
    ----------
    q : torch.Tensor, shape (B, H, Sq, D)
        Query tensor.
    k : torch.Tensor, shape (B, H, Sk, D)
        Key tensor.
    v : torch.Tensor, shape (B, H, Sk, D)
        Value tensor.
    is_causal : bool, default False
        When True, apply the causal mask (position i attends to keys j <= i).

    Returns
    -------
    torch.Tensor, shape (B, H, Sq, D)
        Attention output.

    Memory vs the two-pass kernel
    -----------------------------
    Two-pass (``attn.py``):  O(Sk * D) per program — the full K/V tiles.
    Online (this module):    O(BLOCK_M * D + BLOCK_N * D) per program —
                             independent of Sk. For (B=2, H=16, Sq=2048,
                             Sk=2048, D=64, BLOCK_M=BLOCK_N=32):
                             two-pass: 2 * 2048 * 64 * 4 = 1 MB (K+V per program)
                             online:   2 * 32 * 64 * 4 = 16 KB (tiles per program)
    """
    assert q.device.type == "cuda"
    assert k.device.type == "cuda"
    assert v.device.type == "cuda"
    B, H, Sq, D = q.shape
    _, _, Sk, _ = k.shape
    BLOCK_N = 16
    Sk_pad = int(triton.cdiv(Sk, BLOCK_N) * BLOCK_N)
    # Pad D to power-of-2 for tl.dot.
    D_pad = int(max(_MIN_KERNEL_DIM, triton.next_power_of_2(D)))

    # Zero-pad K and V to (B, H, Sk_pad, D_pad).
    k_pad = torch.zeros((B, H, Sk_pad, D_pad), device=q.device, dtype=q.dtype)
    v_pad = torch.zeros((B, H, Sk_pad, D_pad), device=q.device, dtype=q.dtype)
    k_pad[:, :, :Sk, :D] = k
    v_pad[:, :, :Sk, :D] = v
    BLOCK_M = 16
    grid = (B, H, triton.cdiv(Sq, BLOCK_M))
    # Output: (B, H, Sq, D_pad) — the kernel writes D_pad columns; crop after.
    out = torch.zeros((B, H, Sq, D_pad), device=q.device, dtype=torch.float32)

    _flash_attn_fwd_kernel[grid](
        q,
        k_pad,
        v_pad,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_pad.stride(0),
        k_pad.stride(1),
        k_pad.stride(2),
        k_pad.stride(3),
        v_pad.stride(0),
        v_pad.stride(1),
        v_pad.stride(2),
        v_pad.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        H,
        Sq,
        D,
        Sk,
        1 if is_causal else 0,
        D_pad=D_pad,  # pyright: ignore[reportArgumentType]
        Sk_pad=Sk_pad,  # pyright: ignore[reportArgumentType]
        scale=D**-0.5,
        BLOCK_M=BLOCK_M,  # pyright: ignore[reportArgumentType]
        BLOCK_N=BLOCK_N,  # pyright: ignore[reportArgumentType]
    )
    # Note: the kernel is launched with default num_stages; for very long
    # sequences the shared-memory usage can exceed the GPU limit. The
    # caller can reduce BLOCK_M/BLOCK_N to fit.

    # Crop to the actual D columns.
    return out[:, :, :, :D].to(q.dtype)


@triton.jit
def _flash_attn_fwd_kernel(
    Q,
    K,
    V,
    Out,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    H,
    Sq,
    D,
    Sk,
    IS_CAUSAL,
    D_pad: tl.constexpr,
    Sk_pad: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Triton kernel: FlashAttention online-softmax attention.

    For each (batch, head, Q-tile), process the keys in BLOCK_N-sized blocks,
    updating the running max/sum/accumulator online (no full score matrix).

    Algorithm (per Q-tile):
        m = -inf  (running max, [BLOCK_M, 1])
        l = 0     (running sum of exp, [BLOCK_M, 1])
        acc = 0   (running accumulator, [BLOCK_M, D_pad])

        for n in range(0, Sk_pad, BLOCK_N):
            k_block = K[:, :, n:n+BLOCK_N, :]   # (BLOCK_N, D_pad)
            v_block = V[:, :, n:n+BLOCK_N, :]   # (BLOCK_N, D_pad)
            s = q_block @ k_block^T * scale     # (BLOCK_M, BLOCK_N)
            if IS_CAUSAL: mask s where key_idx > q_idx
            m_new = max(m, max(s, axis=1, keepdims=True))
            # Rescale the running state to the new max.
            alpha = exp(m - m_new)               # (BLOCK_M, 1)
            p = exp(s - m_new)                   # (BLOCK_M, BLOCK_N)
            l = l * alpha + sum(p, axis=1, keepdims=True)
            acc = acc * alpha + p @ v_block
            m = m_new

        out = acc / l

    Parameters
    ----------
    See the two-pass kernel in ``attn.py`` for the full parameter list.
    The only additions are BLOCK_N (key-tile size) and the online-softmax
    state (m, l, acc) which live in registers, not shared memory.
    """
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=2)

    row_start = pid_m * BLOCK_M

    # ---- Load Q tile: [BLOCK_M, D_pad] ----
    q_row = (row_start + tl.arange(0, BLOCK_M))[:, None]  # [BLOCK_M, 1]
    q_col = tl.arange(0, D_pad)[None, :]  # [1, D_pad]
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + q_row * stride_qs + q_col * stride_qd
    q_mask = (q_row < Sq) & (q_col < D)
    q_block = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)  # [BLOCK_M, D_pad]

    # ---- Online-softmax state (in registers, [BLOCK_M, 1] or [BLOCK_M, D_pad]) ----
    m = tl.full([BLOCK_M, 1], float("-inf"), dtype=tl.float32)  # running max
    lsum = tl.zeros([BLOCK_M, 1], dtype=tl.float32)  # running sum of exp
    acc = tl.zeros([BLOCK_M, D_pad], dtype=tl.float32)  # running accumulator

    k_col = tl.arange(0, D_pad)[None, :]  # [1, D_pad]

    # ---- Iterate over key blocks ----
    for n in range(0, Sk_pad, BLOCK_N):
        # Load K block: [BLOCK_N, D_pad]
        k_row = (n + tl.arange(0, BLOCK_N))[:, None]  # [BLOCK_N, 1]
        k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + k_row * stride_ks + k_col * stride_kd
        k_mask = (k_row < Sk) & (k_col < D)
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK_N, D_pad]

        # Load V block: [BLOCK_N, D_pad]
        v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + k_row * stride_vs + k_col * stride_vd
        v_block = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK_N, D_pad]

        # Scores: [BLOCK_M, D_pad] @ [D_pad, BLOCK_N] → [BLOCK_M, BLOCK_N]
        s = tl.dot(q_block, k_block.T, input_precision="ieee") * scale  # [BLOCK_M, BLOCK_N]

        # Mask padded keys.
        key_idx = tl.arange(0, BLOCK_N)[None, :]  # [1, BLOCK_N]
        s = tl.where(key_idx + n < Sk, s, float("-inf"))  # absolute key position < Sk

        # Causal mask: query row i attends only to keys j <= i.
        if IS_CAUSAL:
            q_idx = (row_start + tl.arange(0, BLOCK_M))[:, None]  # [BLOCK_M, 1]
            s = tl.where((key_idx + n) <= q_idx, s, float("-inf"))

        # Online-softmax update.
        m_new = tl.maximum(m, s.max(axis=1, keep_dims=True))  # [BLOCK_M, 1]
        # Avoid exp(-inf - -inf) = NaN: where m is -inf, use 0 for alpha.
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - m_new))  # [BLOCK_M, 1]
        p = tl.exp(s - m_new)  # [BLOCK_M, BLOCK_N]
        lsum = lsum * alpha + p.sum(axis=1, keep_dims=True)  # [BLOCK_M, 1]
        acc = acc * alpha + tl.dot(p, v_block, input_precision="ieee")  # [BLOCK_M, D_pad]
        m = m_new

    # ---- Final normalization: out = acc / lsum ----
    # Guard against lsum == 0 (all keys masked for this row).
    l_safe = tl.where(lsum > 0, lsum, 1.0)  # [BLOCK_M, 1]
    out = acc / l_safe  # [BLOCK_M, D_pad]

    # ---- Store output ----
    col_idx = tl.arange(0, D_pad)[None, :]  # [1, D_pad]
    out_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + q_row * stride_os + col_idx * stride_od
    out_mask = (q_row < Sq) & (col_idx < D)
    tl.store(out_ptrs, out, mask=out_mask)
