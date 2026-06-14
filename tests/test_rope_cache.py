from __future__ import annotations

import numpy as np
from model.rope import apply_rope


def apply_rope_with_positions(x: np.ndarray, theta: np.ndarray, positions: np.ndarray) -> np.ndarray:
    r"""
    Apply RoPE where `positions[i]` is the absolute position for axis 1.
    
    This is the version needed for KV cache mode, where absolute positions
    must be tracked explicitly (since sequence length may be 1 but the token
    is at a large absolute position).
    
    Dimension tracking:
    =====================  ============================  ==================
    Symbol                 Input                        Output
    =====================  ============================  ==================
    x                      | :math:`[B, L, D]`           | :math:`[B, L, D]`
                           | :math:`[B, L, H, D]`        | :math:`[B, L, H, D]`
    theta (precomputed)    | :math:`[MaxPos, D//2]`      | used internally
    positions              | [:math:`L`]                 | used for indexing
    =====================  ============================  ==================
    
    >>> pos = np.array([0, 1, 2, 3], dtype=np.float64)
    >>> theta = pos.reshape(-1, 1)  # [4, 1] for dim=2
    >>> x = np.array([[[0., 1.]]], dtype=np.float64)  # [1, 1, 2]
    >>> result = apply_rope_with_positions(x, theta, np.array([0]))
    >>> np.allclose(result, x)  # pos 0 = identity
    True
    """
    # x: [B, L, D] or [B, L, H, D]
    # theta: [max_pos, D//2]  — precomputed for all possible absolute positions
    # positions: [L]  — absolute position index for each sequence element
    
    # Index theta by absolute positions to get per-position cos/sin
    cos = np.cos(theta[positions, :])  # [L, D//2]
    sin = np.sin(theta[positions, :])  # [L, D//2]
    
    if x.ndim == 3:
        # [L, D//2] → [1, L, D//2] for 3D x
        cos = cos[None, :, :]
        sin = sin[None, :, :]
    else:  # 4D
        # [L, D//2] → [1, L, 1, D//2] for x [B, L, H, D]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    
    # Split even and odd dimensions (along last axis)
    x_even = x[..., 0::2]  # [B, L, D//2] or [B, L, H, D//2]
    x_odd = x[..., 1::2]   # [B, L, D//2] or [B, L, H, D//2]
    
    # Apply 2D rotation: each pair (2i, 2i+1) is rotated independently
    x_rope_even = cos * x_even - sin * x_odd
    x_rope_odd = sin * x_even + cos * x_odd
    
    # Reconstruct by interleaving
    x_rope = np.empty_like(x)
    x_rope[..., 0::2] = x_rope_even
    x_rope[..., 1::2] = x_rope_odd
    
    return x_rope


def _compute_attn_scores(xq, xk):
    """
    Compute attention scores for [B, L, H, D] tensors.
    
    scores[b, m, n] = sum_{h, d} xq[b, m, h, d] * xk[b, n, h, d]
    
    Result: [B, L, L]
    """
    xq2 = xq.reshape(xq.shape[0], xq.shape[1], -1)  # [B, L, H*D]
    xk2 = xk.reshape(xk.shape[0], xk.shape[1], -1)  # [B, L, H*D]
    return xq2 @ xk2.transpose(0, 2, 1)  # [B, L, H*D] @ [B, H*D, L] → [B, L, L]


# ── Helpers ──────────────────────────────────────────────────────────

def _make_x(vec, l):
    """Tile a base vector into [1, L, 1, D]."""
    vec4 = vec.reshape(1, 1, 1, -1)  # [1, 1, 1, D]
    return np.repeat(vec4, l, axis=1)  # [1, L, 1, D]


def _precompute_theta(max_pos, head_dim, base=10000.0):
    """Compute theta for all positions 0..max_pos-1."""
    pair_count = head_dim // 2
    power = np.arange(pair_count, dtype=np.float64) * -2.0 / head_dim
    return np.arange(max_pos, dtype=np.float64).reshape(-1, 1) * (base ** power)


# ── Tests ────────────────────────────────────────────────────────────

def test_full_abs_position_invariant():
    """
    RoPE rotation for absolute position 3 is the same whether it is:
    - at local index 3 in a 5-token sequence
    - at local index 3 in a 20-token sequence
    
    This is the foundation of absolute position encoding.
    """
    head_dim = 16
    base = 10000.0
    theta_all = _precompute_theta(20, head_dim, base)
    
    base_vec = np.random.randn(head_dim).astype(np.float64)
    
    x_short = _make_x(base_vec, 5)   # [1, 5, 1, 16
    x_long  = _make_x(base_vec, 20)  # [1, 20, 1, 16]
    
    xq_short = apply_rope_with_positions(x_short, theta_all, np.arange(5))
    xq_long  = apply_rope_with_positions(x_long, theta_all, np.arange(20))
    
    # Position 3 must be identical in both sequences
    np.testing.assert_allclose(
        xq_short[0, 3, 0], xq_long[0, 3, 0], rtol=1e-10
    )


def test_cache_step_q_matches_full():
    """
    Q arriving at cache_idx=3 (absolute position 3) in cache mode
    must equal Q[3] from full mode.
    """
    head_dim = 8
    theta_all = _precompute_theta(5, head_dim)
    
    base_q = np.random.randn(head_dim).astype(np.float64)
    base_k = np.random.randn(head_dim).astype(np.float64)
    
    # ── Full mode ──
    xq_full = _make_x(base_q, 5)
    xk_full = _make_x(base_k, 5)
    xq_f = apply_rope_with_positions(xq_full, theta_all, np.arange(5))
    xk_f = apply_rope_with_positions(xk_full, theta_all, np.arange(5))
    
    # ── Cache mode: Q arrives at absolute pos 3 ──
    q_cache = _make_x(base_q, 1)  # [1, 1, 1, 8]
    abs_q_pos = np.array([3])
    xq_c = apply_rope_with_positions(q_cache, theta_all, abs_q_pos)  # [1, 1, 1, 8

    # k_cache: K at absolute positions 1, 2 (pre-cached)
    k_cache = _make_x(base_k, 2)  # [1, 2, 1, 8]  → K at absolute positions 1, 2
    k_cache_rope = apply_rope_with_positions(
        k_cache, theta_all, np.array([1, 2])  # absolute positions 1, 2
    )  # [1, 2, 1, 8]

    # Q(3) rotated in cache mode should equal Q(3) from full mode
    np.testing.assert_allclose(xq_c[0, 0, 0, 0], xq_f[0, 3, 0, 0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(xq_c[0, 0, 0, 1], xq_f[0, 3, 0, 1], rtol=1e-10, atol=1e-10)


def test_cache_attn_matches_full_attn():
    """
    RoPE KV cache invariant:
    
    Q arriving at stage 3 (absolute position 3) with K at absolute positions
    1, 2 pre-cached must produce attention scores equal to the full-sequence
    attention at position 3 attending to positions 1, 2.
    """
    head_dim = 8
    theta_all = _precompute_theta(5, head_dim)
    
    base_q = np.random.randn(head_dim).astype(np.float64)
    base_k = np.random.randn(head_dim).astype(np.float64)
    
    # ── Full mode ──
    xq_f = apply_rope_with_positions(
        _make_x(base_q, 5), theta_all, np.arange(5)
    )
    xk_f = apply_rope_with_positions(
        _make_x(base_k, 5), theta_all, np.arange(5)
    )
    full_scores = _compute_attn_scores(xq_f, xk_f)  # [1, 5, 5]

    # ── Cache mode ──
    # Q arrives at absolute position 3 (cache_idx=3)
    q_cache = apply_rope_with_positions(
        _make_x(base_q, 1), theta_all, np.array([3])
    )
    
    # K at absolute positions 1, 2 pre-cached
    k_cache = apply_rope_with_positions(
        _make_x(base_k, 2), theta_all, np.array([1, 2])
    )
    
    cache_scores = _compute_attn_scores(q_cache, k_cache)  # [1, 1, 2]  (or similar, depends on shapes)
    
    np.testing.assert_allclose(
        cache_scores[0, 0, 0], full_scores[0, 3, 1], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        cache_scores[0, 0, 1], full_scores[0, 3, 2], rtol=1e-10, atol=1e-10
    )


def test_all_qk_offsets():
    """
    For arbitrary Q and K, every (q_pos, k_pos) pair must match between
    cache and full mode.
    """
    np.random.seed(99)
    head_dim = 12
    seq_len = 4
    theta_all = _precompute_theta(seq_len, head_dim)
    
    base_q = np.random.randn(head_dim).astype(np.float64)
    base_k = np.random.randn(head_dim).astype(np.float64)
    
    # Full mode
    xq_f = apply_rope_with_positions(_make_x(base_q, seq_len), theta_all, np.arange(seq_len))
    xk_f = apply_rope_with_positions(_make_x(base_k, seq_len), theta_all, np.arange(seq_len))
    full_scores = _compute_attn_scores(xq_f, xk_f)  # [1, 4, 4]

    # For each possible arrival position, run cache mode
    for q_abs in range(seq_len):
        q_cache = apply_rope_with_positions(
            _make_x(base_q, 1), theta_all, np.array([q_abs])
        )
        for k_abs in range(q_abs + 1):
            k_cache = apply_rope_with_positions(
                _make_x(base_k, 1), theta_all, np.array([k_abs])
            )
            cache_scores = _compute_attn_scores(q_cache, k_cache)
            
            np.testing.assert_allclose(
                cache_scores[0, 0, 0],
                full_scores[0, q_abs, k_abs],
                rtol=1e-10, atol=1e-10,
            )
