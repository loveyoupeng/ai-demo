from __future__ import annotations

import numpy as np
from model.rope import apply_rope


def _attention_scores(xq_rope, xk_rope):
    """Compute attention scores for [B, L, H, D] tensors.
    
    scores[b, m, n] = sum over h, d of xq_rope[b, m, h, d] * xxk_rope[b, n, h, d]
    
    Result shape: [B, L, L] where L = seq_len
    """
    # [B, L, H, D] → flatten H, D → [B, L, H*D]
    xq = xq_rope.reshape(xq_rope.shape[0], xq_rope.shape[1], -1)  # [B, L, H*D]
    xk = xk_rope.reshape(xk_rope.shape[0], xk_rope.shape[1], -1)  # [B, L, H*D]
    # scores[b, m, n] = xq[b, m, :] @ xk[b, n, :] → [B, L, L]
    return xq @ xk.transpose(0, 2, 1)  # [B, L, H*D] @ [B, H*D, L] → [B, L, L]


def test_relative_position_property():
    """
    RoPE's defining property: dot(q_rope(m), k_rope(n)) depends ONLY on (m - n).
    
    dot(q_rope(m), k_rope(n)) = dot(q(0), R_{-(m-n)} * k(0))
    
    Therefore, for the same relative offset (m-n), the dot product is identical.
    """
    np.random.seed(42)
    head_dim = 16
    seq_len = 8

    pair_count = head_dim // 2
    power = np.arange(pair_count, dtype=np.float64) * -2.0 / head_dim
    div_term = 10000.0 ** power

    # theta: [seq_len, pair_count] = [8, 8]
    theta = np.arange(seq_len, dtype=np.float64).reshape(-1, 1) * div_term

    # Each position has the same base vector
    base_q = np.random.randn(head_dim).astype(np.float64)
    base_k = np.random.randn(head_dim).astype(np.float64)

    # Use [B=1, L, H=1, D] for MHA-style RoPE
    x_q = np.repeat(base_q[np.newaxis, np.newaxis, ...], seq_len, axis=1)  # [1, 8, 1, 16]
    x_k = np.repeat(base_k[np.newaxis, np.newaxis, ...], seq_len, axis=1)  # [1, 8, 1, 16]

    # Apply RoPE
    xq_rope = apply_rope(x_q, theta)  # [1, 8, 1, 16]
    xk_rope = apply_rope(x_k, theta)  # [1, 8, 1, 16]

    # Compute attention scores: scores[m, n] = xq_rope[m] · xk_rope[n]
    scores = _attention_scores(xq_rope, xk_rope)  # [1, 8, 8]

    # For relative offset +1:
    # score[0,1,0] == score[0,2,1] == score[0,3,2] == ... == score[0,7,6]
    # For relative offset -1:
    # score[0,0,1] == score[0,1,2] == score[0,2,3] == ... == score[0,6,7]

    for offset in list(range(1, seq_len)) + list(range(-1, 0)):
        values = []
        for i in range(seq_len):
            j = i + offset
            if 0 <= j < seq_len:
                values.append(scores[0, i, j])
        # All values for the same offset should be identical
        np.testing.assert_allclose(
            values[0], values, rtol=1e-10, atol=1e-10,
            err_msg=(
                f"Relative offset {offset}: score should be constant, "
                f"got {values[:3]}..."
            ),
        )



def test_relative_position_all_offsets():
    """Verify the relative position property for ALL offsets (including 0)."""
    np.random.seed(123)
    head_dim = 16
    seq_len = 6

    pair_count = head_dim // 2
    div_term = (
        10000.0
        ** (np.arange(pair_count, dtype=np.float64) * -2.0 / head_dim)
    )
    theta = np.arange(seq_len, dtype=np.float64).reshape(-1, 1) * div_term

    base_q = np.random.randn(head_dim).astype(np.float64)
    base_k = np.random.randn(head_dim).astype(np.float64)

    x_q = np.repeat(base_q[np.newaxis, np.newaxis, ...], seq_len, axis=1)
    x_k = np.repeat(base_k[np.newaxis, np.newaxis, ...], seq_len, axis=1)

    xq_rope = apply_rope(x_q, theta)
    xk_rope = apply_rope(x_k, theta)

    scores = _attention_scores(xq_rope, xk_rope)  # [1, 6, 6]

    # Check ALL offsets including 0 (diagonal)
    for offset in range(-5, 6):
        values = []
        for i in range(seq_len):
            j = i + offset
            if 0 <= j < seq_len:
                values.append(scores[0, i, j])
        if len(values) >= 2:
            np.testing.assert_allclose(
                values[0], values, rtol=1e-10, atol=1e-10,
                err_msg=f"Relative offset {offset} not constant: {values}",
            )
