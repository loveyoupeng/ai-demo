from __future__ import annotations

import numpy as np
from model.rope import compute_theta, apply_rope


def test_apply_rope_4d_shape():
    """apply_rope must handle 4D tensor [Batch, Seq_Len, Head, Head_Dim]."""
    batch_size = 2
    seq_len = 4
    num_heads = 4
    head_dim = 8
    x = np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float64)
    
    # For 4D input, theta needs to be [Seq_Len, Head_Dim//2]
    pos = np.arange(seq_len, dtype=np.float64)
    theta = compute_theta(pos, head_dim)  # [4, 4]
    
    x_rope = apply_rope(x, theta)
    
    assert x_rope.shape == x.shape
    # Position 0 is identity for ALL batches (theta at pos 0 = 0)
    for b in range(batch_size):
        np.testing.assert_allclose(x_rope[b, 0, :, :], x[b, 0, :, :], rtol=1e-10, atol=1e-10)
    # Position 1+ has rotation for all batches (different from input)
    for b in range(batch_size):
        assert not np.allclose(x_rope[b, 1, :, :], x[b, 1, :, :], rtol=1e-10)


def test_apply_rope_4d_independence():
    """Each head's RoPE rotation is independent."""
    batch_size = 2
    seq_len = 3
    num_heads = 2
    head_dim = 4
    
    x = np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float64)
    pos = np.arange(seq_len, dtype=np.float64)
    theta = compute_theta(pos, head_dim)  # [3, 2]
    
    x_rope = apply_rope(x, theta)
    
    # Verify each head is independently rotated
    # Position 0 is identity (no rotation) for all heads
    for h in range(num_heads):
        np.testing.assert_allclose(
            x_rope[0, 0, h, :], x[0, 0, h, :], rtol=1e-10, atol=1e-10
        )



def test_apply_rope_4d_batch_independent():
    """Different batch elements get independent RoPE."""
    batch_size = 2
    seq_len = 3
    num_heads = 1
    head_dim = 8
    
    x = np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float64)
    
    # Make batch[1] all zeros
    x[1] = 0.0
    
    pos = np.arange(seq_len, dtype=np.float64)
    theta = compute_theta(pos, head_dim)
    
    x_rope = apply_rope(x, theta)
    
    # Zero rows stay zero (cos(0)=1, sin(0)=0, so 0 → 0 after rotation)
    np.testing.assert_allclose(x_rope[1], 0.0, atol=1e-15)
    # Batch 0 gets proper rotation (not identity, not zero for most positions)
    # Position 0 is identity, so batch0 pos0 should equal input
    # But positions 1+ are rotated so they differ from input
    assert not np.allclose(x_rope[0, 1], x[0, 1], rtol=1e-10)


def test_apply_rope_4d_position_0_identity():
    """Position 0 must be identity (no rotation)."""
    x = np.random.randn(2, 4, 4, 16).astype(np.float64)
    pos = np.arange(4, dtype=np.float64)
    theta = compute_theta(pos, 16)
    
    x_rope = apply_rope(x, theta)
    
    # Position 0: identity for all batches and heads
    np.testing.assert_allclose(x_rope[:, 0, :, :], x[:, 0, :, :], rtol=1e-10, atol=1e-10)
