from __future__ import annotations

import numpy as np
from model.rope import compute_theta, apply_rope


def test_apply_rope_2d_basic():
    """
    Apply RoPE to [Batch, Seq_Len, Embed_Dim].
    
    For each sequence position l at absolute position m:
    [x'(2i), x'(2i+1)] = [cos  -sin] [x(2i)  ]
                          [sin   cos] [x(2i+1)]
    
    Position 0 should have NO rotation (identity: x' = x).
    """
    batch_size = 2
    seq_len = 4
    embed_dim = 8
    np.random.seed(42)
    x = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float64)

    # Positions: 0, 1, 2, 3
    pos = np.arange(seq_len, dtype=np.float64)
    theta = compute_theta(pos, embed_dim)

    x_rope = apply_rope(x, theta)

    # Position 0: no rotation → x_rope[0, 0, :] == x[0, 0, :]
    np.testing.assert_allclose(x_rope[0, 0, :], x[0, 0, :], rtol=1e-10, atol=1e-10)

    # Other positions: rotation changes values (not identity for arbitrary input)
    # At pos=1, theta[1, 0] = 1.0 → cos(1) ≈ 0.54, sin(1) ≈ 0.84 — significant rotation
    assert not np.allclose(x_rope[0, 1, :], x[0, 1, :], rtol=1e-5)


def test_apply_rope_2d_output_shape():
    """Output shape must equal input shape."""
    batch_size = 3
    seq_len = 5
    embed_dim = 16
    x = np.random.randn(batch_size, seq_len, embed_dim)
    pos = np.arange(seq_len, dtype=np.float64)
    theta = compute_theta(pos, embed_dim)

    x_rope = apply_rope(x, theta)
    assert x_rope.shape == x.shape


def test_apply_rope_2d_unit_rotation():
    """
    When theta angles are chosen so that rotation is ~90 degrees,
    verify the rotation is applied correctly.
    
    At pos=0, theta = 0 → cos=1, sin=0 → identity.
    """
    x = np.array([[[0, 1]]], dtype=np.float64)  # [B=1, L=1, D=2], x = [0, 1]
    pos = np.array([0.0], dtype=np.float64)
    theta = compute_theta(pos, 2)  # [0, 0] for dim=2 (1 pair)

    x_rope = apply_rope(x, theta)

    # At theta=0: cos(0)=1, sin(0)=0
    # [x'(0), x'(1)] = [1*0 - 0*1, 0*0 + 1*1] = [0, 1] = x
    np.testing.assert_allclose(x_rope, x, rtol=1e-15, atol=1e-15)
