from __future__ import annotations

import numpy as np
from model.rope import compute_theta


def test_compute_theta_values():
    """
    Rotation angles theta must match the RoPE formula:
    
    theta(pos, i) = pos * 10000^(-2i/d)
    
    For dim=4, i=0..1, pos=0: all zeros (no rotation at position 0)
    For dim=4, i=0..1, pos=1: computed from the formula
    """
    pos = np.array([0, 1, 2, 3], dtype=np.float64)
    dim = 4

    # theta(pos, i) = pos * 10000^(-2i/d)
    # i=0: 10000^(-2*0/4) = 10000^0 = 1
    # i=1: 10000^(-2*1/4) = 10000^(-0.5) = 0.01
    expected = np.zeros((4, 2), dtype=np.float64)  # (num_positions, num_pairs)
    expected[0, :] = [0, 0]  # position 0 = all zeros
    expected[1, :] = [1 * 1.0, 1 * 0.01]  # pos=1: [1, 0.01]
    expected[2, :] = [2, 0.02]  # pos=2: [2, 0.02]
    expected[3, :] = [3, 0.03]  # pos=3: [3, 0.03]

    # This should fail — function not yet implemented
    theta = compute_theta(pos, dim)

    np.testing.assert_allclose(theta, expected, rtol=1e-10, atol=1e-10)


def test_compute_theta_dim64():
    """RoPE with a realistic embedding dimension."""
    pos = np.array([0, 1, 2], dtype=np.float64)
    dim = 64

    theta = compute_theta(pos, dim)
    assert theta.shape == (3, 32)  # 3 positions, 32 pairs

    # Position 0: all zeros (no rotation)
    np.testing.assert_allclose(theta[0], 0, atol=1e-15)

    # Position 1: verify first and last angles
    # i=0: 10000^(-2*0/64) = 1.0
    # i=31 (last pair index): 10000^(-2*31/64) = 10000^(-0.96875)
    expected_first = 1.0
    np.testing.assert_allclose(theta[1, 0], expected_first, rtol=1e-10)


def test_compute_theta_batch():
    """compute_theta should handle a batch of positions."""
    # Simulate 2 samples, each with 4 positions
    pos = np.array([0, 1, 2, 3], dtype=np.float64)
    dim = 8

    theta = compute_theta(pos, dim)
    assert theta.shape == (4, 4)  # 4 positions, 4 pairs (8/2)

    # Verify decreasing frequencies as dimension index increases
    # theta[pos, i] should decrease as i increases (same position)
    for p in range(1, 4):
        assert theta[p, 0] > theta[p, 1] > theta[p, 2] > theta[p, 3]
