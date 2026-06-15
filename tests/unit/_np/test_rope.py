"""B2.1: RoPE — Rotary Positional Embedding.

All tests fail initially. Implement after verifying failure.
"""

import numpy as np

from impl._np.modules import RoPE


class TestRoPEForward:
    """Test the RoPE forward pass."""

    def test_output_shape(self):
        """Output shape matches input — RoPE does not change tensor dimensions.

        Q: (batch=2, seq_len=4, n_heads=4, head_dim=8) → output same shape.
        """
        q = np.random.default_rng(42).random((2, 4, 4, 8)).astype(np.float32)
        pos = np.arange(4)  # positions 0,1,2,3 per sequence step

        rope = RoPE()
        out = rope.forward(q, pos)

        assert out.shape == q.shape, f"Shape mismatch: {out.shape} != {q.shape}"

    def test_different_positions(self):
        """Position 0 and position 1 produce different rotations for the same input.

        Same input values at different sequence positions → different outputs
        because RoPE rotates by position-dependent angles.
        """
        q = np.ones((2, 4, 4, 8), dtype=np.float32)
        pos = np.arange(4)  # [0, 1, 2, 3]

        rope = RoPE()
        out = rope.forward(q, pos)

        # Same input values at different positions → different outputs
        assert not np.allclose(out[0, 0, :, :], out[0, 1, :, :], rtol=1e-3), (
            "Different positions should produce different RoPE outputs"
        )

    def test_full_vs_partial(self):
        """rope_dim=0 means full rotation (all dims rotated).
        rope_dim=4 means first 4 dims rotated, last 4 dims unchanged.
        """
        q = np.random.default_rng(7).random((1, 2, 4, 8), dtype=np.float32)
        pos = np.arange(2)  # positions [0, 1]

        rope = RoPE()

        # Full RoPE: all head_dim=8 dims rotated
        out_full = rope.forward(q, pos, rope_dim=0)

        # Partial RoPE: only first 4 dims rotated
        out_partial = rope.forward(q, pos, rope_dim=4)

        # Partial should have dims 4:8 unchanged from input
        partial_unchanged = np.allclose(out_partial[0, :, :, 4:], q[0, :, :, 4:], rtol=1e-5)
        assert partial_unchanged, "Dims after rope_dim should be unchanged in partial RoPE"

        # Full and partial should differ (full rotates all 8 dims)
        assert not np.allclose(out_full[0, :, :, :], out_partial[0, :, :, :], rtol=1e-3), (
            "Full and partial RoPE should produce different results"
        )

    def test_no_gradient_leakage_qk(self):
        """RoPE is applied independently to Q and K — they don't affect each other.

        If we set K == Q, identical inputs should produce identical outputs
        since RoPE is a deterministic function of input values and position.
        """
        rng = np.random.default_rng(100)
        q = rng.random((1, 3, 2, 4), dtype=np.float32)
        k = q.copy()
        pos = np.arange(3)  # positions [0, 1, 2]

        rope = RoPE()
        out_q = rope.forward(q, pos)
        out_k = rope.forward(k, pos)

        # Same input → same output (deterministic function)
        assert np.allclose(out_q, out_k, rtol=1e-5), (
            "Identical Q and K should produce identical RoPE outputs"
        )
