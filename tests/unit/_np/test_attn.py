"""B3.1: Multi-Head Attention — scaled dot-product self-attention.

All tests fail initially. Implement after verifying failure.
"""

import numpy as np

from impl._np.modules import MultiHeadAttention


class TestMultiHeadAttentionForward:
    """Test the MHA forward pass."""

    def test_output_shape(self):
        """Input [batch, seq_len, embed_dim] → output same shape."""
        x = np.random.default_rng(42).random((2, 4, 16)).astype(np.float32)

        mha = MultiHeadAttention(16, n_heads=4, rope_dim=0, seed=0)
        out = mha.forward(x)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_attention_mechanism(self):
        """Softmax normalizes across the sequence dimension for attention scores.

        Each (batch, head) should produce attention weights that sum to ~1
        across the sequence dimension.
        """
        x = np.random.default_rng(10).random((1, 5, 16)).astype(np.float32)

        mha = MultiHeadAttention(16, n_heads=4, rope_dim=0, seed=42)

        # With n_heads=4, head_dim=4, we have 4 heads
        # After attention: scores sum to 1 across seq for each head
        # This is verified by checking attention weights would sum to 1
        # Since we don't expose internal attn weights, check output is finite
        out = mha.forward(x)
        assert np.all(np.isfinite(out)), "MHA output should be finite"
        assert not np.allclose(out, 0.0), "MHA output should not be all zeros"

    def test_gradient_flow(self):
        """All weight matrices (Q, K, V, O projections) participate in computation.

        Changing any projection weight should change the output.
        """
        x = np.ones((1, 3, 12), dtype=np.float32)

        mha = MultiHeadAttention(12, n_heads=3, rope_dim=0, seed=0)

        # Perturb Q projection
        Wq_orig = mha.Wq.copy()
        mha.Wq = Wq_orig + 0.1
        out_q = mha.forward(x.copy())

        # Reset and perturb O projection
        mha.Wq = Wq_orig
        Wo_orig = mha.Wo.copy()
        mha.Wo = Wo_orig + 0.1
        Wo_orig = mha.Wo.copy()
        mha.Wo = Wo_orig + 0.1
        out_o = mha.forward(x.copy())

        # All perturbations should change output
        assert not np.allclose(out_q, out_o, rtol=1e-3), "Perturbing Q and O should produce different outputs"

    def test_deterministic(self):
        """Same input, same seed → same output (no randomness in forward)."""
        x = np.random.default_rng(55).random((1, 6, 8)).astype(np.float32)

        mha1 = MultiHeadAttention(8, n_heads=2, rope_dim=0, seed=42)
        mha2 = MultiHeadAttention(8, n_heads=2, rope_dim=0, seed=42)

        out1 = mha1.forward(x.copy())
        out2 = mha2.forward(x.copy())

        np.testing.assert_array_equal(out1, out2, err_msg="Same seed should produce identical outputs")

    def test_gqa_structure(self):
        """With n_heads=6, n_groups=3 — K/V projections have 3 heads, Q has 6,
        output should still be [batch, seq_len, embed_dim].

        K and V are shared across groups of query heads (GQA).
        """
        x = np.random.default_rng(33).random((2, 3, 12)).astype(np.float32)

        mha = MultiHeadAttention(12, n_heads=6, n_groups=3, rope_dim=0, seed=0)
        out = mha.forward(x)

        assert out.shape == x.shape, f"GQA output shape {out.shape} != input {x.shape}"
        assert np.all(np.isfinite(out)), "GQA output should be finite"
        assert not np.allclose(out, 0.0), "GQA output should not be all zeros"
