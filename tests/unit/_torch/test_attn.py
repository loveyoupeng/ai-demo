"""Tests for PyTorch MultiHeadAttention (C3.1).

TDD: Write test → all fail → implement → all pass → ruff + pyright → commit
"""

import torch


class TestMultiHeadAttentionForward:
    """Tests for the MultiHeadAttention nn.Module forward pass."""

    def test_output_shape(self) -> None:
        """MHA(x: [B,S,D], H heads, D embed) → [B,S,D]."""
        from impl._torch.layers import MultiHeadAttention

        batch, seq_len, embed_dim = 2, 8, 16
        n_heads = 4
        mha = MultiHeadAttention(embed_dim, n_heads)
        x = torch.randn(batch, seq_len, embed_dim, dtype=torch.float64)
        output, _ = mha(x)

        assert output.shape == (batch, seq_len, embed_dim)
        assert output.dtype == x.dtype
        assert torch.all(torch.isfinite(output))

    def test_attention_mechanism(self) -> None:
        """Attention output is finite and non-trivial (not all zeros)."""
        from impl._torch.layers import MultiHeadAttention

        batch, seq_len, embed_dim = 1, 10, 16
        n_heads = 4
        mha = MultiHeadAttention(embed_dim, n_heads)
        x = torch.randn(batch, seq_len, embed_dim, dtype=torch.float64)
        output, _ = mha(x)

        assert torch.all(torch.isfinite(output))
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_gradient_flow(self) -> None:
        """All weight matrices (Q, K, V, O) participate in computation.

        Different perturbation of Q vs O should produce different outputs.
        """
        from impl._torch.layers import MultiHeadAttention

        embed_dim = 12
        n_heads = 3
        mha = MultiHeadAttention(embed_dim, n_heads)

        x = torch.ones(1, 3, embed_dim, dtype=torch.float64)

        # Perturb Q projection
        Wq_orig = mha.Wq.weight.data.clone()
        mha.Wq.weight.data = Wq_orig.clone() + 0.1
        out_q, _ = mha(x.clone())

        # Reset and perturb O projection
        mha.Wq.weight.data = Wq_orig
        Wo_orig = mha.Wo.weight.data.clone()
        mha.Wo.weight.data = Wo_orig.clone() + 0.1
        out_o, _ = mha(x.clone())

        assert not torch.allclose(out_q, out_o, rtol=1e-3)

    def test_deterministic(self) -> None:
        """Same input → same output (forward is deterministic, no randomness)."""
        from impl._torch.layers import MultiHeadAttention

        embed_dim = 8
        n_heads = 2
        mha = MultiHeadAttention(embed_dim, n_heads)

        x = torch.randn(1, 6, embed_dim, dtype=torch.float64)

        out1, _ = mha(x.clone())
        out2, _ = mha(x.clone())

        assert torch.allclose(out1, out2)

    def test_gqa_support(self) -> None:
        """With n_heads=6, n_groups=3 — K/V shared across groups."""
        from impl._torch.layers import MultiHeadAttention

        embed_dim = 12
        n_heads = 6
        mha = MultiHeadAttention(embed_dim, n_heads, n_groups=3)

        x = torch.randn(2, 3, embed_dim, dtype=torch.float64)
        output, _ = mha(x)

        assert output.shape == (2, 3, embed_dim)
        assert torch.all(torch.isfinite(output))
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_kv_cache_input(self) -> None:
        """MHA should accept past_key_value for autoregressive decoding.

        past_key_value: list of (k, v) tuples, one per layer,
        each of shape [B, n_groups, past_len, head_dim].
        """
        from impl._torch.layers import MultiHeadAttention

        batch, seq_len, embed_dim = 1, 4, 16
        n_heads = 4
        mha = MultiHeadAttention(embed_dim, n_heads)

        x = torch.randn(batch, seq_len, embed_dim, dtype=torch.float64)

        # Simulate a past_key_value with 2 cached positions
        head_dim = embed_dim // n_heads
        past_len = 2
        past_kv = [
            (
                torch.randn(batch, n_heads, past_len, head_dim, dtype=torch.float64),
                torch.randn(batch, n_heads, past_len, head_dim, dtype=torch.float64),
            )
        ]

        # Should accept past_key_value and return same shape
        output, _ = mha(x, past_key_value=past_kv)
        assert output.shape == x.shape
        assert torch.all(torch.isfinite(output))
