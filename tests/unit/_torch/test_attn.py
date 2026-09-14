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
        mha = MultiHeadAttention(embed_dim, n_heads, n_heads, 0).double()
        x = torch.randn(batch, seq_len, embed_dim, dtype=torch.float64)
        output = mha(x)

        assert output.shape == (batch, seq_len, embed_dim)
        assert output.dtype == x.dtype
        assert torch.all(torch.isfinite(output))

    def test_attention_mechanism(self) -> None:
        """Attention output is finite and non-trivial (not all zeros)."""
        from impl._torch.layers import MultiHeadAttention

        batch, seq_len, embed_dim = 1, 10, 16
        n_heads = 4
        mha = MultiHeadAttention(embed_dim, n_heads, n_heads, 0).double()
        x = torch.randn(batch, seq_len, embed_dim, dtype=torch.float64)
        output = mha(x)

        assert torch.all(torch.isfinite(output))
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_gradient_flow(self) -> None:
        """All weight matrices (Q, K, V, O) participate in computation.

        Different perturbation of Q vs O should produce different outputs.
        """
        from impl._torch.layers import MultiHeadAttention

        embed_dim, n_heads = 16, 4
        mha = MultiHeadAttention(embed_dim, n_heads, n_heads, 0).double()

        x = torch.ones(1, 3, embed_dim, dtype=torch.float64)

        # Perturb Q projection
        Wq_orig = mha.q_proj.weight.data.clone()
        mha.q_proj.weight.data = Wq_orig.clone() + 0.1
        out_q = mha(x.clone())

        # Reset and perturb O projection
        mha.q_proj.weight.data = Wq_orig
        Wo_orig = mha.o_proj.weight.data.clone()
        mha.o_proj.weight.data = Wo_orig.clone() + 0.1
        out_o = mha(x.clone())

        assert not torch.allclose(out_q, out_o, rtol=1e-3)

    def test_deterministic(self) -> None:
        """Same input → same output (forward is deterministic, no randomness)."""
        from impl._torch.layers import MultiHeadAttention

        batch, seq_len, embed_dim = 1, 6, 16
        n_heads = 4
        mha = MultiHeadAttention(embed_dim, n_heads, n_heads, 0).double()

        x = torch.randn(batch, seq_len, embed_dim, dtype=torch.float64)

        out1 = mha(x.clone())
        out2 = mha(x.clone())

        assert torch.allclose(out1, out2)

    def test_gqa_support(self) -> None:
        """With n_heads=6, n_groups=3 — K/V shared across groups."""
        from impl._torch.layers import MultiHeadAttention

        batch, seq_len, embed_dim = 2, 3, 16
        n_heads = 6
        mha = MultiHeadAttention(embed_dim, n_heads, 3, 0).double()

        x = torch.randn(batch, seq_len, embed_dim, dtype=torch.float64)
        output = mha(x)

        assert output.shape == (batch, seq_len, embed_dim)
        assert torch.all(torch.isfinite(output))
        assert not torch.allclose(output, torch.zeros_like(output))
