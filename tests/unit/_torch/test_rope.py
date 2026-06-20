"""Tests for PyTorch RoPE (Rotary Position Embedding) — C2.1.

TDD: Write test → all fail → implement → all pass → ruff + pyright → commit
"""

import torch


class TestRoPEForward:
    """Tests for the RoPE nn.Module forward pass."""

    def test_output_shape(self) -> None:
        """RoPE preserves shape: (B, S, H, D) → (B, S, H, D)."""
        from impl._torch.layers import RoPE

        batch, seq_len, n_heads, head_dim = 2, 8, 4, 16
        rope = RoPE()
        x = torch.randn(batch, seq_len, n_heads, head_dim, dtype=torch.float64)
        positions = torch.arange(seq_len)
        output = rope(x, positions)

        assert output.shape == x.shape
        assert output.dtype == x.dtype
        assert torch.all(torch.isfinite(output))

    def test_rotates_by_position(self) -> None:
        """Position 0 and position 1 produce different outputs."""
        from impl._torch.layers import RoPE

        batch, seq_len, n_heads, head_dim = 1, 4, 2, 8
        rope = RoPE()

        x = torch.randn(batch, seq_len, n_heads, head_dim, dtype=torch.float64)

        # All zeros position
        pos_zero = torch.zeros(seq_len, dtype=torch.long)
        out_zero = rope(x, pos_zero)

        # Incrementing positions
        pos_inc = torch.arange(seq_len, dtype=torch.long)
        out_inc = rope(x, pos_inc)

        # Positions 0 should be the same in both
        assert torch.allclose(out_zero[:, 0], out_inc[:, 0], atol=1e-10)
        # Position 1 should differ (position 1 rotates, position 0 doesn't)
        assert not torch.allclose(out_zero[:, 1], out_inc[:, 1], atol=1e-10)

    def test_partial_rope(self) -> None:
        """Only the first rope_dim dimensions are rotated; rest pass through unchanged."""
        from impl._torch.layers import RoPE

        batch, seq_len, n_heads, head_dim = 2, 4, 2, 8
        rope = RoPE()

        x = torch.randn(batch, seq_len, n_heads, head_dim, dtype=torch.float64)
        positions = torch.arange(seq_len, dtype=torch.long)

        # Partial: first 4 dims rotated, last 4 pass through
        out_partial = rope(x, positions, rope_dim=4)

        # Unrotated dims (4-8) should be identical to input (pass-through)
        assert torch.allclose(out_partial[:, :, :, 4:], x[:, :, :, 4:], atol=1e-10)
        # First 4 dims should differ (rotated)
        assert not torch.allclose(out_partial[:, :, :, :4], x[:, :, :, :4], atol=1e-10)

    def test_deterministic(self) -> None:
        """Same input + same position → same output."""
        from impl._torch.layers import RoPE

        batch, seq_len, n_heads, head_dim = 2, 4, 2, 8
        rope = RoPE()
        x = torch.randn(batch, seq_len, n_heads, head_dim, dtype=torch.float64)
        positions = torch.arange(seq_len, dtype=torch.long)

        out1 = rope(x, positions)
        out2 = rope(x, positions)

        assert torch.allclose(out1, out2, atol=1e-12)
