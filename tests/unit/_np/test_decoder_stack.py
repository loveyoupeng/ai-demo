"""B6.1: DecoderStack — stack of TransformerBlocks.

Forward: X → block_0 → block_1 → ... → block_{n-1}
"""

import numpy as np

from impl._np.modules import DecoderStack


class TestDecoderStackForward:
    """Test DecoderStack forward pass."""

    def test_output_shape(self):
        """Input [B, S, D] → output [B, S, D]."""
        x = np.random.default_rng(0).random((2, 4, 16)).astype(np.float32)

        stack = DecoderStack(n_layers=2, embed_dim=16, n_heads=4, n_experts=4, ff_dim=32, k=2, seed=0)
        out = stack.forward(x)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_gradient_chaining(self):
        """Gradients flow through all stacked layers.

        Changing any block's parameters should change the output.
        """
        x = np.random.default_rng(10).random((1, 3, 8)).astype(np.float32)

        stack = DecoderStack(n_layers=2, embed_dim=8, n_heads=2, n_experts=2, ff_dim=16, k=1, seed=0)

        baseline = stack.forward(x.copy())

        # Perturb first block's ln1_gamma
        stack.blocks[0].ln1_gamma = stack.blocks[0].ln1_gamma * 2.0
        out_perturbed = stack.forward(x.copy())

        assert not np.allclose(baseline, out_perturbed, atol=1e-3), "Gradients should flow through all blocks"
        assert np.all(np.isfinite(out_perturbed)), "Output should be finite"

    def test_single_layer(self):
        """Works with n_layers=1."""
        x = np.random.default_rng(20).random((1, 3, 8)).astype(np.float32)

        stack = DecoderStack(n_layers=1, embed_dim=8, n_heads=2, n_experts=2, ff_dim=16, k=1, seed=42)
        out = stack.forward(x)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
        assert not np.allclose(out, x, atol=1e-2), "Single-layer stack should produce non-identity output"
        assert np.all(np.isfinite(out)), "Output should be finite"
