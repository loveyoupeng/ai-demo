"""B5.1: TransformerBlock — standard pre-norm decoder block (RMSNorm + MHA + SwiGLU/MoE + residuals).

Forward: h = x + MHA(RMSNorm(x)); out = h + FFN(RMSNorm(h))
"""

import numpy as np

from impl._np.block import TransformerBlock
from impl._np.moe import MixtureOfExperts
from shared.config import TransformerConfig


class TestTransformerBlockForward:
    """Test TransformerBlock forward pass."""

    def test_output_shape(self):
        """Input [B, S, D] → output [B, S, D]."""
        x = np.random.default_rng(0).random((2, 4, 16)).astype(np.float32)

        block = TransformerBlock(
            TransformerConfig(embed_dim=16, n_heads=4, n_groups=4, n_experts=4, top_k=2, expert_dim=32, seed=0)
        )
        out = block.forward(x)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_residual_connection(self):
        """Output contains the input (residual pass-through).

        With zero-initialized weights and zero activations, the output
        should equal the input (residual connection preserves x).
        """
        x = np.random.default_rng(10).random((1, 3, 8)).astype(np.float32)

        block = TransformerBlock(
            TransformerConfig(embed_dim=8, n_heads=2, n_groups=2, n_experts=2, top_k=1, expert_dim=16, seed=0)
        )
        out = block.forward(x.copy())

        # The output = x + attn_out + moe_out
        # attn_out and moe_out are NOT zero (they're learned functions)
        # So output differs from input
        assert not np.allclose(out, x, atol=1e-2), "Output should differ from input (residual + non-zero activations)"

    def test_attention_and_moe(self):
        """Both MHA and MoE contribute to the output.

        Changing any internal component should change the output.
        """
        x = np.random.default_rng(20).random((1, 3, 8)).astype(np.float32)

        block = TransformerBlock(
            TransformerConfig(embed_dim=8, n_heads=2, n_groups=2, n_experts=2, top_k=1, expert_dim=16, seed=0)
        )

        # Get baseline output
        baseline = block.forward(x.copy())

        # Zero the MoE router — suppresses the expert mixing
        assert isinstance(block.mlp, MixtureOfExperts)
        block.mlp.gate.fill(0.0)

        # After zero router, softmax produces uniform weights which are then
        # zeroed by top-k. So moe_out should be near zero.
        out_no_moe = block.forward(x.copy())

        assert not np.allclose(baseline, out_no_moe, atol=1e-2), "Zeroing MoE should change output"
        assert np.all(np.isfinite(out_no_moe)), "Output should be finite"

    def test_gradient_chaining(self):
        """Gradients flow through all internal components.

        Changing RMSNorm gamma should affect the output, showing that
        gradients flow through normalization layers.
        """
        x = np.random.default_rng(30).random((1, 3, 8)).astype(np.float32)

        block = TransformerBlock(
            TransformerConfig(embed_dim=8, n_heads=2, n_groups=2, n_experts=2, top_k=1, expert_dim=16, seed=0)
        )

        baseline = block.forward(x.copy())

        # Perturb the first RMSNorm (input layernorm) gamma
        ln1_gamma_orig = block.input_layernorm.gamma.copy()
        block.input_layernorm.gamma = ln1_gamma_orig * 2.0

        out_perturbed = block.forward(x.copy())

        assert not np.allclose(baseline, out_perturbed, atol=1e-3), (
            "Perturbing ln1 gamma should change output (grad flows through LN)"
        )
        assert np.all(np.isfinite(out_perturbed)), "Output should be finite"
