"""Phase 3: Transformer without additive position embedding.

Positional information is now encoded via RoPE in MHA Q/K.
The Transformer should NOT have any additive positional embedding anymore.
"""
from __future__ import annotations

import numpy as np
from model.transformer import Transformer


class TestTransformerNoPosPE:
    """Tests confirming Transformer does NOT apply additive positional embedding."""

    def setup_method(self):
        np.random.seed(42)
        self.model = Transformer(
            vocab_size=50, embed_dim=32, num_layers=1,
            num_heads=2, num_experts=2, max_seq_len=32,
        )

    def test_forward_no_pos_pe(self):
        """Transformer forward without PE should run cleanly."""
        input_ids = np.array([[0, 1, 2, 3, 4]], dtype=np.int32)
        logits, cache = self.model.forward(input_ids)

        assert logits.shape == (1, 5, 50)
        assert not np.any(np.isnan(logits))
        # Verify PE is not in params
        params = self.model.get_params()
        for k in params:
            assert "pos" not in k.lower(), f"Unexpected pos param: {k}"

    def test_backward_no_pos_pe(self):
        """Transformer backward without PE should work."""
        input_ids = np.array([[0, 1, 2]], dtype=np.int32)
        logits, cache = self.model.forward(input_ids)
        grad_logits = np.ones_like(logits)

        grads = self.model.backward(grad_logits, cache)

        assert "token_embedding.weights" in grads
        assert "lm_head" in grads
        assert "blocks.0.mha.W_q" in grads
        assert np.all(np.isfinite(grads["token_embedding.weights"]))

    def test_no_positional_params(self):
        """Transformer params should never contain pos-related keys."""
        params = self.model.get_params()
        for k in params:
            assert "pos" not in k.lower(), f"Unexpected pos param: {k}"

    def test_repeated_tokens_no_pe_bias(self):
        """Without additive PE, repeated tokens at different positions should
        still have consistent behavior."""
        input_ids = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        logits, _ = self.model.forward(input_ids, mask=None)
        assert logits.shape == (1, 5, 50)

    def test_no_pe_attribute(self):
        """Transformer should NOT have a pos_embedding attribute."""
        assert not hasattr(self.model, 'pos_embedding')
