"""TritonModel parameter naming consistency tests.

Verify that TritonModel uses the same attribute naming as TorchModel,
enabling cross-backend parity via named_parameters().
"""

from __future__ import annotations

import pytest

from impl._torch.layers import TorchModel
from impl._triton.model import TritonModel
from shared.config import TransformerConfig


def _cfg() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=16, embed_dim=8, n_layers=2, n_heads=2, n_groups=2, n_experts=2, expert_dim=16, top_k=2
    )


def _keys(model) -> set[str]:
    return {k for k, _ in model.named_parameters()}


@pytest.mark.timeout(60)
class TestTritonModelNamingParity:
    """Check that TritonModel produces matching named_parameters() keys."""

    def test_model_level_keys_match_torch(self) -> None:
        """TritonModel and TorchModel expose identical parameter key names."""
        triton_params = _keys(TritonModel(_cfg()))
        torch_params = _keys(TorchModel(_cfg()))

        assert triton_params == torch_params, (
            f"Key sets differ.\nonly in triton: {sorted(triton_params - torch_params)}\nonly in torch: {sorted(torch_params - triton_params)}"
        )

    def test_model_level_final_ln_matches(self) -> None:
        """final layernorm uses nn.RMSNorm instance naming (final_norm.weight)."""
        triton_params = _keys(TritonModel(_cfg()))

        final_key = [k for k in triton_params if "final_norm" in k]
        assert final_key == ["final_norm.weight"], f"Expected final_norm.weight, got {final_key}"

    def test_model_level_lm_head_matches(self) -> None:
        """lm head uses nn.Linear naming (lm_head.weight, no bias)."""
        triton_params = _keys(TritonModel(_cfg()))

        head_keys = sorted(k for k in triton_params if "lm_head" in k)
        assert head_keys == ["lm_head.weight"], f"Expected lm_head.weight only, got {head_keys}"
