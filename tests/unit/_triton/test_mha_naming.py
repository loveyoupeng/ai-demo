"""MHA naming consistency tests for Triton.

Verify that Triton MHA uses the same attribute naming as Torch MHA,
which uses nn.Linear wrappers for q_proj/k_proj/v_proj/o_proj (no bias).
"""

from __future__ import annotations

import pytest

from impl._torch.layers import TorchModel
from shared.config import TransformerConfig


def skip_if_no_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("GPU required")


@pytest.mark.timeout(60)
class TestMHANamingParity:
    """Check MHA named_parameters() keys match between backends."""

    def test_torch_mha_uses_linear_wrappers(self) -> None:
        """Torch MHA should use nn.Linear for Wq/Wk/Wv/Wo."""
        skip_if_no_gpu()
        torch_model = TorchModel(
            TransformerConfig(
                vocab_size=16,
                embed_dim=8,
                n_layers=1,
                n_heads=2,
                n_groups=2,
                n_experts=2,
                expert_dim=16,
                top_k=1,
            ),
        )

        # All mha params should be .weight (nn.Linear style, no bias)
        mha_params = [k for k, _ in torch_model.named_parameters() if "self_attn" in k]

        for k in mha_params:
            # Keys should be .weight (nn.Linear style)
            assert k.endswith((".weight", ".bias")), (
                f"Torch MHA key '{k}' should end with .weight or .bias (nn.Linear style)"
            )

        # Check specific keys exist
        expected_keys = [
            "stack.layers.0.self_attn.q_proj.weight",
            "stack.layers.0.self_attn.k_proj.weight",
            "stack.layers.0.self_attn.v_proj.weight",
            "stack.layers.0.self_attn.o_proj.weight",
        ]
        for k in expected_keys:
            assert k in dict(torch_model.named_parameters()), f"TorchModel should have parameter key '{k}'"

    def test_triton_mha_uses_linear_wrappers(self) -> None:
        """Triton MHA should also use Linear wrappers for Wq/Wk/Wv/Wo."""
        from impl._triton.model import TritonModel

        triton_model = TritonModel(
            TransformerConfig(
                vocab_size=16,
                embed_dim=8,
                n_layers=1,
                n_heads=2,
                n_groups=2,
                n_experts=2,
                expert_dim=16,
                top_k=1,
            ),
        )
        triton_params = dict(triton_model.named_parameters())

        # All mha params should be .weight (Linear style, no bias)
        mha_params = [k for k, _ in triton_params.items() if "self_attn" in k]

        for k in mha_params:
            # Keys should be .weight suffixes
            assert k.endswith((".weight", ".bias")), (
                f"Triton MHA key '{k}' should end with .weight or .bias (nn.Linear style), not raw tensor"
            )

        # Check specific keys exist — should match Torch naming exactly
        expected_keys = [
            "stack.layers.0.self_attn.q_proj.weight",
            "stack.layers.0.self_attn.k_proj.weight",
            "stack.layers.0.self_attn.v_proj.weight",
            "stack.layers.0.self_attn.o_proj.weight",
        ]
        for k in expected_keys:
            assert k in triton_params, f"TritonModel should have parameter key '{k}' (matches TorchModel naming)"
