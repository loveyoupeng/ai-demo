"""MHA naming consistency tests for Triton.

Verify that Triton MHA uses the same attribute naming as Torch MHA,
which uses nn.Linear wrappers for Wq, Wk, Wv, Wo.
"""
from __future__ import annotations

import pytest

from impl._torch.layers import TorchModel


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
            vocab_size=16, embed_dim=8, n_layers=1, n_heads=2,
            n_experts=2, ff_dim=16, k=1,
        )

        # All mha params should be .weight or .bias (nn.Linear style)
        mha_params = [
            k for k, _ in torch_model.named_parameters()
            if "mha" in k and "router" not in k
        ]

        for k in mha_params:
            # Keys should be .weight and .bias suffixes
            assert k.endswith((".weight", ".bias")), (
                f"Torch MHA key '{k}' should end with .weight or .bias "
                f"(nn.Linear style)"
            )

        # Check specific keys exist
        expected_keys = [
            "stack.layers.0.mha.Wq.weight",
            "stack.layers.0.mha.Wq.bias",
            "stack.layers.0.mha.Wk.weight",
            "stack.layers.0.mha.Wk.bias",
            "stack.layers.0.mha.Wv.weight",
            "stack.layers.0.mha.Wv.bias",
            "stack.layers.0.mha.Wo.weight",
            "stack.layers.0.mha.Wo.bias",
        ]
        for k in expected_keys:
            assert k in dict(torch_model.named_parameters()), (
                f"TorchModel should have parameter key '{k}'"
            )

    def test_triton_mha_uses_linear_wrappers(self) -> None:
        """Triton MHA should also use Linear wrappers for Wq/Wk/Wv/Wo."""
        from impl._triton.model import TritonModel

        triton_model = TritonModel(
            vocab_size=16, embed_dim=8, n_layers=1, n_heads=2,
            n_experts=2, ff_dim=16, k=1,
        )
        triton_params = dict(triton_model.named_parameters())

        # All mha params should be .weight or .bias (Linear style)
        mha_params = [
            k for k, _ in triton_params.items()
            if "mha" in k and "router" not in k
        ]

        for k in mha_params:
            # Keys should be .weight and .bias suffixes
            assert k.endswith((".weight", ".bias")), (
                f"Triton MHA key '{k}' should end with .weight or .bias "
                f"(nn.Linear style), not raw tensor"
            )

        # Check specific keys exist — should match Torch naming exactly
        expected_keys = [
            "stack.layers.0.mha.Wq.weight",
            "stack.layers.0.mha.Wq.bias",
            "stack.layers.0.mha.Wk.weight",
            "stack.layers.0.mha.Wk.bias",
            "stack.layers.0.mha.Wv.weight",
            "stack.layers.0.mha.Wv.bias",
            "stack.layers.0.mha.Wo.weight",
            "stack.layers.0.mha.Wo.bias",
        ]
        for k in expected_keys:
            assert k in triton_params, (
                f"TritonModel should have parameter key '{k}' "
                f"(matches TorchModel naming)"
            )
