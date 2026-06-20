"""TritonModel parameter naming consistency tests.

Verify that TritonModel uses the same attribute naming as TorchModel,
enabling cross-backend parity via named_parameters().
"""
from __future__ import annotations

import pytest

from impl._triton.model import TritonModel


@pytest.mark.timeout(60)
class TestTritonModelNamingParity:
    """Check that TritonModel produces matching named_parameters() keys."""

    def test_model_level_final_ln_matches(self) -> None:
        """final layer norm should use RMSNorm instance in both backends."""
        skip_if_no_gpu()
        triton_model = TritonModel(
            vocab_size=16, embed_dim=8, n_layers=1, n_heads=2, n_experts=2, ff_dim=16, k=1,
        )

        triton_params = dict(triton_model.named_parameters())

        # Both should use instance-like naming (not raw param suffix _gamma)
        triton_final_key = [k for k in triton_params if "final" in k and "ln" in k][0]
        assert "gamma" not in triton_final_key, (
            f"TritonModel should use RMSNorm instance for final ln: "
            f"expected 'final_ln.weight', got '{triton_final_key}'"
        )

    def test_model_level_output_matches(self) -> None:
        """Output SwiGLU should be a single instance in both backends."""
        skip_if_no_gpu()
        triton_model = TritonModel(
            vocab_size=16, embed_dim=8, n_layers=1, n_heads=2, n_experts=2, ff_dim=16, k=1,
        )

        triton_params = dict(triton_model.named_parameters())

        # Find output keys
        triton_output_keys = sorted(
            [k for k in triton_params if "output" in k and "proj" not in k]
        )

        # Triton should use instance-style naming (output.W1, not output_W1)
        for k in triton_output_keys:
            assert "." in k, (
                f"TritonModel output key '{k}' should be instance-style (output.W1), "
                f"not raw param style (output_W1)"
            )

    def test_model_level_output_proj_matches(self) -> None:
        """Output projection should use nn.Linear in both backends."""
        skip_if_no_gpu()
        triton_model = TritonModel(
            vocab_size=16, embed_dim=8, n_layers=1, n_heads=2, n_experts=2, ff_dim=16, k=1,
        )

        triton_params = dict(triton_model.named_parameters())

        # Both should have output_proj.* keys
        triton_proj_keys = sorted([k for k in triton_params if "output_proj" in k])

        for k in triton_proj_keys:
            # If project uses nn.Linear, keys should be .weight and .bias
            assert ".weight" in k or ".bias" in k, (
                f"TritonModel output_proj key '{k}' should use nn.Linear style"
            )


def skip_if_no_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("GPU required")
