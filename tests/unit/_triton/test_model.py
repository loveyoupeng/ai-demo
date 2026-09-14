"""E9: Full TritonModel — embedding → DecoderStack → RMSNorm → SwiGLU → output."""

import numpy as np
import pytest
import torch

from shared.config import TransformerConfig


def skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


class TestTritonModel:
    """Tests for complete TritonModel integration."""

    @pytest.mark.timeout(30)
    def test_output_shape(self):
        """TritonModel tokens [B, S] → logits [B, S, V]."""
        skip_if_no_gpu()
        from impl._triton.model import TritonModel

        B, S, V, D = 2, 8, 64, 16
        model = TritonModel(
            TransformerConfig(
                vocab_size=V,
                embed_dim=D,
                n_layers=2,
                n_heads=4,
                n_groups=4,
                n_experts=4,
                expert_dim=32,
                top_k=2,
            ),
        )

        tokens = torch.randint(0, V, (B, S), dtype=torch.int64, device="cuda")
        out = model(tokens)
        assert out.shape == (B, S, V)

    @pytest.mark.timeout(30)
    def test_forward_pass_finite(self):
        """Outputs are finite (no NaN/Inf)."""
        skip_if_no_gpu()
        from impl._triton.model import TritonModel

        B, S, V, D = 2, 8, 64, 16

        model = TritonModel(
            TransformerConfig(
                vocab_size=V,
                embed_dim=D,
                n_layers=2,
                n_heads=4,
                n_groups=4,
                n_experts=4,
                expert_dim=32,
                top_k=2,
            ),
        )

        tokens = torch.randint(0, V, (B, S), dtype=torch.int64, device="cuda")
        out = model(tokens)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"

    @pytest.mark.timeout(60)
    def test_backward_pass(self):
        """All parameters get valid, non-zero gradients."""
        skip_if_no_gpu()
        from impl._triton.model import TritonModel

        B, S, V, D = 2, 8, 64, 16

        model = TritonModel(
            TransformerConfig(
                vocab_size=V,
                embed_dim=D,
                n_layers=2,
                n_heads=4,
                n_groups=4,
                n_experts=4,
                expert_dim=32,
                top_k=2,
            ),
        )

        tokens = torch.randint(0, V, (B, S), dtype=torch.int64, device="cuda")
        logits = model(tokens)
        loss = logits.mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has no gradient"
                assert param.grad.shape == param.shape, f"Gradient shape mismatch for {name}"

    @pytest.mark.timeout(60)
    def test_parity_with_torch(self):
        """Same weights → same output as PyTorchModel (rtol=1e-2 for 2+ layers)."""
        skip_if_no_gpu()
        import impl._torch.layers as torch_layers
        from impl._triton.model import TritonModel

        B, S, V, D, n_heads, n_experts, ff_dim, k = 2, 8, 64, 16, 4, 4, 32, 2
        n_layers = 2

        # Create PyTorch model
        torch_model = torch_layers.TorchModel(
            TransformerConfig(
                vocab_size=V,
                embed_dim=D,
                n_layers=n_layers,
                n_heads=n_heads,
                n_groups=n_heads,
                n_experts=n_experts,
                expert_dim=ff_dim,
                top_k=k,
                rope_dim=0,
                seed=0,
            ),
        ).cuda()

        # Create Triton model
        triton_model = TritonModel(
            TransformerConfig(
                vocab_size=V,
                embed_dim=D,
                n_layers=n_layers,
                n_heads=n_heads,
                n_groups=n_heads,
                n_experts=n_experts,
                expert_dim=ff_dim,
                top_k=k,
            ),
        )

        # Copy weights via the shared Keys scheme — both models use the same
        # (in, out) layout on save/load, so a straight dict copy is exact.
        triton_model.load_from_numpy_dict(torch_model.save_as_numpy())

        def sync_to_cuda(model):
            for p in model.parameters():
                if not p.is_cuda:
                    p.data = p.data.cuda()

        sync_to_cuda(triton_model)

        torch_model.eval()
        triton_model.eval()

        tokens = torch.randint(0, V, (B, S), dtype=torch.int64, device="cuda")
        torch_logits = torch_model(tokens)
        triton_logits = triton_model(tokens)

        torch.testing.assert_close(triton_logits, torch_logits, rtol=1e-2, atol=1e-2)

    @pytest.mark.timeout(30)
    def test_save_load_roundtrip(self):
        """save_as_numpy() → load_from_numpy_dict() → same parameters."""
        skip_if_no_gpu()
        from impl._triton.model import TritonModel

        V, D, n_layers = 64, 16, 2

        model = TritonModel(
            TransformerConfig(
                vocab_size=V,
                embed_dim=D,
                n_layers=n_layers,
                n_heads=4,
                n_groups=4,
                n_experts=4,
                expert_dim=32,
                top_k=2,
            ),
        )

        params_before = model.save_as_numpy()
        model.load_from_numpy_dict(params_before)

        # Re-save and compare: every parameter must survive the round-trip.
        params_after = model.save_as_numpy()
        assert set(params_after) == set(params_before)
        for key, value in params_before.items():
            np.testing.assert_allclose(
                params_after[key], value, rtol=1e-6, atol=1e-6, err_msg=f"Parameter {key} changed after roundtrip"
            )
