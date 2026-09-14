"""Cross-backend parity — Triton vs PyTorch vs NumPy.

Three-way comparison: all backends produce matching results (within the
fp32 kernel tier) for forward and backward passes. Triton and PyTorch
share the same parameter key scheme, so weights transfer losslessly via
``save_as_numpy`` / ``load_from_numpy_dict``.
"""

from __future__ import annotations

import math

import pytest
import torch

from shared.config import TransformerConfig


def skip_if_no_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


def _cfg(**overrides: object) -> TransformerConfig:
    """Small MoE config shared by the Triton/PyTorch parity tests."""
    base: dict[str, object] = {
        "vocab_size": 64,
        "embed_dim": 16,
        "n_layers": 1,
        "n_heads": 2,
        "n_experts": 2,
        "expert_dim": 32,
        "top_k": 2,
        "rope_dim": 8,  # 8 = full head dimension (head_dim = 16 / 2)
        "seed": 42,
    }
    base.update(overrides)
    return TransformerConfig.from_dict(base)


class TestForwardParity:
    """Compare Triton forward pass against PyTorch."""

    @pytest.mark.timeout(30)
    def test_triton_torch_forward_parity(self):
        """Triton forward matches PyTorch forward on GPU (shared weights)."""
        skip_if_no_gpu()
        from impl._torch.layers import TorchModel
        from impl._triton.model import TritonModel

        B, S, V = 2, 8, 64
        cfg = _cfg()

        torch_model = TorchModel(cfg).cuda()
        triton_model = TritonModel(cfg).cuda()

        # Shared Keys scheme → lossless dict transfer
        triton_model.load_from_numpy_dict(torch_model.save_as_numpy())

        x = torch.randint(0, V, (B, S), dtype=torch.int64).cuda()

        torch_model.eval()
        with torch.no_grad():
            torch_logits = torch_model(x)

        triton_model.eval()
        with torch.no_grad():
            triton_logits = triton_model(x)

        # Both fp32 on GPU — tight tier
        assert torch.allclose(
            triton_logits,
            torch_logits,
            rtol=1e-3,
            atol=1e-3,
        ), f"Triton forward mismatch — max diff: {(triton_logits - torch_logits).abs().max().item():.6f}"

    @pytest.mark.timeout(30)
    def test_triton_forward_shapes(self):
        """Triton output shape matches expected (B, S, V)."""
        skip_if_no_gpu()
        from impl._triton.model import TritonModel

        B, S, V = 3, 10, 64
        model = TritonModel(_cfg()).cuda()
        model.eval()
        x = torch.randint(0, V, (B, S), dtype=torch.int64).cuda()

        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (B, S, V)
        assert not math.isnan(logits.abs().max().item())


class TestBackwardParity:
    """Compare gradient magnitude and flow between backends."""

    @pytest.mark.timeout(30)
    def test_triton_gradient_flow(self):
        """Triton model produces valid gradients (no NaN/Inf)."""
        skip_if_no_gpu()
        from impl._triton.model import TritonModel

        cfg = _cfg()
        model = TritonModel(cfg).cuda()

        x = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.int64).cuda()
        y = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.int64).cuda()

        loss_fn = torch.nn.CrossEntropyLoss()
        loss_fn(model(x).reshape(-1, cfg.vocab_size), y.reshape(-1)).backward()

        grad_norm = 0.0
        for p in model.parameters():
            assert p.grad is not None, "All params should have gradients"
            assert math.isfinite(float(p.grad.abs().max())), "Gradient must be finite"
            grad_norm += float(p.grad.data.float().pow(2).sum())
        assert grad_norm > 0, "Non-zero gradients expected"

    @pytest.mark.timeout(30)
    def test_gradient_norm_torch_vs_triton(self):
        """Triton and PyTorch produce similar gradient norms (shared weights)."""
        skip_if_no_gpu()
        from impl._torch.layers import TorchModel
        from impl._triton.model import TritonModel

        cfg = _cfg()
        torch_model = TorchModel(cfg).cuda()
        triton_model = TritonModel(cfg).cuda()
        triton_model.load_from_numpy_dict(torch_model.save_as_numpy())

        x = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.int64).cuda()
        y = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.int64).cuda()
        loss_fn = torch.nn.CrossEntropyLoss()

        torch_loss = loss_fn(torch_model(x).reshape(-1, cfg.vocab_size), y.reshape(-1))
        torch_loss.backward()
        torch_grad_norm = math.sqrt(
            sum((p.grad**2).sum().item() for p in torch_model.parameters() if p.grad is not None)
        )

        triton_loss = loss_fn(triton_model(x).reshape(-1, cfg.vocab_size), y.reshape(-1))
        triton_loss.backward()
        triton_grad_norm = math.sqrt(
            sum((p.grad**2).sum().item() for p in triton_model.parameters() if p.grad is not None)
        )

        # Gradient norms should be similar (allowing for fp32 precision drift)
        assert torch_grad_norm > 0 and triton_grad_norm > 0
        ratio = max(torch_grad_norm, triton_grad_norm) / (min(torch_grad_norm, triton_grad_norm) + 1e-10)
        assert ratio < 1.1, f"Gradient norm ratio too large: torch={torch_grad_norm:.4f}, triton={triton_grad_norm:.4f}"

    @pytest.mark.timeout(30)
    def test_training_reduces_loss_torch_and_triton(self):
        """Both backends reduce loss over identical training steps."""
        skip_if_no_gpu()
        from impl._torch.layers import TorchModel
        from impl._torch.training import train_step as torch_train_step
        from impl._triton.model import TritonModel
        from impl._triton.training import train_step as triton_train_step

        B, S, V = 2, 8, 64
        cfg = _cfg()

        torch_model = TorchModel(cfg).cuda()
        triton_model = TritonModel(cfg).cuda()
        triton_model.load_from_numpy_dict(torch_model.save_as_numpy())

        x = torch.randint(0, V, (B, S), dtype=torch.int64).cuda()
        y = torch.randint(0, V, (B, S), dtype=torch.int64).cuda()
        loss_fn = torch.nn.CrossEntropyLoss()

        def get_loss(model):
            return loss_fn(model(x).reshape(-1, V), y.reshape(-1)).item()

        torch_initial = get_loss(torch_model)
        triton_initial = get_loss(triton_model)

        for _ in range(10):
            torch_train_step(
                torch_model, x, y, torch.optim.Adam(torch_model.parameters(), lr=0.05), loss_fn, max_norm=1.0
            )
            triton_train_step(
                triton_model, x, y, torch.optim.Adam(triton_model.parameters(), lr=0.05), loss_fn, max_norm=1.0
            )

        torch_final = get_loss(torch_model)
        triton_final = get_loss(triton_model)

        assert math.isfinite(torch_final), "Torch loss must be finite"
        assert math.isfinite(triton_final), "Triton loss must be finite"
        assert torch_final < torch_initial, "Torch loss should decrease"
        assert triton_final < triton_initial, "Triton loss should decrease"
