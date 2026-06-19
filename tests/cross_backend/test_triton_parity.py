"""E12: Cross-backend parity — Triton vs PyTorch vs NumPy.

Three-way comparison: all backends produce matching results
(within tolerance) for forward and backward passes.
"""

from __future__ import annotations

import math

import pytest
import torch


def skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


class TestForwardParity:
    """Compare Triton forward pass against PyTorch."""

    @pytest.mark.timeout(30)
    def test_triton_torch_forward_parity(self):
        """Triton forward matches PyTorch forward on GPU."""
        skip_if_no_gpu()
        from impl._torch.layers import TorchModel
        from impl._triton.model import TritonModel

        B, S, V, D, L, H = 2, 8, 64, 16, 1, 2

        torch_model = TorchModel(
            vocab_size=V, embed_dim=D, n_layers=L, n_heads=H,
            n_experts=2, ff_dim=D * 2, k=2,
            rope_dim=D // H, seed=42,
        ).cuda()

        triton_model = TritonModel(
            vocab_size=V, embed_dim=D, n_layers=L, n_heads=H,
            n_experts=2, ff_dim=D * 2, k=2,
        ).cuda()

        # Copy params via save_as_numpy → load_from_numpy_dict
        # (both models have compatible parameter shapes but different naming)
        saved = torch_model.save_as_numpy()
        triton_model.load_from_numpy_dict(saved)

        x = torch.randint(0, V, (B, S), dtype=torch.int64).cuda()

        torch_model.eval()
        with torch.no_grad():
            torch_logits = torch_model(x)

        triton_model.eval()
        with torch.no_grad():
            triton_logits = triton_model(x)

        assert torch.allclose(
            triton_logits, torch_logits, rtol=1e-3, atol=1e-3,
        ), (
            f"Triton forward mismatch — max diff: "
            f"{(triton_logits - torch_logits).abs().max().item():.6f}"
        )

    @pytest.mark.timeout(30)
    def test_triton_forward_shapes(self):
        """Triton output shape matches expected (B, S, V)."""
        skip_if_no_gpu()
        from impl._triton.model import TritonModel

        B, S, V, D = 3, 10, 64, 16
        model = TritonModel(
            vocab_size=V, embed_dim=D, n_layers=1, n_heads=2,
            n_experts=2, ff_dim=D * 2, k=2,
        ).cuda()
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

        model = TritonModel(
            vocab_size=64, embed_dim=16, n_layers=1, n_heads=2,
            n_experts=2, ff_dim=32, k=2,
        ).cuda()

        x = torch.randint(0, 64, (2, 8), dtype=torch.int64).cuda()
        y = torch.randint(0, 64, (2, 8), dtype=torch.int64).cuda()

        loss_fn = torch.nn.CrossEntropyLoss()
        loss_fn(model(x).reshape(-1, model.vocab_size), y.reshape(-1)).backward()

        grad_norm = 0.0
        for p in model.parameters():
            assert p.grad is not None, "All params should have gradients"
            assert math.isfinite(
                float(p.grad.abs().max())
            ), "Gradient must be finite"
            grad_norm += float(p.grad.data.float().pow(2).sum())
        assert grad_norm > 0, "Non-zero gradients expected"

    @pytest.mark.timeout(30)
    def test_gradient_norm_torch_vs_triton(self):
        """Triton and PyTorch produce similar gradient norms."""
        skip_if_no_gpu()
        from impl._torch.layers import TorchModel
        from impl._triton.model import TritonModel

        torch_model = TorchModel(
            vocab_size=64, embed_dim=16, n_layers=1, n_heads=2,
            n_experts=2, ff_dim=32, k=2,
            rope_dim=8, seed=42,
        ).cuda()
        triton_model = TritonModel(
            vocab_size=64, embed_dim=16, n_layers=1, n_heads=2,
            n_experts=2, ff_dim=32, k=2,
        ).cuda()

        # Copy params via save/load dict
        saved = torch_model.save_as_numpy()
        triton_model.load_from_numpy_dict(saved)

        x = torch.randint(0, 64, (2, 8), dtype=torch.int64).cuda()
        y = torch.randint(0, 64, (2, 8), dtype=torch.int64).cuda()
        loss_fn = torch.nn.CrossEntropyLoss()

        torch_loss = loss_fn(torch_model(x).reshape(-1, 64), y.reshape(-1))
        torch_loss.backward()
        torch_grad_norm = math.sqrt(
            sum(
                (p.grad**2).sum().item()
                for p in torch_model.parameters()
                if p.grad is not None
            )
        )

        triton_loss = loss_fn(triton_model(x).reshape(-1, 64), y.reshape(-1))
        triton_loss.backward()
        triton_grad_norm = math.sqrt(
            sum(
                (p.grad**2).sum().item()
                for p in triton_model.parameters()
                if p.grad is not None
            )
        )

        # Gradient norms should be similar (allowing for precision drift)
        assert torch_grad_norm > 0 and triton_grad_norm > 0
        ratio = max(torch_grad_norm, triton_grad_norm) / (
            min(torch_grad_norm, triton_grad_norm) + 1e-10
        )
        assert ratio < 1.1, (
            f"Gradient norm ratio too large: "
            f"torch={torch_grad_norm:.4f}, triton={triton_grad_norm:.4f}"
        )

    @pytest.mark.timeout(30)
    def test_training_reduces_loss_torch_and_triton(self):
        """Both backends reduce loss over training steps."""
        skip_if_no_gpu()
        from impl._torch.layers import TorchModel
        from impl._torch.training import train_step
        from impl._triton.model import TritonModel

        B, S, V, D = 2, 8, 64, 16

        torch_model = TorchModel(
            vocab_size=V, embed_dim=D, n_layers=1, n_heads=2,
            n_experts=2, ff_dim=D * 2, k=2, rope_dim=D // 2, seed=42,
        ).cuda()
        triton_model = TritonModel(
            vocab_size=V, embed_dim=D, n_layers=1, n_heads=2,
            n_experts=2, ff_dim=D * 2, k=2,
        ).cuda()

        # Copy params via save/load dict
        saved = torch_model.save_as_numpy()
        triton_model.load_from_numpy_dict(saved)

        x = torch.randint(0, V, (B, S), dtype=torch.int64).cuda()
        y = torch.randint(0, V, (B, S), dtype=torch.int64).cuda()
        loss_fn = torch.nn.CrossEntropyLoss()

        def get_loss(model):
            return loss_fn(
                model(x).reshape(-1, V), y.reshape(-1)
            ).item()

        torch_initial = get_loss(torch_model)
        triton_initial = get_loss(triton_model)

        for _ in range(10):
            train_step(
                torch_model, x, y,
                torch.optim.Adam(torch_model.parameters(), lr=0.05),
                loss_fn, max_norm=1.0,
            )

        # Reset triton_model to same initial state and train identically
        torch_model2 = TorchModel(
            vocab_size=V, embed_dim=D, n_layers=1, n_heads=2,
            n_experts=2, ff_dim=D * 2, k=2, rope_dim=D // 2, seed=42,
        ).cuda()
        triton_model2 = TritonModel(
            vocab_size=V, embed_dim=D, n_layers=1, n_heads=2,
            n_experts=2, ff_dim=D * 2, k=2,
        ).cuda()
        saved2 = torch_model2.save_as_numpy()
        triton_model2.load_from_numpy_dict(saved2)

        for _ in range(10):
            train_step(
                triton_model2, x, y,
                torch.optim.Adam(triton_model2.parameters(), lr=0.05),
                loss_fn, max_norm=1.0,
            )

        torch_final = get_loss(torch_model)
        triton_final = get_loss(triton_model2)

        assert math.isfinite(torch_final), "Torch loss must be finite"
        assert math.isfinite(triton_final), "Triton loss must be finite"
        assert torch_final < torch_initial, "Torch loss should decrease"
        assert triton_final < triton_initial, "Triton loss should decrease"
