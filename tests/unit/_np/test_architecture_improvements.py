"""Phase 3++: Tests for Post-Norm, Gated Residuals, and Dropout.

Tests the new architecture improvements:
1. Post-Norm (residual add first, then normalize)
2. Gated Residuals (learnable gate controls signal flow)
3. Dropout (random regularization during training)
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import numpy as np
import torch

from impl._np.modules import TransformerBlock as NumPyTransformerBlock
from impl._torch.layers import TransformerBlock as TorchTransformerBlock

# ─────────────────────────────────────────────
# Section 1: Gated Residuals
# ─────────────────────────────────────────────


class TestGatedResidualsInit:
    """Test that gates are properly initialized to near-zero."""

    def test_torch_gate_param_exists(
        self,
    ) -> None:
        """TransformerBlock should have gate1 and gate2 as learnable parameters."""
        block = TorchTransformerBlock(8, n_heads=2, n_experts=2, ff_dim=16)
        gate1: torch.Tensor = block.gate1
        gate2: torch.Tensor = block.gate2
        assert gate1.shape == torch.Size([1]), "gate1 should be [1]-shaped tensor"
        assert gate2.shape == torch.Size([1]), "gate2 should be [1]-shaped tensor"

    def test_torch_gate_init_value(
        self,
    ) -> None:
        """Gate parameters should be initialized to near-zero."""
        block = TorchTransformerBlock(8, n_heads=2, n_experts=2, ff_dim=16)
        gate1_val: float = float(block.gate1.detach().item())
        gate2_val: float = float(block.gate2.detach().item())
        assert abs(gate1_val) < 1e-4, f"gate1 not at zero: {gate1_val}"
        assert abs(gate2_val) < 1e-4, f"gate2 not at zero: {gate2_val}"

    def test_torch_gate_has_grad_tracking(
        self,
    ) -> None:
        """Gate parameters should be learnable (track gradients)."""
        block = TorchTransformerBlock(8, n_heads=2, n_experts=2, ff_dim=16)
        gate1: torch.Tensor = block.gate1
        gate2: torch.Tensor = block.gate2
        assert gate1.requires_grad, "gate1 should track gradients"
        assert gate2.requires_grad, "gate2 should track gradients"

    def test_numpy_gate_shapes(
        self,
    ) -> None:
        """NumPy gates should have correct shapes."""
        block = NumPyTransformerBlock(8, n_heads=2, n_experts=2, ff_dim=16, seed=0)
        assert hasattr(block, "gate1"), "NumPy block missing gate1"
        assert hasattr(block, "gate2"), "NumPy block missing gate2"


class TestGatedResidualsLearn:
    """Test that gates learn during training."""

    def test_gates_change_after_training(
        self,
    ) -> None:
        """Gates should not stay at zero after training steps."""
        block = TorchTransformerBlock(8, n_heads=2, n_experts=2, ff_dim=16)
        gate1_init: torch.Tensor = block.gate1.clone()
        gate2_init: torch.Tensor = block.gate2.clone()

        x = torch.randn(1, 4, 8)
        out = block(x)
        loss = out.sum()
        loss.backward()
        with torch.no_grad():
            for name, p in block.named_parameters():
                if "gate" in name and p.grad is not None:
                    p -= 0.1 * p.grad

        diff1: float = float((block.gate1.abs() - gate1_init.abs()).max())
        diff2: float = float((block.gate2.abs() - gate2_init.abs()).max())
        assert diff1 > 1e-5, f"Gate1 should have changed (diff={diff1})"
        assert diff2 > 1e-5, f"Gate2 should have changed (diff={diff2})"

    def test_gate_controls_output(
        self,
    ) -> None:
        """Gated forward should produce different output with zero gates."""
        block = TorchTransformerBlock(8, n_heads=2, n_experts=2, ff_dim=16)
        x = torch.randn(1, 4, 8)

        out_standard = block(x)

        with torch.no_grad():
            block.gate1.zero_()
            block.gate2.zero_()
        out_gated = block(x)
        max_diff: float = float((out_standard - out_gated).abs().max())
        assert max_diff > 1e-4, "Gates should affect output"


class TestGatedResidualsStability:
    """Test that gates prevent gradient explosion."""

    def test_gradient_norm_stable(
        self,
    ) -> None:
        """Gated residuals should keep gradient norms stable."""
        block = TorchTransformerBlock(embed_dim=8, n_heads=2, n_experts=2, ff_dim=16)
        x = torch.randn(1, 4, 8)

        for step in range(20):
            out = block(x)
            loss = out.sum()
            loss.backward()
            max_grad = max(p.grad.abs().max() for p in block.parameters() if p.grad is not None)
            assert max_grad < 500, f"Gradient exploded after {step + 1} steps: {max_grad:.2f}"


# ─────────────────────────────────────────────
# Section 2: Post-Norm
# ─────────────────────────────────────────────


class TestPostNormOrder:
    """Test that Post-Norm has correct operation order."""

    def test_norm_type_is_post(
        self,
    ) -> None:
        """Block should default to Post-Norm."""
        block = TorchTransformerBlock(8, n_heads=2, n_experts=2, ff_dim=16)
        assert block.norm_type == "post", f"Expected 'post', got '{block.norm_type}'"

    def test_output_changes(
        self,
    ) -> None:
        """Post-Norm output should differ from input."""
        block = TorchTransformerBlock(8, n_heads=2, n_experts=2, ff_dim=16)
        x = torch.randn(1, 4, 8)
        out = block(x)
        diff: float = float((out - x).abs().max())
        assert diff > 1e-4, f"Output should differ from input (diff={diff})"

    def test_gradient_flow(
        self,
    ) -> None:
        """Gradients should flow through all layers in Post-Norm."""
        block = TorchTransformerBlock(8, n_heads=2, n_experts=2, ff_dim=16)
        x = torch.randn(1, 4, 8, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "Gradients should flow to input"
        grad_max: float = float(x.grad.abs().max())
        assert grad_max > 1e-6, f"Gradients should not vanish (max={grad_max})"


class TestPostNormNumPy:
    """Post-Norm behavior in NumPy implementation."""

    def test_output_shape(
        self,
    ) -> None:
        """Input [B, S, D] -> output [B, S, D]."""
        rng = np.random.default_rng(0)
        x: np.ndarray = rng.random((2, 4, 16)).astype(np.float32)
        block = NumPyTransformerBlock(16, n_heads=4, n_experts=4, ff_dim=32, k=2, seed=0)
        out = block.forward(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_output_differs_from_input(
        self,
    ) -> None:
        """Post-Norm output should not equal input."""
        rng = np.random.default_rng(10)
        x: np.ndarray = rng.random((1, 3, 8)).astype(np.float32)
        block = NumPyTransformerBlock(8, n_heads=2, n_experts=2, ff_dim=16, k=1, seed=0)
        out = block.forward(x)
        diff: float = float(np.abs(out - x).max())
        assert diff > 1e-4, f"Output should differ from input (diff={diff})"

    def test_gradient_flows_to_norm(
        self,
    ) -> None:
        """Changing LN gamma should affect output in Post-Norm."""
        rng = np.random.default_rng(30)
        x: np.ndarray = rng.random((1, 3, 8)).astype(np.float32)
        block = NumPyTransformerBlock(8, n_heads=2, n_experts=2, ff_dim=16, k=1, seed=0)
        baseline = block.forward(x.copy())
        gamma: np.ndarray = block.ln1_gamma * 2.0
        block.ln1_gamma = gamma.astype(np.float32)
        out_perturbed = block.forward(x.copy())
        assert not np.allclose(baseline, out_perturbed, atol=1e-3), "Perturbing ln1 gamma should change output"


# ─────────────────────────────────────────────
# Section 3: Dropout
# ─────────────────────────────────────────────


class TestDropoutIdentity:
    """Test that dropout is disabled during inference."""

    def test_torch_inference_deterministic(
        self,
    ) -> None:
        """With dropout disabled, outputs should be identical."""
        block = TorchTransformerBlock(8, n_heads=2, n_experts=2, ff_dim=16)
        block.eval()
        x = torch.randn(1, 4, 8)
        out1 = block(x)
        out2 = block(x)
        assert torch.allclose(out1, out2), "Inference should be deterministic"

    def test_numpy_inference_deterministic(
        self,
    ) -> None:
        """NumPy inference should be deterministic."""
        block = NumPyTransformerBlock(8, n_heads=2, n_experts=2, ff_dim=16, seed=0)
        rng = np.random.default_rng(0)
        x: np.ndarray = rng.random((1, 4, 8)).astype(np.float32)
        out1 = block.forward(x.copy())
        out2 = block.forward(x.copy())
        assert np.allclose(out1, out2), "NumPy inference should be deterministic"


class TestDropoutActive:
    """Test that dropout actually affects training."""

    def test_torch_train_mode_not_deterministic(
        self,
    ) -> None:
        """In train mode, outputs should vary across forward passes."""
        block = TorchTransformerBlock(8, n_heads=2, n_experts=2, ff_dim=16)
        block.train()
        x = torch.randn(1, 4, 8)
        out1 = block(x)
        out2 = block(x)
        max_diff: float = float((out1 - out2).abs().max())
        assert max_diff > 1e-6, "Train mode should not be deterministic"

    def test_gradient_no_nan(
        self,
    ) -> None:
        """Training with dropout enabled should not produce NaN gradients."""
        block = TorchTransformerBlock(8, n_heads=2, n_experts=2, ff_dim=16)
        block.train()
        x = torch.randn(1, 4, 8, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        for p in block.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN gradient in {p.shape}"


# ─────────────────────────────────────────────
# Section 4: Combined — All Three
# ─────────────────────────────────────────────


class TestCombinedArchitecture:
    """Test all three improvements working together."""

    def test_post_norm_gated_deterministic_inference(
        self,
    ) -> None:
        """Post-Norm + Gated + Eval mode = deterministic and stable."""
        block = TorchTransformerBlock(8, n_heads=2, n_experts=2, ff_dim=16)
        block.eval()
        x = torch.randn(1, 4, 8)
        out1 = block(x)
        out2 = block(x)
        assert torch.allclose(out1, out2), "Post-Norm + gated + eval should be deterministic"

    def test_post_norm_gated_grad_flow(
        self,
    ) -> None:
        """Post-Norm + Gated should have gradient flow through all layers."""
        block = TorchTransformerBlock(embed_dim=16, n_heads=4, n_experts=2, ff_dim=32)
        x = torch.randn(1, 8, 16, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        for name, p in block.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN gradient in {name}"
        max_grad_float: float = float(max(p.grad.abs().max() for p in block.parameters() if p.grad is not None))
        assert max_grad_float > 1e-8, "Gradients should flow through Post-Norm + Gated"


class TestAllThreeTogether:
    """Integration test: Post-Norm + Gated + Dropout working together."""

    def test_train_loop_stability(
        self,
    ) -> None:
        """Full training loop should be stable with all three."""
        block = TorchTransformerBlock(embed_dim=64, n_heads=4, n_experts=2, ff_dim=128)
        block.train()

        losses: list[float] = []
        for _step in range(10):
            x = torch.randn(1, 8, 64)
            out = block(x)
            loss = out.sum()
            loss.backward()
            losses.append(float(loss))
            with torch.no_grad():
                for p in block.parameters():
                    if p.grad is not None:
                        p -= 0.01 * p.grad

        for i, loss_val in enumerate(losses):
            assert np.isfinite(loss_val), f"Loss at step {i} should be finite: {loss_val}"

    def test_gates_open_during_training(
        self,
    ) -> None:
        """Gates should open (move away from zero) during training loop."""
        block = TorchTransformerBlock(embed_dim=64, n_heads=4, n_experts=2, ff_dim=128)
        gate1_init = block.gate1.clone()
        block.train()

        for _ in range(5):
            x = torch.randn(1, 8, 64)
            out = block(x)
            loss = out.sum()
            loss.backward()
            with torch.no_grad():
                for p in block.parameters():
                    if p.grad is not None:
                        p -= 0.1 * p.grad

        gate1_final: float = float((block.gate1.abs() - gate1_init.abs()).max())
        assert gate1_final > 0.01, f"Gate1 should open during training (diff={gate1_final})"
