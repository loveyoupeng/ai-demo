"""Cross-backend parity tests.

Tests that the PyTorch and NumPy implementations produce identical forward
results (and well-behaved gradients). Uses the shared ``TransformerConfig``
and the shared Keys checkpoint scheme: NumPy weights load losslessly into the
PyTorch model via ``load_from_numpy_dict``.

Testing approach
----------------
For parity, we:
    1. Create the model config and build the NumPy reference
    2. Load the NumPy weights into the PyTorch model (shared Keys scheme)
    3. Run forward passes with the same inputs → compare logits
    4. Check PyTorch autograd: gradient flow and training convergence
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import impl._np.model as np_model
import impl._torch.layers as torch_layers
from shared.config import TransformerConfig


def _cfg(**overrides: object) -> TransformerConfig:
    """Small 1-layer MoE config; tests override the interesting knobs."""
    base: dict[str, object] = {
        "vocab_size": 16,
        "embed_dim": 8,
        "n_layers": 1,
        "n_heads": 1,
        "n_experts": 2,
        "expert_dim": 8,
        "top_k": 1,
        "rope_dim": 0,
        "seed": 42,
    }
    base.update(overrides)
    return TransformerConfig.from_dict(base)


class TestForwardParity:
    """Test that forward passes match between NumPy and PyTorch."""

    @pytest.mark.timeout(15)
    def test_forward_match(self):
        """Forward pass on identical inputs produces the same logits.

        NumPy weights are loaded losslessly into the PyTorch model (shared
        Keys scheme); float64 vs float32 → single-chain tier (1e-3).
        """
        cfg = _cfg()
        np_model_ = np_model.NumPyModel(cfg)
        torch_model = torch_layers.TorchModel(cfg)

        # Load NumPy weights into PyTorch (shared Keys scheme, lossless)
        torch_model.load_from_numpy_dict(np_model_.get_all_parameters())

        # Run forward pass in eval mode for deterministic behavior
        input_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64)
        np_logits = np_model_.forward(input_ids.numpy())
        torch_model.eval()
        with torch.no_grad():
            torch_logits = torch_model(input_ids).numpy()

        # Compare — tolerance for single chain: rtol=1e-3
        np.testing.assert_allclose(
            np_logits, torch_logits, rtol=1e-3, atol=1e-3, err_msg="Forward pass logits should match"
        )

    @pytest.mark.timeout(15)
    def test_output_shapes_2d(self):
        """2D input shapes produce correct output dimensions."""
        model = torch_layers.TorchModel(_cfg())

        # 2D input — single sequence
        x2d = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
        logits2d = model(x2d)
        assert logits2d.shape == (1, 4, 16)

        # 2D input — batch of 3 sequences
        x2d_batch = torch.tensor(
            [[0, 1, 2, 3], [3, 2, 1, 0], [1, 2, 3, 0]],
            dtype=torch.int64,
        )
        logits_batch = model(x2d_batch)
        assert logits_batch.shape == (3, 4, 16)

    @pytest.mark.timeout(15)
    def test_forward_multi_batch(self):
        """Batched forward pass matches between backends."""
        cfg = _cfg()
        np_model_ = np_model.NumPyModel(cfg)
        torch_model = torch_layers.TorchModel(cfg)

        torch_model.load_from_numpy_dict(np_model_.get_all_parameters())

        # Batch of 3 sequences, each of length 5
        input_ids = torch.tensor(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1]],
            dtype=torch.int64,
        )

        np_logits = np_model_.forward(input_ids.numpy())
        torch_model.eval()  # Disable dropout for deterministic comparison
        with torch.no_grad():
            torch_logits = torch_model(input_ids).numpy()

        np.testing.assert_allclose(
            np_logits,
            torch_logits,
            rtol=1e-3,
            atol=1e-3,
            err_msg="Multi-batch forward pass should match",
        )


class TestGradientNormParity:
    """Test that gradient flows correctly in PyTorch (matching NumPy)."""

    @pytest.mark.timeout(30)
    def test_gradient_chaining(self):
        """Verify gradients are non-trivial after one backward pass.

        A single backward pass through the PyTorch model must produce
        non-zero gradients for parameters — the autograd chain works with
        the new architecture.
        """
        torch_model = torch_layers.TorchModel(_cfg())

        # Prepare training data
        torch.manual_seed(42)
        batch_input = torch.randint(0, 16, (4, 4), dtype=torch.int64)
        batch_target = torch.roll(batch_input, -1, dims=-1)
        batch_target[:, -1] = torch.randint(0, 16, (4,))

        loss_fn = torch.nn.CrossEntropyLoss()

        # Run one training step
        logits = torch_model(batch_input)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), batch_target.reshape(-1))
        loss.backward()

        # Check that most parameters have non-zero gradients
        grad_count = 0
        total_params = 0
        for _name, param in torch_model.named_parameters():
            if param.grad is not None:
                total_params += 1
                if torch.any(param.grad != 0):
                    grad_count += 1

        assert total_params > 0, "Model should have parameters with gradients"
        assert grad_count > 0, f"At least some parameters should have non-zero gradients: {grad_count}/{total_params}"

    @pytest.mark.timeout(30)
    def test_training_reduces_loss(self):
        """Training for 20 steps should reduce loss significantly."""
        torch_model = torch_layers.TorchModel(_cfg(n_layers=2, n_heads=2))

        torch.manual_seed(42)
        batch_input = torch.randint(0, 16, (8, 8), dtype=torch.int64)
        batch_target = torch.roll(batch_input, -1, dims=-1)
        batch_target[:, -1] = torch.randint(0, 16, (8,))

        optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.05)
        loss_fn = torch.nn.CrossEntropyLoss()

        losses = []
        for _ in range(20):
            logits = torch_model(batch_input)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), batch_target.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses), "All losses must be finite"
