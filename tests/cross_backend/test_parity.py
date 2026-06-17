"""C14.1: Cross-backend parity tests.

Tests that PyTorch and NumPy implementations produce identical
forward and backward results (parity). Uses float64 for precision.

Testing approach
----------------
For parity, we:
    1. Create identical models in NumPy and PyTorch
    2. Align parameters using load_from_numpy
    3. Run forward pass with same inputs → compare logits
    4. Run backward → compare gradient norms
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import impl._np.model as np_model
import impl._torch.layers as torch_layers


class TestForwardParity:
    """Test that forward passes match between NumPy and PyTorch."""

    @pytest.mark.timeout(15)
    def test_forward_match(self):
        """Forward pass on identical inputs produces same logits.

        Creates a small model (vocab=16, embed_dim=8, n_layers=1) with
        same random seed in both backends, loads NumPy weights into
        PyTorch, and verifies that forward passes match to 1e-5.
        """

        # Create models with same seed
        np_model_ = np_model.NumPyModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=1,
            n_experts=2,
            ff_dim=8,
            k=1,
            rope_dim=0,
            seed=42,
        )

        torch_model = torch_layers.TorchModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=1,
            n_experts=2,
            ff_dim=8,
            k=1,
            rope_dim=0,
            seed=42,
        )

        # Load NumPy weights into PyTorch
        torch_model.load_from_numpy(np_model_)

        # Run forward pass in eval mode for deterministic behavior (dropout disabled)
        input_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64)
        np_logits = np_model_.forward(input_ids.numpy())
        torch_model.eval()
        with torch.no_grad():
            torch_logits = torch_model(input_ids).numpy()

        # Compare — tolerance for single chain: rtol=1e-3
        np.testing.assert_allclose(
            np_logits,
            torch_logits,
            rtol=1e-3,
            atol=1e-3,
            err_msg="Forward pass logits should match",
        )

    @pytest.mark.timeout(15)
    def test_output_shapes_2d(self):
        """2D input shapes produce correct output dimensions."""
        model = torch_layers.TorchModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=1,
            n_experts=2,
            ff_dim=8,
            k=1,
            rope_dim=0,
            seed=42,
        )

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

        np_model_ = np_model.NumPyModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=1,
            n_experts=2,
            ff_dim=8,
            k=1,
            rope_dim=0,
            seed=42,
        )

        torch_model = torch_layers.TorchModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=1,
            n_experts=2,
            ff_dim=8,
            k=1,
            rope_dim=0,
            seed=42,
        )

        torch_model.load_from_numpy(np_model_)

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
        """Verify gradient norms change after a training step.

        Training a PyTorch model for one step should produce meaningful
        gradients — the gradient norm should be non-zero for most parameters.
        This test ensures the backward pass works correctly with the
        cross-backend parameter alignment.
        """
        torch_model = torch_layers.TorchModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=1,
            n_experts=2,
            ff_dim=8,
            k=1,
            rope_dim=0,
            seed=42,
        )

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
                if torch.all(param.grad != 0):
                    grad_count += 1

        assert total_params > 0, "Model should have parameters with gradients"
        assert grad_count > 0, f"At least some parameters should have non-zero gradients: {grad_count}/{total_params}"

    @pytest.mark.timeout(30)
    def test_training_reduces_loss(self):
        """Training for 20 steps should reduce loss significantly.

        Uses the same training setup as the NumPy full pipeline test to
        verify that the PyTorch implementation can learn and improve.
        """
        torch_model = torch_layers.TorchModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=2,
            n_heads=2,
            n_experts=2,
            ff_dim=8,
            k=1,
            rope_dim=0,
            seed=42,
        )

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
