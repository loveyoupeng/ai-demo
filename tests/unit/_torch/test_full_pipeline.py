"""C13.1: Full pipeline — end-to-end training and save/load for PyTorch.

Tests the complete training workflow of the PyTorch-based decoder-only
transformer: training with loss tracking and model serialization.
"""

from __future__ import annotations

import pytest
import torch

from impl._torch.layers import TorchModel


class TestFullTraining:
    """Test end-to-end training pipeline with loss tracking."""

    @pytest.mark.timeout(30)
    def test_loss_decreases(self):
        """Train on synthetic data — loss should decrease after 50 steps.

        Uses a tiny model (vocab=16, embed_dim=8, layers=2) with 16 random
        sequences of length 8. Runs 50 training steps with Adam and verifies
        that the average loss in the second half of training drops significantly
        relative to the first half.
        """

        model = TorchModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=2,
            n_heads=2,
            n_experts=2,
            ff_dim=8,
            k=1,
            rope_dim=0,
            seed=0,
        )

        # Create deterministic synthetic data
        torch.manual_seed(42)
        batch_input = torch.randint(0, 16, (16, 8), dtype=torch.int64)
        # Target is roll of input (next-token prediction pattern)
        batch_target = torch.roll(batch_input, -1, dims=-1)
        batch_target[:, -1] = torch.randint(0, 16, (16,))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        loss_fn = torch.nn.CrossEntropyLoss()

        # Run 50 training steps, recording loss after each step
        losses: list[float] = []
        for _ in range(50):
            logits = model(batch_input)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), batch_target.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        # Loss should decrease by at least 10% across 50 steps
        first_quarter = sum(losses[:10]) / 10
        last_quarter = sum(losses[40:]) / 10
        reduction = (first_quarter - last_quarter) / first_quarter
        assert reduction >= 0.10, (
            f"Loss did not decrease 10%: first_quarter={first_quarter:.4f}, "
            f"last_quarter={last_quarter:.4f}, reduction={reduction:.3f}"
        )
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses), "All losses must be finite"


class TestModelSerialization:
    """Test model parameter save/load round-trip."""

    def test_save_load_round_trip(self):
        """Parameter round-trip through save_as_numpy must produce matching forward pass.

        Procedure:
            1. Create model, run forward pass → save logits
            2. Save parameters via save_as_numpy()
            3. Create new model with same config, load parameters
            4. Run forward pass with same input
            5. Verify logits match to 5 decimal places
        """
        model = TorchModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=1,
            n_experts=2,
            ff_dim=8,
            k=1,
            rope_dim=0,
            seed=0,
        )
        input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)

        # Get initial logits (no grad)
        with torch.no_grad():
            initial_logits = model(input_ids).clone()

        # Save all parameters via save_as_numpy()
        saved_params = model.save_as_numpy()

        # Load into a new model with identical config
        model2 = TorchModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=1,
            n_experts=2,
            ff_dim=8,
            k=1,
            rope_dim=0,
            seed=0,
        )
        model2.load_from_numpy_dict(saved_params)

        # Forward pass should produce identical logits
        with torch.no_grad():
            recovered_logits = model2(input_ids)

        torch.testing.assert_close(
            initial_logits,
            recovered_logits,
            rtol=1e-5,
            atol=1e-5,
            msg="Forward pass after save/load should match",
        )

    def test_save_load_preserves_inference(self):
        """Generate tokens before and after save/load must match.

        After training, generate tokens from model. Then save, load, and
        generate again — outputs must be identical.
        """
        model = TorchModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=1,
            n_experts=2,
            ff_dim=8,
            k=1,
            rope_dim=0,
            seed=0,
        )

        # Train for 5 steps
        torch.manual_seed(42)
        batch_input = torch.randint(0, 16, (4, 4), dtype=torch.int64)
        batch_target = torch.roll(batch_input, -1, dims=-1)
        batch_target[:, -1] = torch.randint(0, 16, (4,))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(5):
            logits = model(batch_input)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), batch_target.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Generate greedy tokens before save
        prompt = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
        with torch.no_grad():
            before_logits = model(prompt)
            before_tokens = torch.argmax(before_logits[:, -1, :], dim=-1)

        # Save parameters
        saved_params = model.save_as_numpy()

        # Load into new model
        model2 = TorchModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=1,
            n_experts=2,
            ff_dim=8,
            k=1,
            rope_dim=0,
            seed=0,
        )
        model2.load_from_numpy_dict(saved_params)

        # Generate greedy tokens after load
        with torch.no_grad():
            after_logits = model2(prompt)
            after_tokens = torch.argmax(after_logits[:, -1, :], dim=-1)

        torch.testing.assert_close(before_tokens, after_tokens)
