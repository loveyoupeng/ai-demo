"""Full pipeline: training, save/load round-trip, and inference.

End-to-end tests that validate the complete training workflow of the
NumPy-based decoder-only transformer.

Testing levels
--------------
1. Numerical stability — forward pass produces finite values
2. Training dynamics — loss decreases across gradient steps
3. Model serialization — save/load preserves all parameters exactly
4. Autoregressive inference — generate_tokens runs without errors after training
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from impl._np.cross_entropy import CrossEntropyLoss
from impl._np.inference import TextGenerator
from impl._np.model import NumPyModel
from impl._np.optimizer import AdamW
from impl._np.training import train_step


class TestFullTraining:
    """Test end-to-end training pipeline with loss tracking."""

    @pytest.mark.timeout(30)
    def test_loss_decreases(self):
        """Train on synthetic data — loss should decrease after 5 steps.

        Uses a tiny model (vocab=16, embed_dim=8, layers=1) with 100 random
        sequences of length 8. Runs 5 training steps with AdamW and verifies
        that loss drops by at least 5% relative to the initial loss.
        """

        model = NumPyModel(
            vocab_size=16,
            embed_dim=8,  # slightly larger for tractable optimization
            n_layers=1,
            n_heads=1,  # minimum: stable with tiny embed_dim
            n_experts=2,
            ff_dim=8,  # reduced to keep params low for numerical backward
            k=1,
            rope_dim=0,
            seed=0,
        )

        # 10 sequences of length 8 — small but enough for stable gradient signal
        rng = np.random.default_rng(42)
        batch_input = rng.integers(0, 16, (10, 8), dtype=np.int32)
        batch_target = (np.roll(batch_input, 1, axis=-1) % 16).copy()
        batch_target[:, 0] = rng.integers(0, 16, (10,), dtype=np.int32)

        loss_fn = CrossEntropyLoss()
        optimizer = AdamW(lr=0.05)  # high enough for significant 5-step decrease

        # Run 5 training steps, recording loss after each step
        losses: list[float] = []
        for _ in range(5):
            losses.append(float(train_step(model, batch_input, batch_target, loss_fn, optimizer)))

        # Loss should decrease by at least 5% across 5 steps
        reduction = (losses[0] - losses[-1]) / losses[0]
        assert reduction >= 0.05, (
            f"Loss did not decrease 5%: {losses} (initial={losses[0]:.4f}, final={losses[-1]:.4f})"
        )
        assert all(np.isfinite(loss_val) for loss_val in losses), "All losses must be finite"


class TestModelSerialization:
    """Test model parameter save/load round-trip."""

    def test_save_load_round_trip(self):
        """Parameter round-trip through npz serialization must match exactly.

        Procedure:
            1. Create model, run forward pass → save logits
            2. Serialize all parameters to npz checkpoint
            3. Create new model with same config, load parameters
            4. Run forward pass with same input
            5. Verify logits match to 5 decimal places
        """

        model = NumPyModel(
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

        input_ids = np.array([[0, 1, 2, 3]], dtype=np.int32)

        # Get initial logits
        initial_logits = model.forward(input_ids)

        # Serialize all parameters + embedding weights to disk
        params = model.get_all_parameters()
        checkpoint_dir = Path(tempfile.mkdtemp())

        try:
            # Serialize parameters to individual .npy files (avoids np.savez
            # pyright false-positive about allow_pickle with **kwargs)
            for name, param in params.items():
                np.save(str(checkpoint_dir / f"{name}.npy"), param, allow_pickle=False)

            # Load into a new model with identical config
            model2 = NumPyModel(
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
            loaded: dict[str, np.ndarray] = {}
            for name in params:
                loaded[name] = np.load(str(checkpoint_dir / f"{name}.npy"), allow_pickle=False)

            for name, param in model2.get_all_parameters().items():
                if name in loaded:
                    param[:] = loaded[name]

            # Forward pass should produce identical logits
            recovered_logits = model2.forward(input_ids)
            np.testing.assert_array_almost_equal(
                initial_logits,
                recovered_logits,
                decimal=5,
                err_msg="Forward pass after save/load should match",
            )
        finally:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

    def test_save_load_preserves_inference(self):
        """Generate tokens before and after save/load must match.

        After training, generate tokens from model. Then save, load, and
        generate again — outputs must be identical.
        """

        model = NumPyModel(
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
        loss_fn = CrossEntropyLoss()
        optimizer = AdamW(lr=0.01)

        # Train for 3 steps
        batch_input = np.array([[0, 1, 2, 3]], dtype=np.int32)
        batch_target = np.array([[1, 2, 3, 0]], dtype=np.int32)
        for _ in range(3):
            train_step(model, batch_input, batch_target, loss_fn, optimizer)

        # Generate before save
        generator = TextGenerator(model, max_new_tokens=3, temperature=0.0)
        prompt = np.array([[0, 1, 2, 3]], dtype=np.int32)
        sequence_before = generator.generate_greedy(prompt)

        # Save and load
        checkpoint_dir = Path(tempfile.mkdtemp())
        try:
            model_params = model.get_all_parameters()
            for name, param in model_params.items():
                np.save(str(checkpoint_dir / f"{name}.npy"), param, allow_pickle=False)

            model2 = NumPyModel(
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
            loaded: dict[str, np.ndarray] = {}
            for name in model_params:
                loaded[name] = np.load(str(checkpoint_dir / f"{name}.npy"), allow_pickle=False)

            for name, param in model2.get_all_parameters().items():
                if name in loaded:
                    param[:] = loaded[name]
        finally:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

        # Generate after load
        generator2 = TextGenerator(model2, max_new_tokens=3, temperature=0.0)
        sequence_after = generator2.generate_greedy(prompt)

        np.testing.assert_array_equal(
            sequence_before,
            sequence_after,
            err_msg="Greedy generation must match before and after save/load",
        )


class TestInference:
    """Test autoregressive generation after training."""

    def test_inference_after_training(self):
        """After training, generate_tokens runs without errors.

        Train a tiny model for 2 steps on a small batch, then use the
        TextGenerator to generate tokens autoregressively with greedy
        decoding. Verify output shape and token ranges.
        """

        model = NumPyModel(
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
        loss_fn = CrossEntropyLoss()
        optimizer = AdamW(lr=0.01)

        # Create training batch: 2 sequences of length 4
        batch_input = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int32)
        batch_target = np.array([[1, 2, 3, 0], [2, 3, 0, 1]], dtype=np.int32)

        # Train for 2 steps
        for _ in range(2):
            train_step(model, batch_input, batch_target, loss_fn, optimizer)

        # Create generator and produce sequences
        generator = TextGenerator(model, max_new_tokens=3, temperature=0.0)
        prompt = np.array([[0, 1, 2, 3]], dtype=np.int32)

        sequence = generator.generate_greedy(prompt)

        # Verify output shape: (batch=1, prompt_len=4 + new_tokens=3) = (1, 7)
        assert sequence.shape == (1, 7), f"Expected (1, 7), got {sequence.shape}"
        assert sequence.shape[0] == 1  # single sequence
        assert sequence.shape[1] == 7  # 4 prompt tokens + 3 generated tokens

        # All values must be in valid vocab range
        assert np.all(sequence >= 0), "Token IDs must be >= 0"
        assert np.all(sequence < 16), "Token IDs must be < vocab_size"

        assert np.all(np.isfinite(sequence)), "Sequence tokens must be finite"

    def test_generate_sampled(self):
        """Temperature-sampled generation produces valid token sequences."""

        model = NumPyModel(
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
        loss_fn = CrossEntropyLoss()
        optimizer = AdamW(lr=0.01)

        batch_input = np.array([[0, 1, 2, 3]], dtype=np.int32)
        batch_target = np.array([[1, 2, 3, 0]], dtype=np.int32)

        for _ in range(5):
            train_step(model, batch_input, batch_target, loss_fn, optimizer)

        # Sampled generation with temperature
        generator = TextGenerator(model, max_new_tokens=4, temperature=1.0)
        prompt = np.array([[0, 1, 2, 3]], dtype=np.int32)

        sequence = generator.generate_sampled(prompt, temperature=1.0)

        assert sequence.shape == (1, 8), f"Expected (1, 8), got {sequence.shape}"
        assert np.all(sequence >= 0), "Token IDs must be >= 0"
        assert np.all(sequence < 16), "Token IDs must be < vocab_size"

    def test_multi_batch_generate(self):
        """Greedy generation with batch_size > 1 works correctly."""
        model = NumPyModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=16,
            k=1,
            rope_dim=0,
            seed=0,
        )

        generator = TextGenerator(model, max_new_tokens=2, temperature=0.0)
        prompt = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)

        sequence = generator.generate_greedy(prompt)

        # batch=2, prompt_len=4 + new_tokens=2 = 6
        assert sequence.shape == (2, 6), f"Expected (2, 6), got {sequence.shape}"
        assert np.all(sequence >= 0), "Token IDs must be >= 0"
        assert np.all(sequence < 16), "Token IDs must be < vocab_size"
