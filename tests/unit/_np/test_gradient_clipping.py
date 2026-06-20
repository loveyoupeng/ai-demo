"""Tests for gradient clipping in the NumPy training loop.

Verifies:
    - clip_gradients reduces norms correctly (below max_norm → no change)
    - clip_gradients zero max_norm → no clipping
    - train_step integrates clipping (patch and verify call)
    - No gradient norm explosion with Post-Norm for 50 steps
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np


class TestGradientClipping:
    """Unit tests for clip_gradients and compute_gradient_norm."""

    def test_clip_gradients_below_max_norm_no_change(self) -> None:
        """When gradient norm < max_norm, no clipping occurs — grads unchanged."""
        from impl._np.training import clip_gradients

        grads = {
            "w1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            "b1": np.array([0.5, 0.5], dtype=np.float64),
        }
        original = {k: v.copy() for k, v in grads.items()}

        # Global norm = sqrt(1+4+9+16+0.25+0.25) = ~5.48, so max_norm=10.0 won't clip
        clip_gradients(grads, max_norm=10.0)

        for k in original:
            np.testing.assert_allclose(grads[k], original[k], rtol=1e-12)

    def test_clip_gradients_reduces_norm(self) -> None:
        """When gradient norm > max_norm, all grads are scaled so norm == max_norm."""
        from impl._np.training import clip_gradients, compute_gradient_norm

        grads = {
            "w": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        }
        original_norm = compute_gradient_norm(grads)
        assert original_norm > 1.0, "Baseline: grad norm must be > 1.0"

        clip_gradients(grads, max_norm=1.0)

        after_norm = compute_gradient_norm(grads)
        np.testing.assert_allclose(after_norm, 1.0, rtol=1e-6)
        assert after_norm <= 1.0 + 1e-6, f"Clipped norm {after_norm} exceeds max_norm 1.0"

    def test_clip_gradients_zero_max_norm(self) -> None:
        """max_norm=0 should skip all clipping."""
        from impl._np.training import clip_gradients

        grads = {
            "w": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            "b": np.array([0.5, 0.5], dtype=np.float64),
        }
        original = {k: v.copy() for k, v in grads.items()}

        clip_gradients(grads, max_norm=0.0)

        for k in original:
            np.testing.assert_allclose(grads[k], original[k], rtol=1e-12)

    def test_zero_gradient_norm_is_harmless(self) -> None:
        """Zero gradients must not trigger division-by-zero or other errors."""
        from impl._np.training import clip_gradients, compute_gradient_norm

        grads = {
            "w": np.zeros((3, 3), dtype=np.float64),
        }
        norm = compute_gradient_norm(grads)
        assert norm == 0.0
        # Should not raise
        clip_gradients(grads, max_norm=1.0)
        np.testing.assert_allclose(grads["w"], 0.0, rtol=1e-12)


class TestTrainStepIntegration:
    """Verify gradient clipping is wired into train_step."""

    def test_train_step_calls_clip_gradients(self) -> None:
        """train_step must call clip_gradients before optimizer.step."""
        from impl._np.cross_entropy import CrossEntropyLoss
        from impl._np.model import NumPyModel
        from impl._np.optimizer import AdamW
        from impl._np.training import train_step

        # Use tiny model so numerical backward completes within test timeout
        model = NumPyModel(
            vocab_size=4,
            embed_dim=4,
            n_layers=1,
            n_heads=1,
            n_experts=2,
            ff_dim=4,
            k=1,
            rope_dim=0,
            seed=0,
        )
        x = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int32)
        t = np.array([[1, 2, 3, 0], [2, 3, 0, 1]], dtype=np.int32)
        optimizer = AdamW(lr=0.01)
        loss_fn = CrossEntropyLoss()

        max_norm_value = 1.0
        with patch("impl._np.training.clip_gradients") as mock_clip:
            train_step(model, x, t, loss_fn, optimizer, max_norm=max_norm_value)
            mock_clip.assert_called_once()
            call_kwargs = mock_clip.call_args[1]
            assert call_kwargs.get("max_norm") == max_norm_value, (
                f"Expected max_norm={max_norm_value}, got {call_kwargs}"
            )

    def test_train_step_uses_config_max_grad_norm(self) -> None:
        """When max_norm=0 is passed, clipping is skipped and parameters still update."""
        from impl._np.cross_entropy import CrossEntropyLoss
        from impl._np.model import NumPyModel
        from impl._np.optimizer import AdamW
        from impl._np.training import train_step

        model = NumPyModel(
            vocab_size=4,
            embed_dim=4,
            n_layers=1,
            n_heads=1,
            n_experts=2,
            ff_dim=4,
            k=1,
            rope_dim=0,
            seed=0,
        )
        x = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int32)
        t = np.array([[1, 2, 3, 0], [2, 3, 0, 1]], dtype=np.int32)
        optimizer = AdamW(lr=0.01)
        loss_fn = CrossEntropyLoss()

        initial_params = {k: v.copy() for k, v in model.get_all_parameters().items()}

        # max_norm=0 → no clipping, but params should still update
        train_step(model, x, t, loss_fn, optimizer, max_norm=0.0)

        current = model.get_all_parameters()
        total_delta = sum(np.sum(np.abs(current[k] - initial_params[k])) for k in current)
        assert total_delta > 0, "Parameters must change even without clipping"


class TestNoExplosion:
    """Verify no gradient norm explosion with Post-Norm for many steps."""

    def test_no_grad_norm_explosion_50_steps(self) -> None:
        """Run 50 training steps with Post-Norm — loss must stay finite.

        The train_step function internally calls clip_gradients with max_norm=1.0,
        so gradient norms are bounded by construction.  This tests that the full
        forward → backward → clip → optimizer loop stays numerically stable.
        """
        from impl._np.cross_entropy import CrossEntropyLoss
        from impl._np.model import NumPyModel
        from impl._np.optimizer import AdamW
        from impl._np.training import train_step

        model = NumPyModel(
            vocab_size=4,
            embed_dim=4,
            n_layers=1,
            n_heads=1,
            n_experts=2,
            ff_dim=4,
            k=1,
            rope_dim=0,
            seed=0,
        )
        x = np.random.randint(0, 4, (2, 4), dtype=np.int32)
        t = np.random.randint(0, 4, (2, 4), dtype=np.int32)
        optimizer = AdamW(lr=0.05)
        loss_fn = CrossEntropyLoss()

        for step_idx in range(50):
            loss = train_step(model, x, t, loss_fn, optimizer, max_norm=1.0)
            assert np.isfinite(loss), f"Loss not finite at step {step_idx}: {loss}"

    def test_no_clip_still_stable_for_50_steps(self) -> None:
        """Even without clipping (max_norm=0), 50 steps remain stable with tiny model."""
        from impl._np.cross_entropy import CrossEntropyLoss
        from impl._np.model import NumPyModel
        from impl._np.optimizer import AdamW
        from impl._np.training import train_step

        model = NumPyModel(
            vocab_size=4,
            embed_dim=4,
            n_layers=1,
            n_heads=1,
            n_experts=2,
            ff_dim=4,
            k=1,
            rope_dim=0,
            seed=0,
        )
        x = np.random.randint(0, 4, (2, 4), dtype=np.int32)
        t = np.random.randint(0, 4, (2, 4), dtype=np.int32)
        optimizer = AdamW(lr=0.01)
        loss_fn = CrossEntropyLoss()

        for step_idx in range(50):
            loss = train_step(model, x, t, loss_fn, optimizer, max_norm=0.0)
            assert np.isfinite(loss), f"Loss not finite at step {step_idx}: {loss}"
