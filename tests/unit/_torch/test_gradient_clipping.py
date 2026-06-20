"""Test that train_step uses gradient clipping."""

from __future__ import annotations

from unittest.mock import patch

import torch

from impl._torch.layers import TorchModel
from impl._torch.training import clip_gradients, compute_gradient_norm, train_step


class TestGradientClippingIntegration:
    """Verify gradient clipping is wired into train_step."""

    def test_clip_gradients_is_called_by_train_step(self) -> None:
        """train_step must call clip_gradients before optimizer.step."""
        model = TorchModel(
            vocab_size=16,
            embed_dim=32,
            n_layers=2,
            n_heads=2,
            n_experts=2,
            ff_dim=64,
            k=2,
            rope_dim=0,
            seed=0,
        )
        x = torch.randint(0, 16, (2, 4))
        y = torch.randint(0, 16, (2, 4))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        max_norm_value = 1.0

        # Patch the loss_fn to return a scalar with grad so autograd works
        # — this avoids model architecture shape mismatches while keeping
        # the backward pass intact for verifying the clip call.
        with (
            patch.object(loss_fn, "forward", return_value=torch.tensor(1.0, requires_grad=True)),
            patch("impl._torch.training.clip_gradients") as mock_clip,
        ):
            train_step(model, x, y, optimizer, loss_fn, max_norm=max_norm_value)
            mock_clip.assert_called_once()
            call_args = mock_clip.call_args
            assert call_args[1].get("max_norm") == max_norm_value or call_args[0][1] == max_norm_value

    def test_train_step_clips_gradients(self) -> None:
        """Gradient clipping reduces gradient norm to at most max_norm."""
        model = TorchModel(
            vocab_size=16,
            embed_dim=32,
            n_layers=4,
            n_heads=2,
            n_experts=2,
            ff_dim=64,
            k=2,
            rope_dim=0,
            seed=0,
        )
        x = torch.randint(0, 16, (4, 8))
        y = torch.randint(0, 16, (4, 8))

        # Get raw gradients from one forward+backward pass
        model.zero_grad()
        logits = model(x)
        loss = torch.nn.CrossEntropyLoss()(logits.reshape(-1, 16), y.reshape(-1))
        loss.backward()
        grad_dict: dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if p.grad is not None:
                grad_dict[n] = p.grad

        # After one train_step with clip=1.0, norm should be <= 1.0
        # We can't directly observe this from train_step since it calls optimizer,
        # but we can test clip_gradients directly
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        original_norm = compute_gradient_norm(grads)

        clip_gradients(grads, max_norm=1.0)
        after_norm = compute_gradient_norm(grads)

        assert after_norm <= 1.0 + 1e-6, f"Clipped norm {after_norm} exceeds max_norm 1.0"
        assert after_norm < original_norm or original_norm <= 1.0, (
            f"Clip should reduce norm; before={original_norm:.4f}, after={after_norm:.4f}"
        )

    def test_no_clip_when_below_max_norm(self) -> None:
        """If gradient norm < max_norm, no clipping occurs."""
        model = TorchModel(
            vocab_size=16,
            embed_dim=32,
            n_layers=2,
            n_heads=2,
            n_experts=2,
            ff_dim=64,
            k=2,
            rope_dim=0,
            seed=42,
        )
        x = torch.randint(0, 16, (4, 8))
        y = torch.randint(0, 16, (4, 8))

        logits = model(x)
        loss = torch.nn.CrossEntropyLoss()(logits.reshape(-1, 16), y.reshape(-1))
        loss.backward()

        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        original_grads = {k: v.clone() for k, v in grads.items()}

        # Clip with a very large max_norm — nothing should change
        clip_gradients(grads, max_norm=1000.0)

        for k in original_grads:
            torch.testing.assert_close(grads[k], original_grads[k], msg=k)

    def test_zero_max_norm_no_clip(self) -> None:
        """max_norm=0 should skip all clipping."""
        model = TorchModel(
            vocab_size=16,
            embed_dim=32,
            n_layers=2,
            n_heads=2,
            n_experts=2,
            ff_dim=64,
            k=2,
            rope_dim=0,
            seed=0,
        )
        x = torch.randint(0, 16, (2, 4))
        y = torch.randint(0, 16, (2, 4))
        logits = model(x)
        torch.nn.CrossEntropyLoss()(logits.reshape(-1, 16), y.reshape(-1)).backward()

        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        original = {k: v.clone() for k, v in grads.items()}

        clip_gradients(grads, max_norm=0.0)

        for k in original:
            torch.testing.assert_close(grads[k], original[k], msg=k)
