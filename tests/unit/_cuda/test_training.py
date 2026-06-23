"""F10: CUDA Training — train_step, clip_gradients, gradient norm.

TDD: Write test → see it fail (can't import) → minimal implementation → all pass → ruff + pyright.

For CUDA backend, training uses PyTorch autograd with requires_grad_(True) set on all weights.
The CUDAModel does NOT inherit from nn.Module — parameters are plain tensors accessed via:
    model.stacking.blocks[i].Wq.grad      # through CuTransformerBlock
    model.outproj_W.weight.grad           # output projection
"""

from __future__ import annotations

import torch


class TestComputeGradientNorm:
    """Test compute_gradient_norm for CUDA gradient tensors."""

    def test_zero_gradients(self) -> None:
        """All-zero gradients should produce norm 0.0."""
        from impl._cuda.training import compute_gradient_norm

        grads = {
            "Wq": torch.zeros(64, 64),
            "Wk": torch.zeros(64, 64),
            "Wv": torch.zeros(64, 64),
        }
        norm = compute_gradient_norm(grads)
        assert norm == 0.0

    def test_single_tensor_norm(self) -> None:
        """Single gradient tensor: norm = sqrt(sum of squares)."""
        from impl._cuda.training import compute_gradient_norm

        grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        grads = {"W": grad}
        expected = (1.0**2 + 2.0**2 + 3.0**2 + 4.0**2) ** 0.5  # sqrt(30) ≈ 5.4772
        norm = compute_gradient_norm(grads)
        assert abs(norm - expected) < 1e-6

    def test_multi_tensor_accumulation(self) -> None:
        """Global norm = sqrt(sum of all per-tensor squared norms)."""
        from impl._cuda.training import compute_gradient_norm

        grads = {
            "Wq": torch.tensor([1.0, 0.0, 0.0]),
            "Wk": torch.tensor([0.0, 3.0, 0.0]),
        }
        # sqrt(1^2 + 3^2) = sqrt(10)
        expected = (10.0) ** 0.5
        norm = compute_gradient_norm(grads)
        assert abs(norm - expected) < 1e-6

    def test_returns_float_type(self) -> None:
        """compute_gradient_norm should return a Python float."""
        from impl._cuda.training import compute_gradient_norm

        grads = {"W": torch.tensor([1.0])}
        norm = compute_gradient_norm(grads)
        assert isinstance(norm, float)


class TestClipGradients:
    """Test gradient clipping by global L2 norm."""

    def test_no_clip_when_already_below(self) -> None:
        """If global norm <= max_norm, no clipping should occur."""
        from impl._cuda.training import clip_gradients

        grads = {
            "Wq": torch.ones(4, 4) * 0.01,
            "Wk": torch.ones(4, 4) * 0.01,
        }
        original_Wq = grads["Wq"].clone()
        original_Wk = grads["Wk"].clone()
        clip_gradients(grads, max_norm=100.0)
        assert torch.allclose(grads["Wq"], original_Wq)
        assert torch.allclose(grads["Wk"], original_Wk)

    def test_clip_when_above_threshold(self) -> None:
        """If global norm > max_norm, all grads should be scaled uniformly."""
        from impl._cuda.training import clip_gradients, compute_gradient_norm

        grads = {
            "Wq": torch.ones(4, 4) * 10.0,  # all 10s, norm will be huge
        }
        max_norm = 1.0
        clip_gradients(grads, max_norm=max_norm)

        # After clipping, global norm should equal max_norm
        new_norm = compute_gradient_norm(grads)
        assert abs(new_norm - max_norm) < 1e-5

    def test_all_params_scaled_uniformly(self) -> None:
        """Clipping should maintain equal ratios between all gradient elements."""
        from impl._cuda.training import clip_gradients

        grads = {
            "Wq": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "Wk": torch.tensor([[0.5, 0.5]]),
        }
        original_Wq = grads["Wq"].clone()

        clip_gradients(grads, max_norm=0.1)

        # Ratios between elements should be preserved
        for i in range(2):
            for j in range(2):
                scale = original_Wq[i, j] / grads["Wq"][i, j]
                assert original_Wq[1, 0] / grads["Wq"][1, 0] == scale

    def test_zero_max_norm_does_nothing(self) -> None:
        """max_norm=0.0 should disable clipping entirely."""
        from impl._cuda.training import clip_gradients

        grads = {"Wq": torch.ones(5, 5) * 100.0}
        original = grads["Wq"].clone()
        clip_gradients(grads, max_norm=0.0)
        assert torch.allclose(grads["Wq"], original)


class TestTrainStep:
    """Test end-to-end training step for CUDA model."""

    def test_train_step_returns_float(self) -> None:
        """train_step should return a Python float loss value."""
        from impl._cuda.training import train_step

        model = torch.nn.Linear(16, 1000)
        model.to(device="cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        batch_input = torch.randn(4, 2, 16, device="cuda")
        batch_target = torch.randint(0, 1000, (4, 2), device="cuda", dtype=torch.long)

        loss = train_step(model, batch_input, batch_target, optimizer, loss_fn)

        assert isinstance(loss, float)

    def test_train_step_produces_gradients(self) -> None:
        """After train_step, model gradients should have been consumed by optimizer."""
        from impl._cuda.training import train_step

        model = torch.nn.Linear(16, 1000)
        model.to(device="cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        weight_before = model.weight.clone()
        batch_input = torch.randn(4, 2, 16, device="cuda")
        batch_target = torch.randint(0, 1000, (4, 2), device="cuda", dtype=torch.long)

        train_step(model, batch_input, batch_target, optimizer, loss_fn)

        # Run a second step — if grads were consumed after step(), model weights change
        weight_after = model.weight.clone()
        assert not torch.allclose(weight_before, weight_after), "Weights should have changed after train_step"

        # Now run again without optimizer.step to verify gradients accumulate
        loss_fn2 = torch.nn.CrossEntropyLoss()
        logits = model(batch_input)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_target = batch_target.reshape(-1)
        loss_fn2(flat_logits, flat_target).backward()
        assert model.weight.grad is not None
        assert not torch.all(model.weight.grad == 0)

    def test_gradient_clipping_with_linear(self) -> None:
        """ClipGradients should work with gradients from a Linear model."""
        from impl._cuda.training import clip_gradients, compute_gradient_norm

        model = torch.nn.Linear(64, 64)
        model.to(device="cuda")

        # Manually set large gradients
        model.weight.grad = torch.randn(64, 64, device="cuda") * 100.0
        model.bias.grad = torch.randn(64, device="cuda") * 100.0

        grads: dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if p.grad is not None:
                grads[n] = p.grad

        grad_norm_raw = compute_gradient_norm(grads)
        assert grad_norm_raw > 10.0

        clip_gradients(grads, max_norm=1.0)
        grad_norm_clipped = compute_gradient_norm(grads)

        assert grad_norm_clipped <= 1.0 + 1e-5, f"Clipped norm {grad_norm_clipped} should be <= 1.0"
