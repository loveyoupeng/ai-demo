"""E10: Training loop — train_step, clip_gradients, compute_gradient_norm."""

import math

import pytest
import torch
import torch.nn as nn


def skip_if_no_gpu():
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


def _move_to_device(tensor, device):
    return tensor.cuda() if device == "cuda" else tensor


class TestClipGradients:
    @pytest.mark.timeout(10)
    def test_no_clipping_when_below_max_norm(self):
        """Gradients unchanged when norm <= max_norm."""
        from impl._triton.training import clip_gradients

        grads = {"w": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])}
        original = {k: v.clone() for k, v in grads.items()}
        clip_gradients(grads, max_norm=10.0)
        for k, v in grads.items():
            assert torch.allclose(v, original[k], atol=1e-7)

    @pytest.mark.timeout(10)
    def test_clipping_when_above_max_norm(self):
        """Gradients scaled down proportionally."""
        from impl._triton.training import clip_gradients

        grads = {"w": torch.tensor([3.0, 4.0]), "b": torch.tensor([1.0, 1.0])}
        # norm = sqrt(9+16+1+1) = 5.0, max_norm = 2.0
        clip_gradients(grads, max_norm=2.0)
        global_norm = torch.sqrt(sum((g**2).sum() for g in grads.values())).item()
        assert abs(global_norm - 2.0) < 1e-4

    @pytest.mark.timeout(10)
    def test_zero_max_norm_disables_clipping(self):
        """max_norm=0.0 skips clipping."""
        from impl._triton.training import clip_gradients

        grads = {"w": torch.tensor([3.0, 4.0])}
        original = grads["w"].clone()
        clip_gradients(grads, max_norm=0.0)
        assert torch.allclose(grads["w"], original, atol=1e-7)

    @pytest.mark.timeout(10)
    def test_in_place_modification(self):
        """Gradients dict is modified in-place."""
        from impl._triton.training import clip_gradients

        grads = {"w": torch.tensor([3.0, 4.0, 0.0])}
        ptr = id(grads["w"])
        clip_gradients(grads, max_norm=2.0)
        assert id(grads["w"]) == ptr


class TestComputeGradientNorm:
    @pytest.mark.timeout(10)
    def test_empty_grads(self):
        """All zeros → norm 0.0."""
        from impl._triton.training import compute_gradient_norm

        grads = {"w": torch.zeros(2, 2)}
        norm = compute_gradient_norm(grads)
        assert abs(norm - 0.0) < 1e-6

    @pytest.mark.timeout(10)
    def test_single_tensor(self):
        """Norm of [3, 4] tensor is 5.0."""
        from impl._triton.training import compute_gradient_norm

        grads = {"w": torch.tensor([3.0, 4.0, 0.0, 0.0])}
        norm = compute_gradient_norm(grads)
        assert abs(norm - 5.0) < 1e-3

    @pytest.mark.timeout(10)
    def test_multiple_tensors(self):
        """Multiple tensors: each sum of squares adds up."""
        from impl._triton.training import compute_gradient_norm

        grads = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 3.0])}
        norm = compute_gradient_norm(grads)
        assert abs(norm - math.sqrt(1.0 + 9.0)) < 1e-3

    @pytest.mark.timeout(10)
    def test_returns_float(self):
        """returns float, not Tensor."""
        from impl._triton.training import compute_gradient_norm

        grads = {"w": torch.tensor([1.0, 2.0])}
        assert isinstance(compute_gradient_norm(grads), float)


class TestTrainStep:
    @pytest.mark.timeout(60)
    def test_single_step_reduces_loss(self):
        """One training step with a simple model — logits are reshaped for CE."""
        skip_if_no_gpu()
        from impl._triton.training import train_step

        B, S, V, D = 2, 8, 64, 16

        model = SimpleModel(V, D)
        model = model.cuda()
        x = torch.randint(0, V, (B, S), dtype=torch.int64).cuda()
        target = torch.randint(0, V, (B, S), dtype=torch.int64).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        loss = train_step(model, x, target, optimizer, loss_fn, max_norm=1.0)
        assert isinstance(loss, float)
        assert not math.isnan(loss)

    @pytest.mark.timeout(60)
    def test_gradient_clipping_applied(self):
        """train_step with max_norm=0 disables clipping (gradients preserved)."""
        skip_if_no_gpu()
        from impl._triton.training import train_step

        B, S, V, D = 2, 8, 64, 16

        model = SimpleModel(V, D)
        model = model.cuda()
        x = torch.randint(0, V, (B, S), dtype=torch.int64).cuda()
        target = torch.randint(0, V, (B, S), dtype=torch.int64).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        loss = train_step(model, x, target, optimizer, loss_fn, max_norm=0.0)
        assert isinstance(loss, float)
        assert not math.isnan(loss)

    @pytest.mark.timeout(60)
    def test_optimizer_zerograd(self):
        """optimizer.zero_grad() is called after each step."""
        skip_if_no_gpu()
        from impl._triton.training import train_step

        B, S, V, D = 2, 8, 64, 16

        model = SimpleModel(V, D)
        model = model.cuda()
        x = torch.randint(0, V, (B, S), dtype=torch.int64).cuda()
        target = torch.randint(0, V, (B, S), dtype=torch.int64).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        train_step(model, x, target, optimizer, loss_fn, max_norm=1.0)

        for param in model.parameters():
            assert param.grad is None, "Gradient should be None after zero_grad()"


class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        return self.linear(x)
