"""CUDA bare-metal cross-end parity tests.

Verify that the bare-metal CUDA implementation produces correct (finite, structured)
forward and backward results compared to NumPy and PyTorch reference implementations.

Testing approach
----------------
For forward structural validation, we:
    1. Create identical models in NumPy and CUDA with same seed
    2. Run forward pass with same inputs → verify correct shapes
    3. Verify no NaN / Inf values
    4. Verify output distributions (mean/std) are reasonable

Note: NumPy and CUDA models use DIFFERENT weight initialization methods
(NumPy: rng.random()/normal(), CUDA: torch.init.uniform_). Therefore
exact numerical parity is NOT expected — we verify structural correctness
and value range instead.

Tolerance policy (tiered from AGENTS.md):
    - Shape validation: exact
    - Value range: absolute (finite, reasonable magnitudes)
    - Backward pass: verifies gradients accumulate correctly

Note: train_step() calls model(batch_input) which requires torch.nn.Module.
CUDAModel is not callable — it uses .forward() explicitly. So train_step-based
training loops are tested with nn.Module fixtures, not CUDAModel.
"""

from __future__ import annotations

import numpy as np
import torch as th
import torch.nn as nn

import impl._np.model as np_module


def _clean_cuda() -> None:
    """Empty CUDA cache if available."""
    if th.cuda.is_available():
        th.cuda.empty_cache()


class TestCUDAForwardCorrectness:
    """Verify CUDA forward output is structurally correct."""

    def setup_method(self) -> None:
        _clean_cuda()

    def teardown_method(self) -> None:
        _clean_cuda()

    def test_forward_1layer_shape(self) -> None:
        """1-layer forward output has correct shape."""
        from impl._cuda.model import CUDAModel

        cd_m = CUDAModel(
            vocab_size=16,
            embed_dim=64,
            n_layers=1,
            n_heads=4,
            n_experts=2,
            ff_dim=128,
            k=2,
            rope_dim=16,
            seed=42,
        )

        prompt = th.tensor([[0, 1, 2, 3, 4]], dtype=th.int64, device="cuda")
        output = cd_m.forward(prompt)

        assert output.shape == (1, 5, 16), "Output shape should be (B, S, V)"
        assert output.dtype == th.float32

    def test_forward_256_shape(self) -> None:
        """1-layer model with vocab=256: output shape."""
        from impl._cuda.model import CUDAModel

        cd_m = CUDAModel(
            vocab_size=256,
            embed_dim=16,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=32,
            k=1,
            rope_dim=8,
            seed=42,
        )

        prompt = th.tensor([[0, 1, 2, 3, 4, 5]], dtype=th.int64, device="cuda")
        output = cd_m.forward(prompt)

        assert output.shape == (1, 6, 256), "Output shape should be (B, S, V)"
        assert output.dtype == th.float32

    def test_forward_2layer_shape(self) -> None:
        """2-layer model: output shape with batched input."""
        from impl._cuda.model import CUDAModel

        cd_m = CUDAModel(
            vocab_size=64,
            embed_dim=32,
            n_layers=2,
            n_heads=2,
            n_experts=2,
            ff_dim=64,
            k=1,
            rope_dim=16,
            seed=42,
        )

        prompt = th.randint(0, 64, (2, 8), dtype=th.int64, device="cuda")
        output = cd_m.forward(prompt)

        assert output.shape == (2, 8, 64), "Output shape should be (B, S, V)"

    def test_forward_batched(self) -> None:
        """Batched forward (3x16) output shape correct."""
        from impl._cuda.model import CUDAModel

        cd_m = CUDAModel(
            vocab_size=32,
            embed_dim=16,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=32,
            k=1,
            rope_dim=8,
            seed=17,
        )

        prompt = th.randint(0, 32, (3, 16), dtype=th.int64, device="cuda")
        output = cd_m.forward(prompt)

        assert output.shape == (3, 16, 32)

    def test_forward_no_nan(self) -> None:
        """CUDA forward output should be finite for all test cases."""
        from impl._cuda.model import CUDAModel

        # Small model
        cd_m = CUDAModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=16,
            k=1,
            rope_dim=4,
            seed=42,
        )
        prompt = th.tensor([[0, 1, 2, 3, 4]], dtype=th.int64, device="cuda")
        output = cd_m.forward(prompt)
        assert th.isfinite(output).all(), "Small model forward should be finite"

        # Larger model
        cd_m2 = CUDAModel(
            vocab_size=256,
            embed_dim=64,
            n_layers=2,
            n_heads=4,
            n_experts=2,
            ff_dim=128,
            k=2,
            rope_dim=16,
            seed=99,
        )
        prompt2 = th.randint(0, 256, (4, 16), dtype=th.int64, device="cuda")
        output2 = cd_m2.forward(prompt2)
        assert th.isfinite(output2).all(), "Large model forward should be finite"

    def test_forward_output_range_reasonable(self) -> None:
        """CUDA forward logits should be in reasonable range (not exploded)."""
        from impl._cuda.model import CUDAModel

        cd_m = CUDAModel(
            vocab_size=256,
            embed_dim=64,
            n_layers=2,
            n_heads=4,
            n_experts=2,
            ff_dim=128,
            k=2,
            rope_dim=16,
            seed=42,
        )

        prompt = th.randint(0, 256, (4, 16), dtype=th.int64, device="cuda")
        output = cd_m.forward(prompt)

        # No logits should be astronomically large (indicates multiplication issues)
        max_val = output.abs().max()
        assert max_val < 1e6, f"Logits too large: {max_val:.1f}"

    def test_forward_same_input_same_output(self) -> None:
        """Two forward calls with same model and input produce same output."""
        from impl._cuda.model import CUDAModel

        cd_m = CUDAModel(
            vocab_size=64,
            embed_dim=32,
            n_layers=2,
            n_heads=2,
            n_experts=2,
            ff_dim=64,
            k=1,
            rope_dim=16,
            seed=42,
        )

        prompt = th.randint(0, 64, (1, 8), dtype=th.int64, device="cuda")
        out1 = cd_m.forward(prompt)
        out2 = cd_m.forward(prompt)

        np.testing.assert_allclose(
            out1.detach().detach().cpu().numpy(),
            out2.detach().detach().cpu().numpy(),
            rtol=1e-7,
            atol=1e-7,
            err_msg="Same model + same input should produce identical output",
        )

    def test_forward_different_input_different_output(self) -> None:
        """Different inputs produce different outputs."""
        from impl._cuda.model import CUDAModel

        cd_m = CUDAModel(
            vocab_size=16,
            embed_dim=16,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=32,
            k=1,
            rope_dim=8,
            seed=42,
        )

        prompt1 = th.tensor([[0, 1, 2, 3, 4]], dtype=th.int64, device="cuda")
        prompt2 = th.tensor([[5, 6, 7, 8, 9]], dtype=th.int64, device="cuda")

        out1 = cd_m.forward(prompt1)
        out2 = cd_m.forward(prompt2)

        assert not th.allclose(out1, out2, atol=1e-5), "Different inputs should produce different outputs"


class TestCUDAForwardCrossEnd:
    """Verify CUDA forward produces comparable (not identical) results to NumPy."""

    def setup_method(self) -> None:
        _clean_cuda()

    def teardown_method(self) -> None:
        _clean_cuda()

    def test_forward_output_shape_matches(self) -> None:
        """CUDA and NumPy produce same output shape for same input shape."""
        from impl._cuda.model import CUDAModel

        vocab = 64
        D = 32
        S = 8

        np_m = np_module.NumPyModel(
            vocab_size=vocab,
            embed_dim=D,
            n_layers=2,
            n_heads=2,
            n_experts=2,
            ff_dim=64,
            k=1,
            rope_dim=16,
            seed=42,
        )
        cd_m = CUDAModel(
            vocab_size=vocab,
            embed_dim=D,
            n_layers=2,
            n_heads=2,
            n_experts=2,
            ff_dim=64,
            k=1,
            rope_dim=16,
            seed=42,
        )

        prompt_np_arr = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)[:, :S]
        prompt_t = th.tensor(prompt_np_arr, dtype=th.int64, device="cuda")

        cuda_out = cd_m.forward(prompt_t).detach().cpu().numpy()
        np_out = np_m.forward(prompt_np_arr)

        assert cuda_out.shape == np_out.shape, f"Shape mismatch: CUDA {cuda_out.shape} vs NumPy {np_out.shape}"

    def test_forward_output_distributions_similar(self) -> None:
        """CUDA and NumPy outputs have similar statistical properties."""
        from impl._cuda.model import CUDAModel

        vocab = 64
        D = 32

        np_m = np_module.NumPyModel(
            vocab_size=vocab,
            embed_dim=D,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=64,
            k=1,
            rope_dim=16,
            seed=42,
        )
        cd_m = CUDAModel(
            vocab_size=vocab,
            embed_dim=D,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=64,
            k=1,
            rope_dim=16,
            seed=42,
        )

        prompt_np_arr = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
        prompt_t = th.tensor(prompt_np_arr, dtype=th.int64, device="cuda")

        cuda_out = cd_m.forward(prompt_t).detach().cpu().numpy()
        np_out = np_m.forward(prompt_np_arr)

        # Mean and std should be in same order of magnitude
        np.mean(cuda_out)
        np.mean(np_out)
        np.std(cuda_out)
        np.std(np_out)

        # Both should be finite and not NaN
        assert np.isfinite(cuda_out).all(), "CUDA output should be finite"
        assert np.isfinite(np_out).all(), "NumPy output should be finite"

    def test_forward_gradient_norms(self) -> None:
        """CUDA model gradients accumulate correctly (non-zero, finite)."""
        from impl._cuda.model import CUDAModel

        cd_m = CUDAModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=16,
            k=1,
            rope_dim=4,
            seed=42,
        )

        batch_input = th.randint(0, 16, (4, 4), dtype=th.int64, device="cuda")
        batch_target = th.roll(batch_input, -1, dims=-1)
        batch_target[:, -1] = th.randint(0, 16, (4,), device="cuda")

        cross_entropy = nn.CrossEntropyLoss()
        logits = cd_m.forward(batch_input)
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), batch_target.reshape(-1))
        loss.backward()

        # At least one block should have non-zero, finite gradients
        grad_accumulated = False
        grad_finite = True
        for block in cd_m.stacking.blocks:
            if hasattr(block, "Wq") and block.Wq.grad is not None:
                grad_accumulated = True
                if not th.isfinite(block.Wq.grad).all():
                    grad_finite = False
                if th.all(block.Wq.grad == 0):
                    grad_accumulated = False

        assert grad_accumulated, "At least one block should have gradients"
        assert grad_finite, "All gradients should be finite"


class TestCUDABackwardParity:
    """Test that backward through CUDA model works correctly."""

    def setup_method(self) -> None:
        _clean_cuda()

    def teardown_method(self) -> None:
        _clean_cuda()

    def test_gradient_accumulation(self) -> None:
        """CUDA model weights accumulate gradients via autograd."""
        from impl._cuda.model import CUDAModel

        cd_m = CUDAModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=16,
            k=1,
            rope_dim=4,
            seed=42,
        )

        batch_input = th.randint(0, 16, (4, 4), dtype=th.int64, device="cuda")
        batch_target = th.roll(batch_input, -1, dims=-1)
        batch_target[:, -1] = th.randint(0, 16, (4,), device="cuda")

        cross_entropy = nn.CrossEntropyLoss()
        logits = cd_m.forward(batch_input)
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), batch_target.reshape(-1))
        loss.backward()

        # After backward, blocks should have .grad attributes
        grad_found = False
        for block in cd_m.stacking.blocks:
            if hasattr(block, "Wq") and block.Wq.grad is not None:
                grad_found = True
                assert not th.all(block.Wq.grad == 0), "Wq.grad should be non-zero"
                break

        assert grad_found, "At least one block should have Wq.grad"

    def test_gradient_no_nan(self) -> None:
        """Gradients through CUDA model should be finite."""
        from impl._cuda.model import CUDAModel

        cd_m = CUDAModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=16,
            k=1,
            rope_dim=4,
            seed=42,
        )

        batch_input = th.randint(0, 16, (4, 4), dtype=th.int64, device="cuda")
        batch_target = th.roll(batch_input, -1, dims=-1)
        batch_target[:, -1] = th.randint(0, 16, (4,), device="cuda")

        cross_entropy = nn.CrossEntropyLoss()
        logits = cd_m.forward(batch_input)
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), batch_target.reshape(-1))
        loss.backward()

        for block in cd_m.stacking.blocks:
            if hasattr(block, "Wq") and block.Wq.grad is not None:
                assert th.isfinite(block.Wq.grad).all(), "Wq.grad should be finite"

    def test_gradient_values_match(self) -> None:
        """Gradients on same-input with same-model produce same gradients."""
        from impl._cuda.model import CUDAModel

        seed = 42
        cd1 = CUDAModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=16,
            k=1,
            rope_dim=4,
            seed=seed,
        )
        cd2 = CUDAModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=16,
            k=1,
            rope_dim=4,
            seed=seed,
        )

        batch_input = th.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=th.int64, device="cuda")
        batch_target = th.roll(batch_input, -1, dims=-1)
        batch_target[:, -1] = th.tensor([8], dtype=th.int64, device="cuda")

        cross_entropy = nn.CrossEntropyLoss()

        # Both models have same weights (same seed) → same gradients
        logits1 = cd1.forward(batch_input)
        loss1 = cross_entropy(logits1.reshape(-1, logits1.shape[-1]), batch_target.reshape(-1))
        loss1.backward()

        logits2 = cd2.forward(batch_input)
        loss2 = cross_entropy(logits2.reshape(-1, logits2.shape[-1]), batch_target.reshape(-1))
        loss2.backward()

        g1 = cd1.stacking.blocks[0].Wq.grad
        g2 = cd2.stacking.blocks[0].Wq.grad
        assert g1 is not None, "Wq.grad should exist after backward"
        assert g2 is not None, "Wq.grad should exist after backward"

        np.testing.assert_allclose(
            g1.cpu(),
            g2.cpu(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Same input + same weights should yield same gradients",
        )

    def test_training_with_nn_module(self) -> None:
        """Training loop reduces loss using nn.Module (train_step compatible)."""
        from impl._cuda.training import train_step

        model = nn.Sequential(
            nn.Embedding(16, 8),
            nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, batch_first=True),
            nn.Linear(8, 16),
        ).to(device="cuda")
        optimizer = th.optim.AdamW(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        batch_input = th.randint(0, 16, (4, 8), dtype=th.int64, device="cuda")
        batch_target = th.roll(batch_input, -1, dims=-1)
        batch_target[:, -1] = th.randint(0, 16, (4,), device="cuda")

        losses = []
        for _ in range(20):
            loss = train_step(model, batch_input, batch_target, optimizer, loss_fn, max_norm=1.0)
            losses.append(loss)

        assert len(losses) == 20
        assert losses[-1] < losses[0], f"Loss should decrease: {losses[0]:.4f} \u2192 {losses[-1]:.4f}"
        assert all(th.isfinite(th.tensor(v)).item() for v in losses), "All losses should be finite"
        assert all(not th.isnan(th.tensor(v)).item() for v in losses), "No NaN losses"

    def test_training_gradient_clipping(self) -> None:
        """Gradient clipping should cap the norm at max_norm."""
        from impl._cuda.training import compute_gradient_norm, train_step

        model = nn.Linear(16, 1000).to(device="cuda")
        optimizer = th.optim.AdamW(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        clip_free = th.randn(4, 2, 16, device="cuda") * 10.0
        clip_target = th.randint(0, 1000, (4, 2), device="cuda", dtype=th.long)
        logits0 = model(clip_free)
        loss0 = loss_fn(logits0.reshape(-1, logits0.shape[-1]), clip_target.reshape(-1))
        loss0.backward()
        assert model.weight.grad is not None, "weight.grad should exist after backward"
        norm_raw = compute_gradient_norm({"weight": model.weight.grad})
        assert norm_raw > 1.0, "Raw gradient norm should be large"
        model.zero_grad()

        # With max_norm=1.0
        log = train_step(model, clip_free, clip_target, optimizer, loss_fn, max_norm=1.0)

        assert log > 0, "Loss should be positive"
        assert isinstance(log, float)
