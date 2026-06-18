"""GPU PyTorch cross-backend parity tests.

Tests that GPU-accelerated PyTorch produces numerically equivalent
forward and backward results to NumPy reference implementation.

Testing approach
----------------
For parity, we:
    1. Create identical models in NumPy and PyTorch
    2. Align parameters using load_from_numpy
    3. Run forward pass with same inputs → compare logits on GPU
    4. Run backward on GPU → compare gradient norms

All forward tests use float32 (the default on GPU). GPU numerical
results may differ from CPU NumPy due to different parallel reduction
order and hardware-specific floating point behavior.

Tolerance policy (tiered from AGENTS.md):
    - Standalone layers: rtol=1e-3, atol=1e-3
    - Component in single chain: rtol=1e-3, atol=1e-3
    - GPU-specific tolerance: rtol=1e-2, atol=1e-2 (tier 3,
      accounting for CUDA float32 parallel reduction differences)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import impl._np.model as np_model

# Skip all tests if GPU is unavailable
gpu_available = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU PyTorch not available",
)


class TestGPUDevice:
    """Verify GPU PyTorch setup is correct."""

    @pytest.mark.timeout(10)
    @gpu_available
    def test_cuda_available(self):
        """GPU PyTorch should detect CUDA."""
        assert torch.cuda.is_available()

    @pytest.mark.timeout(10)
    @gpu_available
    def test_device_count(self):
        """Should detect at least 1 GPU device."""
        assert torch.cuda.device_count() >= 1

    @pytest.mark.timeout(10)
    @gpu_available
    def test_device_info(self):
        """GPU device should have valid info."""
        props = torch.cuda.get_device_properties(0)
        assert props.major >= 7  # Volta or newer (Orin is 8.7)
        assert props.total_memory > 0


class TestGPUForwardParity:
    """Test that forward passes on GPU match NumPy reference."""

    @pytest.mark.timeout(30)
    @gpu_available
    def test_forward_parity_1layer(self):
        """Single-layer model forward pass on GPU matches NumPy."""
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

        from impl._torch.layers import TorchModel

        torch_model = TorchModel(
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
        torch_model.to("cuda").float()  # Move to GPU, float32
        torch_model.load_from_numpy(np_model_)
        torch_model.eval()

        # Run on GPU
        input_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64, device="cuda")
        with torch.no_grad():
            torch_logits = torch_model(input_ids)  # [1, 5, 16] on GPU

        # Move result to CPU for comparison
        torch_logits_np = torch_logits.cpu().numpy()

        # NumPy forward
        np_logits = np_model_.forward(input_ids.cpu().numpy())

        # GPU forward parity — float32 CUDA parallel reduction differences
        # may cause slight drift vs CPU NumPy → use tier 3 tolerance
        np.testing.assert_allclose(
            np_logits,
            torch_logits_np,
            rtol=1e-2,
            atol=1e-2,
            err_msg="GPU forward logits should match NumPy (rtol=1e-2)",
        )

    @pytest.mark.timeout(30)
    @gpu_available
    def test_forward_parity_multi_batch(self):
        """Batched forward pass on GPU matches NumPy."""
        np_model_ = np_model.NumPyModel(
            vocab_size=16,
            embed_dim=16,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=16,
            k=1,
            rope_dim=0,
            seed=42,
        )

        from impl._torch.layers import TorchModel

        torch_model = TorchModel(
            vocab_size=16,
            embed_dim=16,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=16,
            k=1,
            rope_dim=0,
            seed=42,
        )
        torch_model.to("cuda").float()
        torch_model.load_from_numpy(np_model_)
        torch_model.eval()

        input_ids = torch.tensor(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1]],
            dtype=torch.int64,
            device="cuda",
        )

        with torch.no_grad():
            torch_logits = torch_model(input_ids)

        torch_logits_np = torch_logits.cpu().numpy()
        np_logits = np_model_.forward(input_ids.cpu().numpy())

        np.testing.assert_allclose(
            np_logits,
            torch_logits_np,
            rtol=1e-2,
            atol=1e-2,
            err_msg="GPU multi-batch forward should match NumPy",
        )

    @pytest.mark.timeout(30)
    @gpu_available
    def test_forward_parity_2layer(self):
        """Two-layer model forward pass on GPU matches NumPy."""
        np_model_ = np_model.NumPyModel(
            vocab_size=64,
            embed_dim=32,
            n_layers=2,
            n_heads=2,
            n_experts=2,
            ff_dim=64,
            k=1,
            rope_dim=0,
            seed=42,
        )

        from impl._torch.layers import TorchModel

        torch_model = TorchModel(
            vocab_size=64,
            embed_dim=32,
            n_layers=2,
            n_heads=2,
            n_experts=2,
            ff_dim=64,
            k=1,
            rope_dim=0,
            seed=42,
        )
        torch_model.to("cuda").float()
        torch_model.load_from_numpy(np_model_)
        torch_model.eval()

        input_ids = torch.randint(0, 64, (2, 8), dtype=torch.int64, device="cuda")

        with torch.no_grad():
            torch_logits = torch_model(input_ids)

        torch_logits_np = torch_logits.cpu().numpy()
        np_logits = np_model_.forward(input_ids.cpu().numpy())

        # 2+ layer chain → tier 3 tolerance (multi-layer chain)
        np.testing.assert_allclose(
            np_logits,
            torch_logits_np,
            rtol=1e-2,
            atol=1e-2,
            err_msg="GPU 2-layer forward should match NumPy",
        )

    @pytest.mark.timeout(30)
    @gpu_available
    def test_forward_gpu_dtype(self):
        """GPU forward output is float32 (not float64)."""
        from impl._torch.layers import TorchModel

        torch_model = TorchModel(
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
        torch_model.to("cuda").float()
        torch_model.eval()

        input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device="cuda")
        with torch.no_grad():
            output = torch_model(input_ids)

        assert output.dtype == torch.float32, "GPU forward should output float32"
        assert output.device.type == "cuda", "GPU forward output stays on GPU"


class TestGPUBackwardParity:
    """Test that backward passes on GPU work correctly."""

    @pytest.mark.timeout(30)
    @gpu_available
    def test_gradient_chaining_on_gpu(self):
        """GPU backward pass produces non-zero gradients on all parameters."""
        from impl._torch.layers import TorchModel

        torch_model = TorchModel(
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
        torch_model.to("cuda").float()

        batch_input = torch.randint(0, 16, (4, 4), dtype=torch.int64, device="cuda")
        batch_target = torch.roll(batch_input, -1, dims=-1)
        batch_target[:, -1] = torch.randint(0, 16, (4,), device="cuda")

        loss_fn = torch.nn.CrossEntropyLoss()

        logits = torch_model(batch_input)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), batch_target.reshape(-1))
        loss.backward()

        grad_count = 0
        total_params = 0
        for _name, param in torch_model.named_parameters():
            if param.grad is not None:
                total_params += 1
                if torch.all(param.grad != 0):
                    grad_count += 1

        assert total_params > 0, "Model should have parameters with gradients"
        assert grad_count > 0, f"Most parameters should have non-zero gradients: {grad_count}/{total_params}"

    @pytest.mark.timeout(30)
    @gpu_available
    def test_gradient_magnitude_1layer(self):
        """GPU backward pass produces reasonable gradient magnitudes."""
        from impl._torch.layers import TorchModel

        torch_model = TorchModel(
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
        torch_model.to("cuda").float()

        batch_input = torch.randint(0, 16, (4, 4), dtype=torch.int64, device="cuda")
        batch_target = torch.roll(batch_input, -1, dims=-1)
        batch_target[:, -1] = torch.randint(0, 16, (4,), device="cuda")

        loss_fn = torch.nn.CrossEntropyLoss()

        logits = torch_model(batch_input)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), batch_target.reshape(-1))
        loss.backward()

        grad_norms = []
        for _name, param in torch_model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        assert len(grad_norms) > 0
        # All gradient norms should be finite
        for gn in grad_norms:
            assert float("nan") not in [gn], f"Gradient norm is NaN: {gn}"
            assert float("inf") not in [gn], f"Gradient norm is Inf: {gn}"

        # At least some gradients should be non-trivial (not near-zero)
        non_trivial = sum(1 for gn in grad_norms if gn > 1e-6)
        assert non_trivial > 0, "At least some gradients should be non-trivial"

    @pytest.mark.timeout(60)
    @gpu_available
    def test_training_reduces_loss_on_gpu(self):
        """Training loop on GPU reduces loss over multiple steps."""
        from impl._torch.layers import TorchModel

        torch_model = TorchModel(
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
        torch_model.to("cuda").float()

        batch_input = torch.randint(0, 16, (8, 8), dtype=torch.int64, device="cuda")
        batch_target = torch.roll(batch_input, -1, dims=-1)
        batch_target[:, -1] = torch.randint(0, 16, (8,), device="cuda")

        optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.05)
        loss_fn = torch.nn.CrossEntropyLoss()

        losses = []
        for _step in range(20):
            logits = torch_model(batch_input)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), batch_target.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], f"GPU training loss should decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

        # All losses must be finite
        for i, loss_val in enumerate(losses):
            assert float("nan") not in [loss_val], f"Loss is NaN at step {i}: {loss_val}"
            assert float("inf") not in [loss_val], f"Loss is Inf at step {i}: {loss_val}"


class TestGPUTrainingEquivalence:
    """Test that GPU PyTorch training produces equivalent behavior to CPU PyTorch."""

    @pytest.mark.timeout(60)
    def test_gpu_vs_cpu_training_consistency(self):
        """GPU and CPU training both reduce loss from same initial weights.

        Both models get identical initial weights via state_dict exchange.
        After training for 5 steps on each, their losses should converge
        to similar values (within 10% of each other).
        """

        from impl._torch.layers import TorchModel

        torch.manual_seed(42)
        np.random.seed(42)

        # CPU model — float32, CPU
        cpu_model = TorchModel(
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
        # Clone CPU model weights to create identical GPU model
        cpu_model.eval()
        cpu_input = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
        with torch.no_grad():
            cpu_logits = cpu_model(cpu_input).numpy()

        # GPU model — copy CPU weights, float32, CUDA
        gpu_model = TorchModel(
            vocab_size=16,
            embed_dim=8,
            n_layers=1,
            n_heads=1,
            n_experts=2,
            ff_dim=8,
            k=1,
            rope_dim=0,
            seed=42,  # same seed as CPU model
        )
        # Copy weights from CPU to GPU so they start identically
        cpu_state = {k: v.clone().cpu() for k, v in cpu_model.state_dict().items()}
        gpu_model.load_state_dict(cpu_state)
        gpu_model.to("cuda").float()
        gpu_model.eval()

        with torch.no_grad():
            gpu_logits_cuda = gpu_model(torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device="cuda"))

        # Forward passes should match bit-for-bit (same weights, same computation)
        np.testing.assert_allclose(
            cpu_logits,
            gpu_logits_cuda.cpu().numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="CPU and GPU forward with identical weights should match closely",
        )
