from __future__ import annotations

import numpy as np
import torch
from model.numpy.moe import MoELayer as NumPyMoELayer
from model.pytorch.moe import PyTorchMoELayer


class TestMoEParity:
    """Parity tests comparing NumPy MoE layer against PyTorch MoE layer."""

    def setup_method(self):
        self.embed_dim = 64
        self.num_experts = 4
        self.dim_ff = 128
        self.num_experts_per_token = 2
        self.batch_size = 2
        self.seq_len = 16

        self.numpy_moe = NumPyMoELayer(
            self.embed_dim,
            self.num_experts,
            dim_ff=self.dim_ff,
            num_experts_per_token=self.num_experts_per_token,
        )
        self.pytorch_moe = PyTorchMoELayer(
            self.embed_dim,
            self.num_experts,
            dim_ff=self.dim_ff,
            num_experts_per_token=self.num_experts_per_token,
        )
        # Use float64 parity with NumPy - cast all parameters
        self.pytorch_moe.double()

        # Sync parameters: NumPy -> PyTorch using torch.from_numpy
        # Sync parameters: NumPy -> PyTorch using torch.from_numpy
        numpy_params = self.numpy_moe.get_params()
        for name, param in numpy_params.items():
            with torch.no_grad():
                if name.startswith("router."):
                    param_name = name.split(".", 1)[1]
                    self.pytorch_moe.router.set_params(
                        {param_name: torch.from_numpy(param)}
                    )
                elif name.startswith("expert."):
                    # e.g. "expert.0.w1" -> expert_idx=0, param_name="w1"
                    parts = name.split(".", 2)
                    expert_idx = int(parts[1])
                    param_name = parts[2]
                    self.pytorch_moe.experts[expert_idx].set_params(
                        {param_name: torch.from_numpy(param)}
                    )

    def test_forward_parity(self):
        """Forward pass should match with tolerance rtol=1e-4, atol=1e-4."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(
            np.float64
        )

        numpy_out, numpy_cache = self.numpy_moe.forward(x)
        pytorch_out, pytorch_cache = self.pytorch_moe.forward(torch.from_numpy(x))

        assert torch.from_numpy(numpy_out).dtype == torch.float64
        np.testing.assert_allclose(
            numpy_out,
            pytorch_out.detach().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_backward_router_w(self):
        """Backward w.r.t. router.w should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(
            np.float64
        )
        grad_output = np.random.randn(
            self.batch_size, self.seq_len, self.embed_dim
        ).astype(np.float64)

        _, numpy_cache = self.numpy_moe.forward(x)
        _, numpy_grads = self.numpy_moe.backward(x, grad_output, numpy_cache)

        self.pytorch_moe.forward(torch.from_numpy(x))
        _, pytorch_grads = self.pytorch_moe.backward(
            torch.from_numpy(x),
            torch.from_numpy(grad_output),
            {
                "routing_weights": torch.from_numpy(numpy_cache["routing_weights"]),
                "top_k_indices": torch.from_numpy(numpy_cache["top_k_indices"]),
                "top_k_weights": torch.from_numpy(numpy_cache["top_k_weights"]),
                "top_k_sum": torch.from_numpy(numpy_cache["top_k_sum"]),
                "all_expert_outputs": torch.from_numpy(
                    numpy_cache["all_expert_outputs"]
                ),
            },
        )

        np.testing.assert_allclose(
            numpy_grads["router.w"],
            pytorch_grads["router.w"].detach().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_backward_expert_0_w1(self):
        """Backward w.r.t. expert.0.w1 should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(
            np.float64
        )
        grad_output = np.random.randn(
            self.batch_size, self.seq_len, self.embed_dim
        ).astype(np.float64)

        _, numpy_cache = self.numpy_moe.forward(x)
        _, numpy_grads = self.numpy_moe.backward(x, grad_output, numpy_cache)

        self.pytorch_moe.forward(torch.from_numpy(x))
        _, pytorch_grads = self.pytorch_moe.backward(
            torch.from_numpy(x),
            torch.from_numpy(grad_output),
            {
                "routing_weights": torch.from_numpy(numpy_cache["routing_weights"]),
                "top_k_indices": torch.from_numpy(numpy_cache["top_k_indices"]),
                "top_k_weights": torch.from_numpy(numpy_cache["top_k_weights"]),
                "top_k_sum": torch.from_numpy(numpy_cache["top_k_sum"]),
                "all_expert_outputs": torch.from_numpy(
                    numpy_cache["all_expert_outputs"]
                ),
            },
        )

        np.testing.assert_allclose(
            numpy_grads["expert.0.w1"],
            pytorch_grads["expert.0.w1"].detach().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_backward_expert_1_b2(self):
        """Backward w.r.t. expert.1.b2 should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(
            np.float64
        )
        grad_output = np.random.randn(
            self.batch_size, self.seq_len, self.embed_dim
        ).astype(np.float64)

        _, numpy_cache = self.numpy_moe.forward(x)
        _, numpy_grads = self.numpy_moe.backward(x, grad_output, numpy_cache)

        self.pytorch_moe.forward(torch.from_numpy(x))
        _, pytorch_grads = self.pytorch_moe.backward(
            torch.from_numpy(x),
            torch.from_numpy(grad_output),
            {
                "routing_weights": torch.from_numpy(numpy_cache["routing_weights"]),
                "top_k_indices": torch.from_numpy(numpy_cache["top_k_indices"]),
                "top_k_weights": torch.from_numpy(numpy_cache["top_k_weights"]),
                "top_k_sum": torch.from_numpy(numpy_cache["top_k_sum"]),
                "all_expert_outputs": torch.from_numpy(
                    numpy_cache["all_expert_outputs"]
                ),
            },
        )

        np.testing.assert_allclose(
            numpy_grads["expert.1.b2"],
            pytorch_grads["expert.1.b2"].detach().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_backward_input(self):
        """Backward w.r.t. input x (dx) should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(
            np.float64
        )
        grad_output = np.random.randn(
            self.batch_size, self.seq_len, self.embed_dim
        ).astype(np.float64)

        _, numpy_cache = self.numpy_moe.forward(x)
        dx_numpy, _ = self.numpy_moe.backward(x, grad_output, numpy_cache)

        self.pytorch_moe.forward(torch.from_numpy(x))
        dx_pytorch, _ = self.pytorch_moe.backward(
            torch.from_numpy(x),
            torch.from_numpy(grad_output),
            {
                "routing_weights": torch.from_numpy(numpy_cache["routing_weights"]),
                "top_k_indices": torch.from_numpy(numpy_cache["top_k_indices"]),
                "top_k_weights": torch.from_numpy(numpy_cache["top_k_weights"]),
                "top_k_sum": torch.from_numpy(numpy_cache["top_k_sum"]),
                "all_expert_outputs": torch.from_numpy(
                    numpy_cache["all_expert_outputs"]
                ),
            },
        )

        np.testing.assert_allclose(
            dx_numpy,
            dx_pytorch.detach().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_no_mask_parity(self):
        """Forward pass without mask should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(
            np.float64
        )

        numpy_out, _ = self.numpy_moe.forward(x)
        pytorch_out, _ = self.pytorch_moe.forward(torch.from_numpy(x))

        np.testing.assert_allclose(
            numpy_out,
            pytorch_out.detach().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_cache_integrity(self):
        """Verify cache keys are identical between NumPy and PyTorch."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(
            np.float64
        )

        _, numpy_cache = self.numpy_moe.forward(x)
        _, pytorch_cache = self.pytorch_moe.forward(torch.from_numpy(x))

        numpy_keys = set(numpy_cache.keys())
        pytorch_keys = set(pytorch_cache.keys())
        assert numpy_keys == pytorch_keys, (
            f"Cache keys differ: NumPy has {numpy_keys}, PyTorch has {pytorch_keys}"
        )

        for key in numpy_keys:
            np.testing.assert_allclose(
                numpy_cache[key],
                pytorch_cache[key].detach().numpy(),
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Cache mismatch for key: {key}",
            )
