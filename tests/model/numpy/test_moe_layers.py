from __future__ import annotations

import numpy as np
import pytest
import torch
from model.numpy.moe import MoELayer, Expert, Router
from model.pytorch.moe import PyTorchMoELayer


class TestRouter:
    """Tests for the Router component."""

    def setup_method(self):
        self.embed_dim = 16
        self.num_experts = 4
        np.random.seed(42)
        self.router = Router(self.embed_dim, self.num_experts)

    @pytest.mark.timeout(30)
    def test_forward_shape(self):
        x = np.random.randn(2, 8, self.embed_dim)
        probs = self.router.forward(x)
        assert probs.shape == (2, 8, self.num_experts)

    @pytest.mark.timeout(10)
    def test_probabilities_are_valid(self):
        x = np.random.randn(2, 8, self.embed_dim)
        probs = self.router.forward(x)
        assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
        assert np.all(probs >= -1e-5)

    @pytest.mark.timeout(60)
    def test_backward_numerical(self):
        x = np.random.randn(2, 8, self.embed_dim)
        self.router.forward(x)
        d_probs = np.random.randn(2, 8, self.num_experts)
        dx, grads = self.router.backward(x, d_probs)

        eps = 1e-5

        def loss_fn(flat_s):
            self.router.w[:] = self.router.w
            x_ = flat_s.reshape(x.shape)
            out = self.router.forward(x_)
            return np.sum(out * d_probs)

        flat = x.ravel().astype(np.float64)
        grad = np.zeros_like(flat)
        for i in range(flat.size):
            flat[i] += eps
            loss_p = loss_fn(flat.copy())
            flat[i] -= 2 * eps
            loss_m = loss_fn(flat.copy())
            flat[i] += eps
            grad[i] = (loss_p - loss_m) / (2 * eps)

        dx_numerical = grad.reshape(x.shape)
        np.testing.assert_allclose(dx, dx_numerical, rtol=1e-2, atol=1e-2)

    @pytest.mark.timeout(30)
    def test_backward_gradient_shapes(self):
        x = np.random.randn(2, 8, self.embed_dim)
        self.router.forward(x)
        d_probs = np.random.randn(2, 8, self.num_experts)
        dx, grads = self.router.backward(x, d_probs)

        assert dx.shape == x.shape
        assert grads["w"].shape == self.router.w.shape


class TestExpert:
    """Tests for the Expert component."""

    def setup_method(self):
        self.embed_dim = 16
        self.dim_ff = 32
        np.random.seed(42)
        self.expert = Expert(self.embed_dim, self.dim_ff)

    @pytest.mark.timeout(30)
    def test_forward_shape(self):
        x = np.random.randn(2, 8, self.embed_dim)
        out = self.expert.forward(x)
        assert out.shape == x.shape

    @pytest.mark.timeout(30)
    def test_backward_input_shape(self):
        x = np.random.randn(2, 8, self.embed_dim)
        self.expert.forward(x)
        d_out = np.random.randn(2, 8, self.embed_dim)
        dx, grads = self.expert.backward(x, d_out)
        assert dx.shape == x.shape

    @pytest.mark.timeout(30)
    def test_backward_gradient_shapes(self):
        x = np.random.randn(2, 8, self.embed_dim)
        self.expert.forward(x)
        d_out = np.random.randn(2, 8, self.embed_dim)
        _, grads = self.expert.backward(x, d_out)

        assert grads["w1"].shape == (self.embed_dim, self.dim_ff)
        assert grads["b1"].shape == (self.dim_ff,)
        assert grads["w2"].shape == (self.dim_ff, self.embed_dim)
        assert grads["b2"].shape == (self.embed_dim,)

    @pytest.mark.timeout(60)
    def test_backward_dx_numerical(self):
        x = np.random.randn(2, 8, self.embed_dim)
        self.expert.forward(x)
        d_out = np.random.randn(2, 8, self.embed_dim)
        dx, _ = self.expert.backward(x, d_out)

        eps = 1e-5
        flat = x.ravel().astype(np.float64)
        grad = np.zeros_like(flat)
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + eps
            y_p = self.expert.forward(flat.copy().reshape(x.shape))
            loss_p = np.sum(y_p * d_out)

            flat[i] = old - eps
            y_m = self.expert.forward(flat.copy().reshape(x.shape))
            loss_m = np.sum(y_m * d_out)

            flat[i] = old
            grad[i] = (loss_p - loss_m) / (2 * eps)

        dx_numerical = grad.reshape(x.shape)
        np.testing.assert_allclose(dx, dx_numerical, rtol=1e-2, atol=1e-2)

    @pytest.mark.timeout(120)
    def test_backward_params_numerical_single(self):
        """Test expert param gradients using element-wise numerical differentiation."""
        x = np.random.randn(2, 8, self.embed_dim)
        d_out = np.random.randn(2, 8, self.embed_dim)
        self.expert.forward(x)
        _, grad_dict = self.expert.backward(x, d_out)

        eps = 1e-5
        for key in ["w1", "b2"]:
            orig = self.expert.get_params()[key].copy()
            num_grad = np.zeros_like(orig)

            for idx in np.ndindex(orig.shape):
                orig[idx] += eps
                self.expert.set_params({key: orig})
                y_p = self.expert.forward(x)
                loss_p = np.sum(y_p * d_out)

                orig[idx] -= 2 * eps
                self.expert.set_params({key: orig})
                y_m = self.expert.forward(x)
                loss_m = np.sum(y_m * d_out)

                orig[idx] += eps
                num_grad[idx] = (loss_p - loss_m) / (2 * eps)

            np.testing.assert_allclose(
                num_grad,
                grad_dict[key],
                rtol=1e-2,
                atol=1e-2,
                err_msg=f"param {key} mismatch",
            )


class TestMoELayer:
    """Tests for the MoELayer component - parity against PyTorch."""

    def setup_method(self):
        self.embed_dim = 16
        self.num_experts = 4
        self.dim_ff = 32
        self.k = 2
        self.batch_size = 2
        self.seq_len = 8

        np.random.seed(42)
        self.numpy_moe = MoELayer(
            self.embed_dim,
            self.num_experts,
            dim_ff=self.dim_ff,
            num_experts_per_token=self.k,
        )
        self.pytorch_moe = PyTorchMoELayer(
            self.embed_dim,
            self.num_experts,
            dim_ff=self.dim_ff,
            num_experts_per_token=self.k,
        )
        self.pytorch_moe.double()

        numpy_params = self.numpy_moe.get_params()
        for name, param in numpy_params.items():
            with torch.no_grad():
                if name.startswith("router."):
                    param_name = name.split(".", 1)[1]
                    self.pytorch_moe.router.set_params(
                        {param_name: torch.from_numpy(param)}
                    )
                elif name.startswith("expert."):
                    parts = name.split(".", 2)
                    expert_idx = int(parts[1])
                    param_name = parts[2]
                    self.pytorch_moe.experts[expert_idx].set_params(
                        {param_name: torch.from_numpy(param)}
                    )

    @pytest.mark.timeout(30)
    def test_forward_shape(self):
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim)
        out, cache = self.numpy_moe.forward(x)
        assert out.shape == (self.batch_size, self.seq_len, self.embed_dim)
        assert "routing_weights" in cache
        assert "top_k_indices" in cache
        assert "top_k_weights" in cache
        assert "all_expert_outputs" in cache

    @pytest.mark.timeout(30)
    def test_forward_parity(self):
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

    @pytest.mark.timeout(60)
    def test_backward_router_w_parity(self):
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(
            np.float64
        )
        d_out = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(
            np.float64
        )

        _, cache = self.numpy_moe.forward(x)
        _, numpy_grads = self.numpy_moe.backward(x, d_out, cache)

        self.pytorch_moe.forward(torch.from_numpy(x))
        _, pytorch_grads = self.pytorch_moe.backward(
            torch.from_numpy(x),
            torch.from_numpy(d_out),
            {
                k: torch.from_numpy(v)
                for k, v in cache.items()
                if isinstance(v, np.ndarray)
            },
        )

        np.testing.assert_allclose(
            numpy_grads["router.w"],
            pytorch_grads["router.w"].detach().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

    @pytest.mark.timeout(60)
    def test_backward_expert_w1_parity(self):
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(
            np.float64
        )
        d_out = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(
            np.float64
        )

        _, cache = self.numpy_moe.forward(x)
        _, numpy_grads = self.numpy_moe.backward(x, d_out, cache)

        self.pytorch_moe.forward(torch.from_numpy(x))
        _, pytorch_grads = self.pytorch_moe.backward(
            torch.from_numpy(x),
            torch.from_numpy(d_out),
            {
                k: torch.from_numpy(v)
                for k, v in cache.items()
                if isinstance(v, np.ndarray)
            },
        )

        np.testing.assert_allclose(
            numpy_grads["expert.0.w1"],
            pytorch_grads["expert.0.w1"].detach().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

    @pytest.mark.timeout(60)
    def test_backward_expert_b2_parity(self):
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(
            np.float64
        )
        d_out = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(
            np.float64
        )

        _, cache = self.numpy_moe.forward(x)
        _, numpy_grads = self.numpy_moe.backward(x, d_out, cache)

        self.pytorch_moe.forward(torch.from_numpy(x))
        _, pytorch_grads = self.pytorch_moe.backward(
            torch.from_numpy(x),
            torch.from_numpy(d_out),
            {
                k: torch.from_numpy(v)
                for k, v in cache.items()
                if isinstance(v, np.ndarray)
            },
        )

        np.testing.assert_allclose(
            numpy_grads["expert.2.b2"],
            pytorch_grads["expert.2.b2"].detach().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

    @pytest.mark.timeout(60)
    def test_backward_dx_parity(self):
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(
            np.float64
        )
        d_out = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(
            np.float64
        )

        _, cache = self.numpy_moe.forward(x)
        dx_numpy, _ = self.numpy_moe.backward(x, d_out, cache)

        self.pytorch_moe.forward(torch.from_numpy(x))
        dx_pytorch, _ = self.pytorch_moe.backward(
            torch.from_numpy(x),
            torch.from_numpy(d_out),
            {
                k: torch.from_numpy(v)
                for k, v in cache.items()
                if isinstance(v, np.ndarray)
            },
        )

        np.testing.assert_allclose(
            dx_numpy,
            dx_pytorch.detach().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )

    @pytest.mark.timeout(10)
    def test_get_params_keys(self):
        params = self.numpy_moe.get_params()
        assert "router.w" in params
        for i in range(self.num_experts):
            assert f"expert.{i}.w1" in params
            assert f"expert.{i}.b1" in params
            assert f"expert.{i}.w2" in params
            assert f"expert.{i}.b2" in params

    @pytest.mark.timeout(10)
    def test_set_params(self):
        params = self.numpy_moe.get_params()
        new_params = {k: v.copy() * 2.0 for k, v in params.items()}
        self.numpy_moe.set_params(new_params)
        for k, v in new_params.items():
            np.testing.assert_array_equal(self.numpy_moe.get_params()[k], v)

    @pytest.mark.timeout(10)
    def test_expert_has_params(self):
        exp = Expert(32, 64)
        assert hasattr(exp, "w1")
        assert hasattr(exp, "b1")
        assert hasattr(exp, "w2")
        assert hasattr(exp, "b2")
