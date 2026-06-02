from __future__ import annotations

import pytest
import numpy as np
from model.numpy.moe import Router, Expert, MoELayer


def _numerical_grad_x(func, x_val, eps=1e-5):
    """Numerical gradient of a scalar-valued func(x) w.r.t. x.

    Uses element-wise finite-difference on the flat array then reshapes
    back to ``x_val.shape``, calling ``func`` with *reshaped* 3-D arrays
    each iteration so the forward pass never sees a 1-D tensor.
    """
    flat = x_val.ravel().astype(np.float64)
    grad = np.zeros_like(flat)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        loss_p = func(flat.copy())

        flat[i] = old - eps
        loss_m = func(flat.copy())

        flat[i] = old
        grad[i] = (loss_p - loss_m) / (2 * eps)
    return grad.reshape(x_val.shape)


@pytest.mark.timeout(20)
def test_router_backward_numerical():
    """Check analytical dx matches finite-difference dx for the router."""
    batch_size, seq_len, embed_dim, num_experts = 2, 3, 4, 5

    router = Router(embed_dim, num_experts)
    w_orig = router.w.copy()
    x = np.random.randn(batch_size, seq_len, embed_dim)

    probs = router.forward(x)
    grad_output = np.random.randn(batch_size, seq_len, num_experts)
    dx_analytical, grads = router.backward(x, grad_output)

    eps = 1e-5
    def loss_fn(flat_s):
        router.w[:] = w_orig
        x_ = flat_s.reshape(x.shape)
        out = router.forward(x_)
        return np.sum(out * grad_output)

    dx_numerical = _numerical_grad_x(loss_fn, x, eps=eps)
    np.testing.assert_allclose(dx_analytical, dx_numerical, rtol=1e-2, atol=1e-2)


@pytest.mark.timeout(10)
def test_router_produces_valid_probabilities():
    router = Router(8, 4)
    x = np.random.randn(2, 5, 8)
    probs = router.forward(x)

    assert probs.shape == (2, 5, 4)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)
    assert np.all(probs >= -1e-5)


@pytest.mark.timeout(10)
def test_expert_param_perturbation_works():
    """Verify that perturbing Expert.params actually changes forward output."""
    exp = Expert(4, 8)
    x = np.random.randn(2, 3, 4)
    eps = 1e-5
    orig_out = exp.forward(x).copy()
    exp.w1 += eps
    out_p = exp.forward(x)
    assert not np.allclose(out_p, orig_out, atol=1e-10), (
        "Perturbing w1 did not change forward output"
    )


@pytest.mark.timeout(30)
def test_expert_backward_numerical():
    batch_size, seq_len, embed_dim, dim_ff = 2, 3, 4, 8

    exp = Expert(embed_dim, dim_ff)
    x = np.random.randn(batch_size, seq_len, embed_dim)

    exp.forward(x)
    grad_output = np.random.randn(batch_size, seq_len, embed_dim)
    dx_analytical, grads = exp.backward(x, grad_output)

    eps = 1e-5
    def loss_fn(flat_s):
        out = exp.forward(flat_s.reshape(x.shape))
        return np.sum(out * grad_output)

    dx_numerical = _numerical_grad_x(loss_fn, x, eps=eps)
    np.testing.assert_allclose(dx_analytical, dx_numerical, rtol=1e-3, atol=1e-3)

    # Verify parameter gradients using the same loss_fn
    params = exp.get_params()
    for key in params:
        grad = np.zeros_like(params[key])
        orig = params[key].astype(np.float64, copy=True)
        flat = grad.ravel()
        orig_flat = orig.ravel().astype(np.float64)
        for i in range(flat.size):
            flat[i] = eps
            params[key][:] = (orig_flat + flat).reshape(orig.shape)
            out_p = exp.forward(x)
            loss_p = np.sum(out_p * grad_output)

            flat.fill(0)
            flat[i] = -2 * eps
            params[key][:] = (orig_flat + flat).reshape(orig.shape)
            out_m = exp.forward(x)
            loss_m = np.sum(out_m * grad_output)

            flat.fill(0)
            grad.ravel()[i] = (loss_p - loss_m) / (2 * eps)
        np.testing.assert_allclose(grad, grads[key], rtol=1e-2, atol=1e-2)


@pytest.mark.timeout(60)
def test_moe_layer_backward_numerical():
    batch_size, seq_len, embed_dim, num_experts = 2, 3, 4, 4
    k = 2

    moe = MoELayer(embed_dim, num_experts, num_experts_per_token=k)
    x = np.random.randn(batch_size, seq_len, embed_dim)

    output, cache = moe.forward(x)

    grad_output = np.random.randn(batch_size, seq_len, embed_dim)
    dx_analytical, grads = moe.backward(x, grad_output, cache)

    eps = 1e-5
    def loss_fn(flat_s):
        out, _ = moe.forward(flat_s.reshape(x.shape))
        return np.sum(out * grad_output)

    dx_numerical = _numerical_grad_x(loss_fn, x, eps=eps)
    np.testing.assert_allclose(dx_analytical, dx_numerical, rtol=1e-2, atol=1e-2)


@pytest.mark.timeout(60)
def test_moe_layer_params_numerical():
    batch_size, seq_len, embed_dim, num_experts = 2, 3, 4, 4
    k = 2

    moe = MoELayer(embed_dim, num_experts, num_experts_per_token=k)
    x = np.random.randn(batch_size, seq_len, embed_dim)

    output, cache = moe.forward(x)
    grad_output = np.random.randn(batch_size, seq_len, embed_dim)
    _, grads_out = moe.backward(x, grad_output, cache)

    params = moe.get_params()

    eps = 1e-5

    for key in params:
        grad = np.zeros_like(params[key])
        orig = params[key].astype(np.float64, copy=True)
        flat = grad.ravel()
        orig_flat = orig.ravel().astype(np.float64)

        # Grab the internal object referenced by the dotted key, e.g. "router.w"
        parts = key.split(".")
        obj = moe
        for p in parts[:-1]:
            obj = getattr(obj, p)
        param_arr = getattr(obj, parts[-1])

        for i in range(flat.size):
            flat[i] = eps
            param_arr.ravel()[:] = (orig_flat + flat).astype(param_arr.dtype)
            out_p = moe.forward(x)[0]
            loss_p = np.sum(out_p * grad_output)

            flat.fill(0)
            flat[i] = -2 * eps
            param_arr.ravel()[:] = (orig_flat + flat).astype(param_arr.dtype)
            out_m = moe.forward(x)[0]
            loss_m = np.sum(out_m * grad_output)

            flat.fill(0)
            grad.ravel()[i] = (loss_p - loss_m) / (2 * eps)

        np.testing.assert_allclose(grad, grads_out[key], rtol=1e-2, atol=1e-2)


@pytest.mark.timeout(5)
def test_expert_shape():
    exp = Expert(16, 64)
    x = np.random.randn(2, 10, 16)
    out = exp.forward(x)
    assert out.shape == (2, 10, 16)


@pytest.mark.timeout(5)
def test_moe_layer_output_shape():
    moe = MoELayer(16, 8, num_experts_per_token=2)
    x = np.random.randn(2, 10, 16)
    out, _ = moe.forward(x)
    assert out.shape == (2, 10, 16)


@pytest.mark.timeout(5)
def test_moe_layer_cache_keys():
    moe = MoELayer(16, 4, num_experts_per_token=2)
    x = np.random.randn(2, 5, 16)
    _, cache = moe.forward(x)
    assert "routing_weights" in cache
    assert "top_k_indices" in cache
    assert "top_k_weights" in cache
    assert "all_expert_outputs" in cache


@pytest.mark.timeout(5)
def test_moe_layer_get_params_keys():
    moe = MoELayer(8, 3, dim_ff=16, num_experts_per_token=2)
    params = moe.get_params()
    assert "router.w" in params
    for i in range(3):
        assert f"expert.{i}.w1" in params
        assert f"expert.{i}.b1" in params
        assert f"expert.{i}.w2" in params
        assert f"expert.{i}.b2" in params


@pytest.mark.timeout(10)
def test_moe_layer_set_params():
    moe = MoELayer(8, 3, dim_ff=16)
    params = moe.get_params()
    new_params = {k: v.copy() * 2.0 for k, v in params.items()}
    moe.set_params(new_params)
    for k, v in new_params.items():
        np.testing.assert_array_equal(moe.get_params()[k], v)


@pytest.mark.timeout(5)
def test_moe_expert_has_direct_params():
    exp = Expert(32, 64)
    assert hasattr(exp, "w1")
    assert hasattr(exp, "b1")
    assert hasattr(exp, "w2")
    assert hasattr(exp, "b2")


@pytest.mark.timeout(5)
def test_moe_layer_has_router_and_experts_attr():
    moe = MoELayer(16, 4)
    assert hasattr(moe, "router")
    assert hasattr(moe, "experts")
    assert len(moe.experts) == 4
