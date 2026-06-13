from __future__ import annotations

import pytest
import numpy as np
from model.moe import MoELayer


def numerical_gradient_moe(moe, x, eps=1e-6):
    grad_x = np.zeros_like(x)

    def get_loss(x_val):
        output, _ = moe.forward(x_val)
        return np.sum(output)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        x[idx] = old_val + eps
        loss_plus = get_loss(x)

        x[idx] = old_val - eps
        loss_minus = get_loss(x)

        grad_x[idx] = (loss_plus - loss_minus) / (2 * eps)

        x[idx] = old_val
        it.iternext()
    return grad_x


@pytest.mark.timeout(20)
def test_moe_layer_backward_numerical():
    np.random.seed(42)
    batch_size = 2
    seq_len = 3
    embed_dim = 4
    num_experts = 4
    k = 2

    moe = MoELayer(embed_dim, num_experts, num_experts_per_token=k)
    x = np.random.randn(batch_size, seq_len, embed_dim)

    # Forward pass to populate cache
    output, cache = moe.forward(x)

    # Dummy gradient for loss w.r.t output
    grad_output = np.random.randn(batch_size, seq_len, embed_dim)

    # Analytical gradient
    # We need to pass the cache to backward
    grad_x_analytical, _ = moe.backward(x, grad_output, cache)

    # Numerical gradient
    # L = sum(output * grad_output)
    def get_loss(x_val):
        out, _ = moe.forward(x_val)
        return np.sum(out * grad_output)

    grad_x_numerical = np.zeros_like(x)
    eps = 1e-6
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        x[idx] = old_val + eps
        loss_plus = get_loss(x)

        x[idx] = old_val - eps
        loss_minus = get_loss(x)

        grad_x_numerical[idx] = (loss_plus - loss_minus) / (2 * eps)

        x[idx] = old_val
        it.iternext()

    np.testing.assert_allclose(
        grad_x_analytical, grad_x_numerical, rtol=1e-4, atol=1e-4
    )
