from __future__ import annotations

import numpy as np
import pytest
from model.layers import FeedForward


def numerical_gradient(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        x[idx] = old_val + eps
        pos = f(x).copy()

        x[idx] = old_val - eps
        neg = f(x).copy()

        grad[idx] = np.sum((pos - neg) / (2 * eps))
        x[idx] = old_val
        it.iternext()
    return grad


def test_feedforward_gradients():
    embed_dim = 4
    dim_ff = 8
    batch_size = 2
    seq_len = 3

    ff = FeedForward(embed_dim, dim_ff)
    x = np.random.randn(batch_size, seq_len, embed_dim)

    # Forward
    out = ff.forward(x)

    # Analytical gradient
    grad_out = np.random.randn(*out.shape)
    grad_x_analytical = ff.backward(grad_out)

    # Check grad_W1, grad_b1, grad_W2, grad_b2
    # We need a function that takes weights and returns output for a fixed x
    def f_W1(W1):
        # Use the same h logic as forward
        h = np.dot(x, W1) + ff.b1
        h = np.maximum(0, h)
        return np.dot(h, ff.W2) + ff.b2

    def f_W2(W2):
        h = np.dot(x, ff.W1) + ff.b1
        h = np.maximum(0, h)
        return np.dot(h, W2) + ff.b2

    # Check W1 gradient
    def loss_W1(W1):
        return np.sum(f_W1(W1) * grad_out)

    grad_W1_num = numerical_gradient(lambda w: loss_W1(w), ff.W1)
    np.testing.assert_allclose(ff.grad_W1, grad_W1_num, rtol=1e-5, atol=1e-5)

    # Check W2 gradient
    def loss_W2(W2):
        return np.sum(f_W2(W2) * grad_out)

    grad_W2_num = numerical_gradient(lambda w: loss_W2(w), ff.W2)
    np.testing.assert_allclose(ff.grad_W2, grad_W2_num, rtol=1e-5, atol=1e-5)

    # Check x gradient
    def loss_x(x_in):
        h = np.dot(x_in, ff.W1) + ff.b1
        h = np.maximum(0, h)
        return np.sum(
            np.dot(h, ff.W2) + ff.b2 * grad_out
        )  # Wait, loss is sum(out * grad_out)

    # Correct loss function for grad_x:
    # We want grad of L = sum(out * grad_out) with respect to x
    def loss_x_correct(x_in):
        h = np.dot(x_in, ff.W1) + ff.b1
        h = np.maximum(0, h)
        out = np.dot(h, ff.W2) + ff.b2
        return np.sum(out * grad_out)

    grad_x_num = numerical_gradient(loss_x_correct, x)
    np.testing.assert_allclose(grad_x_analytical, grad_x_num, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
