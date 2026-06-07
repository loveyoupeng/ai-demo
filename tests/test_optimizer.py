from __future__ import annotations

import numpy as np
from optimizer import Adam, SGD


def test_sgd_update():
    """
    Verifies SGD update rule: param = param - lr * grad
    """
    np.random.seed(42)
    lr = 0.1
    optimizer = SGD(learning_rate=lr)

    params = {"w": np.array([1.0, 2.0], dtype=np.float64)}
    grads = {"w": np.array([0.5, -0.5], dtype=np.float64)}

    optimizer.step(params, grads)

    np.testing.assert_allclose(params["w"], np.array([0.95, 2.05]))


def test_adam_update():
    """
    Verifies Adam update rule for the first step (t=1).
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    m_hat = m_t / (1 - beta1^t)
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    v_hat = v_t / (1 - beta2^t)
    param = param - lr * m_hat / (sqrt(v_hat) + eps)
    """
    np.random.seed(42)
    lr = 0.1
    beta1 = 0.9
    beta2 = 0.99
    eps = 1e-8
    optimizer = Adam(learning_rate=lr, beta1=beta1, beta2=beta2, eps=eps)

    params = {"w": np.array([1.0], dtype=np.float64)}
    grads = {"w": np.array([0.5], dtype=np.float64)}

    # Step 1
    optimizer.step(params, grads)

    # t=1
    # m = 0.9 * 0 + (1-0.9) * 0.5 = 0.05
    # m_hat = 0.05 / (1 - 0.9^1) = 0.05 / 0.1 = 0.5
    # v = 0.99 * 0 + (1-0.99) * 0.5^2 = 0.01 * 0.25 = 0.0025
    # v_hat = 0.0025 / (1 - 0.99^1) = 0.0025 / 0.01 = 0.25
    # update = 0.1 * 0.5 / (sqrt(0.25) + 1e-8) = 0.1 * 0.5 / 0.5 = 0.1
    # new_param = 1.0 - 0.1 = 0.9

    np.testing.assert_allclose(params["w"], np.array([0.9]))


def test_adam_momentum_consistency():
    """
    Verifies that Adam maintains state (m and v) across multiple steps.
    """
    np.random.seed(42)
    optimizer = Adam(learning_rate=0.1)
    params = {"w": np.array([1.0], dtype=np.float64)}
    grads = {"w": np.array([0.5], dtype=np.float64)}

    optimizer.step(params, grads)
    assert "w" in optimizer.m
    assert "w" in optimizer.v
    assert optimizer.t == 1

    optimizer.step(params, grads)
    assert optimizer.t == 2


def test_adam_new_parameters():
    """
    Verifies that Adam can handle new parameters introduced after the first step.
    This should fail currently due to the initialization bug.
    """
    np.random.seed(42)
    optimizer = Adam(learning_rate=0.1)

    # First step with param 'w'
    params1 = {"w": np.array([1.0], dtype=np.float64)}
    grads1 = {"w": np.array([0.5], dtype=np.float64)}
    optimizer.step(params1, grads1)

    # Second step with new param 'b'
    params2 = {
        "w": np.array([0.9], dtype=np.float64),
        "b": np.array([2.0], dtype=np.float64),
    }
    grads2 = {
        "w": np.array([0.1], dtype=np.float64),
        "b": np.array([0.5], dtype=np.float64),
    }

    # This is expected to fail with KeyError: 'b'
    optimizer.step(params2, grads2)

    assert "b" in optimizer.m
    assert "b" in optimizer.v
