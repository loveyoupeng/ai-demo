"""Tests for AdamW Optimizer module."""

import numpy as np


class TestAdamW:
    """Tests for AdamW optimizer."""

    def test_zero_lr_no_update(self):
        """With lr=0, parameters do not change after step."""
        from impl._np.optimizer import AdamW

        params = {"w": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
        grads = {"w": np.array([0.1, 0.2, 0.3], dtype=np.float64)}

        optimizer = AdamW(lr=0.0)
        optimizer.step(params, grads)

        assert np.allclose(params["w"], np.array([1.0, 2.0, 3.0], dtype=np.float64))

    def test_basic_update(self):
        """With lr>0, parameters change by expected amount under constant gradients."""
        from impl._np.optimizer import AdamW

        params = {"w": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
        grads = {"w": np.array([0.1, 0.1, 0.1], dtype=np.float64)}

        optimizer = AdamW(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
        optimizer.step(params, grads)

        # After step 1:
        # m = (1-beta1)*g = 0.1*0.1 = 0.01    (first moment)
        # v = (1-beta2)*g^2 = 0.001*0.01 = 0.00001  (second moment)
        # m_hat = m / (1 - beta1^1) = 0.01 / 0.1 = 0.1
        # v_hat = v / (1 - beta2^1) = 0.00001 / 0.001 = 0.01
        # update = lr * (m_hat / sqrt(v_hat + eps)) = 0.1 * (0.1 / sqrt(0.01 + 1e-8))
        #        = 0.1 * (0.1 / 0.10000005) ≈ 0.09999995
        # expected_w = 1.0 - update ≈ 0.90000005
        expected_update = 0.1 * (0.1 / np.sqrt(0.01 + 1e-8))
        expected_w0 = 1.0 - expected_update
        expected_w1 = 2.0 - expected_update
        expected_w2 = 3.0 - expected_update

        assert np.isclose(params["w"][0], expected_w0, rtol=1e-4)
        assert np.isclose(params["w"][1], expected_w1, rtol=1e-4)
        assert np.isclose(params["w"][2], expected_w2, rtol=1e-4)

    def test_weight_decay(self):
        """Weight decay applies L2 regularization to each parameter."""
        from impl._np.optimizer import AdamW

        params = {"w": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
        grads = {"w": np.array([0.1, 0.1, 0.1], dtype=np.float64)}

        optimizer = AdamW(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)
        optimizer.step(params, grads)

        # After step 1 (weight_decay=0.01):
        # m = 0.01, v = 0.00001 (same as above)
        # m_hat = 0.1, v_hat = 0.01
        # update = lr * (m_hat / sqrt(v_hat + eps) + weight_decay * param)
        # For param[0]=1.0: update = 0.1 * (0.1/0.1 + 0.01*1.0) = 0.1 * (1.0 + 0.01) = 0.101
        # For param[1]=2.0: update = 0.1 * (1.0 + 0.02) = 0.102
        # For param[2]=3.0: update = 0.1 * (1.0 + 0.03) = 0.103
        expected_w0 = 1.0 - 0.101  # = 0.899
        expected_w1 = 2.0 - 0.102  # = 1.898
        expected_w2 = 3.0 - 0.103  # = 2.897

        assert np.isclose(params["w"][0], expected_w0, rtol=1e-4)
        assert np.isclose(params["w"][1], expected_w1, rtol=1e-4)
        assert np.isclose(params["w"][2], expected_w2, rtol=1e-4)

    def test_bias_correction_convergence(self):
        """After multiple steps with constant grad, effective update converges to lr."""
        from impl._np.optimizer import AdamW

        params = {"w": np.array([1.0, 2.0, 3.0], dtype=np.float64)}
        grads = {"w": np.array([0.1, 0.1, 0.1], dtype=np.float64)}

        n_steps = 100
        optimizer = AdamW(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)

        for _ in range(n_steps):
            optimizer.step(params, grads)

        # After many steps with constant grad=0.1:
        # m_hat -> 0.1, v_hat -> 0.01 (true gradient values)
        # m_hat / sqrt(v_hat + eps) = 0.1 / sqrt(0.01 + 1e-8) ≈ 0.9999995
        # So per-element update ≈ lr * 0.9999995 ≈ 0.09999995
        expected_per_element = 0.1 * (0.1 / np.sqrt(0.01 + 1e-8))

        # Each element should have been updated approximately n_steps * expected_per_element
        # from the same gradient (since the gradient does not change in our test)
        # After 100 steps, the parameters should have changed by ~100 * 0.1 = ~10 total
        # For element 0 (starting at 1.0): expected ≈ 0.000005
        # For element 1 (starting at 2.0): expected ≈ -0.999995
        # For element 2 (starting at 3.0): expected ≈ -1.999995
        expected_w0 = 1.0 - n_steps * expected_per_element
        expected_w1 = 2.0 - n_steps * expected_per_element
        expected_w2 = 3.0 - n_steps * expected_per_element

        assert np.isclose(params["w"][0], expected_w0, rtol=1e-3)
        assert np.isclose(params["w"][1], expected_w1, rtol=1e-3)
        assert np.isclose(params["w"][2], expected_w2, rtol=1e-3)
