"""AdamW optimizer with bias correction and decoupled weight decay.

AdamW (Adam with decoupled weight decay) combines the Adam optimizer's
adaptive learning rates with L2 regularization applied directly to
parameters (decoupled from the gradient).

Mathematical Background
-----------------------
The update rule for each parameter is:

    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t        # first moment estimate
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2       # second moment estimate
    m_hat = m_t / (1 - beta1^t)                        # bias-corrected first moment
    v_hat = v_t / (1 - beta2^t)                        # bias-corrected second moment
    theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta_{t-1})

Where:
    m, v     : running moment estimates (dimension = shape of parameter)
    g_t      : gradient at step t (dimension = shape of parameter)
    theta    : parameter tensor
    lr       : learning rate (scalar)
    beta1    : exponential decay rate for first moment (scalar, ~0.9)
    beta2    : exponential decay rate for second moment (scalar, ~0.999)
    eps      : numerical stability term (scalar, ~1e-8)

Bias correction is necessary because m_0 = v_0 = 0 initializes both moments
at zero, causing a downward bias in early steps. The correction terms
(1 - beta^t) scale up the estimates to compensate.
"""

from __future__ import annotations

import numpy as np


class AdamW:
    """AdamW optimizer with bias correction and decoupled weight decay.

    Parameters
    ----------
    lr : float, default 3e-4
        Learning rate.
    beta1 : float, default 0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, default 0.999
        Exponential decay rate for the second moment estimates.
    eps : float, default 1e-8
        Term added for numerical stability.
    weight_decay : float, default 0.0
        Weight decay coefficient (L2 regularization, decoupled).

    Usage
    -----
    >>> optimizer = AdamW(lr=1e-3, weight_decay=0.01)
    >>> optimizer.step(params_dict, grads_dict)

    """

    def __init__(
        self,
        lr: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        # Internal state: first and second moment estimates per parameter
        # m[name] : first moment estimate (same shape as params[name])
        # v[name] : second moment estimate (same shape as params[name])
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

        # Step counter (shared across all parameters)
        self._count: int = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """Perform one optimization step.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            Model parameters to update (modified in-place).
            Each key maps to a numpy array of arbitrary shape.
        grads : dict[str, np.ndarray]
            Gradients for each parameter (must have same keys as params).
            Each key maps to a numpy array of the same shape as the
            corresponding parameter.

        """
        self._count += 1  # increment step counter before computing bias correction

        # Pre-compute bias correction denominators once per step
        # (1 - beta1^t) and (1 - beta2^t) — same for all parameters
        # These values approach 1.0 as t grows, reducing correction over time
        bias_correction_1 = 1.0 - self.beta1**self._count  # scalar
        bias_correction_2 = 1.0 - self.beta2**self._count  # scalar

        for name, grad in grads.items():
            param = params[name]

            # Initialize moment estimates on first visit
            if name not in self.m:
                # m[name] and v[name] start at zero arrays of same shape as param
                # Shape: param.shape — e.g. (embed_dim,) for embeddings,
                #        (n_layers, n_heads, head_dim) for attention weights, etc.
                self.m[name] = np.zeros_like(param, dtype=np.float64)
                self.v[name] = np.zeros_like(param, dtype=np.float64)

            # Step 1: Update biased first moment estimate
            # m = beta1 * m + (1 - beta1) * g
            # Shape: param.shape (element-wise update)
            # This tracks the exponential moving average of gradients
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad

            # Step 2: Update biased second moment estimate
            # v = beta2 * v + (1 - beta2) * g^2
            # Shape: param.shape (element-wise, squares each gradient element)
            # This tracks the exponential moving average of squared gradients
            # (i.e., the uncentered variance of gradients)
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * grad**2

            # Step 3: Compute bias-corrected moment estimates
            # Since m_0 = v_0 = 0, early estimates are biased toward zero.
            # Dividing by (1 - beta^t) corrects for this initialization bias.
            # As t → infinity, (1 - beta^t) → 1, so correction fades.
            m_hat = self.m[name] / bias_correction_1  # shape: param.shape
            v_hat = self.v[name] / bias_correction_2  # shape: param.shape

            # Step 4: Parameter update with Adam + decoupled weight decay
            #   theta = theta - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta)
            #
            # The Adam term: m_hat / (sqrt(v_hat) + eps) is an adaptive step size
            #   per parameter — large for slowly changing params, small for volatile ones.
            # The weight decay term: weight_decay * theta applies L2 regularization
            #   directly to parameters (decoupled, unlike standard Adam's L2).
            #
            # Both terms are scaled by lr (shared learning rate).
            # Shape: param.shape — element-wise update.
            adam_step = m_hat / (np.sqrt(v_hat) + self.eps)  # adaptive gradient step
            decay_step = self.weight_decay * param  # L2 weight decay

            params[name] -= self.lr * (adam_step + decay_step)
