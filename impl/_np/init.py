"""Weight initialization for the NumPy reference implementation.

The NumPy track initializes every weight matrix with Xavier (Glorot) uniform
so that each layer starts with comparable input/output variance — the
standard starting point for a network before training.
"""

from __future__ import annotations

import numpy as np


def xavier_uniform(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    """Xavier (Glorot) uniform init: U(-limit, limit), limit = sqrt(6/(fan_in+fan_out)).

    Keeps the variance of the output comparable to the variance of the input
    when the input is zero-mean, which stabilizes training.

    Returns: array of shape (fan_in, fan_out), dtype float32.
    """
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)
