"""RMSNorm for the NumPy reference implementation.

Root Mean Square Layer Normalization (Zhang & Sennrich, 2019) with the
analytic backward.
"""

from __future__ import annotations

import numpy as np


class RMSNorm:
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Formula (applied over the last dimension of x):

        out = x / sqrt(mean(x^2) + eps) * gamma

    Unlike LayerNorm there is no mean-centering (no beta bias); each row is
    scaled so that its root-mean-square is ~1, then scaled per-dimension by the
    learned gain ``gamma``. It is cheaper than LayerNorm (one mean, no
    variance) and is what LLaMA uses.

    Forward
    -------
    x : (B, S, D)  (any leading batch dims work: (..., D))
    Returns: (..., D), same shape as x.

    Step shapes:
        x**2                 (..., D)
        mean(x**2, axis=-1)  (..., 1)   — keepdims so it broadcasts back
        rms                  (..., 1)
        x / rms              (..., D)
        * gamma              (..., D)   — gamma is (D,), broadcast over batch

    Weight: gamma, shape (D,), initialized to ones (identity at start).

    Backward
    --------
    Write x_hat = x / rms (the normalized values, (..., D)) and
    d_x_hat = dout * gamma (upstream scaled by the learned gain, (..., D)).

    Because rms depends on *every* element of x, the chain rule adds a
    correction so that the result stays on the same "scale-1" surface — the
    same structure as the LayerNorm backward:

        dx = (1 / rms) * ( d_x_hat - x_hat * mean(d_x_hat * x_hat, axis=-1) )
        d_gamma = sum over all batch/seq positions of ( dout * x_hat )

    Intuition: the first term passes the gradient through as-is; the second
    removes the component that would change the row's RMS (the Jacobian of
    x/rms is a scaled projection, and projections have this "deflate the
    radial component" form).
    """

    def __init__(self, embed_dim: int, eps: float = 1e-6, seed: int = 0) -> None:
        self.eps = eps
        self.gamma: np.ndarray = np.ones(embed_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply RMSNorm. x: (..., D) → out: (..., D)."""
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)  # (..., 1)
        return (x / rms) * self.gamma  # (..., D)

    def backward(self, dout: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Analytic backward.

        dout: (..., D) upstream gradient.
        x: (..., D) the forward input (recomputes rms — no cached state).

        Returns: (dx, d_gamma) with dx shaped like x and d_gamma shaped (D,).
        """
        # Recompute the intermediates (cheap; keeps backward stateless).
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)  # (..., 1)
        x_hat = x / rms  # (..., D) normalized values

        # d(out)/d(gamma): out = x_hat * gamma, so the row-wise product of the
        # upstream gradient with the normalized values, summed over batch/seq.
        # (dout * x_hat): (..., D) → sum over all leading dims → (D,)
        d_gamma = np.sum(dout * x_hat, axis=tuple(range(dout.ndim - 1)))  # (D,)

        # d(out)/d(x): chain rule with the rms dependency (see docstring).
        d_x_hat = dout * self.gamma  # (..., D)
        mean_term = np.mean(d_x_hat * x_hat, axis=-1, keepdims=True)  # (..., 1)
        dx = (1.0 / rms) * (d_x_hat - x_hat * mean_term)  # (..., D)
        return dx, d_gamma
