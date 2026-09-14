"""SwiGLU feed-forward network for the NumPy reference implementation.

The dense feed-forward layer with its analytic backward.
"""

from __future__ import annotations

import numpy as np

from impl._np.init import xavier_uniform


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish activation: x * sigmoid(x) (Elfwing et al., 2017).

    SiLU is a smooth, unbounded gate: it behaves like ReLU for x >> 0 and
    decays toward 0 (with a small negative dip) for x << 0.

    x: (..., FF) → out: (..., FF)
    """
    return x / (1.0 + np.exp(-x))  # (..., FF)


def silu_backward(dout: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Derivative of SiLU: silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))).

    dout: (..., FF) upstream gradient.
    x: (..., FF) the pre-activation input.

    Returns: dx (..., FF).
    """
    sigma = 1.0 / (1.0 + np.exp(-x))  # (..., FF)
    return dout * sigma * (1.0 + x * (1.0 - sigma))  # (..., FF)


class SwiGLUFFN:
    """SwiGLU feed-forward network (GLU: Dauphin et al. 2016; SwiGLU: Shazeer 2020).

    A gated two-projection design: one projection produces a "gate" that is
    smoothed by SiLU (x * sigmoid(x)) and multiplied element-wise with a
    second ("up") projection; a third projection maps back to the model width.

        gate = SiLU(x @ W_gate)     (..., FF)
        up   = x @ W_up             (..., FF)
        out  = (gate * up) @ W_down (..., D)

    Weights: W_gate, W_up: (D, FF)   W_down: (FF, D).  FF is typically 4D.

    Backward (chain rule, reverse order)
    ------------------------------------
    Given dout (..., D):

        dW_down  = gated^T @ dout          (D, FF) → (FF, D)
        dgated   = dout @ W_down^T         (..., FF)
        dup      = dgated * gate           (element-wise: out = gate * up)
        dgate    = dgated * up
        dpre     = silu_backward(dgate, pre_gate)   where pre_gate = x @ W_gate
        dW_gate  = x^T @ dpre              (D, FF)
        dW_up    = x^T @ dup               (D, FF)
        dx       = dpre @ W_gate^T + dup @ W_up^T   (..., D)

    x: (..., D) — flattened to (T, D) with T = B*S for the matrix products.
    """

    def __init__(self, embed_dim: int, ff_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.gate_proj = xavier_uniform(rng, embed_dim, ff_dim)  # (D, FF)
        self.up_proj = xavier_uniform(rng, embed_dim, ff_dim)  # (D, FF)
        self.down_proj = xavier_uniform(rng, ff_dim, embed_dim)  # (FF, D)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """SwiGLU forward. x: (..., D) → out: (..., D)."""
        gate = x @ self.gate_proj  # (..., FF)
        # SiLU: x / (1 + exp(-x)) — a smooth, unbounded gate
        gate = silu(gate)  # (..., FF)
        up = x @ self.up_proj  # (..., FF)
        gated = gate * up  # (..., FF) element-wise gating
        return gated @ self.down_proj  # (..., D)

    def backward(self, dout: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Analytic backward.

        dout: (..., D) upstream gradient.
        x: (..., D) the forward input (recomputes the intermediates).

        Returns: (dx, dparams) where dparams maps local names
        {"gate_proj", "up_proj", "down_proj"} → gradient arrays.
        """
        pre_gate = x @ self.gate_proj  # (..., FF) pre-activation gate logits
        gate = silu(pre_gate)  # (..., FF)
        up = x @ self.up_proj  # (..., FF)
        gated = gate * up  # (..., FF)

        # dW_down: out = gated @ W_down → dW_down = gated^T @ dout
        # (T, FF)^T @ (T, D) → (FF, D)
        dW_down = gated.reshape(-1, gated.shape[-1]).T @ dout.reshape(-1, dout.shape[-1])  # (FF, D)

        # Back through the gating element-wise product.
        dgated = dout @ self.down_proj.T  # (..., FF)
        dup = dgated * gate  # (..., FF)
        dgate = dgated * up  # (..., FF)

        # Back through SiLU.
        dpre = silu_backward(dgate, pre_gate)  # (..., FF)

        # Linear-layer backwards: dW = x^T @ d(out), dx = d(out) @ W^T.
        x_flat = x.reshape(-1, x.shape[-1])  # (T, D)
        dW_gate = x_flat.T @ dpre.reshape(-1, dpre.shape[-1])  # (D, FF)
        dW_up = x_flat.T @ dup.reshape(-1, dup.shape[-1])  # (D, FF)
        dx = (
            dpre.reshape(-1, dpre.shape[-1]) @ self.gate_proj.T + dup.reshape(-1, dup.shape[-1]) @ self.up_proj.T
        )  # (T, D)

        return dx.reshape(x.shape), {
            "gate_proj": dW_gate,
            "up_proj": dW_up,
            "down_proj": dW_down,
        }
