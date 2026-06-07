from __future__ import annotations

from typing import cast

import numpy as np


class Router:
    r"""
    The Routing / Gating network (NumPy).

    Mathematical context:
    The router computes logits $z = X W_{router}$ where
    $W_{router} \in \mathbb{R}^{D \times N}$ and $N$ is the number of
    experts.  These logits are passed through a softmax to produce routing
    probabilities $P_{b,s,i} = \text{softmax}(z_{b,s,i})$.

    Dimension tracking:

    - Input :math:`X` — :math:`[B, L, D]` (Batch, Seq\_Len, Embed\_Dim)
    - Weights :math:`W_{router}` — :math:`[D, N]` (Embed\_Dim, Num\_Experts)
    - Logits :math:`z` — :math:`[B, L, N]`
    - Routing probabilities :math:`P` — :math:`[B, L, N]`
    """

    def __init__(self, embed_dim: int, num_experts: int) -> None:
        self.embed_dim: int = embed_dim
        self.num_experts: int = num_experts
        # routing weights [D, N]
        self.w: np.ndarray = np.random.randn(embed_dim, num_experts) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input tensor :math:`[B, L, D]`.
        Returns:
            Routing probabilities :math:`[B, L, N]`.
        """
        logits: np.ndarray = np.dot(x, self.w)  # [B, L, N]
        self.last_routing_weights: np.ndarray = self._softmax(logits, axis=-1)
        return self.last_routing_weights

    @staticmethod
    def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
        """Numerically stable softmax along *axis*."""
        e_x: np.ndarray = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def backward(
        self, x: np.ndarray, d_probs: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Softmax backward pass.

        Args:
            x          : Input tensor :math:`[B, L, D]`.
            d_probs    : Gradient w.r.t. output probabilities :math:`[B, L, N]`.

        Returns:
            dx           : Gradient w.r.t. input :math:`[B, L, D]`.
            grads        : dict with ``"w"``  — :math:`[D, N]`.
        """
        w: np.ndarray = self.last_routing_weights  # [B, L, N]

        # Softmax derivative:  d_logits = w * (d_probs - w * sum(d_probs, axis=-1, keepdims=True))
        prod: np.ndarray = d_probs * w  # [B, L, N]
        term2: np.ndarray = np.sum(prod, axis=-1, keepdims=True)  # [B, L, 1]
        d_logits: np.ndarray = w * (d_probs - term2)  # [B, L, N]

        # dW = x^T @ d_logits   →   [D, N]
        d_weights: np.ndarray = np.dot(
            x.reshape(-1, self.embed_dim).T,  # [D, B*L]
            d_logits.reshape(-1, self.num_experts),  # [B*L, N]
        )

        # dx = d_logits @ W^T   →   [B, L, D]
        dx: np.ndarray = np.dot(d_logits, self.w.T)

        grads: dict[str, np.ndarray] = {"w": d_weights}
        return dx, grads

    def get_params(self) -> dict[str, np.ndarray]:
        """Return all learnable parameters."""
        return {"w": self.w}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        """Assign parameters from a dict produced by :meth:`get_params`."""
        for name, param in params.items():
            if name == "w":
                self.w = param.copy()


class Expert:
    r"""
    A single feed-forward expert (NumPy).

    Each expert is a two-layer MLP with ReLU:

    .. math::
        h = \text{ReLU}(x W_1 + b_1) \\
        y = h W_2 + b_2

    Dimension tracking:

    - Input  :math:`x` — :math:`[B, L, D]`
    - Hidden :math:`z_1, h` — :math:`[B, L, D_{ff}]`
    - Output :math:`y` — :math:`[B, L, D]`
    """

    def __init__(self, embed_dim: int, dim_ff: int) -> None:
        self.embed_dim: int = embed_dim
        self.dim_ff: int = dim_ff

        # learnable parameters
        # W1: [D, D_ff]   b1: [D_ff]
        self.w1: np.ndarray = np.random.randn(embed_dim, dim_ff) * 0.01
        self.b1: np.ndarray = np.zeros(dim_ff)
        # W2: [D_ff, D]   b2: [D]
        self.w2: np.ndarray = np.random.randn(dim_ff, embed_dim) * 0.01
        self.b2: np.ndarray = np.zeros(embed_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input tensor :math:`[B, L, D]`.
        Returns:
            Output :math:`[B, L, D]`.
        """
        self.x: np.ndarray = x
        # Linear 1  [B, L, D_ff]
        self.z1: np.ndarray = np.dot(x, self.w1) + self.b1
        # ReLU
        self.h: np.ndarray = np.maximum(0, self.z1)
        # Linear 2  [B, L, D]
        output: np.ndarray = np.dot(self.h, self.w2) + self.b2
        return output

    def backward(
        self, x: np.ndarray, d_out: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Full FFN backward pass.

        Args:
            x       : Input tensor :math:`[B, L, D]`.
            d_out   : Gradient w.r.t. output :math:`[B, L, D]`.

        Returns:
            dx       : Gradient w.r.t. input :math:`[B, L, D]`.
            grads    : dict with ``"w1"[D,D_ff], "b1"[D_ff], "w2"[D_ff,D], "b2"[D]``.
        """
        batch_size, seq_len, embed_dim = x.shape

        # Flatten for matrix operations
        h_flat: np.ndarray = self.h.reshape(-1, self.dim_ff)  # [B*L, D_ff]
        d_out_flat: np.ndarray = d_out.reshape(-1, embed_dim)  # [B*L, D]

        # --- gradient w.r.t. w2, b2 ---
        grad_w2: np.ndarray = np.dot(h_flat.T, d_out_flat)  # [D_ff, D]
        grad_b2: np.ndarray = np.sum(d_out_flat, axis=0)  # [D]

        # --- gradient through W2 back to h ---
        grad_h: np.ndarray = np.dot(d_out_flat, self.w2.T)  # [B*L, D_ff]

        # --- ReLU backward ---
        z1_flat: np.ndarray = self.z1.reshape(-1, self.dim_ff)  # [B*L, D_ff]
        grad_z1: np.ndarray = grad_h * (z1_flat > 0).astype(np.float64)  # [B*L, D_ff]

        # --- gradient w.r.t. w1, b1 ---
        x_flat: np.ndarray = x.reshape(-1, self.embed_dim)  # [B*L, D]
        grad_w1: np.ndarray = np.dot(x_flat.T, grad_z1)  # [D, D_ff]
        grad_b1: np.ndarray = np.sum(grad_z1, axis=0)  # [D_ff]

        # --- gradient w.r.t. x ---
        dx: np.ndarray = np.dot(grad_z1, self.w1.T).reshape(
            batch_size, seq_len, self.embed_dim
        )

        grads: dict[str, np.ndarray] = {
            "w1": grad_w1,
            "b1": grad_b1,
            "w2": grad_w2,
            "b2": grad_b2,
        }
        return dx, grads

    def get_params(self) -> dict[str, np.ndarray]:
        """Return all learnable parameters."""
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        """Assign parameters from a dict produced by :meth:`get_params`."""
        for name, param in params.items():
            if hasattr(self, name):
                setattr(self, name, param.copy())


class MoELayer:
    r"""
    Mixture-of-Experts layer (NumPy).

    For each token :math:`x_{b,s}`, the router selects the top-*k* experts
    and the output is a weighted sum:

    .. math::
        y_{b,s} = \sum_{j \in \text{top}_k} \tilde{P}_{b,s,j} \cdot
                  \text{Expert}_j(x_{b,s})

    where :math:`\tilde{P}` are the **normalised** top-*k* routing weights.

    Dimension tracking:

    - Input :math:`x` — :math:`[B, L, D]`
    - Top-*k* indices :math:`[B, L, K]`
    - Top-*k* weights :math:`[B, L, K]`  (normalised)
    - All experts' outputs :math:`[N, B, L, D]`
    - Combined output :math:`[B, L, D]`
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        dim_ff: int = 128,
        num_experts_per_token: int = 2,
    ) -> None:
        self.embed_dim: int = embed_dim
        self.num_experts: int = num_experts
        self.k: int = min(num_experts_per_token, num_experts)

        self.router: Router = Router(embed_dim, num_experts)
        self.experts: list[Expert] = [
            Expert(embed_dim, dim_ff) for _ in range(num_experts)
        ]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        """
        MoE forward pass.

        Args:
            x: Input tensor :math:`[B, L, D]`.

        Returns:
            combined_output :math:`[B, L, D]`
            cache: dict with ``routing_weights``, ``top_k_indices``,
                   ``top_k_weights``, ``all_expert_outputs``.
        """
        batch_size, seq_len, _ = x.shape

        # 1. Router probabilities  [B, L, N]
        routing_weights: np.ndarray = self.router.forward(x)

        # 2. Top-k indices   [B, L, K]
        top_k_indices: np.ndarray = np.argsort(routing_weights, axis=-1)[..., -self.k :]
        # 3. Unnormalised top-k weights   [B, L, K]
        top_k_raw: np.ndarray = np.take_along_axis(
            routing_weights, top_k_indices, axis=-1
        )
        # 4. Normalise   [B, L, K]
        top_k_sum: np.ndarray = np.sum(top_k_raw, axis=-1, keepdims=True) + 1e-8
        top_k_weights: np.ndarray = top_k_raw / top_k_sum

        # 5. All expert outputs  [N, B, L, D]
        all_expert_outputs: np.ndarray = np.array(
            [exp.forward(x) for exp in self.experts]
        )

        # 6. Weighted combination  [B, L, D]
        #    For each position (b, s):  out +=  top_k_weights[b,s,:] @ expert_outputs[top_k_indices[b,s,:]]
        batch_idx = np.arange(batch_size)[:, np.newaxis, np.newaxis]  # [B,1,1]
        seq_idx = np.arange(seq_len)[np.newaxis, :, np.newaxis]  # [1,L,1]
        combined_output: np.ndarray = (
            top_k_weights[..., np.newaxis]
            * all_expert_outputs[top_k_indices, batch_idx, seq_idx]
        ).sum(axis=2)  # [B, L, D]

        cache: dict[str, object] = {
            "x": x,
            "routing_weights": routing_weights,
            "top_k_indices": top_k_indices,
            "top_k_weights": top_k_weights,
            "top_k_sum": top_k_sum,
            "all_expert_outputs": all_expert_outputs,
        }

        return combined_output, cache

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def backward(
        self, x: np.ndarray, d_out: np.ndarray, cache: dict[str, object]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Full MoE backward pass.

        Args:
            x       : Input :math:`[B, L, D]`.
            d_out   : Gradient w.r.t. combined output :math:`[B, L, D]`.
            cache   : Cache dict from forward.

        Returns:
            dx       : Gradient w.r.t. input :math:`[B, L, D]`.
            grads    : Flat parameter-gradient dict (router keys prefixed with ``"router."``).
        """
        top_k_indices: np.ndarray = cast(np.ndarray, cache["top_k_indices"])
        top_k_weights: np.ndarray = cast(np.ndarray, cache["top_k_weights"])
        top_k_sum: np.ndarray = cast(np.ndarray, cache["top_k_sum"])
        routing_weights: np.ndarray = cast(np.ndarray, cache["routing_weights"])
        all_expert_outputs: np.ndarray = cast(np.ndarray, cache["all_expert_outputs"])

        batch_size, seq_len, embed_dim = x.shape

        # ------------------------------------------------------------------
        # 1. Gradients w.r.t. all_expert_outputs
        #    d_all_expert_outputs[i, b, s, :] += top_k_weights[b, s, k] * d_out[b, s, :]
        #    for each (b, s, k) where top_k_indices[b, s, k] == i
        # ------------------------------------------------------------------
        d_all_expert_outputs: np.ndarray = np.zeros_like(
            all_expert_outputs
        )  # [N, B, L, D]
        d_top_k_weights: np.ndarray = np.zeros_like(top_k_weights)  # [B, L, K]

        # Use flat 2D views for consistent broadcasting
        b_flat = np.arange(batch_size)[:, np.newaxis]  # [B, 1]
        s_flat = np.arange(seq_len)[np.newaxis, :]  # [1, L]

        # d_all_expert_outputs[i, b, s, :] += weight * d_out[b, s, :]
        for k_idx in range(top_k_indices.shape[2]):
            expert_idx = top_k_indices[:, :, k_idx]  # [B, L]
            # d_out[b_flat, s_flat] -> [B, L, D]
            d_out_flat: np.ndarray = d_out[b_flat, s_flat]  # [B, L, D]
            # top_k_weights[:, :, k_idx, np.newaxis] -> [B, L, 1]
            weight_for_k: np.ndarray = top_k_weights[:, :, k_idx, np.newaxis]
            # w * d_out[bs, :] -> [B, L, D]
            grad_contrib: np.ndarray = weight_for_k * d_out_flat
            # Assign to all_expert_outputs[expert_idx, b, s, :]
            for b_i in range(batch_size):
                for s_i in range(seq_len):
                    d_all_expert_outputs[expert_idx[b_i, s_i], b_i, s_i, :] += (
                        grad_contrib[b_i, s_i, :]
                    )

        for k_idx in range(self.k):
            # all_expert_outputs[expert for k at b, s, :] -> [B, L, D]
            expert_for_k = top_k_indices[:, :, k_idx]
            expert_out_at_pos: np.ndarray = all_expert_outputs[
                expert_for_k, b_flat, s_flat
            ]  # [B, L, D]
            # d_top_k_weights[b, s, k] = sum_d top_k_weights[b,s,k] * d_out[b,s,d] * expert[b,s,d]
            d_top_k_weights[:, :, k_idx] = np.sum(
                top_k_weights[:, :, k_idx, np.newaxis]
                * d_out[b_flat, s_flat]
                * expert_out_at_pos,
                axis=-1,
            )

        # ------------------------------------------------------------------
        # 2. Back-propagate through each expert
        # ------------------------------------------------------------------
        d_x_from_experts: np.ndarray = np.zeros_like(x)  # [B, L, D]
        grads_experts: dict[str, np.ndarray] = {}

        for i in range(self.num_experts):
            mask: np.ndarray = top_k_indices == i  # [B, L, K]
            if np.any(mask):
                dx_i: np.ndarray
                grads_i: dict[str, np.ndarray]
                dx_i, grads_i = self.experts[i].backward(x, d_all_expert_outputs[i])
                d_x_from_experts += dx_i
                for name, grad in grads_i.items():
                    grads_experts[f"expert.{i}.{name}"] = grad
            else:
                for name in self.experts[i].get_params():
                    grads_experts[f"expert.{i}.{name}"] = np.zeros_like(
                        self.experts[i].get_params()[name]
                    )

        # ------------------------------------------------------------------
        # 3. Back-propagate through top-k normalisation
        #
        #    Let :math:`R` be the raw (unnormalised) top-k logits,
        #    :math:`S = \sum R`, and :math:`w = R / S`.
        #
        #    .. math::
        #        \frac{\partial w_k}{\partial R_j} = \frac{\delta_{jk} - w_k}{S}
        #
        #    So for the *selected* positions:
        #        dR_k = (d_w_k - \sum_m w_m \, d_w_m) / S
        #    For non-selected positions dR = 0.
        # ------------------------------------------------------------------
        d_top_k_sum: np.ndarray = -np.sum(
            top_k_weights * d_top_k_weights,
            axis=-1,
            keepdims=True,
        ) / (top_k_sum + 1e-8)  # [B, L, 1]

        d_top_k_raw: np.ndarray = (d_top_k_weights + d_top_k_sum) / (
            top_k_sum + 1e-8
        )  # [B, L, K]

        # Place d_raw into d_routing_weights only at top-k positions
        d_routing_weights: np.ndarray = np.zeros(
            (batch_size, seq_len, self.num_experts),
            dtype=np.float64,
        )
        for k_idx in range(self.k):
            expert_indices = top_k_indices[:, :, k_idx]  # [B, L]
            b_coords = np.arange(batch_size)[:, np.newaxis]  # [B, 1]
            s_coords = np.arange(seq_len)[np.newaxis, :]  # [1, L]
            # d_routing_weights shape is (B, L, N) -> index as (b, s, expert)
            d_routing_weights[b_coords, s_coords, expert_indices] += d_top_k_raw[
                :, :, k_idx
            ]
        # Zero out non-selected positions (keep only top-k gradients)
        selected_mask = np.zeros_like(routing_weights)  # [B, L, N]
        for k_idx in range(self.k):
            selected_mask[
                np.arange(batch_size)[:, np.newaxis],
                np.arange(seq_len)[np.newaxis, :],
                top_k_indices[:, :, k_idx],
            ] = 1.0
        d_routing_weights *= selected_mask

        # ------------------------------------------------------------------
        # 4. Router backward
        # ------------------------------------------------------------------
        dx_router: np.ndarray
        grads_router: dict[str, np.ndarray]
        dx_router, grads_router = self.router.backward(x, d_routing_weights)

        # ------------------------------------------------------------------
        # 5. Combine all input gradients
        # ------------------------------------------------------------------
        dx: np.ndarray = d_x_from_experts + dx_router

        # Prefix router grads
        combined_grads: dict[str, np.ndarray] = {}
        for name, grad in grads_router.items():
            combined_grads[f"router.{name}"] = grad
        combined_grads.update(grads_experts)

        return dx, combined_grads

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------
    def get_params(self) -> dict[str, np.ndarray]:
        """Return a flat dict of all learnable parameters."""
        params: dict[str, np.ndarray] = {}
        for name, param in self.router.get_params().items():
            params[f"router.{name}"] = param
        for i, expert in enumerate(self.experts):
            for name, param in expert.get_params().items():
                params[f"expert.{i}.{name}"] = param
        return params

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        """Assign parameters from a flat dict produced by :meth:`get_params`."""
        for name, param in params.items():
            if name.startswith("router."):
                key: str = name.replace("router.", "", 1)
                self.router.set_params({key: param})
            elif name.startswith("expert."):
                # e.g. "expert.2.w1" -> expert_idx=2, key="w1"
                parts: list[str] = name.split(".", 2)
                assert parts[0] == "expert" and parts[1].isdigit()
                idx: int = int(parts[1])
                key: str = parts[2]
                self.experts[idx].set_params({key: param})
