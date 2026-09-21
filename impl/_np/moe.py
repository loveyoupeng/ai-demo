"""Mixture of Experts feed-forward for the NumPy reference implementation.

A router + top-k SwiGLU experts, with the analytic backward.
"""

from __future__ import annotations

import numpy as np

from impl._np.ffn import SwiGLUFFN
from impl._np.init import xavier_uniform


class MixtureOfExperts:
    """Mixture of Experts (Shazeer et al. 2017; Switch/Mixtral style).

    Instead of one dense feed-forward, the block owns E feed-forward *experts*
    (here SwiGLU FFNs) and a small *router* that picks, per token, which
    experts process it.

        scores = x @ W_gate                  (B, S, E)   raw router logits
        probs  = softmax(scores)             (B, S, E)   over ALL experts
        mask   = keep only the top-k probs per token (rest → 0)
        weights = mask / sum(mask)           (B, S, E)   renormalized to sum 1
        out    = sum_j weights_j * expert_j(x)        (B, S, D)

    With ``n_shared_experts > 0`` (ADR 0002, DeepSeek-V2/V3 style) the block
    also has always-active, UNGATED shared experts whose outputs are averaged
    and added to the routed sum — the router never sees them:

        out = out_routed + (1/N_s) * sum_s E_shared_s(x)

    Note (reference implementation): like the dense path, this computes every
    expert's output and multiplies by its (possibly zero) weight. Real
    systems gather tokens per expert to skip the zero-weight compute; the
    math is identical. The point here is the *routing* math.

    Weights: router W_gate: (D, E); each expert: SwiGLU (D, FF) projections.

    Backward
    --------
    out = Σ_j w_j(x) · E_j(x),  where w = top-k-renormalized softmax(x @ W_gate)
    and E_j(x) is expert j's SwiGLU output.

    1. Per-expert upstream: w_j multiplies the whole expert output, so the
       gradient arriving at expert j is w_j ⊙ dout (broadcast over D).
       For the router we also need each expert's *output energy*
           g_j = sum_d (dout ⊙ E_j(x))_d     (B, S) per expert,
       which measures how much the loss "wants" expert j's output to move.
    2. Router gradient. With p = softmax(scores), support A (the selected
       experts, fixed under small perturbations) and S = sum_{j in A} p_j:
           w_j = p_j / S   (j in A),   w_j = 0  (j not in A)
       The chain rule (softmax Jacobian + renormalization Jacobian) gives,
       for k in A:
           dscores_k = c_k - p_k * A1 / S**2,
       with c_k = g_k p_k / S and A1 = sum_{j in A} g_j p_j (the two
       Jacobian terms combine: 1/S + (1-S)/S^2 = 1/S^2); and dscores_k = 0
       for k not in A (the weight w_k is exactly 0 and stays 0 a.e.).
    3. dW_gate = x^T @ dscores  (D, E).
    4. dx gets contributions from the router input gradient (dscores @
       W_gate) plus each expert's input gradient, weighted by w_j.
    """

    def __init__(
        self, embed_dim: int, n_experts: int, ff_dim: int, top_k: int, seed: int = 0, n_shared_experts: int = 0
    ) -> None:
        self.embed_dim = embed_dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.n_shared_experts = n_shared_experts
        rng = np.random.default_rng(seed)
        # Router: (D, E) — no bias (Mixtral convention)
        self.gate = xavier_uniform(rng, embed_dim, n_experts)
        # Experts: one SwiGLU FFN each, distinct seeds per expert
        self.experts = [SwiGLUFFN(embed_dim, ff_dim, seed=seed + 1 + j) for j in range(n_experts)]
        # Shared experts (ADR 0002): always active, ungated — every token gets
        # their output added to the routed sum (DeepSeek-V2/V3 style).
        self.shared_experts = [SwiGLUFFN(embed_dim, ff_dim, seed=seed + 50 + s) for s in range(n_shared_experts)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """MoE forward. x: (B, S, D) → out: (B, S, D)."""
        out, _state = self._forward_state(x)
        return out

    def _forward_state(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """MoE forward + the router state the record and the backward need.

        Returns (out, state) where state holds:
            scores      : (B, S, E) stable-softmax input (after the max subtract)
            probs       : (B, S, E) after the top-k mask (raw softmax if top_k == E)
            topk_idx    : (B, S, k) the selected expert per token
            weights     : (B, S, E) renormalized routing weights
            expert_outs : E × (B, S, D) all expert outputs (intentionally full)
        """
        E = self.n_experts

        # Router scores and softmax over all experts: (B, S, E)
        scores = x @ self.gate  # (B, S, E)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)  # (B, S, E)

        # Top-k selection (tie-safe: the threshold is the k-th largest value.)
        if self.top_k < E:
            order = np.argsort(probs, axis=-1)[:, :, ::-1]  # (B, S, E) descending
            topk_idx = order[:, :, : self.top_k]  # (B, S, k)
            kth_idx = order[:, :, self.top_k - 1 : self.top_k]  # (B, S, 1)
            threshold = np.take_along_axis(probs, kth_idx, axis=-1)  # (B, S, 1)
            probs = np.where(probs >= threshold, probs, 0.0)  # (B, S, E)
            weights = probs / np.maximum(np.sum(probs, axis=-1, keepdims=True), 1e-8)  # (B, S, E)
        else:
            topk_idx = np.argsort(probs, axis=-1)[:, :, ::-1][:, :, : self.top_k]  # (B, S, k)
            weights = probs  # (B, S, E)

        # Weighted sum of expert outputs (all experts computed; zeros masked).
        # PROD: production gathers tokens per expert and runs only the selected top-k.
        expert_outs = [expert.forward(x) for expert in self.experts]  # E × (B, S, D)
        out = np.zeros_like(x)  # (B, S, D)
        for expert_idx, e_out in enumerate(expert_outs):
            w = weights[:, :, expert_idx : expert_idx + 1]  # (B, S, 1)
            out = out + w * e_out  # (B, S, D)

        # Shared experts (ADR 0002): ungated, always active. Averaged so the
        # branch magnitude does not grow with n_shared_experts.
        if self.shared_experts:
            shared_outs = [se.forward(x) for se in self.shared_experts]  # n_shared × (B, S, D)
            shared_sum = np.sum(shared_outs, axis=0)  # (B, S, D)
            out = out + shared_sum / len(shared_outs)
        else:
            shared_outs = []
        state = {
            "scores": scores,
            "probs": probs,
            "topk_idx": topk_idx,
            "weights": weights,
            "expert_outs": expert_outs,
            "shared_outs": shared_outs,
        }
        return out, state

    def backward(self, dout: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """Analytic backward.

        dout: (B, S, D) upstream gradient.
        x: (B, S, D) the forward input (recomputes router + expert outputs).

        Returns: (dx, dparams) where dparams has
        {"gate": (D, E), "experts": [ {"gate_proj", "up_proj", "down_proj"}, ... ]}.
        """
        E = self.n_experts
        x_flat = x.reshape(-1, x.shape[-1])  # (T, D)

        # --- Recompute router internals (same math as forward) ---
        scores = x @ self.gate  # (B, S, E)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)  # (B, S, E)
        mask = np.ones_like(probs, dtype=bool)  # support A (default: all experts)
        if self.top_k < E:
            order = np.argsort(probs, axis=-1)[:, :, ::-1]  # (B, S, E) descending
            kth_idx = order[:, :, self.top_k - 1 : self.top_k]  # (B, S, 1)
            threshold = np.take_along_axis(probs, kth_idx, axis=-1)  # (B, S, 1)
            mask = probs >= threshold  # (B, S, E) — same tie rule as forward
        S = np.sum(probs * mask, axis=-1, keepdims=True)  # (B, S, 1) pre-renorm support sum
        weights = probs * mask / np.maximum(S, 1e-8)  # (B, S, E) the w_j used in forward

        dx = np.zeros_like(x)  # (B, S, D) — accumulates expert + router contributions
        energies = np.empty((x.shape[0], x.shape[1], E), dtype=np.float64)
        expert_grads: list[dict[str, np.ndarray]] = []
        for j, expert in enumerate(self.experts):
            e_out = expert.forward(x)  # (B, S, D)
            energies[:, :, j] = (dout * e_out).sum(axis=-1).astype(np.float64)  # (B, S)
            # Expert j's upstream is w_j ⊙ dout; its input gradient already carries w_j.
            d_e_in, e_grads = expert.backward(weights[:, :, j : j + 1] * dout, x)  # (B, S, D)
            dx = dx + d_e_in  # (B, S, D) — already carries the w_j factor
            expert_grads.append(e_grads)

        # --- Router backward (docstring formula) ---
        # The Jacobian vanishes outside the selected support A: w_j is exactly 0
        # (and stays 0 under small perturbations) for j not in A.
        c = np.where(mask, (energies * probs / np.maximum(S, 1e-8)), 0.0)  # (B, S, E)
        a1 = np.sum(energies * probs * mask, axis=-1, keepdims=True)  # (B, S, 1)
        dscores = (c - probs * (a1 / np.maximum(S**2, 1e-16))) * mask  # (B, S, E)
        dx = dx + dscores @ self.gate.T  # (B, S, D)
        dW_gate = x_flat.T @ dscores.reshape(-1, E)  # (D, E)

        # --- Shared experts (ADR 0002) ---
        # out += mean_s(E_shared_s(x)) is a plain sum of independent branches:
        # each shared expert sees upstream dout / n_shared, no router Jacobian.
        shared_grads: list[dict[str, np.ndarray]] = []
        if self.shared_experts:
            n_sh = len(self.shared_experts)
            for se in self.shared_experts:
                d_se_in, se_grads = se.backward(dout / n_sh, x)  # (B, S, D)
                dx = dx + d_se_in
                shared_grads.append(se_grads)

        return dx, {
            "gate": dW_gate.astype(np.float32),
            "experts": expert_grads,
            "shared_experts": shared_grads,
        }
