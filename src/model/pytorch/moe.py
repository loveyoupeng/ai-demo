from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from src.core.registry import registry


class PyTorchRouter(nn.Module):
    r"""
    The Routing / Gating network (PyTorch).

    Computes logits z = X @ W_router and softmax to produce routing
    probabilities.  Parity-tested against NumPy Router.

    Dimensions:
    - Input :math:`X` — :math:`[B, L, D]`
    - Weights :math:`W` — :math:`[D, N]`
    - Routing probabilities — :math:`[B, L, N]`
    """

    def __init__(self, embed_dim: int, num_experts: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.w = nn.Parameter(torch.randn(embed_dim, num_experts) * 0.01)
        registry.register("pytorch", "router.w", "w")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor :math:`[B, L, D]`.
        Returns:
            Routing probabilities :math:`[B, L, N]`.
        """
        logits = torch.matmul(x, self.w)  # [B, L, N]
        self.last_routing_weights = torch.softmax(logits, dim=-1)
        return self.last_routing_weights

    def backward(
        self, x: torch.Tensor, d_probs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Softmax backward pass.

        Args:
            x       : Input tensor :math:`[B, L, D]`.
            d_probs : Gradient w.r.t. output probabilities :math:`[B, L, N]`.

        Returns:
            dx      : Gradient w.r.t. input :math:`[B, L, D]`.
            grads   : dict with ``"w"`` — :math:`[D, N]`.
        """
        w = self.last_routing_weights  # [B, L, N]
        prod = d_probs * w  # [B, L, N]
        term2 = torch.sum(prod, dim=-1, keepdim=True)  # [B, L, 1]
        d_logits = w * (d_probs - term2)  # [B, L, N]

        # dW = x^T @ d_logits  →  [D, N]
        d_weights = torch.matmul(
            x.reshape(-1, self.embed_dim).T,  # [D, B*L]
            d_logits.reshape(-1, self.num_experts),  # [B*L, N]
        )

        # dx = d_logits @ W^T  →  [B, L, D]
        dx = torch.matmul(d_logits, self.w.T)

        grads: dict[str, torch.Tensor] = {"w": d_weights}
        return dx, grads

    def get_params(self) -> dict[str, torch.Tensor]:
        return {"w": self.w}

    def set_params(self, params: dict[str, object]) -> None:
        for name, param in params.items():
            if name == "w":
                if isinstance(param, np.ndarray):
                    param = torch.from_numpy(param)
                with torch.no_grad():
                    self.w.copy_(param)


class PyTorchExpert(nn.Module):
    r"""
    A single feed-forward expert (PyTorch).

    Two-layer MLP with ReLU:

    .. math::
        h = \text{ReLU}(x W_1 + b_1) \\
        y = h W_2 + b_2

    Parity-tested against NumPy Expert.

    Dimensions:
    - Input :math:`x` — :math:`[B, L, D]`
    - Hidden — :math:`[B, L, D_{ff}]`
    - Output — :math:`[B, L, D]`
    """

    def __init__(self, embed_dim: int, dim_ff: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dim_ff = dim_ff
        self.w1 = nn.Parameter(torch.randn(embed_dim, dim_ff) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(dim_ff))
        self.w2 = nn.Parameter(torch.randn(dim_ff, embed_dim) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor :math:`[B, L, D]`.
        Returns:
            Output :math:`[B, L, D]`.
        """
        self.x = x
        # Linear 1  [B, L, D_ff]
        self.z1 = torch.matmul(x, self.w1) + self.b1
        # ReLU
        self.h = torch.nn.functional.relu(self.z1)
        # Linear 2  [B, L, D]
        output = torch.matmul(self.h, self.w2) + self.b2
        return output

    def backward(
        self, x: torch.Tensor, d_out: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Full FFN backward pass.

        Args:
            x       : Input tensor :math:`[B, L, D]`.
            d_out   : Gradient w.r.t. output :math:`[B, L, D]`.

        Returns:
            dx      : Gradient w.r.t. input :math:`[B, L, D]`.
            grads   : dict with "w1", "b1", "w2", "b2".
        """
        batch_size, seq_len, embed_dim = x.shape

        # Flatten for matrix operations
        h_flat = self.h.reshape(-1, self.dim_ff)       # [B*L, D_ff]
        d_out_flat = d_out.reshape(-1, embed_dim)       # [B*L, D]

        # --- gradient w.r.t. w2, b2 ---
        grad_w2 = torch.matmul(h_flat.T, d_out_flat)        # [D_ff, D]
        grad_b2 = d_out_flat.sum(dim=0)                     # [D]

        # --- gradient through W2 back to h ---
        grad_h = torch.matmul(d_out_flat, self.w2.T)        # [B*L, D_ff]

        # --- ReLU backward ---
        z1_flat = self.z1.reshape(-1, self.dim_ff)        # [B*L, D_ff]
        grad_z1 = grad_h * (z1_flat > 0).float()          # [B*L, D_ff]

        # --- gradient w.r.t. w1, b1 ---
        x_flat = x.reshape(-1, self.embed_dim)              # [B*L, D]
        grad_w1 = torch.matmul(x_flat.T, grad_z1)           # [D, D_ff]
        grad_b1 = grad_z1.sum(dim=0)                        # [D_ff]

        # --- gradient w.r.t. x ---
        dx = torch.matmul(grad_z1, self.w1.T).reshape(
            batch_size, seq_len, self.embed_dim
        )

        grads: dict[str, torch.Tensor] = {
            "w1": grad_w1,
            "b1": grad_b1,
            "w2": grad_w2,
            "b2": grad_b2,
        }
        return dx, grads

    def get_params(self) -> dict[str, torch.Tensor]:
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    def set_params(self, params: dict[str, object]) -> None:
        for name, param in params.items():
            if isinstance(param, np.ndarray):
                param = torch.from_numpy(param)
            if name in ("w1", "b1", "w2", "b2"):
                with torch.no_grad():
                    getattr(self, name).copy_(param)


class PyTorchMoELayer(nn.Module):
    r"""
    Mixture-of-Experts layer (PyTorch).

    For each token, the router selects the top-k experts and the output is
    a weighted sum:

    .. math::
        y = \sum_{j \in \text{top}_k} \tilde{P}_{j} \cdot \text{Expert}_j(x)

    Parity-tested against NumPy MoELayer.

    Dimensions:
    - Input :math:`x` — :math:`[B, L, D]`
    - Top-k indices — :math:`[B, L, K]`
    - Top-k weights — :math:`[B, L, K]` (normalised)
    - All expert outputs — :math:`[N, B, L, D]`
    - Combined output — :math:`[B, L, D]`
    """

    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        dim_ff: int = 128,
        num_experts_per_token: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.k = min(num_experts_per_token, num_experts)

        self.router: PyTorchRouter = PyTorchRouter(embed_dim, num_experts)
        self.experts: nn.ModuleList = nn.ModuleList(
            [PyTorchExpert(embed_dim, dim_ff) for _ in range(num_experts)]
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
        """
        MoE forward pass.

        Args:
            x: Input tensor :math:`[B, L, D]`.

        Returns:
            combined_output :math:`[B, L, D]`
            cache: dict with routing info.
        """
        batch_size, seq_len, _ = x.shape

        # 1. Router probabilities  [B, L, N]
        routing_weights: torch.Tensor = self.router.forward(x)

        # 2. Top-k indices   [B, L, K]
        top_k_indices: torch.Tensor = torch.argsort(
            routing_weights, dim=-1
        )[..., -self.k:]

        # 3. Unnormalised top-k weights   [B, L, K]
        top_k_raw: torch.Tensor = torch.gather(
            routing_weights, -1, top_k_indices
        )

        # 4. Normalise   [B, L, K]
        top_k_sum: torch.Tensor = top_k_raw.sum(dim=-1, keepdim=True) + 1e-8
        top_k_weights: torch.Tensor = top_k_raw / top_k_sum

        # 5. All expert outputs  [N, B, L, D]
        all_expert_outputs: torch.Tensor = torch.stack(
            [exp.forward(x) for exp in self.experts], dim=0
        )

        # 6. Weighted combination  [B, L, D]
        batch_idx: torch.Tensor = torch.arange(
            batch_size, device=x.device
        ).view(batch_size, 1, 1)  # [B,1,1]
        seq_idx: torch.Tensor = torch.arange(
            seq_len, device=x.device
        ).view(1, seq_len, 1)  # [1,L,1]
        # index: [N,B,L,D][top_k_indices, batch_idx, seq_idx] -> [B,L,K,D]
        expert_outputs_for_tokens: torch.Tensor = all_expert_outputs[
            top_k_indices, batch_idx, seq_idx
        ]  # [B, L, K, D]
        combined_output: torch.Tensor = (
            top_k_weights.unsqueeze(-1) * expert_outputs_for_tokens
        ).sum(dim=2)  # [B, L, D]

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
        self, x: torch.Tensor, d_out: torch.Tensor, cache: dict[str, object]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Full MoE backward pass using autograd-compatible manual gradients
        to match the NumPy implementation exactly.

        Args:
            x       : Input :math:`[B, L, D]`.
            d_out   : Gradient w.r.t. combined output :math:`[B, L, D]`.
            cache   : Cache dict from forward.

        Returns:
            dx      : Gradient w.r.t. input :math:`[B, L, D]`.
            grads   : Flat parameter-gradient dict.
        """
        top_k_indices: torch.Tensor = cache["top_k_indices"]      # [B, L, K]
        top_k_weights: torch.Tensor = cache["top_k_weights"]      # [B, L, K]
        top_k_sum: torch.Tensor = cache["top_k_sum"]              # [B, L, 1]
        all_expert_outputs: torch.Tensor = cache["all_expert_outputs"]  # [N, B, L, D]

        batch_size, seq_len, embed_dim = x.shape

        # 1. Gradients w.r.t. all_expert_outputs
        d_all_expert_outputs: torch.Tensor = torch.zeros_like(
            all_expert_outputs
        )  # [N, B, L, D]
        d_top_k_weights: torch.Tensor = torch.zeros_like(top_k_weights)  # [B, L, K]

        # Gather d_out[b, s, :] with shape [B, L, D]
        b_idx = torch.arange(batch_size, device=x.device).unsqueeze(-1)  # [B, 1]
        s_idx = torch.arange(seq_len, device=x.device).unsqueeze(0)     # [1, L]
        d_out_gathered: torch.Tensor = d_out[b_idx, s_idx]               # [B, L, D]

        # Accumulate weight-scaled d_out into d_all_expert_outputs[expert_idx, b, s, :]
        for k_idx in range(self.k):
            expert_for_k = top_k_indices[:, :, k_idx]  # [B, L]
            # broadcast expert_for_k [B,L], b_idx [B,1], s_idx [1,L] -> [B,L,D]
            d_all_expert_outputs[expert_for_k, b_idx, s_idx] += (
                top_k_weights[:, :, k_idx].unsqueeze(-1)
                * d_out_gathered
            )

        # d_top_k_weights[b, s, k] = sum_d top_k_w * d_out[b,s,d] * expert_out[b,s,d]
        for k_idx in range(self.k):
            expert_for_k = top_k_indices[:, :, k_idx]
            # all_expert_outputs[expert_for_k, b_idx, s_idx] -> [B, L, D]
            expert_out_for_k = all_expert_outputs[expert_for_k, b_idx, s_idx]  # [B, L, D]
            d_top_k_weights[:, :, k_idx] = torch.sum(
                top_k_weights[:, :, k_idx, None]
                * d_out_gathered
                * expert_out_for_k,
                dim=-1,
            )

        # 2. Back-propagate through each expert
        d_x_from_experts: torch.Tensor = torch.zeros_like(x)  # [B, L, D]
        grads_experts: dict[str, torch.Tensor] = {}

        for i in range(self.num_experts):
            mask: torch.Tensor = top_k_indices == i  # [B, L, K]
            if mask.any():
                dx_i: torch.Tensor
                grads_i: dict[str, torch.Tensor]
                dx_i, grads_i = self.experts[i].backward(x, d_all_expert_outputs[i])
                d_x_from_experts = d_x_from_experts + dx_i
                for name, grad in grads_i.items():
                    grads_experts[f"expert.{i}.{name}"] = grad

        # 3. Back-propagate through top-k normalisation
        d_top_k_sum: torch.Tensor = -torch.sum(
            top_k_weights * d_top_k_weights, dim=-1, keepdim=True,
        ) / (top_k_sum + 1e-8)  # [B, L, 1]

        d_top_k_raw: torch.Tensor = (d_top_k_weights + d_top_k_sum) / (
            top_k_sum + 1e-8
        )  # [B, L, K]

        # Place d_raw into d_routing_weights only at top-k positions
        d_routing_weights: torch.Tensor = torch.zeros(
            (batch_size, seq_len, self.num_experts), dtype=x.dtype, device=x.device,
        )
        for k_idx in range(self.k):
            d_routing_weights[
                torch.arange(batch_size).unsqueeze(1).unsqueeze(2),
                torch.arange(seq_len).unsqueeze(0).unsqueeze(2),
                top_k_indices[:, :, k_idx].unsqueeze(-1),
            ] = d_top_k_raw[:, :, k_idx].unsqueeze(-1)

        # 4. Router backward
        dx_router: torch.Tensor
        grads_router: dict[str, torch.Tensor]
        dx_router, grads_router = self.router.backward(x, d_routing_weights)

        # 5. Combine all input gradients
        dx: torch.Tensor = d_x_from_experts + dx_router

        # Prefix router grads
        combined_grads: dict[str, torch.Tensor] = {}
        for name, grad in grads_router.items():
            combined_grads[f"router.{name}"] = grad
        combined_grads.update(grads_experts)

        return dx, combined_grads

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------
    def get_params(self) -> dict[str, torch.Tensor]:
        """Return a flat dict of all learnable parameters."""
        params: dict[str, torch.Tensor] = {}
        for name, param in self.router.get_params().items():
            params[f"router.{name}"] = param
        for i, expert in enumerate(self.experts):
            for name, param in expert.get_params().items():
                params[f"expert.{i}.{name}"] = param
        return params

    def set_params(self, params: dict[str, object]) -> None:
        """Assign parameters from a flat dict produced by :meth:`get_params`."""
        for name, param in params.items():
            if name.startswith("router."):
                key = name[len("router."):]
                self.router.set_params({key: param})
            elif name.startswith("expert."):
                # e.g. "expert.2.w1" -> expert_idx=2, key="w1"
                parts = name.split(".", 2)
                assert parts[0] == "expert" and parts[1].isdigit()
                idx = int(parts[1])
                key = parts[2]
                self.experts[idx].set_params({key: param})
