import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

class Router(nn.Module):
    """
    The Routing/Gating network for PyTorch MoE.
    """
    def __init__(self, embed_dim: int, num_experts: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        # Weights shape: [Embed_Dim, Num_Experts] to match NumPy dot product
        self.weights = nn.Parameter(torch.randn(embed_dim, num_experts) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [Batch, Seq_Len, Embed_Dim]
        Returns:
            [Batch, Seq_Len, Num_Experts] routing probabilities
        """
        # logits shape: [Batch, Seq_Len, Num_Experts]
        logits = torch.matmul(x, self.weights)
        return F.softmax(logits, dim=-1)

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {"weights": self.weights}

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            if "weights" in params:
                self.weights.copy_(params["weights"])

    def get_grads(self) -> Dict[str, torch.Tensor]:
        return {"weights": self.weights.grad} if self.weights.grad is not None else {}

class Expert(nn.Module):
    """
    An individual expert in the MoE layer.
    """
    def __init__(self, embed_dim: int, dim_ff: int):
        super().__init__()
        from model.pytorch.layers import FeedForward
        self.ffn = FeedForward(embed_dim, dim_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

    def get_params(self) -> Dict[str, torch.Tensor]:
        return self.ffn.get_params()

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        self.ffn.set_params(params)

    def get_grads(self) -> Dict[str, torch.Tensor]:
        return self.ffn.get_grads()

class MoELayer(nn.Module):
    """
    Mixture of Experts (MoE) layer for PyTorch.
    """
    def __init__(
        self,
        embed_dim: int,
        num_experts: int,
        dim_ff: int = 128,
        num_experts_per_token: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.k = min(num_experts_per_token, num_experts)

        self.router = Router(embed_dim, num_experts)
        self.experts = nn.ModuleList([Expert(embed_dim, dim_ff) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Seq_Len, Embed_Dim]
        Returns:
            combined_output: [Batch, Seq_Len, Embed_Dim]
        """
        batch_size, seq_len, _ = x.shape

        # 1. Get routing probabilities [Batch, Seq_Len, Num_Experts]
        routing_weights = self.router(x)

        # 2. Identify top-k experts
        # top_k_weights, top_k_indices: [Batch, Seq_Len, K]
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.k, dim=-1)

        # 3. Normalize top-k weights
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 4. Compute expert outputs
        combined_output = torch.zeros_like(x)
        
        # We iterate over experts to compute outputs for tokens that use them.
        # This is more efficient than looping over all tokens.
        for i in range(self.num_experts):
            # Find where expert i is selected in the top-k
            # mask: [Batch, Seq_Len, K]
            mask = (top_k_indices == i)
            if not mask.any():
                continue

            # indices of (b, s, k) where expert i is selected
            # indices: [Num_matches, 3]
            indices = torch.nonzero(mask)
            b_indices = indices[:, 0]
            s_indices = indices[:, 1]
            k_indices = indices[:, 2]

            # Get the weights for these matches: [Num_matches]
            weights = top_k_weights[b_indices, s_indices, k_indices]

            # Get the tokens: [Num_matches, Embed_Dim]
            tokens = x[b_indices, s_indices]

            # Expert forward: [Num_matches, Embed_Dim]
            expert_out = self.experts[i](tokens)

            # Accumulate to combined_output
            # Since for a fixed expert i, each (b, s) appears at most once in the top-k,
            # we can safely add.
            combined_output[b_indices, s_indices] += weights.unsqueeze(-1) * expert_out

        return combined_output

    def get_params(self) -> Dict[str, torch.Tensor]:
        params = {}
        for k, v in self.router.get_params().items():
            params[f"router.{k}"] = v
        for i, expert in enumerate(self.experts):
            for k, v in expert.get_params().items():
                params[f"experts.{i}.{k}"] = v
        return params

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        for k, v in params.items():
            if k.startswith("router."):
                param_name = k.replace("router.", "")
                self.router.set_params({param_name: v})
            elif k.startswith("experts."):
                parts = k.split(".")
                i = int(parts[1])
                param_name = ".".join(parts[2:])
                self.experts[i].set_params({param_name: v})

