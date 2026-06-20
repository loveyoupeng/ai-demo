"""E7: TransformerBlock — Python wiring of Triton kernels.

Assembles multi-head attention (TritonMHA), MoE (TritonMoE),
and RMSNorm (TritonRMSNorm) into a complete decoder-only transformer block.

Architecture (same as _torch TransformerBlock):
    Stream 1: x → MHA(x) → h = x + attn → rmsnorm(h) →
              gate1 = sigmoid(gate1) * h → dropout(h)
    Stream 2: h → MoE(h) → out = h + moe → rmsnorm(out) →
              gate2 = sigmoid(gate2) * out → dropout(out)

No new Triton kernels — this is pure PyTorch module wiring.
"""

import torch
import torch.nn as nn

from impl._triton.attn import scaled_dot_product_attention
from impl._triton.ffn import swiglu_ffn
from impl._torch.layers import Linear


class TritonMultiHeadAttention(nn.Module):
    """Multi-head attention using Triton scaled-dot-product kernel."""

    def __init__(self, embed_dim: int, n_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.Wq = Linear(embed_dim, embed_dim, bias=True)
        self.Wk = Linear(embed_dim, embed_dim, bias=True)
        self.Wv = Linear(embed_dim, embed_dim, bias=True)
        self.Wo = Linear(embed_dim, embed_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.Wq.weight, a=0.01)
        nn.init.kaiming_uniform_(self.Wk.weight, a=0.01)
        nn.init.kaiming_uniform_(self.Wv.weight, a=0.01)
        nn.init.kaiming_uniform_(self.Wo.weight, a=0.01)

    def _move_to_device(self, x: torch.Tensor) -> None:
        """Move parameters to the same device and dtype as x on first forward pass."""
        if not x.is_cuda:
            return
        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point or x.dtype.is_complex else None
        for param in [self.Wq, self.Wk, self.Wv, self.Wo]:
            if param is not None:
                if dtype is not None:
                    param = param.to(device, dtype)
                else:
                    param = param.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._move_to_device(x)
        B, S, D = x.shape

        # Q, K, V projections: [B, S, D] -> [B, S, embed_dim]
        q = self.Wq(x)  # [B, S, D]
        k = self.Wk(x)  # [B, S, D]
        v = self.Wv(x)  # [B, S, D]

        # Reshape for MHA: [B, S, H, head_dim] -> [B, H, S, head_dim]
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention: [B, H, S, S] x [B, H, S, head_dim] -> [B, H, S, head_dim]
        attn_out = scaled_dot_product_attention(q, k, v)

        # Reshape back: [B, H, S, head_dim] -> [B, S, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        # Output projection: [B, S, D] -> [B, S, D]
        out = self.Wo(attn_out)  # [B, S, D]
        return out


class TritonTransformerBlock(nn.Module):
    """TransformerBlock using Triton kernels internally.

    Parameters
    ----------
    embed_dim : int
        Hidden dimension.
    n_heads : int
        Number of attention heads.
    n_experts : int
        Number of MoE experts.
    ff_dim : int
        Feed-forward hidden dimension.
    k : int
        Number of top experts to activate.
    dropout : float
        Dropout rate (disabled during eval).
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_experts: int,
        ff_dim: int,
        k: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_experts = n_experts
        self.k = k
        self.norm_type = "post"

        # RMSNorm instances — matching _torch layer structure
        self.ln1 = nn.RMSNorm(embed_dim, eps=1e-5)
        self.ln2 = nn.RMSNorm(embed_dim, eps=1e-5)

        # Gated residuals
        self.gate1 = nn.Parameter(torch.zeros(1))
        self.gate2 = nn.Parameter(torch.zeros(1))

        # Multi-head attention
        self.mha = TritonMultiHeadAttention(embed_dim, n_heads)

        # MoE
        self.moe = TritonMoE(embed_dim, n_experts, ff_dim, k)

        # Dropout (training only)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point or x.dtype.is_complex else None
        if dtype is not None:
            self.ln1 = self.ln1.to(device, dtype)
            self.ln2 = self.ln2.to(device, dtype)
        elif device is not None:
            self.ln1 = self.ln1.to(device)
            self.ln2 = self.ln2.to(device)
        for param in [self.gate1, self.gate2]:
            if param.device != device or (dtype is not None and param.dtype != dtype):
                param.data = param.data.to(device, dtype or param.dtype)
        self.moe._move_to_device(x)

        # Stream 1: Attention
        attn_out = self.mha(x)  # [B, S, D]
        h = x + attn_out
        h = self.ln1(h)  # [B, S, D]
        gate1 = torch.sigmoid(self.gate1)
        h = h + gate1 * h  # gated residual
        h = self.dropout1(h)  # dropout (training only)

        # Stream 2: MoE
        moe_out = self.moe(h)  # [B, S, D]
        out = h + moe_out
        out = self.ln2(out)  # [B, S, D]
        gate2 = torch.sigmoid(self.gate2)
        out = out + gate2 * out  # gated residual
        out = self.dropout2(out)  # dropout (training only)

        return out

    def _move_to_device(self, x: torch.Tensor) -> None:
        """Move all parameters to the same device and dtype as x (skip if int)."""
        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point or x.dtype.is_complex else None
        if dtype is not None:
            self.ln1 = self.ln1.to(device, dtype)
            self.ln2 = self.ln2.to(device, dtype)
        elif device is not None:
            self.ln1 = self.ln1.to(device)
            self.ln2 = self.ln2.to(device)
        for param in [self.gate1, self.gate2]:
            if param.device != device or (dtype is not None and param.dtype != dtype):
                param.data = param.data.to(device, dtype or param.dtype)
        self.mha._move_to_device(x)
        self.moe._move_to_device(x)


class TritonExpert(nn.Module):
    """Single SwiGLU expert."""

    def __init__(self, embed_dim: int, ff_dim: int) -> None:
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(embed_dim, ff_dim))
        self.W3 = nn.Parameter(torch.empty(embed_dim, ff_dim))
        self.W2 = nn.Parameter(torch.empty(ff_dim, embed_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W1, a=0.01)
        nn.init.kaiming_uniform_(self.W3, a=0.01)
        nn.init.kaiming_uniform_(self.W2, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swiglu_ffn(x, self.W1, self.W3, self.W2)


class TritonMoE(nn.Module):
    """Mixture of Experts using Triton kernels."""

    def __init__(
        self,
        embed_dim: int,
        n_experts: int,
        ff_dim: int,
        k: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_experts = n_experts
        self.k = k
        self.W_router = nn.Parameter(torch.empty(embed_dim, n_experts))
        self.b_router = nn.Parameter(torch.zeros(n_experts))
        self.experts = nn.ModuleList([
            TritonExpert(embed_dim, ff_dim) for _ in range(n_experts)
        ])

    def _move_to_device(self, x: torch.Tensor) -> None:
        """Move all MoE parameters to the same device and dtype as x (skip if int)."""
        if not x.is_cuda:
            return
        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point or x.dtype.is_complex else None
        for param in [self.W_router, self.b_router]:
            if param.device != device or (dtype is not None and param.dtype != dtype):
                param.data = param.data.to(device, dtype or param.dtype)
        for expert in self.experts:
            for p in [expert.W1, expert.W3, expert.W2]:
                if p.device != device or (dtype is not None and p.dtype != dtype):
                    p.data = p.data.to(device, dtype or p.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._move_to_device(x)

        # Router scores: [B, S, E]
        scores = x @ self.W_router + self.b_router

        # Softmax routing weights: [B, S, E]
        scores_max = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - scores_max)
        routing_weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True)

        # Top-k selection and renormalization
        n_experts = self.n_experts
        k = self.k
        if k < n_experts:
            top_k_values, _ = torch.topk(routing_weights, k, dim=-1)
            threshold = top_k_values.min(dim=-1, keepdim=True).values
            routing_weights = routing_weights * (routing_weights >= threshold).float()
            renorm_sum = routing_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            routing_weights = routing_weights / renorm_sum

        # Compute expert outputs: [E, B, S, D]
        expert_inputs = x.unsqueeze(0).expand(n_experts, -1, -1, -1)  # [E, B, S, D]
        expert_outs = torch.stack([expert(expert_inputs[i]) for expert, i in zip(self.experts, range(n_experts), strict=True)])

        # Weighted sum: [B, S, E] x [E, B, S, D] -> [B, S, D]
        out = torch.einsum("bse,ebsd->bsd", routing_weights, expert_outs)
        return out


class TritonDecoderStack(nn.Module):
    """Stack of TritonTransformerBlocks — chains n_layers of decoder blocks.

    Architecture:
        Input:  x [B, S, D]
        |
        +-> block_0 -> block_1 -> ... -> block_{n_layers-1} -> output [B, S, D]

        Each block:
          h = x + MHA(RMSNorm(x) + gated) + MoE(RMSNorm(residual) + gated)

    Parameters:
        n_layers: Number of transformer blocks.
        embed_dim: Input/output dimension.
        n_heads: Number of attention heads per block.
        n_experts: Number of MoE experts per block.
        ff_dim: Feed-forward hidden dimension per expert.
        k: Number of top experts per token.
    """

    def __init__(
        self,
        n_layers: int,
        embed_dim: int,
        n_heads: int,
        n_experts: int,
        ff_dim: int,
        k: int = 2,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            TritonTransformerBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_experts=n_experts,
                ff_dim=ff_dim,
                k=k,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.layers:
            out = block(out)
        return out

    def _move_to_device(self, x: torch.Tensor) -> None:
        for block in self.layers:
            block._move_to_device(x)
