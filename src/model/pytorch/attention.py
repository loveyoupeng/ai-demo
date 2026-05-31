import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, cast


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA) mechanism using PyTorch.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        # KV Cache: Dict[int, Tuple[Tensor, Tensor]]
        self.kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: Input tensor [Batch, Seq_Len, Embed_Dim]
            mask: Causal mask [Seq_Len, Seq_Len] (1 for keep, 0 for mask)
            use_cache: Whether to use/update KV cache
            cache_idx: Index of the current token for KV cache update
        Returns:
            output: [Batch, Seq_Len, Embed_Dim]
            cache: Dictionary containing intermediate values for backward pass
        """
        batch_size, seq_len, _ = x.shape

        # 1. Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Split into multiple heads
        # Shape: [Batch, Num_Heads, Seq_Len, Head_Dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # --- KV CACHE LOGIC ---
        if use_cache and cache_idx is not None:
            if cache_idx > 0 and (cache_idx - 1) in self.kv_cache:
                prev_K, prev_V = self.kv_cache[cache_idx - 1]
                K = torch.cat([prev_K, K], dim=2)  # type: ignore
                V = torch.cat([prev_V, V], dim=2)  # type: ignore
            self.kv_cache[cache_idx] = (K, V)
        # ----------------------

        # 3. Scaled Dot-Product Attention
        d_k = self.head_dim
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)

        # 4. Apply causal mask if provided
        if mask is not None:
            # mask [Seq_Len, Seq_Len] -> [1, 1, Seq_Len, Seq_Len] for broadcasting
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 5. Softmax
        attn_weights = torch.softmax(scores, dim=-1)  # type: ignore

        # 6. Weighted sum
        context = torch.matmul(attn_weights, V)

        # 7. Concatenate heads and final projection
        context_out = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        output = self.W_o(context_out)

        cache = {
            "Q": Q,
            "K": K,
            "V": V,
            "attn_weights": attn_weights,
            "context": context_out,
            "mask": mask,
        }

        return output, cache

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {
            "W_q": self.W_q.weight,
            "W_k": self.W_k.weight,
            "W_v": self.W_v.weight,
            "W_o": self.W_o.weight,
        }

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        for k, v in params.items():
            if k == "W_q":
                self.W_q.weight.data.copy_(v)
            elif k == "W_k":
                self.W_k.weight.data.copy_(v)
            elif k == "W_v":
                self.W_v.weight.data.copy_(v)
            elif k == "W_o":
                self.W_o.weight.data.copy_(v)

    def get_grads(self) -> Dict[str, torch.Tensor]:
        return {
            "W_q": cast(torch.Tensor, self.W_q.weight.grad),
            "W_k": cast(torch.Tensor, self.W_k.weight.grad),
            "W_v": cast(torch.Tensor, self.W_v.weight.grad),
            "W_o": cast(torch.Tensor, self.W_o.weight.grad),
        }
