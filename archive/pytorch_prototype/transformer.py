import torch
import torch.nn as nn
from typing import Optional, Dict, cast, Any
from model.pytorch.attention import MultiHeadAttention
from model.pytorch.layers import (
    LayerNorm,
    TokenEmbedding,
    PositionalEmbedding,
)
from model.pytorch.moe import MoELayer

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_experts: int):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.moe = MoELayer(embed_dim, num_experts)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-LN residual connection
        attn_out, _ = self.attn(self.ln1(x), mask=mask)
        x = x + attn_out
        x = x + self.moe(self.ln2(x))
        return x

    def get_params(self) -> Dict[str, torch.Tensor]:
        params = {}
        for k, v in self.ln1.get_params().items():
            params[f"ln1.{k}"] = v
        for k, v in self.ln2.get_params().items():
            params[f"ln2.{k}"] = v
        for k, v in self.attn.get_params().items():
            params[f"attn.{k}"] = v
        for k, v in self.moe.get_params().items():
            params[f"moe.{k}"] = v
        return params

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        for k, v in params.items():
            if k.startswith("ln1."):
                self.ln1.set_params({k.replace("ln1.", ""): v})
            elif k.startswith("ln2."):
                self.ln2.set_params({k.replace("ln2.", ""): v})
            elif k.startswith("attn."):
                self.attn.set_params({k.replace("attn.", ""): v})
            elif k.startswith("moe."):
                self.moe.set_params({k.replace("moe.", ""): v})

class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        num_experts: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_embedding = PositionalEmbedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, num_experts)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self, indices: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = indices.shape
        x = self.token_embedding(indices)
        pe = self.positional_embedding()
        x = x + pe[:seq_len, :].unsqueeze(0)

        if mask is None:
            mask = torch.tril(torch.ones((seq_len, seq_len), device=indices.device))

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def get_params(self) -> Dict[str, torch.Tensor]:
        params = {}
        for k, v in self.token_embedding.get_params().items():
            params[f"token_embedding.{k}"] = v
        for i, layer in enumerate(self.layers):
            for k, v in layer.get_params().items():
                params[f"layers.{i}.{k}"] = v
        for k, v in self.ln_f.get_params().items():
            params[f"ln_f.{k}"] = v
        params["lm_head.weight"] = self.lm_head.weight
        return params

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        for k, v in params.items():
            if k.startswith("token_embedding."):
                self.token_embedding.set_params({k.replace("token_embedding.", ""): v})
            elif k.startswith("layers."):
                parts = k.split(".")
                idx = int(parts[1])
                self.layers[idx].set_params({ ".".join(parts[2:]): v })
            elif k.startswith("ln_f."):
                self.ln_f.set_params({k.replace("ln_f.", ""): v})
            elif k == "lm_head.weight":
                self.lm_head.weight.data.copy_(v)
