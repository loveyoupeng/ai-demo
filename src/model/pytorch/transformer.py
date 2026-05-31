import torch
import torch.nn as nn
from typing import Optional, Dict, cast
from model.pytorch.attention import MultiHeadAttention
from model.pytorch.layers import FeedForward, LayerNorm, TokenEmbedding, PositionalEmbedding

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_intermediate_dim: int):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_intermediate_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LN residual connection
        attn_out, _ = self.attn(self.ln1(x), mask=mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {}

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        pass

    def get_grads(self) -> Dict[str, torch.Tensor]:
        return {}

class Transformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, max_seq_len: int, ffn_intermediate_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_embedding = PositionalEmbedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_intermediate_dim)
            for _ in range(num_layers)
        ])
        self.ln_f = LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, indices: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = indices.shape
        x = self.token_embedding(indices)
        pe = self.positional_embedding()
        x = x + pe[:seq_len, :].unsqueeze(0)
        
        # If no mask is provided, create a causal mask for decoder-only transformer
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=indices.device))
            
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {"lm_head_weight": self.lm_head.weight}

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        pass

    def get_grads(self) -> Dict[str, torch.Tensor]:
        return {"lm_head_weight": cast(torch.Tensor, self.lm_head.weight.grad)}
