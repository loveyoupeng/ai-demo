import torch
import torch.nn as nn
from typing import Dict, cast, Optional


class TokenEmbedding(nn.Module):
    """
    Learned token embeddings using PyTorch.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {"weights": cast(torch.Tensor, self.embedding.weight)}

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        if "weights" in params:
            with torch.no_grad():
                self.embedding.weight.copy_(params["weights"])

    def get_grads(self) -> Dict[str, torch.Tensor]:
        return (
            {"weights": cast(torch.Tensor, self.embedding.weight.grad)}
            if self.embedding.weight.grad is not None
            else {}
        )


class PositionalEmbedding(nn.Module):
    """
    Fixed sinusoidal positional embeddings using PyTorch.
    """

    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        pe = torch.empty((max_seq_len, embed_dim))  # type: ignore
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)  # type: ignore
        div_term = torch.exp(  # type: ignore
            torch.arange(0, embed_dim, 2, dtype=torch.float32)  # type: ignore
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)  # type: ignore
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # type: ignore
        pe[:, 1::2] = torch.cos(position * div_term)  # type: ignore
        self.register_buffer("pe", pe)

    def forward(self) -> torch.Tensor:
        return cast(torch.Tensor, self.pe)

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {}

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        pass

    def get_grads(self) -> Dict[str, torch.Tensor]:
        return {}


class LayerNorm(nn.Module):
    """
    Layer Normalization using PyTorch.
    """

    def __init__(self, embed_dim: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {
            "gamma": cast(torch.Tensor, self.ln.weight),
            "beta": cast(torch.Tensor, self.ln.bias),
        }

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            if "gamma" in params:
                self.ln.weight.copy_(params["gamma"])
            if "beta" in params:
                self.ln.bias.copy_(params["beta"])

    def get_grads(self) -> Dict[str, torch.Tensor]:
        return {
            "gamma": cast(torch.Tensor, self.ln.weight.grad)
            if self.ln.weight.grad is not None
            else torch.zeros_like(self.ln.weight),  # type: ignore
            "beta": cast(torch.Tensor, self.ln.bias.grad)
            if self.ln.bias.grad is not None
            else torch.zeros_like(self.ln.bias),  # type: ignore
        }


class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) using PyTorch.
    Matches the structure: Linear(dim, dim*4) -> GeLU -> Linear(dim*4, dim).
    """

    def __init__(self, embed_dim: int, intermediate_dim: Optional[int] = None):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = embed_dim * 4

        self.w1 = nn.Linear(embed_dim, intermediate_dim)
        self.w2 = nn.Linear(intermediate_dim, embed_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.activation(self.w1(x)))

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {
            "w1_weight": cast(torch.Tensor, self.w1.weight),
            "w1_bias": cast(torch.Tensor, self.w1.bias),
            "w2_weight": cast(torch.Tensor, self.w2.weight),
            "w2_bias": cast(torch.Tensor, self.w2.bias),
        }

    def set_params(self, params: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            if "w1_weight" in params:
                self.w1.weight.copy_(params["w1_weight"])
            if "w1_bias" in params:
                self.w1.bias.copy_(params["w1_bias"])
            if "w2_weight" in params:
                self.w2.weight.copy_(params["w2_weight"])
            if "w2_bias" in params:
                self.w2.bias.copy_(params["w2_bias"])

    def get_grads(self) -> Dict[str, torch.Tensor]:
        return {
            "w1_weight": cast(torch.Tensor, self.w1.weight.grad)
            if self.w1.weight.grad is not None
            else torch.zeros_like(self.w1.weight),  # type: ignore
            "w1_bias": cast(torch.Tensor, self.w1.bias.grad)
            if self.w1.bias.grad is not None
            else torch.zeros_like(self.w1.bias),  # type: ignore
            "w2_weight": cast(torch.Tensor, self.w2.weight.grad)
            if self.w2.weight.grad is not None
            else torch.zeros_like(self.w2.weight),  # type: ignore
            "w2_bias": cast(torch.Tensor, self.w2.bias.grad)
            if self.w2.bias.grad is not None
            else torch.zeros_like(self.w2.bias),  # type: ignore
        }
