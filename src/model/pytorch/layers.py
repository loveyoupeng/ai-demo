from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from core.registry import registry

class PyTorchTokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.indices = None
        registry.register("pytorch", "embedding.weights", "embedding.weight")

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        self.indices = indices
        return self.embedding(indices)

    def backward(self, grad_output: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output = self.embedding(self.indices)
        loss = (output * grad_output).sum()
        loss.backward()
        if self.embedding.weight.grad is None:
            raise RuntimeError("embedding.weight.grad is None")
        dx = grad_output 
        grads = self.get_grads()
        return dx, grads

    def get_params(self) -> dict[str, torch.Tensor]:
        return {"embedding.weights": self.embedding.weight}

    def set_params(self, params: dict[str, np.ndarray | torch.Tensor]) -> None:
        if "embedding.weights" in params:
            val = params["embedding.weights"]
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            with torch.no_grad():
                self.embedding.weight.copy_(val)

    def get_grads(self) -> dict[str, torch.Tensor]:
        grads = {}
        if self.embedding.weight.grad is not None:
            grads["embedding.weights"] = self.embedding.weight.grad
        return grads

class PyTorchLayerNorm(nn.Module):
    def __init__(self, embed_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        registry.register("pytorch", "ln.gamma", "gamma")
        registry.register("pytorch", "ln.beta", "beta")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x = x
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        self.x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * self.x_norm + self.beta

    def backward(self, grad_output: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Manual backward to preserve the computation graph for chaining.
        # Matching the NumPy backward computation from
        # https://arxiv.org/abs/1607.06450
        x = self.x
        mean = self.x.mean(dim=-1, keepdim=True)
        var = self.x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = self.x_norm

        eps = self.eps
        gamma = self.gamma
        beta = self.beta

        # Compute gradients manually (matches NumPy LayerNorm backward)
        N = x.shape[-1]
        grad_x_norm = grad_output * gamma
        sum_grad_x_norm = torch.sum(grad_x_norm, dim=-1, keepdim=True)
        sum_grad_x_norm_x_norm = torch.sum(grad_x_norm * x_norm, dim=-1, keepdim=True)
        grad_x = (1.0 / torch.sqrt(var + eps)) * (
            (N * grad_x_norm - sum_grad_x_norm - x_norm * sum_grad_x_norm_x_norm) / N
        )

        grads = {
            "weight": torch.sum(grad_output * x_norm, dim=0),
            "bias": torch.sum(grad_output, dim=0),
        }

        return grad_x, grads

    def get_params(self) -> dict[str, torch.Tensor]:
        return {"ln.gamma": self.gamma, "ln.beta": self.beta}

    def set_params(self, params: dict[str, np.ndarray | torch.Tensor]) -> None:
        if "ln.gamma" in params:
            val = params["ln.gamma"]
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            with torch.no_grad():
                self.gamma.copy_(val)
        if "ln.beta" in params:
            val = params["ln.beta"]
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            with torch.no_grad():
                self.beta.copy_(val)

    def get_grads(self) -> dict[str, torch.Tensor]:
        grads = {}
        if self.gamma.grad is not None:
            grads["ln.gamma"] = self.gamma.grad
        if self.beta.grad is not None:
            grads["ln.beta"] = self.beta.grad
        return grads

class PyTorchFeedForward(nn.Module):
    def __init__(self, embed_dim: int, dim_ff: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.dim_ff = dim_ff
        self.w1 = nn.Parameter(torch.randn(embed_dim, dim_ff) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(dim_ff))
        self.w2 = nn.Parameter(torch.randn(dim_ff, embed_dim) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(embed_dim))
        registry.register("pytorch", "ffn.w1", "w1")
        registry.register("pytorch", "ffn.b1", "b1")
        registry.register("pytorch", "ffn.w2", "w2")
        registry.register("pytorch", "ffn.b2", "b2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x = x
        self.z1 = torch.matmul(x, self.w1) + self.b1
        self.h = torch.nn.functional.relu(self.z1)
        self.output = torch.matmul(self.h, self.w2) + self.b2
        return self.output

    def backward(self, grad_output: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.x.detach().requires_grad_(True)
        z1 = torch.matmul(x, self.w1) + self.b1
        h = torch.nn.functional.relu(z1)
        output = torch.matmul(h, self.w2) + self.b2
        loss = (output * grad_output).sum()
        
        self.zero_grad()
        loss.backward()
        
        grad_x = x.grad
        assert grad_x is not None, "x.grad should not be None after backward"
        grads = self.get_grads()
        return grad_x, grads

    def get_params(self) -> dict[str, torch.Tensor]:
        return {
            "ffn.w1": self.w1,
            "ffn.b1": self.b1,
            "ffn.w2": self.w2,
            "ffn.b2": self.b2
        }

    def set_params(self, params: dict[str, np.ndarray | torch.Tensor]) -> None:
        for k in ["w1", "b1", "w2", "b2"]:
            canonical_key = f"ffn.{k}"
            if canonical_key in params:
                val = params[canonical_key]
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                with torch.no_grad():
                    getattr(self, k).copy_(val)

    def get_grads(self) -> dict[str, torch.Tensor]:
        grads = {}
        if self.w1.grad is not None:
            grads["ffn.w1"] = self.w1.grad
        if self.b1.grad is not None:
            grads["ffn.b1"] = self.b1.grad
        if self.w2.grad is not None:
            grads["ffn.w2"] = self.w2.grad
        if self.b2.grad is not None:
            grads["ffn.b2"] = self.b2.grad
        return grads

class PyTorchPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        pe = torch.zeros((max_seq_len, embed_dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x = x
        pe = self.get_buffer("pe")  # type: ignore[arg-type]
        return x + pe[:x.shape[1], :]

    def backward(self, grad_output: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pe = self.get_buffer("pe")  # type: ignore[arg-type]
        return grad_output, {"pe": torch.zeros_like(pe)}

    def get_params(self) -> dict[str, torch.Tensor]:
        return {"pos.pe": self.get_buffer("pe")}  # type: ignore[return-value]

    def set_params(self, params: dict[str, object]) -> None:
        pass

    def get_grads(self) -> dict[str, torch.Tensor]:
        return {}
