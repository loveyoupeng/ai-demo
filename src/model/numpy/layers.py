from __future__ import annotations

import numpy as np
from src.core.registry import registry

class NumPyTokenEmbedding:
    def __init__(self, vocab_size: int, embed_dim: int):
        registry.register("numpy", "embedding.weights", "weights")
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weights = np.random.randn(vocab_size, embed_dim) * 0.01

    def forward(self, indices: np.ndarray) -> np.ndarray:
        self.indices = indices
        return self.weights[indices]

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        grad_weights = np.zeros_like(self.weights)
        np.add.at(grad_weights, self.indices, grad_output)
        return grad_output, {"weights": grad_weights}

    def get_params(self) -> dict[str, np.ndarray]:
        return {"weights": self.weights}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        if "weights" in params:
            self.weights = params["weights"].copy()

    def get_grads(self) -> dict[str, np.ndarray]:
        return {"weights": np.zeros_like(self.weights)}

class NumPyPositionalEmbedding:
    def __init__(self, max_seq_len: int, embed_dim: int):
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        pe = np.zeros((max_seq_len, embed_dim))
        position = np.arange(0, max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x + self.pe[:x.shape[1], :]

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return grad_output, {}

    def get_params(self) -> dict[str, np.ndarray]:
        return {}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        pass

class NumPyFeedForward:
    def __init__(self, embed_dim: int, dim_ff: int):
        self.embed_dim = embed_dim
        self.dim_ff = dim_ff
        self.w1 = np.random.randn(embed_dim, dim_ff) * 0.01
        self.b1 = np.zeros(dim_ff)
        self.w2 = np.random.randn(dim_ff, embed_dim) * 0.01
        self.b2 = np.zeros(embed_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z1 = np.dot(x, self.w1) + self.b1
        self.h = np.maximum(0, self.z1)
        self.output = np.dot(self.h, self.w2) + self.b2
        return self.output

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        N, D = self.x.shape
        grad_w2 = np.dot(self.h.reshape(-1, self.dim_ff).T, grad_output.reshape(-1, self.embed_dim))
        grad_b2 = np.sum(grad_output, axis=0)
        
        grad_h = np.dot(grad_output, self.w2.T)
        grad_z1 = grad_h * (self.z1 > 0)
        
        grad_w1 = np.dot(self.x.reshape(-1, self.embed_dim).T, grad_z1.reshape(-1, self.dim_ff))
        grad_b1 = np.sum(grad_z1, axis=0)
        
        grad_x = np.dot(grad_z1, self.w1.T)
        
        grads = {
            "w1": grad_w1,
            "b1": grad_b1,
            "w2": grad_w2,
            "b2": grad_b2
        }
        return grad_x, grads

    def get_params(self) -> dict[str, np.ndarray]:
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        for k, v in params.items():
            setattr(self, k, v.copy())

class NumPyLayerNorm:
    def __init__(self, embed_dim: int, eps: float = 1e-5):
        self.embed_dim = embed_dim
        self.eps = eps
        self.gamma = np.ones(embed_dim)
        self.beta = np.zeros(embed_dim)
        registry.register("numpy", "ln.gamma", "gamma")
        registry.register("numpy", "ln.beta", "beta")

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        self.output = self.gamma * self.x_norm + self.beta
        return self.output

    def backward(self, grad_output: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        # LayerNorm backward gradients
        # https://arxiv.org/abs/1607.06450
        N = self.x.shape[-1]
        grad_x_norm = grad_output * self.gamma
        
        sum_grad_x_norm = np.sum(grad_x_norm, axis=-1, keepdims=True)
        sum_grad_x_norm_x_norm = np.sum(grad_x_norm * self.x_norm, axis=-1, keepdims=True)
        
        grad_x = (1.0 / np.sqrt(self.var + self.eps)) * (
            (N * grad_x_norm - sum_grad_x_norm - self.x_norm * sum_grad_x_norm_x_norm) / N
        )
        
        grads = {
            "gamma": np.sum(grad_output * self.x_norm, axis=0),
            "beta": np.sum(grad_output, axis=0)
        }
        return grad_x, grads

    def get_params(self) -> dict[str, np.ndarray]:
        return {"gamma": self.gamma, "beta": self.beta}

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        if "gamma" in params:
            self.gamma = params["gamma"].copy()
        if "beta" in params:
            self.beta = params["beta"].copy()
