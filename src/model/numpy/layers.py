import numpy as np
from typing import Tuple, Dict
from src.core.base_backend import BaseTransformerBackend
from src.core.registry import registry

# Define Canonical Names
# ----------------------
# For TokenEmbedding: 'embedding.weights'
# For PositionalEmbedding: (none)
# For LayerNorm: 'ln.gamma', 'ln.beta'
# For FeedForward: 'ffn.w1', 'ffn.b1', 'ffn.w2', 'ffn.b2'
# For Attention: 'attn.w_q', 'attn.w_k', 'attn.w_v', 'attn.proj'
# For MoE: 'moe.router.weights', 'moe.experts.X.ffn.w1' ...

class NumPyTokenEmbedding:
    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weights = np.random.randn(vocab_size, embed_dim) * 0.01
        self.grad_weights = np.zeros_like(self.weights)

    def forward(self, indices: np.ndarray) -> np.ndarray:
        self.indices = indices
        return self.weights[indices]

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        batch_size, seq_len, embed_dim = grad_output.shape
        rows = self.indices.ravel()
        grad_output_flat = grad_output.reshape(-1, embed_dim)
        np.add.at(self.grad_weights, rows, grad_output_flat)
        return grad_output, {"weights": self.grad_weights}

    def get_params(self) -> Dict[str, np.ndarray]:
        return {"weights": self.weights}

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        if "weights" in params:
            self.weights = params["weights"]

    def get_grads(self) -> Dict[str, np.ndarray]:
        return {"weights": self.grad_weights}

class NumPyBackend:
    """
    A lightweight wrapper around NumPy components to satisfy the BaseTransformerBackend.
    In a full implementation, this would reside in src/backends/numpy/numpy_backend.py.
    """
    def __init__(self, vocab_size: int, embed_dim: int):
        self.embedding = NumPyTokenEmbedding(vocab_size, embed_dim)
        
        # Register mappings for NumPy
        # Canonical Name <-> NumPy internal name
        registry.register("numpy", "embedding.weights", "weights")
        
    def get_params(self) -> Dict[str, np.ndarray]:
        # We return parameters in CANONICAL form
        params = {}
        # Embedding
        params["embedding.weights"] = self.embedding.get_params()["weights"]
        return params

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        # params are in CANONICAL form
        if "embedding.weights" in params:
            self.embedding.set_params({"weights": params["embedding.weights"]})

    # Placeholder for full backend implementation
    def forward(self, *args, **kwargs): pass
    def backward(self, *args, **kwargs): pass
