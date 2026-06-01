import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, cast
from src.core.registry import registry

class PyTorchTokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Register mappings for PyTorch
        # Canonical Name <-> PyTorch internal name
        registry.register("pytorch", "embedding.weights", "embedding.weight")

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)

    def get_params(self) -> Dict[str, torch.Tensor]:
        # Return parameters in CANONICAL form
        return {"embedding.weights": self.embedding.weight}

    def set_params(self, params: Dict[str, Any]) -> None:
        # params are in CANONICAL form
        if "embedding.weights" in params:
            val = params["embedding.weights"]
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            with torch.no_grad():
                self.embedding.weight.copy_(val)

    def get_grads(self) -> Dict[str, torch.Tensor]:
        grads = {}
        if self.embedding.weight.grad is not None:
            grads["embedding.weights"] = self.embedding.weight.grad
        return grads
