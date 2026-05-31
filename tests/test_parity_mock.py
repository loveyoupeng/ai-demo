import numpy as np
from typing import Any, Dict, Optional, Tuple
from src.utils.backend_interface import BaseTransformerBackend

class MinimalBackend(BaseTransformerBackend):
    def __init__(self, val=1.0):
        self.params = {"a": np.array([val], dtype=np.float64)}
    def forward(self, input_ids, mask=None, use_cache=False, cache_idx=None):
        return np.zeros((input_ids.shape[0], input_ids.shape[1], 1)), {}
    def backward(self, grad_logits, cache):
        return {"a": np.array([0.0], dtype=np.float64)}
    def get_params(self): return self.params
    def set_params(self, p): self.params = p
