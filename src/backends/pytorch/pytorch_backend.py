import numpy as np
from typing import Any, Dict, Optional, Tuple
from utils.backend_interface import BaseTransformerBackend


class PyTorchBackend(BaseTransformerBackend):
    """
    Skeleton for the PyTorch-based backend implementation of the Transformer.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        num_experts: int,
        max_seq_len: int = 512,
    ):
        # To be implemented: Initialize PyTorch-native Transformer
        pass

    def forward(
        self,
        input_ids: np.ndarray,
        mask: Optional[np.ndarray] = None,
        use_cache: bool = False,
        cache_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # To be implemented
        raise NotImplementedError()

    def backward(
        self, grad_logits: np.ndarray, cache: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        # To be implemented
        raise NotImplementedError()

    def get_params(self) -> Dict[str, np.ndarray]:
        # To be implemented
        raise NotImplementedError()

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        # To be implemented
        raise NotImplementedError()
