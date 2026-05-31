import numpy as np
from typing import Any, Dict, Optional, Tuple
from src.utils.backend_interface import BaseTransformerBackend
from src.model.transformer import Transformer


class NumPyBackend(BaseTransformerBackend):
    """
    NumPy-based backend implementation of the Transformer.
    This serves as the pedagogical baseline/ground truth for all other backends.
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
        """
        Initializes the NumPy-based Transformer.
        """
        self.model = Transformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_experts=num_experts,
            max_seq_len=max_seq_len,
        )

    def forward(
        self,
        input_ids: np.ndarray,
        mask: Optional[np.ndarray] = None,
        use_cache: bool = False,
        cache_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Wraps the NumPy Transformer forward pass.
        """
        return self.model.forward(
            input_ids, mask=mask, use_cache=use_cache, cache_idx=cache_idx
        )

    def backward(
        self, grad_logits: np.ndarray, cache: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Wraps the NumPy Transformer backward pass.
        """
        return self.model.backward(grad_logits, cache)

    def get_params(self) -> Dict[str, np.ndarray]:
        """
        Wraps the NumPy Transformer param retrieval.
        """
        return self.model.get_params()

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """
        Wraps the NumPy Transformer param setting.
        """
        self.model.set_params(params)
