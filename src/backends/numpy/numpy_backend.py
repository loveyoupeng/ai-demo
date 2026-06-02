from __future__ import annotations

import numpy as np
from utils.backend_interface import BaseTransformerBackend
from model.transformer import Transformer


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
        mask: np.ndarray | None = None,
        use_cache: bool = False,
        cache_idx: int | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """
        Wraps the NumPy Transformer forward pass.
        """
        return self.model.forward(
            input_ids, mask=mask, use_cache=use_cache, cache_idx=cache_idx
        )

    def backward(
        self, grad_logits: np.ndarray, cache: dict[str, object]
    ) -> dict[str, np.ndarray]:
        """
        Wraps the NumPy Transformer backward pass.
        """
        return self.model.backward(grad_logits, cache)

    def get_params(self) -> dict[str, np.ndarray]:
        """
        Wraps the NumPy Transformer param retrieval.
        """
        return self.model.get_params()

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        """
        Wraps the NumPy Transformer param setting.
        """
        self.model.set_params(params)
