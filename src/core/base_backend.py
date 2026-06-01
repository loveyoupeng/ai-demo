from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


class BaseTransformerBackend(ABC):
    """
    Abstract Base Class for all Transformer backends (NumPy, PyTorch, Triton, CUDA).
    Ensures a unified interface for training and inference.
    """

    @abstractmethod
    def forward(
        self,
        input_ids: np.ndarray,
        mask: Optional[np.ndarray] = None,
        use_cache: bool = False,
        cache_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass for the transformer.

        Args:
            input_ids: [Batch, Seq_Len] integer token IDs.
            mask: Causal mask [Seq_Len, Seq_Len].
            use_cache: Whether to use/update KV cache.
            cache_idx: Index of the current token for KV cache update.

        Returns:
            logits: [Batch, Seq_Len, Vocab_Size] the model output.
            cache: Dictionary containing intermediate values for backward pass.
        """
        pass

    @abstractmethod
    def backward(
        self, grad_logits: np.ndarray, cache: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Backward pass for the transformer.

        Args:
            grad_logits: [Batch, Seq_Len, Vocab_Size] gradient of loss w.r.t. logits.
            cache: Dictionary of intermediate values from the forward pass.

        Returns:
            grads: Dictionary of all parameter gradients.
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, np.ndarray]:
        """
        Returns all model parameters in canonical form.

        Returns:
            params: Dictionary of canonical name -> parameter values.
        """
        pass

    @abstractmethod
    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """
        Sets all model parameters using canonical names.

        Args:
            params: Dictionary of canonical name -> parameter values.
        """
        pass
