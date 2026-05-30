import pytest
import numpy as np
from typing import Any, Dict, Optional, Tuple
from src.utils.backend_interface import BaseTransformerBackend


class MockBackend(BaseTransformerBackend):
    """A minimal mock implementation for testing the interface contract."""
    def __init__(self):
        self.params = {"weight": np.array([1.0])}
        self.grads = {"weight": np.array([0.0])}

    def forward(
        self,
        input_ids: np.ndarray,
        mask: Optional[np.ndarray] = None,
        use_cache: bool = False,
        cache_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Returns dummy logits [Batch, Seq_Len, Vocab] and an empty cache
        batch_size, seq_len = input_ids.shape
        logits = np.zeros((batch_size, seq_len, 1))
        return logits, {}

    def backward(
        self, grad_logits: np.ndarray, cache: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        return self.grads

    def get_params(self) -> Dict[str, np.ndarray]:
        return self.params

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        self.params = params


def test_cannot_instantiate_abstract_class():
    """Invariant: The BaseTransformerBackend should not be instantiable directly."""
    with pytest.raises(TypeError) as excinfo:
        BaseTransformerBackend()
    assert "Can't instantiate abstract class BaseTransformerBackend" in str(excinfo.value)


def test_mock_backend_contract():
    """Contract: MockBackend should satisfy the interface signature and return types."""
    backend = MockBackend()
    
    # Test forward
    input_ids = np.array([[1, 2, 3]], dtype=np.int32)
    logits, cache = backend.forward(input_ids)
    
    assert isinstance(logits, np.ndarray)
    assert isinstance(cache, dict)
    assert logits.shape == (1, 3, 1)
    
    # Test backward
    grad_logits = np.ones((1, 3, 1))
    grads = backend.backward(grad_logits, cache)
    assert isinstance(grads, dict)
    assert "weight" in grads

    # Test params
    params = backend.get_params()
    assert params == {"weight": np.array([1.0])}
    
    new_params = {"weight": np.array([2.0])}
    backend.set_params(new_params)
    assert backend.get_params() == new_params
