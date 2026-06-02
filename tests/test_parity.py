from __future__ import annotations

import pytest
import numpy as np
from backends.numpy.numpy_backend import NumPyBackend
from tests.test_parity_utils import ParityTester

class TestParityTesterLogic:
    """
    Tests the internal logic of the ParityTester using minimal mock-like objects
    to avoid the complexity of the full Transformer parameter tree.
    """

    class MinimalBackend:
        def __init__(self, val=1.0):
            self.params = {"a": np.array([val], dtype=np.float64)}
        def forward(self, input_ids, mask=None, use_cache=False, cache_idx=None):
            return np.zeros((input_ids.shape[0], input_ids.shape[1], 1)), {}
        def backward(self, grad_logits, cache):
            return {"a": np.array([0.0], dtype=np.float64)}
        def get_params(self): return self.params
        def set_params(self, p): self.params = p

    def test_sync_params(self):
        base = self.MinimalBackend(val=1.0)
        target = self.MinimalBackend(val=2.0)
        tester = ParityTester(base, target)
        
        tester.sync_params()
        assert np.allclose(base.get_params()["a"], target.get_params()["a"])
        assert target.get_params()["a"] == 1.0

    def test_compare_forward_mismatch(self):
        base = self.MinimalBackend(val=1.0)
        target = self.MinimalBackend(val=1.0)
        tester = ParityTester(base, target)
        
        def target_forward_ones(self, input_ids, mask=None, use_cache=False, cache_idx=None):
            return np.ones((input_ids.shape[0], input_ids.shape[1], 1)), {}
        
        import types
        target.forward = types.MethodType(target_forward_ones, target)
        
        input_ids = np.array([[1]], dtype=np.int32)
        is_equal, details = tester.compare_forward(input_ids)
        
        assert bool(is_equal) is False
        assert details["max_diff"] == 1.0

    def test_compare_backward_mismatch(self):
        base = self.MinimalBackend(val=1.0)
        target = self.MinimalBackend(val=1.0)
        tester = ParityTester(base, target)
        
        def target_backward_ones(self, grad_logits, cache):
            return {"a": np.array([1.0], dtype=np.float64)}
        
        import types
        target.backward = types.MethodType(target_backward_ones, target)
        
        input_ids = np.array([[1]], dtype=np.int32)
        grad_logits = np.ones((1, 1, 1))
        
        is_equal, details = tester.compare_backward(input_ids, grad_logits)
        
        assert bool(is_equal) is False
        assert details["max_diff"] == 1.0

    def test_key_mismatch(self):
        base = self.MinimalBackend()
        target = self.MinimalBackend()
        
        # Fix: Use a proper method wrapper or explicit lambda that handles all args
        # even if it ignores them.
        def target_backward_mismatch(self, grad_logits, cache):
            return {"b": np.array([0.0])}
            
        import types
        target.backward = types.MethodType(target_backward_mismatch, target)
    
        tester = ParityTester(base, target)
        input_ids = np.array([[1]], dtype=np.int32)
        grad_logits = np.ones((1, 1, 1))
        
        is_equal, details = tester.compare_backward(input_ids, grad_logits)
        assert is_equal is False
        assert "error" in details
        assert "Key mismatch" in details["error"]
