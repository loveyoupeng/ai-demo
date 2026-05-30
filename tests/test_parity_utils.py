import numpy as np
from typing import Any, Dict, Type, Tuple
from src.utils.backend_interface import BaseTransformerBackend

class ParityTester:
    """
    Utility to compare two Transformer backends for mathematical parity.
    """

    def __init__(self, baseline: BaseTransformerBackend, target: BaseTransformerBackend, tol: float = 1e-5):
        self.baseline = baseline
        self.target = target
        self.tol = tol

    def sync_params(self) -> None:
        """Synchronizes parameters from baseline to target."""
        params = self.baseline.get_params()
        self.target.set_params(params)

    def compare_forward(self, input_ids: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Compares forward pass results.
        Returns (is_equal, error_details).
        """
        logits_base, cache_base = self.baseline.forward(input_ids)
        logits_target, cache_target = self.target.forward(input_ids)

        diff = np.abs(logits_base - logits_target)
        max_diff = np.max(diff)
        
        is_equal = max_diff <= self.tol
        
        details = {
            "max_diff": max_diff,
            "logits_base_shape": logits_base.shape,
            "logits_target_shape": logits_target.shape
        }
        
        return is_equal, details

    def compare_backward(self, input_ids: np.ndarray, grad_logits: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Compares backward pass gradients.
        Returns (is_equal, error_details).
        """
        # Ensure params are synced
        self.sync_params()

        # Forward pass to get caches
        _, cache_base = self.baseline.forward(input_ids)
        _, cache_target = self.target.forward(input_ids)

        # Backward pass
        grads_base = self.baseline.backward(grad_logits, cache_base)
        grads_target = self.target.backward(grad_logits, cache_target)

        # Check if all keys exist in both
        all_keys = set(grads_base.keys()) | set(grads_target.keys())
        
        max_diff = 0.0
        for key in all_keys:
            if key not in grads_base or key not in grads_target:
                return False, {"error": f"Key mismatch: {key} not in both"}
            
            diff = np.abs(grads_base[key] - grads_target[key])
            max_diff = max(max_diff, np.max(diff))

        is_equal = max_diff <= self.tol
        
        details = {
            "max_diff": max_diff,
            "keys_compared": list(all_keys)
        }
        
        return is_equal, details
