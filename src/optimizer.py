import numpy as np
from typing import Dict, List, Any

class SGD:
    """
    Stochastic Gradient Descent optimizer.
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """
        Update parameters using gradients.
        """
        for key in params:
            if key in grads:
                params[key] -= self.learning_rate * grads[key]

class Adam:
    """
    Adam optimizer.
    """

    def __init__(
        self, 
        learning_rate: float = 0.001, 
        beta1: float = 0.9, 
        beta2: float = 0.999, 
        eps: float = 1e-8
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}
        self.t = 0

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """
        Update parameters using Adam update rule.
        """
        self.t += 1
        
        if not self.m:
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        for key in params:
            if key in grads:
                g = grads[key]
                
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g**2)
                
                m_hat = self.m[key] / (1 - self.beta1**self.t)
                v_hat = self.v[key] / (1 - self.beta2**self.t)
                
                params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
