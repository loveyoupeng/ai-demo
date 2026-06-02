from __future__ import annotations

import pytest
import numpy as np
import torch
from src.model.numpy.layers import NumPyLayerNorm
from src.model.pytorch.layers import PyTorchLayerNorm
from src.core.registry import registry

@pytest.fixture
def registry_layer_norm():
    # Reset registry for clean state
    registry.clear()

class TestLayerNormParity:
    def setup_method(self):
        self.embed_dim = 16
        self.eps = 1e-5
        self.numpy_ln = NumPyLayerNorm(self.embed_dim, self.eps)
        self.pytorch_ln = PyTorchLayerNorm(self.embed_dim, self.eps)
        
        # Sync parameters
        numpy_params = self.numpy_ln.get_params()
        self.pytorch_ln.set_params(numpy_params)

    def test_forward_parity(self):
        # Create random input
        np.random.seed(42)
        x = np.random.randn(4, self.embed_dim).astype(np.float32)
        
        numpy_out = self.numpy_ln.forward(x)
        pytorch_out = self.pytorch_ln.forward(torch.from_numpy(x))
        
        np.testing.assert_allclose(numpy_out, pytorch_out.detach().numpy(), rtol=1e-5, atol=1e-5)

    def test_backward_gamma_parity(self):
        np.random.seed(42)
        x = np.random.randn(4, self.embed_dim).astype(np.float32)
        grad_output = np.random.randn(4, self.embed_dim).astype(np.float32)
        
        self.numpy_ln.forward(x)
        _, numpy_grads = self.numpy_ln.backward(grad_output)
        
        self.pytorch_ln.forward(torch.from_numpy(x))
        _, pytorch_grads = self.pytorch_ln.backward(torch.from_numpy(grad_output))
        
        np.testing.assert_allclose(numpy_grads["gamma"], pytorch_grads["ln.gamma"].detach().numpy(), rtol=1e-5, atol=1e-5)

    def test_backward_beta_parity(self):
        np.random.seed(42)
        x = np.random.randn(4, self.embed_dim).astype(np.float32)
        grad_output = np.random.randn(4, self.embed_dim).astype(np.float32)
        
        self.numpy_ln.forward(x)
        _, numpy_grads = self.numpy_ln.backward(grad_output)
        
        self.pytorch_ln.forward(torch.from_numpy(x))
        _, pytorch_grads = self.pytorch_ln.backward(torch.from_numpy(grad_output))
        
        np.testing.assert_allclose(numpy_grads["beta"], pytorch_grads["ln.beta"].detach().numpy(), rtol=1e-5, atol=1e-5)

    def test_backward_x_parity(self):
        np.random.seed(42)
        x = np.random.randn(4, self.embed_dim).astype(np.float32)
        grad_output = np.random.randn(4, self.embed_dim).astype(np.float32)
        
        self.numpy_ln.forward(x)
        dx_numpy, _ = self.numpy_ln.backward(grad_output)
        
        self.pytorch_ln.forward(torch.from_numpy(x))
        dx_pytorch, _ = self.pytorch_ln.backward(torch.from_numpy(grad_output))
        
        np.testing.assert_allclose(dx_numpy, dx_pytorch.detach().numpy(), rtol=1e-5, atol=1e-5)
