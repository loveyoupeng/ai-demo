from __future__ import annotations

import pytest
import numpy as np
import torch
from src.model.numpy.layers import NumPyFeedForward
from src.model.pytorch.layers import PyTorchFeedForward
from src.core.registry import registry

class TestFeedForwardParity:
    def setup_method(self):
        self.embed_dim = 16
        self.dim_ff = 64
        self.numpy_ffn = NumPyFeedForward(self.embed_dim, self.dim_ff)
        self.pytorch_ffn = PyTorchFeedForward(self.embed_dim, self.dim_ff)
        
        # Sync parameters manually (NumPy internal names -> PyTorch internal names)
        with torch.no_grad():
            for k in ["w1", "b1", "w2", "b2"]:
                getattr(self.pytorch_ffn, k).copy_(torch.from_numpy(getattr(self.numpy_ffn, k)))

    def test_forward_parity(self):
        np.random.seed(42)
        x = np.random.randn(4, self.embed_dim).astype(np.float32)
        
        numpy_out = self.numpy_ffn.forward(x)
        pytorch_out = self.pytorch_ffn.forward(torch.from_numpy(x))
        
        np.testing.assert_allclose(numpy_out, pytorch_out.detach().numpy(), rtol=1e-5, atol=1e-5)

    def test_backward_w1_parity(self):
        np.random.seed(42)
        x = np.random.randn(4, self.embed_dim).astype(np.float32)
        grad_output = np.random.randn(4, self.embed_dim).astype(np.float32)
        
        self.numpy_ffn.forward(x)
        _, numpy_grads = self.numpy_ffn.backward(grad_output)
        
        self.pytorch_ffn.forward(torch.from_numpy(x))
        _, pytorch_grads = self.pytorch_ffn.backward(torch.from_numpy(grad_output))
        
        np.testing.assert_allclose(numpy_grads["w1"], pytorch_grads["ffn.w1"].detach().numpy(), rtol=1e-5, atol=1e-5)

    def test_backward_w2_parity(self):
        np.random.seed(42)
        x = np.random.randn(4, self.embed_dim).astype(np.float32)
        grad_output = np.random.randn(4, self.embed_dim).astype(np.float32)
        
        self.numpy_ffn.forward(x)
        _, numpy_grads = self.numpy_ffn.backward(grad_output)
        
        self.pytorch_ffn.forward(torch.from_numpy(x))
        _, pytorch_grads = self.pytorch_ffn.backward(torch.from_numpy(grad_output))
        
        np.testing.assert_allclose(numpy_grads["w2"], pytorch_grads["ffn.w2"].detach().numpy(), rtol=1e-5, atol=1e-5)

    def test_backward_b1_parity(self):
        np.random.seed(42)
        x = np.random.randn(4, self.embed_dim).astype(np.float32)
        grad_output = np.random.randn(4, self.embed_dim).astype(np.float32)
        
        self.numpy_ffn.forward(x)
        _, numpy_grads = self.numpy_ffn.backward(grad_output)
        
        self.pytorch_ffn.forward(torch.from_numpy(x))
        _, pytorch_grads = self.pytorch_ffn.backward(torch.from_numpy(grad_output))
        
        np.testing.assert_allclose(numpy_grads["b1"], pytorch_grads["ffn.b1"].detach().numpy(), rtol=1e-5, atol=1e-5)

    def test_backward_b2_parity(self):
        np.random.seed(42)
        x = np.random.randn(4, self.embed_dim).astype(np.float32)
        grad_output = np.random.randn(4, self.embed_dim).astype(np.float32)
        
        self.numpy_ffn.forward(x)
        _, numpy_grads = self.numpy_ffn.backward(grad_output)
        
        self.pytorch_ffn.forward(torch.from_numpy(x))
        _, pytorch_grads = self.pytorch_ffn.backward(torch.from_numpy(grad_output))
        
        np.testing.assert_allclose(numpy_grads["b2"], pytorch_grads["ffn.b2"].detach().numpy(), rtol=1e-5, atol=1e-5)

    def test_backward_x_parity(self):
        np.random.seed(42)
        x = np.random.randn(4, self.embed_dim).astype(np.float32)
        grad_output = np.random.randn(4, self.embed_dim).astype(np.float32)
        
        self.numpy_ffn.forward(x)
        dx_numpy, _ = self.numpy_ffn.backward(grad_output)
        
        self.pytorch_ffn.forward(torch.from_numpy(x))
        dx_pytorch, _ = self.pytorch_ffn.backward(torch.from_numpy(grad_output))
        
        np.testing.assert_allclose(dx_numpy, dx_pytorch.detach().numpy(), rtol=1e-5, atol=1e-5)
