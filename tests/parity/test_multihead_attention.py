from __future__ import annotations

import pytest
import numpy as np
import torch
from model.attention import MultiHeadAttention as NumPyAttention
from model.pytorch.attention import PyTorchMultiHeadAttention as PyTorchAttention

class TestMultiHeadAttentionParity:
    def setup_method(self):
        self.embed_dim = 64
        self.num_heads = 4
        self.head_dim = self.embed_dim // self.num_heads
        self.numpy_mha = NumPyAttention(self.embed_dim, self.num_heads)
        self.pytorch_mha = PyTorchAttention(self.embed_dim, self.num_heads)
        self.seq_len = 16
        self.batch_size = 2
        
        # Sync parameters: W_q/W_k/W_v -> qkv.W_q/qkv.W_k/qkv.W_v, W_o -> o.W_o
        numpy_params = self.numpy_mha.get_params()
        
        # Map NumPy internal names -> PyTorch attribute names
        attr_mapping = {
            "W_q": "W_q",
            "W_k": "W_k",
            "W_v": "W_v",
            "W_o": "W_o",
        }
        for k, v in numpy_params.items():
            attr_name = attr_mapping.get(k)
            if attr_name:
                with torch.no_grad():
                    getattr(self.pytorch_mha, attr_name).copy_(torch.from_numpy(v))

    def test_forward_parity(self):
        """Forward pass should match between NumPy and PyTorch."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float32)
        mask = np.tril(np.ones((self.seq_len, self.seq_len)))
        
        numpy_out, self.numpy_cache = self.numpy_mha.forward(x, mask)
        pytorch_out, self.pytorch_cache = self.pytorch_mha.forward(torch.from_numpy(x), torch.from_numpy(mask))
        
        np.testing.assert_allclose(numpy_out, pytorch_out.detach().numpy(), rtol=1e-4, atol=1e-4)

    def test_backward_q_parity(self):
        """Backward pass for W_q should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float32)
        mask = np.tril(np.ones((self.seq_len, self.seq_len)))
        
        _, numpy_cache = self.numpy_mha.forward(x, mask)
        grad_output = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float32)
        cache = {
            "context": numpy_cache["context"],
            "Q": numpy_cache["Q"],
            "K": numpy_cache["K"],
            "V": numpy_cache["V"],
            "attn_weights": numpy_cache["attn_weights"],
            "x": x,
        }
        _, numpy_grads = self.numpy_mha.backward(x, grad_output, mask, **{k: v for k, v in cache.items() if k != "x"})
        
        self.pytorch_mha.forward(torch.from_numpy(x), torch.from_numpy(mask))
        _, pytorch_grads = self.pytorch_mha.backward(torch.from_numpy(grad_output), torch.from_numpy(mask))
        
        np.testing.assert_allclose(numpy_grads["W_q"], pytorch_grads["qkv.W_q"].detach().numpy(), rtol=1e-4, atol=1e-4)

    def test_backward_k_parity(self):
        """Backward pass for W_k should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float32)
        mask = np.tril(np.ones((self.seq_len, self.seq_len)))
        _, numpy_cache = self.numpy_mha.forward(x, mask)
        grad_output = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float32)
        cache = {
            "context": numpy_cache["context"],
            "Q": numpy_cache["Q"],
            "K": numpy_cache["K"],
            "V": numpy_cache["V"],
            "attn_weights": numpy_cache["attn_weights"],
            "x": x,
        }
        _, numpy_grads = self.numpy_mha.backward(x, grad_output, mask, **{k: v for k, v in cache.items() if k != "x"})
        
        self.pytorch_mha.forward(torch.from_numpy(x), torch.from_numpy(mask))
        _, pytorch_grads = self.pytorch_mha.backward(torch.from_numpy(grad_output), torch.from_numpy(mask))
        
        np.testing.assert_allclose(numpy_grads["W_k"], pytorch_grads["qkv.W_k"].detach().numpy(), rtol=1e-4, atol=1e-4)

    def test_backward_v_parity(self):
        """Backward pass for W_v should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float32)
        mask = np.tril(np.ones((self.seq_len, self.seq_len)))
        _, numpy_cache = self.numpy_mha.forward(x, mask)
        grad_output = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float32)
        cache = {
            "context": numpy_cache["context"],
            "Q": numpy_cache["Q"],
            "K": numpy_cache["K"],
            "V": numpy_cache["V"],
            "attn_weights": numpy_cache["attn_weights"],
            "x": x,
        }
        _, numpy_grads = self.numpy_mha.backward(x, grad_output, mask, **{k: v for k, v in cache.items() if k != "x"})
        
        self.pytorch_mha.forward(torch.from_numpy(x), torch.from_numpy(mask))
        _, pytorch_grads = self.pytorch_mha.backward(torch.from_numpy(grad_output), torch.from_numpy(mask))
        
        np.testing.assert_allclose(numpy_grads["W_v"], pytorch_grads["qkv.W_v"].detach().numpy(), rtol=1e-4, atol=1e-4)

    def test_backward_o_parity(self):
        """Backward pass for W_o should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float32)
        mask = np.tril(np.ones((self.seq_len, self.seq_len)))
        _, numpy_cache = self.numpy_mha.forward(x, mask)
        grad_output = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float32)
        cache = {
            "context": numpy_cache["context"],
            "Q": numpy_cache["Q"],
            "K": numpy_cache["K"],
            "V": numpy_cache["V"],
            "attn_weights": numpy_cache["attn_weights"],
            "x": x,
        }
        _, numpy_grads = self.numpy_mha.backward(x, grad_output, mask, **{k: v for k, v in cache.items() if k != "x"})
        
        self.pytorch_mha.forward(torch.from_numpy(x), torch.from_numpy(mask))
        _, pytorch_grads = self.pytorch_mha.backward(torch.from_numpy(grad_output), torch.from_numpy(mask))
        
        np.testing.assert_allclose(numpy_grads["W_o"], pytorch_grads["o.W_o"].detach().numpy(), rtol=1e-4, atol=1e-4)

    def test_backward_x_parity(self):
        """Backward pass for input x should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float32)
        mask = np.tril(np.ones((self.seq_len, self.seq_len)))
        _, numpy_cache = self.numpy_mha.forward(x, mask)
        grad_output = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float32)
        cache = {
            "context": numpy_cache["context"],
            "Q": numpy_cache["Q"],
            "K": numpy_cache["K"],
            "V": numpy_cache["V"],
            "attn_weights": numpy_cache["attn_weights"],
            "x": x,
        }
        dx_numpy, _ = self.numpy_mha.backward(x, grad_output, mask, **{k: v for k, v in cache.items() if k != "x"})
        
        self.pytorch_mha.forward(torch.from_numpy(x), torch.from_numpy(mask))
        dx_pytorch, _ = self.pytorch_mha.backward(torch.from_numpy(grad_output), torch.from_numpy(mask))
        
        np.testing.assert_allclose(dx_numpy, dx_pytorch.detach().numpy(), rtol=1e-4, atol=1e-4)

    def test_no_mask_parity(self):
        """Forward pass without mask should match."""
        np.random.seed(42)
        x = np.random.randn(self.batch_size, self.seq_len, self.embed_dim).astype(np.float32)
        
        numpy_out, _ = self.numpy_mha.forward(x)
        pytorch_out, _ = self.pytorch_mha.forward(torch.from_numpy(x))
        
        np.testing.assert_allclose(numpy_out, pytorch_out.detach().numpy(), rtol=1e-4, atol=1e-4)
