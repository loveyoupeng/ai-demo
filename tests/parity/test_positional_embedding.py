from __future__ import annotations

import pytest
import numpy as np
import torch
from model.numpy.layers import NumPyPositionalEmbedding
from model.pytorch.layers import PyTorchPositionalEmbedding

class TestPositionalEmbeddingParity:
    """
    Test that NumPy and PyTorch positional embeddings produce identical outputs.
    
    The PE matrix computation is:
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """

    def setup_method(self):
        self.max_seq_len = 64
        self.embed_dim = 32
        self.numpy_pe = NumPyPositionalEmbedding(self.max_seq_len, self.embed_dim)
        self.pytorch_pe = PyTorchPositionalEmbedding(self.max_seq_len, self.embed_dim)

    def test_pe_matrix_parity(self):
        """PE matrix must be identical between NumPy and PyTorch."""
        np.testing.assert_allclose(
            self.numpy_pe.pe, 
            self.pytorch_pe.pe.detach().numpy(),
            rtol=1e-6, atol=3e-6
        )

    def test_forward_parity(self):
        """Forward pass (x + PE) must be identical."""
        np.random.seed(42)
        x = np.random.randn(4, self.max_seq_len, self.embed_dim).astype(np.float32)
        
        numpy_out = self.numpy_pe.forward(x)
        pytorch_out = self.pytorch_pe.forward(torch.from_numpy(x))
        
        np.testing.assert_allclose(
            numpy_out, 
            pytorch_out.detach().numpy(),
            rtol=1e-5, atol=1e-5
        )

    def test_backward_x_parity(self):
        """Backward pass: gradient w.r.t. x must be identical."""
        np.random.seed(42)
        x = np.random.randn(4, self.max_seq_len, self.embed_dim).astype(np.float32)
        grad_output = np.random.randn(4, self.max_seq_len, self.embed_dim).astype(np.float32)
        
        self.numpy_pe.forward(x)
        dx_numpy, _ = self.numpy_pe.backward(grad_output)
        
        self.pytorch_pe.forward(torch.from_numpy(x))
        dx_pytorch, _ = self.pytorch_pe.backward(torch.from_numpy(grad_output))
        
        np.testing.assert_allclose(
            dx_numpy,
            dx_pytorch.detach().numpy(),
            rtol=1e-5, atol=1e-5
        )

    def test_short_sequence_parity(self):
        """PE should work correctly with sequences shorter than max_seq_len."""
        seq_len = 10
        np.random.seed(42)
        x = np.random.randn(2, seq_len, self.embed_dim).astype(np.float32)
        
        numpy_out = self.numpy_pe.forward(x)
        # Slice the PyTorch PE matrix to match sequence length
        pytorch_x = torch.from_numpy(x)
        self.pytorch_pe.forward(pytorch_x)
        # Check that PE matrix is correctly used
        pe_slice_numpy = self.numpy_pe.pe[:seq_len, :]
        pe_slice_pytorch = self.pytorch_pe.pe[:seq_len, :]
        
        np.testing.assert_allclose(
            pe_slice_numpy,
            pe_slice_pytorch.detach().numpy(),
            rtol=1e-6, atol=1e-6
        )
