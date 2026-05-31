import pytest
import numpy as np
import torch
from backends.numpy.numpy_backend import NumPyBackend
from backends.pytorch.pytorch_backend import PyTorchBackend
from tests.test_parity_utils import ParityTester

@pytest.fixture
def configuration():
    return {
        "vocab_size": 32,
        "embed_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "num_experts": 4,
        "max_seq_len": 32,
    }

def test_pytorch_numpy_parity(configuration):
    """
    Test that PyTorchBackend produces results identical to NumPyBackend
    within a specified tolerance.
    """
    numpy_backend = NumPyBackend(**configuration)
    pytorch_backend = PyTorchBackend(**configuration)
    
    tester = ParityTester(numpy_backend, pytorch_backend, tol=1e-5)
    
    # Sync parameters so they start from the same state
    tester.sync_params()
    
    # Test input
    batch_size = 2
    seq_len = 16
    input_ids = np.random.randint(0, configuration["vocab_size"], (batch_size, seq_len)).astype(np.int32)
    
    # 1. Test Forward Pass Parity
    is_equal_fwd, details_fwd = tester.compare_forward(input_ids)
    assert is_equal_fwd, f"Forward pass mismatch: {details_fwd}"
    
    # 2. Test Backward Pass Parity
    # Create a dummy gradient for logits [Batch, Seq, Vocab]
    grad_logits = np.random.randn(batch_size, seq_len, configuration["vocab_size"]).astype(np.float32)
    
    is_equal_bwd, details_bwd = tester.compare_backward(input_ids, grad_logits)
    assert is_equal_bwd, f"Backward pass mismatch: {details_bwd}"

if __name__ == "__main__":
    pytest.main([__file__])
