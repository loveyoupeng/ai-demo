from __future__ import annotations

import pytest
import numpy as np
import torch
from src.model.numpy.layers import NumPyTokenEmbedding
from src.model.pytorch.layers import PyTorchTokenEmbedding
from src.core.registry import registry

def test_token_embedding_parity():
    vocab_size = 32
    embed_dim = 64
    batch_size = 2
    seq_len = 16

    # 1. Initialize
    np_emb = NumPyTokenEmbedding(vocab_size, embed_dim)
    pt_emb = PyTorchTokenEmbedding(vocab_size, embed_dim)

    # 2. Sync parameters via Canonical Names
    # Get NumPy params (Internal -> Canonical is handled by NumPy wrapper if used, 
    # but here we do it manually for the layer)
    np_params = np_emb.get_params()
    # For the sake of this simple test, we'll assume the internal names are already what we want 
    # or we use the registry.
    # In a real backend, get_params() returns canonical names.
    
    # Let's pretend we are the registry for this test
    # mapping np: "weights" -> "embedding.weights"
    registry.register("numpy", "embedding.weights", "weights")
    registry.register("pytorch", "embedding.weights", "embedding.weight")

    # Sync: NumPy -> PyTorch
    # Get NumPy params in canonical form
    np_canonical_params = {}
    for k, v in np_params.items():
        canonical_k = registry.get_canonical_name("numpy", k)
        np_canonical_params[canonical_k] = v

    # Apply to PyTorch
    pt_emb.set_params(np_canonical_params)

    # 3. Test Forward Pass
    input_ids_np = np.random.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int32)
    input_ids_pt = torch.from_numpy(input_ids_np)

    out_np = np_emb.forward(input_ids_np)
    out_pt = pt_emb.forward(input_ids_pt).detach().numpy()

    assert np.allclose(out_np, out_pt, atol=1e-5), f"Forward mismatch: np={out_np.mean()}, pt={out_pt.mean()}"

    # 4. Test Backward Pass
    grad_out_np = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
    grad_out_pt = torch.from_numpy(grad_out_np)

    dx_np, grads_np = np_emb.backward(grad_out_np)
    dx_pt, grads_pt = pt_emb.backward(grad_out_pt)

    # Convert pt grads (canonical) to np grads (internal) for comparison
    # (Actually we compare the weights gradient)
    # pt_grads: {"embedding.weights": Tensor}
    # np_grads: {"weights": array}
    
    # Sync pt grads to canonical, then to np internal
    pt_canonical_grads = {}
    for k, v in grads_pt.items():
        canonical_k = registry.get_canonical_name("pytorch", k)
        pt_canonical_grads[canonical_k] = v.detach().numpy()

    # Sync np grads to canonical
    np_canonical_grads = {}
    for k, v in grads_np.items():
        canonical_k = registry.get_canonical_name("numpy", k)
        np_canonical_grads[canonical_k] = v

    # Compare canonical grads
    assert np.allclose(np_canonical_grads["embedding.weights"], pt_canonical_grads["embedding.weights"], atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])
