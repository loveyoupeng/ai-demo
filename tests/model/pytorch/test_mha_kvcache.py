from __future__ import annotations

"""Phase B: MHA Integration Tests — KV cache through forward() and backward()."""
import torch
import numpy as np

from model.pytorch.attention import PyTorchMultiHeadAttention
from model.pytorch.attention_kvcache import PyTorchTurboQuantCache


# B1: test_mha_no_cache — backward compatibility
def test_mha_no_cache():
    """Forward without kv_cache works exactly as before (no cache wired).

    Shape: [B,L,D] → [B,L,D]
    """
    mha = PyTorchMultiHeadAttention(embed_dim=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    mask = torch.tril(torch.ones((8, 8), dtype=torch.float32))

    output, cache = mha(x, mask=mask, kv_cache=None)
    assert output.shape == (2, 8, 64)
    # Backward must work
    grad = torch.ones_like(output)
    dx, grads = mha.backward(grad, mask=mask)
    assert dx.shape == (2, 8, 64)
    assert "qkv.W_q" in grads


# B2: test_mha_single_token_append — accumulation 1→2→3
def test_mha_single_token_append():
    """Feed one token at a time, cache grows monotonically."""
    embed_dim = 64
    num_heads = 4
    head_dim = embed_dim // num_heads

    mha = PyTorchMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    cache = PyTorchTurboQuantCache(
        embed_dim=embed_dim, num_heads=num_heads,
        max_seq_len=10, head_dim=head_dim, batch_size=1,
    )

    # Token 0: append [1,1,1,hdim] → cache has 1 token in get_kv()
    x0 = torch.randn(1, 1, embed_dim)
    mask0 = torch.tril(torch.ones((1, 1), dtype=torch.float32))
    out0, _ = mha(x0, mask=mask0, kv_cache=cache)
    k0, v0 = cache.get_kv()
    assert cache.size == 1
    assert k0.shape == (1, num_heads, 1, head_dim)
    assert out0.shape == (1, 1, embed_dim)

    # Token 1: append [1,1,1,hdim] → cache has 2 tokens
    x1 = torch.randn(1, 1, embed_dim)
    mask1 = torch.tril(torch.ones((1, 1), dtype=torch.float32))
    out1, _ = mha(x1, mask=mask1, kv_cache=cache)
    k1, v1 = cache.get_kv()
    assert cache.size == 2
    assert k1.shape == (1, num_heads, 2, head_dim)
    assert out1.shape == (1, 1, embed_dim)

    # Token 2: append [1,1,1,hdim] → cache has 3 tokens
    x2 = torch.randn(1, 1, embed_dim)
    mask2 = torch.tril(torch.ones((1, 1), dtype=torch.float32))
    out2, _ = mha(x2, mask=mask2, kv_cache=cache)
    k2, v2 = cache.get_kv()
    assert cache.size == 3
    assert k2.shape == (1, num_heads, 3, head_dim)
    assert out2.shape == (1, 1, embed_dim)


# B3: test_mha_batch_append_3 — append 3 tokens at once
def test_mha_batch_append_3():
    """Append 3 tokens in one call, get_kv() returns all 3."""
    embed_dim = 64
    num_heads = 4
    head_dim = embed_dim // num_heads

    mha = PyTorchMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    cache = PyTorchTurboQuantCache(
        embed_dim=embed_dim, num_heads=num_heads,
        max_seq_len=10, head_dim=head_dim, batch_size=1,
    )

    x3 = torch.randn(1, 3, embed_dim)
    mask3 = torch.tril(torch.ones((3, 3), dtype=torch.float32))
    out, _ = mha(x3, mask=mask3, kv_cache=cache)

    assert cache.size == 3
    k, v = cache.get_kv()
    assert k.shape == (1, num_heads, 3, head_dim)
    assert v.shape == (1, num_heads, 3, head_dim)
    assert out.shape == (1, 3, embed_dim)

    # Verify appended K matches what MHA computed
    K_computed = torch.matmul(x3, mha.W_k)
    K_computed = K_computed.reshape(1, 3, num_heads, head_dim).transpose(1, 2)
    assert torch.allclose(k[0], K_computed[0])


# B4: test_mha_cache_output_shape — output shape = input shape
def test_mha_cache_output_shape():
    """[B,L,D] input with cache → [B,L,D] output (output depends on Q seq_len, not cache)."""
    embed_dim = 64
    num_heads = 4
    head_dim = embed_dim // num_heads

    mha = PyTorchMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    cache = PyTorchTurboQuantCache(
        embed_dim=embed_dim, num_heads=num_heads,
        max_seq_len=20, head_dim=head_dim, batch_size=1,
    )

    # First, fill cache to 5 tokens
    for i in range(5):
        x_fill = torch.randn(1, 1, embed_dim)
        m = torch.tril(torch.ones((1, 1), dtype=torch.float32))
        mha(x_fill, mask=m, kv_cache=cache)

    # Now forward with 3-token batch (new tokens), cache has 5 old + 3 new = 8
    x3 = torch.randn(1, 3, embed_dim)

    # Build causal mask for Q_Len=3, K_Len=8 (autoregressive)
    # Each query position can attend to all previous + current.
    # For autoregressive: causal mask of size [L_current, L_total] with tril
    m3 = torch.tril(torch.ones((3, 8), dtype=torch.float32))
    out, _ = mha(x3, mask=m3, kv_cache=cache)

    # Output shape matches input Q seq_len (3), NOT total cache size (8)
    assert out.shape == (1, 3, embed_dim)
    assert cache.size == 8  # accumulated
