"""Test that KV cache stores raw K/V before RoPE is applied."""

import torch

from model.pytorch.attention import PyTorchMultiHeadAttention
from model.pytorch.attention_kvcache import PyTorchTurboQuantCache


def test_kv_cache_stores_raw_k():
    """KV cache append must store raw K (x @ W_k), not post-RoPE K.

    RoPE is applied to Q/K for attention computation only.
    The cache should contain the pre-rotation values so that
    cross-backend parity holds.
    """
    embed_dim = 64
    num_heads = 4
    head_dim = embed_dim // num_heads

    mha = PyTorchMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    cache = PyTorchTurboQuantCache(
        embed_dim=embed_dim, num_heads=num_heads,
        max_seq_len=10, head_dim=head_dim, batch_size=1,
    )

    x = torch.randn(1, 3, embed_dim)
    mask = torch.tril(torch.ones((3, 3), dtype=torch.float32))

    out, _ = mha(x, mask=mask, kv_cache=cache)

    k_from_cache, v_from_cache = cache.get_kv()

    # Expected: raw K without RoPE
    K_expected = torch.matmul(x, mha.W_k)
    K_expected = K_expected.reshape(1, 3, num_heads, head_dim).transpose(1, 2)

    # The cache should store RAW K, not post-RoPE K
    # Currently: cache stores K AFTER RoPE -> FAILS
    # After fix: cache stores K BEFORE RoPE -> PASSES
    assert torch.allclose(k_from_cache, K_expected, rtol=1e-4, atol=1e-4), (
        f"KV cache stores post-RoPE K. Max diff: {torch.max(torch.abs(k_from_cache - K_expected)):.6e}"
    )
