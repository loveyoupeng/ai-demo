from __future__ import annotations

"""Phase C: Transformer Block Tests — KV cache propagation to MHA."""
import torch

from model.pytorch.attention import PyTorchMultiHeadAttention
from model.pytorch.attention_kvcache import PyTorchTurboQuantCache
from model.pytorch.moe import PyTorchMoELayer
from model.pytorch.transformer import PyTorchTransformerBlock


# C1: test_block_no_cache_train — backward compatibility
def test_block_no_cache_train():
    """Block without cache → forward + backward work as before."""
    torch.manual_seed(42)
    mha = PyTorchMultiHeadAttention(embed_dim=64, num_heads=4)
    moe = PyTorchMoELayer(embed_dim=64, num_experts=4)
    block = PyTorchTransformerBlock(embed_dim=64, mha=mha, moe=moe)

    x = torch.randn(2, 8, 64)
    mask = torch.tril(torch.ones((8, 8), dtype=torch.float32))

    out, cache = block(x, mask=mask, kv_cache=None)
    assert out.shape == (2, 8, 64)

    # Backward
    torch.manual_seed(42)
    grad = torch.ones_like(out)
    dx, grads = block.backward(grad, cache)
    assert dx.shape == (2, 8, 64)
    assert "ln1.weight" in grads
    assert "mha.qkv.W_q" in grads
    # Check at least one MoE gradient key exists (which experts are active
    # depends on the router weights + input data)
    moe_keys = [k for k in grads if k.startswith("moe.")]
    assert len(moe_keys) > 0
    assert "moe.router.w" in grads


# C2: test_block_cache_accumulation — append through block
def test_block_cache_accumulation():
    """Block forward 3 tokens, then 2 more → MHA cache has 5 total."""
    embed_dim = 64
    num_heads = 4
    head_dim = embed_dim // num_heads

    mha = PyTorchMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    moe = PyTorchMoELayer(embed_dim=embed_dim, num_experts=4)
    block = PyTorchTransformerBlock(embed_dim=embed_dim, mha=mha, moe=moe)

    cache = PyTorchTurboQuantCache(
        embed_dim=embed_dim, num_heads=num_heads,
        max_seq_len=10, head_dim=head_dim, batch_size=1,
    )

    # First: feed 3 tokens
    x3 = torch.randn(1, 3, embed_dim)
    m3 = torch.tril(torch.ones((3, 3), dtype=torch.float32))
    out1, _ = block(x3, mask=m3, kv_cache=cache)
    assert out1.shape == (1, 3, embed_dim)
    assert cache.size == 3

    # Second: feed 2 more tokens (autoregressive)
    x2 = torch.randn(1, 2, embed_dim)
    m2 = torch.tril(torch.ones((2, 2), dtype=torch.float32))
    out2, _ = block(x2, mask=m2, kv_cache=cache)
    assert out2.shape == (1, 2, embed_dim)
    assert cache.size == 5


# C3: test_block_output_shape_with_cache — output = input shape
def test_block_output_shape_with_cache():
    """[B,1,D] input with cache → [B,1,D] output (output = current token count, not cache size)."""
    embed_dim = 64
    num_heads = 4
    head_dim = embed_dim // num_heads

    mha = PyTorchMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    moe = PyTorchMoELayer(embed_dim=embed_dim, num_experts=4)
    block = PyTorchTransformerBlock(embed_dim=embed_dim, mha=mha, moe=moe)

    cache = PyTorchTurboQuantCache(
        embed_dim=embed_dim, num_heads=num_heads,
        max_seq_len=20, head_dim=head_dim, batch_size=1,
    )

    # Fill cache to 4 tokens
    for _ in range(4):
        x = torch.randn(1, 1, embed_dim)
        m = torch.tril(torch.ones((1, 1), dtype=torch.float32))
        block(x, mask=m, kv_cache=cache)

    assert cache.size == 4

    # Now forward with 1 token, cache has 4+1=5
    x1 = torch.randn(1, 1, embed_dim)
    m1 = torch.tril(torch.ones((1, 5), dtype=torch.float32))  # causal: 1 vs 5 total
    out1, _ = block(x1, mask=m1, kv_cache=cache)

    # Output shape matches Q (current token) = 1, not cache (5)
    assert out1.shape == (1, 1, embed_dim)
    assert cache.size == 5


# C4: test_block_reset_clears_cache — reset works
def test_block_reset_clears_cache():
    """Cache reset after appending tokens clears all history."""
    embed_dim = 64
    num_heads = 4
    head_dim = embed_dim // num_heads

    mha = PyTorchMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
    moe = PyTorchMoELayer(embed_dim=embed_dim, num_experts=4)
    block = PyTorchTransformerBlock(embed_dim=embed_dim, mha=mha, moe=moe)

    cache = PyTorchTurboQuantCache(
        embed_dim=embed_dim, num_heads=num_heads,
        max_seq_len=10, head_dim=head_dim, batch_size=1,
    )

    # Fill cache
    for _ in range(3):
        x = torch.randn(1, 1, embed_dim)
        m = torch.tril(torch.ones((1, 1), dtype=torch.float32))
        block(x, mask=m, kv_cache=cache)
    assert cache.size == 3

    # Reset
    cache.reset()
    assert cache.size == 0
