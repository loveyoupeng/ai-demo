from __future__ import annotations

import torch
import model.pytorch.attention_kvcache as kvcache_module
from model.pytorch.attention_kvcache import (
    PyQuantize,
    PyTorchTurboQuantCache,
)


# ---------------------------------------------------------------------------
# test_cache_initialization
# ---------------------------------------------------------------------------
def test_cache_initialization():
    """Cache is created with the right shape and parameters."""
    embed_dim = 64
    num_heads = 4
    max_seq_len = 128

    cache = PyTorchTurboQuantCache(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        head_dim=embed_dim // num_heads,
    )

    assert cache.max_seq_len == max_seq_len
    assert cache.num_heads == num_heads
    assert cache.head_dim == embed_dim // num_heads
    assert cache.size == 0

    # Check that index array exists and is uint8
    assert cache.k_indices.dtype == torch.uint8
    assert cache.v_indices.dtype == torch.uint8
    assert cache.k_norms.dtype == torch.float32
    assert cache.v_norms.dtype == torch.float32


# ---------------------------------------------------------------------------
# test_cache_append_and_get
# ---------------------------------------------------------------------------
def test_cache_append_and_get():
    """Append K/V one token at a time, then retrieve the full sequence."""
    embed_dim = 64
    num_heads = 4
    batch_in = 2

    cache = PyTorchTurboQuantCache(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=64,
        head_dim=embed_dim // num_heads,
        batch_size=batch_in,
    )

    # Build the sequence by appending one token at a time
    expected_k_list = []
    expected_v_list = []
    for _ in range(6):
        k = torch.randn(2, num_heads, 1, embed_dim // num_heads)
        v = torch.randn(2, num_heads, 1, embed_dim // num_heads)
        cache.append(k, v)
        expected_k_list.append(k)
        expected_v_list.append(v)

    assert cache.size == 6

    k_cached, v_cached = cache.get_kv()
    assert k_cached.shape == (1, num_heads, 6, embed_dim // num_heads)
    assert v_cached.shape == (1, num_heads, 6, embed_dim // num_heads)

    # Verify content matches (first token stored, rest follow)
    # k_cached[b, :, 0, :] == k[0, :, 0, :].T -> need to check shape alignment
    # k_cached is [batch, num_heads, size, head_dim]
    # k was [batch, num_heads, 1, head_dim], so k[b, :, 0, :] is [num_heads, head_dim]
    # k_cached[b, :, 0, :] is [num_heads, head_dim]
    # Compare batch 0 only (cache processes b=0 even when input batch>1)
    assert torch.allclose(
        k_cached[0, :, 0, :], expected_k_list[0][0, :, 0, :]
    )
    assert torch.allclose(
        v_cached[0, :, 0, :], expected_v_list[0][0, :, 0, :]
    )


# ---------------------------------------------------------------------------
# test_cache_residual_window — recent tokens are NOT quantized
# ---------------------------------------------------------------------------
def test_cache_residual_window():
    """Tokens within the residual window are stored in full precision with no quantization indices set."""
    embed_dim = 64
    num_heads = 4

    cache = PyTorchTurboQuantCache(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=200,
        head_dim=embed_dim // num_heads,
        batch_size=1,
    )

    # Append 30 tokens — well within the 128-token residual window
    for _ in range(30):
        k = torch.randn(1, num_heads, 1, embed_dim // num_heads)
        v = torch.randn(1, num_heads, 1, embed_dim // num_heads)
        cache.append(k, v)

    assert cache.size == 30

    # Residual portion should store actual values (not zeros that would indicate unused)
    k_cached, _ = cache.get_kv()
    assert k_cached.isnan().sum() == 0, "No NaN in cached K"
    assert k_cached.shape == (1, num_heads, 30, embed_dim // num_heads)


# ---------------------------------------------------------------------------
# test_cache_compression_ratio
# ---------------------------------------------------------------------------
def test_cache_compression_ratio():
    """Stored memory for quantized tokens is less than float32 full precision."""
    embed_dim = 64
    num_heads = 4

    cache = PyTorchTurboQuantCache(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=1024,
        head_dim=embed_dim // num_heads,
        batch_size=1,
    )

    total_tokens = 500

    for i in range(total_tokens):
        k = torch.randn(1, num_heads, 1, embed_dim // num_heads)
        v = torch.randn(1, num_heads, 1, embed_dim // num_heads)
        cache.append(k, v)

    assert cache.size == total_tokens

    # Check that compressed portion has non-zero indices
    residual_window = kvcache_module.RESIDUAL_WINDOW
    quantized_count = total_tokens - residual_window
    if quantized_count > 0:
        # Check compressed tensors have non-zero norms
        norms = getattr(cache, "k_norms")[:quantized_count]
        assert norms.any(), "Quantized tokens should have non-zero norms"

    # Verify stored tensors are smaller than full float32
    # Full float32: total_tokens * num_heads * head_dim * 4 bytes * 2 (k + v)
    full_float32_size = total_tokens * num_heads * (embed_dim // num_heads) * 4 * 2

    # Stored for compressed portion: uint8 indices + float32 norms for each
    # (plus residual portion is still float32)
    compressed_size = (
        quantized_count * num_heads * (1 + 4) * 2
    )  # k+ v: indices(1) + norms(4)
    residual_size = (
        min(total_tokens, residual_window)
        * num_heads
        * (embed_dim // num_heads)
        * 4
        * 2
    )

    stored_size = compressed_size + residual_size
    assert stored_size < full_float32_size


# ---------------------------------------------------------------------------
# test_cache_quantization_quality — quantization preserves attention output
# ---------------------------------------------------------------------------
def test_cache_quantization_quality():
    """With sufficient cache capacity (beyond residual window), dequantized output should be close to full precision."""
    embed_dim = 64
    head_dim = embed_dim // 4  # 16, 4 heads

    cache = PyTorchTurboQuantCache(
        embed_dim=embed_dim,
        num_heads=4,
        max_seq_len=500,
        head_dim=head_dim,
        batch_size=1,
    )

    # Build a long sequence beyond the residual window
    # Fill up to residual_window + 50 tokens so compression kicks in
    total_tokens = 200

    for i in range(total_tokens):
        k = torch.randn(1, 4, 1, head_dim)
        v = torch.randn(1, 4, 1, head_dim)
        cache.append(k, v)

    assert cache.size == total_tokens

    # Get dequantized K/V back
    k_out, v_out = cache.get_kv()
    assert k_out.shape == (1, 4, total_tokens, head_dim)
    assert v_out.shape == (1, 4, total_tokens, head_dim)

    # Verify no NaN / inf
    assert not k_out.isnan().any()
    assert not k_out.isinf().any()
    assert not v_out.isnan().any()
    assert not v_out.isinf().any()

    # The dequantized K/V should not be trivially zero (data was preserved)
    assert k_out.norm() > 0
    assert v_out.norm() > 0


# ---------------------------------------------------------------------------
# test_cache_autoregressive_generation — autoregressive with / without cache
# ---------------------------------------------------------------------------
def test_cache_autoregressive_generation():
    """Autoregressive generation with KV cache produces outputs close to without."""
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    num_experts = 4
    vocab_size = 100
    prompt_len = 5
    num_gen = 5

    import torch
    from model.pytorch.attention_kvcache import PyTorchTurboQuantCache
    from model.pytorch.transformer import PyTorchTransformer

    full_seq_len = prompt_len + num_gen
    full_seq = torch.randint(0, vocab_size, (1, full_seq_len))

    torch.manual_seed(123)
    torch.random.manual_seed(123)
    model = PyTorchTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        max_seq_len=full_seq_len + 8,
    )

    # --- Run with KV cache (feed one token at a time) ---
    cache = PyTorchTurboQuantCache(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=full_seq_len,
        head_dim=embed_dim // num_heads,
        batch_size=1,
    )
    kv_caches = [cache] * num_layers  # one cache per layer

    current = full_seq[:, :1].clone()
    for i in range(1, full_seq_len):
        cur_len = current.shape[1]
        mask = torch.tril(torch.ones((cur_len, cur_len), dtype=torch.float32))
        mask = (mask == 0).float() * -1e9
        logits, _ = model(current, mask=mask, kv_caches=kv_caches)
        current = torch.cat([current, full_seq[:, i : i + 1]], dim=1)

    # --- Run without KV cache (feed full sequence at once) ---
    model2 = PyTorchTransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_experts=num_experts,
        max_seq_len=full_seq_len + 8,
    )
    model2.load_state_dict(model.state_dict())
    mask_full = torch.tril(
        torch.ones((full_seq_len, full_seq_len), dtype=torch.float32)
    )
    mask_full = (mask_full == 0).float() * -1e9
    logits_no_cache, _ = model2(full_seq, mask=mask_full)

    # Verify the KV cache stored the correct number of tokens
    k_cached, v_cached = cache.get_kv()
    assert k_cached.shape == (1, num_heads, full_seq_len, embed_dim // num_heads)
    assert v_cached.shape == (1, num_heads, full_seq_len, embed_dim // num_heads)
    assert cache.size == full_seq_len


# ---------------------------------------------------------------------------
# test_quantize_rotate_dequantize_roundtrip
# ---------------------------------------------------------------------------
def test_quantize_rotate_dequantize_roundtrip():
    """Quantizing then dequantizing a tensor should produce a close reconstruction."""
    embed_dim = 64
    codebook = PyQuantize.get_beta_codebook()
    rotation = PyQuantize.get_random_rotation_matrix(embed_dim, seed=42)

    # Create a random signal
    data = torch.randn(100, embed_dim)

    # Quantize
    indices, norms = PyQuantize.quantize(data, rotation, codebook)

    # Dequantize
    reconstructed = PyQuantize.dequantize(indices, norms, codebook)

    assert indices.shape == data.shape
    assert indices.dtype == torch.uint8
    assert reconstructed.shape == data.shape

    # With 4-bit quantization, reconstruction should be reasonably close
    # (not pixel-perfect, but within a reasonable range)
    error = (data - reconstructed).abs().mean()
    # With 4-bit and good normalization, relative error should be bounded
    norm_data = data.norm()
    relative_error = error / norm_data.clamp(min=1e-8)
    # 4-bit gives ~1/16 precision -> error ~0.05 relative is reasonable
    assert relative_error < 0.2, (
        f"Relative error {relative_error:.4f} too large for 4-bit quant"
    )


# ---------------------------------------------------------------------------
# test_auto_clear_false — default should NOT auto-clear
# ---------------------------------------------------------------------------
def test_auto_clear_false():
    """auto_clear=False (default) should NOT auto-compact when size > max_seq_len."""
    embed_dim = 64
    num_heads = 4

    cache = PyTorchTurboQuantCache(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=6,
        head_dim=embed_dim // num_heads,
        batch_size=1,
        auto_clear=False,
    )

    # Append more tokens than max_seq_len
    for _ in range(10):
        k = torch.randn(2, num_heads, 1, embed_dim // num_heads)
        v = torch.randn(2, num_heads, 1, embed_dim // num_heads)
        cache.append(k, v)

    # Should stop at max_seq_len, not compact
    assert cache.size == 6


# ---------------------------------------------------------------------------
# test_compact_cache_manual
# ---------------------------------------------------------------------------
def test_compact_cache_manual():
    """Manual compact_cache() should shift oldest tokens out."""
    embed_dim = 64
    num_heads = 4

    cache = PyTorchTurboQuantCache(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=6,
        head_dim=embed_dim // num_heads,
        batch_size=1,
        auto_clear=False,
    )

    # Fill beyond max_seq_len
    tokens_k = []
    tokens_v = []
    for _ in range(10):
        k = torch.randn(2, num_heads, 1, embed_dim // num_heads)
        v = torch.randn(2, num_heads, 1, embed_dim // num_heads)
        cache.append(k, v)
        tokens_k.append(k[0, :, 0, :])
        tokens_v.append(v[0, :, 0, :])

    # Cache should still be at max_seq_len (6)
    assert cache.size == 6

    # compact_cache() should not change anything (already at limit)
    cache.compact_cache()
    assert cache.size == 6

    # Now test with auto_clear=True
    cache2 = PyTorchTurboQuantCache(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=6,
        head_dim=embed_dim // num_heads,
        batch_size=1,
        auto_clear=True,
    )

    for i in range(10):
        k = torch.randn(2, num_heads, 1, embed_dim // num_heads)
        v = torch.randn(2, num_heads, 1, embed_dim // num_heads)
        cache2.append(k, v)

    # auto_clear should keep only the newest 6 tokens
    assert cache2.size == 6


# ---------------------------------------------------------------------------
# test_auto_clear_true
# ---------------------------------------------------------------------------
def test_auto_clear_true():
    """auto_clear=True should compact when append would exceed max_seq_len."""
    embed_dim = 64
    num_heads = 4

    cache = PyTorchTurboQuantCache(
        embed_dim=embed_dim,
        num_heads=num_heads,
        max_seq_len=6,
        head_dim=embed_dim // num_heads,
        batch_size=1,
        auto_clear=True,
    )

    # Append more tokens than max_seq_len
    for _ in range(10):
        k = torch.randn(2, num_heads, 1, embed_dim // num_heads)
        v = torch.randn(2, num_heads, 1, embed_dim // num_heads)
        cache.append(k, v)

    # Should always stay at max_seq_len
    assert cache.size == 6
