from __future__ import annotations

"""Phase A: KV Cache Lifecycle Unit Tests — MHA integration.

Tests A1–A5 establish the KV cache lifecycle at the MHA layer level:
- A1: Cache creation and reset (lifecycle states)
- A2: Append one token, retrieve (data preservation)
- A3: Growing sequence accumulation (append preserves previous)
- A4: Multi-token batch append (append 3 at once)
- A5: Empty cache append (edge case: no-op)
"""

import torch

from test_turboquant_cache import PyTorchTurboQuantCache


# A1: test_cache_create_reset — lifecycle
def test_cache_create_reset():
    """Cache starts empty → reset() works → append works again."""
    cache = PyTorchTurboQuantCache(
        embed_dim=64, num_heads=4, max_seq_len=32, head_dim=16, batch_size=1,
    )

    # Lifecycle state 1: empty after creation
    assert cache.size == 0

    # Lifecycle state 2: reset on empty should be no-op
    cache.reset()
    assert cache.size == 0

    # Append one token → size becomes 1
    k = torch.randn(1, 4, 1, 16)
    v = torch.randn(1, 4, 1, 16)
    cache.append(k, v)
    assert cache.size == 1

    # Lifecycle state 3: reset clears
    cache.reset()
    assert cache.size == 0

    # Lifecycle state 4: can reuse after reset
    k2 = torch.randn(1, 4, 2, 16)
    v2 = torch.randn(1, 4, 2, 16)
    cache.append(k2, v2)
    assert cache.size == 2


# A2: test_append_one_token — data preservation
def test_append_one_token():
    """Append [1,h,1,d] K/V → get_kv() returns matching [1,h,1,d]."""
    cache = PyTorchTurboQuantCache(
        embed_dim=64, num_heads=4, max_seq_len=32, head_dim=16, batch_size=1,
    )

    k = torch.randn(1, 4, 1, 16)
    v = torch.randn(1, 4, 1, 16)
    cache.append(k, v)

    k_cached, v_cached = cache.get_kv()
    assert k_cached.shape == (1, 4, 1, 16)
    assert v_cached.shape == (1, 4, 1, 16)

    # Data must match exactly (within residual window, full precision)
    assert torch.allclose(k_cached[0], k[0])
    assert torch.allclose(v_cached[0], v[0])


# A3: test_append_growing_sequence — accumulation preserves previous
def test_append_growing_sequence():
    """Sequential appends grow cache monotonically: [1]→[2]→[3]."""
    cache = PyTorchTurboQuantCache(
        embed_dim=64, num_heads=4, max_seq_len=32, head_dim=16, batch_size=1,
    )

    # Append token 0
    k0 = torch.randn(1, 4, 1, 16)
    v0 = torch.randn(1, 4, 1, 16)
    cache.append(k0, v0)
    assert cache.size == 1
    k1, v1 = cache.get_kv()
    assert k1.shape == (1, 4, 1, 16)
    assert torch.allclose(k1[0], k0[0])

    # Append token 1 → cache has 2
    k1 = torch.randn(1, 4, 1, 16)
    v1 = torch.randn(1, 4, 1, 16)
    cache.append(k1, v1)
    assert cache.size == 2
    k2, v2 = cache.get_kv()
    assert k2.shape == (1, 4, 2, 16)
    assert torch.allclose(k2[0, :, 0, :], k0[0, :, 0, :])
    assert torch.allclose(k2[0, :, 1, :], k1[0, :, 0, :])

    # Append token 2 → cache has 3
    k2_actual = torch.randn(1, 4, 1, 16)
    v2_actual = torch.randn(1, 4, 1, 16)
    cache.append(k2_actual, v2_actual)
    assert cache.size == 3
    k3, v3 = cache.get_kv()
    assert k3.shape == (1, 4, 3, 16)
    assert torch.allclose(k3[0, :, 0, :], k0[0, :, 0, :])
    assert torch.allclose(k3[0, :, 1, :], k1[0, :, 0, :])
    assert torch.allclose(k3[0, :, 2, :], k2_actual[0, :, 0, :])


# A4: test_append_batch_tokens — multi-token batch
def test_append_batch_tokens():
    """Append [1,h,3,d] (3 tokens at once) → get_kv() returns all 3."""
    cache = PyTorchTurboQuantCache(
        embed_dim=64, num_heads=4, max_seq_len=32, head_dim=16, batch_size=1,
    )

    k_batch = torch.randn(1, 4, 3, 16)
    v_batch = torch.randn(1, 4, 3, 16)
    cache.append(k_batch, v_batch)

    assert cache.size == 3
    k_out, v_out = cache.get_kv()
    assert k_out.shape == (1, 4, 3, 16)
    assert v_out.shape == (1, 4, 3, 16)
    assert torch.allclose(k_out[0], k_batch[0])
    assert torch.allclose(v_out[0], v_batch[0])


# A5: test_append_empty — empty append is no-op
def test_append_empty():
    """Appending empty K/V does not change cache state."""
    cache = PyTorchTurboQuantCache(
        embed_dim=64, num_heads=4, max_seq_len=32, head_dim=16, batch_size=1,
    )

    # Empty append on fresh cache
    k_empty = torch.randn(1, 4, 0, 16)
    v_empty = torch.randn(1, 4, 0, 16)
    cache.append(k_empty, v_empty)
    assert cache.size == 0

    # Append 2 tokens, then empty → still 2
    k2 = torch.randn(1, 4, 2, 16)
    v2 = torch.randn(1, 4, 2, 16)
    cache.append(k2, v2)
    assert cache.size == 2

    cache.append(k_empty, v_empty)
    assert cache.size == 2

    k_out, v_out = cache.get_kv()
    assert k_out.shape == (1, 4, 2, 16)
