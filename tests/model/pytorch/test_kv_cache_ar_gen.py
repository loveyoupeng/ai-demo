"""KV cache: step-by-step cache accumulation and output shape verification."""
from __future__ import annotations

import torch
from src.model.pytorch.transformer import PyTorchTransformer
from src.model.pytorch.attention_kvcache import PyTorchTurboQuantCache


def test_cache_size_increases_by_input_length():
    """
    Each forward call: cache size += input_seq_len for each layer.
    """
    model = PyTorchTransformer(
        vocab_size=50, embed_dim=64, num_layers=3,
        num_heads=4, num_experts=4, max_seq_len=20,
    )
    torch.manual_seed(0)
    with torch.no_grad():
        model.token_embedding.weight.data.fill_(0.0)
    torch.manual_seed(0)

    seq = torch.randint(0, 50, (1, 6))
    caches = [
        PyTorchTurboQuantCache(
            embed_dim=64, num_heads=4, max_seq_len=20,
            head_dim=16, batch_size=1,
        )
        for _ in range(3)
    ]

    # Check empty cache
    for cache in caches:
        assert cache.size == 0

    # Step 0: feed 3 tokens
    x3 = seq[:, :3]
    model(x3, mask=torch.tril(torch.ones((3, 3))), kv_caches=caches)
    for i, cache in enumerate(caches):
        assert cache.size == 3, f"Layer {i} cache size={cache.size}, expected 3"

    # Step 1: feed 2 more tokens
    x2 = seq[:, 3:5]
    total_len = 5
    model(x2, mask=torch.tril(torch.ones((total_len, total_len))), kv_caches=caches)
    for i, cache in enumerate(caches):
        assert cache.size == 5, f"Layer {i} cache size={cache.size}, expected 5"

    # Step 2: feed 1 more token
    x1 = seq[:, 5:6]
    total_len = 6
    model(x1, mask=torch.tril(torch.ones((total_len, total_len))), kv_caches=caches)
    for i, cache in enumerate(caches):
        assert cache.size == 6, f"Layer {i} cache size={cache.size}, expected 6"


def test_single_token_input_single_token_output():
    """[B,1,D] input → [B,1,V] output, [B,1,D] output in forward pass."""
    model = PyTorchTransformer(
        vocab_size=50, embed_dim=64, num_layers=2,
        num_heads=4, num_experts=4, max_seq_len=10,
    )
    torch.manual_seed(0)
    with torch.no_grad():
        model.token_embedding.weight.data.fill_(0.0)
    torch.manual_seed(0)

    seq = torch.randint(0, 50, (1, 4))

    # Feed 2 tokens first to fill cache
    x2 = seq[:, :2]
    caches = [
        PyTorchTurboQuantCache(
            embed_dim=64, num_heads=4, max_seq_len=10,
            head_dim=16, batch_size=1,
        )
        for _ in range(2)
    ]
    logits2, _ = model(x2, mask=torch.tril(torch.ones((2, 2))), kv_caches=caches)
    assert logits2.shape == (1, 2, 50), f"Expected [1,2,50], got {logits2.shape}"

    # Now feed 1 token → output [1,1,50]
    x1 = seq[:, 2:3]
    logits1, _ = model(x1, mask=torch.tril(torch.ones((3, 3))), kv_caches=caches)
    assert logits1.shape == (1, 1, 50), f"Expected [1,1,50], got {logits1.shape}"

    # Feed 1 token → output [1,1,50]
    x1 = seq[:, 3:4]
    logits1, _ = model(x1, mask=torch.tril(torch.ones((4, 4))), kv_caches=caches)
    assert logits1.shape == (1, 1, 50)


def test_cache_all_layers_same_size():
    """After one forward pass, ALL layer caches have identical size."""
    model = PyTorchTransformer(
        vocab_size=50, embed_dim=64, num_layers=4,
        num_heads=4, num_experts=4, max_seq_len=20,
    )
    torch.manual_seed(0)
    with torch.no_grad():
        model.token_embedding.weight.data.fill_(0.0)
    torch.manual_seed(0)

    seq = torch.randint(0, 50, (1, 5))

    caches = [
        PyTorchTurboQuantCache(
            embed_dim=64, num_heads=4, max_seq_len=20,
            head_dim=16, batch_size=1,
        )
        for _ in range(4)
    ]

    model(seq, mask=torch.tril(torch.ones((5, 5))), kv_caches=caches)

    sizes = [cache.size for cache in caches]
    assert all(s == 5 for s in sizes), f"Expected all 5, got {sizes}"
    assert len(set(sizes)) == 1, f"All cache sizes should be identical: {sizes}"


def test_reset_clears_all_layers():
    """Calling reset() on each cache sets size to 0."""
    model = PyTorchTransformer(
        vocab_size=50, embed_dim=64, num_layers=2,
        num_heads=4, num_experts=4, max_seq_len=10,
    )
    torch.manual_seed(0)
    with torch.no_grad():
        model.token_embedding.weight.data.fill_(0.0)
    torch.manual_seed(0)

    seq = torch.randint(0, 50, (1, 4))
    caches = [
        PyTorchTurboQuantCache(
            embed_dim=64, num_heads=4, max_seq_len=10,
            head_dim=16, batch_size=1,
        )
        for _ in range(2)
    ]

    model(seq, mask=torch.tril(torch.ones((4, 4))), kv_caches=caches)
    for cache in caches:
        assert cache.size == 4

    # Reset each cache
    for cache in caches:
        cache.reset()

    for cache in caches:
        assert cache.size == 0


def test_cache_independent_across_forward_passes():
    """Two consecutive forward passes without reset accumulate on top of each other."""
    model = PyTorchTransformer(
        vocab_size=50, embed_dim=64, num_layers=2,
        num_heads=4, num_experts=4, max_seq_len=20,
    )
    torch.manual_seed(0)
    with torch.no_grad():
        model.token_embedding.weight.data.fill_(0.0)
    torch.manual_seed(0)

    seq1 = torch.randint(0, 50, (1, 3))
    caches = [
        PyTorchTurboQuantCache(
            embed_dim=64, num_heads=4, max_seq_len=20,
            head_dim=16, batch_size=1,
        )
        for _ in range(2)
    ]

    # First forward pass: 3 tokens
    model(seq1, mask=torch.tril(torch.ones((3, 3))), kv_caches=caches)
    for cache in caches:
        assert cache.size == 3

    # Second forward pass: 2 more tokens, no reset
    seq2 = torch.randint(0, 50, (1, 2))
    model(seq2, mask=torch.tril(torch.ones((5, 5))), kv_caches=caches)
    for cache in caches:
        assert cache.size == 5, f"Expected 5 (3+2), got {cache.size}"
