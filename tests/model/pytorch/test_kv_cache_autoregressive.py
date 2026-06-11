"""Autoregressive step: 1 token in, cache grows by 1 per layer."""
from __future__ import annotations

import pytest
import torch
from src.model.pytorch.transformer import PyTorchTransformer
from src.model.pytorch.attention_kvcache import PyTorchTurboQuantCache


def test_autoregressive_cache_accumulation_per_layer():
    """
    Feed 1 token, then feed 1 token 3 more times.
    After step S (0-indexed), ALL layer caches should hold exactly S+1 tokens.
    """
    model = PyTorchTransformer(
        vocab_size=50, embed_dim=64, num_layers=3,
        num_heads=4, num_experts=4, max_seq_len=10,
    )
    torch.manual_seed(0)
    with torch.no_grad():
        model.token_embedding.weight.data.fill_(0.0)
    torch.manual_seed(0)

    caches = [
        PyTorchTurboQuantCache(
            embed_dim=64, num_heads=4, max_seq_len=10,
            head_dim=16, batch_size=1,
        )
        for _ in range(3)
    ]

    seq = torch.randint(0, 50, (1, 4))

    # Step 0: feed token 0
    current = seq[:, 0:1]
    model(current, mask=torch.ones((1, 1)), kv_caches=caches)
    for i, cache in enumerate(caches):
        assert cache.size == 1, f"Step 0: layer {i} cache size={cache.size}, expected 1"

    # Step 1: feed token 1 (new token, cache has 1 from step 0)
    current = seq[:, 1:2]
    model(current, mask=torch.tril(torch.ones((2, 2))), kv_caches=caches)
    for i, cache in enumerate(caches):
        assert cache.size == 2, f"Step 1: layer {i} cache size={cache.size}, expected 2"

    # Step 2: feed token 2
    current = seq[:, 2:3]
    model(current, mask=torch.tril(torch.ones((3, 3))), kv_caches=caches)
    for i, cache in enumerate(caches):
        assert cache.size == 3, f"Step 2: layer {i} cache size={cache.size}, expected 3"

    # Step 3: feed token 3
    current = seq[:, 3:4]
    model(current, mask=torch.tril(torch.ones((4, 4))), kv_caches=caches)
    for i, cache in enumerate(caches):
        assert cache.size == 4, f"Step 3: layer {i} cache size={cache.size}, expected 4"
