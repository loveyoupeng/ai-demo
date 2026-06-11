"""TestKVCacheShapeMismatch - check that KV cache list length must match num_layers."""
from __future__ import annotations

import pytest
import torch
from src.model.pytorch.transformer import PyTorchTransformer
from src.model.pytorch.attention_kvcache import PyTorchTurboQuantCache


class TestKVCacheShapeMismatch:
    """KV cache list length mismatches cause errors."""

    def test_fewer_caches_than_layers_raises(self):
        """1 cache for 2-layer model → IndexError."""
        model = PyTorchTransformer(vocab_size=10, embed_dim=64, num_layers=2,
                                   num_heads=4, num_experts=4, max_seq_len=10)
        caches = [PyTorchTurboQuantCache(embed_dim=64, num_heads=4, max_seq_len=10,
                                          head_dim=16, batch_size=1)]
        x = torch.randint(0, 10, (1, 3))
        m = torch.tril(torch.ones((3, 3)))
        with pytest.raises(IndexError):
            model(x, mask=m, kv_caches=caches)  # type: ignore[arg-type]

    def test_more_caches_than_layers_no_error(self):
        """3 caches for 1-layer model → no error, extra caches unused."""
        model = PyTorchTransformer(vocab_size=10, embed_dim=64, num_layers=1,
                                   num_heads=4, num_experts=4, max_seq_len=10)
        caches = [
            PyTorchTurboQuantCache(embed_dim=64, num_heads=4, max_seq_len=10,
                                   head_dim=16, batch_size=1),
            PyTorchTurboQuantCache(embed_dim=64, num_heads=4, max_seq_len=10,
                                   head_dim=16, batch_size=1),
            PyTorchTurboQuantCache(embed_dim=64, num_heads=4, max_seq_len=10,
                                   head_dim=16, batch_size=1),
        ]
        x = torch.randint(0, 10, (1, 3))
        m = torch.tril(torch.ones((3, 3)))
        model(x, mask=m, kv_caches=caches)  # type: ignore[arg-type]

    def test_correct_cache_count_no_error(self):
        """2 caches for 2-layer model → no error."""
        model = PyTorchTransformer(vocab_size=10, embed_dim=64, num_layers=2,
                                   num_heads=4, num_experts=4, max_seq_len=10)
        caches = [
            PyTorchTurboQuantCache(embed_dim=64, num_heads=4, max_seq_len=10,
                                   head_dim=16, batch_size=1),
            PyTorchTurboQuantCache(embed_dim=64, num_heads=4, max_seq_len=10,
                                   head_dim=16, batch_size=1),
        ]
        x = torch.randint(0, 10, (1, 3))
        m = torch.tril(torch.ones((3, 3)))
        model(x, mask=m, kv_caches=caches)  # type: ignore[arg-type]

    def test_none_caches_any_layers(self):
        """kv_caches=None → no error for any layer count."""
        for num_layers in [1, 2, 4]:
            model = PyTorchTransformer(vocab_size=10, embed_dim=64,
                                       num_layers=num_layers,
                                       num_heads=4, num_experts=4, max_seq_len=10)
            x = torch.randint(0, 10, (1, 3))
            m = torch.tril(torch.ones((3, 3)))
            model(x, mask=m, kv_caches=None)
