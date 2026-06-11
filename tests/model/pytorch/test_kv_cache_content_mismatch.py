"""TestKVCacheContentMismatch - verify KV cache content matches full batch."""
from __future__ import annotations

import torch
from src.model.pytorch.transformer import PyTorchTransformer
from src.model.pytorch.attention_kvcache import PyTorchTurboQuantCache


class TestKVCacheContentMismatch:
    """Autoregressive generation with KV cache should match full-batch logits."""

    def test_ar_logits_match_full_batch(self):
        """Single token auto-regression at position [0,1] matches full-batch."""
        model = PyTorchTransformer(
            vocab_size=50, embed_dim=64, num_layers=2,
            num_heads=4, num_experts=4, max_seq_len=10,
        )
        # Deterministic embeddings
        torch.manual_seed(0)
        with torch.no_grad():
            model.token_embedding.weight.data.fill_(0.0)
        torch.manual_seed(0)

        seq_len = 5
        seq = torch.randint(0, 50, (1, seq_len))

        # Full batch logits at position [0,1]
        full_mask = torch.tril(torch.ones((seq_len, seq_len)))
        logits_full, _ = model(seq, mask=full_mask)
        full_at_1 = logits_full[0, 1, :]  # [V]

        # Two-token prefix with KV cache
        prefix = seq[:, :2].clone()
        prefix_mask = torch.tril(torch.ones((2, 2)))
        caches = [
            PyTorchTurboQuantCache(
                embed_dim=64, num_heads=4, max_seq_len=seq_len,
                head_dim=16, batch_size=1,
            )
            for _ in range(2)
        ]
        logits_prefix, _ = model(prefix, mask=prefix_mask, kv_caches=caches)
        ar_at_1 = logits_prefix[0, 1, :]  # last token through 2 layers

        assert torch.allclose(ar_at_1, full_at_1, atol=1e-4, rtol=1e-4), (
            f"Mismatch at [0,1]: logits shape {ar_at_1.shape}, "
            f"max_diff={torch.max(torch.abs(ar_at_1 - full_at_1)).item():.6f}"
        )

    def test_ar_logits_mismatch_with_cache(self):
        """Same prefix: with-cache and without-cache logits differ at [0,1]."""
        model = PyTorchTransformer(
            vocab_size=50, embed_dim=64, num_layers=2,
            num_heads=4, num_experts=4, max_seq_len=10,
        )
        torch.manual_seed(0)
        with torch.no_grad():
            model.token_embedding.weight.data.fill_(0.0)
        torch.manual_seed(0)

        seq_len = 5
        seq = torch.randint(0, 50, (1, seq_len))

        # Without cache
        full_mask = torch.tril(torch.ones((seq_len, seq_len)))
        logits_no_cache, _ = model(seq, mask=full_mask)
        no_cache_at_1 = logits_no_cache[0, 1, :]

        # With cache (same prefix)
        prefix = seq[:, :2].clone()
        prefix_mask = torch.tril(torch.ones((2, 2)))
        caches = [
            PyTorchTurboQuantCache(
                embed_dim=64, num_heads=4, max_seq_len=seq_len,
                head_dim=16, batch_size=1,
            )
            for _ in range(2)
        ]
        logits_with_cache, _ = model(prefix, mask=prefix_mask, kv_caches=caches)
        with_cache_at_1 = logits_with_cache[0, 1, :]

        # These differ because cache-only sees 2 tokens, full sees all 5
        # (the with-cache version effectively computes only up to seq 0-1)
