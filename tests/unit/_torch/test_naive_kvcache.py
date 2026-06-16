"""C10: Tests for PyTorch Naive KV Cache.

TDD: Write test → all fail → implement → all pass → ruff + pyright → commit
"""

import torch


class TestTorchNaiveKVCache:
    """Test TorchNaiveKVCache forward behavior."""

    def test_output_shape(self) -> None:
        """k_cache, v_cache have correct shapes after update."""
        from impl._torch.kv_cache import TorchNaiveKVCache

        B, S, H, D = 2, 6, 4, 8
        cache = TorchNaiveKVCache(max_length=S, n_layers=3, n_heads=H, head_dim=D)

        # Create new K,V for a single token at pos=0
        k = torch.randn(B, H, 1, D, dtype=torch.float32)
        v = torch.randn(B, H, 1, D, dtype=torch.float32)
        cache.update(k, v, 0)

        k_result, v_result = cache.get()

        assert len(k_result) == 3
        assert len(v_result) == 3
        for ki, vi in zip(k_result, v_result):
            assert ki.shape == (B, H, 1, D)
            assert vi.shape == (B, H, 1, D)

    def test_positional_storage(self) -> None:
        """update at pos=0, pos=1, pos=2 → cache has 3 positions."""
        from impl._torch.kv_cache import TorchNaiveKVCache

        B, H, D = 1, 2, 4
        cache = TorchNaiveKVCache(max_length=5, n_layers=2, n_heads=H, head_dim=D)

        for pos in range(3):
            k = torch.full((B, H, 1, D), float(pos + 1), dtype=torch.float32)
            v = torch.full((B, H, 1, D), float(pos + 1), dtype=torch.float32)
            cache.update(k, v, pos)

        k_result, v_result = cache.get()

        assert len(k_result) == 2  # 2 layers
        assert k_result[0].shape == (B, H, 3, D)  # 3 cached positions

        # Verify stored values: position 0 should be 1.0, pos 1 should be 2.0, etc.
        for layer_k in k_result:
            assert torch.allclose(
                layer_k[:, :, 0, :], torch.tensor([[[[1.0]]]] * H, dtype=torch.float32)
            )
            assert torch.allclose(
                layer_k[:, :, 1, :], torch.tensor([[[[2.0]]]] * H, dtype=torch.float32)
            )
            assert torch.allclose(
                layer_k[:, :, 2, :], torch.tensor([[[[3.0]]]] * H, dtype=torch.float32)
            )

    def test_incremental_growth(self) -> None:
        """Cache grows sequentially, positions not overwritten."""
        from impl._torch.kv_cache import TorchNaiveKVCache

        B, H, D = 1, 2, 4
        cache = TorchNaiveKVCache(max_length=5, n_layers=1, n_heads=H, head_dim=D)

        # Insert at positions 0, 2, 4 (skip positions)
        for pos in [0, 2, 4]:
            k = torch.full((B, H, 1, D), float(pos), dtype=torch.float32)
            v = torch.full((B, H, 1, D), float(pos), dtype=torch.float32)
            cache.update(k, v, pos)

        k_result, _ = cache.get()
        # get() should return 0..max_length (since pos=4 was set, length=5)
        assert k_result[0].shape == (B, H, 5, D)
        # Position 0 has 0.0, position 2 has 2.0, position 4 has 4.0
        assert torch.allclose(k_result[0][:, :, 0, :], torch.tensor([[[[0.0]]]], dtype=torch.float32))
        assert torch.allclose(k_result[0][:, :, 2, :], torch.tensor([[[[2.0]]]], dtype=torch.float32))
        assert torch.allclose(k_result[0][:, :, 4, :], torch.tensor([[[[4.0]]]], dtype=torch.float32))

    def test_clear_behavior(self) -> None:
        """After clear, cache is empty."""
        from impl._torch.kv_cache import TorchNaiveKVCache

        B, H, D = 2, 3, 16
        cache = TorchNaiveKVCache(max_length=10, n_layers=2, n_heads=H, head_dim=D)
        k = torch.randn(B, H, 1, D, dtype=torch.float32)
        v = torch.randn(B, H, 1, D, dtype=torch.float32)
        cache.update(k, v, 0)

        k_result, v_result = cache.get()
        assert k_result[0].shape[2] == 1

        cache.clear()

        assert cache.is_empty()
        assert cache.current_length() == 0

        k_result, v_result = cache.get()
        for ki, vi in zip(k_result, v_result):
            assert ki.shape == (B, H, 0, D)
            assert vi.shape == (B, H, 0, D)

    def test_batch_size_stickiness(self) -> None:
        """Once batch size is established, all updates must match."""
        from impl._torch.kv_cache import TorchNaiveKVCache

        B, H, D = 4, 2, 8
        cache = TorchNaiveKVCache(max_length=5, n_layers=1, n_heads=H, head_dim=D)
        k = torch.randn(B, H, 1, D, dtype=torch.float32)
        v = torch.randn(B, H, 1, D, dtype=torch.float32)
        cache.update(k, v, 0)

        # Second update with same batch size
        k2 = torch.randn(B, H, 1, D, dtype=torch.float32)
        v2 = torch.randn(B, H, 1, D, dtype=torch.float32)
        cache.update(k2, v2, 1)

        k_result, _ = cache.get()
        assert k_result[0].shape[0] == B
