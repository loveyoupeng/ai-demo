"""Tests for Naive KV Cache implementation."""

import numpy as np


class TestNaiveKVCache:
    def test_initial_state_is_empty(self):
        """Cache starts empty."""
        from impl._np.kv_cache import NaiveKVCache

        cache = NaiveKVCache(max_length=10, n_layers=2, n_heads=2, head_dim=4)
        assert cache.is_empty()
        assert cache.current_length() == 0

    def test_update_and_get(self):
        """After updating, get() returns stored values."""
        from impl._np.kv_cache import NaiveKVCache

        cache = NaiveKVCache(max_length=10, n_layers=2, n_heads=2, head_dim=4)

        # Update layer 0 at pos 0
        k = np.ones((1, 2, 1, 4), dtype=np.float32)
        v = np.ones((1, 2, 1, 4), dtype=np.float32)
        cache.update(k, v, pos=0)

        k_cache, v_cache = cache.get()
        assert len(k_cache) == 2  # 2 layers
        assert k_cache[0].shape == (1, 2, 1, 4)
        np.testing.assert_array_equal(k_cache[0], k)
        np.testing.assert_array_equal(v_cache[0], v)
        assert cache.current_length() == 1

    def test_incremental_growth(self):
        """Cache grows sequentially, positions not overwritten."""
        from impl._np.kv_cache import NaiveKVCache

        cache = NaiveKVCache(max_length=10, n_layers=1, n_heads=2, head_dim=4)

        # Update 3 positions
        for i in range(3):
            k = np.full((1, 2, 1, 4), float(i + 1), dtype=np.float32)
            v = np.full((1, 2, 1, 4), float(i + 1) * 2, dtype=np.float32)
            cache.update(k, v, pos=i)

        k_cache, _ = cache.get()
        # k_cache is a list of ndarrays, one per layer
        assert len(k_cache) == 1
        k_arr = k_cache[0]
        # Shape should be (1, 2, 3, 4)
        assert k_arr.shape == (1, 2, 3, 4)
        # Check positional values
        np.testing.assert_array_equal(k_arr[0, 0, 0, 0], 1.0)
        np.testing.assert_array_equal(k_arr[0, 0, 1, 0], 2.0)
        np.testing.assert_array_equal(k_arr[0, 0, 2, 0], 3.0)

    def test_clear_resets_state(self):
        """After clear, cache is empty."""
        from impl._np.kv_cache import NaiveKVCache

        cache = NaiveKVCache(max_length=10, n_layers=2, n_heads=2, head_dim=4)

        # Add some data
        k = np.ones((1, 2, 1, 4), dtype=np.float32)
        v = np.ones((1, 2, 1, 4), dtype=np.float32)
        cache.update(k, v, pos=0)
        assert not cache.is_empty()

        cache.clear()
        assert cache.is_empty()
        assert cache.current_length() == 0
        k_cache, v_cache = cache.get()
        assert len(k_cache) == 2
        assert all(kc.size == 0 for kc in k_cache)
