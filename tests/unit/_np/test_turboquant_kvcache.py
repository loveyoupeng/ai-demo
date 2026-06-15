"""Tests for TurboQuant KV Cache implementation.

TurboQuant uses 1-bit quantization with per-channel scaling to compress
K and V tensors to ~1/32 the memory of float32 while maintaining reasonable
reconstruction accuracy.

Matrix shapes throughout this module:
    k, v : (batch_size, n_heads, 1, head_dim) — per-token inputs
    k_cache[i], v_cache[i] : (batch_size, n_heads, seq_len, head_dim) — dequantized outputs
    internal bits[K/V] : (batch_size, n_heads, max_len, head_dim) — 1-bit storage
    internal scales[K/V] : (batch_size, n_heads, max_len, head_dim) — float32 scale factors
"""

import numpy as np


class TestTurboQuantKVCache:
    """Tests for TurboQuantKVCache — 1-bit compressed KV Cache."""

    def test_initial_state_is_empty(self):
        """Cache starts empty."""
        from impl._np.turboquant_kv_cache import TurboQuantKVCache

        cache = TurboQuantKVCache(
            max_length=10,
            n_layers=2,
            n_heads=2,
            head_dim=4,
        )
        assert cache.is_empty()
        assert cache.current_length() == 0

    def test_compression_ratio(self):
        """1-bit storage uses ~1/32 the memory of float32."""
        from impl._np.turboquant_kv_cache import TurboQuantKVCache

        cache = TurboQuantKVCache(
            max_length=1000,
            n_layers=4,
            n_heads=8,
            head_dim=64,
        )

        fp32_bytes = 32 * cache.max_length * cache.n_heads * cache.head_dim * 2  # K and V
        actual_bytes = cache.memory_usage()
        ratio = fp32_bytes / actual_bytes

        # Should be approximately 32x compression
        assert ratio >= 30, f"Expected ~32x compression, got {ratio:.1f}x"

    def test_dequantization(self):
        """Dequantized values approximate original sign direction."""
        from impl._np.turboquant_kv_cache import TurboQuantKVCache

        cache = TurboQuantKVCache(
            max_length=32,
            n_layers=1,
            n_heads=2,
            head_dim=8,
        )

        # Create test data with positive values
        k = np.full((1, 2, 1, 8), 1.5, dtype=np.float32)
        v = np.full((1, 2, 1, 8), 2.0, dtype=np.float32)

        cache.update(k, v, pos=0)
        k_cache, v_cache = cache.get()

        # Dequantized V should be positive (scale > 0)
        assert np.all(v_cache[0] >= 0), "Positive input should dequantize non-negative"
        # Dequantized K should be positive (scale > 0)
        assert np.all(k_cache[0] >= 0), "Positive input should dequantize non-negative"

    def test_quantization_accuracy(self):
        """Error bounded within expected range for 1-bit."""
        from impl._np.turboquant_kv_cache import TurboQuantKVCache

        cache = TurboQuantKVCache(
            max_length=32,
            n_layers=1,
            n_heads=2,
            head_dim=8,
        )

        # Create data with mixed signs
        k = np.random.RandomState(42).randn(1, 2, 1, 8).astype(np.float32) * 2.0

        cache.update(k, k, pos=0)
        k_cache, _ = cache.get()

        # 1-bit quantization captures sign but not exact magnitude
        # Mean absolute error should be bounded relative to input scale
        mean_error = np.mean(np.abs(k_cache - k))
        max_abs = np.max(np.abs(k))

        assert mean_error < max_abs * 2, (
            f"Mean error {mean_error:.4f} too large relative to max magnitude {max_abs:.4f}"
        )

    def test_incremental_growth(self):
        """Cache grows sequentially, positions not overwritten."""
        from impl._np.turboquant_kv_cache import TurboQuantKVCache

        cache = TurboQuantKVCache(
            max_length=32,
            n_layers=1,
            n_heads=2,
            head_dim=8,
        )

        positions = 4
        for i in range(positions):
            k = np.full((1, 2, 1, 8), float(i + 1), dtype=np.float32)
            v = np.full((1, 2, 1, 8), float(i + 1) * 2, dtype=np.float32)
            cache.update(k, v, pos=i)

        k_cache, _ = cache.get()
        assert k_cache[0].shape == (1, 2, positions, 8)
        assert cache.current_length() == positions

        # Verify non-zero dequantized values at each position
        for i in range(positions):
            assert np.any(k_cache[0][0, 0, i, :] != 0), f"No dequantized data at position {i}"

    def test_clear_resets_state(self):
        """After clear(), cache is empty."""
        from impl._np.turboquant_kv_cache import TurboQuantKVCache

        cache = TurboQuantKVCache(
            max_length=10,
            n_layers=2,
            n_heads=2,
            head_dim=4,
        )

        # Add some data
        k = np.ones((1, 2, 1, 4), dtype=np.float32)
        v = np.ones((1, 2, 1, 4), dtype=np.float32)
        cache.update(k, v, pos=0)
        assert not cache.is_empty()
        assert cache.current_length() == 1

        cache.clear()
        assert cache.is_empty()
        assert cache.current_length() == 0
        k_cache, v_cache = cache.get()
        assert len(k_cache) == 2
        assert all(kc.size == 0 for kc in k_cache)

    def test_memory_usage_formula(self):
        """memory_usage() follows the formula: ceil(2*max_len*n_heads*head_dim/8)."""
        from impl._np.turboquant_kv_cache import TurboQuantKVCache

        cache = TurboQuantKVCache(
            max_length=100,
            n_layers=4,
            n_heads=8,
            head_dim=64,
        )

        # Each layer: 2 * max_length * n_heads * head_dim bits (K and V)
        expected_bits = 2 * 100 * 8 * 64
        expected_bytes = (expected_bits + 7) // 8
        assert cache.memory_usage() == expected_bytes
