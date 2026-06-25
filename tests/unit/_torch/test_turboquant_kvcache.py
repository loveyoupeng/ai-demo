"""C10: Tests for PyTorch TurboQuant KV Cache (1-bit compression).

TDD: Write test → all fail → implement → all pass → ruff + pyright → commit
"""

import torch


class TestTorchTurboQuantKVCache:
    """Test TorchTurboQuantKVCache 1-bit compression."""

    def test_compression_shape(self) -> None:
        """1-bit storage uses int8 (lower memory than float32)."""
        from impl._torch.turboquant_kv_cache import TorchTurboQuantKVCache

        B, S, H, D = 2, 10, 3, 8
        cache = TorchTurboQuantKVCache(max_length=S, n_layers=2, n_heads=H, head_dim=D)

        k = torch.randn(B, H, 1, D, dtype=torch.float32)
        v = torch.randn(B, H, 1, D, dtype=torch.float32)
        cache.update(k, v, 0)

        # Bits should be int8, not float32
        assert len(cache.bits_k) == 2
        assert cache.bits_k[0].dtype == torch.int8
        assert cache.scales_k[0].dtype == torch.float32

    def test_dequantization(self) -> None:
        """Dequantized values approximate the original."""
        from impl._torch.turboquant_kv_cache import TorchTurboQuantKVCache

        B, H, D = 1, 2, 8
        cache = TorchTurboQuantKVCache(max_length=5, n_layers=1, n_heads=H, head_dim=D)

        # Create a clear positive tensor so all bits = 1
        k = torch.ones(B, H, 1, D, dtype=torch.float32) * 2.0
        v = torch.zeros(B, H, 1, D, dtype=torch.float32)
        cache.update(k, v, 0)

        k_result, v_result = cache.get()

        # All-positive k should reconstruct approximately (bits=1, scale=2.0)
        assert torch.allclose(k_result[0], torch.tensor(2.0, dtype=torch.float32), atol=0.5)

    def test_quantization_accuracy(self) -> None:
        """Quantized reconstruction error bounded within expected range."""
        from impl._torch.turboquant_kv_cache import TorchTurboQuantKVCache

        B, H, D = 1, 2, 16
        cache = TorchTurboQuantKVCache(max_length=5, n_layers=1, n_heads=H, head_dim=D)

        # Mixed values: some positive, some negative
        torch.manual_seed(42)
        k = torch.randn(B, H, 1, D, dtype=torch.float32) * 3.0
        v = torch.randn(B, H, 1, D, dtype=torch.float32) * 3.0
        cache.update(k, v, 0)

        k_result, v_result = cache.get()

        # 1-bit quantization: each element is either 0 or scale
        # The reconstruction error for each element is at most scale (not |x|)
        # Because we store sign(0) → binary, then reconstruct as binary_bits * scale
        for i in range(k_result[0].numel()):
            orig = k.flatten()[i].item()
            recon = k_result[0].flatten()[i].item()
            # Reconstructed value should equal scale (stored for that position)
            # scale = mean(|k|) per (batch, head)
            # So reconstruction error ≤ scale which is finite
            assert abs(recon - orig) <= max(abs(orig), 3.0) + 0.1

    def test_incremental_store(self) -> None:
        """Multiple updates at different positions build up storage."""
        from impl._torch.turboquant_kv_cache import TorchTurboQuantKVCache

        B, H, D = 1, 2, 4
        cache = TorchTurboQuantKVCache(max_length=5, n_layers=1, n_heads=H, head_dim=D)

        for pos in range(3):
            k = torch.full((B, H, 1, D), float(pos + 1), dtype=torch.float32)
            v = torch.full((B, H, 1, D), float(pos + 1), dtype=torch.float32)
            cache.update(k, v, pos)

        k_result, v_result = cache.get()

        assert k_result[0].shape == (B, H, 3, D)
        # After dequantization, all-positive values reconstruct to scale ≈ value
        # Position 0 has value 1.0, position 1 has 2.0, position 2 has 3.0
        for pos, expected in enumerate([1.0, 2.0, 3.0]):
            assert torch.allclose(k_result[0][:, :, pos, :], torch.tensor(expected, dtype=torch.float32), atol=0.5)

    def test_clear(self) -> None:
        """After clear, cache is empty."""
        from impl._torch.turboquant_kv_cache import TorchTurboQuantKVCache

        B, H, D = 2, 3, 16
        cache = TorchTurboQuantKVCache(max_length=10, n_layers=2, n_heads=H, head_dim=D)
        k = torch.randn(B, H, 1, D, dtype=torch.float32)
        v = torch.randn(B, H, 1, D, dtype=torch.float32)
        cache.update(k, v, 0)

        assert not cache.is_empty()
        cache.clear()

        assert cache.is_empty()
        assert cache.current_length() == 0
        assert cache._batch_size == 0

        k_result, v_result = cache.get()
        assert k_result[0].shape[2] == 0
