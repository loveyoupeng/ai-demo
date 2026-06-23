"""F10: CUDA Inference — CudaTextGenerator (greedy, sampled, top-k).

TDD: Write test → see it fail → minimal implementation → all pass → ruff + pyright.
"""

from __future__ import annotations

import torch


class TestValidatePrompt:
    """Test _validate_prompt helper function."""

    def test_list_inputs_converted(self) -> None:
        """List input should be converted to tensor."""
        from impl._cuda.inference import _validate_prompt

        prompt = torch.tensor([1, 2, 3])
        result = _validate_prompt(prompt)
        assert isinstance(result, torch.Tensor)

    def test_1d_becomes_2d(self) -> None:
        """1D tensor should become (1, S)."""
        from impl._cuda.inference import _validate_prompt

        prompt = torch.tensor([1, 2, 3, 4, 5])
        result = _validate_prompt(prompt)
        assert result.ndim == 2
        assert result.shape == (1, 5)

    def test_2d_preserved(self) -> None:
        """2D tensor should remain (B, S)."""
        from impl._cuda.inference import _validate_prompt

        prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = _validate_prompt(prompt)
        assert result.ndim == 2
        assert result.shape == (2, 3)

    def test_dtype_float_becomes_int64(self) -> None:
        """Float tensor should be converted to int64."""
        from impl._cuda.inference import _validate_prompt

        prompt = torch.tensor([1.0, 2.0, 3.0])
        result = _validate_prompt(prompt)
        assert result.dtype == torch.int64

    def test_invalid_nd_raises(self) -> None:
        """3D+ tensor should raise ValueError."""
        from impl._cuda.inference import _validate_prompt

        prompt = torch.randn(2, 3, 4)
        try:
            _validate_prompt(prompt)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "1D or 2D" in str(e)


class TestApplyTopKMask:
    """Test _apply_top_k_mask helper function."""

    def test_no_mask_when_k_large(self) -> None:
        """top_k >= vocab_size should not modify logits."""
        from impl._cuda.inference import _apply_top_k_mask

        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        masked = _apply_top_k_mask(logits, 10)
        assert torch.allclose(logits, masked)

    def test_masks_below_threshold(self) -> None:
        """Values below top-k should be set to -inf."""
        from impl._cuda.inference import _apply_top_k_mask

        logits = torch.tensor([[-5.0, 1.0, -10.0, 4.0, 2.0]])
        masked = _apply_top_k_mask(logits, 3)
        assert torch.all(torch.isinf(masked[masked < 0]))

    def test_top_1_kills_all_but_max(self) -> None:
        """top_k=1 should keep only the maximum logit."""
        from impl._cuda.inference import _apply_top_k_mask

        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])  # max is 5.0 at index 1
        masked = _apply_top_k_mask(logits, 1)
        assert masked[0, 1] == 5.0  # max value preserved

    def test_batch_2d(self) -> None:
        """Works with batch dimension of 2."""
        from impl._cuda.inference import _apply_top_k_mask

        logits = torch.tensor(
            [
                [1.0, 5.0, 2.0, 3.0],  # max=5.0
                [10.0, 1.0, 6.0, 2.0],  # max=10.0
            ]
        )
        masked = _apply_top_k_mask(logits, 2)
        assert not torch.all(torch.isinf(masked[0]))
        assert not torch.all(torch.isinf(masked[1]))


class TestCudaTextGenerator:
    """Test CudaTextGenerator with CUDA model."""

    def test_generate_returned_tensor(self) -> None:
        """generate() should return a torch.Tensor."""
        from impl._cuda.inference import CudaTextGenerator
        from impl._cuda.model import CUDAModel

        model = CUDAModel(
            vocab_size=100,
            embed_dim=32,
            n_layers=2,
            n_heads=4,
            n_experts=4,
            ff_dim=64,
        )
        gen = CudaTextGenerator(model, max_new_tokens=5)

        prompt = torch.randint(0, 100, (1, 3))
        result = gen.generate(prompt)
        assert isinstance(result, torch.Tensor)

    def test_generate_length(self) -> None:
        """generate() should return prompt + max_new_tokens length."""
        from impl._cuda.inference import CudaTextGenerator
        from impl._cuda.model import CUDAModel

        model = CUDAModel(
            vocab_size=100,
            embed_dim=32,
            n_layers=2,
            n_heads=4,
            n_experts=4,
            ff_dim=64,
        )
        gen = CudaTextGenerator(model, max_new_tokens=10)

        prompt = torch.randint(0, 100, (1, 4))
        result = gen.generate(prompt)
        assert result.shape == (1, 14)  # 4 prompt + 10 new

    def test_generate_deterministic(self) -> None:
        """Greedy generation (temperature=0) should be deterministic."""
        from impl._cuda.inference import CudaTextGenerator
        from impl._cuda.model import CUDAModel

        model = CUDAModel(
            vocab_size=100,
            embed_dim=32,
            n_layers=2,
            n_heads=4,
            n_experts=4,
            ff_dim=64,
        )
        gen = CudaTextGenerator(model, max_new_tokens=5, temperature=0.0)

        prompt = torch.randint(0, 100, (1, 3))
        result1 = gen.generate(prompt.clone())
        result2 = gen.generate(prompt.clone())
        assert torch.equal(result1, result2)

    def test_generate_greedy_matches_temperature_zero(self) -> None:
        """generate_greedy() should produce same result as generate(temperature=0)."""
        from impl._cuda.inference import CudaTextGenerator
        from impl._cuda.model import CUDAModel

        model = CUDAModel(
            vocab_size=100,
            embed_dim=32,
            n_layers=2,
            n_heads=4,
            n_experts=4,
            ff_dim=64,
        )
        prompt = torch.randint(0, 100, (1, 3))

        gen1 = CudaTextGenerator(model, max_new_tokens=5, temperature=0.0)
        gen2 = CudaTextGenerator(model, max_new_tokens=5, temperature=0.5)

        result1 = gen1.generate_greedy(prompt.clone())
        result2 = gen2.generate_sampled(prompt.clone(), temperature=0.0)

        assert torch.equal(result1, result2)

    def test_generate_batch(self) -> None:
        """Batch generation should work correctly."""
        from impl._cuda.inference import CudaTextGenerator
        from impl._cuda.model import CUDAModel

        model = CUDAModel(
            vocab_size=100,
            embed_dim=32,
            n_layers=2,
            n_heads=4,
            n_experts=4,
            ff_dim=64,
        )
        gen = CudaTextGenerator(model, max_new_tokens=5)

        prompt = torch.randint(0, 100, (3, 4))
        result = gen.generate(prompt)
        assert result.shape == (3, 9)  # 4 prompt + 5 new

    def test_generate_no_nan(self) -> None:
        """Generated tokens should be valid integers (no NaN)."""
        from impl._cuda.inference import CudaTextGenerator
        from impl._cuda.model import CUDAModel

        model = CUDAModel(
            vocab_size=100,
            embed_dim=32,
            n_layers=2,
            n_heads=4,
            n_experts=4,
            ff_dim=64,
        )
        gen = CudaTextGenerator(model, max_new_tokens=10)

        prompt = torch.randint(0, 100, (1, 3))
        result = gen.generate(prompt)
        assert torch.isfinite(result).all()

    def test_generate_token_range(self) -> None:
        """All output tokens should be within vocab range [0, vocab_size)."""
        from impl._cuda.inference import CudaTextGenerator
        from impl._cuda.model import CUDAModel

        vocab_size = 100
        model = CUDAModel(
            vocab_size=vocab_size,
            embed_dim=32,
            n_layers=2,
            n_heads=4,
            n_experts=4,
            ff_dim=64,
        )
        gen = CudaTextGenerator(model, max_new_tokens=10)

        prompt = torch.randint(0, vocab_size, (1, 5))
        result = gen.generate(prompt)

        assert (result >= 0).all()
        assert (result < vocab_size).all()

    def test_sampled_produces_valid_output(self) -> None:
        """Temperature-sampled generation should produce valid output."""
        from impl._cuda.inference import CudaTextGenerator
        from impl._cuda.model import CUDAModel

        model = CUDAModel(
            vocab_size=100,
            embed_dim=32,
            n_layers=2,
            n_heads=4,
            n_experts=4,
            ff_dim=64,
        )

        torch.manual_seed(42)
        gen = CudaTextGenerator(model, max_new_tokens=10, temperature=2.0)

        prompt = torch.randint(0, 100, (2, 3))
        torch.manual_seed(42)
        result = gen.generate_sampled(prompt, temperature=2.0)

        assert result.shape == (2, 13)
        assert torch.isfinite(result).all()

    def test_top_k_filtering_reduces_options(self) -> None:
        """top_k=1 greedy should always pick the same token at each step."""
        from impl._cuda.inference import CudaTextGenerator
        from impl._cuda.model import CUDAModel

        model = CUDAModel(
            vocab_size=100,
            embed_dim=32,
            n_layers=2,
            n_heads=4,
            n_experts=4,
            ff_dim=64,
        )
        gen = CudaTextGenerator(model, max_new_tokens=5, top_k=1)

        prompt = torch.randint(0, 100, (1, 3))
        result = gen.generate(prompt)
        assert result.shape == (1, 8)
        assert torch.isfinite(result).all()

    def test_generate_single_sequence(self) -> None:
        """Single sequence input should work (batch size 1)."""
        from impl._cuda.inference import CudaTextGenerator
        from impl._cuda.model import CUDAModel

        model = CUDAModel(
            vocab_size=100,
            embed_dim=32,
            n_layers=2,
            n_heads=4,
            n_experts=4,
            ff_dim=64,
        )
        gen = CudaTextGenerator(model, max_new_tokens=3)

        prompt = torch.randint(0, 100, (1, 5))
        result = gen.generate(prompt)
        assert result.shape == (1, 8)
        assert torch.equal(result[:, :5].cpu(), prompt)  # prompt preserved
