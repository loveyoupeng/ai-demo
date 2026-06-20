"""E11: CLI inference — text_to_tokens, text_from_tokens, generate_text."""

import pytest
import torch
import torch.nn as nn


class TestTextConversion:
    def test_text_to_tokens(self):
        """Each byte becomes a token ID."""
        from impl._triton.cli import text_to_tokens

        result = text_to_tokens("hello")
        assert isinstance(result, list)
        assert all(0 <= t <= 255 for t in result)
        assert len(result) == 5

    def test_text_from_tokens(self):
        """Token IDs decode back to text."""
        from impl._triton.cli import text_from_tokens

        text = "hello"
        tokens = [ord(c) for c in text]
        result = text_from_tokens(tokens)
        assert result == text

    def test_roundtrip(self):
        """text -> tokens -> text lossless."""
        from impl._triton.cli import text_from_tokens, text_to_tokens

        text = "abcdefgh"
        tokens = text_to_tokens(text)
        result = text_from_tokens(tokens)
        assert result == text


class SimpleModel(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        return self.linear(self.embedding(x))


class TestInferenceGenerate:
    @pytest.mark.timeout(10)
    def test_greedy_returns_sequence(self):
        """Greedy generation returns (batch, seq_len) tensor with prompt + new tokens."""
        from impl._triton.inference import TritonTextGenerator

        model = SimpleModel(256, 16)
        gen = TritonTextGenerator(model, max_new_tokens=5, temperature=0.0)

        prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        output = gen.generate(prompt)

        assert output.shape == (1, 3 + 5), f"Expected (1, 8), got {output.shape}"
        assert output.dtype == torch.int64

    @pytest.mark.timeout(10)
    def test_greedy_deterministic(self):
        """Same prompt + greedy → same output."""
        from impl._triton.inference import TritonTextGenerator

        model = SimpleModel(256, 16)
        model = model.eval()
        gen = TritonTextGenerator(model, max_new_tokens=10, temperature=0.0)

        prompt = torch.tensor([[10, 20, 30]], dtype=torch.int64)
        out1 = gen.generate(prompt)
        out2 = gen.generate(prompt)

        assert torch.equal(out1, out2)

    @pytest.mark.timeout(10)
    def test_greedy_preserves_prompt(self):
        """Greedy output starts with the prompt tokens."""
        from impl._triton.inference import TritonTextGenerator

        model = SimpleModel(256, 16)
        gen = TritonTextGenerator(model, max_new_tokens=3, temperature=0.0)

        prompt = torch.tensor([[42, 43, 44]], dtype=torch.int64)
        output = gen.generate(prompt)

        assert torch.equal(output[0, :3], prompt[0])

    @pytest.mark.timeout(10)
    def test_sampled_output_shape(self):
        """Sampled generation returns (batch, seq_len) tensor."""
        from impl._triton.inference import TritonTextGenerator

        model = SimpleModel(256, 16)
        gen = TritonTextGenerator(model, max_new_tokens=5, temperature=0.7, top_k=20)

        prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        output = gen.generate(prompt)

        assert output.shape == (1, 3 + 5)
        assert output.dtype == torch.int64
        assert torch.all(output >= 0)
        assert torch.all(output < 256)

    @pytest.mark.timeout(10)
    def test_temperature_zero_uses_greedy(self):
        """Temperature=0 falls back to greedy even with top_k set."""
        from impl._triton.inference import TritonTextGenerator

        model = SimpleModel(256, 16)
        model = model.eval()
        gen = TritonTextGenerator(model, max_new_tokens=5, temperature=0.0, top_k=10)

        out1 = gen.generate(torch.tensor([[10, 20, 30]], dtype=torch.int64))
        out2 = gen.generate(torch.tensor([[10, 20, 30]], dtype=torch.int64))

        assert torch.equal(out1, out2)

    @pytest.mark.timeout(10)
    def test_batched_generation(self):
        """Generation works with batch size > 1."""
        from impl._triton.inference import TritonTextGenerator

        model = SimpleModel(256, 16)
        gen = TritonTextGenerator(model, max_new_tokens=4, temperature=0.0)

        prompt = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        output = gen.generate(prompt)

        assert output.shape == (2, 2 + 4)

    @pytest.mark.timeout(10)
    def test_top_k_filtering(self):
        """top_k limits sampling to k highest-probability tokens."""
        from impl._triton.inference import TritonTextGenerator

        model = SimpleModel(10, 8)  # Small vocab for easier testing
        gen = TritonTextGenerator(model, max_new_tokens=10, temperature=0.1, top_k=3)

        prompt = torch.tensor([[1]], dtype=torch.int64)
        output = gen.generate(prompt)

        assert output.shape == (1, 1 + 10)
        assert torch.all(output >= 0)
        assert torch.all(output < 10)

    @pytest.mark.timeout(10)
    def test_one_token_generation(self):
        """max_new_tokens=1 adds exactly one token."""
        from impl._triton.inference import TritonTextGenerator

        model = SimpleModel(256, 16)
        gen = TritonTextGenerator(model, max_new_tokens=1, temperature=0.0)

        prompt = torch.tensor([[99]], dtype=torch.int64)
        output = gen.generate(prompt)

        assert output.shape == (1, 1 + 1)

    @pytest.mark.timeout(10)
    def test_1d_prompt_automatic_batch_reshape(self):
        """1D prompt gets reshaped to (1, seq_len)."""
        from impl._triton.inference import TritonTextGenerator

        model = SimpleModel(256, 16)
        gen = TritonTextGenerator(model, max_new_tokens=2, temperature=0.0)

        # 1D input
        prompt = torch.tensor([5, 6, 7], dtype=torch.int64)
        output = gen.generate(prompt)

        assert output.shape == (1, 3 + 2)
