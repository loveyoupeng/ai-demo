"""C11: Tests for the shared text generator (torch-family inference).

The torch/triton/cuda generators collapsed into ``shared.generator``; the
fixture here is a minimal model implementing the KV-step interface (the
same one the NumPy track publicizes) so the tests exercise the real
decode path, not a re-forward stub.
"""

from __future__ import annotations

import torch


class TinyStepModel(torch.nn.Module):
    """Minimal model implementing the KV-step interface.

    forward_prefill fills a dict cache with the last position's embedding
    as a stand-in K/V; forward_step attends nothing — it just returns the
    next token's logits (deterministic function of the input token). The
    generator's decode loop, temperature/top-k math, and length contract
    are what these tests defend.
    """

    def __init__(self, vocab_size: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = torch.nn.Embedding(vocab_size, 8)
        self.fc = torch.nn.Linear(8, vocab_size)

    def make_cache(self, batch_size: int) -> list[dict[str, torch.Tensor]]:
        return [{"k": torch.zeros(batch_size, 1, 0, 8), "v": torch.zeros(batch_size, 1, 0, 8)} for _ in range(1)]

    def forward_prefill(
        self, input_ids: torch.Tensor, cache: list[dict[str, torch.Tensor]] | None = None
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        if cache is None:
            cache = self.make_cache(input_ids.shape[0])
        logits = self.fc(self.emb(input_ids))
        return logits, cache

    def forward_step(
        self, input_ids: torch.Tensor, position: int, cache: list[dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        # Deterministic per-token logits: token id → a distinct logit row.
        logits = self.fc(self.emb(input_ids))  # (B, 1, V)
        # Append to the cache so the step path is exercised.
        cache[0]["k"] = torch.cat([cache[0]["k"], self.emb(input_ids).unsqueeze(1).permute(0, 2, 1, 3)], dim=2)
        return logits


class TestTorchInference:
    """Test the shared generator through the KV-step interface."""

    def test_output_length(self) -> None:
        """Generated tokens have correct length (prompt + max_new_tokens)."""
        from impl._torch.inference import TorchTextGenerator

        gen = TorchTextGenerator(TinyStepModel(), max_new_tokens=10, temperature=0)
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        output = gen.generate(prompt)
        assert output.shape[1] == 3 + 10  # prompt_len + max_new_tokens

    def test_greedy_deterministic(self) -> None:
        """Same prompt -> same output (temperature=0)."""
        from impl._torch.inference import TorchTextGenerator

        gen = TorchTextGenerator(TinyStepModel(), max_new_tokens=5, temperature=0)
        prompt = torch.tensor([[1, 2]], dtype=torch.int64)
        assert torch.equal(gen.generate(prompt.clone()), gen.generate(prompt.clone()))

    def test_temperature_sampling(self) -> None:
        """Higher temperature -> more diverse outputs (non-deterministic)."""
        from impl._torch.inference import TorchTextGenerator

        gen = TorchTextGenerator(TinyStepModel(vocab_size=100), max_new_tokens=5, temperature=2.0)
        prompt = torch.tensor([[1, 2]], dtype=torch.int64)

        outputs = []
        for seed in range(3):
            torch.manual_seed(seed)
            outputs.append(gen.generate(prompt.clone()))

        differs = any(not torch.equal(outputs[i], outputs[j]) for i in range(3) for j in range(i + 1, 3))
        assert differs

    def test_1d_prompt_rejected(self) -> None:
        """The KV-step interface is strict: 1-D prompts raise, not reshape."""
        import pytest

        from impl._torch.inference import TorchTextGenerator

        gen = TorchTextGenerator(TinyStepModel(), max_new_tokens=2, temperature=0)
        with pytest.raises(ValueError, match="2-D"):
            gen.generate(torch.tensor([1, 2], dtype=torch.int64))

    def test_top_k_filtering(self) -> None:
        """top_k=1 with temperature≈0 should produce identical greedy output."""
        from impl._torch.inference import TorchTextGenerator

        model = TinyStepModel()
        gen_greedy = TorchTextGenerator(model, max_new_tokens=5, temperature=0, top_k=0)
        gen_topk1 = TorchTextGenerator(model, max_new_tokens=5, temperature=1e-8, top_k=1)
        prompt = torch.tensor([[1, 2]], dtype=torch.int64)
        assert torch.equal(gen_greedy.generate(prompt.clone()), gen_topk1.generate(prompt.clone()))

    def test_temperature_zero_fallback(self) -> None:
        """Temperature=0 falls back to greedy decoding."""
        from impl._torch.inference import TorchTextGenerator

        model = TinyStepModel()
        gen0 = TorchTextGenerator(model, max_new_tokens=5, temperature=0, top_k=0)
        gen0f = TorchTextGenerator(model, max_new_tokens=5, temperature=0.0, top_k=0)
        prompt = torch.tensor([[1, 2]], dtype=torch.int64)
        assert torch.equal(gen0.generate(prompt.clone()), gen0f.generate(prompt.clone()))
