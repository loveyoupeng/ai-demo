"""C11: Tests for PyTorch Inference Engine.

TDD: Write test -> all fail -> implement -> all pass -> ruff + pyright -> commit
"""

import torch


class TestTorchInference:
    """Test the inference engine orchestration."""

    def test_output_length(self) -> None:
        """Generated tokens have correct length (prompt + max_new_tokens)."""
        from impl._torch.inference import TorchTextGenerator

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(16, 8)
                self.fc = torch.nn.Linear(8, 16)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(self.emb(x))

        model = TinyModel()
        gen = TorchTextGenerator(model, max_new_tokens=10, temperature=0)
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        output = gen.generate(prompt)

        assert output.shape[1] == 3 + 10  # prompt_len + max_new_tokens

    def test_greedy_deterministic(self) -> None:
        """Same prompt -> same output (temperature=0)."""
        from impl._torch.inference import TorchTextGenerator

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(16, 8)
                self.fc = torch.nn.Linear(8, 16)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(self.emb(x))

        model = TinyModel()
        gen = TorchTextGenerator(model, max_new_tokens=5, temperature=0)
        prompt = torch.tensor([[1, 2]], dtype=torch.int64)

        out1 = gen.generate(prompt.clone())
        out2 = gen.generate(prompt.clone())

        assert torch.equal(out1, out2)

    def test_temperature_sampling(self) -> None:
        """Higher temperature -> more diverse outputs (non-deterministic)."""
        from impl._torch.inference import TorchTextGenerator

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(100, 8)
                self.fc = torch.nn.Linear(8, 100)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(self.emb(x))

        model = TinyModel()
        gen = TorchTextGenerator(model, max_new_tokens=5, temperature=2.0)
        prompt = torch.tensor([[1, 2]], dtype=torch.int64)

        # Run multiple times with same seed
        outputs = []
        for seed in range(3):
            torch.manual_seed(seed)
            outputs.append(gen.generate(prompt.clone()))

        # With high temperature, outputs should differ across runs
        differs = any(not torch.equal(outputs[i], outputs[j]) for i in range(3) for j in range(i + 1, 3))
        assert differs

    def test_1d_prompt_reshaped(self) -> None:
        """1D prompt gets reshaped to 2D."""
        from impl._torch.inference import TorchTextGenerator

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(16, 8)
                self.fc = torch.nn.Linear(8, 16)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(self.emb(x))

        model = TinyModel()
        gen = TorchTextGenerator(model, max_new_tokens=2, temperature=0)
        prompt = torch.tensor([1, 2], dtype=torch.int64)
        output = gen.generate(prompt)

        assert output.ndim == 2

    def test_top_k_filtering(self) -> None:
        """top_k=1 with temperature=0 should produce identical greedy output."""
        from impl._torch.inference import TorchTextGenerator

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(16, 8)
                self.fc = torch.nn.Linear(8, 16)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(self.emb(x))

        model = TinyModel()
        gen_greedy = TorchTextGenerator(model, max_new_tokens=5, temperature=0, top_k=0)
        gen_topk1 = TorchTextGenerator(model, max_new_tokens=5, temperature=1e-8, top_k=1)
        prompt = torch.tensor([[1, 2]], dtype=torch.int64)

        out1 = gen_greedy.generate(prompt.clone())
        out2 = gen_topk1.generate(prompt.clone())

        assert torch.equal(out1, out2)

    def test_temperature_zero_fallback(self) -> None:
        """Temperature=0 falls back to greedy decoding."""
        from impl._torch.inference import TorchTextGenerator

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(16, 8)
                self.fc = torch.nn.Linear(8, 16)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc(self.emb(x))

        model = TinyModel()
        gen0 = TorchTextGenerator(model, max_new_tokens=5, temperature=0, top_k=0)
        gen0f = TorchTextGenerator(model, max_new_tokens=5, temperature=0.0, top_k=0)
        prompt = torch.tensor([[1, 2]], dtype=torch.int64)

        out1 = gen0.generate(prompt.clone())
        out2 = gen0f.generate(prompt.clone())

        assert torch.equal(out1, out2)
