"""Tests for the PyTorch Embedding layer (C1.1).

TDD: Write test → all fail → implement → all pass → ruff + pyright → commit
"""

import torch


class TestEmbeddingLayer:
    """Tests for the Embedding nn.Module in impl._torch.layers."""

    def test_output_shape(self) -> None:
        """Embedding[input_ids: [B,S]] → [B,S,embed_dim]."""
        from impl._torch.layers import Embedding

        vocab_size, embed_dim = 64, 16
        layer = Embedding(vocab_size, embed_dim)
        input_ids = torch.randint(0, vocab_size, size=(2, 8))
        output = layer(input_ids)

        assert output.shape == (2, 8, 16)
        assert output.dtype == torch.float32
        assert torch.all(torch.isfinite(output))

    def test_lookup_correctness(self) -> None:
        """embedding(token_id) equals the correct row of weight."""
        from impl._torch.layers import Embedding

        vocab_size, embed_dim = 8, 4
        layer = Embedding(vocab_size, embed_dim)

        # With a small embedding, verify specific lookups
        for token_id in range(vocab_size):
            single = torch.tensor([[token_id]], dtype=torch.long)
            lookup = layer(single)[0, 0]
            weight_row = layer.weight[token_id]
            assert torch.allclose(lookup, weight_row)

    def test_batch_sequential_parity(self) -> None:
        """Batch processing gives same results as sequential lookups."""
        from impl._torch.layers import Embedding

        vocab_size, embed_dim = 16, 8
        layer = Embedding(vocab_size, embed_dim)

        input_ids = torch.randint(0, vocab_size, size=(3, 5))
        batch_output = layer(input_ids)

        for b in range(3):
            for s in range(5):
                single = input_ids[b, s].unsqueeze(0).unsqueeze(0)
                assert torch.allclose(batch_output[b, s], layer(single)[0, 0])

    def test_gradient_flow(self) -> None:
        """Gradients flow back through embedding to weight."""
        from impl._torch.layers import Embedding

        vocab_size, embed_dim = 16, 8
        layer = Embedding(vocab_size, embed_dim)

        input_ids = torch.randint(0, vocab_size, size=(2, 4))
        output = layer(input_ids)
        loss = output.sum()
        loss.backward()

        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == layer.weight.shape
        assert torch.all(torch.isfinite(layer.weight.grad))
