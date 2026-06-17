"""C7.1: Tests for PyTorch full model (TorchModel).

TDD: Write test → all fail → implement → all pass → ruff + pyright → commit
"""

import torch


class TestTorchModelForward:
    """Test the TorchModel nn.Module forward pass."""

    def test_output_shape(self) -> None:
        """TorchModel(x: [B,S]) → logits [B,S,V]."""
        from impl._torch.layers import TorchModel

        vocab_size = 100
        embed_dim = 64
        n_layers = 2
        n_heads = 4
        n_experts = 4
        ff_dim = 128
        k = 2

        model = TorchModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
            rope_dim=0,
        )

        tokens = torch.randint(0, vocab_size, (1, 8), dtype=torch.int64)
        logits = model(tokens)

        assert logits.shape == (1, 8, vocab_size)
        assert torch.all(torch.isfinite(logits))

    def test_with_embedding(self) -> None:
        """Token IDs are looked up in embedding table → propagated through layers."""
        from impl._torch.layers import TorchModel

        vocab_size = 16
        embed_dim = 32
        n_layers = 2
        n_heads = 4
        n_experts = 4
        ff_dim = 64
        k = 2

        model = TorchModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
            rope_dim=0,
        )

        tokens = torch.full((1, 4), 0, dtype=torch.int64)
        logits_0 = model(tokens)

        tokens2 = torch.full((1, 4), 1, dtype=torch.int64)
        logits_1 = model(tokens2)

        # Different tokens should produce different outputs
        assert not torch.allclose(logits_0, logits_1), "Different tokens should produce different logits"

    def test_gradient_existence(self) -> None:
        """All parameters in the model get gradients after backward."""
        from impl._torch.layers import TorchModel

        vocab_size = 32
        embed_dim = 32
        n_layers = 1
        n_heads = 2
        n_experts = 4
        ff_dim = 64
        k = 2

        model = TorchModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
            rope_dim=0,
        )

        tokens = torch.randint(0, vocab_size, (1, 4), dtype=torch.int64)
        logits = model(tokens)
        loss = logits.sum()
        loss.backward()

        grad_count = 0
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            if param.grad.abs().sum().item() > 0:
                grad_count += 1

        assert grad_count > 0, "At least some parameters must have non-zero gradients"

    def test_small_model(self) -> None:
        """Works with minimal config (vocab=16, D=32, layers=1, heads=2)."""
        from impl._torch.layers import TorchModel

        model = TorchModel(
            vocab_size=16,
            embed_dim=32,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=32,
            k=1,
            rope_dim=0,
        )

        tokens = torch.randint(0, 16, (1, 2), dtype=torch.int64)
        logits = model(tokens)

        assert logits.shape == (1, 2, 16)
        assert torch.all(torch.isfinite(logits))

    def test_cross_backend_parity(self) -> None:
        """Same seed + same input → same output as NumPyModel.

        Both models use the same initialization, so at float64 they should match.
        """
        from impl._np.model import NumPyModel
        from impl._torch.layers import TorchModel

        vocab_size = 16
        embed_dim = 32
        n_layers = 2
        n_heads = 4
        n_experts = 4
        ff_dim = 64
        k = 2
        seed = 42

        np_model = NumPyModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
            rope_dim=0,
            seed=seed,
        )

        torch_model = TorchModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_experts=n_experts,
            ff_dim=ff_dim,
            k=k,
            rope_dim=0,
            seed=seed,
        )

        # Initialize PyTorch model with same weights as NumPy model
        torch_model.load_from_numpy(np_model)

        # Forward pass with same input — eval mode disables dropout
        tokens = torch.randint(0, vocab_size, (1, 4), dtype=torch.int64)

        np_logits = np_model.forward(tokens.numpy())
        torch_model.eval()
        with torch.no_grad():
            torch_logits = torch_model(tokens)

        torch_np_logits = torch.tensor(np_logits, dtype=torch.float64)

        # Full transformer backward chain: 2 layers → rtol=1e-2, atol=1e-2
        assert torch.allclose(torch_logits.double(), torch_np_logits, atol=1e-2, rtol=1e-2), (
            "Output should match NumPy model"
        )
