"""Tests for CLI interface."""


class TestCliGeneration:
    """Tests that the CLI generate workflow is structurally correct."""

    def test_main_signature(self) -> None:
        """main is a callable function."""
        from impl._np.cli import main

        assert callable(main)

    def test_model_creation(self) -> None:
        """Building a NumPyModel with CLI parameters works."""
        from impl._np.model import NumPyModel
        from shared.config import TransformerConfig

        model = NumPyModel(
            TransformerConfig(
                vocab_size=256,
                embed_dim=16,
                n_layers=1,
                n_heads=2,
                n_groups=2,
                n_experts=2,
                expert_dim=16,
                top_k=2,
                rope_dim=8,
                seed=42,
            ),
        )
        assert model.vocab_size == 256
        assert model.embed_dim == 16
