"""Tests for CLI interface."""


class TestPromptTokens:
    """Tests for text_to_tokens and text_from_tokens."""

    def test_text_to_tokens_returns_list(self) -> None:
        """text_to_tokens returns a list of int."""
        from impl._np.cli import text_to_tokens

        result = text_to_tokens("hello")
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)
        assert len(result) == 5  # 'hello' = 5 chars

    def test_text_to_tokens_valid_range(self) -> None:
        """All token IDs are valid byte values (0-255)."""
        from impl._np.cli import text_to_tokens

        result = text_to_tokens("Hello World!")
        assert all(0 <= t < 256 for t in result)

    def test_text_from_tokens(self) -> None:
        """Token IDs convert back to the correct text."""
        from impl._np.cli import text_from_tokens

        tokens = [72, 101, 108, 108, 111]  # "Hello"
        result = text_from_tokens(tokens)
        assert isinstance(result, str)
        assert result == "Hello"

    def test_round_trip(self) -> None:
        """Encode then decode returns the original ASCII text."""
        from impl._np.cli import text_from_tokens, text_to_tokens

        original = "Hello World!"
        tokens = text_to_tokens(original)
        result = text_from_tokens(tokens)
        assert result == original

    def test_cli_import(self) -> None:
        """Importing the CLI module succeeds without side effects."""
        import impl._np.cli as cli_module

        assert hasattr(cli_module, "main")
        assert callable(cli_module.main)
        assert hasattr(cli_module, "text_to_tokens")
        assert hasattr(cli_module, "text_from_tokens")


class TestCliGeneration:
    """Tests that the CLI generate workflow is structurally correct."""

    def test_main_signature(self) -> None:
        """main is a callable function."""
        from impl._np.cli import main

        assert callable(main)

    def test_model_creation(self) -> None:
        """Building a NumPyModel with CLI parameters works."""
        from impl._np.model import NumPyModel

        model = NumPyModel(
            vocab_size=256,
            embed_dim=16,
            n_layers=1,
            n_heads=2,
            n_experts=2,
            ff_dim=16,
            k=2,
            rope_dim=8,
            seed=42,
        )
        assert model.vocab_size == 256
        assert model.embed_dim == 16
