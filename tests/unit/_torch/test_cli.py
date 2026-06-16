"""C12: Tests for PyTorch CLI.

TDD: Write test -> all fail -> implement -> all pass -> ruff + pyright -> commit
"""

import subprocess
import sys


class TestCLI:
    """Test the CLI interface."""

    def test_help_text(self) -> None:
        """CLI --help exits with code 0."""
        result = subprocess.run(
            [sys.executable, "-m", "impl._torch.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Generate text with PyTorch LLM" in result.stdout

    def test_prompt_parsing(self) -> None:
        """CLI --prompt correctly parsed and generates output."""
        result = subprocess.run(
            [sys.executable, "-m", "impl._torch.cli", "--prompt", "hi", "--max_new_tokens", "3"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Prompt:" in result.stdout
        assert "Generated:" in result.stdout


def test_cli_imports() -> None:
    """CLI module can be imported without errors."""
    from impl._torch.cli import main, text_from_tokens, text_to_tokens

    assert callable(main)
    assert callable(text_from_tokens)
    assert callable(text_to_tokens)

    tokens = text_to_tokens("hello")
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)

    assert text_from_tokens(tokens) == "hello"
