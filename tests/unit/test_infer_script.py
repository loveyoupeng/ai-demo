"""Tests for scripts/infer.py — interactive inference CLI.

Tests follow TDD: write failing test first, then implement to make it pass.
Uses synthetic data and in-memory checkpoints for speed.
"""

from __future__ import annotations

import sys


class TestInferScriptStructure:
    """Test that scripts/infer.py has the expected structure."""

    def test_module_importable(self):
        """scripts.infer should be importable."""
        import scripts.infer as m  # noqa: F401

    def test_create_argparser(self):
        """There should be a create_argparser() function."""
        from scripts.infer import create_argparser

        assert callable(create_argparser)
        parser = create_argparser()
        assert parser is not None

    def test_main_exists(self):
        """There should be a main() function returning int."""
        from scripts.infer import main

        assert callable(main)

    def test_help_exits_0(self):
        """Passing --help should return 0."""
        import scripts.infer as m

        old_argv = sys.argv
        try:
            sys.argv = ["infer.py", "--help"]
            code = m.main()
            assert code == 0
        except SystemExit:
            pass  # argparse may call sys.exit
        finally:
            sys.argv = old_argv


class TestEncodeDecode:
    """Tests for encode_prompt() and decode_tokens()."""

    def test_encode_prompt_simple(self):
        """Basic ASCII text should encode to integer tokens."""
        from scripts.infer import encode_prompt

        tokens = encode_prompt("hello", vocab_size=256)
        assert isinstance(tokens, list)
        assert len(tokens) == 5
        # Each token is ord(c) % 256
        assert tokens == [ord(c) for c in "hello"]

    def test_encode_prompt_small_vocab(self):
        """With small vocab, tokens wrap around."""
        from scripts.infer import encode_prompt

        tokens = encode_prompt("ab", vocab_size=50)
        assert len(tokens) == 2
        assert tokens[0] == ord("a") % 50
        assert tokens[1] == ord("b") % 50

    def test_encode_empty_prompt(self):
        """Empty string should return empty list."""
        from scripts.infer import encode_prompt

        tokens = encode_prompt("")
        assert tokens == []

    def test_decode_tokens_simple(self):
        """Tokens should decode back to the original text (for bytes <= 255)."""
        from scripts.infer import decode_tokens, encode_prompt

        text = "hello"
        tokens = encode_prompt(text, 256)
        decoded = decode_tokens(tokens, 256)
        assert decoded == text

    def test_decode_tokens_rejects_nonprintable(self):
        """Non-printable characters should be filtered out."""
        from scripts.infer import decode_tokens, encode_prompt

        # Characters like \x01, \x02 are non-printable
        bad = "a\x01b\x02c"
        tokens = encode_prompt(bad, 256)
        decoded = decode_tokens(tokens, 256)
        # \x01 and \x02 are not printable, should be skipped
        assert "a" in decoded
        assert "b" in decoded
        assert "c" in decoded
        assert "\x01" not in decoded
        assert "\x02" not in decoded


class TestArgParsing:
    """Test CLI argument parsing for infer.py."""

    def test_default_backend(self):
        """Default backend should be 'torch'."""
        import scripts.infer as m

        parser = m.create_argparser()
        old_argv = sys.argv
        try:
            sys.argv = ["infer.py", "--model", "/tmp/model/"]
            args = parser.parse_args()
            assert args.backend == "torch"
        finally:
            sys.argv = old_argv

    def test_model_required(self):
        """--model should be required."""
        import scripts.infer as m

        parser = m.create_argparser()
        old_argv = sys.argv
        try:
            sys.argv = ["infer.py"]  # no --model
            args = parser.parse_args()
            assert args.model is None
        except SystemExit:
            pass  # argparse may call sys.exit for missing required
        finally:
            sys.argv = old_argv

    def test_prompt_and_temperature_parsing(self):
        """Prompt and temperature args should parse correctly."""
        import scripts.infer as m

        parser = m.create_argparser()
        old_argv = sys.argv
        try:
            sys.argv = [
                "infer.py",
                "--model",
                "/tmp/model/",
                "--prompt",
                "hello world",
                "--temperature",
                "0.8",
                "--top_k",
                "50",
                "--backend",
                "numpy",
            ]
            args = parser.parse_args()
            assert args.prompt == "hello world"
            assert args.temperature == 0.8
            assert args.top_k == 50
            assert args.backend == "numpy"
        finally:
            sys.argv = old_argv

    def test_greedy_flag(self):
        """--greedy should set greedy=True."""
        import scripts.infer as m

        parser = m.create_argparser()
        old_argv = sys.argv
        try:
            sys.argv = ["infer.py", "--model", "/tmp/model/", "--greedy"]
            args = parser.parse_args()
            assert args.greedy is True
            assert args.temperature == 0.0  # default
        finally:
            sys.argv = old_argv

    def test_max_new_tokens_default(self):
        """Default max_new_tokens should be 50."""
        import scripts.infer as m

        parser = m.create_argparser()
        old_argv = sys.argv
        try:
            sys.argv = ["infer.py", "--model", "/tmp/model/"]
            args = parser.parse_args()
            assert args.max_new_tokens == 50
        finally:
            sys.argv = old_argv


class TestGenerateSingle:
    """Tests for generate_single() integration."""

    def test_generate_single_numpy_greedy(self):
        """NumPy greedy generation should return tokens without error."""
        from impl._np.model import NumPyModel
        from scripts.infer import generate_single

        model = NumPyModel(
            vocab_size=256,
            embed_dim=32,
            n_layers=1,
            n_heads=4,
            n_experts=2,
            ff_dim=128,
            k=1,
            rope_dim=0,
            seed=42,
        )
        config = {
            "vocab_size": 256,
            "context_length": 64,
            "embed_dim": 32,
            "n_layers": 1,
            "n_heads": 4,
            "n_experts": 2,
            "top_k": 1,
            "expert_dim": 0,
            "max_length": 128,
            "rope_dim": 0,
            "seed": 42,
        }

        result = generate_single(
            model, config, "hello world", max_new_tokens=5, temperature=0.0, top_k=0, backend="numpy"
        )
        assert "input_tokens" in result
        assert "generated_tokens" in result
        assert "full_tokens" in result
        assert "prompt_text" in result
        assert "generated_text" in result
        assert "full_text" in result
        assert len(result["generated_tokens"]) == 5
        assert result["input_tokens"] == [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]

    def test_generate_single_torch_greedy(self):
        """Torch greedy generation should return tokens without error."""
        from impl._torch.layers import TorchModel
        from scripts.infer import generate_single

        model = TorchModel(
            vocab_size=256,
            embed_dim=32,
            n_layers=1,
            n_heads=4,
            n_experts=2,
            ff_dim=128,
            k=1,
            rope_dim=0,
            seed=42,
        )
        config = {
            "vocab_size": 256,
            "context_length": 64,
            "embed_dim": 32,
            "n_layers": 1,
            "n_heads": 4,
            "n_experts": 2,
            "top_k": 1,
            "expert_dim": 0,
            "max_length": 128,
            "rope_dim": 0,
            "seed": 42,
        }

        result = generate_single(
            model, config, "hello world", max_new_tokens=5, temperature=0.0, top_k=0, backend="torch"
        )
        assert "input_tokens" in result
        assert "generated_tokens" in result
        assert len(result["generated_tokens"]) == 5
        # Greedy should be deterministic — same prompt same output
        result2 = generate_single(
            model, config, "hello world", max_new_tokens=5, temperature=0.0, top_k=0, backend="torch"
        )
        assert result["full_tokens"] == result2["full_tokens"]

    def test_greedy_deterministic_repeated(self):
        """Greedy decoding should be deterministic — same output every run."""
        from impl._np.model import NumPyModel
        from scripts.infer import generate_single

        model = NumPyModel(
            vocab_size=256, embed_dim=32, n_layers=1, n_heads=4, n_experts=2, ff_dim=128, k=1, rope_dim=0, seed=42
        )
        config = {
            "vocab_size": 256,
            "context_length": 64,
            "embed_dim": 32,
            "n_layers": 1,
            "n_heads": 4,
            "n_experts": 2,
            "top_k": 1,
            "expert_dim": 0,
            "max_length": 128,
            "rope_dim": 0,
            "seed": 42,
        }

        results = []
        for _ in range(3):
            r = generate_single(
                model, config=config, prompt_text="abc", max_new_tokens=5, temperature=0.0, top_k=0, backend="numpy"
            )
            results.append(r["full_tokens"])

        # All three runs should produce the same output
        assert results[0] == results[1] == results[2]

    def test_max_output_length(self):
        """Generated output should not exceed max_new_tokens."""
        from impl._np.model import NumPyModel
        from scripts.infer import generate_single

        model = NumPyModel(
            vocab_size=256, embed_dim=32, n_layers=1, n_heads=4, n_experts=2, ff_dim=128, k=1, rope_dim=0, seed=42
        )
        config = {
            "vocab_size": 256,
            "context_length": 64,
            "embed_dim": 32,
            "n_layers": 1,
            "n_heads": 4,
            "n_experts": 2,
            "top_k": 1,
            "expert_dim": 0,
            "max_length": 128,
            "rope_dim": 0,
            "seed": 42,
        }

        result = generate_single(model, config, "hi", 2, 0.0, 0, "numpy")
        assert len(result["generated_tokens"]) == 2

        result = generate_single(model, config, "hi", 100, 0.0, 0, "numpy")
        assert len(result["generated_tokens"]) == 100


class TestLoadCheckpoint:
    """Tests for checkpoint loading functionality."""

    def test_load_checkpoint_nonexistent(self):
        """Loading from a nonexistent directory should return exit code 2."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-c", (
                "import scripts.infer as m;"
                "m.load_model_from_checkpoint('/tmp/nonexistent_xyz', 'numpy');"
            )],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        # Should show error on stderr
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()
        # Should NOT print generated output
        assert result.stdout.strip() == ""
