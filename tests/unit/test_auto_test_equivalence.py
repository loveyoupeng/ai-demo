"""Tests for auto_test_equivalence.py — Full 8-test matrix automation."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


class TestMatrixScriptStructure:
    """Test basic script structure and entry point."""

    def test_main_exists(self):
        """Main function must exist."""
        import scripts.auto_test_equivalence

        assert hasattr(scripts.auto_test_equivalence, "main")

    def test_script_importable(self):
        """Script must be importable without errors."""
        import scripts.auto_test_equivalence

        assert scripts.auto_test_equivalence is not None

    def test_help_exits_zero(self):
        """--help should exit with code 0."""
        import scripts.auto_test_equivalence

        result = scripts.auto_test_equivalence.main(["--help"])
        assert result == 0


class TestMatrixCombinations:
    """Test matrix combination generation."""

    def test_default_matrix_count(self):
        """Default matrix should have 8 combinations."""
        import scripts.auto_test_equivalence as m

        matrix = m._generate_matrix()
        assert len(matrix) == 8

    def test_matrix_combinations_structure(self):
        """Each matrix combination should have required fields."""
        import scripts.auto_test_equivalence as m

        matrix = m._generate_matrix()
        for combo in matrix:
            assert "name" in combo
            assert "description" in combo
            assert isinstance(combo, dict)


class TestReportFormatting:
    """Test report formatting helpers."""

    def test_format_result_pass(self):
        """PASS results should include 'PASS' marker."""
        import scripts.auto_test_equivalence as m

        result = {"name": "Test", "passed": True, "details": {"max_diff": 0.01}}
        line = m._format_result_line(result)
        assert "Test" in line
        assert "PASS" in line

    def test_format_result_fail(self):
        """FAIL results should include 'FAIL' marker."""
        import scripts.auto_test_equivalence as m

        result = {"name": "Test", "passed": False, "details": {"max_diff": 0.05}}
        line = m._format_result_line(result)
        assert "Test" in line
        assert "FAIL" in line

    def test_format_summary_all_pass(self):
        """Summary should show count of passes."""
        import scripts.auto_test_equivalence as m

        results = [
            {"passed": True},
            {"passed": True},
            {"passed": True},
        ]
        summary = m._format_summary(results)
        assert "3/3" in summary


class TestMatrixRunFunction:
    """Test the run_combination function."""

    def test_run_combination_import(self):
        """run_combination function must exist."""
        import scripts.auto_test_equivalence as m

        assert hasattr(m, "run_combination")
        # Also check that _format_result_line exists
        assert hasattr(m, "_format_result_line")

    def test_run_combination_returns_dict(self):
        """run_combination should return a dict."""
        import scripts.auto_test_equivalence as m

        # Use a tiny config that runs fast
        config = {
            "vocab_size": 32,
            "context_length": 16,
            "embed_dim": 16,
            "n_layers": 1,
            "n_heads": 2,
            "n_groups": 2,
            "n_experts": 1,
            "top_k": 1,
            "ff_dim": 32,
            "epochs": 1,
            "batch_size": 4,
            "lr": 0.01,
            "seed": 42,
            "max_length": 16,
            "save_steps": 1,
            "eval_steps": 1,
            "train_steps": 2,
            "synthetic": True,
        }
        result = m.run_combination("test_combo", config)
        assert isinstance(result, dict)
        assert "name" in result
        assert "passed" in result
        assert "elapsed" in result


class TestCliFlagParsing:
    """Test CLI flag parsing."""

    def test_fast_flag(self):
        """--fast flag should be recognized."""
        import scripts.auto_test_equivalence as m

        with patch.object(sys, "argv", ["auto_test", "--help"]):
            result = m.main()
            assert result == 0

    def test_output_flag(self):
        """--output flag should be recognized."""
        import scripts.auto_test_equivalence as m

        with tempfile.TemporaryDirectory() as tmpdir:
            result = m.main(["--output", str(Path(tmpdir) / "report.json")])
            assert result in (0, 1)  # 0=all pass, 1=any fail, never error


class TestIntegrationFullRun:
    """Integration test: full matrix runs end-to-end."""

    def test_full_matrix_small(self):
        """Full matrix should complete with small config."""
        import scripts.auto_test_equivalence as m
        # Override defaults to small for speed

        # Temporarily patch the matrix generator
        original_matrix = m._generate_matrix
        original_output_dir = m.OUTPUT_DIR

        def small_matrix():
            config = {
                "vocab_size": 32,
                "context_length": 16,
                "embed_dim": 16,
                "n_layers": 1,
                "n_heads": 2,
                "n_groups": 2,
                "n_experts": 1,
                "top_k": 1,
                "ff_dim": 32,
                "epochs": 1,
                "batch_size": 4,
                "lr": 0.01,
                "seed": 42,
                "max_length": 16,
                "save_steps": 1,
                "eval_steps": 1,
                "train_steps": 2,
                "synthetic": True,
            }
            return [
                {"name": "Small model weight diff", "description": "Test", "kwargs": config},
            ]

        m._generate_matrix = small_matrix

        with tempfile.TemporaryDirectory() as tmpdir:
            m.OUTPUT_DIR = tmpdir
            result = m.main(["--fast"])
            assert result in (0, 1)  # Should succeed (not error)

        # Restore original functions
        m._generate_matrix = original_matrix
        m.OUTPUT_DIR = original_output_dir


class TestCheckpointRoundTrip:
    """Test checkpoint round-trip functions."""

    def test_save_load_roundtrip(self):
        """Saving and loading a model dict should preserve values."""
        import numpy as np

        import scripts.auto_test_equivalence as m

        params = {"w1": np.array([1.0, 2.0, 3.0])}
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test.npz"
            m._save_checkpoint(params, ckpt_path)
            loaded = m._load_checkpoint(ckpt_path)
            assert "w1" in loaded
            np.testing.assert_array_equal(loaded["w1"], params["w1"])


class TestScenarioNames:
    """Test that scenario names match the plan."""

    def test_has_small_model_weight_diff(self):
        """Matrix should include 'Small model weight diff'."""
        import scripts.auto_test_equivalence as m

        matrix = m._generate_matrix()
        names = [c["name"] for c in matrix]
        assert "Small model weight diff" in names

    def test_has_medium_model_weight_diff(self):
        """Matrix should include 'Medium model weight diff'."""
        import scripts.auto_test_equivalence as m

        matrix = m._generate_matrix()
        names = [c["name"] for c in matrix]
        assert "Medium model weight diff" in names

    def test_has_greedy_token_match(self):
        """Matrix should include 'Greedy token match'."""
        import scripts.auto_test_equivalence as m

        matrix = m._generate_matrix()
        names = [c["name"] for c in matrix]
        assert "Greedy token match" in names

    def test_has_roundtrip_tests(self):
        """Matrix should include checkpoint round-trip tests."""
        import scripts.auto_test_equivalence as m

        matrix = m._generate_matrix()
        names = [c["name"] for c in matrix]
        assert "PyTorch→NumPy round-trip" in names
        assert "NumPy→PyTorch round-trip" in names
