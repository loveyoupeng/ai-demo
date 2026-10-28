# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Unit tests for the doc-vs-code alignment check (scripts/check_docs).

The checker is the docs' own test surface: it must catch path drift and
count drift, and it must not fire on legitimate prose (globs, hypothetical
paths named in design docs, module-style references without .py).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_docs", Path(__file__).resolve().parents[2] / "scripts" / "check_docs.py"
)
assert _SPEC is not None and _SPEC.loader is not None
check_docs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_docs)

REPO_ROOT = check_docs.REPO_ROOT


def _texts_with_readme(payload: str) -> list[tuple[Path, str]]:
    """The real md corpus, with README.md's content swapped for ``payload``."""
    texts = []
    for path, text in check_docs._iter_md_files():
        texts.append((path, payload if path.name == "README.md" else text))
    return texts


class TestPathDrift:
    def test_clean_repo_passes(self) -> None:
        assert check_docs.check_paths(check_docs._iter_md_files()) == []

    def test_missing_path_is_caught(self) -> None:
        texts = _texts_with_readme("see [docs/math/nope.md](docs/math/nope.md)\n")
        errs = check_docs.check_paths(texts)
        assert len(errs) == 1 and "docs/math/nope.md" in errs[0]

    def test_file_line_is_reported(self) -> None:
        texts = _texts_with_readme("line1\n\nbroken ref: impl/_np/ghost.py here\n")
        errs = check_docs.check_paths(texts)
        assert errs[0].startswith("README.md:3:")

    def test_glob_tail_not_flagged(self) -> None:
        texts = _texts_with_readme("for f in tests/unit/_cuda/test_*.py; do\n")
        assert check_docs.check_paths(texts) == []

    def test_hypothetical_path_not_flagged(self) -> None:
        # The seam doc names a hypothetical impl/shared_torch/ module.
        texts = _texts_with_readme("a shared `impl/shared_torch/` module would\n")
        assert check_docs.check_paths(texts) == []

    def test_module_style_reference_accepted_when_module_exists(self) -> None:
        texts = _texts_with_readme("run scripts.train_demo_model_v3\n")
        assert check_docs.check_paths(texts) == []

    def test_module_style_reference_flagged_when_module_missing(self) -> None:
        texts = _texts_with_readme("run scripts/ghost_module\n")
        errs = check_docs.check_paths(texts)
        assert len(errs) == 1 and "scripts/ghost_module" in errs[0]


class TestScenarioCount:
    def test_actual_count_matches_source(self) -> None:
        # The source of truth: count Scenario( constructors in SCENARIOS.
        n = check_docs._load_scenario_count()
        assert n >= 9, "verify_equivalence lost scenarios?"

    def test_wrong_count_is_caught(self) -> None:
        n = check_docs._load_scenario_count()
        texts = _texts_with_readme("verify_equivalence runs 7 scenarios today\n")
        errs = check_docs.check_scenario_counts(texts, n)
        assert len(errs) == 1 and "claims 7" in errs[0] and f"has {n}" in errs[0]

    def test_correct_count_passes(self) -> None:
        n = check_docs._load_scenario_count()
        texts = _texts_with_readme(f"verify_equivalence runs {n} scenarios today\n")
        assert check_docs.check_scenario_counts(texts, n) == []


class TestMain:
    def test_main_exit_code_zero_on_clean_repo(self, capsys: pytest.CaptureFixture[str]) -> None:
        # The whole point: the real repo's docs are aligned right now.
        assert check_docs.main() == 0
        assert "docs aligned" in capsys.readouterr().out
