# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Doc-vs-code alignment check: the docs' own test.

Every file path referenced in the repo's markdown docs must exist in the
tree, and every count the docs claim that has a source of truth in the code
must match it. Docs drift silently otherwise — this script makes the drift
fail loudly instead.

Checks:
    1. Path references — extract repo-relative path tokens (``docs/…``,
       ``impl/…``, ``scripts/…``, ``tests/…``, ``shared/…``, ``resource/…``)
       from every ``.md`` file and verify each exists. Globs (``*``) and
       URL/anchor tails are skipped.
    2. Scenario count — the docs claim a number of verify_equivalence
       scenarios in several places; the source of truth is
       ``len(SCENARIOS)`` in scripts/verify_equivalence.py. Every
       ``<N>-scenario`` / ``<N> scenarios`` mention must match.

Exit status 0 = aligned; non-zero = drift (print the offending file:line).

Usage:
    uv run python -m scripts.check_docs
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MD_FILES = [
    "README.md",
    "CONTEXT.md",
    "AGENTS.md",
]

MD_GLOBS = [
    "docs/**/*.md",
]

# Path prefixes that mean a repo-relative file/dir reference.
PATH_PREFIXES = ("docs/", "impl/", "scripts/", "tests/", "shared/", "resource/")

# Deliberately hypothetical paths named in design prose (the seam doc's
# "a shared impl/shared_torch/ module *would* ..." thought experiment).
HYPOTHETICAL = {"impl/shared_torch/"}

# Matches a path token starting at a word boundary; greedy up to whitespace
# or common punctuation.  Examples it must catch:
#   docs/architecture/design.md   impl/_np/model.py   scripts/check_docs.py
#   docs/architecture/adr/        tests/unit/_cuda/
_PATH_RE = re.compile(r"(?<![\w./-])((?:" + "|".join(re.escape(p) for p in PATH_PREFIXES) + r")[\w/*./-]*)")

# "<N>-scenario" / "<N> scenarios" claims (any N, so the count is checked
# against the source of truth, not hardcoded here).
_SCENARIO_RE = re.compile(r"\b(\d+)[ -]scenario\b|\b(\d+)\s+scenarios\b", re.IGNORECASE)


def _load_scenario_count() -> int:
    """The source of truth: len(SCENARIOS) in scripts/verify_equivalence.py."""
    src = (REPO_ROOT / "scripts" / "verify_equivalence.py").read_text()
    # Find the SCENARIOS list and count Scenario( constructors inside it.
    start = src.index("SCENARIOS:")
    end = src.index("\n]", start)
    return len(re.findall(r"Scenario\(\n", src[start:end])) or src[start:end].count("Scenario(")


def _iter_md_files() -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for rel in MD_FILES:
        p = REPO_ROOT / rel
        if p.is_file():
            files.append((p, p.read_text()))
    for pattern in MD_GLOBS:
        for p in sorted(REPO_ROOT.glob(pattern)):
            files.append((p, p.read_text()))
    return files


def check_paths(texts: list[tuple[Path, str]]) -> list[str]:
    """Every repo-relative path token in the docs must exist on disk."""
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for path, text in texts:
        for lineno, line in enumerate(text.splitlines(), start=1):
            for m in _PATH_RE.finditer(line):
                token = m.group(1).rstrip(".,;:)\u2019\u201d`")
                # Skip globs (docs/**/*.md style) and stripped-empty tails.
                if "*" in token or not token or token in HYPOTHETICAL:
                    continue
                key = (str(path.relative_to(REPO_ROOT)), token)
                if key in seen:
                    continue
                seen.add(key)
                if (REPO_ROOT / token).exists():
                    continue
                # Module-style reference (scripts/foo, shared/foo without
                # .py, e.g. "scripts/train_demo_model_v2" in prose): accept
                # when the module file exists.
                if "." not in token and (REPO_ROOT / (token + ".py")).is_file():
                    continue
                errors.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: referenced path does not exist: {token}")
    return errors


def check_scenario_counts(texts: list[tuple[Path, str]], actual: int) -> list[str]:
    """Every '<N>-scenario' / '<N> scenarios' claim must equal the source."""
    errors: list[str] = []
    for path, text in texts:
        for lineno, line in enumerate(text.splitlines(), start=1):
            for m in _SCENARIO_RE.finditer(line):
                claimed = int(m.group(1) or m.group(2))
                if claimed != actual:
                    errors.append(
                        f"{path.relative_to(REPO_ROOT)}:{lineno}: claims {claimed} scenario(s), "
                        f"but scripts/verify_equivalence.py has {actual}"
                    )
    return errors


def main() -> int:
    texts = _iter_md_files()
    errors = check_paths(texts)
    n_scenarios = _load_scenario_count()
    errors += check_scenario_counts(texts, n_scenarios)

    if errors:
        print(f"doc drift: {len(errors)} problem(s)")
        for e in errors:
            print(f"  {e}")
        return 1
    print(
        f"docs aligned: {len(texts)} md files, "
        f"{n_scenarios} verify_equivalence scenario(s), every referenced path exists"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
