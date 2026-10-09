"""Infer CLI --spec test: threads the flag into the NumPy spec engine.

The lossless oracle per track × family × k is pinned by
tests/cross_backend/test_spec_parity.py; this file only proves the CLI
threads `--spec` into the right engine with the right defaults.
"""

from __future__ import annotations

import numpy as np
import pytest

from impl._np.learning_server import load_learning_model


@pytest.fixture(scope="module")
def tool_model():
    return load_learning_model("resource/models/learning_tool", backend="numpy")[0]


def test_infer_spec_mtp_runs_and_matches_plain(tool_model) -> None:
    """--spec mtp on the NumPy backend runs the NumPy engine: lossless greedy."""
    import contextlib
    import io

    from scripts import infer

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = infer.generate_speculative(
            tool_model, "resource/models/learning_tool", "Once upon a time", 8, "mtp", None
        )

    assert "spec_stats" in result
    assert result["spec_stats"]["spec"] == "mtp"
    assert result["spec_stats"]["n_rounds"] > 0

    # The engine path is token-identical to plain greedy (lossless
    # regardless of drafter quality — the contract we pin).
    from impl._np.inference import TextGenerator as NpGen

    plain = NpGen(tool_model, max_new_tokens=8, temperature=0.0).generate_greedy(
        np.array([result["input_tokens"]], dtype=np.int32)
    )
    np.testing.assert_array_equal(np.array(result["full_tokens"]), plain[0])


def test_infer_spec_plain_routes_to_generate_single(tool_model) -> None:
    """--spec plain (the default) keeps the classic single generator."""
    from scripts import infer

    result = infer.generate_single(
        tool_model, tool_model.config.to_dict(), "hi", 8, temperature=0.0, top_k=0, backend="numpy"
    )
    assert "spec_stats" not in result
    assert len(result["generated_tokens"]) == 8
