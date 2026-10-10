"""Speculative-decoding engine accounting + endpoint validation tests.

Two contracts pinned here (from the 2026-10-09 architecture review):

1. **Honest forward accounting**: ``stats["n_target_forwards"]`` must equal
   the number of target forward passes ACTUALLY made — 1 (prefill) +
   2 × rounds (verify chunk + commit pass; the correction token's hidden
   does not exist until it is forwarded). A counting stub proves the
   metric can never silently drift from reality again (it drifted 2×).

2. **Endpoint validation** (the server seam): /api/inference and
   /api/record answer "what requests are legal" IDENTICALLY — bad spec
   values, k outside the trained block size, and sampled-spec on non-NumPy
   backends all raise the same 400-style ValueError on both endpoints.
   (Before this, /api/record silently degraded sampled spec to greedy.)
"""

from __future__ import annotations

import numpy as np
import pytest

from impl._np.drafters import DSparkDrafter, MTPDrafter, make_drafter_meta
from impl._np.learning_server import load_learning_model, make_handler
from impl._np.model import NumPyModel
from impl._np.spec import SpeculativeGenerator
from shared.config import TransformerConfig


def _model(seed: int = 42) -> NumPyModel:
    return NumPyModel(
        TransformerConfig(vocab_size=64, embed_dim=32, n_layers=2, n_heads=4, context_length=64, seed=seed)
    )


class TestHonestForwardAccounting:
    """reported n_target_forwards == actual target passes (the metric)."""

    @pytest.mark.parametrize("family,k", [(MTPDrafter, 4), (DSparkDrafter, 8)])
    def test_forwards_counted_honestly(self, family: type, k: int) -> None:
        model = _model()
        drafter = family(
            make_drafter_meta("mtp" if family is MTPDrafter else "dspark", model.config, k, 48),
            model.embedding.weight,
            model.lm_head_weight,
            seed=5,
        )
        gen = SpeculativeGenerator(model, drafter)

        calls = {"n": 0}
        orig = model.forward_chunk

        def counting(*a, **kw):
            calls["n"] += 1
            return orig(*a, **kw)

        model.forward_chunk = counting
        try:
            _seq, stats = gen.generate_greedy(np.array([[5, 12, 33, 7]], dtype=np.int32), 20, k=k)
        finally:
            model.forward_chunk = orig

        assert stats["n_target_forwards"] == calls["n"], (
            f"reported {stats['n_target_forwards']} but {calls['n']} target passes ran — "
            "the metric drifted from reality"
        )
        # the accounting formula: 1 prefill + 2 per round (verify + commit)
        assert stats["n_target_forwards"] == 1 + 2 * stats["n_rounds"]
        # tokens-per-target-forward is built from the HONEST count
        n_new = sum(r["new_tokens"] for r in stats["rounds"])
        assert stats["tokens_per_target_forward"] == round(n_new / stats["n_target_forwards"], 4)


class _HandlerHarness:
    """A handler instance without a socket (endpoint methods callable)."""

    def __new__(cls, backend: str = "numpy"):
        model, vocab, cfg, tok = load_learning_model("resource/models/learning_tool", backend="numpy")
        Handler = make_handler(
            model, vocab, backend="numpy", model_dir="resource/models/learning_tool", default_spec="mtp"
        )
        inst = Handler.__new__(Handler)
        inst.model = model
        inst.vocab = vocab
        inst.tokenizer = tok
        inst.backend = backend
        inst.model_dir = "resource/models/learning_tool"
        inst._drafter_cache = {}
        return inst


class TestEndpointValidation:
    """Both endpoints answer 'what requests are legal' identically."""

    @pytest.mark.parametrize(
        "body,match",
        [
            ({"spec": "bogus"}, "spec must be"),
            ({"spec": "mtp", "k": 99}, "k must be"),
            ({"spec": "mtp", "k": 0}, "k must be"),
            ({"spec": "mtp", "k": -1}, "k must be"),
        ],
    )
    def test_bad_requests_raise(self, body: dict, match: str) -> None:
        inst = _HandlerHarness()
        with pytest.raises(ValueError, match=match):
            inst._decode_spec_params(body)

    def test_legal_requests_pass(self) -> None:
        inst = _HandlerHarness()
        spec, k = inst._decode_spec_params({"spec": "mtp", "k": 4})
        assert (spec, k) == ("mtp", 4)
        spec, k = inst._decode_spec_params({})
        assert spec == "mtp" and k is None  # default_spec, no k

    def test_record_sampled_spec_on_torch_raises(self) -> None:
        """The silent-greedy hole: /api/record with temperature + spec on a
        torch-family model must raise (same answer as /api/inference)."""
        inst = _HandlerHarness()
        tm, *_ = load_learning_model("resource/models/learning_tool", backend="torch")
        inst.model = tm
        with pytest.raises(ValueError, match="NumPy track only"):
            inst._run_record({"text": "Once upon a time", "n_tokens": 6, "temperature": 0.8, "spec": "mtp", "k": 4})
