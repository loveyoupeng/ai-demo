"""Learning-mode tests: instrumented forward + generate_with_records.

Validates that the learning overlay (impl/_np/learning.py) captures
intermediates that are numerically consistent with the model's own forward,
and that recorded generation matches direct generation token-for-token.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from impl._np.learning import generate_with_records, instrumented_forward
from impl._np.model import NumPyModel
from shared.config import TransformerConfig

DEMO_MODEL = Path(__file__).resolve().parents[3] / "resource" / "models" / "learning_demo"


def tiny_model() -> NumPyModel:
    """Tiny MoE model (V=8, D=8, L=2, H=2, E=3) — fast for parity checks."""
    return NumPyModel(
        TransformerConfig(
            vocab_size=8,
            context_length=16,
            embed_dim=8,
            n_layers=2,
            n_heads=2,
            n_experts=3,
            top_k=1,
            seed=42,
        )
    )


def sample_ids(model: NumPyModel, n: int = 5) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, model.vocab_size, size=(1, n)).astype(np.int32)


class TestInstrumentedForward:
    """The step record must be consistent with the model's real forward."""

    @pytest.mark.timeout(30)
    def test_logits_match_model_forward(self):
        """Record logits equal model.forward() (within display rounding).

        The record rounds values to 8 decimal places, so exact equality is
        not expected — but the deviation must be far below any real
        numerical difference (atol=1e-6).
        """
        model = tiny_model()
        x = sample_ids(model)
        ref = model.forward(x)
        rec = instrumented_forward(model, x)
        got = np.array(rec["logits"])
        assert got.shape == ref.shape
        assert np.abs(got - ref).max() < 1e-6

    @pytest.mark.timeout(30)
    def test_softmax_rows_sum_to_one(self):
        """Every softmax row in the record sums to 1 and covers the vocab."""
        model = tiny_model()
        rec = instrumented_forward(model, sample_ids(model))
        p = np.array(rec["softmax"])
        assert p.shape == (1, 5, model.vocab_size)
        assert np.abs(p.sum(axis=-1) - 1.0).max() < 1e-6
        assert (p >= 0).all()

    @pytest.mark.timeout(30)
    def test_record_structure_moe(self):
        """MoE model record has every block/field the page needs."""
        model = tiny_model()
        rec = instrumented_forward(model, sample_ids(model, 4))
        assert sorted(rec.keys()) == [
            "blocks",
            "embedding",
            "final_norm",
            "input_ids",
            "logits",
            "positions",
            "softmax",
            "top_tokens",
        ]
        for block in rec["blocks"]:
            assert sorted(block.keys()) == ["attn", "h", "ln1", "ln2", "moe", "out"]
            attn = block["attn"]
            for key in [
                "q_pre",
                "k_pre",
                "v_pre",
                "q",
                "k",
                "v",
                "q_rope_in",
                "k_rope_in",
                "scores",
                "scores_masked",
                "causal_mask",
                "attn_weights",
                "ctx",
                "k_cache",
                "v_cache",
                "scale",
                "rope",
            ]:
                assert key in attn, f"missing attn key {key}"
            rope = attn["rope"]
            assert sorted(rope.keys()) == ["angles", "cos", "freqs", "sin"]
            moe = block["moe"]
            assert sorted(moe.keys()) == [
                "expert_outs",
                "n_experts",
                "out",
                "probs",
                "scores",
                "shared_outs",
                "top_k",
                "topk_idx",
                "weights",
            ]
            assert moe["n_experts"] == 3
            assert len(moe["expert_outs"]) == 3
            assert moe["shared_outs"] == []
        for norm in [rec["final_norm"]] + [b["ln1"] for b in rec["blocks"]]:
            assert sorted(norm.keys()) == ["gamma", "out", "rms"]

    @pytest.mark.timeout(30)
    def test_record_structure_dense(self):
        """Dense (no-MoE) model record exposes the ffn field instead of moe."""
        model = NumPyModel(TransformerConfig(vocab_size=8, embed_dim=8, n_layers=1, n_heads=2, seed=1))
        rec = instrumented_forward(model, sample_ids(model, 4))
        for block in rec["blocks"]:
            assert "ffn" in block and "moe" not in block
            assert sorted(block["ffn"].keys()) == ["ff_dim", "gate", "gated", "out", "pre_gate", "up"]

    @pytest.mark.timeout(30)
    def test_top_tokens_sorted(self):
        """top_tokens at the last position is sorted by descending probability."""
        model = tiny_model()
        rec = instrumented_forward(model, sample_ids(model, 4))
        top = rec["top_tokens"]
        probs = [p for _, p in top]
        assert probs == sorted(probs, reverse=True)
        assert len(top) == 8  # V=8 → min(10, V)


class TestGenerateWithRecords:
    """Recorded generation must equal direct generation, token for token."""

    def _reference_greedy(self, model: NumPyModel, prompt: list[int], n: int) -> list[int]:
        """Greedy reference loop straight from model.forward."""
        seq = list(prompt)
        for _ in range(n):
            x = np.array([seq[-model.config.context_length :]], dtype=np.int32)
            seq.append(int(np.argmax(model.forward(x)[0, -1])))
        return seq

    @pytest.mark.timeout(30)
    def test_greedy_matches_direct_forward(self):
        """Greedy recorded generation reproduces a direct greedy loop exactly."""
        model = tiny_model()
        prompt = [1, 3, 5, 2]
        ref = self._reference_greedy(model, prompt, 6)
        rec = generate_with_records(model, [chr(97 + i) for i in range(8)], prompt, 6, seed=42)
        assert rec["generated"]["tokens"] == ref[4:]
        assert rec["steps"][0]["token"] == ref[4]

    @pytest.mark.timeout(30)
    def test_sampling_seeded_deterministic(self):
        """Same seed → identical sampled sequence; the rng is consumed per step."""
        model = tiny_model()
        vocab = [chr(97 + i) for i in range(8)]
        prompt = [2, 4, 6]
        a = generate_with_records(model, vocab, prompt, 8, temp=1.0, top_k=4, seed=7)
        b = generate_with_records(model, vocab, prompt, 8, temp=1.0, top_k=4, seed=7)
        c = generate_with_records(model, vocab, prompt, 8, temp=1.0, top_k=4, seed=8)
        assert a["generated"]["tokens"] == b["generated"]["tokens"]
        # different seed may (or may not) differ — but each must be a valid length
        assert len(c["generated"]["tokens"]) == 8

    @pytest.mark.timeout(30)
    def test_context_window_capped(self):
        """Prompts longer than context_length truncate to the last ctx tokens,
        so generation from a long prompt equals generation from its tail."""
        model = tiny_model()  # ctx = 16
        vocab = [chr(97 + i) for i in range(8)]
        rng = np.random.default_rng(3)
        long_prompt = [int(v) for v in rng.integers(0, 8, size=30)]
        tail = long_prompt[-16:]
        a = generate_with_records(model, vocab, long_prompt, 3, seed=1)
        b = generate_with_records(model, vocab, tail, 3, seed=1)
        assert a["generated"]["tokens"] == b["generated"]["tokens"]
        assert a["steps"][0]["input_tokens"] == tail

    @pytest.mark.timeout(30)
    def test_prefill_then_cached_decode_steps(self):
        """Step 0 prefills the prompt; each decode step processes only the new
        token, and its K/V cache grows by exactly one row per step."""
        model = tiny_model()
        prompt = [1, 2, 3]
        rec = generate_with_records(model, [chr(97 + i) for i in range(8)], prompt, 4, seed=42)
        gen = rec["generated"]["tokens"]
        s0 = rec["steps"][0]
        assert s0["kind"] == "prefill"
        assert s0["input_tokens"] == prompt
        assert s0["forward"]["positions"] == [0, 1, 2]
        # prefill record already holds the initialized cache: one row per prompt token
        k0 = s0["forward"]["blocks"][0]["attn"]["k_cache"][0]
        assert len(k0) == model.config.n_heads and len(k0[0]) == len(prompt)
        for i, step in enumerate(rec["steps"][1:], start=1):
            assert step["kind"] == "decode"
            assert step["input_tokens"] == [gen[i - 1]]  # only the new token is computed
            assert step["forward"]["positions"] == [len(prompt) + i - 1]
            kc = step["forward"]["blocks"][0]["attn"]["k_cache"][0][0]
            assert len(kc) == len(prompt) + i  # cache grew by one row per step
        assert rec["prompt"]["tokens"] == prompt

    @pytest.mark.timeout(30)
    def test_cached_generation_matches_naive_full_recompute(self):
        """The KV-cached path generates the same tokens (and near-identical
        final logits) as the naive loop that re-forwards the whole sequence."""
        import numpy as np

        from impl._np.learning import instrumented_forward

        model = tiny_model()
        vocab = [chr(97 + i) for i in range(8)]
        prompt = [1, 3, 5, 2]
        rec = generate_with_records(model, vocab, prompt, 6)  # greedy
        # naive reference: full forward every step, argmax
        seq = list(prompt)
        naive = []
        for _ in range(6):
            x = np.array([seq[-model.config.context_length :]], dtype=np.int32)
            f = instrumented_forward(model, x)
            p = np.asarray(f["softmax"], dtype=np.float64)[0][-1]
            t = int(np.argmax(p))
            naive.append(t)
            seq.append(t)
        assert rec["generated"]["tokens"] == naive
        # the last recorded step's logits vs a full forward at the SAME
        # position (its input sequence is the prefix ending at that token)
        last_pos = rec["steps"][-1]["position"]
        prefix = (prompt + rec["generated"]["tokens"])[: last_pos + 1]
        whole = model.forward(np.array([prefix], dtype=np.int32))[0, last_pos]
        assert np.allclose(np.array(rec["steps"][-1]["forward"]["logits"][0][0]), whole, rtol=1e-8, atol=1e-8)


@pytest.mark.timeout(30)
class TestDemoModelCheckpoint:
    """The shipped demo checkpoint loads and serves the learning page."""

    @pytest.mark.timeout(30)
    def test_demo_model_loads_and_generates(self):
        """learning_demo checkpoint loads with its vocab sidecar and generates text."""
        import json

        from impl._np.learning_server import load_learning_model

        assert DEMO_MODEL.is_dir(), "demo checkpoint missing — run scripts/train_demo_model.py"
        model, vocab, cfg = load_learning_model(str(DEMO_MODEL))
        assert vocab is not None and len(vocab) == cfg.vocab_size
        assert " " in vocab  # space is part of the char vocab
        rec = generate_with_records(model, vocab, [vocab.index("t"), vocab.index("h"), vocab.index("e")], 5, seed=42)
        assert len(rec["generated"]["tokens"]) == 5
        text = rec["generated"]["text"]
        assert all(ch in vocab for ch in text)
        assert json.loads((DEMO_MODEL / "vocab.json").read_text()) == vocab
