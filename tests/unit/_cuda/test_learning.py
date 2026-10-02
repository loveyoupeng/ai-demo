"""CUDA learning-adapter tests: the record contract against a tiny random model.

Extends the torch-adapter contract (tests/unit/test_learning_adapter.py) to
the CUDA track: the adapter must emit the NumPy-owned record JSON shape,
its float64 display path must agree with the model's real float32 CUDA
forward (fp32 whole-model tolerance, AGENTS.md rule 2), and generation must
be KV-cache-consistent with a fresh full forward. Cross-backend: same
weights in the NumPy track must yield identical greedy tokens.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from impl._cuda import learning as cuda_learning
from impl._cuda.model import CUDAModel
from shared.config import TransformerConfig

VOCAB = list("abcdefghijklm")  # 13 chars
PROMPT = [3, 7, 11]
SEED = 42

DENSE = TransformerConfig(
    vocab_size=len(VOCAB), context_length=16, n_layers=2, embed_dim=16, n_heads=2, n_groups=1, expert_dim=8, seed=7
)
MOE = TransformerConfig(
    vocab_size=len(VOCAB),
    context_length=16,
    n_layers=2,
    embed_dim=16,
    n_heads=2,
    n_groups=1,
    n_experts=4,
    top_k=2,
    n_shared_experts=1,
    expert_dim=8,
    seed=7,
)


def _model(cfg: TransformerConfig) -> CUDAModel:
    torch.manual_seed(cfg.seed)
    return CUDAModel(cfg)


def _keys_equal(a: object, b: object, path: str, errs: list[str]) -> None:
    """Deep key-set + list-length equality (the page's structure contract)."""
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            errs.append(f"keys at {path}: {sorted(set(a) ^ set(b)) if isinstance(b, dict) else type(b)}")
            return
        for k in a:
            _keys_equal(a[k], b[k], f"{path}.{k}", errs)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            errs.append(f"list len at {path}: {len(a)} vs {len(b) if isinstance(b, list) else type(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            if isinstance(x, (dict, list)) or isinstance(y, (dict, list)):
                _keys_equal(x, y, f"{path}[{i}]", errs)


class TestInstrumentedForward:
    @pytest.mark.timeout(120)
    def test_record_shape(self):
        model = _model(DENSE)
        rec = cuda_learning.instrumented_forward(model, torch.tensor([PROMPT]))
        assert rec["input_ids"] == [PROMPT]
        assert rec["positions"] == list(range(len(PROMPT)))
        assert len(rec["blocks"]) == DENSE.n_layers
        blk = rec["blocks"][0]
        assert set(blk) == {"ln1", "attn", "h", "ln2", "ffn", "out"}
        attn = blk["attn"]
        n_kv = len(attn["k_pre"][0][0])
        assert n_kv == DENSE.kv_heads * DENSE.head_dim  # GQA: k_pre is per-group
        S = len(PROMPT)
        assert len(attn["scores"][0][0]) == S and len(attn["scores"][0][0][0]) == S
        assert len(attn["causal_mask"][0]) == S  # prefill: (S, S) mask
        assert len(attn["k_cache"][0][0]) == S  # cache holds the whole prompt
        assert np.allclose(np.asarray(rec["softmax"]).sum(-1), 1.0, atol=1e-6)
        probs = [p for _, p in rec["top_tokens"]]
        assert probs == sorted(probs, reverse=True)

    @pytest.mark.timeout(120)
    def test_moe_record(self):
        model = _model(MOE)
        rec = cuda_learning.instrumented_forward(model, torch.tensor([PROMPT]))
        moe = rec["blocks"][0]["moe"]
        assert "ffn" not in rec["blocks"][0]
        assert moe["n_experts"] == MOE.n_experts and moe["top_k"] == MOE.top_k
        assert len(moe["expert_outs"]) == MOE.n_experts
        assert len(moe["shared_outs"]) == MOE.n_shared_experts
        S = len(PROMPT)
        assert np.asarray(moe["topk_idx"]).shape == (1, S, MOE.top_k)
        # Top-k masked + renormalized: every row's weights sum to 1.
        assert np.allclose(np.asarray(moe["weights"]).sum(-1), 1.0, atol=1e-6)

    @pytest.mark.timeout(120)
    def test_matches_real_cuda_forward(self):
        """The float64 display path agrees with the real float32 CUDA forward."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for the real-forward check")
        model = _model(DENSE)
        rec = cuda_learning.instrumented_forward(model, torch.tensor([PROMPT]))
        real = model.forward(torch.tensor([PROMPT], device="cuda")).detach().cpu()
        assert np.allclose(rec["logits"], real.numpy().astype(np.float64), rtol=1e-2, atol=1e-2)
        # The record's chosen token is the one the CUDA model would produce.
        assert int(np.argmax(rec["logits"][0][-1])) == int(real[0, -1].argmax())


class TestGenerateWithRecords:
    @pytest.mark.timeout(120)
    def test_prefill_then_decode(self):
        model = _model(DENSE)
        rec = cuda_learning.generate_with_records(model, VOCAB, PROMPT, 4, seed=SEED)
        steps = rec["steps"]
        assert len(steps) == 4
        assert steps[0]["kind"] == "prefill" and steps[0]["position"] == len(PROMPT) - 1
        assert steps[0]["input_tokens"] == PROMPT
        for i, step in enumerate(steps[1:], 1):
            assert step["step"] == i and step["kind"] == "decode"
            assert step["position"] == len(PROMPT) - 1 + i
            assert len(step["input_tokens"]) == 1  # decode embeds ONLY the new token
            attn = step["forward"]["blocks"][0]["attn"]
            t = len(PROMPT) + i  # cache length after this step's append
            assert len(attn["k_cache"][0][0]) == t
            assert len(attn["scores"][0][0][0]) == t  # decode: one row over the whole cache
            assert not np.asarray(attn["causal_mask"]).any()  # nothing masked in decode
            assert step["forward"]["input_ids"] == [step["input_tokens"]]
        assert rec["generated"]["tokens"] == [s["token"] for s in steps]
        assert rec["generated"]["text"] == "".join(VOCAB[t] for t in rec["generated"]["tokens"])
        assert rec["prompt"]["text"] == "".join(VOCAB[t] for t in PROMPT)

    @pytest.mark.timeout(120)
    def test_cache_consistency_with_full_forward(self):
        """Decode-step logits equal a fresh instrumented forward over the prefix."""
        model = _model(DENSE)
        rec = cuda_learning.generate_with_records(model, VOCAB, PROMPT, 4, seed=SEED)
        for step in rec["steps"][1:]:
            prefix = PROMPT + rec["generated"]["tokens"][: step["step"]]
            fresh = cuda_learning.instrumented_forward(model, torch.tensor([prefix]))
            assert np.allclose(step["forward"]["logits"][0][0], fresh["logits"][0][-1], rtol=1e-6, atol=1e-6)

    @pytest.mark.timeout(120)
    def test_greedy_determinism(self):
        model = _model(DENSE)
        a = cuda_learning.generate_with_records(model, VOCAB, PROMPT, 4, seed=SEED)
        b = cuda_learning.generate_with_records(model, VOCAB, PROMPT, 4, seed=SEED)
        assert a["generated"]["tokens"] == b["generated"]["tokens"]


class TestNumPyContractParity:
    """Same weights + same prompt + same seed ⇒ same tokens as the NumPy track."""

    @pytest.mark.timeout(120)
    def test_greedy_token_identity_and_structure(self):
        from impl._np.learning import generate_with_records as np_generate
        from impl._np.model import NumPyModel

        model = _model(DENSE)
        np_model = NumPyModel(DENSE)
        np_model.load_from_numpy_dict(model.get_all_parameters())

        np_rec = np_generate(np_model, VOCAB, PROMPT, 5, seed=SEED)
        cuda_rec = cuda_learning.generate_with_records(model, VOCAB, PROMPT, 5, seed=SEED)

        # Greedy: identical token sequences; logits within fp32-vs-64 parity.
        assert cuda_rec["generated"]["tokens"] == np_rec["generated"]["tokens"]
        errs: list[str] = []
        _keys_equal(np_rec, cuda_rec, "root", errs)
        assert not errs, "; ".join(errs[:5])

    @pytest.mark.timeout(120)
    def test_seeded_sampling_identity(self):
        """Seeded temp/top-k sampling draws the same tokens as the NumPy track."""
        from impl._np.learning import generate_with_records as np_generate
        from impl._np.model import NumPyModel

        model = _model(DENSE)
        np_model = NumPyModel(DENSE)
        np_model.load_from_numpy_dict(model.get_all_parameters())

        np_rec = np_generate(np_model, VOCAB, PROMPT, 5, temp=0.9, top_k=3, seed=SEED)
        cuda_rec = cuda_learning.generate_with_records(model, VOCAB, PROMPT, 5, temp=0.9, top_k=3, seed=SEED)
        assert cuda_rec["generated"]["tokens"] == np_rec["generated"]["tokens"]
