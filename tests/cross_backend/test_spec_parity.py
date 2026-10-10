"""Cross-backend speculative-decoding tests (ADR 0003).

The seams (per the spec's Testing Decisions):

1. *KV-step interface* — ``forward_chunk`` on the torch-family tracks
   (torch / triton / cuda) matches the NumPy reference: prefill+chunk ==
   full forward, and a second chunk keeps parity. Verified at the
   standalone component tier (rtol=atol=1e-4, float32 GPU vs float64
   reference).
2. *Sidecar round-trip* — a sidecar saved from the NUMPY drafter loads
   into the TORCH drafter (via the shared scheme's transpose rule) and
   produces the same draft tokens and distributions. The cross-track
   round-trip guarantee extends to drafters.
3. *Lossless greedy via the shared engine* — the torch-family engine
   (shared/spec_engine.py) over TorchModel + sidecar drafter produces
   token-identical output to plain greedy via the shared generator.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from impl._np.drafters import DSparkDrafter as NpDSpark
from impl._np.drafters import MTPDrafter as NpMTP
from impl._np.drafters import make_drafter_meta as np_make_meta
from impl._np.model import NumPyModel
from shared.config import TransformerConfig
from shared.draft import DSHARK, MTP, load_drafter, save_drafter
from shared.generator import TextGenerator as TorchGen

pytestmark = pytest.mark.gpu


def _cfg(seed: int = 42, **overrides) -> TransformerConfig:
    base = {
        "vocab_size": 64,
        "embed_dim": 32,
        "n_layers": 2,
        "n_heads": 4,
        "seed": seed,
    }
    base.update(overrides)
    return TransformerConfig.from_dict(base)


def _params_np(model_np: NumPyModel) -> dict[str, np.ndarray]:
    """The NumPy model's parameter dict (registry key scheme)."""
    return {k: np.asarray(v) for k, v in model_np.get_all_parameters().items()}


def _tail_chunk_reference(ref: NumPyModel, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NumPy reference: full-sequence chunk pass + one tail chunk (positions
    6..7) — the exact logits the torch-family chunk path must reproduce."""
    cache = ref.make_cache(1)
    ref_logits, _ = ref.forward_chunk(ids, 0, cache)
    ref_tail, _ = ref.forward_chunk(np.array([[10, 11]], dtype=np.int32), 6, cache)
    return ref_logits, ref_tail


class TestTorchFamilyChunkParity:
    """forward_chunk on torch/triton/cuda vs the NumPy reference."""

    @pytest.mark.parametrize("backend", ["torch", "triton", "cuda"])
    @pytest.mark.parametrize("gqa", [False, True])
    def test_chunk_matches_numpy(self, backend: str, gqa: bool) -> None:
        overrides = {"n_groups": 2} if gqa else {}
        cfg = _cfg(seed=7, **overrides)
        ref = NumPyModel(cfg)
        ids = np.random.default_rng(1).integers(0, 64, size=(1, 6)).astype(np.int32)
        ref_logits, ref_tail = _tail_chunk_reference(ref, ids)

        if backend == "torch":
            from impl._torch.layers import TorchModel

            model = TorchModel(cfg)
            device = torch.device("cpu")
        elif backend == "triton":
            from impl._triton.model import TritonModel

            model = TritonModel(cfg).cuda()
            device = torch.device("cuda")
        else:
            from impl._cuda.model import CUDAModel

            model = CUDAModel(cfg)
            device = torch.device("cuda")
        model.load_from_numpy_dict({k: v.copy() for k, v in _params_np(ref).items()})  # type: ignore
        if hasattr(model, "eval"):
            model.eval()  # type: ignore

        cache = model.make_cache(1)
        ids_t = torch.tensor(ids, dtype=torch.int64, device=device)
        tail_t = torch.tensor([[10, 11]], dtype=torch.int64, device=device)
        with torch.no_grad():
            logits, _hidden = model.forward_chunk(ids_t, 0, cache)
            tail_logits, _h2 = model.forward_chunk(tail_t, 6, cache)
        got = logits[0].float().cpu().numpy()
        got_tail = tail_logits[0].float().cpu().numpy()

        np.testing.assert_allclose(got, ref_logits[0], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(got_tail, ref_tail[0], rtol=1e-4, atol=1e-4)


class TestSidecarCrossTrack:
    """NumPy-saved sidecar → torch drafter: same tokens, same distributions."""

    @pytest.mark.parametrize("family,np_cls", [(MTP, NpMTP), (DSHARK, NpDSpark)])
    def test_numpy_sidecar_loads_in_torch_drafter(self, tmp_path, family: str, np_cls: type) -> None:
        from impl._torch.drafters import TorchDSparkDrafter, TorchMTPDrafter

        cfg = _cfg(seed=3)
        ref = NumPyModel(cfg)
        k = 4 if family == MTP else 8
        meta = np_make_meta(family, cfg, block_size=k, ff_dim=48)
        np_drafter = np_cls(meta, ref.embedding.weight, ref.lm_head_weight, seed=5)
        save_drafter(tmp_path, meta, np_drafter.get_all_parameters())

        # Round trip through the file (not the in-memory dict).
        meta2, params2 = load_drafter(tmp_path)
        emb = torch.tensor(ref.embedding.weight, dtype=torch.float32)
        lm = torch.tensor(ref.lm_head_weight, dtype=torch.float32)
        torch_cls = TorchMTPDrafter if family == MTP else TorchDSparkDrafter
        t_drafter = torch_cls(meta2, emb, lm, seed=5)
        t_drafter.load_from_numpy_dict(params2)
        t_drafter.eval()

        hidden = np.random.default_rng(0).standard_normal((2, cfg.embed_dim)).astype(np.float32)
        tok = np.array([7, 9], dtype=np.int64)
        np_drafter.reset()
        np_t, np_p = np_drafter.draft(tok, hidden, k)
        with torch.no_grad():
            t_drafter.reset()
            t_t, t_p = t_drafter.draft(torch.tensor(tok), torch.tensor(hidden), k)

        np.testing.assert_array_equal(np_t, t_t.numpy())
        for a, b in zip(np_p, [p.numpy() for p in t_p], strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)


class TestSharedEngineLossless:
    """The torch-family engine: spec-greedy == plain greedy (TorchModel)."""

    @pytest.mark.parametrize("family", [MTP, DSHARK])
    @pytest.mark.parametrize("k_req", [1, 4])
    def test_torch_engine_greedy_parity(self, tmp_path, family: str, k_req: int) -> None:
        from impl._torch.drafters import TorchDSparkDrafter, TorchMTPDrafter
        from shared.spec_engine import SpeculativeGenerator as TorchSpecGen

        cfg = _cfg(seed=11)
        ref = NumPyModel(cfg)
        from impl._torch.layers import TorchModel

        model = TorchModel(cfg)  # float32 — the production dtype
        model.load_from_numpy_dict({kk: v.copy() for kk, v in _params_np(ref).items()})
        model.eval()

        k = 4 if family == MTP else 8
        meta = np_make_meta(family, cfg, block_size=k, ff_dim=48)
        np_cls = NpMTP if family == MTP else NpDSpark
        np_drafter = np_cls(meta, ref.embedding.weight, ref.lm_head_weight, seed=5)
        save_drafter(tmp_path, meta, np_drafter.get_all_parameters())
        meta2, params2 = load_drafter(tmp_path)
        emb = torch.tensor(ref.embedding.weight, dtype=torch.float32)
        lm = torch.tensor(ref.lm_head_weight, dtype=torch.float32)
        torch_cls = TorchMTPDrafter if family == MTP else TorchDSparkDrafter
        t_drafter = torch_cls(meta2, emb, lm, seed=5)
        t_drafter.load_from_numpy_dict(params2)
        t_drafter.eval()

        prompt = torch.tensor([[5, 12, 33, 7]], dtype=torch.int64)
        plain = TorchGen(model, max_new_tokens=20, temperature=0.0).generate_greedy(prompt)
        with torch.no_grad():
            seq, stats = TorchSpecGen(model, t_drafter).generate_greedy(prompt, 20, k=k_req)
        np.testing.assert_array_equal(seq.numpy(), plain.numpy())
        assert stats["spec"] == family
