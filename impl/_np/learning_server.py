"""Learning mode HTTP server — stdlib ``http.server`` only, zero dependencies.

Serves the learning page (static files from ``impl/_np/web/``) plus a small
JSON API on top of a loaded model — either the NumPy track (``NumPyModel``)
or the PyTorch track (``TorchModel``; the page consumes both backends'
records interchangeably because the record shapes are identical):

- ``GET  /``             → the page (and other static assets)
- ``GET  /api/model``    → model config + vocab + backend (for the page)
- ``GET  /api/tokenize`` → tokenize-or-detokenize via the model's tokenizer
- ``POST /api/inference`` → run generation with per-layer activation records
- ``POST /api/record``    → a single recorded forward/step
- ``POST /api/compare``   → run two backends on the same prompt for diffing

Request body for the POST inference/record/compare endpoints:
    {"text": "the she", "n_tokens": 20, "temperature": 0.8|null,
     "top_k": 10|null, "seed": 42}

User text is tokenized through the model's char vocab; characters outside the
vocab are skipped (and reported) so the page can explain why some input
characters disappear.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from tokenizers import Tokenizer

from impl._np.learning import generate_with_records
from impl._np.model import NumPyModel
from shared.checkpoint import load_checkpoint
from shared.config import TransformerConfig
from shared.constants import REP_PENALTY

if TYPE_CHECKING:
    from impl._torch.layers import TorchModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "resource/models/learning_tool"  # BPE-trained (mirrors scripts/learning.py DEFAULT_MODEL)
WEB_DIR = Path(__file__).parent / "web"


Backend = Literal["numpy", "torch", "triton", "cuda"]


def load_learning_model(
    model_dir: str, backend: Backend = "numpy"
) -> tuple[NumPyModel | TorchModel, list[str] | None, TransformerConfig, Tokenizer | None]:
    """Load a checkpoint (+ sidecars: vocab.json + tokenizer.json) into one of
    the four tracks (numpy/torch/triton/cuda).

    Returns (model, vocab, config, tokenizer). Triton/CUDA paths import lazily so
    the learning page works on the NumPy/Torch tracks untouched even on a machine
    without a full GPU stack; they fall back only if the user picks them.
    """
    params, cfg = load_checkpoint(model_dir)
    if cfg is None:
        raise ValueError(f"checkpoint {model_dir} is missing config.json")
    if backend == "numpy":
        model: NumPyModel | TorchModel | Any = NumPyModel(cfg)
        model.load_from_numpy_dict({k: v.copy() for k, v in params.items()})
    elif backend == "torch":
        from impl._torch.layers import TorchModel as _TorchModel

        torch_model = _TorchModel(cfg).double()
        torch_model.load_from_numpy_dict({k: v.copy() for k, v in params.items()})
        torch_model.eval()
        model = torch_model
    elif backend == "triton":
        from impl._triton.model import TritonModel

        triton_model = TritonModel(cfg).cuda()
        triton_model.load_from_numpy_dict({k: v.copy() for k, v in params.items()})
        triton_model.eval()
        model = triton_model
    elif backend == "cuda":
        from impl._cuda.model import CUDAModel as _CUDAModel

        cuda_model = _CUDAModel(cfg)
        cuda_model.load_from_numpy_dict({k: v.copy() for k, v in params.items()})
        model = cuda_model
    else:
        raise ValueError(f"backend must be 'numpy'|'torch'|'triton'|'cuda', got {backend!r}")

    vocab: list[str] | None = None
    vocab_path = Path(model_dir) / "vocab.json"
    if vocab_path.is_file():
        vocab = json.loads(vocab_path.read_text())

    tok: Tokenizer | None = None
    tok_path = Path(model_dir) / "tokenizer.json"
    if tok_path.is_file():
        tok = Tokenizer.from_file(str(tok_path))
    return model, vocab, cfg, tok


def tokenize_text(text: str, vocab: list[str], tok: Tokenizer | None = None) -> tuple[list[int], int]:
    """Map text to token IDs.

    If the checkpoint carries a BPE ``tokenizer.json`` (all the SFT demo
    checkpoints do), encode with it — the model was trained on those token
    streams, so this is the only correct encode path for them. Otherwise
    fall back to the per-character vocab lookup (the old learning-demo
    path, which preserves spaces as their own vocab entries).

    Returns (token_ids, n_skipped).
    """
    if tok is not None:
        return tok.encode(text).ids, 0
    idx = {c: i for i, c in enumerate(vocab)}
    ids: list[int] = []
    skipped = 0
    for c in text.lower():
        if c in idx:
            ids.append(idx[c])
        else:
            skipped += 1
    return ids, skipped


def _sample(
    logits: np.ndarray,
    temp: float | None,
    top_k: int | None,
    rng: np.random.Generator,
    recent: list[int] | None = None,
) -> int:
    """Pick the next token from a raw logits vector.

    Greedy (argmax) when ``temp is None``; otherwise temperature-scaled
    softmax with optional top-k filtering, drawn from the caller's seeded
    RNG. This is the server's single sampling formula — the same for both
    backends, so a given (seed, temp, top_k) picks identically.

    ``recent`` carries the tokens already emitted in this response; they get
    the Holtzman repetition penalty so a tiny LM cannot lock into a
    "... ... ..." loop — without it the learning page shows degenerate
    repeats, which is precisely what it must avoid teaching.
    """
    z = logits.astype(np.float64)
    if recent:
        for tok_id in set(recent):
            if 0 <= tok_id < len(z):
                z[tok_id] = z[tok_id] / REP_PENALTY if z[tok_id] > 0 else z[tok_id] * REP_PENALTY
        # Hard block on immediate repeats: a 1-gram/2-gram no-repeat guard, the
        # same trick as HF ``no_repeat_ngram_size=2``. A tiny LM on a short
        # prompt overfits to one dominant token; penalty alone cannot beat a
        # 7.8-vs-3.0 logit gap, but banning the just-emitted 2-gram forces the
        # runner-up — which is exactly the diversity the learning page shows.
        if len(recent) >= 1:
            z[recent[-1]] = -np.inf  # never repeat the immediately previous token
    if temp is None:
        return int(np.argmax(z))
    z = z / temp
    if top_k is not None:
        kth = np.partition(z, -top_k)[-1]
        z = np.where(z < kth, -np.inf, z)
    z = z - z.max()
    p = np.exp(z)
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


class _LearningHandler(BaseHTTPRequestHandler):
    """Static file + JSON API handler with the model(s) bound at construction."""

    model: NumPyModel | TorchModel  # the "default" model (the old backend semantics)
    backend: str  # default backend (= startup -backend)
    vocab: list[str] | None
    tokenizer: Tokenizer | None  # main model's BPE (None on old char-only demos)
    web_dir: Path
    models: dict[str, NumPyModel | TorchModel]  # label → model (compare tab)
    vocabs: dict[str, list[str] | None]  # label → vocab (compare tab)
    tokenizers: dict[str, Tokenizer | None]  # label → BPE (compare tab)
    backend_models: dict[str, Any]  # backend → model of the default checkpoint
    model_dir: str  # the target checkpoint dir (sidecar drafters live under it)
    default_spec: str  # page default: plain | mtp | dspark (CLI decides; requests override)
    _drafter_cache: dict[str, Any]  # backend|family → drafter, lazily loaded

    def log_message(self, format: str, *args: object) -> None:  # noqa: D102 — quiet the default stderr noise
        logger.debug(format, *args)

    @contextmanager
    def _per_backend(self, backend: str | None):
        """Swap self.model/vocab/tokenizer for the named backend.

        Lazily loads the named track's model from the SAME default
        checkpoint on first use. Triton/CUDA loading is GPU-gated: when the
        GPU is missing or the backend couldn't be loaded, `backend_models.get`
        returns None and the endpoint raises a 400 with a user-facing
        "backend unavailable" message — never silently serving the default
        backend's output under the wrong name.
        """
        if not backend or backend == self.backend:
            yield
            return
        if backend not in ("numpy", "torch", "triton", "cuda"):
            raise ValueError(f"unknown backend {backend!r}")
        if backend not in self.backend_models:
            self.backend_models[backend] = self._try_load_backend(backend)
        bm = self.backend_models[backend]
        if bm is None:
            raise ValueError(f"backend {backend!r} unavailable on this host (needs a CUDA GPU for triton / cuda)")
        saved = (self.model, self.vocab, self.tokenizer)
        try:
            self.model = bm
            self.vocab = self._vocab_for_backend(backend)
            yield
        finally:
            self.model, self.vocab, self.tokenizer = saved

    def _try_load_backend(self, backend: str) -> Any | None:
        """Load the named track's model for the default checkpoint; None if it can't run here."""
        try:
            ckpt_dir = os.environ.get("AI_DEMO_LEARNING_MODEL", DEFAULT_MODEL_DIR)
            # Same factory the CLI uses; returns (model, vocab, cfg, tokenizer).
            model, _vocab, _cfg, _tok = load_learning_model(ckpt_dir, backend=backend)
            return model
        except Exception:
            logger.exception("backend %r not loadable on this host", backend)
            return None

    def _vocab_for_backend(self, backend: str) -> list[str] | None:
        """Vocab is checkpoint-side, not backend-side — return the shared one."""
        return self.vocab

    # --- static files ------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 — http.server API
        path = self.path.split("?", 1)[0]
        if path == "/api/model":
            self._send_json(self._model_info())
            return
        if path == "/api/tokenize":
            from urllib.parse import unquote_plus

            text = self.path.split("?", 1)[1].split("=", 1)[1] if "?" in self.path else ""
            text = unquote_plus(text)
            tokens = self.tokenizer.encode(text).ids if self.tokenizer else []
            self._send_json({"tokens": tokens, "text": text})
            return
        # static: serve from the web dir, '/' → index.html
        rel = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
        target = (self.web_dir / rel).resolve()
        if not str(target).startswith(str(self.web_dir.resolve())) or not target.is_file():
            self._send_error(404, "not found")
            return
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".map": "application/json; charset=utf-8",
            ".woff2": "font/woff2",
            ".woff": "font/woff",
            ".ttf": "font/ttf",
        }.get(target.suffix, "application/octet-stream")
        self._send_bytes(target.read_bytes(), ctype)

    # --- JSON API ----------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_error(400, "invalid JSON body")
            return
        try:
            if self.path == "/api/inference":
                self._send_json(self._api_inference(body))
            elif self.path == "/api/record":
                self._send_json(self._api_record(body))
            elif self.path == "/api/compare":
                self._send_json(self._api_compare(body))
            else:
                self._send_error(404, "unknown endpoint")
        except ValueError as e:  # user-facing validation errors from the endpoint body
            self._send_error(400, str(e))

    # --- endpoint bodies ----------------------------------------------------

    def _model_info(self) -> dict:
        cfg = self.model.config
        if isinstance(self.model, NumPyModel):
            n_params = int(sum(v.size for v in self.model.get_all_parameters().values()))
        else:
            n_params = int(sum(p.numel() for p in self.model.parameters()))
        return {
            "config": cfg.to_dict(),
            "vocab": self.vocab,
            "has_vocab": self.vocab is not None,
            "backend": self.backend,
            "n_params": n_params,
            "has_moe": cfg.has_moe(),
            "spec_available": self._spec_available(),
            "default_spec": self.default_spec if self.default_spec in self._spec_available() else "plain",
            "spec_block_size": self._spec_block_sizes(),
        }

    def _spec_block_sizes(self) -> dict[str, int]:
        """Trained block size per available family (the k slider's cap)."""
        import json as _json

        from shared.draft import DSHARK, MTP, sidecar_dir

        out: dict[str, int] = {}
        if not self.model_dir:
            return out
        for family in (MTP, DSHARK):
            meta_path = sidecar_dir(self.model_dir, family) / "draft.json"
            if meta_path.is_file():
                out[family] = int(_json.loads(meta_path.read_text())["block_size"])
        return out

    def _load_drafter(self, family: str) -> Any:
        """Load the ``family`` sidecar drafter for the CURRENT model.

        NumPy model → the NumPy drafters (``impl/_np/drafters.py``); the
        torch family (torch/triton/cuda tracks) → the torch drafters over
        the shared sidecar scheme (``impl/_torch/drafters.py``). Cached
        per (backend, family); missing sidecar → ValueError with a
        user-facing message (the page falls back to plain).
        """
        import torch as _torch

        from shared.draft import load_drafter, sidecar_dir

        if not self.model_dir:
            raise ValueError("no model_dir bound — speculative decoding unavailable")
        key = f"{self.backend}:{family}"
        if key in self._drafter_cache:
            return self._drafter_cache[key]
        meta, params = load_drafter(sidecar_dir(self.model_dir, family))
        if isinstance(self.model, NumPyModel):
            from impl._np.drafters import DSparkDrafter as NpDSpark
            from impl._np.drafters import MTPDrafter as NpMTP

            cls = NpMTP if meta.family == "mtp" else NpDSpark
            drafter = cls(meta, self.model.embedding.weight, self.model.lm_head_weight)
            drafter.load_from_numpy_dict(params)
        else:
            from impl._torch.drafters import drafter_from_sidecar

            with _torch.no_grad():
                emb = self.model.embed_tokens.weight.detach().clone().float()
                lm = self.model.lm_head.weight.detach().clone().float()
            drafter = drafter_from_sidecar(meta, params, emb, lm)
            drafter.eval()
        self._drafter_cache[key] = drafter
        return drafter

    def _spec_available(self) -> list[str]:
        """Which drafter families have sidecars for this checkpoint."""
        from shared.draft import DSHARK, MTP, sidecar_dir

        if not self.model_dir:
            return []
        out = []
        for family in (MTP, DSHARK):
            if (sidecar_dir(self.model_dir, family) / "draft.npz").is_file():
                out.append(family)
        return out

    def _decode_params(self, body: dict) -> tuple[list[int], float | None, int | None, int, int, str | None]:
        """Shared request parsing; raises ValueError with a user-facing message."""
        if self.vocab is None:
            raise ValueError("this checkpoint has no vocab.json — text input is unavailable (raw token IDs only)")
        text = str(body.get("text", ""))
        n_tokens = int(body.get("n_tokens", 20))
        temp = body.get("temperature")
        top_k = body.get("top_k")
        seed = int(body.get("seed", 42))
        backend = body.get("backend")  # numpy | torch | triton | cuda (allowed)
        if not (1 <= n_tokens <= 512):
            raise ValueError("n_tokens must be in [1, 512]")
        ids, skipped = tokenize_text(text, self.vocab, self.tokenizer)
        if not ids:
            raise ValueError("no in-vocab characters in the input text")
        t: float | None = None if temp is None else float(temp)
        k: int | None = None if top_k is None else int(top_k)
        return ids, t, k, seed, skipped, backend

    def _api_inference(self, body: dict) -> dict:
        with self._per_backend(body.get("backend")):
            return self._run_inference(body)

    def _run_inference(self, body: dict) -> dict:
        """Quick generation through the track's own generator path.

        NumPy track: one O(S) ``forward_prefill`` + O(1) per-token
        ``forward_step`` calls against the full-history KV cache. The prompt
        is prefilled at its absolute position and every generated token
        attends to the ENTIRE history — the old inline loop re-windowed to
        the last ``context_length`` characters and recomputed from scratch
        every step (identical results while len(seq) <= context_length;
        beyond that the cache keeps the full history instead of dropping
        the oldest tokens).

        PyTorch/Triton/CUDA tracks: a full-window forward per step.
        PROD: every backend has a KV cache (``make_cache`` /
        ``forward_prefill`` / ``forward_step`` in ``layers.py``); this
        display path deliberately re-forwards the window per step for
        simplicity instead of threading per-track caches.

        Response shape (unchanged):
            {"prompt": {...}, "skipped_chars": n, "generated": {...},
             "last_step": {"logits": [...], "top_tokens": [[id, p], ...]}}
        ``last_step`` is the distribution over the token that would come
        AFTER the generated ones (one extra decode position).
        """
        spec = str(body.get("spec") or self.default_spec)
        if spec not in ("plain", "mtp", "dspark"):
            raise ValueError(f"spec must be plain|mtp|dspark, got {spec!r}")
        ids, t, k, seed, skipped, backend = self._decode_params(body)
        if spec != "plain":
            return self._run_spec_inference(
                body, ids, n=int(body.get("n_tokens", 20)), temp=t, top_k=k, seed=seed, skipped=skipped, spec=spec
            )
        n = int(body.get("n_tokens", 20))
        vocab = self.vocab
        assert vocab is not None
        model = self.model
        ctx = model.config.context_length
        rng = np.random.default_rng(seed)
        seq = list(ids)
        if isinstance(model, NumPyModel):
            logits, cache = model.forward_prefill(
                np.array([seq[-ctx:]], dtype=np.int32), position_offset=max(0, len(seq) - ctx)
            )
            logits = logits[0, -1].astype(np.float64)
            for _ in range(n):
                tok = _sample(logits, t, k, rng, recent=seq[len(ids) :])
                seq.append(tok)
                logits = model.forward_step(np.array([[tok]], dtype=np.int32), len(seq) - 1, cache)
                logits = logits[0, 0].astype(np.float64)
            last = model.forward_step(np.array([seq[-1:]], dtype=np.int32), len(seq) - 1, cache)
            last = last[0, 0].astype(np.float64)
        else:
            import torch

            def _window_logits(window: list[int]) -> np.ndarray:
                with torch.no_grad():
                    out = model(torch.tensor([window], dtype=torch.int64))
                return out[0, -1].double().detach().cpu().numpy().astype(np.float64)

            logits = _window_logits(seq[-ctx:])
            for _ in range(n):
                tok = _sample(logits, t, k, rng, recent=seq[len(ids) :])
                seq.append(tok)
                logits = _window_logits(seq[-ctx:])
            last = _window_logits(seq[-ctx:])
        z = last - last.max()
        p = np.exp(z)
        p /= p.sum()
        top = np.argsort(p)[::-1][:10]
        return {
            "prompt": {
                "tokens": list(ids),
                "text": self.tokenizer.decode(list(ids)) if self.tokenizer else "".join(vocab[i] for i in ids),
            },
            "skipped_chars": skipped,
            "generated": {
                "tokens": seq[len(ids) :],
                "text": self.tokenizer.decode(seq[len(ids) :])
                if self.tokenizer
                else "".join(vocab[i] for i in seq[len(ids) :]),
            },
            "last_step": {
                "logits": [round(float(v), 6) for v in last],
                "top_tokens": [[int(i), float(p[i])] for i in top],
            },
        }

    def _run_spec_inference(
        self,
        body: dict,
        ids: list[int],
        n: int,
        temp: float | None,
        top_k: int | None,
        seed: int,
        skipped: int,
        spec: str,
    ) -> dict:
        """Speculative generation (ADR 0003) + the paired plain baseline.

        Greedy speculation (the lossless rule) over the track's spec engine:
        the NumPy engine (``impl/_np/spec.py`` — carries the
        rejection-sampling theorem for temperature mode) or the shared
        torch-family engine (``shared/spec_engine.py``). For the honest
        same-machine speed comparison the page needs, the response also
        carries a paired PLAIN greedy run of the same prompt/length: its
        wall-clock and token count are what "tokens/sec without a
        drafter" actually means on this host (spec decision: measured
        TPS of the active mode + the plain counterfactual, never a stored
        average). The two runs share nothing (fresh caches, fresh RNG).
        """
        import time as _time

        if temp is not None and not isinstance(self.model, NumPyModel):
            # Sampled speculation is the NumPy track's theorem path; the
            # torch family runs greedy verification only (production rule).
            raise ValueError("sampled speculative decoding is implemented on the NumPy track only")
        drafter = self._load_drafter(spec)
        prompt_arr = np.array([ids], dtype=np.int32)
        k_req = body.get("k")

        t0 = _time.perf_counter()
        if isinstance(self.model, NumPyModel):
            from impl._np.spec import SpeculativeGenerator

            gen = SpeculativeGenerator(self.model, drafter)
            if temp is not None:
                _seq, stats = gen.generate_sampled(prompt_arr, n, temperature=temp, k=k_req, seed=seed)
            else:
                _seq, stats = gen.generate_greedy(prompt_arr, n, k=k_req)
        else:
            import torch as _torch

            from shared.spec_engine import SpeculativeGenerator as TorchSpecGen

            with _torch.no_grad():
                gen = TorchSpecGen(self.model, drafter)
                _seq, stats = gen.generate_greedy(prompt_arr, n, k=k_req)
        spec_ms = (_time.perf_counter() - t0) * 1000
        gen_ids = [int(x) for x in np.asarray(_seq)[0][len(ids) :]]

        # Paired plain baseline: same prompt, same length, greedy, no drafter.
        t1 = _time.perf_counter()
        plain_ids = self._plain_greedy_ids(ids, n)
        plain_ms = (_time.perf_counter() - t1) * 1000

        n_gen = len(gen_ids)
        vocab = self.vocab or []
        decode = self.tokenizer.decode if self.tokenizer else (lambda toks: "".join(vocab[i] for i in toks))
        return {
            "prompt": {"tokens": list(ids), "text": decode(list(ids))},
            "skipped_chars": skipped,
            "generated": {"tokens": gen_ids, "text": decode(gen_ids)},
            "spec": {
                "mode": spec,
                "k": k_req,
                **{kk: vv for kk, vv in stats.items() if kk != "rounds"},
                "rounds": stats.get("rounds", []),
                "tokens_per_sec": round(n_gen / max(spec_ms / 1000, 1e-9), 2),
                "elapsed_ms": round(spec_ms, 2),
            },
            "plain_baseline": {
                "tokens": plain_ids,
                "tokens_per_sec": round(len(plain_ids) / max(plain_ms / 1000, 1e-9), 2),
                "elapsed_ms": round(plain_ms, 2),
            },
        }

    def _plain_greedy_ids(self, ids: list[int], n: int) -> list[int]:
        """Plain greedy baseline (no drafter) on the CURRENT model, fresh cache.

        Uses the track's own generator path — the same one /api/inference
        runs with temperature=None — so the comparison is same-model,
        same-machine, same-decode-rule.
        """
        model = self.model
        if isinstance(model, NumPyModel):
            from impl._np.inference import TextGenerator

            gen = TextGenerator(model, max_new_tokens=n, temperature=0.0)
            seq = gen.generate_greedy(np.array([ids], dtype=np.int32))
            return [int(x) for x in seq[0][len(ids) :]]
        import torch

        from shared.generator import TextGenerator as TorchGen

        with torch.no_grad():
            gen = TorchGen(model, max_new_tokens=n, temperature=0.0)
            seq = gen.generate_greedy(torch.tensor([ids], dtype=torch.int64))
        return [int(x) for x in seq[0][len(ids) :].tolist()]

    def _api_record(self, body: dict) -> dict[str, object]:
        with self._per_backend(body.get("backend")):
            return self._run_record(body)

    def _run_record(self, body: dict) -> dict[str, object]:
        """Full inference record via the track's own record adapter.

        All four tracks produce the identical JSON shape (the record TypedDicts
        are shared), so the page consumes any of them interchangeably.
        Dispatch is by model class (the backend the last `_per_backend`
        swap installed), not by the `backend` request key — the two are kept
        in sync by `_per_backend`, and using the class keeps this honest
        even if a call site forgot the swap.
        """
        spec = str(body.get("spec") or self.default_spec)
        drafter = None
        if spec in ("mtp", "dspark"):
            drafter = self._load_drafter(spec)
        ids, t, k, seed, skipped, backend = self._decode_params(body)
        n = int(body.get("n_tokens", 20))
        model = self.model
        if isinstance(model, NumPyModel):
            record = generate_with_records(
                model,
                self.vocab or [],
                list(ids),
                n,
                temp=t,
                top_k=k,
                seed=seed,
                spec=spec,
                k=body.get("k"),
                drafter=drafter,
            )
        else:
            # Class name shown only for diagnostics; all four records share one schema.
            cls_name = type(model).__name__
            if cls_name == "TorchModel":
                from impl._torch import learning as torch_learning

                record = torch_learning.generate_with_records(
                    model,
                    self.vocab or [],
                    list(ids),
                    n,
                    temp=t,
                    top_k=k,
                    seed=seed,
                    spec=spec,
                    k=body.get("k"),
                    drafter=drafter,
                )
            elif cls_name == "TritonModel":
                from impl._triton import learning as triton_learning

                record = triton_learning.generate_with_records(
                    model,
                    self.vocab or [],
                    list(ids),
                    n,
                    temp=t,
                    top_k=k,
                    seed=seed,
                    spec=spec,
                    k=body.get("k"),
                    drafter=drafter,
                )
            elif cls_name == "CUDAModel":
                from impl._cuda import learning as cuda_learning

                record = cuda_learning.generate_with_records(
                    model,
                    self.vocab or [],
                    list(ids),
                    n,
                    temp=t,
                    top_k=k,
                    seed=seed,
                    spec=spec,
                    k=body.get("k"),
                    drafter=drafter,
                )
            else:
                raise ValueError(f"no record adapter for model class {cls_name!r}")
        record_out: dict[str, object] = dict(record)
        record_out["skipped_chars"] = skipped
        return record_out

    def _api_compare(self, body: dict[str, object]) -> dict[str, object]:
        """Run the same prompt over every loaded model and return their records.

        The compare tab is where the model's weight update *means* something:
        same input, different records (per-step logits, decoded tokens, and
        the same for every model in ``self.models`` (the default model is
        excluded; it lives at position 0 in the tab skeleton)).
        """
        saved_model, saved_vocab, saved_tok = self.model, self.vocab, self.tokenizer
        results: dict[str, object] = {}
        try:
            for label, model in self.models.items():
                self.model = model
                self.vocab = self.vocabs.get(label)
                self.tokenizer = self.tokenizers.get(label)
                results[label] = self._api_record(body)  # includes skipped_chars + per-model record
        finally:
            self.model, self.vocab, self.tokenizer = saved_model, saved_vocab, saved_tok
        return results

    # --- response helpers ----------------------------------------------------

    def _send_json(self, obj: object) -> None:
        data = json.dumps(obj).encode("utf-8")
        self._send_bytes(data, "application/json; charset=utf-8")

    def _send_bytes(self, data: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, code: int, message: str) -> None:
        data = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_handler(
    model: NumPyModel | TorchModel,
    vocab: list[str] | None,
    backend: str = "numpy",
    web_dir: Path = WEB_DIR,
    models: dict[str, NumPyModel | TorchModel] | None = None,
    vocabs: dict[str, list[str] | None] | None = None,
    tokenizer: Tokenizer | None = None,
    tokenizers: dict[str, Tokenizer | None] | None = None,
    model_dir: str = "",
    default_spec: str = "plain",
) -> type[BaseHTTPRequestHandler]:
    """Bind the model + backend + vocab + web dir into a handler class
    (closure over state). The compare tab adds its secondary models via
    ``models``; model/vocab stay as defaults for the ""-keyed endpoints.
    """
    return type(
        "LearningHandler",
        (_LearningHandler,),
        {
            "model": model,
            "backend": backend,
            "vocab": vocab,
            "web_dir": web_dir,
            "tokenizer": tokenizer,
            "models": models if models is not None else {},
            "vocabs": vocabs if vocabs is not None else {},
            "tokenizers": tokenizers if tokenizers is not None else {},
            "backend_models": {},  # lazy per-backend caches, populated on first request
            "model_dir": model_dir,
            "default_spec": default_spec,
            "_drafter_cache": {},
        },
    )


def start_server(
    model: NumPyModel | TorchModel,
    vocab: list[str] | None,
    host: str = "0.0.0.0",
    port: int = 8080,
    backend: str = "numpy",
    models: dict[str, NumPyModel | TorchModel] | None = None,
    vocabs: dict[str, list[str] | None] | None = None,
    tokenizer: Tokenizer | None = None,
    tokenizers: dict[str, Tokenizer | None] | None = None,
    model_dir: str = "",
    default_spec: str = "plain",
) -> ThreadingHTTPServer:
    """Create the learning-mode HTTP server (bound and listening, not yet serving).

    The caller owns the serving loop: ``server.serve_forever()`` (CLI) or a
    daemon thread (tests). The ``models``/``vocabs``/``tokenizers`` sets are
    the named compare-tab models; the main model is the default ""-keyed
    endpoints.

    ``model_dir`` is the target checkpoint directory — the root the
    speculative-decoding sidecar drafters (``draft_mtp/``, ``draft_dspark/``)
    are loaded from when a request picks ``spec: mtp|dspark`` (ADR 0003).
    ``default_spec`` is the mode the page starts in (the CLI default is
    ``mtp`` when drafters exist; requests override per call).
    """
    handler = make_handler(
        model,
        vocab,
        backend=backend,
        models=models,
        vocabs=vocabs,
        tokenizer=tokenizer,
        tokenizers=tokenizers,
        model_dir=model_dir,
        default_spec=default_spec,
    )
    server = ThreadingHTTPServer((host, port), handler)
    logger.info("learning mode: serving on http://%s:%d", host, port)
    return server
