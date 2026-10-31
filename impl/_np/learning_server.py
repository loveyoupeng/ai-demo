# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
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
- ``GET  /api/params``   → the trained parameters (shape list, or one
  tensor's values with ?key=<registry key>)

Request body for the POST inference/record endpoints:
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

DEFAULT_MODEL_DIR = (
    "resource/models/learning_moe_rl"  # the full v3 pipeline model (mirrors scripts/learning.py DEFAULT_MODEL)
)
WEB_DIR = Path(__file__).parent / "web"


Backend = Literal["numpy", "torch", "triton", "cuda"]


def count_active_params(cfg: TransformerConfig) -> int:
    """Parameters that process ONE token (the MoE total-vs-active trade).

    Dense: every parameter is active. MoE (ADR 0005): the routed slice is
    top_k of the n_experts experts (plus the always-on shared experts);
    everything else — embeddings, attention, norms, lm_head, router — is
    fully active. This is the number the page's badge line shows next to
    the TOTAL so the param-matched MoE's whole point is visible.
    """
    D, V, L = cfg.embed_dim, cfg.vocab_size, cfg.n_layers
    H, G = cfg.n_heads, cfg.kv_heads
    hd = cfg.head_dim
    n = V * D  # embedding
    n += D  # final norm gamma
    n += D * V  # lm_head
    n += L * (2 * D + 4 * D * D)  # two norm gammas + q/k/v/o per layer (MHA-sized)
    if cfg.has_moe():
        FF, E = cfg.expert_dim, cfg.n_experts
        routed = cfg.top_k
        shared = cfg.n_shared_experts
        n += L * (D * E)  # router: always active (it scores every expert)
        n += L * (2 * D * FF + FF * D) * (routed + shared)  # active experts
    else:
        FF = cfg.expert_dim
        n += L * (2 * D * FF + FF * D)  # dense FFN: fully active
    # (GQA shaves the k/v columns; counted at MHA size — the badge is a
    # teaching number, the exact GQA correction is the diagram's job)
    if G != H:
        n -= L * 2 * (H - G) * hd * D
    return int(n)


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
    # Constructor injection on all four tracks: the model is BORN with the
    # checkpoint's trained weights (no post-construction load step).
    if backend == "numpy":
        from impl._np.model import NumPyModel

        model: NumPyModel | TorchModel | Any = NumPyModel(cfg, params)
    elif backend == "torch":
        from impl._torch.layers import TorchModel as _TorchModel

        model = _TorchModel(cfg, params)
        # Cast to float32 for display compatibility (the learning page
        # converts torch tensors to numpy for display; numpy doesn't
        # support bfloat16). The checkpoint itself is stored in float32;
        # BF16 compute happens during training, not serving.
        model = model.float()
        model.eval()
    elif backend == "triton":
        from impl._triton.model import TritonModel

        model = TritonModel(cfg, params).cuda().float()
        model.eval()
    elif backend == "cuda":
        import torch

        from impl._cuda.model import CUDAModel as _CUDAModel
        from impl._cuda.model import _cuda_to_device, _cuda_to_dtype

        model = _CUDAModel(cfg, params)
        # Same serving convention as the torch/triton tracks: float32.
        # The checkpoint's BF16 dtype marker is a TRAINING (AMP) property;
        # serving (and the drafter sidecars beside it, which load as
        # float32 and align to the target via ``drafter_from_sidecar``)
        # computes in float32 — a BF16 target against float32 drafter
        # weights is exactly the mixed-dtype matmul the page hit.
        _cuda_to_dtype(model, torch.float32)  # in place
        _cuda_to_device(model, torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # in place
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
    backend_models: dict[str, Any]  # backend → model of the default checkpoint
    model_dir: str  # the target checkpoint dir (sidecar drafters live under it)
    default_spec: str  # page default: plain | mtp | dspark (CLI decides; requests override)
    _drafter_cache: dict[str, Any]  # backend|family → drafter, lazily loaded
    variant_models: dict[str, Any]  # variant label ("dense"/"moe") → loaded model
    variant_dirs: dict[str, str]  # variant label → checkpoint dir (lazily loaded)
    _variant_state: dict[str, dict[str, Any]]  # variant → {vocab, tokenizer}
    default_variant: str  # which variant the page starts on
    # The label of the variant bound to self.model. EXPLICIT STATE, not model
    # identity: _per_backend swaps in a track's materialization of the SAME
    # checkpoint, and that object is not the one registered in variant_models —
    # identity probing would return None there and collapse the per-variant
    # cache keys (the dense drafter served for the moe target).
    _active_variant: str | None

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
        bkey = f"{self._active_variant}:{backend}"
        if bkey not in self.backend_models:
            self.backend_models[bkey] = self._try_load_backend(backend)
        bm = self.backend_models[bkey]
        if bm is None:
            raise ValueError(f"backend {backend!r} unavailable on this host (needs a CUDA GPU for triton / cuda)")
        saved = (self.model, self.vocab, self.tokenizer)
        try:
            self.model = bm
            self.vocab = self._vocab_for_backend(backend)
            yield
        finally:
            self.model, self.vocab, self.tokenizer = saved

    @contextmanager
    def _per_variant(self, variant: str | None):
        """Swap self.model/vocab/tokenizer for the named VARIANT checkpoint.

        Variants are DIFFERENT CHECKPOINTS of the demo families served by
        one page (the v3 pipeline, ADRs 0004/0005): "moe-rl" (the DEFAULT —
        the full pipeline's param-matched MoE: E=8 routed + 1 shared at
        dim 32 after GRPO RL), "dense-rl" (the dense FFN twin), and the
        two SFT-final checkpoints "moe" / "dense" the RL runs started
        from — so *-rl-vs-* isolates RL's delta and moe-vs-dense isolates
        the FFN geometry, both controlled. Lazily loaded per backend so
        switching stays cheap; an unknown variant raises a 400 the page
        surfaces, and a variant still training in the background
        bootstrap raises an actionable "retry shortly" 400.
        """
        target = variant or self.default_variant
        if not self.variant_dirs:
            # No registry configured (a custom --model serves one checkpoint):
            # any request runs the startup model — the dropdown is hidden.
            yield
            return
        if target not in self.variant_dirs:
            raise ValueError(f"unknown model variant {target!r}")
        if target not in self.variant_models and not os.path.isdir(self.variant_dirs[target]):
            # The background bootstrap (scripts/learning.py) is still
            # training this family's pipeline — a distinct, actionable
            # answer rather than a generic load failure.
            raise ValueError(f"model variant {target!r} is still training in the background — retry in a few minutes")
        # The startup model IS a registry entry when its dir matches: bind
        # it (record vocab/tokenizer so swapping back is symmetric).
        if target not in self.variant_models and os.path.abspath(self.model_dir) == os.path.abspath(
            self.variant_dirs[target]
        ):
            self.variant_models[target] = self.model
            self._variant_state[target] = {"vocab": self.vocab, "tokenizer": self.tokenizer}
        if self._active_variant == target:
            yield
            return
        if target not in self.variant_models:
            try:
                model, vocab, _cfg, tok = load_learning_model(self.variant_dirs[target], backend=self.backend)  # type: ignore[arg-type]
            except Exception as e:  # noqa: BLE001 — surfaced to the page as a 400
                raise ValueError(f"model variant {target!r} not loadable: {e}") from e
            self.variant_models[target] = model
            self._variant_state[target] = {"vocab": vocab, "tokenizer": tok}
        saved = (self.model, self.vocab, self.tokenizer, self.model_dir, self._active_variant)
        try:
            vs = self._variant_state[target]
            self.model = self.variant_models[target]
            self.vocab = vs["vocab"]
            self.tokenizer = vs["tokenizer"]
            self.model_dir = self.variant_dirs[target]
            self._active_variant = target
            yield
        finally:
            self.model, self.vocab, self.tokenizer, self.model_dir, self._active_variant = saved

    def _try_load_backend(self, backend: str) -> Any | None:
        """Load the named track's model for the CURRENTLY BOUND checkpoint
        (the default dir, or the variant's dir when a variant is swapped in);
        None if it can't run here."""
        try:
            ckpt_dir = self.model_dir or os.environ.get("AI_DEMO_LEARNING_MODEL", DEFAULT_MODEL_DIR)
            # Same factory the CLI uses; returns (model, vocab, cfg, tokenizer).
            model, _vocab, _cfg, _tok = load_learning_model(ckpt_dir, backend=backend)  # type: ignore
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
            from urllib.parse import parse_qs

            qs = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
            with self._per_variant(qs.get("variant", [None])[0]):
                self._send_json(self._model_info())
            return
        if path == "/api/params":
            from urllib.parse import parse_qs

            qs = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
            try:
                with self._per_variant(qs.get("variant", [None])[0]), self._per_backend(qs.get("backend", [None])[0]):
                    self._send_json(self._api_params(qs.get("key", [None])[0]))
            except (ValueError, KeyError) as e:
                self._send_error(404, str(e))
            return
        if path == "/api/tokenize":
            from urllib.parse import parse_qs

            qs = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
            with self._per_variant(qs.get("variant", [None])[0]):
                # parse_qs handles the percent-encoding AND the parameter
                # separation; the old naive split("=", 1) glued every later
                # query param onto the text (e.g. text=intentional&variant=moe
                # tokenized "intentional&variant=moe" — the &|v|ar|i|an|t|=|m|o|e
                # chips the page showed).
                text = qs.get("text", [""])[0]
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
            # the total-vs-active trade the param-matched MoE exists to
            # show (ADR 0005): which slice of n_params stands behind ONE
            # token (routed top_k of E + the shared experts).
            "n_params_active": count_active_params(cfg),
            "has_moe": cfg.has_moe(),
            "spec_available": self._spec_available(),
            "default_spec": self.default_spec if self.default_spec in self._spec_available() else "plain",
            "spec_block_size": self._spec_block_sizes(),
            # model-variant registry: which checkpoints this page can serve,
            # and which of them are still training in a background bootstrap
            # (the page shows "training…" instead of failing the switch)
            "variants": sorted(self.variant_dirs.keys()),
            "training_variants": sorted(v for v, d in self.variant_dirs.items() if not os.path.isdir(os.path.join(d))),
            "variant": self._active_variant or self.default_variant,
            "default_variant": self.default_variant,
        }

    def _api_params(self, key: str | None) -> dict[str, object]:
        """The trained parameters (the checkpoint's weights), serialized.

        GET /api/params            → {key: shape} for every registry entry
        GET /api/params?key=<key>  → {key: {shape, values}} — one tensor

        Same registry key scheme as the checkpoint (shared/registry.py —
        the HF-Llama names: model.layers.0.self_attn.q_proj.weight, ...),
        read through the track's get_all_parameters() (one storage binding
        per track). Values are the LIVE bound weights — variant- and
        backend-scoped by the request's contexts — rounded to 6 decimals
        like every record.
        """
        import numpy as _np

        params = self.model.get_all_parameters()
        if key is not None:
            arr = _np.asarray(params[key])  # KeyError → the 404 handler
            return {key: {"shape": list(arr.shape), "values": _np.round(arr.astype(_np.float64), 6).tolist()}}
        return {k: list(_np.asarray(v).shape) for k, v in params.items()}

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
        per (backend, variant, family) — each checkpoint has its own
        sidecars; missing sidecar → ValueError with a
        user-facing message (the page falls back to plain).
        """

        from shared.draft import load_drafter, sidecar_dir

        if not self.model_dir:
            raise ValueError("no model_dir bound — speculative decoding unavailable")
        fam = "np" if isinstance(self.model, NumPyModel) else "torch"
        sub = ""
        if fam == "torch":
            p = next(self.model.parameters(), None) if hasattr(self.model, "parameters") else None  # type: ignore
            if p is not None:
                sub = f":{p.device}:{p.dtype}"
        key = f"{fam}{sub}:{self._active_variant}:{family}"
        if key in self._drafter_cache:
            return self._drafter_cache[key]
        meta, params = load_drafter(sidecar_dir(self.model_dir, family))
        if isinstance(self.model, NumPyModel):
            from impl._np.drafters import DSparkDrafter as NpDSpark
            from impl._np.drafters import MTPDrafter as NpMTP

            cls = NpMTP if meta.family == "mtp" else NpDSpark
            drafter = cls(meta, self.model.embedding.weight, self.model.lm_head_weight, params=params)
        else:
            from impl._torch.drafters import drafter_from_sidecar
            from shared.device_util import compute_device, shared_embedding_and_head

            emb, lm = shared_embedding_and_head(self.model)
            drafter = drafter_from_sidecar(meta, params, emb, lm)  # type: ignore
            drafter = drafter.to(device=compute_device(self.model))
            if hasattr(drafter, "eval"):
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
        # T=0.0 means GREEDY (the page's convention) — collapse it to
        # None here so the sampled paths (which would divide by T) are
        # never entered with a degenerate temperature.
        t: float | None = None if temp is None or float(temp) <= 0.0 else float(temp)
        k: int | None = None if top_k is None else int(top_k)
        return ids, t, k, seed, skipped, backend

    def _decode_spec_params(self, body: dict) -> tuple[str, int | None]:
        """Validate the speculative-decoding request fields (BOTH endpoints
        run through this — one place answers "what requests are legal").

        Returns (spec, k). Raises ValueError with a user-facing message on:
          - spec not in plain|mtp|dspark;
          - k outside [1, the drafter's trained block size] (a k past the
            trained max would silently draft garbage; k<=0 is a no-op).

        The temperature rule (sampled speculation = NumPy track only) is
        enforced at the call sites — it depends on the CURRENT backend.
        """
        spec = str(body.get("spec") or self.default_spec)
        if spec not in ("plain", "mtp", "dspark"):
            raise ValueError(f"spec must be plain|mtp|dspark, got {spec!r}")
        raw_k = body.get("k")
        k: int | None = None if raw_k is None else int(raw_k)
        if k is not None and spec != "plain":
            cap = self._spec_block_sizes().get(spec)
            if cap is None:
                # No trained drafter for this checkpoint: every k is
                # unverifiable, so reject the request instead of silently
                # accepting an arbitrary one (the validation must not
                # depend on which sidecars happen to exist on disk).
                raise ValueError(f"spec={spec!r} has no trained drafter for this checkpoint (k must be omitted)")
            if not 1 <= k <= cap:
                raise ValueError(f"k must be in [1, {cap}] for spec={spec!r}, got {k}")
        return spec, k

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
        ids, t, k, seed, skipped, backend = self._decode_params(body)
        spec, spec_k = self._decode_spec_params(body)
        if t is not None and spec != "plain" and not isinstance(self.model, NumPyModel):
            # Sampled speculation is the NumPy track's theorem path; the
            # torch family runs greedy verification (production at T=0).
            raise ValueError("sampled speculative decoding is implemented on the NumPy track only")
        if spec != "plain":
            return self._run_spec_inference(
                body,
                ids,
                n=int(body.get("n_tokens", 20)),
                temp=t,
                top_k=k,
                seed=seed,
                skipped=skipped,
                spec=spec,
                spec_k=spec_k,
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
                np.array([seq[-ctx:]], dtype=np.int32), position=max(0, len(seq) - ctx)
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
            # Non-Numpy backends: use the track's OWN generate_with_records
            # (the same path _run_record uses) so "generate" and "generate +
            # record" produce IDENTICAL tokens — the old inline loop used
            # the real CUDA/torch forward while _run_record used the
            # float64 display path, producing different results.

            cls_name = type(model).__name__
            if cls_name == "TorchModel":
                from impl._torch import learning as torch_learning

                record = torch_learning.generate_with_records(
                    model,
                    vocab or [],
                    list(ids),
                    n,
                    temp=t,
                    top_k=k,
                    seed=seed,
                    spec="plain",
                    k=None,
                    drafter=None,
                )
            elif cls_name == "TritonModel":
                from impl._triton import learning as triton_learning

                record = triton_learning.generate_with_records(
                    model,
                    vocab or [],
                    list(ids),
                    n,
                    temp=t,
                    top_k=k,
                    seed=seed,
                    spec="plain",
                    k=None,
                    drafter=None,
                )
            elif cls_name == "CUDAModel":
                from impl._cuda import learning as cuda_learning

                record = cuda_learning.generate_with_records(
                    model,
                    vocab or [],
                    list(ids),
                    n,
                    temp=t,
                    top_k=k,
                    seed=seed,
                    spec="plain",
                    k=None,
                    drafter=None,
                )
            else:
                raise ValueError(f"no inference path for backend {cls_name!r}")

            gen = record["generated"]
            seq[len(ids) :] = gen["tokens"]
            # last_step: reuse the last step's forward logits (the record
            # nests logits inside steps[-1]["forward"], not at the top level)
            last_step_rec = record["steps"][-1]["forward"]
            last = np.array(last_step_rec["logits"][0][-1], dtype=np.float64)
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
        spec_k: int | None = None,
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
        k_req = spec_k

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

            from shared.device_util import compute_device
            from shared.spec_engine import SpeculativeGenerator as TorchSpecGen

            prompt_t = _torch.tensor([ids], dtype=_torch.long, device=compute_device(self.model))
            with _torch.no_grad():
                gen = TorchSpecGen(self.model, drafter)
                _seq, stats = gen.generate_greedy(prompt_t, n, k=k_req)
            _seq = _seq.detach().cpu().tolist()
        spec_ms = (_time.perf_counter() - t0) * 1000
        if isinstance(_seq, np.ndarray):
            gen_ids = [int(x) for x in _seq[0][len(ids) :]]
        else:
            gen_ids = [int(x) for x in _seq[0][len(ids) :]]

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

        from shared.device_util import compute_device
        from shared.generator import TextGenerator as TorchGen

        with torch.no_grad():
            gen = TorchGen(model, max_new_tokens=n, temperature=0.0)
            seq = gen.generate_greedy(torch.tensor([ids], dtype=torch.int64, device=compute_device(model)))
        return [int(x) for x in seq[0][len(ids) :].tolist()]

    def _api_record(self, body: dict) -> dict[str, object]:
        with self._per_variant(body.get("variant")), self._per_backend(body.get("backend")):
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
        ids, t, k, seed, skipped, backend = self._decode_params(body)
        spec, spec_k = self._decode_spec_params(body)
        drafter = None
        if spec in ("mtp", "dspark"):
            if t is not None and not isinstance(self.model, NumPyModel):
                raise ValueError("sampled speculative decoding is implemented on the NumPy track only")
            drafter = self._load_drafter(spec)
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
                k=spec_k,
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
                    k=spec_k,
                    drafter=drafter,
                )
            elif cls_name == "TritonModel":
                from impl._triton import learning as triton_learning

                record = triton_learning.generate_with_records(
                    model,  # type: ignore
                    self.vocab or [],
                    list(ids),
                    n,
                    temp=t,
                    top_k=k,
                    seed=seed,
                    spec=spec,
                    k=spec_k,
                    drafter=drafter,
                )
            elif cls_name == "CUDAModel":
                from impl._cuda import learning as cuda_learning

                record = cuda_learning.generate_with_records(
                    model,  # type: ignore
                    self.vocab or [],
                    list(ids),
                    n,
                    temp=t,
                    top_k=k,
                    seed=seed,
                    spec=spec,
                    k=spec_k,
                    drafter=drafter,
                )
            else:
                raise ValueError(f"no record adapter for model class {cls_name!r}")
        record_out: dict[str, object] = dict(record)
        record_out["skipped_chars"] = skipped
        return record_out

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
    tokenizer: Tokenizer | None = None,
    model_dir: str = "",
    default_spec: str = "plain",
    variant_dirs: dict[str, str] | None = None,
    default_variant: str = "dense",
) -> type[BaseHTTPRequestHandler]:
    """Bind the model + backend + vocab + web dir into a handler class
    (closure over state).
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
            "backend_models": {},  # lazy per-(variant, backend) caches, populated on first request
            "model_dir": model_dir,
            "default_spec": default_spec,
            "_drafter_cache": {},
            "variant_models": {},  # lazy: variant label → loaded model (default backend)
            "_active_variant": None,  # set by _per_variant; None = the startup model
            "_variant_state": {},  # variant → {vocab, tokenizer}
            "variant_dirs": dict(variant_dirs) if variant_dirs else {},
            "default_variant": default_variant,
        },
    )


def start_server(
    model: NumPyModel | TorchModel,
    vocab: list[str] | None,
    host: str = "0.0.0.0",
    port: int = 8080,
    backend: str = "numpy",
    tokenizer: Tokenizer | None = None,
    model_dir: str = "",
    default_spec: str = "plain",
    variant_dirs: dict[str, str] | None = None,
    default_variant: str = "dense",
) -> ThreadingHTTPServer:
    """Create the learning-mode HTTP server (bound and listening, not yet serving).

    The caller owns the serving loop: ``server.serve_forever()`` (CLI) or a
    daemon thread (tests).

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
        tokenizer=tokenizer,
        model_dir=model_dir,
        default_spec=default_spec,
        variant_dirs=variant_dirs,
        default_variant=default_variant,
    )
    server = ThreadingHTTPServer((host, port), handler)
    logger.info("learning mode: serving on http://%s:%d", host, port)
    return server
