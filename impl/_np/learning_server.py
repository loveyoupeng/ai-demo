"""Learning mode HTTP server — stdlib ``http.server`` only, zero dependencies.

Serves the learning page (static files from ``impl/_np/web/``) plus a small
JSON API on top of a loaded model — either the NumPy track (``NumPyModel``)
or the PyTorch track (``TorchModel``; the page consumes both backends'
records interchangeably because the record shapes are identical):

- ``GET  /``            → the page (and other static assets)
- ``GET  /api/model``   → model config + vocab + backend (for the page)

Request body for both POSTs:
    {"text": "the she", "n_tokens": 20, "temperature": 0.8|null,
     "top_k": 10|null, "seed": 42}

User text is tokenized through the model's char vocab; characters outside the
vocab are skipped (and reported) so the page can explain why some input
characters disappear.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from impl._np.learning import generate_with_records
from impl._np.model import NumPyModel
from shared.checkpoint import load_checkpoint
from shared.config import TransformerConfig

if TYPE_CHECKING:
    from impl._torch.layers import TorchModel

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "resource/models/learning_demo"
WEB_DIR = Path(__file__).parent / "web"


Backend = Literal["numpy", "torch"]


def load_learning_model(
    model_dir: str, backend: Backend = "numpy"
) -> tuple[NumPyModel | TorchModel, list[str] | None, TransformerConfig]:
    """Load a checkpoint (+ optional ``vocab.json`` sidecar) into the requested backend.

    ``backend`` selects which track materializes the weights:
      - ``"numpy"`` (default) → ``NumPyModel`` (float32, the NumPy track).
      - ``"torch"`` → ``TorchModel`` (float64 via ``.double()`` so the record
        adapter's float64 math matches the NumPy track's records).

    The vocab sidecar is what makes text I/O possible: without it the model
    still loads and can run on raw token IDs, but the page's text box has
    nothing to encode/decode.
    """
    params, cfg = load_checkpoint(model_dir)
    if cfg is None:
        raise ValueError(f"checkpoint {model_dir} failed registry validation")
    if backend == "numpy":
        model: NumPyModel | TorchModel = NumPyModel(cfg)
        model.load_from_numpy_dict({k: v.copy() for k, v in params.items()})
    elif backend == "torch":
        from impl._torch.layers import TorchModel as _TorchModel

        torch_model = _TorchModel(cfg).double()
        torch_model.load_from_numpy_dict({k: v.copy() for k, v in params.items()})
        torch_model.eval()
        model = torch_model
    else:
        raise ValueError(f"backend must be 'numpy' or 'torch', got {backend!r}")
    vocab: list[str] | None = None
    vocab_path = Path(model_dir) / "vocab.json"
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text())
    return model, vocab, cfg


def tokenize_text(text: str, vocab: list[str]) -> tuple[list[int], int]:
    """Map text to in-vocab token IDs; OOV characters are skipped.

    Returns (token_ids, n_skipped).
    """
    idx = {c: i for i, c in enumerate(vocab)}
    ids: list[int] = []
    skipped = 0
    for c in text.lower():
        if c in idx:
            ids.append(idx[c])
        else:
            skipped += 1
    return ids, skipped


def _sample(logits: np.ndarray, temp: float | None, top_k: int | None, rng: np.random.Generator) -> int:
    """Pick the next token from a raw logits vector.

    Greedy (argmax) when ``temp is None``; otherwise temperature-scaled
    softmax with optional top-k filtering, drawn from the caller's seeded
    RNG. This is the server's single sampling formula — the same for both
    backends, so a given (seed, temp, top_k) picks identically.
    """
    z = logits.astype(np.float64)
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
    backend: str
    vocab: list[str] | None
    web_dir: Path
    models: dict[str, NumPyModel | TorchModel]  # label → model (compare tab)
    vocabs: dict[str, list[str] | None]  # label → vocab (compare tab)

    def log_message(self, format: str, *args: object) -> None:  # noqa: D102 — quiet the default stderr noise
        logger.debug(format, *args)

    # --- static files ------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 — http.server API
        path = self.path.split("?", 1)[0]
        if path == "/api/model":
            self._send_json(self._model_info())
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
        }

    def _decode_params(self, body: dict) -> tuple[list[int], float | None, int | None, int, int | None]:
        """Shared request parsing; raises ValueError with a user-facing message."""
        if self.vocab is None:
            raise ValueError("this checkpoint has no vocab.json — text input is unavailable (raw token IDs only)")
        text = str(body.get("text", ""))
        n_tokens = int(body.get("n_tokens", 20))
        temp = body.get("temperature")
        top_k = body.get("top_k")
        seed = int(body.get("seed", 42))
        if not (1 <= n_tokens <= 512):
            raise ValueError("n_tokens must be in [1, 512]")
        ids, skipped = tokenize_text(text, self.vocab)
        if not ids:
            raise ValueError("no in-vocab characters in the input text")
        t: float | None = None if temp is None else float(temp)
        k: int | None = None if top_k is None else int(top_k)
        return ids, t, k, seed, skipped

    def _api_inference(self, body: dict) -> dict:
        """Quick generation through the track's own generator path.

        NumPy track: one O(S) ``forward_prefill`` + O(1) per-token
        ``forward_step`` calls against the full-history KV cache. The prompt
        is prefilled at its absolute position and every generated token
        attends to the ENTIRE history — the old inline loop re-windowed to
        the last ``context_length`` characters and recomputed from scratch
        every step (identical results while len(seq) <= context_length;
        beyond that the cache keeps the full history instead of dropping
        the oldest tokens).

        PyTorch track: the track's production decode — a full-window forward
        per step (the torch track has no per-token step; its flash-attention
        production path makes the recompute cheap in practice).

        Response shape (unchanged):
            {"prompt": {...}, "skipped_chars": n, "generated": {...},
             "last_step": {"logits": [...], "top_tokens": [[id, p], ...]}}
        ``last_step`` is the distribution over the token that would come
        AFTER the generated ones (one extra decode position).
        """
        ids, t, k, seed, skipped = self._decode_params(body)
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
                tok = _sample(logits, t, k, rng)
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
                tok = _sample(logits, t, k, rng)
                seq.append(tok)
                logits = _window_logits(seq[-ctx:])
            last = _window_logits(seq[-ctx:])
        z = last - last.max()
        p = np.exp(z)
        p /= p.sum()
        top = np.argsort(p)[::-1][:10]
        return {
            "prompt": {"tokens": list(ids), "text": "".join(vocab[i] for i in ids)},
            "skipped_chars": skipped,
            "generated": {"tokens": seq[len(ids) :], "text": "".join(vocab[i] for i in seq[len(ids) :])},
            "last_step": {
                "logits": [round(float(v), 6) for v in last],
                "top_tokens": [[int(i), float(p[i])] for i in top],
            },
        }

    def _api_record(self, body: dict) -> dict[str, object]:
        """Full inference record via the track's own record adapter.

        Both backends produce the identical JSON shape (the record TypedDicts
        are shared), so the page consumes either interchangeably.
        """
        ids, t, k, seed, skipped = self._decode_params(body)
        n = int(body.get("n_tokens", 20))
        if isinstance(self.model, NumPyModel):
            record = generate_with_records(self.model, self.vocab or [], list(ids), n, temp=t, top_k=k, seed=seed)
        else:
            from impl._torch import learning as torch_learning

            record = torch_learning.generate_with_records(
                self.model, self.vocab or [], list(ids), n, temp=t, top_k=k, seed=seed
            )
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
        saved_model, saved_vocab = self.model, self.vocab
        results: dict[str, object] = {}
        try:
            for label, model in self.models.items():
                self.model = model
                self.vocab = self.vocabs.get(label)
                results[label] = self._api_record(body)  # includes skipped_chars + per-model record
        finally:
            self.model, self.vocab = saved_model, saved_vocab
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
            "models": models if models is not None else {},
            "vocabs": vocabs if vocabs is not None else {},
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
) -> ThreadingHTTPServer:
    """Create the learning-mode HTTP server (bound and listening, not yet serving).

    The caller owns the serving loop: ``server.serve_forever()`` (CLI) or a
    daemon thread (tests). The ``models``/``vocabs`` sets are the named
    compare-tab models; the main model is the default ""-keyed endpoints.
    """
    handler = make_handler(model, vocab, backend=backend, models=models, vocabs=vocabs)
    server = ThreadingHTTPServer((host, port), handler)
    logger.info("learning mode: serving on http://%s:%d", host, port)
    return server
