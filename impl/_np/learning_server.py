"""Learning mode HTTP server — stdlib ``http.server`` only, zero dependencies.

Serves the learning page (static files from ``impl/_np/web/``) plus a small
JSON API on top of a loaded ``NumPyModel``:

- ``GET  /``            → the page (and other static assets)
- ``GET  /api/model``   → model config + vocab (for the page to render)
- ``POST /api/inference`` → quick generation (tokens + last-step logits/top)
- ``POST /api/record``  → full inference record (every intermediate, JSON)

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

import numpy as np

from impl._np.learning import generate_with_records
from impl._np.model import NumPyModel
from shared.checkpoint import load_checkpoint
from shared.config import TransformerConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "resource/models/learning_demo"
WEB_DIR = Path(__file__).parent / "web"


def load_learning_model(model_dir: str) -> tuple[NumPyModel, list[str] | None, TransformerConfig]:
    """Load a checkpoint (+ optional ``vocab.json`` sidecar) into the NumPy model.

    The vocab sidecar is what makes text I/O possible: without it the model
    still loads and can run on raw token IDs, but the page's text box has
    nothing to encode/decode.
    """
    params, cfg = load_checkpoint(model_dir)
    if cfg is None:
        raise ValueError(f"checkpoint {model_dir} failed registry validation")
    model = NumPyModel(cfg)
    model.load_from_numpy_dict({k: v.copy() for k, v in params.items()})
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


class _LearningHandler(BaseHTTPRequestHandler):
    """Static file + JSON API handler with the model bound at construction."""

    model: NumPyModel
    vocab: list[str] | None
    web_dir: Path

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
            else:
                self._send_error(404, "unknown endpoint")
        except ValueError as e:  # user-facing validation errors from the endpoint body
            self._send_error(400, str(e))

    # --- endpoint bodies ----------------------------------------------------

    def _model_info(self) -> dict:
        cfg = self.model.config
        n_params = int(sum(v.size for v in self.model.get_all_parameters().values()))
        return {
            "config": cfg.to_dict(),
            "vocab": self.vocab,
            "has_vocab": self.vocab is not None,
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
        ids, t, k, seed, skipped = self._decode_params(body)
        n = int(body.get("n_tokens", 20))
        model, vocab = self.model, self.vocab
        assert vocab is not None
        rng = np.random.default_rng(seed)
        seq = list(ids)
        for _ in range(n):
            logits = model.forward(np.array([seq[-model.config.context_length :]], dtype=np.int32))[0, -1]
            z = logits.astype(np.float64)
            if t is None:
                tok = int(np.argmax(z))
            else:
                z = z / t
                if k is not None:
                    kth = np.partition(z, -k)[-1]
                    z = np.where(z < kth, -np.inf, z)
                z = z - z.max()
                p = np.exp(z)
                p /= p.sum()
                tok = int(rng.choice(len(p), p=p))
            seq.append(tok)
        last = model.forward(np.array([seq[-model.config.context_length :]], dtype=np.int32))
        z = last[0, -1].astype(np.float64)
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        top = np.argsort(p)[::-1][:10]
        return {
            "prompt": {"tokens": list(ids), "text": "".join(vocab[i] for i in ids)},
            "skipped_chars": skipped,
            "generated": {"tokens": seq[len(ids) :], "text": "".join(vocab[i] for i in seq[len(ids) :])},
            "last_step": {
                "logits": [round(float(v), 6) for v in last[0, -1]],
                "top_tokens": [[int(i), float(p[i])] for i in top],
            },
        }

    def _api_record(self, body: dict) -> dict[str, object]:
        ids, t, k, seed, skipped = self._decode_params(body)
        n = int(body.get("n_tokens", 20))
        record = generate_with_records(self.model, self.vocab or [], list(ids), n, temp=t, top_k=k, seed=seed)
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


def make_handler(model: NumPyModel, vocab: list[str] | None, web_dir: Path = WEB_DIR) -> type[BaseHTTPRequestHandler]:
    """Bind the model + vocab + web dir into a handler class (closure over state)."""
    return type("LearningHandler", (_LearningHandler,), {"model": model, "vocab": vocab, "web_dir": web_dir})


def start_server(
    model: NumPyModel, vocab: list[str] | None, host: str = "0.0.0.0", port: int = 8080
) -> ThreadingHTTPServer:
    """Create the learning-mode HTTP server (bound and listening, not yet serving).

    The caller owns the serving loop: ``server.serve_forever()`` (CLI) or a
    daemon thread (tests).
    """
    handler = make_handler(model, vocab)
    server = ThreadingHTTPServer((host, port), handler)
    logger.info("learning mode: serving on http://%s:%d", host, port)
    return server
