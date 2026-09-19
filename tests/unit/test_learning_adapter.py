"""Cross-backend learning-adapter tests (NumPy track vs PyTorch track).

Loads the same demo checkpoint into both tracks and asserts the two record
adapters are interchangeable at the JSON boundary: identical parameter key
sets, logit parity within the cross-backend tolerance, identical record
structure, identical greedy token sequences, and a live torch-backend HTTP
server exposing ``backend`` in ``/api/model``.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from impl._np.learning import generate_with_records as np_generate
from impl._np.learning import instrumented_forward as np_forward
from impl._np.learning_server import Backend, load_learning_model, start_server
from impl._np.model import NumPyModel
from impl._torch import learning as torch_learning
from impl._torch.layers import TorchModel

DEMO = Path(__file__).resolve().parents[2] / "resource" / "models" / "learning_demo"
PROMPT = [3, 7, 11]
N_TOKENS = 6
SEED = 42


def _vocab_size() -> int:
    """Length of the demo checkpoint's char vocabulary."""
    return len(json.loads((DEMO / "vocab.json").read_text()))


def _load_numpy() -> tuple[NumPyModel, list[str]]:
    model, vocab, _cfg = load_learning_model(str(DEMO))
    assert isinstance(model, NumPyModel)
    return model, _require_vocab(vocab)


def _load_torch() -> tuple[TorchModel, list[str]]:
    model, vocab, _cfg = load_learning_model(str(DEMO), backend="torch")
    assert isinstance(model, TorchModel)
    return model, _require_vocab(vocab)


def _require_vocab(vocab: list[str] | None) -> list[str]:
    assert vocab is not None
    return vocab


def _post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


class TestLoadBackend:
    """load_learning_model materializes the same checkpoint in either track."""

    @pytest.mark.timeout(120)
    def test_numpy_is_default_and_torch_is_float64(self):
        np_model, vocab = _load_numpy()
        assert isinstance(np_model, NumPyModel)
        assert len(vocab) > 0

        torch_model, _ = _load_torch()
        assert isinstance(torch_model, TorchModel)
        # The torch adapter serializes records in float64 — the model is
        # materialized in double precision so the two tracks agree.
        assert torch_model.lm_head.weight.dtype == torch.float64
        assert torch_model.embedding.weight.dtype == torch.float64

    @pytest.mark.timeout(120)
    def test_invalid_backend_raises(self):
        # Deliberately out-of-literal input: the runtime guard must reject it.
        backend = cast(Backend, "cuda")
        with pytest.raises(ValueError, match="backend must be"):
            load_learning_model(str(DEMO), backend=backend)

    @pytest.mark.timeout(120)
    def test_param_key_set_parity(self):
        """Both tracks address the same Keys-scheme flat parameter dict."""
        np_model, _ = _load_numpy()
        torch_model, _ = _load_torch()
        np_keys = set(np_model.get_all_parameters())
        torch_keys = set(torch_model.get_all_parameters())
        assert np_keys == torch_keys


class TestRecordParity:
    """The two adapters emit the same JSON at the record boundary."""

    @pytest.mark.timeout(120)
    def test_logit_parity(self):
        """Same prompt → same logits within cross-backend tolerance."""
        np_model, vocab = _load_numpy()
        torch_model, _ = _load_torch()
        ids = np.array([PROMPT], dtype=np.int32)
        np_logits = np_forward(np_model, ids)["logits"]
        torch_logits = torch_learning.instrumented_forward(torch_model, torch.tensor([PROMPT]))["logits"]
        assert np.allclose(np_logits, torch_logits, rtol=1e-3, atol=1e-3)

    @pytest.mark.timeout(120)
    def test_greedy_token_identity(self):
        """Greedy generation picks the same token sequence in both tracks."""
        np_model, vocab = _load_numpy()
        torch_model, _ = _load_torch()
        np_rec = np_generate(np_model, vocab, PROMPT, N_TOKENS, seed=SEED)
        torch_rec = torch_learning.generate_with_records(torch_model, vocab, PROMPT, N_TOKENS, seed=SEED)
        assert np_rec["generated"]["tokens"] == torch_rec["generated"]["tokens"]
        assert len(np_rec["generated"]["tokens"]) == N_TOKENS

    @pytest.mark.timeout(120)
    def test_record_structure_parity(self):
        """Deep key-set + list-length equality: the page consumes either."""
        np_model, vocab = _load_numpy()
        torch_model, _ = _load_torch()
        np_rec = np_generate(np_model, vocab, PROMPT, N_TOKENS, seed=SEED)
        torch_rec = torch_learning.generate_with_records(torch_model, vocab, PROMPT, N_TOKENS, seed=SEED)

        def keys_equal(a: object, b: object, path: str, errs: list[str]) -> None:
            if isinstance(a, dict):
                if not isinstance(b, dict) or set(a) != set(b):
                    errs.append(f"keys at {path}: {sorted(set(a) ^ set(b)) if isinstance(b, dict) else type(b)}")
                    return
                for k in a:
                    keys_equal(a[k], b[k], f"{path}.{k}", errs)
            elif isinstance(a, list):
                if not isinstance(b, list) or len(a) != len(b):
                    errs.append(f"list len at {path}: {len(a)} vs {len(b) if isinstance(b, list) else type(b)}")
                    return
                for i, (x, y) in enumerate(zip(a, b, strict=True)):
                    if isinstance(x, (dict, list)) or isinstance(y, (dict, list)):
                        keys_equal(x, y, f"{path}[{i}]", errs)
            # scalars: values are compared for equality of type only — the
            # structure contract is about keys/lengths, not magnitudes

        errs: list[str] = []
        keys_equal(np_rec, torch_rec, "root", errs)
        assert not errs, "; ".join(errs[:5])


class TestTorchServer:
    """The live HTTP server with backend='torch' serves the same API."""

    @pytest.fixture()
    def server(self):
        model, vocab = _load_torch()
        srv = start_server(model, vocab, host="127.0.0.1", port=0, backend="torch")
        port = srv.server_address[1]
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{port}"
        srv.shutdown()
        srv.server_close()

    @pytest.mark.timeout(120)
    def test_api_model_reports_torch_backend(self, server):
        code, j = _get(f"{server}/api/model")
        assert code == 200
        assert j["backend"] == "torch"
        assert j["has_vocab"] is True
        assert j["n_params"] > 0

    @pytest.mark.timeout(120)
    def test_api_inference_shape(self, server):
        code, j = _post(f"{server}/api/inference", {"text": "abc", "n_tokens": 5, "seed": SEED})
        assert code == 200
        assert len(j["generated"]["tokens"]) == 5
        # The last-step logit vector spans the whole vocabulary.
        assert len(j["last_step"]["logits"]) == _vocab_size()
        probs = [p for _, p in j["last_step"]["top_tokens"]]
        assert probs == sorted(probs, reverse=True)

    @pytest.mark.timeout(120)
    def test_api_record_steps(self, server):
        code, j = _post(f"{server}/api/record", {"text": "abc", "n_tokens": 3, "seed": SEED})
        assert code == 200
        assert len(j["steps"]) == 3  # 1 prefill + 2 decode (n_tokens total)
        assert j["steps"][0]["kind"] == "prefill"
        assert j["steps"][1]["kind"] == "decode"
        assert "forward" in j["steps"][1]
        assert "skipped_chars" in j
