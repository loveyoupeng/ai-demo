"""Learning-mode HTTP server tests.

Starts the real server (stdlib http.server) on an ephemeral port and drives
the JSON API + static serving with urllib — the same surface the webpage
uses.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from impl._np.learning_server import load_learning_model, start_server, tokenize_text
from impl._np.model import NumPyModel
from shared.config import TransformerConfig

VOCAB = list("abcdefg ")  # 7 letters + space


def make_model() -> NumPyModel:
    return NumPyModel(
        TransformerConfig(
            vocab_size=len(VOCAB),
            context_length=16,
            embed_dim=8,
            n_layers=1,
            n_heads=2,
            seed=7,
        )
    )


def post_json(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def get_json(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


class TestTokenizeText:
    """Char → token mapping used by the text API."""

    @pytest.mark.timeout(10)
    def test_in_vocab_chars_map_in_order(self):
        ids, skipped = tokenize_text("ab cd", VOCAB)  # space is in the vocab (index 7)
        assert ids == [0, 1, 7, 2, 3]
        assert skipped == 0

    @pytest.mark.timeout(10)
    def test_oov_chars_are_skipped_and_counted(self):
        ids, skipped = tokenize_text("az bz", VOCAB)  # z ∉ vocab
        assert ids == [VOCAB.index("a"), VOCAB.index(" "), VOCAB.index("b")]
        assert skipped == 2

    @pytest.mark.timeout(10)
    def test_uppercase_normalized_to_lowercase(self):
        ids, skipped = tokenize_text("AB", VOCAB)
        assert ids == [VOCAB.index("a"), VOCAB.index("b")]
        assert skipped == 0


@pytest.mark.timeout(60)
class TestLearningServer:
    """The live HTTP API against a real model."""

    @pytest.fixture()
    def server(self):
        model = make_model()
        srv = start_server(model, VOCAB, host="127.0.0.1", port=0)
        port = srv.server_address[1]
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{port}"
        srv.shutdown()
        srv.server_close()

    def test_api_model_reports_config_vocab_params(self, server):
        code, j = get_json(f"{server}/api/model")
        assert code == 200
        assert j["has_vocab"] is True
        assert j["vocab"] == VOCAB
        assert j["config"]["vocab_size"] == len(VOCAB)
        assert j["n_params"] > 0
        assert j["has_moe"] is False

    def test_api_inference_generates_and_reports_last_step(self, server):
        code, j = post_json(f"{server}/api/inference", {"text": "ab c", "n_tokens": 5, "seed": 42})
        assert code == 200
        assert j["prompt"]["tokens"] == [VOCAB.index(c) for c in "ab c"]
        assert len(j["generated"]["tokens"]) == 5
        assert all(t in VOCAB for t in j["generated"]["text"])
        # last_step carries the full logit vector + sorted top tokens
        assert len(j["last_step"]["logits"]) == len(VOCAB)
        probs = [p for _, p in j["last_step"]["top_tokens"]]
        assert probs == sorted(probs, reverse=True)

    def test_api_inference_rejects_all_oov_input(self, server):
        code, j = post_json(f"{server}/api/inference", {"text": "xyz123", "n_tokens": 5})
        assert code == 400
        assert "in-vocab" in j["error"]

    def test_api_inference_rejects_bad_token_count(self, server):
        code, _ = post_json(f"{server}/api/inference", {"text": "ab", "n_tokens": 0})
        assert code == 400

    def test_api_record_returns_full_step_records(self, server):
        code, j = post_json(f"{server}/api/record", {"text": "ab", "n_tokens": 3, "seed": 42})
        assert code == 200
        assert len(j["steps"]) == 3
        step = j["steps"][0]
        fwd = step["forward"]
        assert len(fwd["logits"][0][-1]) == len(VOCAB)
        assert "attn" in fwd["blocks"][0]
        # the record must be a valid, self-contained JSON document
        assert json.loads(json.dumps(j))["generated"]["tokens"] == j["generated"]["tokens"]

    def test_unknown_endpoint_404(self, server):
        code, _ = get_json(f"{server}/api/nope")
        assert code == 404

    def test_static_index_served(self, server):
        with urllib.request.urlopen(f"{server}/", timeout=30) as r:
            html = r.read().decode()
        assert r.status == 200
        assert "Learning Mode" in html

    def test_static_missing_file_404(self, server):
        code, _ = get_json(f"{server}/no-such-file.html")
        assert code == 404

    def test_path_traversal_blocked(self, server):
        code, _ = get_json(f"{server}/../models/learning_demo/vocab.json")
        assert code == 404


@pytest.mark.timeout(30)
class TestLoadLearningModel:
    """Checkpoint loading for the learning page."""

    @pytest.mark.timeout(30)
    def test_demo_checkpoint_loads_with_vocab(self):
        from pathlib import Path

        demo = Path(__file__).resolve().parents[3] / "resource" / "models" / "learning_demo"
        if not demo.is_dir():
            pytest.skip("demo checkpoint not trained yet")
        model, vocab, cfg = load_learning_model(str(demo))
        assert vocab is not None
        assert len(vocab) == cfg.vocab_size
        # registry-validated load: parameter count matches the registry
        assert model.get_all_parameters()
