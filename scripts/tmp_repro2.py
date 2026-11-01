"""Throwaway: reproduce (a) record inspector content, (b) tag presence on default prompt."""
import json
import urllib.request

BASE = "http://127.0.0.1:8099"


def post(path, body):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


# (b) default prompt on the page = "What is 3 plus 4, times 2?" — check tags
for v in ("moe-rl", "dense-rl", "moe", "dense"):
    body = {"text": "What is 3 plus 4, times 2?", "n_tokens": 32, "seed": 42,
            "backend": "numpy", "variant": v, "spec": "plain", "top_k": 0, "temperature": None}
    r = post("/api/inference", body)
    t = r["generated"]["text"]
    print(f"{v:9s} think={'<|think|>' in t} answer={'<|answer|>' in t} tool={'<|tool_call|>' in t}: {t!r}")

# (a) record inspector: /api/record steps content
body = {"text": "What is 3 plus 4, times 2?", "n_tokens": 8, "seed": 42,
        "backend": "numpy", "variant": "moe-rl", "spec": "plain", "top_k": 0, "temperature": None}
rec = post("/api/record", body)
print()
print("record keys:", sorted(rec.keys()))
print("steps:", len(rec["steps"]), "step0 keys:", sorted(rec["steps"][0].keys()))
print("step0 forward keys:", sorted(rec["steps"][0]["forward"].keys()))
b0 = rec["steps"][0]["forward"]["blocks"][0]
print("block0 keys:", sorted(b0.keys()))
for k in ("attn", "deltanet"):
    if k in b0:
        print(f"block0[{k}] keys:", sorted(b0[k].keys()))
print("top_tokens[:3]:", rec["steps"][0]["top_tokens"][:3])
print("gen text:", repr(rec["generated"]["text"]))
