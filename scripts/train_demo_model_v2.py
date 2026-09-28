#!/usr/bin/env python3
"""Retrain the demo-model pipeline on a small but *realistic* mixed corpus.

Stages (mirrors the documented pre → SFT-code → SFT-tool pipeline):

  1. BPE tokenizer, trained WITHOUT lowercasing so "Write"/"Python" survive.
     Sources: tinystories (English) + code_instructions (code) + tool_calls
     (JSON schemas) — same files the SFT data pipeline already downloaded.
  2. Base model: pre-train on tinystories (English fluency).
  3. SFT model: fine-tune base on code_instructions (instruction→code).
  4. Tool model: fine-tune SFT on tool_calls (OpenAI-message JSON shape).

Everything stays tiny: 512-token vocab, 3-layer model, a few thousand
sequences — minutes on the Jetson. Output: resource/models/{learning_base,
learning_sft,learning_tool} each with model.npz + config.json + vocab.json +
tokenizer.json.

Usage:
    uv run python -m scripts.train_demo_model_v2
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from impl._torch.layers import TorchModel
from shared.config import TransformerConfig

RESOURCE = Path(__file__).resolve().parent.parent / "resource"
N_STORIES = 4000
BATCH = 32
VOCAB = 512


def train_tokenizer() -> Tokenizer:
    """BPE over the mixed corpus, case preserved (no lowercasing this time)."""
    if (RESOURCE / "bpe_tokenizer.json").exists():
        tok = Tokenizer.from_file(str(RESOURCE / "bpe_tokenizer.json"))
        # detect the lowercase-fold vocabulary: 'W' would be missing
        if tok.token_to_id("W") is not None:
            return tok
    texts: list[str] = []
    for story in json.loads((RESOURCE / "tinystories_train.json").read_text())[:N_STORIES]:
        texts.append(story)
    for row in json.loads((RESOURCE / "code_instructions.json").read_text()):
        texts.append(f"{row['instruction']}\n{row['output']}")
    for row in json.loads((RESOURCE / "tool_calls.json").read_text()):
        for msg in row["messages"]:
            texts.append(msg.get("content") or "")
            for tc in msg.get("tool_calls") or []:
                texts.append(json.dumps(tc["function"], ensure_ascii=False))
        for t in row["tools"]:
            texts.append(json.dumps(t["function"], ensure_ascii=False))

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB,
        special_tokens=[
            "<|endoftext|>",
            "<|prompt|>",
            "```output",
            "```input",
            "```system",
            "<|tool|>",
            "<|tool_call|>",
            "<|tool_response|>",
        ],
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(RESOURCE / "bpe_tokenizer.json"))
    print(f"tokenizer retrained: {tok.get_vocab_size()} tokens, case preserved (W id={tok.token_to_id('W')})")
    return tok


def load_corpus() -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Tokenized sequences for the three training stages."""

    def enc(text: str) -> list[int]:
        return tok.encode(text).ids[:128]

    stories = [enc(s) for s in json.loads((RESOURCE / "tinystories_train.json").read_text())[:N_STORIES]]
    stories = [s for s in stories if len(s) > 8]

    code_rows = json.loads((RESOURCE / "code_instructions.json").read_text())
    code = [enc(f"```output{r['instruction']}```input{r['output']}") for r in code_rows]
    code = [s for s in code if len(s) > 8]

    tool_rows = json.loads((RESOURCE / "tool_calls.json").read_text())
    tool: list[list[int]] = []
    for row in tool_rows:
        for msg in row["messages"]:
            if msg.get("tool_calls"):
                body = "<|tool_call|>" + json.dumps([c["function"] for c in msg["tool_calls"]], ensure_ascii=False)
            elif msg["role"] == "tool":
                body = "<|tool_response|>" + (msg.get("content") or "")
            elif msg.get("content"):
                body = msg["content"]
            else:
                continue
            tool.append(enc(body))
    tool = [s for s in tool if len(s) > 8]
    print(f"corpus: {len(stories)} stories, {len(code)} code rows, {len(tool)} tool rows")
    return stories, code, tool


def make_batch(seqs: list[list[int]], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced next-token batch: x = s[:-1], y = s[1:] (the TRUE shift).

    At position t the model sees tokens s[0..t] and must predict s[t+1].
    (An off-by-one here trains a copy-echo model: loss ~0, generations
    repeat the last token forever — the exact bug this fixes.)
    """
    L = max(len(s) for s in seqs) - 1
    x = torch.zeros((len(seqs), L), dtype=torch.int64, device=device)
    y = torch.full((len(seqs), L), -100, dtype=torch.int64, device=device)
    for j, s in enumerate(seqs):
        n = len(s) - 1
        x[j, :n] = torch.tensor(s[:n], device=device)
        y[j, :n] = torch.tensor(s[1:], device=device)
    return x, y


def train(model: torch.nn.Module, seqs: list[list[int]], epochs: int, lr: float, tag: str) -> list[float]:
    """Next-token CE training; returns per-epoch mean losses."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses_all = []
    for ep in range(epochs):
        ls = []
        for i in range(0, len(seqs), BATCH):
            x, y = make_batch(seqs[i : i + BATCH], "cuda")
            loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            ls.append(float(loss.detach()))
        losses_all.append(float(np.mean(ls)))
        print(f"  {tag} epoch {ep}: loss={losses_all[-1]:.4f}")
    return losses_all


def save_checkpoint(model: TorchModel, name: str, tok: Tokenizer) -> None:
    out = RESOURCE / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "model.npz", **model.get_all_parameters())
    (out / "config.json").write_text(json.dumps(cfg.to_dict()))
    (out / "vocab.json").write_text(json.dumps([tok.decode([i]) for i in range(VOCAB)]))
    tok.save(str(out / "tokenizer.json"))
    print(f"saved resource/models/{name}/")


def sample(model: torch.nn.Module, prompt: str, n: int = 14) -> str:
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.int64, device="cuda")
    with torch.no_grad():
        for _ in range(n):
            logits = model(x)[0, -1, :]
            probs = torch.softmax(logits / 0.9, dim=-1)
            topv, topi = torch.topk(probs, 40)
            nxt = int(topi[0, torch.multinomial(topv[0], 1)] if topi.dim() == 2 else topi[torch.multinomial(topv, 1)])
            nxt = int(nxt.item()) if torch.is_tensor(nxt) else int(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device="cuda")], dim=1)
    return tok.decode(x[0].cpu().tolist())


if __name__ == "__main__":
    tok = train_tokenizer()
    stories, code, tool = load_corpus()

    cfg = TransformerConfig.from_dict(
        {"vocab_size": VOCAB, "embed_dim": 64, "n_layers": 3, "n_heads": 4, "context_length": 128, "seed": 42}
    )

    # ── stage 1: base (English fluency) ─────────────────────────────────
    base = TorchModel(cfg).cuda().train()
    print("stage 1: pre-train on stories")
    train(base, stories, epochs=4, lr=1e-3, tag="base")
    base.eval()
    save_checkpoint(base, "learning_base", tok)

    # ── stage 2: SFT on code instructions ───────────────────────────────
    sft = TorchModel(cfg).cuda().train()
    sft.load_from_numpy_dict(dict(np.load(RESOURCE / "models" / "learning_base" / "model.npz")))
    print("stage 2: SFT on code instructions")
    train(sft, code, epochs=4, lr=5e-4, tag="sft")
    sft.eval()
    save_checkpoint(sft, "learning_sft", tok)

    # ── stage 3: SFT on tool calls ───────────────────────────────────────
    tool_m = TorchModel(cfg).cuda().train()
    tool_m.load_from_numpy_dict(dict(np.load(RESOURCE / "models" / "learning_sft" / "model.npz")))
    print("stage 3: SFT on tool calls")
    train(tool_m, tool, epochs=3, lr=5e-4, tag="tool")
    tool_m.eval()
    save_checkpoint(tool_m, "learning_tool", tok)

    # ── generation probes (sampled, not argmax) ─────────────────────────
    for name, m in (("base", base), ("sft", sft), ("tool", tool_m)):
        print(f"[{name}] 'Once upon a time'      →", repr(sample(m, "Once upon a time")))
        print(f"[{name}] 'Write a Python function' →", repr(sample(m, "Write a Python function")))
