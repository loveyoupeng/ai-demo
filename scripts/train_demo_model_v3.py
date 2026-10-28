#!/usr/bin/env python3
# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Retrain the demo-model pipeline v3: pre → SFT-code → SFT-tool →
SFT-think → GRPO RL → drafter distillation (ADR 0004 + 0005).

Supersedes ``train_demo_model_v2.py`` (deleted). The six stages, in the
documented pipeline order (docs/math/training-pipeline.md):

1. BPE tokenizer, WITH the thinking-section specials ``<|think|>`` and
   ``<|answer|>`` (``shared.thinking`` owns the tag constants). Sources:
   tinystories + code_instructions + tool_calls — plus GSM8K rationales
   so the reasoning vocabulary (digits, "=") gets merge coverage.
2. Base model: pre-train on tinystories (English fluency).
3. SFT model: fine-tune base on code_instructions (instruction→code).
4. Tool model: fine-tune SFT on tool_calls (OpenAI-message JSON shape)
   → ``learning_tool`` / ``learning_moe_tool``.
5. Think model (reasoning distillation framing, ADR 0004): fine-tune on
   GSM8K gold rationales + synthetic two-step arithmetic — each row
   ``question <|think|> rationale <|answer|> number``. The "teacher
   traces" are gold rationales/templates — no real teacher at this
   scale, and the logs say so. → ``learning_think`` /
   ``learning_moe_think``.
6. RL model (GRPO, ADR 0004): reinforce the think model on the same
   prompts (reward: 1.0 correctness + 0.2 format, group-relative
   advantage, KL-to-frozen-think β=0.04) → ``learning_rl`` /
   ``learning_moe_rl`` — THE final checkpoints the learning page serves.
7. Drafter sidecars: re-distilled from the RL targets
   (``scripts/train_drafters`` — ADR 0003's per-target contract).

The MoE twin (``--variant moe``) runs the same stages with the FFN
swapped for the param-matched MoE of ADR 0005: E=8 routed + 1
always-on shared expert, expert_dim=32, top-k=1 — ~1.08× the dense
model's TOTAL parameters but only ~54% ACTIVE per token, the
total-vs-active trade the page renders.

Every checkpoint config.json carries ``"pipeline": 3`` — the staleness
marker ``scripts/learning.py`` checks (a missing/older marker retrains).

Everything stays tiny: 512-token vocab, 3-layer model, a few thousand
sequences — minutes per stage on the Jetson. Outputs under
``resource/models/`` each carry model.npz + config.json + vocab.json +
tokenizer.json (+ rl_log.json for the RL stages).

Usage:
    uv run python -m scripts.train_demo_model_v3                 # dense family
    uv run python -m scripts.train_demo_model_v3 --variant moe    # the MoE twin
    uv run python -m scripts.train_demo_model_v3 --stage think    # resume from tool
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from impl._torch.layers import TorchModel  # noqa: E402
from shared.config import TransformerConfig  # noqa: E402
from shared.thinking import ANSWER_TAG, THINK_TAG, format_thinking_row  # noqa: E402

RESOURCE = _project_root / "resource"
N_STORIES = 4000
BATCH = 32
VOCAB = 512
PIPELINE_MARKER = 3  # the config.json staleness marker (ADR 0004)
CTX = 128

# Stage order; ``--stage`` resumes AT a named stage (all later stages run).
STAGES = ["base", "sft", "tool", "think", "rl", "drafters"]


def _corpus_texts() -> list[str]:
    """The mixed corpus the BPE trainer sees (stories, code, tool, GSM8K)."""
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
    # GSM8K rationales (if downloaded): digit/equals coverage for the think stage
    gsm = RESOURCE / "gsm8k.json"
    if gsm.is_file():
        for row in json.loads(gsm.read_text()):
            texts.append(format_thinking_row(row["question"], row["reasoning"], row["answer"]))
    else:
        print("note: resource/gsm8k.json missing — run scripts.download_gsm8k for think-stage coverage")
    return texts


def train_tokenizer() -> Tokenizer:
    """BPE over the mixed corpus + the thinking-section specials."""
    if (RESOURCE / "bpe_tokenizer.json").exists():
        tok = Tokenizer.from_file(str(RESOURCE / "bpe_tokenizer.json"))
        # the v3 vocabulary must know the think tags
        if tok.token_to_id(THINK_TAG) is not None and tok.token_to_id(ANSWER_TAG) is not None:
            return tok
    texts = _corpus_texts()
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
            THINK_TAG,
            ANSWER_TAG,
        ],
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(RESOURCE / "bpe_tokenizer.json"))
    print(f"tokenizer retrained: {tok.get_vocab_size()} tokens (think tags included)")
    return tok


def load_corpus(tok: Tokenizer) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Tokenized sequences for stages 2-4 (stories, code, tool)."""

    def enc(text: str) -> list[int]:
        return tok.encode(text).ids[:CTX]

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


def load_think_corpus(tok: Tokenizer) -> list[list[int]]:
    """The thinking-stage sequences: GSM8K + synthetic two-step arithmetic.

    Each row is the full ``question <|think|> rationale <|answer|> number``
    stream (teacher-forced next-token training; every position graded —
    the pre-training loss shape, on distillation-shaped data).
    """
    seqs: list[list[int]] = []
    gsm = RESOURCE / "gsm8k.json"
    if gsm.is_file():
        for row in json.loads(gsm.read_text()):
            ids = tok.encode(format_thinking_row(row["question"], row["reasoning"], row["answer"])).ids[:CTX]
            if len(ids) > 8:
                seqs.append(ids)
    # synthetic two-step arithmetic: (a + b) * c with template rationales —
    # the guaranteed-learnable floor so RL always has a signal to reinforce
    rng = np.random.default_rng(42)
    for _ in range(2000):
        a, b, c = int(rng.integers(1, 6)), int(rng.integers(1, 6)), int(rng.integers(2, 5))
        q = f"{a} plus {b}, then times {c}."
        mid = a + b
        ans = mid * c
        rationale = f"First {a} plus {b} makes {mid}. Then {mid} times {c} makes {ans}."
        ids = tok.encode(format_thinking_row(q, rationale, str(ans))).ids[:CTX]
        if len(ids) > 8:
            seqs.append(ids)
    print(f"think corpus: {len(seqs)} rows (gsm8k + synthetic)")
    return seqs


def make_batch(seqs: list[list[int]], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced next-token batch: x = s[:-1], y = s[1:] (the TRUE shift)."""
    L = max(len(s) for s in seqs) - 1
    x = torch.zeros((len(seqs), L), dtype=torch.int64, device=device)
    y = torch.full((len(seqs), L), -100, dtype=torch.int64, device=device)
    for j, s in enumerate(seqs):
        n = len(s) - 1
        x[j, :n] = torch.tensor(s[:n], device=device)
        y[j, :n] = torch.tensor(s[1:], device=device)
    return x, y


def train(model: torch.nn.Module, seqs: list[list[int]], epochs: int, lr: float, tag: str, device: str) -> list[float]:
    """Next-token CE training; returns per-epoch mean losses."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses_all = []
    for ep in range(epochs):
        ls = []
        for i in range(0, len(seqs), BATCH):
            x, y = make_batch(seqs[i : i + BATCH], device)
            loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            ls.append(float(loss.detach()))
        losses_all.append(float(np.mean(ls)))
        print(f"  {tag} epoch {ep}: loss={losses_all[-1]:.4f}")
    return losses_all


def save_checkpoint(model: TorchModel, name: str, tok: Tokenizer, cfg: TransformerConfig) -> Path:
    out = RESOURCE / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "model.npz", **model.get_all_parameters())
    config = cfg.to_dict()
    config["pipeline"] = PIPELINE_MARKER  # the staleness marker (not a model knob)
    (out / "config.json").write_text(json.dumps(config))
    (out / "vocab.json").write_text(
        json.dumps([tok.decode([i], skip_special_tokens=False) for i in range(cfg.vocab_size)])
    )
    tok.save(str(out / "tokenizer.json"))
    print(f"saved resource/models/{name}/")
    return out


def sample(model: torch.nn.Module, tok: Tokenizer, prompt: str, n: int, device: str) -> str:
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.int64, device=device)
    with torch.no_grad():
        for _ in range(n):
            logits = model(x)[0, -1, :]
            probs = torch.softmax(logits / 0.9, dim=-1)
            topv, topi = torch.topk(probs, 40)
            nxt = int(topi[0, torch.multinomial(topv[0], 1)] if topi.dim() == 2 else topi[torch.multinomial(topv, 1)])
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return tok.decode(x[0].cpu().tolist(), skip_special_tokens=False)


def run_rl_stage(think_dir: Path, out_dir: Path, tok: Tokenizer, args: argparse.Namespace) -> None:
    """The GRPO stage via scripts.rl.py (the unified 4-backend entry —
    the pipeline runs it on the torch track, the production shape)."""
    dataset = RESOURCE / "gsm8k.json"
    # scripts.rl trains on the gsm8k prompts; the synthetic arithmetic floor
    # is generated HERE and merged into the dataset the stage consumes.
    if not dataset.is_file():
        print("error: resource/gsm8k.json missing — run scripts.download_gsm8k first", file=sys.stderr)
        raise SystemExit(1)
    merged = RESOURCE / "gsm8k_rl.json"
    rows = json.loads(dataset.read_text())
    rng = np.random.default_rng(43)
    for _ in range(1000):  # the synthetic floor joins the RL prompts
        a, b, c = int(rng.integers(1, 6)), int(rng.integers(1, 6)), int(rng.integers(2, 5))
        rows.append({"question": f"{a} plus {b}, then times {c}.", "reasoning": "", "answer": str((a + b) * c)})
    merged.write_text(json.dumps(rows))
    cmd = [
        sys.executable,
        "-m",
        "scripts.rl",
        "--backend",
        "torch",
        "--model",
        str(think_dir),
        "--save-dir",
        str(out_dir),
        "--dataset",
        str(merged),
        "--iterations",
        str(args.rl_iterations),
        # Per ADR 0004: batch = 8 prompts × G=8 = 64 rollouts/step, a few
        # hundred optimizer steps per iteration — minutes on the Jetson,
        # not the full 1500-prompt epoch (which would be a production
        # run's scale, not a demo's).
        "--prompts-per-iter",
        str(args.rl_prompts),
        "--group-size",
        "8",
        "--max-new-tokens",
        "80",
        "--kl-beta",
        "0.04",
        "--lr",
        "1e-4",
    ]
    print(f"stage 5 (rl): GRPO, {args.rl_prompts} prompts/iteration × {args.rl_iterations} iterations (torch track)")
    subprocess.run(cmd, check=True)
    # scripts.rl copies the tokenizer/vocab sidecars; add the pipeline marker
    config = json.loads((out_dir / "config.json").read_text())
    config["pipeline"] = PIPELINE_MARKER
    (out_dir / "config.json").write_text(json.dumps(config))


def run_drafter_stage(target_dir: Path) -> None:
    """Re-distill the drafter sidecars from the RL target (ADR 0003: every
    served checkpoint needs its OWN sidecars, distilled from it)."""
    subprocess.run([sys.executable, "-m", "scripts.train_drafters", "--model", str(target_dir)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the demo-model pipeline v3 (dense or MoE twin).")
    parser.add_argument(
        "--variant",
        choices=["dense", "moe"],
        default="dense",
        help="dense: plain SwiGLU FFN; moe: E=8 routed + 1 shared expert, expert_dim=32 (ADR 0005)",
    )
    parser.add_argument("--stage", choices=STAGES, default="base", help="Resume AT this stage (later stages run too)")
    parser.add_argument("--epochs", type=int, default=0, help="Override the per-SFT-stage epoch counts (quick smoke)")
    parser.add_argument("--rl-iterations", type=int, default=2, help="GRPO epochs over the prompt set")
    parser.add_argument(
        "--rl-prompts", type=int, default=150, help="Prompts per GRPO iteration (default 150 — minutes, not hours)"
    )
    args = parser.parse_args()

    moe = args.variant == "moe"
    names = {
        "base": "learning_moe_base" if moe else "learning_base",
        "sft": "learning_moe_sft" if moe else "learning_sft",
        "tool": "learning_moe_tool" if moe else "learning_tool",
        "think": "learning_moe_think" if moe else "learning_think",
        "rl": "learning_moe_rl" if moe else "learning_rl",
    }
    start = STAGES.index(args.stage)

    tok = train_tokenizer()
    stories, code, tool = load_corpus(tok)

    cfg = TransformerConfig.from_dict(
        {
            "vocab_size": VOCAB,
            "embed_dim": 64,
            "n_layers": 3,
            "n_heads": 4,
            "context_length": CTX,
            "seed": 42,
            # the ONE knob that differs between the variants — the
            # param-matched geometry of ADR 0005 (E=8+1 shared @ dim 32):
            "n_experts": 8 if moe else 1,
            "n_shared_experts": 1 if moe else 0,
            "expert_dim": 32 if moe else 0,  # 0 → 4*D = dense FFN 256
            "top_k": 1,
        }
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ep = args.epochs or 4

    # ── stage 1: base (English fluency) ─────────────────────────────────
    if start <= 0:
        base = TorchModel(cfg).to(device).train()
        print(f"stage 1 ({args.variant}): pre-train on stories")
        train(base, stories, epochs=ep, lr=1e-3, tag="base", device=device)
        base.eval()
        save_checkpoint(base, names["base"], tok, cfg)
    else:
        base = None

    # ── stage 2: SFT on code instructions ──────────────────────────────
    if start <= 1:
        sft = TorchModel(cfg, dict(np.load(RESOURCE / "models" / names["base"] / "model.npz"))).to(device).train()
        print(f"stage 2 ({args.variant}): SFT on code instructions")
        train(sft, code, epochs=ep, lr=5e-4, tag="sft", device=device)
        sft.eval()
        save_checkpoint(sft, names["sft"], tok, cfg)
    else:
        sft = None

    # ── stage 3: SFT on tool calls ──────────────────────────────────────
    if start <= 2:
        tool_m = TorchModel(cfg, dict(np.load(RESOURCE / "models" / names["sft"] / "model.npz"))).to(device).train()
        print(f"stage 3 ({args.variant}): SFT on tool calls")
        train(tool_m, tool, epochs=args.epochs or 3, lr=5e-4, tag="tool", device=device)
        tool_m.eval()
        save_checkpoint(tool_m, names["tool"], tok, cfg)
    else:
        tool_m = None

    # ── stage 4: thinking SFT (reasoning distillation, ADR 0004) ────────
    if start <= 3:
        think_seqs = load_think_corpus(tok)
        think_m = TorchModel(cfg, dict(np.load(RESOURCE / "models" / names["tool"] / "model.npz"))).to(device).train()
        print(f"stage 4 ({args.variant}): thinking SFT (reasoning distillation — gold rationales + templates)")
        train(think_m, think_seqs, epochs=args.epochs or 3, lr=5e-4, tag="think", device=device)
        think_m.eval()
        save_checkpoint(think_m, names["think"], tok, cfg)
        # the distillation teaching artifact: print trace pairs
        for row in think_seqs[:2]:
            print("  [distill trace]", repr(tok.decode(row, skip_special_tokens=False)))
    else:
        think_m = None

    # ── stage 5: GRPO RL (ADR 0004) ─────────────────────────────────────
    if start <= 4:
        run_rl_stage(RESOURCE / "models" / names["think"], RESOURCE / "models" / names["rl"], tok, args)

    # ── stage 6: drafter sidecars from the RL target ────────────────────
    if start <= 5:
        run_drafter_stage(RESOURCE / "models" / names["rl"])

    # ── generation probes ───────────────────────────────────────────────
    for name, m in (("base", base), ("sft", sft), ("tool", tool_m), ("think", think_m)):
        if m is None:
            continue
        print(f"[{name}] 'Once upon a time'      →", repr(sample(m, tok, "Once upon a time", 14, device)))
        print(f"[{name}] 'Write a Python function' →", repr(sample(m, tok, "Write a Python function", 14, device)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
