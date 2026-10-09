#!/usr/bin/env python3
"""Distill the MTP and DSpark drafters from the frozen learning_tool target.

Speculative decoding needs a drafter whose proposals the target actually
accepts. Training a drafter on the raw corpus alone misses — the target's
own greedy continuations are the distribution to match. So this is
DISTILLATION (ADR 0003, stage 4 of the demo pipeline):

  1. Roll the FROZEN target greedily over prompts sampled from the mixed
     corpus, capturing at each position the target's final-norm hidden
     state and its greedy next token.
  2. Train each drafter (MTP k=4, DSpark k=8) to reproduce, from the
     anchor's hidden state + token, the target's greedy continuation at
     every drafted offset — per-position cross-entropy against the
     target's own argmax choices.
  3. Save each drafter as a sidecar checkpoint
     (``<model_dir>/draft_mtp/``, ``<model_dir>/draft_dspark/`` —
     ``draft.json`` + ``draft.npz``, see shared/draft.py), loadable by
     every track (round-trip guarantee).
  4. Report the drafter's top-1 agreement with the target on held-out
     rollouts (the acceptance-rate estimate). Below 50% prints a WARNING —
     a teaching artifact, never a hard failure (spec decision).

Rollouts run on the NumPy track (the oracle — one math path); drafter
training runs on the torch track (autograd, GPU); acceptance validation
loads the SAVED sidecars back into the NumPy engine — proving the
torch-trained → NumPy-verified round trip on every run.

Usage:
    uv run python -m scripts.train_drafters                          # default target learning_tool
    uv run python -m scripts.train_drafters --model resource/models/learning_sft
    uv run python -m scripts.train_drafters --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from impl._np.drafters import DSparkDrafter as NpDSpark  # noqa: E402
from impl._np.drafters import MTPDrafter as NpMTP  # noqa: E402
from impl._np.model import NumPyModel  # noqa: E402
from impl._np.spec import SpeculativeGenerator as NpSpecGen  # noqa: E402
from shared.checkpoint import load_checkpoint  # noqa: E402
from shared.config import TransformerConfig  # noqa: E402
from shared.draft import DSHARK, MTP, DrafterMeta, load_drafter, save_drafter, sidecar_dir  # noqa: E402
from shared.utils.logger_setup import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

RESOURCE = _project_root / "resource"
DEFAULT_MODEL = "resource/models/learning_tool"
MTP_K = 4
DSHARK_K = 8
ACCEPT_WARN = 0.5  # warn below, never fail (spec decision)


def load_frozen_target(model_dir: str) -> tuple[NumPyModel, TransformerConfig]:
    """Load the frozen target into the NumPy track (the distillation oracle
    and acceptance validator — one math path, no torch/numpy drift)."""
    params, cfg = load_checkpoint(model_dir)
    if cfg is None:
        raise ValueError(f"checkpoint {model_dir} is missing config.json")
    model = NumPyModel(cfg)
    model.load_from_numpy_dict({k: v.copy() for k, v in params.items()})
    return model, cfg


def rollouts(
    model: NumPyModel,
    corpus: list[list[int]],
    n_rollouts: int,
    max_len: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Greedy rollouts of the frozen target from corpus prompts.

    Returns one record per rollout: {"tokens": (S,) the full generated
    sequence, "hiddens": (S, D) the final-norm hidden states}. The hidden
    at position i is the drafter's conditioning for the continuation after
    token i (the anchor contract: anchor_token = tokens[i]).
    """
    out: list[dict] = []
    prompts = rng.choice(len(corpus), size=n_rollouts, replace=True)
    for idx in prompts:
        seq = list(corpus[int(idx)][:16])  # 16-token prompt from the corpus
        if len(seq) < 4:
            continue
        tokens: list[int] = list(seq)
        hiddens: list[np.ndarray] = []
        cache = model.make_cache(1)
        # Prefill via forward_chunk: logits + the final-norm hidden stream
        # in one pass (chunk parity guarantees it equals the plain forward).
        logits, x_final = model.forward_chunk(np.array([seq], dtype=np.int32), 0, cache)
        hiddens.extend(x_final[0])  # S0 × (D,)
        pos = len(seq)
        while pos < max_len:
            nxt = int(np.argmax(logits[0, -1]))
            tokens.append(nxt)
            logits, x_final = model.forward_chunk(np.array([[nxt]], dtype=np.int32), pos, cache)
            hiddens.extend(x_final[0])
            pos += 1
        out.append({"tokens": np.array(tokens, dtype=np.int64), "hiddens": np.stack(hiddens)})
    return out


def _distill_batch(
    rolls: list[dict], batch: int, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a distillation batch: (anchor_tokens (N,), anchor_hiddens
    (N, D), targets (N, k)).

    targets[j] = the target's greedy token at anchor+j+1 (-100 beyond the
    sequence end). The drafter's draft position j is trained to match the
    target continuation at offset j+1 — exactly the acceptance test it
    must pass at inference.
    """
    picks = [rolls[i] for i in rng.choice(len(rolls), size=batch, replace=True)]
    anchor_t = np.empty(batch, dtype=np.int64)
    anchor_h = []
    targets = np.full((batch, k), -100, dtype=np.int64)
    for n, roll in enumerate(picks):
        toks = roll["tokens"]
        i = int(rng.integers(0, len(toks) - 1))  # anchor position
        anchor_t[n] = toks[i]
        anchor_h.append(roll["hiddens"][i])
        for j in range(min(k, len(toks) - 1 - i)):
            targets[n, j] = toks[i + 1 + j]
    return anchor_t, np.stack(anchor_h).astype(np.float32), targets


def _train_family_torch(
    torch_cls,
    meta: DrafterMeta,
    emb_np: np.ndarray,
    lm_np: np.ndarray,
    rolls: list[dict],
    steps: int,
    lr: float,
    device: str,
    seed: int,
    batch: int = 64,
) -> dict[str, np.ndarray]:
    """Train one torch drafter by distillation; return its sidecar params.

    From the anchor (token id + the target's final-norm hidden), draft
    position j should equal the target's token at anchor+j+1. Loss =
    per-position CE over the drafter's per-position distributions, masked
    at -100. The target is frozen — conditioning inputs come from the
    rollout records; the drafter trains alone.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    emb = torch.tensor(emb_np, dtype=torch.float32, device=device)
    lm = torch.tensor(lm_np, dtype=torch.float32, device=device)
    drafter = torch_cls(meta, emb, lm, seed=seed).to(device)
    drafter.train()
    opt = torch.optim.AdamW(drafter.parameters(), lr=lr)
    k_eff = meta.block_size
    losses: list[float] = []
    for step in range(steps):
        a_t, a_h, tgt = _distill_batch(rolls, batch, k_eff, rng)
        a_t_t = torch.tensor(a_t, dtype=torch.long, device=device)
        a_h_t = torch.tensor(a_h, dtype=torch.float32, device=device)
        tgt_t = torch.tensor(tgt, dtype=torch.long, device=device)
        drafter.reset()  # MTP's KV cache would otherwise retain step N-1's
        # autograd graph into step N (backward-through-freed-graph error).
        _draft_tokens, draft_probs = drafter.draft(a_t_t, a_h_t, k_eff)
        probs = torch.stack(draft_probs, dim=1)  # (N, k, V)
        logp = torch.log(probs.clamp_min(1e-12))
        loss = torch.nn.functional.cross_entropy(logp.reshape(-1, logp.shape[-1]), tgt_t.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if (step + 1) % max(steps // 4, 1) == 0:
            print(f"  step {step + 1}/{steps} loss {losses[-1]:.4f}")
    drafter.eval()
    return {key: np.asarray(val) for key, val in drafter.get_all_parameters().items()}


def _load_corpus_for(cfg: TransformerConfig) -> list[list[int]]:
    """The mixed corpus as token-id lists (the pipeline's BPE tokenizer)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(RESOURCE / "bpe_tokenizer.json"))
    seqs: list[list[int]] = []
    stories = json.loads((RESOURCE / "tinystories_train.json").read_text())[:400]
    seqs += [tok.encode(s).ids[:128] for s in stories]
    code = json.loads((RESOURCE / "code_instructions.json").read_text())
    seqs += [tok.encode(f"{r['instruction']}\n{r['output']}").ids[:128] for r in code]
    tool = json.loads((RESOURCE / "tool_calls.json").read_text())
    for row in tool:
        for msg in row["messages"]:
            if msg.get("content"):
                seqs.append(tok.encode(msg["content"]).ids[:128])
    seqs = [s for s in seqs if len(s) > 8]
    if not seqs:
        raise FileNotFoundError("no corpus under resource/ (run scripts.train_demo_model_v2 first)")
    return seqs


def _report_acceptance(
    family: str,
    model_dir: str,
    target_np: NumPyModel,
    np_cls: type,
    k: int,
    val_rolls: list[dict],
) -> float:
    """Load the SAVED sidecar into the NumPy engine and measure acceptance
    over held-out rollouts (also proves the round trip: torch-trained,
    NumPy-loaded, NumPy-verified)."""
    meta, sc_params = load_drafter(sidecar_dir(model_dir, family))
    drafter = np_cls(meta, target_np.embedding.weight, target_np.lm_head_weight)
    drafter.load_from_numpy_dict(sc_params)
    gen = NpSpecGen(target_np, drafter)
    accepted = drafted = 0
    for roll in val_rolls[:16]:
        toks = roll["tokens"]
        prompt = toks[:16].reshape(1, -1).astype(np.int32)
        _seq, stats = gen.generate_greedy(prompt, min(32, max(len(toks) - 16, 1)), k=k)
        accepted += stats["n_accepted"]
        drafted += stats["n_draft_tokens"]
    rate = accepted / max(drafted, 1)
    line = f"{family}: acceptance {rate:.1%} ({accepted}/{drafted})"
    if rate < ACCEPT_WARN:
        print(f"WARNING: {line} — below {ACCEPT_WARN:.0%}; the demo still runs (spec: warn, never fail)")
    else:
        print(line)
    return rate


def main() -> int:
    """Entry point — distill both drafters, save sidecars, report acceptance."""
    setup_logging()
    parser = argparse.ArgumentParser(
        prog="train_drafters.py",
        description="Distill MTP + DSpark drafters from the frozen learning-mode target.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Target checkpoint directory")
    parser.add_argument("--n-rollouts", type=int, default=512, help="Greedy rollouts for distillation")
    parser.add_argument("--max-len", type=int, default=64, help="Max rollout length")
    parser.add_argument("--steps", type=int, default=400, help="Optimizer steps per drafter")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument("--ff-dim", type=int, default=128, help="Drafter SwiGLU width")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true", help="Retrain even if sidecars exist")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    mtp_dir = sidecar_dir(args.model, MTP)
    dspark_dir = sidecar_dir(args.model, DSHARK)
    if not args.force and (mtp_dir / "draft.npz").is_file() and (dspark_dir / "draft.npz").is_file():
        print(f"drafters already present: {mtp_dir}, {dspark_dir} (use --force to retrain)")
        return 0

    # 1. Frozen target + rollouts (NumPy track = the oracle).
    print(f"loading frozen target: {args.model}")
    target_np, cfg = load_frozen_target(args.model)
    corpus = _load_corpus_for(cfg)
    print(f"rolling out {args.n_rollouts} greedy continuations (max_len={args.max_len})...")
    rolls = rollouts(target_np, corpus, args.n_rollouts, args.max_len, rng)
    if not rolls:
        print("error: no usable rollouts (corpus too short?)", file=sys.stderr)
        return 1

    # 2. Train each family on the torch track (autograd), save sidecars.
    from impl._torch.drafters import TorchDSparkDrafter, TorchMTPDrafter, make_drafter_meta

    emb_np, lm_np = target_np.embedding.weight, target_np.lm_head_weight
    for family, torch_cls, k in ((MTP, TorchMTPDrafter, MTP_K), (DSHARK, TorchDSparkDrafter, DSHARK_K)):
        meta = make_drafter_meta(family, cfg, block_size=k, ff_dim=args.ff_dim)
        print(f"training {family} drafter (k={k}, steps={args.steps}) on {args.device}...")
        params = _train_family_torch(torch_cls, meta, emb_np, lm_np, rolls, args.steps, args.lr, args.device, args.seed)
        save_drafter(sidecar_dir(args.model, family), meta, params)
        print(f"saved {sidecar_dir(args.model, family)}/")

    # 3. Acceptance on held-out rollouts, NumPy engine over the saved sidecars.
    val_rolls = rollouts(target_np, corpus, max(16, args.n_rollouts // 8), args.max_len, rng)
    for family, np_cls, k in ((MTP, NpMTP, MTP_K), (DSHARK, NpDSpark, DSHARK_K)):
        _report_acceptance(family, args.model, target_np, np_cls, k, val_rolls)

    print("done: draft_mtp/ + draft_dspark/ sidecars written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
