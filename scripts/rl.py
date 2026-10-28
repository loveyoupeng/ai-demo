#!/usr/bin/env python3
# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""RL pipeline entry point: GRPO on a thinking-SFT checkpoint (ADR 0004).

The RL stage of the demo pipeline, as a single user-facing command
covering all four backends (the SFT convention: NumPy = the teaching
baseline, torch/triton/cuda = the production shapes):

    uv run python -m scripts.rl --backend numpy --model resource/models/learning_think \
        --save-dir resource/models/learning_rl
    uv run python -m scripts.rl --backend torch  --model resource/models/learning_think \
        --save-dir resource/models/learning_rl

What it does per iteration:

  1. Roll a GROUP of G completions per prompt (temperature/top-k sampling
     through the track's own KV-step path).
  2. Score each completion with the two-component reward
     (``shared.thinking.reward``: 1.0 correctness + 0.2 format).
  3. Advantage = group-normalized reward (no critic — GRPO's point).
  4. One optimizer step reweighting the sampled tokens' CE gradient by
     the clipped advantage, with the β-weighted k3 KL penalty against
     the frozen reference (the thinking-SFT model the run started from).

Data: ``resource/gsm8k.json`` (``scripts.download_gsm8k.py``) — the
prompts are the questions, the gold answers drive the reward. The model
is expected to have been thinking-SFT-trained (``<|think|>`/`<|answer|>`'
sections already in its behavior — RL sharpens, SFT seeds).

Outputs the reinforced checkpoint to ``--save-dir`` (registry-valid;
loadable in all four tracks) plus ``rl_log.json`` — the per-iteration
reward/format/correctness curves the docs and the learning page cite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from shared.config import TransformerConfig

RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resource"


def _load_items(dataset: Path, tokenizer: Tokenizer, max_len: int) -> list[tuple[np.ndarray, str]]:
    """(prompt_ids, gold_answer) pairs from the gsm8k JSON, ctx-filtered.

    The prompt is the QUESTION alone (no tags) — the policy must open
    ``<|think|>`` itself; the gold answer is the parseable final number
    the reward compares against.
    """
    rows = json.loads(dataset.read_text())
    items: list[tuple[np.ndarray, str]] = []
    for row in rows:
        ids = np.asarray(tokenizer.encode(row["question"]).ids[:max_len], dtype=np.int64)
        if ids.size >= 4:
            items.append((ids, str(row["answer"])))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["numpy", "torch", "triton", "cuda"], required=True)
    ap.add_argument(
        "--model", type=Path, required=True, help="Thinking-SFT checkpoint dir (the RL start + KL reference)"
    )
    ap.add_argument(
        "--save-dir", type=Path, default=None, help="Where to write the RL checkpoint (default: <model>_rl)"
    )
    ap.add_argument("--dataset", type=Path, default=RESOURCE_DIR / "gsm8k.json")
    ap.add_argument("--iterations", type=int, default=2, help="Epochs over the prompt set")
    ap.add_argument("--prompts-per-iter", type=int, default=0, help="Cap prompts per iteration (0 = all)")
    ap.add_argument("--group-size", type=int, default=8, help="Completions per prompt (G)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--kl-beta", type=float, default=0.04)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=128)
    args = ap.parse_args()

    # The checkpoint carries its own config; the tokenizer must match.
    cfg = TransformerConfig.from_dict(json.loads((args.model / "config.json").read_text()))
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    items = _load_items(args.dataset, tokenizer, args.max_len)
    if not items:
        print(f"error: no usable prompts in {args.dataset} (run scripts.download_gsm8k first)", flush=True)
        return 1
    if args.prompts_per_iter:
        items = items[: args.prompts_per_iter]

    def decode(ids: np.ndarray) -> str:
        # keep the special tokens: the think/answer tags ARE the reward
        # signal's delimiters — dropping them (decode's default) would
        # make every completion unparseable and the reward always 0.
        return tokenizer.decode([int(t) for t in ids], skip_special_tokens=False)

    save_dir = args.save_dir or args.model.with_name(args.model.name + "_rl")
    rng = np.random.default_rng(args.seed)

    if args.backend == "numpy":
        out, curve = _run_numpy(args, cfg, items, decode, rng)
    else:
        out, curve = _run_torch_family(args, cfg, items, decode)

    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_dir / "model.npz", **out)
    (save_dir / "config.json").write_text(json.dumps(cfg.to_dict()))
    import shutil

    for side in ("tokenizer.json", "vocab.json"):
        if (args.model / side).is_file():
            shutil.copy(args.model / side, save_dir / side)
    (save_dir / "rl_log.json").write_text(json.dumps(curve, indent=1))
    print(f"saved {save_dir} (reward curve: {len(curve)} iterations)")
    return 0


def _print(it: int, total: int, stats: dict[str, float]) -> None:
    print(
        f"iter {it + 1}/{total}: reward={stats['reward']:.3f} "
        f"format={stats['format']:.2f} correct={stats['correct']:.2f} kl={stats['kl']:.4f}",
        flush=True,
    )


def _run_numpy(args, cfg, items, decode, rng) -> tuple[dict, list[dict[str, float]]]:
    """The NumPy track: ``grpo_iteration_scored`` per epoch (the teaching
    path — every formula visible in impl/_np/rl.py)."""
    from impl._np.model import NumPyModel
    from impl._np.optimizer import AdamW
    from impl._np.rl import GRPOConfig, grpo_iteration_scored

    params = {k: v.copy() for k, v in np.load(args.model / "model.npz").items()}
    model = NumPyModel(cfg, params)
    reference = NumPyModel(cfg, {k: v.copy() for k, v in params.items()})
    grpo = GRPOConfig(
        group_size=args.group_size,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
    )
    opt = AdamW(lr=args.lr)
    curve: list[dict[str, float]] = []
    for it in range(args.iterations):
        stats = grpo_iteration_scored(model, reference, items, decode, grpo, opt, rng, log_every=10)
        curve.append(stats)
        _print(it, args.iterations, stats)
    return {k: v for k, v in model.get_all_parameters().items()}, curve


def _run_torch_family(args, cfg, items, decode) -> tuple[dict, list[dict[str, float]]]:
    """The torch/triton/cuda tracks: the framework loop (one trainer per
    track, rollouts through ``shared.generator.TextGenerator``'s KV-step
    decode — except CUDA, which runs its own decode loop below)."""
    import torch

    params = {k: v.copy() for k, v in np.load(args.model / "model.npz").items()}
    if args.backend == "cuda":
        from impl._cuda.model import CUDAModel

        model = CUDAModel(cfg, params)
        reference = CUDAModel(cfg, {k: v.copy() for k, v in params.items()})
        return _run_cuda_loop(args, model, reference, items, decode)
    if args.backend == "triton":
        from impl._triton.model import TritonModel
        from impl._triton.rl import TritonGRPOTrainer

        Model = TritonModel
        Trainer: type[TritonGRPOTrainer] | type = TritonGRPOTrainer
    else:
        from impl._torch.layers import TorchModel
        from impl._torch.rl import TorchGRPOTrainer

        Model = TorchModel  # type: ignore[assignment]
        Trainer = TorchGRPOTrainer

    device = "cuda" if torch.cuda.is_available() or args.backend == "triton" else "cpu"
    model = Model(cfg, params)
    reference = Model(cfg, {k: v.copy() for k, v in params.items()})
    trainer = Trainer(
        model,
        reference,
        device=device,
        lr=args.lr,
        group_size=args.group_size,
        temperature=args.temperature,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
    )
    generator = trainer._gen
    from impl._np.rl import group_advantage
    from shared.thinking import reward as score_reward

    torch.manual_seed(args.seed)
    curve: list[dict[str, float]] = []
    for it in range(args.iterations):
        ep_reward = ep_format = ep_correct = ep_kl = 0.0
        for prompt_ids, gold in items:
            prompt = torch.from_numpy(prompt_ids.reshape(1, -1)).to(trainer.device)
            with torch.no_grad():
                streams = trainer.rollout(prompt, generator)  # (G, S+T)
            S = prompt_ids.shape[0]
            rmask = torch.zeros_like(streams)
            rmask[:, S:] = 1
            completions = streams[:, S:].cpu().numpy()
            rewards = np.zeros(args.group_size)
            for g in range(args.group_size):
                r, det = score_reward(decode(completions[g]), gold)
                rewards[g] = r
                ep_format += 1.0 if det["format_ok"] else 0.0
            adv = group_advantage(rewards, 1e-4)
            st = trainer.train_step(streams, rmask, torch.from_numpy(adv).to(streams.device))
            ep_reward += float(rewards.mean())
            ep_correct += float(np.mean(rewards >= 1.0))
            ep_kl += st["kl"]
        n = len(items)
        stats = {
            "reward": ep_reward / n,
            "format": ep_format / n / args.group_size,
            "correct": ep_correct / n,
            "kl": ep_kl / n,
        }
        curve.append(stats)
        _print(it, args.iterations, stats)
    out = {k: v for k, v in model.get_all_parameters().items()}
    return out, curve


def _run_cuda_loop(args, model, reference, items, decode) -> tuple[dict, list[dict[str, float]]]:
    """The CUDA track's own loop: CUDAModel is not an nn.Module, so the
    torch-family trainer (and its TextGenerator) does not bind; the
    rollout decodes step-by-step over forward_prefill/forward_step —
    the same KV-step path, driven explicitly (the bare-metal way)."""
    import torch

    from impl._cuda.rl import CudaGRPOTrainer
    from impl._np.rl import group_advantage
    from shared.thinking import reward as score_reward

    trainer = CudaGRPOTrainer(model, reference, lr=args.lr, clip_eps=args.clip_eps, kl_beta=args.kl_beta)
    device = model.device
    torch.manual_seed(args.seed)
    curve: list[dict[str, float]] = []
    for it in range(args.iterations):
        ep_reward = ep_format = ep_correct = ep_kl = 0.0
        for prompt_ids, gold in items:
            prompt = torch.from_numpy(prompt_ids.reshape(1, -1))
            streams = []
            with torch.no_grad():
                for _ in range(args.group_size):
                    seq = prompt.to(device)
                    cache = model.make_cache(seq.shape[0])
                    logits, _ = model.forward_prefill(seq, cache)
                    nxt = logits[:, -1, :]
                    for t in range(args.max_new_tokens):
                        scaled = nxt / max(args.temperature, 1e-8)
                        if 0 < args.top_k < scaled.shape[-1]:
                            kth = torch.sort(scaled, dim=-1).values[:, -args.top_k]
                            scaled = torch.where(scaled < kth[:, None], scaled.new_full((), -float("inf")), scaled)
                        probs = torch.softmax(scaled, dim=-1)
                        nxt_tok = torch.multinomial(probs, 1)  # (1, 1)
                        seq = torch.cat([seq, nxt_tok], dim=1)
                        if t < args.max_new_tokens - 1:
                            nxt = model.forward_step(nxt_tok, seq.shape[1] - 1, cache)[:, 0, :]
                    streams.append(seq)
            streams = torch.cat(streams, dim=0)  # (G, S+T)
            S = prompt_ids.shape[0]
            rmask = torch.zeros_like(streams)
            rmask[:, S:] = 1
            completions = streams[:, S:].cpu().numpy()
            rewards = np.zeros(args.group_size)
            for g in range(args.group_size):
                r, det = score_reward(decode(completions[g]), gold)
                rewards[g] = r
                ep_format += 1.0 if det["format_ok"] else 0.0
            adv = group_advantage(rewards, 1e-4)
            st = trainer.train_step(streams, rmask, torch.from_numpy(adv))
            ep_reward += float(rewards.mean())
            ep_correct += float(np.mean(rewards >= 1.0))
            ep_kl += st["kl"]
        n = len(items)
        stats = {
            "reward": ep_reward / n,
            "format": ep_format / n / args.group_size,
            "correct": ep_correct / n,
            "kl": ep_kl / n,
        }
        curve.append(stats)
        _print(it, args.iterations, stats)
    out = {k: v for k, v in model.get_all_parameters().items()}
    return out, curve


if __name__ == "__main__":
    raise SystemExit(main())
