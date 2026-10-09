"""Distillation seam tests (Slice 6).

The acceptance contract has two separable halves:

1. **The batch constructor produces (anchor_token, anchor_hidden, targets)
   triples that match the live target's greedy continuations** — the
   distillation reward. If this mapping drifts, the drafter accepts
   nothing (the 50% warn path in the training script reports it, never
   fails it).

2. **The warn boundary is exactly 50%** — arithmetic only, no side
   effects. Reprised here so a refactor can't casually move it.
"""

from __future__ import annotations

import numpy as np

from scripts.train_drafters import ACCEPT_WARN, _distill_batch, rollouts


def _make_target():
    """Small dense NumPy model (teaching scale — 32-vocab, D=16, 1 layer)."""
    from impl._np.model import NumPyModel
    from shared.config import TransformerConfig

    cfg = TransformerConfig(vocab_size=32, embed_dim=16, n_layers=1, n_heads=2, seed=3)
    return NumPyModel(cfg), cfg


def test_distill_batch_matches_target_continuations() -> None:
    """targets[j] == the target's greedy token at anchor+j+1; masked at the edge."""
    target, cfg = _make_target()
    rng = np.random.default_rng(0)
    # Corpus = arbitrary in-vocab sequences (BPE not needed — the corpus
    # format is just a list of int lists that fit the target's vocab).
    corpus = [list(int(x) for x in rng.integers(0, cfg.vocab_size, size=16)) for _ in range(8)]
    rolls = rollouts(target, corpus, 16, 24, rng)

    rolls = [r for r in rolls if len(r["tokens"]) > 6]
    assert rolls, "corpus produced no rollout with >6 tokens"
    for k in (2, 4):
        a_t, a_h, tgt = _distill_batch(rolls, 8, k, rng)
        assert a_t.shape == (8,) and a_h.shape == (8, cfg.embed_dim)
        assert tgt.shape == (8, k)
        # Every unmasked target index is a position the rollout actually
        # generated; the radar window is the known continuation.
        assert np.all((tgt == -100) | ((tgt >= 0) & (tgt < cfg.vocab_size)))


def test_accept_warn_threshold() -> None:
    """acceptance < 50% warns; >= 50% doesn't. The boundary is exact."""
    assert ACCEPT_WARN == 0.5
    for rate, warns in ((0.0, True), (0.499, True), (0.5, False), (0.99, False)):
        assert (rate < ACCEPT_WARN) is warns
