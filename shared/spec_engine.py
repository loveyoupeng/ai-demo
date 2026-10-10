"""Torch-family speculative decoding engine — the production engine (ADR 0003).

The counterpart of ``shared.generator.TextGenerator`` for the speculative
feature: ONE implementation in ``shared/`` (the cross-track backbone)
consumed by the torch, triton, and cuda tracks, mirroring the NumPy
reference ``impl/_np/spec.py`` — same records, same stats keys, same loop.

**The loop, in plain words.** Ordinary generation repeats: model forward →
one word. This engine repeats: the drafter proposes k candidate words, the
target verifies all k in ONE chunk pass, the leading agreeing prefix
commits, the first mismatch is replaced by the target's own word, and the
rejected tail's KV rows are trimmed. Accepted words + the correction all
join the output — one target pass confirms up to k+1 words.

**Target-forward accounting (honest, same as the NumPy reference):** each
round is TWO target passes — the verify chunk over the draft block, plus
the commit pass that appends the correction/bonus token and computes the
next anchor's hidden (the drafter conditions on the target's hidden AT the
anchor; the correction token's hidden does not exist until it is
forwarded — the same shape production MTP serving uses)::

    n_target_forwards = 1 (prefill) + 2 × n_rounds

**Greedy verification only** (what production runs at temperature 0 — the
torch family runs what production runs; the rejection-sampling theorem is
NumPy-only, the reference's teaching centerpiece):

    accept d_j while argmax(target logits at position t+j-1) == d_j

The target's token at the first mismatch (or the bonus token after a
fully accepted block) always wins, so the output is **token-identical to
plain greedy decoding** — the lossless contract pinned by the parity
tests and the ``spec_mtp`` / ``spec_dspark`` verify_equivalence scenarios.

**Repetition guard**: every walked position's prediction is the guarded
argmax — the SAME rule the plain greedy generator applies
(``impl/_np.inference._apply_rep_guard`` (REP_PENALTY on tokens already
emitted + a hard block on the immediately-previous token)), so the
lossless contract holds against the shipped plain decoder.

**Confidence-scheduled verification** (DSpark): per-prefix-length survival
statistics tracked online; each round's draft length capped at the longest
prefix whose survival estimate meets 0.8 (see the NumPy reference for the
plain-words version).

Batching: the engine asserts batch 1 (per-row accepted lengths differ in
batched speculation — the reason DSpark schedules per request).

Logging (logger ``shared.spec_engine``): per-round stats + totals.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

import torch

from shared.constants import REP_PENALTY
from shared.draft import DSHARK

logger = logging.getLogger(__name__)

SCHEDULE_TARGET = 0.8
"""DSpark prefix-survival target (mirrors impl/_np/spec.py)."""


class _ChunkModel(Protocol):
    """The chunk contract the engine consumes (all torch-family tracks)."""

    def make_cache(self, batch_size: int) -> list[dict[str, torch.Tensor]]: ...

    def forward_chunk(
        self, input_ids: torch.Tensor, position: int, cache: list[dict[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def _apply_rep_guard(step_logits: torch.Tensor, emitted: list[int]) -> torch.Tensor:
    """Repetition guard on one row: penalty on already-emitted tokens + a
    hard block on the immediately previous token.

    Same rule as ``shared.generator.TextGenerator._decode`` /
    ``impl._np.inference._apply_rep_guard`` (server ``_sample``, record
    ``_pick``): v / REP_PENALTY when v > 0 else v * REP_PENALTY, then the
    previous token set to -inf — so all tracks decode equivalently.

    step_logits: (V,) one row (the engine is single-sequence); returns a
    fresh guarded (V,) tensor (the input is not mutated).
    """
    guarded = step_logits.clone()
    for tid in set(emitted):
        if 0 <= tid < guarded.shape[-1]:
            v = guarded[tid]
            guarded[tid] = v / REP_PENALTY if v > 0 else v * REP_PENALTY
    if emitted:
        guarded[emitted[-1]] = -float("inf")  # never immediately repeat
    return guarded


def _top_slice(p: torch.Tensor, n_top: int = 8) -> list[list[float]]:
    """A distribution's top-n as [[token_id, prob], ...] (the page's bars)."""
    pv = p.detach().reshape(-1).float().cpu()
    idx = torch.argsort(pv, descending=True)[:n_top]
    return [[int(i), round(float(pv[i]), 6)] for i in idx]


class _SurvivalSchedule:
    """Online per-prefix survival estimate (the DSpark schedule state).

    Plain words: a little bookkeeper that watches past rounds and counts,
    for each prefix length, how often the verifier kept at least that many
    words; counts[length] / rounds is the observed survival fraction. The
    engine caps the round's draft length at the longest length whose
    fraction still clears the 0.8 bar (at least 1 — always draft).
    """

    def __init__(self, k_max: int) -> None:
        self.counts = [0] * (k_max + 1)
        self.rounds = 0

    def update(self, accepted: int) -> None:
        self.rounds += 1
        for length in range(1, accepted + 1):
            self.counts[length] += 1

    def verify_len(self) -> int:
        """Longest prefix length whose survival estimate meets the target."""
        if self.rounds == 0:
            return len(self.counts) - 1  # no data yet → full block
        best = 1
        for length in range(1, len(self.counts)):
            if self.counts[length] / self.rounds >= SCHEDULE_TARGET:
                best = length
        return best

    def state(self) -> dict:
        """JSON-friendly snapshot for the record / UI."""
        p = [c / self.rounds for c in self.counts] if self.rounds else [0.0] * len(self.counts)
        return {
            "rounds": self.rounds,
            "survival": [round(v, 4) for v in p],
            "verify_len": self.verify_len(),
            "target": SCHEDULE_TARGET,
        }


class SpeculativeGenerator:
    """Drafter + target verification loop over the torch-family chunk contract.

    ONE round loop (``_generate``) drives the greedy walk;
    ``generate_greedy`` is the thin named wrapper (the teaching API).
    ``stats`` carries the per-round records and the aggregate block with
    the SAME keys as the NumPy reference (``impl/_np/spec.py``) — the
    learning-mode spec block. The returned sequence INCLUDES the prompt,
    like every generator in the repo.
    """

    def __init__(self, model: _ChunkModel, drafter) -> None:
        self.model = model
        self.drafter = drafter

    # ── helpers ────────────────────────────────────────────────────────────
    def _prefill(self, prompt: torch.Tensor, cache: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Prefill = one full chunk pass over the prompt on an empty cache.

        forward_chunk over the whole prompt at position 0 IS the plain
        forward (pinned by chunk parity), and it returns the final-norm
        hidden stream — the drafter's conditioning at the anchor.
        Returns (pending_logits (1, 1, V) — the ANCHOR position's logits
        only, anchor_hidden (1, D)).
        """
        logits, x_final = self.model.forward_chunk(prompt, 0, cache)
        return logits[:, -1:, :], x_final[:, -1, :]

    @staticmethod
    def _trim_cache(cache: list[dict[str, torch.Tensor]], n: int) -> None:
        """Drop every cached position >= n (rejected draft tail)."""
        for layer in cache:
            layer["k"] = layer["k"][:, :, :n]
            layer["v"] = layer["v"][:, :, :n]

    def _commit(
        self, cache: list[dict[str, torch.Tensor]], position: int, token: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append the committed anchor token to the target cache and return
        (its logits (1, 1, V), its final-norm hidden (1, D)) — the next
        round's pending_logits / anchor_hidden.

        This is the round's SECOND target pass (see the module docstring's
        accounting): the correction/bonus token was never part of the draft
        block, so its K/V row and its hidden state do not exist until this
        pass — production MTP serving has the same shape.
        """
        device = cache[0]["k"].device
        token_ids = torch.tensor([[token]], dtype=torch.long, device=device)
        logits, x_final = self.model.forward_chunk(token_ids, position, cache)
        return logits, x_final[:, -1, :]  # (1, 1, V), (1, D) — the anchor's

    @staticmethod
    def _greedy_walk(
        pending_logits: torch.Tensor,
        logits_v: torch.Tensor,
        draft: list[int] | torch.Tensor,
        k_eff: int,
        emitted: list[int],
    ) -> tuple[list[bool], int, torch.Tensor]:
        """Greedy acceptance walk over one draft block.

        pred(j) = the target's logits at the position BEFORE draft token j:
        the anchor's pending logits for j=0, otherwise logits_v[j-1] (the
        verification pass scored positions t..t+k-1, so logits_v[j-1] is
        the target's prediction AT draft position j-1 — the distribution
        draft token j must match). The prediction at each walked position
        is the repetition-guarded argmax — the SAME rule the plain greedy
        generator applies, so the lossless contract holds against the
        shipped TextGenerator. The guard's view of "emitted" grows as the
        walk accepts tokens: a mid-block position is guarded by everything
        committed so far plus the block prefix above it.

        Returns (accept_mask, a = accepted count, corr (V,) — the
        correction position: the target's guarded logits at the first
        disagreement, or the last draft position's when the whole block
        accepted (the bonus token)).
        """
        accept_mask: list[bool] = []
        a = 0
        seen = list(emitted)  # the guard grows as the walk accepts
        while a < k_eff:
            pred = pending_logits[0, 0] if a == 0 else logits_v[0, a - 1]  # (V,)
            guarded = _apply_rep_guard(pred, seen)
            if int(torch.argmax(guarded).item()) == int(draft[a]):
                accept_mask.append(True)
                seen.append(int(draft[a]))
                a += 1
            else:
                break
        corr_source = pending_logits[0, 0] if a == 0 else logits_v[0, a - 1]
        if a == k_eff:
            corr_source = logits_v[0, k_eff - 1]  # bonus position
        corr = _apply_rep_guard(corr_source, seen)
        return accept_mask, a, corr

    # ── the round loop (the greedy walk drives it) ─────────────────────────
    def _generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        walk,
        k_req: int,
        sched: _SurvivalSchedule | None,
    ) -> tuple[torch.Tensor, dict]:
        """The draft → verify → accept → trim → commit → record loop.

        ``walk(pending_logits, logits_v, draft, k_eff, emitted) ->
        (mask, a, corr)`` is the verification rule. Everything else — cache
        trimming, drafter rollback, the round record with the drafter's
        inputs/outputs — is shared bookkeeping, written exactly once.
        """
        ids = torch.as_tensor(prompt, dtype=torch.long).reshape(1, -1)
        assert ids.shape[0] == 1, "speculative engine is single-sequence (see docstring)"
        t = int(ids.shape[1])
        scheduled = sched is not None

        cache = self.model.make_cache(1)
        pending_logits, anchor_hidden = self._prefill(ids, cache)
        anchor_token = ids[:, -1]  # (1,)
        self.drafter.reset()

        out_tokens: list[int] = []
        rounds: list[dict] = []
        t_draft_ms = t_verify_ms = 0.0

        while len(out_tokens) < max_new_tokens:
            remaining = max_new_tokens - len(out_tokens)
            k_eff = min(k_req, remaining)
            if scheduled:
                k_eff = min(k_eff, sched.verify_len())
            if k_eff <= 0:
                break

            # 1. DRAFT.
            t0 = time.perf_counter()
            draft_tokens, d_probs = self.drafter.draft(anchor_token, anchor_hidden, k_eff)
            d = torch.as_tensor(draft_tokens)[0].tolist()  # (k_eff,) greedy proposals
            t1 = time.perf_counter()

            # 2. VERIFY: one parallel pass scores every drafted position.
            draft_ids = torch.as_tensor(draft_tokens, dtype=torch.long, device=ids.device).reshape(1, k_eff)
            logits_v, _x_final = self.model.forward_chunk(draft_ids, t, cache)  # (1, k_eff, V)
            t2 = time.perf_counter()
            t_draft_ms += t1 - t0
            t_verify_ms += t2 - t1

            # 3. ACCEPT (the walk) + the correction/bonus token.
            accept_mask, a, corr = walk(pending_logits, logits_v, d, k_eff, emitted=out_tokens)
            committed = d[:a] + [int(torch.argmax(corr).item())]
            committed = committed[:remaining]
            n_new = len(committed)

            # The drafter's own inputs/outputs this round (the learning
            # page's drafter-node views) — same schema as the NumPy engine.
            round_drafter = {
                "anchor_token": int(anchor_token.reshape(-1)[0].item()),
                "anchor_hidden": [round(float(x), 6) for x in anchor_hidden.reshape(-1).float().cpu()],
                "draft_probs_top": [_top_slice(p) for p in d_probs],
            }

            # 4. ROLL BACK the rejected tail, then COMMIT the correction/
            #    bonus token — the round's second target pass; the new
            #    anchor's hidden/logits come from it.
            self._trim_cache(cache, t + a)
            if n_new > 0:
                pending_logits, anchor_hidden = self._commit(cache, t + a, committed[-1])
                anchor_token = torch.tensor([committed[-1]], dtype=torch.long, device=ids.device)
            self.drafter.rollback(keep=a)
            if scheduled:
                sched.update(a)

            out_tokens.extend(committed)
            rounds.append(
                {
                    "round": len(rounds),
                    "draft_tokens": [int(x) for x in d],
                    "accept_mask": accept_mask + [False] * (k_eff - len(accept_mask)),
                    "accepted": a,
                    "committed": committed,
                    "new_tokens": n_new,
                    "k_eff": k_eff,
                    "draft_ms": round((t1 - t0) * 1000, 3),
                    "verify_ms": round((t2 - t1) * 1000, 3),
                    "schedule": sched.state() if scheduled else None,
                    "drafter": round_drafter,
                }
            )
            t += n_new
            if n_new == 0:
                break

        stats = self._stats(out_tokens, rounds, t_draft_ms, t_verify_ms, scheduled, sched)
        return torch.cat(
            [ids, torch.tensor(out_tokens, dtype=torch.long, device=ids.device).reshape(1, -1)], dim=1
        ), stats

    # ── stats ──────────────────────────────────────────────────────────────
    def _stats(
        self,
        out_tokens: list[int],
        rounds: list[dict],
        t_draft_ms: float,
        t_verify_ms: float,
        scheduled: bool,
        sched: _SurvivalSchedule | None,
    ) -> dict:
        """Aggregate the per-round records into the spec stats block (same
        keys as the NumPy reference — the records/UI contract).

        n_target_forwards = 1 (prefill) + 2 × n_rounds (verify + commit per
        round) — the honest count; see the module docstring's accounting.
        """
        n_rounds = len(rounds)
        n_draft = sum(r["k_eff"] for r in rounds)
        n_accepted = sum(r["accepted"] for r in rounds)
        n_forwards = 1 + 2 * n_rounds
        return {
            "spec": self.drafter.family,
            "rounds": rounds,
            "n_rounds": n_rounds,
            "n_target_forwards": n_forwards,
            "n_draft_tokens": n_draft,
            "n_accepted": n_accepted,
            "accept_rate": round(n_accepted / max(n_draft, 1), 4),
            "mean_accepted": round(n_accepted / max(n_rounds, 1), 4),
            "tokens_per_target_forward": round(len(out_tokens) / n_forwards, 4),
            "draft_ms_total": round(t_draft_ms * 1000, 3),
            "verify_ms_total": round(t_verify_ms * 1000, 3),
            "schedule": sched.state() if scheduled and sched is not None else None,
        }

    # ── the thin named wrapper (the teaching API) ──────────────────────────
    def generate_greedy(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        k: int | None = None,
        schedule: _SurvivalSchedule | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Greedy speculative decoding — token-identical to plain greedy.

        Round (t = current length; cache holds positions 0..t-1):
          1. DRAFT  d = drafter.draft(anchor_token, anchor_hidden, k_eff)
          2. VERIFY logits_v = target.forward_chunk(d, position=t, cache)
          3. ACCEPT the longest agreeing prefix (a tokens); the target's own
             guarded argmax at the first disagreement (or the bonus position
             after a full block) is committed as the correction token.
          4. ROLL BACK the target cache to t+a, the drafter to the a kept
             drafts, append the committed token — the new anchor.
        """
        k_req = min(k if k is not None else self.drafter.block_size, self.drafter.block_size)
        sched = schedule if schedule is not None else _SurvivalSchedule(k_req)
        scheduled = schedule is not None or self.drafter.family == DSHARK
        seq, stats = self._generate(
            prompt,
            max_new_tokens,
            walk=self._greedy_walk,
            k_req=k_req,
            sched=sched if scheduled else None,
        )
        logger.info(
            "spec-greedy family=%s rounds=%d tokens=%d accepted=%d (%.0f%%) draft_ms=%.1f verify_ms=%.1f",
            self.drafter.family,
            stats["n_rounds"],
            int(sum(r["new_tokens"] for r in stats["rounds"])),
            stats["n_accepted"],
            100 * stats["accept_rate"],
            stats["draft_ms_total"],
            stats["verify_ms_total"],
        )
        return seq, stats
