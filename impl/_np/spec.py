"""NumPy speculative decoding engine — the math reference (ADR 0003).

**The loop, in plain words.** Ordinary generation repeats: big model
forward → one word. This engine repeats a fatter loop instead:
  1. DRAFT — the tiny drafter proposes k candidate words in a row;
  2. VERIFY — the big model runs ONE forward over all k at once and
     scores each position (what would *I* have written here?);
  3. ACCEPT — keep the leading run of words where the big model's pick
     equals the draft's word; at the first mismatch, take the big
     model's word instead, throw the rest of the draft away, and erase
     the thrown-away positions' internal memory (the KV cache) so the
     next round starts clean.
Accepted words + the correction word all become part of the output. If
the drafter is terrible, almost everything is rejected — the loop then
just degenerates to ordinary generation speed, never slower *in
content*: the committed words are always the big model's own choices.

**Target-forward accounting (honest):** each round is TWO target passes —
the verify chunk over the draft block, plus the commit pass that appends
the correction/bonus token and computes the next anchor's hidden state
(the drafter conditions on the target's hidden AT the anchor; the
correction token's hidden does not exist until it is forwarded — the
same shape production MTP serving uses). So::

    n_target_forwards = 1 (prefill) + 2 × n_rounds

and ``tokens_per_target_forward = n_new_tokens / n_target_forwards`` —
the honest per-target-pass metric the learning page displays. Speculation
therefore wins while acceptance beats ~1 token/round; a counting-stub
test pins the stat to reality.

**Target-forward accounting (honest):** each round is TWO target passes —
the verify chunk over the draft block, plus the commit pass that appends
the correction/bonus token and computes the next anchor's hidden state
(the drafter conditions on the target's hidden AT the anchor; the
correction token's hidden does not exist until it is forwarded — the
same shape production MTP serving uses). So::

    n_target_forwards = 1 (prefill) + 2 × n_rounds

and ``tokens_per_target_forward = n_new_tokens / n_target_forwards`` —
the honest per-target-pass metric the learning page displays.

Shape letters (CONTEXT.md → "Shape notation"): B = batch size (the
speculative engine is B=1), S = sequence length, D = embed_dim, V =
vocab_size, k = the drafter's block size, t = KV-cache depth.

Two verification rules:

**Greedy** (all tracks; the production rule at temperature 0):

    accept d_j while argmax(target logits at position t+j-1) == d_j

The target's token at the first mismatch (or the bonus token after a fully
accepted block) always wins, so the output is **token-identical to plain
greedy decoding** — the lossless contract pinned by the parity tests and the
``spec_mtp`` / ``spec_dspark`` verify_equivalence scenarios.

**Sampled** (NumPy only — the rejection-sampling theorem, Leviathan et al.
ICML 2023; the best teaching math in the feature):

    x ~ p_d          (the drafter samples its proposal)
    accept x with probability  min(1, p_t(x) / p_d(x))
    on rejection, resample from  norm(max(0, p_t − p_d))

Every committed token is distributed exactly as the target would have
sampled it — the drafter only changes *how many* tokens one target pass
confirms, never the distribution. Temperature note: renormalizing
softmax(l)^(1/T) IS softmax(l/T) (the normalizer Z^(1/T) cancels), so the
drafter's scaled distribution below is exactly its temperature-scaled
softmax. No repetition guard in this mode: the guard is a sampler feature,
not part of the target distribution the theorem preserves.

**Confidence-scheduled verification** (DSpark, arXiv 2607.05147): the
engine tracks per-prefix-length survival statistics online and caps each
round's draft length at the longest prefix whose estimated survival
probability still meets 0.8 — verification capacity is not wasted on block
suffixes likely to be rejected. Active for the DSpark drafter (its paper
feature); MTP drafts at the requested k.

Batching: the engine asserts batch 1. Per-row accepted lengths differ in
batched speculation (a genuine production problem — the reason DSpark
schedules *per request*); the teaching engine keeps the loop readable.

Logging (logger ``impl._np.spec``): per-round draft/accept/verify stats,
schedule state, and totals.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from impl._np.drafters import DSparkDrafter, MTPDrafter
from shared.draft import DSHARK

logger = logging.getLogger(__name__)

SCHEDULE_TARGET = 0.8
"""DSpark prefix-survival target: cap the verified length where the
estimated survival probability of the prefix drops below this."""


def _softmax(z: np.ndarray) -> np.ndarray:
    """Stable softmax over the last axis."""
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def _scaled(probs_or_logits: np.ndarray, temperature: float, *, is_logits: bool) -> np.ndarray:
    """Temperature-scale a distribution: softmax(l/T) (is_logits=True) or
    renormalized p^(1/T) (is_logits=False) — identical math for p =
    softmax(l), see the module docstring."""
    if is_logits:
        return _softmax(probs_or_logits / temperature)
    p = np.clip(probs_or_logits, 1e-12, None) ** (1.0 / temperature)
    return p / np.sum(p, axis=-1, keepdims=True)


def _top_slice(p: np.ndarray, n_top: int = 8) -> list[list[float]]:
    """A distribution's top-n as [[token_id, prob], ...] (the page's bars)."""
    flat = np.asarray(p).reshape(-1)
    idx = np.argsort(flat)[::-1][:n_top]
    return [[int(i), round(float(flat[i]), 6)] for i in idx]


class _SurvivalSchedule:
    """Online per-prefix survival estimate (the DSpark schedule state).

    Plain words: a little bookkeeper that watches past rounds and counts,
    for each prefix length, how often the verifier kept at least that
    many words. counts[length] / rounds is the observed fraction of rounds
    where a draft of that length survived — the empirical probability.
    The engine asks "how long may the drafter run this round?" and the
    answer is the longest length whose survival fraction still clears
    the 0.8 bar (at least 1 — always draft something). The point: stop
    paying verification compute on block suffixes that history says
    almost never survive (the DSpark paper's throughput observation).

    counts[length] = rounds whose accepted length was >= length;
    rounds = total.  p(length) = counts[length] / rounds is the empirical
    probability that a draft prefix of that length survives verification.
    The verified-length cap is the longest length with p >= SCHEDULE_TARGET
    (at least 1 — always draft something).
    """

    def __init__(self, k_max: int) -> None:
        self.counts = np.zeros(k_max + 1, dtype=np.int64)
        self.rounds = 0

    def update(self, accepted: int) -> None:
        """Record one round's outcome: accepted = how many drafted tokens
        survived (0 = all rejected — still counts as a round of history)."""
        self.rounds += 1
        for length in range(1, accepted + 1):
            self.counts[length] += 1

    def verify_len(self) -> int:
        """Longest prefix length whose survival estimate meets the target."""
        if self.rounds == 0:
            return int(self.counts.shape[0] - 1)  # no data yet → full block
        p = self.counts / self.rounds
        best = 1
        for length in range(1, p.shape[0]):
            if p[length] >= SCHEDULE_TARGET:
                best = length
        return int(best)

    def state(self) -> dict:
        """JSON-friendly snapshot for the record / UI."""
        p = (self.counts / self.rounds).tolist() if self.rounds else [0.0] * int(self.counts.shape[0])
        return {
            "rounds": self.rounds,
            "survival": [round(v, 4) for v in p],
            "verify_len": self.verify_len(),
            "target": SCHEDULE_TARGET,
        }


class SpeculativeGenerator:
    """Drafter + target verification loop over the KV-step interface.

    ONE round loop drives both verification rules: ``generate_greedy`` and
    ``generate_sampled`` are thin named wrappers that pick the walk
    (greedy argmax comparison vs the rejection-sampling theorem) and hand
    it to :meth:`_generate` — the draft/verify/trim/rollback/record
    bookkeeping lives exactly once.

    Usage:
        gen = SpeculativeGenerator(model, drafter)
        seq, stats = gen.generate_greedy(prompt, max_new_tokens, k=4)
        seq, stats = gen.generate_sampled(prompt, max_new_tokens, temperature=0.8, k=4)

    ``stats`` carries the per-round record (draft tokens, accept mask,
    accepted length, timings, the drafter's inputs/outputs, schedule
    state) — the learning-mode spec block. The returned sequence INCLUDES
    the prompt, like every generator in the repo.
    """

    def __init__(self, model, drafter: MTPDrafter | DSparkDrafter) -> None:
        self.model = model
        self.drafter = drafter

    # ── shared helpers ─────────────────────────────────────────────────────
    def _prefill(self, prompt: np.ndarray, cache: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Prefill = one full chunk pass over the prompt on an empty cache.

        forward_chunk over the whole prompt at position 0 IS the plain
        forward (pinned by chunk parity), and unlike forward_prefill it
        also returns the final-norm hidden stream — the drafter's
        conditioning at the anchor. Returns (pending_logits (1, 1, V) — the
        ANCHOR position's logits only, anchor_hidden (1, D)).
        """
        logits, x_final = self.model.forward_chunk(prompt, 0, cache)
        return logits[:, -1:, :], x_final[:, -1, :]

    @staticmethod
    def _trim_cache(cache: list[dict], n: int) -> None:
        """Drop every cached position >= n (rejected draft tail)."""
        for layer in cache:
            layer["k"] = layer["k"][:, :, :n]
            layer["v"] = layer["v"][:, :, :n]

    def _commit(self, cache: list[dict], position: int, token: int) -> tuple[np.ndarray, np.ndarray]:
        """Append the committed anchor token to the target cache and return
        (its logits (1, 1, V), its final-norm hidden (1, D)) — the next
        round's pending_logits / anchor_hidden.

        cache: the per-layer cache list — MUST hold exactly positions
            0..position-1 (the caller trims first); mutated in place: ends
            holding 0..position.
        position: the absolute token index the committed token occupies
            (0-based; = the trimmed cache depth).
        token: the committed token id (the correction or bonus token).

        This is the round's SECOND target pass (see the module docstring's
        accounting): the correction/bonus token was never part of the draft
        block, so its K/V row and its hidden state do not exist until this
        pass — production MTP serving has the same shape.
        """
        logits, x_final = self.model.forward_chunk(np.array([[token]], dtype=np.int32), position, cache)
        return logits, x_final[:, -1, :]  # (1, 1, V), (1, D) — the anchor's

    @staticmethod
    def _greedy_walk(
        pending_logits: np.ndarray,
        logits_v: np.ndarray,
        draft: np.ndarray,
        k_eff: int,
        emitted: list[int],
    ) -> tuple[list[bool], int, np.ndarray]:
        """Greedy acceptance walk over one draft block.

        pending_logits: (1, 1, V) the ANCHOR position's target logits
            (the prediction that scores draft position 0) — carried from
            the previous round's commit pass or the prefill.
        logits_v: (1, k_eff, V) this round's verification logits (the
            chunk pass over the draft block).
        draft: (k_eff,) the drafter's proposed token ids.
        k_eff: this round's draft length.
        emitted: the committed tokens so far (the repetition guard's
            view; grows as the walk accepts).

        pred(j) = the target's logits at the position BEFORE draft token j:
        the anchor's pending logits for j=0, otherwise logits_v[j-1] (the
        verification pass scored positions t..t+k-1, so logits_v[j-1] is
        the target's prediction AT draft position j-1 — the distribution
        draft token j must match). The prediction at each walked position
        is the repetition-guarded argmax — the SAME rule the plain greedy
        generator applies (impl/_np/inference.py ``_apply_rep_guard``), so
        the lossless contract holds against the shipped plain decoder.
        The guard's view of "emitted" grows as the walk accepts tokens: a
        mid-block position is guarded by everything committed so far plus
        the block prefix above it.

        Returns (accept_mask, a = accepted count, corr_logits (V,) — the
        correction position: the target's guarded logits at the first
        disagreement, or the last draft position's when the whole block
        accepted (the bonus token)).
        """
        from impl._np.inference import _apply_rep_guard

        accept_mask: list[bool] = []
        a = 0
        seen = list(emitted)  # the guard grows as the walk accepts
        while a < k_eff:
            pred = pending_logits[0, 0] if a == 0 else logits_v[0, a - 1]  # (V,)
            guarded = _apply_rep_guard(pred.reshape(1, -1).astype(np.float64), [seen])[0]
            if int(np.argmax(guarded)) == int(draft[a]):
                accept_mask.append(True)
                seen.append(int(draft[a]))
                a += 1
            else:
                break
        corr_source = pending_logits[0, 0] if a == 0 else logits_v[0, a - 1]
        if a == k_eff:
            corr_source = logits_v[0, k_eff - 1]  # bonus position
        corr = _apply_rep_guard(corr_source.reshape(1, -1).astype(np.float64), [seen])[0]
        return accept_mask, a, corr

    @staticmethod
    def _sampled_walk(
        pending_logits: np.ndarray,
        logits_v: np.ndarray,
        draft: np.ndarray,
        k_eff: int,
        emitted: list[int],
        draft_probs: list[np.ndarray],
        temperature: float = 1.0,
        rng: np.random.Generator = None,
    ) -> tuple[list[bool], int, np.ndarray]:
        """Rejection-sampling acceptance walk (the theorem; NumPy only).

        Same (pending_logits, logits_v, draft, k_eff, emitted) head as
        ``_greedy_walk`` — the one round loop drives both walks through one
        call shape — plus the sampled-only extras (draft_probs, temperature,
        rng). ``emitted`` is accepted for signature parity and unused: no
        repetition guard in this mode (the guard is a sampler feature, not
        part of the target distribution the theorem preserves).

        Accept draft[j] with probability min(1, p_t/p_d); at the first
        rejection return the residual distribution norm(max(0, p_t−p_d));
        on a full block return the target's own scaled distribution at the
        last position (the bonus draw). Returns (accept_mask, a, p_next (V,)).
        """
        T = temperature
        accept_mask: list[bool] = []
        a = 0
        while a < k_eff:
            pt = _scaled(pending_logits[0, 0] if a == 0 else logits_v[0, a - 1], T, is_logits=True)
            pd = _scaled(draft_probs[a][0], T, is_logits=False)
            x = int(draft[a])
            if float(rng.random()) < min(1.0, float(pt[x]) / float(pd[x])):
                accept_mask.append(True)
                a += 1
            else:
                break
        if a < k_eff:
            pt = _scaled(pending_logits[0, 0] if a == 0 else logits_v[0, a - 1], T, is_logits=True)
            pd = _scaled(draft_probs[a][0], T, is_logits=False)
            resid = np.clip(pt - pd, 0.0, None)
            return accept_mask, a, resid / max(float(resid.sum()), 1e-12)
        return accept_mask, a, _scaled(logits_v[0, k_eff - 1], T, is_logits=True)

    @staticmethod
    def _drafter_record(anchor_token: np.ndarray, anchor_hidden: np.ndarray, d_probs: list[np.ndarray]) -> dict:
        """The drafter's own inputs/outputs for one round (the learning
        page's drafter-node views): its two inputs (anchor token id +
        anchor hidden) and its per-position output distributions, trimmed
        to top-8 per position (the page shows distributions, not a k×V wall).

        anchor_token: (1,) the anchor's token id; anchor_hidden: (1, D)
        the target's final-norm hidden at the anchor; d_probs: the
        drafter's k per-position distributions, each (1, V)."""
        return {
            "anchor_token": int(anchor_token.reshape(-1)[0]),
            "anchor_hidden": [round(float(x), 6) for x in np.asarray(anchor_hidden).reshape(-1)],
            "draft_probs_top": [_top_slice(p) for p in d_probs],
        }

    # ── the ONE round loop (both walks drive it) ───────────────────────────
    def _generate(
        self,
        prompt: np.ndarray,
        max_new_tokens: int,
        walk,
        propose,
        nxt_pick,
        temperature: float,
        rng: np.random.Generator,
        k_req: int,
        sched: _SurvivalSchedule | None,
    ) -> tuple[np.ndarray, dict]:
        """The draft → verify → accept → trim → commit → record loop.

        prompt: (S,) or (1, S) the starting token ids (B=1, asserted).
        max_new_tokens: the output length cap (truncates mid-round).
        walk: the verification rule — greedy or sampled (see the wrappers;
            both share the (pending_logits, logits_v, draft, k_eff,
            emitted) head).
        propose: the proposal strategy — (draft_tokens, d_probs) -> the
            proposed token ids (greedy: the argmax row; sampled: draws
            from the temperature-scaled p_d).
        nxt_pick: the correction strategy — the walk's third output ->
            the committed token id (greedy: argmax; sampled: a draw).
        temperature: the sampling temperature (greedy ignores it).
        rng: the seeded RNG.
        k_req: the draft block size (capped at the drafter's).
        sched: the DSpark survival schedule (None = unscheduled; MTP
            without a passed schedule is unscheduled).

        Everything else — cache trimming, drafter rollback, the round
        record with the drafter's inputs/outputs — is shared bookkeeping,
        written exactly once. Returns (seq (1, S+n), stats dict)."""
        ids = np.asarray(prompt, dtype=np.int32).reshape(1, -1)
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

            # 1. DRAFT (+ sample the proposals when the walk is stochastic).
            t0 = time.perf_counter()
            draft_tokens, d_probs = self.drafter.draft(anchor_token, anchor_hidden, k_eff)
            d = propose(draft_tokens, d_probs)
            t1 = time.perf_counter()

            # 2. VERIFY: one parallel pass scores every drafted position.
            logits_v, _x_final = self.model.forward_chunk(
                np.asarray(draft_tokens, dtype=np.int32), t, cache
            )  # (1, k_eff, V)
            t2 = time.perf_counter()
            t_draft_ms += t1 - t0
            t_verify_ms += t2 - t1

            # 3. ACCEPT (the walk) + the correction/bonus token. Both walks
            #    share the 5-arg head (the sampled extras — draft_probs,
            #    temperature, rng — ride through a partial built per mode).
            accept_mask, a, nxt = walk(pending_logits, logits_v, d, k_eff, emitted=out_tokens)
            new_token = nxt_pick(nxt)
            committed = [int(x) for x in d[:a]] + [new_token]
            committed = committed[:remaining]
            n_new = len(committed)

            round_drafter = self._drafter_record(anchor_token, anchor_hidden, d_probs)

            # 4. ROLL BACK the target cache to t+a (the rejected tail), then
            #    COMMIT the correction/bonus token — the round's second
            #    target pass; the new anchor's hidden/logits come from it.
            self._trim_cache(cache, t + a)
            if n_new > 0:
                pending_logits, anchor_hidden = self._commit(cache, t + a, committed[-1])
                anchor_token = np.array([committed[-1]], dtype=np.int32)
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
        return np.concatenate([ids, np.array(out_tokens, dtype=np.int32).reshape(1, -1)], axis=1), stats

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
        """Aggregate the per-round records into the spec stats block.

        out_tokens: the committed tokens (excluding the prompt); rounds:
            the per-round records; t_draft_ms/t_verify_ms: accumulated
            timings (seconds); scheduled: whether a schedule ran; sched:
            the schedule object (state() snapshot goes in the block).

        n_target_forwards = 1 (prefill) + 2 × n_rounds (verify + commit per
        round) — the honest count; see the module docstring's accounting.
        Returns the spec block dict (the records/UI contract — same keys as
        the torch-family engine)."""
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

    # ── the two thin named wrappers (the teaching API) ─────────────────────
    def generate_greedy(
        self,
        prompt: np.ndarray,
        max_new_tokens: int,
        k: int | None = None,
        schedule: _SurvivalSchedule | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Greedy speculative decoding — token-identical to plain greedy.

        Round (t = current length; cache holds positions 0..t-1):
          1. DRAFT  d = drafter.draft(anchor_token, anchor_hidden, k_eff)
          2. VERIFY logits_v = target.forward_chunk(d, position=t, cache)
          3. ACCEPT the longest agreeing prefix (a tokens); the target's own
             argmax at the first disagreement (or the bonus position after
             a full block) is committed as the correction token.
          4. ROLL BACK the target cache to t+a, the drafter to the a kept
             drafts, append the committed token — the new anchor.

        Parameters
        ----------
        prompt : np.ndarray, shape (S,) or (1, S)
            The token ids to start from (B=1 — the engine is
            single-sequence; asserted).
        max_new_tokens : int
            How many tokens to generate (the output is truncated here even
            mid-round).
        k : int, optional
            The draft block size (default: the drafter's trained
            block_size; capped at it — the page's slider drags DOWN only).
        schedule : _SurvivalSchedule, optional
            An existing survival schedule to keep updating (DSpark gets one
            automatically; passing one to MTP makes MTP scheduled too).

        Returns
        -------
        seq : np.ndarray, shape (1, S + generated)
            The full sequence INCLUDING the prompt.
        stats : dict
            The spec block: rounds (each with draft_tokens, accept_mask,
            committed, timings, the drafter's inputs/outputs), n_rounds,
            n_target_forwards, n_accepted, accept_rate, mean_accepted,
            tokens_per_target_forward, schedule state.
        """
        k_req = min(k if k is not None else self.drafter.block_size, self.drafter.block_size)
        sched = schedule if schedule is not None else _SurvivalSchedule(k_req)
        scheduled = schedule is not None or self.drafter.family == DSHARK
        seq, stats = self._generate(
            prompt,
            max_new_tokens,
            walk=self._greedy_walk,
            propose=lambda draft_tokens, d_probs: np.asarray(draft_tokens)[0],  # the argmax row
            nxt_pick=lambda p_next: int(np.argmax(p_next)),
            temperature=1.0,
            rng=np.random.default_rng(0),
            k_req=k_req,
            sched=sched if scheduled else None,
        )
        logger.info(
            "spec-greedy family=%s rounds=%d tokens=%d accepted=%d (%.0f%%) draft_ms=%.1f verify_ms=%.1f",
            self.drafter.family,
            stats["n_rounds"],
            int(stats["rounds"] and sum(r["new_tokens"] for r in stats["rounds"]) or 0),
            stats["n_accepted"],
            100 * stats["accept_rate"],
            stats["draft_ms_total"] / 1000,
            stats["verify_ms_total"] / 1000,
        )
        return seq, stats

    def generate_sampled(
        self,
        prompt: np.ndarray,
        max_new_tokens: int,
        temperature: float = 1.0,
        k: int | None = None,
        seed: int = 42,
    ) -> tuple[np.ndarray, dict]:
        """Lossless sampled speculative decoding (Leviathan et al. 2023).

        Committed tokens are distributed exactly as the target's own
        temperature sampling — the drafter never changes the distribution,
        only how many tokens one target pass confirms. No schedule, no
        repetition guard (see the module docstring).

        Parameters
        ----------
        prompt : np.ndarray, shape (S,) or (1, S)
            The token ids to start from (B=1, asserted).
        max_new_tokens : int
            How many tokens to generate (output truncated here).
        temperature : float
            The sampling temperature for BOTH the drafter's proposals and
            the target's acceptance/residual distributions (the theorem's
            p_t and p_d must use the same scaling); clamped to >= 1e-8.
        k : int, optional
            The draft block size (default: the drafter's trained
            block_size, capped at it).
        seed : int
            Seeds the RNG — reproducible sampling.

        Returns
        -------
        seq : np.ndarray, shape (1, S + generated)
            The full sequence INCLUDING the prompt.
        stats : dict
            The spec block (same keys as greedy, plus mode="sampled" and
            the temperature).
        """
        T = max(temperature, 1e-8)
        k_req = min(k if k is not None else self.drafter.block_size, self.drafter.block_size)
        rng = np.random.default_rng(seed)

        round_state: dict = {}

        def _propose_sampled(draft_tokens, d_probs):
            # The drafter SAMPLES its proposal from p_d^(1/T) renormalized
            # (= its temperature-scaled softmax; see the module docstring).
            # The walk consumes the same per-position distributions — stash
            # them where the partial below can hand them over.
            round_state["draft_probs"] = d_probs
            d = np.empty(len(d_probs), dtype=np.int64)
            for j in range(len(d_probs)):
                pd = _scaled(d_probs[j][0], T, is_logits=False)  # (V,)
                d[j] = int(rng.choice(pd.shape[0], p=pd))
            return d

        def _sampled_walk_bound(pending_logits, logits_v, draft, k_eff, emitted):
            return self._sampled_walk(
                pending_logits,
                logits_v,
                draft,
                k_eff,
                emitted,
                draft_probs=round_state["draft_probs"],
                temperature=T,
                rng=rng,
            )

        seq, stats = self._generate(
            prompt,
            max_new_tokens,
            walk=_sampled_walk_bound,
            propose=_propose_sampled,
            nxt_pick=lambda p_next: int(rng.choice(p_next.shape[0], p=p_next)),
            temperature=T,
            rng=rng,
            k_req=k_req,
            sched=None,
        )
        stats["mode"] = "sampled"
        stats["temperature"] = temperature
        logger.info(
            "spec-sampled family=%s T=%.2f rounds=%d tokens=%d accepted=%d",
            self.drafter.family,
            temperature,
            stats["n_rounds"],
            int(sum(r["new_tokens"] for r in stats["rounds"])),
            stats["n_accepted"],
        )
        return seq, stats
