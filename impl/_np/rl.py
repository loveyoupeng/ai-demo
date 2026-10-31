# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""GRPO RL for the NumPy reference implementation — the math owner (ADR 0004).

GRPO (Group Relative Policy Optimization; Shao et al., DeepSeekMath — the
DeepSeek-R1 recipe) turns "make the reward go up" into a language-model
gradient. The whole algorithm is three formulas plus the existing
training machinery:

1. **Sample a group**: G completions per prompt from the current policy
   (``rollout`` below — prefill + temperature/top-k decode steps).
2. **Score and baseline**: each completion gets a reward r_i
   (``shared.thinking.reward``: correctness + format); the advantage is
   the group-normalized reward,

       A_i = (r_i − μ) / (σ + 1e-4)        μ, σ = group mean / std

   *Why this is allowed (the group-baseline theorem):* the policy
   gradient ∇θ J = E[A · ∇θ log π(y|x)] is invariant to subtracting any
   baseline that does not depend on the action — E[b(x)·∇θ log π] =
   b(x)·E[∇θ log π] = 0 because π is a normalized density over the
   action space. The group mean is an *empirical* baseline (it does
   depend on the actions), which is why GRPO normalizes per group and,
   in exchange, never trains a critic network — the G completions of
   ONE prompt carry the counterfactual "how good would the others have
   been?"

3. **Reweight the CE gradient**: with y_t the sampled tokens,

       ∇θ L = E[ w_t · ∇θ(−log π_θ(y_t|·)) ] + β · ∇θ KL(π_θ ‖ π_ref)

   where w = clip(π_θ(y_t)/π_old(y_t), 1±ε) · A. The ratio corrects for
   the policy having moved since the rollout — at the first step of a
   fresh batch it is 1 (on-policy); the ε-clip guards drift within an
   epoch over a fixed batch of rollouts.

**The key implementation insight (why this file is small):** the CE
gradient w.r.t. the logits is ``softmax(logits) − onehot(y)`` (see
``impl/_np/cross_entropy.py``). Scaling that by a per-position weight is
the whole policy gradient — so GRPO enters the model's analytic backward
through ``NumPyModel.backward_from_dlogits`` (the shared chain, one step
below the CE entry) with a *weighted* dlogits. Same chain, different
scalar weights: the ADR 0004 statement, executable.

**KL estimator (k3, Schulman et al. "Approximating KL Divergence"):**
with p = policy, r = reference (per-token log-probabilities of the
sampled token y),

    KL(p ‖ r) ≈ k3 = exp(r − p) − (r − p) − 1

Unbiased in expectation under p, and — the reason production uses it —
bounded variance when r(y) → 0 where p samples y (naive log-ratio
estimators explode there). Its gradient w.r.t. the policy logits is
derived inline in ``grpo_dlogits``.

Shape letters (CONTEXT.md → "Shape notation"): B batch, S sequence, V
vocab, G group size, T new tokens.
"""

from __future__ import annotations

import logging

import numpy as np

from impl._np.model import NumPyModel
from impl._np.optimizer import AdamW
from impl._np.softmax import stable_softmax
from impl._np.training import clip_gradients
from shared.thinking import reward as score_reward

logger = logging.getLogger(__name__)


class GRPOConfig:
    """The knobs of one GRPO run (defaults = the ADR 0004 decision).

    group_size : completions sampled per prompt (G=8).
    temperature, top_k : rollout sampling parameters (exploration).
    max_new_tokens : rollout budget; truncation mid-think scores 0 —
        the honest lesson that reasoning length is a budget.
    clip_eps : importance-ratio clip (ε=0.2, PPO heritage).
    kl_beta : KL-to-reference weight (β=0.04; 0 disables the penalty).
    adv_eps : std guard in the advantage denominator (1e-4).
    """

    def __init__(
        self,
        group_size: int = 8,
        temperature: float = 1.0,
        top_k: int = 40,
        max_new_tokens: int = 80,
        clip_eps: float = 0.2,
        kl_beta: float = 0.04,
        adv_eps: float = 1e-4,
    ) -> None:
        self.group_size = group_size
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
        self.adv_eps = adv_eps


# ── 1. Rollout: the policy samples G completions ──────────────────────────────


def _sample(logits: np.ndarray, temperature: float, top_k: int, rng: np.random.Generator) -> np.ndarray:
    """Temperature + top-k multinomial sampling over (G, V) logits → (G,).

    The same recipe as the learning page's ``_sample`` and
    ``impl._np.inference.TextGenerator.generate_sampled``: logits/T →
    top-k mask → stable softmax → one draw per row.
    """
    scaled = logits / max(temperature, 1e-8)  # (G, V)
    if 0 < top_k < scaled.shape[-1]:
        kth = np.sort(scaled, axis=-1)[:, -top_k]  # (G,) the k-th best logit
        scaled = np.where(scaled < kth[:, None], -np.inf, scaled)
    probs = stable_softmax(scaled, axis=-1)
    cdf = np.cumsum(probs, axis=-1)
    cdf /= cdf[:, -1:]
    u = rng.random((probs.shape[0], 1))
    return np.argmax(u < cdf, axis=-1).astype(np.int64)


def rollout(
    model: NumPyModel,
    prompt_ids: np.ndarray,
    grpo: GRPOConfig,
    rng: np.random.Generator,
    stop_ids: tuple[int, ...] = (),
) -> np.ndarray:
    """Sample ``group_size`` completions for ONE prompt.

    prompt_ids: (S,) int — the tokenized question (tags NOT appended;
        the policy must learn to open ``<|think|>`` itself).
    stop_ids: token ids that end a completion early (e.g. the tokenizer's
        EOS); rows past a stop are frozen.
    Returns: (G, T) int — the sampled new tokens (T = max_new_tokens,
        the actual per-row lengths are recoverable from the stop tags).

    Generation is the KV-step path (the same O(1)-per-token interface the
    learning page serves): one prefill of the G-row batch, then
    ``forward_step`` per position — the prompt's K/V are computed once.
    """
    G, T, S = grpo.group_size, grpo.max_new_tokens, prompt_ids.shape[0]
    ctx = model.config.context_length
    T = min(T, ctx - S - 1)  # prompt + sampled tokens must fit the cache
    seqs = np.tile(prompt_ids, (G, 1))  # (G, S)
    out = np.zeros((G, T), dtype=np.int64)
    done = np.zeros(G, dtype=bool)
    cache = model.make_cache(G)
    logits, _ = model.forward_prefill(seqs, cache=cache)  # (G, S, V)
    nxt_logits: np.ndarray = logits[:, -1, :]  # (G, V) — after the prompt
    for t in range(T):
        if t > 0:
            # step with the token sampled at t−1 (position S+t−1); the
            # finished rows keep stepping (their samples are discarded).
            nxt_logits = model.forward_step(seqs[:, -1:], S + t - 1, cache)[:, 0, :]
        nxt = _sample(nxt_logits, grpo.temperature, grpo.top_k, rng)  # (G,)
        if stop_ids:
            done |= np.isin(nxt, stop_ids)
        out[:, t] = np.where(done, 0, nxt)
        seqs = np.concatenate([seqs, out[:, t : t + 1]], axis=1)  # (G, S+t+1)
        if done.all() or seqs.shape[1] >= ctx:
            break
    return out


# ── 2. Score + group-relative advantage ───────────────────────────────────────


def group_advantage(rewards: np.ndarray, adv_eps: float = 1e-4) -> np.ndarray:
    """A = (r − μ_group)/(σ_group + adv_eps) — the critic-free baseline.

    rewards: (G,) one scalar per completion of the SAME group.
    Zero-sum within the group by construction: Σ A_i·(σ+ε) = Σ r_i − G·μ = 0.
    """
    return (rewards - rewards.mean()) / (rewards.std() + adv_eps)


# ── 3. The GRPO gradient on dlogits (the reweighted-CE insight) ───────────────


def logprobs_at(logits: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """log π(id | ·) at every position: (B, S, V) logits, (B, S) ids → (B, S).

    The stable log-softmax (the same logsumexp form as
    ``CrossEntropyLoss.forward``); ids out of range (masked -100) score 0.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    in_range = (ids >= 0) & (ids < logits.shape[-1])
    safe = np.where(in_range, ids, 0)
    target = np.take_along_axis(shifted, safe[..., None], axis=-1)[..., 0]
    return np.where(in_range, target - logsumexp[..., 0], 0.0)


def kl_k3(policy_logprobs: np.ndarray, ref_logprobs: np.ndarray) -> np.ndarray:
    """Per-token k3 KL estimate: ``exp(Δ) − Δ − 1`` with Δ = r − p.

    policy_logprobs/ref_logprobs: (B, S) — log π_θ(y_t|·), log π_ref(y_t|·).
    E_p[k3] = KL(p ‖ ref) — unbiased under the policy.
    """
    delta = ref_logprobs - policy_logprobs
    return np.exp(delta) - delta - 1.0


def grpo_dlogits(
    logits: np.ndarray,
    graded_ids: np.ndarray,
    token_weights: np.ndarray,
    mask: np.ndarray,
    ref_logprobs: np.ndarray | None = None,
    kl_beta: float = 0.04,
) -> np.ndarray:
    """dL/dlogits for the GRPO objective — THE reference formula.

    Per valid position t the loss is (p = log π_θ(y_t|·), r = log π_ref):

        L_t = −w_t · p + β · k3_t,       k3 = exp(r−p) − (r−p) − 1

    so with ``resid = onehot(y) − softmax`` (the CE gradient core — note
    d(−p)/dlogits = resid, since p is the target's log-prob):

        dL_t/dlogits = −w_t · resid + β · (1 − exp(r−p)) · resid

    The KL factor's gradient: with u = r − p, dk3/du = e^u − 1 and
    du/dp = −1, so dk3/dp = 1 − e^(r−p) (NEGATIVE when the reference
    likes y more than the policy — pulling π toward π_ref), and
    dp/dlogits = resid, so dk3/dlogits = (1 − e^(r−p)) · resid. The
    loss ADDS β·k3, so this term enters with the + sign above.

    logits : (B, S, V) — the policy's logits on the rollout batch.
    graded_ids : (B, S) — the token graded at each position.
    token_weights : (B, S) — w = ratio_clipped advantage broadcast over
        the completion's tokens (sequence-level advantage, GRPO's
        per-response convention).
    mask : (B, S) — 1 on graded positions, 0 elsewhere (prompt/pad).
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_ = np.exp(shifted)
    softmax = exp_ / exp_.sum(axis=-1, keepdims=True)  # (B, S, V)
    onehot = np.zeros_like(softmax)
    np.put_along_axis(onehot, graded_ids[..., None], 1.0, axis=-1)
    resid = onehot - softmax  # d(log π(y))/dlogits

    dlogits = -token_weights[..., None] * resid  # (B, S, V)

    if ref_logprobs is not None and kl_beta > 0.0:
        policy_lp = logprobs_at(logits, graded_ids)  # (B, S)
        delta = ref_logprobs - policy_lp
        kl_factor = 1.0 - np.exp(delta)  # (B, S) — dk3/dp (see derivation)
        dlogits = dlogits + kl_beta * kl_factor[..., None] * resid

    dlogits = np.where(mask[..., None] == 1.0, dlogits, 0.0)
    n_valid = float(mask.sum())
    if n_valid == 0:
        return np.zeros_like(dlogits)
    return dlogits / n_valid  # mean over graded positions (CE convention)


def grpo_step(
    model: NumPyModel,
    reference: NumPyModel,
    batch_ids: np.ndarray,
    response_mask: np.ndarray,
    advantages: np.ndarray,
    optimizer: AdamW,
    grpo: GRPOConfig,
    old_logprobs: np.ndarray | None = None,
) -> dict[str, float]:
    """One synchronous GRPO update on a fixed batch of scored rollouts.

    batch_ids : (B, S) — prompt + sampled tokens per rollout, the full
        teacher-forced stream (the model re-forwards it: the rollout's KV
        cache holds K/V from OLD weights, unusable after any update).
    response_mask : (B, S) — 1 on the SAMPLED tokens (the positions that
        produced them are the graded ones).
    advantages : (B,) — per-rollout group-relative advantage.
    old_logprobs : (B, S) or None — log π_old at the graded positions;
        None → ratio 1 (fully on-policy, the first step of a batch).

    Grading convention (the SFT shift, reused): the logits at position t
    score token t+1, so a sampled token at index i is graded by the
    logits at position i−1; the mask shift below implements this.
    """
    # 1. Policy forward + the graded positions (mask shifted onto logits).
    logits, trace = model.forward_with_trace(batch_ids)  # (B, S, V)
    B, S = batch_ids.shape
    graded_mask = np.zeros((B, S), dtype=np.float64)
    graded_mask[:, :-1] = response_mask[:, 1:]  # token i graded by logits i−1
    graded_ids = np.full_like(batch_ids, -100)
    graded_ids[:, :-1] = batch_ids[:, 1:]  # what position t must predict
    valid = (graded_mask == 1.0) & (graded_ids >= 0)
    graded_mask = valid.astype(np.float64)
    graded_ids = np.where(valid, graded_ids, 0)
    policy_lp = logprobs_at(logits, graded_ids)  # (B, S)

    # 2. Importance ratio (1 when on-policy) with ε-clip.
    ratio = np.ones_like(policy_lp) if old_logprobs is None else np.exp(policy_lp - old_logprobs)
    clipped = np.clip(ratio, 1.0 - grpo.clip_eps, 1.0 + grpo.clip_eps)
    token_weights = clipped * advantages[:, None] * graded_mask  # (B, S)

    # 3. Reference forward (frozen) for the k3 KL term.
    ref_lp = None
    if grpo.kl_beta > 0.0:
        ref_lp = logprobs_at(reference.forward(batch_ids), graded_ids)

    # 4. GRPO dlogits → the shared analytic backward chain.
    dlogits = grpo_dlogits(logits, graded_ids, token_weights, graded_mask, ref_logprobs=ref_lp, kl_beta=grpo.kl_beta)
    grads = model.backward_from_dlogits(dlogits, trace, batch_ids)
    clip_gradients(grads, max_norm=1.0)
    optimizer.step(model.get_all_parameters(), grads)

    # 5. Diagnostics (the teaching log line).
    if valid.any():
        kl = float(np.mean(kl_k3(policy_lp, ref_lp)[valid])) if ref_lp is not None else 0.0
        frac_clipped = float(np.mean((np.abs(ratio - 1.0) > grpo.clip_eps)[valid]))
        loss_proxy = float(-np.mean((policy_lp * token_weights)[valid]))
    else:
        kl = frac_clipped = loss_proxy = 0.0
    return {"loss_proxy": loss_proxy, "kl": kl, "frac_clipped": frac_clipped, "n_valid": float(valid.sum())}


# ── 4. One full GRPO iteration over a prompt batch ────────────────────────────


def grpo_iteration_scored(
    model: NumPyModel,
    reference: NumPyModel,
    items: list[tuple[np.ndarray, str]],
    decode,
    grpo: GRPOConfig,
    optimizer: AdamW,
    rng: np.random.Generator,
    log_every: int = 1,
) -> dict[str, float]:
    """One GRPO iteration over (prompt_ids, gold_answer) items.

    For each item: sample G completions (``rollout``), score them
    (``shared.thinking.reward``: 1.0·correctness + 0.2·format), compute
    the group advantage, and take one optimizer step on the group.
    Every ``log_every`` items, log one full sample group — the "watch
    the policy think" moment (prompt, completions, rewards, advantages).
    """
    stats: dict[str, float] = {"reward": 0.0, "format": 0.0, "correct": 0.0, "kl": 0.0, "frac_clipped": 0.0}
    n_groups = 0
    for idx, (prompt_ids, gold) in enumerate(items):
        new_tokens = rollout(model, prompt_ids, grpo, rng)  # (G, T)
        # (G, S+T) full streams; response mask marks the sampled tokens.
        S = prompt_ids.shape[0]
        streams = np.concatenate([np.tile(prompt_ids, (grpo.group_size, 1)), new_tokens], axis=1)
        response_mask = np.zeros_like(streams)
        response_mask[:, S:] = 1

        rewards = np.zeros(grpo.group_size)
        formats = np.zeros(grpo.group_size)
        for g in range(grpo.group_size):
            text = decode(new_tokens[g])
            r, details = score_reward(text, gold)
            rewards[g] = r
            formats[g] = 1.0 if details["format_ok"] else 0.0
        adv = group_advantage(rewards, grpo.adv_eps)

        step_stats = grpo_step(model, reference, streams, response_mask, adv, optimizer, grpo)
        n_groups += 1
        stats["reward"] += float(rewards.mean())
        stats["format"] += float(formats.mean())
        stats["correct"] += float(np.mean(rewards >= 1.0))
        stats["kl"] += step_stats["kl"]
        stats["frac_clipped"] += step_stats["frac_clipped"]

        if log_every and idx % log_every == 0:
            logger.info(
                "grpo item %d/%d: reward=%.3f format=%.2f correct=%.2f kl=%.4f clipped=%.2f",
                idx + 1,
                len(items),
                rewards.mean(),
                formats.mean(),
                float(np.mean(rewards >= 1.0)),
                step_stats["kl"],
                step_stats["frac_clipped"],
            )
            for g in range(min(2, grpo.group_size)):
                logger.info("  [%d] reward=%.2f adv=%+.2f | %s", g, rewards[g], adv[g], decode(new_tokens[g])[:120])
    n = max(n_groups, 1)
    return {k: v / n for k, v in stats.items()}
