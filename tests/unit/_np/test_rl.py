# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Tests for the thinking-format contract and the GRPO math (ADR 0004).

Three layers:

1. ``shared.thinking`` — the reward parser's edge cases (the tags, the
   truncation, the reversed tags, the $/comma cleanup).
2. ``impl._np.rl`` — the GRPO formulas against finite differences: the
   weighted-CE gradient (the whole point: "CE backward, reweighted"),
   the k3 KL estimator's unbiasedness, the group advantage, the rollout
   determinism.
3. cross-track gradient parity on FIXED rollouts (no sampling in the
   parity path — the same trick as the drafter acceptance tests): the
   torch trainer's loss and gradient must match the NumPy reference on
   the same weights and the same fixed batch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from impl._np.model import NumPyModel
from impl._np.optimizer import AdamW
from impl._np.rl import (
    GRPOConfig,
    _sample,
    group_advantage,
    grpo_dlogits,
    grpo_step,
    kl_k3,
    logprobs_at,
    rollout,
)
from shared.config import TransformerConfig
from shared.constants import Keys
from shared.thinking import ANSWER_TAG, THINK_TAG, format_thinking_row, parse_completion, reward

# ── 1. the thinking-format contract ──────────────────────────────────────────


class TestRewardParser:
    def test_full_row_formats(self) -> None:
        row = format_thinking_row("How many?", "3 + 2 = 5", "5")
        assert THINK_TAG in row and ANSWER_TAG in row
        parsed = parse_completion(f"q {THINK_TAG} 3 + 2 = 5 {ANSWER_TAG} 5")
        assert parsed["format_ok"] is True
        assert parsed["answer"] == 5.0

    def test_correct_answer_full_reward(self) -> None:
        r, det = reward(f"{THINK_TAG} so 3+2=5 {ANSWER_TAG} 5", "5")
        assert r == pytest.approx(1.2) and det["correct"] is True

    def test_wrong_answer_format_only(self) -> None:
        r, det = reward(f"{THINK_TAG} so 3+2=5 {ANSWER_TAG} 6", "5")
        assert r == pytest.approx(0.2) and det["correct"] is False and det["format_ok"] is True

    def test_truncated_mid_think_zero(self) -> None:
        r, det = reward(f"{THINK_TAG} so 3+2=", "5")
        assert r == 0.0 and det["correct"] is False and det["format_ok"] is False

    def test_missing_think_tag_zero(self) -> None:
        r, _ = reward(f"{ANSWER_TAG} 5", "5")
        assert r == 0.0

    def test_reversed_tags_zero(self) -> None:
        r, _ = reward(f"{ANSWER_TAG} 5 {THINK_TAG} backwards", "5")
        assert r == 0.0

    def test_last_answer_tag_wins(self) -> None:
        # draft-and-revise inside think: the LAST answer tag is the commitment
        parsed = parse_completion(f"{THINK_TAG} hmm {ANSWER_TAG} 4 {ANSWER_TAG} 5")
        assert parsed["answer"] == 5.0

    def test_currency_and_commas_stripped(self) -> None:
        r, det = reward(f"{THINK_TAG} x {ANSWER_TAG} $1,234.", "1234")
        assert det["correct"] is True and r == pytest.approx(1.2)

    def test_nonnumeric_answer_format_zero(self) -> None:
        r, det = reward(f"{THINK_TAG} x {ANSWER_TAG} banana", "5")
        assert r == 0.0 and det["format_ok"] is False


# ── 2. the GRPO math (NumPy reference) ───────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_cfg() -> TransformerConfig:
    return TransformerConfig(vocab_size=32, embed_dim=16, n_layers=2, n_heads=2, context_length=48, seed=0)


@pytest.fixture()
def tiny_model(tiny_cfg: TransformerConfig) -> NumPyModel:
    return NumPyModel(tiny_cfg).to(np.float64)


class TestGroupAdvantage:
    def test_zero_sum_within_group(self) -> None:
        a = group_advantage(np.array([1.2, 0.0, 1.2, 0.2]))
        assert abs(a.sum()) < 1e-9

    def test_identical_rewards_stay_zero(self) -> None:
        # σ=0 → the guard keeps the advantage finite and zero
        a = group_advantage(np.array([0.2, 0.2, 0.2]))
        assert np.allclose(a, 0.0)

    def test_best_reward_gets_positive_advantage(self) -> None:
        a = group_advantage(np.array([1.2, 0.0, 0.2]))
        assert a[0] > 0 and a[1] < 0


class TestK3Estimator:
    def test_unbiased_against_exact_kl(self) -> None:
        rng = np.random.default_rng(0)
        lp = rng.normal(size=64)
        lp = np.exp(lp - lp.max())
        lp /= lp.sum()
        lp = np.log(lp)
        lr_ = lp + rng.normal(scale=0.2, size=64)
        lr_ = np.exp(lr_ - lr_.max())
        lr_ /= lr_.sum()
        lr_ = np.log(lr_)
        exact = float(np.sum(np.exp(lp) * (lp - lr_)))
        y = np.random.default_rng(1).choice(64, size=200_000, p=np.exp(lp))
        est = float(np.mean(kl_k3(lp[y][None, :], lr_[y][None, :])))
        assert est == pytest.approx(exact, abs=0.01)

    def test_zero_when_identical(self) -> None:
        lp = np.zeros((1, 5))
        assert np.allclose(kl_k3(lp, lp), 0.0)


class TestGRPOGradient:
    def test_weighted_ce_gradient_matches_finite_differences(self, tiny_model: NumPyModel) -> None:
        """The ADR 0004 statement, executable: the GRPO dlogits is the CE
        gradient core scaled per position — verified end-to-end through
        the shared backward chain against central differences."""
        rng = np.random.default_rng(0)
        B, S = 3, 7
        ids = rng.integers(0, 32, size=(B, S)).astype(np.int64)
        graded = rng.integers(0, 32, size=(B, S)).astype(np.int64)
        w = rng.normal(size=(B, S))
        mask = (rng.random((B, S)) < 0.8).astype(float)
        logits, trace = tiny_model.forward_with_trace(ids)
        # a PERTURBED reference — with the identical weights the k3 term
        # is 0 and both signs of its gradient pass vacuously
        ref_lp = logprobs_at(tiny_model.forward(ids), graded) + rng.normal(scale=0.3, size=(3, 7))
        beta = 0.04

        def loss() -> float:
            lg, _ = tiny_model.forward_with_trace(ids)
            plp = logprobs_at(lg, graded)
            return float((-w * plp * mask).sum() / mask.sum() + beta * (kl_k3(plp, ref_lp) * mask).sum() / mask.sum())

        dl = grpo_dlogits(logits, graded, w, mask, ref_logprobs=ref_lp, kl_beta=beta)
        grads = tiny_model.backward_from_dlogits(dl, trace, ids)
        params = tiny_model.get_all_parameters()
        eps = 1e-6
        for key in (
            Keys.lm_head(),
            Keys.embed(),
            Keys.attn(0, "q_proj"),
            Keys.ffn(0, "gate_proj"),
        ):
            arr = params[key]
            o = arr[0, 0]
            arr[0, 0] = o + eps
            hi = loss()
            arr[0, 0] = o - eps
            lo = loss()
            arr[0, 0] = o
            fd = (hi - lo) / (2 * eps)
            assert fd == pytest.approx(float(grads[key][0, 0]), abs=1e-5), key

    def test_moe_router_gradient_flows(self) -> None:
        cfg = TransformerConfig(
            vocab_size=32,
            embed_dim=16,
            n_layers=2,
            n_heads=2,
            context_length=48,
            seed=0,
            n_experts=4,
            n_shared_experts=1,
            expert_dim=8,
            top_k=1,
        )
        m = NumPyModel(cfg).to(np.float64)
        rng = np.random.default_rng(0)
        ids = rng.integers(0, 32, size=(2, 6)).astype(np.int64)
        graded = rng.integers(0, 32, size=(2, 6)).astype(np.int64)
        w = rng.normal(size=(2, 6))
        mask = np.ones((2, 6))
        logits, trace = m.forward_with_trace(ids)
        dl = grpo_dlogits(logits, graded, w, mask, kl_beta=0.0)
        grads = m.backward_from_dlogits(dl, trace, ids)
        router = Keys.moe_gate(0)
        assert router in grads  # the MoE branch sees the RL gradient
        assert np.isfinite(grads[router]).all()


class TestRollout:
    def test_shape_and_determinism(self, tiny_model: NumPyModel) -> None:
        g = GRPOConfig(group_size=4, max_new_tokens=10, top_k=5)
        prompt = np.array([1, 2, 3, 4], dtype=np.int64)
        t1 = rollout(tiny_model, prompt, g, np.random.default_rng(7))
        t2 = rollout(tiny_model, prompt, g, np.random.default_rng(7))
        assert t1.shape == (4, 10)
        assert np.array_equal(t1, t2)

    def test_sampler_follows_distribution(self) -> None:
        """One logits row, many draws: empirical frequencies track the
        softmax (and top-k keeps everything else out)."""
        rng = np.random.default_rng(3)
        logits = rng.normal(size=(1, 32))
        counts = np.zeros(32)
        for _ in range(2000):
            counts[_sample(logits, 1.0, 8, np.random.default_rng(_))] += 1
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        top8 = np.argsort(probs[0])[-8:]
        # all draws landed inside the top-8 support
        assert counts[~np.isin(np.arange(32), top8)].sum() == 0
        # and the top token dominates a soft-ish distribution
        assert counts[np.argmax(probs[0])] > counts[np.argmin(probs[0])]

    def test_respects_context_window(self, tiny_model: NumPyModel) -> None:
        g = GRPOConfig(group_size=2, max_new_tokens=100, top_k=5)
        prompt = np.arange(40, dtype=np.int64) % 32  # ctx=48 → ≤7 new tokens
        out = rollout(tiny_model, prompt, g, np.random.default_rng(1))
        assert out.shape[1] <= 7


class TestGRPOStep:
    def test_step_updates_weights_and_reports(self, tiny_model: NumPyModel, tiny_cfg: TransformerConfig) -> None:
        ref = NumPyModel(tiny_cfg, {k: v.copy() for k, v in tiny_model.get_all_parameters().items()}).to(np.float64)
        g = GRPOConfig(group_size=4, max_new_tokens=8, top_k=5)
        rng = np.random.default_rng(2)
        streams = rng.integers(0, 32, size=(4, 14)).astype(np.int64)
        rmask = np.zeros_like(streams)
        rmask[:, 6:] = 1
        adv = group_advantage(np.array([1.2, 0.0, 0.2, 0.0]))
        before = tiny_model.get_all_parameters()[Keys.lm_head()].copy()
        opt = AdamW(lr=1e-4)
        st = grpo_step(tiny_model, ref, streams, rmask, adv, opt, g)
        after = tiny_model.get_all_parameters()[Keys.lm_head()]
        assert not np.array_equal(before, after)
        assert st["n_valid"] > 0
        assert 0.0 <= st["frac_clipped"] <= 1.0

    def test_kl_penalty_pulls_toward_reference(self, tiny_model: NumPyModel, tiny_cfg: TransformerConfig) -> None:
        """A pure-KL step (β>0, advantage 0) decreases the k3 KL between
        policy and reference — the regularizer's sign is right."""
        ref = NumPyModel(tiny_cfg, {k: v.copy() for k, v in tiny_model.get_all_parameters().items()}).to(np.float64)
        # perturb the policy away from the reference
        params = tiny_model.get_all_parameters()
        params[Keys.lm_head()] += np.random.default_rng(5).normal(scale=0.05, size=params[Keys.lm_head()].shape)
        g = GRPOConfig(group_size=2, max_new_tokens=4, top_k=5, kl_beta=1.0)
        streams = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)
        rmask = np.zeros_like(streams)
        rmask[:, 3:] = 1
        opt = AdamW(lr=1e-2)

        def measure_kl() -> float:
            # mean k3 over the GRADED positions only (the masking
            # grpo_step applies — the -100 positions are excluded)
            logits, _ = tiny_model.forward_with_trace(streams)
            graded_ids = np.full_like(streams, -100)
            graded_ids[:, :-1] = streams[:, 1:]
            graded_mask = np.zeros_like(streams)
            graded_mask[:, :-1] = rmask[:, 1:]
            valid = (graded_mask == 1) & (graded_ids >= 0)
            gid = np.where(valid, graded_ids, 0)
            plp = logprobs_at(logits, gid)[valid]
            rlp = logprobs_at(ref.forward(streams), gid)[valid]
            return float(np.mean(kl_k3(plp, rlp)))

        kl_before = measure_kl()
        for _ in range(20):
            grpo_step(tiny_model, ref, streams, rmask, np.zeros(1), opt, g)
        assert measure_kl() < kl_before


# ── 3. cross-track parity on FIXED rollouts (no sampling) ────────────────────


class TestCrossTrackParity:
    def test_torch_loss_matches_numpy_reference(self) -> None:
        """Same weights, same fixed batch → the torch trainer's loss equals
        the NumPy reference formula (float64 tight)."""
        from impl._torch.layers import TorchModel
        from impl._torch.rl import TorchGRPOTrainer

        cfg = TransformerConfig(vocab_size=32, embed_dim=16, n_layers=2, n_heads=2, context_length=48, seed=0)
        tm = TorchModel(cfg)
        tr = TorchGRPOTrainer(tm, TorchModel(cfg), device="cpu", lr=1e-4, kl_beta=0.04)
        params_np = {k: v for k, v in tm.get_all_parameters().items()}
        ref_module = tr.reference
        assert isinstance(ref_module, torch.nn.Module)
        ref_np = {k: v for k, v in ref_module.get_all_parameters().items()}  # type: ignore[attr-defined]

        rng = np.random.default_rng(0)
        B, S = 4, 10
        batch = rng.integers(0, 32, size=(B, S)).astype(np.int64)
        rmask = np.zeros_like(batch)
        rmask[:, 5:] = 1
        adv = rng.normal(size=(B,))

        loss_t, stats = tr.grpo_loss(
            torch.from_numpy(batch), torch.from_numpy(rmask.astype(np.int64)), torch.from_numpy(adv)
        )

        nm = NumPyModel(cfg, {k: v.copy() for k, v in params_np.items()}).to(np.float64)
        refm = NumPyModel(cfg, {k: v.copy() for k, v in ref_np.items()}).to(np.float64)
        logits, _ = nm.forward_with_trace(batch)
        graded_mask = np.zeros((B, S))
        graded_mask[:, :-1] = rmask[:, 1:]
        graded_ids = np.full_like(batch, -100)
        graded_ids[:, :-1] = batch[:, 1:]
        valid = (graded_mask == 1) & (graded_ids >= 0)
        gid = np.where(valid, graded_ids, 0)
        gm = valid.astype(float)
        plp = logprobs_at(logits, gid)
        rlp = logprobs_at(refm.forward(batch), gid)
        w = adv[:, None] * gm
        k3 = np.exp(rlp - plp) - (rlp - plp) - 1
        loss_n = float((-(w * plp)[valid]).sum() / valid.sum() + 0.04 * k3[valid].sum() / valid.sum())

        assert float(loss_t.item()) == pytest.approx(loss_n, abs=1e-4)
        assert stats["n_valid"] == pytest.approx(float(valid.sum()))

    def test_torch_gradient_matches_numpy_reference(self) -> None:
        """The gradient path too: one step on both tracks from the same
        weights/batch — the lm_head grads must agree to float tolerance."""
        from impl._torch.layers import TorchModel
        from impl._torch.rl import TorchGRPOTrainer

        cfg = TransformerConfig(vocab_size=32, embed_dim=16, n_layers=2, n_heads=2, context_length=48, seed=0)
        rng = np.random.default_rng(0)
        B, S = 2, 8
        batch = rng.integers(0, 32, size=(B, S)).astype(np.int64)
        rmask = np.zeros_like(batch)
        rmask[:, 4:] = 1
        adv = np.array([1.0, -1.0])

        # torch: compute grads only (no optimizer state pollution)
        tm = TorchModel(cfg)
        tr = TorchGRPOTrainer(tm, TorchModel(cfg), device="cpu", lr=1e-4, kl_beta=0.0)
        params_np = {k: v for k, v in tm.get_all_parameters().items()}
        loss_t, _ = tr.grpo_loss(
            torch.from_numpy(batch), torch.from_numpy(rmask.astype(np.int64)), torch.from_numpy(adv)
        )
        loss_t.backward()
        # nn.Linear stores (out, in); the registry layout is (in, out) —
        # transpose for the comparison (the torch_transpose rule).
        assert tm.lm_head.weight.grad is not None  # backward populated it
        g_torch = tm.lm_head.weight.grad.detach().cpu().numpy().T

        # numpy: the same objective through the reference formulas
        nm = NumPyModel(cfg, {k: v.copy() for k, v in params_np.items()}).to(np.float64)
        logits, trace = nm.forward_with_trace(batch)
        graded_mask = np.zeros((B, S))
        graded_mask[:, :-1] = rmask[:, 1:]
        graded_ids = np.full_like(batch, -100)
        graded_ids[:, :-1] = batch[:, 1:]
        valid = (graded_mask == 1) & (graded_ids >= 0)
        gid = np.where(valid, graded_ids, 0)
        gm = valid.astype(float)
        token_w = adv[:, None] * gm
        dl = grpo_dlogits(logits, gid, token_w, gm, ref_logprobs=None, kl_beta=0.0)
        grads = nm.backward_from_dlogits(dl, trace, batch)
        g_np = grads[Keys.lm_head()]

        assert g_torch.shape == g_np.shape
        assert np.abs(g_torch - g_np).max() < 1e-5
