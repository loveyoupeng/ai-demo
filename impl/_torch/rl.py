# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""GRPO RL for the PyTorch track — the production idiom (ADR 0004).

Same GRPO math as the NumPy reference (``impl/_np/rl.py`` — read that
docstring for the derivations), expressed the framework way:

- no hand-rolled gradient assembly — the weighted-CE + KL objective is
  built from tensor ops and ``loss.backward()`` does the chain;
- no manual sampler — rollouts reuse ``shared.generator.TextGenerator``
  (the exact KV-step decode path all torch-family tracks already serve);
- no manual optimizer — ``torch.optim.AdamW`` + ``clip_grad_norm_``.

The loss mirrors ``grpo_dlogits`` exactly:

    L = mean_valid( −w · log π_θ(y_t|·) + β · k3(π_θ, π_ref) )

with w = clip(π_θ/π_old, 1±ε) · A (the group-relative advantage), the
k3 KL estimator, and the same next-token grading shift as SFT (position
t's logit scores token t+1 — the response mask shifts onto the graded
logit positions identically).
"""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
import torch.nn.functional as F

from shared.generator import TextGenerator

logger = logging.getLogger(__name__)


class TorchGRPOTrainer:
    """The PyTorch GRPO trainer (the Triton track subclasses this over its
    own kernel-backed stack — the SFT convention: one trainer family,
    model injected)."""

    IGNORE_INDEX = -100

    def __init__(
        self,
        model: torch.nn.Module,
        reference: torch.nn.Module,
        device: str = "cpu",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        group_size: int = 8,
        temperature: float = 1.0,
        top_k: int = 40,
        max_new_tokens: int = 80,
        clip_eps: float = 0.2,
        kl_beta: float = 0.04,
    ) -> None:
        self.model = model.to(device)
        self.reference = reference.to(device).eval()
        for p in self.reference.parameters():
            p.requires_grad_(False)
        self.device = device
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        # The rollout sampler: same KV-step decode as every track's
        # generation (no grad — sampling is data collection). ``cast``:
        # the protocol is structural and TorchModel satisfies it, but
        # pyright cannot see through the nn.Module annotation above.
        self._gen = TextGenerator(
            cast(Any, model),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    @torch.no_grad()
    def rollout(self, prompt: torch.Tensor, generator=None) -> torch.Tensor:
        """Sample ``group_size`` completions for one (1, S) prompt.

        ONE batched decode over G rows (the generator's KV-step path is
        batched already — the rows never interact), not G sequential
        decodes: same distribution, a fraction of the launch overhead.

        Returns: (G, S+T) — the G full streams (prompt + sampled tokens).
        """
        gen = generator or self._gen
        rows = prompt.expand(self.group_size, -1)  # (G, S) — a read-only view
        return gen.generate(rows)  # (G, S+T)

    def grpo_loss(
        self,
        batch_ids: torch.Tensor,
        response_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_logprobs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """The GRPO objective on a fixed batch of scored rollouts.

        batch_ids : (B, S) — prompt + sampled tokens (teacher-forced).
        response_mask : (B, S) — 1 on the SAMPLED tokens; the graded
            logit positions are the mask shifted right by one (position
            t's logit scores token t+1 — the SFT shift).
        advantages : (B,) — group-relative advantage per rollout.
        old_logprobs : (B, S) or None — log π_old at graded positions;
            None → ratio 1 (fully on-policy).

        Returns (loss, diagnostics) — the loss is a scalar tensor ready
        for ``.backward()``; diagnostics carry the teaching log numbers.
        """
        logits = self.model(batch_ids)  # (B, S, V)
        B, S = batch_ids.shape
        graded_ids = torch.full_like(batch_ids, self.IGNORE_INDEX)
        graded_ids[:, :-1] = batch_ids[:, 1:]  # what position t must predict
        graded_mask = torch.zeros((B, S), dtype=batch_ids.dtype, device=batch_ids.device)
        graded_mask[:, :-1] = response_mask[:, 1:]  # token i graded by logits i−1
        valid = (graded_mask == 1) & (graded_ids >= 0)

        # log π_θ(y_t|·) at every graded position — gather on log_softmax.
        logp = F.log_softmax(logits, dim=-1)  # (B, S, V)
        policy_lp = logp.gather(-1, graded_ids.clamp(min=0).unsqueeze(-1)).squeeze(-1)  # (B, S)

        # Importance ratio (1 when on-policy) with ε-clip.
        ratio = torch.ones_like(policy_lp) if old_logprobs is None else torch.exp(policy_lp - old_logprobs)
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        token_weights = clipped * advantages.reshape(B, 1) * graded_mask  # (B, S)

        # k3 KL estimator vs the frozen reference (0 contribution when β=0).
        if self.kl_beta > 0.0:
            with torch.no_grad():
                ref_lp = (
                    F.log_softmax(self.reference(batch_ids), dim=-1)
                    .gather(-1, graded_ids.clamp(min=0).unsqueeze(-1))
                    .squeeze(-1)
                )
            delta = ref_lp - policy_lp
            k3 = torch.exp(delta) - delta - 1.0
        else:
            k3 = torch.zeros_like(policy_lp)
            ref_lp = None

        # L = mean over graded positions of (−w·logπ + β·k3).
        n_valid = valid.sum()
        if n_valid == 0:
            z = logits.sum() * 0.0
            return z, {"kl": 0.0, "frac_clipped": 0.0, "n_valid": 0.0}
        loss = (-(token_weights * policy_lp)[valid].sum() + self.kl_beta * (k3[valid]).sum()) / n_valid

        with torch.no_grad():
            kl = float(k3[valid].mean()) if ref_lp is not None else 0.0
            frac = float(((ratio - 1.0).abs() > self.clip_eps)[valid].float().mean())
        return loss, {"kl": kl, "frac_clipped": frac, "n_valid": float(n_valid)}

    def train_step(
        self,
        batch_ids: torch.Tensor,
        response_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_logprobs: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """One optimizer step on one batch of scored rollouts."""
        loss, stats = self.grpo_loss(batch_ids, response_mask, advantages, old_logprobs)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {"loss": float(loss.item()), **stats}
