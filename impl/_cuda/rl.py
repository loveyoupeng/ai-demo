# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""GRPO RL for the CUDA track — NVRTC kernels + torch autograd glue.

Same GRPO objective as the torch trainer (``impl/_torch/rl.py``; the
derivations live in ``impl/_np/rl.py``). The CUDA track's constraint:
``CUDAModel`` is not an
``nn.Module`` — the forward runs NVRTC kernels that return plain
tensors, so the optimizer and the autograd attachments are built from
the model's registry binding (``_param_tensors``).

The loss is assembled here with the same tensor ops as the torch
trainer (gather on log_softmax, ε-clipped ratio, k3 estimator); only
the plumbing (parameter collection, device handling, the ``forward``
entry) follows the CUDA conventions.

Usage::

    trainer = CudaGRPOTrainer(model, reference, lr=1e-4)
    stats = trainer.train_step(batch_ids, response_mask, advantages)
    # batch_ids (B, S) int64; response_mask (B, S) int64; advantages (B,)
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from impl._cuda.model import CUDAModel

logger = logging.getLogger(__name__)


class CudaGRPOTrainer:
    """CUDA-backed GRPO: NVRTC forward, torch-autograd between kernels."""

    IGNORE_INDEX = -100

    def __init__(
        self,
        model: CUDAModel,
        reference: CUDAModel,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        clip_eps: float = 0.2,
        kl_beta: float = 0.04,
    ) -> None:
        self.model = model
        self.reference = reference  # frozen by construction: never in an optimizer
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
        # CUDA convention: collect the parameter tensors from the
        # registry binding and attach autograd — that is the trainer's
        # parameter set. The reference's tensors simply never get grads.
        params = model._param_tensors()
        self._param_refs = [p for p in params.values() if p.is_floating_point()]
        for p in self._param_refs:
            p.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(self._param_refs, lr=lr, weight_decay=weight_decay)

    def grpo_loss(
        self,
        batch_ids: torch.Tensor,
        response_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_logprobs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """The GRPO objective on a fixed batch of scored rollouts.

        Identical formula to ``TorchGRPOTrainer.grpo_loss`` (see that
        docstring); the CUDA model's ``forward`` runs the kernels and
        returns the (B, S, V) logits the loss reads.
        """
        device = self.model.device
        batch_ids = batch_ids.to(device)
        response_mask = response_mask.to(device)
        advantages = advantages.to(device)
        logits = self.model(batch_ids)  # (B, S, V) — NVRTC kernels
        B, S = batch_ids.shape
        graded_ids = torch.full_like(batch_ids, self.IGNORE_INDEX)
        graded_ids[:, :-1] = batch_ids[:, 1:]
        graded_mask = torch.zeros((B, S), dtype=batch_ids.dtype, device=device)
        graded_mask[:, :-1] = response_mask[:, 1:]
        valid = (graded_mask == 1) & (graded_ids >= 0)

        logp = F.log_softmax(logits, dim=-1)
        policy_lp = logp.gather(-1, graded_ids.clamp(min=0).unsqueeze(-1)).squeeze(-1)

        ratio = torch.ones_like(policy_lp) if old_logprobs is None else torch.exp(policy_lp - old_logprobs)
        clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        token_weights = clipped * advantages.reshape(B, 1) * graded_mask

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

        n_valid = valid.sum()
        if n_valid == 0:
            return logits.sum() * 0.0, {"kl": 0.0, "frac_clipped": 0.0, "n_valid": 0.0}
        loss = (-(token_weights * policy_lp)[valid].sum() + self.kl_beta * k3[valid].sum()) / n_valid

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
        """One step: clear grads → CUDA forward → GRPO loss → backward → AdamW."""
        for p in self._param_refs:
            p.grad = None
        loss, stats = self.grpo_loss(batch_ids, response_mask, advantages, old_logprobs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._param_refs, max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {"loss": float(loss.item()), **stats}
