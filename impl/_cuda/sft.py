"""SFT trainer for the CUDA track — NVRTC kernels + torch autograd glue.

The CUDA track has no ``torch.autograd.Function`` wrapper around its kernels —
the forward pass runs on CUDA via NVRTC, and the backward pass lives in
``torch`` because the kernels *return plain Tensors* (the gradients get back
through normal autograd on those tensors only if you hold the graph).  We
don't: the compute graph is built per-call in ``forward``.  So this trainer
uses SGD-style moment via torch's ``optim.AdamW`` *on the ``parameters``* —
that is, we collect the tensor references from the model's registry binding
and treat them as the parameter set.

NumPy        — the teaching baseline: mask the target, same loss.
PyTorch      — the framework idiom: `(loss).backward()`.
Triton/CUDA  — kernels for compute, autograd between them.

Same masking contract as the other tracks (targets == -100 at prompt
positions).

Usage::

    trainer = CudaSFTTrainer(model, lr=1e-3)
    loss = trainer.train_step(batch)  # batch: input_ids, target_ids, mask
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from impl._cuda.model import CUDAModel


class CudaSFTTrainer:
    """CUDA SFT trainer: CUDA forward, torch-autograd between kernels."""

    IGNORE_INDEX = -100

    def __init__(self, model: CUDAModel, lr: float = 1e-3, weight_decay: float = 0.01) -> None:
        self.model = model
        # CUDAModel is not an nn.Module — collect its tensor references
        # the same way save/load/io does (via its registry binding).
        params = model._param_tensors()
        self._param_refs = [p for p in params.values() if p.is_floating_point()]
        for p in self._param_refs:  # attach autograd so backward() has somewhere to flow
            p.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(self._param_refs, lr=lr, weight_decay=weight_decay)

    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor, response_mask: torch.Tensor) -> float:
        """One step: forward (CUDA kernels) → masked CE → backward (torch) → AdamW step.

        The model's `forward` is what runs the CUDA kernels; the gradients it
        lands on are the ones attached to the parameter tensors we registered
        above.  ``grad`` accumulation happens on the tensors themselves — we
        just zero it out before each step.
        """
        for p in self._param_refs:
            p.grad = None

        logits = self.model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1).to(logits.device),
            ignore_index=self.IGNORE_INDEX,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._param_refs, max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return float(loss.item())
