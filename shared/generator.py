"""TextGenerator — the shared autoregressive generator for the torch family.

Track intent: **one deep generator behind one interface** — the torch,
triton, and cuda tracks were three shallow copies (identical sampling math,
~370 lines each); this module is the single implementation, and each track
keeps a thin adapter only where it genuinely varies (device placement,
cache access).

The generator consumes any model exposing the KV-step interface (the
same one the NumPy track publicizes): ``make_cache(batch)``,
``forward_prefill(input_ids) -> (logits, cache)``, and
``forward_step(input_ids, position, cache) -> logits``. Generation is
O(1) per token — only the new token's K/V are computed.

Interface
---------
``generate(prompt)`` dispatches on temperature; ``generate_greedy`` and
``generate_sampled(prompt, temperature)`` are the two decoding modes.
Sampling math (temperature scaling, top-k mask, stable softmax, entropy
logging) lives inside — callers and tests cross the same seam.

Logging (logger ``shared.generator``):
    - INFO:  generation mode, first/last token, completion
    - DEBUG: per-token logits (top-5), temperature scaling, top-k masking,
      softmax entropy, sampled token probabilities
"""

from __future__ import annotations

import logging
from typing import Protocol

import torch

from shared.constants import REP_PENALTY

logger = logging.getLogger(__name__)


class _StepModel(Protocol):
    """The KV-step interface the generator consumes (all torch-family tracks).

    make_cache(batch_size) — an empty per-layer KV cache (one dict per
        layer; batch_size rows).
    forward_prefill(input_ids (B, S) int, cache?) — one full-sequence
        pass; fills the cache; returns (logits (B, S, V), cache).
    forward_step(input_ids (B, 1) int, position int, cache) — ONE new
        token against the cached K/V; returns logits (B, 1, V)."""

    def make_cache(self, batch_size: int) -> list[dict[str, torch.Tensor]]: ...

    def forward_prefill(
        self, input_ids: torch.Tensor, cache: list[dict[str, torch.Tensor]] | None = None
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]: ...

    def forward_step(
        self, input_ids: torch.Tensor, position: int, cache: list[dict[str, torch.Tensor]]
    ) -> torch.Tensor: ...


def _validate_prompt(prompt: torch.Tensor) -> torch.Tensor:
    """Assert the prompt is a 2-D int tensor; return it unchanged."""
    if prompt.dim() != 2:
        raise ValueError(f"prompt must be 2-D (batch, seq_len), got {prompt.dim()}-D")
    if prompt.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"prompt must be integer token IDs (int32/int64), got {prompt.dtype}")
    return prompt


def _apply_top_k_mask(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep only the k largest logits per row; set the rest to -inf.

    logits: (B, V) → (B, V). top_k <= 0 or >= V returns logits unchanged.
    """
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]  # (B, 1)
    return torch.where(logits >= kth, logits, torch.full_like(logits, float("-inf")))


def _compute_entropy(probs: torch.Tensor) -> float:
    """Shannon entropy of a batch-averaged distribution: -Σ p·log(p)."""
    p = probs.mean(dim=0).clamp(min=1e-12)
    return float(-(p * p.log()).sum().item())


def _top_k_values(probs: torch.Tensor, k: int) -> list[str]:
    """Top-k probabilities of a batch-averaged distribution, as strings."""
    p = probs.mean(dim=0)
    top = torch.topk(p, min(k, p.shape[-1]), dim=-1).values
    return [f"{v:.4f}" for v in top.tolist()]


class TextGenerator:
    """Autoregressive text generation over the KV-step interface.

    Deep module: the sampling math, temperature/top-k handling, cache
    threading, and educational logging all live behind one small
    interface (``generate`` / ``generate_greedy`` / ``generate_sampled``).

    Parameters
    ----------
    model : the KV-step interface (TorchModel, TritonModel, or CUDAModel)
    max_new_tokens : maximum tokens to generate after the prompt
    temperature : sampling temperature (0.0 = greedy/argmax)
    top_k : top-k filtering before softmax (0 = off)
    """

    def __init__(
        self,
        model: _StepModel,
        max_new_tokens: int = 20,
        temperature: float = 0.0,
        top_k: int = 0,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k

    def generate(self, prompt: torch.Tensor) -> torch.Tensor:
        """Generate tokens autoregressively (dispatch on temperature).

        prompt: (B, S) int token IDs → (B, S + max_new_tokens).
        """
        prompt = _validate_prompt(prompt)
        mode = "greedy" if self.temperature == 0.0 else f"sampled_T={self.temperature:.3f}"
        if self.top_k > 0:
            mode = f"top_k={self.top_k}_" + mode
        logger.info(
            "TextGenerator.generate() mode=%s batch_size=%d prompt_len=%d max_new=%d",
            mode,
            prompt.shape[0],
            prompt.shape[1],
            self.max_new_tokens,
        )
        if self.temperature == 0.0:
            return self.generate_greedy(prompt)
        return self.generate_sampled(prompt, self.temperature)

    # ── the decode loop: one implementation, both modes ──────────────────
    def _decode(self, prompt: torch.Tensor, temperature: float) -> torch.Tensor:
        """Prefill the prompt, then step one token at a time (O(1) per token).

        This is the exact KV-step path the NumPy track publicizes: only the
        new token's K/V are computed per step; attention runs against the
        cached K/V. Mirrors ``impl._np.inference.TextGenerator``.
        """
        batch_size, prompt_len = prompt.shape
        sequence = prompt.clone()
        cache = self.model.make_cache(batch_size)

        logits, cache = self.model.forward_prefill(sequence)
        # Tracks may relocate inputs (the CUDA track owns device placement) —
        # the sequence follows the model's emitted logits so torch.cat keeps
        # one device downstream.
        sequence = sequence.to(logits.device)
        next_logits = logits[:, -1, :]  # (B, V)

        # Repetition guard — same rule as every other sampler in the repo
        # (server _sample, NumPy/Torch record _pick): penalty on tokens
        # already emitted in this response + a hard block on the immediately
        # previous token. Without it a tiny LM degenerates into repeat
        # loops; with it, all four tracks decode equivalently.
        emitted = torch.zeros((batch_size, 0), dtype=torch.long, device=next_logits.device)

        for step in range(self.max_new_tokens):
            guarded = next_logits.clone()
            for b in range(batch_size):
                if emitted.shape[1] > 0:
                    seen = emitted[b].unique()
                    for tid in seen.tolist():
                        v = guarded[b, tid]
                        guarded[b, tid] = v / REP_PENALTY if v > 0 else v * REP_PENALTY
                    guarded[b, int(emitted[b, -1].item())] = -float("inf")  # never immediately repeat

            if temperature == 0.0:
                # Greedy: argmax picks the highest-logit token
                next_token = torch.argmax(guarded, dim=-1)  # (B,)
                probs = None
            else:
                # Temperature scaling: softmax(z/T)
                scaled = guarded / max(temperature, 1e-8)
                # Top-k filtering: keep only the top-k logits
                if self.top_k > 0:
                    scaled = _apply_top_k_mask(scaled, self.top_k)
                # Stable softmax: subtract max per row to avoid overflow
                logits_max = torch.max(scaled, dim=-1, keepdim=True).values
                exp_logits = torch.exp(scaled - logits_max)
                probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)  # (B, V)
                next_token = torch.stack(
                    [torch.multinomial(probs[b].float(), num_samples=1) for b in range(batch_size)]
                ).squeeze(-1)  # (B,)
            emitted = torch.cat([emitted, next_token.reshape(batch_size, 1)], dim=1)

            # Educational logging (first/last step, or short generations)
            if step == 0 or step == self.max_new_tokens - 1 or self.max_new_tokens <= 5:
                if probs is None:
                    top_idx = torch.argsort(next_logits[0], descending=True)[:5]
                    logger.debug(
                        "decode() step=%d top5_logits=%s",
                        step + 1,
                        [f"{v:.4f}" for v in next_logits[0, top_idx].tolist()],
                    )
                else:
                    logger.debug(
                        "decode() step=%d entropy=%.4f probs_top5=%s",
                        step + 1,
                        _compute_entropy(probs),
                        _top_k_values(probs, 5),
                    )

            # Append the sampled token and step (the new token's K/V only)
            sequence = torch.cat([sequence, next_token.reshape(batch_size, 1)], dim=1)
            if step < self.max_new_tokens - 1:
                # (B, 1, V) → (B, V): the next step's logits
                next_logits = self.model.forward_step(
                    next_token.reshape(batch_size, 1), prompt_len + step, cache
                ).squeeze(1)

        logger.info(
            "decode() complete batch_size=%d final_len=%d",
            batch_size,
            sequence.shape[1],
        )
        return sequence

    def generate_greedy(self, prompt: torch.Tensor) -> torch.Tensor:
        """Generate using greedy decoding (argmax). Deterministic.

        prompt: (B, S) int token IDs → (B, S + max_new_tokens).
        """
        prompt = _validate_prompt(prompt)
        logger.info("TextGenerator.generate_greedy() batch_size=%d", prompt.shape[0])
        return self._decode(prompt, temperature=0.0)

    def generate_sampled(self, prompt: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample from the (temperature-scaled, top-k filtered) softmax.

        prompt: (B, S) int token IDs → (B, S + max_new_tokens).
        """
        prompt = _validate_prompt(prompt)
        logger.info("TextGenerator.generate_sampled() T=%.3f batch_size=%d", temperature, prompt.shape[0])
        return self._decode(prompt, temperature=temperature)
