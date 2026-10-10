"""Autoregressive inference engine for a decoder-only transformer.

Implements TextGenerator with greedy decoding, temperature-sampled decoding,
top-k filtering, and batch processing support.

Logging
-------
- INFO:  first/last token generation, temperature/sample mode
- DEBUG: per-token logits (top-5), temperature scaling, top-k masking, softmax probs

Architecture
-------------
Generation loop (KV-cached, O(1) attention work per new token):
    cache = model.make_cache(B, quantize=quantize)
    for i in range(prompt_len - 1):
        model.forward_step(sequence[:, [i]], i, cache)     # thread the prompt
    for step in range(max_new_tokens):
        step_logits = model.forward_step(sequence[:, [-1]], len-1, cache)  # (B, 1, V)
        if sampled:
            probs = softmax(step_logits / temperature)     # (B, V)
            if top_k > 0: logits = top_k_filter(logits, top_k)
            token = rng.choice(V, p=probs)                 # sample
        else:
            token = argmax(step_logits, axis=-1)           # greedy
        sequence = concat(sequence, token)                 # (B, S+1)

Only the newest token is forwarded each step; the cached K/V of all earlier
tokens is reused. (``model.forward_prefill`` can fill the same cache from a
whole prompt in one pass — equivalent to the per-token threading above.)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from shared.constants import REP_PENALTY

if TYPE_CHECKING:
    from impl._np.model import NumPyModel

logger = logging.getLogger(__name__)


def _apply_rep_guard(step_logits: np.ndarray, emitted: list[list[int]]) -> np.ndarray:
    """Per-row repetition guard (in place): penalty on already-emitted tokens
    + a hard block on the immediately previous token.

    Same rule as every other sampler in the repo — server ``_sample``, record
    ``_pick``, shared ``generator.py`` — so all four tracks decode equivalently.
    Rows are independent sequences, so the guard is applied per row.
    """
    for b in range(step_logits.shape[0]):
        for tid in set(emitted[b]):
            if 0 <= tid < step_logits.shape[-1]:
                v = step_logits[b, tid]
                step_logits[b, tid] = v / REP_PENALTY if v > 0 else v * REP_PENALTY
        if emitted[b]:
            step_logits[b, emitted[b][-1]] = -np.inf  # never immediately repeat
    return step_logits


class TextGenerator:
    """Autoregressive text generation for a decoder-only transformer.

    Logs per-token generation progress, sampling statistics, and
    model output characteristics when DEBUG/TRACE level is enabled.
    """

    def __init__(
        self,
        model: NumPyModel,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        quantize: bool = False,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.quantize = quantize

    def generate(self, prompt: np.ndarray) -> np.ndarray:
        """Generate tokens autoregressively.

        Logs the generation mode (greedy vs sampled) and batch context.

        Parameters
        ----------
        prompt : np.ndarray, shape (batch_size, prompt_len)
            Initial token IDs to start from.

        Returns
        -------
        output : np.ndarray, shape (batch_size, prompt_len + new_tokens)
            Full sequence including the original prompt.

        """
        prompt = self._validate_prompt(prompt)
        batch_size, prompt_len = prompt.shape
        mode = "greedy" if self.temperature == 0.0 else f"sampled_T={self.temperature:.3f}"

        if self.top_k > 0:
            mode = f"top_k={self.top_k}_" + mode

        logger.info(
            "TextGenerator.generate() mode=%s batch_size=%d prompt_len=%d max_new_tokens=%d",
            mode,
            batch_size,
            prompt_len,
            self.max_new_tokens,
        )

        if self.temperature == 0.0:
            return self.generate_greedy(prompt)
        return self.generate_sampled(prompt, self.temperature)

    def generate_greedy(self, prompt: np.ndarray) -> np.ndarray:
        """Generate using greedy decoding (argmax).

        Logs each token selection for traceability.

        Parameters
        ----------
        prompt : np.ndarray, shape (batch_size, prompt_len)
            Initial token IDs.

        Returns
        -------
        output : np.ndarray, shape (batch_size, prompt_len + new_tokens)
            Generated sequence including prompt.

        """
        prompt = self._validate_prompt(prompt)
        batch_size, seq_len = prompt.shape
        sequence = prompt.copy()

        # Build the KV cache by threading the prompt (all but the last token)
        # one token at a time, then step the last token to get the first
        # generated logits. Each subsequent step appends one new token.
        # quantize=True uses the 1-bit TurboQuant cache (lossy; int8 storage
        # ≈4x smaller than the float cache);
        # quantize=False uses the full-precision naive cache (exact).
        cache = self.model.make_cache(batch_size, quantize=self.quantize)

        for i in range(seq_len - 1):
            self.model.forward_step(sequence[:, [i]], i, cache, quantize=self.quantize)

        # Per-row emitted tokens (repeat guard, cf. shared/generator.py —
        # rows are independent sequences, so the guard must be per row).
        emitted = [[] for _ in range(sequence.shape[0])]

        for step in range(self.max_new_tokens):
            step_logits = self.model.forward_step(
                sequence[:, [-1]], sequence.shape[1] - 1, cache, quantize=self.quantize
            )  # (B, 1, V)
            step_logits = step_logits[:, 0, :].astype(np.float64)  # (B, V)

            # Repetition guard — same rule as every other sampler in the repo
            # (server _sample, record _pick, shared generator).
            step_logits = _apply_rep_guard(step_logits, emitted)

            next_token = np.argmax(step_logits, axis=-1)  # (B,)
            for b in range(sequence.shape[0]):
                emitted[b].append(int(next_token[b]))

            # Log top-5 tokens for traceability on first/last step
            if step == 0 or step == self.max_new_tokens - 1 or self.max_new_tokens <= 5:
                top_idx = np.argsort(step_logits[0], axis=-1)[::-1][:5]
                top_log = step_logits[0, top_idx]
                logger.debug(
                    "generate_greedy() step=%d batch=0 top5_tokens=%s top5_logits=%s",
                    step + 1,
                    top_idx.tolist(),
                    [f"{v:.4f}" for v in top_log.tolist()],
                )

            next_token_2d = next_token.reshape(batch_size, 1)
            sequence = np.concatenate([sequence, next_token_2d], axis=1)

            if step == 0 or step == self.max_new_tokens - 1:
                logger.info(
                    "generate_greedy() step=%d/%d token=%s new_len=%d",
                    step + 1,
                    self.max_new_tokens,
                    next_token.tolist(),
                    sequence.shape[1],
                )

        logger.info(
            "generate_greedy() complete batch_size=%d final_len=%d",
            batch_size,
            sequence.shape[1],
        )
        return sequence

    def generate_sampled(self, prompt: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Generate using temperature-sampled token selection.

        Logs temperature scaling, top-k filtering, softmax distribution,
        and sampled tokens for reproducibility.

        Parameters
        ----------
        prompt : np.ndarray, shape (batch_size, prompt_len)
            Initial token IDs.
        temperature : float
            Sampling temperature. Lower = more confident predictions.
            0.0 falls back to greedy decoding.

        Returns
        -------
        output : np.ndarray, shape (batch_size, prompt_len + new_tokens)
            Generated sequence including prompt.

        """
        prompt = self._validate_prompt(prompt)
        batch_size, seq_len = prompt.shape
        effective_temperature = max(temperature, 1e-8)

        logger.info(
            "TextGenerator.generate_sampled() batch_size=%d prompt_len=%d temperature=%.4f top_k=%d",
            batch_size,
            seq_len,
            temperature,
            self.top_k,
        )

        logger.debug("generate_sampled() effective_temperature=%.6f", effective_temperature)

        sequence = prompt.copy()
        rng = np.random.default_rng(self.model.config.seed)

        # Build the KV cache by threading the prompt (all but the last token)
        # one token at a time, then step the last token to get the first
        # generated logits. Each subsequent step appends one new token.
        # quantize=True uses the 1-bit TurboQuant cache (lossy; int8 storage
        # ≈4x smaller than the float cache);
        # quantize=False uses the full-precision naive cache (exact).
        cache = self.model.make_cache(batch_size, quantize=self.quantize)

        for i in range(seq_len - 1):
            self.model.forward_step(sequence[:, [i]], i, cache, quantize=self.quantize)

        # Per-row emitted tokens (repeat guard; rows are independent).
        emitted = [[] for _ in range(sequence.shape[0])]

        for step in range(self.max_new_tokens):
            step_logits = self.model.forward_step(
                sequence[:, [-1]], sequence.shape[1] - 1, cache, quantize=self.quantize
            )  # (B, 1, V)
            step_logits = step_logits[:, 0, :].astype(np.float64)  # (B, V)

            # Repetition guard — same rule as every other sampler in the repo.
            step_logits = _apply_rep_guard(step_logits, emitted)

            scaled_logits = step_logits / effective_temperature
            logger.debug(
                "generate_sampled() step=%d scaled_logits_batch0=%s",
                step + 1,
                [f"{v:.4f}" for v in scaled_logits[0].tolist()],
            )

            if self.top_k > 0:
                scaled_logits = self._apply_top_k_mask(scaled_logits, self.top_k)
                logger.debug("generate_sampled() step=%d top_k_masked (top_k=%d)", step + 1, self.top_k)

            # Stable softmax
            logits_max = np.max(scaled_logits, axis=-1, keepdims=True)
            exp_logits = np.exp(scaled_logits - logits_max)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            logger.debug(
                "generate_sampled() step=%d probs_entropy=%.4f probs_top5=%s",
                step + 1,
                self._compute_entropy(probs),
                self._top_k_values(probs, 5),
            )

            if batch_size == 1:
                # Wrap in an array so downstream batched indexing is uniform.
                next_token = np.array([rng.choice(self.model.vocab_size, p=probs[0])])
            else:
                next_token = np.array([rng.choice(self.model.vocab_size, p=probs[b]) for b in range(batch_size)])
            for b in range(sequence.shape[0]):
                emitted[b].append(int(next_token[b]))

            if step == 0 or step == self.max_new_tokens - 1 or self.max_new_tokens <= 5:
                # Use np.asarray to ensure proper numpy array type
                nt_array = np.asarray(next_token)
                selected_probs = [f"{probs[b, int(nt_array[b])]:.4f}" for b in range(batch_size)]
                logger.info(
                    "generate_sampled() step=%d/%d sampled=%s selected_probs=%s",
                    step + 1,
                    self.max_new_tokens,
                    list(nt_array),
                    selected_probs,
                )

            next_token_2d = np.asarray(next_token).reshape(batch_size, 1)
            sequence = np.concatenate([sequence, next_token_2d], axis=1)

        logger.info(
            "generate_sampled() complete batch_size=%d final_len=%d",
            batch_size,
            sequence.shape[1],
        )
        return sequence

    def _validate_prompt(self, prompt: np.ndarray) -> np.ndarray:
        """Validate and normalize a prompt to expected 2D int32 format."""
        prompt = np.asarray(prompt, dtype=np.int32)
        if prompt.ndim == 1:
            prompt = prompt.reshape(1, -1)
        elif prompt.ndim != 2:
            raise ValueError(f"Prompt must be 1D or 2D, got {prompt.ndim}D with shape {prompt.shape}")
        return prompt

    @staticmethod
    def _apply_top_k_mask(logits: np.ndarray, top_k: int) -> np.ndarray:
        """Mask all logits below the top-k values to minus infinity.

        logits: (B, V) raw logits (one row per sequence); top_k: keep the
            k largest per row (1 <= top_k <= V). Returns the masked
            (B, V) logits — the sampler's constrained distribution."""
        batch_size, vocab_size = logits.shape
        sorted_indices = np.argsort(logits, axis=-1)[:, ::-1]
        kth_indices = sorted_indices[:, top_k - 1 : top_k]
        kth_values = np.take_along_axis(logits, kth_indices, axis=-1)
        masked = np.where(logits >= kth_values, logits, -np.inf)
        return masked

    @staticmethod
    def _compute_entropy(probs: np.ndarray) -> float:
        """Compute mean entropy of a probability distribution.

        probs: (B, V) or (V,) a probability distribution (rows sum to 1).
        Entropy = -sum(p * log(p)) — low = peaked (confident), high = uniform (uncertain).
        """
        safe_probs = np.clip(probs, 1e-10, 1.0)
        if probs.ndim == 1:
            return float(-np.sum(safe_probs * np.log(safe_probs)))
        return float(-np.mean(np.sum(safe_probs * np.log(safe_probs), axis=-1)))

    @staticmethod
    def _top_k_values(probs: np.ndarray, k: int) -> list[str]:
        """Top-k probabilities of batch 0's distribution, as strings.

        probs: (B, V) probabilities (rows sum to 1); k: how many of the
        largest to report."""
        """Get top-k probability values as formatted strings for batch 0."""
        top_idx = np.argsort(probs[0], axis=-1)[::-1][:k]
        return [f"{probs[0, i]:.4f}" for i in top_idx]
