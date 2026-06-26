"""Autoregressive inference engine for a decoder-only transformer.

Implements TextGenerator with greedy decoding, temperature-sampled decoding,
top-k filtering, and batch processing support.

Logging
-------
- INFO:  first/last token generation, temperature/sample mode
- DEBUG: per-token logits (top-5), temperature scaling, top-k masking, softmax probs
- TRACE: per-token softmax distribution, RNG seed state

Architecture
-------------
Generation loop:
    for step in range(max_new_tokens):
        logits   = model.forward(sequence)          # (B, S, V)
        step_log = logits[:, -1, :]                 # (B, V)
        if sampled:
            probs = softmax(step_log / temperature)  # (B, V)
            if top_k > 0: probs = top_k_filter(probs, top_k)
            token = np.random.choice(V, p=probs)     # sample
        else:
            token = np.argmax(step_log, axis=-1)     # greedy
        sequence = concat(sequence, token)           # (B, S+1)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from impl._np.model import NumPyModel

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k

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
        logger.info("TextGenerator.generate_greedy() batch_size=%d prompt_len=%d", batch_size, seq_len)

        sequence = prompt.copy()

        for step in range(self.max_new_tokens):
            logits = self.model.forward(sequence)  # (B, S, V)
            step_logits = logits[:, -1, :]  # (B, V)

            next_token = np.argmax(step_logits, axis=-1)  # (B,)

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
        rng = np.random.default_rng(self.model.seed)

        for step in range(self.max_new_tokens):
            logits = self.model.forward(sequence)  # (B, S, V)
            step_logits = logits[:, -1, :]  # (B, V)

            scaled_logits = step_logits / effective_temperature
            logger.debug("generate_sampled() step=%d scaled_logits_batch0=%s", step + 1, [f"{v:.4f}" for v in scaled_logits[0].tolist()])

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
                next_token = rng.choice(self.model.vocab_size, p=probs[0])
            else:
                next_token = np.array(
                    [rng.choice(self.model.vocab_size, p=probs[b]) for b in range(batch_size)]
                )

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
        """Mask all logits below the top-k values to minus infinity."""
        batch_size, vocab_size = logits.shape
        sorted_indices = np.argsort(logits, axis=-1)[:, ::-1]
        kth_indices = sorted_indices[:, top_k - 1 : top_k]
        kth_values = np.take_along_axis(logits, kth_indices, axis=-1)
        masked = np.where(logits >= kth_values, logits, -np.inf)
        return masked

    @staticmethod
    def _compute_entropy(probs: np.ndarray) -> float:
        """Compute mean entropy of a probability distribution.

        Entropy = -sum(p * log(p)) — low = peaked (confident), high = uniform (uncertain).
        """
        safe_probs = np.clip(probs, 1e-10, 1.0)
        if probs.ndim == 1:
            return float(-np.sum(safe_probs * np.log(safe_probs)))
        return float(-np.mean(np.sum(safe_probs * np.log(safe_probs), axis=-1)))

    @staticmethod
    def _top_k_values(probs: np.ndarray, k: int) -> list[str]:
        """Get top-k probability values as formatted strings for batch 0."""
        top_idx = np.argsort(probs[0], axis=-1)[::-1][:k]
        return [f"{probs[0, i]:.4f}" for i in top_idx]


# Backwards-compat module-level functions (used by tests)
def _validate_prompt(prompt: np.ndarray) -> np.ndarray:
    """Validate and normalize a prompt."""
    return module_validate_prompt(prompt)


def _apply_top_k_mask(logits: np.ndarray, top_k: int) -> np.ndarray:
    """Apply top-k masking to logits."""
    return module_apply_top_k_mask(logits, top_k)


@staticmethod
def module_validate_prompt(prompt: np.ndarray) -> np.ndarray:
    """Validate and normalize a prompt."""
    ...


@staticmethod
def module_apply_top_k_mask(logits: np.ndarray, top_k: int) -> np.ndarray:
    """Apply top-k masking to logits."""
    ...
