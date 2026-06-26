"""Autoregressive inference engine for a Triton decoder-only transformer.

Since TritonModel inherits from nn.Module and runs on CUDA, the generation
logic is identical to the torch backend — this module provides the same
interface with Triton-idiomatic naming.

Logging via triton_inference:
    - INFO:  first/last token generation, temperature/sample mode, device
    - DEBUG: per-token logits (top-5), temperature scaling, top-k masking
    - TRACE: softmax entropy, sampled token probabilities

Architecture
-------------
Generation loop:
    for step in range(max_new_tokens):
        logits   = model(sequence)              # (B, S, V)
        step_log = logits[:, -1, :]             # (B, V)
        if sampled:
            probs = softmax(step_log / temp)    # (B, V)
            if top_k > 0: probs = top_k_filter(probs)
            token = torch.multinomial(probs)     # sample
        else:
            token = torch.argmax(step_log)       # greedy
        sequence = concat(sequence, token)       # (B, S+1)
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class TritonTextGenerator:
    """Autoregressive text generation for a Triton model.

    Logs per-token generation progress, sampling statistics, and
    device context when DEBUG/TRACE level is enabled.

    Parameters
    ----------
    model : torch.nn.Module
        The model must accept integer tensor of shape (batch_size, seq_len)
        and return floating-point logits of shape (batch_size, seq_len, vocab_size).
    max_new_tokens : int
        Maximum number of tokens to generate after the prompt.
    temperature : float
        Sampling temperature for token selection (0.0 = greedy/argmax).
    top_k : int
        Keep only top-k logits before sampling. 0 = disabled.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k

    def generate(self, prompt: torch.Tensor) -> torch.Tensor:
        """Generate tokens autoregressively.

        Logs generation mode, batch context, and device placement.
        """
        prompt = _validate_prompt(prompt)
        device = prompt.device
        batch_size, prompt_len = prompt.shape
        mode = "greedy" if self.temperature == 0.0 else f"sampled_T={self.temperature:.3f}"

        if self.top_k > 0:
            mode = f"top_k={self.top_k}_" + mode

        logger.info(
            "TritonTextGenerator.generate() mode=%s batch=%d prompt_len=%d max_new=%d device=%s",
            mode,
            batch_size,
            prompt_len,
            self.max_new_tokens,
            device,
        )

        if self.temperature == 0.0:
            return self._generate_greedy(prompt)
        return self._generate_sampled(prompt, self.temperature)

    def _generate_greedy(self, prompt: torch.Tensor) -> torch.Tensor:
        """Generate using argmax — deterministic, same prompt always gives same output.

        Logs first/last token selection for traceability.
        """
        prompt = _validate_prompt(prompt)
        batch_size = prompt.shape[0]
        logger.info("TritonTextGenerator._generate_greedy() batch_size=%d", batch_size)

        sequence = prompt.clone()
        self.model.eval()

        for step in range(self.max_new_tokens):
            logits = self.model(sequence)  # (B, S, V)
            step_logits = logits[:, -1, :]  # (B, V)
            next_token = torch.argmax(step_logits, dim=-1)  # (B,)

            # Log top-5 on first/last step for traceability
            if step == 0 or step == self.max_new_tokens - 1 or self.max_new_tokens <= 5:
                top_idx = torch.argsort(step_logits[0], dim=-1, descending=True)[:5]
                top_log = step_logits[0, top_idx]
                logger.debug(
                    "_generate_greedy() step=%d top5_tokens=%s top5_logits=%s",
                    step + 1,
                    top_idx.tolist(),
                    [f"{v:.4f}" for v in top_log.tolist()],
                )

            next_token_2d = next_token.reshape(batch_size, 1)
            sequence = torch.cat([sequence, next_token_2d], dim=1)

            if step == 0 or step == self.max_new_tokens - 1:
                logger.info(
                    "_generate_greedy() step=%d/%d token=%s new_len=%d",
                    step + 1,
                    self.max_new_tokens,
                    next_token.tolist(),
                    sequence.shape[1],
                )

        logger.info(
            "_generate_greedy() complete batch=%d final_len=%d",
            batch_size,
            sequence.shape[1],
        )
        return sequence

    def _generate_sampled(self, prompt: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Generate using temperature-sampled token selection.

        Logs temperature scaling, top-k filtering, softmax distribution,
        and sampled tokens for reproducibility.
        """
        prompt = _validate_prompt(prompt)
        batch_size = prompt.shape[0]
        device = prompt.device
        effective_temperature = max(temperature, 1e-8)

        logger.info(
            "TritonTextGenerator._generate_sampled() batch=%d temp=%.4f top_k=%d device=%s",
            batch_size,
            temperature,
            self.top_k,
            device,
        )

        logger.debug("_generate_sampled() effective_temperature=%.6f", effective_temperature)

        sequence = prompt.clone()
        self.model.eval()

        for step in range(self.max_new_tokens):
            logits = self.model(sequence)  # (B, S, V)
            step_logits = logits[:, -1, :]  # (B, V)

            # Apply temperature scaling: lower temp = sharper distribution
            scaled_logits = step_logits / effective_temperature
            logger.debug(
                "_generate_sampled() step=%d scaled_logits_batch0=%s",
                step + 1,
                [f"{v:.4f}" for v in scaled_logits[0].tolist()],
            )

            # Top-k filtering: keep only top-k logits
            if self.top_k > 0:
                scaled_logits = _apply_top_k_mask(scaled_logits, self.top_k)
                logger.debug("_generate_sampled() step=%d top_k=%d", step + 1, self.top_k)

            # Stable softmax
            logits_max = torch.max(scaled_logits, dim=-1, keepdim=True).values
            exp_logits = torch.exp(scaled_logits - logits_max)
            probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)

            # Log entropy and top-5 probabilities
            logger.debug(
                "_generate_sampled() step=%d probs_entropy=%.4f probs_top5=%s",
                step + 1,
                _compute_entropy(probs),
                _top_k_values(probs, 5),
            )

            # Sample from categorical distribution for each sequence
            next_token = torch.stack(
                [torch.multinomial(probs[b].float(), num_samples=1) for b in range(batch_size)]
            )

            # Log sampled token on first/last step
            if step == 0 or step == self.max_new_tokens - 1 or self.max_new_tokens <= 5:
                selected_probs = [f"{probs[b, int(next_token[b])]:.4f}" for b in range(batch_size)]
                logger.info(
                    "_generate_sampled() step=%d/%d sampled=%s selected_probs=%s",
                    step + 1,
                    self.max_new_tokens,
                    next_token.tolist(),
                    selected_probs,
                )

            sequence = torch.cat([sequence, next_token], dim=1)

        logger.info(
            "_generate_sampled() complete batch=%d final_len=%d",
            batch_size,
            sequence.shape[1],
        )
        return sequence


def _validate_prompt(prompt: torch.Tensor) -> torch.Tensor:
    """Validate and normalize a prompt to expected 2D int64 format."""
    prompt = torch.as_tensor(prompt, dtype=torch.int64)
    if prompt.ndim == 1:
        prompt = prompt.reshape(1, -1)
    elif prompt.ndim != 2:
        raise ValueError(f"Prompt must be 1D or 2D, got {prompt.ndim}D")
    return prompt


def _apply_top_k_mask(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask all logits below the top-k values to minus infinity.

    Parameters
    ----------
    logits : torch.Tensor, shape (B, V)
        Raw logits before top-k filtering.
    top_k : int
        Number of highest-logit tokens to keep.

    Returns
    -------
    torch.Tensor
        Masked logits with non-top-k values set to -inf.
    """
    # Sort each row descending to find k-th threshold
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)
    # Get the threshold value per row: the k-th highest logit
    kth_indices = sorted_indices[:, top_k - 1 : top_k]
    kth_values = logits.gather(dim=-1, index=kth_indices)
    # Keep only logits >= k-th threshold, set others to -inf
    masked = torch.where(logits >= kth_values, logits, float("-inf"))
    return masked


def _compute_entropy(probs: torch.Tensor) -> float:
    """Compute mean entropy of probability distribution.

    Entropy = -sum(p * log(p)) — low entropy = peaked (confident),
    high entropy = uniform (uncertain).

    Parameters
    ----------
    probs : torch.Tensor, shape (B, V)
        Probability distribution.

    Returns
    -------
    entropy : float
        Mean entropy across batch.
    """
    safe_probs = torch.clip(probs, min=1e-10)
    return float(torch.mean(-torch.sum(safe_probs * torch.log(safe_probs), dim=-1)))


def _top_k_values(probs: torch.Tensor, k: int) -> list[str]:
    """Get top-k probability values as formatted strings for batch 0."""
    top_idx = torch.argsort(probs[0], dim=-1, descending=True)[:k]
    return [f"{probs[0, i]:.4f}" for i in top_idx]
