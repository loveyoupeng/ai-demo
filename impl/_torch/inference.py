"""Autoregressive inference engine for a PyTorch decoder-only transformer.

Implements TorchTextGenerator with greedy decoding, temperature-sampled decoding,
top-k filtering, and batch processing support.

Logging via torch_inference:
    - INFO:  first/last token generation, temperature/sample mode, device info
    - DEBUG: per-token logits (top-5), temperature scaling, top-k masking, softmax probs
    - TRACE: per-token softmax distribution, RNG state

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
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TorchTextGenerator:
    """Autoregressive text generation for a PyTorch model.

    Logs per-token generation progress, sampling statistics, and
    device context when DEBUG/TRACE level is enabled.

    Parameters
    ----------
    model : torch.nn.Module
        The trained decoder-only transformer model. Must accept
        integer tensor of shape (batch_size, seq_len) and return
        floating-point logits of shape (batch_size, seq_len, vocab_size).
    max_new_tokens : int
        Maximum number of tokens to generate after the prompt.
    temperature : float
        Sampling temperature for token selection (0.0 = greedy/argmax).
    top_k : int
        Keep only top-k logits before sampling. 0 = disabled (all logits used).

    Usage
    -----
    >>> import torch
    >>> from impl._torch.model import TorchModel
    >>> from impl._torch.inference import TorchTextGenerator
    >>> config = TorchModel.make_config(vocab_size=50257, embed_dim=768)
    >>> model = TorchModel(config)
    >>> gen = TorchTextGenerator(model, max_new_tokens=20)
    >>> tokens = gen.generate(prompt)  # uses greedy if temperature=0

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

        Parameters
        ----------
        prompt : torch.Tensor, shape (batch_size, prompt_len)
            Initial token IDs to start from.

        Returns
        -------
        output : torch.Tensor, shape (batch_size, prompt_len + new_tokens)
            Full sequence including the original prompt.

        """
        prompt = _validate_prompt(prompt)
        device = prompt.device
        batch_size, prompt_len = prompt.shape
        mode = "greedy" if self.temperature == 0.0 else f"sampled_T={self.temperature:.3f}"

        if self.top_k > 0:
            mode = f"top_k={self.top_k}_" + mode

        logger.info(
            "TorchTextGenerator.generate() mode=%s batch_size=%d prompt_len=%d max_new=%d device=%s",
            mode,
            batch_size,
            prompt_len,
            self.max_new_tokens,
            device,
        )

        if self.temperature == 0.0:
            return self.generate_greedy(prompt)
        return self.generate_sampled(prompt, self.temperature)

    def generate_greedy(self, prompt: torch.Tensor) -> torch.Tensor:
        """Generate using greedy decoding (argmax).

        Logs first/last token selection for traceability.

        Parameters
        ----------
        prompt : torch.Tensor, shape (batch_size, prompt_len)
            Initial token IDs.

        Returns
        -------
        output : torch.Tensor, shape (batch_size, prompt_len + new_tokens)
            Generated sequence including prompt.

        """
        prompt = _validate_prompt(prompt)
        batch_size = prompt.shape[0]
        logger.info("TorchTextGenerator.generate_greedy() batch_size=%d", batch_size)

        sequence = prompt.clone()
        self.model.eval()

        for step in range(self.max_new_tokens):
            logits = self.model(sequence)  # (B, S, V)
            # Get logits for last position: (B, V)
            step_logits = logits[:, -1, :]

            # Greedy: argmax picks the highest-logit token
            next_token = torch.argmax(step_logits, dim=-1)  # (B,)

            # Log top-5 on first/last step for traceability
            if step == 0 or step == self.max_new_tokens - 1 or self.max_new_tokens <= 5:
                top_idx = torch.argsort(step_logits[0], dim=-1, descending=True)[:5]
                top_log = step_logits[0, top_idx]
                logger.debug(
                    "generate_greedy() step=%d batch=0 top5_tokens=%s top5_logits=%s",
                    step + 1,
                    top_idx.tolist(),
                    [f"{v:.4f}" for v in top_log.tolist()],
                )

            # Append next token to sequence
            next_token_2d = next_token.reshape(batch_size, 1)
            sequence = torch.cat([sequence, next_token_2d], dim=1)

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

    def generate_sampled(self, prompt: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Generate using temperature-sampled token selection.

        Logs temperature scaling, top-k filtering, softmax distribution,
        and sampled tokens for reproducibility.

        Parameters
        ----------
        prompt : torch.Tensor, shape (batch_size, prompt_len)
            Initial token IDs.
        temperature : float
            Sampling temperature. Lower = more confident predictions.
            0.0 falls back to greedy decoding.

        Returns
        -------
        output : torch.Tensor, shape (batch_size, prompt_len + new_tokens)
            Generated sequence including prompt.

        """
        prompt = _validate_prompt(prompt)
        batch_size = prompt.shape[0]
        device = prompt.device
        effective_temperature = max(temperature, 1e-8)

        logger.info(
            "TorchTextGenerator.generate_sampled() batch_size=%d prompt_len=%d temperature=%.4f top_k=%d device=%s",
            batch_size,
            prompt.shape[1],
            temperature,
            self.top_k,
            device,
        )

        logger.debug("generate_sampled() effective_temperature=%.6f", effective_temperature)

        sequence = prompt.clone()
        self.model.eval()

        for step in range(self.max_new_tokens):
            logits = self.model(sequence)  # (B, S, V)
            # Get logits for last position: (B, V)
            step_logits = logits[:, -1, :]

            # Apply temperature scaling: lower temp = sharper distribution
            scaled_logits = step_logits / effective_temperature
            logger.debug(
                "generate_sampled() step=%d scaled_logits_batch0=%s",
                step + 1,
                [f"{v:.4f}" for v in scaled_logits[0].tolist()],
            )

            # Top-k filtering: keep only top-k logits
            if self.top_k > 0:
                scaled_logits = _apply_top_k_mask(scaled_logits, self.top_k)
                logger.debug("generate_sampled() step=%d top_k_masked (top_k=%d)", step + 1, self.top_k)

            # Stable softmax: subtract max per row to avoid overflow
            # softmax(z_i) = exp(z_i - max(z)) / sum(exp(z_j - max(z)))
            logits_max = torch.max(scaled_logits, dim=-1, keepdim=True).values
            exp_logits = torch.exp(scaled_logits - logits_max)
            probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)

            # Log entropy and top-5 probabilities
            logger.debug(
                "generate_sampled() step=%d probs_entropy=%.4f probs_top5=%s",
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
                    "generate_sampled() step=%d/%d sampled=%s selected_probs=%s",
                    step + 1,
                    self.max_new_tokens,
                    next_token.tolist(),
                    selected_probs,
                )

            # Append next token to sequence
            sequence = torch.cat([sequence, next_token], dim=1)

        logger.info(
            "generate_sampled() complete batch_size=%d final_len=%d",
            batch_size,
            sequence.shape[1],
        )
        return sequence


def _validate_prompt(prompt: torch.Tensor) -> torch.Tensor:
    """Validate and normalize a prompt to the expected 2D int64 format.

    Parameters
    ----------
    prompt : torch.Tensor
        Input prompt — may be 1D or 2D, various integer dtypes.

    Returns
    -------
    torch.Tensor
        2D prompt with shape (batch_size, seq_len) and dtype int64.

    Raises
    ------
    ValueError
        If prompt has fewer than 2 dimensions or a non-integer dtype.

    """
    prompt = torch.as_tensor(prompt, dtype=torch.int64)
    if prompt.ndim == 1:
        prompt = prompt.reshape(1, -1)
    elif prompt.ndim != 2:
        raise ValueError(f"Prompt must be 1D or 2D, got {prompt.ndim}D with shape {prompt.shape}")
    return prompt


def _apply_top_k_mask(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask all logits below the top-k values to minus infinity.

    This implements top-k filtering: only keep the top-k logits,
    set the rest to -inf so they have zero probability after softmax.

    Parameters
    ----------
    logits : torch.Tensor, shape (batch_size, vocab_size)
        Raw logits before top-k filtering.
    top_k : int
        Number of highest-logit tokens to keep.

    Returns
    -------
    torch.Tensor, shape (batch_size, vocab_size)
        Masked logits with non-top-k values set to -inf.

    Algorithm
    ---------
    1. Sort each row descending to find k-th threshold
    2. Keep values >= k-th threshold, set others to -inf

    """
    # Sort each row descending to find the k-th threshold
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
    probs : torch.Tensor, shape (B, V) or (V,)
        Probability distribution(s).

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
