"""Autoregressive inference engine for a CUDA decoder-only transformer.

Implements CudaTextGenerator with greedy decoding, temperature-sampled decoding,
top-k filtering, and batch processing support.

Unlike the PyTorch backend, the CUDAModel does NOT inherit from nn.Module,
so there is no .eval()/ .train() mode. Inference is always deterministic
unless temperature sampling introduces randomness.

Logging via cuda_inference:
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


class CudaTextGenerator:
    """Autoregressive text generation for a CUDA model.

    Logs per-token generation progress, sampling statistics, and
    device context when DEBUG/TRACE level is enabled.

    Parameters
    ----------
    model : object
        The trained decoder-only transformer model (CUDAModel type).
        Must accept integer tensor of shape (batch_size, seq_len) and
        return floating-point logits of shape (batch_size, seq_len, vocab_size).
    max_new_tokens : int
        Maximum number of tokens to generate after the prompt.
    temperature : float
        Sampling temperature for token selection (0.0 = greedy/argmax).
    top_k : int
        Keep only top-k logits before sampling. 0 = disabled (all logits used).

    Usage
    -----
    >>> import torch
    >>> from impl._cuda.model import CUDAModel
    >>> from impl._cuda.inference import CudaTextGenerator
    >>> model = CUDAModel(vocab_size=1000, embed_dim=64, n_layers=2, n_heads=4)
    >>> gen = CudaTextGenerator(model, max_new_tokens=20)
    >>> tokens = gen.generate(prompt)  # uses greedy if temperature=0
    """

    def __init__(
        self,
        model: object,  # CUDAModel
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> None:
        # pyright: ignore[reportUnknownArgumentType]
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self._device = "cuda"

    def generate(self, prompt: torch.Tensor) -> torch.Tensor:
        """Generate tokens autoregressively.

        Logs generation mode, batch context, and device placement.
        """
        prompt = _validate_prompt(prompt)
        prompt = prompt.to(self._device)

        batch_size, prompt_len = prompt.shape
        mode = "greedy" if self.temperature == 0.0 else f"sampled_T={self.temperature:.3f}"

        if self.top_k > 0:
            mode = f"top_k={self.top_k}_" + mode

        logger.info(
            "CudaTextGenerator.generate() mode=%s batch=%d prompt_len=%d max_new=%d device=%s",
            mode,
            batch_size,
            prompt_len,
            self.max_new_tokens,
            self._device,
        )

        if self.temperature == 0.0:
            return self.generate_greedy(prompt)
        return self.generate_sampled(prompt, self.temperature)

    def generate_greedy(self, prompt: torch.Tensor) -> torch.Tensor:
        """Generate using greedy decoding (argmax).

        Logs first/last token selection for traceability.
        """
        prompt = _validate_prompt(prompt)
        prompt = prompt.to(self._device)
        batch_size = prompt.shape[0]
        logger.info("CudaTextGenerator.generate_greedy() batch_size=%d", batch_size)

        sequence = prompt.clone()

        for step in range(self.max_new_tokens):
            # Forward pass — model always on CUDA
            logits = self.model.forward(sequence)  # pyright: ignore[reportAttributeAccessIssue, reportCallIssue]
            # Get logits for last position: (B, V)
            step_logits = logits[:, -1, :]

            # Greedy: argmax picks the highest-logit token
            next_token = torch.argmax(step_logits, dim=-1)  # (B,)

            # Log top-5 on first/last step for traceability
            if step == 0 or step == self.max_new_tokens - 1 or self.max_new_tokens <= 5:
                top_idx = torch.argsort(step_logits[0], dim=-1, descending=True)[:5]
                top_log = step_logits[0, top_idx]
                logger.debug(
                    "generate_greedy() step=%d top5_tokens=%s top5_logits=%s",
                    step + 1,
                    top_idx.tolist(),
                    [f"{v:.4f}" for v in top_log.tolist()],
                )

            # Append next token to sequence: (B, 1)
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
            "generate_greedy() complete batch=%d final_len=%d",
            batch_size,
            sequence.shape[1],
        )
        return sequence

    def generate_sampled(self, prompt: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Generate using temperature-sampled token selection.

        Logs temperature scaling, top-k filtering, softmax distribution,
        and sampled tokens for reproducibility.
        """
        prompt = _validate_prompt(prompt)
        prompt = prompt.to(self._device)
        batch_size = prompt.shape[0]
        effective_temperature = max(temperature, 1e-8)

        logger.info(
            "CudaTextGenerator.generate_sampled() batch=%d temp=%.4f top_k=%d device=%s",
            batch_size,
            temperature,
            self.top_k,
            self._device,
        )

        logger.debug("generate_sampled() effective_temperature=%.6f", effective_temperature)

        sequence = prompt.clone()

        for step in range(self.max_new_tokens):
            # Forward pass
            logits = self.model.forward(sequence)  # pyright: ignore[reportAttributeAccessIssue, reportCallIssue]
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
                logger.debug(
                    "generate_sampled() step=%d top_k=%d (vocab=%d)",
                    step + 1,
                    self.top_k,
                    scaled_logits.shape[-1],
                )

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
            "generate_sampled() complete batch=%d final_len=%d",
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
        raise ValueError(f"Prompt must be 1D or 2D, got {prompt.ndim}D with shape {prompt.shape}")
    return prompt


def _apply_top_k_mask(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask all logits below the top-k values to minus infinity.

    Adds logging to trace top-k filtering behavior.

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
    logger.debug("_apply_top_k_mask() logits_shape=%s top_k=%d vocab=%d", logits.shape, top_k, logits.shape[-1])

    # Sort each row descending to find k-th threshold
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)

    # Clamp top_k to avoid out-of-bounds slicing when top_k >= vocab_size
    vocab_size = logits.shape[-1]
    effective_top_k = min(top_k, vocab_size)

    # Get the threshold value per row: the k-th highest logit
    kth_indices = sorted_indices[:, effective_top_k - 1 : effective_top_k]
    kth_values = logits.gather(dim=-1, index=kth_indices)

    # If top_k >= vocab_size, no masking needed — return as-is
    if effective_top_k >= vocab_size:
        logger.debug("_apply_top_k_mask() top_k=%d >= vocab=%d skipping mask", effective_top_k, vocab_size)
        return logits

    # Keep only logits >= k-th threshold, set others to -inf
    masked = torch.where(logits >= kth_values, logits, float("-inf"))

    if top_k < vocab_size:
        logger.debug(
            "_apply_top_k_mask() threshold=%.4f masked %d/%d values to -inf",
            kth_values[0, 0].item(),
            (logits < kth_values).sum().item(),
            logits.numel(),
        )

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
