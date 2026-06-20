"""Autoregressive inference engine for a Triton decoder-only transformer.

Since TritonModel inherits from nn.Module and runs on CUDA, the generation
logic is identical to the torch backend — this module provides the same
interface with Triton-idiomatic naming.
"""

from __future__ import annotations

import torch


class TritonTextGenerator:
    """Autoregressive text generation for a Triton model.

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

        Parameters
        ----------
        prompt : torch.Tensor, shape (batch_size, prompt_len)
            Initial token IDs.

        Returns
        -------
        output : torch.Tensor, shape (batch_size, prompt_len + new_tokens)
            Full sequence including the original prompt.

        """
        prompt = _validate_prompt(prompt)

        if self.temperature == 0.0:
            return self._generate_greedy(prompt)
        return self._generate_sampled(prompt, self.temperature)

    def _generate_greedy(self, prompt: torch.Tensor) -> torch.Tensor:
        """Generate using argmax — deterministic, same prompt always gives same output."""
        prompt = _validate_prompt(prompt)
        batch_size = prompt.shape[0]

        sequence = prompt.clone()

        self.model.eval()
        for _step in range(self.max_new_tokens):
            logits = self.model(sequence)
            step_logits = logits[:, -1, :]
            next_token = torch.argmax(step_logits, dim=-1)
            next_token_2d = next_token.reshape(batch_size, 1)
            sequence = torch.cat([sequence, next_token_2d], dim=1)

        return sequence

    def _generate_sampled(self, prompt: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Generate using temperature-sampled token selection."""
        prompt = _validate_prompt(prompt)
        batch_size = prompt.shape[0]

        if temperature == 0.0:
            return self._generate_greedy(prompt)

        effective_temperature = max(temperature, 1e-8)
        sequence = prompt.clone()

        self.model.eval()
        for _step in range(self.max_new_tokens):
            logits = self.model(sequence)
            step_logits = logits[:, -1, :]
            scaled_logits = step_logits / effective_temperature

            if self.top_k > 0:
                scaled_logits = _apply_top_k_mask(scaled_logits, self.top_k)

            logits_max = torch.max(scaled_logits, dim=-1, keepdim=True).values
            exp_logits = torch.exp(scaled_logits - logits_max)
            probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)

            next_token = torch.stack([torch.multinomial(probs[b].float(), num_samples=1) for b in range(batch_size)])

            sequence = torch.cat([sequence, next_token], dim=1)

        return sequence


def _validate_prompt(prompt: torch.Tensor) -> torch.Tensor:
    """Validate prompt to (batch_size, seq_len) int64 tensor."""
    prompt = torch.as_tensor(prompt, dtype=torch.int64)
    if prompt.ndim == 1:
        prompt = prompt.reshape(1, -1)
    elif prompt.ndim != 2:
        raise ValueError(f"Prompt must be 1D or 2D, got {prompt.ndim}D")
    return prompt


def _apply_top_k_mask(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask all logits below the top-k values to minus infinity."""
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)
    kth_indices = sorted_indices[:, top_k - 1 : top_k]
    kth_values = logits.gather(dim=-1, index=kth_indices)
    masked = torch.where(logits >= kth_values, logits, float("-inf"))
    return masked
