"""Autoregressive inference engine for a PyTorch decoder-only transformer.

Implements TorchTextGenerator with greedy decoding, temperature-sampled decoding,
top-k filtering, and batch processing support.
"""

from __future__ import annotations

import torch


class TorchTextGenerator:
    """Autoregressive text generation for a PyTorch model.

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

        Uses greedy decoding if temperature is 0.0, otherwise uses
        temperature-sampled decoding.

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

        if self.temperature == 0.0:
            return self.generate_greedy(prompt)
        return self.generate_sampled(prompt, self.temperature)

    def generate_greedy(self, prompt: torch.Tensor) -> torch.Tensor:
        """Generate using greedy decoding (argmax).

        Each next token is selected by argmax of the model logits.
        This is deterministic — same prompt always produces same output.

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

        # Initialize sequence buffer with the prompt
        # We build up the sequence by appending tokens
        sequence = prompt.clone()  # (B, S0)

        self.model.eval()
        for _step in range(self.max_new_tokens):
            logits = self.model(sequence)  # (B, S, V)

            # Get logits for last position
            # Shape: (B, V) — logits for each vocab token at the current last position
            step_logits = logits[:, -1, :]  # (B, V)

            # Greedy: argmax picks the highest-logit token per sequence in batch
            # Shape: (B,) — integer token indices
            next_token = torch.argmax(step_logits, dim=-1)  # (B,)

            # Append next token to sequence
            # Shape: (B, 1) — one new token per sequence
            next_token_2d = next_token.reshape(batch_size, 1)  # (B, 1)

            # Concatenate along sequence dimension to extend
            # (B, S) @ (B, 1) -> (B, S+1)
            sequence = torch.cat([sequence, next_token_2d], dim=1)  # (B, S+1)

        return sequence

    def generate_sampled(self, prompt: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Generate using temperature-sampled token selection.

        Applies temperature scaling to logits, optional top-k filtering,
        then samples from the resulting probability distribution.

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

        # Temperature 0 falls back to greedy
        if temperature == 0.0:
            return self.generate_greedy(prompt)

        # Effective temperature (clipped to avoid division by zero or negative)
        effective_temperature = max(temperature, 1e-8)

        # Initialize sequence buffer with the prompt
        sequence = prompt.clone()  # (B, S0)

        self.model.eval()
        for _step in range(self.max_new_tokens):
            logits = self.model(sequence)  # (B, S, V)

            # Get logits for last position
            # Shape: (B, V) — logits for each vocab token at current last position
            step_logits = logits[:, -1, :]  # (B, V)

            # Apply temperature scaling: lower temp = sharper distribution
            # Shape: (B, V)
            scaled_logits = step_logits / effective_temperature  # (B, V)

            # Top-k filtering: keep only top-k logits, zero out the rest
            if self.top_k > 0:
                scaled_logits = _apply_top_k_mask(scaled_logits, self.top_k)  # (B, V)

            # Convert logits to probabilities via softmax
            # Stable softmax: subtract max per row to avoid overflow
            # softmax(z_i) = exp(z_i - max(z)) / sum(exp(z_j - max(z)))
            logits_max = torch.max(scaled_logits, dim=-1, keepdim=True).values  # (B, 1)
            exp_logits = torch.exp(scaled_logits - logits_max)  # (B, V)
            probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)  # (B, V)

            # Sample from categorical distribution for each sequence
            # shape (B,) — sampled token indices from [0, V)
            next_token = torch.stack(
                [torch.multinomial(probs[b].float(), num_samples=1) for b in range(batch_size)]
            )  # (B, 1) -> (B,)

            # Append next token to sequence
            sequence = torch.cat([sequence, next_token], dim=1)  # (B, S+1)

        return sequence


def _validate_prompt(prompt: torch.Tensor) -> torch.Tensor:
    """Validate and normalize a prompt to the expected format.

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
    # sorted_values: (B, V) — sorted logit values descending
    # sorted_indices: (B, V) — the column indices that sort each row descending
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)  # (B, V)

    # Get the threshold value per row: the k-th highest logit
    # kth_values: (B, 1) — the k-th largest logit per batch element
    kth_indices = sorted_indices[:, top_k - 1 : top_k]  # (B, 1)
    kth_values = logits.gather(dim=-1, index=kth_indices)  # (B, 1)

    # Keep only logits >= k-th threshold, set others to -inf
    masked = torch.where(logits >= kth_values, logits, float("-inf"))  # (B, V)

    return masked
