"""Autoregressive inference engine for a decoder-only transformer.

Implements TextGenerator with greedy decoding, temperature-sampled decoding,
top-k filtering, and batch processing support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from impl._np.model import NumPyModel


class TextGenerator:
    """Autoregressive text generation for a NumPyModel.

    Parameters
    ----------
    model : NumPyModel  # noqa: F821 — forward-ref, defined in model.py
        The trained decoder-only transformer model.
    max_new_tokens : int
        Maximum number of tokens to generate after the prompt.
    temperature : float
        Sampling temperature for token selection (0.0 = greedy/argmax).
    top_k : int
        Keep only top-k logits before sampling. 0 = disabled (all logits used).

    Usage
    -----
    >>> generator = TextGenerator(model, max_new_tokens=20)
    >>> tokens = generator.generate(prompt)  # uses greedy if temperature=0

    """

    def __init__(
        self,
        model: NumPyModel,  # noqa: F821 — forward-ref, defined in model.py
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

        Uses greedy decoding if temperature is 0.0, otherwise uses
        temperature-sampled decoding.

        Parameters
        ----------
        prompt : np.ndarray, shape (batch_size, prompt_len)
            Initial token IDs to start from.

        Returns
        -------
        output : np.ndarray, shape (batch_size, prompt_len + new_tokens)
            Full sequence including the original prompt.

        Examples
        --------
        >>> import numpy as np
        >>> from impl._np.model import NumPyModel
        >>> from impl._np.inference import TextGenerator
        >>> model = NumPyModel(vocab_size=16, embed_dim=8, n_layers=1,
        ...                    n_heads=2, n_experts=2, ff_dim=16, k=1,
        ...                    rope_dim=0, seed=0)
        >>> gen = TextGenerator(model, max_new_tokens=3)
        >>> prompt = np.array([[0, 1]], dtype=np.int32)
        >>> result = gen.generate(prompt)
        >>> result.shape
        (1, 5)

        """
        prompt = _validate_prompt(prompt)

        if self.temperature == 0.0:
            return self.generate_greedy(prompt)
        return self.generate_sampled(prompt, self.temperature)

    def generate_greedy(self, prompt: np.ndarray) -> np.ndarray:
        """Generate using greedy decoding (argmax).

        Each next token is selected by argmax of the model logits.
        This is deterministic — same prompt always produces same output.

        Parameters
        ----------
        prompt : np.ndarray, shape (batch_size, prompt_len)
            Initial token IDs.

        Returns
        -------
        output : np.ndarray, shape (batch_size, prompt_len + new_tokens)
            Generated sequence including prompt.

        """
        prompt = _validate_prompt(prompt)
        batch_size, seq_len = prompt.shape

        # Initialize sequence buffer with the prompt
        # shape: (batch_size, 0) — empty buffer on top of prompt
        # We build up the sequence by appending tokens
        sequence = prompt.copy()  # (B, S0)

        for _step in range(self.max_new_tokens):
            logits = self.model.forward(sequence)  # (B, S, V)

            # Get logits for last position
            # Shape: (B, V) — logits for each vocab token at the current last position
            step_logits = logits[:, -1, :]  # (B, V)

            # Greedy: argmax picks the highest-logit token per sequence in batch
            # Shape: (B,) — integer token indices
            next_token = np.argmax(step_logits, axis=-1)  # (B,)

            # Append next token to sequence
            # Shape: (B, 1) — one new token per sequence
            next_token_2d = next_token.reshape(batch_size, 1)  # (B, 1)

            # Concatenate along sequence dimension to extend
            # (B, S) @ (B, 1) -> (B, S+1)
            sequence = np.concatenate([sequence, next_token_2d], axis=1)  # (B, S+1)

        return sequence

    def generate_sampled(self, prompt: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Generate using temperature-sampled token selection.

        Applies temperature scaling to logits, optional top-k filtering,
        then samples from the resulting probability distribution.

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
        prompt = _validate_prompt(prompt)
        batch_size, seq_len = prompt.shape

        # Temperature 0 falls back to greedy
        if temperature == 0.0:
            return self.generate_greedy(prompt)

        # Effective temperature (clipped to avoid division by zero or negative)
        effective_temperature = max(temperature, 1e-8)

        # Initialize sequence buffer with the prompt
        sequence = prompt.copy()  # (B, S0)

        # Random state for reproducibility
        rng = np.random.default_rng(self.model.seed)

        for _step in range(self.max_new_tokens):
            logits = self.model.forward(sequence)  # (B, S, V)

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
            logits_max = np.max(scaled_logits, axis=-1, keepdims=True)  # (B, 1)
            exp_logits = np.exp(scaled_logits - logits_max)  # (B, V)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # (B, V)

            # Sample from categorical distribution for each sequence
            # shape (B,) — sampled token indices from [0, V)
            if batch_size == 1:
                next_token = rng.choice(self.model.vocab_size, p=probs[0])  # scalar int
            else:
                next_token = np.array(
                    [rng.choice(self.model.vocab_size, p=probs[b]) for b in range(batch_size)]
                )  # (B,)

            # Append next token to sequence
            next_token_2d = np.asarray(next_token).reshape(batch_size, 1)  # (B, 1)
            sequence = np.concatenate([sequence, next_token_2d], axis=1)  # (B, S+1)

        return sequence


def _validate_prompt(prompt: np.ndarray) -> np.ndarray:
    """Validate and normalize a prompt to the expected format.

    Parameters
    ----------
    prompt : np.ndarray
        Input prompt — may be 1D or 2D, various integer dtypes.

    Returns
    -------
    np.ndarray
        2D prompt with shape (batch_size, seq_len) and dtype int32.

    Raises
    ------
    ValueError
        If prompt has fewer than 2 dimensions or an integer dtype.

    """
    prompt = np.asarray(prompt, dtype=np.int32)
    if prompt.ndim == 1:
        prompt = prompt.reshape(1, -1)
    elif prompt.ndim != 2:
        raise ValueError(f"Prompt must be 1D or 2D, got {prompt.ndim}D with shape {prompt.shape}")
    return prompt


def _apply_top_k_mask(logits: np.ndarray, top_k: int) -> np.ndarray:
    """Mask all logits below the top-k values to minus infinity.

    This implements top-k filtering: only keep the top-k logits,
    set the rest to -inf so they have zero probability after softmax.

    Parameters
    ----------
    logits : np.ndarray, shape (batch_size, vocab_size)
        Raw logits before top-k filtering.
    top_k : int
        Number of highest-logit tokens to keep.

    Returns
    -------
    np.ndarray, shape (batch_size, vocab_size)
        Masked logits with non-top-k values set to -inf.

    Algorithm
    ---------
    1. Sort each row descending to find k-th threshold
    2. Zero out all values below the threshold
    3. Set remaining non-top-k values to -inf for clean softmax

    """
    batch_size, vocab_size = logits.shape

    # Sort each row descending to find the k-th threshold
    # sorted_indices: (B, V) — the order of indices that sort each row descending
    sorted_indices = np.argsort(logits, axis=-1)[:, ::-1]  # (B, V)

    # Get the threshold value per row: the k-th highest logit
    # kth_indices: (B, 1) — column index of k-th largest element
    kth_indices = sorted_indices[:, top_k - 1 : top_k]  # (B, 1)

    # kth_values: (B, 1) — the actual k-th largest logit per batch element
    kth_values = np.take_along_axis(logits, kth_indices, axis=-1)  # (B, 1)

    # Zero out values below the k-th threshold
    # Only keep logits >= k-th threshold
    masked = np.where(logits >= kth_values, logits, -np.inf)  # (B, V)

    return masked
