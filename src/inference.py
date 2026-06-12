from __future__ import annotations

import numpy as np
from model.transformer import Transformer
from tokenizer.char_tokenizer import CharTokenizer


class AutoregressiveGenerator:
    """
    Implements autoregressive text generation.
    Uses the model to predict one token at a time, then feeds that token back as input.
    """

    def __init__(
        self, model: Transformer, tokenizer: CharTokenizer, temperature: float = 1.0, use_cache: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.use_cache = use_cache

    def generate(self, prompt: str, num_new_tokens: int, use_cache: bool | None = None) -> np.ndarray:
        """
        Generates new tokens given a prompt.

        Args:
            prompt: The starting text.
            num_new_tokens: How many tokens to generate.
            use_cache: Whether to use KV cache. Defaults to ``self.use_cache``.

        Returns:
            A numpy array of all token IDs (prompt + generated).
        """
        # 1. Encode the prompt
        # [1, prompt_len]
        current_ids = self.tokenizer.encode(prompt).reshape(1, -1)

        # Record the original prompt length
        prompt_len = current_ids.shape[1]

        # Handle empty prompt by ensuring at least one token (padding with index 0)
        # We keep track of whether we added this dummy token so we can strip it later.
        is_empty_prompt = prompt_len == 0

        if is_empty_prompt:
            current_ids = np.array([[0]], dtype=np.int32)

        for step in range(num_new_tokens):
            # 2. Get logits from the model with KV cache enabled
            # [1, current_len, vocab_size]
            use_kv = self.use_cache if use_cache is None else use_cache
            logits, _ = self.model.forward(current_ids, use_cache=use_kv, cache_idx=step + prompt_len)

            # 3. Focus on the LAST predicted token
            # [1, vocab_size]
            next_token_logits = logits[:, -1, :]

            # 4. Apply Temperature scaling
            # Higher temp = more randomness, Lower temp = more deterministic
            next_token_logits = next_token_logits / max(self.temperature, 1e-8)

            # 5. Sample from the distribution
            # We use softmax to get probabilities
            probs = self._softmax(next_token_logits)

            # Sample one index based on probabilities
            probs_1d = probs.flatten()

            # Normalize to sum to 1, handle numerical edge case
            probs_1d = probs_1d / (np.sum(probs_1d) + 1e-12)

            # Use the dimension of the logits (which is model's vocab_size)
            choices = np.arange(len(probs_1d))
            next_token_id = np.random.choice(choices, p=probs_1d)

            # 6. Append the new token to the sequence
            # [1, current_len + 1]
            next_token_id_array = np.array([[next_token_id]], dtype=np.int32)
            current_ids = np.concatenate([current_ids, next_token_id_array], axis=1)

        # Return only the newly generated tokens.
        # For empty prompts, we injected a dummy [0] at index 0
        # so skip it; otherwise skip the prompt prefix.
        if is_empty_prompt:
            return current_ids[0, 1:]
        else:
            # Return only the generated tokens, not the prompt.
            return current_ids[0, prompt_len:]

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerical stable softmax."""
        if x.size == 0:
            return x
        # Numerical stability: subtract max
        max_val = np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(x - max_val)
        return e_x / (np.sum(e_x, axis=-1, keepdims=True) + 1e-12)
