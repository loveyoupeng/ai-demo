"""Token embedding for the NumPy reference implementation.

The embedding is a learned lookup table from token IDs to dense vectors.
This module owns the forward pass *and* the analytic backward (the chain
rule for a row lookup).
"""

from __future__ import annotations

import numpy as np


class Embedding:
    """Token embedding: a learned lookup table from token IDs to dense vectors.

    The weight matrix ``W`` has one row per vocabulary entry. "Embedding" a
    token is just row lookup: the one-hot vector e_t of the token (shape (V,))
    multiplied by W gives e_t @ W = the t-th row, so the lookup table *is*
    the linear map from one-hot to the dense space.

    Forward
    -------
    input_ids : (B, S) int array, values in [0, V)
    Returns: (B, S, D)

    out[b, s, :] = W[input_ids[b, s]]

    Backward (chain rule for a row lookup)
    --------------------------------------
    out[b, s, :] = W[input_ids[b, s]]

    The only parameter is W, and each token's row is used wherever that token
    appears. Summing the upstream gradient over every occurrence of a token
    gives the row's gradient:

        dW[t] = sum_{(b,s) : input_ids[b,s] == t} dout[b, s, :]

    (B, S) → flat: dW[ids.flatten()] += dout.reshape(-1, D)  — this is exactly
    the matrix product d(one_hot) ^T @ dout, where one_hot is (B*S, V).
    """

    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        # (V, D) — one learned vector per token
        self.weight: np.ndarray = rng.normal(0.0, 1.0 / np.sqrt(embed_dim), (vocab_size, embed_dim)).astype(np.float32)

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Look up the embedding vector for every token.

        input_ids: (B, S) → output: (B, S, D).
        NumPy advanced indexing broadcasts the (B, S) index array over the
        rows of W, producing (B, S, D) directly.
        """
        return self.weight[input_ids]  # (B, S, D)

    def backward(self, dout: np.ndarray, input_ids: np.ndarray) -> np.ndarray:
        """Gradient of the loss w.r.t. the weight table.

        dout: (B, S, D) upstream gradient.
        input_ids: (B, S) the same indices used in forward.

        Returns: dW (V, D).

        np.add.at accumulates duplicate indices (a token may appear many
        times in the batch), which is exactly the "sum over occurrences"
        rule above.
        """
        dW = np.zeros_like(self.weight)  # (V, D)
        # Flatten the (B, S) index array and the (B, S, D) gradient so both
        # are 2-D: indices (B*S,) and gradients (B*S, D).
        flat_ids = input_ids.reshape(-1)  # (B*S,)
        flat_dout = dout.reshape(-1, dout.shape[-1])  # (B*S, D)
        np.add.at(dW, flat_ids, flat_dout)  # dW[t] += sum of flat_dout rows where flat_ids == t
        return dW
