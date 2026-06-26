"""Shared weight initialization utilities for all backends.

Provides a single canonical initialization scheme so that models across
NumPy, PyTorch, Triton, and CUDA backends start with (approximately)
identical initial weights. All distributions use the same seed, ensuring
reproducible results.

Initialization Scheme
---------------------

All weight matrices use **Kaiming (Xavier) uniform** with bound:
    bound = sqrt(6 / (fan_in + fan_out))
    values in [-bound, +bound]
This is the canonical formula used by torch.nn.Linear.

Biases are initialized to **zeros** (identically across all backends).

LayerNorm/RMSNorm gamma parameters are always initialized to **ones**.
Gates are always initialized to **zeros**.

This scheme ensures:
1. Round-trip test validity (save from any backend, load into any other)
2. Cross-backend weight diff tests measure training behavior, not init
3. Deterministic results from a given seed

Seed Handling
-------------

Each major component uses a different offset from the model seed:

  Component          | Seed offset   | Purpose
  --------------------|---------------|-------------------------------
  token embedding    | 0             | Vocabulary lookup table
  transformer blocks | seed+100+n    | Per-layer unique seed
  output SwiGLU      | seed+200      | Output projection FFN
  output projection  | seed+300      | Linear projection to vocab

Within transformer blocks:
  - MHA (Wq/Wk/Wv/Wo): seed+100+layer + {0,1,2,3}
  - MoE router/router: seed+100+layer + {2,3,4,5} → uses seed+6
  - MoE expert weights: seed+100+layer + {7,8,...} → uses seed+7

Reference
---------
He et al. "Delivering Deep Learning" (2015) — Kaiming initialization
Glorot & Bengio "Understanding the difficulty of..." (2010) — Xavier
"""  # noqa: D400

from __future__ import annotations

import math

import numpy as np
import torch


def _xavier_uniform_tensor(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Xavier uniform initialization for weight matrices.

    Computes the bound as::

        bound = sqrt(6 / (fan_in + fan_out))

    Parameters
    ----------
    shape : tuple[int, ...]
        Desired tensor shape (row_dim, col_dim, ...) — only first two dims matter.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Float32 array with values in [-bound, +bound].
    """
    fan_in, fan_out = shape[0], shape[1]
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    arr = rng.uniform(-bound, bound, size=shape).astype(np.float32)
    return arr


def _xavier_uniform_torch(
    shape: tuple[int, ...], seed: int, device: str = "cpu"
) -> torch.Tensor:
    """Xavier uniform initialization for torch tensors.

    Parameters
    ----------
    shape : tuple[int, ...]
        Desired tensor shape.
    seed : int
        Random seed for reproducibility.
    device : str
        Device to create the tensor on.

    Returns
    -------
    torch.Tensor
        Float32 tensor with values in [-bound, +bound].
    """
    fan_in, fan_out = shape[0], shape[1]
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    t = torch.empty(shape, dtype=torch.float32, device=device)
    torch.nn.init.uniform_(t, -bound, bound, generator=torch.Generator().manual_seed(seed))
    return t


def create_embedding(vocab_size: int, embed_dim: int, seed: int) -> torch.Tensor:
    """Create token embedding weights.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    embed_dim : int
        Embedding dimension.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor, shape (V, D)
        Token embedding weight matrix.
    """
    D = embed_dim
    V = vocab_size
    return torch.empty(V, D).uniform_(-math.sqrt(1.0 / D), math.sqrt(1.0 / D))