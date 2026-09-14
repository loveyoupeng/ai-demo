"""CuDecoderStack — chained transformer blocks (F8).

Assembles n_layers of CuTransformerBlock into a decoder stack.

Architecture:
    x [B, S, D] → block_0 → block_1 → ... → block_{n_layers-1} → out [B, S, D]

    - No position embeddings (RoPE handles positional info inside attention)
    - No final RMSNorm (belongs to the parent model)
    - Post-norm gated residual with MoE in each block

Reference
---------
Vaswani et al. "Attention Is All You Need" (2017)
https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

import torch

from impl._cuda.block import CuTransformerBlock
from shared.config import TransformerConfig


class CuDecoderStack:
    """CUDA DecoderStack — chain of n_layers CuTransformerBlock modules.

    This class follows the same architecture as the NumPy/PyTorch DecoderStack
    but uses CuTransformerBlock for all computation (all CUDA kernels).

    It does NOT inherit from nn.Module — weights are stored as plain tensors
    (attributes or __dict__) for parity checking against the NumPy implementation.

    Parameters
    ----------
    n_layers : int
        Number of transformer blocks.
    embed_dim : int
        Input/output embedding dimension.
    n_heads : int
        Number of attention heads per block.
    n_experts : int
        Number of MoE experts per block.
    ff_dim : int
        Hidden dimension per MoE expert.
    k : int, optional
        Number of top experts to activate per token (default: 2).
    rope_dim : int, optional
        Number of head dimensions for RoPE (0 = disabled, default: 0).

    Attributes
    ----------
    blocks : list[CuTransformerBlock]
        List of transformer blocks chained in sequence.
    n_layers : int
        Number of blocks (len(blocks)).
    embed_dim : int
        Input/output embedding dimension.

    Forward
    -------
    x : torch.Tensor, shape (batch_size, seq_len, embed_dim) on CUDA

    Returns
    -------
    out : torch.Tensor, shape (batch_size, seq_len, embed_dim) on CUDA

    """

    def __init__(self, config: TransformerConfig) -> None:
        self.config = config
        self.n_layers = config.n_layers
        self.embed_dim = config.embed_dim
        self.head_dim = config.head_dim

        # Create transformer blocks in sequence (distinct seed per layer,
        # matching the NumPy/PyTorch offset scheme)
        self.blocks = [CuTransformerBlock(config, seed=100 + layer_idx) for layer_idx in range(config.n_layers)]

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through all stacked blocks.

        Chains n_layers of CuTransformerBlock sequentially.
        Input and output have the same shape: (B, S, D).

        Shape flow
        ----------
          x:      (B, S, D)
          block_0: (B, S, D)
          block_1: (B, S, D)
          ...
          block_{n-1}: (B, S, D) = out

        Parameters
        ----------
        x : torch.Tensor, shape (B, S, D)
            Input activations on CUDA device.
        positions : torch.Tensor, shape (S,) or None
            Position indices for RoPE. If None, each block uses arange(S).

        Returns
        -------
        out : torch.Tensor, shape (B, S, D)
            Output from the final block.

        """
        out = x
        for block in self.blocks:
            out = block.forward(out, positions=positions)
        return out
