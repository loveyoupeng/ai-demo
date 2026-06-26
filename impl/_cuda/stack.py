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

    def __init__(
        self,
        n_layers: int,
        embed_dim: int,
        n_heads: int,
        n_experts: int,
        ff_dim: int,
        k: int = 2,
        rope_dim: int = 0,
    ) -> None:
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_experts = n_experts
        self.ff_dim = ff_dim
        self.k = k
        self.rope_dim = rope_dim
        self.head_dim = embed_dim // n_heads

        # Create transformer blocks in sequence
        # Each block uses the same architecture (no per-layer parameter separation)
        self.blocks = [
            CuTransformerBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_experts=n_experts,
                ff_dim=ff_dim,
                k=k,
                rope_dim=rope_dim,
                seed=100 + layer_idx,  # Same offset scheme as NumPy/PyTorch
            )
            for layer_idx in range(n_layers)
        ]

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
