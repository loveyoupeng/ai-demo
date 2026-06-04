from __future__ import annotations

from typing import cast

import numpy as np
from model.attention import MultiHeadAttention
from model.numpy.layers import NumPyLayerNorm
from model.numpy.moe import MoELayer


class NumPyTransformerBlock:
    r"""
    A single Transformer Decoder Block (NumPy).

    Combines Multi-Head Attention (MHA) and a Mixture of Experts (MoE) layer,
    wrapped in residual connections and Layer Normalization.

    Uses a pre-normal architecture: LayerNorm is applied BEFORE the sub-layers.
    Forward: x + MHA(LN1(x)) then x + MoE(LN2(x))

    Dimension tracking:
    - Input x: [B, L, D]
    - MHA Output: [B, L, D]
    - MoE Output: [B, L, D]
    - Final Block Output: [B, L, D]
    """

    def __init__(
        self,
        embed_dim: int,
        mha: MultiHeadAttention,
        moe: MoELayer,
    ):
        self.embed_dim: int = embed_dim
        self.ln1: NumPyLayerNorm = NumPyLayerNorm(embed_dim)
        self.ln2: NumPyLayerNorm = NumPyLayerNorm(embed_dim)
        self.mha: MultiHeadAttention = mha
        self.moe: MoELayer = moe

    def forward(
        self,
        x: np.ndarray,
        mask: np.ndarray | None = None,
        use_cache: bool = False,
        cache_idx: int | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """
        Args:
            x: [Batch, Seq_Len, Embed_Dim]
            mask: Causal mask [Seq_Len, Seq_Len]
            use_cache: Whether to use/update KV cache
            cache_idx: Index of the current token for KV cache update
        Returns:
            output: [Batch, Seq_Len, Embed_Dim]
            cache: Dictionary containing intermediate values for backward pass
        """
        # 1. Self-Attention Sub-layer (Pre-Norm) - x = x + MHA(LN(x))
        residual1: np.ndarray = x
        ln1_x: np.ndarray = self.ln1.forward(x)
        mha_out: np.ndarray
        mha_cache: dict[str, object]
        mha_out, mha_cache = self.mha.forward(ln1_x, mask)
        x_after_mha: np.ndarray = residual1 + mha_out

        # 2. Feed-Forward / MoE Sub-layer (Pre-Norm) - x = x + MoE(LN(x))
        residual2: np.ndarray = x_after_mha
        ln2_x: np.ndarray = self.ln2.forward(x_after_mha)
        moe_out: np.ndarray
        moe_cache: dict[str, object]
        moe_out, moe_cache = self.moe.forward(ln2_x)
        x_after_moe: np.ndarray = residual2 + moe_out

        cache: dict[str, object] = {
            "ln1_input": x,
            "mha_input": ln1_x,
            "mha_output": mha_out,
            "residual1": residual1,
            "ln2_input": x_after_mha,
            "moe_input": ln2_x,
            "moe_output": moe_out,
            "residual2": residual2,
            "mha_cache": mha_cache,
            "moe_cache": moe_cache,
        }

        return x_after_moe, cache

    def backward(
        self,
        grad_output: np.ndarray,
        cache: dict[str, object],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Backward pass for NumPy TransformerBlock.

        Args:
            grad_output: Gradient w.r.t. output [B, L, D]
            cache: Cache dict from forward pass
        Returns:
            dx: Gradient w.r.t. input x [B, L, D]
            combined_grads: Dict of gradients for all parameters
        """
        # 1. Gradient w.r.t. MoE sub-layer
        d_moe_out: np.ndarray = grad_output
        dx_moe: np.ndarray
        grads_moe: dict[str, np.ndarray]
        dx_moe, grads_moe = self.moe.backward(
            cast(np.ndarray, cache["moe_input"]), d_moe_out, cast(dict[str, object], cache["moe_cache"])
        )

        # 2. Gradient w.r.t. residual 2 and ln2
        # d_ln2_input = d_residual2 + dx_moe
        d_ln2_input: np.ndarray = grad_output + dx_moe
        dx_ln2: np.ndarray
        grads_ln2: dict[str, np.ndarray]
        dx_ln2, grads_ln2 = self.ln2.backward(d_ln2_input)

        # 3. Gradient w.r.t. MHA sub-layer
        # d_residual1 = d_ln2_input (since residual2 = x_after_mha)
        # d_mha_out = d_ln2_input
        d_mha_out: np.ndarray = d_ln2_input

        mha_cache = cache["mha_cache"]
        mask: np.ndarray | None = mha_cache.get("mask") if isinstance(mha_cache, dict) else None
        dx_mha: np.ndarray
        grads_mha: dict[str, np.ndarray]
        dx_mha, grads_mha = self.mha.backward(
            x=cast(np.ndarray, cache["mha_input"]),
            d_out=d_mha_out,
            mask=mask,
            Q=mha_cache.get("Q") if isinstance(mha_cache, dict) else None,
            K=mha_cache.get("K") if isinstance(mha_cache, dict) else None,
            V=mha_cache.get("V") if isinstance(mha_cache, dict) else None,
            attn_weights=mha_cache.get("attn_weights") if isinstance(mha_cache, dict) else None,
            context=mha_cache.get("context") if isinstance(mha_cache, dict) else None,
        )

        # 4. Gradient w.r.t. residual 1 and ln1
        # d_ln1_input = d_residual1 + dx_mha
        d_ln1_input: np.ndarray = d_ln2_input + dx_mha
        dx_ln1: np.ndarray
        grads_ln1: dict[str, np.ndarray]
        dx_ln1, grads_ln1 = self.ln1.backward(d_ln1_input)

        # 5. Total gradient w.r.t. input x
        dx: np.ndarray = dx_ln1

        # Combine all gradients
        combined_grads: dict[str, np.ndarray] = {}
        for k, v in grads_ln1.items():
            combined_grads[f"ln1.{k}"] = v
        for k, v in grads_mha.items():
            combined_grads[f"mha.{k}"] = v
        for k, v in grads_moe.items():
            combined_grads[f"moe.{k}"] = v
        for k, v in grads_ln2.items():
            combined_grads[f"ln2.{k}"] = v

        # Store gradients as attributes for get_grads()
        self._grads: dict[str, np.ndarray] = combined_grads

        return dx, combined_grads

    def get_params(self) -> dict[str, np.ndarray]:
        """Return a flat dict of all learnable parameters."""
        params: dict[str, np.ndarray] = {}
        for k, v in self.ln1.get_params().items():
            params[f"ln1.{k}"] = v
        for k, v in self.ln2.get_params().items():
            params[f"ln2.{k}"] = v
        for k, v in self.mha.get_params().items():
            params[f"mha.{k}"] = v
        for k, v in self.moe.get_params().items():
            params[f"moe.{k}"] = v
        return params
