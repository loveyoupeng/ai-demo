from __future__ import annotations

from typing import Any, Optional, cast

import torch
import torch.nn as nn
import numpy as np
from core.registry import registry
from model.pytorch.attention import PyTorchMultiHeadAttention
from model.pytorch.moe import PyTorchMoELayer


class PyTorchTransformerBlock(nn.Module):
    """
    A single Transformer Decoder Block (PyTorch).

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
        mha: PyTorchMultiHeadAttention,
        moe: PyTorchMoELayer,
    ):
        super().__init__()
        self.mha = mha
        self.moe = moe

        # Pre-norm architecture: LayerNorm is applied BEFORE the sub-layers
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache_idx: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
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
        batch_size, seq_len, _ = x.shape

        # 1. Self-Attention Sub-layer (Pre-Norm)
        # x = x + MHA(LN(x))
        residual1 = x
        ln1_x = self.ln1(x)
        mha_out, mha_cache = self.mha.forward(ln1_x, mask=mask)
        x_after_mha = residual1 + mha_out

        # 2. Feed-Forward / MoE Sub-layer (Pre-Norm)
        # x = x + MoE(LN(x))
        residual2 = x_after_mha
        ln2_x = self.ln2(x_after_mha)
        moe_out, moe_cache = self.moe.forward(ln2_x)
        x_after_moe = residual2 + moe_out

        # Cache for backward pass
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
        grad_output: torch.Tensor,
        cache: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Backward pass for TransformerBlock.
        """
        # 1. Gradient w.r.t. MoE sub-layer
        d_residual2 = grad_output
        d_moe_out = grad_output

        dx_moe, grads_moe = self.moe.backward(
            cast(torch.Tensor, cache["moe_input"]),
            d_moe_out,
            cast(dict[str, object], cache["moe_cache"]),
        )

        # 2. Gradient w.r.t. residual 2 and ln2
        # d_ln2_input = d_residual2 + dx_moe
        d_ln2_input = d_residual2 + dx_moe
        dx_ln2, grads_ln2 = self.ln2.backward(d_ln2_input)

        # 3. Gradient w.r.t. mha sub-layer
        # x_after_mha = residual1 + mha_out
        # d_residual1 = d_ln2_input (since residual2 = x_after_mha)
        # d_mha_out = d_ln2_input
        d_residual1 = d_ln2_input
        d_mha_out = d_ln2_input

        dx_mha, grads_mha = self.mha.backward(
            cast(torch.Tensor, cache["mha_input"]),
            d_mha_out,
            cast(torch.Tensor, cache["mha_cache"]).get("mask"),
        )

        # 4. Gradient w.r.t. residual 1 and ln1
        # d_ln1_input = d_residual1 + dx_mha
        d_ln1_input = d_residual1 + dx_mha
        dx_ln1, grads_ln1 = self.ln1.backward(d_ln1_input)

        # 5. Total gradient w.r.t. input x
        dx = dx_ln1

        # Combine all gradients
        combined_grads: dict[str, torch.Tensor] = {}
        for k, v in grads_ln1.items():
            combined_grads[f"ln1.{k}"] = v
        for k, v in grads_mha.items():
            combined_grads[f"mha.{k}"] = v
        for k, v in grads_moe.items():
            combined_grads[f"moe.{k}"] = v
        for k, v in grads_ln2.items():
            combined_grads[f"ln2.{k}"] = v

        return dx, combined_grads

    def get_params(self) -> dict[str, torch.Tensor]:
        params: dict[str, torch.Tensor] = {}
        for k, v in self.ln1.state_dict().items():
            params[f"ln1.{k}"] = v
        for k, v in self.ln2.state_dict().items():
            params[f"ln2.{k}"] = v
        for k, v in self.mha.get_params().items():
            params[f"mha.{k}"] = v
        for k, v in self.moe.get_params().items():
            params[f"moe.{k}"] = v
        return params

    def set_params(self, params: dict[str, object]) -> None:
        """
        Sets the model parameters from a dictionary.
        """
        for k, v in params.items():
            if k.startswith("ln1."):
                param_name = k[4:]  # len("ln1.") == 4
                val = v
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                with torch.no_grad():
                    if param_name == "weight":
                        self.ln1.weight.copy_(cast(torch.Tensor, val))
                    elif param_name == "bias":
                        self.ln1.bias.copy_(cast(torch.Tensor, val))
            elif k.startswith("ln2."):
                param_name = k[4:]
                val = v
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                with torch.no_grad():
                    if param_name == "weight":
                        self.ln2.weight.copy_(cast(torch.Tensor, val))
                    elif param_name == "bias":
                        self.ln2.bias.copy_(cast(torch.Tensor, val))
            elif k.startswith("mha."):
                param_name = k[4:]
                self.mha.set_params({param_name: v})
            elif k.startswith("moe."):
                param_name = k[4:]
                self.moe.set_params({param_name: v})


class PyTorchTransformer(nn.Module):
    """
    The full Decoder-only Transformer model (PyTorch).

    Composed of a stack of Transformer blocks, token/positional embeddings,
    and a language model head.

    Dimension tracking:
    - Token/Pos Embedding: [B, L, D]
    - Transformer Block stack: [B, L, D]
    - LM Head Input: [B, L, D]
    - Logits: [B, L, V]
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        num_experts: int,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # 1. Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Sequential(
            _PositionalEmbedding(max_seq_len, embed_dim),
            # Wrap so it behaves like a module with get_params/set_params
        )
        self._pos_embedding_module = _PositionalEmbedding(max_seq_len, embed_dim)

        # 2. Transformer Stack
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            mha = PyTorchMultiHeadAttention(embed_dim, num_heads)
            moe = PyTorchMoELayer(embed_dim, num_experts)
            self.blocks.append(PyTorchTransformerBlock(embed_dim, mha, moe))

        # 3. Language Model Head (Linear projection to vocab)
        # [Embed_Dim, Vocab_Size]
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Registry mappings
        registry.register("pytorch", "token_embedding.embedding.weight", "token_embedding.weight")
        registry.register("pytorch", "pos_embedding.pe", "pos_embedding.pe")
        registry.register("pytorch", "lm_head.weight", "lm_head.weight")

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache_idx: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        """
        Args:
            input_ids: [Batch, Seq_Len] integer token IDs
            mask: Causal mask [Seq_Len, Seq_Len]
            use_cache: Whether to use/update KV cache
            cache_idx: Index of the current token for KV cache update
        Returns:
            logits: [Batch, Seq_Len, Vocab_Size]
            cache: Dictionary containing intermediate values for backward pass
        """
        batch_size, seq_len = input_ids.shape

        # 1. Token + Positional Embeddings
        # [Batch, Seq_Len, Embed_Dim]
        x = self.token_embedding(input_ids)

        # Positional embedding from buffer
        pe = self._pos_embedding_module.get_buffer("pe")  # type: ignore[return-value]
        x = x + pe[:seq_len, :].unsqueeze(0)  # [1, Seq_Len, Embed_Dim]

        # 2. Causal Mask
        if mask is None:
            mask = torch.tril(torch.ones((seq_len, seq_len), dtype=x.dtype, device=x.device))

        # 3. Transformer Blocks
        block_caches: list[dict[str, object]] = []
        for block in self.blocks:
            block_out, block_cache = block.forward(x, mask=mask, use_cache=use_cache, cache_idx=cache_idx)
            block_caches.append(block_cache)
            x = block_out

        # 4. LM Head (Logits)
        # [Batch, Seq_Len, Vocab_Size]
        logits = self.lm_head(x)

        # Cache for backward pass
        cache: dict[str, object] = {
            "token_embedding_input": input_ids,
            "pos_embedding_input": seq_len,
            "lm_head_input": x,
            "blocks_cache": block_caches,
        }

        return logits, cache

    def backward(
        self, grad_logits: torch.Tensor, cache: dict[str, object]
    ) -> dict[str, torch.Tensor]:
        """
        Returns all gradients collected from the backward pass.
        """
        # 1. LM Head
        lm_head_input = cast(torch.Tensor, cache["lm_head_input"])
        # Re-execute lm_head forward to build the graph
        lm_head_output = self.lm_head(lm_head_input)
        loss = (lm_head_output * grad_logits).sum()
        loss.backward()

        grads: dict[str, torch.Tensor] = {}
        grads["lm_head"] = self.lm_head.weight.grad.clone() if self.lm_head.weight.grad is not None else torch.zeros_like(self.lm_head.weight)

        # 2. Transformer Blocks
        # Gradient w.r.t. lm_head_input
        d_lm_head_input = torch.matmul(grad_logits, self.lm_head.weight)

        dx = d_lm_head_input
        block_caches = cast(list[dict[str, object]], cache["blocks_cache"])
        for i in range(self.num_layers - 1, -1, -1):
            block = self.blocks[i]
            block_cache = block_caches[i]
            dx, block_grads = block.backward(dx, block_cache)

            # Prefix block grads
            for k, v in block_grads.items():
                grads[f"blocks.{i}.{k}"] = v

        # 3. Token Embedding
        # Backward through token embedding
        self.token_embedding.zero_grad()
        input_ids = cast(torch.Tensor, cache["token_embedding_input"])
        embed_output = self.token_embedding(input_ids)
        embed_loss = (embed_output * dx).sum()
        embed_loss.backward()

        if self.token_embedding.weight.grad is not None:
            grads["token_embedding.embedding.weight"] = self.token_embedding.weight.grad

        return grads

    def get_params(self) -> dict[str, torch.Tensor]:
        params: dict[str, torch.Tensor] = {}
        # Token Embedding
        if self.token_embedding.weight is not None:
            params["token_embedding.embedding.weight"] = self.token_embedding.weight
        # Transformer Blocks
        for i, block in enumerate(self.blocks):
            for k, v in block.get_params().items():
                params[f"blocks.{i}.{k}"] = v
        # LM Head
        params["lm_head"] = self.lm_head.weight
        # Positional Embedding (buffer, not a trainable param, but included for parity)
        pe = self._pos_embedding_module.get_buffer("pe")  # type: ignore[return-value]
        if pe is not None:
            params["pos_embedding.pe"] = pe
        return params

    def set_params(self, params: dict[str, object]) -> None:
        """
        Sets the model parameters from a dictionary.
        """
        for k, v in params.items():
            if k.startswith("token_embedding."):
                param_name = k[len("token_embedding."):]
                val = v
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                with torch.no_grad():
                    if self.token_embedding.weight is not None:
                        self.token_embedding.weight.copy_(cast(torch.Tensor, val))
            elif k.startswith("blocks."):
                # blocks.{i}.{sublayer}.{param_name}
                parts = k.split(".")
                i = int(parts[1])
                sublayer = parts[2]
                param_name = ".".join(parts[3:])

                block = self.blocks[i]
                if sublayer == "ln1":
                    val = v
                    if isinstance(val, np.ndarray):
                        val = torch.from_numpy(val)
                    with torch.no_grad():
                        if param_name == "weight":
                            cast(nn.LayerNorm, block.ln1).weight.copy_(cast(torch.Tensor, val))
                        elif param_name == "bias":
                            cast(nn.LayerNorm, block.ln1).bias.copy_(cast(torch.Tensor, val))
                elif sublayer == "ln2":
                    val = v
                    if isinstance(val, np.ndarray):
                        val = torch.from_numpy(val)
                    with torch.no_grad():
                        if param_name == "weight":
                            cast(nn.LayerNorm, block.ln2).weight.copy_(cast(torch.Tensor, val))
                        elif param_name == "bias":
                            cast(nn.LayerNorm, block.ln2).bias.copy_(cast(torch.Tensor, val))
                elif sublayer == "mha":
                    block.mha.set_params({param_name: v})
                elif sublayer == "moe":
                    block.moe.set_params({param_name: v})
            elif k == "lm_head":
                val = v
                if isinstance(val, np.ndarray):
                    val = torch.from_numpy(val)
                with torch.no_grad():
                    self.lm_head.weight.copy_(cast(torch.Tensor, val))


class _PositionalEmbedding(nn.Module):
    """Internal helper for positional encoding as a registered buffer."""

    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        pe = torch.zeros(max_seq_len, embed_dim, dtype=torch.float32)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * -(torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.get_buffer("pe")  # type: ignore[arg-type]
        return x + pe[: x.shape[1], :]

    def get_params(self) -> dict[str, torch.Tensor]:
        return {"pe": self.get_buffer("pe")}  # type: ignore[return-value]

    def set_params(self, params: dict[str, object]) -> None:
        pass
