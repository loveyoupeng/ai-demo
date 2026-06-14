from __future__ import annotations

from typing import Any, Optional, cast

import numpy as np
from model.attention import MultiHeadAttention
from model.moe import MoELayer
from model.layers import LayerNorm

_CacheDict = dict[str, Any]


class TransformerBlock(object):
    """
    A single Transformer Decoder Block.
    Combines Multi-Head Attention (MHA) and a Mixture of Experts (MoE) layer,
    wrapped in residual connections and Layer Normalization.

    Dimension tracking:
    - Input $x$: $[B, L, D]$
    - MHA Output: $[B, L, D]$
    - MoE Output: $[B, L, D]$
    - Final Block Output: $[B, L, D]$
    """

    def __init__(self, embed_dim: int, mha: MultiHeadAttention, moe: MoELayer):
        self.mha = mha
        self.moe = moe

        # Pre-norm architecture: LayerNorm is applied BEFORE the sub-layers
        self.ln1 = LayerNorm(embed_dim)
        self.ln2 = LayerNorm(embed_dim)

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
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
        # 1. Self-Attention Sub-layer (Pre-Norm)
        # x = x + MHA(LN(x))
        residual1 = x
        ln1_x = self.ln1.forward(x)
        mha_out, mha_cache = self.mha.forward(
            ln1_x, mask=mask, use_cache=use_cache, cache_idx=cache_idx
        )
        x_after_mha = residual1 + mha_out

        # 2. Feed-Forward / MoE Sub-layer (Pre-Norm)
        # x = x + MoE(LN(x))
        residual2 = x_after_mha
        ln2_x = self.ln2.forward(x_after_mha)
        moe_out, moe_cache = self.moe.forward(ln2_x)
        x_after_moe = residual2 + moe_out

        # Cache for backward pass
        cache = {
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
        self, grad_output: np.ndarray, cache: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Backward pass for TransformerBlock.
        """
        # 1. Gradient w.r.t. MoE sub-layer
        # x_after_moe = residual2 + moe_out
        # d_residual2 = grad_output
        # d_moe_out = grad_output
        d_residual2 = grad_output
        d_moe_out = grad_output

        dx_moe, grads_moe = self.moe.backward(
            cache["moe_input"], d_moe_out, cache["moe_cache"]
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
            x=cache["mha_input"],
            d_out=d_mha_out,
            mask=cache["mha_cache"].get("mask"),
            Q=cache["mha_cache"].get("Q"),
            K=cache["mha_cache"].get("K"),
            V=cache["mha_cache"].get("V"),
            attn_weights=cache["mha_cache"].get("attn_weights"),
            context=cache["mha_cache"].get("context"),
        )

        # 4. Gradient w.r.t. residual 1 and ln1
        # d_ln1_input = d_residual1 + dx_mha
        d_ln1_input = d_residual1 + dx_mha
        dx_ln1, grads_ln1 = self.ln1.backward(d_ln1_input)

        # 5. Total gradient w.r.t. input x
        dx = dx_ln1

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

        return dx, combined_grads

    def get_params(self) -> dict[str, np.ndarray]:
        params = {}
        for k, v in self.ln1.get_params().items():
            params[f"ln1.{k}"] = v
        for k, v in self.ln2.get_params().items():
            params[f"ln2.{k}"] = v
        for k, v in self.mha.get_params().items():
            params[f"mha.{k}"] = v
        for k, v in self.moe.get_params().items():
            params[f"moe.{k}"] = v
        return params

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        """
        Sets the model parameters from a dictionary.
        """
        for k, v in params.items():
            if k.startswith("ln1."):
                param_name = k.replace("ln1.", "")
                self.ln1.set_params({param_name: v})
            elif k.startswith("ln2."):
                param_name = k.replace("ln2.", "")
                self.ln2.set_params({param_name: v})
            elif k.startswith("mha."):
                param_name = k.replace("mha.", "")
                self.mha.set_params({param_name: v})
            elif k.startswith("moe."):
                param_name = k.replace("moe.", "")
                self.moe.set_params({param_name: v})


class Transformer:
    """
    The full Decoder-only Transformer model.
    Composed of a stack of Transformer blocks, token/positional embeddings,
    and a language model head.

    Dimension tracking:
    - Token/Pos Embedding: $[B, L, D]$
    - Transformer Block stack: $[B, L, D]$
    - LM Head Input: $[B, L, D]$
    - Logits: $[B, L, V]$
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

        from model.layers import TokenEmbedding

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # 1. Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)

        # 2. Transformer Stack
        self.blocks = []
        for _ in range(num_layers):
            mha = MultiHeadAttention(embed_dim, num_heads)
            moe = MoELayer(embed_dim, num_experts)
            self.blocks.append(TransformerBlock(embed_dim, mha, moe))

        # 3. Language Model Head (Linear projection to vocab)
        # [Embed_Dim, Vocab_Size]
        self.lm_head = np.random.randn(embed_dim, vocab_size) * 0.01

    def forward(
        self,
        input_ids: np.ndarray,
        mask: Optional[np.ndarray] = None,
        use_cache: bool = False,
        cache_idx: int | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
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

        # 1. Token Embedding (RoPE provides positional encoding in MHA)
        # [Batch, Seq_Len, Embed_Dim]
        x = self.token_embedding.forward(input_ids)

        # 2. Causal Mask
        if mask is None:
            # When using KV cache, only the new token is processed but the
            # mask must cover Q_Len x K_Len where K_Len includes accumulated
            # cached tokens.  Full causal mask at the effective sequence length.
            if use_cache and cache_idx is not None:
                effective_len = cache_idx
            else:
                effective_len = seq_len
            mask = np.tril(np.ones((effective_len, effective_len)))

        # 3. Transformer Blocks
        block_caches = []
        for block in self.blocks:
            block_out, block_cache = block.forward(
                x, mask=mask, use_cache=use_cache, cache_idx=cache_idx
            )
            block_caches.append(block_cache)
            x = block_out

        # 4. LM Head (Logits)
        # [Batch, Seq_Len, Vocab_Size]
        logits = np.dot(x, self.lm_head)

        # Cache for backward pass
        cache = {
            "token_embedding_input": input_ids,
            "pos_embedding_input": seq_len,
            "lm_head_input": x,
            "blocks_cache": block_caches,
        }

        return logits, cache

    def backward(
        self, grad_logits: np.ndarray, cache: dict[str, object]
    ) -> dict[str, np.ndarray]:
        """
        Returns all gradients collected from the backward pass.
        """
        # 1. LM Head
        lm_head_input = cast(np.ndarray, cache["lm_head_input"])
        d_lm_head = np.dot(
            lm_head_input.reshape(-1, self.embed_dim).T,
            grad_logits.reshape(-1, self.vocab_size),
        )
        d_lm_head_input = np.dot(grad_logits, self.lm_head.T)

        grads = {"lm_head": d_lm_head}

        # 2. Transformer Blocks
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
        self.token_embedding.backward(dx)
        for k, v in self.token_embedding.get_grads().items():
            grads[f"token_embedding.{k}"] = v

        return grads

    def get_params(self) -> dict[str, np.ndarray]:
        params = {}
        # Token Embedding
        for k, v in self.token_embedding.get_params().items():
            params[f"token_embedding.{k}"] = v
        # Transformer Blocks
        for i, block in enumerate(self.blocks):
            for k, v in block.get_params().items():
                params[f"blocks.{i}.{k}"] = v
        # LM Head
        params["lm_head"] = self.lm_head
        return params

    def set_params(self, params: dict[str, np.ndarray]) -> None:
        """
        Sets the model parameters from a dictionary.
        """
        for k, v in params.items():
            if k.startswith("token_embedding."):
                param_name = k.replace("token_embedding.", "")
                self.token_embedding.set_params({param_name: v})
            elif k.startswith("blocks."):
                # blocks.{i}.{sublayer}.{param_name}
                parts = k.split(".")
                i = int(parts[1])
                sublayer = parts[2]
                param_name = ".".join(parts[3:])

                block = self.blocks[i]
                if sublayer == "ln1":
                    block.ln1.set_params({param_name: v})
                elif sublayer == "ln2":
                    block.ln2.set_params({param_name: v})
                elif sublayer == "mha":
                    block.mha.set_params({param_name: v})
                elif sublayer == "moe":
                    block.moe.set_params({param_name: v})
            elif k == "lm_head":
                self.lm_head = v
