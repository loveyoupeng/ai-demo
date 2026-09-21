"""CUDAModel — full decoder-only transformer.

Track intent: **the bare metal** — NVRTC-compiled kernels with explicit
launches and explicit device memory; no framework between you and the GPU.
The API mirrors the NumPy track's ``NumPyModel`` (plain numpy in, torch is
used only as the host-side driver).

Forward (same layout as the NumPy/PyTorch/Triton tracks):
    tokens → embedding → stack → final RMSNorm → lm_head → logits

Architecture:
    Input:  tokens [B, S] (int64)
    │
    ├→ Embedding table lookup       [B, S, D]
    ├→ CuDecoderStack (n_layers)    [B, S, D]
    ├→ RMSNorm (final_ln)           [B, S, D]
    └→ lm_head linear D → V         [B, S, V]
    │
    Output: logits [B, S, V]

This class does NOT inherit from nn.Module — weights are stored as plain
tensors (attributes) for parity checking against the NumPy implementation.
All parameters are addressed by the ``shared.constants.Keys`` scheme (the
cross-backend checkpoint contract).

Reference
---------
Vaswani et al. "Attention Is All You Need" (2017)
https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

import math

import numpy as np
import torch

from impl._cuda.layernorm import rmsnorm as _rmsnorm
from impl._cuda.stack import CuDecoderStack
from shared.config import TransformerConfig
from shared.constants import Attn, Keys, LayerNorm, Mlp
from shared.registry import ParameterRegistry


class CUDAModel:
    """CUDA decoder-only transformer — full model with embedding and lm_head.

    Uses CuDecoderStack for the transformer blocks and CUDA kernels for all
    heavy computation.

    Attributes
    ----------
    embedding_weights : torch.Tensor, shape (V, D)
        Token embedding weight matrix.
    final_norm_gamma : torch.Tensor, shape (D,)
        Final RMSNorm gamma.
    lm_head_weight : torch.Tensor, shape (D, V)
        Language-model head (linear D → V, no bias).
    stacking : CuDecoderStack
        The transformer blocks.
    """

    def __init__(self, config: TransformerConfig, stacking: CuDecoderStack | None = None) -> None:
        """Initialize CUDAModel from a :class:`TransformerConfig`.

        For parity checking, weights are initialized with the config's seed,
        mirroring the NumPy/PyTorch models.
        """
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        seed = config.seed

        # Create or use provided stack
        self.stacking = stacking if stacking is not None else CuDecoderStack(config)

        # Embedding table (V, D)
        gen = torch.Generator().manual_seed(seed)
        self.embedding_weights = torch.normal(
            0.0, 1.0 / math.sqrt(config.embed_dim), size=(config.vocab_size, config.embed_dim), generator=gen
        )
        self.embedding_weights.requires_grad_(True)

        # Final RMSNorm gamma (D,)
        self.final_norm_gamma = torch.ones(config.embed_dim, dtype=torch.float32)
        self.final_norm_gamma.requires_grad_(True)

        # lm_head (D, V) — no bias (Llama convention)
        gen2 = torch.Generator().manual_seed(seed + 300)
        self.lm_head_weight = torch.normal(
            0.0, 1.0 / math.sqrt(config.embed_dim), size=(config.embed_dim, config.vocab_size), generator=gen2
        )
        self.lm_head_weight.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the complete model.

        x: (B, S) int token IDs on CUDA → logits (B, S, V).

        Shape flow:
          x: (B, S)
            → embed: (B, S, D)
            → stack: (B, S, D)
            → RMSNorm: (B, S, D)
            → lm_head @: (B, S, V)
        """
        device = x.device
        # (B, S) → (B, S, D)
        x = self.embedding_weights.to(device)[x]
        # (B, S, D) → (B, S, D)
        x = self.stacking.forward(x)
        # (B, S, D) → (B, S, D)
        x = _rmsnorm(x, self.final_norm_gamma.to(device), eps=self.config.norm_eps)  # (B, S, D)
        # (B, S, D) @ (D, V) → (B, S, V)
        logits = x @ self.lm_head_weight.to(device)
        return logits  # (B, S, V)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Make the model callable — delegates to forward."""
        return self.forward(x)

    def _param_tensors(self) -> dict[str, torch.Tensor]:
        """Storage binding: registry key → owning tensor (the track's only traversal)."""
        t: dict[str, torch.Tensor] = {
            Keys.embed(): self.embedding_weights,
            Keys.final_norm(): self.final_norm_gamma,
            Keys.lm_head(): self.lm_head_weight,
        }
        for i, block in enumerate(self.stacking.blocks):
            t[Keys.ln(i, LayerNorm.INPUT)] = block.input_layernorm_gamma
            t[Keys.ln(i, LayerNorm.POST_ATTENTION)] = block.post_attention_layernorm_gamma
            t[Keys.attn(i, Attn.Q_PROJ)] = block.q_proj
            t[Keys.attn(i, Attn.K_PROJ)] = block.k_proj
            t[Keys.attn(i, Attn.V_PROJ)] = block.v_proj
            t[Keys.attn(i, Attn.O_PROJ)] = block.o_proj
            if self.config.has_moe():
                t[Keys.moe_gate(i)] = block.router
                for e in range(self.config.n_experts):
                    t[Keys.moe_expert(i, e, Mlp.GATE_PROJ)] = block.expert_gate_proj[e]
                    t[Keys.moe_expert(i, e, Mlp.UP_PROJ)] = block.expert_up_proj[e]
                    t[Keys.moe_expert(i, e, Mlp.DOWN_PROJ)] = block.expert_down_proj[e]
                for s in range(self.config.n_shared_experts):
                    t[Keys.moe_shared_expert(i, s, Mlp.GATE_PROJ)] = block.shared_gate_proj[s]
                    t[Keys.moe_shared_expert(i, s, Mlp.UP_PROJ)] = block.shared_up_proj[s]
                    t[Keys.moe_shared_expert(i, s, Mlp.DOWN_PROJ)] = block.shared_down_proj[s]
            else:
                t[Keys.ffn(i, Mlp.GATE_PROJ)] = block.gate_proj
                t[Keys.ffn(i, Mlp.UP_PROJ)] = block.up_proj
                t[Keys.ffn(i, Mlp.DOWN_PROJ)] = block.down_proj
        return t

    def get_all_parameters(self) -> dict[str, np.ndarray]:
        """Flat dict of all parameters keyed by ``shared.constants.Keys``.

        Registry-driven; CUDA stores every projection in the checkpoint's
        ``(in, out)`` layout, so no transposition is needed.
        """
        tensors = self._param_tensors()
        return {
            entry.key: tensors[entry.key].detach().cpu().numpy() for entry in ParameterRegistry(self.config).entries
        }

    def load_from_numpy_dict(self, params: dict[str, np.ndarray]) -> None:
        """Load parameters from a flat dict keyed by ``shared.constants.Keys``.

        Validates against the registry first (stale checkpoints fail fast);
        every key loads as a direct copy — CUDA uses the ``(in, out)`` layout.
        """
        registry = ParameterRegistry(self.config)
        registry.validate(params)
        tensors = self._param_tensors()
        for entry in registry.entries:
            loaded = torch.from_numpy(params[entry.key]).to(tensors[entry.key].dtype)
            tensors[entry.key].data.copy_(loaded)
