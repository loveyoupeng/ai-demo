"""TritonModel — complete decoder-only transformer on Triton kernels.

Forward (same layout as the NumPy/PyTorch tracks):

    tokens (B, S)
    → embed_tokens                          (B, S, D)
    → TritonDecoderStack (n_layers blocks)  (B, S, D)
    → final RMSNorm                         (B, S, D)
    → lm_head (linear D → V)                (B, S, V)  = logits

Parameter keys follow the ``shared.constants.Keys`` scheme (the
cross-backend checkpoint contract).
"""

import numpy as np
import torch
import torch.nn as nn

from impl._triton.transformer import (
    TritonDecoderStack,
    TritonMixtureOfExperts,
)
from shared.config import TransformerConfig
from shared.constants import Attn, Keys, LayerNorm, Mlp
from shared.registry import ParameterRegistry


class TritonModel(nn.Module):
    """Complete decoder-only transformer using Triton kernels.

    The attention core runs on the Triton SDPA kernel and the feed-forward on
    the Triton SwiGLU kernel; embedding, norms, and the lm_head are plain
    PyTorch. Parameters are addressed by the ``shared.constants.Keys``
    scheme so weights load across tracks.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Build the model from a :class:`TransformerConfig`."""
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim

        torch.manual_seed(config.seed)

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.stack = TritonDecoderStack(config)
        self.final_norm = nn.RMSNorm(config.embed_dim, eps=config.norm_eps)
        # lm_head: (in=D, out=V) per nn.Linear convention, no bias
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def _move_to_device(self, x: torch.Tensor) -> None:
        """Move all parameters to x's device/dtype (triton kernels need it)."""
        if not x.is_cuda:
            return
        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point or x.dtype.is_complex else None
        if dtype is None:
            self.embedding.to(device)
            self.final_norm.to(device)
            self.lm_head.to(device)
        else:
            self.embedding.to(device, dtype)
            self.final_norm.to(device, dtype)
            self.lm_head.to(device, dtype)
        self.stack._move_to_device(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, S) int → logits: (B, S, V)."""
        self._move_to_device(x)
        # (B, S) → (B, S, D)
        x = self.embedding(x)
        # (B, S, D) → (B, S, D)
        x = self.stack(x)
        # (B, S, D) → (B, S, D)
        x = self.final_norm(x)
        # (B, S, D) → (B, S, V)
        return self.lm_head(x)

    def _param_tensors(self) -> dict[str, torch.Tensor]:
        """Storage binding: registry key → owning tensor (the track's only traversal)."""
        t: dict[str, torch.Tensor] = {
            Keys.embed(): self.embedding.weight,
            Keys.final_norm(): self.final_norm.weight,
            Keys.lm_head(): self.lm_head.weight,
        }
        for layer_idx, block in enumerate(self.stack.blocks):
            t[Keys.ln(layer_idx, LayerNorm.INPUT)] = block.input_layernorm.weight
            t[Keys.ln(layer_idx, LayerNorm.POST_ATTENTION)] = block.post_attention_layernorm.weight
            attn = block.self_attn
            for proj, module in (
                (Attn.Q_PROJ, attn.q_proj),
                (Attn.K_PROJ, attn.k_proj),
                (Attn.V_PROJ, attn.v_proj),
                (Attn.O_PROJ, attn.o_proj),
            ):
                t[Keys.attn(layer_idx, proj)] = module.weight
            mlp = block.mlp
            if isinstance(mlp, TritonMixtureOfExperts):
                t[Keys.moe_gate(layer_idx)] = mlp.gate.weight
                for expert_idx, expert in enumerate(mlp.expert_list):
                    t[Keys.moe_expert(layer_idx, expert_idx, Mlp.GATE_PROJ)] = expert.gate_proj
                    t[Keys.moe_expert(layer_idx, expert_idx, Mlp.UP_PROJ)] = expert.up_proj
                    t[Keys.moe_expert(layer_idx, expert_idx, Mlp.DOWN_PROJ)] = expert.down_proj
            else:
                t[Keys.ffn(layer_idx, Mlp.GATE_PROJ)] = mlp.gate_proj
                t[Keys.ffn(layer_idx, Mlp.UP_PROJ)] = mlp.up_proj
                t[Keys.ffn(layer_idx, Mlp.DOWN_PROJ)] = mlp.down_proj
        return t

    def save_as_numpy(self) -> dict[str, np.ndarray]:
        """Save all parameters as a NumPy-compatible flat dict (Keys scheme).

        Registry-driven: key set, expected shapes, and the ``nn.Linear``
        ``(out, in) → (in, out)`` transpose rule come from
        ``ParameterRegistry`` — this track only supplies storage.
        """
        tensors = self._param_tensors()
        params: dict[str, np.ndarray] = {}
        for entry in ParameterRegistry(self.config).entries:
            array = tensors[entry.key].detach().cpu().numpy()
            params[entry.key] = array.T if entry.torch_transpose else array
        return params

    def get_all_parameters(self) -> dict[str, np.ndarray]:
        """All parameters as a flat numpy dict (interface parity with NumPy/CUDA)."""
        return self.save_as_numpy()

    def load_from_numpy_dict(self, params: dict[str, np.ndarray]) -> None:
        """Load parameters from a flat dict (inverse of ``save_as_numpy``).

        Validates against the registry first, so stale or mismatched
        checkpoints fail fast instead of half-loading.
        """
        registry = ParameterRegistry(self.config)
        registry.validate(params)
        tensors = self._param_tensors()
        for entry in registry.entries:
            loaded = torch.from_numpy(params[entry.key])
            if entry.torch_transpose:
                loaded = loaded.T  # checkpoint (in, out) → nn.Linear (out, in)
            loaded = loaded.contiguous().to(tensors[entry.key].dtype)
            tensors[entry.key].data.copy_(loaded)
