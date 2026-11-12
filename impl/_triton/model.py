# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""TritonModel — complete decoder-only transformer on Triton kernels.

Track intent: **how to write the kernels** — memory-traffic-aware GPU
kernels (online softmax / FlashAttention in ``flash_attn.py``) over the
PyTorch model skeleton. The RoPE operator is imported from the PyTorch track
via the documented shared seam (``docs/architecture/seam_triton_to_torch.md``).

Forward (same layout as the NumPy/PyTorch tracks):

    tokens (B, S)
    → embed_tokens                          (B, S, D)
    → TritonDecoderStack (n_layers blocks)  (B, S, D)
    → final RMSNorm                         (B, S, D)
    → lm_head (linear D → V)                (B, S, V)  = logits

Parameter keys follow the ``shared.constants.Keys`` scheme (the
cross-backend checkpoint contract).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from impl._triton.transformer import (
    TritonDecoderStack,
    TritonGatedDeltaNet,
    TritonMixtureOfExperts,
    copy_in_place,
)
from shared.checkpoint import load_checkpoint
from shared.config import TransformerConfig
from shared.constants import Attn, DeltaNet, Keys, LayerNorm, Mlp
from shared.registry import ParameterRegistry


class TritonModel(nn.Module):
    """Complete decoder-only transformer using Triton kernels.

    The attention core runs on the Triton SDPA kernel and the feed-forward on
    the Triton SwiGLU kernel; embedding, norms, and the lm_head are plain
    PyTorch. Parameters are addressed by the ``shared.constants.Keys``
    scheme so weights load across tracks.

    Shape letters (CONTEXT.md → "Shape notation"): B = batch size (the
    speculative engine is B=1), S = sequence length, D = embed_dim, V =
    vocab_size, k = the drafter's block size.
    """

    def __init__(self, config: TransformerConfig, params: Mapping[str, np.ndarray] | None = None) -> None:
        """Build the model from a :class:`TransformerConfig`.

        params: optional checkpoint arrays (the shared Keys scheme)
        **injected at construction** — validated against the registry, then
        threaded DOWN the chain (embedding → stack → block → RMSNorm /
        attention / FFN / MoE), so every operator is born with its trained
        tensor (the inference path; see ``from_checkpoint``). When None
        (default) every operator keeps its seed-initialized values (the
        training path). There is no third lifecycle: weights are either
        injected here or trained in place.
        """
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim

        if params is not None:
            # Fail fast on a stale/mismatched checkpoint BEFORE building, so a
            # bad slice can never silently leave an operator at its defaults.
            ParameterRegistry(config).validate(params)
        p = params

        torch.manual_seed(config.seed)

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        if p is not None:
            copy_in_place(self.embedding.weight, p[Keys.embed()])  # (V, D) — same layout

        # Decoder stack (n_layers TritonTransformerBlocks) — each block gets its slice.
        self.stack = TritonDecoderStack(config, p)

        self.final_norm = nn.RMSNorm(config.embed_dim, eps=config.norm_eps)
        if p is not None:
            copy_in_place(self.final_norm.weight, p[Keys.final_norm()])  # (D,) gain

        # lm_head: (V, D) per nn.Linear convention, no bias; checkpoint keeps (D, V).
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        if p is not None:
            copy_in_place(self.lm_head.weight, p[Keys.lm_head()], transpose=True)

        # Cast model parameters to the configured dtype — BF16 AMP: the Triton
        # kernels' matmuls then run on the GPU tensor cores in BF16 (~2x FP32 on
        # Ampere+); the checkpoint itself stays FP32 (see design.md
        # "Precision & tensor cores"; serving casts back to float32).
        if isinstance(config.dtype, torch.dtype):
            self.to(dtype=config.dtype)

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path, device: torch.device | str | None = None) -> TritonModel:
        """Construct a model from a checkpoint directory (the explicit
        inference factory).

        Reads ``config.json`` + ``model.npz`` and injects the trained
        arrays at construction. ``device`` optionally moves the tensors
        (e.g. ``"cuda"``); defaults to the constructor device.
        """
        arrays, cfg = load_checkpoint(checkpoint_dir)
        if cfg is None:
            raise ValueError(f"checkpoint {checkpoint_dir} is missing config.json")
        model = cls(cfg) if device is None else cls(cfg).to(device)
        model.load_from_numpy_dict(arrays)
        return model

    def make_cache(self, batch_size: int) -> list[list[dict[str, torch.Tensor]]]:
        """Create the per-layer, per-mixer cache for the per-token step path.

        Same shape contract as ``impl._torch.layers.TorchModel.make_cache``
        (which mirrors ``impl._np.model.NumPyModel.make_cache``): one LIST
        per layer with one dict per mixer — DeltaNet mixers get
        {"s": (B, H, hd, hd), "s_hist": []}; the attention mixer gets
        {"k": (B, G, 0, hd), "v": (B, G, 0, hd)} — K/V *per group*.

        Returns: one list per block (n_layers entries).
        """
        B = batch_size
        G = self.config.kv_heads
        H = self.config.n_heads
        hd = self.embed_dim // H
        device = self.embedding.weight.device
        dtype = self.embedding.weight.dtype
        caches: list[list[dict[str, Any]]] = []
        for _i in range(self.config.n_layers):
            layer_caches: list[dict[str, Any]] = []
            for m in range(self.config.n_mixers()):
                if self.config.mixer_kind(m) == "d":
                    layer_caches.append({"s": torch.zeros(B, H, hd, hd, device=device, dtype=dtype), "s_hist": []})
                elif self.config.has_mla_at(_i):
                    # MLA (ADR 0009): the cache stores ONE latent c per
                    # token — (B, L, 0) — instead of per-group K/V.
                    L = self.config.mla_latent_dim
                    layer_caches.append({"c": torch.zeros(B, L, 0, device=device, dtype=dtype)})
                else:
                    layer_caches.append(
                        {
                            "k": torch.zeros(B, G, 0, hd, device=device, dtype=dtype),
                            "v": torch.zeros(B, G, 0, hd, device=device, dtype=dtype),
                        }
                    )
            caches.append(layer_caches)
        return caches

    def forward_prefill(
        self, input_ids: torch.Tensor, cache: list[list[dict[str, torch.Tensor]]] | None = None, position: int = 0
    ) -> tuple[torch.Tensor, list[list[dict[str, torch.Tensor]]]]:
        """Run a full-sequence forward and fill the per-layer KV cache.

        Mirrors ``impl._torch.layers.TorchModel.forward_prefill`` (which
        mirrors ``impl._np.model.NumPyModel.forward_prefill``): the cache is
        backfilled from the attention state the blocks capture during this
        very forward pass — no second pass needed.

        input_ids: (B, S) int token IDs.
        cache: optional pre-allocated cache (make_cache); created when None.

        Returns (logits (B, S, V), cache) ready for per-token steps.
        """
        self._move_to_device(input_ids)
        B, S = input_ids.shape
        if cache is None:
            cache = self.make_cache(B)
        states: list[list[dict[str, Any]]] = []
        x = self.embedding(input_ids)  # (B, S, D)
        for block in self.stack.blocks:
            block_res = block.forward_state(x, position)
            x = block_res.value
            states.append(block_res.ctx.display)  # type: ignore[arg-type]
        x_final = self.final_norm(x)  # (B, S, D)
        logits = self.lm_head(x_final)  # (B, S, V)
        for i, mixer_states in enumerate(states):
            layer_cache = cache[i]
            for m, mixer_state in enumerate(mixer_states):
                mixer_cache: dict[str, Any] = layer_cache[m]
                if self.config.mixer_kind(m) == "d":
                    mixer_cache["s"] = mixer_state["s_final"]
                    mixer_cache["s_hist"] = [
                        mixer_state["s_hist"][:, :, j] for j in range(mixer_state["s_hist"].shape[2])
                    ]  # type: ignore[assignment]
                elif self.config.has_mla_at(i):
                    # MLA (ADR 0009): backfill the latent cache (B, L, S).
                    mixer_cache["c"] = mixer_state["c_group"].transpose(1, 2)  # (B, L, S)
                else:
                    mixer_cache["k"] = mixer_state["k_group"]  # (B, G, S, hd)
                    mixer_cache["v"] = mixer_state["v_group"]  # (B, G, S, hd)
        return logits, cache

    def forward_step(
        self,
        input_ids: torch.Tensor,
        position: int,
        cache: list[list[dict[str, torch.Tensor]]],
    ) -> torch.Tensor:
        """Process ONE token per batch row against the cached K/V.

        Mirrors ``impl._torch.layers.TorchModel.forward_step`` (which
        mirrors ``impl._np.model.NumPyModel.forward_step``) — the
        O(1)-per-token inference path.

        input_ids: (B, 1) int token IDs.
        position: the absolute token index of the token (0-based).
        cache: the per-layer cache from ``make_cache``/``forward_prefill``.

        Returns: logits (B, 1, V).
        """
        self._move_to_device(input_ids)
        x = self.embedding(input_ids)  # (B, 1, D)
        stack_out = self.stack.forward_step(x, position, cache)  # (B, 1, D)
        x_final = self.final_norm(stack_out)  # (B, 1, D)
        return self.lm_head(x_final)  # (B, 1, V)

    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        position: int,
        cache: list[list[dict[str, torch.Tensor]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a CHUNK of c tokens against the cached K/V (verification path).

        Mirrors ``impl._torch.layers.TorchModel.forward_chunk`` (which
        mirrors ``impl._np.model.NumPyModel.forward_chunk``) — the target
        side of speculative decoding (ADR 0003): one call runs a draft block
        of c tokens through the whole stack, appending every token's K/V in
        a single pass and scoring all c positions in parallel. Naive cache
        only (the lossless verification contract needs full precision).

        input_ids: (B, c) int token IDs, c >= 1.
        position: the absolute token index of the chunk's FIRST token (0-based).
        cache: the per-layer cache; the chunk's K/V are appended to each
            layer's cache before attention runs.

        Returns (logits (B, c, V), x_final (B, c, D)) — the final-norm
        hidden stream rides along because the drafter conditions on it at
        the next anchor.
        """
        self._move_to_device(input_ids)
        x = self.embedding(input_ids)  # (B, c, D)
        stack_out = self.stack.forward_chunk(x, position, cache)  # (B, c, D)
        x_final = self.final_norm(stack_out)  # (B, c, D)
        logits = self.lm_head(x_final)  # (B, c, V)
        return logits, x_final

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
            t[Keys.ln(layer_idx, LayerNorm.POST_ATTENTION)] = block.post_attention_layernorm.weight
            for m, (norm, mixer) in enumerate(zip(block.mixer_norms, block.mixers, strict=True)):
                t[Keys.mixer_ln(layer_idx, m)] = norm.weight  # type: ignore[assignment]
                if isinstance(mixer, TritonGatedDeltaNet):
                    inner = mixer._impl
                    t[Keys.deltanet(layer_idx, m, DeltaNet.Q_PROJ)] = inner.q_proj.weight
                    t[Keys.deltanet(layer_idx, m, DeltaNet.K_PROJ)] = inner.k_proj.weight
                    t[Keys.deltanet(layer_idx, m, DeltaNet.V_PROJ)] = inner.v_proj.weight
                    t[Keys.deltanet(layer_idx, m, DeltaNet.O_PROJ)] = inner.o_proj.weight
                    t[Keys.deltanet(layer_idx, m, DeltaNet.A_PROJ)] = inner.a_proj.weight
                    t[Keys.deltanet(layer_idx, m, DeltaNet.GATE_PROJ)] = inner.gate_proj.weight
                    t[Keys.deltanet(layer_idx, m, DeltaNet.DT_BIAS)] = inner.dt_bias
                    t[Keys.deltanet(layer_idx, m, DeltaNet.NORM)] = inner.norm.weight
                else:
                    if self.config.has_mla_at(layer_idx):
                        assert (
                            mixer.kv_down_proj is not None
                            and mixer.k_up_proj is not None
                            and mixer.v_up_proj is not None
                        )
                        for attr, module in (
                            (Attn.Q_PROJ, mixer.q_proj),
                            (Attn.O_PROJ, mixer.o_proj),
                            (Attn.KV_DOWN_PROJ, mixer.kv_down_proj),
                            (Attn.K_UP_PROJ, mixer.k_up_proj),
                            (Attn.V_UP_PROJ, mixer.v_up_proj),
                        ):
                            t[Keys.attn(layer_idx, m, attr)] = module.weight
                    else:
                        assert mixer.k_proj is not None and mixer.v_proj is not None
                        for proj, module in (
                            (Attn.Q_PROJ, mixer.q_proj),
                            (Attn.K_PROJ, mixer.k_proj),
                            (Attn.V_PROJ, mixer.v_proj),
                            (Attn.O_PROJ, mixer.o_proj),
                        ):
                            t[Keys.attn(layer_idx, m, proj)] = module.weight
            mlp = block.mlp
            if isinstance(mlp, TritonMixtureOfExperts):
                t[Keys.moe_gate(layer_idx)] = mlp.gate.weight
                for expert_idx, expert in enumerate(mlp.expert_list):
                    t[Keys.moe_expert(layer_idx, expert_idx, Mlp.GATE_PROJ)] = expert.gate_proj
                    t[Keys.moe_expert(layer_idx, expert_idx, Mlp.UP_PROJ)] = expert.up_proj
                    t[Keys.moe_expert(layer_idx, expert_idx, Mlp.DOWN_PROJ)] = expert.down_proj
                for s, shared in enumerate(mlp.shared_expert_list):
                    t[Keys.moe_shared_expert(layer_idx, s, Mlp.GATE_PROJ)] = shared.gate_proj
                    t[Keys.moe_shared_expert(layer_idx, s, Mlp.UP_PROJ)] = shared.up_proj
                    t[Keys.moe_shared_expert(layer_idx, s, Mlp.DOWN_PROJ)] = shared.down_proj
            else:
                t[Keys.ffn(layer_idx, Mlp.GATE_PROJ)] = mlp.gate_proj
                t[Keys.ffn(layer_idx, Mlp.UP_PROJ)] = mlp.up_proj
                t[Keys.ffn(layer_idx, Mlp.DOWN_PROJ)] = mlp.down_proj
        return t

    def save_as_numpy(self) -> dict[str, np.ndarray]:
        """Save all parameters as a NumPy-compatible flat dict (Keys scheme).

        Registry-driven: key set, expected shapes, and the ``nn.Linear``
        ``(out, in) → (in, out)`` transpose rule come from
        ``ParameterRegistry`` — this track only supplies storage
        via the track's binding map.
        """
        tensors = ParameterRegistry(self.config).bind(self._param_tensors())
        params: dict[str, np.ndarray] = {}
        for entry in ParameterRegistry(self.config).entries:
            array = tensors[entry.key].detach().cpu()
            # Checkpoint/page storage is float32 (np.bfloat16 doesn't exist);
            # the served model may run in BF16 (GPU tracks).
            if array.dtype == torch.bfloat16:
                array = array.float()
            array = array.numpy()
            params[entry.key] = array.T if entry.torch_transpose else array
        return params

    def get_all_parameters(self) -> dict[str, np.ndarray]:
        """All parameters as a flat numpy dict (interface parity with NumPy/CUDA)."""
        return self.save_as_numpy()

    def load_from_numpy_dict(self, params: Mapping[str, np.ndarray]) -> None:
        """Load parameters from a flat dict (inverse of ``save_as_numpy``).

        Validates against the registry first, so stale or mismatched
        checkpoints fail fast instead of half-loading.
        """
        registry = ParameterRegistry(self.config)
        registry.validate(params)
        tensors = ParameterRegistry(self.config).bind(self._param_tensors())
        for entry in registry.entries:
            loaded = torch.from_numpy(params[entry.key])
            if entry.torch_transpose:
                loaded = loaded.T  # checkpoint (in, out) → nn.Linear (out, in)
            loaded = loaded.contiguous().to(tensors[entry.key].dtype)
            tensors[entry.key].data.copy_(loaded)
