"""The NumPy reference model: decoder-only transformer.

Track intent: **how the math works** — every operator is hand-derived
(forward + closed-form analytic backward), with formula citations and shape
comments on each matrix operation. This track is the teaching reference the
other three tracks mirror.

Forward pass (the standard LLaMA-style layout):

    input_ids (B, S)
    → embed_tokens                          (B, S, D)
    → DecoderStack (n_layers blocks)        (B, S, D)
    → final RMSNorm                         (B, S, D)
    → lm_head (linear D → V)                (B, S, V)  = logits

Parameter keys follow the flat-dict scheme in ``shared.constants``
(HuggingFace Llama naming), which is the cross-backend checkpoint contract.
"""

import logging

import numpy as np

from impl._np.cross_entropy import CrossEntropyLoss
from impl._np.embedding import Embedding
from impl._np.ffn import SwiGLUFFN
from impl._np.layernorm import RMSNorm
from impl._np.moe import MixtureOfExperts
from impl._np.stack import DecoderStack
from shared.config import TransformerConfig
from shared.constants import ATTN_PROJS, FFN_PROJS, Attn, Keys, LayerNorm, Mlp
from shared.registry import ParameterRegistry

logger = logging.getLogger(__name__)


class NumPyModel:
    """Complete decoder-only transformer in NumPy.

    This class is the *reference implementation*: it owns all parameter
    arrays and the forward/backward/optimization loop, and it is the track
    that the PyTorch/Triton/CUDA tracks are verified against by the
    cross-backend parity tests.

    All parameters are stored as NumPy float32 arrays and addressed by the
    flat keys defined in ``shared.constants.Keys`` (see that module for the
    full key → shape table).
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Build the model from a :class:`TransformerConfig`.

        The config is the single source of truth for every dimension; the
        model derives head_dim, K/V group width, and FFN/MoE width from it.
        """
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        seed = config.seed

        # Embedding table: (V, D)
        self.embedding = Embedding(vocab_size=config.vocab_size, embed_dim=config.embed_dim, seed=seed)

        # Decoder stack (n_layers TransformerBlocks)
        self.stack = DecoderStack(config)

        # Final RMSNorm gamma: (D,)
        self.final_norm = RMSNorm(config.embed_dim, eps=config.norm_eps)

        # Language-model head: linear D → V (no bias, Llama convention)
        rng = np.random.default_rng(seed + 300)
        self.lm_head_weight: np.ndarray = rng.normal(
            0.0, 1.0 / np.sqrt(config.embed_dim), (config.embed_dim, config.vocab_size)
        ).astype(np.float32)

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Full forward pass.

        input_ids : (B, S) int token IDs.
        Returns: logits (B, S, V).

        Step shapes:
            embed        (B, S) → (B, S, D)
            stack        (B, S, D) → (B, S, D)
            final norm   (B, S, D) → (B, S, D)
            lm_head      (B, S, D) @ (D, V) → (B, S, V)
        """
        logits, _trace = self.forward_with_trace(input_ids)
        return logits

    def _param_arrays(self) -> dict[str, np.ndarray]:
        """Storage binding: registry key → owning array (the track's only traversal)."""
        p: dict[str, np.ndarray] = {
            Keys.embed(): self.embedding.weight,
            Keys.final_norm(): self.final_norm.gamma,
            Keys.lm_head(): self.lm_head_weight,
        }
        for i, block in enumerate(self.stack.layers):
            p[Keys.ln(i, LayerNorm.INPUT)] = block.input_layernorm.gamma
            p[Keys.ln(i, LayerNorm.POST_ATTENTION)] = block.post_attention_layernorm.gamma
            attn = block.self_attn
            p[Keys.attn(i, Attn.Q_PROJ)] = attn.q_proj
            p[Keys.attn(i, Attn.K_PROJ)] = attn.k_proj
            p[Keys.attn(i, Attn.V_PROJ)] = attn.v_proj
            p[Keys.attn(i, Attn.O_PROJ)] = attn.o_proj
            mlp = block.mlp
            if isinstance(mlp, MixtureOfExperts):
                p[Keys.moe_gate(i)] = mlp.gate
                for j, expert in enumerate(mlp.experts):
                    p[Keys.moe_expert(i, j, Mlp.GATE_PROJ)] = expert.gate_proj
                    p[Keys.moe_expert(i, j, Mlp.UP_PROJ)] = expert.up_proj
                    p[Keys.moe_expert(i, j, Mlp.DOWN_PROJ)] = expert.down_proj
                for s, shared in enumerate(mlp.shared_experts):
                    p[Keys.moe_shared_expert(i, s, Mlp.GATE_PROJ)] = shared.gate_proj
                    p[Keys.moe_shared_expert(i, s, Mlp.UP_PROJ)] = shared.up_proj
                    p[Keys.moe_shared_expert(i, s, Mlp.DOWN_PROJ)] = shared.down_proj
            else:
                p[Keys.ffn(i, Mlp.GATE_PROJ)] = mlp.gate_proj
                p[Keys.ffn(i, Mlp.UP_PROJ)] = mlp.up_proj
                p[Keys.ffn(i, Mlp.DOWN_PROJ)] = mlp.down_proj
        return p

    def get_all_parameters(self) -> dict[str, np.ndarray]:
        """Flat dict of all parameters, keyed by ``shared.constants.Keys``.

        This is the checkpoint contract: the same dict is what
        ``load_from_numpy_dict`` and the cross-track save/load paths use.
        The key set and shapes are owned by ``ParameterRegistry``; this
        method only supplies storage via the track's binding map.
        """
        return ParameterRegistry(self.config).bind(self._param_arrays())

    def load_from_numpy_dict(self, params: dict[str, np.ndarray]) -> None:
        """Load parameters from a flat dict (inverse of ``get_all_parameters``).

        Validates against the registry first, so stale or mismatched
        checkpoints fail fast instead of half-loading. The assignment walk
        mirrors the track's binding map (``_param_arrays``) exactly — both
        address the same objects, so a binding drift fails at one of the
        two, never silently.
        """
        registry = ParameterRegistry(self.config)
        registry.validate(params)
        # Copy each array: the caller keeps ownership of ``params`` (e.g. a
        # shared checkpoint dict) — aliasing would let in-place training
        # mutate it.
        self.embedding.weight = params[Keys.embed()].copy()
        self.final_norm.gamma = params[Keys.final_norm()].copy()
        self.lm_head_weight = params[Keys.lm_head()].copy()
        for i, block in enumerate(self.stack.layers):
            block.input_layernorm.gamma = params[Keys.ln(i, LayerNorm.INPUT)].copy()
            block.post_attention_layernorm.gamma = params[Keys.ln(i, LayerNorm.POST_ATTENTION)].copy()
            attn = block.self_attn
            attn.q_proj = params[Keys.attn(i, Attn.Q_PROJ)].copy()
            attn.k_proj = params[Keys.attn(i, Attn.K_PROJ)].copy()
            attn.v_proj = params[Keys.attn(i, Attn.V_PROJ)].copy()
            attn.o_proj = params[Keys.attn(i, Attn.O_PROJ)].copy()
            mlp = block.mlp
            if isinstance(mlp, MixtureOfExperts):
                mlp.gate = params[Keys.moe_gate(i)].copy()
                for j, expert in enumerate(mlp.experts):
                    expert.gate_proj = params[Keys.moe_expert(i, j, Mlp.GATE_PROJ)].copy()
                    expert.up_proj = params[Keys.moe_expert(i, j, Mlp.UP_PROJ)].copy()
                    expert.down_proj = params[Keys.moe_expert(i, j, Mlp.DOWN_PROJ)].copy()
                for s, shared in enumerate(mlp.shared_experts):
                    shared.gate_proj = params[Keys.moe_shared_expert(i, s, Mlp.GATE_PROJ)].copy()
                    shared.up_proj = params[Keys.moe_shared_expert(i, s, Mlp.UP_PROJ)].copy()
                    shared.down_proj = params[Keys.moe_shared_expert(i, s, Mlp.DOWN_PROJ)].copy()
            elif isinstance(mlp, SwiGLUFFN):
                mlp.gate_proj = params[Keys.ffn(i, Mlp.GATE_PROJ)].copy()
                mlp.up_proj = params[Keys.ffn(i, Mlp.UP_PROJ)].copy()
                mlp.down_proj = params[Keys.ffn(i, Mlp.DOWN_PROJ)].copy()

    def forward_with_trace(
        self, input_ids: np.ndarray, positions: np.ndarray | None = None, record: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Forward pass plus the intermediates the analytic backward needs.

        Returns (logits, trace) where trace holds:
            x_in0    (B, S, D) embedding output (the stack input)
            stack_out (B, S, D) pre-final-norm activations
            positions (S,) the RoPE positions used

        record: optional dict for the learning-mode instrumented forward;
            when given, it is filled with the raw (unserialized)
            intermediates of this pass: x_in0, stack_out, positions, the
            per-block state dicts ("blocks"), x_final, and logits — the same
            values the operators kept for the backward, captured once along
            the way. The math (and its bit-level result) is identical either
            way.

        ``forward`` is a thin wrapper over this (it drops the trace).
        """
        if positions is None:
            positions = np.arange(input_ids.shape[1], dtype=np.int32)
        blocks_rec = [{} for _ in self.stack.layers] if record is not None else None
        x_in0 = self.embedding.forward(input_ids)  # (B, S, D)
        stack_out = self.stack.forward(x_in0, positions, record=blocks_rec)  # (B, S, D)
        x_final = self.final_norm.forward(stack_out)  # (B, S, D)
        logits = x_final @ self.lm_head_weight  # (B, S, V)
        if record is not None:
            record.update(
                {
                    "x_in0": x_in0,
                    "stack_out": stack_out,
                    "positions": positions,
                    "blocks": blocks_rec,
                    "x_final": x_final,
                    "logits": logits,
                }
            )
        return logits, {"x_in0": x_in0, "stack_out": stack_out, "positions": positions}

    def make_cache(self, batch_size: int, quantize: bool = False) -> list[dict]:
        """Create an empty per-layer KV cache for the per-token step path.

        Naive cache (quantize=False): each layer gets
            {"k": (B, G, 0, hd), "v": (B, G, 0, hd)}
        TurboQuant cache (quantize=True): each layer gets
            {"bits_k": (B, H, 0, hd) int8, "scales_k": (B, H, 0, hd) float,
             "bits_v": (B, H, 0, hd) int8, "scales_v": (B, H, 0, hd) float}

        G = the K/V head count (config.kv_heads) and hd = the head dim — the
        naive cache stores K/V *per group*, so GQA caches are H // G times
        smaller. The TurboQuant cache stores K/V *per head* (H heads) because
        it quantizes the full (B, H, t, hd) tensor after GQA repeat; a
        per-group variant is a straightforward extension.
        """
        B = batch_size
        D = self.embed_dim
        H = self.config.n_heads
        G = self.config.kv_heads
        hd = D // H
        dtype = self.embedding.weight.dtype
        if not quantize:
            empty_k = np.zeros((B, G, 0, hd), dtype=dtype)
            empty_v = np.zeros((B, G, 0, hd), dtype=dtype)
            return [{"k": empty_k.copy(), "v": empty_v.copy()} for _ in range(self.config.n_layers)]
        empty_bits = np.zeros((B, H, 0, hd), dtype=np.int8)
        # Scales are stored per (B, H) — a single scalar per head, broadcast
        # over the sequence and head-dim axes at dequantize time.
        empty_scales = np.zeros((B, H, 0, 1), dtype=dtype)
        return [
            {
                "bits_k": empty_bits.copy(),
                "scales_k": empty_scales.copy(),
                "bits_v": empty_bits.copy(),
                "scales_v": empty_scales.copy(),
            }
            for _ in range(self.config.n_layers)
        ]

    def forward_prefill(
        self,
        input_ids: np.ndarray,
        cache: list[dict] | None = None,
        position_offset: int = 0,
        record: dict | None = None,
    ) -> tuple[np.ndarray, list[dict]]:
        """Run a full-sequence forward and fill the per-layer KV cache.

        input_ids: (B, S) int token IDs.
        cache: optional pre-allocated cache (make_cache); created when None.
        position_offset: added to the 0-based RoPE positions — so a windowed
            prompt (longer prompt than the context) prefills at its absolute
            positions while the decode steps keep using absolute ones.
        record: optional dict filled with the raw intermediates of this pass
            (same keys as ``forward_with_trace``'s record) — the
            learning-mode prefill step.

        Returns (logits (B, S, V), cache) where the cache holds the K/V of
        every position, ready for per-token steps. The cache is backfilled
        from the attention state the blocks capture during this very forward
        pass (per-group, RoPE'd K / un-rotated V) — no second pass needed.
        """
        B, S = input_ids.shape
        if cache is None:
            cache = self.make_cache(B)
        positions = np.arange(S, dtype=np.int32) + position_offset
        blocks_rec = [{} for _ in self.stack.layers]
        x = self.embedding.forward(input_ids)  # (B, S, D)
        stack_out = self.stack.forward(x, positions, record=blocks_rec)  # (B, S, D)
        x_final = self.final_norm.forward(stack_out)  # (B, S, D)
        logits = x_final @ self.lm_head_weight  # (B, S, V)
        # Backfill each layer's cache from this pass's attention state:
        # K/V per group (GQA keeps it small), K already RoPE'd, V un-rotated.
        for i, block_state in enumerate(blocks_rec):
            attn_state = block_state["attn"]
            cache[i]["k"] = attn_state["k_group"]  # (B, G, S, hd)
            cache[i]["v"] = attn_state["v_group"]  # (B, G, S, hd)
        if record is not None:
            record.update(
                {
                    "x_in0": x,
                    "stack_out": stack_out,
                    "positions": positions,
                    "blocks": blocks_rec,
                    "x_final": x_final,
                    "logits": logits,
                }
            )
        return logits, cache

    def forward_step(
        self,
        input_ids: np.ndarray,
        position: int,
        cache: list[dict],
        quantize: bool = False,
        record: dict | None = None,
    ) -> np.ndarray:
        """Process ONE token per batch row against the cached K/V.

        input_ids: (B, 1) int token IDs.
        position: the absolute token index of the token (0-based).
        cache: the per-layer cache; the token's K/V are *appended* to each
            layer's cache before attention runs.
        quantize: if True, append the new K/V to the cache in 1-bit
            TurboQuant form (bits + per-channel scale) and dequantize the
            full cached tensor before attention, so the step attends against
            the (lossy) quantized cache. If False, append the full-precision
            K/V (the default naive path).

        Returns: logits (B, 1, V).

        Usage:
          - **Prefill:** call with the *last* token of the sequence after
            ``forward_prefill`` (or after threading all but the last token),
            to recompute the last position's logits against the full cache.
          - **Generate:** call with each *new* token (position = current
            sequence length), to extend the sequence by one.

        This is the O(1)-per-token inference path: only the new token's K/V
        are computed; attention runs against the cached (B, G, t, hd) K/V
        (naive) or the dequantized (B, H, t, hd) K/V (TurboQuant).
        """
        x = self.embedding.forward(input_ids)  # (B, 1, D)
        blocks_rec = [{} for _ in self.stack.layers] if record is not None else None
        stack_out = self.stack.forward_step(x, position, cache, quantize=quantize, record=blocks_rec)  # (B, 1, D)
        x_final = self.final_norm.forward(stack_out)  # (B, 1, D)
        logits = x_final @ self.lm_head_weight  # (B, 1, V)
        if record is not None:
            record.update(
                {
                    "x_in0": x,
                    "stack_out": stack_out,
                    "positions": np.array([position], dtype=np.int32),
                    "blocks": blocks_rec,
                    "x_final": x_final,
                    "logits": logits,
                }
            )
        return logits

    def backward(self, input_ids: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
        """Analytic gradients of the loss w.r.t. every parameter.

        The chain rule runs in reverse of the forward:

            logits = final_norm(stack(embed(x))) @ W_lm

        1. dlogits = CrossEntropyLoss(shift=False).backward(logits, targets)   (B, S, V)
           (targets are pre-shifted next-token labels; no internal shift)
        2. lm_head (linear D → V):
               dW_lm  = h^T @ dlogits          (D, V)
               dh     = dlogits @ W_lm^T       (B, S, D)
        3. final RMSNorm: (dh, d_gamma) = final_norm.backward(dh, stack_out)
        4. stack: per-layer backwards in reverse order (see DecoderStack.backward)
        5. embedding: dW_emb[t] = sum of dh rows where input_ids == t

        This is O(forward) — the finite-difference loop that used to live
        here is now a test-only gradient checker (``impl._np.gradcheck``).
        """
        logits, trace = self.forward_with_trace(input_ids)
        dlogits = CrossEntropyLoss(shift=False).backward(logits, targets)  # (B, S, V)

        stack_out = trace["stack_out"]  # (B, S, D)
        positions = trace["positions"]  # (S,)
        D, V = self.embed_dim, self.vocab_size

        # lm_head: logits = h @ W_lm where h = final_norm(stack_out)
        h_final = self.final_norm.forward(stack_out)  # (B, S, D)
        dW_lm = h_final.reshape(-1, D).T @ dlogits.reshape(-1, V)  # (D, V)
        dh = dlogits @ self.lm_head_weight.T  # (B, S, D)

        # final RMSNorm
        dh, d_gamma_final = self.final_norm.backward(dh, stack_out)  # (B, S, D), (D,)

        # decoder stack (reverse order internally)
        d_stack_in, per_layer_grads = self.stack.backward(dh, trace["x_in0"], positions)

        # embedding
        dW_emb = self.embedding.backward(d_stack_in, input_ids)  # (V, D)

        # Assemble the flat Keys dict.
        grads: dict[str, np.ndarray] = {
            Keys.embed(): dW_emb,
            Keys.final_norm(): d_gamma_final,
            Keys.lm_head(): dW_lm,
        }
        for i, block_grads in enumerate(per_layer_grads):
            grads[Keys.ln(i, LayerNorm.INPUT)] = block_grads["input_layernorm.gamma"]
            grads[Keys.ln(i, LayerNorm.POST_ATTENTION)] = block_grads["post_attention_layernorm.gamma"]
            for proj in ATTN_PROJS:
                grads[Keys.attn(i, proj)] = block_grads[f"self_attn.{proj}"]
            block = self.stack.layers[i]
            if isinstance(block.mlp, MixtureOfExperts):
                grads[Keys.moe_gate(i)] = block_grads["mlp.gate"]
                for j, expert_grads in enumerate(block_grads["mlp.experts"]):
                    for proj in FFN_PROJS:
                        grads[Keys.moe_expert(i, j, proj)] = expert_grads[proj]
                for s, shared_grads in enumerate(block_grads["mlp.shared_experts"]):
                    for proj in FFN_PROJS:
                        grads[Keys.moe_shared_expert(i, s, proj)] = shared_grads[proj]
            else:
                for proj in FFN_PROJS:
                    grads[Keys.ffn(i, proj)] = block_grads[f"mlp.{proj}"]
        return grads

    def _compute_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Cross-entropy loss between logits and target token IDs.

        Targets are pre-shifted next-token labels (standard LM convention —
        the caller aligns them), so no internal shift is applied; this must
        match ``backward`` exactly.

        logits: (B, S, V)  targets: (B, S) int
        """
        return float(CrossEntropyLoss(shift=False).forward(logits, targets))

    def train_step(self, input_ids: np.ndarray, targets: np.ndarray, optimizer) -> float:
        """One training step: loss = CE(forward(x), y); grads = backward; optimizer.step.

        The optimizer updates the parameter arrays in place (AdamW mutates
        each array's values), so the model state changes in place as well.
        Returns the step's loss.
        """
        logits = self.forward(input_ids)
        loss = self._compute_loss(logits, targets)
        grads = self.backward(input_ids, targets)
        optimizer.step(self.get_all_parameters(), grads)
        return loss
