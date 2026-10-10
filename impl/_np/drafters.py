"""NumPy drafters — the MTP and DSpark draft models (the math reference).

**The problem, in plain words.** Generating text with a transformer is a
repeating loop: feed the model everything written so far, it computes a
score (a "logit") for every possible next word, pick one, append it,
repeat. The expensive part is the model forward — and it buys only ONE
word per pass. The speculative idea: let a *tiny* model (the drafter)
write several candidate words first — cheaply, possibly badly — then run
the big model ONCE over the whole candidate block. Wherever the big model
agrees, several words are confirmed for the price of one pass; at the
first disagreement the big model's own word wins and the rest of the
draft is thrown away. The output is exactly what the big model alone
would have written — the tiny model can only affect *speed*, never
content (that guarantee is checked by tests, not promised).

**What the drafter is made of.** A small stack of the same parts the big
model uses (an input projection, one or two transformer blocks, a final
scale, and the big model's own word-matching matrix borrowed — so it
scores words in exactly the big model's vocabulary). Its two inputs are
the anchor: the last confirmed word id, and the big model's internal
summary vector of everything before it (the "hidden state"). Its output
is a probability distribution over the vocabulary for each proposed
position — the same kind of row the big model produces, just cheaper.

Implements the ``shared.draft.Drafter`` protocol (ADR 0003; the ONE new
seam of the speculative-decoding feature):

    drafter.draft(anchor_token, anchor_hidden, k) -> (tokens (B, k'), probs)
    drafter.rollback(keep)   — align internal state after a rejection
    drafter.reset()          — fresh sequence

Both drafters condition on the target's final-norm hidden state at the
anchor (the last VERIFIED token) and **share the target's embedding matrix
and lm_head** (DeepSeek-V3 MTP design — no duplicated V×D weights; they are
handed in at construction, never copied into the sidecar).

**MTP** (multi-token prediction, DeepSeek-V3 style — sequential drafting):

    h_norm = RMSNorm(h_anchor)                       (B, D)
    x_anchor = block.step(in_proj([h_norm; emb(anchor)]))   — seeds the
               drafter's own tiny KV cache with the anchor position
    ...then for j = 0..k-1, ONE block step per drafted token, each step
    conditioned on h_norm AND the previous drafted token t_j:
        x_{j+1}  = block.step(in_proj([h_norm; emb(t_j)]), pos = anchor+j)
        logits_j = out_norm(x_{j+1}) @ W_lm           (B, V)
        t_{j+1}   = argmax(logits_j)                  (the proposal)
    where t_0 = the anchor token. Every step's input is the SAME anchor
    hidden state paired with the *latest* token so far — one proposal per
    block step, sequential by construction.

**DSpark** (arXiv 2607.05147 — semi-autoregressive parallel drafting):

    h_norm  = RMSNorm(h_anchor)                       (B, D)
    u_j     = in_proj(concat(h_norm, pos_emb[j]))     (B, D), j = 0..k-1
    p_j     = parallel_backbone(u)                   (B, k, D)  NON-causal:
              every block position attends to every other — the positions
              are told apart by pos_emb, not by order
    s_j     = seq_module(p)                          (B, k, D)  CAUSAL over
              the block — the lightweight sequential module that adds the
              intra-token dependency the pure parallel drafter lacks
    logits_j = out_norm(s_j) @ W_lm                   (B, k, V)
    draft_j = argmax(logits_j)  — the whole block in ONE drafter forward

Confidence-scheduled verification (the second DSpark idea) lives in the
speculative engine (``impl/_np/spec.py``) — the drafter only reports its
per-position token probabilities so the schedule can estimate prefix
survival. The engine drives the same interface for MTP and DSpark (and a
future DFlash plug-in).

Weights: the sidecar key scheme (``shared/draft.py``) —
``{family}.norm.weight``, ``{family}.in_proj.weight``, the drafter block(s)
mirroring the main per-layer names, ``{family}.out_norm.weight`` (+
``dspark.pos_emb.weight``). Drafters train by distillation
(``scripts/train_drafters.py``); inference only.
"""

from __future__ import annotations

import numpy as np

from impl._np.attention import MultiHeadAttention
from impl._np.init import xavier_uniform
from impl._np.layernorm import RMSNorm
from shared.config import TransformerConfig
from shared.draft import DSHARK, MTP, DrafterMeta


def _stable_probs(logits: np.ndarray) -> np.ndarray:
    """Stable softmax over the last axis: logits (…, V) → probs (…, V)."""
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


class _DrafterBlock:
    """One transformer block of a drafter, over sidecar-flavored keys.

    A thin parameter owner reusing the track's real operators (RMSNorm,
    MultiHeadAttention, SwiGLU) so the forward math is the same code the
    target runs — the sidecar key scheme maps onto these attributes.
    ``causal`` selects the attention mask for the DSpark backbone
    (parallel → non-causal; seq module → causal).
    """

    def __init__(self, D: int, H: int, G: int, ff_dim: int, rope_dim: int, causal: bool, seed: int) -> None:
        self.causal = causal
        self.input_layernorm = RMSNorm(D)
        self.post_attention_layernorm = RMSNorm(D)
        self.self_attn = MultiHeadAttention(embed_dim=D, n_heads=H, n_groups=G, rope_dim=rope_dim, seed=seed + 1)
        from impl._np.ffn import SwiGLUFFN

        self.mlp = SwiGLUFFN(embed_dim=D, ff_dim=ff_dim, seed=seed + 2)

    def forward(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Standard pre-norm block forward with the configured mask."""
        ln1_out = self.input_layernorm.forward(x)  # (B, S, D)
        attn_out = self.self_attn.forward(ln1_out, positions, causal=self.causal)  # (B, S, D)
        h = x + attn_out  # (B, S, D)
        ln2_out = self.post_attention_layernorm.forward(h)  # (B, S, D)
        return h + self.mlp.forward(ln2_out)  # (B, S, D)

    def forward_step(self, x: np.ndarray, position: int, cache: dict) -> np.ndarray:
        """One-token step with the drafter's own tiny KV cache (MTP path)."""
        ln1_out = self.input_layernorm.forward(x)  # (B, 1, D)
        attn_out = self.self_attn.forward_step(ln1_out, position, cache)  # (B, 1, D)
        h = x + attn_out  # (B, 1, D)
        ln2_out = self.post_attention_layernorm.forward(h)  # (B, 1, D)
        return h + self.mlp.forward(ln2_out)  # (B, 1, D)

    def params(self, prefix: str) -> dict[str, np.ndarray]:
        """The sidecar key → array map for this block (see shared/draft.py)."""
        return {
            f"{prefix}.input_layernorm.weight": self.input_layernorm.gamma,
            f"{prefix}.self_attn.q_proj.weight": self.self_attn.q_proj,
            f"{prefix}.self_attn.k_proj.weight": self.self_attn.k_proj,
            f"{prefix}.self_attn.v_proj.weight": self.self_attn.v_proj,
            f"{prefix}.self_attn.o_proj.weight": self.self_attn.o_proj,
            f"{prefix}.post_attention_layernorm.weight": self.post_attention_layernorm.gamma,
            f"{prefix}.mlp.gate_proj.weight": self.mlp.gate_proj,
            f"{prefix}.mlp.up_proj.weight": self.mlp.up_proj,
            f"{prefix}.mlp.down_proj.weight": self.mlp.down_proj,
        }

    def load(self, prefix: str, params: dict[str, np.ndarray]) -> None:
        """Copy sidecar arrays into the operators (no aliasing)."""
        self.input_layernorm.gamma = np.array(params[f"{prefix}.input_layernorm.weight"], copy=True)
        self.self_attn.q_proj = np.array(params[f"{prefix}.self_attn.q_proj.weight"], copy=True)
        self.self_attn.k_proj = np.array(params[f"{prefix}.self_attn.k_proj.weight"], copy=True)
        self.self_attn.v_proj = np.array(params[f"{prefix}.self_attn.v_proj.weight"], copy=True)
        self.self_attn.o_proj = np.array(params[f"{prefix}.self_attn.o_proj.weight"], copy=True)
        self.post_attention_layernorm.gamma = np.array(params[f"{prefix}.post_attention_layernorm.weight"], copy=True)
        self.mlp.gate_proj = np.array(params[f"{prefix}.mlp.gate_proj.weight"], copy=True)
        self.mlp.up_proj = np.array(params[f"{prefix}.mlp.up_proj.weight"], copy=True)
        self.mlp.down_proj = np.array(params[f"{prefix}.mlp.down_proj.weight"], copy=True)


class MTPDrafter:
    """Sequential MTP drafter (DeepSeek-V3 style; the DSpark paper's baseline).

    How it works, mechanically: ONE small transformer block (the same
    parts the target uses, just fewer of them) keeps its own tiny memory
    of past positions (the K/V cache). ``draft`` runs that block k times:
    each step reads the anchor summary vector + the newest word proposed
    so far, and outputs a distribution over the next word. Because step
    j's *output word* becomes step j+1's *input*, order is automatic —
    word 3 can only be proposed after words 1 and 2 exist.

    Why "sequential" matters: this is the same left-to-right chain as
    ordinary generation, in miniature — the proposal quality per word is
    as good as the tiny block can make it, but the *latency* is k block
    runs per draft block (the cost DSpark trades away for parallelism).
    """

    family = MTP

    def __init__(
        self,
        meta: DrafterMeta,
        embedding_weight: np.ndarray,  # (V, D) — the TARGET's embedding (shared)
        lm_head_weight: np.ndarray,  # (D, V) — the TARGET's lm_head (shared)
        seed: int = 0,
    ) -> None:
        D, H = meta.embed_dim, meta.n_heads
        G = meta.n_groups if meta.n_groups is not None else meta.n_heads
        self.meta = meta
        self.block_size = meta.block_size
        self.embedding_weight = embedding_weight
        self.lm_head_weight = lm_head_weight
        self.norm = RMSNorm(D)
        self.in_proj = xavier_uniform(np.random.default_rng(seed), 2 * D, D)
        self.block = _DrafterBlock(D, H, G, meta.ff_dim, meta.rope_dim, causal=True, seed=seed + 10)
        self.out_norm = RMSNorm(D)
        self.reset()

    def reset(self) -> None:
        """Fresh sequence: empty the drafter's KV cache."""
        B, G, hd = 1, self.block.self_attn.n_groups, self.block.self_attn.head_dim
        self._cache = {
            "k": np.zeros((B, G, 0, hd), dtype=np.float32),
            "v": np.zeros((B, G, 0, hd), dtype=np.float32),
        }

    def draft(self, anchor_token: np.ndarray, anchor_hidden: np.ndarray, k: int) -> tuple[np.ndarray, list[np.ndarray]]:
        """Propose up to k tokens after the anchor, sequentially.

        anchor_token: (B,) ints — the last verified token.
        anchor_hidden: (B, D) — the target's final-norm hidden at the anchor.
        Returns (tokens (B, k'), probs — k' arrays (B, V), the drafter's own
        per-position distributions). k' = min(k, block_size).
        """
        k = min(k, self.block_size)
        B = anchor_hidden.shape[0]
        # Rebuild the tiny cache when the batch changes (the distillation
        # path drafts batch-N anchors; inference is batch-1). Mirrors the
        # torch drafter's contract.
        if self._cache["k"].shape[0] != B:
            G = self.block.self_attn.n_groups
            hd = self.block.self_attn.head_dim
            self._cache = {
                "k": np.zeros((B, G, 0, hd), dtype=np.float32),
                "v": np.zeros((B, G, 0, hd), dtype=np.float32),
            }
        h_norm = self.norm.forward(anchor_hidden.reshape(B, 1, self.meta.embed_dim))  # (B, 1, D)
        anchor_pos = self._cache["k"].shape[2]  # absolute position of the anchor in the drafter's stream
        # Seed the stream: the anchor token's projected vector is step 0.
        x = self.in_proj_forward(h_norm, anchor_token.reshape(B, 1))
        self.block.forward_step(x, anchor_pos, self._cache)  # append the anchor's K/V
        tokens = []
        probs_list = []
        prev = anchor_token  # (B,)
        for j in range(k):
            x = self.in_proj_forward(h_norm, prev.reshape(B, 1))
            x = self.block.forward_step(x, anchor_pos + 1 + j, self._cache)  # (B, 1, D)
            logits = self.out_norm.forward(x) @ self.lm_head_weight  # (B, 1, V)
            probs = _stable_probs(logits[:, 0, :])  # (B, V)
            prev = np.argmax(probs, axis=-1)  # (B,) greedy proposal
            tokens.append(prev)
            probs_list.append(probs)
        self._last_len = 1 + k  # anchor step + k drafts written to the cache
        return np.stack(tokens, axis=1), probs_list  # (B, k), k × (B, V)

    def in_proj_forward(self, h_norm: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        """x = in_proj(concat(h_norm, emb(token))): (B, s, D) → (B, s, D)."""
        emb = self.embedding_weight[token_ids]  # (B, s, D)
        cat = np.concatenate([h_norm, emb], axis=-1)  # (B, s, 2D)
        return cat @ self.in_proj  # (B, s, D) @ (2D, D)

    def rollback(self, keep: int) -> None:
        """After a rejection: keep the anchor + first ``keep`` drafted steps
        in the drafter's KV cache, drop the rest (the next draft continues
        from the new anchor)."""
        n = 1 + keep
        self._cache["k"] = self._cache["k"][:, :, :n]
        self._cache["v"] = self._cache["v"][:, :, :n]

    # ── sidecar I/O ────────────────────────────────────────────────────────
    def get_all_parameters(self) -> dict[str, np.ndarray]:
        p = {
            "mtp.norm.weight": self.norm.gamma,
            "mtp.in_proj.weight": self.in_proj,
            "mtp.out_norm.weight": self.out_norm.gamma,
        }
        p.update(self.block.params("mtp.block"))
        return p

    def load_from_numpy_dict(self, params: dict[str, np.ndarray]) -> None:
        self.norm.gamma = np.array(params["mtp.norm.weight"], copy=True)
        self.in_proj = np.array(params["mtp.in_proj.weight"], copy=True)
        self.out_norm.gamma = np.array(params["mtp.out_norm.weight"], copy=True)
        self.block.load("mtp.block", params)


class DSparkDrafter:
    """Semi-autoregressive parallel drafter (arXiv 2607.05147, teaching scale).

    How it works, mechanically: two small transformer blocks run over the
    k draft slots AT ONCE. The first (the *parallel backbone*) has no
    left-to-right mask: every slot sees every other slot, so k words are
    proposed in a single pass. Slots are told apart only by learned
    position vectors (pos_emb), one per slot. The second block (the
    *causal refine*) DOES have the left-to-right mask: slot j re-reads
    slots 0..j-1 and repairs its own proposal — that is the
    "semi-autoregressive" part.

    Why two passes: a fully parallel proposal is fast but blind to
    word order inside its own block — later slots drift (suffix decay,
    the DSpark paper's motivating observation). The causal repair pass
    reintroduces order at a fraction of the sequential cost: 2 passes
    for k words instead of k. Stateless across rounds — rollback is a
    no-op (nothing is cached between draft calls).
    """

    family = DSHARK

    def __init__(
        self,
        meta: DrafterMeta,
        embedding_weight: np.ndarray,  # (V, D) — the TARGET's embedding (shared)
        lm_head_weight: np.ndarray,  # (D, V) — the TARGET's lm_head (shared)
        seed: int = 0,
    ) -> None:
        D, H = meta.embed_dim, meta.n_heads
        G = meta.n_groups if meta.n_groups is not None else meta.n_heads
        self.meta = meta
        self.block_size = meta.block_size
        self.embedding_weight = embedding_weight
        self.lm_head_weight = lm_head_weight
        self.norm = RMSNorm(D)
        self.in_proj = xavier_uniform(np.random.default_rng(seed), 2 * D, D)
        rng = np.random.default_rng(seed + 5)
        self.pos_emb = rng.normal(0.0, 0.02, size=(meta.block_size, D)).astype(np.float32)
        self.parallel = _DrafterBlock(D, H, G, meta.ff_dim, meta.rope_dim, causal=False, seed=seed + 10)
        self.seq = _DrafterBlock(D, H, G, meta.ff_dim, meta.rope_dim, causal=True, seed=seed + 30)
        self.out_norm = RMSNorm(D)

    def reset(self) -> None:
        """Stateless drafter — nothing to clear."""

    def draft(self, anchor_token: np.ndarray, anchor_hidden: np.ndarray, k: int) -> tuple[np.ndarray, list[np.ndarray]]:
        """Propose up to k tokens after the anchor in ONE forward pass.

        Same contract as ``MTPDrafter.draft``; the whole block is predicted
        in parallel (non-causal backbone) then refined causally.
        """
        k = min(k, self.block_size)
        B, D = anchor_hidden.shape
        h_norm = self.norm.forward(anchor_hidden.reshape(B, 1, D))  # (B, 1, D)
        # Replicate the anchor hidden to all k block positions, each paired
        # with its own learned block-position embedding.
        h_rep = np.repeat(h_norm, k, axis=1)  # (B, k, D)
        pos = np.broadcast_to(self.pos_emb[:k].reshape(1, k, D), (B, k, D))  # (B, k, D)
        x = np.concatenate([h_rep, pos], axis=-1) @ self.in_proj  # (B, k, 2D) @ (2D, D) → (B, k, D)
        positions = np.arange(k, dtype=np.int32)  # block-local positions (RoPE over the block)
        p = self.parallel.forward(x, positions)  # (B, k, D) — NON-causal
        s = self.seq.forward(p, positions)  # (B, k, D) — causal refinement
        logits = self.out_norm.forward(s) @ self.lm_head_weight  # (B, k, V)
        probs = _stable_probs(logits)  # (B, k, V)
        tokens = np.argmax(logits, axis=-1)  # (B, k) greedy proposal
        self._last_len = k
        return tokens, [probs[:, j, :] for j in range(k)]  # k × (B, V)

    def rollback(self, keep: int) -> None:
        """Stateless — nothing to roll back."""

    # ── sidecar I/O ────────────────────────────────────────────────────────
    def get_all_parameters(self) -> dict[str, np.ndarray]:
        p = {
            "dspark.norm.weight": self.norm.gamma,
            "dspark.in_proj.weight": self.in_proj,
            "dspark.pos_emb.weight": self.pos_emb,
            "dspark.out_norm.weight": self.out_norm.gamma,
        }
        p.update(self.parallel.params("dspark.parallel"))
        p.update(self.seq.params("dspark.seq"))
        return p

    def load_from_numpy_dict(self, params: dict[str, np.ndarray]) -> None:
        self.norm.gamma = np.array(params["dspark.norm.weight"], copy=True)
        self.in_proj = np.array(params["dspark.in_proj.weight"], copy=True)
        self.pos_emb = np.array(params["dspark.pos_emb.weight"], copy=True)
        self.out_norm.gamma = np.array(params["dspark.out_norm.weight"], copy=True)
        self.parallel.load("dspark.parallel", params)
        self.seq.load("dspark.seq", params)


def drafter_from_sidecar(
    meta: DrafterMeta,
    params: dict[str, np.ndarray],
    embedding_weight: np.ndarray,
    lm_head_weight: np.ndarray,
) -> MTPDrafter | DSparkDrafter:
    """Build a NumPy drafter from a validated sidecar + the target's shared
    embedding/lm_head. Any track's sidecar loads here (round-trip guarantee)."""
    if meta.family == MTP:
        d: MTPDrafter | DSparkDrafter = MTPDrafter(meta, embedding_weight, lm_head_weight)
    elif meta.family == DSHARK:
        d = DSparkDrafter(meta, embedding_weight, lm_head_weight)
    else:
        raise ValueError(f"no NumPy drafter for family {meta.family!r}")
    d.load_from_numpy_dict(params)
    return d


def make_drafter_meta(family: str, target_config: TransformerConfig, block_size: int, ff_dim: int) -> DrafterMeta:
    """DrafterMeta for a new drafter matching the target's D/V."""
    return DrafterMeta(
        family=family,
        block_size=block_size,
        embed_dim=target_config.embed_dim,
        n_heads=max(1, target_config.n_heads // 2),  # half the target's heads (tiny drafter)
        n_groups=None,
        ff_dim=ff_dim,
        rope_dim=target_config.rope_dim,
        vocab_size=target_config.vocab_size,
    )
