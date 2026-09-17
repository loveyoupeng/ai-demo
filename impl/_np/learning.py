"""Learning mode — instrumented forward pass + inference records.

The **instrumented forward** runs the model's own components (so every number
is bit-identical to ``NumPyModel.forward``) and, in parallel, recomputes each
component's math from its public parameters to capture the intermediate
tensors the components discard internally. ``impl/_np`` itself is never
modified — this module is a pure overlay, so the NumPy track stays readable
as teaching code.

Two public entry points:

- ``instrumented_forward(model, input_ids)`` — one forward pass, returning
  every intermediate as a JSON-ready nested dict (the "step record").
- ``generate_with_records(...)`` — multi-token generation where each token
  step is captured as a step record; this is the **inference record** the
  learning-mode page displays and exports as JSON.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from impl._np.attention import MultiHeadAttention
from impl._np.ffn import SwiGLUFFN, silu
from impl._np.model import NumPyModel
from impl._np.moe import MixtureOfExperts
from impl._np.rope import RoPE

# ---------------------------------------------------------------------------
# Tensor → JSON helpers
# ---------------------------------------------------------------------------


def arr(a: np.ndarray) -> list:
    """Convert a NumPy array to a nested list of floats (JSON-ready).

    Values are rounded to 8 decimal places to keep the JSON compact without
    losing any displayable precision.
    """
    return np.round(a.astype(np.float64), 8).tolist()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Record types — the JSON shapes the page consumes
# ---------------------------------------------------------------------------


class NormRecord(TypedDict):
    """RMSNorm capture: per-row rms, the gamma vector, and the output."""

    rms: list
    gamma: list
    out: list


class RopeRecord(TypedDict):
    """RoPE capture: per-pair frequencies, angles, cos, and sin."""

    freqs: list
    angles: list
    cos: list
    sin: list


class AttnRecord(TypedDict):
    """Attention capture: projections, RoPE, scores, weights, context."""

    q_pre: list
    k_pre: list
    v_pre: list
    q_rope_in: list
    k_rope_in: list
    q: list
    k: list
    v: list
    scale: float
    rope: RopeRecord
    scores: list
    attn_out: list
    causal_mask: list
    scores_masked: list
    attn_weights: list
    ctx: list
    k_cache: list
    v_cache: list


class FFNRecord(TypedDict):
    """Dense SwiGLU capture: gate/up intermediates and output."""

    pre_gate: list
    gate: list
    up: list
    gated: list
    out: list
    ff_dim: int


class MoERecord(TypedDict):
    """MoE capture: router, top-k selection, renormalized weights, experts."""

    scores: list
    probs: list
    topk_idx: list
    weights: list
    expert_outs: list
    out: list
    n_experts: int
    top_k: int


class BlockRecord(TypedDict, total=False):
    ln1: NormRecord
    attn: AttnRecord
    h: list
    ln2: NormRecord
    ffn: FFNRecord
    moe: MoERecord
    out: list


class ForwardRecord(TypedDict):
    """The full instrumented-forward step record (see instrumented_forward)."""

    input_ids: list
    embedding: list
    positions: list
    blocks: list
    final_norm: NormRecord
    logits: list
    softmax: list
    top_tokens: list


class StepRecord(TypedDict):
    """One generated token: its input, full forward record, and the pick."""

    step: int
    kind: str
    position: int
    input_tokens: list
    forward: ForwardRecord
    top_tokens: list
    token: int
    text: str


class TokenSpan(TypedDict):
    """A token sequence + its decoded text (prompt / generated)."""

    tokens: list
    text: str


class GenerationRecord(TypedDict):
    """The complete inference record the page displays and exports."""

    config: dict
    vocab: list
    prompt: TokenSpan
    steps: list
    generated: TokenSpan


# ---------------------------------------------------------------------------
# Instrumented forward — one pass, every intermediate captured
# ---------------------------------------------------------------------------


def _rmsnorm_record(gamma: np.ndarray, x: np.ndarray, eps: float, out: np.ndarray) -> NormRecord:
    """Capture an RMSNorm: the rms scalar per row, gamma, output."""
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)  # (B, S, 1)
    return {"rms": arr(rms), "gamma": arr(gamma), "out": arr(out)}


def _rope_record(x_rot: np.ndarray, positions: np.ndarray, rope_dim: int) -> RopeRecord:
    """Recompute the RoPE frequencies/angles/cos/sin (same math as RoPE.forward).

    x_rot: (B, S, H, d) the tensor that will be rotated.
    rope_dim: 0 → rotate all head dims; 0 < rope_dim < d → rotate the first
    rope_dim dims only (the rest pass through, un-rotated).
    """
    d = x_rot.shape[-1]
    d_rot = rope_dim if (0 < rope_dim < d) else d
    pair_dim = d_rot // 2
    pos = np.asarray(positions, dtype=np.int32)
    if pos.ndim == 1:
        pos = np.broadcast_to(pos, (x_rot.shape[0], x_rot.shape[1]))
    freqs = 1.0 / (10000.0 ** (np.arange(pair_dim, dtype=np.float32) * 2.0 / d_rot))  # (pair_dim,)
    angles = pos[:, :, np.newaxis] * freqs[np.newaxis, np.newaxis, :]  # (B, S, pair_dim)
    return {"freqs": arr(freqs), "angles": arr(angles), "cos": arr(np.cos(angles)), "sin": arr(np.sin(angles))}


def _attn_record(
    model_attn: MultiHeadAttention, ln1_out: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, AttnRecord]:
    """Run the block's real MHA (via its state hook) and capture everything.

    Returns (attn_out, record) where the record holds the Q/K/V math the
    page displays: projections, RoPE inputs/outputs, the raw score matrix
    (recomputed, pre- and post-mask), the attention weights, and the
    context.
    """
    attn_out, state = model_attn._forward_state(ln1_out, positions)  # (B, S, D), state dict

    # Raw scores: q·kᵀ/sqrt(hd) with the causal mask (recomputed; the
    # component keeps only the softmax output).
    q = state["q"]  # (B, H, S, hd)
    k = state["k"]  # (B, H, S, hd) after GQA repeat
    scale = state["scale"]  # float, sqrt(hd) — the scores DIVISOR
    S = q.shape[2]
    scores = np.einsum("bhid,bhjd->bhij", q, k) / scale  # (B, H, S, S)
    mask = np.triu(np.ones((S, S), dtype=bool), k=1)  # (S, S), True where j > i
    scores_masked = np.where(mask, -np.inf, scores)  # (B, H, S, S)

    return attn_out, {
        "q_pre": arr(state["q_pre"]),  # (B, S, H·hd) — ln1_out @ Wq
        "k_pre": arr(state["k_pre"]),  # (B, S, G·hd)
        "v_pre": arr(state["v_pre"]),  # (B, S, G·hd)
        "q_rope_in": arr(state["q_rope_in"].transpose(0, 2, 1, 3)),  # (B, S, H, hd) pre-RoPE
        "k_rope_in": arr(state["k_rope_in"].transpose(0, 2, 1, 3)),  # (B, S, G, hd) pre-RoPE
        "q": arr(q),  # (B, H, S, hd) post-RoPE
        "k": arr(k),  # (B, H, S, hd) post-RoPE (GQA-expanded)
        "v": arr(state["v"]),  # (B, H, S, hd)
        "scale": float(scale),  # sqrt(hd) — divide scores by this
        "rope": _rope_record(state["q_rope_in"].transpose(0, 2, 1, 3), positions, model_attn.rope_dim),
        "scores": arr(np.nan_to_num(scores, neginf=-1e4)),  # (B, H, S, S)
        "scores_masked": arr(np.nan_to_num(scores_masked, neginf=-1e4)),  # (B, H, S, S)
        "causal_mask": arr(mask.astype(np.float64)),  # (S, S)
        "attn_weights": arr(state["attn"]),  # (B, H, S, S) — softmax output
        "ctx": arr(state["ctx"]),  # (B, S, H·hd)
        "attn_out": arr(attn_out),  # (B, S, D) — ctx @ Wo
        # The KV cache state AFTER this step: in prefill the whole prompt's
        # K/V are stored (GQA-expanded to H heads so the page can browse heads).
        "k_cache": arr(k),  # (B, H, S, hd)
        "v_cache": arr(state["v"]),  # (B, H, S, hd)
    }


def _init_kv_cache(attn: MultiHeadAttention, k_pre: np.ndarray, v_pre: np.ndarray, positions: np.ndarray) -> dict:
    """Build a block's KV cache from a prefill pass.

    k_pre/v_pre are the pre-RoPE, un-split projections (B, S, G·hd). The
    cache stores K/V *per group* (GQA keeps it small); RoPE is applied to K
    with the prefill positions so later decode steps only rotate the NEW
    token and can attend to these cached rows unchanged.
    """
    B, S, _ = k_pre.shape
    G, hd = attn.n_groups, attn.head_dim
    k_heads = k_pre.reshape(B, S, G, hd).transpose(0, 2, 1, 3)  # (B, G, S, hd) pre-RoPE
    v_heads = v_pre.reshape(B, S, G, hd).transpose(0, 2, 1, 3)  # (B, G, S, hd)
    k_heads = (
        RoPE().forward(k_heads.transpose(0, 2, 1, 3), positions, rope_dim=attn.rope_dim).transpose(0, 2, 1, 3)
    )  # (B, G, S, hd) post-RoPE
    return {"k": k_heads, "v": v_heads}  # each (B, G, S, hd)


def _attn_step_record(
    attn: MultiHeadAttention, x: np.ndarray, position: int, cache: dict
) -> tuple[np.ndarray, AttnRecord]:
    """Run ONE decode step through the block's real ``forward_step`` (which
    appends the new K/V to ``cache``) and capture the record the page shows.

    The intermediates are recomputed with the same math as ``forward_step``
    (same parameters, same operations) — the cache is already appended, so
    the displayed scores/weights are exactly what produced the output.
    Nothing is masked in decode: every cached row is in the past of the new
    token, so the causal mask is all zeros and the row spans positions
    1..t (the whole cache).
    """
    out = attn.forward_step(x, position, cache)  # (B, 1, D); cache mutated
    B = x.shape[0]
    H, G, hd = attn.n_heads, attn.n_groups, attn.head_dim
    positions = np.array([position], dtype=np.int32)  # (1,)

    q_pre = x @ attn.q_proj  # (B, 1, H·hd)
    k_pre = x @ attn.k_proj  # (B, 1, G·hd)
    v_pre = x @ attn.v_proj  # (B, 1, G·hd)
    q_rope_in = q_pre.reshape(B, 1, H, hd).transpose(0, 2, 1, 3)  # (B, H, 1, hd) pre-RoPE
    k_rope_in = k_pre.reshape(B, 1, G, hd).transpose(0, 2, 1, 3)  # (B, G, 1, hd) pre-RoPE
    v = v_pre.reshape(B, 1, G, hd).transpose(0, 2, 1, 3)  # (B, G, 1, hd)
    q = (
        RoPE().forward(q_rope_in.transpose(0, 2, 1, 3), positions, rope_dim=attn.rope_dim).transpose(0, 2, 1, 3)
    )  # (B, H, 1, hd)
    k = (
        RoPE().forward(k_rope_in.transpose(0, 2, 1, 3), positions, rope_dim=attn.rope_dim).transpose(0, 2, 1, 3)
    )  # (B, G, 1, hd)
    if G != H:  # GQA: broadcast each group to its query heads (display parity with _attn_record)
        k = np.repeat(k, H // G, axis=1)  # (B, H, 1, hd)
        v = np.repeat(v, H // G, axis=1)  # (B, H, 1, hd)

    # The cache AFTER the append (GQA-expanded) is what the new query attended to.
    k_r = np.repeat(cache["k"], H // G, axis=1) if G != H else cache["k"]  # (B, H, t, hd)
    v_r = np.repeat(cache["v"], H // G, axis=1) if G != H else cache["v"]  # (B, H, t, hd)
    t = k_r.shape[2]

    scale = float(np.sqrt(hd))
    scores = (q @ k_r.transpose(0, 1, 3, 2)) / scale  # (B, H, 1, t)
    mask = np.zeros((1, t), dtype=bool)  # (1, t) — nothing masked in decode
    z = scores - np.max(scores, axis=-1, keepdims=True)
    attn_w = np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)  # (B, H, 1, t)
    ctx = (attn_w @ v_r).transpose(0, 2, 1, 3).reshape(B, 1, H * hd)  # (B, 1, H·hd)

    return out, {
        "q_pre": arr(q_pre),  # (B, 1, H·hd)
        "k_pre": arr(k_pre),  # (B, 1, G·hd)
        "v_pre": arr(v_pre),  # (B, 1, G·hd)
        "q_rope_in": arr(q_rope_in.transpose(0, 2, 1, 3)),  # (B, 1, H, hd) pre-RoPE
        "k_rope_in": arr(k_rope_in.transpose(0, 2, 1, 3)),  # (B, 1, G, hd) pre-RoPE
        "q": arr(q),  # (B, H, 1, hd) post-RoPE
        "k": arr(k),  # (B, H, 1, hd) post-RoPE (GQA-expanded)
        "v": arr(v),  # (B, H, 1, hd)
        "scale": scale,  # sqrt(hd)
        "rope": _rope_record(q_rope_in.transpose(0, 2, 1, 3), positions, attn.rope_dim),
        "scores": arr(np.nan_to_num(scores, neginf=-1e4)),  # (B, H, 1, t) over the WHOLE cache
        "scores_masked": arr(np.nan_to_num(scores, neginf=-1e4)),  # (B, H, 1, t) — mask is all zeros
        "causal_mask": arr(mask.astype(np.float64)),  # (1, t) — all 0 in decode
        "attn_weights": arr(attn_w),  # (B, H, 1, t)
        "ctx": arr(ctx),  # (B, 1, H·hd)
        "attn_out": arr(out),  # (B, 1, D)
        "k_cache": arr(k_r),  # (B, H, t, hd) — the full cache after this append
        "v_cache": arr(v_r),  # (B, H, t, hd)
    }


def _ffn_record(mlp: SwiGLUFFN, x: np.ndarray, out: np.ndarray) -> FFNRecord:
    """Capture dense SwiGLU: pre_gate, gate (post-SiLU), up, gated, output."""
    pre_gate = x @ mlp.gate_proj  # (B, S, FF)
    gate = silu(pre_gate)  # (B, S, FF)
    up = x @ mlp.up_proj  # (B, S, FF)
    gated = gate * up  # (B, S, FF)
    return {
        "pre_gate": arr(pre_gate),
        "gate": arr(gate),
        "up": arr(up),
        "gated": arr(gated),
        "out": arr(out),
        "ff_dim": int(mlp.down_proj.shape[0]),
    }


def _moe_record(mlp: MixtureOfExperts, x: np.ndarray, out: np.ndarray) -> MoERecord:
    """Capture MoE: router scores, softmax probs, top-k selection, per-expert outputs."""
    E = mlp.n_experts
    scores = x @ mlp.gate  # (B, S, E)
    scores_stable = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_stable)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)  # (B, S, E)
    if mlp.top_k < E:
        order = np.argsort(probs, axis=-1)[:, :, ::-1]  # (B, S, E) descending
        topk_idx = order[:, :, : mlp.top_k]  # (B, S, k)
        kth_idx = order[:, :, mlp.top_k - 1 : mlp.top_k]
        threshold = np.take_along_axis(probs, kth_idx, axis=-1)  # (B, S, 1)
        probs = np.where(probs >= threshold, probs, 0.0)
        weights = probs / np.maximum(np.sum(probs, axis=-1, keepdims=True), 1e-8)  # (B, S, E)
    else:
        topk_idx = np.argsort(probs, axis=-1)[:, :, ::-1][:, :, : mlp.top_k]
        weights = probs
    expert_outs = [expert.forward(x) for expert in mlp.experts]  # E × (B, S, D)
    return {
        "scores": arr(scores_stable),  # (B, S, E) stable-softmaxed
        "probs": arr(probs),  # (B, S, E) after top-k mask
        "topk_idx": np.round(topk_idx).astype(int).tolist(),  # (B, S, k)
        "weights": arr(weights),  # (B, S, E) renormalized
        "expert_outs": [arr(e) for e in expert_outs],
        "out": arr(out),
        "n_experts": E,
        "top_k": mlp.top_k,
    }


def instrumented_forward(model: NumPyModel, input_ids: np.ndarray) -> ForwardRecord:
    """One forward pass with every intermediate captured (the step record).

    The output tensors are identical to ``model.forward(input_ids)`` — the
    capture is done by calling the model's own components and recomputing
    their (discarded) intermediates from the same public parameters.

    Returns a nested dict (all arrays as nested float lists):
      input_ids, embedding, positions,
      blocks[i]: {ln1, attn, h, ln2, ffn|moe, out},
      final_norm, logits, softmax, top_tokens
    """
    x = np.asarray(input_ids, dtype=np.int32)
    S = x.shape[1]
    positions = np.arange(S, dtype=np.int32)

    # Embedding: (B, S) → (B, S, D)
    emb = model.embedding.forward(x)

    # Blocks: each runs through the real component; intermediates captured.
    stream = emb
    blocks: list[BlockRecord] = []
    for block in model.stack.layers:
        ln1 = block.input_layernorm
        ln1_out = ln1.forward(stream)  # (B, S, D)
        attn_out, attn_rec = _attn_record(block.self_attn, ln1_out, positions)  # (B, S, D)
        h = stream + attn_out  # (B, S, D) residual after attention
        ln2 = block.post_attention_layernorm
        ln2_out = ln2.forward(h)  # (B, S, D)
        mlp = block.mlp
        ff_out = mlp.forward(ln2_out)  # (B, S, D)
        out = h + ff_out  # (B, S, D) block output
        block_rec: BlockRecord = {
            "ln1": _rmsnorm_record(ln1.gamma, stream, ln1.eps, ln1_out),
            "attn": attn_rec,
            "h": arr(h),
            "ln2": _rmsnorm_record(ln2.gamma, h, ln2.eps, ln2_out),
            "out": arr(out),
        }
        if isinstance(mlp, MixtureOfExperts):
            block_rec["moe"] = _moe_record(mlp, ln2_out, ff_out)
        else:
            block_rec["ffn"] = _ffn_record(mlp, ln2_out, ff_out)
        blocks.append(block_rec)
        stream = out

    # Final norm + lm_head: (B, S, D) → (B, S, D) → (B, S, V)
    normed = model.final_norm.forward(stream)
    logits = normed @ model.lm_head_weight

    # Display extras: the softmax distribution + top tokens at the last position.
    z = logits.astype(np.float64) - np.max(logits, axis=-1, keepdims=True)
    p = np.exp(z)
    p /= np.sum(p, axis=-1, keepdims=True)  # (B, S, V)
    V = p.shape[-1]
    top_n = min(10, V)
    top_idx = np.argsort(p[0, -1])[::-1][:top_n]  # top-10 at the last position
    top_tokens = [[int(i), float(p[0, -1, i])] for i in top_idx]

    return {
        "input_ids": x.tolist(),
        "embedding": arr(emb),
        "positions": positions.tolist(),
        "blocks": blocks,
        "final_norm": _rmsnorm_record(model.final_norm.gamma, stream, model.final_norm.eps, normed),
        "logits": arr(logits),
        "softmax": arr(p),
        "top_tokens": top_tokens,
    }


# ---------------------------------------------------------------------------
# Multi-token generation with a record per token step
# ---------------------------------------------------------------------------


def generate_with_records(
    model: NumPyModel,
    vocab: list[str],
    prompt_ids: list[int],
    n_tokens: int,
    temp: float | None = None,
    top_k: int | None = None,
    seed: int = 42,
) -> GenerationRecord:
    """Generate ``n_tokens`` with a real KV cache, recording every step.

    Step 0 is **prefill**: one full forward over the prompt (the same
    ``instrumented_forward`` as before), after which each block's KV cache
    is initialized with the prompt's RoPE'd K/V. Steps 1..n are
    **autoregressive decode**: each step embeds ONLY the new token and runs
    the blocks' real ``forward_step``, which appends the new K/V row to the
    cache and attends the single new query to the whole cache — no
    recomputation of past positions, and no causal mask needed (every cached
    row is in the past). The generated tokens are identical to the naive
    full-recompute loop (same math, same parameters), but each decode step's
    record shows the single new row plus the full cache it attended to.

    ``temp is None`` → greedy (argmax); otherwise sample with temperature
    scaling and optional top-k filtering (seeded).

    Returns:
      {
        "config": {...}, "vocab": [...],
        "prompt": {"tokens": [...], "text": "..."},
        "steps": [ {"step": i, "kind": "prefill"|"decode", "position": int,
                     "input_tokens": [...], "forward": {...},
                     "top_tokens": [[id, prob]...], "token": id, "text": str}, ... ],
        "generated": {"tokens": [...], "text": "..."},
      }
    """
    rng = np.random.default_rng(seed)
    ctx = model.config.context_length
    seq = list(prompt_ids)
    steps: list[StepRecord] = []

    def _pick(p_last: np.ndarray) -> int:
        """Greedy argmax, or seeded temperature/top-k sampling."""
        if temp is None:
            return int(np.argmax(p_last))
        z = np.log(p_last + 1e-30) / temp
        if top_k is not None:
            kth = np.partition(z, -top_k)[-1]
            z = np.where(z < kth, -np.inf, z)
        z = z - z.max()
        pk = np.exp(z)
        pk /= pk.sum()
        return int(rng.choice(len(pk), p=pk))

    # ---- Step 0: prefill the prompt (full forward, initializes the caches)
    x = np.array([seq[-ctx:]], dtype=np.int32)
    rec = instrumented_forward(model, x)
    tok = _pick(np.asarray(rec["softmax"], dtype=np.float64)[0][-1])
    steps.append(
        {
            "step": 0,
            "kind": "prefill",
            "position": int(x.shape[1] - 1),  # last prompt position produced the token
            "input_tokens": list(seq[-ctx:]),
            "forward": rec,
            "top_tokens": rec["top_tokens"],
            "token": tok,
            "text": vocab[tok],
        }
    )
    seq.append(tok)
    p0 = list(x[0])
    positions0 = np.arange(len(p0), dtype=np.int32) + (len(seq) - 1 - len(p0))  # absolute
    caches: list[dict] = []
    for i, block in enumerate(model.stack.layers):
        r = rec["blocks"][i]["attn"]
        caches.append(_init_kv_cache(block.self_attn, np.array(r["k_pre"]), np.array(r["v_pre"]), positions0))

    # ---- Steps 1..n: autoregressive decode, one new token per step
    for i in range(1, n_tokens):
        t = len(seq) - 1  # absolute position of the new token
        x1 = np.array([seq[-1:]], dtype=np.int32)  # (1, 1)
        emb = model.embedding.forward(x1)  # (1, 1, D)
        stream = emb
        blocks: list[BlockRecord] = []
        for j, block in enumerate(model.stack.layers):
            ln1 = block.input_layernorm
            ln1_out = ln1.forward(stream)  # (1, 1, D)
            attn_out, attn_rec = _attn_step_record(block.self_attn, ln1_out, t, caches[j])  # (1, 1, D)
            h = stream + attn_out  # (1, 1, D)
            ln2 = block.post_attention_layernorm
            ln2_out = ln2.forward(h)  # (1, 1, D)
            mlp = block.mlp
            ff_out = mlp.forward(ln2_out)  # (1, 1, D)
            out = h + ff_out  # (1, 1, D)
            block_rec: BlockRecord = {
                "ln1": _rmsnorm_record(ln1.gamma, stream, ln1.eps, ln1_out),
                "attn": attn_rec,
                "h": arr(h),
                "ln2": _rmsnorm_record(ln2.gamma, h, ln2.eps, ln2_out),
                "out": arr(out),
            }
            if isinstance(mlp, MixtureOfExperts):
                block_rec["moe"] = _moe_record(mlp, ln2_out, ff_out)
            else:
                block_rec["ffn"] = _ffn_record(mlp, ln2_out, ff_out)
            blocks.append(block_rec)
            stream = out

        normed = model.final_norm.forward(stream)  # (1, 1, D)
        logits = normed @ model.lm_head_weight  # (1, 1, V)
        z = logits.astype(np.float64) - np.max(logits, axis=-1, keepdims=True)
        p = np.exp(z)
        p /= np.sum(p, axis=-1, keepdims=True)  # (1, 1, V)
        V = p.shape[-1]
        top_n = min(10, V)
        top_idx = np.argsort(p[0, 0])[::-1][:top_n]  # top-10 (single position)
        top_tokens = [[int(k), float(p[0, 0, k])] for k in top_idx]

        rec: ForwardRecord = {
            "input_ids": x1.tolist(),
            "embedding": arr(emb),
            "positions": [t],
            "blocks": blocks,
            "final_norm": _rmsnorm_record(model.final_norm.gamma, stream, model.final_norm.eps, normed),
            "logits": arr(logits),
            "softmax": arr(p),
            "top_tokens": top_tokens,
        }
        tok = _pick(p[0, 0])
        steps.append(
            {
                "step": i,
                "kind": "decode",
                "position": t,
                "input_tokens": [int(seq[-1])],  # only the new token is computed
                "forward": rec,
                "top_tokens": top_tokens,
                "token": tok,
                "text": vocab[tok],
            }
        )
        seq.append(tok)

    return {
        "config": model.config.to_dict(),
        "vocab": list(vocab),
        "prompt": {"tokens": list(prompt_ids), "text": "".join(vocab[t] for t in prompt_ids)},
        "steps": steps,
        "generated": {"tokens": seq[len(prompt_ids) :], "text": "".join(vocab[t] for t in seq[len(prompt_ids) :])},
    }
