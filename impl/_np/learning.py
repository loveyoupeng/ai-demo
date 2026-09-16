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
    causal_mask: list
    scores_masked: list
    attn_weights: list
    ctx: list


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
    """One block's capture; exactly one of ``ffn`` / ``moe`` is present."""

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
        "causal_mask": arr(mask.astype(np.float64)),  # (S, S)
        "scores_masked": arr(np.nan_to_num(scores_masked, neginf=-1e4)),  # (B, H, S, S)
        "attn_weights": arr(state["attn"]),  # (B, H, S, S) — softmax output
        "ctx": arr(state["ctx"]),  # (B, S, H·hd)
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
    """Generate ``n_tokens`` tokens, capturing an inference record per step.

    Each step is a FULL forward over the current sequence (no KV cache) —
    the page shows every position, so the record is the whole matrix, not a
    single-row slice. ``temp is None`` → greedy (argmax); otherwise sample
    with temperature scaling and optional top-k filtering (seeded).

    Returns:
      {
        "config": {...}, "vocab": [...],
        "prompt": {"tokens": [...], "text": "..."},
        "steps": [ {"step": i, "input_tokens": [...], "forward": {...},
                     "top_tokens": [[id, prob]...], "token": id, "text": str}, ... ],
        "generated": {"tokens": [...], "text": "..."},
      }
    """
    rng = np.random.default_rng(seed)
    ctx = model.config.context_length
    seq = list(prompt_ids)
    steps: list[StepRecord] = []
    for i in range(n_tokens):
        x = np.array([seq[-ctx:]], dtype=np.int32)
        rec = instrumented_forward(model, x)
        p_last = np.asarray(rec["softmax"], dtype=np.float64)[0][-1]  # (V,) at the last position
        if temp is None:
            tok = int(np.argmax(p_last))
        else:
            z = np.log(p_last + 1e-30) / temp
            if top_k is not None:
                kth = np.partition(z, -top_k)[-1]
                z = np.where(z < kth, -np.inf, z)
            z = z - z.max()
            pk = np.exp(z)
            pk /= pk.sum()
            tok = int(rng.choice(len(pk), p=pk))
        steps.append(
            {
                "step": i,
                "input_tokens": list(seq[-ctx:]),
                "forward": rec,
                "top_tokens": rec["top_tokens"],
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
