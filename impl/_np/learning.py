"""Learning mode — instrumented forward pass + inference records.

The **instrumented forward** runs the track's own state paths — the same
``_forward_state`` hooks the analytic backward uses — so every displayed
number is a value the operators actually computed (bit-identical to
``NumPyModel.forward``), not a re-derivation. This module is the thin
serializer between the raw state the track captures and the JSON shapes the
learning-mode page consumes; it holds no operator math of its own (the one
exception is the one-line RMSNorm ``rms`` formula).

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

from impl._np.block import TransformerBlock
from impl._np.model import NumPyModel
from impl._np.moe import MixtureOfExperts

# ---------------------------------------------------------------------------
# Tensor → JSON helpers


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
    shared_outs: list  # per-shared-expert outputs (ADR 0002); [] when absent
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
# State → record serializers (no math: the state holds everything)
# ---------------------------------------------------------------------------


def _rmsnorm_record(gamma: np.ndarray, x: np.ndarray, eps: float, out: np.ndarray) -> NormRecord:
    """Capture an RMSNorm: the rms scalar per row, gamma, output.

    The one-line ``rms = sqrt(mean(x^2) + eps)`` is the only formula kept in
    this module; it is a display scalar, not a multi-step operator.
    """
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)  # (B, S, 1)
    return {"rms": arr(rms), "gamma": arr(gamma), "out": arr(out)}


def _rope_record(rope_state: dict) -> RopeRecord:
    """Serialize the RoPE state the operator already computed."""
    return {
        "freqs": arr(rope_state["freqs"]),  # (pair_dim,)
        "angles": arr(rope_state["angles"]),  # (B, S, pair_dim)
        "cos": arr(rope_state["cos"]),
        "sin": arr(rope_state["sin"]),
    }


def _attn_record(state: dict, attn_out: np.ndarray) -> AttnRecord:
    """Serialize the attention state the operator already computed.

    ``state`` is either the dense ``_forward_state`` dict or the
    ``forward_step`` state dict; both carry the same keys (the step state
    additionally carries ``k_cache``/``v_cache`` — the full GQA-expanded
    cache the query attended to — which the dense path derives from the
    GQA-expanded k/v, identical to what the prefill stores).
    """
    k_cache = state.get("k_cache", state["k"])  # (B, H, S|t, hd)
    v_cache = state.get("v_cache", state["v"])  # (B, H, S|t, hd)
    return {
        "q_pre": arr(state["q_pre"]),  # (B, S, H·hd) — ln1_out @ Wq
        "k_pre": arr(state["k_pre"]),  # (B, S, G·hd)
        "v_pre": arr(state["v_pre"]),  # (B, S, G·hd)
        "q_rope_in": arr(state["q_rope_in"].transpose(0, 2, 1, 3)),  # (B, S, H, hd) pre-RoPE
        "k_rope_in": arr(state["k_rope_in"].transpose(0, 2, 1, 3)),  # (B, S, G, hd) pre-RoPE
        "q": arr(state["q"]),  # (B, H, S, hd) post-RoPE
        "k": arr(state["k"]),  # (B, H, S, hd) post-RoPE (GQA-expanded)
        "v": arr(state["v"]),  # (B, H, S, hd)
        "scale": float(state["scale"]),  # sqrt(hd) — the scores divisor
        "rope": _rope_record(state["rope"]),
        "scores": arr(np.nan_to_num(state["scores"], neginf=-1e4)),  # (B, H, S, S)
        "scores_masked": arr(np.nan_to_num(state["scores_masked"], neginf=-1e4)),  # (B, H, S, S)
        "causal_mask": arr(state["causal_mask"].astype(np.float64)),  # (S, S) or (1, t)
        "attn_weights": arr(state["attn"]),  # (B, H, S, S) — softmax output
        "ctx": arr(state["ctx"]),  # (B, S, H·hd)
        "attn_out": arr(attn_out),  # (B, S, D) — ctx @ Wo
        "k_cache": arr(k_cache),
        "v_cache": arr(v_cache),
    }


def _ffn_record(state: dict, out: np.ndarray, ff_dim: int) -> FFNRecord:
    """Serialize the dense-SwiGLU state the operator already computed."""
    return {
        "pre_gate": arr(state["pre_gate"]),
        "gate": arr(state["gate"]),
        "up": arr(state["up"]),
        "gated": arr(state["gated"]),
        "out": arr(out),
        "ff_dim": int(ff_dim),
    }


def _moe_record(state: dict, out: np.ndarray, n_experts: int, top_k: int) -> MoERecord:
    """Serialize the MoE state the operator already computed (all experts
    are computed in the track for display — see the PROD note in moe.py)."""
    return {
        "scores": arr(state["scores"]),  # (B, S, E) stable-softmax input
        "probs": arr(state["probs"]),  # (B, S, E) after the top-k mask
        "topk_idx": np.round(state["topk_idx"]).astype(int).tolist(),  # (B, S, k)
        "weights": arr(state["weights"]),  # (B, S, E) renormalized
        "expert_outs": [arr(e) for e in state["expert_outs"]],
        "shared_outs": [arr(e) for e in state.get("shared_outs", [])],  # (B, S, D) each
        "out": arr(out),
        "n_experts": n_experts,
        "top_k": top_k,
    }


def _block_record(block: TransformerBlock, bs: dict) -> BlockRecord:
    """Serialize one block's raw state (filled by ``TransformerBlock``)."""
    rec: BlockRecord = {
        "ln1": _rmsnorm_record(block.input_layernorm.gamma, bs["x"], block.input_layernorm.eps, bs["ln1_out"]),
        "attn": _attn_record(bs["attn"], bs["attn_out"]),
        "h": arr(bs["h"]),
        "ln2": _rmsnorm_record(
            block.post_attention_layernorm.gamma, bs["h"], block.post_attention_layernorm.eps, bs["ln2_out"]
        ),
        "out": arr(bs["out"]),
    }
    if isinstance(block.mlp, MixtureOfExperts):
        rec["moe"] = _moe_record(bs["ff"], bs["mlp_out"], block.mlp.n_experts, block.mlp.top_k)
    else:
        rec["ffn"] = _ffn_record(bs["ff"], bs["mlp_out"], int(block.mlp.down_proj.shape[0]))
    return rec


def _forward_record(model: NumPyModel, raw: dict, input_ids: np.ndarray) -> ForwardRecord:
    """Serialize one pass' raw state (filled by the model's forward path).

    ``raw`` holds x_in0, stack_out, positions, per-block state dicts
    ("blocks"), x_final, logits — the values the operators captured along
    the way. The softmax + top tokens are display extras derived from the
    logits (one stable softmax, no operator math).
    """
    logits = raw["logits"]
    # Display extras: the softmax distribution + top tokens at the last position.
    z = logits.astype(np.float64) - np.max(logits, axis=-1, keepdims=True)
    p = np.exp(z)
    p /= np.sum(p, axis=-1, keepdims=True)  # (B, S, V)
    V = p.shape[-1]
    top_n = min(10, V)
    top_idx = np.argsort(p[0, -1])[::-1][:top_n]  # top-10 at the last position
    top_tokens = [[int(i), float(p[0, -1, i])] for i in top_idx]

    return {
        "input_ids": np.asarray(input_ids).tolist(),
        "embedding": arr(raw["x_in0"]),
        "positions": raw["positions"].tolist(),
        "blocks": [_block_record(model.stack.layers[i], raw["blocks"][i]) for i in range(len(model.stack.layers))],
        "final_norm": _rmsnorm_record(model.final_norm.gamma, raw["stack_out"], model.final_norm.eps, raw["x_final"]),
        "logits": arr(logits),
        "softmax": arr(p),
        "top_tokens": top_tokens,
    }


# ---------------------------------------------------------------------------
# Instrumented forward — one pass, every intermediate captured
# ---------------------------------------------------------------------------


def instrumented_forward(model: NumPyModel, input_ids: np.ndarray) -> ForwardRecord:
    """One forward pass with every intermediate captured (the step record).

    The output tensors are identical to ``model.forward(input_ids)`` — the
    capture rides on the track's own state hooks (the same ``_forward_state``
    the analytic backward uses), so no intermediate is recomputed here.

    Returns a nested dict (all arrays as nested float lists):
      input_ids, embedding, positions,
      blocks[i]: {ln1, attn, h, ln2, ffn|moe, out},
      final_norm, logits, softmax, top_tokens
    """
    x = np.asarray(input_ids, dtype=np.int32)
    raw: dict = {}
    model.forward_with_trace(x, None, record=raw)
    return _forward_record(model, raw, x)


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

    Step 0 is **prefill**: one full forward over the prompt window (the same
    pass that initializes each block's KV cache — no second pass). Steps
    1..n are **autoregressive decode**: each step embeds ONLY the new token
    and runs the track's ``forward_step``, which appends the new K/V row to
    the cache and attends the single new query to the whole cache — no
    recomputation of past positions, and no causal mask needed (every cached
    row is in the past). The generated tokens are identical to the naive
    full-recompute loop (same math, same parameters), but each decode step's
    record shows the single new row plus the full cache it attended to.

    When the prompt is longer than ``context_length`` it is windowed to the
    last ``ctx`` tokens; the prefill then runs at the window's *absolute*
    RoPE positions (``position_offset``), so the decode steps — which always
    use absolute positions — see the same relative geometry as generating
    from the tail directly.

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

    # ---- Step 0: prefill the prompt window (initializes the caches)
    P = len(seq)
    P0 = min(ctx, P)
    window = list(seq[-P0:])
    x = np.array([window], dtype=np.int32)  # (1, P0)
    raw: dict = {}
    _logits, cache = model.forward_prefill(x, None, position_offset=P - P0, record=raw)
    rec = _forward_record(model, raw, x)
    tok = _pick(np.asarray(rec["softmax"], dtype=np.float64)[0][-1])
    steps.append(
        {
            "step": 0,
            "kind": "prefill",
            "position": int(x.shape[1] - 1) + (P - P0),  # last prompt position (absolute)
            "input_tokens": window,
            "forward": rec,
            "top_tokens": rec["top_tokens"],
            "token": tok,
            "text": vocab[tok],
        }
    )
    seq.append(tok)

    # ---- Steps 1..n: autoregressive decode, one new token per step
    for i in range(1, n_tokens):
        t = len(seq) - 1  # absolute position of the new token
        x1 = np.array([seq[-1:]], dtype=np.int32)  # (1, 1)
        raw = {}
        model.forward_step(x1, t, cache, record=raw)
        rec = _forward_record(model, raw, x1)
        tok = _pick(np.asarray(rec["softmax"], dtype=np.float64)[0][0])
        steps.append(
            {
                "step": i,
                "kind": "decode",
                "position": t,
                "input_tokens": [int(seq[-1])],  # only the new token is computed
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
