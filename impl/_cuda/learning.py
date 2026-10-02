"""CUDA learning-mode adapter — instrumented forward + inference records.

The CUDA sibling of ``impl._np.learning`` / ``impl._torch.learning``: the
exact same JSON record shapes (the record ``TypedDict``s are imported from
the NumPy track, so the learning page can consume every backend's output
interchangeably).

The CUDA model is bare metal: ``CUDAModel`` is a plain object holding raw
float32 tensors, and the production forward dispatches to NVRTC-compiled
kernels that expose no intermediates. For **records**, every block's
intermediates (RMSNorm rms, Q/K/V projections, RoPE, GQA repeat, raw
scores, causal mask, softmax weights, context, SwiGLU/MoE internals) are
recomputed with direct torch ops on the model's own parameter tensors,
in float64 — the display path, same as the torch adapter.

Because the display path is a parallel computation, every
``instrumented_forward`` also runs the model's real float32 CUDA forward
(when a GPU is available) and asserts the record's logits agree within the
repo's documented fp32 whole-model tolerance (``AGENTS.md`` rule 2:
``rtol=1e-2, atol=1e-2``). The record's chosen values are therefore exactly
the ones the CUDA model itself produces — not an unchecked recomputation.

Generation mirrors the NumPy/Torch record paths: step 0 is a full prefill
(``instrumented_forward``), after which each block's KV cache is
initialized from the prefill's recorded K/V, and steps 1..n are
autoregressive decode steps that embed ONLY the new token.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from shared.constants import REP_PENALTY

if TYPE_CHECKING:
    # Record shape contract: the JSON shapes are owned by the NumPy track.
    from impl._cuda.block import CuTransformerBlock
    from impl._cuda.model import CUDAModel
    from impl._np.learning import (
        AttnRecord,
        BlockRecord,
        FFNRecord,
        ForwardRecord,
        GenerationRecord,
        MoERecord,
        NormRecord,
        RopeRecord,
        StepRecord,
    )

# fp32 whole-model tolerance (AGENTS.md rule 2, top tier): the float64
# display path must agree with the real CUDA forward within this band.
_TOLERANCES = {"rtol": 1e-2, "atol": 1e-2}


# ---------------------------------------------------------------------------
# Tensor → JSON helpers (same rounding contract as impl._np.learning.arr)
# ---------------------------------------------------------------------------


def arr(t: torch.Tensor) -> list:
    """Convert a tensor to a nested list of floats (JSON-ready).

    All record values are float64, rounded to 8 decimals — identical to the
    NumPy/Torch records' ``arr`` so all backends serialize the same way.
    """
    return np.round(t.detach().cpu().numpy().astype(np.float64), 8).tolist()


def _w(t: torch.Tensor) -> torch.Tensor:
    """A parameter tensor as float64 for the display path."""
    return t.detach().cpu().to(torch.float64)


# ---------------------------------------------------------------------------
# Display-path math — same math as the track's kernels, in float64 torch ops
# ---------------------------------------------------------------------------


def _rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm recomputed in float64; returns (out, rms)."""
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)  # (B, S, 1)
    return x / rms * gamma, rms


def _rmsnorm_record(gamma: torch.Tensor, x: torch.Tensor, eps: float, out: torch.Tensor) -> NormRecord:
    """Capture an RMSNorm: the rms scalar per row, gamma, output."""
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)  # (B, S, 1)
    return {"rms": arr(rms), "gamma": arr(gamma), "out": arr(out)}


def _rope_tables(d_rot: int, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RoPE frequencies/angles/cos/sin (same math as kernels/rope.cu).

    positions: 1-D long tensor of S absolute positions. The CUDA kernel
    rotates adjacent pairs (x_2m, x_2m+1) with θ_m = 10000^(-2m/d_rot).
    """
    pair_dim = d_rot // 2
    freqs = 1.0 / (10000.0 ** (torch.arange(pair_dim, dtype=torch.float64) * 2.0 / d_rot))
    angles = positions.to(torch.float64).unsqueeze(-1) * freqs.unsqueeze(0)  # (S, pair_dim)
    return freqs, angles, (torch.cos(angles), torch.sin(angles))


def _rope(x: torch.Tensor, positions: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """Apply RoPE in float64 to x (B, S, H, hd); positions: 1-D (S,).

    rope_dim: 0 → rotate all head dims; 0 < rope_dim < hd → rotate the first
    rope_dim dims only (the rest pass through, un-rotated).
    """
    d = x.shape[-1]
    d_rot = rope_dim if (0 < rope_dim < d) else d
    _, _, (cos, sin) = _rope_tables(d_rot, positions)
    cos = cos.unsqueeze(1).unsqueeze(0)  # (1, S, 1, pair_dim)
    sin = sin.unsqueeze(1).unsqueeze(0)  # (1, S, 1, pair_dim)
    xr = x[..., :d_rot]
    x_even, x_odd = xr[..., 0::2], xr[..., 1::2]  # (B, S, H, pair_dim)
    y = torch.empty_like(xr)
    y[..., 0::2] = x_even * cos - x_odd * sin
    y[..., 1::2] = x_even * sin + x_odd * cos
    if d_rot < d:
        return torch.cat([y, x[..., d_rot:]], dim=-1)
    return y


def _rope_record(x_rot: torch.Tensor, positions: torch.Tensor, rope_dim: int) -> RopeRecord:
    """Recompute the RoPE frequencies/angles/cos/sin (same math as the track's RoPE).

    x_rot: (B, S, H, d) the tensor that will be rotated.
    rope_dim: 0 → rotate all head dims; 0 < rope_dim < d → rotate the first
    rope_dim dims only (the rest pass through, un-rotated).
    """
    d = x_rot.shape[-1]
    d_rot = rope_dim if (0 < rope_dim < d) else d
    pos = torch.as_tensor(positions, dtype=torch.long)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0).expand(x_rot.shape[0], x_rot.shape[1])  # (B, S)
    pair_dim = d_rot // 2
    freqs = 1.0 / (10000.0 ** (torch.arange(pair_dim, dtype=torch.float64) * 2.0 / d_rot))
    angles = pos.to(torch.float64).unsqueeze(-1) * freqs.unsqueeze(0)  # (B, S, pair_dim)
    cos, sin = torch.cos(angles), torch.sin(angles)
    return {
        "freqs": arr(freqs),
        "angles": arr(angles),
        "cos": arr(cos),
        "sin": arr(sin),
    }


def _attention_forward(
    block: CuTransformerBlock,
    ln1_out: torch.Tensor,
    positions: torch.Tensor,
    cache: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, AttnRecord]:
    """Run the block's attention math (float64 display path) + record.

    Two modes sharing one body:
    - ``cache is None``: prefill over the whole sequence — causal mask, and
      the record's k_cache/v_cache hold the full post-RoPE K/V.
    - ``cache is not None``: ONE decode step — the token's post-RoPE K/V are
      appended to the block's cache (mutated in place, like the NumPy
      track's ``forward_step``); nothing is masked, since every cached row
      precedes the new token.
    """
    B, S, _ = ln1_out.shape
    H, G, hd = block.n_heads, block.n_groups, block.head_dim
    Wq, Wk, Wv, Wo = (block.q_proj, block.k_proj, block.v_proj, block.o_proj)
    Wq, Wk, Wv, Wo = _w(Wq), _w(Wk), _w(Wv), _w(Wo)  # float64 display path

    # Projections: (B, S, D) → (B, S, ·)
    q_pre = ln1_out @ Wq  # (B, S, H·hd)
    k_pre = ln1_out @ Wk  # (B, S, G·hd)
    v_pre = ln1_out @ Wv  # (B, S, G·hd)
    q_rope_in = q_pre.view(B, S, H, hd).permute(0, 2, 1, 3)  # (B, H, S, hd) pre-RoPE
    k_rope_in = k_pre.view(B, S, G, hd).permute(0, 2, 1, 3)  # (B, G, S, hd) pre-RoPE
    v = v_pre.view(B, S, G, hd).permute(0, 2, 1, 3)  # (B, G, S, hd)

    # RoPE (the track's convention: adjacent pairs, θ_m over d_rot)
    q = _rope(q_rope_in.permute(0, 2, 1, 3), positions, block.rope_dim).permute(0, 2, 1, 3)  # (B, H, S, hd)
    k = _rope(k_rope_in.permute(0, 2, 1, 3), positions, block.rope_dim).permute(0, 2, 1, 3)  # (B, G, S, hd)

    if cache is not None:
        # Decode: append this token's K/V to the block's cache, then attend
        # over the WHOLE cache (no mask — all cached rows are in the past).
        cache["k"] = torch.cat([cache["k"], k], dim=2)  # (B, G, t, hd)
        cache["v"] = torch.cat([cache["v"], v], dim=2)  # (B, G, t, hd)
        k_all, v_all = cache["k"], cache["v"]
    else:
        k_all, v_all = k, v

    # GQA: broadcast each K/V group to its H // G query heads (display parity
    # with the NumPy record, which stores the GQA-expanded heads).
    k_r = k_all.repeat_interleave(H // G, dim=1) if G != H else k_all  # (B, H, t, hd)
    v_r = v_all.repeat_interleave(H // G, dim=1) if G != H else v_all  # (B, H, t, hd)
    k_x = k.repeat_interleave(H // G, dim=1) if G != H else k  # (B, H, S, hd) this step's K
    v_x = v.repeat_interleave(H // G, dim=1) if G != H else v  # (B, H, S, hd)
    t = k_r.shape[2]

    # Raw scores + mask (recomputed; the record shows pre/post mask)
    scale = math.sqrt(hd)  # the scores DIVISOR
    scores = (q @ k_r.transpose(-1, -2)) / scale  # (B, H, S, t)
    if cache is None:
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)  # True where j > i
        neg = torch.tensor(-1e4, dtype=ln1_out.dtype)
        scores_masked = torch.where(mask, neg, scores)  # (B, H, S, S)
    else:
        mask = torch.zeros(S, t, dtype=torch.bool)  # nothing masked in decode
        scores_masked = scores  # (B, H, S, t)

    z = scores_masked - scores_masked.max(dim=-1, keepdim=True).values
    attn_w = torch.exp(z)
    attn_w = attn_w / attn_w.sum(dim=-1, keepdim=True)  # (B, H, S, t)
    ctx = (attn_w @ v_r).permute(0, 2, 1, 3).reshape(B, S, H * hd)  # (B, S, H·hd)
    attn_out = ctx @ Wo  # (B, S, D)

    return attn_out, {
        "q_pre": arr(q_pre),  # (B, S, H·hd) — ln1_out @ Wq
        "k_pre": arr(k_pre),  # (B, S, G·hd)
        "v_pre": arr(v_pre),  # (B, S, G·hd)
        "q_rope_in": arr(q_rope_in.permute(0, 2, 1, 3)),  # (B, S, H, hd) pre-RoPE
        "k_rope_in": arr(k_rope_in.permute(0, 2, 1, 3)),  # (B, S, G, hd) pre-RoPE
        "q": arr(q),  # (B, H, S, hd) post-RoPE
        "k": arr(k_x),  # (B, H, S, hd) post-RoPE (GQA-expanded), this step's rows
        "v": arr(v_x),  # (B, H, S, hd)
        "scale": float(scale),  # sqrt(hd) — divide scores by this
        "rope": _rope_record(q_rope_in.permute(0, 2, 1, 3), positions, block.rope_dim),
        "scores": arr(scores),  # (B, H, S, t)
        "scores_masked": arr(scores_masked),  # (B, H, S, t)
        "causal_mask": arr(mask.to(torch.float64)),  # (S, t) — all 0 in decode
        "attn_weights": arr(attn_w),  # (B, H, S, t) — softmax output
        "ctx": arr(ctx),  # (B, S, H·hd)
        "attn_out": arr(attn_out),  # (B, S, D) — ctx @ Wo
        # The KV cache state AFTER this step: in prefill the whole prompt's
        # K/V are stored (GQA-expanded to H heads so the page can browse heads).
        "k_cache": arr(k_r),  # (B, H, t, hd)
        "v_cache": arr(v_r),  # (B, H, t, hd)
    }


def _init_kv_cache(
    block: CuTransformerBlock, k_pre: torch.Tensor, v_pre: torch.Tensor, positions: torch.Tensor
) -> dict:
    """Build a block's KV cache from a prefill pass.

    k_pre/v_pre are the pre-RoPE, un-split projections (B, S, G·hd). The
    cache stores K/V *per group* (GQA keeps it small); RoPE is applied to K
    with the prefill positions so later decode steps only rotate the NEW
    token and can attend to these cached rows unchanged.
    """
    B, S, _ = k_pre.shape
    G, hd = block.n_groups, block.head_dim
    k_heads = k_pre.view(B, S, G, hd)  # (B, S, G, hd) pre-RoPE
    v_heads = v_pre.view(B, S, G, hd).permute(0, 2, 1, 3)  # (B, G, S, hd)
    k_heads = _rope(k_heads, positions, block.rope_dim).permute(0, 2, 1, 3)  # (B, G, S, hd) post-RoPE
    return {"k": k_heads, "v": v_heads}  # each (B, G, S, hd)


# ---------------------------------------------------------------------------
# Feed-forward records (dense SwiGLU and MoE)
# ---------------------------------------------------------------------------


def _swiglu(x: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """SwiGLU FFN in float64: SiLU(x @ gate) * (x @ up) @ down."""
    return (F.silu(x @ gate) * (x @ up)) @ down


def _ffn_record(block: CuTransformerBlock, x: torch.Tensor) -> tuple[torch.Tensor, FFNRecord]:
    """Capture dense SwiGLU: pre_gate, gate (post-SiLU), up, gated, output."""
    gate_w, up_w, down_w = _w(block.gate_proj), _w(block.up_proj), _w(block.down_proj)
    pre_gate = x @ gate_w  # (B, S, FF)
    gate = F.silu(pre_gate)  # (B, S, FF)
    up = x @ up_w  # (B, S, FF)
    gated = gate * up  # (B, S, FF)
    out = gated @ down_w  # (B, S, D)
    return out, {
        "pre_gate": arr(pre_gate),
        "gate": arr(gate),
        "up": arr(up),
        "gated": arr(gated),
        "out": arr(out),
        "ff_dim": int(down_w.shape[0]),
    }


def _moe_record(block: CuTransformerBlock, x: torch.Tensor) -> tuple[torch.Tensor, MoERecord]:
    """Capture MoE: router scores, softmax probs, top-k selection, per-expert outputs.

    Returns (out, record); ``out`` reproduces the block's MoE math in
    float64: Σ_j w_j·E_j(x) + mean(shared experts) (ADR 0002).
    """
    E = block.n_experts
    top_k = block.config.top_k
    scores = x @ _w(block.router)  # (B, S, E)
    scores_stable = scores - scores.max(dim=-1, keepdim=True).values  # (B, S, E)
    probs = torch.softmax(scores, dim=-1)  # (B, S, E)
    if top_k < E:
        order = torch.argsort(probs, dim=-1, descending=True)  # (B, S, E) descending
        topk_idx = order[..., :top_k]  # (B, S, k)
        kth_idx = order[..., top_k - 1 : top_k]
        threshold = torch.gather(probs, -1, kth_idx)  # (B, S, 1)
        probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
        weights = probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1e-8)  # (B, S, E)
    else:
        topk_idx = torch.argsort(probs, dim=-1, descending=True)[..., :top_k]
        weights = probs
    eg, eu, ed = _w(block.expert_gate_proj), _w(block.expert_up_proj), _w(block.expert_down_proj)
    expert_outs = [_swiglu(x, eg[e], eu[e], ed[e]) for e in range(E)]  # E × (B, S, D)
    # Weighted sum (same loop math as CuTransformerBlock._moe_forward).
    out = torch.zeros_like(x)  # (B, S, D)
    for e in range(E):
        out = out + weights[..., e : e + 1] * expert_outs[e]
    shared_outs: list[torch.Tensor] = []
    n_shared = block.config.n_shared_experts
    if n_shared > 0:
        sg, su, sd = _w(block.shared_gate_proj), _w(block.shared_up_proj), _w(block.shared_down_proj)
        shared_outs = [_swiglu(x, sg[s], su[s], sd[s]) for s in range(n_shared)]  # n_shared × (B, S, D)
        out = out + torch.stack(shared_outs).mean(dim=0)  # (B, S, D)
    return out, {
        "scores": arr(scores_stable),  # (B, S, E) stable-softmaxed
        "probs": arr(probs),  # (B, S, E) after top-k mask
        "topk_idx": topk_idx.tolist(),  # (B, S, k)
        "weights": arr(weights),  # (B, S, E) renormalized
        "expert_outs": [arr(e) for e in expert_outs],
        "shared_outs": [arr(e) for e in shared_outs],
        "out": arr(out),
        "n_experts": E,
        "top_k": top_k,
    }


# ---------------------------------------------------------------------------
# Instrumented forward — one pass with every intermediate captured
# ---------------------------------------------------------------------------


def _softmax64(logits: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax in float64 (the record's display distribution)."""
    z = logits - logits.max(dim=-1, keepdim=True).values
    p = torch.exp(z)
    return p / p.sum(dim=-1, keepdim=True)


def _top_tokens(p_last: torch.Tensor, top_n: int = 10) -> list[list]:
    """[token_id, prob] pairs for the top_n tokens of a probability vector."""
    top_n = min(top_n, p_last.shape[-1])
    idx = torch.argsort(p_last, descending=True)[:top_n]
    return [[int(i), float(p_last[i])] for i in idx.tolist()]


def _validate_against_real_forward(model: CUDAModel, input_ids: torch.Tensor, logits64: torch.Tensor) -> None:
    """Assert the float64 display logits match the model's real CUDA forward.

    The record's chosen values must be the ones the CUDA model itself
    produces — this guard catches any drift between the NVRTC kernels and
    the recomputed display path (fp32 whole-model tolerance, AGENTS.md 2).
    No-op when no GPU is available (tests then still exercise the math).
    """
    if not torch.cuda.is_available():
        return
    real = model.forward(input_ids.to("cuda")).detach().cpu().to(torch.float64)  # (B, S, V)
    assert torch.allclose(real, logits64, **_TOLERANCES), (
        f"CUDA record display path diverged from the real forward: "
        f"max|Δ|={float((real - logits64).abs().max()):.6f} exceeds {_TOLERANCES['atol']}"
    )


def _instrumented_blocks(
    model: CUDAModel,
    stream: torch.Tensor,
    positions: torch.Tensor,
    caches: list[dict] | None = None,
) -> tuple[torch.Tensor, list[BlockRecord]]:
    """Run every block of the stack in float64, capturing each block record.

    caches: per-block KV caches for decode mode; None for a full prefill.
    """
    blocks: list[BlockRecord] = []
    for i, block in enumerate(model.stacking.blocks):
        eps = model.config.norm_eps
        ln1_out, _ = _rmsnorm(stream, _w(block.input_layernorm_gamma), eps)  # (B, S, D)
        attn_out, attn_rec = _attention_forward(
            block, ln1_out, positions, cache=None if caches is None else caches[i]
        )  # (B, S, D)
        h = stream + attn_out  # (B, S, D) residual after attention
        ln2_out, _ = _rmsnorm(h, _w(block.post_attention_layernorm_gamma), eps)  # (B, S, D)
        if block.config.has_moe():
            ff_out, ffn_rec = _moe_record(block, ln2_out)  # (B, S, D)
        else:
            ff_out, ffn_rec = _ffn_record(block, ln2_out)  # (B, S, D)
        out = h + ff_out  # (B, S, D) block output
        block_rec: BlockRecord = {
            "ln1": _rmsnorm_record(_w(block.input_layernorm_gamma), stream, eps, ln1_out),
            "attn": attn_rec,
            "h": arr(h),
            "ln2": _rmsnorm_record(_w(block.post_attention_layernorm_gamma), h, eps, ln2_out),
            "out": arr(out),
        }
        if block.config.has_moe():
            block_rec["moe"] = ffn_rec
        else:
            block_rec["ffn"] = ffn_rec
        blocks.append(block_rec)
        stream = out
    return stream, blocks


def instrumented_forward(model: CUDAModel, input_ids: torch.Tensor, *, validate: bool = True) -> ForwardRecord:
    """One forward pass with every intermediate captured (the step record).

    Same record shape as ``impl._np.learning.instrumented_forward``; every
    block's internals are the explicit float64 display path (recomputed from
    the model's own parameter tensors) since the production CUDA kernels
    expose none of them. ``validate`` (default) cross-checks the result
    against the model's real float32 CUDA forward.

    Returns a nested dict (all arrays as nested float lists):
      input_ids, embedding, positions,
      blocks[i]: {ln1, attn, h, ln2, ffn|moe, out},
      final_norm, logits, softmax, top_tokens
    """
    x = input_ids.detach().cpu().to(torch.long)
    S = x.shape[1]
    positions = torch.arange(S, dtype=torch.long)

    # Embedding: (B, S) → (B, S, D)
    emb = _w(model.embedding_weights)[x]

    # Blocks: each recomputed with direct torch ops; intermediates captured.
    stream, blocks = _instrumented_blocks(model, emb, positions)

    # Final norm + lm_head: (B, S, D) → (B, S, D) → (B, S, V)
    normed, _ = _rmsnorm(stream, _w(model.final_norm_gamma), model.config.norm_eps)
    logits = normed @ _w(model.lm_head_weight)  # (B, S, V)

    if validate:
        _validate_against_real_forward(model, x, logits)

    # Display extras: the softmax distribution + top tokens at the last position.
    p = _softmax64(logits)  # (B, S, V)
    top_tokens = _top_tokens(p[0, -1])  # top-10 at the last position

    return {
        "input_ids": x.tolist(),
        "embedding": arr(emb),
        "positions": positions.tolist(),
        "blocks": blocks,
        "final_norm": _rmsnorm_record(_w(model.final_norm_gamma), stream, model.config.norm_eps, normed),
        "logits": arr(logits),
        "softmax": arr(p),
        "top_tokens": top_tokens,
    }


# ---------------------------------------------------------------------------
# Multi-token generation with a record per token step
# ---------------------------------------------------------------------------


def generate_with_records(
    model: CUDAModel,
    vocab: list[str],
    prompt_ids: list[int],
    n_tokens: int,
    temp: float | None = None,
    top_k: int | None = None,
    seed: int = 42,
) -> GenerationRecord:
    """Generate ``n_tokens`` with a real KV cache, recording every step.

    Same record shape and semantics as the NumPy/Torch adapters: step 0 is
    a **prefill** (one full ``instrumented_forward`` over the prompt, after
    which each block's KV cache is initialized from the recorded pre-RoPE
    K/V projections), then steps 1..n are autoregressive decode steps that
    embed ONLY the new token and attend to the whole cache (no recomputation
    of past positions, no causal mask — every cached row is in the past).

    ``temp is None`` → greedy (argmax); otherwise sample with temperature
    scaling and optional top-k filtering (seeded, via the same NumPy RNG
    formula as the other tracks so identically distributed steps draw the
    same token).

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

    emitted: list[int] = []  # generated tokens so far (repeat guard, cf. NumPy track)

    def _pick(p_last: np.ndarray) -> int:
        """Greedy argmax, or seeded temperature/top-k sampling.

        Identical formula AND identical repetition guard as the NumPy
        track's ``_pick`` (impl/_np/learning.py) and the server's
        ``_sample``: penalty on already-emitted tokens + a hard block on the
        immediately previous token. All record adapters must stay equivalent
        — the page consumes any backend's records interchangeably.
        """
        z = np.log(p_last + 1e-30)
        if temp is not None:
            z = z / temp
        if emitted:
            for tid in set(emitted):
                if 0 <= tid < len(z):
                    z[tid] = z[tid] / REP_PENALTY if z[tid] > 0 else z[tid] * REP_PENALTY
            z[emitted[-1]] = -np.inf  # never immediately repeat
        if temp is None:
            return int(np.argmax(z))
        if top_k is not None:
            kth = np.partition(z, -top_k)[-1]
            z = np.where(z < kth, -np.inf, z)
        z = z - z.max()
        pk = np.exp(z)
        pk /= pk.sum()
        return int(rng.choice(len(pk), p=pk))

    # ---- Step 0: prefill the prompt (full forward, initializes the caches)
    x = torch.tensor([seq[-ctx:]], dtype=torch.int64)
    rec = instrumented_forward(model, x)
    tok = _pick(np.asarray(rec["softmax"], dtype=np.float64)[0][-1])
    emitted.append(tok)
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
    p0 = list(x[0].tolist())
    positions0 = torch.arange(len(p0), dtype=torch.long) + (len(seq) - 1 - len(p0))  # absolute
    caches: list[dict] = []
    for i, block in enumerate(model.stacking.blocks):
        r = rec["blocks"][i]["attn"]
        # (as in the NumPy track) the cache is initialized from the recorded
        # pre-RoPE projections, so the display and the state agree exactly.
        caches.append(_init_kv_cache(block, torch.tensor(r["k_pre"]), torch.tensor(r["v_pre"]), positions0))

    # ---- Steps 1..n: autoregressive decode, one new token per step
    for i in range(1, n_tokens):
        t = len(seq) - 1  # absolute position of the new token
        x1 = torch.tensor([[seq[-1]]], dtype=torch.int64)  # (1, 1)
        emb = _w(model.embedding_weights)[x1]  # (1, 1, D)

        stream, blocks = _instrumented_blocks(model, emb, torch.tensor([t]), caches)

        normed, _ = _rmsnorm(stream, _w(model.final_norm_gamma), model.config.norm_eps)  # (1, 1, D)
        logits = normed @ _w(model.lm_head_weight)  # (1, 1, V)
        p = _softmax64(logits)  # (1, 1, V)
        top_tokens = _top_tokens(p[0, 0])  # top-10 (single position)

        rec: ForwardRecord = {
            "input_ids": x1.tolist(),
            "embedding": arr(emb),
            "positions": [t],
            "blocks": blocks,
            "final_norm": _rmsnorm_record(_w(model.final_norm_gamma), stream, model.config.norm_eps, normed),
            "logits": arr(logits),
            "softmax": arr(p),
            "top_tokens": top_tokens,
        }
        tok = _pick(p[0, 0].detach().cpu().numpy())
        emitted.append(tok)
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
