"""PyTorch learning-mode adapter — instrumented forward + inference records.

The PyTorch sibling of ``impl._np.learning``: the exact same JSON record
shapes (the record ``TypedDict``s are imported from there, so the learning
page can consume either backend's output interchangeably).

The model's own forward is the **production** path — the attention block
uses the fused ``F.scaled_dot_product_attention`` call, which exposes no
intermediates. For **records**, the attention intermediates (Q/K/V
projections, RoPE, GQA repeat, raw scores, causal mask, softmax weights,
context) are recomputed with direct torch ops on the model's own weights —
a display path, documented as such, with the math in float64 (the adapter
runs the model in float64 via ``model.double()``).

Generation mirrors the NumPy record path: step 0 is a full prefill
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

from impl._torch.layers import MixtureOfExperts, MultiHeadAttention, SwiGLUFFN

if TYPE_CHECKING:
    # Record shape contract: the JSON shapes are owned by the NumPy track.
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
    from impl._torch.layers import TorchModel


# ---------------------------------------------------------------------------
# Tensor → JSON helpers (same rounding contract as impl._np.learning.arr)
# ---------------------------------------------------------------------------


def arr(t: torch.Tensor) -> list:
    """Convert a tensor to a nested list of floats (JSON-ready).

    All record values are float64, rounded to 8 decimals — identical to the
    NumPy record's ``arr`` so the two backends serialize the same way.
    """
    return np.round(t.detach().cpu().numpy().astype(np.float64), 8).tolist()


# ---------------------------------------------------------------------------
# Record helpers — same math as impl._np.learning, on torch tensors
# ---------------------------------------------------------------------------


def _rmsnorm_record(gamma: torch.Tensor, x: torch.Tensor, eps: float, out: torch.Tensor) -> NormRecord:
    """Capture an RMSNorm: the rms scalar per row, gamma, output."""
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)  # (B, S, 1)
    return {"rms": arr(rms), "gamma": arr(gamma), "out": arr(out)}


def _rope_record(x_rot: torch.Tensor, positions: torch.Tensor, rope_dim: int) -> RopeRecord:
    """Recompute the RoPE frequencies/angles/cos/sin (same math as the track's RoPE).

    x_rot: (B, S, H, d) the tensor that will be rotated.
    rope_dim: 0 → rotate all head dims; 0 < rope_dim < d → rotate the first
    rope_dim dims only (the rest pass through, un-rotated).
    """
    d = x_rot.shape[-1]
    d_rot = rope_dim if (0 < rope_dim < d) else d
    pair_dim = d_rot // 2
    pos = torch.as_tensor(positions, dtype=torch.long, device=x_rot.device)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0).expand(x_rot.shape[0], x_rot.shape[1])
    freqs = 1.0 / (10000.0 ** (torch.arange(pair_dim, dtype=torch.float64, device=x_rot.device) * 2.0 / d_rot))
    angles = pos.to(torch.float64).unsqueeze(-1) * freqs.unsqueeze(0)  # (B, S, pair_dim)
    return {
        "freqs": arr(freqs),
        "angles": arr(angles),
        "cos": arr(torch.cos(angles)),
        "sin": arr(torch.sin(angles)),
    }


def _attn_record(
    attn: MultiHeadAttention, ln1_out: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, AttnRecord]:
    """Run the block's attention math and capture every intermediate.

    Returns (attn_out, record). The Q/K/V projections, RoPE, GQA repeat,
    scores, causal mask, softmax weights, and context are recomputed with
    direct torch ops on the model's own weights — the display path for the
    record (the fused SDPA call exposes none of these).
    # PROD: production path is F.scaled_dot_product_attention inside
    MultiHeadAttention.forward — the fused call is the default and unchanged.
    """
    B, S, _ = ln1_out.shape
    H, G, hd = attn.n_heads, attn.n_groups, attn.head_dim

    # Projections: (B, S, D) → (B, S, ·)
    q_pre = attn.q_proj(ln1_out)  # (B, S, H·hd)
    k_pre = attn.k_proj(ln1_out)  # (B, S, G·hd)
    v_pre = attn.v_proj(ln1_out)  # (B, S, G·hd)
    q_rope_in = q_pre.view(B, S, H, hd).permute(0, 2, 1, 3)  # (B, H, S, hd) pre-RoPE
    k_rope_in = k_pre.view(B, S, G, hd).permute(0, 2, 1, 3)  # (B, G, S, hd) pre-RoPE
    v = v_pre.view(B, S, G, hd).permute(0, 2, 1, 3)  # (B, G, S, hd)

    # RoPE (the model's own module): (B, H, S, hd) → (B, S, H, hd) → back
    q = attn.rope(q_rope_in.permute(0, 2, 1, 3), positions, rope_dim=attn.rope_dim).permute(0, 2, 1, 3)  # (B, H, S, hd)
    k = attn.rope(k_rope_in.permute(0, 2, 1, 3), positions, rope_dim=attn.rope_dim).permute(0, 2, 1, 3)  # (B, G, S, hd)

    # GQA: broadcast each K/V group to its H // G query heads (display parity
    # with the NumPy record, which stores the GQA-expanded heads).
    if G != H:
        k = k.repeat_interleave(H // G, dim=1)  # (B, H, S, hd)
        v = v.repeat_interleave(H // G, dim=1)  # (B, H, S, hd)

    # Raw scores + causal mask (recomputed; the record shows pre/post mask)
    scale = math.sqrt(hd)  # the scores DIVISOR
    scores = (q @ k.transpose(-1, -2)) / scale  # (B, H, S, S)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=ln1_out.device), diagonal=1)  # True where j > i
    neg = torch.tensor(-1e4, dtype=ln1_out.dtype, device=ln1_out.device)
    scores_masked = torch.where(mask, neg, scores)  # (B, H, S, S)

    z = scores_masked - scores_masked.max(dim=-1, keepdim=True).values
    attn_w = torch.exp(z)
    attn_w = attn_w / attn_w.sum(dim=-1, keepdim=True)  # (B, H, S, S)
    ctx = (attn_w @ v).permute(0, 2, 1, 3).reshape(B, S, H * hd)  # (B, S, H·hd)
    attn_out = attn.o_proj(ctx)  # (B, S, D)

    return attn_out, {
        "q_pre": arr(q_pre),  # (B, S, H·hd) — ln1_out @ Wq
        "k_pre": arr(k_pre),  # (B, S, G·hd)
        "v_pre": arr(v_pre),  # (B, S, G·hd)
        "q_rope_in": arr(q_rope_in.permute(0, 2, 1, 3)),  # (B, S, H, hd) pre-RoPE
        "k_rope_in": arr(k_rope_in.permute(0, 2, 1, 3)),  # (B, S, G, hd) pre-RoPE
        "q": arr(q),  # (B, H, S, hd) post-RoPE
        "k": arr(k),  # (B, H, S, hd) post-RoPE (GQA-expanded)
        "v": arr(v),  # (B, H, S, hd)
        "scale": float(scale),  # sqrt(hd) — divide scores by this
        "rope": _rope_record(q_rope_in.permute(0, 2, 1, 3), positions, attn.rope_dim),
        "scores": arr(scores),  # (B, H, S, S)
        "scores_masked": arr(scores_masked),  # (B, H, S, S)
        "causal_mask": arr(mask.to(torch.float64)),  # (S, S)
        "attn_weights": arr(attn_w),  # (B, H, S, S) — softmax output
        "ctx": arr(ctx),  # (B, S, H·hd)
        "attn_out": arr(attn_out),  # (B, S, D) — ctx @ Wo
        # The KV cache state AFTER this step: in prefill the whole prompt's
        # K/V are stored (GQA-expanded to H heads so the page can browse heads).
        "k_cache": arr(k),  # (B, H, S, hd)
        "v_cache": arr(v),  # (B, H, S, hd)
    }


def _init_kv_cache(attn: MultiHeadAttention, k_pre: torch.Tensor, v_pre: torch.Tensor, positions: torch.Tensor) -> dict:
    """Build a block's KV cache from a prefill pass.

    k_pre/v_pre are the pre-RoPE, un-split projections (B, S, G·hd). The
    cache stores K/V *per group* (GQA keeps it small); RoPE is applied to K
    with the prefill positions so later decode steps only rotate the NEW
    token and can attend to these cached rows unchanged.
    """
    B, S, _ = k_pre.shape
    G, hd = attn.n_groups, attn.head_dim
    k_heads = k_pre.view(B, S, G, hd).permute(0, 2, 1, 3)  # (B, G, S, hd) pre-RoPE
    v_heads = v_pre.view(B, S, G, hd).permute(0, 2, 1, 3)  # (B, G, S, hd)
    k_heads = attn.rope(k_heads.permute(0, 2, 1, 3), positions, rope_dim=attn.rope_dim).permute(0, 2, 1, 3)  # post-RoPE
    return {"k": k_heads, "v": v_heads}  # each (B, G, S, hd)


def _attn_step_record(
    attn: MultiHeadAttention, x: torch.Tensor, position: int, cache: dict
) -> tuple[torch.Tensor, AttnRecord]:
    """Run ONE decode step of the attention math and capture the record.

    The intermediates are recomputed with the same math as the production
    MHA (same weights, same RoPE, same GQA repeat) — the cache is appended
    first, so the displayed scores/weights are exactly what produced the
    output. Nothing is masked in decode: every cached row is in the past of
    the new token, so the causal mask is all zeros and the row spans
    positions 1..t (the whole cache).
    # PROD: production path is F.scaled_dot_product_attention inside
    MultiHeadAttention.forward — the fused call is the default and unchanged.
    """
    B = x.shape[0]
    H, G, hd = attn.n_heads, attn.n_groups, attn.head_dim
    positions = torch.tensor([position], dtype=torch.long, device=x.device)  # (1,)

    q_pre = attn.q_proj(x)  # (B, 1, H·hd)
    k_pre = attn.k_proj(x)  # (B, 1, G·hd)
    v_pre = attn.v_proj(x)  # (B, 1, G·hd)
    q_rope_in = q_pre.view(B, 1, H, hd).permute(0, 2, 1, 3)  # (B, H, 1, hd) pre-RoPE
    k_rope_in = k_pre.view(B, 1, G, hd).permute(0, 2, 1, 3)  # (B, G, 1, hd) pre-RoPE
    v = v_pre.view(B, 1, G, hd).permute(0, 2, 1, 3)  # (B, G, 1, hd)
    q = attn.rope(q_rope_in.permute(0, 2, 1, 3), positions, rope_dim=attn.rope_dim).permute(0, 2, 1, 3)  # (B, H, 1, hd)
    k = attn.rope(k_rope_in.permute(0, 2, 1, 3), positions, rope_dim=attn.rope_dim).permute(0, 2, 1, 3)  # (B, G, 1, hd)

    # Append the new row to the block's cache (mutated in place, like the
    # NumPy track's forward_step).
    cache["k"] = torch.cat([cache["k"], k], dim=2)  # (B, G, t, hd)
    cache["v"] = torch.cat([cache["v"], v], dim=2)  # (B, G, t, hd)

    if G != H:  # GQA: broadcast each group to its query heads (display parity with _attn_record)
        k = k.repeat_interleave(H // G, dim=1)  # (B, H, 1, hd)
        v = v.repeat_interleave(H // G, dim=1)  # (B, H, 1, hd)
    k_r = cache["k"].repeat_interleave(H // G, dim=1) if G != H else cache["k"]  # (B, H, t, hd)
    v_r = cache["v"].repeat_interleave(H // G, dim=1) if G != H else cache["v"]  # (B, H, t, hd)
    t = k_r.shape[2]

    scale = math.sqrt(hd)
    scores = (q @ k_r.transpose(-1, -2)) / scale  # (B, H, 1, t) over the WHOLE cache
    mask = torch.zeros(1, t, dtype=torch.bool, device=x.device)  # nothing masked in decode
    z = scores - scores.max(dim=-1, keepdim=True).values
    attn_w = torch.exp(z)
    attn_w = attn_w / attn_w.sum(dim=-1, keepdim=True)  # (B, H, 1, t)
    ctx = (attn_w @ v_r).permute(0, 2, 1, 3).reshape(B, 1, H * hd)  # (B, 1, H·hd)
    out = attn.o_proj(ctx)  # (B, 1, D)

    return out, {
        "q_pre": arr(q_pre),  # (B, 1, H·hd)
        "k_pre": arr(k_pre),  # (B, 1, G·hd)
        "v_pre": arr(v_pre),  # (B, 1, G·hd)
        "q_rope_in": arr(q_rope_in.permute(0, 2, 1, 3)),  # (B, 1, H, hd) pre-RoPE
        "k_rope_in": arr(k_rope_in.permute(0, 2, 1, 3)),  # (B, 1, G, hd) pre-RoPE
        "q": arr(q),  # (B, H, 1, hd) post-RoPE
        "k": arr(k),  # (B, H, 1, hd) post-RoPE (GQA-expanded)
        "v": arr(v),  # (B, H, 1, hd)
        "scale": scale,  # sqrt(hd)
        "rope": _rope_record(q_rope_in.permute(0, 2, 1, 3), positions, attn.rope_dim),
        "scores": arr(scores),  # (B, H, 1, t)
        "scores_masked": arr(scores),  # (B, H, 1, t) — mask is all zeros
        "causal_mask": arr(mask.to(torch.float64)),  # (1, t) — all 0 in decode
        "attn_weights": arr(attn_w),  # (B, H, 1, t)
        "ctx": arr(ctx),  # (B, 1, H·hd)
        "attn_out": arr(out),  # (B, 1, D)
        "k_cache": arr(k_r),  # (B, H, t, hd) — the full cache after this append
        "v_cache": arr(v_r),  # (B, H, t, hd)
    }


def _ffn_record(mlp: SwiGLUFFN, x: torch.Tensor, out: torch.Tensor) -> FFNRecord:
    """Capture dense SwiGLU: pre_gate, gate (post-SiLU), up, gated, output."""
    pre_gate = x @ mlp.gate_proj  # (B, S, FF)
    gate = F.silu(pre_gate)  # (B, S, FF)
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


def _moe_record(mlp: MixtureOfExperts, x: torch.Tensor, out: torch.Tensor) -> MoERecord:
    """Capture MoE: router scores, softmax probs, top-k selection, per-expert outputs."""
    E = mlp.n_experts
    scores = mlp.gate(x)  # (B, S, E)
    scores_stable = scores - scores.max(dim=-1, keepdim=True).values  # (B, S, E)
    probs = torch.softmax(scores, dim=-1)  # (B, S, E)
    if mlp.top_k < E:
        order = torch.argsort(probs, dim=-1, descending=True)  # (B, S, E) descending
        topk_idx = order[..., : mlp.top_k]  # (B, S, k)
        kth_idx = order[..., mlp.top_k - 1 : mlp.top_k]
        threshold = torch.gather(probs, -1, kth_idx)  # (B, S, 1)
        probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
        weights = probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1e-8)  # (B, S, E)
    else:
        topk_idx = torch.argsort(probs, dim=-1, descending=True)[..., : mlp.top_k]
        weights = probs
    expert_outs = [expert(x) for expert in mlp.experts]  # E × (B, S, D)
    return {
        "scores": arr(scores_stable),  # (B, S, E) stable-softmaxed
        "probs": arr(probs),  # (B, S, E) after top-k mask
        "topk_idx": topk_idx.tolist(),  # (B, S, k)
        "weights": arr(weights),  # (B, S, E) renormalized
        "expert_outs": [arr(e) for e in expert_outs],
        "out": arr(out),
        "n_experts": E,
        "top_k": mlp.top_k,
    }


def instrumented_forward(model: TorchModel, input_ids: torch.Tensor) -> ForwardRecord:
    """One forward pass with every intermediate captured (the step record).

    Same record shape as ``impl._np.learning.instrumented_forward``; the
    attention block's intermediates are the explicit display path (see
    ``_attn_record``) since the production SDPA call exposes none of them.

    Returns a nested dict (all arrays as nested float lists):
      input_ids, embedding, positions,
      blocks[i]: {ln1, attn, h, ln2, ffn|moe, out},
      final_norm, logits, softmax, top_tokens
    """
    x = input_ids
    S = x.shape[1]
    positions = torch.arange(S, dtype=torch.long, device=x.device)

    # Embedding: (B, S) → (B, S, D)
    emb = model.embedding(x)

    # Blocks: each runs through the real component; intermediates captured.
    stream = emb
    blocks: list[BlockRecord] = []
    for block in model.stack.blocks:
        ln1 = block.input_layernorm
        ln1_out = ln1(stream)  # (B, S, D)
        attn_out, attn_rec = _attn_record(block.self_attn, ln1_out, positions)  # (B, S, D)
        h = stream + attn_out  # (B, S, D) residual after attention
        ln2 = block.post_attention_layernorm
        ln2_out = ln2(h)  # (B, S, D)
        mlp = block.mlp
        ff_out = mlp(ln2_out)  # (B, S, D)
        out = h + ff_out  # (B, S, D) block output
        block_rec: BlockRecord = {
            "ln1": _rmsnorm_record(ln1.weight, stream, ln1.eps, ln1_out),
            "attn": attn_rec,
            "h": arr(h),
            "ln2": _rmsnorm_record(ln2.weight, h, ln2.eps, ln2_out),
            "out": arr(out),
        }
        if isinstance(mlp, MixtureOfExperts):
            block_rec["moe"] = _moe_record(mlp, ln2_out, ff_out)
        else:
            block_rec["ffn"] = _ffn_record(mlp, ln2_out, ff_out)
        blocks.append(block_rec)
        stream = out

    # Final norm + lm_head: (B, S, D) → (B, S, D) → (B, S, V)
    normed = model.final_norm(stream)
    logits = model.lm_head(normed)

    # Display extras: the softmax distribution + top tokens at the last position.
    z = logits - logits.max(dim=-1, keepdim=True).values
    p = torch.exp(z)
    p = p / p.sum(dim=-1, keepdim=True)  # (B, S, V)
    p = p.detach()
    V = p.shape[-1]
    top_n = min(10, V)
    top_idx = torch.argsort(p[0, -1], descending=True)[:top_n]  # top-10 at the last position
    top_tokens = [[int(i), float(p[0, -1, i])] for i in top_idx.tolist()]

    return {
        "input_ids": x.tolist(),
        "embedding": arr(emb),
        "positions": positions.tolist(),
        "blocks": blocks,
        "final_norm": _rmsnorm_record(model.final_norm.weight, stream, model.final_norm.eps, normed),
        "logits": arr(logits),
        "softmax": arr(p),
        "top_tokens": top_tokens,
    }


# ---------------------------------------------------------------------------
# Multi-token generation with a record per token step
# ---------------------------------------------------------------------------


def generate_with_records(
    model: TorchModel,
    vocab: list[str],
    prompt_ids: list[int],
    n_tokens: int,
    temp: float | None = None,
    top_k: int | None = None,
    seed: int = 42,
) -> GenerationRecord:
    """Generate ``n_tokens`` with a real KV cache, recording every step.

    Same record shape and semantics as ``impl._np.learning.generate_with_records``:
    step 0 is a **prefill** (one full ``instrumented_forward`` over the
    prompt, after which each block's KV cache is initialized from the
    recorded pre-RoPE K/V projections), then steps 1..n are autoregressive
    decode steps that embed ONLY the new token and attend to the whole cache
    (no recomputation of past positions, no causal mask — every cached row
    is in the past).

    ``temp is None`` → greedy (argmax); otherwise sample with temperature
    scaling and optional top-k filtering (seeded, via the same NumPy RNG
    formula as the NumPy track so identically distributed steps draw the
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

    def _pick(p_last: np.ndarray) -> int:
        """Greedy argmax, or seeded temperature/top-k sampling (same formula as the NumPy track)."""
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
    x = torch.tensor([seq[-ctx:]], dtype=torch.int64)
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
    p0 = list(x[0].tolist())
    positions0 = torch.arange(len(p0), dtype=torch.long) + (len(seq) - 1 - len(p0))  # absolute
    caches: list[dict] = []
    for i, block in enumerate(model.stack.blocks):
        r = rec["blocks"][i]["attn"]
        # (as in the NumPy track) the cache is initialized from the recorded
        # pre-RoPE projections, so the display and the state agree exactly.
        caches.append(_init_kv_cache(block.self_attn, torch.tensor(r["k_pre"]), torch.tensor(r["v_pre"]), positions0))

    # ---- Steps 1..n: autoregressive decode, one new token per step
    for i in range(1, n_tokens):
        t = len(seq) - 1  # absolute position of the new token
        x1 = torch.tensor([[seq[-1]]], dtype=torch.int64)  # (1, 1)
        emb = model.embedding(x1)  # (1, 1, D)
        stream = emb
        blocks: list[BlockRecord] = []
        for j, block in enumerate(model.stack.blocks):
            ln1 = block.input_layernorm
            ln1_out = ln1(stream)  # (1, 1, D)
            attn_out, attn_rec = _attn_step_record(block.self_attn, ln1_out, t, caches[j])  # (1, 1, D)
            h = stream + attn_out  # (1, 1, D)
            ln2 = block.post_attention_layernorm
            ln2_out = ln2(h)  # (1, 1, D)
            mlp = block.mlp
            ff_out = mlp(ln2_out)  # (1, 1, D)
            out = h + ff_out  # (1, 1, D)
            block_rec: BlockRecord = {
                "ln1": _rmsnorm_record(ln1.weight, stream, ln1.eps, ln1_out),
                "attn": attn_rec,
                "h": arr(h),
                "ln2": _rmsnorm_record(ln2.weight, h, ln2.eps, ln2_out),
                "out": arr(out),
            }
            if isinstance(mlp, MixtureOfExperts):
                block_rec["moe"] = _moe_record(mlp, ln2_out, ff_out)
            else:
                block_rec["ffn"] = _ffn_record(mlp, ln2_out, ff_out)
            blocks.append(block_rec)
            stream = out

        normed = model.final_norm(stream)  # (1, 1, D)
        logits = model.lm_head(normed)  # (1, 1, V)
        z = logits - logits.max(dim=-1, keepdim=True).values
        p = torch.exp(z)
        p = p / p.sum(dim=-1, keepdim=True)  # (1, 1, V)
        p = p.detach()
        V = p.shape[-1]
        top_n = min(10, V)
        top_idx = torch.argsort(p[0, 0], descending=True)[:top_n]  # top-10 (single position)
        top_tokens = [[int(k), float(p[0, 0, k])] for k in top_idx.tolist()]

        rec: ForwardRecord = {
            "input_ids": x1.tolist(),
            "embedding": arr(emb),
            "positions": [t],
            "blocks": blocks,
            "final_norm": _rmsnorm_record(model.final_norm.weight, stream, model.final_norm.eps, normed),
            "logits": arr(logits),
            "softmax": arr(p),
            "top_tokens": top_tokens,
        }
        tok = _pick(p[0, 0].detach().cpu().numpy())
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
