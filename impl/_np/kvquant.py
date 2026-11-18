# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""KV-cache quantization schemes for the NumPy reference track.

Three production-grade lossy cache layouts replace the old 1-bit
TurboQuant experiment (removed 2026-11): the repo now teaches the schemes
serving stacks actually deploy in 2026 (vLLM/SGLang `--kv-cache-dtype`):

- ``"fp8"``  — FP8 e4m3 code + one max-abs scale per (batch, head, token).
  The industry default: 2x cache capacity at negligible accuracy loss.
- ``"int4"`` — asymmetric int4 (scale + zero-point) per (batch, head, token),
  with the first ``SINK_TOKENS`` positions kept full-precision ("attention
  sinks", KIVI-style). ~4x capacity.
- ``"int2"`` — asymmetric int2, per (batch, head, token, channel-group of
  ``GROUP``), sinks likewise full-precision (KIVI-2bit / Kitty-style).
  ~8x capacity, measurable degradation on long contexts.

Cache entry (per layer, per full-attention mixer, replacing ``{"k", "v"}``):

    {"q_k": entry, "q_v": entry, "quant": scheme}

where ``entry`` is a dict of int codes + float scales whose exact keys
depend on the scheme (see ``empty_entry``). Quantized K/V are stored PER
HEAD (B, H, t, hd) — after the GQA repeat — because the scale grids below
are per-head; the naive cache stays per-group (B, G, t, hd).

All schemes dequantize the FULL cache before attention runs (the teaching
path; PROD: fused dequant inside the attention kernel — vLLM/SGLang do
exactly this for int8/int4/fp8 caches).

Selection: ``kv_quant`` is a string threaded from the CLI
(``scripts/learning.py --kv-quant``) through ``make_cache`` /
``forward_prefill`` / ``forward_step``; ``"none"`` (the default) is the
exact full-precision cache. The speculative-decoding verification path
(``forward_chunk``) always uses the exact cache — the lossless contract
needs full precision.
"""

from __future__ import annotations

import numpy as np

KV_QUANT_SCHEMES = ("none", "fp8", "int4", "int2")
"""Valid values of the ``kv_quant`` flag (``"none"`` = exact cache)."""

SINK_TOKENS = 4
"""First positions kept full-precision in the int4/int2 caches.

Attention sinks (Xi et al. 2023, KIVI): the first few tokens absorb a
disproportionate share of attention mass; quantizing them hurts far more
than quantizing the tail. Stored raw alongside the codes, applied by
absolute position.
"""

GROUP = 16
"""Channel-group size for the int2 scheme (per-group scale + zero-point)."""

# FP8 e4m3 constants (IEEE-style 1-4-3 float: 1 sign, 4 exp, 3 mantissa).
_E4M3_MAX = 448.0  # largest finite e4m3 magnitude
_E4M3_MIN_SUB = 2.0**-9  # smallest subnormal (exp=0, mantissa=1/8): 2^-6 * 2^-3


def _validate(kv_quant: str) -> str:
    if kv_quant not in KV_QUANT_SCHEMES:
        raise ValueError(f"kv_quant must be one of {KV_QUANT_SCHEMES}, got {kv_quant!r}")
    return kv_quant


def empty_entry(kv_quant: str, B: int, H: int, hd: int, dtype: np.dtype | type) -> dict:
    """Allocate an empty quantized cache entry (t = 0) for one mixer.

    kv_quant: the scheme (validated). B, H, hd: cache axes. dtype: the
    float dtype of the model (scales are stored in it).

    Returns a dict whose CODE arrays all have t = 0 on axis 2; the raw
    sink pool is pre-sized to SINK_TOKENS rows (it fills by absolute
    position, not by append order, so it cannot grow via concat).
    """
    _validate(kv_quant)
    if kv_quant == "fp8":
        return {
            "codes": np.zeros((B, H, 0, hd), dtype=np.uint8),
            "scales": np.zeros((B, H, 0, 1), dtype=dtype),
            "raw": np.zeros((B, H, SINK_TOKENS, hd), dtype=dtype),
        }
    if kv_quant == "int4":
        return {
            "codes": np.zeros((B, H, 0, hd), dtype=np.int8),  # 0..15
            "scales": np.zeros((B, H, 0, 1), dtype=dtype),
            "zeros": np.zeros((B, H, 0, 1), dtype=dtype),
            "raw": np.zeros((B, H, SINK_TOKENS, hd), dtype=dtype),
        }
    # int2: per-group scales/zeros over the head-dim axis
    ng = -(-hd // GROUP)
    return {
        "codes": np.zeros((B, H, 0, hd), dtype=np.int8),  # 0..3
        "scales": np.zeros((B, H, 0, ng), dtype=dtype),
        "zeros": np.zeros((B, H, 0, ng), dtype=dtype),
        "raw": np.zeros((B, H, SINK_TOKENS, hd), dtype=dtype),
    }


def entry_keys(kv_quant: str) -> list[str]:
    """The array keys a quantized entry of this scheme carries (besides 'raw')."""
    return {"fp8": ["codes", "scales"], "int4": ["codes", "scales", "zeros"], "int2": ["codes", "scales", "zeros"]}[
        kv_quant
    ]


# ---------------------------------------------------------------------------
# FP8 e4m3 encode/decode (hand-rolled for teaching; PROD: torch.float8_e4m3fn
# or a fused kernel — the bit layout below IS the OCP FP8 e4m3 layout).
# ---------------------------------------------------------------------------


def fp8_e4m3_encode(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize x to FP8 e4m3 codes + a per-vector max-abs scale.

    x: (..., hd) — the scale is computed over the LAST axis (one scale per
    token vector). Returns (codes uint8, scales (..., 1) float64).
    """
    amax = np.max(np.abs(x), axis=-1, keepdims=True)  # (..., 1)
    scale = np.where(amax > 0, amax / _E4M3_MAX, 1.0).astype(np.float64)  # (..., 1)
    y = x.astype(np.float64) / scale  # (..., hd), |y| <= 448
    # Round-to-nearest-even on the e4m3 grid: clamp to the finite range,
    # then quantize mantissa to 3 bits and exponent to 4 bits (bias 7).
    y = np.clip(y, -_E4M3_MAX, _E4M3_MAX)
    sign = y < 0
    ay = np.abs(y)
    # Normal range: exp field 1..15 represents 2^(e-7) with e in [-6, 8];
    # subnormal (exp field 0) covers [0, 2^-6) in steps of 2^-9.
    e = np.floor(np.log2(np.maximum(ay, _E4M3_MIN_SUB))).astype(np.int32)  # (..., hd)
    e = np.clip(e, -6, 8)
    m = ay / np.power(2.0, e)  # in [1, 2) for normals
    frac = m - 1.0  # in [0, 1)
    # 3 mantissa bits, round-half-away is fine here because the scale makes
    # the max value land exactly on 448 = grid max (no overflow past it).
    q = np.round(frac * 8.0)
    carry = q >= 8.0
    q = np.where(carry, 0.0, q)
    e = np.clip(e + carry.astype(np.int32), -6, 15)
    # Subnormals: ay < 2^-6 → exp field 0, value = mant/8 * 2^-6 (step 2^-9).
    sub = e <= -6
    q_sub = np.round(ay / _E4M3_MIN_SUB)  # subnormal grid step is 2^-9
    # Rebuild e4m3 code: sign | exp(4) | mantissa(3)
    exp_field = np.where(sub, 0, np.clip(e + 7, 0, 15)).astype(np.uint8)
    mant = np.where(sub, np.clip(q_sub, 0, 7), q).astype(np.uint8)
    code = (exp_field << 3) | mant
    code = np.where(sign, code | 0x80, code).astype(np.uint8)
    return code, scale


def fp8_e4m3_decode(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Decode e4m3 codes * per-vector scale back to float.

    codes: (..., hd) uint8; scales: (..., 1). Returns (..., hd) float64.
    """
    sign = (codes & 0x80) != 0
    exp_field = ((codes >> 3) & 0x0F).astype(np.int32)
    mant = (codes & 0x07).astype(np.float64) / 8.0
    # subnormal: exp_field == 0 → value = 2^-6 * mant; normal: 2^(e-7) * (1+mant)
    val = np.where(exp_field > 0, np.power(2.0, (exp_field - 7).astype(np.float64)) * (1.0 + mant), 2.0**-6 * mant)
    val = np.where(sign, -val, val)
    return val * scales.astype(np.float64)  # broadcast (..., 1) over hd


# ---------------------------------------------------------------------------
# Quantize / dequantize a whole block or one token (shared by prefill + step)
# ---------------------------------------------------------------------------


def quantize_block(kv_quant: str, x: np.ndarray, sink_from: int = 0) -> dict:
    """Quantize a block of K or V rows: x (B, H, S, hd) → entry dict (t = S).

    sink_from: absolute position of x's first row; rows with absolute
    position < SINK_TOKENS go to the raw pool instead of being quantized
    (fp8/int4/int2 all keep sinks exact).
    """
    _validate(kv_quant)
    B, H, S, hd = x.shape
    x = x.astype(np.float64)
    entry = empty_entry(kv_quant, B, H, hd, np.float64)
    if kv_quant == "fp8":
        codes, scales = fp8_e4m3_encode(x)  # (B,H,S,hd), (B,H,S,1)
        entry["codes"], entry["scales"] = codes, scales
    elif kv_quant == "int4":
        amax = np.max(np.abs(x), axis=-1, keepdims=True)  # (B,H,S,1)
        scale = np.where(amax > 0, amax / 7.5, 1.0)  # 15 levels, |x| <= 7.5*scale
        q = np.clip(np.round(x / scale) + 8.0, 0.0, 15.0)  # zero-point 8: value = (code-8)*scale
        entry["codes"] = q.astype(np.int8)
        entry["scales"] = scale
        entry["zeros"] = np.full((B, H, S, 1), 8.0)
    else:  # int2, per channel-group over the head-dim axis
        ng = -(-hd // GROUP)
        # Split the head-dim axis into groups of GROUP; the tail group may
        # be shorter — slice per group instead of zero-padding (padding
        # would corrupt the group min/max).
        codes = np.zeros((B, H, S, hd), dtype=np.int8)
        scales = np.zeros((B, H, S, ng))
        zeros = np.zeros((B, H, S, ng))
        for g in range(ng):
            sl = slice(g * GROUP, min((g + 1) * GROUP, hd))
            xg = x[..., sl]  # (B,H,S,glen)
            amin = xg.min(axis=-1)  # (B,H,S)
            amax = xg.max(axis=-1)
            scale = np.where(amax > amin, (amax - amin) / 3.0, 1.0)  # 4 levels
            zero = amin  # value at code 0
            codes[..., sl] = np.clip(np.round((xg - zero[..., None]) / scale[..., None]), 0.0, 3.0).astype(np.int8)
            scales[..., g] = scale
            zeros[..., g] = zero
        entry["codes"] = codes
        entry["scales"] = scales
        entry["zeros"] = zeros
    # Sinks: rows whose ABSOLUTE position < SINK_TOKENS go to the raw pool
    # (indexed by absolute position, not by order within this block).
    for j in range(S):
        if sink_from + j < SINK_TOKENS:
            entry["raw"][:, :, sink_from + j] = x[:, :, j]
            entry["codes"][:, :, j] = 0
            entry["scales"][:, :, j] = 0.0
            if kv_quant != "fp8":
                entry["zeros"][:, :, j] = 0.0
    return entry


def append_token(kv_quant: str, entry: dict, x: np.ndarray, position: int) -> None:
    """Quantize ONE new token's K/V row and append it to the entry (in place).

    x: (B, H, 1, hd); position: the row's absolute index (sink check).
    """
    block = quantize_block(kv_quant, x, sink_from=position)
    for key, arr in block.items():
        if key == "raw":
            if position < SINK_TOKENS:
                entry["raw"][:, :, position] = arr[:, :, position]  # sink slot by absolute position
        else:
            entry[key] = np.concatenate([entry[key], arr], axis=2)


def dequantize_block(kv_quant: str, entry: dict, sink_from: int = 0) -> np.ndarray:
    """Dequantize the whole entry back to float: (B, H, t, hd).

    Rows whose absolute position < SINK_TOKENS come from the raw pool;
    the rest from codes/scales(/zeros). sink_from: absolute position of
    the entry's first row (0 for a cache built from position 0).
    """
    _validate(kv_quant)
    codes = entry["codes"]
    B, H, t, hd = codes.shape
    if kv_quant == "fp8":
        out = fp8_e4m3_decode(codes, entry["scales"])  # (B,H,t,hd)
    elif kv_quant == "int4":
        # asymmetric int4 with fixed zero-point 8: value = (code - 8) * scale
        out = (codes.astype(np.float64) - entry["zeros"]) * entry["scales"]
    else:  # int2
        # per-group: value = code * scale + zero (zero = group min); expand
        # each group's scale/zero over its ACTUAL channels (tail group may
        # be shorter than GROUP when hd % GROUP != 0)
        ng = entry["scales"].shape[-1]
        sc = np.zeros((B, H, t, hd))
        zc = np.zeros((B, H, t, hd))
        for g in range(ng):
            sl = slice(g * GROUP, min((g + 1) * GROUP, hd))
            sc[..., sl] = entry["scales"][..., g : g + 1]
            zc[..., sl] = entry["zeros"][..., g : g + 1]
        out = codes.astype(np.float64) * sc + zc
    n_sink = max(0, min(t, SINK_TOKENS - sink_from))
    if n_sink > 0:
        out[:, :, :n_sink] = entry["raw"][:, :, sink_from : sink_from + n_sink]
    return out
