"""Drafter protocol + sidecar checkpoints for speculative decoding.

A **drafter** is the small, fast model that proposes candidate tokens for
the target model to verify (see CONTEXT.md → Inference / Speculative
decoding, and ADR 0003). This module owns:

1. The ``Drafter`` protocol — the ONE new seam of the speculative-decoding
   feature. Every drafter family (MTP, DSpark, a future DFlash) implements
   it; the speculative engines (NumPy reference + the torch-family
   production engine) consume it without knowing the family.
2. The **sidecar checkpoint format** — draft weights deliberately live
   OUTSIDE the target's ``model.npz`` (the main ``ParameterRegistry`` key
   set is untouched, so every existing checkpoint stays valid). A sidecar
   is its own directory beside the target checkpoint:

       <model_dir>/draft_mtp/    draft.json + draft.npz
       <model_dir>/draft_dspark/ draft.json + draft.npz

   ``draft.json`` carries the drafter's own hyperparameters (the meta
   below); ``draft.npz`` is a flat dict of weights under this scheme:

       mtp:    mtp.norm.weight                (D,)      RMSNorm over the target hidden
               mtp.in_proj.weight              (2D, D)   concat(norm(h), emb(prev)) → D
               mtp.block.<block keys>                    ONE transformer block (own KV cache)
               mtp.out_norm.weight             (D,)
       dspark: dspark.norm.weight              (D,)
               dspark.in_proj.weight           (2D, D)   concat(norm(h), pos_emb[j]) → D
               dspark.pos_emb.weight           (k_max, D)  learned block-position embeddings
               dspark.parallel.<block keys>              parallel backbone (non-causal over the block)
               dspark.seq.<block keys>                    sequential module (causal over the block)
               dspark.out_norm.weight          (D,)

   where ``<block keys>`` mirror the main scheme's per-layer names
   (``input_layernorm.weight``, ``self_attn.{q,k,v,o}_proj.weight``,
   ``post_attention_layernorm.weight``, ``mlp.{gate,up,down}_proj.weight``).

   Layout rule (the same one the main registry uses): keys marked
   ``torch_transpose`` are stored ``(in, out)`` in the npz and transposed
   on load into an ``nn.Linear``-backed torch module; everything else is
   stored exactly as used.

The drafter SHARES the target's embedding matrix and lm_head (referenced
at bind time, never copied into the sidecar) — the DeepSeek-V3 MTP design:
no duplicated V×D weights.

Round-trip guarantee (ADR 0003): a sidecar saved by any track loads and
drafts on all four.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

MTP = "mtp"
DSHARK = "dspark"
FAMILIES = (MTP, DSHARK)


@dataclass(frozen=True)
class DrafterMeta:
    """A drafter sidecar's hyperparameters (``draft.json``)."""

    family: str  # "mtp" | "dspark"
    block_size: int  # trained max draft length k
    embed_dim: int  # D — must match the target
    n_heads: int  # H of the drafter's transformer block(s)
    n_groups: int | None  # G (None → MHA), like the main config
    ff_dim: int  # FF of the drafter's SwiGLU
    rope_dim: int  # RoPE prefix length (0 = full head dim), like the main config
    vocab_size: int  # V — must match the target (shared lm_head)

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown drafter family {self.family!r} (expected one of {FAMILIES})")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "block_size": self.block_size,
            "embed_dim": self.embed_dim,
            "n_heads": self.n_heads,
            "n_groups": self.n_groups,
            "ff_dim": self.ff_dim,
            "rope_dim": self.rope_dim,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DrafterMeta:
        """Deserialize draft.json's dict (missing rope_dim → 0; unknown
        families raise in __post_init__)."""
        return cls(
            family=str(data["family"]),
            block_size=int(data["block_size"]),
            embed_dim=int(data["embed_dim"]),
            n_heads=int(data["n_heads"]),
            n_groups=None if data.get("n_groups") is None else int(data["n_groups"]),
            ff_dim=int(data["ff_dim"]),
            rope_dim=int(data.get("rope_dim", 0)),
            vocab_size=int(data["vocab_size"]),
        )


def _block_param_keys(prefix: str, D: int, H: int, G: int, hd: int, FF: int) -> list[tuple[str, tuple[int, ...], bool]]:
    """(key, npz shape, torch_transpose) for ONE drafter transformer block."""
    return [
        (f"{prefix}.input_layernorm.weight", (D,), False),
        (f"{prefix}.self_attn.q_proj.weight", (D, H * hd), True),
        (f"{prefix}.self_attn.k_proj.weight", (D, G * hd), True),
        (f"{prefix}.self_attn.v_proj.weight", (D, G * hd), True),
        (f"{prefix}.self_attn.o_proj.weight", (H * hd, D), True),
        (f"{prefix}.post_attention_layernorm.weight", (D,), False),
        (f"{prefix}.mlp.gate_proj.weight", (D, FF), False),
        (f"{prefix}.mlp.up_proj.weight", (D, FF), False),
        (f"{prefix}.mlp.down_proj.weight", (FF, D), False),
    ]


def expected_drafter_params(meta: DrafterMeta) -> list[tuple[str, tuple[int, ...], bool]]:
    """Every sidecar key for a drafter, in stable order, with its npz shape
    and torch_transpose flag — the single owner of the sidecar format.

    meta: the drafter's hyperparameters (the key set derives from it:
    MTP = one block; DSpark = pos_emb + parallel + seq blocks)."""
    D, H = meta.embed_dim, meta.n_heads
    G = meta.n_groups if meta.n_groups is not None else meta.n_heads
    if D % H != 0:
        raise ValueError(f"embed_dim {D} not divisible by n_heads {H}")
    if H % G != 0:
        raise ValueError(f"n_heads {H} not divisible by n_groups {G}")
    hd = D // H
    keys: list[tuple[str, tuple[int, ...], bool]] = [
        (f"{meta.family}.norm.weight", (D,), False),
        (f"{meta.family}.in_proj.weight", (2 * D, D), True),
    ]
    if meta.family == MTP:
        keys += _block_param_keys(f"{meta.family}.block", D, H, G, hd, meta.ff_dim)
    else:  # dspark
        keys.append((f"{meta.family}.pos_emb.weight", (meta.block_size, D), False))
        keys += _block_param_keys(f"{meta.family}.parallel", D, H, G, hd, meta.ff_dim)
        keys += _block_param_keys(f"{meta.family}.seq", D, H, G, hd, meta.ff_dim)
    keys.append((f"{meta.family}.out_norm.weight", (D,), False))
    return keys


@runtime_checkable
class Drafter(Protocol):
    """The drafter protocol — the one new seam of speculative decoding.

    Shape letters (CONTEXT.md → "Shape notation"): B = batch size, D =
    embed_dim (model width), V = vocab_size, k = the drafter's block size.

    Implementations: NumPy (``impl/_np/drafters.py``), torch family
    (``impl/_torch/drafters.py``). Arrays are ``np.ndarray`` for the NumPy
    track and ``torch.Tensor`` for the torch family — same shapes either way.

    Contract:
        family       — "mtp" | "dspark" (a future plug-in adds its own).
        block_size   — trained max draft length k (the caller may request less).
        draft(anchor_token, anchor_hidden, k)
                       — propose up to k tokens after ``anchor_token``
                         (the last verified token id, (B,) ints), conditioned
                         on ``anchor_hidden`` (the target's final-norm hidden
                         state at the anchor position, (B, D)). Returns
                         (tokens (B, k'), probs: k' distributions (B, V)) —
                         the drafter's own next-token distribution per drafted
                         position (needed by the rejection-sampling rule).
        rollback(keep) — align internal state after a rejected verification:
                         keep the anchor + the first ``keep`` accepted drafts
                         of the LAST call, drop the rest (KV-cache alignment;
                         MTP keeps 1+keep rows). Stateless drafters no-op.
        reset()        — clear all internal state for a fresh sequence.
    """

    family: str
    block_size: int

    def draft(self, anchor_token: Any, anchor_hidden: Any, k: int) -> tuple[Any, list[Any]]: ...

    def rollback(self, keep: int) -> None: ...

    def reset(self) -> None: ...


def sidecar_dir(model_dir: str | Path, family: str) -> Path:
    """The conventional sidecar directory for a family under a target checkpoint.

    model_dir: the target checkpoint's directory; family: "mtp" |
    "dspark" — returns <model_dir>/draft_<family>/ (draft.json + draft.npz)."""
    return Path(model_dir) / f"draft_{family}"


def save_drafter(directory: str | Path, meta: DrafterMeta, params: dict[str, np.ndarray]) -> None:
    """Write a drafter sidecar: draft.json + draft.npz (validated first).

    directory: the sidecar directory (sidecar_dir's output); meta: the
    drafter's hyperparameters; params: the sidecar key → array map (npz
    layout — torch_transpose keys stored (in, out)). Raises ValueError on
    key-set or shape drift — never writes a malformed sidecar."""
    _validate_params(meta, params)
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "draft.json", "w") as f:
        json.dump(meta.to_dict(), f, indent=2)
    np.savez(str(path / "draft.npz"), **{k: np.asarray(v) for k, v in params.items()})  # pyright: ignore[reportArgumentType]


def load_drafter(directory: str | Path) -> tuple[DrafterMeta, dict[str, np.ndarray]]:
    """Load a drafter sidecar; fail fast on key-set or shape drift.

    directory: the sidecar directory. Returns (meta, params) — the
    hyperparameters and the validated npz-layout arrays (a torch-track
    loader applies the torch_transpose rule on top)."""
    path = Path(directory)
    with open(path / "draft.json") as f:
        meta = DrafterMeta.from_dict(json.load(f))
    with np.load(str(path / "draft.npz")) as f:
        params = {k: f[k] for k in f.files}
    _validate_params(meta, params)
    return meta, params


def _validate_params(meta: DrafterMeta, params: dict[str, np.ndarray]) -> None:
    """Exact key-set + shape validation (mirrors ParameterRegistry.validate)."""
    expected = expected_drafter_params(meta)
    expected_keys = {k for k, _shape, _t in expected}
    got_keys = set(params)
    missing = sorted(expected_keys - got_keys)
    unexpected = sorted(got_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "Drafter sidecar does not match the expected key set"
            f" (missing: {missing[:4]}, unexpected: {unexpected[:4]})"
        )
    for key, shape, _t in expected:
        got = tuple(np.asarray(params[key]).shape)
        if got != shape:
            raise ValueError(f"drafter param {key!r} has shape {got}, expected {shape}")
