"""Parameter registry — single owner of the flat-dict checkpoint format.

The registry knows, for one ``TransformerConfig``:

- every parameter's **key** (HF-Llama convention, from
  ``shared.constants.Keys``),
- its **shape** as a function of the config,
- its **layout rule** — PyTorch ``nn.Linear`` stores weights as
  ``(out, in)`` while every other track (NumPy/Triton/CUDA and the
  checkpoint format itself) uses ``(in, out)``; the affected keys are
  flagged ``torch_transpose`` and transposed on save/load only in the
  PyTorch track.

All save/load paths in all four tracks walk ``ParameterRegistry.entries``
instead of maintaining their own key lists, and load paths call
``validate()`` so a checkpoint whose keys or shapes do not match the
current registry (e.g. a pre-migration checkpoint) fails fast with a
clear message instead of producing a half-loaded model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from shared.config import TransformerConfig
from shared.constants import FFN_PROJS, Attn, Keys, LayerNorm, Mlp


@dataclass(frozen=True)
class ParamEntry:
    """One parameter of the flat-dict checkpoint format.

    Attributes:
        key: Registry key (HF-Llama convention, see ``shared.constants``).
        shape: Expected array shape in the checkpoint layout ``(in, out)``.
        torch_transpose: True when the PyTorch track stores this weight
            transposed (``nn.Linear`` keeps ``(out, in)``).
    """

    key: str
    shape: tuple[int, ...]
    torch_transpose: bool = False


class ParameterRegistry:
    """Every parameter of one config: key, shape, and layout rules.

    The registry is rebuilt per config (cheap: a list of dataclasses) and
    shared by the save/load adapters of all tracks plus the on-disk
    checkpoint reader.
    """

    def __init__(self, config: TransformerConfig) -> None:
        self.config = config
        self._entries = self._build_entries(config)
        self._key_set = frozenset(e.key for e in self._entries)

    @staticmethod
    def _build_entries(cfg: TransformerConfig) -> list[ParamEntry]:
        """Enumerate every parameter of ``cfg`` in a stable order."""
        D, V = cfg.embed_dim, cfg.vocab_size
        H, G, hd = cfg.n_heads, cfg.kv_heads, cfg.head_dim
        FF, E = cfg.expert_dim, cfg.n_experts

        def ffn_shape(proj: str) -> tuple[int, ...]:
            # SwiGLU: gate/up project D→FF, down projects FF→D.
            return (FF, D) if proj == Mlp.DOWN_PROJ else (D, FF)

        entries: list[ParamEntry] = [ParamEntry(Keys.embed(), (V, D))]
        for i in range(cfg.n_layers):
            entries.append(ParamEntry(Keys.ln(i, LayerNorm.INPUT), (D,)))
            entries.append(ParamEntry(Keys.ln(i, LayerNorm.POST_ATTENTION), (D,)))
            # Attention projections are nn.Linear in the PyTorch track → transposed there.
            entries.append(ParamEntry(Keys.attn(i, Attn.Q_PROJ), (D, H * hd), torch_transpose=True))
            entries.append(ParamEntry(Keys.attn(i, Attn.K_PROJ), (D, G * hd), torch_transpose=True))
            entries.append(ParamEntry(Keys.attn(i, Attn.V_PROJ), (D, G * hd), torch_transpose=True))
            entries.append(ParamEntry(Keys.attn(i, Attn.O_PROJ), (H * hd, D), torch_transpose=True))
            if cfg.has_moe():
                entries.append(ParamEntry(Keys.moe_gate(i), (D, E), torch_transpose=True))
                for j in range(E):
                    for proj in FFN_PROJS:
                        entries.append(ParamEntry(Keys.moe_expert(i, j, proj), ffn_shape(proj)))
                for s in range(cfg.n_shared_experts):
                    for proj in FFN_PROJS:
                        entries.append(ParamEntry(Keys.moe_shared_expert(i, s, proj), ffn_shape(proj)))
            else:
                for proj in FFN_PROJS:
                    entries.append(ParamEntry(Keys.ffn(i, proj), ffn_shape(proj)))
        entries.append(ParamEntry(Keys.final_norm(), (D,)))
        entries.append(ParamEntry(Keys.lm_head(), (D, V), torch_transpose=True))
        return entries

    @property
    def entries(self) -> list[ParamEntry]:
        """All parameters in a stable, documented order."""
        return self._entries

    def keys(self) -> list[str]:
        """Every registry key, in entry order."""
        return [e.key for e in self._entries]

    def expected_shapes(self) -> dict[str, tuple[int, ...]]:
        """Key → expected checkpoint shape (the format contract)."""
        return {e.key: e.shape for e in self._entries}

    def __contains__(self, key: object) -> bool:
        return key in self._key_set

    def bind(self, binding: Mapping[str, object]) -> dict[str, object]:
        """Materialize a track's storage binding as a full key→param dict.

        binding: registry key → owning array/tensor, as each track's
        ``_param_arrays``/``_param_tensors`` map builds it. Returns the
        full dict in entry order, failing fast when the binding misses or
        adds a key (the stale-walker bug class: a track dict that drifted
        from this registry fails here, not silently at save/load).

        This is the single traversal every save/load/grad path walks: the
        registry owns the key set; each track supplies only storage.
        """
        missing = self._key_set - frozenset(binding)
        extra = frozenset(binding) - self._key_set
        if missing or extra:
            raise ValueError(
                f"Storage binding does not match the registry"
                f" (missing: {sorted(missing)[:4]}, extra: {sorted(extra)[:4]})"
            )
        return {e.key: binding[e.key] for e in self._entries}

    def validate(self, params: Mapping[str, object]) -> None:
        """Fail fast when a checkpoint does not match this registry.

        Catches pre-migration checkpoints (stale/extra keys, missing keys)
        and shape drift (wrong config) with a single clear message.

        Raises:
            ValueError: If keys or shapes do not match the registry.
        """
        param_keys = set(params)
        missing = [e.key for e in self._entries if e.key not in param_keys]
        unexpected = sorted(param_keys - self._key_set)
        if missing or unexpected:
            parts = []
            if missing:
                parts.append(f"missing keys: {missing}")
            if unexpected:
                parts.append(f"unexpected keys (stale checkpoint?): {unexpected}")
            raise ValueError("Checkpoint does not match the parameter registry: " + "; ".join(parts))
        for e in self._entries:
            actual = tuple(params[e.key].shape)  # type: ignore[union-attr]
            if actual != e.shape:
                raise ValueError(f"Shape mismatch for '{e.key}': expected {e.shape}, got {actual}")
