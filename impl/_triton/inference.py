"""Triton-track inference — thin adapter over the shared deep generator.

The torch/triton/cuda generators were three shallow copies (identical
sampling math); they collapsed into ``shared.generator.TextGenerator`` —
one deep module behind one interface, consuming the KV-step interface
this track's ``TritonModel`` now exposes (``make_cache`` /
``forward_prefill`` / ``forward_step``, O(1) per token).

``TritonTextGenerator`` remains as an alias so the single-track smoke demo
(``impl._triton.cli``, which never imports across tracks per the
entry-point contract) keeps its per-track naming.
"""

from __future__ import annotations

from shared.generator import TextGenerator

TritonTextGenerator = TextGenerator
"""Alias: the shared deep generator, under this track's naming."""

__all__ = ["TextGenerator", "TritonTextGenerator"]
