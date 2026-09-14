# Spec: Architecture Fixes — Standardize, Teach, Interchange

**Status:** complete (2026-09-14 — all phases done: Architecture, Interchange, Backprop, KV cache, Causal mask, PyTorch Idioms, Flash Kernel, Docs & Naming; see Status & Progress)
**Date:** 2026-09-12
**Source:** architecture review (`architecture-review-20260912-213529.html`) + user decisions
**Tracks affected:** NumPy (math reference), PyTorch (production reference), Triton (kernel reference), CUDA (bare-metal reference)

## Status & Progress (2026-09-14)

### Completed

- **Platform:** cu132 torch installed, index pins moved off the JetPack 6 / cu126 mirror, GPU smoke green (Orin sm_87, driver 595.78, CUDA 13.2).
- **Architecture (all four tracks, identical math):** pre-norm LLaMA block `h = x + attn(ln1(x)); out = h + mlp(ln2(h))` in NumPy/PyTorch/Triton/CUDA. RoPE on by default with one semantic (`rope_dim`: 0 = rotate the full head dimension, else the prefix length; validated even and ≤ head_dim). GQA wired end-to-end via `n_groups` (`None` → `n_heads`, resolved in `__post_init__`; `config.kv_heads` exposes the resolved int; divisibility validated). Dense SwiGLU feed-forward is the default; MoE is the opt-in (`n_experts > 1`, softmax-over-all + top-k mask + renormalize). Plain linear `lm_head` (D→V), no weight tying, no biases anywhere.
- **Keys scheme (`shared/constants.py`):** one HF-Llama-style checkpoint key scheme (`model.embed_tokens`, `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`, …, `model.norm.weight`, `model.lm_head.weight`) shared by all four tracks; `Keys` static builders + `all_param_keys()` generate the full set for a config (order aligned with the registry). Every track exposes `get_all_parameters()` / `load_from_numpy_dict()` (torch + triton also `save_as_numpy()`).
- **Interchange (checkpoint seam) — complete:**
  - `shared/registry.py`: `ParameterRegistry(config)` is the single owner of the checkpoint format — `entries` (stable order: embed → per-layer [ln1, ln2, q/k/v/o, MoE(gate + experts) | FFN(gate/up/down)] → final norm → lm_head), `expected_shapes()` derived from the config, and `validate()` which raises `ValueError` on missing keys, stale keys (message: "unexpected keys (stale checkpoint?)"), or shape mismatch. `ParamEntry.torch_transpose` encodes the one layout rule: only `nn.Linear`-backed weights (attn q/k/v/o, MoE router, lm_head) are transposed for the PyTorch track.
  - All four tracks are registry-driven with exactly one storage-binding map each (`_param_tensors()` torch/triton/cuda, `_param_arrays()` np). `load_from_numpy_dict` validates against the registry before assigning; `load_checkpoint` validates whenever `config.json` is present (skipped for config-less weight-only checkpoints, pinned by a test). Pre-migration checkpoints are rejected by design.
  - Fossils deleted: `model_config.py` stub (+ stale `.pyc`), `scripts/auto_test_equivalence.py` + its test, the `save_checkpoint` key-normalizer (`_param_key_for_npz`) and `_CHECKPOINT_META_KEYS`. `save_checkpoint(checkpoint_dir, config=None, params: dict)` now takes the parameter dict as a named argument.
  - `tests/unit/test_registry.py` (15 tests): enumeration (dense + MoE), key-set equality with `all_param_keys`, config-derived shapes (incl. GQA), the transpose rule, and validate() accept/reject paths.
- **Triton SDPA:** `_attn_fwd_kernel`/`_attn_bwd_kernel` verified against `F.scaled_dot_product_attention` (forward diff 1.8e-7, grad diff 0.0 at reference settings). Device handling: blocks/stack/model call `_move_to_device(x)` at forward entry.
- **Script migration:** `infer.py` / `train.py` / `verify_equivalence.py` all on the shared `TransformerConfig` + Keys loading; verify_equivalence exposes `Scenario`/`SCENARIOS` (6)/`_scenarios()`/`distribution_check()`/`format_report`, all 6 scenarios pass (incl. `all_four_backends` with fresh CUDA context).
- **Cross-backend suite:** `tests/cross_backend/` rewritten on the shared `TransformerConfig.from_dict` config (42 tests: dense/GQA/MoE parity, GPU parity incl. grad checks, CUDA parity, 3-way equivalence demo as a real pytest test).
- **Tests:** unit 488 (incl. 15 registry tests), cuda 125 (9 files, one subprocess per file via NVRTC context isolation), cross_backend 42. All green.
- **Pyright: 0 errors** — the triton 3.7.1 stub false-positives (`tl.constexpr`/`pointer_type` on kernel launch args) are silenced per-line with documented `# pyright: ignore` comments; C901 complexity in `verify_equivalence.py`/`train.py` fixed with small extracted helpers.
- **Real bug fixed:** NumPy `generate_sampled` produced a scalar `next_token` for batch 1 → `IndexError`; now wrapped in a 1-row array.

- **Backprop — complete:**
  - `impl/_np/modules.py` (448-line god module) deleted; split into per-operator modules under `impl/_np/`: `init.py` (xavier_uniform), `embedding.py`, `layernorm.py` (RMSNorm), `rope.py`, `ffn.py` (SwiGLUFFN + silu), `attention.py` (MultiHeadAttention, dense + GQA), `moe.py` (MixtureOfExperts), `block.py` (TransformerBlock), `stack.py` (DecoderStack), plus `cross_entropy.py` (backward) and `model.py` (analytic chain). `impl/_np/__init__.py` documents the layout.
  - Every operator has an analytic `backward(dout, x[, positions]) -> (dinput, dparams)` with shape comments and formula references; backwards recompute intermediates (stateless, no cached activations — documented tradeoff). `NumPyModel.backward` is now an O(forward) analytic chain (CE → lm_head → final norm → stack (reverse) → embedding): ~3.5 ms for a 2-layer V16/D8 model (was ~120 s with finite differences).
  - `impl/_np/gradcheck.py` demotes finite differences to a test reference: `finite_difference_gradient` + `check_model_gradients` (per-parameter worst relative error, `max(1,|a|,|n|)` denominator, automatic MoE top-k flip detection/skip — the loss is piecewise smooth and the numeric derivative is undefined at selection kinks).
  - `tests/unit/_np/test_gradient_check.py` (17 tests): every operator (RMSNorm, SwiGLU, RoPE full+partial, MHA dense+GQA, MoE top-k 1/2/3 flip-aware, Embedding, CrossEntropy shift×mask) + full-model checks (dense, MoE, GQA), all float64, all passing at ~1e-10.
  - **Real bugs found by the gradient checks:** (1) `TransformerBlock.backward` chained the residual wrong (fed `d_h` into both the ln2 and mlp upstreams; dropped the direct `x→h` path) — fixed to `d_h = dout + ln2.backward(mlp.backward(dout))[0]`, `dx = d_h + ln1.backward(...)[0]`; (2) `dW_lm` used the pre-final-norm activations instead of `h = final_norm(stack_out)`; (3) `forward_with_trace` stored the post-norm vector as `x_in0` (variable shadowing) — the stack backward was recomputing from the wrong input; (4) `CrossEntropyLoss.forward` crashed on `ignore_index` targets (out-of-range `take_along_axis`) — gathered with an in-range guard (backward had the same hazard); (5) `NumPyModel.load_from_numpy_dict` aliased the caller's arrays — in-place training mutated the shared checkpoint dict, which corrupted cross-backend comparisons in verify_equivalence; it now copies.
  - **verify_equivalence:** NumPy training re-enabled (was skipped because of the finite-difference cost; now ~4 ms/step). All 6 scenarios pass with `numpy_last_loss` reported alongside the autograd tracks.
  - **Tests:** unit 505 (488 + 17 gradient-check), cuda 99 (9 files × 11, per-file subprocess), cross_backend 42. ruff clean, pyright 0 errors.

- **Causal mask (all four tracks):** the pre-existing attention had *no* causal mask — every position attended over all positions, so the exact KV-cache step path (which is only exact for causal attention) diverged from a full re-forward at layer ≥ 1. Added the standard lower-triangular mask to attention in all four tracks (NumPy `attention.py` forward + docstring; PyTorch `layers.py`; Triton `attn.py` kernel via an `IS_CAUSAL` flag with `is_causal` plumbed through the autograd function and `transformer.py` call site; CUDA `attention.py` with `is_causal` plumbed through the autograd function and `block.py` call site). The backward needs no form change — the masked scores produce zero attention weights at masked positions, so the same softmax-backward math applies (verified by the gradient checker at ~1e-10). New causal tests: `test_causal_prefix_invariance` (np + triton), `test_causal_uniform_attention` (triton). Full green bar re-verified: unit 408 + cuda 99 + triton 98 + cross_backend 42, ruff clean, pyright 0, verify_equivalence 6/6.
- **KV Cache — complete:** the per-token step path is exact (`forward_prefill` + `forward_step` == full re-forward, 0.0 diff, all of dense/GQA/MoE). The generators (`generate_greedy`, `generate_sampled`) now thread the prompt through a fresh cache one token at a time and then step one token per iteration (O(1) per step instead of O(T) re-forward); token-identical to the old re-feed path. **TurboQuant** is wired as an alternative cache behind the same `forward_step(quantize=True)` interface: the new K/V are 1-bit quantized (sign + per-channel scale) per head, appended to the cache, and the full cached tensor is dequantized before attention — the step attends against the lossy cache. The parity-budget test (`tests/unit/_np/test_turboquant_parity.py`) asserts the TurboQuant output is degraded but bounded (max-abs logit diff ≤ 2.0 at the last prompt token; a broken cache would exceed it). **Eps alignment:** the CUDA track's `rmsnorm` now takes an `eps` parameter (default 1e-6) and the block/model call sites pass `config.norm_eps`, so all four tracks use the single `norm_eps` constant from the config.
- **PyTorch Idioms — complete:**
  - `MultiHeadAttention.forward` (PyTorch track) now calls `F.scaled_dot_product_attention(q, k, v, is_causal=True)` — the hand-rolled max-subtract softmax + `torch.triu` mask + `torch.where` is gone. The GQA `repeat_interleave` stays before the SDPA call: this PyTorch build (2.13.0+cu132) does not broadcast K/V heads to query heads (raises `RuntimeError` on H≠G), so K/V are expanded to H heads first.
  - The manual `AdamW` class was deleted from `impl/_torch/layers.py` (+ removed from `__init__`/`__all__`, test `test_adamw.py` deleted). `scripts/train.py` already used `torch.optim.AdamW` for the torch/triton/cuda backends; the NumPy track's hand-rolled `AdamW` (the teaching artifact) is unchanged.
  - **Shared seam documented:** `docs/seam_triton_to_torch.md` records the one-way Triton→PyTorch dependency (the only cross-track import: `RoPE` in `impl/_triton/transformer.py`, marked `# shared-seam:`). The Triton track is a "PyTorch model with a Triton attention core"; the seam is a documented one-way import, not a shared module.
- **Flash Kernel — complete:**
  - `impl/_triton/flash_attn.py`: `flash_attention(q, k, v, is_causal)` public function + `_flash_attn_fwd_kernel` Triton kernel. Single-pass online softmax (running max/sum/accumulator per key block) — no materialized (BLOCK_M, Sk) score matrix. BLOCK_M = BLOCK_N = 16 (fits the Orin sm_87 shared-memory limit). Causal mask uses the **absolute** key index: `key_idx + n <= q_idx`. `scale` is a keyword argument at launch (a positional `scale` misaligns onto the `D_pad` kwarg).
  - `tests/unit/_triton/test_flash_attn.py` (5 tests): forward parity against both the two-pass Triton kernel and the framework `F.scaled_dot_product_attention` (causal on/off, diff < 1e-4) + a long-sequence memory note (B=1, H=1, Sq=256, Sk=1024, D=16: flash peak ≤ 2× two-pass peak, verified).
- **Docs & Naming — complete:**
  - `CONTEXT.md` — domain glossary (architecture, attention, norms, training, inference, tracks, cross-cutting).
  - Naming convergence: CUDA `final_ln_gamma` → `final_norm_gamma` (model, training, `scripts/train.py` ×3, test) to match the other tracks' `final_norm`. MoE class names verified consistent (NumPy/PyTorch `MixtureOfExperts`; Triton `TritonMixtureOfExperts`/`TritonExpert`) — no change needed.
  - `docs/docstring_style.md` — the one docstring style (numpydoc sections + the shape-comment convention `# (B,S,D) @ (D,H·hd) → (B,S,H·hd)`; required in the NumPy track, encouraged elsewhere).
  - `docs/adr/0001-gated-residual-abandonment.md` — records the decision to drop the non-standard gated residual (scalar × stream) for the standard additive residual + pre-norm (matches LLaMA; the old form had no published equivalent).
  - `README.md` re-aligned to the four-backend reality (structure, CLI, test commands, links to CONTEXT.md / spec / seam / docstring docs).
- **Final green bar (2026-09-14):** unit np+torch 406, triton 103 (98 + 5 flash), cuda 99 (9 files × 11, per-file subprocess), cross_backend 42, ruff clean, pyright 0 errors (1 pre-existing warning `impl/_triton/attn.py:267`), verify_equivalence 6/6 (numpy_last_loss 4.4593, torch/triton 4.2277, greedy match ✓ all three ways).

### Known debt

- Torch prints a harmless cuDNN warning on this GPU ("No published PyTorch CUDA builds for release 2.13.0+cu132 support this GPU (CC 8.7)") — ignored.
- `save_checkpoint` skips registry validation when `config is None` (weights-only checkpoints have no config to derive keys from) — documented and pinned by `test_save_checkpoint_without_config`.

## Problem Statement

The project's stated goal is pedagogical: the NumPy track should teach how the math of a
decoder-only transformer works (shapes at every step, references for every formula), the
PyTorch/Triton/CUDA tracks should show how to implement the *same* model properly with each
tech stack, all tracks must be interchangeable (one trained model runs on any track; same
data → functionally identical models), and all naming must follow industry convention so
every concept is web-searchable.

The current code has drifted away from that goal on every axis:

1. **The reference architecture is non-standard.** The decoder block is post-norm with
   always-on MoE, a "gated residual" that is actually a scalar multiplier of the whole
   stream, and a SwiGLU+Linear output head. The docstring in the NumPy track even describes
   the *wrong* wiring (pre-norm) for code that is post-norm. A learner Googling any of these
   names finds nothing, because nothing by that name exists outside this repo.
2. **The math reference cannot teach the math.** `NumPyModel.backward` is per-element
   finite-difference numerical differentiation — O(parameter count) full forward passes per
   step. There is no analytic chain rule anywhere in the NumPy track, which is exactly the
   math the project exists to teach.
3. **The headline features are dead knobs.** Grouped-Query Attention (`n_groups`) exists in
   the config, the CLI flag, and the MHA module — and is hardwired to full MHA before it can
   ever reach attention. RoPE is *off by default* and its `rope_dim` semantics contradict
   between config (0 = full) and the model code (0 = disabled).
4. **The interchange contract is fragmented.** The "any track can load any track's model"
   guarantee lives in ~6 hand-written save/load traversals with per-key transpose flags;
   CUDA can save but not load through the format; fossil modules (a key normalizer mapping
   names nothing uses, a stub config "to pass an import test") remain in the tree.
5. **The KV cache is unwired.** Three cache modules exist with tests, but the attention
   module's cache argument is dead code (a design monologue is shipped in the source), and
   the generators re-feed the full sequence at every step — O(T²) generation.
6. **The "production" track is not idiomatic.** The PyTorch track hand-rolls scaled
   dot-product attention (materializing the full score matrix) where the framework's
   SDPA dispatch exists, and hand-rolls AdamW where `torch.optim` exists. The Triton track
   imports the PyTorch track's building blocks, so "four stacks" is actually one stack plus
   one kernel; its headline attention kernel is a tiled matmul that cites FlashAttention
   without implementing the online-softmax algorithm.
7. **Docs describe a different project.** The README describes a two-backend project with a
   nonexistent layers file; the design doc names a class and a learning-rate scheduler that
   do not exist; naming diverges per track (MoE vs MixtureOfExperts; router vs
   routing_weights); docstring style drifts per track.
8. **Dependencies are rotting on the old platform.** The project was built for JetPack 6.2.2
   (CUDA 12.6); the host is now JetPack 7.2.1 (CUDA 13.2), and the dependency pins still
   point at the JetPack 6 / cu126 mirror, so the GPU tracks cannot even be imported in the
   current environment.

## Solution

Rebuild the project around the canonical industry-standard decoder-only transformer, in
three layers of change:

1. **One standard architecture, mirrored across all four tracks.** A pre-norm decoder block
   (RMSNorm → attention → residual; RMSNorm → feed-forward → residual), RoPE on by default,
   Grouped-Query Attention wired end-to-end from config to the attention module, dense
   SwiGLU feed-forward as the default with MoE as the documented opt-in, and a plain linear
   language-model head. Every parameter name becomes a name a learner can search on the web.
2. **The NumPy track becomes a genuine math reference.** Every operator gets an analytic
   backward pass with shape-annotated equations and literature references, split out of the
   god module into per-operator modules. The finite-difference method is demoted to a
   gradient-checking utility that *verifies* the analytic math — the standard role.
3. **One interchange mechanism.** A single parameter-registry module behind the existing
   flat-dict checkpoint format owns the key scheme, shapes, and per-track layout rules; all
   save/load paths derive from it, and CUDA gets a real load path. The KV cache is wired
   into inference as the real per-token step path (it is a key piece of the transformer and
   stays). The PyTorch track adopts framework idioms (SDPA dispatch, `torch.optim`), and the
   Triton track earns its name with a true online-softmax attention kernel verified at the
   existing parity seam. Docs, naming, and the domain glossary converge; dependencies are
   upgraded to the JetPack 7.2.1 / CUDA 13.2 toolchain.

The cross-backend equivalence contract is unchanged in *meaning* and is the acceptance
criterion throughout: a model trained on any track loads on any other and produces
near-identical outputs; models trained on the same data with the same seed are
functionally identical within the established tolerance tiers.

## User Stories

1. As a learner with basic linear algebra, I want the model to be the standard LLaMA-style
   decoder (pre-norm block, RMSNorm, RoPE, SwiGLU), so that every concept I encounter in
   this repo matches the papers and web articles I find.
2. As a learner, I want every step of the NumPy forward pass annotated with the tensor shape
   before and after (e.g. `(B, S, D) → (B, H, S, d)`), so that I can trace exactly how data
   moves through the math.
3. As a learner, I want each NumPy operator to have an analytic backward pass with the chain
   rule written out and referenced, so that I learn how backpropagation actually works
   instead of watching finite differences perturb elements.
4. As a learner, I want the finite-difference method kept as a gradient checker, so that I
   can see the analytic gradients verified against numerical ones and trust the math.
5. As a learner, I want an intro-level explanation of each transformer concept (attention,
   RoPE, GQA, MoE, KV cache) assuming only basic linear algebra, so that the repo is
   self-contained for someone new to transformers.
6. As a learner, I want RoPE enabled by default with one consistent semantic for the
   partial-rotation knob, so that the positional-encoding I am taught is the one real
   models use and it actually runs.
7. As a learner, I want Grouped-Query Attention to be a real, reachable feature (config →
   model → attention), so that I can set fewer K/V heads than query heads and see the
   K/V-sharing broadcast happen.
8. As a learner, I want MoE to be an explicit opt-in on top of the standard dense feed-forward
   default, so that the default model matches what I will find in the literature and MoE is
   the documented extension.
9. As a learner, I want the language-model head to be a plain linear layer named after the
   industry convention, so that "lm_head" in this repo means the same thing as in every
   framework I will ever use.
10. As a learner, I want the "gated residual" decoration removed (or made a real gate), so
    that I am not taught a concept whose implementation does what its name does not say.
11. As a learner, I want the KV cache wired into the generation loop (step on one new token,
    append to the cache, attend over the cached keys/values), so that I see the canonical
    autoregressive inference optimization and generation stops re-reading the whole sequence.
12. As a learner, I want the quantized KV cache kept as a documented experiment with an
    honest parity budget, so that I can see how memory compression trades off against
    output quality.
13. As a learner, I want the PyTorch track to use framework idioms (framework SDPA,
    `torch.optim`), so that the track teaches "how to do it properly in PyTorch" rather than
    a parallel hand-rolled implementation.
14. As a learner, I want the Triton track's attention to be a true online-softmax
    (FlashAttention-style) kernel, so that the track's headline kernel matches its name and
    the memory/throughput story is real.
15. As a learner, I want the CUDA track's bare-metal pipeline (nvrtc compile → PTX → module
    → kernel launch) to keep teaching reductions, coalesced access, and grid/block
    configuration, so that the perf reference for the CUDA track is preserved.
16. As a learner, I want every module name and parameter name to follow industry convention
    consistently across all four tracks, so that a name found in one track resolves to the
    same concept (and the same web results) in every track.
17. As a learner, I want one domain glossary that all docs and code speak from, so that
    terms like block, GQA, RoPE, MoE, KV cache, and checkpoint mean one thing.
18. As a learner, I want the docs to describe the project that actually exists, so that the
    README, design doc, and plan do not contradict the code.
19. As a learner, I want one docstring style with the shape-annotation convention, so that
    every module reads the same way and the math is always visible.
20. As a developer, I want one parameter-registry module that owns the checkpoint key
    scheme, so that adding a parameter is a one-line change instead of six hand-rolled
    edits across the tracks.
21. As a developer, I want CUDA to load checkpoints through the same format as the other
    tracks, so that the four-way interchange story is actually four-way.
22. As a developer, I want the save/load traversals to derive from the registry, so that the
    equivalence contract has one implementation and one test surface.
23. As a developer, I want the fossil modules (the unused key normalizer, the stub config,
    the redundant constant wrappers) deleted, so that the tree does not teach dead
    conventions.
24. As a developer, I want the Triton track's dependency on the PyTorch track's building
    blocks to be an explicit, documented seam (or moved to a shared location), so that the
    track boundaries are honest.
25. As a developer, I want the checkpoint format validated once at the interchange seam, so
    that a malformed or version-skewed checkpoint fails fast with a clear message.
26. As a developer, I want the JetPack 7.2.1 / CUDA 13.2 toolchain adopted (torch cu132
    build, corrected index pins, compute-capability flags), so that the GPU tracks run on
    the actual hardware this repo now lives on.
27. As a developer, I want the cross-backend parity tests to keep their tiered tolerance
    policy, so that the equivalence guarantee is measured consistently across standalone
    ops, single chains, and full backward chains.
28. As a developer, I want old checkpoints declared incompatible (new key scheme after the
    architecture change), so that nobody silently loads a pre-migration checkpoint into the
    new architecture.
29. As a maintainer, I want the NumPy god module split into per-operator modules, so that
    each operator's forward, backward, and tests live together and the math for one concept
    is one file.
30. As a maintainer, I want the model constructors to take the shared config object rather
    than a long parameter list, so that adding a knob is a change in one place per track,
    not a signature change across scripts and tests.
31. As a maintainer, I want the parity machinery (test files + verification script) to
    derive the track list and the parameter access from the same registry, so that the
    "three ways to check equivalence" stop drifting apart.
32. As a maintainer, I want an ADR recorded for abandoning the gated-residual design, so
    that future architecture reviews do not re-suggest it without the rationale on file.
33. As a maintainer, I want the generation loop to expose the per-token step path
    (forward one token → sample → append), so that the KV cache, the sampling, and the
    teacher-forced training path share one concept of "one step".
34. As a maintainer, I want the attention module's dead cache argument removed, so that the
    interface only promises what it does.
35. As a maintainer, I want the design monologues and broken docstrings fixed, so that the
    shipped source reads as documentation, not as a work in progress.

## Implementation Decisions

**Architecture (all four tracks, one shape):**

- The decoder block becomes the canonical pre-norm block: input → RMSNorm → attention →
  residual add; then RMSNorm → feed-forward → residual add. This replaces the current
  post-norm wiring and the "gated residual" scalar-gating, which is deleted (not renamed):
  it multiplied the whole stream, it was not a gate, and its removal passes the deletion
  test (complexity vanishes; nothing moves). This explicitly supersedes the old design doc's
  "gated residual" decision — an ADR will be recorded (see Further Notes).
- Attention keeps scaled dot-product semantics with numerical-stability max-subtraction,
  RoPE applied to queries and keys, and Grouped-Query Attention wired end-to-end: the config
  and CLI already carry the group count; the model constructors and the block will now
  accept and forward it so that fewer K/V groups than query heads actually changes the K/V
  projection shapes and the K/V head-sharing broadcast.
- RoPE is on by default. The partial-rotation knob keeps one semantic (0 = rotate the full
  head dimension, a value between 0 and the head dimension rotates a prefix), and every
  docstring and the CLI help agree with it. The current contradiction (config says 0 = full,
  model code treats 0 = disabled) is resolved in favor of the config/CLI semantic.
- The feed-forward default is the dense SwiGLU feed-forward (the industry standard); MoE
  (router + top-k over SwiGLU experts) becomes the documented opt-in selected by a config
  flag. The MoE algorithm keeps softmax-over-all-experts with top-k masking and
  renormalization, but the top-k selection is made tie-stable and the expert computation is
  restricted to the selected experts where the track allows it (the NumPy reference keeps
  the simple all-experts-then-mask form only if the math section says so explicitly —
  decision: the NumPy track computes all experts and masks, with a documented note that
  production tracks compute selected experts only, because the point is routing math, not
  expert dispatch; the PyTorch track computes all experts with a single einsum as today for
  parity simplicity, documented as such).
- The output head becomes a single linear layer (the industry "lm_head"), replacing the
  SwiGLU-plus-linear double head. Embedding/lm_head weight tying is out of scope (the
  head stays a separate matrix) but the name and shape (`D → V`) match convention.
- All four tracks mirror the same architecture and the same parameter key scheme. Track-
  specific implementation style is preserved: NumPy = manual arrays + manual analytic
  gradients; PyTorch = `nn.Module` + autograd + framework idioms; Triton = PyTorch wiring
  with Triton kernels at the named seam points (attention, FFN, layernorm, RoPE, activation);
  CUDA = bare-metal nvrtc pipeline with cuBLAS matmuls where the current hybrid design uses
  them.

**NumPy track (the math reference):**

- The current single 1100-line module file is split into per-operator modules: embedding,
  normalization (RMSNorm), attention (including the GQA broadcast), RoPE, feed-forward
  (SwiGLU), MoE, and the block/stack. Each operator module owns its forward *and* analytic
  backward, with the chain-rule equations in the docstring, shape annotations at every
  intermediate, and a reference (Vaswani §3.2.1/§5, the RMSNorm paper, the RoPE paper, the
  SwiGLU/GLU paper) where a real reference exists.
- The model's backward composes the per-operator backwards in reverse order; the model
  interface gains a `train_step` so that forward, loss, backward, and optimizer step are one
  documented path. The optimizer (manual AdamW) stays in the NumPy track with its formula
  docstring.
- The existing per-element finite-difference loop is demoted to a gradient-checking
  utility (relative-error comparison against the analytic gradients) used by tests; it is
  no longer the training path.

**Interchange (checkpoint seam):**

- A new shared module — the parameter registry — is the single owner of the flat-dict
  checkpoint format: every parameter's key, shape (as a function of the config), dtype, and
  per-track layout rule (e.g. the PyTorch linear-weight transpose) live there. All
  save/load in all four tracks become thin adapters over the registry; the three separate
  PyTorch traversals collapse into one, and CUDA gains a real load path through the same
  format (its attribute-level naming converges on the registry keys).
- The checkpoint format keeps its two-file shape (config JSON + flat npz); the config is
  validated against the registry on load, and a checkpoint whose keys do not match the
  current registry (e.g. a pre-migration checkpoint containing the deleted gate
  parameters) fails fast with a clear message. Pre-migration checkpoints are declared
  incompatible.
- Fossils are deleted: the checkpoint key normalizer that maps names nothing emits, the stub
  config module that exists to pass an import test, and the redundant constant wrappers that
  forward to the real key helpers. The constants module keeps one generation of key
  constants, owned by the registry.

**KV cache (kept, wired):**

- The KV cache is a first-class part of inference, not a side module: the attention module
  gains a real cache path (given the new token, project k/v, RoPE the new key, append to
  the cache, attend over cached keys and values), and the generator steps on one token per
  iteration instead of re-running the whole sequence. The dead cache argument in the
  PyTorch attention (and its design monologue) is removed in favor of this path.
- The naive cache is the default; the 1-bit quantized cache (TurboQuant) is kept as the
  documented compression experiment, wired as an alternative cache implementation behind the
  same interface, with an explicit parity budget (it degrades outputs; the docs state how
  much and why) instead of being an untested orphan.
- The final-norm epsilon mismatch between tracks (two different values) is resolved by one
  constant in the config.

**PyTorch track (idioms):**

- Attention uses the framework's scaled-dot-product function (which dispatches to
  flash/efficient kernels on GPU) for the computation; the hand-rolled max-subtract softmax
  stays only in the NumPy and (where pedagogically needed) the Triton/CUDA kernel tracks.
  The attention module keeps returning the attention weights for the educational logging
  when requested, without making the hot path pay for them by default.
- Training uses `torch.optim` instead of the manual optimizer in the PyTorch module file
  (the manual optimizer stays in the NumPy track, where it is the teaching artifact).
- The Triton track's imports of PyTorch building blocks become an explicit, documented seam:
  a shared location for the small building blocks both tracks need (or a documented
  one-way dependency with the PyTorch track as the reference). The Triton track's
  final-norm implementation is aligned with the PyTorch track's (same class, same epsilon)
  so the two "torch-family" tracks cannot drift.
- The broken docstring in the Triton model (a statement preceding the docstring, making it
  a no-op string) is fixed.

**Triton track (true flash attention):**

- The Triton attention kernel is upgraded from the current tiled matmul (which materializes
  per-tile score blocks and cites FlashAttention without implementing it) to an
  online-softmax kernel: running max and running sum per query row, rescaled accumulation of
  the value-weighted output, O(S) memory, single pass over K/V tiles.
- The kernel keeps the same interface as the existing SDPA wrapper so the cross-backend
  parity seam is unchanged; a parity + memory/throughput comparison against the framework
  SDPA is captured in the docs. The backward stays where it is (framework SDPA) for v1; a
  Triton backward kernel is out of scope.

**Naming, docs, domain model:**

- One docstring style (numpydoc) with the existing shape-annotation convention, applied
  consistently across tracks; the per-track style drift ends.
- A domain glossary file (CONTEXT.md) is created as the single place for the project's
  terms (block, GQA, RoPE, MoE, KV cache, checkpoint, parity tier); all docs and the ADR
  speak from it.
- README, design doc, and plan are re-aligned to one status source: what exists, what the
  four tracks are, the canonical block diagram (matching the code), and the CLI that
  actually runs. The plan's "remaining work" table is refreshed.
- Per-track naming converges on the industry names: one name for the MoE class across
  tracks, the CUDA track's parameter attributes renamed to the registry keys, and the
  "gated residual" concept removed from all docs (or, if revived later, a real gate with an
  ADR).

**Platform / dependencies (JetPack 6.2.2 → 7.2.1, CUDA 12.6 → 13.2):**

- PyTorch is installed as the cu132 build (torch 2.13.0+cu132 from the official PyTorch
  index) per the user's instruction; the project's index pin moves off the JetPack 6 / cu126
  mirror to the index that serves the current platform.
- The CUDA track's compile flags are checked against the new toolchain (the hardcoded
  compute-capability flag is verified/updated for the Orin under CUDA 13; the nvrtc
  pipeline is re-verified end to end).
- The Triton and cuda-python pins are re-verified against the CUDA 13.2 host (import + a
  smoke kernel each), and the platform notes in the plan/design docs (which still say
  JetPack 6.2.2 / CUDA 12.6) are updated.

## Testing Decisions

- **Good tests here** assert external behavior through the confirmed seams only: logits for
  fixed inputs and weights, parameter round-trips through the checkpoint format, and
  cross-track output equality. They do not assert source text, do not pin docstring
  wording, and do not test the registry's internals directly (the registry is
  implementation behind the checkpoint seam; it is covered by round-trips and by the
  gradient checker on the NumPy track).
- **Seam 1 — per-track model interface (existing, highest):** forward / train_step /
  save / load. Every architecture change is accepted here: same config + same seed + same
  inputs → identical logits within the existing tiered tolerances; a trained model's
  checkpoint loaded into another track → near-identical outputs. This is where the GQA,
  RoPE, KV-cache, and lm_head changes are all proven.
- **Seam 2 — flat-dict checkpoint (existing):** `ckpt.npz` + config JSON. Round-trip
  save→load is a no-op; cross-track load produces parity; a stale-key checkpoint (the
  deleted gate parameters) is rejected with a clear error. The parameter registry is
  exercised *through* this seam only.
- **Seam 3 — per-operator interface, NumPy track only (existing pattern):** each operator's
  analytic backward is verified against the finite-difference gradient checker
  (relative error within tolerance) at fixed small sizes; operators are also unit-tested
  for forward shape and value against hand-computed cases. This seam is internal to the
  NumPy track and mirrors the existing per-operator unit tests.
- **Prior art:** the existing cross-backend parity suite (tiered tolerances per AGENTS.md:
  1e-4 standalone, 1e-3 single chain, 1e-2 full backward chain), the per-operator NumPy
  unit tests, the 3-way equivalence test (train on one track, load into others, compare
  greedy outputs), and the per-track GPU parity tests. New tests are added *next to* these
  suites in the same style, with the existing per-test timeout convention; GPU-marked tests
  are deselected when no GPU is available (the current behavior).
- **Gradient checker as test, not training:** the finite-difference utility gains a
  relative-error comparison mode; the tests use it on the NumPy operators. It is removed
  from the training path entirely.
- **Flash kernel:** the Triton online-softmax kernel is accepted at Seam 1 (parity with the
  framework SDPA within the standalone tier) with a memory/throughput measurement recorded
  in the docs; the measurement is reported, not asserted (GPU variance).
- **Platform smoke:** after the dependency upgrade, an import + tiny-forward smoke per GPU
  track (marked `gpu`, skipped without CUDA) proves the cu132 toolchain works on the
  host.

## Out of Scope

- Embedding/lm_head weight tying (the head stays a separate matrix; naming and shape match
  convention, tying is a future option).
- Learning-rate scheduling (still deferred per the plan; fixed LR remains).
- The design doc's "multi-level KV cache" (LRU/LFU tiers): only the naive cache (default)
  and the 1-bit quantized cache (documented experiment) exist.
- A Triton backward kernel for the flash attention (backward stays on the framework SDPA).
- Expert-parallel / distributed MoE dispatch; the all-experts-then-mask teaching form is
  documented as intentional in the reference track.
- Retraining on the real TinyStories corpus as part of this change (parity is proven with
  the existing synthetic/small-data regime; a real training run is a follow-up).
- The tokenizer, the dataset loader, and the byte-level BPE scheme are unchanged.
- New public CLI surface: existing entry points keep their flags; dead flags are fixed to
  work, not added to.
- UI, notebooks, or non-Python artifacts.

## Further Notes

- **Execution order** (per the review's top recommendation): architecture standardization →
  parameter registry → NumPy analytic backprop → KV cache wiring → PyTorch idioms → flash
  kernel → docs/naming/glossary, with the dependency upgrade running in parallel from the
  start (it gates only the GPU-track verification).
- **ADR to record:** abandoning the gated-residual design (supersedes the old design doc's
  key design decision #6). Rationale on file: the "gate" multiplied the entire post-norm
  stream (a scalar scale, not a gate), the docstring described wiring the code did not
  perform, and the standard pre-norm block is what the project's learners will search for.
  The KV-cache "keep it" decision is a user decision recorded in this spec.
- **Checkpoint compatibility:** pre-migration checkpoints are incompatible (deleted gate
  parameters, new head, changed K/V projection shapes under GQA). The registry's load-time
  validation makes this a loud failure, not a silent one.
- **Issue tracker:** no tracker integration was available at authoring time (no gh CLI, no
  token, no triage-label vocabulary provided). This spec is published in the repo as the
  planning artifact of record; if issue-based tracking is wanted, run the skills setup so
  this can be filed with the `ready-for-agent` label.
- **Environment:** host is a Jetson AGX Orin on JetPack 7.2.1 / CUDA 13.2. The GPU tracks'
  tests are `gpu`-marked and run when CUDA is visible in the shell; the CPU-visible
  verification (NumPy + PyTorch tracks, full parity) runs everywhere.
