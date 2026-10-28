# Repository Guidelines

## Project Overview

Pedagogical decoder-only transformer implemented in **four equivalent backends**:
NumPy (manual math + analytic backward — the teaching reference), PyTorch
(autograd idiom), Triton (FlashAttention-style kernels over a PyTorch model),
and CUDA (bare-metal NVRTC kernels). All tracks share one architecture, one
config (`shared/config.py`), and one checkpoint key scheme
(`shared/constants.py` + `shared/registry.py`), so a checkpoint trained on
one track loads and runs on any other. Code is deliberately heavy on math
documentation: formulas cite references and every matrix operation carries a
shape comment.

**Where each kind of knowledge lives** (docs reorganization of 2026-10-10 —
link, never duplicate):

- `CONTEXT.md` (root) — the domain glossary: every term defined once, shape
  notation included. The single owner of definitions.
- `README.md` (root) — project interface: what this is, quickstart, the
  learning-mode model card, the full project-structure tree, usage.
- `docs/architecture/` — `design.md` (doctrine + decisions), the
  `seam_triton_to_torch.md` shared-seam contract, `adr/` (irreversible
  decisions).
- `docs/math/` — `transformer-walkthrough.md` (the forward-pass derivation
  tour, code-anchored), `training-pipeline.md` (pre-train → SFT → tool →
  thinking → RL → drafter pipeline, ADRs 0003/0004).
- `docs/reference/` — `docstring_style.md` (the numpydoc + shape-comment
  convention this repo's docstrings follow).
- Finished task specs are **deleted** — the git history is the record.
  Doc-vs-code alignment is enforced by `scripts/check_docs.py` (referenced
  paths must exist; claimed counts must match the source).

## Architecture & Data Flow

**Model (identical in all 4 tracks):** LLaMA-style pre-norm block
`h = x + attn(ln1(x)); out = h + mlp(ln2(h))`; RMSNorm (no biases, no weight
tying), RoPE (`rope_dim=0` rotates the full head dim), causal mask, GQA via
`n_groups` (None → MHA), dense SwiGLU FFN by default with opt-in MoE
(`n_experts > 1`: softmax router + top-k mask + renormalize; opt-in
always-on ungated shared experts via `n_shared_experts > 0`, ADR 0002).

Forward: `tokens (B,S) → embedding (B,S,D) → DecoderStack → final RMSNorm →
lm_head (D→V) → logits (B,S,V)`.

Inference: exact KV-cache step path — `forward_prefill` + `forward_step`
equals a full re-forward; generators step one token per iteration (O(1)).
TurboQuant (1-bit quantized K/V cache) is wired behind the same
`forward_step(quantize=True)` interface with a parity-budget test.
Speculative decoding (ADR 0003): all four tracks also expose
`forward_chunk(ids, position, cache) -> (logits, x_final)` — k tokens'
K/V appended in ONE pass (the verification path); drafters (MTP k=4
sequential, DSpark k=8 semi-AR + confidence-scheduled verification with a
0.8 survival target) implement the `shared/draft.py` Drafter protocol and
live as sidecar checkpoints (`<model>/draft_mtp/`, `<model>/draft_dspark/`
— the target's registry is untouched). Greedy spec output is
token-identical to plain greedy (the lossless contract; NumPy additionally
implements the rejection-sampling theorem for temperature mode). Drafters
train by distillation (`scripts/train_drafters.py`, bootstrapped by
learning mode); learning mode starts in MTP mode when sidecars exist
(`--spec plain|mtp|dspark` overrides), showing acceptance/TPS vs a paired
plain run.

**NumPy backward:** every operator has a closed-form
`backward(dout, x) -> (dinput, dparams)` (stateless — recomputes
intermediates); `impl/_np/gradcheck.py` verifies it against finite
differences (float64, ~1e-10, MoE top-k flip-aware).

**Weight interchange (the checkpoint seam):**

- `ParameterRegistry(config)` in `shared/registry.py` is the single owner of
  the checkpoint format: key (HF-Llama naming), shape derived from config,
  and the `torch_transpose` layout rule — only `nn.Linear`-backed weights are
  transposed for the PyTorch track.
- Each track has exactly **one** storage-binding map
  (`_param_slots()`/`_param_arrays()` np / `_param_tensors()` torch, triton,
  cuda → `ParameterRegistry.bind()`); all save/load walk the registry, not a
  per-track key list. Load validates key-set + shapes (stale checkpoints fail fast).
- **Construction lifecycle (constructor injection, all four tracks)** — every
  model is born in exactly one of two states: `Model(config)` = fresh
  seed-initialized weights (training), or `Model(config, params)` /
  `Model.from_checkpoint(dir)` = the checkpoint arrays injected **at
  construction** (inference). Injection is threaded **down the whole chain**:
  the model validates against the registry, then passes the flat `Keys`-keyed
  mapping to `DecoderStack`/`Embedding`/final norm/lm_head; each
  `TransformerBlock` takes its own `(layer_index, params)` slice and hands each
  operator its array (`RMSNorm(gamma=…)`, `MultiHeadAttention(q_proj=…, …)`,
  `SwiGLUFFN(gate_proj=…, …)`, MoE reads `Keys.moe_*` itself) — so no operator
  is ever built with defaults and patched afterwards, and `params=None` is
  visibly the training start (gammas = ones). Injection copies **in place**
  over the storage the constructor created (the `load_state_dict` convention:
  storage identity never changes, dtype/device moves are the only re-binding
  paths — `NumPyModel.to()` mirrors torch's `.double()`), and a stale
  checkpoint fails the registry validation before anything is built.
  `load_from_numpy_dict` stays the explicit in-place loading interface (weight
  swap into a live model, round-trip tests); nothing loads by reassigning
  attributes. Drafters (sidecars) follow the same pattern with their sidecar
  key paths.
- `shared/checkpoint.py` does disk I/O: `config.json` + flat `model.npz`.

**Learning mode (all four backends):** the web page runs records on all four
backends, selected per request via the in-page dropdown; each track has its
own record adapter (`impl/_np/learning.py`, `impl/_torch/learning.py`,
`impl/_triton/learning.py`, `impl/_cuda/learning.py`). The NumPy one consumes
the track's own `forward_state` hooks (bit-identical to the forward); the
PyTorch one runs the track's real forward and recomputes the attention
intermediates with explicit ops (a documented display path, since the fused
SDPA call exposes none). All four serialize into one shared JSON shape ("step
records" / inference records), so the page consumes any backend
interchangeably. `impl/_np/learning_server.py` serves the web page
(`impl/_np/web/`) plus a small JSON API (`/api/model`, `/api/inference`,
`/api/record`) via stdlib `http.server`, listening on `0.0.0.0:8080` by
default; the CLI `--backend` (`numpy`|`torch`) only picks which track the
*startup* load materializes. The default served checkpoint is
`learning_moe_rl` — the MoE family's FULL v3 pipeline model (BPE-512
vocab with the `<|think|>`/`<|answer|>` specials, `D=64, L=3, H=4`;
built by `scripts/train_demo_model_v3.py --variant moe`: pretrain →
code SFT → tool SFT → thinking SFT (reasoning distillation, ADR 0004) →
GRPO RL (ADR 0004) → drafter distillation). The page's Model dropdown
carries four variants — `moe-rl` (default), `dense-rl`, `moe`, `dense` —
giving TWO controlled comparisons: `*-rl` vs `*` isolates RL's delta
(the RL run started from exactly the shown SFT checkpoint), and
`moe-rl` vs `dense-rl` isolates the FFN geometry — the param-matched
MoE of ADR 0005 (E=8 routed + 1 shared at `expert_dim=32`: ~1.08× the
dense total at ~54% active/token). Missing/stale checkpoints (the
`"pipeline": 3` config.json marker) bootstrap: the default family at
startup, the other family in a background thread after the server is
up (a still-training variant answers "training, retry shortly").

The full directory tree lives in [README.md](README.md) → "Project
structure" (one owner, kept aligned by `scripts/check_docs.py`).

## Development Commands

```bash
uv sync   # Python 3.10; torch==2.13.0+cu132 via the pinned custom uv index

# Lint / format / typecheck (must stay clean)
uv run ruff format . && uv run ruff check .
uv run pyright .   # repo-wide, 0 errors + 0 warnings

# Doc-vs-code alignment check (paths/counts claimed in md must match reality)
uv run python -m scripts.check_docs
```

Full inference / training / learning-mode / verification commands live in
[README.md](README.md) → "Usage" (one owner). The four commands above are
the dev loop.

## Code Conventions & Common Patterns

1. **Quick iteration feedback loop** — when debugging, run the minimal
   failing test first, capture the actual error, then make a targeted fix.
   Every hypothesis is validated by a test result, not more reading.
2. **Tiered parity tolerances** (all parity tests use float64):
   - Standalone components: `rtol=1e-4, atol=1e-4` (LayerNorm, FFN, MoE, MHA
     tested in isolation).
   - Component in a single chain (e.g. MHA inside a TransformerBlock):
     `rtol=1e-3, atol=1e-3`.
   - Full backward chains (gradient through 2+ layers):
     `rtol=1e-2, atol=1e-2`.
3. **Key constants, never raw strings** — parameter-dict keys come from
   `shared/constants.py`: `Keys.embed()`, `Keys.lm_head()`,
   `Keys.attn(0, Attn.Q_PROJ)`, `Keys.ln(0, LayerNorm.INPUT)`,
   `Keys.ffn(0, Mlp.GATE_PROJ)`, `Keys.moe_gate(0)`,
   `Keys.moe_expert(0, 0, Mlp.UP_PROJ)`. Never literals like `"gamma"` or
   `"blocks.0.ln1.gamma"`.
4. **Strict type hints** on all interfaces; `from __future__ import
   annotations` at the top of every module.
5. **Docstrings + shape comments** — numpydoc style (see
   `docs/reference/docstring_style.md`); math code cites its formula
   reference; matrix ops carry shape comments, e.g.
   `# (B,S,H,hd) @ (H,hd,hd) -> (B,S,H,hd)`.
   Required in the NumPy track, encouraged elsewhere.
6. **One storage binding per track** — checkpoint save/load walks
   `ParameterRegistry.entries`; adding a parameter means extending the
   registry, not a key list in a track.
7. **Cross-track imports** — the only allowed *model-math* seam is
   `impl/_triton → impl/_torch` (RoPE); it is documented in
   `docs/architecture/seam_triton_to_torch.md`
   with a `# shared-seam:` marker. One documented reverse exception: the
   learning-record dispatch (`impl/_torch/learning.py` function-locally
   imports the triton/cuda record adapters, dispatching on model class) —
   display plumbing, not model math. Do not add other cross-track imports.
8. **Logging** — `logger = logging.getLogger(__name__)` per module; dotted
   logger names map 1:1 to file paths (`impl._np.attention` =
   `impl/_np/attention.py`). Educational logs show the math in motion (shape
   chains, attention entropy, grad stats, top-5 sampling).
9. **`# PROD:` notes** — where the teaching implementation deliberately takes
   the readable path instead of the production one, mark it with a one-line
   `# PROD: <what production would do> — <why/where>` comment; where the repo
   contains the production version, point at it (see
   `docs/reference/docstring_style.md`).
10. **Entry-point contract** — per-track `cli.py` files are single-track
    smoke demos on random weights (no checkpoint loading, no learning mode);
    all checkpoint-based or cross-backend entry points live in `scripts/`
    (`train.py`, `infer.py`, `learning.py`). New user-facing commands go in
    `scripts/`, never into a track's `cli.py`.
11. After any change: `ruff format`, `ruff check`, `pyright`, and
    `scripts.check_docs` must be clean. If a request is unclear or has
    materially different approaches, confirm with the user first; do not
    make technical or business assumptions.
12. **Subagents** — never run more than 3 subagents in parallel at a time.

## Important Files

- `shared/config.py` — `TransformerConfig`: single source of truth for every
  dimension (GQA, RoPE, MoE); validated in `__post_init__`, derived fields
  computed.
- `shared/constants.py` + `shared/registry.py` — the checkpoint contract
  (HF-Llama key scheme; format owner with `validate()`).
- `impl/_np/model.py` — reference `NumPyModel` (forward, analytic backward,
  `load_from_numpy_dict`); per-track `cli.py` files are single-track
  inference demos on random weights.
- `impl/_torch/layers.py` — PyTorch operator mirror (SDPA, GQA repeat before
  SDPA); `impl/_triton/transformer.py` — Triton stack + the documented seam.
- `scripts/train.py`, `scripts/infer.py`, `scripts/learning.py` — unified
  4-backend entry points; `scripts/verify_equivalence.py` — 11-scenario
  parity check (`scripts.check_docs` pins the count).
- `tests/unit/_cuda/conftest.py` — NVRTC per-file context isolation
  (CUDA tests run one file per invocation).
- `CONTEXT.md` — the domain glossary (every term + the shape notation);
  `docs/architecture/adr/` — the irreversible decisions (residual,
  shared-expert MoE, drafter sidecars, thinking-SFT + GRPO RL,
  param-matched MoE scale).

## Runtime/Tooling Preferences

- `uv` only (the repo `.venv`); Python 3.10; `torch==2.13.0+cu132` from the
  pinned `pytorch-cu132` uv index — do not re-resolve torch from PyPI.
  Optional extras: `triton`, `cuda` (`cuda-python`).
- Ruff: `line-length = 120`, rules `E, W, F, I, UP, B, SIM, C90` (E501
  ignored — the formatter owns line length); `__init__.py` ignores F401.
- Pyright: repo-wide clean (`uv run pyright .` — 0 errors, 0 warnings;
  the `include = ["shared"]` pyproject entry only sets the default IDE
  scope).
- Pytest: global `timeout = 300` (pytest-timeout), `gpu` marker (deselect
  with `-m "not gpu"`), `pythonpath = ["shared"]`, `testpaths = ["tests"]`.
- Target hardware: fully tested on Jetson AGX Orin 64GB (JetPack 7.2.2,
  CUDA 13.2). The CUDA track compiles to per-device PTX (native arch, CC 8.0
  baseline fallback) and runs on any NVIDIA GPU with CC >= 8.0 (Ampere or
  newer) under a CUDA 13.x driver — see docs/architecture/design.md →
  "Platform Target". GPU tests are slow and stateful — run them in separate
  invocations (see below).
- `resource/` is fully git-ignored: TinyStories data and all checkpoints are
  local artifacts. Recreate: `scripts/train_real_tinystories` (real-data
  checkpoints), `scripts/train_demo_model_v3` (learning-mode pipeline +
  `--variant moe`; the thinking/RL stages need `scripts/download_gsm8k`
  first), `scripts/download_tinystories` (dataset).

## Testing & QA

```bash
# CPU unit tests (NumPy + PyTorch + the shared package + repo-root tests)
uv run pytest tests/unit/ -q --timeout=120 \
    --ignore=tests/unit/_cuda --ignore=tests/unit/_triton

# Triton unit tests (GPU)
uv run pytest tests/unit/_triton/ -q --timeout=120

# CUDA unit tests (GPU; one file per invocation for NVRTC context isolation)
for f in tests/unit/_cuda/test_*.py; do
    uv run pytest "$f" -q --timeout=120
done

# Cross-backend parity tests (GPU)
uv run pytest tests/cross_backend/ -q --timeout=120

# Doc-vs-code alignment (the docs' own test)
uv run python -m scripts.check_docs
```

- Every unit test must have a timeout (the global 300 s default applies;
  GPU suites keep an explicit `--timeout=120`).
- Parity tests: float64, tiered tolerances (rule 2). `tests/cross_backend/`
  covers dense/GQA/MoE parity, speculative decoding (chunk parity, lossless
  greedy, sidecar round-trip), GPU/CUDA parity, 3-way equivalence — run
  `pytest tests/cross_backend/ --collect-only -q | tail -1` for the live
  count rather than pinning it here.
- Gradient correctness: `tests/unit/_np/test_gradient_check.py` — analytic
  backward vs finite differences at ~1e-10 (incl. MoE kink handling).
- End-to-end equivalence: `uv run python -m scripts.verify_equivalence` —
  11 scenarios (`dense_np_torch`, `gqa_np_torch`, `moe_np_torch`,
  `moe_shared_experts_np_torch`, `gqa_torch_triton`, `spec_mtp_np_torch`,
  `spec_dspark_np_torch`, `moe_param_matched_np_torch`,
  `rl_dense_np_torch`, `cuda_shared_weights`, `all_four_backends`);
  greedy outputs must match.
- Round-trip guarantee to preserve: save from any backend → load into any
  other → identical logits.
