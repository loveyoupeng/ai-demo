# Repository Guidelines

## Project Overview

Pedagogical decoder-only transformer implemented in **four equivalent backends**:
NumPy (manual math + analytic backward — the teaching reference), PyTorch
(autograd idiom), Triton (FlashAttention-style kernels over a PyTorch model),
and CUDA (bare-metal NVRTC kernels). All tracks share one architecture, one
config (`shared/config.py`), and one checkpoint key scheme
(`shared/constants.py` + `shared/registry.py`), so a checkpoint trained on one
track loads and runs on any other. Code is deliberately heavy on math
documentation: formulas cite references and every matrix operation carries a
shape comment.

See [docs/task_plan.md](docs/task_plan.md) for phase status and [CONTEXT.md](CONTEXT.md)
for the domain glossary.

## Architecture & Data Flow

**Model (identical in all 4 tracks):** LLaMA-style pre-norm block
`h = x + attn(ln1(x)); out = h + mlp(ln2(h))`; RMSNorm (no biases, no weight
tying), RoPE (`rope_dim=0` rotates the full head dim), causal mask, GQA via
`n_groups` (None → MHA), dense SwiGLU FFN by default with opt-in MoE
(`n_experts > 1`: softmax router + top-k mask + renormalize).

Forward: `tokens (B,S) → embedding (B,S,D) → DecoderStack → final RMSNorm →
lm_head (D→V) → logits (B,S,V)`.

Inference: exact KV-cache step path — `forward_prefill` + `forward_step`
equals a full re-forward; generators step one token per iteration (O(1)).
TurboQuant (1-bit quantized K/V cache) is wired behind the same
`forward_step(quantize=True)` interface with a parity-budget test.

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
  (`_param_arrays()` np / `_param_tensors()` torch, triton, cuda); all
  save/load paths walk `ParameterRegistry.entries` — never a parallel key
  list. Load validates key-set + shapes (stale checkpoints fail fast).
- `shared/checkpoint.py` does disk I/O: `config.json` + flat `model.npz`.

**Learning mode (NumPy + PyTorch tracks):** `impl/_np/learning.py` and
`impl/_torch/learning.py` are the two record adapters. The NumPy one consumes
the track's own `_forward_state` hooks (bit-identical to the forward); the
PyTorch one runs the track's real forward and recomputes the attention
intermediates with explicit ops (a documented display path, since the fused
SDPA call exposes none). Both serialize into one shared JSON shape ("step
records" / inference records), so the page consumes either backend
interchangeably. `impl/_np/learning_server.py` serves the web page
(`impl/_np/web/`) plus a small JSON API (`/api/model`, `/api/inference`,
`/api/record`) via stdlib `http.server`, with a `--backend` flag selecting the
materialized track. The default demo model is a tiny char-level MoE LM
(`D=8, H=4, L=3, E=3, V=20`).

## Key Directories

| Path | Purpose |
| --- | --- |
| `impl/_np/` | NumPy reference track: per-operator modules, `model.py` (analytic backward), `training.py`, `inference.py`, KV cache (+ TurboQuant), `gradcheck.py`, `cli.py`, learning mode |
| `impl/_torch/` | PyTorch track: `layers.py` (all nn.Modules, `TorchModel`), `training.py`, `inference.py`, `kv_cache.py`, `turboquant_kv_cache.py`, `learning.py`, `cli.py` |
| `impl/_triton/` | Triton kernels over a PyTorch model: `attn.py`, `flash_attn.py` (online softmax), `transformer.py`, `cli.py` |
| `impl/_cuda/` | NVRTC bare-metal kernels: `compiler.py`, `attention.py`, `layernorm.py`, `rope.py`, `ffn.py`, `moe.py`, `model.py`, `training.py`, `cli.py` |
| `shared/` | Cross-track backbone: `config.py` (`TransformerConfig`), `constants.py` (Keys), `registry.py`, `checkpoint.py`, `init.py`, `tokenizer.py`, `dataset.py`, `config_utils.py`, `utils/` |
| `scripts/` | Unified `train.py`/`infer.py` (all backends), `verify_equivalence.py`, `train_real_tinystories.py`, `train_demo_model.py`, `download_tinystories.py`, `run_learning_mode.sh` |
| `tests/` | `unit/` (root: shared, scripts, registry; plus `_np`, `_torch`, `_triton`, `_cuda`) and `cross_backend/` (42 parity tests) |
| `docs/` | `specs/architecture-fixes.md` (spec + progress of record), `theory/transformer-walkthrough.md` (forward-pass tour), `seam_triton_to_torch.md`, `docstring_style.md`, `adr/`, `design.md` |
| `resource/` | **git-ignored**: TinyStories JSON + `models/{numpy,torch,triton,cuda}_real` and `models/learning_demo` checkpoints — recreate via scripts |
| `docs/task_plan.md`, `CONTEXT.md` | Plan/phase status, glossary |

## Development Commands

```bash
uv sync   # Python 3.10; torch==2.13.0+cu132 via the pinned custom uv index

# Lint / format / typecheck (must stay clean)
uv run ruff format . && uv run ruff check .
uv run pyright .   # pyright config include = ["shared"] only

# Inference (any track; defaults are a tiny model: vocab 256, D=16, L=1, H=2)
uv run python -m impl._np.cli    --prompt "the" --max_new_tokens 10
uv run python -m impl._torch.cli --prompt "the" --max_new_tokens 10
uv run python -m impl._triton.cli --prompt "the" --max_new_tokens 10   # GPU
uv run python -m impl._cuda.cli  --prompt "the" --max_new_tokens 10   # GPU
uv run python -m impl._torch.cli --prompt "Once upon a" --max_new_tokens 50 \
    --temperature 0.9 --top_k 20 --embed_dim 64 --n_layers 4 --n_heads 8

# Learning mode (NumPy web page; auto-trains the demo model on first run)
bash scripts/run_learning_mode.sh          # http://127.0.0.1:8080
uv run python -m impl._np.cli --learning --port 8080 --model resource/models/learning_demo

# Training / data / verification (unified scripts, --backend numpy|torch|triton|cuda)
uv run python -m scripts.train --backend torch [--synthetic] [--n_layers 2 --embed_dim 128]
uv run python -m scripts.infer --model resource/models/torch_real/ --backend torch --prompt "hello"
uv run python -m scripts.verify_equivalence [--fast] [--scenario gqa]
uv run python -m scripts.train_real_tinystories              # ~150 MB, 4 backends
uv run python -m scripts.train_real_tinystories --backends numpy,torch --num_batches 5 --suffix tmp
uv run python -m scripts.train_demo_model [--backend cuda]   # learning-mode demo model
uv run python -m scripts.download_tinystories
```

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
   `docs/docstring_style.md`); math code cites its formula reference; matrix
   ops carry shape comments, e.g. `# (B,S,H,hd) @ (H,hd,hd) -> (B,S,H,hd)`.
   Required in the NumPy track, encouraged elsewhere.
6. **One storage binding per track** — checkpoint save/load walks
   `ParameterRegistry.entries`; adding a parameter means extending the
   registry, not a key list in a track.
7. **Cross-track imports** — the only allowed seam is `impl/_triton →
   impl/_torch` (RoPE); it is documented in `docs/seam_triton_to_torch.md`
   with a `# shared-seam:` marker. Do not add other cross-track imports.
8. **Logging** — `logger = logging.getLogger(__name__)` per module; dotted
   logger names map 1:1 to file paths (`impl._np.attention` =
   `impl/_np/attention.py`). Educational logs follow
   `docs/task_plan.md` §H (shape chains, attention entropy, grad stats, top-5
   sampling).
9. **`# PROD:` notes** — where the teaching implementation deliberately takes
   the readable path instead of the production one, mark it with a one-line
   `# PROD: <what production would do> — <why/where>` comment; where the repo
   contains the production version, point at it (see
   `docs/docstring_style.md`).
10. After any change: `ruff format`, `ruff check`, and `pyright` must be
    clean. If a request is unclear or has materially different approaches,
    confirm with the user first; do not make technical or business
    assumptions.
11. **Subagents** — never run more than 3 subagents in parallel at a time.

## Important Files

- `shared/config.py` — `TransformerConfig`: single source of truth for every
  dimension (GQA, RoPE, MoE, KV-cache quant knobs); validated in
  `__post_init__`, derived fields computed.
- `shared/constants.py` + `shared/registry.py` — the checkpoint contract
  (HF-Llama key scheme; format owner with `validate()`).
- `shared/checkpoint.py` — `save_checkpoint` / `load_checkpoint`
  (`config.json` + `model.npz`).
- `impl/_np/model.py` — reference `NumPyModel` (forward, analytic backward,
  `load_from_numpy_dict`); `impl/_np/cli.py` — inference entry point incl.
  `--learning`.
- `impl/_torch/layers.py` — PyTorch operator mirror (SDPA, GQA repeat before
  SDPA); `impl/_triton/transformer.py` — Triton stack + the documented seam.
- `scripts/train.py`, `scripts/infer.py` — unified 4-backend entry points;
  `scripts/verify_equivalence.py` — 6-scenario parity check.
- `tests/conftest.py` (GPU isolation fixture) and
  `tests/unit/_cuda/conftest.py` (NVRTC per-file context isolation).
- `docs/specs/architecture-fixes.md` — spec + progress of record;
  `docs/adr/0001-gated-residual-abandonment.md` — why the block uses the
  standard additive residual.

## Runtime/Tooling Preferences

- `uv` only (the repo `.venv`); Python 3.10; `torch==2.13.0+cu132` from the
  pinned `pytorch-cu132` uv index — do not re-resolve torch from PyPI.
  Optional extras: `triton`, `cuda` (`cuda-python`).
- Ruff: `line-length = 120`, rules `E, W, F, I, UP, B, SIM, C90` (E501
  ignored — the formatter owns line length); `__init__.py` ignores F401.
- Pyright: config `include = ["shared"]` — strictly checked is `shared/`;
  keep `impl/` and `scripts/` clean anyway.
- Pytest: global `timeout = 300` (pytest-timeout), `gpu` marker (deselect
  with `-m "not gpu"`), `pythonpath = ["shared"]`, `testpaths = ["tests"]`.
- Target hardware: Jetson (Orin sm_87 / GB10); GPU tests are slow and
  stateful — run them in separate invocations (see below).
- `resource/` is fully git-ignored: TinyStories data and all checkpoints are
  local artifacts. Recreate: `scripts/train_real_tinystories` (real-data
  checkpoints), `scripts/train_demo_model` (learning-mode demo),
  `scripts/download_tinystories` (dataset).

## Testing & QA

```bash
# CPU unit tests (NumPy + PyTorch + shared/root)
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

# Run only non-GPU tests anywhere
uv run pytest tests/ -q -m "not gpu"
```

- Every unit test must have a timeout (the global 300 s default applies;
  GPU suites keep an explicit `--timeout=120`).
- Parity tests: float64, tiered tolerances (rule 2). `tests/cross_backend/`
  has 42 tests: dense/GQA/MoE parity, GPU/CUDA parity, 3-way equivalence.
- Gradient correctness: `tests/unit/_np/test_gradient_check.py` — analytic
  backward vs finite differences at ~1e-10 (incl. MoE kink handling).
- End-to-end equivalence: `uv run python -m scripts.verify_equivalence` —
  6 scenarios (`dense_np_torch`, `gqa_np_torch`, `moe_np_torch`, `gqa_torch_triton`,
  `cuda_shared_weights`, `all_four_backends`); greedy outputs must match.
- Round-trip guarantee to preserve: save from any backend → load into any
  other → identical logits.
