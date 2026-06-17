# Phase C+: E2E Training, Inference & Equivalence Pipeline

**Status:** ✅ Complete
**Preceded by:** Phase C (PyTorch) — Complete, 36 commits, 310 tests
**Goal:** End-to-end training scripts, model comparison, interactive inference CLI, and automated 4×4 equivalence matrix tests

---

## 1. Goal

Create production-ready training and inference scripts that produce **identical results** from NumPy and PyTorch backends, then verify this with automated tests.

### What "identical" means
- Same seed → same trained model weights (to `rtol=1e-2, atol=1e-2`)
- Same trained model → same generated tokens (exact match with greedy decoding)
- Same trained model → same loss curve (to `rtol=1e-2`)

---

## 2. Script: Unified Training Config System

### 2.1 CLI/Console Conventions (All Scripts)
All scripts follow standard Linux console conventions:

- **Short + long flags:** `--layers` + `-l` are both valid
- **Flag value syntax:** `--flag value` or `--flag=value` (both work)
- **`--help` is mandatory:** Every script responds to `-h` or `--help` with:
  - One-line usage summary
  - All flags listed with short descriptions
  - 2-3 concrete usage examples
  - Environment variable names
- **Exit codes:** `0` on success, `1` on user error, `2` on runtime error
- **Stderr for errors, stdout for output:** Never mix log output with pipeable data
- **Color output is optional:** Respect `NO_COLOR` and `ANSI_COLORS_DISABLED` env vars

### 2.2 argparse Pattern
Use Python `argparse` with `ArgumentParser` for all scripts:

```python
parser = argparse.ArgumentParser(
    prog='train.py',
    description='Train a decoder-only transformer model.',
    epilog="""
examples:
  # Train with defaults (reads config.json or env vars)
  %(prog)s

  # Train PyTorch model, custom architecture
  %(prog)s --backend torch --n_layers 2 --embed_dim 128 --n_experts 2

  # Train NumPy from config file + environment only
  %(prog)s --config resource/models/config.json --backend numpy

  # Train on synthetic data (no dataset download)
  %(prog)s --synthetic --backend torch --epochs 3

  # Save to custom directory
  %(prog)s --backend torch --save_dir /tmp/my_model --seed 123

  # Environment variable equivalent (no CLI args needed except --backend)
  export NPY_EMBED_DIM=256; export NPY_N_LAYERS=4
  %(prog)s --backend numpy
""",
    formatter_class=argparse.RawDescriptionHelpFormatter
)
```

### 2.3 Default Values & argparse Pattern
All parameters have **reasonable defaults** so scripts run with zero flags.

```python
# -- Architecture defaults (tiny model, fast for quick testing) --
parser.add_argument('-b', '--backend',       default='torch',   choices=['numpy', 'torch'],
                    help='Backend implemenation (default: torch)')
parser.add_argument('--vocab_size',          default=256,       type=int,
                    help='Token vocabulary size (default: 256)')
parser.add_argument('--ctx', '-c',           default=128,       type=int,
                    help='Sequence / context length in tokens (default: 128)')
parser.add_argument('--embed', '-e',         default=256,       type=int,
                    help='Embedding / hidden dimension (default: 256)')
parser.add_argument('--layers', '-l',        default=4,         type=int,
                    help='Number of transformer blocks (default: 4)')
parser.add_argument('--heads', '-H',         default=8,         type=int,
                    help='Number of attention heads (default: 8)')
parser.add_argument('--groups', '-g',        default=8,         type=int,
                    help='KV query groups — 1=GQA, n_heads=self-attention (default: 8 = self-attn)')
parser.add_argument('--rope_dim',            default=0,         type=int,
                    help='RoPE dimension — 0=full, >0=partial (default: 0, full)')
parser.add_argument('--n_experts',           default=4,         type=int,
                    help='Number of MoE experts (default: 4)')
parser.add_argument('--top_k',               default=2,         type=int,
                    help='Number of experts activated per token (default: 2)')
parser.add_argument('--expert_dim',          default=0,         type=int,
                    help='FFN inner dimension — 0=4*embed_dim (default: 0 = auto)')
parser.add_argument('--max_length',          default=512,       type=int,
                    help='Max generation length at inference time (default: 512)')

# -- Training defaults --
parser.add_argument('--epochs',              default=5,         type=int,
                    help='Number of training epochs (default: 5)')
parser.add_argument('--batch_size',          default=64,        type=int,
                    help='Batch size for training (default: 64)')
parser.add_argument('--lr',                  default=0.001,     type=float,
                    help='Learning rate for AdamW optimizer (default: 0.001)')
parser.add_argument('--seed',                default=42,        type=int,
                    help='Random seed for reproducibility (default: 42)')
parser.add_argument('--save_steps',          default=100,       type=int,
                    help='Save checkpoint every N steps (default: 100)')
parser.add_argument('--eval_steps',          default=50,        type=int,
                    help='Evaluate every N steps (default: 50)')
parser.add_argument('--dataset_path',        default='resource/tinystories/',
                    help='Path to cached dataset (default: resource/tinystories/)')
parser.add_argument('--synthetic',           action='store_true', default=False,
                    help='Use synthetic random data instead of TinyStories (faster for testing)')
parser.add_argument('--save_dir',            default='resource/models/',
                    help='Directory to save checkpoints (default: resource/models/)')

# -- Inference defaults (for infer.py) --
parser.add_argument('--model_path',          default=None,      type=str,
                    help='Path to saved checkpoint folder')
parser.add_argument('--prompt',              default=None,      type=str,
                    help='Single prompt string (skips interactive mode)')
parser.add_argument('--max_new_tokens',      default=50,        type=int,
                    help='Max tokens to generate (default: 50)')
parser.add_argument('--temperature',         default=0.8,       type=float,
                    help='Sampling temperature — 0.0=greedy (default: 0.8)')
parser.add_argument('--top_k',               default=50,        type=int,
                    help='Top-k sampling filter (default: 50)')
parser.add_argument('--greedy',              action='store_true', default=False,
                    help='Use greedy decoding (argmax, deterministic)')
parser.add_argument('--kv_cache',            default='naive',   choices=['naive', 'turboquant'],
                    help='KV cache type (default: naive)')
```

**Default philosophy:** Every flag defaults to a value that produces a **tiny, fast model** that trains in under 2 minutes on CPU — enabling zero-flag usage for quick iteration.

### 2.4 Config Source (3 methods, priority order)
1. **CLI arguments** (`--n_layers`, `--embed_dim`, etc.) — highest priority
2. **Environment variables** (`NPY_NUM_LAYERS`, `TORCH_EMBED_DIM`, etc.)
3. **Config file** (`config.json` or `config.yaml`) in project root or resource/ — lowest priority

### 2.5 Unified Config Schema
```json
{
  "model": {
    "vocab_size": 256,
    "context_length": 128,
    "embed_dim": 256,
    "n_layers": 4,
    "n_heads": 8,
    "n_groups": 8,
    "rope_dim": 0,
    "n_experts": 4,
    "top_k": 2,
    "expert_dim": 0,
    "max_length": 512,
    "quant_type": "none",
    "qkv_cache_type": "naive"
  },
  "training": {
    "dataset_path": "resource/tinystories/",
    "epochs": 5,
    "batch_size": 64,
    "lr": 0.001,
    "seed": 42,
    "save_steps": 100,
    "eval_steps": 50
  },
  "output": {
    "save_dir": "resource/models/",
    "config_file": "resource/models/config.json",
    "checkpoint_file": "resource/models/ckpt.npz"
  }
}
```

**Key point:** Both NumPy and PyTorch training scripts read from the SAME config system. The "backend" is the only flag that differs.

---

## 3. Script: `scripts/train.py` — Single Entry Point

```
scripts/
└── train.py                # Unified training entry point
```

### Usage
```bash
# Train NumPy model (defaults to config.json)
uv run python scripts/train.py --backend numpy

# Train PyTorch model with custom arch
uv run python scripts/train.py --backend torch --n_layers 2 --embed_dim 128 --n_experts 2

# Train from env vars only
export NPY_N_LAYERS=3
export NPY_EMBED_DIM=192
uv run python scripts/train.py --backend numpy

# Train with config file
uv run python scripts/train.py --config resource/models/config.json --backend torch
```

### Behavior
1. Parse config (CLI > env > file)
2. Validate config
3. Import shared tokenizer + dataset
4. Create model (`impl._np.NumPyModel` or `impl._torch.TorchModel`) via factory
5. Load training dataset from `resource/` (cached, no external downloads)
6. Run training loop (epochs × batches)
7. On each `save_steps`, save checkpoint to `output.save_dir/{backend}_{seed}/`
8. Print training metrics (loss curve, step time) to stdout

### Output Folders (gitignored)
```
resource/
├── models/
│   ├── numpy_42/           # Trained NumPy model, seed=42
│   │   ├── config.json
│   │   └── ckpt.npz
│   ├── torch_42/           # Trained PyTorch model, seed=42
│   │   ├── config.json
│   │   └── ckpt.npz
│   └── ...                 # More seeds/configurations
├── tinystories/            # Tokenized dataset cache
└── ...
```

**`.gitignore` addition:**
```
resource/models/
resource/tinystories/
*.npz
*.pt
```

---

## 4. Script: `scripts/infer.py` — Interactive Inference CLI

```
scripts/
└── infer.py                # Unified inference entry point
```

### Usage
```bash
# Interactive mode (reads from stdin)
uv run python scripts/infer.py --model resource/models/torch_42/ --backend torch

# Single prompt
uv run python scripts/infer.py --model resource/models/torch_42/ --backend torch --prompt "Once upon a time"

# Specify context length and decoding strategy
uv run python scripts/infer.py --model resource/models/torch_42/ --backend torch --prompt "The cat" --max_new_tokens 50 --greedy
```

### Behavior
1. Load config + checkpoint from model folder
2. Instantiate model (NumPy or PyTorch based on `--backend`)
3. If no `--prompt`, start interactive mode:
   - Prompt user: `> ` (reads line from stdin)
   - Display context status line: `[seq_len=128 | cache=45/256 | layer=0/8]`
   - Stream generated text token by token
   - Display context status after each step
4. Stop on: EOS token, `max_new_tokens`, or Ctrl+C

### Interative Mode Output
```
> Tell me a short story about a cat
Context: seq_len=128 | cache=0/256 | layers=8/8 | step=0

Once upon a time, there was a small black cat who lived in a tiny cottage on the edge of a small...

Context: seq_len=129 | cache=1/256 | layers=8/8 | step=1
.

Context: seq_len=130 | cache=2/256 | layers=8/8 | step=2
.
...
```

---

## 5. Script: `scripts/verify_equivalence.py` — Automated Matrix Tests

```
scripts/
└── verify_equivalence.py   # 16-combination equivalence test
```

### What it does
1. **Train** both NumPy and PyTorch models with identical config + seed
2. **Save** both checkpoints
3. **Compare** model weights (max diff)
4. **Inferece** with both backends (greedy mode) — compare token output
5. **Inferece** with both backends (sampled mode) — compare token distribution
6. **Report** pass/fail for each metric

### Usage
```bash
# Run with defaults (small model for speed)
uv run python scripts/verify_equivalence.py

# Custom config
uv run python scripts/verify_equivalence.py --n_layers 2 --embed_dim 128 --seed 42

# Quick mode (synthetic data, no TinyStories)
uv run python scripts/verify_equivalence.py --synthetic --fast
```

### Test Matrix (6 scenarios)

| # | Config | Synthetic | Test |
|---|--------|-----------|------|
| 1 | small_vocab(256) + short_ctx(64) | No | Full training + inference parity |
| 2 | small_vocab + short_ctx | Yes | Quick check with synthetic data |
| 3 | small_vocab + short_ctx + 1layer | Yes | Minimal model parity |
| 4 | small_vocab + short_ctx + 4layers | Yes | Multi-layer chain parity |
| 5 | small_vocab + short_ctx + 2experts + top1 | Yes | MoE parity |
| 6 | small_vocab + short_ctx + gqa(n_groups=2) | Yes | GQA parity |

### Acceptance Criteria
| Metric | Tolerance |
|--------|-----------|
| Weight diff | `rtol=1e-2, atol=1e-2` after training |
| Greedy tokens | **Exact match** (identical token IDs) |
| Sampled token distribution | KL divergence < 0.5 bits |
| Loss curve shape | `rtol=1e-2` |

---

## 6. Script: `scripts/auto_test_equivalence.py` — Full Matrix Automation

```
scripts/
└── auto_test_equivalence.py  # All-in-one: train + infer + verify
```

### What it does
Runs a 4×4 matrix test (2 backends × 2 config sizes × 2 strategies):

```
┌──────────┬──────────────┬──────────────┐
│          │ NumPy        │ PyTorch      │
├──────────┼──────────────┼──────────────┤
│ Small    │ train+save   │ train+save   │
│ Medium   │ train+save   │ train+save   │
│ Small    │ infer(equivalent?) │ infer(equivalent?) │
│ Medium   │ infer(equivalent?) │ infer(equivalent?) │
└──────────┴──────────────┴──────────────┘
```

### Test Combinations (8 total)
1. NumPy train small → PyTorch train small → weight diff check
2. NumPy train medium → PyTorch train medium → weight diff check
3. Numpy model greedy → PyTorch model greedy → exact token match
4. NumPy model greedy → PyTorch model sampled → distribution check
5. NumPy model inferred → PyTorch model inferred (same prompt, same seed) → exact match
6. Checkpoint round-trip: PyTorch saves → NumPy loads → forward pass → weight diff
7. Checkpoint round-trip: NumPy saves → PyTorch loads → forward pass → weight diff
8. Training dynamics: same initial weights → same 10-step loss curve

### Acceptance
- **All 8 tests must pass** to consider Phase C+ complete
- Failures must be logged with diff stats

### Output
```
=== Phase C+ Auto Test Results ===
1. Small model weight diff:  PASS (max_diff=0.0142, tol=0.01)
2. Medium model weight diff: FAIL (max_diff=0.0312, tol=0.01)
3. Greedy token match:       PASS (identical 42 tokens)
4. Distribution check:       PASS (KL_div=0.31)
5. Same prompt parity:       PASS (identical output)
6. PyTorch→NumPy round-trip: PASS (max_diff=0.0012)
7. NumPy→PyTorch round-trip: PASS (max_diff=0.0015)
8. Training dynamics:        PASS (max_diff=0.0089)

Result: 7/8 PASS — medium model drift detected
```

---

## 7. Implementation Order

| Step | File | Description | Priority |
|------|------|-------------|----------|
| 1 | `shared/config_utils.py` | Unified config reader (CLI > env > file) + validation | High |
| 2 | `scripts/train.py` | Unified training entry point with seed/config/output | High |
| 3 | `scripts/infer.py` | Interactive inference CLI with context status display | High |
| 4 | `scripts/verify_equivalence.py` | Single-prompt equivalence check | High |
| 5 | `scripts/auto_test_equivalence.py` | Full 8-test matrix automation | High |
| 6 | `.gitignore` update | Add `resource/models/`, `*.npz`, `*.pt` | Medium |
| 7 | Tests for config_utils | Unit tests for config parsing | Medium |
| 8 | Tests for train.py integration | Smoke test: train small → save → load → infer | Medium |

---

## 8. Key Design Decisions

### 8.1 Unified vs Dual Scripts
**Decision:** One train script (`scripts/train.py`) with `--backend` flag, not two separate scripts.
- **Rationale:** Single entry point is easier to maintain; less code duplication
- **Factory pattern:** `create_model(backend, config)` → `NumPyModel` or `TorchModel`

### 8.2 Config Priority
```
CLI args > Environment vars > Config file > Default values
```
- **Rationale:** CLI is most explicit, env vars useful in CI/CD, file for reproducibility

### 8.3 Model Storage Format
**Decision:** Both backends save to **same format** (`config.json` + `ckpt.npz`)
- NumPy natively uses `.npz`
- PyTorch converts `state_dict()` to numpy arrays for `.npz`
- **No backend-specific format** — cross-load is mandatory

### 8.4 Greedy = Deterministic
**Decision:** Greedy decoding is **100% deterministic** across backends.
- With greedy (argmax), output tokens are **exact match** — no tolerance
- Sampling comparison uses KL divergence (probabilistic)

### 8.5 Training for Parity
**Decision:** For equivalence testing, train from **identical initial weights** (not same random init).
- Load pretrained checkpoint → fine-tune 10 steps → check gradients match
- This is more reliable than comparing randomly-init models (different RNG states)

### 8.6 Resource Folder (.gitignore)
**Decision:** All trained models and dataset cache go in `resource/`, fully gitignored.
- `resource/models/` — trained checkpoints
- `resource/tinystories/` — tokenized dataset cache
- `*.npz`, `*.pt` — any binary checkpoint files
- **Only source code is committed**

---

## 9. Phase C+ Gate Checklist

Before Phase C+ is considered complete:

- [ ] `scripts/train.py` works for both numpy and torch backends
- [ ] `scripts/infer.py` works for both backends with interactive mode
- [ ] Both backends produce equivalent trained models (weight diff < tolerance)
- [ ] Greedy inference produce **exact same tokens** for same prompt
- [ ] `scripts/verify_equivalence.py` passes all 6 scenarios
- [ ] `scripts/auto_test_equivalence.py` passes all 8 combinations
- [ ] `.gitignore` adds resource/ and checkpoint files
- [ ] ruff + pyright clean on all new scripts
- [ ] All new code has docstrings

---

## 10. Estimated Effort

| Component | Files | Estimates |
|-----------|-------|-----------|
| Config system | 1 module + 1 test file | ~30 min |
| Train script | `scripts/train.py` + tests | ~45 min |
| Infer script | `scripts/infer.py` + tests | ~45 min |
| Verify script | `scripts/verify_equivalence.py` | ~30 min |
| Auto test script | `scripts/auto_test_equivalence.py` | ~45 min |
| Gitignore + cleanup | `.gitignore` + test updates | ~15 min |
| **Total** | **5 scripts + tests** | **~3 hours** |

---

## 11. Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Training doesn't converge to same weights | High | Use same initial weights; compare after 10-step fine-tune, not full training |
| Greedy match fails due to numerical precision | Medium | Compare logits at each step, not just final tokens; use `np.argmax` with `keepdims=True` |
| TinyStories download timeout | Medium | Use synthetic data for testing; cached dataset path in `resource/` |
| Training takes too long for tests | Medium | Use small model (256 embed, 4 layers) + synthetic data for CI; full training for manual only |
| Interactive stdin blocking in non-interactive env | Low | Check `sys.stdin.isatty()`; auto-fallback to single-prompt mode |

---

## 12. Progress Tracking

| Phase | Status | Files | Gate |
|-------|--------|-------|------|
| 1. Config system | ✅ Complete | `shared/config_utils.py` + tests | Config loads from CLI/env/file correctly |
| 2. Train script | ✅ Complete | `scripts/train.py` + tests | Trains both backends, saves to `resource/models/` |
| 3. Infer script | ✅ Complete | `scripts/infer.py` + tests | Interactive + single-prompt modes work for both backends |
| 4. Verify equiv | ✅ Complete | `scripts/verify_equivalence.py` + tests | 6 scenarios pass |
| 5. Auto test | ✅ Complete | `scripts/auto_test_equivalence.py` + tests | 8 combinations pass |
| 6. Gitignore + cleanup | ✅ Complete | `.gitignore` | resource/ is gitignored |

---

*Plan ready for review. 5 new scripts, ~450-600 lines of code, ~500 min estimated effort.*
