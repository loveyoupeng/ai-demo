# AGENTS.md

This is a demo repository to learn AI concepts and tooling.
It is building a decoder-only transformer demo project to show how the
LLM works in detail.
It is well documented and commented in the code to let users learn the
details and theories of LLM.

See [task_plan.md](task_plan.md) for the current project plan, progress tracking, and issue status.

## Tooling

- Use `uv` for dependency management and running scripts.
- The `.venv` in the repository is the `uv` virtual environment for this
  project.
- Use `pytest` for testing.
- Use `ruff` for linting, formatting, and reorganizing imports.
- Use `pyright` for type checking.

## Execution

### CLI Commands

All commands run via `uv run`:

```bash
# Train model (default: NumPy backend)
uv run src/train.py train --checkpoint_name my_model

# Advanced training
uv run src/train.py train \
    --embed_dim 64 --layers 4 --heads 8 --experts 8 \
    --max_context 128 --epochs 10 --lr 0.001 \
    --backend numpy  # or "torch"

# Text generation from trained checkpoint
uv run src/train.py infer --checkpoint_name my_model --prompt "the"

# E2E cross-backend validation (4 scenarios)
uv run src/validate_e2e.py
```

### Testing

```bash
# All tests
uv run pytest tests/ -v

# Cross-backend parity tests
uv run pytest tests/test_cross_backend.py -v

# E2E cross-load validation
uv run pytest tests/test_e2e_cross_backend.py -v
```

## Rules and Principles

1. **Quick iteration feedback loop over repetitive thinking** — When debugging, always run the minimal failing test first, capture the actual error, then make a targeted fix. Never spend time reading and re-reading code without running a test to get feedback. Every hypothesis must be validated with a test result, not with more thinking. Prefer:
   - Write/run minimal failing test → observe error → fix → verify pass
   - Over: read code → reason about what might be wrong → guess → read more code
    
2. **Tiered tolerance policy for parity tests** — All parity tests use float64. Acceptable tolerances depend on computational chain depth:
   - **Standalone components** (tested in isolation): `rtol=1e-4, atol=1e-4` — e.g., LayerNorm, FeedForward, MoE, MHA tested independently without gradient chaining through multiple layers.
   - **Component in single chain** (e.g., MHA inside TransformerBlock with single residual): `rtol=1e-3, atol=1e-3` — one level of gradient accumulation.
   - **Full transformer backward chains** (e.g., `blocks.0` params when gradient flows through `lm_head → block.1 → block.0`): `rtol=1e-2, atol=1e-2` — gradient compound through 2+ layers, float64 precision limits accumulate to ~0.001–0.01 drift.

3. All Python code must be free of `pyright` and `ruff` issues. After any
   code change, use `ruff` to reorganize imports and format the code.

4. If any request is unclear or has alternative approaches, confirm with the
   user before making changes.

5. Do not make technical or business assumptions.

6. Follow Python best practices and existing patterns in the repository.
   Prefer short and clean code and avoid unnecessary complexity.

7. All unit tests must have a reasonable timeout using `pytest-timeout` to
   prevent hung tests and ensure performance.

8. All code need be well documented and comment, especially for code related
   to math need intuitive explanations, and for matrix calculation should put
   with comments to indicate the dimension shape of the matrix.

9. All code need to be using strict type hint to indicate the interfaces.

10. All dictionary keys for parameter names (e.g., dict["gamma"], dict["lm_head"]) must use constants from `src/model/parameters.py` — never use raw string literals like `"gamma"`, `"lm_head"`, `"blocks.0.ln1.gamma"`. This prevents typos, ensures consistency across NumPy/PyTorch implementations and tests, and makes refactoring trivial. Common constants: `LayerNorm.NP_GAMMA`, `Transformer.LM_HEAD`, `block_param(0, "ln1", LayerNorm.NP_GAMMA)`, `expert_param(0, 0, Expert.NP_W1)`.

11. When using `edit`, ensure `oldString` is unique and matches the file content exactly, including whitespace and indentation. If `edit` fails, use `write` to overwrite the file.
