# AGENTS.md

This is a demo repository to learn AI concepts and tooling.
It is building a decoder-only transformer demo project to show how the
LLM works in detail.
It is well documented and commented in the code to let users learn the
details and theories of LLM.

## Plan

### Core Objectives

1. **Core Math (NumPy)**: Implement all transformer components (Attention,
   MoE, etc.) using NumPy with manual backward passes for pedagogical
   clarity.
2. **Training Orchestration**: Implement `Trainer`, `Optimizer`, and
   `Loss` functions.
3. **Verification**: Use TDD to ensure mathematical correctness of all layers
   and gradients.
4. **Evaluation**: Implement perplexity and accuracy metrics.
5. **Multi-Level Implementation (Backend Agnostic)**: Implement a pluggable
   architecture to demonstrate the trade-offs between abstraction and
   performance across three levels:
   - **Level 1: PyTorch** (High-level API, standard practitioner tool).
   - **Level 2: Triton** (Kernel-level optimization for compute bottlenecks).
   - **Level 3: CUDA** (Low-level hardware control).
6. **Benchmarking & Profiling**: Implement a specialized profiler to compare
   latency, throughput, and memory usage across all backends.
7. **Educational Synthesis**: Create a "Concept-to-Tool" mapping guide to
   explain how mathematical concepts translate to specific tool primitives.

### Progress

#### Finished

- [x] Project structure and repository initialization.
- [x] Implementation of core transformer components using NumPy.
- [x] Implementation of manual backward passes for NumPy.
- [x] Archiving of failed PyTorch prototype.
- [x] Phase 0: Infrastructure (Canonical Registry & BaseBackend).
- [x] Phase 1: PyTorch Re-implementation (TDD approach)
  - [x] TokenEmbedding (parity test passing ✅ 1/1)
  - [x] LayerNorm (parity test passing ✅ 4/4)
  - [x] FeedForward (parity test passing ✅ 6/6)
  - [x] PositionalEmbedding (parity test passing - PE matrix parity ✅, forward/backward ✅)
  - [x] MultiHeadAttention (parity test passing - forward + all backward params ✅ 7/7)
  - [x] MoELayer (parity test passing - forward + router + expert grads ✅ 7/7)
- [x] 29/29 parity tests passing (TokenEmbedding + LayerNorm + FeedForward + PositionalEmbedding + MultiHeadAttention + MoE).

#### In Progress

- [ ] TransformerBlock (attention + FFN + LayerNorm in composition)
- [ ] Full Transformer

#### Directory Structure

- All test files live in `tests/` at the repo root (NOT in `src/tests/`).
- Parity test files live in `tests/parity/`, not in any nested directory.
- Debug scripts (`debug_*.py`) live in `debug/` for cleanup and removal later.

#### Next Steps

- [ ] TransformerBlock (attention + FFN + LayerNorm in composition).
- [ ] Full Transformer.
- [ ] Training Orchestration: Trainer, Optimizer, Loss functions.
- [ ] Evaluation: perplexity and accuracy metrics.
- [ ] Level 2: Triton Implementation (Kernel-level optimization).
- [ ] Level 3: CUDA Implementation (Low-level hardware control).
- [ ] Benchmarking & Profiling.
- [ ] Final Educational Synthesis: Concept-to-Tool mapping guide.

## Tooling

- Use `uv` for dependency management and running scripts.
- The `.venv` in the repository is the `uv` virtual environment for this
  project.
- Use `pytest` for testing.
- Use `ruff` for linting, formatting, and reorganizing imports.
- Use `pyright` for type checking.

## Execution

- Main entry point: `python src/main.py` or `uv run src/main.py`

## Rules and Principles

1. Use Test Driven Design (TDD) / Behavior Driven Design (BDD) to derive
   clean and high-quality application interfaces via tests. After any code
   change all the test case should pass.
2. All Python code must be free of `pyright` and `ruff` issues. After any
   code change, use `ruff` to reorganize imports and format the code.
3. If any request is unclear or has alternative approaches, confirm with the
   user before making changes.
4. Do not make technical or business assumptions.
5. Follow Python best practices and existing patterns in the repository.
   Prefer short and clean code and avoid unnecessary complexity.
6. All unit tests must have a reasonable timeout using `pytest-timeout` to
   prevent hung tests and ensure performance.
7. All code need be well documented and comment, especially for code related
   to math need intuitive explanations, and for matrix calculation should put
   with comments to indicate the dimension shape of the matrix.
8. All code need to be using strict type hint to indicate the interfaces.
9. When using `edit`, ensure `oldString` is unique and matches the file content exactly, including whitespace and indentation. If `edit` fails, use `write` to overwrite the file.
