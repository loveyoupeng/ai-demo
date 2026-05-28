# AGENTS.md

This is a demo repository to learn AI concepts and tooling.

## Tooling

- Use `uv` for dependency management and running scripts.
- The `.venv` in the repository is the `uv` virtual environment for this project.
- Use `pytest` for testing.
- Use `ruff` for linting, formatting, and reorganizing imports.
- Use `pyright` for type checking.

## Execution

- Main entry point: `python src/main.py` or `uv run src/main.py`

## Rules and Principles

1. Use Test Driven Design (TDD) / Behavior Driven Design (BDD) to derive clean
   and high-quality application interfaces via tests.
2. All Python code must be free of `pyright` and `ruff` issues. After any code
   change, use `ruff` to reorganize imports and format the code.
3. If any request is unclear or has alternative approaches, confirm with the
   user before making changes.
4. Do not make technical or business assumptions.
5. Follow Python best practices and existing patterns in the repository.
