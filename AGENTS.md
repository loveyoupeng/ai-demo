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
