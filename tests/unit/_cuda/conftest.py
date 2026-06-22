"""Run CUDA tests in batched subprocesses to prevent NVRTC context pollution.

On Jetson (JetPack 6.2.2, CUDA 12.6), NVRTC driver state accumulates within
a process as kernels are compiled, causing INVALID_HANDLE errors.

Solution: the parent conftest spawns ONE subprocess per test FILE. Each
subprocess runs ALL tests in that file at once. With merged test files,
this yields a single subprocess for the entire suite (~71 tests in one run).
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

CUDA_DIR = Path(__file__).parent
PYTHON = sys.executable
_ENV_KEY = "CUDA_TESTS_IN_SUBPROCESS"
_IN_SUBPROCESS = os.environ.get(_ENV_KEY, "0") == "1"

# Prevent pytest from importing test modules in the parent process.
# test files import impl._cuda.* which initializes CUDA/NVRTC in the parent,
# and that state leaks into child subprocesses via fork. The parent must only
# orchestrate subprocess runs — never importing CUDA modules itself.
# Only apply collect_ignore_glob in the parent (not in child subprocesses).
if not _IN_SUBPROCESS:
    collect_ignore_glob = ["test_*.py"]  # noqa: ANN201

# Vars inherited from `uv run` that interfere with subprocess CUDA state
# on Jetson Tegra. Strip these before spawning child pytest processes.
_ENV_BLACKLIST = frozenset({"UV", "VIRTUAL_ENV", "VIRTUAL_ENV_PROMPT", "UV_RUN"})


def _clean_env() -> dict[str, str]:
    """Return a copy of os.environ with uv/virtualenv vars stripped, plus CUDA cache isolation env vars."""
    env = {k: v for k, v in os.environ.items() if not any(k.startswith(ns) for ns in _ENV_BLACKLIST)}
    env[_ENV_KEY] = "1"  # Children run tests normally (no recursion into pytest_runtestloop)
    return env


def _spawn_file_subprocess(test_file: Path, batch_id: str) -> int:
    """Spawn a subprocess that runs ALL tests in a single file with a fresh CUDA context.

    Parameters
    ----------
    test_file : Path
        Path to the test file, e.g. Path("tests/unit/_cuda/test_all_cuda.py")
    batch_id : str
        Unique batch ID for cache directory isolation

    Returns
    -------
    int
        subprocess return code (0 = all tests passed)
    """
    cmd = [
        PYTHON, "-m", "pytest",
        str(test_file),
        "-q",
        "--timeout=120",
        "--tb=short",
        "-p", "no:cacheprovider",
    ]
    env = _clean_env()
    env["CUDA_CACHE_DISABLE"] = "1"
    env["CUDA_CACHE_PATH"] = str(Path(f"/tmp/.cuda_test_cache_{batch_id}") / "__pid__")
    return subprocess.run(cmd, env=env).returncode


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtestloop(session):  # noqa: ANN001, D103
    """Run each test FILE in its own subprocess for CUDA isolation."""
    if _IN_SUBPROCESS:
        yield
        return

    test_files: list[Path] = sorted(CUDA_DIR.glob("test_*.py"))

    if not test_files:
        return

    exit_code = 0
    uid = uuid.uuid4().hex[:8]
    for i, test_file in enumerate(test_files, 1):
        short = test_file.name.removesuffix(".py")
        batch_id = f"{uid}_{i:03d}"
        print(f"  [cuda] ({i:>3}/{len(test_files)}) {short}", file=sys.stderr, flush=True)
        rc = _spawn_file_subprocess(test_file, batch_id)
        if rc != 0:
            exit_code = rc

    os._exit(exit_code)
