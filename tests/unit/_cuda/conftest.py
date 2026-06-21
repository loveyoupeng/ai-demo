"""Run CUDA tests in per-test subprocesses to prevent NVRTC context pollution.

On Jetson (JetPack 6.2.2, CUDA 12.6), NVRTC driver state accumulates within
a process as kernels are compiled, causing INVALID_HANDLE errors.

Solution: the parent conftest spawns one subprocess per TEST FILE. Each of
those subprocesses, instead of running tests directly, spawns one subprocess
PER TEST using the _run_single_test.py helper. This gives complete per-test
isolation while keeping the pytest infrastructure intact.
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


def _spawn_test_subprocess(test_id: str, cache_uid: str) -> int:
    """Spawn a subprocess that runs a single test with a fresh CUDA context.

    Parameters
    ----------
    test_id : str
        Full test identifier, e.g. 'tests/unit/_cuda/test_block.py::test_shape'
    cache_uid : str
        Unique cache directory suffix to prevent NVRTC cross-talk

    Returns
    -------
    int
        subprocess return code (0 = all tests passed)
    """
    cmd = [
        PYTHON, "-m", "pytest",
        test_id,
        "-q",
        "--timeout=120",
        "--tb=short",
        "-p", "no:cacheprovider",
    ]
    env = _clean_env()
    env["CUDA_CACHE_DISABLE"] = "1"
    env["CUDA_CACHE_PATH"] = str(Path(f"/tmp/.cuda_test_cache_{cache_uid}") / "__pid__")
    return subprocess.run(cmd, env=env).returncode


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtestloop(session):  # noqa: ANN001, D103
    """Run each test in its own subprocess for complete CUDA isolation."""
    if _IN_SUBPROCESS:
        yield
        return

    test_ids: list[str] = []
    for test_file in CUDA_DIR.glob("test_*.py"):
        _collect_test_ids(str(test_file), test_ids)

    if not test_ids:
        return

    exit_code = 0
    uid = uuid.uuid4().hex[:8]
    for i, test_id in enumerate(test_ids, 1):
        short = test_id.split("::")[-1]
        test_uid = f"{uid}_{i:04d}"
        print(f"  [cuda] ({i:>3}/{len(test_ids)}) {short}", file=sys.stderr, flush=True)
        rc = _spawn_test_subprocess(test_id, test_uid)
        if rc != 0:
            exit_code = rc

    sys.exit(exit_code)


def _collect_test_ids(test_file: str, test_ids: list[str]) -> None:
    """Collect one test ID per line from a test file's collection output."""
    env = _clean_env()
    env["CUDA_CACHE_DISABLE"] = "1"
    env["CUDA_CACHE_PATH"] = str(Path(f"/tmp/.cuda_test_cache_collect_{uuid.uuid4().hex[:8]}") / "__pid__")
    cmd = [PYTHON, "-m", "pytest", test_file, "--collect-only", "-q", "--tb=no"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if "::test_" in line or (line.split() and "::test_" in line.split()[-1]):
            test_ids.append(line)