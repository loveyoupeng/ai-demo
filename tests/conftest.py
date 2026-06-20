"""Fixtures and hooks for GPU test isolation."""

from contextlib import contextmanager

import pytest
import torch

_cuda_available = None


def _is_cuda_available():
    """Check if CUDA is available (cached for performance)."""
    global _cuda_available
    if _cuda_available is None:
        _cuda_available = torch.cuda.is_available()
    return _cuda_available


@contextmanager
def cuda_isolated():
    """Context manager to isolate GPU state between operations.

    Usage:
        with cuda_isolated():
            # Do GPU work here
            pass
    """
    if _is_cuda_available():
        try:
            yield
        finally:
            # Clean up after GPU work
            import gc
            torch.cuda.empty_cache()
            gc.collect()
    else:
        yield
        return  # No-op for CPU-only systems


def make_isolation_fixture():
    """Create a fixture that cleans CUDA state between tests."""
    @pytest.fixture
    def gpu_isolation(request):
        if _is_cuda_available():
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            yield
            torch.cuda.empty_cache()
        else:
            yield
    return gpu_isolation


# Create and expose the fixture
gpu_isolation = make_isolation_fixture()
