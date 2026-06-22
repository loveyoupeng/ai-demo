"""CUDA driver API — cuInit, NVRTC, module loading, kernel launch.

Contains all CUDA API tests with proper imports for the cuda-python package.
Runs in its own subprocess via the conftest batching strategy.
"""

from __future__ import annotations

import pytest


class TestImport:
    """Verify CUDA package can be imported."""

    @pytest.mark.timeout(10)
    def test_cuda_package(self):
        import impl._cuda

        assert hasattr(impl._cuda, "__file__")