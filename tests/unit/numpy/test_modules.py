"""T1: Import the impl.numpy.modules package."""

import impl.numpy.modules as modules


def test_import_modules_succeeds():
    """Verify the impl.numpy.modules package is importable."""
    assert hasattr(modules, "Embedding")


def test_import_utils_succeeds():
    """Verify the impl.numpy.utils package is importable."""
    import impl.numpy.utils

    assert hasattr(impl.numpy.utils, "initialize_linear")
