"""Test that the impl._np modules package exists."""

import impl._np.modules as modules


def test_import_modules_succeeds():
    """Verify the impl._np.modules package is importable."""
    assert hasattr(modules, "Embedding")


def test_import_utils_succeeds():
    """Verify the impl._np.utils package is importable."""
    import impl._np.utils

    assert hasattr(impl._np.utils, "initialize_linear")
