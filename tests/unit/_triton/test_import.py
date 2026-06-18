import pytest


class TestImport:
    @pytest.mark.timeout(10)
    def test_triton_package(self):
        """Verify impl._triton package exists and is importable."""
        import impl._triton
        assert hasattr(impl._triton, "__file__")
