import pytest


class TestImport:
    @pytest.mark.timeout(10)
    def test_cuda_package(self):
        import impl._cuda

        assert hasattr(impl._cuda, "__file__")
