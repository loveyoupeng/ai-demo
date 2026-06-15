class TestPackageImport:
    """Verify the PyTorch package structure is importable.

    Gate: C0.2 — after creating the package, the first test verifies
    that the TorchModel and ModelConfig can be imported from impl._torch.
    These will be stub implementations initially, developed in C1–C7.
    """

    def test_import_torch_package(self) -> None:
        """The impl._torch package must exist."""
        from impl import _torch

        assert hasattr(_torch, "__file__")
        assert "_torch" in _torch.__file__

    def test_import_torch_model_and_config(self) -> None:
        """TorchModel and ModelConfig must be importable from impl._torch.

        These are the main public API entries — models are built with
        ModelConfig and instantiated as TorchModel. See C7 for full impl.
        """
        from impl._torch import ModelConfig, TorchModel

        assert TorchModel is not None
        assert ModelConfig is not None
