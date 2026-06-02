from __future__ import annotations

class ParameterRegistry:
    """
    The 'Rosetta Stone' for the Transformer architecture.
    Maps canonical parameter names (used by the Trainer/Tester) to
    backend-specific names (used by NumPy, PyTorch, Triton, or CUDA).
    """

    def __init__(self):
        # mappings[backend][internal_name] = canonical_name
        self._mappings: dict[str, dict[str, str]] = {}
        # rev_mappings[backend][canonical_name] = internal_name
        self._rev_mappings: dict[str, dict[str, str]] = {}

    def register(self, backend: str, canonical_name: str, internal_name: str) -> None:
        """
        Registers a mapping between a canonical name and a backend-specific name.
        """
        if backend not in self._mappings:
            self._mappings[backend] = {}
            self._rev_mappings[backend] = {}

        self._mappings[backend][internal_name] = canonical_name
        self._rev_mappings[backend][canonical_name] = internal_name

    def clear(self) -> None:
        """Clear all registered mappings."""
        self._mappings = {}
        self._rev_mappings = {}

    def get_canonical_name(self, backend: str, internal_name: str) -> str:
        """
        Translates an internal backend name to a canonical name.
        Returns the internal name if no mapping is found.
        """
        return self._mappings.get(backend, {}).get(internal_name, internal_name)

    def get_internal_name(self, backend: str, canonical_name: str) -> str:
        """
        Translates a canonical name to an internal backend name.
        Returns the canonical name if no mapping is found.
        """
        return self._rev_mappings.get(backend, {}).get(canonical_name, canonical_name)


# Global registry instance
registry = ParameterRegistry()
