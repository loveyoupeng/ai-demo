# Copyright (c) 2026 loveyoupeng <loveyoupeng@gmail.com>. MIT License — see LICENSE.
"""Constructor-injection contract: a model is BORN with its weights.

The repository's load convention is dependency injection through the
constructor: ``Model(config, params)`` (or ``Model.from_checkpoint(dir)``)
threads the checkpoint arrays down the chain — embedding → stack → block →
RMSNorm / attention / FFN / MoE — so every operator is born with its trained
array, while ``Model(config)`` alone is the fresh seed-initialized training
start (RMSNorm gammas = ones).

These tests pin the two observable consequences:

1. **Completeness** — after construction with ``params``, every checkpoint key
   is the live model storage (nothing was silently left at its default).
2. **Equivalence** — injection at construction is the same model as the
   explicit in-place load path (``Model(config)`` + ``load_from_numpy_dict``),
   so the two entry points cannot drift apart.

CPU tracks only (NumPy, PyTorch); the GPU tracks' parity suites cover the
same contract on device.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.config import TransformerConfig

_DENSE = TransformerConfig(vocab_size=64, embed_dim=32, n_layers=3, n_heads=4, context_length=16, seed=7)
_GQA = TransformerConfig(vocab_size=64, embed_dim=32, n_layers=2, n_heads=4, n_groups=2, context_length=16, seed=7)
_MOE = TransformerConfig(
    vocab_size=64,
    embed_dim=32,
    n_layers=2,
    n_heads=4,
    n_experts=3,
    expert_dim=16,
    top_k=1,
    n_shared_experts=2,
    context_length=16,
    seed=7,
)
_CONFIGS = {"dense": _DENSE, "gqa": _GQA, "moe": _MOE}


def _perturbed(params: dict[str, np.ndarray], seed: int = 0) -> dict[str, np.ndarray]:
    """A copy of ``params`` nudged off its init values (so injection is observable)."""
    rng = np.random.default_rng(seed)
    return {k: (v + rng.normal(0.0, 0.01, v.shape)).astype(np.float32) for k, v in params.items()}


@pytest.mark.parametrize("name", list(_CONFIGS))
def test_numpy_injection_matches_explicit_load(name: str) -> None:
    """NumPyModel(cfg, params) ≡ NumPyModel(cfg) + load_from_numpy_dict(params)."""
    from impl._np.model import NumPyModel

    cfg = _CONFIGS[name]
    params = _perturbed(NumPyModel(cfg).get_all_parameters())

    injected = NumPyModel(cfg, params)
    explicit = NumPyModel(cfg)
    explicit.load_from_numpy_dict(params)

    # 1. completeness: injection reached every parameter
    got = injected.get_all_parameters()
    assert set(got) == set(params)
    for key, value in params.items():
        assert np.array_equal(got[key], value), f"injection missed {key}"

    # 2. equivalence: same forward result as the explicit load path
    tokens = np.random.default_rng(1).integers(0, cfg.vocab_size, (1, 6)).astype(np.int32)
    assert np.array_equal(injected.forward(tokens), explicit.forward(tokens))


@pytest.mark.parametrize("name", list(_CONFIGS))
def test_torch_injection_matches_explicit_load(name: str) -> None:
    """TorchModel(cfg, params) ≡ TorchModel(cfg) + load_from_numpy_dict(params)."""
    torch = pytest.importorskip("torch")
    from impl._torch.layers import TorchModel

    cfg = _CONFIGS[name]
    params = _perturbed(TorchModel(cfg).save_as_numpy())

    injected = TorchModel(cfg, params)
    explicit = TorchModel(cfg)
    explicit.load_from_numpy_dict(params)

    # 1. completeness — save_as_numpy returns the checkpoint layout, so the
    #    comparison is apples-to-apples (the nn.Linear transpose rule cancels).
    got = injected.save_as_numpy()
    assert set(got) == set(params)
    for key, value in params.items():
        assert np.array_equal(got[key], value), f"injection missed {key}"

    # 2. equivalence: same logits as the explicit load path
    tokens = torch.randint(0, cfg.vocab_size, (1, 6), dtype=torch.int64)
    injected.eval()
    explicit.eval()
    with torch.no_grad():
        assert torch.equal(injected(tokens), explicit(tokens))


def test_numpy_fresh_model_has_identity_gains() -> None:
    """The training start point: gammas are ones (identity) when nothing is injected."""
    from impl._np.model import NumPyModel

    model = NumPyModel(_DENSE)
    assert np.all(model.final_norm.gamma == 1.0)
    assert all(np.all(block.input_layernorm.gamma == 1.0) for block in model.stack.layers)


def test_numpy_mismatched_injection_fails_fast() -> None:
    """A stale/foreign checkpoint is rejected before any operator is built."""
    from impl._np.model import NumPyModel

    params = NumPyModel(_DENSE).get_all_parameters()
    with pytest.raises(ValueError, match="registry"):
        NumPyModel(_MOE, params)  # wrong topology → stale keys + missing keys
