"""Tests for the parx.methods registry and cross-method equivalence."""
# juliacall must be imported before torch
from parx._julia_init import ensure_julia

import numpy as np
import pytest
import torch
import torch.nn as nn

from parx import compute_partition, list_methods
from parx.methods import RegionFindResult, get_method, register_method


@pytest.fixture(scope="module", autouse=True)
def _julia(julia_session):
    """Ensure Julia is ready for every test in this module."""


# ── Registry plumbing ─────────────────────────────────────────────────────────

def test_list_methods_contains_builtins():
    names = list_methods()
    for expected in (
        "sparse_julia",
        "sparse_python",
        "exact_julia",
        "exact_julia_fast",
        "exact_python",
    ):
        assert expected in names


def test_get_method_unknown_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        get_method("does_not_exist")


def test_register_duplicate_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_method("sparse_julia")(lambda *_: None)


# ── Cross-method equivalence on a small MLP ──────────────────────────────────

def _identity_2x2():
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.ReLU(), nn.Linear(2, 1))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))
    return model


def _region_keys(partition):
    return {tuple(b.tobytes() for b in r.activation_path) for r in partition.regions}


def test_sparse_julia_matches_sparse_python():
    model = _identity_2x2()
    X = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    pj = compute_partition(model, X, method="sparse_julia")
    pp = compute_partition(model, X, method="sparse_python")
    assert _region_keys(pj) == _region_keys(pp)


def test_exact_methods_agree_on_identity_network():
    model = _identity_2x2()
    x0 = np.array([1.0, 1.0])

    keys = {
        m: _region_keys(compute_partition(model, x0, method=m))
        for m in ("exact_julia", "exact_julia_fast", "exact_python")
    }
    assert keys["exact_julia"] == keys["exact_julia_fast"] == keys["exact_python"]
    assert len(keys["exact_julia"]) == 4


def test_exact_methods_agree_on_random_mlp():
    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Linear(2, 4), nn.ReLU(),
        nn.Linear(4, 4), nn.ReLU(),
        nn.Linear(4, 1),
    )
    x0 = np.zeros(2)

    keys_j  = _region_keys(compute_partition(model, x0, method="exact_julia"))
    keys_jf = _region_keys(compute_partition(model, x0, method="exact_julia_fast"))
    keys_py = _region_keys(compute_partition(model, x0, method="exact_python"))
    assert keys_j == keys_jf == keys_py


# ── state_dict input path ────────────────────────────────────────────────────

def test_compute_partition_accepts_state_dict():
    """compute_partition takes a single state_dict (the canonical input)."""
    model = _identity_2x2()
    sd = model.state_dict()
    X = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    p = compute_partition(sd, X, method="sparse_julia")
    assert len(p) == 4


# ── RegionFindResult shape contract ──────────────────────────────────────────

def test_region_find_result_shapes():
    model = _identity_2x2()
    X = np.array([[1.0, 1.0], [-1.0, -1.0]])
    fn = get_method("sparse_python")
    weights = [model[0].weight.detach().numpy().astype(np.float64)]
    biases  = [np.zeros(2, dtype=np.float64)]
    res = fn(weights, biases, X)

    assert isinstance(res, RegionFindResult)
    assert res.patterns.ndim == 2
    assert res.offsets.shape == (2,)            # n_layers + 1
    assert res.centroids.shape == (res.patterns.shape[0], 2)
