"""Tests for the parx.methods registry and cross-method equivalence."""
# juliacall must be imported before torch

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
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
    )
    x0 = np.zeros(2)

    keys_j = _region_keys(compute_partition(model, x0, method="exact_julia"))
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
    biases = [np.zeros(2, dtype=np.float64)]
    res = fn(weights, biases, X)

    assert isinstance(res, RegionFindResult)
    assert res.patterns.ndim == 2
    assert res.offsets.shape == (2,)  # n_layers + 1
    assert res.centroids.shape == (res.patterns.shape[0], 2)


# ── Phase 8: exact metadata (active indices, bounded flag) ───────────────────


def test_exact_julia_has_active_indices():
    """Exact methods must populate active_indices for every region.

    The active_indices identify non-redundant constraints in the halfspace
    system for each region.
    """
    model = _identity_2x2()
    x0 = np.array([1.0, 1.0])
    partition = compute_partition(model, x0, method="exact_julia_fast")

    assert len(partition) == 4, "identity network has 4 quadrant regions"
    for region in partition.regions:
        assert region.active_indices is not None, "exact method must set active_indices"
        assert len(region.active_indices) > 0, "every region must have active idx"
        assert region.active_indices.dtype == np.int32, "active_indices must be int32"


def test_exact_julia_bounded_flag():
    """Exact methods must populate bounded flag for every region.

    The identity 2x2 network creates 4 quadrant regions in R^2; all four
    quadrants extend to infinity so every region is unbounded.
    """
    model = _identity_2x2()
    x0 = np.array([1.0, 1.0])
    result = get_method("exact_julia_fast")(
        [model[0].weight.detach().numpy().astype(np.float64)],
        [np.zeros(2, dtype=np.float64)],
        x0,
    )

    assert result.bounded is not None, "exact method must populate bounded"
    assert result.bounded.dtype == bool, "bounded must be bool array"
    assert len(result.bounded) == result.patterns.shape[0], "one flag per region"
    assert not np.any(result.bounded), "all quadrant regions are unbounded"


def test_active_only_halfspaces():
    """Partition.halfspaces(region, active_only=True) must index into full rows.

    active_only=True must return exactly the rows of the full system indexed
    by region.active_indices.  The identity 2x2 network has 2 constraints per
    region (all active), so no row-count reduction is expected; we test the
    indexing contract instead.
    """
    model = _identity_2x2()
    x0 = np.array([1.0, 1.0])
    partition = compute_partition(model, x0, method="exact_julia_fast")

    for region in partition.regions:
        D_full, g_full = partition.halfspaces(region, active_only=False)
        D_active, g_active = partition.halfspaces(region, active_only=True)

        assert D_active.shape[1] == D_full.shape[1], "width unchanged"
        assert len(g_active) == D_active.shape[0], "g rows match D rows"
        assert D_active.shape[0] <= D_full.shape[0], "active ≤ full row count"
        assert np.allclose(D_active, D_full[region.active_indices]), \
            "active rows must match D_full[active_indices]"
        assert np.allclose(g_active, g_full[region.active_indices]), \
            "active g must match g_full[active_indices]"


def test_exact_parity_bounded():
    """Bounded flag must agree between exact_python and exact_julia_fast.

    Both methods use the same definition (Chebyshev radius < 1e3), so regions
    with the same activation path should agree on boundedness.
    """
    model = _identity_2x2()
    x0 = np.array([1.0, 1.0])

    partition_py = compute_partition(model, x0, method="exact_python")
    partition_jf = compute_partition(model, x0, method="exact_julia_fast")

    bounded_py = {}
    for region in partition_py.regions:
        key = tuple(q.tobytes() for q in region.activation_path)
        bounded_py[key] = region.bounded

    bounded_jf = {}
    for region in partition_jf.regions:
        key = tuple(q.tobytes() for q in region.activation_path)
        bounded_jf[key] = region.bounded

    assert set(bounded_py.keys()) == set(bounded_jf.keys()), "same regions"
    for key in bounded_py:
        assert bounded_py[key] == bounded_jf[key], (
            f"bounded mismatch for region {key}: "
            f"exact_python={bounded_py[key]}, exact_julia_fast={bounded_jf[key]}"
        )
