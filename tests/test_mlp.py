"""Tests on a randomly initialised [2→5→5→5→1] MLP."""
# juliacall must be imported before torch
from parx._julia_init import ensure_julia

import numpy as np
import pytest
import torch
import torch.nn as nn

from parx import compute_partition


@pytest.fixture(scope="module", autouse=True)
def _julia(julia_session):
    """Ensure Julia is ready for every test in this module."""


def _mlp555(seed: int = 42) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(2, 5), nn.ReLU(),
        nn.Linear(5, 5), nn.ReLU(),
        nn.Linear(5, 5), nn.ReLU(),
        nn.Linear(5, 1),
    )


# ── Sparse mode ───────────────────────────────────────────────────────────────

def test_sparse_555_finds_regions():
    model = _mlp555()
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (300, 2))

    partition = compute_partition(model, X, method="sparse_julia")

    assert len(partition) > 0
    assert partition.input_dim == 2
    assert partition.n_layers == 3   # 3 hidden layers; output excluded


def test_sparse_555_centroids_satisfy_halfspaces():
    model = _mlp555()
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (300, 2))

    partition = compute_partition(model, X, method="sparse_julia")

    for region in partition.regions:
        D, g = partition.halfspaces(region)
        slack = g - D @ region.centroid
        assert np.all(slack >= -1e-8), "centroid violates its own halfspace system"


def test_sparse_555_route_recovers_all_data():
    """Every data point used to build the partition routes back to a known region."""
    model = _mlp555()
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (100, 2))

    partition = compute_partition(model, X, method="sparse_julia")
    routed = partition.route(X)

    assert all(r is not None for r in routed), "some data points route to None"


def test_sparse_555_more_data_more_regions():
    """Passing more data can only find the same or more regions."""
    model = _mlp555()
    rng = np.random.default_rng(2)
    X_small = rng.uniform(-1, 1, (20,  2))
    X_large = rng.uniform(-1, 1, (500, 2))

    p_small = compute_partition(model, X_small, method="sparse_julia")
    p_large = compute_partition(model, X_large, method="sparse_julia")

    assert len(p_large) >= len(p_small)


# ── Exact mode ────────────────────────────────────────────────────────────────

def test_exact_555_finds_regions():
    model = _mlp555()
    x0 = np.zeros(2)

    partition = compute_partition(model, x0, method="exact_julia")

    assert len(partition) > 0
    assert partition.input_dim == 2
    assert partition.n_layers == 3


def test_exact_555_centroids_satisfy_halfspaces():
    model = _mlp555()
    x0 = np.zeros(2)

    partition = compute_partition(model, x0, method="exact_julia")

    for region in partition.regions:
        D, g = partition.halfspaces(region)
        slack = g - D @ region.centroid
        assert np.all(slack >= -1e-8), "centroid violates its own halfspace system"


def test_exact_555_no_overlaps():
    """No two exact-mode regions of the 5,5,5 MLP should share an interior point."""
    from parx.verify import check_no_overlaps

    model = _mlp555()
    partition = compute_partition(model, np.zeros(2), method="exact_julia")

    rng = np.random.default_rng(7)
    X = rng.uniform(-1, 1, (1000, 2))
    ok, counts = check_no_overlaps(partition, X)
    assert ok, (
        f"Overlapping regions in exact partition — "
        f"max membership count: {counts.max()}, "
        f"affected points: {(counts > 1).sum()}"
    )


def test_exact_555_covers_space():
    """Every point in the sample domain must belong to exactly one exact region."""
    from parx.verify import check_covers_space

    model = _mlp555()
    partition = compute_partition(model, np.zeros(2), method="exact_julia")

    rng = np.random.default_rng(7)
    X = rng.uniform(-1, 1, (1000, 2))
    ok, counts = check_covers_space(partition, X)
    assert ok, (
        f"Exact partition does not tile the space — "
        f"{(counts == 0).sum()} uncovered points, "
        f"{(counts > 1).sum()} overlapping points"
    )


def test_exact_555_finds_at_least_as_many_as_sparse():
    """Exact mode must find at least as many regions as sparse over the same domain."""
    model = _mlp555()
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, (200, 2))
    x0 = X[0]

    p_sparse = compute_partition(model, X,  method="sparse_julia")
    p_exact  = compute_partition(model, x0, method="exact_julia")

    assert len(p_exact) >= len(p_sparse), (
        f"exact found {len(p_exact)} regions, sparse found {len(p_sparse)}"
    )
