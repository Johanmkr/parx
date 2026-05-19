"""Smoke tests — Julia bridge, sparse region finder, and compute_partition."""
# juliacall must be imported before torch to avoid a signal-handling conflict.
from parx._julia_init import ensure_julia

import numpy as np
import pytest
import torch
import torch.nn as nn


@pytest.fixture(scope="module", autouse=True)
def _julia(julia_session):
    """Ensure Julia is ready for every test in this module."""


# ── Phase 1: bridge ───────────────────────────────────────────────────────────

def test_julia_loads():
    jl = ensure_julia()
    assert jl is not None


def test_julia_threads():
    jl = ensure_julia()
    n = jl.seval("Threads.nthreads()")
    assert n >= 1


def test_hello():
    jl = ensure_julia()
    msg = jl.LinearRegions.hello()
    assert "LinearRegions.jl" in str(msg)


def test_network_info():
    """Verify numpy weight arrays survive the Python→Julia round-trip."""
    jl = ensure_julia()
    W1 = np.random.randn(4, 2)
    b1 = np.random.randn(4)
    W2 = np.random.randn(3, 4)
    b2 = np.random.randn(3)

    input_dim, n_layers = jl.LinearRegions.network_info([W1, W2], [b1, b2])
    assert int(input_dim) == 2
    assert int(n_layers) == 2


# ── Phase 2: sparse region finder ────────────────────────────────────────────

def test_sparse_regions_quadrants():
    """Identity 2→2 single-layer network must produce exactly 4 quadrant regions."""
    jl = ensure_julia()
    W = np.eye(2)
    b = np.zeros(2)
    X = np.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])

    result = jl.LinearRegions.find_regions_sparse([W], [b], X)
    patterns  = np.array(result[0])
    offsets   = np.array(result[1])
    centroids = np.array(result[2])

    assert patterns.shape  == (4, 2), "4 regions × 2 bits"
    assert list(offsets)   == [0, 2], "1 layer → offsets [0, 2]"
    assert centroids.shape == (4, 2), "4 centroids in 2D"

    # All 4 activation patterns must be distinct
    rows = set(map(tuple, patterns.tolist()))
    assert len(rows) == 4


def test_sparse_regions_duplicate_points():
    """Repeated points from the same region count as one region."""
    jl = ensure_julia()
    W = np.eye(2)
    b = np.zeros(2)
    # 8 points but only 2 distinct regions
    X = np.array([
        [1., 1.], [2., 3.], [0.5, 0.1],   # all q=[1,1]
        [-1., -1.], [-2., -3.], [-0.1, -0.5],  # all q=[0,0]
        [1., 1.],   # duplicate
        [-1., -1.],  # duplicate
    ])

    result = jl.LinearRegions.find_regions_sparse([W], [b], X)
    patterns = np.array(result[0])
    assert patterns.shape[0] == 2


def test_sparse_regions_two_layers():
    """Two-layer network: verify offsets and activation path slicing."""
    jl = ensure_julia()
    W1 = np.eye(2)
    b1 = np.zeros(2)
    W2 = np.array([[1., 1.], [-1., 1.]])
    b2 = np.zeros(2)
    X  = np.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])

    result  = jl.LinearRegions.find_regions_sparse([W1, W2], [b1, b2], X)
    patterns = np.array(result[0])
    offsets  = np.array(result[1])

    assert list(offsets) == [0, 2, 4], "2 layers × 2 neurons → [0, 2, 4]"
    assert patterns.shape[1] == 4, "4 total bits"

    # Layer 0 bits = patterns[:, 0:2], layer 1 bits = patterns[:, 2:4]
    for i in range(patterns.shape[0]):
        l0 = patterns[i, offsets[0]:offsets[1]]
        l1 = patterns[i, offsets[1]:offsets[2]]
        assert len(l0) == 2
        assert len(l1) == 2


# ── Phase 3: exact region finder ─────────────────────────────────────────────

def test_exact_regions_quadrants():
    """Exact mode on identity [2→2] must find all 4 quadrant regions."""
    jl = ensure_julia()
    W  = np.eye(2)
    b  = np.zeros(2)
    x0 = np.array([1.0, 1.0])

    result   = jl.LinearRegions.find_regions_exact([W], [b], x0)
    patterns = np.array(result[0])
    offsets  = np.array(result[1])

    assert patterns.shape[0] == 4, "should find all 4 quadrant regions"
    assert list(offsets) == [0, 2]
    rows = set(map(tuple, patterns.tolist()))
    assert len(rows) == 4, "all 4 patterns must be distinct"


def test_exact_finds_more_than_sparse():
    """Exact mode discovers regions not covered by the data; sparse does not."""
    jl = ensure_julia()
    W  = np.eye(2)
    b  = np.zeros(2)

    # Sparse: only points in the positive quadrant → sees just 1 region.
    X_partial = np.array([[1.0, 1.0], [2.0, 0.5]])
    n_sparse  = np.array(jl.LinearRegions.find_regions_sparse([W], [b], X_partial)[0]).shape[0]

    # Exact: starting from the same region, traverses all neighbours.
    x0      = np.array([1.0, 1.0])
    n_exact = np.array(jl.LinearRegions.find_regions_exact([W], [b], x0)[0]).shape[0]

    assert n_sparse == 1
    assert n_exact  == 4


def test_exact_two_layers():
    """Two-layer identity network: exact mode returns correct number of regions."""
    jl = ensure_julia()
    W1 = np.eye(2)
    b1 = np.zeros(2)
    W2 = np.eye(2)
    b2 = np.zeros(2)
    x0 = np.array([1.0, 1.0])

    result   = jl.LinearRegions.find_regions_exact([W1, W2], [b1, b2], x0)
    patterns = np.array(result[0])
    offsets  = np.array(result[1])

    assert list(offsets) == [0, 2, 4]
    # With identity weights, layer 2 activation is determined by layer 1, so still 4 regions.
    assert patterns.shape[0] == 4


def test_exact_centroids_satisfy_halfspaces():
    """Chebyshev centres returned by exact mode must satisfy their halfspace systems."""
    from parx import compute_partition

    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.ReLU(), nn.Linear(2, 1))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))

    x0        = np.array([1.0, 1.0])
    partition = compute_partition(model, x0, mode="exact")

    assert len(partition) == 4
    for region in partition.regions:
        D, g  = partition.halfspaces(region)
        slack = g - D @ region.centroid
        assert np.all(slack >= -1e-8), "centroid must satisfy all halfspaces"


# ── Phase 6: compute_partition ────────────────────────────────────────────────

def test_compute_partition_sparse():
    """End-to-end: compute_partition returns a usable Partition."""
    from parx import compute_partition

    model = nn.Sequential(
        nn.Linear(2, 2, bias=False),
        nn.ReLU(),
        nn.Linear(2, 1),
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))

    X = np.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
    partition = compute_partition(model, X, mode="sparse")

    assert len(partition) == 4
    assert partition.input_dim == 2
    assert partition.n_layers  == 1   # output layer excluded

    # Each point should route back to a known region
    routed = partition.route(X)
    assert all(r is not None for r in routed)


def test_compute_partition_halfspaces():
    """Halfspaces reconstructed for each region must contain its centroid."""
    from parx import compute_partition

    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.ReLU(), nn.Linear(2, 1))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))

    X = np.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
    partition = compute_partition(model, X)

    for region in partition.regions:
        D, g = partition.halfspaces(region)
        slack = g - D @ region.centroid
        assert np.all(slack >= -1e-9), "centroid must satisfy D*x <= g"
