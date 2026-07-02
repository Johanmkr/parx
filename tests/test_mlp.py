"""Tests on a randomly initialised [2→5→5→5→1] MLP."""
# juliacall must be imported before torch

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
        nn.Linear(2, 5),
        nn.ReLU(),
        nn.Linear(5, 5),
        nn.ReLU(),
        nn.Linear(5, 5),
        nn.ReLU(),
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
    assert partition.n_layers == 3  # 3 hidden layers; output excluded


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
    X_small = rng.uniform(-1, 1, (20, 2))
    X_large = rng.uniform(-1, 1, (500, 2))

    p_small = compute_partition(model, X_small, method="sparse_julia")
    p_large = compute_partition(model, X_large, method="sparse_julia")

    assert len(p_large) >= len(p_small)


# ── Exact mode ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def exact_555_partition():
    """The [2→5→5→5→1] MLP's exact partition, shared across tests that only

    read from it — avoids re-running a full Julia DFS per test.
    """
    return compute_partition(_mlp555(), np.zeros(2), method="exact_julia")


def test_exact_555_finds_regions(exact_555_partition):
    partition = exact_555_partition

    assert len(partition) > 0
    assert partition.input_dim == 2
    assert partition.n_layers == 3


def test_exact_555_centroids_satisfy_halfspaces(exact_555_partition):
    partition = exact_555_partition

    for region in partition.regions:
        D, g = partition.halfspaces(region)
        slack = g - D @ region.centroid
        assert np.all(slack >= -1e-8), "centroid violates its own halfspace system"


def test_exact_555_regions_nonempty(exact_555_partition):
    """Every exact-mode region must have a strictly positive Chebyshev radius."""
    from parx.verify import check_regions_nonempty

    partition = exact_555_partition
    ok, bad, radii = check_regions_nonempty(partition, min_radius=1e-6)
    assert ok, (
        f"{len(bad)} degenerate region(s); smallest radii: "
        f"{[(int(i), float(radii[i])) for i in bad[:5]]}"
    )
    assert float(radii.min()) > 1e-6, f"smallest radius = {radii.min():.3g}"


def test_exact_555_no_overlaps(exact_555_partition):
    """No two exact-mode regions of the 5,5,5 MLP should share an interior point.

    Combines uniform sampling, boundary-targeted sampling (which stress-tests
    the seams where numerical overlaps would manifest), and a sweep across
    every region's centroid.
    """
    from parx.verify import check_no_overlaps, sample_near_boundaries

    partition = exact_555_partition

    rng = np.random.default_rng(7)
    X_uniform = rng.uniform(-1, 1, (5000, 2))
    X_boundary = sample_near_boundaries(partition, eps=1e-3)
    X_centroid = np.array([r.centroid for r in partition.regions])
    X = np.vstack([X_uniform, X_boundary, X_centroid])

    ok, counts = check_no_overlaps(partition, X)
    if not ok:
        bad_idx = np.where(counts > 1)[0]
        offending = []
        for i in bad_idx[:3]:
            members = [
                j
                for j, r in enumerate(partition.regions)
                if np.all(
                    partition.halfspaces(r)[0] @ X[i]
                    <= partition.halfspaces(r)[1] + 1e-8
                )
            ]
            offending.append((X[i].tolist(), members))
        raise AssertionError(
            f"Overlapping regions in exact partition. "
            f"max membership = {counts.max()}, "
            f"{(counts > 1).sum()} of {len(X)} sampled points overlap "
            f"({len(X_uniform)} uniform, {len(X_boundary)} boundary, "
            f"{len(X_centroid)} centroids). "
            f"Examples (point, region_ids): {offending}"
        )


def test_exact_555_covers_space(exact_555_partition):
    """Every sampled point must belong to exactly one exact region — including
    boundary-targeted samples that stress the seams between adjacent regions."""
    from parx.verify import check_covers_space, sample_near_boundaries

    partition = exact_555_partition

    rng = np.random.default_rng(7)
    X_uniform = rng.uniform(-1, 1, (5000, 2))
    X_boundary = sample_near_boundaries(partition, eps=1e-3)
    X = np.vstack([X_uniform, X_boundary])

    ok, counts = check_covers_space(partition, X)
    if not ok:
        uncovered = X[counts == 0]
        overlap = X[counts > 1]
        raise AssertionError(
            f"Exact partition does not tile the space. "
            f"{len(uncovered)} uncovered (e.g. {uncovered[:3].tolist()}); "
            f"{len(overlap)} overlapping (e.g. {overlap[:3].tolist()}). "
            f"Sample mix: {len(X_uniform)} uniform + {len(X_boundary)} boundary."
        )


def test_exact_555_route_matches_halfspace_membership(exact_555_partition):
    """route(x) must return the same region as the halfspace membership test."""
    from parx.verify import check_routing_consistency

    partition = exact_555_partition
    rng = np.random.default_rng(11)
    X = rng.uniform(-1, 1, (2000, 2))

    ok, bad = check_routing_consistency(partition, X, tol=1e-8)
    assert ok, (
        f"route() disagreed with halfspace membership for {len(bad)} of "
        f"{len(X)} sample points. First offenders: {bad[:5]}"
    )


def test_exact_555_finds_at_least_as_many_as_sparse():
    """Exact mode must find at least as many regions as sparse over the same domain."""
    model = _mlp555()
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, (200, 2))
    x0 = X[0]

    p_sparse = compute_partition(model, X, method="sparse_julia")
    p_exact = compute_partition(model, x0, method="exact_julia")

    assert len(p_exact) >= len(p_sparse), (
        f"exact found {len(p_exact)} regions, sparse found {len(p_sparse)}"
    )
