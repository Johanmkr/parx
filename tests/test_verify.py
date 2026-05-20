"""Tests for parx.verify — both the existing sample-based checks and the
LP-based ones (regions_nonempty, routing_consistency, sample_near_boundaries).

A few tests build *broken* partitions on purpose to confirm the new checks
actually fail when something is wrong (a check that never fails is no check
at all).  These tests stay in pure Python — no Julia required.
"""

from __future__ import annotations

import numpy as np
import pytest

from parx.partition import Partition
from parx.region import Region
from parx.verify import (
    check_covers_space,
    check_no_overlaps,
    check_regions_nonempty,
    check_routing_consistency,
    region_chebyshev_radii,
    sample_near_boundaries,
)


# ── Fixtures: hand-built partitions ──────────────────────────────────────────

def _quadrant_partition() -> Partition:
    """Healthy 4-quadrant partition (single identity layer, no bias)."""
    W = np.eye(2)
    b = np.zeros(2)
    regions = [
        Region([np.array([True,  True])],  centroid=np.array([ 1.0,  1.0])),
        Region([np.array([True,  False])], centroid=np.array([ 1.0, -1.0])),
        Region([np.array([False, True])],  centroid=np.array([-1.0,  1.0])),
        Region([np.array([False, False])], centroid=np.array([-1.0, -1.0])),
    ]
    return Partition(regions=regions, weights=[W], biases=[b])


def _partition_with_empty_region() -> Partition:
    """A partition that includes an inconsistent activation pattern.

    Hidden layer is x_1 ≥ 0 AND x_1 < 0 (q = [True, False] with weight = [[1,0],[1,0]]),
    which is infeasible — the region is empty.  Used to verify
    ``check_regions_nonempty`` catches it.
    """
    W = np.array([[1.0, 0.0], [1.0, 0.0]])  # both neurons see same input feature
    b = np.zeros(2)
    # q=[True, False] requires x_1 > 0 AND x_1 ≤ 0 → empty.
    bad = Region([np.array([True, False])], centroid=np.array([0.5, 0.0]))
    ok  = Region([np.array([True, True])],  centroid=np.array([1.0, 0.0]))
    return Partition(regions=[ok, bad], weights=[W], biases=[b])


# ── region_chebyshev_radii ───────────────────────────────────────────────────

def test_radii_strictly_positive_for_healthy_partition():
    p = _quadrant_partition()
    radii = region_chebyshev_radii(p)
    assert radii.shape == (4,)
    assert np.all(radii > 0)


def test_empty_region_gets_zero_radius():
    p = _partition_with_empty_region()
    radii = region_chebyshev_radii(p)
    # ok region is unbounded (no constraints binding from above) → near max_radius
    # bad region (infeasible) → 0.
    assert radii[1] == 0.0
    assert radii[0] > 0.0


# ── check_regions_nonempty ───────────────────────────────────────────────────

def test_nonempty_passes_for_quadrants():
    p = _quadrant_partition()
    ok, bad, radii = check_regions_nonempty(p, min_radius=1e-6)
    assert ok
    assert len(bad) == 0
    assert (radii > 0).all()


def test_nonempty_catches_degenerate_region():
    p = _partition_with_empty_region()
    ok, bad, radii = check_regions_nonempty(p, min_radius=1e-6)
    assert not ok, "empty region should fail the nonempty check"
    assert list(bad) == [1]
    assert radii[1] == 0.0


# ── check_routing_consistency ────────────────────────────────────────────────

def test_routing_consistent_on_quadrants():
    p = _quadrant_partition()
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, (200, 2))
    ok, bad = check_routing_consistency(p, X)
    assert ok, f"unexpected routing failures: {bad[:5]}"


def test_routing_flags_points_not_in_partition():
    """If a sample point has no matching activation pattern, route returns None
    and the check should flag it."""
    W = np.eye(2)
    b = np.zeros(2)
    only = Region([np.array([True, True])], centroid=np.array([1.0, 1.0]))
    p = Partition(regions=[only], weights=[W], biases=[b])

    X = np.array([[1.0, 1.0], [-1.0, -1.0]])
    ok, bad = check_routing_consistency(p, X)
    assert not ok
    assert bad == [1]


# ── sample_near_boundaries ───────────────────────────────────────────────────

def test_boundary_samples_are_in_their_region():
    """Every boundary sample is generated to sit just inside its source region."""
    p = _quadrant_partition()
    X = sample_near_boundaries(p, eps=1e-3)
    assert X.ndim == 2 and X.shape[1] == 2
    # Each region contributes one inside-sample per halfspace row (2 rows).
    assert len(X) == 4 * 2

    ok, counts = check_covers_space(p, X)
    assert ok, "boundary samples should each lie inside exactly one region"
    assert (counts == 1).all()


def test_boundary_samples_empty_when_partition_has_no_constraints():
    """An empty-path region has no halfspaces → no boundary samples to take."""
    W = np.eye(2)
    b = np.zeros(2)
    r = Region(activation_path=[], centroid=np.zeros(2))
    p = Partition(regions=[r], weights=[W], biases=[b])
    X = sample_near_boundaries(p)
    assert X.shape == (0, 2)


# ── Sample-based checks (existing functions) ─────────────────────────────────

def test_overlap_detection_on_constructed_overlap():
    """Two identical regions should overlap on every sample point."""
    W = np.eye(2)
    b = np.zeros(2)
    r = Region([np.array([True, True])], centroid=np.array([1.0, 1.0]))
    # Same region added twice → guaranteed overlap on positive quadrant.
    p = Partition(regions=[r, r], weights=[W], biases=[b])

    X = np.array([[1.0, 1.0], [2.0, 3.0]])
    ok, counts = check_no_overlaps(p, X)
    assert not ok
    assert (counts == 2).all()


def test_coverage_gap_detection():
    """Missing one quadrant must produce uncovered samples in that quadrant."""
    W = np.eye(2)
    b = np.zeros(2)
    regions = [
        Region([np.array([True,  True])],  centroid=np.array([ 1.0,  1.0])),
        Region([np.array([True,  False])], centroid=np.array([ 1.0, -1.0])),
        Region([np.array([False, True])],  centroid=np.array([-1.0,  1.0])),
        # Intentionally drop the [False, False] region.
    ]
    p = Partition(regions=regions, weights=[W], biases=[b])

    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (500, 2))
    ok, counts = check_covers_space(p, X)
    assert not ok
    # All uncovered samples must live in the negative quadrant.
    uncovered = X[counts == 0]
    assert len(uncovered) > 0
    assert np.all(uncovered[:, 0] <= 0) and np.all(uncovered[:, 1] <= 0)
