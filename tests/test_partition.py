"""Tests for Region and Partition (pure Python, no Julia)."""
import numpy as np
import pytest

from parx.partition import Partition
from parx.region import Region


def _identity_partition() -> Partition:
    """Two-layer identity-weight network [2→2→2], single all-active region."""
    W = np.eye(2)
    b = np.zeros(2)
    region = Region(
        activation_path=[np.array([True, True]), np.array([True, True])],
        centroid=np.array([1.0, 1.0]),
    )
    return Partition(regions=[region], weights=[W, W], biases=[b, b])


def _quadrant_partition() -> Partition:
    """One-layer [2→2] network that splits input into four quadrants."""
    W = np.eye(2)
    b = np.zeros(2)
    regions = [
        Region([np.array([True, True])],  centroid=np.array([ 1.0,  1.0])),
        Region([np.array([True, False])], centroid=np.array([ 1.0, -1.0])),
        Region([np.array([False, True])], centroid=np.array([-1.0,  1.0])),
        Region([np.array([False, False])],centroid=np.array([-1.0, -1.0])),
    ]
    return Partition(regions=regions, weights=[W], biases=[b])


class TestPartitionBasics:
    def test_repr(self):
        p = _identity_partition()
        assert "n_regions=1" in repr(p)

    def test_len(self):
        p = _quadrant_partition()
        assert len(p) == 4

    def test_regions_at_layer(self):
        p = _identity_partition()
        assert len(p.regions_at_layer(2)) == 1
        assert len(p.regions_at_layer(1)) == 0


class TestHalfspaces:
    def test_output_shape(self):
        p = _identity_partition()
        D, g = p.halfspaces(p.regions[0])
        assert D.shape == (4, 2)  # 2 layers × 2 neurons = 4 constraints
        assert g.shape == (4,)

    def test_active_point_satisfies_constraints(self):
        p = _identity_partition()
        D, g = p.halfspaces(p.regions[0])
        x = np.array([2.0, 3.0])
        assert np.all(D @ x <= g + 1e-10), "centroid must satisfy D*x <= g"

    def test_active_only_with_indices(self):
        p = _identity_partition()
        r = p.regions[0]
        r.active_indices = np.array([0, 2])
        D, g = p.halfspaces(r, active_only=True)
        assert D.shape[0] == 2

    def test_empty_path_returns_unconstrained(self):
        W = np.eye(2)
        b = np.zeros(2)
        r = Region(activation_path=[], centroid=np.zeros(2))
        p = Partition(regions=[r], weights=[W], biases=[b])
        D, g = p.halfspaces(r)
        assert D.shape == (0, 2)


class TestRoute:
    def test_all_positive_quadrant(self):
        p = _quadrant_partition()
        X = np.array([[1.0, 1.0], [2.0, 3.0]])
        results = p.route(X)
        expected = p.regions[0]  # both neurons active
        assert results[0] is expected
        assert results[1] is expected

    def test_each_quadrant(self):
        p = _quadrant_partition()
        X = np.array([
            [ 1.0,  1.0],  # q = [T, T]
            [ 1.0, -1.0],  # q = [T, F]
            [-1.0,  1.0],  # q = [F, T]
            [-1.0, -1.0],  # q = [F, F]
        ])
        results = p.route(X)
        for i, r in enumerate(results):
            assert r is p.regions[i], f"point {i} routed to wrong region"

    def test_unknown_point_returns_none(self):
        # Partition with only one region; route a point that activates differently
        W = np.eye(2)
        b = np.zeros(2)
        r = Region([np.array([True, True])], centroid=np.array([1.0, 1.0]))
        p = Partition(regions=[r], weights=[W], biases=[b])
        results = p.route(np.array([[-1.0, -1.0]]))
        assert results[0] is None
