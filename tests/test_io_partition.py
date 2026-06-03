"""Tests for parx.io_partition — save_partition / load_partition round-trips.

Pure Python: no Julia required.  All partitions are built by hand so the
expected values are known exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from parx.io_partition import load_partition, save_partition
from parx.partition import Partition
from parx.region import Region

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _quadrant_partition() -> Partition:
    """One-layer [2→2] identity network — four quadrant regions."""
    W = np.eye(2)
    b = np.zeros(2)
    regions = [
        Region([np.array([True, True])], centroid=np.array([1.0, 1.0])),
        Region([np.array([True, False])], centroid=np.array([1.0, -1.0])),
        Region([np.array([False, True])], centroid=np.array([-1.0, 1.0])),
        Region([np.array([False, False])], centroid=np.array([-1.0, -1.0])),
    ]
    return Partition(regions=regions, weights=[W], biases=[b])


def _exact_partition() -> Partition:
    """Quadrant partition with active_indices and bounded populated."""
    W = np.eye(2)
    b = np.zeros(2)
    regions = [
        Region(
            [np.array([True, True])],
            centroid=np.array([1.0, 1.0]),
            active_indices=np.array([0, 1], dtype=np.int32),
            bounded=False,
        ),
        Region(
            [np.array([True, False])],
            centroid=np.array([1.0, -1.0]),
            active_indices=np.array([0], dtype=np.int32),
            bounded=False,
        ),
    ]
    return Partition(regions=regions, weights=[W], biases=[b])


def _empty_partition() -> Partition:
    return Partition(regions=[], weights=[np.eye(2)], biases=[np.zeros(2)])


# ── Round-trip: basic ─────────────────────────────────────────────────────────


def test_roundtrip_region_count(tmp_path):
    p = _quadrant_partition()
    path = tmp_path / "part.npz"
    save_partition(p, path)
    loaded = load_partition(path)
    assert len(loaded) == len(p)


def test_roundtrip_activation_paths(tmp_path):
    p = _quadrant_partition()
    path = tmp_path / "part.npz"
    save_partition(p, path)
    loaded = load_partition(path)
    for orig, load in zip(p.regions, loaded.regions):
        assert len(orig.activation_path) == len(load.activation_path)
        for q_orig, q_load in zip(orig.activation_path, load.activation_path):
            assert np.array_equal(q_orig, q_load)


def test_roundtrip_centroids(tmp_path):
    p = _quadrant_partition()
    path = tmp_path / "part.npz"
    save_partition(p, path)
    loaded = load_partition(path)
    for orig, load in zip(p.regions, loaded.regions):
        assert np.allclose(orig.centroid, load.centroid)


def test_roundtrip_weights_biases(tmp_path):
    p = _quadrant_partition()
    path = tmp_path / "part.npz"
    save_partition(p, path)
    loaded = load_partition(path)
    assert len(loaded.weights) == len(p.weights)
    for w_orig, w_load in zip(p.weights, loaded.weights):
        assert np.allclose(w_orig, w_load)
    for b_orig, b_load in zip(p.biases, loaded.biases):
        assert np.allclose(b_orig, b_load)


def test_roundtrip_halfspaces_identical(tmp_path):
    """Reloaded halfspace systems must match the originals exactly."""
    p = _quadrant_partition()
    path = tmp_path / "part.npz"
    save_partition(p, path)
    loaded = load_partition(path)
    for r_orig, r_load in zip(p.regions, loaded.regions):
        D_orig, g_orig = p.halfspaces(r_orig)
        D_load, g_load = loaded.halfspaces(r_load)
        assert np.allclose(D_orig, D_load)
        assert np.allclose(g_orig, g_load)


# ── Round-trip: optional exact-method fields ──────────────────────────────────


def test_roundtrip_active_indices_preserved(tmp_path):
    p = _exact_partition()
    path = tmp_path / "exact.npz"
    save_partition(p, path)
    loaded = load_partition(path)
    for orig, load in zip(p.regions, loaded.regions):
        assert load.active_indices is not None
        assert np.array_equal(orig.active_indices, load.active_indices)


def test_roundtrip_bounded_preserved(tmp_path):
    p = _exact_partition()
    path = tmp_path / "exact.npz"
    save_partition(p, path)
    loaded = load_partition(path)
    for orig, load in zip(p.regions, loaded.regions):
        assert orig.bounded == load.bounded


def test_sparse_partition_loads_without_active_indices(tmp_path):
    """Partitions without active_indices round-trip cleanly."""
    p = _quadrant_partition()
    path = tmp_path / "sparse.npz"
    save_partition(p, path)
    loaded = load_partition(path)
    for region in loaded.regions:
        assert region.active_indices is None


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_empty_partition_roundtrip(tmp_path):
    p = _empty_partition()
    path = tmp_path / "empty.npz"
    save_partition(p, path)
    loaded = load_partition(path)
    assert len(loaded) == 0


def test_npz_extension_appended(tmp_path):
    """np.savez appends .npz; load_partition must accept the bare path too."""
    p = _quadrant_partition()
    bare = tmp_path / "part"
    save_partition(p, bare)
    loaded = load_partition(tmp_path / "part.npz")
    assert len(loaded) == len(p)


def test_version_mismatch_raises(tmp_path):
    p = _quadrant_partition()
    path = tmp_path / "part.npz"
    save_partition(p, path)

    data = dict(np.load(path))
    data["version"] = np.int64(99)
    np.savez(path, **data)

    with pytest.raises(ValueError, match="Unsupported parx partition format version"):
        load_partition(path)
