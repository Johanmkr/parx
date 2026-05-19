"""Sampling-based partition verification (pure NumPy, no Julia)."""

from __future__ import annotations

import numpy as np

from parx.partition import Partition


def count_region_memberships(
    partition: Partition,
    X: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    """For each row of X, count how many regions contain it.

    A point x is considered inside region r when ``D @ x <= g + tol`` for
    every row of the region's halfspace system.

    Parameters
    ----------
    partition : Partition
    X : array of shape (N, input_dim)
    tol : halfspace tolerance (accounts for floating-point error near boundaries)

    Returns
    -------
    counts : int array of shape (N,)
    """
    X = np.asarray(X, dtype=float)
    counts = np.zeros(len(X), dtype=int)
    for region in partition.regions:
        D, g = partition.halfspaces(region)
        # (N, n_constraints) <= (n_constraints,)  →  (N,) bool
        contained = np.all(X @ D.T <= g + tol, axis=1)
        counts += contained
    return counts


def check_no_overlaps(
    partition: Partition,
    X: np.ndarray,
    tol: float = 1e-8,
) -> tuple[bool, np.ndarray]:
    """Check that no two regions share an interior point in the sample.

    Two regions can share a boundary (a lower-dimensional face) which is
    fine, but they must not share interior points.  Sampling-based: fails only
    if a sample point satisfies two full halfspace systems simultaneously.

    Returns
    -------
    ok : True when every sampled point belongs to at most one region
    counts : (N,) membership counts for diagnosis
    """
    counts = count_region_memberships(partition, X, tol=tol)
    return bool(np.all(counts <= 1)), counts


def check_covers_space(
    partition: Partition,
    X: np.ndarray,
    tol: float = 1e-8,
) -> tuple[bool, np.ndarray]:
    """Check that every sampled point belongs to exactly one region.

    This should hold for exact-mode partitions.  Sparse partitions may
    return ``counts == 0`` for points in regions not covered by the data.

    Returns
    -------
    ok : True when every sampled point is in exactly one region
    counts : (N,) membership counts for diagnosis
    """
    counts = count_region_memberships(partition, X, tol=tol)
    return bool(np.all(counts == 1)), counts
