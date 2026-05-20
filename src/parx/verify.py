"""Partition verification utilities — both sample-based and LP-based.

The sample-based checks (``check_no_overlaps``, ``check_covers_space``) probe
the partition at user-supplied points.  The LP-based checks
(``check_regions_nonempty``, ``region_chebyshev_radii``) interrogate the
geometry directly through the halfspace systems.  ``check_routing_consistency``
cross-checks that ``Partition.route`` agrees with the membership returned by
the halfspace tests.

A correct exact-mode partition should satisfy:

  * every region has a strictly positive Chebyshev radius
    (``check_regions_nonempty``)
  * no two regions claim the same interior point
    (``check_no_overlaps``)
  * every input point is claimed by exactly one region
    (``check_covers_space``)
  * ``route(x)`` returns the region whose halfspace system contains ``x``
    (``check_routing_consistency``)
"""

from __future__ import annotations

import numpy as np

from parx._lp import chebyshev_center
from parx.partition import Partition


# ── Sample-based checks ──────────────────────────────────────────────────────

def count_region_memberships(
    partition: Partition,
    X: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    """For each row of X, count how many regions contain it.

    A point ``x`` is considered inside region ``r`` when ``D @ x ≤ g + tol``
    for every row of the region's halfspace system.
    """
    X = np.asarray(X, dtype=float)
    counts = np.zeros(len(X), dtype=int)
    for region in partition.regions:
        D, g = partition.halfspaces(region)
        contained = np.all(X @ D.T <= g + tol, axis=1)
        counts += contained
    return counts


def check_no_overlaps(
    partition: Partition,
    X: np.ndarray,
    tol: float = 1e-8,
) -> tuple[bool, np.ndarray]:
    """Check that no two regions share an interior point in the sample.

    Two regions can share a boundary (a lower-dimensional face) which is fine,
    but they must not share interior points.  Sample-based: only fails if a
    sample point satisfies two full halfspace systems simultaneously.
    """
    counts = count_region_memberships(partition, X, tol=tol)
    return bool(np.all(counts <= 1)), counts


def check_covers_space(
    partition: Partition,
    X: np.ndarray,
    tol: float = 1e-8,
) -> tuple[bool, np.ndarray]:
    """Check that every sampled point belongs to exactly one region.

    Should hold for exact-mode partitions.  Sparse partitions may return
    ``counts == 0`` for points in regions not covered by the data.
    """
    counts = count_region_memberships(partition, X, tol=tol)
    return bool(np.all(counts == 1)), counts


# ── LP-based checks ──────────────────────────────────────────────────────────

def region_chebyshev_radii(
    partition: Partition,
    *,
    max_radius: float = 1e3,
) -> np.ndarray:
    """Chebyshev (largest inscribed ball) radius for every region.

    A radius of ``0.0`` means the region is empty or degenerate (a bug for
    exact partitions).  A radius at or near ``max_radius`` indicates the
    region is unbounded; the LP caps it for numerical sanity.
    """
    radii = np.zeros(len(partition), dtype=float)
    for i, region in enumerate(partition.regions):
        D, g = partition.halfspaces(region)
        _, r = chebyshev_center(D, g, max_radius=max_radius)
        radii[i] = r
    return radii


def check_regions_nonempty(
    partition: Partition,
    *,
    min_radius: float = 1e-6,
    max_radius: float = 1e3,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """LP-based check that every region has a strictly positive Chebyshev radius.

    Catches degenerate regions that survived construction — a class of bugs
    that sample-based checks miss (an empty region is never selected by a
    sample so it cannot create an overlap or a gap).

    Returns
    -------
    ok : True when every region has radius ≥ ``min_radius``.
    bad_indices : indices of regions failing the test, sorted ascending by
        radius (smallest first — most degenerate first).
    radii : per-region Chebyshev radii (same as ``region_chebyshev_radii``).
    """
    radii = region_chebyshev_radii(partition, max_radius=max_radius)
    mask = radii < min_radius
    bad = np.where(mask)[0]
    bad = bad[np.argsort(radii[bad])]
    return bool(len(bad) == 0), bad, radii


def check_routing_consistency(
    partition: Partition,
    X: np.ndarray,
    *,
    tol: float = 1e-8,
) -> tuple[bool, list[int]]:
    """Verify that ``route(x)`` matches the halfspace-membership region.

    For every sample point, the region returned by :meth:`Partition.route`
    must contain that point under ``D x ≤ g + tol``.  Discrepancies indicate
    a mismatch between the forward-pass routing logic and the halfspace
    reconstruction.

    Returns
    -------
    ok : True when every routed region contains its sample.
    bad : sample indices where the check failed (either ``route`` returned
        ``None`` or the returned region did not contain the point).
    """
    X = np.asarray(X, dtype=float)
    routed = partition.route(X)
    bad: list[int] = []
    for i, region in enumerate(routed):
        if region is None:
            bad.append(i)
            continue
        D, g = partition.halfspaces(region)
        if not np.all(D @ X[i] <= g + tol):
            bad.append(i)
    return len(bad) == 0, bad


# ── Targeted sampling helpers ────────────────────────────────────────────────

def sample_near_boundaries(
    partition: Partition,
    *,
    eps: float = 1e-3,
) -> np.ndarray:
    """Generate sample points near every region's halfspace boundaries.

    For each region and each halfspace ``D[i] · x ≤ g[i]``, project the
    region's centroid onto the boundary plane and step ``eps`` along the
    inward normal direction.  These points sit just inside a region but near
    a seam shared with a neighbour — exactly where numerical overlaps or
    coverage gaps would manifest.

    Returns an ``(M, input_dim)`` array of points.  Useful when used as the
    ``X`` argument to :func:`check_no_overlaps` or :func:`check_covers_space`.
    """
    pts: list[np.ndarray] = []
    for region in partition.regions:
        D, g = partition.halfspaces(region)
        c = region.centroid
        for i in range(D.shape[0]):
            n_i = D[i]
            nrm = np.linalg.norm(n_i)
            if nrm < 1e-10:
                continue
            # Project centroid onto the plane D[i] · x = g[i].
            offset = (n_i @ c - g[i]) / nrm
            x_plane = c - offset * (n_i / nrm)
            # Step eps inside (negative-normal direction satisfies D[i]·x ≤ g[i]).
            step = eps * (n_i / nrm)
            pts.append(x_plane - step)
    if not pts:
        return np.zeros((0, partition.input_dim))
    return np.array(pts)
