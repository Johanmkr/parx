"""Exact region finder — Julia implementation (DFS + facet-flipping LPs)."""

from __future__ import annotations

import numpy as np

from parx._julia_init import ensure_julia
from parx.methods import RegionFindResult, register_method


@register_method("exact_julia")
def find(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    data: np.ndarray,
    **_: object,
) -> RegionFindResult:
    """Exhaustively enumerate all feasible regions reachable from ``data[0]``.

    Parameters
    ----------
    weights, biases:
        Per-layer weight/bias arrays (as returned by
        :func:`parx.network.load_network`).
    data : np.ndarray
        Seed point(s); only ``data[0]`` (or ``data`` itself if 1-D) is used
        as the DFS starting point — it must lie in a feasible region.
    **_ :
        Ignored — accepted so every method shares one call signature.

    Returns
    -------
    RegionFindResult
        All six fields populated: ``patterns``, ``offsets``, ``centroids``,
        plus ``active_indices_flat``/``active_offsets`` (non-redundant
        halfspace rows per region) and ``bounded`` (per-region flag).
    """
    arr = np.asarray(data, dtype=float)
    x0 = arr[0] if arr.ndim == 2 else arr.ravel()

    jl = ensure_julia()
    result = jl.LinearRegions.find_regions_exact(weights, biases, x0)
    return RegionFindResult(
        patterns=np.asarray(result[0], dtype=np.int8),
        offsets=np.asarray(result[1], dtype=np.int64),
        centroids=np.asarray(result[2], dtype=np.float64),
        active_indices_flat=np.asarray(result[3], dtype=np.int32),
        active_offsets=np.asarray(result[4], dtype=np.int64),
        bounded=np.asarray(result[5], dtype=bool),
    )
