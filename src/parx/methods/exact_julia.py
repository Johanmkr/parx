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
    """Exhaustively enumerate all feasible regions reachable from ``data[0]``."""
    arr = np.asarray(data, dtype=float)
    x0 = arr[0] if arr.ndim == 2 else arr.ravel()

    jl = ensure_julia()
    result = jl.LinearRegions.find_regions_exact(weights, biases, x0)
    return RegionFindResult(
        patterns=np.asarray(result[0], dtype=np.int8),
        offsets=np.asarray(result[1], dtype=np.int64),
        centroids=np.asarray(result[2], dtype=np.float64),
    )
