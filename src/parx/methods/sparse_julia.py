"""Sparse region finder — Julia implementation (parallel forward pass)."""

from __future__ import annotations

import numpy as np

from parx._julia_init import ensure_julia
from parx.methods import RegionFindResult, register_method


@register_method("sparse_julia")
def find(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    data: np.ndarray,
    **_: object,
) -> RegionFindResult:
    """Discover only regions that contain at least one point from ``data``."""
    jl = ensure_julia()
    result = jl.LinearRegions.find_regions_sparse(weights, biases, data)
    return RegionFindResult(
        patterns=np.asarray(result[0], dtype=np.int8),
        offsets=np.asarray(result[1], dtype=np.int64),
        centroids=np.asarray(result[2], dtype=np.float64),
    )
