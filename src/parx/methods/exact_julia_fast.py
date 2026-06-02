"""Exact region finder — faster Julia implementation using ``direct_model``.

Same DFS as ``exact_julia`` but constructs each LP with
``direct_model(HiGHS.Optimizer())`` instead of ``Model(HiGHS.Optimizer)``.
Skips JuMP's caching/bridge layer, which dominates the runtime on the tiny
LPs we solve — measured ~6× speed-up on the LP path.
"""

from __future__ import annotations

import numpy as np

from parx._julia_init import ensure_julia
from parx.methods import RegionFindResult, register_method


@register_method("exact_julia_fast")
def find(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    data: np.ndarray,
    **_: object,
) -> RegionFindResult:
    arr = np.asarray(data, dtype=float)
    x0 = arr[0] if arr.ndim == 2 else arr.ravel()

    jl = ensure_julia()
    result = jl.LinearRegions.find_regions_exact_fast(weights, biases, x0)
    return RegionFindResult(
        patterns=np.asarray(result[0], dtype=np.int8),
        offsets=np.asarray(result[1], dtype=np.int64),
        centroids=np.asarray(result[2], dtype=np.float64),
        active_indices_flat=np.asarray(result[3], dtype=np.int32),
        active_offsets=np.asarray(result[4], dtype=np.int64),
        bounded=np.asarray(result[5], dtype=bool),
    )
