"""Lightweight runtime diagnostics: thread count, method benchmarking."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


def thread_info() -> dict[str, Any]:
    """Report Julia and Python thread settings.

    Returns
    -------
    dict with keys:
        ``julia_threads``    : ``Threads.nthreads()`` inside the live Julia
                               runtime (after env vars take effect).
        ``julia_max_threads``: ``Threads.maxthreadid()`` (Julia ≥ 1.10).
        ``env_JULIA_NUM_THREADS``: the env var as Python sees it.
    """
    import os

    from parx._julia_init import ensure_julia

    jl = ensure_julia()
    return {
        "julia_threads": int(jl.seval("Threads.nthreads()")),
        "julia_max_threads": int(jl.seval("Threads.maxthreadid()")),
        "env_JULIA_NUM_THREADS": os.environ.get("JULIA_NUM_THREADS"),
    }


def benchmark_method(
    method: str,
    state_dict_or_model,
    data: np.ndarray,
    *,
    repeats: int = 3,
    warmup: int = 1,
) -> dict[str, Any]:
    """Time a region-finding method.

    Returns
    ``{"method": ..., "n_regions": ..., "best_seconds": ..., "all_seconds": [...]}``.

    A warm-up run is performed and discarded to amortise JIT compilation.
    """
    from parx import compute_partition

    for _ in range(warmup):
        compute_partition(state_dict_or_model, data, method=method)

    timings = []
    n_regions = 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        p = compute_partition(state_dict_or_model, data, method=method)
        timings.append(time.perf_counter() - t0)
        n_regions = len(p)

    return {
        "method": method,
        "n_regions": n_regions,
        "best_seconds": min(timings),
        "all_seconds": timings,
    }
