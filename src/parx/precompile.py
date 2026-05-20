"""Optional smoke-run helper to amortise Julia's TTFX (time-to-first-X).

Julia compiles each function on first call.  In ``parx`` this manifests as:

  1. ~1-2 s for the Julia runtime + ``LinearRegions.jl`` to load
  2. ~3-5 s the first time JuMP + HiGHS are touched
  3. ~0.2-2 s per method the first time it is invoked

Cost (1) is paid implicitly by ``ensure_julia()``.  Cost (2) and (3) are paid
by the first call to each Julia-backed method.  Total cold start from a fresh
process: 5-10 seconds; subsequent calls take 0-1 s.

Call :func:`precompile` once at the top of a session (or notebook) if you
want predictable timing from your first real ``compute_partition`` call.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

# Methods that benefit from the smoke run.  Pure-Python methods aren't listed
# because they don't pay a JIT cost — first-call performance already matches
# subsequent calls.
_JULIA_METHODS = ("sparse_julia", "exact_julia", "exact_julia_fast")


def precompile(*, verbose: bool = False) -> dict[str, float]:
    """Run a tiny problem through every Julia-backed method.

    Returns a dict of ``{method: seconds_taken}`` so callers can confirm the
    smoke run actually executed.  Pure-Python methods are skipped because they
    do not benefit.

    Parameters
    ----------
    verbose : bool
        If True, print per-method timing as it runs.
    """
    from parx import compute_partition
    from parx._julia_init import ensure_julia

    ensure_julia()  # cost (1) + (2)

    weights = {
        "0.weight": np.eye(2),
        "0.bias": np.zeros(2),
        "2.weight": np.eye(2),
        "2.bias": np.zeros(2),
    }
    X = np.array([[1.0, 1.0], [-1.0, -1.0]])
    x0 = np.array([1.0, 1.0])

    timings: dict[str, float] = {}
    for method in _JULIA_METHODS:
        data: Any = X if method == "sparse_julia" else x0
        t0 = time.perf_counter()
        compute_partition(weights, data, method=method)
        dt = time.perf_counter() - t0
        timings[method] = dt
        if verbose:
            print(f"  precompiled {method:<20} {dt:.2f}s")
    return timings
