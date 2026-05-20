"""Shared LP primitives used by region finders and partition verification."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

_ZERO_NORM_TOL = 1e-10
_SLACK_TOL = 1e-8


def chebyshev_center(
    D: np.ndarray,
    g: np.ndarray,
    *,
    max_radius: float = 1e3,
) -> tuple[np.ndarray | None, float]:
    """Chebyshev centre of ``{x : D x ≤ g}``.

    Returns ``(x_interior, radius)`` or ``(None, 0.0)`` when empty / degenerate.

    Zero-norm rows are filtered (they would make the LP unbounded); a
    zero-norm row with strictly negative ``g[i]`` makes the system infeasible
    (returns ``(None, 0.0)``).  Unbounded polytopes are detected by the radius
    hitting ``max_radius``; callers can compare against that value to decide
    whether to treat the region as unbounded.
    """
    m, n = D.shape
    if m == 0:
        return np.zeros(n), float("inf")

    row_norms = np.linalg.norm(D, axis=1)
    valid = row_norms > _ZERO_NORM_TOL

    if np.any(g[~valid] < -_ZERO_NORM_TOL):
        return None, 0.0
    if not valid.any():
        return np.zeros(n), float("inf")

    D_v = D[valid]
    g_v = g[valid]
    norms_v = row_norms[valid]

    # Variables = (x_1, …, x_n, r).  Constraint i: D_v[i] · x + ‖D_v[i]‖ · r ≤ g_v[i].
    A_ub = np.hstack([D_v, norms_v[:, None]])
    b_ub = g_v
    c = np.zeros(n + 1)
    c[-1] = -1.0  # maximise r ↔ minimise -r
    bounds = [(None, None)] * n + [(0.0, max_radius)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success or res.x is None:
        return None, 0.0

    x = res.x[:n]
    r = float(res.x[-1])
    if r < _SLACK_TOL:
        return None, 0.0
    return x, r
