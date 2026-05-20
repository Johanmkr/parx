"""Exact region finder — pure-Python implementation.

Direct port of the Julia ``find_regions_exact`` DFS using
``scipy.optimize.linprog`` (HiGHS backend) for the Chebyshev-ball feasibility
LP and the per-constraint active-index LP.

Expect ~10-50× slower than the Julia version on networks where the per-region
LP cost dominates — scipy's linprog has higher per-call overhead than the
in-process JuMP/HiGHS path.  Ship this as a Julia-free fallback and a
correctness reference.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from parx._lp import chebyshev_center as _chebyshev_center
from parx.methods import RegionFindResult, register_method

_ZERO_NORM_TOL = 1e-10
_SLACK_TOL = 1e-8


# ── LP primitives ─────────────────────────────────────────────────────────────
# Chebyshev centre lives in parx._lp (shared with verify.py).


def _active_local_indices(
    D_full: np.ndarray, g_full: np.ndarray, n_prev: int
) -> list[int]:
    """0-based indices of non-redundant rows within the *local* (newest) block.

    A local row is non-redundant if dropping it strictly enlarges the polytope.
    Solve, for each row, ``max d_i · x`` s.t. the *other* constraints — if the
    optimum exceeds the right-hand side, the constraint is binding.
    """
    n_full = D_full.shape[0]
    n_local = n_full - n_prev
    n_vars = D_full.shape[1]
    active: list[int] = []

    for i in range(n_local):
        full_i = n_prev + i
        if np.linalg.norm(D_full[full_i]) < _ZERO_NORM_TOL:
            continue

        mask = np.ones(n_full, dtype=bool)
        mask[full_i] = False
        A_ub = D_full[mask]
        b_ub = g_full[mask]

        if A_ub.shape[0] == 0:
            # No other constraints → objective is unbounded → facet is active.
            active.append(i)
            continue

        c = -D_full[full_i]  # max d_i · x ↔ min -d_i · x
        res = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=[(None, None)] * n_vars,
            method="highs",
        )
        if res.status == 3:  # unbounded ↔ active
            active.append(i)
        elif res.success and (-res.fun) > g_full[full_i] - _SLACK_TOL:
            active.append(i)

    return active


# ── DFS ───────────────────────────────────────────────────────────────────────


def _dfs_exact(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    layer: int,
    A_prev: np.ndarray,
    c_prev: np.ndarray,
    D_prev: np.ndarray,
    g_prev: np.ndarray,
    q_path: list[np.ndarray],
    x_parent: np.ndarray,
    results: list[tuple[list[np.ndarray], np.ndarray]],
) -> None:
    if layer >= len(weights):
        results.append(([q.copy() for q in q_path], x_parent.copy()))
        return

    W = weights[layer]
    b = biases[layer]
    W_hat = W @ A_prev
    b_hat = W @ c_prev + b

    q_start = (W_hat @ x_parent + b_hat) > 0
    visited: set[bytes] = {q_start.tobytes()}
    queue: list[np.ndarray] = [q_start]

    while queue:
        q_curr = queue.pop(0)
        s = -2.0 * q_curr.astype(np.float64) + 1.0
        D_local = s[:, None] * W_hat
        g_local = -(s * b_hat)

        D_full = np.vstack([D_prev, D_local])
        g_full = np.concatenate([g_prev, g_local])

        x_int, _ = _chebyshev_center(D_full, g_full)
        if x_int is None:
            continue

        A_next = q_curr[:, None].astype(np.float64) * W_hat
        c_next = q_curr.astype(np.float64) * b_hat

        _dfs_exact(
            weights,
            biases,
            layer + 1,
            A_next,
            c_next,
            D_full,
            g_full,
            q_path + [q_curr.copy()],
            x_int,
            results,
        )

        for i in _active_local_indices(D_full, g_full, D_prev.shape[0]):
            q_n = q_curr.copy()
            q_n[i] = not q_n[i]
            key = q_n.tobytes()
            if key not in visited:
                visited.add(key)
                queue.append(q_n)


# ── Public entry point ────────────────────────────────────────────────────────


@register_method("exact_python")
def find(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    data: np.ndarray,
    **_: object,
) -> RegionFindResult:
    """Exhaustively enumerate all feasible regions via DFS + facet-flipping."""
    arr = np.asarray(data, dtype=float)
    x0 = arr[0] if arr.ndim == 2 else arr.ravel()

    n_layers = len(weights)
    layer_sizes = [w.shape[0] for w in weights]
    input_dim = weights[0].shape[1]
    total_bits = sum(layer_sizes)

    offsets = np.zeros(n_layers + 1, dtype=np.int64)
    for layer_idx in range(n_layers):
        offsets[layer_idx + 1] = offsets[layer_idx] + layer_sizes[layer_idx]

    results: list[tuple[list[np.ndarray], np.ndarray]] = []
    _dfs_exact(
        weights=weights,
        biases=biases,
        layer=0,
        A_prev=np.eye(input_dim),
        c_prev=np.zeros(input_dim),
        D_prev=np.zeros((0, input_dim)),
        g_prev=np.zeros(0),
        q_path=[],
        x_parent=x0.astype(np.float64),
        results=results,
    )

    n_regions = len(results)
    patterns = np.zeros((n_regions, total_bits), dtype=np.int8)
    centroids = np.zeros((n_regions, input_dim), dtype=np.float64)
    for i, (path, centroid) in enumerate(results):
        patterns[i] = np.concatenate([q.astype(np.int8) for q in path])
        centroids[i] = centroid

    return RegionFindResult(patterns=patterns, offsets=offsets, centroids=centroids)
