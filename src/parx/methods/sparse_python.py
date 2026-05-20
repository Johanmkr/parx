"""Sparse region finder — pure NumPy implementation.

Same semantics as ``sparse_julia``: forward-pass every input point, dedupe
activation paths, return one centroid per unique region.  Useful as a Julia-
free fallback and as a speed reference for the Julia version.
"""

from __future__ import annotations

import numpy as np

from parx.methods import RegionFindResult, register_method


@register_method("sparse_python")
def find(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    data: np.ndarray,
    **_: object,
) -> RegionFindResult:
    """Discover only regions that contain at least one point from ``data``."""
    X = np.asarray(data, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"data must be 2-D (N, input_dim); got shape {X.shape}")

    n_layers = len(weights)
    layer_sizes = [w.shape[0] for w in weights]
    total_bits = sum(layer_sizes)

    offsets = np.zeros(n_layers + 1, dtype=np.int64)
    for layer_idx in range(n_layers):
        offsets[layer_idx + 1] = (
            offsets[layer_idx] + layer_sizes[layer_idx]
        )

    # Vectorised forward pass: build the concatenated bit pattern per point.
    N = X.shape[0]
    patterns_all = np.empty((N, total_bits), dtype=np.int8)
    A = X
    for layer_idx, (W, b) in enumerate(zip(weights, biases)):
        Z = A @ W.T + b  # (N, layer_sizes[layer_idx])
        Q = (Z > 0).astype(np.int8)
        patterns_all[:, offsets[layer_idx] : offsets[layer_idx + 1]] = Q
        A = Q * Z

    # Dedupe: hash via bytes to preserve first-seen order
    seen: dict[bytes, int] = {}
    unique_indices: list[int] = []
    for i in range(N):
        key = patterns_all[i].tobytes()
        if key not in seen:
            seen[key] = i
            unique_indices.append(i)

    idx = np.array(unique_indices, dtype=np.int64)
    return RegionFindResult(
        patterns=patterns_all[idx].copy(),
        offsets=offsets,
        centroids=X[idx].astype(np.float64, copy=True),
    )
