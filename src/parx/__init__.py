"""
parx — POLyhedral Activation Region Xplorer

Exactly enumerates the linear regions of ReLU neural networks.
"""

from __future__ import annotations

import numpy as np

from parx._check import check_julia
from parx._julia_init import ensure_julia  # noqa: F401
from parx.network import load_network
from parx.partition import Partition
from parx.region import Region
from parx import viz  # noqa: F401  — expose parx.viz

check_julia()

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "compute_partition",
    "ensure_julia",
    "load_network",
    "Partition",
    "Region",
]


def compute_partition(
    model,
    data: np.ndarray,
    *,
    mode: str = "sparse",
    include_output_layer: bool = False,
) -> Partition:
    """Find the linear regions of a ReLU network.

    Parameters
    ----------
    model:
        ``nn.Module``, PyTorch ``state_dict``, or path to a ``.pth`` / ``.h5``
        file.
    data:
        Input points, shape ``(N, input_dim)``.  Required for ``mode="sparse"``;
        used only to pick a starting point for ``mode="exact"``.
    mode:
        ``"sparse"`` (default) — discover only regions that contain at least one
        point from *data*.  Fast; scales with dataset size.
        ``"exact"`` — exhaustively enumerate all feasible regions via DFS and
        facet-flipping.  Not yet implemented.
    include_output_layer:
        Include the final linear layer in the partition.  Defaults to ``False``.

    Returns
    -------
    Partition
    """
    weights, biases = load_network(model, include_output_layer=include_output_layer)

    jl = ensure_julia()

    if mode == "sparse":
        result = jl.LinearRegions.find_regions_sparse(weights, biases, data)
        return Partition._from_sparse_output(result, weights, biases)

    if mode == "exact":
        data_arr = np.asarray(data, dtype=float)
        x0 = data_arr[0] if data_arr.ndim == 2 else data_arr.ravel()
        result = jl.LinearRegions.find_regions_exact(weights, biases, x0)
        return Partition._from_sparse_output(result, weights, biases)

    raise ValueError(f"Unknown mode {mode!r}. Choose 'sparse' or 'exact'.")
