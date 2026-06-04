"""
parx — POLyhedral Activation Region Xplorer

Exactly enumerates the linear regions of ReLU neural networks.
"""

from __future__ import annotations

import numpy as np

from parx import (
    methods,  # noqa: F401 — trigger method registration
    viz,  # noqa: F401 — expose parx.viz
)
from parx.viz import animate_epochs, animate_epochs_video
from parx._check import check_julia
from parx._julia_init import ensure_julia  # noqa: F401
from parx.analysis import (
    always_active_neurons,
    complexity_profile,
    dead_neurons,
    neuron_activity_rates,
    region_size_summary,
)
from parx.io import iter_state_dicts
from parx.io_partition import load_partition, save_partition
from parx.methods import get_method, list_methods
from parx.network import load_network
from parx.partition import Partition
from parx.precompile import precompile
from parx.region import Region

check_julia()

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "always_active_neurons",
    "animate_epochs",
    "animate_epochs_video",
    "complexity_profile",
    "compute_partition",
    "dead_neurons",
    "ensure_julia",
    "iter_state_dicts",
    "list_methods",
    "load_network",
    "load_partition",
    "neuron_activity_rates",
    "Partition",
    "precompile",
    "Region",
    "region_size_summary",
    "save_partition",
]


def compute_partition(
    source,
    data: np.ndarray,
    *,
    method: str = "sparse_julia",
    include_output_layer: bool = False,
    **method_kwargs,
) -> Partition:
    """Find the linear regions of a ReLU network.

    Parameters
    ----------
    source:
        A single network's parameters.  Accepted forms: PyTorch ``state_dict``,
        ``nn.Module``, or path to a ``.pth`` / ``.h5`` file.  For per-epoch
        analyses, iterate over state dicts at the call site (see
        :func:`parx.io.iter_state_dicts`).
    data:
        Input points, shape ``(N, input_dim)``.  Sparse methods scan the array
        for activation patterns; exact methods use ``data[0]`` as the DFS
        starting point.
    method:
        Name of a registered region-finding method.  Built-ins:
        ``"sparse_julia"`` (default), ``"exact_julia"``, ``"sparse_python"``,
        ``"exact_python"``.  See :func:`parx.list_methods`.
    include_output_layer:
        Include the final linear layer in the partition.  Defaults to ``False``
        because only hidden ReLU layers define the polyhedral partition.
    **method_kwargs:
        Forwarded verbatim to the chosen method's function.

    Returns
    -------
    Partition
    """
    weights, biases = load_network(source, include_output_layer=include_output_layer)
    fn = get_method(method)
    result = fn(weights, biases, data, **method_kwargs)
    return Partition.from_result(result, weights, biases)
