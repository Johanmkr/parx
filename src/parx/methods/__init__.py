"""Region-finding methods registry.

Each method finds the linear regions of a ReLU network and returns a
``RegionFindResult``.  Add a new method by decorating a function with
``@register_method(name)`` inside a submodule; importing it from here
triggers registration.

Selecting a method from user code:

    from parx import compute_partition
    p = compute_partition(state_dict, X, method="sparse_julia")
    p = compute_partition(state_dict, X, method="exact_python")

List what is available:

    from parx.methods import list_methods
    list_methods()
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class RegionFindResult:
    """Common output produced by every region-finding method.

    ``patterns[i, offsets[l] : offsets[l + 1]]`` (Python-style, 0-indexed) is
    the activation pattern of layer ``l`` for region ``i``.
    """

    patterns: np.ndarray  # (n_regions, total_bits)      int8
    offsets: np.ndarray  # (n_layers + 1,)              int64
    centroids: np.ndarray  # (n_regions, input_dim)       float64


MethodFn = Callable[..., RegionFindResult]
_METHODS: dict[str, MethodFn] = {}


def register_method(name: str) -> Callable[[MethodFn], MethodFn]:
    """Decorator: register a region-finding method under ``name``."""

    def decorator(fn: MethodFn) -> MethodFn:
        if name in _METHODS:
            raise ValueError(f"Method {name!r} is already registered")
        _METHODS[name] = fn
        return fn

    return decorator


def get_method(name: str) -> MethodFn:
    """Look up a registered method by name."""
    if name not in _METHODS:
        raise ValueError(f"Unknown method {name!r}. Available: {list_methods()}")
    return _METHODS[name]


def list_methods() -> list[str]:
    """All registered method names."""
    return sorted(_METHODS)


# Self-register all bundled methods on import.
for _module in (
    "parx.methods.exact_julia",
    "parx.methods.exact_julia_fast",
    "parx.methods.exact_python",
    "parx.methods.sparse_julia",
    "parx.methods.sparse_python",
):
    importlib.import_module(_module)
