"""Region: a single polytope where the network is a fixed affine map."""

from __future__ import annotations

import numpy as np


class Region:
    """A single linear region of a ReLU network.

    Defined by its activation path: the sequence of per-layer activation
    patterns (one bool array per hidden layer) that is shared by every input
    point inside this polytope.
    """

    __slots__ = ("activation_path", "centroid", "active_indices", "bounded")

    def __init__(
        self,
        activation_path: list[np.ndarray],
        centroid: np.ndarray,
        active_indices: np.ndarray | None = None,
        bounded: bool = False,
    ) -> None:
        self.activation_path = activation_path
        self.centroid = centroid
        self.active_indices = active_indices  # non-redundant rows in D*x≤g (exact only)
        self.bounded = bounded

    @property
    def n_layers(self) -> int:
        """Number of hidden layers spanned by this region's activation path."""
        return len(self.activation_path)

    def __repr__(self) -> str:
        shapes = [q.shape[0] for q in self.activation_path]
        return f"Region(layers={shapes}, bounded={self.bounded})"
