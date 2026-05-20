"""Partition: the collection of linear regions for a ReLU network."""

from __future__ import annotations

import numpy as np

from parx.region import Region


class Partition:
    """The polyhedral partition of a ReLU network's input space.

    Holds a flat list of ``Region`` objects together with the network weights
    needed to reconstruct halfspace systems on demand.  No tree structure is
    stored; the activation paths on each Region encode all needed topology.
    """

    def __init__(
        self,
        regions: list[Region],
        weights: list[np.ndarray],
        biases: list[np.ndarray],
    ) -> None:
        self.regions = regions
        self.weights = weights
        self.biases = biases
        self.input_dim: int = weights[0].shape[1]
        self.n_layers: int = len(weights)

    def __len__(self) -> int:
        return len(self.regions)

    def __repr__(self) -> str:
        return (
            "Partition("
            f"n_regions={len(self)}, "
            f"n_layers={self.n_layers}, "
            f"input_dim={self.input_dim}"
            ")"
        )

    # ── Geometry ──────────────────────────────────────────────────────────────

    def halfspaces(
        self,
        region: Region,
        active_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct the halfspace system D*x ≤ g for a region.

        Direct port of ``compute_path_geometry`` from the original Julia
        implementation.  Pure NumPy; no Julia call required.

        Parameters
        ----------
        region:
            The region whose constraints to compute.
        active_only:
            If ``True`` and ``region.active_indices`` is available (exact
            construction only), return only the non-redundant rows.

        Returns
        -------
        D : ndarray, shape (n_constraints, input_dim)
        g : ndarray, shape (n_constraints,)
            The system D*x ≤ g defines the polytope.
        """
        q_path = region.activation_path
        if not q_path:
            return np.zeros((0, self.input_dim)), np.zeros(0)

        A = np.eye(self.input_dim)  # accumulated linear map from input
        c = np.zeros(self.input_dim)  # accumulated bias from input
        D_blocks: list[np.ndarray] = []
        g_blocks: list[np.ndarray] = []

        for layer_idx, q in enumerate(q_path):
            W, b = self.weights[layer_idx], self.biases[layer_idx]
            W_hat = W @ A  # effective weight:  (out_l, input_dim)
            b_hat = W @ c + b  # effective bias:    (out_l,)

            # s[i] = -1 if neuron i is active (q=1), +1 if inactive (q=0)
            s = -2.0 * q + 1.0
            D_blocks.append(s[:, None] * W_hat)
            g_blocks.append(-(s * b_hat))

            # Propagate the affine map through active neurons only
            A = q[:, None] * W_hat  # (out_l, input_dim)
            c = q * b_hat  # (out_l,)

        D = np.vstack(D_blocks)
        g = np.concatenate(g_blocks)

        if (
            active_only
            and region.active_indices is not None
            and len(region.active_indices) > 0
        ):
            return D[region.active_indices], g[region.active_indices]

        return D, g

    # ── Local linearisation ───────────────────────────────────────────────────

    def local_affine(self, region: Region) -> tuple[np.ndarray, np.ndarray]:
        """The local affine map ``f(x) = A x + b`` for this region.

        Walks the activation path layer by layer, gating inactive neurons.
        ``A`` has shape ``(last_layer_out_dim, input_dim)`` and ``b`` has shape
        ``(last_layer_out_dim,)``.  This is the network's representation up to
        whichever final layer the partition was built with — typically the
        last hidden ReLU layer's output, post-gating.
        """
        A = np.eye(self.input_dim)
        c = np.zeros(self.input_dim)
        for layer_idx, q in enumerate(region.activation_path):
            W, b = self.weights[layer_idx], self.biases[layer_idx]
            W_hat = W @ A
            b_hat = W @ c + b
            A = q[:, None] * W_hat
            c = q * b_hat
        return A, c

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, X: np.ndarray) -> list[Region | None]:
        """Assign each row of X to its region via a forward pass.

        Points that fall outside all known regions (only possible with
        sparse-mode partitions) are returned as ``None``.

        Time complexity: O(N·L) for the forward pass, O(N·n_regions) worst case
        for lookup (typically O(N) with the hash table).
        """
        X = np.asarray(X, dtype=float)
        N = X.shape[0]

        # Vectorised forward pass: collect activation patterns at every layer
        A = X
        q_per_layer: list[np.ndarray] = []
        for W, b in zip(self.weights, self.biases):
            Z = A @ W.T + b  # (N, out_l)
            Q = Z > 0  # (N, out_l), dtype bool
            q_per_layer.append(Q)
            A = Q * Z

        # Build hash lookup: tuple-of-bytes-per-layer → Region
        lookup: dict[tuple[bytes, ...], Region] = {}
        for r in self.regions:
            key = tuple(q.tobytes() for q in r.activation_path)
            lookup[key] = r

        results: list[Region | None] = []
        for i in range(N):
            key = tuple(
                q_per_layer[layer_idx][i].tobytes()
                for layer_idx in range(self.n_layers)
            )
            results.append(lookup.get(key))
        return results

    # ── Filtering ─────────────────────────────────────────────────────────────

    def regions_at_layer(self, layer: int) -> list[Region]:
        """Return regions whose activation path has exactly ``layer`` layers."""
        return [r for r in self.regions if r.n_layers == layer]

    # ── Construction from a method's RegionFindResult ─────────────────────────

    @classmethod
    def from_result(
        cls,
        result,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
    ) -> Partition:
        """Build a Partition from a ``RegionFindResult`` (any registered method)."""
        patterns = np.asarray(result.patterns)
        offsets = np.asarray(result.offsets, dtype=np.int64)
        centroids = np.asarray(result.centroids)

        n_regions = patterns.shape[0]
        n_layers = len(offsets) - 1

        regions = [
            Region(
                activation_path=[
                    patterns[i, offsets[layer_idx] : offsets[layer_idx + 1]].astype(
                        bool
                    )
                    for layer_idx in range(n_layers)
                ],
                centroid=centroids[i],
            )
            for i in range(n_regions)
        ]
        return cls(regions=regions, weights=weights, biases=biases)
