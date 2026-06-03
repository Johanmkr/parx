"""Analysis and statistics API for ReLU network partitions.

This module provides:

- **Cheap** (no LP): neuron activity statistics, complexity profile.
- **Moderate** (one LP per region): Chebyshev radii, size summary.
- **Expensive** (Monte Carlo): volume estimation — *do not call in tight loops*.

All functions accept a :class:`~parx.partition.Partition` object and return
plain NumPy arrays, scalars, or dicts.  No Julia is imported here.
"""

from __future__ import annotations

import numpy as np

from parx._lp import chebyshev_center
from parx.partition import Partition
from parx.region import Region

# ── 9a: Neuron activity statistics ───────────────────────────────────────────


def neuron_activity_rates(partition: Partition) -> dict[int, np.ndarray]:
    """Fraction of regions where each neuron is active, per layer.

    For layer ``l`` (0-based), returns an array of shape ``(layer_width,)``
    where entry ``j`` is the fraction of regions with neuron ``j`` active.

    Complexity: O(n_regions × total_neurons).  No LP calls.

    Parameters
    ----------
    partition:
        The partition to analyse.

    Returns
    -------
    dict mapping layer index (int) to ndarray of shape ``(layer_width,)``.
    """
    if not partition.regions:
        n_layers = partition.n_layers
        return {
            layer: np.zeros(partition.weights[layer].shape[0], dtype=float)
            for layer in range(n_layers)
        }

    n_regions = len(partition.regions)
    # Determine layer widths from the first region's activation path.
    sample_path = partition.regions[0].activation_path
    n_layers = len(sample_path)

    # Accumulate sum of active indicator per layer.
    totals: list[np.ndarray] = [
        np.zeros(sample_path[layer].shape[0], dtype=float)
        for layer in range(n_layers)
    ]
    for region in partition.regions:
        for layer, q in enumerate(region.activation_path):
            totals[layer] += q.astype(float)

    return {layer: totals[layer] / n_regions for layer in range(n_layers)}


def dead_neurons(
    partition: Partition,
    *,
    threshold: float = 0.0,
) -> list[tuple[int, int]]:
    """Neurons with activity rate at or below *threshold*.

    A neuron is "dead" when it is active in at most ``threshold`` fraction of
    regions.  The default ``threshold=0.0`` finds neurons that are **never**
    active across all known regions.

    Parameters
    ----------
    partition:
        The partition to analyse.
    threshold:
        Upper bound on activity rate for a neuron to be reported.  Use
        ``0.0`` for strictly dead neurons, or a small positive value (e.g.
        ``0.05``) for rarely-active neurons.

    Returns
    -------
    List of ``(layer, neuron_index)`` pairs, sorted by layer then index.
    """
    rates = neuron_activity_rates(partition)
    result: list[tuple[int, int]] = []
    for layer_idx, arr in rates.items():
        for j, rate in enumerate(arr):
            if rate <= threshold:
                result.append((layer_idx, j))
    return result


def always_active_neurons(
    partition: Partition,
    *,
    threshold: float = 1.0,
) -> list[tuple[int, int]]:
    """Neurons active in (at least) *threshold* fraction of regions.

    The default ``threshold=1.0`` finds neurons that are **always** active
    across all known regions.

    Parameters
    ----------
    partition:
        The partition to analyse.
    threshold:
        Lower bound on activity rate.  Use ``1.0`` for universally active
        neurons, or a value like ``0.95`` for nearly-always-active neurons.

    Returns
    -------
    List of ``(layer, neuron_index)`` pairs, sorted by layer then index.
    """
    rates = neuron_activity_rates(partition)
    result: list[tuple[int, int]] = []
    for layer_idx, arr in rates.items():
        for j, rate in enumerate(arr):
            if rate >= threshold:
                result.append((layer_idx, j))
    return result


# ── 9b: Partition complexity profile ─────────────────────────────────────────


def complexity_profile(partition: Partition) -> dict:
    """Structural complexity statistics for the partition.

    Returns
    -------
    dict with keys:

    ``n_regions`` : int
        Total number of regions.
    ``n_layers`` : int
        Number of hidden layers (depth of activation paths).
    ``input_dim`` : int
        Input dimensionality of the network.
    ``regions_per_layer`` : list[int]
        At depth ``l`` (1-based), the number of unique activation-path
        *prefixes* of length ``l``.  ``regions_per_layer[0]`` corresponds to
        depth 1 (after the first hidden layer).
    ``total_neurons`` : int
        Sum of widths across all hidden layers.
    ``total_constraints`` : int
        Sum over all regions of the number of halfspace constraints.
    ``mean_constraints_per_region`` : float
        Average number of constraints per region.
    """
    n_regions = len(partition.regions)
    n_layers = partition.n_layers
    input_dim = partition.input_dim
    total_neurons = sum(w.shape[0] for w in partition.weights)

    # regions_per_layer: unique prefixes at each depth.
    regions_per_layer: list[int] = []
    for depth in range(1, n_layers + 1):
        prefixes: set[tuple[bytes, ...]] = set()
        for region in partition.regions:
            prefix = tuple(q.tobytes() for q in region.activation_path[:depth])
            prefixes.add(prefix)
        regions_per_layer.append(len(prefixes))

    # total_constraints: sum of D.shape[0] for each region.
    total_constraints = 0
    for region in partition.regions:
        D, _ = partition.halfspaces(region)
        total_constraints += D.shape[0]

    mean_constraints = total_constraints / n_regions if n_regions > 0 else 0.0

    return {
        "n_regions": n_regions,
        "n_layers": n_layers,
        "input_dim": input_dim,
        "regions_per_layer": regions_per_layer,
        "total_neurons": total_neurons,
        "total_constraints": total_constraints,
        "mean_constraints_per_region": float(mean_constraints),
    }


# ── 9c: Region size proxies ──────────────────────────────────────────────────


def region_chebyshev_radii(
    partition: Partition,
    *,
    max_radius: float = 1e3,
) -> np.ndarray:
    """Chebyshev (largest inscribed ball) radius for every region.

    One LP is solved per region via :func:`parx._lp.chebyshev_center`.
    A radius of ``0.0`` indicates an empty or degenerate region (a bug for
    exact-mode partitions).  A radius at or near ``max_radius`` indicates an
    unbounded region.

    Parameters
    ----------
    partition:
        The partition to analyse.
    max_radius:
        Upper bound on the radius passed to the LP solver.  Unbounded regions
        will have their radius capped at this value.

    Returns
    -------
    ndarray of shape ``(n_regions,)``.
    """
    radii = np.zeros(len(partition), dtype=float)
    for i, region in enumerate(partition.regions):
        D, g = partition.halfspaces(region)
        _, r = chebyshev_center(D, g, max_radius=max_radius)
        radii[i] = r
    return radii


def region_size_summary(
    partition: Partition,
    *,
    max_radius: float = 1e3,
) -> dict:
    """Descriptive statistics of per-region Chebyshev radii.

    Calls :func:`region_chebyshev_radii` (one LP per region) and summarises
    the resulting distribution.

    Parameters
    ----------
    partition:
        The partition to analyse.
    max_radius:
        Passed through to :func:`region_chebyshev_radii`.

    Returns
    -------
    dict with keys:

    ``min`` : float
    ``median`` : float
    ``mean`` : float
    ``max`` : float
    ``fraction_bounded`` : float
        Fraction of regions whose radius is strictly below ``max_radius``.
    """
    radii = region_chebyshev_radii(partition, max_radius=max_radius)
    if len(radii) == 0:
        return {
            "min": float("nan"),
            "median": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
            "fraction_bounded": float("nan"),
        }
    fraction_bounded = float(np.mean(radii < max_radius))
    return {
        "min": float(np.min(radii)),
        "median": float(np.median(radii)),
        "mean": float(np.mean(radii)),
        "max": float(np.max(radii)),
        "fraction_bounded": fraction_bounded,
    }


# ── 9d: Volume estimation (Monte Carlo — SLOW) ───────────────────────────────


def region_volume_estimate(
    partition: Partition,
    region: Region,
    *,
    n_samples: int = 10_000,
    seed: int | None = None,
) -> float:
    """Estimate the hypervolume of a region via rejection sampling.

    .. warning::
        This function is **slow** — it calls the halfspace system for every
        sample and scales poorly with dimensionality.  For ``input_dim > 10``
        or large ``n_samples`` the runtime can be minutes.  Do not call in
        tight loops; use :func:`partition_volume_estimates` for batch
        processing.

    Algorithm
    ---------
    A bounding box is constructed from the Chebyshev ball of radius ``r``
    centred at ``region.centroid``: the axis-aligned box
    ``[x0 - r, x0 + r]^d``.  Points are sampled uniformly in this box and
    tested against ``D @ x <= g``.  The volume estimate is::

        volume = acceptance_rate × (2r)^d

    For unbounded regions the Chebyshev radius is capped at ``max_radius=1e3``
    internally, which can produce misleadingly large or small estimates.

    Parameters
    ----------
    partition:
        The partition containing the region.
    region:
        The specific region to estimate.
    n_samples:
        Number of candidate points to draw.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    float — estimated hypervolume.
    """
    rng = np.random.default_rng(seed)
    D, g = partition.halfspaces(region)
    x0 = region.centroid
    d = len(x0)

    # Find the Chebyshev radius as the bounding scale.
    _, r = chebyshev_center(D, g, max_radius=1e3)
    if r == 0.0:
        return 0.0
    if not np.isfinite(r):
        # No constraints at all — polytope is all of R^d; return inf.
        return float("inf")

    # Sample uniformly in the L-infinity ball [x0 - r, x0 + r]^d.
    samples = rng.uniform(x0 - r, x0 + r, size=(n_samples, d))

    # Accept samples satisfying all halfspace constraints.
    if D.shape[0] == 0:
        acceptance_rate = 1.0
    else:
        inside = np.all(samples @ D.T <= g, axis=1)
        acceptance_rate = float(np.mean(inside))

    box_volume = (2.0 * r) ** d
    return acceptance_rate * box_volume


def partition_volume_estimates(
    partition: Partition,
    *,
    n_samples: int = 5_000,
    seed: int | None = None,
) -> np.ndarray:
    """Volume estimate for every region in the partition.

    .. warning::
        This function is **very slow** — it calls :func:`region_volume_estimate`
        once per region.  For a partition with hundreds of regions and
        ``input_dim > 5``, expect minutes of runtime.

    Parameters
    ----------
    partition:
        The partition to analyse.
    n_samples:
        Number of Monte Carlo samples per region.
    seed:
        Base random seed.  Each region uses a derived seed for reproducibility.

    Returns
    -------
    ndarray of shape ``(n_regions,)`` with per-region volume estimates.
    """
    volumes = np.zeros(len(partition), dtype=float)
    for i, region in enumerate(partition.regions):
        # Derive a per-region seed so results are reproducible regardless of
        # the order regions are processed.
        region_seed = None if seed is None else seed + i
        volumes[i] = region_volume_estimate(
            partition,
            region,
            n_samples=n_samples,
            seed=region_seed,
        )
    return volumes


# ── 9e: Epoch-over-epoch analysis ────────────────────────────────────────────


def complexity_over_epochs(
    partitions: list[Partition],
    labels=None,
) -> dict[str, list]:
    """Track partition complexity statistics across a sequence of epochs.

    Computes :func:`complexity_profile`, mean Chebyshev radius (via
    :func:`region_chebyshev_radii`), and dead-neuron count for each partition
    in the list.

    Parameters
    ----------
    partitions:
        Ordered list of partitions (one per epoch or checkpoint).
    labels:
        Optional epoch labels.  If ``None``, defaults to ``[0, 1, 2, ...]``.

    Returns
    -------
    dict with keys:

    ``labels`` : list
        Epoch identifiers.
    ``n_regions`` : list[int]
        Number of regions per epoch.
    ``mean_chebyshev_radius`` : list[float]
        Mean Chebyshev radius across regions per epoch.
    ``dead_neuron_count`` : list[int]
        Number of dead neurons (activity rate == 0) per epoch.
    """
    if labels is None:
        labels = list(range(len(partitions)))
    else:
        labels = list(labels)

    n_regions_list: list[int] = []
    mean_radius_list: list[float] = []
    dead_count_list: list[int] = []

    for partition in partitions:
        profile = complexity_profile(partition)
        n_regions_list.append(profile["n_regions"])

        radii = region_chebyshev_radii(partition)
        mean_radius_list.append(float(np.mean(radii)) if len(radii) > 0 else 0.0)

        dead = dead_neurons(partition, threshold=0.0)
        dead_count_list.append(len(dead))

    return {
        "labels": labels,
        "n_regions": n_regions_list,
        "mean_chebyshev_radius": mean_radius_list,
        "dead_neuron_count": dead_count_list,
    }
