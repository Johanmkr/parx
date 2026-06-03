"""Save and load Partition objects to/from .npz files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from parx.methods import RegionFindResult
from parx.partition import Partition

_FORMAT_VERSION = 1


def save_partition(partition: Partition, path: str | Path) -> None:
    """Save a Partition to a .npz file.

    Parameters
    ----------
    partition:
        The Partition to serialise.
    path:
        Destination file path.  A ``.npz`` extension is appended by
        ``np.savez`` if not already present.
    """
    regions = partition.regions
    n_regions = len(regions)

    # Build offsets from the activation-path widths of the first region.
    # All regions share the same layer widths, so region 0 is representative.
    if n_regions == 0:
        # Edge case: empty partition — infer layer widths from weights.
        layer_widths = [w.shape[0] for w in partition.weights]
    else:
        layer_widths = [q.shape[0] for q in regions[0].activation_path]

    n_layers = len(layer_widths)
    total_bits = sum(layer_widths)

    offsets = np.zeros(n_layers + 1, dtype=np.int64)
    for l, w in enumerate(layer_widths):  # noqa: E741
        offsets[l + 1] = offsets[l] + w

    # Reconstruct patterns matrix
    patterns = np.zeros((n_regions, total_bits), dtype=np.int8)
    for i, region in enumerate(regions):
        for l, q in enumerate(region.activation_path):  # noqa: E741
            patterns[i, offsets[l] : offsets[l + 1]] = q.astype(np.int8)

    # Reconstruct centroids matrix
    input_dim = partition.input_dim
    centroids = np.zeros((n_regions, input_dim), dtype=np.float64)
    for i, region in enumerate(regions):
        centroids[i] = region.centroid

    arrays: dict[str, np.ndarray] = {
        "version": np.int64(_FORMAT_VERSION),
        "patterns": patterns,
        "offsets": offsets,
        "centroids": centroids,
    }

    # Weights and biases
    for i, (w, b) in enumerate(zip(partition.weights, partition.biases)):
        arrays[f"weights_{i}"] = np.asarray(w, dtype=np.float64)
        arrays[f"biases_{i}"] = np.asarray(b, dtype=np.float64)

    # Optional: active_indices and bounded (exact-method partitions)
    has_active = any(r.active_indices is not None for r in regions)
    if has_active:
        # Build flat active_indices and active_offsets
        active_parts: list[np.ndarray] = []
        active_offsets = np.zeros(n_regions + 1, dtype=np.int64)
        for i, region in enumerate(regions):
            if region.active_indices is not None:
                part = np.asarray(region.active_indices, dtype=np.int32)
            else:
                part = np.empty(0, dtype=np.int32)
            active_parts.append(part)
            active_offsets[i + 1] = active_offsets[i] + len(part)

        active_indices_flat = (
            np.concatenate(active_parts)
            if active_parts
            else np.empty(0, dtype=np.int32)
        )
        arrays["active_indices_flat"] = active_indices_flat
        arrays["active_offsets"] = active_offsets

    # Always save bounded when there are regions so we can round-trip the flag.
    if n_regions > 0:
        arrays["bounded"] = np.array([r.bounded for r in regions], dtype=bool)

    np.savez(path, **arrays)


def load_partition(path: str | Path) -> Partition:
    """Load a Partition from a .npz file produced by :func:`save_partition`.

    Parameters
    ----------
    path:
        Path to the ``.npz`` file.

    Returns
    -------
    Partition

    Raises
    ------
    ValueError
        If the file uses an unsupported format version.
    """
    data = np.load(path, allow_pickle=False)

    version = int(data["version"])
    if version != _FORMAT_VERSION:
        raise ValueError(
            f"Unsupported parx partition format version: {data['version']}"
        )

    patterns = data["patterns"]
    offsets = data["offsets"]
    centroids = data["centroids"]

    # Reconstruct weights and biases
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    i = 0
    while f"weights_{i}" in data:
        weights.append(data[f"weights_{i}"])
        biases.append(data[f"biases_{i}"])
        i += 1

    # Optional fields
    active_indices_flat: np.ndarray | None = (
        data["active_indices_flat"] if "active_indices_flat" in data else None
    )
    active_offsets: np.ndarray | None = (
        data["active_offsets"] if "active_offsets" in data else None
    )
    bounded: np.ndarray | None = data["bounded"] if "bounded" in data else None

    result = RegionFindResult(
        patterns=patterns,
        offsets=offsets,
        centroids=centroids,
        active_indices_flat=active_indices_flat,
        active_offsets=active_offsets,
        bounded=bounded,
    )
    return Partition.from_result(result, weights, biases)
