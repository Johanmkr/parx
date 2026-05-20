"""I/O helpers for iterating over state dicts.

``compute_partition`` consumes one state dict at a time.  When you have many
(e.g. snapshots taken every epoch during training), iterate above
``compute_partition`` using :func:`iter_state_dicts`:

    for ep, sd in iter_state_dicts(source):
        partition = compute_partition(sd, X, method="sparse_julia")
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator


def iter_state_dicts(source: Any) -> Iterator[tuple[Any, dict]]:
    """Yield ``(label, state_dict)`` pairs from epoch-indexed sources.

    Accepted inputs:

    * **Single state dict** — a ``dict`` whose keys end in ``.weight`` /
      ``.bias`` (PyTorch convention).  Yields ``(None, state_dict)``.
    * **Mapping of epoch → state_dict** — a ``dict`` whose values are
      themselves state dicts.  Yields ``(epoch, state_dict)`` in key-sorted
      order.
    * **Sequence/set of state dicts** — yields ``(index, state_dict)`` in
      iteration order (sets are sorted by ``id`` — pass a list for determinism).
    * **Path to a ``.h5`` file** — two layouts auto-detected:
        - **flat** (all parameter datasets at the file root) → single state dict
        - **grouped** (each top-level item is a ``Group`` containing one state
          dict) → one yield per group.  Group names are sorted numerically when
          they look like integers; lexicographically otherwise.

    The h5 path holds the file open for the duration of iteration; consume the
    generator (e.g. via ``list(...)``) before doing anything else if you need
    the handles released sooner.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() not in (".h5", ".hdf5"):
            raise ValueError(
                f"Only .h5/.hdf5 paths are supported; got {path.suffix!r}"
            )
        yield from _iter_h5(path)
        return

    if isinstance(source, dict):
        if _looks_like_state_dict(source):
            yield None, source
            return

        # Mapping of label → state_dict
        try:
            keys = sorted(source.keys())
        except TypeError:
            keys = list(source.keys())
        for k in keys:
            sd = source[k]
            if not _looks_like_state_dict(sd):
                raise TypeError(
                    f"value at key {k!r} is not a state dict "
                    f"(got {type(sd).__name__})"
                )
            yield k, sd
        return

    # Anything else iterable: list, tuple, set, generator
    try:
        items = list(source)
    except TypeError as exc:
        raise TypeError(
            f"Cannot iterate state dicts from {type(source).__name__}"
        ) from exc
    for i, sd in enumerate(items):
        if not _looks_like_state_dict(sd):
            raise TypeError(
                f"item {i} is not a state dict (got {type(sd).__name__})"
            )
        yield i, sd


# ── Internals ─────────────────────────────────────────────────────────────────

def _looks_like_state_dict(obj: Any) -> bool:
    """Heuristic: dict with str keys and at least one ``.weight`` entry."""
    if not isinstance(obj, dict) or not obj:
        return False
    return all(isinstance(k, str) for k in obj) and any(
        k.endswith(".weight") for k in obj
    )


def _h5_sort_key(name: str) -> tuple[int, Any]:
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)


def _iter_h5(path: Path) -> Iterator[tuple[Any, dict]]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for .h5 files: pip install parx[h5]"
        ) from exc

    import numpy as np

    with h5py.File(path, "r") as f:
        groups = [k for k in f if isinstance(f[k], h5py.Group)]
        datasets = [k for k in f if isinstance(f[k], h5py.Dataset)]

        if groups and not datasets:
            for g_name in sorted(groups, key=_h5_sort_key):
                grp = f[g_name]
                sd = {k: np.asarray(grp[k]) for k in grp}
                yield g_name, sd
        else:
            sd = {
                k: np.asarray(f[k])
                for k in f
                if isinstance(f[k], h5py.Dataset)
            }
            yield None, sd
