# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo",
#   "torch",
#   "numpy",
#   "plotly",
#   "parx @ file:///home/johan/Documents/phd/parx",
# ]
# ///

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


# ── imports ───────────────────────────────────────────────────────────────────

@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    # juliacall must be imported before torch — parx handles this at import time
    import parx

    import copy
    import tempfile

    import numpy as np
    import torch
    import torch.nn as nn

    from parx import (
        compute_partition,
        complexity_profile,
        dead_neurons,
        region_size_summary,
        animate_epochs,
        save_partition,
        load_partition,
    )
    from parx.viz import (
        plot_partition_2d,
        plot_region_counts,
        plot_halfspaces,
        plot_partition_slice,
        plot_partition_projection,
        affine_spectral,
    )
    from parx.verify import (
        check_no_overlaps,
        check_covers_space,
        check_regions_nonempty,
    )

    return (
        affine_spectral,
        animate_epochs,
        check_covers_space,
        check_no_overlaps,
        check_regions_nonempty,
        complexity_profile,
        compute_partition,
        copy,
        dead_neurons,
        load_partition,
        nn,
        np,
        parx,
        plot_halfspaces,
        plot_partition_2d,
        plot_partition_projection,
        plot_partition_slice,
        plot_region_counts,
        region_size_summary,
        save_partition,
        tempfile,
        torch,
    )


# ── title ─────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    # parx quick tour — Plotly backend

    A ReLU network is piecewise-affine: fix which neurons fire (its
    "activation pattern") and the rest of the function is just an affine map
    `f(x) = A @ x + b`. `parx` **exactly enumerates** these pieces — the
    maximal convex "linear regions" of input space on which the activation
    pattern, and therefore the affine map, is constant.

    This matters for a few reasons:

    - **Interpretability** — the affine map on any single region tells you
      exactly what the network computes there, with no approximation.
    - **Complexity / expressivity** — the number of regions is a standard
      proxy for how expressive a network is; `parx` can track it across
      training checkpoints.
    - **Verification** — geometric properties (region volume, boundedness,
      Lipschitz constant per region) are computable exactly rather than
      estimated, and `parx.verify` can sanity-check the result itself.

    This notebook walks the public Python API end-to-end: build a toy
    network, compute its partition, inspect regions, compare enumeration
    methods, run analysis and verification functions, visualize a 2D and a
    higher-dimensional example, and save/reload a partition. All figures use
    `parx.viz`'s **Plotly backend** (`backend="plotly"`, the default) —
    interactive figures with hover tooltips and, for the epoch animation, a
    play button and slider.

    **Run interactively:** `uv run notebooks/demo_plotly.py` opens this
    notebook in marimo's editor with both code and outputs visible (the
    inline `# /// script` block at the top declares its dependencies, so
    `uv` builds an isolated environment automatically).
    """)
    return


# ── build a toy network ─────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## Build a toy network

    A small 2-input MLP with two ReLU hidden layers of width 4 — small enough
    that the resulting partition has only a handful of regions, so every plot
    below stays readable.
    """)
    return


@app.cell
def __(nn, torch):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(2, 4), nn.ReLU(),
        nn.Linear(4, 4), nn.ReLU(),
        nn.Linear(4, 1),
    )
    state_dict = model.state_dict()
    return model, state_dict


# ── compute the partition ────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## Compute the partition

    `compute_partition` accepts an `nn.Module`, a `state_dict`, or a path to a
    `.pth`/`.h5` file. Sparse methods (the default, `"sparse_julia"`) scan a
    batch of input points for distinct activation patterns — fast, but may
    miss low-density regions. `"exact_julia"` / `"exact_julia_fast"` instead
    run a complete DFS + facet-flip enumeration from a single starting point.
    """)
    return


@app.cell
def __(compute_partition, mo, np, state_dict):
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(400, 2))

    with mo.status.spinner(title="Computing partition …"):
        partition = compute_partition(state_dict, X, method="sparse_julia")

    mo.md(f"Found **{len(partition)}** linear regions from {X.shape[0]} sample points.")
    return X, partition, rng


# ── inspect the Partition object ─────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## Inspecting the `Partition` object

    A `Partition` is a flat list of `Region`s plus the network weights needed
    to reconstruct geometry on demand — no Julia call is required at query
    time.
    """)
    return


@app.cell
def __(X, mo, partition):
    _region = partition.regions[0]
    _D, _g = partition.halfspaces(_region)
    _A, _b = partition.local_affine(_region)
    _routed = partition.route(X[:5])
    _region_index = {id(r): i for i, r in enumerate(partition.regions)}
    _routed_idx = [_region_index.get(id(r)) for r in _routed]

    mo.md(f"""
    Looking at `partition.regions[0]`, activation pattern
    `{[q.astype(int).tolist() for q in _region.activation_path]}`:

    - `partition.halfspaces(region)` → `D` shape `{_D.shape}`, `g` shape
      `{_g.shape}` — the system `D @ x <= g` defining the polytope.
    - `partition.local_affine(region)` → `A` shape `{_A.shape}`, `b` shape
      `{_b.shape}` — the local affine map `f(x) = A @ x + b`.
    - `partition.route(X[:5])` → region index for each of the first 5 points:
      `{_routed_idx}`.
    """)
    return


# ── analysis functions ───────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## Analysis functions

    A few of the built-in statistics — no Julia calls, cheap enough to run on
    every partition.
    """)
    return


@app.cell
def __(complexity_profile, dead_neurons, mo, partition, region_size_summary):
    _profile = complexity_profile(partition)
    _dead = dead_neurons(partition)
    _size = region_size_summary(partition)

    mo.md(f"""
    **`complexity_profile(partition)`**

    - regions per layer (cumulative depth): `{_profile['regions_per_layer']}`
    - total constraints: `{_profile['total_constraints']}`
      (mean {_profile['mean_constraints_per_region']:.1f} per region)

    **`dead_neurons(partition)`** — neurons never active in any region:
    `{_dead if _dead else 'none'}`

    **`region_size_summary(partition)`** — Chebyshev-radius statistics
    (one LP per region):

    - min `{_size['min']:.3f}`, median `{_size['median']:.3f}`,
      mean `{_size['mean']:.3f}`, max `{_size['max']:.3f}`
    - fraction bounded: `{_size['fraction_bounded']:.0%}`
    """)
    return


# ── comparing enumeration methods ─────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## Comparing enumeration methods

    `parx` ships several interchangeable region-finding methods
    (`parx.list_methods()`). Sparse methods scan a finite batch of points and
    can miss regions that no sample lands in; exact methods run a complete
    DFS + facet-flip search from a single starting point and are guaranteed
    to find every region, at the cost of scaling combinatorially with network
    width. On a network this small, the two should agree exactly.
    """)
    return


@app.cell
def __(X, compute_partition, mo, parx, partition, state_dict):
    _exact = compute_partition(state_dict, X, method="exact_julia_fast")
    _agree = len(_exact) == len(partition)

    mo.md(f"""
    `parx.list_methods()` → `{parx.list_methods()}`

    - `sparse_julia` (used above): **{len(partition)}** regions found from
      {X.shape[0]} samples
    - `exact_julia_fast`: **{len(_exact)}** regions found by exhaustive
      DFS + facet-flip search

    {"✅ They agree" if _agree else "⚠️ They differ"} on this small network —
    exact methods are complete by construction, so any mismatch would mean
    the sparse sample missed a region.
    """)
    return (_exact,)


# ── visualization ─────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## Visualizing with the Plotly backend

    Every `parx.viz` plotting function accepts `backend: Literal["plotly",
    "matplotlib"] = "plotly"`. Plotly is the default, so it is omitted below
    except once for clarity — figures are returned directly from the cell and
    marimo renders them inline, interactively.
    """)
    return


@app.cell
def __(partition, plot_partition_2d):
    fig_partition = plot_partition_2d(
        partition, domain=((-1.0, 1.0), (-1.0, 1.0)), backend="plotly"
    )
    fig_partition
    return (fig_partition,)


@app.cell
def __(partition, plot_region_counts):
    fig_counts = plot_region_counts(partition)
    fig_counts
    return (fig_counts,)


@app.cell
def __(partition, plot_halfspaces):
    fig_halfspaces = plot_halfspaces(
        partition, partition.regions[0], x_range=(-1.0, 1.0), y_range=(-1.0, 1.0)
    )
    fig_halfspaces
    return (fig_halfspaces,)


# ── custom coloring ────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ### Custom coloring

    `color_by` accepts any `(partition, region) -> float` callable. Built-ins
    in `parx.viz` include `affine_frobenius` (the default, ‖A‖_F),
    `affine_spectral` (the local Lipschitz constant — the top singular value
    of `A`), `affine_det`, and `active_neuron_count`. Swap the colorscale
    with `colorscale=`, or hand `plot_partition_2d` a fixed palette via
    `colors=region_palette(partition, scheme=...)`.
    """)
    return


@app.cell
def __(affine_spectral, partition, plot_partition_2d):
    fig_spectral = plot_partition_2d(
        partition,
        domain=((-1.0, 1.0), (-1.0, 1.0)),
        color_by=affine_spectral,
        colorscale="Plasma",
        backend="plotly",
    )
    fig_spectral
    return (fig_spectral,)


# ── higher-dimensional partitions ─────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## Higher-dimensional partitions: slice & projection

    `plot_partition_2d` requires `input_dim == 2`. For higher-dimensional
    networks, `plot_partition_slice` fixes all but two input dimensions and
    renders the induced 2D slice exactly; `plot_partition_projection`
    instead projects the halfspace normals onto an arbitrary 2D subspace —
    an approximation, since projecting a polytope's dual representation
    isn't generally exact (see the function's docstring). Below is a
    3-input network, exactly enumerated with `exact_julia_fast`.
    """)
    return


@app.cell
def __(compute_partition, mo, nn, np, torch):
    torch.manual_seed(1)
    model_3d = nn.Sequential(
        nn.Linear(3, 4), nn.ReLU(),
        nn.Linear(4, 1),
    )
    partition_3d = compute_partition(
        model_3d.state_dict(), np.zeros((1, 3)), method="exact_julia_fast"
    )
    mo.md(f"3-input network → **{len(partition_3d)}** regions in ℝ³.")
    return model_3d, partition_3d


@app.cell
def __(partition_3d, plot_partition_slice):
    fig_slice = plot_partition_slice(
        partition_3d, free_dims=(0, 1), fixed_values={2: 0.0}, backend="plotly"
    )
    fig_slice
    return (fig_slice,)


@app.cell
def __(np, partition_3d, plot_partition_projection):
    projection = np.eye(3)[:, :2]  # drop the 3rd input dimension
    fig_proj = plot_partition_projection(partition_3d, projection, backend="plotly")
    fig_proj
    return fig_proj, projection


# ── training-time tracking ────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## Training-time tracking with `animate_epochs`

    `parx` doesn't run any training itself — the caller computes one
    `Partition` per checkpoint (see `parx.iter_state_dicts` for iterating
    saved checkpoints) and hands the list to `animate_epochs`. Here we fake
    three checkpoints by adding growing Gaussian noise to the same base
    weights, just to have more than one partition to animate.
    """)
    return


@app.cell
def __(X, compute_partition, copy, np, state_dict, torch):
    _rng = np.random.default_rng(1)
    partitions = []
    epoch_labels = []
    for _epoch, _scale in enumerate([0.0, 0.3, 0.6]):
        _sd = copy.deepcopy(state_dict)
        for _k, _v in _sd.items():
            _noise = torch.tensor(_rng.normal(0.0, _scale, size=tuple(_v.shape)), dtype=_v.dtype)
            _sd[_k] = _v + _noise
        partitions.append(compute_partition(_sd, X, method="sparse_julia"))
        epoch_labels.append(f"epoch {_epoch}")
    return epoch_labels, partitions


@app.cell
def __(animate_epochs, epoch_labels, partitions):
    fig_anim = animate_epochs(
        partitions,
        epoch_labels=epoch_labels,
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        backend="plotly",
    )
    fig_anim
    return (fig_anim,)


# ── verification ──────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## Verifying correctness

    `parx.verify` runs geometric sanity checks against the reconstructed
    halfspace systems — useful after changing a region-finding method, or
    just to build confidence that a partition is correct.
    """)
    return


@app.cell
def __(
    check_covers_space,
    check_no_overlaps,
    check_regions_nonempty,
    mo,
    partition,
    rng,
):
    _X_test = rng.uniform(-1.0, 1.0, size=(2000, 2))
    _no_overlap, _ = check_no_overlaps(partition, _X_test)
    _covers, _counts = check_covers_space(partition, _X_test)
    _nonempty, _bad, _radii = check_regions_nonempty(partition)

    mo.md(f"""
    - `check_no_overlaps`: **{"pass ✅" if _no_overlap else "FAIL ✗"}** — no
      sample point lands in two regions' interiors.
    - `check_covers_space`: **{"pass ✅" if _covers else f"{int((_counts == 0).sum())} uncovered points"}**
      — every sample point belongs to exactly one region (sparse partitions
      can miss coverage; this sample is dense enough to pass).
    - `check_regions_nonempty`: **{"pass ✅" if _nonempty else f"{len(_bad)} degenerate regions"}**
      — every region has a strictly positive Chebyshev radius (min radius
      found: `{_radii.min():.4f}`).
    """)
    return


# ── save & reload ─────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## Save & reload

    `save_partition`/`load_partition` serialize a `Partition` to a `.npz`
    bundle — NumPy-native, no Julia required to reload. A saved partition
    can be analyzed or visualized in an environment that never installed
    Julia at all.
    """)
    return


@app.cell
def __(load_partition, mo, partition, save_partition, tempfile):
    _path = tempfile.mktemp(suffix=".npz")
    save_partition(partition, _path)
    reloaded = load_partition(_path)

    mo.md(f"""
    Saved **{len(partition)}** regions to `{_path}` and reloaded them:
    `len(reloaded) == len(partition)` → **{len(reloaded) == len(partition)}**
    """)
    return (reloaded,)


# ── closing note ──────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## Matplotlib backend

    An identical notebook using the matplotlib backend lives at
    `notebooks/demo_plt.py` (also runnable via `uv run notebooks/demo_plt.py`).
    The only difference is `backend="matplotlib"` passed to each `parx.viz`
    call:

    - `plot_partition_2d`, `plot_region_counts`, `plot_halfspaces`,
      `plot_partition_slice`, `plot_partition_projection` return a static
      `matplotlib.figure.Figure` instead of an interactive
      `plotly.graph_objects.Figure` — no hover tooltips, but easily saved
      with `fig.savefig(...)`.
    - `animate_epochs` returns a `matplotlib.animation.FuncAnimation` instead
      of a Plotly figure with a play/pause button and slider — playback in a
      notebook requires `.to_jshtml()`; use `animate_epochs_video(...)` to
      export to MP4/GIF instead.
    - The matplotlib backend requires `pip install "parx[animate]"`.
    """)
    return


if __name__ == "__main__":
    import os

    # Running this file directly (e.g. `uv run notebooks/demo_plotly.py`)
    # normally executes the notebook headlessly with nothing shown. Setting
    # this launches marimo's editor instead, so code and outputs are visible.
    os.environ.setdefault("MARIMO_SCRIPT_EDIT", "1")
    app.run()
