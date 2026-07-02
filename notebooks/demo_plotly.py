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

    import numpy as np
    import torch
    import torch.nn as nn

    from parx import (
        compute_partition,
        complexity_profile,
        dead_neurons,
        region_size_summary,
        animate_epochs,
    )
    from parx.viz import plot_partition_2d, plot_region_counts, plot_halfspaces

    return (
        animate_epochs,
        complexity_profile,
        compute_partition,
        copy,
        dead_neurons,
        nn,
        np,
        parx,
        plot_halfspaces,
        plot_partition_2d,
        plot_region_counts,
        region_size_summary,
        torch,
    )


# ── title ─────────────────────────────────────────────────────────────────────

@app.cell
def __(mo):
    mo.md(r"""
    # parx quick tour — Plotly backend

    `parx` exactly enumerates the **linear (polyhedral activation) regions**
    of a ReLU network: the maximal convex pieces of input space on which the
    network is a single fixed affine map.

    This notebook walks through the public Python API end-to-end — building a
    toy network, computing its partition, inspecting the result, running a
    couple of analysis functions, and visualizing everything. All figures use
    `parx.viz`'s **Plotly backend**, which is the default (`backend="plotly"`)
    for every plotting function: interactive figures with hover tooltips and,
    for the epoch animation, a play button and slider.
    """)
    return


# ── build a toy network ─────────────────────────────────────────────────────────

@app.cell
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

@app.cell
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

@app.cell
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

@app.cell
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


# ── visualization ─────────────────────────────────────────────────────────────────

@app.cell
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


# ── training-time tracking ────────────────────────────────────────────────────────

@app.cell
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


# ── closing note ──────────────────────────────────────────────────────────────────

@app.cell
def __(mo):
    mo.md(r"""
    ## Matplotlib backend

    An identical notebook using the matplotlib backend lives at
    `notebooks/demo_plt.py`. The only difference is `backend="matplotlib"`
    passed to each `parx.viz` call:

    - `plot_partition_2d`, `plot_region_counts`, `plot_halfspaces` return a
      static `matplotlib.figure.Figure` instead of an interactive
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
    app.run()
