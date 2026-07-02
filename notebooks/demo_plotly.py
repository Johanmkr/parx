# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo",
#   "torch",
#   "numpy",
#   "plotly",
#   "parx @ file:.",
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

    This notebook walks the public Python API end-to-end using a single
    running example: a small classifier trained on the **two moons**
    dataset. We compute both the **sparse** and **exact** partitions
    **before, during, and after training**, look at the partition **layer by
    layer**, animate the whole training run, then continue into the rest of
    the API — inspecting a `Partition`, running analysis/verification
    functions, visualizing a higher-dimensional example, and saving/reloading
    a partition. All figures use `parx.viz`'s **Plotly backend**
    (`backend="plotly"`, the default) — interactive figures with hover
    tooltips and, for the epoch animation, a play button and slider.

    **Run interactively:** `uv run notebooks/demo_plotly.py` opens this
    notebook in marimo's editor with both code and outputs visible (the
    inline `# /// script` block at the top declares its dependencies, so
    `uv` builds an isolated environment automatically).
    """)
    return


# ── the dataset ───────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## The dataset: two interleaving moons

    Two interleaving half-circles of points, each labelled by which moon it
    belongs to — a standard nonlinear classification benchmark: no straight
    line separates the classes, so a network needs at least one hidden layer
    to fit it. Generated here with plain NumPy (no extra dependency).
    """)
    return


@app.cell
def __(np):
    def make_two_moons(n_samples=240, noise=0.15, seed=0):
        rng = np.random.default_rng(seed)
        n0 = n_samples // 2
        n1 = n_samples - n0
        theta0 = rng.uniform(0, np.pi, n0)
        upper = np.stack([np.cos(theta0), np.sin(theta0)], axis=1)
        theta1 = rng.uniform(0, np.pi, n1)
        lower = np.stack([1 - np.cos(theta1), 1 - np.sin(theta1) - 0.5], axis=1)
        pts = np.concatenate([upper, lower], axis=0)
        pts = pts + rng.normal(0.0, noise, size=pts.shape)
        labels = np.concatenate([np.zeros(n0), np.ones(n1)])
        perm = rng.permutation(n_samples)
        return pts[perm].astype(np.float32), labels[perm].astype(np.float32)

    X_moons, y_moons = make_two_moons()
    return X_moons, make_two_moons, y_moons


# ── the network ───────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## The network: a small classifier

    A 2-input MLP with two ReLU hidden layers of width 8 (16 neurons total)
    and a single logit output — small enough to enumerate exactly in
    milliseconds, expressive enough to fit the moons.
    """)
    return


@app.cell
def __(nn, torch):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(2, 8), nn.ReLU(),
        nn.Linear(8, 8), nn.ReLU(),
        nn.Linear(8, 1),
    )
    return (model,)


# ── training with checkpoints ─────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## Training with checkpoints

    `parx` doesn't run any training itself — the caller trains however it
    likes and hands a `state_dict` (or several) to `compute_partition`. Here
    we train the classifier for real with Adam + BCE loss, and save three
    checkpoints: the random **initialization** ("before"), an early,
    partially-fit epoch ("during"), and the final, converged network
    ("after"). Everything below computes both partitioning methods for all
    three.
    """)
    return


@app.cell
def __(X_moons, copy, mo, model, nn, torch, y_moons):
    _Xt = torch.tensor(X_moons)
    _yt = torch.tensor(y_moons).unsqueeze(1)
    _opt = torch.optim.Adam(model.parameters(), lr=0.05)
    _loss_fn = nn.BCEWithLogitsLoss()

    n_epochs = 300
    mid_epoch = 10
    checkpoints = {"before": copy.deepcopy(model.state_dict())}
    for _epoch in range(n_epochs):
        _opt.zero_grad()
        _out = model(_Xt)
        _loss = _loss_fn(_out, _yt)
        _loss.backward()
        _opt.step()
        if _epoch == mid_epoch:
            checkpoints["during"] = copy.deepcopy(model.state_dict())
    checkpoints["after"] = copy.deepcopy(model.state_dict())

    with torch.no_grad():
        final_acc = ((model(_Xt) > 0).float() == _yt).float().mean().item()

    mo.md(f"""
    Trained for **{n_epochs}** epochs (Adam, BCE loss) → final training
    accuracy **{final_acc:.1%}**. Checkpoints captured at epoch **0**
    (before), epoch **{mid_epoch}** (during), and epoch **{n_epochs}**
    (after).
    """)
    return checkpoints, mid_epoch, n_epochs


# ── sparse vs. exact, per checkpoint ──────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## Sparse vs. exact, at every checkpoint

    One shared sample grid — a padded box around the moons' bounding box —
    is handed to both `"sparse_julia"` (scans the sample for distinct
    activation patterns) and `"exact_julia_fast"` (a complete DFS +
    facet-flip enumeration) for each of the three checkpoints.
    """)
    return


@app.cell
def __(X_moons, np):
    rng = np.random.default_rng(1)
    _pad = 0.6
    x_range = (float(X_moons[:, 0].min() - _pad), float(X_moons[:, 0].max() + _pad))
    y_range = (float(X_moons[:, 1].min() - _pad), float(X_moons[:, 1].max() + _pad))
    domain = (x_range, y_range)
    X = rng.uniform(
        [x_range[0], y_range[0]], [x_range[1], y_range[1]], size=(1500, 2)
    )
    return X, domain, rng


@app.cell
def __(X, checkpoints, compute_partition, mo, parx):
    with mo.status.spinner(title="Computing sparse & exact partitions for each checkpoint …"):
        partitions_sparse = {
            stage: compute_partition(sd, X, method="sparse_julia")
            for stage, sd in checkpoints.items()
        }
        partitions_exact = {
            stage: compute_partition(sd, X, method="exact_julia_fast")
            for stage, sd in checkpoints.items()
        }

    _rows = "\n".join(
        f"| {stage} | {len(partitions_sparse[stage])} | {len(partitions_exact[stage])} |"
        for stage in ("before", "during", "after")
    )
    mo.md(f"""
    `parx.list_methods()` → `{parx.list_methods()}`

    Same {X.shape[0]}-point sample grid, two methods, three checkpoints:

    | checkpoint | `sparse_julia` regions | `exact_julia_fast` regions |
    |---|---|---|
    {_rows}

    The sparse scan consistently finds fewer regions than the exact
    enumeration — it can only report activation patterns that some sampled
    point actually lands in, while `exact_julia_fast` enumerates every
    region by construction, regardless of how densely the input space
    happens to be sampled.
    """)
    return partitions_exact, partitions_sparse


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    The rest of this notebook uses the fully-trained ("after") exact
    partition as its running example.
    """)
    return


@app.cell
def __(partitions_exact):
    partition = partitions_exact["after"]
    return (partition,)


# ── multiple layers ───────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## Multiple layers: how depth builds the partition

    `plot_partition_2d`'s `layer=` argument collapses every leaf region to
    its activation-path *prefix* of that length, showing the coarser
    partition induced by only the first `layer` ReLU layers. Below: the
    trained network's partition after just the first hidden layer
    (`layer=1`, one arrangement of 8 hyperplanes) versus the full two-layer
    partition (`layer=2`) — the second layer visibly folds the coarse wedges
    into a much finer arrangement that hugs the moons' decision boundary.
    """)
    return


@app.cell
def __(X_moons, domain, go, mo, partition, plot_partition_2d, y_moons):
    _figs = []
    for _layer in (1, 2):
        _fig = plot_partition_2d(
            partition, domain=domain, layer=_layer, backend="plotly"
        )
        _fig.add_trace(
            go.Scatter(
                x=X_moons[:, 0],
                y=X_moons[:, 1],
                mode="markers",
                marker=dict(
                    color=y_moons,
                    colorscale=[[0, "#3b4cc0"], [1, "#b40426"]],
                    size=6,
                    line=dict(width=0.5, color="black"),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        _figs.append(_fig)
    mo.hstack(_figs)
    return


# ── watching the partition emerge ─────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## Watching the partition emerge across training

    Same domain, same coloring, one exact partition per checkpoint —
    scattered on top are the moons themselves. At initialization the
    boundaries are unrelated to the data; by the end they've organized
    themselves densely around the decision boundary between the two
    classes.
    """)
    return


@app.cell
def __(
    X_moons,
    domain,
    go,
    mid_epoch,
    mo,
    n_epochs,
    partitions_exact,
    plot_partition_2d,
    y_moons,
):
    _labels = {
        "before": "before training (random init)",
        "during": f"during training (epoch {mid_epoch})",
        "after": f"after training (epoch {n_epochs}, converged)",
    }
    _figs = []
    for _stage in ("before", "during", "after"):
        _fig = plot_partition_2d(
            partitions_exact[_stage], domain=domain, backend="plotly"
        )
        _fig.add_trace(
            go.Scatter(
                x=X_moons[:, 0],
                y=X_moons[:, 1],
                mode="markers",
                marker=dict(
                    color=y_moons,
                    colorscale=[[0, "#3b4cc0"], [1, "#b40426"]],
                    size=6,
                    line=dict(width=0.5, color="black"),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        _fig.update_layout(title=_labels[_stage])
        _figs.append(_fig)
    mo.hstack(_figs)
    return


@app.cell
def __():
    import plotly.graph_objects as go
    return (go,)


# ── training-time tracking ────────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## Training-time tracking with `animate_epochs`

    The same three checkpoints, animated. `animate_epochs` shares a fixed
    color scale and spatial range across frames so the growth in complexity
    is visually comparable; we use the (fast) sparse partitions here since
    the animation is illustrative rather than exact-count-critical.
    """)
    return


@app.cell
def __(animate_epochs, domain, mid_epoch, n_epochs, partitions_sparse):
    _order = ["before", "during", "after"]
    _epoch_labels = [
        "before (epoch 0)",
        f"during (epoch {mid_epoch})",
        f"after (epoch {n_epochs})",
    ]
    fig_anim = animate_epochs(
        [partitions_sparse[_stage] for _stage in _order],
        epoch_labels=_epoch_labels,
        x_range=domain[0],
        y_range=domain[1],
        backend="plotly",
    )
    fig_anim
    return


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


# ── visualization ─────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    ## Visualizing with the Plotly backend

    Every `parx.viz` plotting function accepts `backend: Literal["plotly",
    "matplotlib"] = "plotly"`. Plotly is the default, so it is omitted below
    except once for clarity — figures are returned directly from the cell and
    marimo renders them inline, interactively. These use the same
    fully-trained partition and domain as above.
    """)
    return


@app.cell
def __(domain, partition, plot_partition_2d):
    fig_partition = plot_partition_2d(partition, domain=domain, backend="plotly")
    fig_partition
    return


@app.cell
def __(partition, plot_region_counts):
    fig_counts = plot_region_counts(partition)
    fig_counts
    return


@app.cell
def __(domain, partition, plot_halfspaces):
    fig_halfspaces = plot_halfspaces(
        partition, partition.regions[0], x_range=domain[0], y_range=domain[1]
    )
    fig_halfspaces
    return


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
def __(affine_spectral, domain, partition, plot_partition_2d):
    fig_spectral = plot_partition_2d(
        partition,
        domain=domain,
        color_by=affine_spectral,
        colorscale="Plasma",
        backend="plotly",
    )
    fig_spectral
    return


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
    return


@app.cell
def __(np, partition_3d, plot_partition_projection):
    projection = np.eye(3)[:, :2]  # drop the 3rd input dimension
    fig_proj = plot_partition_projection(partition_3d, projection, backend="plotly")
    fig_proj
    return fig_proj, projection


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
    domain,
    mo,
    partition,
    rng,
):
    _X_test = rng.uniform(
        [domain[0][0], domain[1][0]], [domain[0][1], domain[1][1]], size=(2000, 2)
    )
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
