# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo",
#   "torch",
#   "numpy",
#   "scikit-learn",
#   "llvmlite>=0.42",
#   "numba>=0.59",
#   "umap-learn>=0.5.5",
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
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.datasets import make_moons
    import parx
    return make_moons, nn, np, parx, torch


# ── title ─────────────────────────────────────────────────────────────────────

@app.cell
def __(mo):
    mo.md(r"""
    # Do linear regions survive into feature space?

    A ReLU network partitions its input space into **linear regions** — maximal
    connected subsets where the network is a fixed affine map. `parx` computes
    this partition exactly.

    **The question explored here:** if you project the network's internal
    activations at some hidden layer down to 2D with tSNE or UMAP, do points
    from the *same linear region* cluster together, and do *neighbouring regions*
    land nearby?

    The answer is yes in a precise sense. Within each linear region the whole
    network is affine, so activations at any layer are an affine function of the
    2D input. Points in the same region therefore lie on the same 2-dimensional
    affine subspace of the activation space — they *must* cluster. The notebook
    makes this visible across different architectures and extraction layers.

    Use the controls below to pick an architecture and a layer. The partition,
    feature extraction, and both embeddings update reactively.
    """)
    return


# ── data ──────────────────────────────────────────────────────────────────────

@app.cell
def __(make_moons, np):
    X, y = make_moons(n_samples=600, noise=0.12, random_state=0)
    X = X.astype(np.float64)
    return X, y


@app.cell
def __(X, go, mo, y):
    _colors = ["#4878d0" if yi == 0 else "#ee854a" for yi in y]
    _fig = go.Figure(go.Scatter(
        x=X[:, 0], y=X[:, 1], mode="markers",
        marker=dict(color=_colors, size=5, opacity=0.7),
        text=[f"class {yi}" for yi in y],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    _fig.update_layout(
        title="Two-moons dataset (600 points, 2D input)",
        xaxis=dict(title="x₁", scaleanchor="y", scaleratio=1),
        yaxis_title="x₂", width=460, height=400,
    )
    mo.center(_fig)
    return


@app.cell
def __():
    import plotly.graph_objects as go
    return (go,)


# ── architecture selector ──────────────────────────────────────────────────────

@app.cell
def __(mo):
    # hidden layer sizes; model is always input(2) → hidden layers → output(2)
    _arch_options = {
        "Tiny     2 → 8 → 8 → 2":               [8, 8],
        "Small    2 → 16 → 16 → 2":              [16, 16],
        "Medium   2 → 32 → 32 → 2":              [32, 32],
        "Wide     2 → 64 → 64 → 2":              [64, 64],
        "Deep-narrow    2 → 16 → 16 → 16 → 2":  [16, 16, 16],
        "Deep-medium    2 → 32 → 32 → 32 → 2":  [32, 32, 32],
        "Bottleneck     2 → 32 → 8 → 32 → 2":   [32, 8, 32],
    }
    arch_dropdown = mo.ui.dropdown(
        options=_arch_options,
        value="Medium   2 → 32 → 32 → 2",
        label="Architecture",
    )
    arch_dropdown
    return (arch_dropdown,)


# ── build and train model ──────────────────────────────────────────────────────

@app.cell
def __(X, arch_dropdown, nn, np, torch, y):
    def _build(hidden_sizes):
        sizes = [2] + hidden_sizes + [2]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _train(model, epochs=500, lr=5e-3):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss_fn(model(Xt), yt).backward()
            opt.step()
        model.eval()

    torch.manual_seed(0)
    model = _build(arch_dropdown.value)
    _train(model)

    with torch.no_grad():
        _logits = model(torch.tensor(X, dtype=torch.float32))
        _acc = (_logits.argmax(dim=1).numpy() == y).mean()

    # label each extractable layer
    _linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    _layer_opts = {}
    for _i in range(1, len(_linears)):
        _dim = _linears[_i].in_features
        if _i == len(_linears) - 1:
            _label = f"Before output head  ({_dim}D activations)"
        else:
            _label = f"After hidden layer {_i}  ({_dim}D activations)"
        _layer_opts[_label] = _i

    arch_summary = (
        f"**Architecture:** `{arch_dropdown.value}`  →  "
        f"{len(_linears) - 1} hidden layer(s), "
        f"{sum(p.numel() for p in model.parameters())} parameters  |  "
        f"training accuracy **{_acc:.1%}**"
    )
    layer_options = _layer_opts
    return arch_summary, layer_options, model


# ── layer selector ────────────────────────────────────────────────────────────

@app.cell
def __(arch_summary, layer_options, mo):
    layer_dropdown = mo.ui.dropdown(
        options=layer_options,
        value=list(layer_options.keys())[-1],   # default: before output head
        label="Extract features from",
    )
    mo.vstack([
        mo.md(arch_summary),
        layer_dropdown,
    ])
    return (layer_dropdown,)


# ── compute partition ──────────────────────────────────────────────────────────

@app.cell
def __(X, mo, model, parx):
    with mo.status.spinner(title="Computing partition …"):
        partition = parx.compute_partition(model, X, method="sparse_julia")
    return (partition,)


# ── colour scheme selector ────────────────────────────────────────────────────

@app.cell
def __(mo):
    colour_dropdown = mo.ui.dropdown(
        options={
            "Spatial (HSV — angle + radius)": "spatial",
            "Frobenius norm (Viridis)": "frobenius",
            "Random (Turbo)": "random",
        },
        value="Spatial (HSV — angle + radius)",
        label="Region colour scheme",
    )
    colour_dropdown
    return (colour_dropdown,)


@app.cell
def __(colour_dropdown, parx, partition):
    region_colors = parx.region_palette(partition, colour_dropdown.value)
    return (region_colors,)


# ── partition plot ─────────────────────────────────────────────────────────────

@app.cell
def __(mo):
    mo.md(r"""
    ### Linear regions in input space

    Each polygon is one linear region, clipped to the data bounding box. The
    colour scheme is selected above:

    - **Spatial (HSV)** — hue encodes the angle of the region's centroid from
      the partition centre; saturation encodes radial distance. Adjacent regions
      in input space get visually adjacent colours.
    - **Frobenius norm** — colour encodes ‖A‖_F, the local affine map's
      Frobenius norm. Dense small regions near the decision boundary tend to
      have larger norms.
    - **Random (Turbo)** — each region gets a distinct hue from the Turbo
      palette in region-index order.
    """)
    return


@app.cell
def __(X, mo, parx, partition, region_colors):
    _pad = 0.35
    _xr = (float(X[:, 0].min()) - _pad, float(X[:, 0].max()) + _pad)
    _yr = (float(X[:, 1].min()) - _pad, float(X[:, 1].max()) + _pad)
    _fig = parx.viz.plot_partition_2d(partition, domain=(_xr, _yr), colors=region_colors)
    _fig.update_layout(
        title=f"Partition — {len(partition)} regions",
        width=500, height=460,
    )
    mo.center(_fig)
    return


# ── feature extraction ────────────────────────────────────────────────────────

@app.cell
def __(mo):
    mo.md(r"""
    ### Feature extraction

    `parx.extract_features` attaches a forward hook to the chosen Linear layer
    and captures its **input tensor** — the post-ReLU activations flowing into
    that layer. Within each linear region the network is affine end-to-end, so
    these activations are an affine function of the 2D input. Points in the same
    region lie on the same 2D affine subspace of the (higher-dimensional)
    activation space.
    """)
    return


@app.cell
def __(X, layer_dropdown, mo, model, parx):
    with mo.status.spinner(title="Extracting features …"):
        features = parx.extract_features(model, X, layer_index=layer_dropdown.value)
    mo.md(
        f"Extracted **{features.shape[1]}D** activations from "
        f"`layer_index={layer_dropdown.value}` "
        f"→ array shape `{features.shape}`"
    )
    return (features,)


# ── embeddings ────────────────────────────────────────────────────────────────

@app.cell
def __(mo):
    mo.md(r"""
    ### tSNE and UMAP of the extracted features

    Both methods project the high-dimensional activations to 2D. Points are
    coloured two ways:

    - **By linear region** (left pair) — same colour = same affine map. Tight
      monochromatic clusters confirm that the region structure is preserved in
      feature space.
    - **By class label** (right pair) — comparing this with the region colouring
      shows how many affine pieces the network used per class and where class
      boundaries fall relative to region boundaries.
    """)
    return


@app.cell
def __(X, features, mo, np, parx, partition, region_colors, y):
    def _embed_fig(method, color_by, title):
        return parx.plot_feature_embedding(
            features, partition, X,
            method=method,
            color_by=color_by,
            title=title,
            random_state=0,
        )

    with mo.status.spinner(title="Running tSNE …"):
        _tsne_region = _embed_fig("tsne", region_colors, "tSNE — by region")
        _tsne_class  = _embed_fig("tsne", y.astype(np.float64), "tSNE — by class")

    with mo.status.spinner(title="Running UMAP …"):
        _umap_region = _embed_fig("umap", region_colors, "UMAP — by region")
        _umap_class  = _embed_fig("umap", y.astype(np.float64), "UMAP — by class")

    for _f in [_tsne_region, _tsne_class, _umap_region, _umap_class]:
        _f.update_layout(width=400, height=380, margin=dict(t=40, b=20, l=20, r=20))

    mo.vstack([
        mo.hstack([_tsne_region, _tsne_class], justify="center"),
        mo.hstack([_umap_region, _umap_class], justify="center"),
    ])
    return


# ── interpretation ────────────────────────────────────────────────────────────

@app.cell
def __(mo):
    mo.md(r"""
    ### Reading the plots

    **Tight monochromatic clusters in the region plot** confirm the geometric
    argument: the affine map within each region collapses the 2D affine subspace
    to a single point in tSNE/UMAP space.

    **Smooth colour gradients at cluster edges** indicate neighbouring regions
    — sharing a face in input space means differing by one ReLU flip, so their
    affine maps are close and their feature clouds land nearby in the embedding.

    **One class colour, multiple region colours** is expected and informative.
    A curved decision boundary requires multiple affine pieces; the number of
    region-colour blobs per class is a direct count of how many pieces the
    network allocated to that class.

    **tSNE vs UMAP** — tSNE preserves local structure (nearby clusters are
    reliable) but distorts global distances. UMAP better preserves the large-scale
    topology of how region groups relate to each other. Using both gives a more
    complete picture.

    **Changing the extraction layer** shows how the representation evolves
    through the network. Early layers produce coarser clusters (fewer active
    neurons, simpler affine maps); later layers produce tighter, more
    class-aligned clusters as the network funnels towards the decision.
    """)
    return


if __name__ == "__main__":
    app.run()
