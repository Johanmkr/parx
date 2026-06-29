# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo",
#   "torch",
#   "numpy",
#   "scikit-learn",
#   "plotly",
#   "parx @ file:///home/johan/Documents/phd/parx",
# ]
# ///

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


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


@app.cell
def __(mo):
    mo.md(r"""
    # Do linear regions survive into feature space?

    A ReLU network partitions its input space into **linear regions** — maximal
    connected subsets where the network is a fixed affine map. `parx` computes
    this partition exactly.

    The question explored here: if you take the activations just before the
    classification head (the *penultimate features*) and project them with tSNE,
    do points from the **same region** cluster together, and do points from
    **neighbouring regions** land nearby?

    The answer must be yes in a precise sense — within each linear region the
    whole network is affine, so the feature map is affine too. Points in the same
    region lie on the same low-dimensional affine subspace of feature space. The
    notebook makes this visible.
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""
    ## Step 1 · Data

    We use the two-moons dataset: 500 points in 2D, two interleaved half-circles.
    The low dimensionality lets us draw the exact partition as polygons and compare
    it directly to the tSNE embedding.
    """)
    return


@app.cell
def __(make_moons, np):
    X, y = make_moons(n_samples=500, noise=0.15, random_state=0)
    X = X.astype(np.float64)
    print(f"X shape: {X.shape}   (500 points, 2D input)")
    print(f"Classes: {np.unique(y).tolist()}   (0 = upper moon, 1 = lower moon)")
    return X, y


@app.cell
def __(X, np, parx, y):
    import plotly.graph_objects as go

    _colors = ["#4878d0" if yi == 0 else "#ee854a" for yi in y]
    _fig_data = go.Figure(go.Scatter(
        x=X[:, 0], y=X[:, 1],
        mode="markers",
        marker=dict(color=_colors, size=5, opacity=0.7),
        text=[f"class {yi}" for yi in y],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    _fig_data.update_layout(
        title="Two-moons dataset",
        xaxis=dict(title="x₁", scaleanchor="y", scaleratio=1),
        yaxis_title="x₂",
        width=480, height=420,
    )
    _fig_data
    return (go,)


@app.cell
def __(mo):
    mo.md(r"""
    ## Step 2 · Network architecture

    We train a small MLP with two hidden layers of **32 neurons** each:

    ```
    Input (2D)
      └─ Linear(2 → 32)  +  ReLU      ← hidden layer 1: 32 neurons
           └─ Linear(32 → 32)  +  ReLU  ← hidden layer 2: 32 neurons
                └─ Linear(32 → 2)         ← output / classification head
    ```

    The output of hidden layer 2 is a **32-dimensional vector**. This is the
    representation that the classifier sees — we will call it the *penultimate
    feature vector*. It is what `parx.extract_features` captures.

    Each ReLU layer introduces hyperplane boundaries. With 32 neurons per layer
    and a 2D input, the network can produce at most a few hundred distinct linear
    regions (fewer in practice because most patterns are unreachable).
    """)
    return


@app.cell
def __(X, make_moons, nn, np, torch, y):
    def make_model():
        return nn.Sequential(
            nn.Linear(2, 32),  nn.ReLU(),   # hidden layer 1
            nn.Linear(32, 32), nn.ReLU(),   # hidden layer 2
            nn.Linear(32, 2),               # classification head
        )

    def _train(model, X, y, epochs=400, lr=5e-3):
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
    model = make_model()
    _train(model, X, y)

    with torch.no_grad():
        _logits = model(torch.tensor(X, dtype=torch.float32))
        acc = (_logits.argmax(dim=1).numpy() == y).mean()

    print(f"Training accuracy: {acc:.1%}")
    return acc, make_model, model


@app.cell
def __(mo):
    mo.md(r"""
    ## Step 3 · Exact polyhedral partition

    `parx.compute_partition` traces through every reachable activation pattern
    and records the resulting linear region. The sparse Julia method does a
    parallel forward-pass scan over the training data; the exact method would
    also find regions not covered by any training point.
    """)
    return


@app.cell
def __(X, model, parx):
    partition = parx.compute_partition(model, X, method="sparse_julia")
    print(f"Regions found: {len(partition)}")
    return (partition,)


@app.cell
def __(mo):
    mo.md(r"""
    ### Partition visualised in input space

    Each coloured polygon is one linear region, clipped to the data bounding box.
    The colour encodes ‖A‖_F — the Frobenius norm of that region's local affine
    map — which measures how strongly the network stretches inputs in that area.
    Small, dense regions near the centre mark where the decision boundary runs.
    """)
    return


@app.cell
def __(X, np, parx, partition):
    _pad = 0.35
    _xr = (float(X[:, 0].min()) - _pad, float(X[:, 0].max()) + _pad)
    _yr = (float(X[:, 1].min()) - _pad, float(X[:, 1].max()) + _pad)

    fig_2d = parx.viz.plot_partition_2d(partition, domain=(_xr, _yr))
    fig_2d.update_layout(width=520, height=480)
    fig_2d
    return (fig_2d,)


@app.cell
def __(mo):
    mo.md(r"""
    ## Step 4 · Extract penultimate features

    `parx.extract_features` attaches a PyTorch forward hook to the **last
    `nn.Linear` layer** (the classification head, `Linear(32 → 2)`), and
    captures the tensor flowing **into** it — i.e., the 32-dimensional output
    of hidden layer 2 after its ReLU.

    ```
    hidden layer 2 output  →  [hook captures here]  →  Linear(32 → 2)  →  logits
    ```

    This gives one 32-dimensional vector per input point. The hook is removed
    immediately after the forward pass; the model is not modified.
    """)
    return


@app.cell
def __(X, model, parx):
    features = parx.extract_features(model, X, layer_index=-1)
    print(f"features shape: {features.shape}")
    print(f"  → {features.shape[0]} points, each described by {features.shape[1]} numbers")
    print(f"    (the 32 activations of hidden layer 2, after ReLU)")
    return (features,)


@app.cell
def __(mo):
    mo.md(r"""
    ## Step 5 · tSNE of features, coloured by linear region

    tSNE projects the 500 × 32 feature matrix down to 2D, preserving local
    neighbourhood structure. We then colour each point by which linear region
    it belongs to, as determined by `partition.route(X)`.

    **What to expect:** Points in the same linear region share an identical
    affine map, so their features differ only by an affine function of their
    2D input displacement. They lie on the same 2-dimensional affine subspace
    of ℝ³². tSNE will collapse each such subspace to a tight cluster — and
    neighbouring regions (one ReLU flip apart) should land in adjacent clusters.

    Grey × markers are points the sparse partition did not cover.
    """)
    return


@app.cell
def __(X, features, parx, partition):
    fig_region = parx.plot_feature_embedding(
        features, partition, X,
        method="tsne",
        color_by="region",
        title="tSNE of hidden-layer-2 features — coloured by linear region",
        random_state=0,
    )
    fig_region
    return (fig_region,)


@app.cell
def __(mo):
    mo.md(r"""
    ## Step 6 · Same tSNE, coloured by class label

    We reuse the same tSNE layout (`random_state=0` makes it reproducible) but
    swap the colour to ground-truth class. This lets us directly compare the
    region structure with the class structure.

    A well-trained network organises its feature space so that different classes
    map to different regions. If the two colour maps agree, the partition and
    the classification are consistent. Where they disagree — e.g., one class
    spans several region colours — the network needed multiple affine pieces to
    represent the curved decision boundary.
    """)
    return


@app.cell
def __(X, features, np, parx, partition, y):
    fig_class = parx.plot_feature_embedding(
        features, partition, X,
        method="tsne",
        color_by=y.astype(np.float64),
        title="tSNE of hidden-layer-2 features — coloured by class label",
        random_state=0,
    )
    fig_class
    return (fig_class,)


@app.cell
def __(mo):
    mo.md(r"""
    ## Step 7 · Reading the results

    **Region coluring clusters tightly** — each region forms a compact blob in
    the tSNE, confirming the affine-subspace argument. Colour transitions between
    adjacent blobs correspond to neighbouring regions in input space.

    **Class and region colourings are largely consistent** — the network has
    learned to place the two moons in different parts of feature space, and those
    parts are further subdivided into regions.

    **One class, multiple region colours** — each moon is not a single flat
    region; the network uses several affine pieces to approximate the curved
    boundary. In the tSNE these appear as a group of differently-coloured blobs
    that are spatially adjacent. Counting how many blobs per class gives a rough
    measure of how many affine pieces the network devoted to each part of the
    decision boundary.

    **Scalability note** — for networks with input dimension > 2 the direct
    partition plot is unavailable, but this tSNE-of-features view works
    regardless of input dimension. It is not a geometric substitute (it cannot
    recover region shapes), but it faithfully reveals adjacency structure and
    how the partition relates to the task.
    """)
    return


if __name__ == "__main__":
    app.run()
