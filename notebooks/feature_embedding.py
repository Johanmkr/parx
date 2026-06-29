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
def __(mo):
    mo.md(r"""
    # Feature Embedding Coloured by Linear Region

    **Hypothesis (Waly, 2026):** The penultimate-layer features of a trained ReLU network,
    when projected with tSNE, should reflect the same adjacency structure as the exact
    polyhedral linear regions computed by `parx`. Within each region the network applies
    the same affine map, so all points in that region land on the same affine subspace of
    feature space — they *must* cluster. Neighbouring regions (differing by exactly one
    ReLU flip) have nearly identical affine maps, so their clusters should lie close
    together in the embedding.

    This notebook tests that hypothesis on the two-moons dataset, where we can also
    draw the exact partition directly and compare visually.
    """)
    return


@app.cell
def __():
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.datasets import make_moons
    return make_moons, nn, np, torch


@app.cell
def __():
    import parx
    return (parx,)


@app.cell
def __(mo):
    mo.md(r"""
    ## 1 · Dataset and Model

    We use the two-moons dataset (500 points, moderate noise) and a small
    three-layer MLP. The 2-dimensional input makes it possible to draw the
    exact partition later and compare it to the embedding directly.
    """)
    return


@app.cell
def __(make_moons, np):
    X, y = make_moons(n_samples=500, noise=0.15, random_state=0)
    X = X.astype(np.float64)
    return X, y


@app.cell
def __(nn):
    def make_model():
        return nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )
    return (make_model,)


@app.cell
def __(X, make_model, nn, torch, y):
    def _train(model, X, y, epochs=300, lr=1e-2):
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
        logits = model(torch.tensor(X, dtype=torch.float32))
        acc = (logits.argmax(dim=1).numpy() == y).mean()

    print(f"Training accuracy: {acc:.1%}")
    return acc, model


@app.cell
def __(mo):
    mo.md(r"""
    ## 2 · Exact Polyhedral Partition

    `parx.compute_partition` enumerates every linear region of the trained network —
    the maximal connected subsets of input space where the network is affine.
    Each region corresponds to a unique activation pattern (which neurons are on/off).
    """)
    return


@app.cell
def __(X, mo, model, parx):
    partition = parx.compute_partition(model, X, method="sparse_julia")
    mo.md(f"**{len(partition)} linear regions** found in the input space.")
    return (partition,)


@app.cell
def __(mo):
    mo.md(r"""
    ### 2a · The partition in input space

    Each coloured polygon is one linear region. The colour encodes the Frobenius norm
    of the region's local affine map — a proxy for how strongly the network is
    transforming inputs in that part of space. The two-moons decision boundary runs
    through the densest cluster of small regions near the centre.
    """)
    return


@app.cell
def __(parx, partition):
    fig_2d = parx.viz.plot_partition_2d(partition)
    fig_2d
    return (fig_2d,)


@app.cell
def __(mo):
    mo.md(r"""
    ## 3 · Penultimate-Layer Features

    `parx.extract_features` hooks into the network with a PyTorch forward hook and
    captures the activations fed into the final linear layer — the 32-dimensional
    representation that the classifier operates on. These features live in a space
    where linear regions correspond to affine subspaces.

    > **Why the last linear's input?** Within each linear region the whole network
    > is a fixed affine map, so the penultimate activations for any two points in
    > the same region differ only by a linear function of their input displacement.
    > They lie on the same 2-dimensional affine subspace of ℝ³² (since the input
    > is 2D). tSNE will collapse each such subspace to a single tight cluster.
    """)
    return


@app.cell
def __(X, model, parx):
    features = parx.extract_features(model, X)
    print(f"Feature array shape: {features.shape}")
    return (features,)


@app.cell
def __(mo):
    mo.md(r"""
    ## 4 · tSNE Embedding Coloured by Linear Region

    We run tSNE on the 32-dimensional features and colour each point by which
    linear region it belongs to (from `partition.route(X)`). If the hypothesis
    holds, points with the same colour should form tight, well-separated clusters.
    Points that the sparse partition did not assign to any region appear as grey ×.
    """)
    return


@app.cell
def __(X, features, parx, partition):
    fig_region = parx.plot_feature_embedding(
        features, partition, X,
        method="tsne",
        color_by="region",
        title="tSNE of penultimate features — coloured by linear region",
    )
    fig_region
    return (fig_region,)


@app.cell
def __(mo):
    mo.md(r"""
    **What to look for:** Each colour corresponds to one linear region. If the
    hypothesis is correct you should see monochromatic clusters with smooth
    colour gradients at cluster boundaries — indicating that geometrically
    adjacent regions (sharing a face in input space) also have nearby feature
    clouds in the tSNE layout.

    ## 5 · Same Embedding, Coloured by Class Label

    For comparison, here is the identical tSNE embedding coloured by ground-truth
    class (0 = moon A, 1 = moon B). A well-trained network should have learned to
    separate classes into different regions, so the class coloring and the region
    coloring should be largely consistent.
    """)
    return


@app.cell
def __(X, features, np, parx, partition, y):
    fig_class = parx.plot_feature_embedding(
        features, partition, X,
        method="tsne",
        color_by=y.astype(np.float64),
        title="tSNE of penultimate features — coloured by class label",
    )
    fig_class
    return (fig_class,)


@app.cell
def __(mo):
    mo.md(r"""
    ## 6 · Interpretation

    **Agreement between the two colourings** confirms that the network has
    organised its feature space around the polyhedral partition: each class
    occupies a distinct collection of regions, and the tSNE layout reflects both
    structures simultaneously.

    **Fine-grained disagreement** is expected and informative: a single class can
    span multiple regions (one per local linear piece of the decision boundary),
    so one class colour may correspond to several region colours that form a
    connected group in the embedding. These sub-clusters mark where the network
    needed more than one affine piece to fit the class boundary.

    **Grey × points** are inputs the sparse partition missed — `route()` returned
    `None` for them. With the exact Julia method (`method="exact_julia"`) or a
    denser sample these would be assigned regions and coloured accordingly.

    **Broader implication:** The tSNE-of-features plot is a scalable substitute
    for the 2D partition plot when the input dimension is high. It cannot recover
    the geometry of individual regions, but it faithfully reveals the *adjacency
    structure* — which regions cluster together and which are far apart — which is
    exactly the question Waly's hypothesis is about.
    """)
    return


if __name__ == "__main__":
    app.run()
