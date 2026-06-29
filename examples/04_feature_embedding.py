"""
Example 04: Feature embedding coloured by linear region.

Trains a small MLP on the two-moons dataset, computes the exact polyhedral
partition, extracts penultimate-layer features, and plots a tSNE embedding
coloured by region membership.

Run:
    python examples/04_feature_embedding.py
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

import parx


def make_model():
    return nn.Sequential(
        nn.Linear(2, 32), nn.ReLU(),
        nn.Linear(32, 32), nn.ReLU(),
        nn.Linear(32, 2),
    )


def train(model, X, y, epochs=200, lr=1e-2):
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


def main():
    X, y = make_moons(n_samples=500, noise=0.15, random_state=0)
    X = X.astype(np.float64)

    model = make_model()
    train(model, X, y)

    print("Computing partition …")
    partition = parx.compute_partition(model, X, method="sparse_julia")
    print(f"  {len(partition)} regions found")

    print("Extracting penultimate features …")
    features = parx.extract_features(model, X)
    print(f"  feature shape: {features.shape}")

    print("Plotting tSNE embedding coloured by region …")
    fig = parx.plot_feature_embedding(
        features, partition, X,
        method="tsne",
        color_by="region",
    )
    fig.write_html("feature_embedding_region.html")
    print("  saved → feature_embedding_region.html")

    print("Plotting tSNE embedding coloured by class label …")
    fig2 = parx.plot_feature_embedding(
        features, partition, X,
        method="tsne",
        color_by=y.astype(float),
        title="Feature embedding (tSNE) — class label",
    )
    fig2.write_html("feature_embedding_class.html")
    print("  saved → feature_embedding_class.html")


if __name__ == "__main__":
    main()
