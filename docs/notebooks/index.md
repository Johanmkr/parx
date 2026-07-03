# Notebooks

Two marimo notebooks walk the public API end-to-end using a single running example: a small classifier trained on the **two-moons** dataset. Both compute the **sparse** and **exact** partitions **before, during, and after training**, look at the partition **layer by layer**, and animate the full training run — then continue into inspecting a `Partition`, analysis/verification functions, a higher-dimensional example, and save/reload. The only difference between them is the plotting backend.

A third notebook asks a more specific question: do linear regions survive into feature-space? It projects hidden-layer activations to 2D with tSNE/UMAP and checks whether points from the same linear region cluster together, with an interactive architecture/layer picker.

These are static, pre-exported snapshots (`marimo export html`) rather than live notebooks — `parx` depends on Julia and PyTorch, neither of which run inside a browser sandbox, so a fully interactive in-browser version isn't possible here. To run any notebook live and edit it, clone the repo and run:

```bash
uv run notebooks/demo_plt.py            # matplotlib backend
uv run notebooks/demo_plotly.py         # plotly backend
uv run notebooks/feature_embedding.py   # feature-space embedding
```

## Matplotlib backend

Static figures, no hover tooltips; the training animation is a plain `FuncAnimation`.

<iframe src="demo_plt.html" style="width: 100%; height: 1400px; border: 0;" title="parx demo notebook — matplotlib backend"></iframe>

## Plotly backend

Interactive figures with hover tooltips; the training animation has a play/pause button and slider.

<iframe src="demo_plotly.html" style="width: 100%; height: 1400px; border: 0;" title="parx demo notebook — Plotly backend"></iframe>

## Feature-space embedding

Do linear regions survive into feature-space? Projects hidden-layer activations down to 2D with tSNE/UMAP and colors points by which linear region they belong to, across a selectable architecture and extraction layer.

<iframe src="feature_embedding.html" style="width: 100%; height: 1400px; border: 0;" title="parx demo notebook — feature-space embedding"></iframe>
