# Examples

Runnable, self-contained scripts in [`examples/`](https://github.com/Johanmkr/parx/tree/main/examples). Each writes its figures to `examples/output/` as standalone HTML (and PNG, where a Chrome install is available for Plotly's `kaleido`/`plotly_get_chrome` export). Run them from the repo root after installing `parx[dev]` (see [Home § Installation](index.md#installation)).

## 01 — Identity network

```bash
python examples/01_identity_network.py
```

A one-hidden-layer network with identity weights (`2 → 2 → 1`) partitions R² into exactly its four quadrants — a partition simple enough to verify by hand. Demonstrates both `sparse_julia` and `exact_julia` modes on the same network, the [verification](reference.md#verification) functions (`check_no_overlaps`, `check_covers_space`), and all three 2D visualizations (`plot_partition_2d`, `plot_region_counts`, `plot_halfspaces`).

## 02 — Random MLP

```bash
python examples/02_random_mlp.py
```

A deeper `2 → 5 → 5 → 5 → 1` network with random weights. Demonstrates the `state_dict` workflow (`model.state_dict()` → `compute_partition`, rather than passing the `nn.Module` directly), the method registry (`parx.list_methods()`, comparing sparse vs. exact and the Julia vs. Python backends), and coloring regions by a metric derived from the local affine map (`affine_frobenius`, `affine_spectral`).

## 03 — Partition evolution over training

```bash
python examples/03_epochs.py
```

Trains a small network to convergence, capturing a `state_dict` snapshot every epoch, then computes the **exact** partition for each snapshot. Produces a static before/after comparison, an interactive Plotly animation with a play button and epoch slider (`animate_epochs`), and an exported GIF (`animate_epochs_video`, requires `pip install "parx[animate]"`). This is the closest example to the primary research use case described in [Usage § Training-time tracking](usage.md#training-time-tracking).

## 04 — Feature embedding

```bash
python examples/04_feature_embedding.py
```

Trains a classifier on the two-moons dataset, computes the exact polyhedral partition, extracts penultimate-layer features, and plots a t-SNE embedding colored by region membership — connecting the partition back to the network's learned representation, not just its input-space geometry.

---

For a guided, narrated walkthrough of the same two-moons workflow (sparse and exact partitions before/during/after training, layer-by-layer views, save/reload), see the [Notebooks](notebooks/index.md) page instead — the scripts here are meant to be read as source code and run standalone, while the notebooks are meant to be read top-to-bottom as a tutorial.
