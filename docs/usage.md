# Usage

## One-shot analysis

```python
import parx
import numpy as np
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 8), nn.ReLU(),
    nn.Linear(8, 8), nn.ReLU(),
    nn.Linear(8, 1),
)

# Warm up Julia JIT (optional but recommended)
parx.precompile()

# Compute the partition
X = np.random.uniform(-1, 1, (1000, 2))
partition = parx.compute_partition(model, X, method="sparse_julia")

# Analyze
print(parx.complexity_profile(partition))
print("Dead neurons:", parx.dead_neurons(partition))
print("Size summary:", parx.region_size_summary(partition))

# Visualize
from parx.viz import plot_partition_2d
plot_partition_2d(partition).show()

# Save (no Julia needed to reload)
parx.save_partition(partition, "partition.npz")
```

## Training-time tracking

The primary research use case: compute the partition at multiple checkpoints and track how it evolves. The [two-moons notebook](notebooks/index.md) walks through this end-to-end, comparing sparse and exact partitions before/during/after training.

```python
partitions = []
labels = []

for epoch, state_dict in parx.iter_state_dicts("checkpoints/"):
    p = parx.compute_partition(state_dict, X, method="sparse_julia")
    partitions.append(p)
    labels.append(epoch)
    parx.save_partition(p, f"partitions/epoch_{epoch}.npz")

# Track complexity over training
stats = parx.analysis.complexity_over_epochs(partitions, labels=labels)
# stats["n_regions"], stats["dead_neuron_count"], stats["mean_chebyshev_radius"]

# Animated visualization
from parx.viz import animate_epochs, animate_epochs_video
animate_epochs(partitions, epoch_labels=labels).show()
animate_epochs_video(partitions, "evolution.gif", epoch_labels=labels)
```

## Geometric analysis with exact partitions

```python
# Exact enumeration: complete partition from a single starting point
partition = parx.compute_partition(model, np.zeros((1, 2)), method="exact_julia")
# Every region has geometry
for region in partition.regions:
    D, g  = partition.halfspaces(region)         # polytope constraints
    A, b  = partition.local_affine(region)        # local linear map
    x0    = region.centroid                       # interior point
    print(f"  bounded={region.bounded}, n_constraints={D.shape[0]}")

# Verify geometric correctness
from parx.verify import check_no_overlaps, check_covers_space
X_test = np.random.uniform(-1, 1, (5000, 2))
assert check_no_overlaps(partition, X_test)[0]
assert check_covers_space(partition, X_test)[0]

# Volume estimates (slow)
from parx.analysis import partition_volume_estimates
volumes = partition_volume_estimates(partition, n_samples=5000)
```

## Visualization

Every plotting function accepts a keyword-only `backend: Literal["plotly", "matplotlib"] = "plotly"` argument. `"plotly"` (default) returns an interactive `plotly.graph_objects.Figure`; `"matplotlib"` returns a static `matplotlib.figure.Figure` (or `matplotlib.animation.FuncAnimation` for `animate_epochs`) and requires `pip install "parx[animate]"`. Matplotlib output has no hover tooltips, and `animate_epochs(backend="matplotlib")` has no play/pause/slider controls — use `.to_jshtml()` in a notebook or `.save(...)` to export it.

```python
from parx.viz import (
    plot_partition_2d,         # 2D input space; color by metric
    plot_partition_slice,      # slice higher-dim partition to 2D
    plot_partition_projection, # project halfspace normals to 2D
    plot_partition_pca,        # PCA projection (requires scikit-learn)
    plot_region_counts,        # bar chart of regions per layer
    plot_halfspaces,           # halfspace boundary overlay
)

fig = plot_partition_2d(partition, backend="matplotlib")
fig.savefig("partition.png")
```

Color-by callables: `affine_frobenius`, `affine_spectral`, `affine_det`, `active_neuron_count` (all in `parx.viz`). `plot_partition_2d(..., layer=l)` collapses regions to their first `l` layers, showing how depth builds the partition — see the notebooks for a worked example.

## Utilities

```python
parx.precompile()         # warm up Julia JIT (call once at startup)
parx.list_methods()       # → list of registered method names

from parx.diagnostics import thread_info, benchmark_method
thread_info()             # Julia thread count and environment info
benchmark_method(model, X, method="sparse_julia")  # timing comparison
```

## Limitations

These are current constraints, not design goals. See the project's [TODO.md](https://github.com/Johanmkr/parx/blob/main/TODO.md) for planned extensions.

**Architecture support:** Only sequential `Linear → ReLU` stacks. ResNets, transformers, CNNs, networks with batch normalization or dropout, and any model with branching structure are not supported.

**Exact enumeration scaling:** The DFS is exponential in the worst case. In practice `input_dim=2` with up to ~20 neurons per layer is reliable; `input_dim=3` is feasible for small networks; beyond that exact enumeration is often infeasible.

**Sparse enumeration completeness:** Sparse mode can only find regions that your sample points land in. With `input_dim > 5` and complex networks, many regions may never appear in any finite sample.

**Julia startup:** The first call to any Julia-backed method in a process incurs 10–30 seconds of JIT compilation. Call `parx.precompile()` once at startup to amortize this. Subsequent calls within the same process are fast.

**Visualization:** `plot_partition_2d` requires `input_dim == 2`. For higher-dimensional networks use `plot_partition_slice` (fix all but two dimensions) or `plot_partition_projection` (project onto a 2D subspace) — both are approximations.
