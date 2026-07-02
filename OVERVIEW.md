# parx — Concepts, Workflow, and Design

This document explains what parx does, why it does it, how data flows through the library, and how to think about using it. For the API reference see [README.md](README.md).

---

## The core idea

A ReLU network is **piecewise linear**. Every input point `x` activates some neurons (pre-activation value > 0) and silences others (pre-activation ≤ 0), producing a binary activation pattern across every hidden layer. All points that share the exact same pattern experience the same fixed linear function `f(x) = Ax + b` — the network is literally a matrix multiply plus bias in that region, with no nonlinearity in play.

The set of all points sharing a given activation pattern forms a **convex polytope** (a bounded or unbounded convex region defined by a finite set of halfspace constraints). These polytopes tile the input space without overlap: every input point belongs to exactly one region. The collection of all such polytopes is the **polyhedral partition** of the network.

parx computes this partition. Knowing it lets you ask questions that are otherwise opaque:

- How many distinct linear pieces does this network have? (complexity)
- Are there neurons that never fire across the whole input domain? (dead neurons)
- How large or small are the regions? (Chebyshev radii, volume estimates)
- How does the partition change as the network trains? (epoch-over-epoch analysis)
- What is the exact halfspace representation of a given region? (geometry)

---

## Inputs

### The network

parx accepts a ReLU neural network in four forms:

| Form | Example |
|---|---|
| PyTorch `nn.Module` | `model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))` |
| PyTorch `state_dict` | `model.state_dict()` |
| Path to `.pth` file | `"checkpoints/epoch_10.pth"` |
| Path to `.h5` file | `"model.h5"` (requires `pip install parx[h5]`) |

**Important constraint:** parx currently only handles sequential stacks of `Linear → ReLU` layers. Residual connections, skip connections, batch normalization, convolutional layers, and attention are not supported. Attempting to load such a model will either raise an error or silently drop the unsupported layers.

By default the output layer is excluded (`include_output_layer=False`) because it does not participate in the piecewise-linear structure — only hidden ReLU layers define the polyhedral partition.

### The sample points

```python
X  # shape (N, input_dim), dtype float64
```

Sample points serve different purposes depending on the enumeration method (see below). They do **not** need to cover the whole input space, and they do not need to be on a grid.

---

## The two enumeration strategies

This is the most important choice a user makes. The two strategies have fundamentally different guarantees.

### Sparse enumeration — `method="sparse_julia"` or `"sparse_python"`

**What it does:** Runs a forward pass on every point in `X`. Each point lands in one region (its activation pattern). The result is the set of distinct patterns seen across all `N` points, one region per pattern, with the centroid set to the mean of all points in that pattern.

**What you get:** Only the regions that your data points happened to land in. If a region is small, low-density, or simply not covered by your sample, it will not appear in the partition.

**When to use it:**
- You want a fast survey — which regions exist in the neighborhood your data occupies?
- You have a large, dense point cloud that likely covers the relevant input space.
- You are willing to miss small or remote regions in exchange for speed.
- `input_dim` can be large; the cost scales with `N` and the number of neurons, not with the number of regions.

**Cost:** O(N × total neurons) — embarrassingly parallel, fast even for deep networks.

```python
X = np.random.uniform(-1, 1, (10_000, input_dim))
partition = compute_partition(model, X, method="sparse_julia")
```

---

### Exact enumeration — `method="exact_julia"`, `"exact_julia_fast"`, or `"exact_python"`

**What it does:** Starts from a single point `x0 = data[0]` and performs a depth-first search (DFS) over the graph of adjacent regions. Two regions are adjacent if they share a facet (a face of codimension 1). At each candidate facet the algorithm solves a linear program (LP via HiGHS) to determine whether the neighboring region is feasible and to find its Chebyshev center.

**What you get:** The complete partition — every region reachable from `x0` by crossing facets. For a connected network this is provably all regions.

**When to use it:**
- You need the complete picture, not just a sample.
- `input_dim` is small (2 is ideal; up to 4–5 is often feasible).
- The network is shallow (1–3 hidden layers, small width).
- You can afford minutes of compute time.

**Cost:** Exponential in the worst case. A 2×5×5×5 network (input_dim=2, three layers of 5 neurons each) typically has hundreds of regions and takes seconds. A 2×20×20 network may have thousands of regions and take minutes. Beyond that, exact enumeration is currently impractical.

```python
x0 = np.zeros(input_dim)              # only data[0] is used
partition = compute_partition(model, x0, method="exact_julia")
```

> **Note:** The exact methods populate `Region.active_indices` (non-redundant constraint indices) and `Region.bounded` (whether the polytope is bounded). The sparse methods do not.

---

## Outputs — the `Partition` object

`compute_partition` returns a `Partition`: a flat list of `Region` objects plus the network weights and biases.

### `Region`

| Field | Type | Description |
|---|---|---|
| `activation_path` | `list[np.ndarray[bool]]` | One boolean array per hidden layer; `True` = neuron active |
| `centroid` | `np.ndarray` shape `(input_dim,)` | An interior point (Chebyshev center) |
| `active_indices` | `np.ndarray[int32]` or `None` | Non-redundant constraint row indices (exact only) |
| `bounded` | `bool` | Whether the polytope is bounded (exact only) |

### `Partition` methods

```python
# Geometry
D, g = partition.halfspaces(region)             # D @ x <= g defines the polytope
D, g = partition.halfspaces(region, active_only=True)  # non-redundant rows only
A, b = partition.local_affine(region)           # local linear map: f(x) = A @ x + b

# Routing
regions = partition.route(X)                    # list[Region | None] per point

# Filtering
subset = partition.regions_at_layer(l)          # regions with path depth == l

# Metadata
partition.n_layers                              # number of hidden layers
partition.input_dim                             # input dimensionality
len(partition)                                  # number of regions
```

---

## Data pipeline

```
┌─────────────────────────────────────┐
│  Network source                     │
│  nn.Module / state_dict / .pth / .h5│
└──────────────────┬──────────────────┘
                   │
                   ▼
            load_network()
            weights: list[ndarray]
            biases:  list[ndarray]
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼ sparse              ▼ exact
  forward pass (N pts)   DFS from x0
  deduplicate patterns   LP at each facet
  O(N × neurons)         O(regions × LP)
        │                     │
        └──────────┬──────────┘
                   │
                   ▼  Julia → Python boundary
             RegionFindResult
             patterns  (n_regions, total_bits) int8
             offsets   (n_layers+1,)           int64
             centroids (n_regions, input_dim)  float64
             [active_indices, active_offsets,
              bounded — exact methods only]
                   │
                   ▼
         Partition.from_result()
                   │
                   ▼
              Partition
         ┌─────────┼─────────────────┐
         │         │                 │
         ▼         ▼                 ▼
     analyze    visualize         persist
     stats      Plotly/mpl figs  save_partition()
                                 load_partition()
                                      .npz
```

The critical boundary is the Python↔Julia bridge: Julia does all combinatorially intensive work (forward passes, DFS, LP solves) and returns plain numeric arrays. Python rebuilds the `Partition` from those arrays using only NumPy — no Julia call is needed after `compute_partition` returns.

`Partition.halfspaces()` reconstructs `D, g` on the fly from the activation path and network weights. This is pure NumPy arithmetic — fast and allocation-light.

---

## User workflow

### One-shot analysis

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

---

### Training-time tracking

The primary research use case: compute the partition at multiple checkpoints and track how it evolves.

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

---

### Geometric analysis with exact partitions

```python
# Exact enumeration: complete partition from a single starting point
partition = parx.compute_partition(model, np.zeros(2), method="exact_julia")

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

---

## Limitations

These are current constraints, not design goals. See [README.md — Future directions](README.md#future-directions) for planned extensions.

**Architecture support:** Only sequential `Linear → ReLU` stacks. ResNets, transformers, CNNs, networks with batch normalization or dropout, and any model with branching structure are not supported.

**Exact enumeration scaling:** The DFS is exponential in the worst case. In practice `input_dim=2` with up to ~20 neurons per layer is reliable; `input_dim=3` is feasible for small networks; beyond that exact enumeration is often infeasible.

**Sparse enumeration completeness:** Sparse mode can only find regions that your sample points land in. With `input_dim > 5` and complex networks, many regions may never appear in any finite sample.

**Julia startup:** The first call to any Julia-backed method in a process incurs 10–30 seconds of JIT compilation. Call `parx.precompile()` once at startup to amortize this. Subsequent calls within the same process are fast.

**Visualization:** `plot_partition_2d` requires `input_dim == 2`. For higher-dimensional networks use `plot_partition_slice` (fix all but two dimensions) or `plot_partition_projection` (project onto a 2D subspace) — both are approximations. Every plotting function defaults to an interactive Plotly figure; pass `backend="matplotlib"` for a static `matplotlib.figure.Figure` instead (requires `parx[animate]`, no hover tooltips, and `animate_epochs` loses its play/pause/slider controls in favor of a plain `FuncAnimation`).

---

## Architecture — the Python/Julia split

The design separates concerns sharply:

| Layer | Language | Responsibility |
|---|---|---|
| User API | Python | `compute_partition`, `Partition`, analysis, viz |
| Combinatorial search | Julia | Forward passes, DFS, LP solves |
| Bridge | juliacall | Zero-copy NumPy↔Julia array passing |
| Geometry queries | Python (NumPy) | `halfspaces()`, `local_affine()`, routing |
| Visualization | Python (Plotly default, matplotlib optional) | All plotting; `backend=` selects engine |
| Serialization | Python (NumPy `.npz`) | Save/load, no Julia at load time |

Julia is only needed during `compute_partition`. Everything after — analysis, visualization, verification, serialization — is pure Python and runs without Julia. A partition saved to `.npz` can be loaded and analyzed in an environment without Julia installed.

Julia threads default to all available cores. Override with `JULIA_NUM_THREADS=4 python script.py`.
