# parx — Polyhedral Affine Region eXplorer

parx exactly enumerates and analyzes the linear (polyhedral activation) regions of ReLU neural networks.

- **Python layer:** network loading, user API, analysis, visualization, serialization
- **Julia layer** (via juliacall): region enumeration algorithms and combinatorial computation

> Alpha software — the API is still evolving.

**Docs site:** [johanmkr.github.io/parx](https://johanmkr.github.io/parx/) — includes the two demo notebooks below, rendered and browsable without cloning the repo.

For a conceptual introduction — what parx computes, how sparse vs. exact enumeration differ, the data pipeline, workflow examples, and current limitations — see **[OVERVIEW.md](OVERVIEW.md)** (also on the docs site).

## What parx computes

A ReLU network is piecewise affine. parx computes the partition of input space into convex polytopes (linear regions), where each region corresponds to a fixed activation pattern. For each region it provides:

- The activation pattern (which neurons are on/off at each layer)
- An interior point (centroid)
- The halfspace representation `D x ≤ g` of the polytope
- Optional: active (non-redundant) constraint indices and boundedness flag (exact methods)

## Requirements

- Python 3.10+
- Julia 1.10+ on `PATH`

Install Julia via [juliaup](https://github.com/JuliaLang/juliaup):

```bash
curl -fsSL https://install.julialang.org | sh
```

## Installation

```bash
pip install parx
```

Optional extras:

```bash
pip install "parx[h5]"        # HDF5 model files (.h5)
pip install "parx[analysis]"  # PCA-based visualization (scikit-learn)
```

## Quick start

```python
# juliacall must be imported before torch — parx handles this at import time
import parx
import numpy as np
import torch.nn as nn
from parx import compute_partition
from parx.viz import plot_partition_2d

model = nn.Sequential(
    nn.Linear(2, 8), nn.ReLU(),
    nn.Linear(8, 8), nn.ReLU(),
    nn.Linear(8, 1),
)

X = np.random.uniform(-1.0, 1.0, size=(500, 2))
partition = compute_partition(model, X, method="sparse_julia")

print(f"Found {len(partition)} regions")

fig = plot_partition_2d(partition, domain=((-1, 1), (-1, 1)))
fig.show()
```

## Enumeration methods

| Method | Description |
|---|---|
| `"sparse_julia"` | Parallel forward-pass scan; fast, may miss low-density regions |
| `"exact_julia"` | DFS + facet-flip; complete enumeration (JuMP/HiGHS) |
| `"exact_julia_fast"` | Same DFS with optimized LP calls |
| `"sparse_python"` | Pure-Python forward-pass scan; slow, no Julia required |
| `"exact_python"` | Pure-Python DFS; slow, no Julia required |

```python
partition = compute_partition(model, X, method="exact_julia")
parx.list_methods()  # → list of available method names
```

## API reference

### Partition

```python
len(partition)                          # number of regions
partition.halfspaces(region)            # → (D, g) for D @ x <= g
partition.halfspaces(region, active_only=True)  # non-redundant rows only (exact methods)
partition.route(X)                      # → list[Region | None]; None if uncovered
partition.regions_at_layer(l)          # → list[Region] filtered by path depth
partition.local_affine(region)          # → (A, b) for the local linear map f(x) = Ax + b
```

### Network loading

```python
weights, biases = parx.load_network(model)         # nn.Module
weights, biases = parx.load_network(state_dict)    # PyTorch state_dict
weights, biases = parx.load_network("model.pth")   # .pth file
weights, biases = parx.load_network("model.h5")    # .h5 file (requires parx[h5])
```

### Analysis

```python
from parx import (
    neuron_activity_rates,   # fraction of regions where each neuron fires
    dead_neurons,            # neurons with activity rate ≤ threshold
    always_active_neurons,   # neurons active in ≥ threshold fraction of regions
    complexity_profile,      # structural statistics dict
    region_size_summary,     # Chebyshev-radius statistics (one LP per region)
)
from parx.analysis import (
    region_chebyshev_radii,      # per-region inscribed-ball radii
    region_volume_estimate,      # Monte Carlo volume estimate for one region
    partition_volume_estimates,  # batch volume estimates (slow)
    complexity_over_epochs,      # track stats across a sequence of partitions
)
```

### Serialization

```python
parx.save_partition(partition, "partition.npz")
partition = parx.load_partition("partition.npz")
```

Partitions are saved as `.npz` bundles (NumPy-native, no extra dependencies). Active indices and boundedness flags are preserved when present.

### Epoch / training analysis

```python
from parx import iter_state_dicts
from parx.viz import animate_epochs, animate_epochs_video

# Iterate over checkpoints saved in a single .pth file or directory
for epoch, state_dict in iter_state_dicts("checkpoints/"):
    partition = compute_partition(state_dict, X)
    ...

# Animated Plotly figure showing how the partition evolves
fig = animate_epochs(partitions, domain=((-1, 1), (-1, 1)))
fig.show()

# Matplotlib FuncAnimation instead — no slider, but playable via .to_jshtml()
anim = animate_epochs(partitions, backend="matplotlib")

# Export to MP4/GIF
animate_epochs_video(partitions, domain=((-1, 1), (-1, 1)), path="partition.gif")
```

### Verification

```python
from parx.verify import (
    check_no_overlaps,         # assert no point lands in two regions
    check_covers_space,        # assert all sample points are covered
    check_regions_nonempty,    # assert all regions contain their centroid
    check_routing_consistency, # route(X) agrees with forward-pass labels
    sample_near_boundaries,    # generate test points near region boundaries
)
```

### Visualization

Every plotting function accepts a keyword-only `backend: Literal["plotly", "matplotlib"] = "plotly"` argument. `"plotly"` (default) returns an interactive `plotly.graph_objects.Figure`; `"matplotlib"` returns a static `matplotlib.figure.Figure` (or `matplotlib.animation.FuncAnimation` for `animate_epochs`) and requires `pip install "parx[animate]"`. Matplotlib output has no hover tooltips, and `animate_epochs(backend="matplotlib")` has no play/pause/slider controls — use `.to_jshtml()` in a notebook or `.save(...)` to export it.

```python
from parx.viz import (
    plot_partition_2d,         # 2D input space; color by metric
    plot_partition_slice,      # slice higher-dim partition to 2D
    plot_partition_projection, # project halfspace normals to 2D
    plot_partition_pca,        # PCA projection (requires parx[analysis])
    plot_region_counts,        # bar chart of regions per layer
    plot_halfspaces,           # halfspace boundary overlay
)

fig = plot_partition_2d(partition, backend="matplotlib")
fig.savefig("partition.png")
```

Color-by callables: `affine_frobenius`, `affine_spectral`, `affine_det`, `active_neuron_count` (all in `parx.viz`).

### Utilities

```python
parx.precompile()         # warm up Julia JIT (call once at startup)
parx.list_methods()       # → list of registered method names

from parx.diagnostics import thread_info, benchmark_method
thread_info()             # Julia thread count and environment info
benchmark_method(model, X, method="sparse_julia")  # timing comparison
```

## Examples

```bash
python examples/01_identity_network.py
python examples/02_random_mlp.py
```

## Development setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor workflow.

## Architecture

```text
src/parx/
  __init__.py         public API
  network.py          load_network() — nn.Module, state_dict, .pth, .h5
  region.py           Region dataclass
  partition.py        Partition — halfspaces, route, filter, local_affine
  analysis.py         neuron stats, complexity profile, volume estimation
  verify.py           overlap/coverage/routing checks
  viz.py              Plotly visualizations (2D, slice, projection, PCA, animation)
  io.py               iter_state_dicts — iterate checkpoints
  io_partition.py     save_partition / load_partition (.npz)
  diagnostics.py      thread_info, benchmark_method
  precompile.py       Julia JIT warm-up
  _lp.py              Chebyshev center LP helper (scipy HiGHS)
  _julia_init.py      lazy Julia runtime initialisation (singleton)
  _check.py           startup check — Julia on PATH
  methods/
    __init__.py       RegionFindResult, register_method, get_method
    sparse_julia.py
    exact_julia.py
    exact_julia_fast.py
    sparse_python.py
    exact_python.py
  julia/
    LinearRegions.jl  module wrapper
    bridge.jl         Python↔Julia helpers
    sparse.jl         parallel forward-pass enumeration
    exact.jl          DFS + facet-flip exact enumeration
    lp.jl             LP helpers (Chebyshev center, active indices)
```

Julia threads default to all available cores; override with `JULIA_NUM_THREADS`.

## Citation

```bibtex
@software{parx,
  author = {Johan Mylius-Kroken},
  title  = {parx — Polyhedral Affine Region eXplorer},
  year   = {2026},
  url    = {https://github.com/Johanmkr/parx}
}
```

## License

MIT. See [LICENCE](LICENCE).

---

## Future directions

The core enumeration and analysis pipeline is complete. For a concrete, prioritized task list with implementation detail see **[TODO.md](TODO.md)**. High-level directions:

### Non-sequential architectures

The current bridge assumes a simple sequential stack of `Linear → ReLU` layers. Supporting residual connections, skip connections, and batch normalization would broaden applicability to modern networks (ResNets, MLPs with normalization).

### Convolutional and recurrent networks

Region enumeration for CNNs and RNNs requires adapting the activation-pattern representation and the facet-flip DFS to handle weight sharing and sequential state. This is a significant algorithmic extension.

### Tighter LP-based geometry

Phase 8 surfaces active (non-redundant) constraint indices from the exact DFS. These could be used to accelerate Chebyshev-center and volume computations, and to implement exact facet enumeration (vertex representation via `pycddlib` or `scipy`).

### Lipschitz and gradient analysis per region

Each linear region has a well-defined local Jacobian. Adding `partition.local_jacobian(region)` and per-region Lipschitz constants (largest singular value) would enable gradient-based network analysis.

### Training-time integration

A PyTorch callback / Lightning hook that computes the partition at configurable intervals during training and logs complexity metrics (region count, dead neurons, mean Chebyshev radius) would make `complexity_over_epochs` usable with standard training loops.

### Symbolic / formal verification interface

Exporting the halfspace representation to SMT-LIB or MILP format (for Gurobi, CPLEX, or `z3`) would connect parx to formal neural-network verification tools.

### Scalability

The exact DFS is exponential in the worst case. Heuristics such as region sampling budgets, early termination, and parallel DFS branches (Julia multi-threading already in place) could make exact enumeration practical for deeper networks.

### Packaging and distribution

- Publish to PyPI once the API stabilizes
- Add a CI matrix covering Julia 1.10/1.11 × Python 3.10–3.12
- Add type stubs (`.pyi`) for IDE autocompletion
