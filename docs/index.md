# parx

**parx** exactly enumerates and analyzes the linear (polyhedral activation) regions of ReLU neural networks.

- **Python layer** — network loading, user API, analysis, visualization, serialization.
- **Julia layer** (via [juliacall](https://github.com/JuliaPy/PythonCall.jl)) — region enumeration algorithms and the combinatorially intensive computation.

!!! note "Alpha software"
    The API is still evolving.

A ReLU network is piecewise affine. parx computes the partition of input space into convex polytopes (linear regions), where each region corresponds to a fixed activation pattern. For each region it provides:

- The activation pattern (which neurons are on/off at each layer)
- An interior point (centroid)
- The halfspace representation `D x ≤ g` of the polytope
- Optional: active (non-redundant) constraint indices and boundedness flag (exact methods)

Read **[Concepts](concepts.md)** for what this means and why it's useful, **[Usage](usage.md)** for the API surface and workflows, **[Examples](examples.md)** for runnable scripts, **[Reference](reference.md)** for exact function signatures, or jump straight to the **[Notebooks](notebooks/index.md)** for a full worked example — training a classifier on the two-moons dataset and watching its partition emerge.

## Requirements

- Python 3.10+
- Julia 1.10+ on `PATH` (install via [juliaup](https://github.com/JuliaLang/juliaup): `curl -fsSL https://install.julialang.org | sh`)

## Installation

`parx` is not yet published on PyPI (see the project's [road2publish.md](https://github.com/Johanmkr/parx/blob/main/road2publish.md) for the plan to get there). Until then, install from source:

```bash
git clone https://github.com/Johanmkr/parx
cd parx
pip install -e .
```

See [CONTRIBUTING.md](https://github.com/Johanmkr/parx/blob/main/CONTRIBUTING.md) for the full dev-environment setup (`uv`, Julia environment instantiation, running the test suite).

Optional extras:

```bash
pip install -e ".[h5]"        # HDF5 model files (.h5)
pip install -e ".[animate]"   # matplotlib backend + video export
pip install -e ".[embed]"     # UMAP-based feature embedding
pip install -e ".[analysis]"  # PCA projection (plot_partition_pca)
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

!!! note "First call is slow"
    The first Julia-backed call in a process takes 10–30 seconds (JIT compilation) — this is expected, not a hang. Call `parx.precompile()` once at startup to pay that cost upfront instead of on your first `compute_partition()`. See [Usage § Limitations](usage.md#limitations) for details.

Source: [github.com/Johanmkr/parx](https://github.com/Johanmkr/parx).
