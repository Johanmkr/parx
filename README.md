# parx - Polyhedral Affine Region eXplorer

parx exactly enumerates and analyzes linear (polyhedral activation) regions of ReLU neural networks.

- Python layer: network loading, user API, visualization, verification
- Julia layer (via juliacall): region enumeration algorithms and heavy combinatorial work

> Alpha software: the API is still evolving.

## What parx computes

A ReLU network is piecewise affine. parx computes the partition of input space into convex polytopes (linear regions), where each region corresponds to a fixed activation pattern.

## Current implementation status

Implemented now:
- Sparse enumeration: `compute_partition(..., mode="sparse")`
- Exact enumeration: `compute_partition(..., mode="exact")`
- Geometry reconstruction: `Partition.halfspaces(region)` returns `D, g` for `D x <= g`
- Routing: `Partition.route(X)` maps points to known regions (or `None` for uncovered sparse regions)
- Region filtering: `Partition.regions_at_layer(l)`
- Verification helpers: `parx.verify.check_no_overlaps`, `parx.verify.check_covers_space`
- 2D plotting helpers in `parx.viz`:
  - `plot_partition_2d(partition, domain=..., layer=...)`
  - `plot_region_counts(partition)`
  - `plot_halfspaces(partition, region, ...)`

## Requirements

- Python 3.10+
- Julia 1.10+ available on PATH

Install Julia (recommended with juliaup):

```bash
curl -fsSL https://install.julialang.org | sh
julia --version
```

## Installation

```bash
pip install parx
```

Optional HDF5 support (`.h5` models):

```bash
pip install "parx[h5]"
```

## Quick start

```python
# juliacall must be imported before torch
import parx

import numpy as np
import torch
import torch.nn as nn

from parx import compute_partition
from parx.viz import plot_partition_2d

model = nn.Sequential(
    nn.Linear(2, 5), nn.ReLU(),
    nn.Linear(5, 5), nn.ReLU(),
    nn.Linear(5, 1),
)

X = np.random.uniform(-1.0, 1.0, size=(300, 2))
partition = compute_partition(model, X, mode="sparse")

print(f"regions: {len(partition)}")

# Explicit plotting domain and layer-specific visualization
fig = plot_partition_2d(
    partition,
    domain=((-1.0, 1.0), (-1.0, 1.0)),
    layer=1,
)
fig.show()
```

## Examples

Run from repository root:

```bash
python examples/01_identity_network.py
python examples/02_random_mlp.py
```

Notebook version of Example 2:

- `examples/02_random_mlp.ipynb`

## Development setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor workflow.

## Architecture

```text
src/parx/
  __init__.py      public API (compute_partition, load_network, etc.)
  network.py       load model/weights from nn.Module, state_dict, .pth, .h5
  partition.py     Partition object + halfspaces/route/filter helpers
  region.py        Region object (activation_path, centroid, metadata)
  verify.py        overlap/coverage checks on sampled points
  viz.py           Plotly visualizations for partitions and halfspaces
  julia/
    LinearRegions.jl
```

Julia threading defaults to available cores (override with `JULIA_NUM_THREADS`).

## Citation

If you use parx in research, please cite the repository and this software name:

```bibtex
@software{parx,
  author = {Johan Mylius-Kroken},
  title = {parx - Polyhedral Affine Region eXplorer},
  year = {2026},
  url = {https://github.com/Johanmkr/parx}
}
```

## License

MIT. See [LICENCE](LICENCE).
