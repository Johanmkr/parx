# polarx

**POLyhedral Activation Region Xplorer**

`polarx` exactly enumerates and represents the linear regions (polyhedral activation regions) of ReLU-based neural networks. The combinatorially heavy computations are performed in Julia for performance, while the Python layer handles network loading, user-facing API, and integration with the PyTorch ecosystem.

> ⚠️ **Alpha software.** The API is not yet stable.

---

## What it does

A neural network with ReLU activations is a continuous piecewise-linear (CPWL) function. Its input space is partitioned into convex polytopes — *linear regions* — inside each of which the network behaves as a single affine map. `polarx` computes that partition exactly, without sampling or approximation.

---

## Requirements

### Python
- Python ≥ 3.10
- Dependencies are managed automatically by pip/uv (see below)

### Julia
Julia must be installed separately and available on your `PATH`. The recommended way is via [juliaup](https://github.com/JuliaLang/juliaup):

**macOS / Linux:**
```bash
curl -fsSL https://install.julialang.org | sh
```

**Windows:**
```powershell
winget install julia -s msstore
```

Verify your installation:
```bash
julia --version   # should print Julia 1.10 or newer
```

> On first use, `polarx` will automatically instantiate the embedded Julia environment and download Julia dependencies. This takes a minute or two and only happens once.

---

## Installation

```bash
pip install polarx
```

With optional HDF5 support (for `.h5` network files):
```bash
pip install polarx[h5]
```

---

## Quick Start

```python
import polarx

# Load a network from a PyTorch checkpoint
network = polarx.load("my_model.pth")

# Compute the polyhedral partition over a bounded input domain
partition = polarx.compute_partition(network, domain=...)

# Inspect regions
print(f"Number of linear regions: {len(partition)}")
for region in partition:
    print(region.affine_map)   # the A, b of Ax + b on this region
    print(region.vertices)     # vertices of the polytope
```

---

## Development Setup

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. Quick version:

```bash
git clone https://github.com/yourname/polarx
cd polarx
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

---

## Architecture

```
polarx/
├── Python layer        network loading, user API, data conversion
└── Julia layer         polyhedral enumeration, linear algebra, threading
```

The Julia runtime is embedded via [juliacall](https://github.com/cjdoris/PythonCall.jl). Julia code runs in parallel using native threads (`Threads.@spawn`). The number of threads defaults to the number of available CPU cores and can be overridden via the `JULIA_NUM_THREADS` environment variable.

---

## Related Work

- [SplineCam](https://github.com/AhmedImtiazPrio/splinecam) — exact 2D-slice visualization of DN geometry (CVPR 2023)
- [relu_edge_subdivision](https://github.com/arturs-berzins/relu_edge_subdivision) — GPU-based polyhedral complex extraction (ICML 2023)

---

## Citation

If you use `polarx` in your research, please cite:

```bibtex
@software{polarx,
  author  = {Your Name},
  title   = {polarx: POLyhedral Activation Region Xplorer},
  year    = {2025},
  url     = {https://github.com/yourname/polarx}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).