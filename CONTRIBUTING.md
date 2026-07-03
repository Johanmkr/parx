# Contributing to parx

## Prerequisites

- Python ≥ 3.10
- Julia ≥ 1.10 (install via [juliaup](https://github.com/JuliaLang/juliaup))
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

---

## Setting Up the Development Environment

### 1. Clone the repository

```bash
git clone https://github.com/Johanmkr/parx
cd parx
```

### 2. Create a virtual environment and install dependencies

**With uv (recommended):**
```bash
uv venv                         # creates .venv/
source .venv/bin/activate       # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

**With pip:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Verify the setup

```bash
pytest
```

All tests should pass.

---

## Project Structure

```
parx/
├── pyproject.toml              # build config, dependencies, tool config
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── .github/
│   └── workflows/
│       ├── ci.yml               # GitHub Actions CI
│       └── docs.yml             # builds & deploys the docs site to GitHub Pages
├── mkdocs.yml                   # docs site config
├── docs/                        # docs site content (see "Docs site" below)
├── tests/
│   ├── conftest.py
│   ├── test_analysis.py
│   ├── test_diagnostics.py
│   ├── test_io.py
│   ├── test_io_partition.py
│   ├── test_julia_bridge.py
│   ├── test_methods.py
│   ├── test_mlp.py
│   ├── test_network.py
│   ├── test_partition.py
│   ├── test_verify.py
│   └── test_viz.py
└── src/
    └── parx/
        ├── __init__.py         # public API surface
        ├── _check.py           # startup checks (Julia on PATH, etc.)
        ├── _julia_init.py      # Julia runtime initialization
        ├── _lp.py              # Chebyshev center LP helper
        ├── network.py          # network loading (.pth, .h5)
        ├── region.py           # Region dataclass
        ├── partition.py        # Partition object + halfspaces/route/filter
        ├── methods/            # region-finding backends
        │   ├── __init__.py
        │   ├── sparse_julia.py
        │   ├── exact_julia.py
        │   ├── exact_julia_fast.py
        │   ├── sparse_python.py
        │   └── exact_python.py
        ├── io.py               # iter_state_dicts helper
        ├── verify.py           # overlap/coverage checks
        ├── viz.py              # Plotly (default) / matplotlib visualizations
        ├── juliapkg.json       # Julia runtime dependencies (for juliacall/juliapkg)
        └── julia/
            ├── LinearRegions.jl
            ├── bridge.jl
            ├── sparse.jl
            ├── exact.jl
            ├── lp.jl
            ├── Project.toml    # standalone Julia environment (for direct Julia testing)
            └── Manifest.toml   # locked deps for standalone environment
```

---

## Development Workflow

### Running tests
```bash
pytest                          # run all tests
pytest tests/test_methods.py   # run a specific file
pytest -x                      # stop on first failure
pytest --cov=parx               # with coverage
```

### Linting and formatting
```bash
ruff check src/ tests/          # lint
ruff format src/ tests/         # format
```

### Testing the Julia code independently

You can test Julia code directly without going through Python:
```bash
cd src/parx/julia
julia --project=. -e "using LinearRegions; LinearRegions.run_tests()"
```

---

## Docs site

The docs site (published at [johanmkr.github.io/parx](https://johanmkr.github.io/parx/)) is built with MkDocs from `docs/` and deployed by `.github/workflows/docs.yml` whenever `docs/` or `mkdocs.yml` changes on `main`.

```bash
uv pip install -e ".[docs]"
mkdocs serve      # live preview at http://127.0.0.1:8000
mkdocs build --strict   # what CI runs; fails on broken nav/links
```

The two demo notebooks are embedded on the Notebooks page as pre-exported static HTML (`docs/notebooks/demo_plt.html`, `docs/notebooks/demo_plotly.html`) — the docs build itself has no Julia/PyTorch dependency, so this export step is manual. **After editing either `notebooks/demo_plt.py` or `notebooks/demo_plotly.py`, re-export before committing:**

```bash
marimo export html notebooks/demo_plt.py -o docs/notebooks/demo_plt.html
marimo export html notebooks/demo_plotly.py -o docs/notebooks/demo_plotly.html
```

---

## Adding Julia Dependencies

parx uses two separate Julia environments:

- **Runtime environment** — managed by juliacall/juliapkg, stored in `.venv/julia_env`.
  Declare new packages in `src/parx/juliapkg.json`.  juliacall resolves and installs them
  automatically on the next `import parx`.  No manual `Pkg.add` needed.

- **Standalone testing environment** — `src/parx/julia/Project.toml`.  Used only for
  running Julia code directly (see "Testing the Julia code independently" above).
  Add packages here with:

```bash
cd src/parx/julia
julia --project=.
```
```julia
using Pkg
Pkg.add("SomePackage")   # updates Project.toml and Manifest.toml
```

Commit both `Project.toml` and `Manifest.toml` when changing the standalone environment.

For packages needed at runtime, edit `src/parx/juliapkg.json` and commit that file.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `JULIA_NUM_THREADS` | `"auto"` | Number of Julia threads |
| `PYTHON_JULIACALL_HANDLE_SIGNALS` | `"yes"` | Suppress harmless segfault at exit when Julia threads are active |

---

## Pull Request Checklist

- [ ] Tests pass (`pytest`)
- [ ] Code is formatted (`ruff format`)
- [ ] No lint errors (`ruff check`)
- [ ] New functionality has tests
- [ ] `Manifest.toml` is committed if Julia deps changed