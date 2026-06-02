# Contributing to parx

## Prerequisites

- Python ≥ 3.10
- Julia ≥ 1.10 (install via [juliaup](https://github.com/JuliaLang/juliaup))
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

---

## Setting Up the Development Environment

### 1. Clone the repository

```bash
git clone https://github.com/yourname/parx
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

### 3. Instantiate the Julia environment

On first use, Julia needs to download and precompile its dependencies:
```bash
julia --project=src/parx/julia -e "using Pkg; Pkg.instantiate()"
```

This only needs to be done once (or after changes to `julia/Project.toml`).

### 4. Verify the setup

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
│       └── ci.yml              # GitHub Actions CI
├── tests/
│   ├── test_network.py
│   ├── test_regions.py
│   └── conftest.py
└── src/
    └── parx/
        ├── __init__.py         # public API surface
        ├── _check.py           # startup checks (Julia on PATH, etc.)
        ├── _julia_init.py      # Julia runtime initialization
        ├── network.py          # network loading (.pth, .h5)
        ├── regions.py          # Python-facing partition API
        └── julia/
            ├── Project.toml    # Julia package environment
            ├── Manifest.toml   # locked Julia dependencies
            └── LinearRegions.jl
```

---

## Development Workflow

### Running tests
```bash
pytest                          # run all tests
pytest tests/test_regions.py   # run a specific file
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

## Adding Julia Dependencies

```bash
cd src/parx/julia
julia --project=.
```

Then inside Julia:
```julia
using Pkg
Pkg.add("SomePackage")   # updates Project.toml and Manifest.toml
```

Commit both `Project.toml` and `Manifest.toml`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `JULIA_NUM_THREADS` | `"auto"` | Number of Julia threads |
| `JULIA_PROJECT` | set automatically | Path to Julia project — do not set manually |

---

## Pull Request Checklist

- [ ] Tests pass (`pytest`)
- [ ] Code is formatted (`ruff format`)
- [ ] No lint errors (`ruff check`)
- [ ] New functionality has tests
- [ ] `Manifest.toml` is committed if Julia deps changed