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
│       ├── docs.yml             # builds & deploys the docs site to GitHub Pages
│       └── release.yml          # builds + publishes to PyPI on GitHub Release (see RELEASING.md)
├── mkdocs.yml                   # docs site config
├── docs/                        # docs site content (see "Docs site" below)
├── tests/
│   ├── conftest.py
│   ├── test_analysis.py
│   ├── test_diagnostics.py
│   ├── test_io.py
│   ├── test_io_partition.py
│   ├── test_julia_bridge.py
│   ├── test_julia_init.py
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
        ├── analysis.py         # neuron stats, complexity profile, volume estimation
        ├── methods/            # region-finding backends
        │   ├── __init__.py
        │   ├── sparse_julia.py
        │   ├── exact_julia.py
        │   ├── exact_julia_fast.py
        │   ├── sparse_python.py
        │   └── exact_python.py
        ├── io.py               # iter_state_dicts helper
        ├── io_partition.py     # save_partition / load_partition (.npz)
        ├── verify.py           # overlap/coverage checks
        ├── viz.py              # Plotly (default) / matplotlib visualizations
        ├── diagnostics.py      # thread_info, benchmark_method
        ├── precompile.py       # Julia JIT warm-up
        ├── juliapkg.json       # Julia runtime dependencies (for juliacall/juliapkg)
        └── julia/
            ├── LinearRegions.jl
            ├── bridge.jl
            ├── sparse.jl
            ├── exact.jl
            ├── lp.jl
            ├── self_test.jl    # standalone smoke test, no Python/juliacall involved
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

The three demo notebooks are embedded on the Notebooks page as pre-exported static HTML (`docs/notebooks/demo_plt.html`, `docs/notebooks/demo_plotly.html`, `docs/notebooks/feature_embedding.html`) — the docs build itself has no Julia/PyTorch dependency, so this export step is manual. **After editing any of `notebooks/demo_plt.py`, `notebooks/demo_plotly.py`, or `notebooks/feature_embedding.py`, re-export before committing:**

```bash
marimo export html notebooks/demo_plt.py -o docs/notebooks/demo_plt.html
marimo export html notebooks/demo_plotly.py -o docs/notebooks/demo_plotly.html
marimo export html notebooks/feature_embedding.py -o docs/notebooks/feature_embedding.html
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
| `PYTHON_JULIACALL_HANDLE_SIGNALS` | `"yes"` | Makes the harmless segfault at exit (when Julia threads are active) happen cleanly after results are already reported, instead of during teardown. It does not prevent the segfault itself — if you see one, check the run's actual output/exit status first; a `"N passed"` line means the run succeeded despite it. |
| `PYTHON_JULIAPKG_EXE` | auto-detected on juliaup (see below) | Force juliacall to use a specific Julia binary instead of auto-resolving one. |

---

## Troubleshooting

### `ERROR: could not load library ".../juliaup/bin/../lib/julia/sys.so": ... No such file or directory`

This used to happen on the **first** call that touches Julia (`ensure_julia()`, any `*_julia`/`exact_julia*` method, or the `julia_session` pytest fixture) on a machine where Julia was installed via [juliaup](https://github.com/JuliaLang/juliaup) — i.e. exactly the install method this repo recommends. **`parx` now works around it automatically** (`_julia_init.py::_resolve_juliaup_shim`) — read on for what it does and what to do if you still hit this.

**Cause:** `julia` on `PATH` is juliaup's launcher shim (`~/.juliaup/bin/julia`), not a real Julia install — juliaup keeps the actual per-version binaries elsewhere (`~/.julia/juliaup/julia-<version>+.../`). `juliapkg` tries to auto-upgrade to the newest Julia release on every resolve; when that opportunistic install fails (network hiccup, a not-yet-fully-available release, etc.) it silently falls back to the raw shim path instead of one of your already-installed, perfectly good Julia versions, and juliacall then derives the system-image path relative to the shim's directory — which never has a `lib/julia/sys.so` next to it. This is a `juliapkg` bug, not a `parx` one; it's invisible in CI because CI installs Julia directly (`julia-actions/setup-julia@v2`), never through juliaup.

**The automatic fix:** before `juliacall` is ever imported, `ensure_julia()` checks whether `julia` on `PATH` resolves to a juliaup launcher shim (`os.path.realpath` ends in `julialauncher`). If so, it reads `~/.julia/juliaup/juliaup.json` directly to find the real binary behind juliaup's *default* channel, and sets `PYTHON_JULIAPKG_EXE` to that real path (via `os.environ.setdefault`, so it never overrides an explicit value you've already set) — juliapkg then uses that binary directly instead of falling back to the broken shim path. Confirmed end-to-end: a from-scratch clone + `uv venv` + `pip install -e ".[dev]"` + `pytest`, with **zero manual env vars**, now passes cleanly (`219 passed`).

**If it still happens anyway** (e.g. juliaup's on-disk metadata format has changed since this was written, or you're using something other than juliaup — a container image with a hand-rolled Julia install, `asdf`, etc.), the detection silently no-ops and you're back to the original failure. Fall back to pointing juliacall at a known-good binary yourself:

```bash
juliaup list                                    # see what's installed, e.g. 1.10.11+0.x64.linux.gnu
export PYTHON_JULIAPKG_EXE="$HOME/.julia/juliaup/julia-1.10.11+0.x64.linux.gnu/bin/julia"
```

(matching the Julia 1.10 that CI and `src/parx/julia/Manifest.toml` are pinned to is the safest bet — a newer version may work but isn't what's tested). Then delete any stale resolution and retry: `rm -rf .venv/julia_env`. If you had to do this, please open an issue — it means the auto-detection missed a case and should be taught about it.

If you'd rather not hardcode a version path, `export PYTHON_JULIAPKG_OFFLINE=yes` also avoids the buggy auto-upgrade path (it makes juliapkg reuse the newest **already-installed** juliaup version instead of trying to fetch a new one) — simpler, but it may still land you on a newer, less-tested Julia than pinning to 1.10 would.

---

## Releasing

See [RELEASING.md](RELEASING.md) for how to cut a new version and what the automation does from there.

---

## Pull Request Checklist

- [ ] Tests pass (`pytest`)
- [ ] Code is formatted (`ruff format`)
- [ ] No lint errors (`ruff check`)
- [ ] New functionality has tests
- [ ] `Manifest.toml` is committed if Julia deps changed