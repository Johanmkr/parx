# parx Implementation Plan

## Design shift from the original `geobin` implementation

| Old (`geobin`) | New (`parx`) |
|---|---|
| Python → HDF5 → Julia → HDF5 → Python | Python controls everything; Julia called in-process via `juliacall` |
| Julia builds and owns the tree | Julia finds activation patterns; Python builds the tree |
| HDF5 files as the data bus | NumPy arrays passed directly to Julia (juliacall handles conversion) |
| Matplotlib plots | Plotly `go.Figure` objects |
| CLI entry point (`run_geobin.jl`) | Importable library (`parx.compute_partition(...)`) |

## Data passing convention

`juliacall` converts NumPy arrays to Julia arrays automatically through PythonCall's
AbstractArray interface. No serialisation needed.

**Python → Julia:** pass `list[np.ndarray]` for weights and biases; Julia receives them as
iterable PythonCall proxies and explicitly constructs `Matrix{Float64}` / `Vector{Float64}`.

**Julia → Python:** Julia functions return a named tuple of plain arrays
(`Matrix{Int8}` for activation patterns, `Vector{Int64}` for offsets,
`Matrix{Float64}` for centroids). juliacall wraps these as Python objects that
`np.array(...)` converts to NumPy arrays.

## Phases

### Phase 1 — Julia data bridge  ✅ done
Establish the Python↔Julia plumbing without any file I/O.

- `LinearRegions.jl`: `_to_weights(py_weights, py_biases)` helper (converts Python
  lists of arrays to `Vector{Matrix{Float64}}`) and `network_info(py_weights,
  py_biases) → (input_dim, n_layers)` smoke-test function.
- Test: `tests/test_julia_bridge.py::test_network_info`

### Phase 5 — Network loading  ✅ done
`src/parx/network.py`: `load_network(model, *, include_output_layer=False)`

Accepts: `nn.Module`, PyTorch `state_dict` dict, path to `.pth`, path to `.h5`
(optional `h5py`).  
Returns: `(weights: list[np.ndarray], biases: list[np.ndarray])` with weights shaped
`(out_features, in_features)`.  
Default: output layer excluded (`include_output_layer=False`) because only hidden ReLU
layers define the polyhedral partition.

### Phase 4 — Python data model  ✅ done
- `src/parx/region.py`: `Region` — activation path, centroid, optional active_indices,
  bounded flag.
- `src/parx/partition.py`: `Partition` — flat list of regions plus weights/biases.
  - `halfspaces(region)` — reconstruct `D*x ≤ g` on the fly (port of Julia's
    `compute_path_geometry`; pure NumPy).
  - `route(X)` — forward pass + hash-lookup to assign each row of X to its region.
  - `regions_at_layer(l)` — filter by activation path length.

### Phase 2 — Julia sparse region finder  ✅ done
`LinearRegions.jl`: `find_regions_sparse(py_weights, py_biases, py_points)`

- Parallel forward pass (`Threads.@threads`) to compute activation paths for all points.
- Deduplicate into unique patterns.
- Returns `(patterns::Matrix{Int8}, offsets::Vector{Int64}, centroids::Matrix{Float64})`
  where `patterns[i, offsets[l]:offsets[l+1]]` is the activation at layer `l` for region `i`.
- No LP, no Julia deps beyond `LinearAlgebra`.
- `Partition._from_sparse_output(...)` class method converts the Julia output.

### Phase 3 — Julia exact region finder (DFS + facet-flip)
`LinearRegions.jl`: `find_regions_exact(py_weights, py_biases, py_x0)`

- Port of `_build_dfs!` from the original `construction.jl`.
- LP feasibility via `JuMP` + `HiGHS` (add to `Project.toml`).
- Returns same shape as Phase 2 plus `active_indices_flat::Vector{Int32}`,
  `active_offsets::Vector{Int64}`, `bounded::Vector{Bool}`.
- `Partition._from_exact_output(...)` class method.

### Phase 6 — Public API  ✅ done (sparse mode)
`parx.compute_partition(model, data=None, *, mode="sparse", include_output_layer=False)`

- `mode="sparse"`: calls Phase 2; `data` is required (the point set).
- `mode="exact"`: calls Phase 3; `data` used only to pick starting point `x0`.
- Returns a `Partition`.

### Phase 7 — Plotly visualisation
`src/parx/viz.py`:

- `plot_partition_2d(partition, x_range, y_range, resolution=200)` — rasterise 2D
  input space by routing a grid; color by region.
- `plot_region_counts(partition)` — bar chart of region count per layer.
- `plot_halfspaces(partition, region, x_range, y_range)` — overlay halfspace boundaries.

All functions return `go.Figure`.

## Suggested implementation order

```
Phase 1 (bridge) → Phase 5 (network load) → Phase 4 (data model)
                                                      ↓
                    Phase 2 (sparse Julia) → Phase 6 (API) → Phase 7 (viz)
                    Phase 3 (exact Julia) ↗
```

Phases 2 and 3 are independent of each other.  
Phase 7 depends only on Phase 4 (the `Partition` interface), not on Julia.
