# parx — Next Phases Development Plan

## Status summary

All seven original phases (PLAN.md) are complete. This document plans the next
development work, organized by theme, with implementation detail and an agent
orchestration strategy for each phase.

---

## Agent strategy

### Built-in agents (use as-is)

| Agent | When to use |
|---|---|
| `Plan` | Design a module's public API before any code is written |
| `Explore` | Search the codebase for an existing pattern before implementing |
| `code-reviewer` | After every phase's implementation; catch correctness bugs |
| `test-writer` | After `code-reviewer` signs off; write targeted unit tests |
| `test-runner` | Verify the full test suite after any phase |

### Custom local agents (define in `.claude/agents/`)

Three project-specific agents are proposed below.  Each lives in
`.claude/agents/<name>.md` and is invoked via `/agents:<name>` or spawned
programmatically via `Agent(subagent_type="<name>")`.

---

#### `.claude/agents/parx-bridge.md`

**Purpose:** Changes that cross the Python↔Julia boundary — updating
`RegionFindResult`, modifying `.jl` files, and ensuring the juliacall data
passing contract holds.

**System prompt:**
```
You are a specialist in the parx Python↔Julia bridge.

Key conventions:
- Julia functions return plain-array named tuples; juliacall wraps them as
  Python objects; `np.asarray(result[i])` converts each field.
- Weights/biases are passed as Python list[np.ndarray]; the Julia side calls
  `_to_weights(py_weights, py_biases)` (bridge.jl) to produce
  Vector{Matrix{Float64}} / Vector{Vector{Float64}}.
- The canonical return shape for any region-finding function is a tuple:
  (patterns::Matrix{Int8}, offsets::Vector{Int64}, centroids::Matrix{Float64})
  where patterns[i, offsets[l]:offsets[l+1]] (0-indexed, Python) is the
  activation at layer l for region i.
- RegionFindResult (methods/__init__.py) is the Python-side dataclass for this.
- Adding optional metadata fields: add them as `np.ndarray | None` with default
  None to RegionFindResult, extend the Julia return tuple to include them, and
  update Partition.from_result to consume them.
- Never call `using LinearRegions` from Python — the module is loaded by path
  via include(); access it as `jl.LinearRegions.<function>`.
- juliacall/juliapkg manages its own Julia env at .venv/julia_env; Julia deps
  must be declared in src/parx/juliapkg.json (not Project.toml).

Your job: make coordinated changes to .jl files and the Python wrappers,
keeping the bridge contract consistent. Run tests with `pytest
tests/test_julia_bridge.py` to verify.
```

---

#### `.claude/agents/parx-analysis.md`

**Purpose:** Implement the analysis and statistics API in `src/parx/analysis.py`.
Pre-loaded with the Partition/Region data model and LP geometry conventions.

**System prompt:**
```
You are implementing the analysis module for parx, a library that enumerates
the linear (polyhedral activation) regions of ReLU neural networks.

Data model:
- Partition: flat list of Region objects + network weights/biases.
- Region.activation_path: list[np.ndarray[bool]], one per hidden layer.
- Partition.halfspaces(region) -> (D, g) such that D @ x <= g defines the
  polytope (pure NumPy, no Julia call).
- Partition.local_affine(region) -> (A, b): the local linear map f(x) = Ax + b.
- parx._lp.chebyshev_center(D, g) -> (x_interior, radius) via scipy HiGHS.

Design principles:
- All analysis functions live in src/parx/analysis.py and are pure Python.
- Functions should accept a Partition and return plain NumPy arrays or dicts.
- Expensive functions (volume estimation, LP-based) should be clearly
  documented as such; offer cheap proxies (e.g. Chebyshev radius) alongside
  expensive exact alternatives.
- Expose useful functions in src/parx/__init__.py under __all__.
- Do not import Julia in this module — all geometry is reconstructed via
  Partition.halfspaces().
```

---

#### `.claude/agents/parx-viz.md`

**Purpose:** Extend `src/parx/viz.py` with higher-dimensional visualization.
Pre-loaded with the existing viz patterns and Plotly conventions.

**System prompt:**
```
You are extending the visualization module (src/parx/viz.py) in parx.

Existing patterns:
- All public functions return go.Figure (Plotly).
- plot_partition_2d requires input_dim==2; it clips polytopes to a view box,
  enumerates pairwise halfspace intersections for exact vertex positions, and
  colors regions by a scalar metric via a `color_by(partition, region) -> float`
  callable.
- _region_vertices_2d(D, g, x_range, y_range): adds bounding-box constraints,
  enumerates pairwise line intersections, filters to those satisfying all
  constraints, sorts CCW. This is the geometry workhorse for 2D.
- Existing color-by helpers: affine_frobenius, affine_spectral, affine_det,
  active_neuron_count.

New work targets higher-dimensional networks by projecting or slicing to 2D:
- Slicing: fix all but 2 input dimensions to chosen values; compute the
  induced 2D halfspace system by substituting the fixed values into D @ x <= g.
- Projection: project halfspace normals onto a 2D subspace (inexact but fast).
- Use Partition.halfspaces(region) to get D, g; then manipulate to 2D.
- Keep the same go.Figure / color_by interface as plot_partition_2d.
```

---

## Phase 8 — Surface exact metadata through the bridge

**Status:** Not started  
**Effort:** Medium (Julia + Python, no new algorithms)  
**Agents:** `parx-bridge` (primary), `code-reviewer`, `test-writer`, `test-runner`

### Motivation

The exact DFS in `exact.jl` already computes `_active_local_indices` for
facet-flipping, but discards the per-region result.  `Region.active_indices`
and `Region.bounded` always stay at their defaults (`None`, `False`).  Exposing
these enables:
- `Partition.halfspaces(region, active_only=True)` to return only the
  non-redundant rows (already has the logic, but `active_indices` is always None).
- Downstream: volume estimation, tighter LP-based checks, bounded-region filters.

### What needs to change

**`src/parx/julia/exact.jl`**
- Accumulate per-region `active_indices` during the DFS:  after the
  `_active_local_indices` call for facet-flipping, record the full set of
  non-redundant rows across all layers for the completed region (not just the
  local layer block).
- Compute `bounded` via an LP: a polytope is bounded iff the Chebyshev LP with
  an unbounded `r` variable still produces a finite optimal (or equivalently,
  solve a feasibility LP with a free variable and check boundedness).
  A cheap proxy: if Chebyshev radius < `max_radius` (1e3), treat as bounded.
- Extend the return tuple to a 6-tuple:
  `(patterns, offsets, centroids, active_indices_flat, active_offsets, bounded)`
  where:
  - `active_indices_flat::Vector{Int32}` — concatenated non-redundant row
    indices across all regions.
  - `active_offsets::Vector{Int64}` — length `n_regions + 1`; region `i` owns
    `active_indices_flat[active_offsets[i]+1 : active_offsets[i+1]]`.
  - `bounded::Vector{Bool}` — length `n_regions`.

**`src/parx/methods/__init__.py`**
- Add optional fields to `RegionFindResult`:
  ```python
  active_indices_flat: np.ndarray | None = None   # int32
  active_offsets: np.ndarray | None = None         # int64
  bounded: np.ndarray | None = None                # bool
  ```

**`src/parx/methods/exact_julia.py` and `exact_julia_fast.py`**
- Extract the extra three arrays from the 6-tuple Julia return and populate
  the new `RegionFindResult` fields.

**`src/parx/methods/exact_python.py`**
- The Python DFS already computes `_active_local_indices` per layer per region.
  Accumulate a flat list; populate `RegionFindResult` consistently with Julia.
- Bounded check: after collecting `D_full, g_full` for the completed region,
  call `chebyshev_center(D_full, g_full, max_radius=1e3)` and check if
  `radius < 1e3`.

**`src/parx/partition.py` — `Partition.from_result`**
- If `result.active_indices_flat` is not None, slice it per-region using
  `active_offsets` and assign to `Region.active_indices`.
- If `result.bounded` is not None, assign `Region.bounded`.

### Tests to add (`tests/test_julia_bridge.py` / `tests/test_methods.py`)

- `test_exact_julia_has_active_indices`: run exact on tiny network, assert
  all regions have `active_indices is not None`.
- `test_exact_julia_bounded_flag`: all regions of a 2D network with finite
  data should have at least some bounded regions.
- `test_active_only_halfspaces`: `partition.halfspaces(r, active_only=True)`
  returns strictly fewer rows than the full system for non-trivial regions.
- Parity test: `exact_python` and `exact_julia_fast` produce the same
  `bounded` values on a shared tiny network.

---

## Phase 9 — Analysis and statistics API

**Status:** Not started  
**Effort:** Medium-large (multiple functions, LP math)  
**Agents:** `parx-analysis` (primary), `Plan` (design API first), `code-reviewer`, `test-writer`, `test-runner`

### New file: `src/parx/analysis.py`

#### 9a — Neuron activity statistics (cheap, no LP)

```python
neuron_activity_rates(partition) -> dict[int, np.ndarray]
```
For each layer `l`, returns an array of shape `(layer_width,)` giving the
fraction of regions in which each neuron is active.  Uses only
`region.activation_path`; O(n_regions × total_neurons).

```python
dead_neurons(partition, *, threshold: float = 0.0) -> list[tuple[int, int]]
```
Returns `(layer, neuron_index)` pairs for neurons whose activity rate is
`<= threshold`.  Wraps `neuron_activity_rates`.

```python
always_active_neurons(partition, *, threshold: float = 1.0) -> list[tuple[int, int]]
```
Neurons active in every region (rate >= threshold).

#### 9b — Partition complexity profile (cheap, no LP)

```python
complexity_profile(partition) -> dict
```
Returns:
```python
{
    "n_regions": int,
    "n_layers": int,
    "input_dim": int,
    "regions_per_layer": list[int],   # unique activation-path prefixes at each depth
    "total_neurons": int,
    "total_constraints": int,         # sum over regions of D.shape[0]
    "mean_constraints_per_region": float,
}
```

#### 9c — Region size proxies (moderate, one LP per region)

```python
region_chebyshev_radii(partition, *, max_radius=1e3) -> np.ndarray
```
Already exists in `verify.py` — move here (keep a re-export in verify.py for
backward compat).

```python
region_size_summary(partition, *, max_radius=1e3) -> dict
```
Returns descriptive statistics (min, median, mean, max, fraction bounded) of
Chebyshev radii across all regions.

#### 9d — Volume estimation (expensive, Monte Carlo)

```python
region_volume_estimate(
    partition,
    region,
    *,
    n_samples: int = 10_000,
    seed: int | None = None,
) -> float
```
Hit-and-run MCMC sampler within the polytope `D @ x <= g`, starting from
`region.centroid`.  Returns an estimate of the region's hypervolume relative
to the unit hypercube.  Clearly documented as approximate and slow.

```python
partition_volume_estimates(
    partition,
    *,
    n_samples: int = 5_000,
    seed: int | None = None,
) -> np.ndarray
```
Vectorized over all regions.

#### 9e — Epoch-over-epoch analysis helpers

```python
complexity_over_epochs(
    partitions: list[Partition],
    labels=None,
) -> dict[str, list]
```
Given a list of partitions (e.g. one per training epoch), returns time-series
of: `n_regions`, mean Chebyshev radius, dead neuron counts.  Designed to
feed directly into Plotly line plots.

#### Expose in public API

Add to `src/parx/__init__.py` `__all__`:
`neuron_activity_rates`, `dead_neurons`, `complexity_profile`,
`region_size_summary`.  Leave volume estimation as `parx.analysis.*` (too
expensive to advertise at top level).

### Tests (`tests/test_analysis.py`)

- Activity rates sum to `n_regions` × layer_width in expectation; rates in [0, 1].
- `dead_neurons` returns empty list for a trivially non-dead network.
- `complexity_profile["n_regions"]` matches `len(partition)`.
- Chebyshev radii are all positive for an exact partition.
- Volume estimate: for a known 2D square polytope, estimate is in reasonable
  range of the true area.

---

## Phase 10 — Higher-dimensional visualization

**Status:** Not started  
**Effort:** Medium (geometry + Plotly)  
**Agents:** `parx-viz` (primary), `code-reviewer`, `test-writer`, `test-runner`

### New functions in `src/parx/viz.py`

#### 10a — Axis-aligned 2D slice

```python
plot_partition_slice(
    partition: Partition,
    free_dims: tuple[int, int],
    fixed_values: dict[int, float],
    *,
    x_range=None,
    y_range=None,
    color_by=affine_frobenius,
    ...
) -> go.Figure
```
Fixes all dimensions except `free_dims[0]` and `free_dims[1]` to the values
in `fixed_values`.  For each region, substitutes fixed values into `D @ x <= g`
to produce a 2D halfspace system, then delegates to `_region_vertices_2d`.
Regions whose fixed-dimension constraints are violated (the slice misses them)
are silently skipped.

**Implementation note:** partition does not need to have `input_dim == 2`.
The induced 2D system is:
```
D_2d = D[:, [free0, free1]]
g_2d = g - D[:, fixed_dims] @ fixed_vals_vector
```
Then pass `D_2d, g_2d` to `_region_vertices_2d`.

#### 10b — Linear projection to 2D

```python
plot_partition_projection(
    partition: Partition,
    projection: np.ndarray,   # shape (input_dim, 2)
    *,
    x_range=None,
    y_range=None,
    color_by=affine_frobenius,
    ...
) -> go.Figure
```
Projects each region's halfspace normals onto the 2D subspace spanned by the
columns of `projection`.  The projected system is:
```
D_proj = D @ projection          # (n_constraints, 2)
```
`g` is unchanged.  This over-approximates the projected polytope (the
projection of a polytope is a polytope, but the dual halfspace representation
is not preserved under projection).  Documented as approximate.

#### 10c — PCA projection helper

```python
plot_partition_pca(
    partition: Partition,
    data: np.ndarray,
    *,
    color_by=affine_frobenius,
    ...
) -> go.Figure
```
Runs `sklearn.decomposition.PCA(n_components=2).fit(data)` and delegates to
`plot_partition_projection` with the PCA components as the projection matrix.
Requires `sklearn` (add as optional dep: `pip install parx[analysis]`).

### Tests (`tests/test_viz.py`)

- `plot_partition_slice` on a 3D partition with `free_dims=(0,1)` returns a
  Figure with at least one trace.
- `plot_partition_projection` runs without error on a non-2D partition.
- Existing 2D tests continue to pass (no regressions).

---

## Phase 11 — Serialization

**Status:** Not started  
**Effort:** Small-medium (pure Python, no new algorithms)  
**Agents:** general-purpose or inline (straightforward enough), `code-reviewer`, `test-writer`, `test-runner`

### Design

Save/load a `Partition` to a `.npz` bundle (NumPy native; no extra deps).

**Format** — one `.npz` archive containing:

| Key | Shape / dtype | Description |
|---|---|---|
| `patterns` | `(n_regions, total_bits)` int8 | activation patterns |
| `offsets` | `(n_layers + 1,)` int64 | layer column boundaries |
| `centroids` | `(n_regions, input_dim)` float64 | interior points |
| `weights_<i>` | `(out, in)` float64 | weight matrix for layer i |
| `biases_<i>` | `(out,)` float64 | bias vector for layer i |
| `active_indices_flat` | `(k,)` int32 or absent | optional |
| `active_offsets` | `(n_regions+1,)` int64 or absent | optional |
| `bounded` | `(n_regions,)` bool or absent | optional |
| `version` | scalar int | format version = 1 |

**API** — new file `src/parx/io_partition.py` (avoids name clash with
existing `src/parx/io.py`):

```python
def save_partition(partition: Partition, path: str | Path) -> None: ...
def load_partition(path: str | Path) -> Partition: ...
```

Expose both in `src/parx/__init__.py` and `__all__`.

**Why `.npz`:**
- Zero extra dependencies (NumPy is already required).
- Human-inspectable with `np.load(path, allow_pickle=False)`.
- Lossless round-trip for all numeric fields.

### Tests (`tests/test_io_partition.py`)

- `save_partition` / `load_partition` round-trip: `len(loaded) == len(original)`,
  all activation paths identical, centroids match.
- Loading a file written with `active_indices` populated preserves
  `Region.active_indices` on all regions.
- Missing optional fields (sparse partition) load cleanly.
- Version mismatch raises a clear `ValueError`.

---

## Phase 12 — Housekeeping

**Status:** Not started  
**Effort:** Small  
**Agents:** inline (no agent overhead needed)

### 12a — Fix stale `CONTRIBUTING.md`

- Remove references to `tests/test_regions.py` and `src/parx/regions.py`
  (these no longer exist).
- Update the project structure diagram to match the actual layout.
- Update "Adding Julia Dependencies" to reference `juliapkg.json` (not
  `Project.toml`) per the current CLAUDE.md guidance.

### 12b — Clarify Julia environment management

CLAUDE.md states deps should go in `src/parx/juliapkg.json`; yet
`src/parx/julia/Project.toml` also defines a separate Julia environment that
CLAUDE.md says is "no longer the active Julia environment."

Steps:
1. Verify which environment is actually loaded at runtime (check `_julia_init.py`).
2. Either: consolidate to `juliapkg.json` only and document that `Project.toml`
   is legacy, or explain why both coexist.
3. Update CLAUDE.md and CONTRIBUTING.md to reflect the final answer.

### 12c — `README.md` API sync

The README still uses `mode="sparse"` / `mode="exact"` (the old API); the
actual API uses `method="sparse_julia"` etc.  Update the Quick Start and
examples sections.

---

## Recommended implementation order

```
Phase 12 (housekeeping)    ← unblock docs confusion first; cheap
        ↓
Phase 8 (exact metadata)   ← unblocks active_only halfspaces + Phase 9c
        ↓
Phase 9a/9b (cheap stats)  ← no LP, can start immediately after bridge
Phase 9c (Chebyshev)       ← moves code from verify.py; needs Phase 8 for active_only
        ↓
Phase 11 (serialization)   ← independent of 9/10; do in parallel
Phase 9d (volume)          ← expensive; do last within Phase 9
        ↓
Phase 10 (viz)             ← independent; parallelizable with Phase 9
```

Phases 11 and 10 are independent of each other and of Phase 9; they can be
worked in any order or in parallel.

---

## Suggested per-phase agent workflow

```
1.  Explore   — locate all files touched by the phase
2.  Plan      — design public API signatures (skip for trivial phases)
3.  [parx-bridge | parx-analysis | parx-viz | inline]  — implement
4.  code-reviewer   — correctness + simplification pass
5.  test-writer     — write tests from reviewer's TEST CASES NEEDED section
6.  test-runner     — verify all tests pass
```

For Phase 8, steps 3–6 should be done once for the Julia side (bridge agent)
and once for the Python side separately, to keep context windows focused.
