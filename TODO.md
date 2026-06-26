# parx — TODO and Future Directions

Concrete next steps, ordered roughly by effort and impact. Each item includes the motivation, a specific implementation description, and pointers to further work it would unlock.

---

## Tier 1 — Housekeeping (hours each, no design decisions)

### 1.1 Fix `_exact_555_partition()` in `test_mlp.py`

**Problem:** `_exact_555_partition()` is a plain helper function called by five separate test functions. Each call triggers a full Julia DFS — roughly 5 redundant exact enumerations per test run. This is the dominant factor in the ~53s test runtime.

**Fix:** Convert to a `@pytest.fixture(scope="module")`:

```python
@pytest.fixture(scope="module")
def exact_555_partition():
    return compute_partition(_mlp555(), np.zeros(2), method="exact_julia")
```

Update the five test functions to accept it as a parameter. The DFS runs once; the result is shared across the module.

**Unlocks:** Faster CI. Makes adding more exact-mode tests cheap.

---

### 1.2 Remove the vacuous volume test

**Problem:** `test_analysis.py::TestRegionVolumeEstimate::test_different_seeds_may_differ` only checks that two floats are finite-or-inf. This is true of any float returned by the function and the test can never fail. It gives false confidence.

**Fix:** Delete it, or replace it with a test that actually checks the Monte Carlo estimate against a known analytic answer. The unit square `[0, 1]^2` has volume 1.0 and can be constructed by hand:

```python
def test_volume_estimate_unit_square():
    # D @ x <= g defines [0,1]^2: x1 >= 0, x2 >= 0, x1 <= 1, x2 <= 1
    D = np.array([[-1,0],[0,-1],[1,0],[0,1]], dtype=float)
    g = np.array([0, 0, 1, 1], dtype=float)
    centroid = np.array([0.5, 0.5])
    # Monkeypatch partition.halfspaces to return this system directly,
    # or build a network whose single region IS [0,1]^2 by appropriate W, b.
    ...
    vol = region_volume_estimate(partition, region, n_samples=50_000, seed=0)
    assert abs(vol - 1.0) < 0.05   # within 5% of true area
```

---

### 1.3 Add tests for `diagnostics.py` and `precompile.py`

**Problem:** Both modules are in `__all__` and completely untested.

**Fix:** Add a `tests/test_diagnostics.py`:

```python
from parx.diagnostics import thread_info, benchmark_method

def test_thread_info_returns_valid_dict(julia_session):
    info = thread_info()
    assert info["n_threads"] >= 1
    assert isinstance(info["julia_version"], str)

def test_benchmark_method_returns_timings(julia_session):
    model = ...  # tiny model
    X = np.zeros((10, 2))
    timings = benchmark_method(model, X, method="sparse_julia")
    assert "elapsed_s" in timings
    assert timings["elapsed_s"] > 0
```

And a test for `precompile()` in `tests/test_julia_bridge.py` (already has the `julia_session` fixture):

```python
def test_precompile_returns_timing_dict():
    result = parx.precompile()
    assert isinstance(result, dict)
    assert all(v >= 0 for v in result.values())
```

---

### 1.4 Add `.pth` file loading test to `test_network.py`

**Problem:** `load_network("model.pth")` is the most common real-world usage and has no test. Only `nn.Module` and `state_dict` are tested.

**Fix:**

```python
def test_load_from_pth_file(tmp_path):
    model = _mlp([2, 4, 2])
    path = tmp_path / "model.pth"
    torch.save(model.state_dict(), path)
    weights, biases = load_network(path)
    assert len(weights) == 1
    assert weights[0].shape == (4, 2)
```

Also add a test for unsupported file type:

```python
def test_load_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "model.pkl"
    bad.write_bytes(b"")
    with pytest.raises((ValueError, RuntimeError)):
        load_network(bad)
```

---

### 1.5 Fix the juliacall-before-torch warning

**Problem:** Several test files import `torch` at module level before juliacall is loaded. When pytest collects those modules, torch is imported first, triggering a `UserWarning: torch was imported before juliacall`. The comment `# juliacall must be imported before torch` at the top of those files is therefore misleading — the files do the opposite.

**Fix:** In `conftest.py`, import `parx` (which loads juliacall) before any test module is imported:

```python
# conftest.py
import parx  # ensure juliacall is loaded before torch in any test module
```

Since `conftest.py` is loaded before test modules are collected, this guarantees import order. Remove the now-redundant comment from individual test files.

---

## Tier 2 — Correctness (days each, high research impact)

### 2.1 Sparse centroid → Chebyshev center

**Problem:** In sparse mode, `region.centroid` is the arithmetic mean of whichever data points landed in that region. For a convex polytope the sample mean is technically inside (by convexity), but:

- It has no radius guarantee — it could be arbitrarily close to a constraint boundary.
- It is inconsistent with exact mode, where the centroid is the true Chebyshev center (largest inscribed ball).
- `region_size_summary`, `region_volume_estimate`, and `plot_partition_2d` all use the centroid as an interior reference point. A centroid near the boundary degrades all of these.

**Fix:** After `Partition._from_sparse_output` builds the regions, run a Chebyshev center LP on each region. The LP infrastructure already exists in `_lp.py`:

```python
# In partition.py, _from_sparse_output:
for region in regions:
    D, g = partition.halfspaces(region)
    center, radius = chebyshev_center(D, g, max_radius=1e3)
    if radius > 0:
        region.centroid = center
```

**Cost:** One LP per region. For a sparse partition with 200 regions this is fast (milliseconds each via HiGHS). Can be disabled with a flag (`recompute_centroids=False`) for users who don't need it.

**Unlocks:** Meaningful `region_size_summary` for sparse partitions. Correct volume estimates. Better `plot_partition_2d` layout (polygons centered properly). Parity between sparse and exact modes on all geometry operations.

---

### 2.2 Better error messages for unsupported architectures

**Problem:** If a user passes a ResNet, a transformer, or any network with BatchNorm/Dropout/Conv layers, `load_network` either silently drops those layers or raises a cryptic error deep in weight extraction. The current behavior is undefined and confusing.

**Fix:** In `network.py`, add an explicit check after extracting layers, raising a descriptive error for any unsupported type:

```python
SUPPORTED = (nn.Linear,)
IGNORABLE = (nn.ReLU, nn.Flatten)

for name, module in model.named_modules():
    if isinstance(module, nn.BatchNorm1d):
        raise ValueError(
            f"Layer '{name}' is BatchNorm1d. parx does not yet support batch "
            f"normalization. At inference time, BatchNorm can be folded into the "
            f"adjacent Linear layer — see TODO.md §3.1 for the planned fix."
        )
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        raise ValueError(
            f"Layer '{name}' is a convolutional layer. parx currently only "
            f"supports fully-connected (Linear) layers."
        )
```

**Unlocks:** Users understand immediately what's wrong instead of getting a KeyError or shape mismatch 10 layers deep.

---

## Tier 3 — Architecture support (days to a week, unlocks real networks)

### 3.1 BatchNorm folding in `load_network`

**Problem:** Virtually every non-toy trained network uses batch normalization. A `BatchNorm1d` layer following a `Linear` layer is at inference time (with `model.eval()`) mathematically equivalent to a rescaled and shifted linear layer. Specifically, for `y = BN(Wx + b)`:

```
y_i = gamma_i * (W_i x + b_i - mu_i) / sqrt(sigma_i^2 + eps) + beta_i
```

This is just another affine map: `W'_i = gamma_i / sqrt(sigma_i^2 + eps) * W_i`, `b'_i = gamma_i * (b_i - mu_i) / sqrt(sigma_i^2 + eps) + beta_i`. The resulting `W'`, `b'` can be passed to Julia exactly as regular weights. No change to the Julia side is needed.

**Implementation:** In `network.py`, detect `Linear → BatchNorm1d` pairs and fold the BN parameters into the preceding weight matrix and bias vector. Only valid when `model.training == False`.

```python
def _fold_batchnorm(W, b, bn: nn.BatchNorm1d) -> tuple[np.ndarray, np.ndarray]:
    gamma = bn.weight.detach().numpy()
    beta  = bn.bias.detach().numpy()
    mu    = bn.running_mean.detach().numpy()
    var   = bn.running_var.detach().numpy()
    scale = gamma / np.sqrt(var + bn.eps)
    W_new = scale[:, None] * W
    b_new = scale * (b - mu) + beta
    return W_new, b_new
```

**Add a test:** `load_network` on a `Linear → BatchNorm1d → ReLU` model should return the same partition as `load_network` on the manually folded equivalent.

**Unlocks:** Support for most trained MLPs. This is the highest-leverage single change for making parx useful beyond toy networks. Also unlocks: networks trained with weight normalization (same folding trick applies).

---

### 3.2 Dropout support

**Problem:** `nn.Dropout` at inference time (after `model.eval()`) is an identity operation. But `load_network` doesn't know to skip it.

**Fix:** Add `nn.Dropout` to the list of ignorable layer types in `network.py`. One line.

---

### 3.3 LeakyReLU support

**Problem:** `nn.LeakyReLU(negative_slope=alpha)` is also piecewise linear: `f(x) = x` if `x > 0`, `f(x) = alpha * x` if `x <= 0`. The polyhedral partition structure is identical — the same binary activation pattern still defines regions. The only difference is that inactive neurons contribute `alpha * W_i x` instead of `0` to the next layer.

**Fix:** In `partition.py`, `halfspaces()` and `local_affine()` currently hard-code the "inactive neuron → zero row" assumption. Parameterize with `negative_slope` per layer, stored in `Partition`. The Julia bridge would need a corresponding change in `bridge.jl` to pass the slope values and use them in the LP constraint construction.

**Unlocks:** A much wider class of architectures. LeakyReLU and PReLU are common in image generation networks.

---

## Tier 4 — Exact enumeration usability (days, research workflow)

### 4.1 Julia precompiled sysimage

**Problem:** The first call to any Julia-backed method in a fresh process triggers 10–30 seconds of JIT compilation. `parx.precompile()` exists but must be called explicitly and still doesn't persist across processes.

**Fix:** Use `PackageCompiler.jl` to build a precompiled Julia sysimage that includes `LinearRegions.jl` and its dependencies (JuMP, HiGHS). Ship the sysimage as part of the parx package. juliacall can be told to use a custom sysimage via an environment variable.

```bash
# One-time build (run during package install or manually):
julia --project=src/parx/julia -e "
    using PackageCompiler
    create_sysimage([:LinearRegions, :JuMP, :HiGHS];
        sysimage_path=\"src/parx/julia/parx.so\",
        precompile_execution_file=\"src/parx/julia/precompile_workload.jl\")
"
```

```python
# In _julia_init.py:
os.environ.setdefault("JULIA_CPU_TARGET", "generic")
os.environ.setdefault("JULIA_SYSIMAGE", str(SYSIMAGE_PATH))
```

**Tradeoff:** The sysimage is ~200MB and platform-specific. This is fine for local use; PyPI distribution would require platform wheels. For now, document the build step in `CONTRIBUTING.md` as optional.

**Unlocks:** Sub-second startup for all Julia methods. Makes interactive use (Jupyter, IPython) practical.

---

### 4.2 Progress reporting for exact DFS

**Problem:** For a network with hundreds of regions, `compute_partition(..., method="exact_julia")` runs silently for minutes with no indication of progress. Users can't tell if it's working or hung.

**Fix:** Add an optional callback in the Julia DFS that fires after each region is found. Surface this in Python as a `tqdm` progress bar or a user-supplied callable:

```python
partition = compute_partition(
    model, x0,
    method="exact_julia",
    progress=True,       # tqdm bar showing regions found
    # or:
    progress=my_callback,  # called with (n_found, latest_region) each step
)
```

Julia side: add a Python callback argument to `find_regions_exact`; call `PythonCall.pycall(callback, n_found)` after each region is appended.

**Unlocks:** Interactive use. Also enables early termination: if the callback raises `StopIteration`, the DFS can exit cleanly and return whatever it found so far (useful for budget-limited exploration).

---

### 4.3 Budget-limited exact enumeration

**Problem:** There is no way to say "find up to 500 regions and then stop." Users with large networks either wait forever or kill the process.

**Fix:** Add a `max_regions` parameter to `find_regions_exact`. When `n_found >= max_regions`, the DFS exits cleanly and returns the partial result. The returned `Partition` is a valid (if incomplete) partition — every region in it is geometrically correct, just not exhaustive.

```python
partition = compute_partition(model, x0, method="exact_julia", max_regions=200)
```

Combined with progress reporting (§4.2), this gives users control over the time/completeness tradeoff.

---

## Tier 5 — Training framework integration (one week, high research value)

### 5.1 PyTorch callback

**Problem:** The primary research use case is tracking partition complexity during training. Currently users must manually checkpoint the model, then loop over checkpoints with `iter_state_dicts`. This is fragile and error-prone.

**Fix:** A `PartitionTracker` callback for PyTorch that hooks into the training loop:

```python
# New file: src/parx/callbacks/pytorch.py
from parx import compute_partition

class PartitionTracker:
    def __init__(self, X_probe, every_n_epochs=1, method="sparse_julia",
                 save_dir=None, metrics=None):
        self.X_probe = X_probe
        self.every_n = every_n_epochs
        self.method = method
        self.save_dir = save_dir
        self.metrics = metrics or ["n_regions", "dead_neuron_count"]
        self.history = []

    def on_epoch_end(self, epoch, model):
        if epoch % self.every_n != 0:
            return
        partition = compute_partition(model, self.X_probe, method=self.method)
        record = {"epoch": epoch, "partition": partition}
        self.history.append(record)
        if self.save_dir:
            parx.save_partition(partition, f"{self.save_dir}/epoch_{epoch}.npz")
```

**Usage:**

```python
tracker = parx.PartitionTracker(X_probe=X, every_n_epochs=5)

for epoch in range(100):
    train_one_epoch(model, optimizer, data)
    tracker.on_epoch_end(epoch, model)

fig = animate_epochs([r["partition"] for r in tracker.history])
fig.show()
```

**Stretch:** Implement the same interface as `pytorch_lightning.Callback` and `transformers.TrainerCallback` so it slots into standard training loops with zero boilerplate.

---

## Tier 6 — Scaling exact enumeration (weeks, algorithmic)

### 6.1 Parallel DFS

**Problem:** The DFS in `exact.jl` is single-threaded. Julia threads are available (defaulting to all cores) but the DFS stack is sequential.

**Approach:** Parallelize at the facet-exploration level. Each region has `n_constraints` candidate facets to check. These checks are independent LP solves that can run concurrently:

```julia
Threads.@threads for facet_idx in pending_facets
    feasible, new_region = _check_facet(region, facet_idx, weights, biases)
    if feasible && !visited(new_region)
        push!(found, new_region)
    end
end
```

The main complication is the shared `visited` set — use a thread-safe set (`Base.Threads.SpinLock` protecting a regular `Set`). Expected speedup: linear in thread count for wide search frontiers (many facets per region).

---

### 6.2 LP warm-starting

**Problem:** Adjacent regions share all but one constraint. The LP at each facet is nearly identical to the LP of the current region — but HiGHS is cold-started each time.

**Fix:** Pass the current region's LP solution as a warm start when checking each adjacent facet. HiGHS supports warm-starting via basis information (`set_basis`). This can reduce LP solve time by 5–10× for well-conditioned problems.

---

### 6.3 Incremental enumeration across training epochs

**Long-term idea:** If the network changes only slightly between epochs, most of the partition is unchanged. An incremental DFS could start from the previous partition, check which regions are still valid (halfspace systems still feasible), and only re-explore regions that changed. This would make per-epoch exact enumeration practical for small networks that are close to convergence.

---

## Tier 7 — Visualization (days, usability)

### 7.1 Fix `test_plot_partition_2d_layer_coarsens_regions`

The test uses a 1-layer network where `layer=1` and `layer=n_layers` are the same — so the `<=` assertion can never fail. Use a 2-layer fixture to actually exercise the coarsening path.

### 7.2 3D visualization

`plot_partition_slice` and `plot_partition_projection` produce 2D figures from higher-dimensional partitions. A true 3D visualization (interactive Plotly 3D scatter/mesh) for `input_dim=3` would be more informative and is achievable with Plotly's `go.Mesh3d` if vertex enumeration (see §7.3) is available.

### 7.3 Vertex enumeration

The current `_region_vertices_2d` function enumerates vertices of a 2D polytope by intersecting pairs of halfspace boundaries. For `input_dim=3` this generalizes to intersecting triples of planes. Beyond 3D, vertex enumeration requires a dedicated algorithm (`pycddlib` or `scipy.spatial.HalfspaceIntersection`). Having exact vertices would also enable:

- Exact volume computation (instead of Monte Carlo)
- Facet enumeration for visualization
- Export to standard polytope formats

---

## Tier 8 — Long-term / research directions

### 8.1 Non-sequential architectures

Residual connections (`x + f(x)`) change the polyhedral structure: the activation pattern no longer uniquely determines the linear map because the skip path adds a non-gated contribution. Handling residual connections requires tracking which paths through the computational graph are gated and which aren't. This is a significant algorithmic change to both the bridge and the Julia DFS.

### 8.2 Lipschitz constants per region

Each linear region has a well-defined local Jacobian `A` (from `local_affine`). The local Lipschitz constant is `||A||_2` (largest singular value). Adding `partition.local_lipschitz(region)` and aggregating across regions gives a Lipschitz profile of the network — useful for robustness analysis.

### 8.3 Formal verification export

The halfspace representation `D @ x <= g` is exactly the format used by neural network verification tools (Reluplex, ERAN, α-β-CROWN). Exporting a `Partition` to VNNLIB or SMT-LIB format would connect parx to the formal verification ecosystem.

### 8.4 PyPI distribution

The package is not yet on PyPI. Blocking issues: Julia startup overhead makes install testing awkward; the sysimage (§4.1) should be resolved first. Once those are addressed, a `pip install parx` workflow is straightforward.

---

## Quick reference

| # | Item | Effort | Blocks |
|---|---|---|---|
| 1.1 | Fix DFS fixture in tests | 1h | Faster CI |
| 1.2 | Replace vacuous volume test | 1h | Test correctness |
| 1.3 | Tests for diagnostics/precompile | 2h | Coverage |
| 1.4 | `.pth` loading test | 1h | Coverage |
| 1.5 | juliacall import order warning | 1h | Clean CI output |
| 2.1 | Sparse centroid → Chebyshev center | 1–2d | Analysis correctness |
| 2.2 | Better unsupported-arch errors | 1d | UX |
| 3.1 | BatchNorm folding | 2–3d | Real network support |
| 3.2 | Dropout skip | 1h | Real network support |
| 3.3 | LeakyReLU support | 3–4d | Architecture coverage |
| 4.1 | Julia precompiled sysimage | 2–3d | Startup time |
| 4.2 | DFS progress reporting | 1–2d | Interactive use |
| 4.3 | Budget-limited exact (`max_regions`) | 1d | Scaling |
| 5.1 | PyTorch training callback | 3–5d | Research workflow |
| 6.1 | Parallel DFS | 1w | Scaling |
| 6.2 | LP warm-starting | 3–4d | Scaling |
| 6.3 | Incremental epoch enumeration | 2–3w | Research |
| 7.1 | Fix coarsening viz test | 1h | Test correctness |
| 7.2–7.3 | 3D viz + vertex enumeration | 1–2w | Visualization |
| 8.x | Long-term research directions | open | Future |
