import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import torch
    import torch.nn as nn
    import parx
    import parx.viz as viz
    import time
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, mo, nn, np, parx, time, torch, viz


@app.cell
def __(mo):
    mo.md(r"""
    # Linear Region Enumeration in ReLU Networks: parx

    **A Python+Julia hybrid library for exactly enumerating polyhedral activation regions**

    This notebook provides a mathematical overview and demonstration of `parx`, a research tool that
    computes the complete linear region decomposition of ReLU neural networks. The library combines a
    Python user interface with high-performance Julia algorithms for combinatorial enumeration.

    ---
    """)


@app.cell
def __(mo):
    mo.md(r"""
    ## Mathematical Background

    ### Linear Regions and Polyhedral Decomposition

    A **ReLU neural network** with $L$ layers defines a piecewise-linear function $f: \mathbb{R}^d \to \mathbb{R}^k$.
    The input space $\mathbb{R}^d$ is partitioned into **linear regions** where $f$ is affine.

    For a network with weight matrices $W^{(\ell)} \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$ and bias
    vectors $b^{(\ell)} \in \mathbb{R}^{n_\ell}$, the forward pass computes:

    $$z^{(\ell)} = W^{(\ell)} a^{(\ell-1)} + b^{(\ell)}$$
    $$a^{(\ell)} = \max(0, z^{(\ell)})$$

    where $a^{(0)} = x$ is the input and $\max(\cdot, \cdot)$ is applied element-wise.

    ### Activation Patterns and Region Definition

    An **activation pattern** $\sigma \in \{0,1\}^N$ (where $N = \sum_{\ell=1}^{L-1} n_\ell$) specifies
    which neurons are active:
    $$\sigma_{\ell,j} = \begin{cases} 1 & \text{if } z^{(\ell)}_j > 0 \\ 0 & \text{if } z^{(\ell)}_j \leq 0 \end{cases}$$

    Each activation pattern $\sigma$ defines a **linear region** $R_\sigma$:
    $$R_\sigma = \{x \in \mathbb{R}^d : \text{activation pattern of } f(x) \text{ is } \sigma\}$$

    ### Halfspace Representation

    Each region $R_\sigma$ is a convex polytope defined by linear inequalities. Given an activation
    pattern, we can derive the **halfspace representation**:
    $$R_\sigma = \{x \in \mathbb{R}^d : D_\sigma x \leq g_\sigma\}$$

    where $D_\sigma \in \mathbb{R}^{m \times d}$ and $g_\sigma \in \mathbb{R}^m$ encode the activation
    constraints. Each row corresponds to one neuron's activation threshold, propagated back through
    the preceding affine layers.

    ### Local Affine Maps

    Within each region $R_\sigma$, the network function is affine:
    $$f(x) = A_\sigma x + b_\sigma \quad \forall x \in R_\sigma$$

    The matrices $A_\sigma$ and $b_\sigma$ are computed by composing only the active linear layers,
    masking out dead neurons.
    """)


@app.cell
def __(mo):
    mo.md(r"""
    ## Architecture Overview

    The `parx` library implements a **Python + Julia hybrid architecture**:

    - **Python layer**: User-facing API, network loading, visualization
    - **Julia core**: Combinatorially intensive enumeration algorithms
    - **Bridge**: `juliacall` for seamless interoperability
    """)


@app.cell
def __(mo):
    mo.mermaid(r"""
    graph TD
        A[PyTorch Model] --> B[parx.load_network]
        B --> C[Weight/Bias Extraction]
        C --> D[Julia Bridge]
        D --> E[Sparse Enumeration]
        D --> F[Exact Enumeration]
        E --> G[Region Collection]
        F --> G
        G --> H[Partition Object]
        H --> I[Query Operations]
        H --> J[Visualization]
        H --> K[Mathematical Analysis]

        classDef python fill:#3776ab,stroke:#fff,color:#fff
        classDef julia fill:#389826,stroke:#fff,color:#fff
        classDef result fill:#ff6b6b,stroke:#fff,color:#fff

        class A,B,C,H,I,J,K python
        class D,E,F julia
        class G result
    """)


@app.cell
def __(mo):
    mo.md(r"""
    ## Core API Demonstration

    ### Network Creation and Loading

    We demonstrate with synthetic fully-connected ReLU networks. In practice, `parx.load_network()`
    can load from:
    - PyTorch `nn.Module` objects
    - State dictionaries (`.pth` files)
    - HDF5 format (`.h5` files, requires `pip install parx[h5]`)
    """)


@app.cell
def __(nn, np, torch):
    def make_net(layer_sizes, seed=0):
        torch.manual_seed(seed)
        layers = []
        for _i in range(len(layer_sizes) - 2):
            layers += [nn.Linear(layer_sizes[_i], layer_sizes[_i + 1]), nn.ReLU()]
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        return nn.Sequential(*layers)

    np.random.seed(42)
    net_2d = make_net([2, 8, 1], seed=7)

    print(f"Sample network: {net_2d}")
    print(f"Parameters: {sum(p.numel() for p in net_2d.parameters())}")

    net_2d


@app.cell
def __(mo):
    mo.md(r"""
    ### Partition Computation

    The core operation is `parx.compute_partition()`, which returns a `Partition` object containing
    all enumerated regions:

    ```python
    partition = parx.compute_partition(network, data, method="exact_julia")
    ```

    **Methods available:**
    - `"sparse_julia"`: Forward-pass sampling (fast, incomplete coverage)
    - `"exact_julia"`: DFS + facet-flipping (complete, exponential time)
    """)


@app.cell
def __(make_net, np, parx):
    _data = np.random.default_rng(1).uniform(-2, 2, (300, 2))
    _net = make_net([2, 8, 1], seed=7)
    partition = parx.compute_partition(_net, _data, method="exact_julia")

    print(f"Network: [2 → 8 → 1] ReLU network")
    print(f"Regions enumerated: {len(partition)}")
    print(f"Sample region: {partition.regions[0]}")

    partition


@app.cell
def __(mo):
    mo.md(r"""
    ### Region Data Structure

    Each `Region` object contains:
    - **`activation_path`**: Binary pattern $\sigma \in \{0,1\}^N$
    - **`centroid`**: Chebyshev center (point equidistant from all boundaries)
    - **`bounded`**: Whether the region is bounded (polytope vs. polyhedron)
    - **`active_indices`**: Indices of active neurons per layer

    The activation path uniquely identifies the region and enables reconstruction of its geometry.
    """)


@app.cell
def __(mo, partition):
    _region = partition.regions[0]

    mo.md(f"""
    **Example Region Analysis:**

    - Activation pattern: `{_region.activation_path[:16]}...` (first 16 bits)
    - Centroid: $({_region.centroid[0]:.3f}, {_region.centroid[1]:.3f})$
    - Bounded: {_region.bounded}
    - Layers with active neurons: {_region.n_layers}
    - Total pattern length: {len(_region.activation_path)} bits
    """)


@app.cell
def __(mo):
    mo.md(r"""
    ## Region Scaling Analysis

    ### Exponential Growth with Network Width

    The number of linear regions grows **exponentially** with network width. The fundamental result
    from Montúfar et al. (2014) shows that a ReLU network with $L$ layers and width $W$ can have up
    to $O(W^L)$ regions.

    We demonstrate this scaling behavior across different architectures:
    """)


@app.cell
def __(make_net, np, parx, time):
    _data_seed = np.random.default_rng(1).uniform(-2, 2, (10, 2))

    _architectures = [
        ([2, 4, 1],        "2→4→1",        4),
        ([2, 8, 1],        "2→8→1",        8),
        ([2, 16, 1],       "2→16→1",      16),
        ([2, 4, 4, 1],     "2→4→4→1",      8),
        ([2, 8, 4, 1],     "2→8→4→1",     12),
        ([2, 16, 8, 1],    "2→16→8→1",    24),
        ([2, 8, 8, 4, 1],  "2→8→8→4→1",   20),
    ]

    scaling_results = []
    for _sizes, _label, _n_hidden in _architectures:
        _net = make_net(_sizes, seed=3)
        _t0 = time.perf_counter()
        _p = parx.compute_partition(_net, _data_seed, method="exact_julia")
        _elapsed = time.perf_counter() - _t0
        scaling_results.append({
            "Architecture": _label,
            "Hidden neurons": _n_hidden,
            "Layers": len(_sizes) - 2,
            "Regions": len(_p),
            "Time (s)": round(_elapsed, 3),
        })

    import pandas as _pd
    _df = _pd.DataFrame(scaling_results)
    print("Region count scaling with architecture:")
    print(_df.to_string(index=False))

    scaling_results


@app.cell
def __(go, make_subplots, scaling_results):
    _rows = scaling_results
    _fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Regions vs Hidden Neurons", "Architecture Comparison"),
    )

    _fig.add_trace(
        go.Scatter(
            x=[r["Hidden neurons"] for r in _rows],
            y=[r["Regions"] for r in _rows],
            mode="markers+lines",
            marker=dict(
                size=12,
                color=[r["Layers"] for r in _rows],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Layers", x=0.45)
            ),
            text=[r["Architecture"] for r in _rows],
            hovertemplate="%{text}<br>Neurons: %{x}<br>Regions: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )

    _fig.add_trace(
        go.Bar(
            x=[r["Architecture"] for r in _rows],
            y=[r["Regions"] for r in _rows],
            marker_color="steelblue",
            hovertemplate="%{x}<br>%{y} regions<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )

    _fig.update_layout(height=400, title="Exponential Region Growth")
    _fig.update_xaxes(title_text="Total Hidden Neurons", row=1, col=1)
    _fig.update_yaxes(title_text="Number of Regions", row=1, col=1)
    _fig.update_xaxes(tickangle=30, row=1, col=2)
    _fig.update_yaxes(title_text="Number of Regions", row=1, col=2)

    _fig


@app.cell
def __(mo):
    mo.md(r"""
    ## 2D Visualization of Partitions

    ### Polyhedral Decomposition Visualization

    For 2D input networks, we can directly visualize the polyhedral decomposition. Each colored
    region corresponds to a unique activation pattern $\sigma$.

    The boundaries between regions are **hyperplanes** defined by neuron activation thresholds:
    $w_i^\top x + b_i = 0$.
    """)


@app.cell
def __(make_net, np, parx, viz):
    _data_vis = np.random.default_rng(1).uniform(-2, 2, (300, 2))

    net_small  = make_net([2, 4, 1],    seed=7)
    net_medium = make_net([2, 8, 1],    seed=7)
    net_wide   = make_net([2, 16, 1],   seed=7)

    part_small  = parx.compute_partition(net_small,  _data_vis, method="exact_julia")
    part_medium = parx.compute_partition(net_medium, _data_vis, method="exact_julia")
    part_wide   = parx.compute_partition(net_wide,   _data_vis, method="exact_julia")

    print(f"Small network [2→4→1]:   {len(part_small)} regions")
    print(f"Medium network [2→8→1]:  {len(part_medium)} regions")
    print(f"Wide network [2→16→1]:   {len(part_wide)} regions")

    part_small, part_medium, part_wide


@app.cell
def __(mo, part_medium, part_small, part_wide, viz):
    _domain = ((-2.0, 2.0), (-2.0, 2.0))

    _fig_small  = viz.plot_partition_2d(part_small,  domain=_domain)
    _fig_medium = viz.plot_partition_2d(part_medium, domain=_domain)
    _fig_wide   = viz.plot_partition_2d(part_wide,   domain=_domain)

    _fig_small.update_layout(
        title=f"[2 → 4 → 1]  ·  {len(part_small)} regions",
        width=340, height=340, margin=dict(t=50, b=20, l=20, r=20),
        showlegend=False,
    )
    _fig_medium.update_layout(
        title=f"[2 → 8 → 1]  ·  {len(part_medium)} regions",
        width=340, height=340, margin=dict(t=50, b=20, l=20, r=20),
        showlegend=False,
    )
    _fig_wide.update_layout(
        title=f"[2 → 16 → 1]  ·  {len(part_wide)} regions",
        width=340, height=340, margin=dict(t=50, b=20, l=20, r=20),
        showlegend=False,
    )

    mo.hstack([_fig_small, _fig_medium, _fig_wide])


@app.cell
def __(mo):
    mo.md(r"""
    ## Query and Analysis Operations

    ### Point-to-Region Mapping

    Given input points, we can efficiently determine which region they belong to using
    **vectorized routing**:

    ```python
    regions = partition.route(points)  # O(1) hash lookup per point
    ```

    This is much faster than naive forward-pass computation for each query.
    """)


@app.cell
def __(np, part_medium):
    _test_points = np.array([
        [ 1.2,  0.8],
        [-0.5,  1.5],
        [ 0.0, -1.0],
        [-1.8, -0.7],
    ])

    _regions = part_medium.route(_test_points)

    _query_results = []
    for _i, (_point, _region) in enumerate(zip(_test_points, _regions)):
        _query_results.append({
            "Point": f"({_point[0]:+.1f}, {_point[1]:+.1f})",
            "Region found": _region is not None,
            "Bounded": _region.bounded if _region else "—",
            "Active layers": _region.n_layers if _region else "—",
            "Centroid": f"({_region.centroid[0]:+.3f}, {_region.centroid[1]:+.3f})" if _region else "—",
        })

    import pandas as _pd2
    _df2 = _pd2.DataFrame(_query_results)
    print("Point-to-region query results:")
    print(_df2.to_string(index=False))

    _test_points, _regions


@app.cell
def __(mo):
    mo.md(r"""
    ### Halfspace Representation

    For any region, we can compute its **halfspace representation** $D x \leq g$:

    $$R_\sigma = \{x \in \mathbb{R}^d : D_\sigma x \leq g_\sigma\}$$

    This is derived by tracing the activation constraints through all layers. Each row of $D_\sigma$
    corresponds to one neuron's activation threshold.
    """)


@app.cell
def __(mo, part_medium):
    _region_hs = part_medium.regions[0]
    D, g = part_medium.halfspaces(_region_hs)

    mo.md(f"""
    **Halfspace representation for Region 0:**

    - Constraint matrix $D$: shape {D.shape}
    - Right-hand side $g$: shape {g.shape}
    - Number of constraints: {len(g)}
    - Region polytope: $\\{{x \\in \\mathbb{{R}}^2 : D x \\leq g\\}}$

    **Sample constraints:**
    """)


@app.cell
def __(D, g):
    print("First 5 halfspace constraints (D[i] · x ≤ g[i]):")
    for _i in range(min(5, len(g))):
        print(f"  [{D[_i,0]:+.3f}, {D[_i,1]:+.3f}] · x ≤ {g[_i]:+.3f}")

    if len(g) > 5:
        print(f"  ... ({len(g)-5} more constraints)")

    D, g


@app.cell
def __(mo):
    mo.md(r"""
    ### Local Affine Transformation

    Within each region $R_\sigma$, the network computes an **affine transformation**:

    $$f(x) = A_\sigma x + b_\sigma$$

    where $A_\sigma$ and $b_\sigma$ are derived by composing only the active linear layers.
    """)


@app.cell
def __(mo, part_medium):
    _region_aff = part_medium.regions[0]
    _A_local, _b_local = part_medium.local_affine(_region_aff)

    mo.md(f"""
    **Local affine map for Region 0:**

    - Linear part $A$: shape {_A_local.shape}
    - Bias part $b$: shape {_b_local.shape}
    - Transformation: $f(x) = A x + b$

    $$A = \\begin{{bmatrix}} {_A_local[0,0]:.3f} & {_A_local[0,1]:.3f} \\end{{bmatrix}}, \\quad b = {_b_local[0]:.3f}$$
    """)


@app.cell
def __(mo):
    mo.md(r"""
    ### Region Boundary Visualization

    We can visualize the halfspace constraints that define a specific region's boundaries:
    """)


@app.cell
def __(part_medium, viz):
    _region_viz = part_medium.regions[0]
    _fig_hs = viz.plot_halfspaces(part_medium, _region_viz, x_range=(-2, 2), y_range=(-2, 2))
    _fig_hs.update_layout(
        width=500, height=450,
        title="Halfspace Boundaries for Region 0 ([2→8→1] network)"
    )
    _fig_hs


@app.cell
def __(mo):
    mo.md(r"""
    ## Enumeration Algorithms

    ### Algorithm 1: Sparse Enumeration (Forward-Pass Sampling)

    **Mathematical approach**: Sample points uniformly in the input domain and compute their
    activation patterns through forward passes.

    $$\text{Sparse}(f, X) = \{\sigma : \exists x \in X \text{ s.t. activation pattern of } f(x) = \sigma\}$$

    **Complexity**: $O(|X| \cdot L \cdot W)$ where $|X|$ is sample size

    **Coverage**: Incomplete but fast — finds only regions that contain sample points
    """)


@app.cell
def __(mo):
    mo.md(r"""
    ### Algorithm 2: Exact Enumeration (DFS + Facet-Flipping)

    **Mathematical approach**: Depth-first search through the arrangement of hyperplanes defined by
    neuron thresholds.

    Starting from a seed region, the algorithm:
    1. **Identifies facets**: Boundaries where exactly one neuron changes activation state
    2. **Flips through facets**: Traverse to adjacent regions by crossing hyperplanes
    3. **Recurses**: Continue until all reachable regions are enumerated

    Formally, two regions $R_\sigma$ and $R_{\sigma'}$ are **adjacent** if their activation patterns
    differ in exactly one bit: $\|\sigma - \sigma'\|_1 = 1$.

    **Complexity**: $O(R \cdot L \cdot W)$ where $R$ is the total number of regions

    **Coverage**: Complete — finds all regions reachable from the seed point
    """)


@app.cell
def __(make_net, mo, np, parx, time, viz):
    _net_cmp = make_net([2, 8, 1], seed=42)
    _data_sparse = np.random.default_rng(7).uniform(-2, 2, (50, 2))
    _data_seed = _data_sparse[:1]

    _t0_s = time.perf_counter()
    _part_sparse = parx.compute_partition(_net_cmp, _data_sparse, method="sparse_julia")
    _t_sparse = time.perf_counter() - _t0_s

    _t0_e = time.perf_counter()
    _part_exact = parx.compute_partition(_net_cmp, _data_seed, method="exact_julia")
    _t_exact = time.perf_counter() - _t0_e

    _coverage = 100 * len(_part_sparse) / len(_part_exact) if len(_part_exact) else 0

    _domain_cmp = ((-2.0, 2.0), (-2.0, 2.0))
    _fig_sparse = viz.plot_partition_2d(_part_sparse, domain=_domain_cmp)
    _fig_exact  = viz.plot_partition_2d(_part_exact,  domain=_domain_cmp)

    _fig_sparse.update_layout(
        title=f"Sparse: {len(_part_sparse)} regions ({_coverage:.1f}% coverage) · {_t_sparse*1000:.0f} ms",
        width=400, height=400, margin=dict(t=60), showlegend=False,
    )
    _fig_exact.update_layout(
        title=f"Exact: {len(_part_exact)} regions (complete) · {_t_exact*1000:.0f} ms",
        width=400, height=400, margin=dict(t=60), showlegend=False,
    )

    mo.hstack([_fig_sparse, _fig_exact])


@app.cell
def __(mo):
    mo.md(r"""
    ### Layer-wise Region Analysis

    The network creates distinct activation patterns at each layer depth. We can analyze how
    regions propagate through the network:
    """)


@app.cell
def __(part_wide, viz):
    _fig_rc = viz.plot_region_counts(part_wide)
    _fig_rc.update_layout(
        title="Region Evolution Through Network Layers [2→16→1]",
        width=550, height=380,
    )
    _fig_rc


@app.cell
def __(mo):
    mo.md(r"""
    ### Architectural Complexity Comparison

    Different network architectures create dramatically different partition structures. Deeper
    networks tend to create more complex, irregular boundaries:
    """)


@app.cell
def __(make_net, mo, np, parx, viz):
    _data_arch = np.random.default_rng(1).uniform(-2, 2, (5, 2))
    _domain_arch = ((-2.0, 2.0), (-2.0, 2.0))

    _configs = [
        ([2, 4, 1],       "2→4→1"),
        ([2, 8, 4, 1],    "2→8→4→1"),
        ([2, 16, 8, 1],   "2→16→8→1"),
        ([2, 8, 8, 4, 1], "2→8→8→4→1"),
    ]

    _figs = []
    for _sizes, _lbl in _configs:
        _p = parx.compute_partition(make_net(_sizes, seed=5), _data_arch, method="exact_julia")
        _f = viz.plot_partition_2d(_p, domain=_domain_arch)
        _f.update_layout(
            title=f"{_lbl} · {len(_p)} regions",
            width=320, height=320, margin=dict(t=50, b=10, l=10, r=10),
            showlegend=False,
        )
        _figs.append(_f)

    mo.hstack(_figs, justify="start")


@app.cell
def __(mo):
    mo.md(r"""
    ## Julia Integration Details

    ### Bridge Architecture

    The Python-Julia integration uses `juliacall` for seamless interoperability:

    ```python
    from parx._julia_init import ensure_julia
    jl = ensure_julia()  # Lazy singleton initialization

    # Call Julia functions directly
    regions = jl.LinearRegions.find_regions_exact(weights, biases, seed_point)
    ```

    ### Performance Characteristics

    - **Thread parallelization**: Julia algorithms use all available cores (`JULIA_NUM_THREADS`)
    - **Memory efficiency**: Sparse representation of activation patterns
    - **Numerical stability**: Careful handling of degenerate cases and floating-point precision

    ### Installation Requirements

    - Julia ≥ 1.10 must be on `PATH`
    - Optional: `pip install parx[h5]` for HDF5 network loading
    - Julia packages auto-installed via `juliapkg.json`
    """)


@app.cell
def __(mo):
    mo.md(r"""
    ## Current Status & Next Phases

    ### Phase 1: Complete ✅
    - [x] Sparse enumeration via forward-pass sampling
    - [x] Exact enumeration via DFS + facet-flipping
    - [x] Python-Julia bridge with `juliacall`
    - [x] Halfspace geometry reconstruction
    - [x] 2D visualization tools

    ### Phase 2: In Development 🚧
    - [ ] **JuMP optimization integration** for exact LP-based construction
    - [ ] **HiGHS solver** for fast constraint satisfaction
    - [ ] **Polyhedral volume computation** using triangulation methods
    - [ ] **Memory optimization** for large-scale networks

    ### Phase 3: Research Extensions 🔬
    - [ ] **Adversarial region analysis**: Find regions most vulnerable to attacks
    - [ ] **Decision boundary characterization**: Analyze margin distributions
    - [ ] **Approximation bounds**: Theoretical guarantees for sparse sampling

    ### Known Limitations

    - **Exponential scaling**: Exact enumeration becomes prohibitive for large networks
    - **2D visualization only**: Higher-dimensional networks need projection methods
    - **Memory usage**: Large partitions require significant RAM for storage
    - **Numerical precision**: Degenerate cases near hyperplane intersections
    """)


@app.cell
def __(mo):
    mo.md(r"""
    ## Getting Started

    ### Installation

    ```bash
    uv venv && source .venv/bin/activate
    uv pip install -e ".[dev,h5]"
    julia --project=src/parx/julia -e "using Pkg; Pkg.instantiate()"
    ```

    ### Basic Usage

    ```python
    import parx
    import torch.nn as nn
    import numpy as np

    network = nn.Sequential(
        nn.Linear(2, 8), nn.ReLU(),
        nn.Linear(8, 1)
    )

    data = np.random.uniform(-2, 2, (100, 2))
    partition = parx.compute_partition(network, data, method="exact_julia")
    print(f"Found {len(partition)} linear regions")

    test_points = np.array([[0.5, -1.0], [1.2, 0.8]])
    regions = partition.route(test_points)
    ```

    ### Testing & Linting

    ```bash
    pytest                              # all tests
    pytest tests/test_julia_bridge.py  # single component
    ruff format src/ tests/
    ruff check src/ tests/
    ```
    """)


if __name__ == "__main__":
    app.run()
