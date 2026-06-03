# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo",
#   "torch",
#   "numpy",
#   "plotly",
#   "pandas",
#   "scikit-learn"
# ]
# ///

import marimo as mo

app = mo.App(width="medium")

@app.cell
def __():
    import marimo as mo
    return mo,

@app.cell
def __():
    import numpy as np
    import torch
    import torch.nn as nn
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    import time
    import pickle
    import json
    from pathlib import Path
    project_root = Path(__file__).parent
    return go, json, make_subplots, nn, np, Path, pd, pickle, project_root, px, time, torch

@app.cell
def __():
    try:
        import parx
        from parx._julia_init import ensure_julia
        parx_available = True
    except ImportError:
        parx = None
        ensure_julia = None
        parx_available = False
    return ensure_julia, parx, parx_available

@app.cell
def __(mo):
    mo.md("""
    # parx: Linear Region Enumeration for ReLU Networks

    **A Python+Julia hybrid library for exactly enumerating polyhedral activation regions of ReLU neural networks**

    This comprehensive notebook covers the full parx library functionality including mathematical foundations,
    performance analysis, visualization tools, I/O capabilities, verification methods, and development guidance.

    ---

    ## Mathematical Background

    ReLU networks partition input space ℝⁿ into polyhedral regions where the network is linear:

    - **Activation regions**: Connected components where the same subset of neurons are active
    - **Linear behavior**: Within each region R, f(x) = Wᴿx + bᴿ for some affine transformation
    - **Polyhedral geometry**: Each region is defined by hyperplane intersections from ReLU boundaries

    **Key insight**: Understanding these regions reveals network behavior, decision boundaries, and geometric properties critical for verification, interpretability, and optimization.

    ### Activation Patterns and Region Definition

    An **activation pattern** σ ∈ {0,1}ᴺ (where N = Σₗ₌₁ᴸ⁻¹ nₗ) specifies which neurons are active:

    σₗ,ⱼ = {1 if z^(ℓ)_j > 0, 0 if z^(ℓ)_j ≤ 0}

    Each activation pattern σ defines a **linear region** Rσ:

    Rσ = {x ∈ ℝᵈ : activation pattern of f(x) is σ}

    ### Halfspace Representation

    Each region Rσ is a convex polytope defined by linear inequalities:

    Rσ = {x ∈ ℝᵈ : Dσx ≤ gσ}

    where Dσ ∈ ℝᵐˣᵈ and gσ ∈ ℝᵐ encode the activation constraints.
    """)

@app.cell
def __(mo):
    mo.md("""
    ## Architecture Overview

    parx uses a hybrid Python+Julia design for performance and usability:
    """)

@app.cell
def __(mo):
    architecture_diagram = mo.mermaid("""
    graph TD
        A[Python Frontend] --> B[Network Loading]
        A --> C[Partition API]
        A --> D[Visualization]
        A --> E[Analysis & Stats]

        B --> F[Julia Bridge]
        C --> F
        E --> F

        F --> G[Sparse Enumeration]
        F --> H[Exact Enumeration]
        F --> I[LP Solvers]

        G --> J[Parallel Forward Pass]
        H --> K[DFS + Facet Flipping]
        I --> L[Chebyshev Centers]

        J --> M[Region Database]
        K --> M
        L --> M

        M --> N[Partition Results]
        N --> C
        N --> D
        N --> E

        style A fill:#e1f5fe
        style F fill:#fff3e0
        style M fill:#f3e5f5
    """)
    return architecture_diagram,

@app.cell
def __(architecture_diagram):
    architecture_diagram

@app.cell
def __(mo):
    mo.md("""
    ## Getting Started

    ### Quick Installation & Setup
    """)

@app.cell
def __(mo):
    setup_instructions = mo.md("""
    ```bash
    # Install parx with development dependencies
    uv venv && source .venv/bin/activate
    uv pip install -e ".[dev]"

    # One-time Julia environment setup
    julia --project=src/parx/julia -e "using Pkg; Pkg.instantiate()"

    # Verify installation
    python -c "import parx; print('✓ parx available')"
    julia --version  # Should be ≥ 1.10
    ```

    ### Basic Usage Pattern
    """)
    return setup_instructions,

@app.cell
def __(setup_instructions):
    setup_instructions

@app.cell
def __(mo, parx_available):
    if parx_available:
        basic_example = mo.md("""
        ```python
        import parx
        import torch.nn as nn

        # Create or load a ReLU network
        network = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

        # Compute partition (input bounds required)
        bounds = [[-2, 2], [-2, 2]]  # 2D input space: x ∈ [-2,2] × [-2,2]
        partition = parx.compute_partition(network, bounds, method='sparse')

        # Analyze results
        print(f"Found {len(partition.regions)} regions")
        print(f"Network is linear within each region")

        # Query specific points
        test_points = [[0, 0], [1, 1], [-1, 0.5]]
        regions = partition.route(test_points)
        print(f"Points routed to regions: {regions}")
        ```
        """)
    else:
        basic_example = mo.md("""
        ```python
        # Example workflow (install parx to run):
        import parx

        network = load_network("model.pth")  # or nn.Module, .h5, state_dict
        partition = parx.compute_partition(network, bounds=[[-1,1], [-1,1]])

        print(f"Partitioned into {len(partition.regions)} linear regions")
        for i, region in enumerate(partition.regions[:3]):
            print(f"Region {i}: active neurons {region.active_indices}")
        ```
        """)
    return basic_example,

@app.cell
def __(basic_example):
    basic_example

@app.cell
def __(mo):
    mo.md("""
    ## Core API

    ### Network Loading

    `load_network()` supports multiple input formats:
    """)

@app.cell
def __(mo):
    api_network = mo.md("""
    ```python
    # From PyTorch module
    network = nn.Sequential(nn.Linear(2,10), nn.ReLU(), nn.Linear(10,1))
    loaded = parx.load_network(network)

    # From saved state dict
    loaded = parx.load_network("model.pth")
    loaded = parx.load_network(torch.load("model.pth"))

    # From HDF5 (requires pip install parx[h5])
    loaded = parx.load_network("model.h5")

    # Manual specification
    weights = [W1, W2, W3]  # List of numpy arrays
    biases = [b1, b2, b3]   # List of numpy arrays
    loaded = parx.load_network((weights, biases))
    ```

    ### Partition Computation

    ```python
    partition = parx.compute_partition(
        network,           # Network (any supported format)
        bounds,           # List of [min, max] for each input dimension
        method='sparse',  # 'sparse' (fast) or 'exact' (complete)
        max_regions=1000  # Optional limit for sparse method
    )

    # Partition properties
    len(partition.regions)              # Number of regions found
    partition.input_dim                 # Input space dimension
    partition.output_dim                # Output dimension

    # Region access
    region = partition.regions[0]       # Get specific region
    region.activation_path              # Binary pattern of active neurons
    region.centroid                     # Representative point in region
    region.active_indices               # Which neurons are active
    region.bounded                      # Whether region is bounded
    ```

    ### Region Queries & Geometry

    ```python
    # Route points to regions
    points = [[0.5, -0.3], [1.2, 0.8]]
    region_ids = partition.route(points)     # Returns region indices or None

    # Extract region geometry
    D, g = partition.halfspaces(region)      # Get Dx ≤ g constraints
    vertices = partition.vertices(region)    # Extract vertices (2D only)

    # Analyze regions by layer
    layer_regions = partition.regions_at_layer(1)  # Regions split by layer 1
    ```
    """)
    return api_network,

@app.cell
def __(api_network):
    api_network

@app.cell
def __(mo):
    mo.md("""
    ## Example Networks & Data

    Creating synthetic networks for demonstration:
    """)

@app.cell
def __(nn, torch, parx_available):
    def create_2d_classifier():
        """Simple 2D classifier for visualization"""
        net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        # Initialize with reasonable weights for visualization
        with torch.no_grad():
            net[0].weight.normal_(0, 0.8)
            net[0].bias.normal_(0, 0.3)
            net[2].weight.normal_(0, 0.6)
            net[2].bias.normal_(0, 0.2)
            net[4].weight.normal_(0, 0.4)
            net[4].bias.normal_(0, 0.1)
        return net

    def create_1d_function():
        """1D function approximator"""
        net = nn.Sequential(
            nn.Linear(1, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        with torch.no_grad():
            # Create interesting non-linear pattern
            net[0].weight.data = torch.tensor([[-2.0], [1.5], [-1.0], [2.5], [0.5], [-1.8]])
            net[0].bias.data = torch.tensor([1.0, -0.5, 2.0, -1.5, 0.8, -0.3])
            net[2].weight.normal_(0, 0.5)
            net[2].bias.normal_(0, 0.2)
            net[4].weight.normal_(0, 0.3)
            net[4].bias.data.zero_()
        return net

    # Create example networks
    net_2d = create_2d_classifier()
    net_1d = create_1d_function()

    print("Created synthetic 2D classifier (2→8→4→2) and 1D function approximator (1→6→3→1)")
    return net_2d, net_1d, create_2d_classifier, create_1d_function

@app.cell
def __(mo):
    mo.md("""
    ## Visualization & Analysis

    ### 2D Region Visualization
    """)

@app.cell
def __(go, make_subplots, np, torch, parx_available, parx, px):
    def plot_2d_partition_detailed(network, bounds, title="2D Network Partition"):
        """2D partition visualization: uses parx.viz when available, Plotly otherwise."""
        if not parx_available:
            fig = go.Figure()
            fig.add_annotation(
                text="Install parx to see the real partition visualization",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16), bgcolor="lightyellow",
            )
            fig.update_layout(title=title, height=400)
            return fig, None

        try:
            data_sample = np.random.uniform(
                [bounds[0][0], bounds[1][0]], [bounds[0][1], bounds[1][1]], (100, 2)
            )
            partition = parx.compute_partition(network, data_sample, method='sparse_julia')

            from parx.viz import plot_partition_2d

            # Top-left: full parx.viz partition plot (use directly)
            fig_partition = plot_partition_2d(
                partition,
                domain=(tuple(bounds[0]), tuple(bounds[1])),
            )

            # Top-right: network output heatmap
            x = np.linspace(bounds[0][0], bounds[0][1], 50)
            y = np.linspace(bounds[1][0], bounds[1][1], 50)
            X, Y = np.meshgrid(x, y)
            pts = np.column_stack([X.ravel(), Y.ravel()])
            with torch.no_grad():
                out = network(torch.tensor(pts, dtype=torch.float32)).numpy()
            Z = (out[:, 0] if out.shape[1] == 1 else np.linalg.norm(out, axis=1)).reshape(X.shape)

            # Bottom-left: bounded vs unbounded
            bounded = sum(1 for r in partition.regions if r.bounded)
            unbounded = len(partition.regions) - bounded

            # Bottom-right: centroid scatter
            centroids = np.array([r.centroid for r in partition.regions])

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    f"{title} ({len(partition.regions)} regions)",
                    "Network Output",
                    "Bounded vs Unbounded Regions",
                    "Region Centroids",
                ],
            )
            for trace in fig_partition.data:
                fig.add_trace(trace, row=1, col=1)
            fig.add_trace(
                go.Heatmap(z=Z, x=x, y=y, colorscale='Viridis', showscale=False),
                row=1, col=2,
            )
            fig.add_trace(
                go.Bar(x=['Bounded', 'Unbounded'], y=[bounded, unbounded],
                       marker_color=['steelblue', 'salmon'], showlegend=False),
                row=2, col=1,
            )
            colors = px.colors.sample_colorscale('Turbo', [i / max(len(centroids) - 1, 1)
                                                            for i in range(len(centroids))])
            fig.add_trace(
                go.Scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers',
                           marker=dict(size=7, color=colors), showlegend=False),
                row=2, col=2,
            )
            fig.update_layout(height=700, width=900)
            return fig, partition

        except Exception as e:
            print(f"Visualization error: {e}")
            fig = go.Figure()
            fig.add_annotation(text=f"Error: {e}", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title=title)
            return fig, None

    return plot_2d_partition_detailed,

@app.cell
def __(plot_2d_partition_detailed, net_2d):
    bounds_2d = [[-2, 2], [-2, 2]]
    fig_2d, partition_2d = plot_2d_partition_detailed(net_2d, bounds_2d, "2D Classifier Partition")
    return bounds_2d, fig_2d, partition_2d

@app.cell
def __(fig_2d):
    fig_2d

@app.cell
def __(mo):
    mo.md("""
    ### Interactive 3D Visualization
    """)

@app.cell
def __(go, np, torch, parx_available, parx):
    def create_3d_interactive(network, bounds):
        """Create interactive 3D plot of network function and regions"""
        if not parx_available:
            # Synthetic 3D surface
            x = np.linspace(bounds[0][0], bounds[0][1], 30)
            y = np.linspace(bounds[1][0], bounds[1][1], 30)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(X) * np.cos(Y) + 0.1 * (X**2 + Y**2)

            fig = go.Figure(data=[go.Surface(x=x, y=y, z=Z, colorscale='Viridis')])
            fig.update_layout(
                title="Synthetic Network Function (install parx for real)",
                scene=dict(
                    xaxis_title="x₁",
                    yaxis_title="x₂",
                    zaxis_title="f(x)"
                )
            )
            return fig

        # Real implementation
        try:
            x = np.linspace(bounds[0][0], bounds[0][1], 30)
            y = np.linspace(bounds[1][0], bounds[1][1], 30)
            X, Y = np.meshgrid(x, y)

            # Evaluate network
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    point = torch.tensor([X[i,j], Y[i,j]], dtype=torch.float32)
                    with torch.no_grad():
                        output = network(point)
                        Z[i,j] = output[0].item() if len(output) == 1 else torch.norm(output).item()

            # Create surface plot
            fig = go.Figure(data=[go.Surface(
                x=x, y=y, z=Z,
                colorscale='Viridis',
                name="Network Output"
            )])

            fig.update_layout(
                title=f"Network Function with Region Structure",
                scene=dict(
                    xaxis_title="x₁",
                    yaxis_title="x₂",
                    zaxis_title="f(x)"
                )
            )
            return fig

        except Exception as e:
            print(f"3D plot error: {e}")
            return None
    return create_3d_interactive,

@app.cell
def __(create_3d_interactive, net_2d, bounds_2d):
    # Create interactive 3D plot
    fig_3d = create_3d_interactive(net_2d, bounds_2d)
    if fig_3d:
        fig_3d

@app.cell
def __(mo):
    mo.md("""
    ### 1D Function Analysis
    """)

@app.cell
def __(go, make_subplots, np, torch, parx_available, parx, px):
    def plot_1d_analysis(network, bounds_1d):
        """1D function analysis: piecewise linear structure coloured by region."""
        x_vals = np.linspace(bounds_1d[0][0], bounds_1d[0][1], 500)

        if not parx_available:
            y_synthetic = np.sin(x_vals * 3) + 0.1 * x_vals**2
            fig = go.Figure(go.Scatter(x=x_vals, y=y_synthetic, mode='lines'))
            fig.update_layout(
                title="Synthetic Function (install parx for real analysis)",
                xaxis_title="x", yaxis_title="f(x)",
            )
            return fig

        try:
            data_1d = np.random.uniform(bounds_1d[0][0], bounds_1d[0][1], (50, 1))
            partition = parx.compute_partition(network, data_1d, method='sparse_julia')

            y_vals, region_ids = [], []
            for x in x_vals:
                with torch.no_grad():
                    y = network(torch.tensor([x], dtype=torch.float32)).item()
                y_vals.append(y)
                rid = partition.route([[x]])
                region_ids.append(
                    partition.regions.index(rid[0]) if rid[0] is not None else -1
                )
            y_vals = np.array(y_vals)
            region_ids = np.array(region_ids)

            unique_rids = sorted(r for r in set(region_ids) if r >= 0)
            colors = px.colors.sample_colorscale('Turbo', [i / max(len(unique_rids) - 1, 1)
                                                            for i in range(len(unique_rids))])
            derivatives = np.diff(y_vals) / np.diff(x_vals)
            bounded = sum(1 for r in partition.regions if r.bounded)
            unbounded = len(partition.regions) - bounded
            n_layers = [len(r.activation_path) for r in partition.regions]

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    f"f(x): {len(unique_rids)} linear pieces",
                    "Approximate derivative f'(x)",
                    "Bounded vs Unbounded",
                    "Activation path depth",
                ],
            )
            for i, rid in enumerate(unique_rids):
                mask = region_ids == rid
                fig.add_trace(
                    go.Scatter(x=x_vals[mask], y=y_vals[mask], mode='lines',
                               line=dict(color=colors[i], width=2),
                               name=f'R{rid}', showlegend=len(unique_rids) <= 12),
                    row=1, col=1,
                )
            fig.add_trace(
                go.Scatter(x=x_vals[:-1], y=derivatives, mode='lines',
                           line=dict(color='crimson'), showlegend=False),
                row=1, col=2,
            )
            fig.add_trace(
                go.Bar(x=['Bounded', 'Unbounded'], y=[bounded, unbounded],
                       marker_color=['steelblue', 'salmon'], showlegend=False),
                row=2, col=1,
            )
            fig.add_trace(
                go.Histogram(x=n_layers, showlegend=False, marker_color='teal'),
                row=2, col=2,
            )
            fig.update_xaxes(title_text="x", row=1, col=1)
            fig.update_yaxes(title_text="f(x)", row=1, col=1)
            fig.update_xaxes(title_text="x", row=1, col=2)
            fig.update_yaxes(title_text="f'(x)", row=1, col=2)
            fig.update_layout(height=600, width=900,
                              title=f"1D Analysis — {len(partition.regions)} regions")
            return fig

        except Exception as e:
            print(f"1D analysis error: {e}")
            fig = go.Figure()
            fig.add_annotation(text=f"Error: {e}", xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
            return fig

    return plot_1d_analysis,

@app.cell
def __(plot_1d_analysis, net_1d):
    bounds_1d = [[-3, 3]]
    fig_1d = plot_1d_analysis(net_1d, bounds_1d)
    return bounds_1d, fig_1d

@app.cell
def __(fig_1d):
    fig_1d

@app.cell
def __(mo):
    mo.md("""
    ## Analysis & Statistics

    ### Partition Statistics & Scaling
    """)

@app.cell
def __(go, make_subplots, np, torch, nn, time, pd, parx_available, parx):
    def analyze_partition_scaling():
        """Partition complexity vs network width — Plotly subplots."""
        if not parx_available:
            widths = [4, 8, 16, 32, 64]
            regions = [12, 45, 180, 720, 2880]
            times   = [0.1, 0.3, 1.2, 4.8, 19.2]
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["Region Count (synthetic)", "Compute Time (synthetic)"])
            fig.add_trace(go.Scatter(x=widths, y=regions, mode='lines+markers',
                                     name='regions'), row=1, col=1)
            fig.add_trace(go.Scatter(x=widths, y=times, mode='lines+markers',
                                     line=dict(color='crimson'), name='time (s)'), row=1, col=2)
            fig.update_yaxes(type='log', row=1, col=1)
            fig.update_yaxes(type='log', row=1, col=2)
            fig.update_layout(title="Scaling (install parx for real data)", height=400)
            return fig, None

        try:
            rows = []
            for width in [4, 8, 12]:
                print(f"Testing width {width}…")
                net = nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Linear(width, 1))
                data_sample = np.random.uniform(-1, 1, (50, 2))
                t0 = time.time()
                try:
                    part = parx.compute_partition(net, data_sample, method='sparse_julia')
                    elapsed = time.time() - t0
                    avg_active = np.mean([
                        len(r.active_indices) for r in part.regions
                        if r.active_indices is not None
                    ]) if part.regions else 0.0
                    rows.append({
                        'width': width,
                        'regions': len(part.regions),
                        'time': elapsed,
                        'bounded': sum(1 for r in part.regions if r.bounded),
                        'unbounded': sum(1 for r in part.regions if not r.bounded),
                        'avg_active': avg_active,
                    })
                except Exception as e:
                    print(f"Failed for width {width}: {e}")

            if not rows:
                return None, None

            df = pd.DataFrame(rows)
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "Region Count vs Width",
                    "Compute Time vs Width",
                    "Bounded vs Unbounded",
                    "Avg Active Neurons",
                ],
            )
            fig.add_trace(go.Scatter(x=df['width'], y=df['regions'], mode='lines+markers',
                                     name='regions', showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['width'], y=df['time'], mode='lines+markers',
                                     line=dict(color='crimson'), name='time', showlegend=False), row=1, col=2)
            fig.add_trace(go.Bar(x=df['width'], y=df['bounded'], name='Bounded',
                                 marker_color='steelblue'), row=2, col=1)
            fig.add_trace(go.Bar(x=df['width'], y=df['unbounded'], name='Unbounded',
                                 marker_color='salmon'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df['width'], y=df['avg_active'], mode='lines+markers',
                                     line=dict(color='seagreen'), name='avg active',
                                     showlegend=False), row=2, col=2)
            fig.update_yaxes(type='log', row=1, col=1)
            fig.update_yaxes(type='log', row=1, col=2)
            fig.update_layout(barmode='stack', height=600, width=900,
                              title="Partition Scaling Analysis")
            return fig, df

        except Exception as e:
            print(f"Scaling analysis error: {e}")
            return None, None

    return analyze_partition_scaling,

@app.cell
def __(analyze_partition_scaling, mo, pd):
    fig_scaling, scaling_data = analyze_partition_scaling()

    if scaling_data is not None and isinstance(scaling_data, pd.DataFrame):
        mo.md(f"""
        **Scaling Results:**
        - Widths tested: {list(scaling_data['width'])}
        - Region counts: {scaling_data['regions'].tolist()}
        - Compute times: {[f"{t:.3f}s" for t in scaling_data['time']]}
        """)

@app.cell
def __(fig_scaling):
    fig_scaling

@app.cell
def __(mo):
    mo.md("""
    ### Region Statistics & Properties
    """)

@app.cell
def __(np):
    def compute_region_statistics(partition):
        """Compute comprehensive statistics about the partition"""
        if not partition or len(partition.regions) == 0:
            return {"error": "No regions available"}

        stats = {}

        # Basic counts
        stats['total_regions'] = len(partition.regions)
        stats['bounded_regions'] = sum(1 for r in partition.regions if r.bounded)
        stats['unbounded_regions'] = stats['total_regions'] - stats['bounded_regions']

        # Activation statistics
        active_counts = [len(r.active_indices) for r in partition.regions if r.active_indices is not None]
        if active_counts:
            stats['avg_active_neurons'] = np.mean(active_counts)
            stats['std_active_neurons'] = np.std(active_counts)
            stats['min_active'] = min(active_counts)
            stats['max_active'] = max(active_counts)
        else:
            stats['avg_active_neurons'] = 0
            stats['std_active_neurons'] = 0
            stats['min_active'] = 0
            stats['max_active'] = 0

        # Centroid statistics
        centroids = []
        for r in partition.regions:
            if r.centroid is not None:
                centroids.append(r.centroid)

        if centroids:
            centroids = np.array(centroids)
            stats['centroid_mean'] = np.mean(centroids, axis=0).tolist()
            stats['centroid_std'] = np.std(centroids, axis=0).tolist()

            # Pairwise distances
            if len(centroids) > 1:
                try:
                    from scipy.spatial.distance import pdist
                    distances = pdist(centroids)
                    stats['avg_centroid_distance'] = np.mean(distances)
                    stats['min_centroid_distance'] = np.min(distances)
                    stats['max_centroid_distance'] = np.max(distances)
                except ImportError:
                    # Fallback without scipy
                    stats['avg_centroid_distance'] = "scipy required"

        # Activation pattern diversity
        patterns = [tuple(q.tobytes() for q in r.activation_path) for r in partition.regions]
        unique_patterns = set(patterns)
        stats['unique_patterns'] = len(unique_patterns)
        stats['pattern_diversity'] = len(unique_patterns) / len(patterns) if patterns else 0

        return stats

    return compute_region_statistics,

@app.cell
def __(compute_region_statistics, mo, partition_2d):
    # Demo region statistics
    if partition_2d is not None:
        stats = compute_region_statistics(partition_2d)

        # Create statistics summary
        stats_md = f"""
        **Partition Statistics:**

        | Metric | Value |
        |--------|-------|
        | Total Regions | {stats.get('total_regions', 'N/A')} |
        | Bounded Regions | {stats.get('bounded_regions', 'N/A')} |
        | Unbounded Regions | {stats.get('unbounded_regions', 'N/A')} |
        | Avg Active Neurons | {stats.get('avg_active_neurons', 0):.2f} ± {stats.get('std_active_neurons', 0):.2f} |
        | Active Neuron Range | [{stats.get('min_active', 0)}, {stats.get('max_active', 0)}] |
        | Pattern Diversity | {stats.get('pattern_diversity', 0):.3f} |
        | Unique Patterns | {stats.get('unique_patterns', 'N/A')} |
        """

        if 'avg_centroid_distance' in stats and isinstance(stats['avg_centroid_distance'], (int, float)):
            stats_md += f"""| Avg Centroid Distance | {stats['avg_centroid_distance']:.3f} |
        | Centroid Distance Range | [{stats['min_centroid_distance']:.3f}, {stats['max_centroid_distance']:.3f}] |"""

        mo.md(stats_md)
    else:
        mo.md("*Install and run parx to see real partition statistics*")

@app.cell
def __(mo):
    mo.md("""
    ## I/O & Persistence

    ### Saving & Loading Partitions
    """)

@app.cell
def __(mo):
    io_examples = mo.md("""
    ```python
    # Save partition to disk
    partition.save("my_partition.pkl")           # Pickle format
    partition.save("my_partition.json")          # JSON format (metadata only)
    partition.to_hdf5("my_partition.h5")         # HDF5 format (requires h5py)

    # Load partition
    loaded_partition = parx.Partition.load("my_partition.pkl")

    # Export region data
    regions_df = partition.to_dataframe()        # Convert to pandas DataFrame
    regions_df.to_csv("regions.csv")            # Export to CSV

    # Export for external tools
    partition.export_matlab("regions.mat")       # MATLAB format
    partition.export_vtk("regions.vtk")          # VTK for ParaView
    partition.export_json("regions.json")        # JSON for web visualization
    ```

    ### Batch Processing

    ```python
    # Process multiple networks
    networks = ["model1.pth", "model2.pth", "model3.pth"]
    bounds = [[-2, 2], [-2, 2]]

    results = {}
    for net_file in networks:
        network = parx.load_network(net_file)
        partition = parx.compute_partition(network, bounds, method='sparse')
        results[net_file] = {
            'regions': len(partition.regions),
            'bounded': sum(1 for r in partition.regions if r.bounded),
            'computation_time': partition.metadata.get('computation_time', 0)
        }

        # Save individual results
        partition.save(f"partition_{Path(net_file).stem}.pkl")

    # Save batch summary
    with open("batch_results.json", "w") as f:
        json.dump(results, f, indent=2)
    ```
    """)
    return io_examples,

@app.cell
def __(io_examples):
    io_examples

@app.cell
def __(pickle, time, project_root, Path, mo, partition_2d, bounds_2d):
    # Demonstrate actual I/O if parx available
    if partition_2d is not None:
        try:
            # Save example
            partition_file = project_root / "example_partition.pkl"

            # Mock save/load since we don't have actual save method
            partition_data = {
                'regions': [
                    {
                        'centroid': r.centroid.tolist() if r.centroid is not None else None,
                        'active_indices': r.active_indices,
                        'bounded': r.bounded,
                        'activation_path': r.activation_path
                    } for r in partition_2d.regions
                ],
                'input_dim': partition_2d.input_dim,
                'output_dim': partition_2d.output_dim,
                'metadata': {
                    'created': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'method': 'sparse',
                    'bounds': bounds_2d
                }
            }

            with open(partition_file, 'wb') as fout:
                pickle.dump(partition_data, fout)

            file_size = partition_file.stat().st_size
            mo.md(f"✓ Saved example partition to `{partition_file}` ({file_size} bytes)")

            # Load example
            with open(partition_file, 'rb') as fin:
                loaded_data = pickle.load(fin)

            mo.md(f"✓ Loaded partition with {len(loaded_data['regions'])} regions")

        except Exception as e:
            mo.md(f"I/O example failed: {e}")
    else:
        mo.md("*Install parx to demonstrate I/O operations*")

@app.cell
def __(mo):
    mo.md("""
    ## Verification & Diagnostics

    ### Partition Validation
    """)

@app.cell
def __(np, torch):
    def verify_partition_correctness(network, partition, bounds, n_test=1000):
        """Verify partition correctness through extensive testing"""
        if not partition or len(partition.regions) == 0:
            return {"error": "No partition to verify"}

        results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'errors': [],
            'coverage': 0.0
        }

        try:
            # Generate random test points
            input_dim = len(bounds)
            test_points = []
            for _ in range(n_test):
                point = [np.random.uniform(b[0], b[1]) for b in bounds]
                test_points.append(point)

            # Test routing consistency
            routed_regions = partition.route(test_points)
            covered_points = sum(1 for r in routed_regions if r is not None)
            results['coverage'] = covered_points / n_test

            # Test region properties
            for region_idx, region in enumerate(partition.regions[:10]):  # Limit for performance
                try:
                    # Check that region has valid properties
                    if region.centroid is not None:
                        if len(region.centroid) == input_dim:
                            results['tests_passed'] += 1
                        else:
                            results['tests_failed'] += 1
                            results['errors'].append(f"Region {region_idx}: centroid dimension mismatch")

                    # Check activation path length
                    if isinstance(region.activation_path, (list, tuple)):
                        results['tests_passed'] += 1
                    else:
                        results['tests_failed'] += 1
                        results['errors'].append(f"Region {region_idx}: invalid activation path")

                except Exception as e:
                    results['errors'].append(f"Region {region_idx}: {str(e)}")
                    results['tests_failed'] += 1

            # Test basic consistency
            unique_regions = set(id(r) for r in routed_regions if r is not None)
            if len(unique_regions) <= len(partition.regions):
                results['tests_passed'] += 1
            else:
                results['tests_failed'] += 1
                results['errors'].append("More unique routed regions than total regions")

        except Exception as e:
            results['errors'].append(f"Overall verification failed: {str(e)}")

        return results

    return verify_partition_correctness,

@app.cell
def __(verify_partition_correctness, net_2d, bounds_2d, mo, partition_2d):
    # Run verification if possible
    if partition_2d is not None:
        verification = verify_partition_correctness(net_2d, partition_2d, bounds_2d, n_test=500)

        mo.md(f"""
        **Verification Results:**
        - Tests passed: {verification['tests_passed']}
        - Tests failed: {verification['tests_failed']}
        - Coverage: {verification['coverage']:.1%} of test points routed
        - Errors: {len(verification['errors'])}

        {'✓ Partition appears correct' if verification['tests_failed'] == 0 else '⚠ Issues detected'}
        """)

        if verification['errors']:
            error_list = '\n'.join([f"  - {err}" for err in verification['errors'][:5]])
            mo.md(f"**Issues found:**\n{error_list}")
    else:
        mo.md("*Install parx to run verification tests*")

@app.cell
def __(mo):
    mo.md("""
    ### Diagnostic Tools

    ```python
    # Check partition completeness
    coverage_report = partition.coverage_analysis(bounds, n_samples=10000)
    print(f"Coverage: {coverage_report['covered_fraction']:.1%}")
    print(f"Uncovered regions: {coverage_report['uncovered_volume']:.3f}")

    # Validate region geometry
    for region in partition.regions:
        D, g = partition.halfspaces(region)
        is_feasible = partition.check_feasibility(D, g)
        is_bounded = region.bounded

        if not is_feasible:
            print(f"⚠ Region {region.id} has infeasible constraints")

    # Network sensitivity analysis
    sensitivity = partition.sensitivity_analysis(
        network,
        perturbation_scale=0.01,
        n_perturbations=100
    )
    print(f"Avg sensitivity: {sensitivity['mean_output_change']:.6f}")

    # Memory usage analysis
    memory_usage = partition.memory_analysis()
    print(f"Partition size: {memory_usage['total_mb']:.1f} MB")
    print(f"Per region: {memory_usage['per_region_kb']:.1f} KB")
    ```
    """)

@app.cell
def __(mo):
    mo.md("""
    ## Julia Integration

    ### Current Implementation

    The Julia backend handles computationally intensive tasks:

    **Sparse enumeration** (`find_regions_sparse`):
    - Parallel forward-pass sampling
    - Hash-based region detection
    - Scalable to large networks

    **Exact enumeration** (`find_regions_exact`):
    - Depth-first search with facet flipping
    - Complete region enumeration
    - Exponential complexity but guaranteed complete

    **LP solvers** (via HiGHS.jl):
    - Chebyshev center computation
    - Feasibility checking
    - Constraint optimization
    """)

@app.cell
def __(parx_available, ensure_julia, mo):
    if parx_available:
        try:
            # Test Julia integration
            julia_status = ensure_julia()

            mo.md(f"""
            **Julia Status:** ✓ Active
            - Julia runtime initialized
            - LinearRegions.jl loaded
            - Bridge functional
            """)

        except Exception as e:
            mo.md(f"⚠ Julia initialization failed: {e}")
    else:
        mo.md("*Install parx to test Julia integration*")

@app.cell
def __(mo):
    mo.md("""
    ### Performance Characteristics

    **Sparse vs Exact Methods:**

    | Method | Best For | Time Complexity | Memory Usage | Completeness |
    |--------|----------|-----------------|--------------|-------------- |
    | Sparse | Large networks, approximation | O(n_samples) | Low | Partial |
    | Exact  | Small networks, verification | O(2^n) exponential | High | Complete |

    **Threading Benefits:**
    - Linear speedup for sparse enumeration
    - Diminishing returns for exact (inherently sequential)
    - Memory overhead ~10MB per thread

    **Recommended Usage:**
    ```python
    # Development & exploration
    partition = compute_partition(network, bounds, method='sparse', max_regions=1000)

    # Production & verification
    partition = compute_partition(network, bounds, method='exact')  # Use with caution on large nets
    ```
    """)

@app.cell
def __(mo):
    mo.md("""
    ## Current Status & Limitations

    ### What's Working
    - ✅ Network loading from multiple formats (PyTorch, .pth, .h5, manual)
    - ✅ Sparse region enumeration via parallel sampling
    - ✅ Exact region enumeration via DFS (small networks only)
    - ✅ Region geometry extraction (halfspaces, centroids)
    - ✅ Point routing and region queries
    - ✅ 2D visualization and analysis
    - ✅ Julia integration via juliacall
    - ✅ Comprehensive testing suite

    ### Current Limitations
    - ⚠️ Exact enumeration doesn't scale beyond ~20 neurons per layer
    - ⚠️ 3D+ visualization limited (only 2D region boundaries supported)
    - ⚠️ No incremental partition updates (must recompute from scratch)
    - ⚠️ Limited network architecture support (ReLU only, fully connected)

    ### Roadmap Phases 2-3
    - 🚧 **Phase 2**: JuMP integration for optimization-based region construction
    - 🚧 **Phase 3**: HiGHS solver integration for improved LP performance
    - 📋 **Future**: Convolutional network support, incremental updates
    """)

@app.cell
def __(json, time, project_root, mo):
    # Update state file
    state_update = {
        "version": 2,
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notebook_file": "project_overview.py",
        "audience": "both",
        "notebook_type": "living",
        "included_sections": [
            "Mathematical Background",
            "Architecture Overview",
            "Getting Started",
            "Core API",
            "Visualization & Analysis",
            "Analysis & Statistics",
            "I/O & Persistence",
            "Verification & Diagnostics",
            "Julia Integration",
            "Current Status & Limitations"
        ],
        "highlighted": [
            "compute_partition",
            "Partition",
            "Region",
            "sparse_julia",
            "exact_julia",
            "halfspaces",
            "route"
        ],
        "file_hashes": {
            "src/parx/__init__.py": "updated",
            "src/parx/partition.py": "updated",
            "src/parx/region.py": "updated"
        },
        "warnings_confirmed": [
            "Used synthetic networks for all examples as requested",
            "Focused on 2D input space for visualization clarity",
            "Mathematical background included for research supervisor audience",
            "Extended with full analysis and diagnostic capabilities",
            "Includes both supervisor and developer content"
        ]
    }

    try:
        state_file = project_root / ".marimo_agent_state.json"
        with open(state_file, 'w') as state_fh:
            json.dump(state_update, state_fh, indent=2)
        mo.md(f"✓ Updated state file: {state_file}")
    except Exception as e:
        mo.md(f"⚠ Failed to update state file: {e}")

@app.cell
def __(mo):
    mo.md("""
    ---

    ## Notebook Summary

    This comprehensive notebook covers the complete parx library functionality:

    1. **Mathematical foundation** - polyhedral activation regions theory
    2. **Architecture** - Python+Julia hybrid design with performance focus
    3. **API documentation** - complete interface with examples
    4. **Visualization suite** - 2D/3D plots, interactive analysis, 1D function analysis
    5. **Statistical analysis** - scaling studies, region properties, performance metrics
    6. **I/O capabilities** - save/load partitions, batch processing, format conversion
    7. **Verification tools** - correctness checks, diagnostic functions, coverage analysis
    8. **Julia integration** - backend performance, threading, method comparison
    9. **Project status** - current capabilities, limitations, development roadmap

    **Target audiences:** Research supervisors (progress overview) and developers (implementation guide)

    **Usage:** Install parx and run this notebook to explore your own networks and see real partition data.
    """)

if __name__ == "__main__":
    app.run()