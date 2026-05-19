"""Plotly visualisations for Partition objects."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from parx.partition import Partition
from parx.region import Region


def plot_partition_2d(
    partition: Partition,
    x_range: tuple[float, float] = (-1.0, 1.0),
    y_range: tuple[float, float] = (-1.0, 1.0),
    resolution: int = 200,
) -> go.Figure:
    """Rasterise the input plane by region membership.

    Routes a ``resolution × resolution`` grid through the partition and colours
    each cell by region index.  Cells that map to no known region (possible for
    sparse partitions that don't cover the full domain) are left transparent.

    Parameters
    ----------
    partition : Partition
        Must have ``input_dim == 2``.
    x_range, y_range : (float, float)
        Axis extents for the plot grid.
    resolution : int
        Number of grid points along each axis.

    Returns
    -------
    go.Figure
    """
    if partition.input_dim != 2:
        raise ValueError(
            f"plot_partition_2d requires input_dim=2, got {partition.input_dim}"
        )

    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    XX, YY = np.meshgrid(xs, ys)
    X_grid = np.column_stack([XX.ravel(), YY.ravel()])

    routed = partition.route(X_grid)
    region_to_idx = {id(r): i for i, r in enumerate(partition.regions)}

    Z = np.full(len(X_grid), np.nan)
    for k, r in enumerate(routed):
        if r is not None:
            Z[k] = float(region_to_idx[id(r)])
    Z = Z.reshape(resolution, resolution)

    fig = go.Figure(
        go.Heatmap(
            x=xs,
            y=ys,
            z=Z,
            colorscale="Turbo",
            zsmooth=False,
            showscale=False,
            hovertemplate="x=%{x:.3f}, y=%{y:.3f}, region=%{z:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="x₁",
        yaxis_title="x₂",
        title=f"Linear regions ({len(partition)} regions)",
        width=520,
        height=500,
    )
    return fig


def plot_region_counts(partition: Partition) -> go.Figure:
    """Bar chart of distinct region count at each depth.

    At depth ``d``, counts the number of unique activation-path prefixes of
    length ``d`` across all leaf regions.  Shows how partition complexity grows
    layer by layer.

    Returns
    -------
    go.Figure
    """
    depths = list(range(1, partition.n_layers + 1))
    counts = []
    for d in depths:
        prefixes = {
            tuple(r.activation_path[l].tobytes() for l in range(d))
            for r in partition.regions
        }
        counts.append(len(prefixes))

    fig = go.Figure(
        go.Bar(
            x=depths,
            y=counts,
            marker_color="steelblue",
        )
    )
    fig.update_layout(
        xaxis=dict(title="Layer depth", tickmode="linear", dtick=1),
        yaxis_title="Distinct regions",
        title=f"Region complexity by depth  (leaf total: {len(partition)})",
    )
    return fig


def plot_halfspaces(
    partition: Partition,
    region: Region,
    x_range: tuple[float, float] = (-2.0, 2.0),
    y_range: tuple[float, float] = (-2.0, 2.0),
) -> go.Figure:
    """Draw the halfspace boundaries that define a single region.

    For each row ``D[i,:] x = g[i]`` of the region's constraint system, draws
    the corresponding line in the 2-D plane clipped to the plot window.

    Parameters
    ----------
    partition : Partition
        Must have ``input_dim == 2``.
    region : Region
        The region whose halfspace boundaries to draw.
    x_range, y_range : (float, float)
        Axis extents.

    Returns
    -------
    go.Figure
    """
    if partition.input_dim != 2:
        raise ValueError(
            f"plot_halfspaces requires input_dim=2, got {partition.input_dim}"
        )

    D, g = partition.halfspaces(region)
    xs = np.linspace(x_range[0], x_range[1], 400)

    fig = go.Figure()

    for i in range(D.shape[0]):
        d0, d1 = D[i, 0], D[i, 1]
        gi = g[i]
        if abs(d1) > 1e-10:
            y_line = (gi - d0 * xs) / d1
            mask = (y_line >= y_range[0]) & (y_line <= y_range[1])
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=xs[mask],
                        y=y_line[mask],
                        mode="lines",
                        line=dict(color="rgba(80,80,80,0.45)", width=1),
                        showlegend=False,
                    )
                )
        elif abs(d0) > 1e-10:
            # Vertical line
            x_val = gi / d0
            if x_range[0] <= x_val <= x_range[1]:
                fig.add_trace(
                    go.Scatter(
                        x=[x_val, x_val],
                        y=[y_range[0], y_range[1]],
                        mode="lines",
                        line=dict(color="rgba(80,80,80,0.45)", width=1),
                        showlegend=False,
                    )
                )

    # Mark centroid if it lies within the plot window
    c = region.centroid
    if x_range[0] <= c[0] <= x_range[1] and y_range[0] <= c[1] <= y_range[1]:
        fig.add_trace(
            go.Scatter(
                x=[c[0]],
                y=[c[1]],
                mode="markers",
                marker=dict(size=10, color="crimson", symbol="x"),
                name="centroid",
            )
        )

    fig.update_layout(
        xaxis=dict(range=list(x_range), title="x₁"),
        yaxis=dict(range=list(y_range), title="x₂"),
        title=f"Halfspaces for region  ({D.shape[0]} constraints)",
    )
    return fig
