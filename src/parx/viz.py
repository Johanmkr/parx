"""Plotly visualisations for Partition objects."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from parx.partition import Partition
from parx.region import Region


# ── Internal helpers ─────────────────────────────────────────────────────────

def _activation_label(region: Region) -> str:
    """Return a compact, per-layer activation pattern string for hover labels."""
    parts = []
    for l, q in enumerate(region.activation_path):
        bits = "".join(str(int(b)) for b in q)
        parts.append(f"L{l + 1}: {bits}")
    return "<br>".join(parts)


def _auto_range_2d(
    partition: Partition,
    pad: float = 0.3,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute a view range that shows all hyperplane intersection points.

    Collects every unique boundary hyperplane from the partition, finds all
    pairwise intersection points in 2-D, and returns a padded bounding box.
    Falls back to (-2, 2) × (-2, 2) when no finite intersections exist.
    """
    # Collect unique normalised hyperplanes across all regions
    rows: list[np.ndarray] = []
    for region in partition.regions:
        D, g = partition.halfspaces(region)
        for i in range(D.shape[0]):
            nrm = np.linalg.norm(D[i, :])
            if nrm > 1e-10:
                row = np.append(D[i, :] / nrm, g[i] / nrm)
                # Canonicalise sign: first large element is positive
                for v in row[:2]:
                    if abs(v) > 1e-10:
                        if v < 0:
                            row = -row
                        break
                rows.append(row)

    if not rows:
        return (-2.0, 2.0), (-2.0, 2.0)

    rows_arr = np.unique(np.round(rows, 8), axis=0)
    D_u, g_u = rows_arr[:, :2], rows_arr[:, 2]
    n = len(g_u)

    # All pairwise intersections
    pts: list[np.ndarray] = []
    for i in range(n):
        for j in range(i + 1, n):
            try:
                x = np.linalg.solve(D_u[[i, j], :], g_u[[i, j]])
                if np.all(np.isfinite(x)):
                    pts.append(x)
            except np.linalg.LinAlgError:
                continue

    if not pts:
        return (-2.0, 2.0), (-2.0, 2.0)

    pts_arr = np.array(pts)
    xs, ys = pts_arr[:, 0], pts_arr[:, 1]
    span   = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1.0)
    margin = pad * span
    return (float(xs.min()) - margin, float(xs.max()) + margin), \
           (float(ys.min()) - margin, float(ys.max()) + margin)


def _region_vertices_2d(
    D: np.ndarray,
    g: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> np.ndarray | None:
    """CCW-ordered vertices of the 2-D polytope {x : D x ≤ g} clipped to the view box.

    Algorithm: add bounding-box constraints, enumerate all pairwise boundary-line
    intersections, keep those that satisfy every constraint (= vertices of the
    clipped polytope), sort counter-clockwise.  Produces crisp vector polygons
    with no sampling artefacts.
    """
    box_D = np.array([[ 1., 0.], [-1., 0.], [ 0.,  1.], [ 0., -1.]])
    box_g = np.array([x_range[1], -x_range[0], y_range[1], -y_range[0]])
    D_all = np.vstack([D, box_D])
    g_all = np.hstack([g, box_g])

    # Drop zero-norm rows (degenerate constraints — can't define a boundary line)
    valid = np.linalg.norm(D_all, axis=1) > 1e-10
    D_all, g_all = D_all[valid], g_all[valid]

    n = len(g_all)
    verts: list[np.ndarray] = []
    for i in range(n):
        for j in range(i + 1, n):
            A = D_all[[i, j], :]
            try:
                x = np.linalg.solve(A, g_all[[i, j]])
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(x)):
                continue
            if np.all(D_all @ x <= g_all + 1e-8):
                verts.append(x)

    if len(verts) < 3:
        return None

    pts = np.unique(np.round(verts, 8), axis=0)
    if len(pts) < 3:
        return None

    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(angles)]


# ── Public API ────────────────────────────────────────────────────────────────

def plot_partition_2d(
    partition: Partition,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    pad: float = 0.3,
    *,
    domain: tuple[tuple[float, float], tuple[float, float]] | None = None,
    layer: int | None = None,
) -> go.Figure:
    """Draw each linear region as a filled, crisp polygon.

    Each region is rendered as a vector-quality ``go.Scatter`` polygon, giving
    sharp boundaries regardless of plot size — no pixellation artefacts.
    Hovering over a region shows its per-layer activation pattern.

    Parameters
    ----------
    partition : Partition
        Must have ``input_dim == 2``.
    domain : ((float, float), (float, float)) or None
        Explicit ``(x_range, y_range)`` bounds for the plot. When provided,
        these bounds are used directly and no auto-ranging is performed.
    x_range, y_range : (float, float) or None
        Axis extents.  When ``None`` (default) the range is auto-computed from
        the arrangement of hyperplane intersection points so all regions are
        visible.  Unbounded regions are clipped to the computed box.
    pad : float
        Fractional padding added around the auto-computed bounding box.
    layer : int or None
        If given (1-indexed), collapse all leaf regions to their activation-path
        prefix of this length and plot the resulting coarser partition.
        ``layer=1`` shows the regions induced by just the first ReLU layer;
        ``layer=partition.n_layers`` (the default) shows full leaf regions.
        Must satisfy ``1 <= layer <= partition.n_layers``.

    Returns
    -------
    go.Figure
    """
    if partition.input_dim != 2:
        raise ValueError(
            f"plot_partition_2d requires input_dim=2, got {partition.input_dim}"
        )

    if layer is not None and not (1 <= layer <= partition.n_layers):
        raise ValueError(
            f"layer must be between 1 and {partition.n_layers}, got {layer}"
        )

    # Build the list of regions to plot (possibly coarser than leaf level)
    if layer is not None:
        seen: dict[tuple[bytes, ...], Region] = {}
        for r in partition.regions:
            key = tuple(r.activation_path[l].tobytes() for l in range(layer))
            if key not in seen:
                seen[key] = Region(
                    activation_path=r.activation_path[:layer],
                    centroid=r.centroid,
                )
        plot_regions = list(seen.values())
    else:
        plot_regions = partition.regions

    n = len(plot_regions)
    if n == 0:
        return go.Figure()

    if domain is not None:
        x_range, y_range = domain
    elif x_range is None or y_range is None:
        auto_x, auto_y = _auto_range_2d(partition, pad=pad)
        x_range = x_range or auto_x
        y_range = y_range or auto_y

    colors = px.colors.sample_colorscale(
        "Turbo", [i / max(n - 1, 1) for i in range(n)]
    )

    depth_str = f"layer {layer}" if layer is not None else "all layers"
    fig = go.Figure()
    for i, region in enumerate(plot_regions):
        D, g = partition.halfspaces(region)
        verts = _region_vertices_2d(D, g, x_range, y_range)
        if verts is None:
            continue

        xs = np.append(verts[:, 0], verts[0, 0])   # close the polygon
        ys = np.append(verts[:, 1], verts[0, 1])

        label = _activation_label(region)
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                fill="toself",
                fillcolor=colors[i],
                line=dict(color="black", width=0.8),
                mode="lines",
                opacity=0.75,
                showlegend=False,
                name=f"region {i}",
                hovertemplate=f"{label}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis=dict(range=list(x_range), title="x₁", constrain="domain"),
        yaxis=dict(range=list(y_range), title="x₂", scaleanchor="x"),
        title=f"Linear regions — {depth_str}  ({n} regions)",
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
        go.Bar(x=depths, y=counts, marker_color="steelblue")
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
                        x=xs[mask], y=y_line[mask],
                        mode="lines",
                        line=dict(color="rgba(80,80,80,0.45)", width=1),
                        showlegend=False,
                    )
                )
        elif abs(d0) > 1e-10:
            x_val = gi / d0
            if x_range[0] <= x_val <= x_range[1]:
                fig.add_trace(
                    go.Scatter(
                        x=[x_val, x_val], y=[y_range[0], y_range[1]],
                        mode="lines",
                        line=dict(color="rgba(80,80,80,0.45)", width=1),
                        showlegend=False,
                    )
                )

    c = region.centroid
    if x_range[0] <= c[0] <= x_range[1] and y_range[0] <= c[1] <= y_range[1]:
        fig.add_trace(
            go.Scatter(
                x=[c[0]], y=[c[1]],
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
