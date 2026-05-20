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


# ── Color-by-metric helpers ──────────────────────────────────────────────────
#
# Each helper has signature ``(partition, region) -> float`` so it can be passed
# as the ``color_by=`` callable to ``plot_partition_2d``.  Helpers derive a
# scalar from the region's local affine map ``f(x) = A x + b``.

def affine_frobenius(partition: Partition, region: Region) -> float:
    """Frobenius norm ``‖A‖_F`` of the region's local affine map."""
    A, _ = partition.local_affine(region)
    return float(np.linalg.norm(A, ord="fro"))


def affine_spectral(partition: Partition, region: Region) -> float:
    """Spectral norm (top singular value) of A — the local Lipschitz constant."""
    A, _ = partition.local_affine(region)
    if A.size == 0:
        return 0.0
    return float(np.linalg.svd(A, compute_uv=False)[0])


def affine_det(partition: Partition, region: Region) -> float:
    """Determinant of A.  Raises ``ValueError`` if A is not square."""
    A, _ = partition.local_affine(region)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"affine_det requires square A, got shape {A.shape}")
    return float(np.linalg.det(A))


def active_neuron_count(_partition: Partition, region: Region) -> float:
    """Total number of active neurons across all layers of the path."""
    return float(sum(int(q.sum()) for q in region.activation_path))


# ── Public API ────────────────────────────────────────────────────────────────

def plot_partition_2d(
    partition: Partition,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    pad: float = 0.3,
    *,
    domain: tuple[tuple[float, float], tuple[float, float]] | None = None,
    layer: int | None = None,
    color_by=affine_frobenius,
    color_label: str | None = None,
    log_color: bool = False,
    colorscale: str = "Viridis",
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
    color_by : callable, default :func:`affine_frobenius`
        Callable ``(partition, region) -> float`` returning the scalar that
        determines each region's colour.  Built-in helpers in this module:
        :func:`affine_frobenius` (default — ``‖A‖_F``),
        :func:`affine_spectral` (local Lipschitz constant),
        :func:`affine_det`, :func:`active_neuron_count`.  Pass ``None`` to
        disable the metric mapping and revert to the discrete Turbo palette.
    color_label : str or None
        Colourbar title.  When ``None``, derived from ``color_by.__name__``.
    log_color : bool
        If ``True``, the metric is mapped onto the colorscale via ``log10``
        (clamped at ``1e-12``).  Useful when ``‖A‖`` spans orders of magnitude.
    colorscale : str
        Any Plotly colorscale name (``"Viridis"`` default, ``"Turbo"``,
        ``"Plasma"``, …).

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

    if color_by is None:
        # Fallback: discrete Turbo palette, no colourbar, no per-region metric.
        colors = px.colors.sample_colorscale(
            "Turbo", [i / max(n - 1, 1) for i in range(n)]
        )
        metrics  = None
        scaled   = None
        m_min    = m_max = None
        label_str = None
    else:
        metrics = np.array(
            [color_by(partition, r) for r in plot_regions], dtype=float
        )
        if log_color:
            scaled = np.log10(np.maximum(metrics, 1e-12))
        else:
            scaled = metrics
        m_min, m_max = float(scaled.min()), float(scaled.max())
        if m_max - m_min < 1e-12:
            normed = np.zeros_like(scaled)
        else:
            normed = (scaled - m_min) / (m_max - m_min)
        colors    = px.colors.sample_colorscale(colorscale, normed)
        label_str = color_label or getattr(color_by, "__name__", "metric")

    depth_str = f"layer {layer}" if layer is not None else "all layers"
    fig = go.Figure()
    for i, region in enumerate(plot_regions):
        D, g = partition.halfspaces(region)
        verts = _region_vertices_2d(D, g, x_range, y_range)
        if verts is None:
            continue

        xs = np.append(verts[:, 0], verts[0, 0])
        ys = np.append(verts[:, 1], verts[0, 1])

        hover_lines = []
        if metrics is not None:
            hover_lines.append(f"{label_str}: {metrics[i]:.4g}")
        hover_lines.append(_activation_label(region))
        hover_text = "<br>".join(hover_lines)

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
                hovertemplate=f"{hover_text}<extra></extra>",
            )
        )

    # Add a hidden marker trace just to expose the colorbar.  The markers
    # themselves are invisible (opacity=0) and placed inside the view; Plotly
    # surfaces the colorscale via showscale=True on the marker.
    if metrics is not None:
        cb_title = label_str + (" (log10)" if log_color else "")
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(
                    color=[m_min, m_max],
                    colorscale=colorscale,
                    cmin=m_min,
                    cmax=m_max,
                    showscale=True,
                    opacity=0,
                    colorbar=dict(title=cb_title, thickness=14),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        xaxis=dict(range=list(x_range), title="x₁", constrain="domain"),
        yaxis=dict(range=list(y_range), title="x₂", scaleanchor="x"),
        title=f"Linear regions — {depth_str}  ({n} regions)",
        width=620 if metrics is not None else 520,
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
