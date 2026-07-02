"""Plotly visualisations for Partition objects."""

from __future__ import annotations

import colorsys

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from parx.partition import Partition
from parx.region import Region

# ── Internal helpers ─────────────────────────────────────────────────────────


def _activation_label(region: Region) -> str:
    """Return a compact, per-layer activation pattern string for hover labels."""
    parts = []
    for layer_idx, q in enumerate(region.activation_path):
        bits = "".join(str(int(b)) for b in q)
        parts.append(f"L{layer_idx + 1}: {bits}")
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
    span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1.0)
    margin = pad * span
    return (float(xs.min()) - margin, float(xs.max()) + margin), (
        float(ys.min()) - margin,
        float(ys.max()) + margin,
    )


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
    box_D = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
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


def _spatial_colors(regions: list[Region]) -> list[str]:
    """HSV colors keyed to centroid position: hue = angle, saturation = radius."""
    centroids = np.array([r.centroid for r in regions], dtype=float)
    cx, cy = centroids.mean(axis=0)
    dx, dy = centroids[:, 0] - cx, centroids[:, 1] - cy
    angles = np.arctan2(dy, dx)
    radii = np.hypot(dx, dy)
    r_max = radii.max() if radii.max() > 1e-12 else 1.0
    hues = (angles / (2 * np.pi) + 0.5) % 1.0
    sats = 0.35 + 0.65 * (radii / r_max)
    out = []
    for h, s in zip(hues, sats):
        r, g, b = colorsys.hsv_to_rgb(h, s, 0.88)
        out.append(f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})")
    return out


def region_palette(partition: Partition, scheme: str = "random") -> list[str]:
    """Return one CSS colour string per region in ``partition.regions``.

    Parameters
    ----------
    partition : Partition
    scheme : 'random', 'frobenius', or 'spatial'
        ``'random'`` — Turbo palette indexed by region order.
        ``'frobenius'`` — Viridis palette ordered by ``‖A‖_F``.
        ``'spatial'`` — HSV colour keyed to centroid angle (hue) and
        radial distance from the mean centroid (saturation).
    """
    regions = partition.regions
    n = len(regions)
    if scheme == "random":
        positions = [i / max(n - 1, 1) for i in range(n)]
        return list(px.colors.sample_colorscale("Turbo", positions))
    if scheme == "frobenius":
        metrics = np.array(
            [affine_frobenius(partition, r) for r in regions], dtype=float
        )
        span = metrics.max() - metrics.min()
        normed = (metrics - metrics.min()) / (span if span > 1e-12 else 1.0)
        return list(px.colors.sample_colorscale("Viridis", normed))
    if scheme == "spatial":
        return _spatial_colors(regions)
    raise ValueError(
        f"unknown scheme {scheme!r}; choose 'random', 'frobenius', or 'spatial'"
    )


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
    colors: list[str] | None = None,
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
    colors : list[str] or None
        Pre-computed CSS colour strings — one per region in
        ``partition.regions`` (or per ``plot_regions`` when ``layer`` is set).
        When provided, ``color_by``, ``colorscale``, and ``log_color`` are
        ignored and no colourbar is shown.  Use :func:`region_palette` to
        generate a matching list.

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
            key = tuple(
                r.activation_path[layer_idx].tobytes()
                for layer_idx in range(layer)
            )
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

    if colors is not None:
        # Pre-computed palette: bypass color_by entirely, no colourbar.
        metrics = None
        scaled = None
        m_min = m_max = None
        label_str = None
    elif color_by is None:
        # Discrete Turbo palette, no colourbar, no per-region metric.
        colors = px.colors.sample_colorscale(
            "Turbo", [i / max(n - 1, 1) for i in range(n)]
        )
        metrics = None
        scaled = None
        m_min = m_max = None
        label_str = None
    else:
        metrics = np.array([color_by(partition, r) for r in plot_regions], dtype=float)
        if log_color:
            scaled = np.log10(np.maximum(metrics, 1e-12))
        else:
            scaled = metrics
        m_min, m_max = float(scaled.min()), float(scaled.max())
        if m_max - m_min < 1e-12:
            normed = np.zeros_like(scaled)
        else:
            normed = (scaled - m_min) / (m_max - m_min)
        colors = px.colors.sample_colorscale(colorscale, normed)
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
                x=xs,
                y=ys,
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
                x=[None],
                y=[None],
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
            tuple(
                r.activation_path[layer_idx].tobytes()
                for layer_idx in range(d)
            )
            for r in partition.regions
        }
        counts.append(len(prefixes))

    fig = go.Figure(go.Bar(x=depths, y=counts, marker_color="steelblue"))
    fig.update_layout(
        xaxis=dict(title="Layer depth", tickmode="linear", dtick=1),
        yaxis_title="Distinct regions",
        title=f"Region complexity by depth  (leaf total: {len(partition)})",
    )
    return fig


def _auto_range_2d_from_systems(
    systems: list[tuple[np.ndarray, np.ndarray]],
    pad: float = 0.3,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute a view range from a collection of 2D (D, g) systems.

    Finds all pairwise intersection points across the unique hyperplanes of
    every system, returns a padded bounding box.  Falls back to (-2, 2) × (-2, 2)
    when no finite intersections exist.
    """
    rows: list[np.ndarray] = []
    for D, g in systems:
        for i in range(D.shape[0]):
            nrm = np.linalg.norm(D[i, :])
            if nrm > 1e-10:
                row = np.append(D[i, :] / nrm, g[i] / nrm)
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
    span = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 1.0)
    margin = pad * span
    return (float(xs.min()) - margin, float(xs.max()) + margin), (
        float(ys.min()) - margin,
        float(ys.max()) + margin,
    )


def _build_colored_figure(
    plot_regions: list,
    systems_2d: list[tuple[np.ndarray, np.ndarray]],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    partition,
    color_by,
    color_label: str | None,
    log_color: bool,
    colorscale: str,
    title: str,
) -> go.Figure:
    """Shared figure-building logic for slice/projection plots.

    Iterates over regions with their corresponding 2D halfspace systems,
    calls ``_region_vertices_2d``, and assembles a ``go.Figure`` with the
    same coloring and layout conventions as ``plot_partition_2d``.
    """
    n = len(plot_regions)

    if color_by is None:
        colors = px.colors.sample_colorscale(
            "Turbo", [i / max(n - 1, 1) for i in range(n)]
        )
        metrics = None
        scaled = None
        m_min = m_max = None
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
        colors = px.colors.sample_colorscale(colorscale, normed)
        label_str = color_label or getattr(color_by, "__name__", "metric")

    fig = go.Figure()
    for i, (region, (D_2d, g_2d)) in enumerate(zip(plot_regions, systems_2d)):
        verts = _region_vertices_2d(D_2d, g_2d, x_range, y_range)
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
                x=xs,
                y=ys,
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

    if metrics is not None:
        cb_title = label_str + (" (log10)" if log_color else "")
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
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
        title=title,
        width=620 if metrics is not None else 520,
        height=500,
    )
    return fig


def _global_range(
    partitions: list,
    pad: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    systems = [
        partition.halfspaces(r)
        for partition in partitions
        for r in partition.regions
    ]
    return _auto_range_2d_from_systems(systems, pad)


def _global_metric_range(
    partitions: list,
    color_by,
    log_color: bool,
) -> tuple[float, float]:
    vals = np.array(
        [color_by(p, r) for p in partitions for r in p.regions], dtype=float
    )
    scaled = np.log10(np.maximum(vals, 1e-12)) if log_color else vals
    return float(scaled.min()), float(scaled.max())


def plot_partition_slice(
    partition: Partition,
    free_dims: tuple[int, int],
    fixed_values: dict[int, float],
    *,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    pad: float = 0.3,
    color_by=affine_frobenius,
    color_label: str | None = None,
    log_color: bool = False,
    colorscale: str = "Viridis",
) -> go.Figure:
    """Draw an axis-aligned 2D slice through a higher-dimensional partition.

    Fixes all input dimensions except ``free_dims[0]`` and ``free_dims[1]``
    to the values in ``fixed_values``, then renders the induced 2D polyhedral
    partition on those two free axes.  Regions whose polytopes do not intersect
    the slice plane are silently skipped.

    Parameters
    ----------
    partition : Partition
        The partition to visualise.  ``input_dim`` must be ≥ 2.
    free_dims : (int, int)
        The two input-dimension indices to keep free (0-indexed).
    fixed_values : dict[int, float]
        Maps every other dimension index to its fixed value.  All dimensions
        not in ``free_dims`` must appear as keys.
    x_range, y_range : (float, float) or None
        Axis extents for the free dimensions.  When ``None`` (default) the
        range is auto-computed from the induced 2D hyperplane intersections.
    pad : float
        Fractional padding added around the auto-computed bounding box.
    color_by : callable or None, default :func:`affine_frobenius`
        Callable ``(partition, region) -> float`` returning the scalar used
        to colour each region.  Pass ``None`` for a discrete Turbo palette.
    color_label : str or None
        Colourbar title.  When ``None``, derived from ``color_by.__name__``.
    log_color : bool
        If ``True``, colour values are mapped via ``log10``.
    colorscale : str
        Any Plotly colorscale name (default ``"Viridis"``).

    Returns
    -------
    go.Figure
    """
    free0, free1 = free_dims
    fixed_dims = [d for d in range(partition.input_dim) if d not in free_dims]
    fixed_vals = np.array([fixed_values[d] for d in fixed_dims])

    systems_2d: list[tuple[np.ndarray, np.ndarray]] = []
    plot_regions = []
    for region in partition.regions:
        D, g = partition.halfspaces(region)
        D_2d = D[:, [free0, free1]]
        g_2d = g - (D[:, fixed_dims] @ fixed_vals if fixed_dims else g * 0.0)
        systems_2d.append((D_2d, g_2d))
        plot_regions.append(region)

    if not plot_regions:
        return go.Figure()

    if x_range is None or y_range is None:
        auto_x, auto_y = _auto_range_2d_from_systems(systems_2d, pad=pad)
        x_range = x_range if x_range is not None else auto_x
        y_range = y_range if y_range is not None else auto_y

    n_visible = sum(
        1
        for D_2d, g_2d in systems_2d
        if _region_vertices_2d(D_2d, g_2d, x_range, y_range) is not None
    )
    fixed_str = ", ".join(f"x{d}={v:.3g}" for d, v in fixed_values.items())
    title = (
        f"Partition slice — free dims ({free0}, {free1}), fixed: {fixed_str}"
        f"  ({n_visible} visible regions)"
    )

    return _build_colored_figure(
        plot_regions,
        systems_2d,
        x_range,
        y_range,
        partition,
        color_by,
        color_label,
        log_color,
        colorscale,
        title,
    )


def plot_partition_projection(
    partition: Partition,
    projection: np.ndarray,
    *,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    pad: float = 0.3,
    color_by=affine_frobenius,
    color_label: str | None = None,
    log_color: bool = False,
    colorscale: str = "Viridis",
) -> go.Figure:
    """Draw an approximate 2D projection of a higher-dimensional partition.

    Projects each region's halfspace normals onto a 2D subspace defined by
    ``projection``:

    .. code-block:: python

        D_proj = D @ projection   # (n_constraints, 2)
        # g unchanged

    .. warning::
        This is an **approximation**.  Projecting the dual representation
        (halfspace normals) does not, in general, yield the exact projected
        polytope — the true projection of a convex polytope requires
        eliminating the non-free variables (e.g. via Fourier-Motzkin), which
        is computationally expensive.  The visualisation is useful for
        qualitative exploration but should not be interpreted as geometrically
        exact in dimensions ≥ 3.

    Parameters
    ----------
    partition : Partition
        The partition to visualise.
    projection : np.ndarray, shape (input_dim, 2)
        Linear map from input space to the 2D display plane.  Columns are the
        two basis vectors of the subspace.  Need not be orthonormal, though
        orthonormal columns give a shape-preserving projection.
    x_range, y_range : (float, float) or None
        Axis extents.  When ``None`` (default), auto-computed from the
        projected hyperplane intersections.
    pad : float
        Fractional padding added around the auto-computed bounding box.
    color_by : callable or None, default :func:`affine_frobenius`
        Callable ``(partition, region) -> float``.  Pass ``None`` for a
        discrete Turbo palette.
    color_label : str or None
        Colourbar title.
    log_color : bool
        If ``True``, colour values are mapped via ``log10``.
    colorscale : str
        Any Plotly colorscale name (default ``"Viridis"``).

    Returns
    -------
    go.Figure
    """
    projection = np.asarray(projection, dtype=float)
    if projection.shape != (partition.input_dim, 2):
        raise ValueError(
            f"projection must have shape (input_dim, 2) = ({partition.input_dim}, 2), "
            f"got {projection.shape}"
        )

    systems_2d: list[tuple[np.ndarray, np.ndarray]] = []
    plot_regions = list(partition.regions)
    for region in plot_regions:
        D, g = partition.halfspaces(region)
        D_proj = D @ projection
        systems_2d.append((D_proj, g))

    if not plot_regions:
        return go.Figure()

    if x_range is None or y_range is None:
        auto_x, auto_y = _auto_range_2d_from_systems(systems_2d, pad=pad)
        x_range = x_range if x_range is not None else auto_x
        y_range = y_range if y_range is not None else auto_y

    n = len(plot_regions)
    title = f"Partition projection (approx.)  ({n} regions)"

    return _build_colored_figure(
        plot_regions,
        systems_2d,
        x_range,
        y_range,
        partition,
        color_by,
        color_label,
        log_color,
        colorscale,
        title,
    )


def plot_partition_pca(
    partition: Partition,
    data: np.ndarray,
    *,
    color_by=affine_frobenius,
    color_label: str | None = None,
    log_color: bool = False,
    colorscale: str = "Viridis",
) -> go.Figure:
    """Draw an approximate 2D PCA projection of a higher-dimensional partition.

    Fits a 2-component PCA on ``data``, uses the principal-component axes as
    the projection matrix, and delegates to :func:`plot_partition_projection`.

    The projection is the same approximation noted in
    :func:`plot_partition_projection` — halfspace normals are projected, not
    the polytopes themselves.

    Parameters
    ----------
    partition : Partition
        The partition to visualise.
    data : np.ndarray, shape (N, input_dim)
        Data used to fit the PCA.  Typically the training or evaluation set.
    color_by : callable or None, default :func:`affine_frobenius`
        Callable ``(partition, region) -> float``.  Pass ``None`` for a
        discrete Turbo palette.
    color_label : str or None
        Colourbar title.
    log_color : bool
        If ``True``, colour values are mapped via ``log10``.
    colorscale : str
        Any Plotly colorscale name (default ``"Viridis"``).

    Returns
    -------
    go.Figure

    Raises
    ------
    ImportError
        If ``scikit-learn`` is not installed.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError(
            "plot_partition_pca requires scikit-learn: pip install scikit-learn"
        ) from e

    pca = PCA(n_components=2)
    pca.fit(np.asarray(data, dtype=float))
    projection = pca.components_.T  # shape (input_dim, 2)

    return plot_partition_projection(
        partition,
        projection,
        color_by=color_by,
        color_label=color_label,
        log_color=log_color,
        colorscale=colorscale,
    )


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


def animate_epochs(
    partitions: list[Partition],
    *,
    epoch_labels: list[str] | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    pad: float = 0.3,
    color_by=affine_frobenius,
    color_label: str | None = None,
    log_color: bool = False,
    colorscale: str = "Viridis",
    frame_duration: int = 500,
) -> go.Figure:
    """Plotly figure with one frame per epoch, a play/pause button, and a slider.

    All frames share a fixed color scale and spatial range so that changes
    across epochs are visually comparable.  Each partition in ``partitions``
    corresponds to one epoch (frame); the list index is used as the epoch label
    unless ``epoch_labels`` is provided.

    Parameters
    ----------
    partitions : list[Partition]
        One Partition per epoch, all must have ``input_dim == 2``.
    epoch_labels : list[str] or None
        Labels shown in the slider.  Defaults to ``["0", "1", …]``.
    x_range, y_range : (float, float) or None
        Axis extents shared across all frames.  Auto-computed when ``None``.
    pad : float
        Fractional padding for auto-computed range.
    color_by : callable or None
        ``(partition, region) -> float`` metric.  Pass ``None`` for a discrete
        Turbo palette (color range not shared across epochs in that case).
    color_label : str or None
        Colourbar title.
    log_color : bool
        Map metric via ``log10`` before colouring.
    colorscale : str
        Any Plotly colorscale name.
    frame_duration : int
        Milliseconds each frame is shown during playback.

    Returns
    -------
    go.Figure
    """
    if not partitions:
        return go.Figure()
    if any(p.input_dim != 2 for p in partitions):
        raise ValueError(
            "animate_epochs requires all partitions to have input_dim == 2"
        )

    labels = epoch_labels or [str(i) for i in range(len(partitions))]
    if len(labels) != len(partitions):
        raise ValueError("epoch_labels length must match number of partitions")

    if x_range is None or y_range is None:
        auto_x, auto_y = _global_range(partitions, pad)
        x_range = x_range or auto_x
        y_range = y_range or auto_y

    if color_by is not None:
        m_min, m_max = _global_metric_range(partitions, color_by, log_color)
        label_str = color_label or getattr(color_by, "__name__", "metric")
        cb_title = label_str + (" (log10)" if log_color else "")
    else:
        m_min = m_max = None
        label_str = None
        cb_title = None

    # Pre-compute per-epoch polygon data
    epoch_traces: list[list[go.Scatter]] = []
    for partition in partitions:
        traces: list[go.Scatter] = []
        regions = partition.regions
        if color_by is not None:
            raw = np.array([color_by(partition, r) for r in regions], dtype=float)
            scaled = np.log10(np.maximum(raw, 1e-12)) if log_color else raw
            if m_max - m_min < 1e-12:
                normed = np.zeros_like(scaled)
            else:
                normed = (scaled - m_min) / (m_max - m_min)
            colors = px.colors.sample_colorscale(colorscale, normed)
        else:
            n = len(regions)
            colors = px.colors.sample_colorscale(
                "Turbo", [i / max(n - 1, 1) for i in range(n)]
            )

        for i, region in enumerate(regions):
            D, g = partition.halfspaces(region)
            verts = _region_vertices_2d(D, g, x_range, y_range)
            if verts is None:
                xs_poly: list = []
                ys_poly: list = []
            else:
                xs_poly = list(np.append(verts[:, 0], verts[0, 0]))
                ys_poly = list(np.append(verts[:, 1], verts[0, 1]))

            hover_lines = []
            if color_by is not None:
                hover_lines.append(f"{label_str}: {raw[i]:.4g}")
            hover_lines.append(_activation_label(region))
            hover_text = "<br>".join(hover_lines)

            traces.append(
                go.Scatter(
                    x=xs_poly,
                    y=ys_poly,
                    fill="toself",
                    fillcolor=colors[i],
                    line=dict(color="black", width=0.8),
                    mode="lines",
                    opacity=0.75,
                    showlegend=False,
                    hovertemplate=f"{hover_text}<extra></extra>",
                )
            )
        epoch_traces.append(traces)

    n_max = max(len(t) for t in epoch_traces)
    # Colorbar trace index is always n_max
    _empty = go.Scatter(
        x=[],
        y=[],
        fill="toself",
        fillcolor="rgba(0,0,0,0)",
        line=dict(color="rgba(0,0,0,0)", width=0),
        mode="lines",
        showlegend=False,
        hoverinfo="skip",
    )

    def _pad_traces(traces: list[go.Scatter]) -> list[go.Scatter]:
        padded = list(traces)
        while len(padded) < n_max:
            padded.append(_empty)
        return padded

    colorbar_trace = go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            color=[m_min, m_max] if m_min is not None else [0, 1],
            colorscale=colorscale,
            cmin=m_min,
            cmax=m_max,
            showscale=(m_min is not None),
            opacity=0,
            colorbar=dict(title=cb_title or "", thickness=14),
        ),
        hoverinfo="skip",
        showlegend=False,
    )

    # Initial figure data (first epoch)
    initial_traces = _pad_traces(epoch_traces[0]) + [colorbar_trace]
    fig = go.Figure(data=initial_traces)

    # Build frames
    frames = []
    for label, traces in zip(labels, epoch_traces):
        frame_data = _pad_traces(traces) + [colorbar_trace]
        frames.append(
            go.Frame(
                data=frame_data,
                name=label,
                traces=list(range(n_max + 1)),
            )
        )
    fig.frames = frames

    # Slider steps
    slider_steps = [
        dict(
            args=[
                [label],
                dict(
                    frame=dict(duration=frame_duration, redraw=True),
                    mode="immediate",
                    transition=dict(duration=0),
                ),
            ],
            label=label,
            method="animate",
        )
        for label in labels
    ]

    fig.update_layout(
        xaxis=dict(range=list(x_range), title="x₁", constrain="domain"),
        yaxis=dict(range=list(y_range), title="x₂", scaleanchor="x"),
        title=f"Partition evolution  ({len(partitions)} epochs)",
        width=640 if m_min is not None else 520,
        height=540,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.08,
                x=0.0,
                xanchor="left",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=frame_duration, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Epoch: ", visible=True, xanchor="center"),
                pad=dict(t=60),
                steps=slider_steps,
            )
        ],
    )
    return fig


def animate_epochs_video(
    partitions: list[Partition],
    path,
    *,
    epoch_labels: list[str] | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    pad: float = 0.3,
    color_by=affine_frobenius,
    color_label: str | None = None,
    log_color: bool = False,
    colorscale: str = "Viridis",
    fps: int = 4,
    dpi: int = 150,
    figsize: tuple[float, float] = (6.0, 5.0),
) -> None:
    """Write an animated ``.gif`` or ``.mp4`` of the partition across epochs.

    Format is inferred from the ``path`` suffix (``.gif`` → Pillow writer,
    ``.mp4`` → ffmpeg writer).

    Parameters
    ----------
    partitions : list[Partition]
        One Partition per epoch, all must have ``input_dim == 2``.
    path : str or os.PathLike
        Output file.  Suffix determines format: ``.gif`` or ``.mp4``.
    epoch_labels : list[str] or None
        Title suffix per frame.  Defaults to ``["0", "1", …]``.
    x_range, y_range : (float, float) or None
        Axis extents shared across all frames.  Auto-computed when ``None``.
    pad : float
        Fractional padding for auto-computed range.
    color_by : callable or None
        ``(partition, region) -> float`` metric.
    color_label : str or None
        Colorbar label.
    log_color : bool
        Map metric via ``log10``.
    colorscale : str
        Plotly colorscale name; lowercased to resolve a matplotlib colormap
        (``"Viridis"`` → ``"viridis"`` etc.).
    fps : int
        Frames per second.
    dpi : int
        Output resolution in dots per inch.
    figsize : (float, float)
        Matplotlib figure size in inches.

    Raises
    ------
    ImportError
        If ``matplotlib`` is not installed.
    """
    try:
        import matplotlib as _mpl
        import matplotlib.cm as mcm
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError as e:
        raise ImportError(
            "animate_epochs_video requires matplotlib: pip install 'parx[animate]'"
        ) from e

    if not partitions:
        return
    if any(p.input_dim != 2 for p in partitions):
        raise ValueError(
            "animate_epochs_video requires all partitions to have input_dim == 2"
        )

    import os
    path = os.fspath(path)
    labels = epoch_labels or [str(i) for i in range(len(partitions))]
    if len(labels) != len(partitions):
        raise ValueError("epoch_labels length must match number of partitions")

    if x_range is None or y_range is None:
        auto_x, auto_y = _global_range(partitions, pad)
        x_range = x_range or auto_x
        y_range = y_range or auto_y

    mpl_cmap = _mpl.colormaps[colorscale.lower()]

    if color_by is not None:
        m_min, m_max = _global_metric_range(partitions, color_by, log_color)
        norm = mcolors.Normalize(vmin=m_min, vmax=m_max)
        label_str = color_label or getattr(color_by, "__name__", "metric")
    else:
        m_min = m_max = None
        norm = None
        label_str = None

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

    if norm is not None:
        sm = mcm.ScalarMappable(cmap=mpl_cmap, norm=norm)
        sm.set_array([])
        cb_label = label_str + (" (log10)" if log_color else "")
        fig.colorbar(sm, ax=ax, label=cb_label, fraction=0.046, pad=0.04)

    def _draw_frame(frame_idx: int):
        for patch in list(ax.patches):
            patch.remove()
        partition = partitions[frame_idx]
        ax.set_title(f"Epoch {labels[frame_idx]}  ({len(partition)} regions)")

        if color_by is not None:
            raw = np.array(
                [color_by(partition, r) for r in partition.regions], dtype=float
            )
            scaled = np.log10(np.maximum(raw, 1e-12)) if log_color else raw
        else:
            raw = scaled = None

        for i, region in enumerate(partition.regions):
            D, g = partition.halfspaces(region)
            verts = _region_vertices_2d(D, g, x_range, y_range)
            if verts is None:
                continue

            if norm is not None and scaled is not None:
                color = mpl_cmap(norm(scaled[i]))
            else:
                n = len(partition.regions)
                color = mpl_cmap(i / max(n - 1, 1))

            patch = mpatches.Polygon(
                verts,
                closed=True,
                facecolor=color,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.75,
            )
            ax.add_patch(patch)

    _draw_frame(0)

    anim = FuncAnimation(
        fig,
        _draw_frame,
        frames=len(partitions),
        interval=1000 // fps,
        blit=False,
    )

    suffix = os.path.splitext(path)[-1].lower()
    if suffix == ".gif":
        anim.save(path, writer="pillow", fps=fps, dpi=dpi)
    elif suffix == ".mp4":
        anim.save(path, writer="ffmpeg", fps=fps, dpi=dpi)
    else:
        anim.save(path, fps=fps, dpi=dpi)

    plt.close(fig)


# ── Feature embedding ──────────────────────────────────────────────────────────


def _compute_embedding(features: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """Reduce ``features`` (N, D) to (N, 2) via t-SNE or UMAP."""
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as e:
            raise ImportError(
                "plot_feature_embedding requires scikit-learn: pip install scikit-learn"
            ) from e
        return TSNE(n_components=2, **kwargs).fit_transform(features)
    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("umap requires umap-learn: pip install 'parx[embed]'")
        return umap.UMAP(n_components=2, **kwargs).fit_transform(features)
    else:
        raise ValueError(
            f"unknown embedding method {method!r}; choose 'tsne' or 'umap'"
        )


def plot_feature_embedding(
    features: np.ndarray,
    partition: Partition,
    X: np.ndarray,
    *,
    method: str = "tsne",
    color_by="region",
    title: str | None = None,
    **embed_kwargs,
) -> go.Figure:
    """Scatter the 2D embedding of penultimate-layer features, coloured by region.

    Parameters
    ----------
    features : np.ndarray, shape (N, D)
        Pre-extracted penultimate-layer activations.
    partition : Partition
        Used to route ``X`` to linear regions via ``partition.route(X)``.
    X : np.ndarray, shape (N, input_dim)
        Original input points corresponding to each row of ``features``.
    method : 'tsne' or 'umap'
        Dimensionality reduction method forwarded to :func:`_compute_embedding`.
    color_by : 'region' or np.ndarray shape (N,)
        ``'region'`` colours discretely by region index (Turbo palette); an
        array colours continuously by scalar value (Viridis + colorbar).
    title : str or None
        Figure title; defaults to embedding method and region count.
    **embed_kwargs
        Forwarded to :func:`_compute_embedding`.
    """
    emb = _compute_embedding(np.asarray(features, dtype=float), method, **embed_kwargs)

    routed = partition.route(X)
    region_map = {
        tuple(q.tobytes() for q in r.activation_path): i
        for i, r in enumerate(partition.regions)
    }
    region_ids = np.array(
        [
            region_map.get(tuple(q.tobytes() for q in r.activation_path), -1)
            if r is not None
            else -1
            for r in routed
        ],
        dtype=int,
    )

    fig = go.Figure()

    is_region_palette = isinstance(color_by, list) or (
        isinstance(color_by, str) and color_by == "region"
    )
    if is_region_palette:
        # Per-region discrete palette: either pre-computed list[str] or Turbo by index.
        n_regions = len(partition.regions)
        routed_mask = region_ids >= 0
        unrouted_mask = ~routed_mask
        routed_indices = np.where(routed_mask)[0]

        if isinstance(color_by, list):
            palette = color_by  # N region colors, index by region_id
            color_list = [palette[region_ids[i]] for i in routed_indices]
        else:
            color_list = px.colors.sample_colorscale(
                "Turbo",
                [region_ids[i] / max(n_regions - 1, 1) for i in routed_indices],
            )
        hover_strings = [
            f"Region {rid}<br>{_activation_label(r)}" if r is not None else "outside"
            for rid, r in zip(region_ids, routed)
        ]

        if routed_mask.any():
            fig.add_trace(
                go.Scattergl(
                    x=emb[routed_mask, 0],
                    y=emb[routed_mask, 1],
                    mode="markers",
                    marker=dict(color=color_list, size=4, opacity=0.7),
                    text=[hover_strings[i] for i in routed_indices],
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                    name="routed",
                )
            )

        if unrouted_mask.any():
            fig.add_trace(
                go.Scattergl(
                    x=emb[unrouted_mask, 0],
                    y=emb[unrouted_mask, 1],
                    mode="markers",
                    marker=dict(color="lightgrey", size=5, symbol="x", opacity=0.7),
                    text=["outside known regions"] * int(unrouted_mask.sum()),
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                    name="unrouted",
                )
            )
    else:
        color_vals = np.asarray(color_by, dtype=float)
        fig.add_trace(
            go.Scattergl(
                x=emb[:, 0],
                y=emb[:, 1],
                mode="markers",
                marker=dict(
                    color=color_vals,
                    colorscale="Viridis",
                    showscale=True,
                    size=4,
                    opacity=0.7,
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        xaxis_title=f"{method.upper()} 1",
        yaxis_title=f"{method.upper()} 2",
        title=title
        or f"Feature embedding ({method.upper()}, {len(partition)} regions)",
        width=600,
        height=520,
    )
    return fig
