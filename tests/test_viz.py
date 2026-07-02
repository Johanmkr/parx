"""Smoke tests for parx.viz — verify figures are returned without error."""
# juliacall must be imported before torch

import numpy as np
import plotly.graph_objects as go
import pytest
import torch
import torch.nn as nn

from parx import compute_partition
from parx.partition import Partition
from parx.region import Region
from parx.viz import (
    animate_epochs,
    animate_epochs_video,
    plot_feature_embedding,
    region_palette,
)


@pytest.fixture(scope="module", autouse=True)
def _julia(julia_session):
    """Ensure Julia is ready for every test in this module."""


@pytest.fixture(scope="module")
def simple_partition():
    """4-region identity network partition, reused across tests."""
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.ReLU(), nn.Linear(2, 1))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))
    X = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    return compute_partition(model, X, method="sparse_julia")


@pytest.fixture(scope="module")
def two_layer_partition():
    """2-layer network where 3 leaf regions share one layer-1 activation
    pattern, so layer=1 strictly coarsens (1 region) vs the leaf level (3)."""
    model = nn.Sequential(
        nn.Linear(2, 2), nn.ReLU(),
        nn.Linear(2, 2), nn.ReLU(),
        nn.Linear(2, 1),
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 1.0], [1.0, -1.0]]))
        model[0].bias.copy_(torch.tensor([10.0, 10.0]))
        model[2].weight.copy_(torch.eye(2))
        model[2].bias.copy_(torch.tensor([-11.0, -11.0]))
    X = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    return compute_partition(model, X, method="sparse_julia")


# ── plot_partition_2d ─────────────────────────────────────────────────────────


def _polygon_traces(fig):
    """Polygon traces only.

    Exclude the hidden marker trace that exposes the colorbar.
    """
    return [t for t in fig.data if t.mode == "lines" and t.fill == "toself"]


def test_plot_partition_2d_returns_figure(simple_partition):
    from parx.viz import plot_partition_2d

    fig = plot_partition_2d(simple_partition)  # auto-range
    assert isinstance(fig, go.Figure)
    polys = _polygon_traces(fig)
    assert len(polys) == len(simple_partition)
    assert all(isinstance(t, go.Scatter) for t in fig.data)


def test_plot_partition_2d_polygons_have_vertices(simple_partition):
    from parx.viz import plot_partition_2d

    fig = plot_partition_2d(simple_partition)
    for trace in _polygon_traces(fig):
        assert len(trace.x) >= 4  # at least triangle + closing point


def test_plot_partition_2d_hover_contains_activation(simple_partition):
    from parx.viz import plot_partition_2d

    fig = plot_partition_2d(simple_partition)
    for trace in _polygon_traces(fig):
        # Every hover template should mention at least one layer activation
        assert "L1:" in trace.hovertemplate


def test_plot_partition_2d_uses_explicit_domain(simple_partition):
    from parx.viz import plot_partition_2d

    domain = ((-1.5, 1.5), (-0.75, 0.75))
    fig = plot_partition_2d(simple_partition, domain=domain)

    assert tuple(fig.layout.xaxis.range) == domain[0]
    assert tuple(fig.layout.yaxis.range) == domain[1]


def test_plot_partition_2d_layer_coarsens_regions(two_layer_partition):
    from parx.viz import plot_partition_2d

    # layer=n_layers (leaf) must match the unfiltered, full-depth partition.
    n_layers = two_layer_partition.n_layers
    fig_leaf = plot_partition_2d(two_layer_partition, layer=n_layers)
    fig_all = plot_partition_2d(two_layer_partition)
    assert len(_polygon_traces(fig_leaf)) == len(_polygon_traces(fig_all))

    # layer=1 must strictly coarsen: all 3 leaf regions share one layer-1 pattern.
    fig_l1 = plot_partition_2d(two_layer_partition, layer=1)
    assert len(_polygon_traces(fig_l1)) < len(_polygon_traces(fig_all))


def test_plot_partition_2d_layer_out_of_range(simple_partition):
    from parx.viz import plot_partition_2d

    with pytest.raises(ValueError, match="layer must be between"):
        plot_partition_2d(simple_partition, layer=0)

    with pytest.raises(ValueError, match="layer must be between"):
        plot_partition_2d(simple_partition, layer=simple_partition.n_layers + 1)


def test_plot_partition_2d_rejects_non_2d():
    from parx.viz import plot_partition_2d

    p = Partition(
        regions=[Region([np.array([True])], np.zeros(3))],
        weights=[np.eye(1, 3)],
        biases=[np.zeros(1)],
    )
    with pytest.raises(ValueError, match="input_dim=2"):
        plot_partition_2d(p)


# ── color_by metric ───────────────────────────────────────────────────────────


def test_color_by_default_adds_colorbar(simple_partition):
    from parx.viz import plot_partition_2d

    fig = plot_partition_2d(simple_partition)
    cbars = [t for t in fig.data if getattr(t.marker, "showscale", False)]
    assert len(cbars) == 1


def test_color_by_none_skips_colorbar(simple_partition):
    from parx.viz import plot_partition_2d

    fig = plot_partition_2d(simple_partition, color_by=None)
    assert len(_polygon_traces(fig)) == len(simple_partition)
    cbars = [t for t in fig.data if getattr(t.marker, "showscale", False)]
    assert cbars == []


def test_color_by_metric_appears_in_hover(simple_partition):
    from parx.viz import affine_spectral, plot_partition_2d

    fig = plot_partition_2d(simple_partition, color_by=affine_spectral)
    for trace in _polygon_traces(fig):
        assert "affine_spectral:" in trace.hovertemplate


def test_color_by_custom_callable(simple_partition):
    from parx.viz import plot_partition_2d

    fig = plot_partition_2d(
        simple_partition,
        color_by=lambda _p, _r: 42.0,
        color_label="constant",
    )
    for trace in _polygon_traces(fig):
        assert "constant: 42" in trace.hovertemplate


def test_color_by_log_color_runs(simple_partition):
    from parx.viz import affine_frobenius, plot_partition_2d

    fig = plot_partition_2d(simple_partition, color_by=affine_frobenius, log_color=True)
    cbars = [t for t in fig.data if getattr(t.marker, "showscale", False)]
    assert cbars and "log10" in cbars[0].marker.colorbar.title.text


# ── plot_region_counts ────────────────────────────────────────────────────────


def test_plot_region_counts_returns_figure(simple_partition):
    from parx.viz import plot_region_counts

    fig = plot_region_counts(simple_partition)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Bar)


def test_plot_region_counts_bar_length(simple_partition):
    from parx.viz import plot_region_counts

    fig = plot_region_counts(simple_partition)
    ys = list(fig.data[0].y)
    # One bar per layer; last bar equals total number of regions
    assert len(ys) == simple_partition.n_layers
    assert ys[-1] == len(simple_partition)


# ── plot_halfspaces ───────────────────────────────────────────────────────────


def test_plot_halfspaces_returns_figure(simple_partition):
    from parx.viz import plot_halfspaces

    region = simple_partition.regions[0]
    fig = plot_halfspaces(simple_partition, region)
    assert isinstance(fig, go.Figure)


def test_plot_halfspaces_rejects_non_2d():
    from parx.viz import plot_halfspaces

    p = Partition(
        regions=[Region([np.array([True])], np.zeros(3))],
        weights=[np.eye(1, 3)],
        biases=[np.zeros(1)],
    )
    with pytest.raises(ValueError, match="input_dim=2"):
        plot_halfspaces(p, p.regions[0])


# ── Higher-dimensional viz (slice, projection, PCA) ───────────────────────────


@pytest.fixture(scope="module")
def partition_3d():
    """One-layer [3→2] network — 4 regions in R^3."""
    W = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b = np.zeros(2)
    regions = [
        Region([np.array([True, True])], centroid=np.array([1.0, 1.0, 0.0])),
        Region([np.array([True, False])], centroid=np.array([1.0, -1.0, 0.0])),
        Region([np.array([False, True])], centroid=np.array([-1.0, 1.0, 0.0])),
        Region([np.array([False, False])], centroid=np.array([-1.0, -1.0, 0.0])),
    ]
    return Partition(regions=regions, weights=[W], biases=[b])


def test_plot_partition_slice_returns_figure(partition_3d):
    from parx.viz import plot_partition_slice

    fig = plot_partition_slice(
        partition_3d,
        free_dims=(0, 1),
        fixed_values={2: 0.0},
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
    )
    assert isinstance(fig, go.Figure)


def test_plot_partition_slice_has_traces(partition_3d):
    from parx.viz import plot_partition_slice

    fig = plot_partition_slice(
        partition_3d,
        free_dims=(0, 1),
        fixed_values={2: 0.0},
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
    )
    polys = [t for t in fig.data if t.mode == "lines" and t.fill == "toself"]
    assert len(polys) >= 1


def test_plot_partition_slice_color_by_none(partition_3d):
    from parx.viz import plot_partition_slice

    fig = plot_partition_slice(
        partition_3d,
        free_dims=(0, 1),
        fixed_values={2: 0.0},
        color_by=None,
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
    )
    assert isinstance(fig, go.Figure)
    cbars = [t for t in fig.data if getattr(t.marker, "showscale", False)]
    assert cbars == []


def test_plot_partition_slice_auto_range(partition_3d):
    """Omitting x_range/y_range should not raise."""
    from parx.viz import plot_partition_slice

    fig = plot_partition_slice(
        partition_3d,
        free_dims=(0, 1),
        fixed_values={2: 0.0},
    )
    assert isinstance(fig, go.Figure)


def test_plot_partition_projection_returns_figure(partition_3d):
    from parx.viz import plot_partition_projection

    proj = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # (3, 2)
    fig = plot_partition_projection(
        partition_3d,
        proj,
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
    )
    assert isinstance(fig, go.Figure)


def test_plot_partition_projection_has_traces(partition_3d):
    from parx.viz import plot_partition_projection

    proj = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    fig = plot_partition_projection(
        partition_3d,
        proj,
        x_range=(-2.0, 2.0),
        y_range=(-2.0, 2.0),
    )
    polys = [t for t in fig.data if t.mode == "lines" and t.fill == "toself"]
    assert len(polys) >= 1


def test_plot_partition_projection_wrong_shape(partition_3d):
    from parx.viz import plot_partition_projection

    bad_proj = np.eye(2)  # (2, 2) — wrong for input_dim=3
    with pytest.raises(ValueError, match="projection must have shape"):
        plot_partition_projection(partition_3d, bad_proj)


def test_plot_partition_pca_returns_figure(partition_3d):
    pytest.importorskip("sklearn")
    from parx.viz import plot_partition_pca

    rng = np.random.default_rng(0)
    data = rng.standard_normal((50, 3))
    fig = plot_partition_pca(partition_3d, data)
    assert isinstance(fig, go.Figure)


def test_plot_partition_pca_missing_sklearn(partition_3d, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _block_sklearn(name, *args, **kwargs):
        if name == "sklearn.decomposition":
            raise ImportError("sklearn not available")
        return real_import(name, *args, **kwargs)

    from parx.viz import plot_partition_pca

    rng = np.random.default_rng(0)
    data = rng.standard_normal((50, 3))

    monkeypatch.setattr(builtins, "__import__", _block_sklearn)
    with pytest.raises(ImportError, match="scikit-learn"):
        plot_partition_pca(partition_3d, data)


# ── Animation tests ───────────────────────────────────────────────────────────


@pytest.fixture
def epoch_partitions(simple_partition):
    return [simple_partition, simple_partition, simple_partition]


class TestAnimateEpochs:
    def test_returns_figure(self, epoch_partitions):
        fig = animate_epochs(epoch_partitions)
        assert isinstance(fig, go.Figure)

    def test_frame_count(self, epoch_partitions):
        fig = animate_epochs(epoch_partitions)
        assert len(fig.frames) == 3

    def test_custom_labels(self, epoch_partitions):
        fig = animate_epochs(epoch_partitions, epoch_labels=["a", "b", "c"])
        assert fig.frames[1].name == "b"

    def test_wrong_label_count(self, epoch_partitions):
        with pytest.raises(ValueError, match="epoch_labels length"):
            animate_epochs(epoch_partitions, epoch_labels=["only_one"])

    def test_wrong_input_dim(self, partition_3d):
        with pytest.raises(ValueError, match="input_dim"):
            animate_epochs([partition_3d])

    def test_empty_list(self):
        fig = animate_epochs([])
        assert isinstance(fig, go.Figure)

    def test_has_slider(self, epoch_partitions):
        fig = animate_epochs(epoch_partitions)
        assert len(fig.layout.sliders) == 1

    def test_has_play_button(self, epoch_partitions):
        fig = animate_epochs(epoch_partitions)
        assert len(fig.layout.updatemenus) == 1


class TestAnimateEpochsVideo:
    def test_gif_output(self, tmp_path, epoch_partitions):
        mpl = pytest.importorskip("matplotlib")  # noqa: F841
        out = tmp_path / "out.gif"
        animate_epochs_video(epoch_partitions, out, fps=2, dpi=72)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_wrong_input_dim(self, tmp_path, partition_3d):
        pytest.importorskip("matplotlib")
        out = tmp_path / "bad.gif"
        with pytest.raises(ValueError, match="input_dim"):
            animate_epochs_video([partition_3d], out)

    def test_wrong_label_count(self, tmp_path, epoch_partitions):
        pytest.importorskip("matplotlib")
        out = tmp_path / "bad.gif"
        with pytest.raises(ValueError, match="epoch_labels length"):
            animate_epochs_video(epoch_partitions, out, epoch_labels=["x"])


# ── region_palette ────────────────────────────────────────────────────────────


class TestRegionPalette:
    @pytest.mark.parametrize("scheme", ["random", "frobenius", "spatial"])
    def test_returns_one_color_per_region(self, simple_partition, scheme):
        colors = region_palette(simple_partition, scheme=scheme)
        assert len(colors) == len(simple_partition.regions)
        assert all(isinstance(c, str) and c.startswith("rgb(") for c in colors)

    def test_unknown_scheme_raises(self, simple_partition):
        with pytest.raises(ValueError, match="unknown scheme"):
            region_palette(simple_partition, scheme="not_a_scheme")

    def test_plot_partition_2d_accepts_precomputed_colors(self, simple_partition):
        from parx.viz import plot_partition_2d

        colors = region_palette(simple_partition, scheme="random")
        fig = plot_partition_2d(simple_partition, colors=colors)
        assert isinstance(fig, go.Figure)
        assert len(_polygon_traces(fig)) == len(simple_partition)


# ── plot_feature_embedding ───────────────────────────────────────────────────


@pytest.fixture
def embedding_inputs(simple_partition):
    """Synthetic penultimate-layer features + inputs routable through
    simple_partition, sized for a valid (low) t-SNE perplexity."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (30, 2))
    features = rng.normal(size=(30, 5))
    return features, simple_partition, X


class TestPlotFeatureEmbedding:
    def test_returns_figure_with_default_region_coloring(self, embedding_inputs):
        pytest.importorskip("sklearn")
        features, partition, X = embedding_inputs
        fig = plot_feature_embedding(features, partition, X, perplexity=5)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_continuous_color_by_array_adds_colorbar(self, embedding_inputs):
        pytest.importorskip("sklearn")
        features, partition, X = embedding_inputs
        values = np.linalg.norm(X, axis=1)
        fig = plot_feature_embedding(
            features, partition, X, color_by=values, perplexity=5
        )
        assert fig.data[0].marker.showscale is True

    def test_umap_method(self, embedding_inputs):
        pytest.importorskip("umap")
        features, partition, X = embedding_inputs
        fig = plot_feature_embedding(features, partition, X, method="umap")
        assert isinstance(fig, go.Figure)

    def test_unknown_method_raises(self, embedding_inputs):
        features, partition, X = embedding_inputs
        with pytest.raises(ValueError, match="unknown embedding method"):
            plot_feature_embedding(features, partition, X, method="pca")
