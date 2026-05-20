"""Smoke tests for parx.viz — verify figures are returned without error."""
# juliacall must be imported before torch
from parx._julia_init import ensure_julia

import numpy as np
import pytest
import torch
import torch.nn as nn
import plotly.graph_objects as go

from parx import compute_partition
from parx.partition import Partition
from parx.region import Region


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
    return compute_partition(model, X, mode="sparse")


# ── plot_partition_2d ─────────────────────────────────────────────────────────

def test_plot_partition_2d_returns_figure(simple_partition):
    from parx.viz import plot_partition_2d
    fig = plot_partition_2d(simple_partition)   # auto-range
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == len(simple_partition)
    assert all(isinstance(t, go.Scatter) for t in fig.data)


def test_plot_partition_2d_polygons_have_vertices(simple_partition):
    from parx.viz import plot_partition_2d
    fig = plot_partition_2d(simple_partition)
    for trace in fig.data:
        assert len(trace.x) >= 4  # at least triangle + closing point


def test_plot_partition_2d_hover_contains_activation(simple_partition):
    from parx.viz import plot_partition_2d
    fig = plot_partition_2d(simple_partition)
    for trace in fig.data:
        # Every hover template should mention at least one layer activation
        assert "L1:" in trace.hovertemplate


def test_plot_partition_2d_uses_explicit_domain(simple_partition):
    from parx.viz import plot_partition_2d

    domain = ((-1.5, 1.5), (-0.75, 0.75))
    fig = plot_partition_2d(simple_partition, domain=domain)

    assert tuple(fig.layout.xaxis.range) == domain[0]
    assert tuple(fig.layout.yaxis.range) == domain[1]


def test_plot_partition_2d_layer_coarsens_regions(simple_partition):
    from parx.viz import plot_partition_2d

    # Identity network has 1 ReLU layer → at layer=1 and all-layers count must match
    fig_leaf = plot_partition_2d(simple_partition, layer=simple_partition.n_layers)
    fig_all  = plot_partition_2d(simple_partition)
    assert len(fig_leaf.data) == len(fig_all.data)

    # layer=1 must produce ≤ leaf count (same here, but at most)
    fig_l1 = plot_partition_2d(simple_partition, layer=1)
    assert len(fig_l1.data) <= len(fig_all.data)


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
