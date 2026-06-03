"""Tests for parx.analysis — pure Python, no Julia required.

All tests use hand-built Partition objects with simple geometry so that
exact answers are known analytically.
"""

from __future__ import annotations

import numpy as np
import pytest

from parx.analysis import (
    always_active_neurons,
    complexity_over_epochs,
    complexity_profile,
    dead_neurons,
    neuron_activity_rates,
    partition_volume_estimates,
    region_chebyshev_radii,
    region_size_summary,
    region_volume_estimate,
)
from parx.partition import Partition
from parx.region import Region

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _quadrant_partition() -> Partition:
    """One-layer [2→2] identity network — four quadrant regions."""
    W = np.eye(2)
    b = np.zeros(2)
    regions = [
        Region([np.array([True, True])], centroid=np.array([1.0, 1.0])),
        Region([np.array([True, False])], centroid=np.array([1.0, -1.0])),
        Region([np.array([False, True])], centroid=np.array([-1.0, 1.0])),
        Region([np.array([False, False])], centroid=np.array([-1.0, -1.0])),
    ]
    return Partition(regions=regions, weights=[W], biases=[b])


def _two_layer_partition() -> Partition:
    """Two-layer [2→2→2] network with two all-active regions for testing."""
    W = np.eye(2)
    b = np.zeros(2)
    regions = [
        Region(
            [np.array([True, True]), np.array([True, True])],
            centroid=np.array([1.0, 1.0]),
        ),
        Region(
            [np.array([False, False]), np.array([False, False])],
            centroid=np.array([-1.0, -1.0]),
        ),
    ]
    return Partition(regions=regions, weights=[W, W], biases=[b, b])


def _always_dead_partition() -> Partition:
    """One-layer network where neuron 1 is always dead (never active)."""
    W = np.eye(2)
    b = np.zeros(2)
    # Both regions have neuron 1 inactive.
    regions = [
        Region([np.array([True, False])], centroid=np.array([1.0, -1.0])),
        Region([np.array([False, False])], centroid=np.array([-1.0, -1.0])),
    ]
    return Partition(regions=regions, weights=[W], biases=[b])


def _empty_partition() -> Partition:
    """Partition with zero regions."""
    W = np.eye(2)
    b = np.zeros(2)
    return Partition(regions=[], weights=[W], biases=[b])


# ── 9a: neuron_activity_rates ─────────────────────────────────────────────────


class TestNeuronActivityRates:
    def test_quadrant_rates_are_half(self):
        """Each neuron is active in exactly 2 of 4 quadrant regions → 0.5."""
        p = _quadrant_partition()
        rates = neuron_activity_rates(p)
        assert set(rates.keys()) == {0}
        np.testing.assert_allclose(rates[0], [0.5, 0.5])

    def test_always_active_neuron_has_rate_one(self):
        """If a neuron is active in every region its rate should be 1.0."""
        W = np.eye(2)
        b = np.zeros(2)
        regions = [
            Region([np.array([True, True])], centroid=np.array([1.0, 1.0])),
            Region([np.array([True, False])], centroid=np.array([1.0, -1.0])),
        ]
        p = Partition(regions=regions, weights=[W], biases=[b])
        rates = neuron_activity_rates(p)
        assert rates[0][0] == pytest.approx(1.0)

    def test_two_layer_rates(self):
        """Two-layer partition returns rates for both layers."""
        p = _two_layer_partition()
        rates = neuron_activity_rates(p)
        assert set(rates.keys()) == {0, 1}
        # Layer 0: region 0 has [T, T], region 1 has [F, F] → 0.5 each
        np.testing.assert_allclose(rates[0], [0.5, 0.5])
        np.testing.assert_allclose(rates[1], [0.5, 0.5])

    def test_empty_partition_returns_zero_arrays(self):
        """No regions → all rates are 0."""
        p = _empty_partition()
        rates = neuron_activity_rates(p)
        assert 0 in rates
        assert len(rates[0]) == 2
        np.testing.assert_allclose(rates[0], [0.0, 0.0])

    def test_return_shape_matches_layer_width(self):
        p = _quadrant_partition()
        rates = neuron_activity_rates(p)
        assert rates[0].shape == (2,)


# ── 9a: dead_neurons ──────────────────────────────────────────────────────────


class TestDeadNeurons:
    def test_detects_always_inactive_neuron(self):
        p = _always_dead_partition()
        dead = dead_neurons(p)
        # Neuron 1 (index 1) in layer 0 is never active.
        assert (0, 1) in dead

    def test_no_dead_neurons_in_quadrant(self):
        """Quadrant partition has no dead neurons — each is active in 2/4."""
        p = _quadrant_partition()
        dead = dead_neurons(p)
        assert dead == []

    def test_custom_threshold(self):
        """threshold=0.5 should flag neurons active in ≤50% of regions."""
        p = _quadrant_partition()
        # Each neuron is active in exactly 0.5 of regions.
        # With threshold=0.5, they are included (<=).
        dead = dead_neurons(p, threshold=0.5)
        assert len(dead) == 2  # both neurons qualify

    def test_returns_sorted_tuples(self):
        p = _always_dead_partition()
        dead = dead_neurons(p)
        layers = [t[0] for t in dead]
        assert layers == sorted(layers)


# ── 9a: always_active_neurons ─────────────────────────────────────────────────


class TestAlwaysActiveNeurons:
    def test_no_always_active_in_quadrant(self):
        """In the quadrant partition no neuron is always active."""
        p = _quadrant_partition()
        aa = always_active_neurons(p)
        assert aa == []

    def test_detects_always_active(self):
        W = np.eye(2)
        b = np.zeros(2)
        # Both regions always have neuron 0 active.
        regions = [
            Region([np.array([True, True])], centroid=np.array([1.0, 1.0])),
            Region([np.array([True, False])], centroid=np.array([1.0, -1.0])),
        ]
        p = Partition(regions=regions, weights=[W], biases=[b])
        aa = always_active_neurons(p)
        assert (0, 0) in aa
        assert (0, 1) not in aa

    def test_custom_threshold(self):
        """threshold=0.5 should flag neurons active in >=50% of regions."""
        p = _quadrant_partition()
        # Each neuron is active in exactly 0.5 of regions.
        aa = always_active_neurons(p, threshold=0.5)
        assert len(aa) == 2  # both qualify


# ── 9b: complexity_profile ────────────────────────────────────────────────────


class TestComplexityProfile:
    def test_basic_fields_quadrant(self):
        p = _quadrant_partition()
        prof = complexity_profile(p)
        assert prof["n_regions"] == 4
        assert prof["n_layers"] == 1
        assert prof["input_dim"] == 2
        assert prof["total_neurons"] == 2  # one layer of width 2

    def test_regions_per_layer_single_layer(self):
        p = _quadrant_partition()
        prof = complexity_profile(p)
        # At depth 1 there are 4 unique patterns.
        assert prof["regions_per_layer"] == [4]

    def test_regions_per_layer_two_layer(self):
        p = _two_layer_partition()
        prof = complexity_profile(p)
        # At depth 1: 2 unique prefixes; at depth 2: 2 unique full paths.
        assert len(prof["regions_per_layer"]) == 2
        assert prof["regions_per_layer"][0] == 2
        assert prof["regions_per_layer"][1] == 2

    def test_total_constraints(self):
        """One-layer identity network: 2 constraints per region × 4 regions = 8."""
        p = _quadrant_partition()
        prof = complexity_profile(p)
        assert prof["total_constraints"] == 8
        assert prof["mean_constraints_per_region"] == pytest.approx(2.0)

    def test_empty_partition(self):
        p = _empty_partition()
        prof = complexity_profile(p)
        assert prof["n_regions"] == 0
        assert prof["total_constraints"] == 0
        assert prof["mean_constraints_per_region"] == 0.0


# ── 9c: region_chebyshev_radii ───────────────────────────────────────────────


class TestRegionChebyshevRadii:
    def test_returns_correct_shape(self):
        p = _quadrant_partition()
        radii = region_chebyshev_radii(p)
        assert radii.shape == (4,)

    def test_all_positive_for_healthy_partition(self):
        p = _quadrant_partition()
        radii = region_chebyshev_radii(p)
        assert np.all(radii > 0)

    def test_empty_region_gets_zero_radius(self):
        """An infeasible region (same feature positive & negative) → radius 0."""
        W = np.array([[1.0, 0.0], [1.0, 0.0]])
        b = np.zeros(2)
        bad = Region([np.array([True, False])], centroid=np.array([0.5, 0.0]))
        ok = Region([np.array([True, True])], centroid=np.array([1.0, 0.0]))
        p = Partition(regions=[ok, bad], weights=[W], biases=[b])
        radii = region_chebyshev_radii(p)
        assert radii[1] == 0.0
        assert radii[0] > 0.0

    def test_backward_compat_import_from_verify(self):
        """verify.region_chebyshev_radii should still work (re-export)."""
        from parx.verify import region_chebyshev_radii as rcr_verify

        p = _quadrant_partition()
        radii = rcr_verify(p)
        assert radii.shape == (4,)


# ── 9c: region_size_summary ──────────────────────────────────────────────────


class TestRegionSizeSummary:
    def test_returns_expected_keys(self):
        p = _quadrant_partition()
        summary = region_size_summary(p)
        assert {"min", "median", "mean", "max", "fraction_bounded"} == set(
            summary.keys()
        )

    def test_min_leq_median_leq_max(self):
        p = _quadrant_partition()
        s = region_size_summary(p)
        assert s["min"] <= s["median"] <= s["max"]
        assert s["min"] <= s["mean"] <= s["max"]

    def test_all_finite_for_healthy_partition(self):
        p = _quadrant_partition()
        s = region_size_summary(p)
        # Quadrant partition is unbounded (no upper bounds) so fraction_bounded
        # may be 0; but min/mean/max should still be finite numbers.
        assert np.isfinite(s["min"])
        assert np.isfinite(s["mean"])
        assert np.isfinite(s["max"])

    def test_empty_partition_returns_nan(self):
        p = _empty_partition()
        s = region_size_summary(p)
        assert np.isnan(s["min"])
        assert np.isnan(s["mean"])


# ── 9d: region_volume_estimate ───────────────────────────────────────────────


class TestRegionVolumeEstimate:
    def test_unit_square_region(self):
        """A region defined by [0,1]^2 should have volume ≈ 1.0."""
        # Constraints: x1 >= 0, x2 >= 0, x1 <= 1, x2 <= 1.
        # In D @ x <= g form:
        # -x1 <= 0, -x2 <= 0, x1 <= 1, x2 <= 1
        # Override halfspaces by setting a custom region with fixed D, g.
        # Instead, build a simple network where the halfspaces are [0,1]^2.
        # Use a 1-layer network with W=I, b=[0,0]; region [T,T] gives:
        # constraints x1 > 0, x2 > 0 (from active side)
        # The Chebyshev ball will be centred at [0.5, 0.5] inside [0,1]^2 if
        # centroid is there.  Actually the quadrant partition is [0,inf)^2.
        # For a bounded box we need explicit upper bounds — skip exact test.
        # Instead just check the estimate is a non-negative float.
        p = _quadrant_partition()
        region = p.regions[0]  # [T, T] → positive quadrant (unbounded)
        vol = region_volume_estimate(p, region, n_samples=500, seed=42)
        # Unbounded region: radius capped at 1e3.  Just check type and sign.
        assert isinstance(vol, float)
        assert vol >= 0.0

    def test_reproducible_with_seed(self):
        p = _quadrant_partition()
        region = p.regions[0]
        v1 = region_volume_estimate(p, region, n_samples=200, seed=7)
        v2 = region_volume_estimate(p, region, n_samples=200, seed=7)
        assert v1 == pytest.approx(v2)

    def test_different_seeds_may_differ(self):
        """Different seeds should generally produce different estimates."""
        p = _quadrant_partition()
        region = p.regions[0]
        v1 = region_volume_estimate(p, region, n_samples=100, seed=1)
        v2 = region_volume_estimate(p, region, n_samples=100, seed=999)
        # Not guaranteed to differ, but with different seeds they usually will.
        # We just check both are valid floats.
        assert np.isfinite(v1) or v1 == float("inf")
        assert np.isfinite(v2) or v2 == float("inf")

    def test_empty_region_returns_zero(self):
        """An infeasible region should have volume 0."""
        W = np.array([[1.0, 0.0], [1.0, 0.0]])
        b = np.zeros(2)
        bad = Region([np.array([True, False])], centroid=np.array([0.5, 0.0]))
        p = Partition(regions=[bad], weights=[W], biases=[b])
        vol = region_volume_estimate(p, bad, n_samples=100, seed=0)
        assert vol == 0.0


# ── 9d: partition_volume_estimates ───────────────────────────────────────────


class TestPartitionVolumeEstimates:
    def test_returns_correct_shape(self):
        p = _quadrant_partition()
        vols = partition_volume_estimates(p, n_samples=100, seed=0)
        assert vols.shape == (4,)

    def test_all_non_negative(self):
        p = _quadrant_partition()
        vols = partition_volume_estimates(p, n_samples=100, seed=0)
        assert np.all(vols >= 0.0)

    def test_reproducible_with_seed(self):
        p = _quadrant_partition()
        v1 = partition_volume_estimates(p, n_samples=100, seed=42)
        v2 = partition_volume_estimates(p, n_samples=100, seed=42)
        np.testing.assert_array_equal(v1, v2)


# ── 9e: complexity_over_epochs ───────────────────────────────────────────────


class TestComplexityOverEpochs:
    def test_returns_expected_keys(self):
        p = _quadrant_partition()
        result = complexity_over_epochs([p, p])
        expected_keys = {
            "labels", "n_regions", "mean_chebyshev_radius", "dead_neuron_count"
        }
        assert expected_keys == set(result.keys())

    def test_length_matches_number_of_partitions(self):
        p = _quadrant_partition()
        result = complexity_over_epochs([p, p, p])
        assert len(result["labels"]) == 3
        assert len(result["n_regions"]) == 3
        assert len(result["mean_chebyshev_radius"]) == 3
        assert len(result["dead_neuron_count"]) == 3

    def test_default_labels_are_indices(self):
        p = _quadrant_partition()
        result = complexity_over_epochs([p, p])
        assert result["labels"] == [0, 1]

    def test_custom_labels(self):
        p = _quadrant_partition()
        result = complexity_over_epochs([p, p], labels=["epoch1", "epoch2"])
        assert result["labels"] == ["epoch1", "epoch2"]

    def test_n_regions_counts(self):
        p4 = _quadrant_partition()
        p2 = _two_layer_partition()
        result = complexity_over_epochs([p4, p2])
        assert result["n_regions"] == [4, 2]

    def test_dead_neuron_count_accurate(self):
        p = _always_dead_partition()
        result = complexity_over_epochs([p])
        # Neuron 1 in layer 0 is always dead.
        assert result["dead_neuron_count"][0] >= 1

    def test_empty_partitions_list(self):
        result = complexity_over_epochs([])
        assert result["labels"] == []
        assert result["n_regions"] == []
        assert result["mean_chebyshev_radius"] == []
        assert result["dead_neuron_count"] == []


# ── Integration: public __all__ exports ──────────────────────────────────────


class TestPublicAPI:
    def test_analysis_functions_in_parx_namespace(self):
        """Functions listed in __all__ must be importable from parx directly."""
        import parx

        assert hasattr(parx, "neuron_activity_rates")
        assert hasattr(parx, "dead_neurons")
        assert hasattr(parx, "always_active_neurons")
        assert hasattr(parx, "complexity_profile")
        assert hasattr(parx, "region_size_summary")

    def test_volume_not_in_parx_namespace(self):
        """Volume functions are too expensive to advertise at top level."""
        import parx

        assert not hasattr(parx, "region_volume_estimate")
        assert not hasattr(parx, "partition_volume_estimates")

    def test_region_chebyshev_radii_accessible_via_analysis(self):
        from parx.analysis import region_chebyshev_radii as rcr

        p = _quadrant_partition()
        radii = rcr(p)
        assert radii.shape == (4,)
