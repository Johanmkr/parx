"""Tests for parx.diagnostics — thread_info() and benchmark_method()."""

import numpy as np
import pytest

from parx.diagnostics import benchmark_method, thread_info


@pytest.fixture(scope="module", autouse=True)
def _julia(julia_session):
    """Ensure Julia is ready for every test in this module."""


def test_thread_info_returns_valid_dict():
    info = thread_info()
    assert info["julia_threads"] >= 1
    assert info["julia_max_threads"] >= 1
    assert info["env_JULIA_NUM_THREADS"] is None or isinstance(
        info["env_JULIA_NUM_THREADS"], str
    )


def test_benchmark_method_returns_timings():
    weights = {
        "0.weight": np.eye(2),
        "0.bias": np.zeros(2),
        "2.weight": np.eye(2),
        "2.bias": np.zeros(2),
    }
    X = np.array([[1.0, 1.0], [-1.0, -1.0]])

    result = benchmark_method("sparse_julia", weights, X, repeats=2, warmup=1)

    assert result["method"] == "sparse_julia"
    assert result["n_regions"] > 0
    assert result["best_seconds"] >= 0.0
    assert len(result["all_seconds"]) == 2
    assert result["best_seconds"] == min(result["all_seconds"])
