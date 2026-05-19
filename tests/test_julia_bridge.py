"""Smoke tests — verify the Julia bridge is alive and the data bridge works."""
import numpy as np
import pytest

from parx._julia_init import ensure_julia


@pytest.fixture(scope="module", autouse=True)
def _julia(julia_session):
    """Ensure Julia is ready for every test in this module."""


def test_julia_loads():
    jl = ensure_julia()
    assert jl is not None


def test_julia_threads():
    jl = ensure_julia()
    n = jl.seval("Threads.nthreads()")
    assert n >= 1


def test_hello():
    jl = ensure_julia()
    msg = jl.LinearRegions.hello()
    assert "LinearRegions.jl" in str(msg)


def test_network_info():
    """Phase 1: verify numpy weight arrays survive the Python→Julia round-trip."""
    jl = ensure_julia()
    W1 = np.random.randn(4, 2)
    b1 = np.random.randn(4)
    W2 = np.random.randn(3, 4)
    b2 = np.random.randn(3)

    input_dim, n_layers = jl.LinearRegions.network_info([W1, W2], [b1, b2])
    assert int(input_dim) == 2
    assert int(n_layers) == 2
