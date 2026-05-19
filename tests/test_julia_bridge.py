"""Smoke tests — verify the Julia bridge is alive."""
from parx._julia_init import ensure_julia


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