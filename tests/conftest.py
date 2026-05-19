import pytest


@pytest.fixture(scope="session")
def julia_session():
    """Initialize Julia once for the entire test session.

    Only injected into tests that explicitly request this fixture, so
    pure-Python tests are never blocked by Julia initialization.
    """
    from parx._julia_init import ensure_julia
    ensure_julia()
