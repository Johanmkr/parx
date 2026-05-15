import pytest


@pytest.fixture(scope="session", autouse=True)
def julia_session():
    """Initialize Julia once for the entire test session."""
    from polarx._julia_init import ensure_julia
    ensure_julia()