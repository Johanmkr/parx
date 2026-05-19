"""
Handles Julia runtime initialization.
Import this module once; subsequent calls to ensure_julia() are no-ops.
"""

import os
from pathlib import Path

_julia_initialized = False
_jl = None


def ensure_julia():
    """Initialize the Julia runtime and load the LinearRegions module.

    juliacall/juliapkg manages its own Julia project environment, so we load
    our LinearRegions module via include() rather than registering it as a
    package.  Safe to call multiple times — only runs once per process.
    """
    global _julia_initialized, _jl

    if _julia_initialized:
        return _jl

    os.environ.setdefault("JULIA_NUM_THREADS", "auto")

    from juliacall import Main as jl

    julia_file = Path(__file__).parent / "julia" / "LinearRegions.jl"
    jl.seval(f'include("{julia_file.as_posix()}")')

    _jl = jl
    _julia_initialized = True

    return _jl
