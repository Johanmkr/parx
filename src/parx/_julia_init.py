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

    Sets JULIA_PROJECT to the embedded julia/ environment before the
    juliacall import so the correct packages are available. Safe to call
    multiple times — only runs once per process.
    """
    global _julia_initialized, _jl

    if _julia_initialized:
        return _jl

    julia_project = Path(__file__).parent / "julia"

    os.environ.setdefault("JULIA_NUM_THREADS", "auto")
    os.environ["JULIA_PROJECT"] = str(julia_project)

    from juliacall import Main as jl

    # Instantiate in case Manifest is missing (e.g. fresh clone without
    # running `julia --project=. -e "using Pkg; Pkg.instantiate()"`)
    jl.seval("using Pkg; Pkg.instantiate()")
    jl.seval("using LinearRegions")

    _jl = jl
    _julia_initialized = True

    return _jl