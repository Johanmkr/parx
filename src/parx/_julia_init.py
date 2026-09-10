"""
Handles Julia runtime initialization.
Import this module once; subsequent calls to ensure_julia() are no-ops.
"""

import json
import os
import shutil
from pathlib import Path

_julia_initialized = False
_jl = None


def _resolve_juliaup_shim() -> str | None:
    """Work around a juliapkg bug when Julia was installed via juliaup.

    juliaup puts a launcher shim (not a real Julia binary) at
    ``~/.juliaup/bin/julia``; the actual per-version installs live under
    ``~/.julia/juliaup/julia-<version>+.../``.  juliapkg tries to
    opportunistically upgrade to the newest Julia release on every resolve;
    when that install attempt fails, it silently falls back to the shim path
    itself instead of one of the already-installed real binaries. juliacall
    then derives the system-image path relative to the shim's directory,
    which never has a ``lib/julia/sys.so`` next to it, and the first Julia
    call crashes with "could not load library ... sys.so ... No such file or
    directory".  See CONTRIBUTING.md's Troubleshooting section.

    Returns the absolute path to the real Julia binary juliaup's default
    channel points at, or ``None`` if ``julia`` isn't on PATH, isn't a
    juliaup shim, or the real binary can't be resolved for any reason — in
    which case juliapkg's normal resolution is left alone.
    """
    julia_on_path = shutil.which("julia")
    if julia_on_path is None:
        return None

    # juliaup's shim is a symlink/copy of a binary literally named
    # "julialauncher"; a plain (non-juliaup) Julia install is not.
    real_target = os.path.realpath(julia_on_path)
    if os.path.basename(real_target) not in ("julialauncher", "julialauncher.exe"):
        return None

    # juliaup does not follow JULIA_DEPOT_PATH, but defines its own
    # override for ~/.julia (matches juliapkg's own lookup).
    depot = os.environ.get("JULIAUP_DEPOT_PATH") or os.path.join(
        os.path.expanduser("~"), ".julia"
    )
    judir = os.path.join(depot, "juliaup")
    try:
        with open(os.path.join(judir, "juliaup.json")) as f:
            meta = json.load(f)
        version = meta["InstalledChannels"][meta["Default"]]["Version"]
        info = meta["InstalledVersions"][version]
        if "BinaryPath" in info:
            exe = os.path.join(judir, info["BinaryPath"])
        else:
            ext = ".exe" if os.name == "nt" else ""
            exe = os.path.join(judir, info["Path"], "bin", "julia" + ext)
        exe = os.path.abspath(exe)
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None

    return exe if os.path.isfile(exe) else None


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
    # Let Julia own signal handling so its GC threads don't conflict with
    # Python's signal machinery.  Must be set before juliacall is imported.
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

    # Work around the juliaup-shim juliapkg bug (see _resolve_juliaup_shim's
    # docstring).  setdefault so an explicit user override always wins.
    juliaup_exe = _resolve_juliaup_shim()
    if juliaup_exe is not None:
        os.environ.setdefault("PYTHON_JULIAPKG_EXE", juliaup_exe)

    from juliacall import Main as jl

    julia_file = Path(__file__).parent / "julia" / "src" / "LinearRegions.jl"
    jl.seval(f'include("{julia_file.as_posix()}")')

    _jl = jl
    _julia_initialized = True

    return _jl
