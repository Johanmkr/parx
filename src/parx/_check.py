import shutil


def check_julia() -> None:
    """Raise a helpful error if Julia is not found on PATH."""
    if shutil.which("julia") is None:
        raise RuntimeError(
            "\n\nparx requires Julia to be installed and available on PATH.\n"
            "Install Julia via juliaup (recommended):\n\n"
            "  macOS/Linux:  curl -fsSL https://install.julialang.org | sh\n"
            "  Windows:      winget install julia -s msstore\n\n"
            "After installing, restart your terminal and try again.\n"
            "See https://github.com/Johanmkr/parx for more details.\n"
        )
