"""
polarx — POLyhedral Activation Region Xplorer

Exactly enumerates the linear regions of ReLU neural networks.
"""

from polarx._check import check_julia
from polarx._julia_init import ensure_julia  # noqa: F401

check_julia()

__version__ = "0.1.0"
__all__ = ["__version__"]