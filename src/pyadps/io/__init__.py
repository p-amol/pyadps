# FILE 2: src/pyadps/io/__init__.py
# ============================================================================

"""
Input/Output module for pyadps.

This module handles reading RDI ADCP binary files and converting them to xarray datasets.

Submodules:
    pd0_parser: Low-level binary file format parsing
    binary_reader: High-level xarray.Dataset conversion
    accessors: xarray accessor classes for domain-specific methods
"""

# Import submodules (makes: import pyadps.io.submodule work)
from . import pd0_parser
from . import binary_reader
from . import accessors

# Import commonly used functions (for convenience: from pyadps.io import read)
from .binary_reader import (
    read,
    read_header,
    read_fixed_leader,
    read_variable_leader,
    read_velocity,
    read_correlation,
    read_echo_intensity,
    read_percent_good,
    read_status,
)


# Define what gets imported with "from pyadps.io import *"
__all__ = [
    # Submodules (for: import pyadps.io.submodule)
    "pd0_parser",
    "binary_reader",
    "accessors",
    # Main functions (for: from pyadps.io import function)
    "read",
    "read_header",
    "read_fixed_leader",
    "read_variable_leader",
    "read_velocity",
    "read_correlation",
    "read_echo_intensity",
    "read_percent_good",
    "read_status",
]
