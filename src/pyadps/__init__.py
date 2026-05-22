# ============================================================================
# FIXED __init__.py FILES FOR pyadps
# Copy these exactly to your project
# ============================================================================

# FILE 1: src/pyadps/__init__.py
# ============================================================================

"""
pyadps - Python tools for reading and processing RDI ADCP data.

Version 1.0.0 - Major architectural update to xarray-based processing.

Main API:
    read(filename): Read ADCP binary file and return xarray.Dataset
    process(dataset): Get a ProcessedDataset for the six-step pipeline

Submodules:
    io: Low-level file reading (binary_reader, pd0_parser)
    pipeline: Six-step processing pipeline
    config: Configuration management
    legacy: v0.4.0 code (deprecated)
    pages: Streamlit UI components

Example:
    >>> import pyadps
    >>> ds = pyadps.read('file.000')
    >>> ds_proc = pyadps.process(ds)
"""

# Import submodules
from . import io
from . import pages
from . import processing

# Main entry point functions
from .io import (
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

from .processing import ProcessedDataset, ProcessingConfig


def process(ds):
    """Create ProcessedDataset from raw xarray Dataset."""
    return ProcessedDataset(ds)


def load_example(name: str = "demo"):
    """
    Load an example ADCP dataset for tutorials and testing.

    Parameters
    ----------
    name : str, optional
        Name of the example dataset. Default is "demo".
        Available examples:
        - "demo" : Small sample deployment (~100 ensembles, 30 cells)

    Returns
    -------
    xarray.Dataset
        ADCP dataset ready for processing

    Examples
    --------
    >>> import pyadps
    >>> ds = pyadps.load_example("demo")
    >>> print(ds.sizes)
    Frozen({'beam': 4, 'cell': 30, 'time': 100})

    Notes
    -----
    Example files are included with the pyadps package for tutorials
    and testing purposes. They are small subsets of real deployments.
    """
    from importlib.resources import files

    # Map example names to filenames
    examples = {
        "demo": "demo.000",
    }

    if name not in examples:
        available = ", ".join(f"'{k}'" for k in examples.keys())
        raise ValueError(f"Unknown example '{name}'. Available examples: {available}")

    # Get path to the example file in io/metadata/
    example_file = files("pyadps.io.metadata").joinpath(examples[name])

    # Read and return the dataset
    # Using as_file() to handle both installed packages and editable installs
    from contextlib import ExitStack
    from importlib.resources import as_file

    file_manager = ExitStack()
    # Note: For binary files that need a real path, we need to extract to temp
    path = file_manager.enter_context(as_file(example_file))

    # Read the file using pyadps.read()
    return read(str(path))


__version__ = "1.0.0"

__all__ = [
    "read",
    "read_header",
    "read_fixed_leader",
    "read_variable_leader",
    "read_velocity",
    "read_correlation",
    "read_echo_intensity",
    "read_percent_good",
    "read_status",
    "io",
    "legacy",
    "pages",
    "ProcessedDataset",
    "ProcessingConfig",
]
