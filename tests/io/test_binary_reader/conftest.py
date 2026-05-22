"""
Shared pytest fixtures for test_filereader test suite.

This module provides fixtures used across all test files in test_filereader,
following patterns established in test_pyreadrdi.
"""

import pytest
import struct
from pathlib import Path
import sys


from .fixtures.ensemble_builder import (
    build_ensemble,
    EnsembleConfig,
    FixedLeaderData,
    VariableLeaderData,
)

import pyadps.io.accessors

# Add parent project directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    """
    Suppress the NumPy reload warning that arises from sys.path manipulation
    above.  Inserting the parent directory at the front of sys.path can cause
    Python to locate numpy (or a package that imports it) via a different path
    entry than the one used during the initial import, which makes Python emit
    a UserWarning about numpy being "imported a second time".  The warning is
    a false positive in this test context — it does not indicate an actual
    problem — so we silence it here rather than in every individual test file.
    """
    config.addinivalue_line(
        "filterwarnings",
        "ignore:The NumPy module was reloaded:UserWarning",
    )


# Import ensemble builder from parent project

# Note: accessors module would need to be imported if available
# import accessors  # IMPORTANT: Register accessor


# ============================================================================
# FIXTURES: Basic Test Files
# ============================================================================


@pytest.fixture(scope="function")
def valid_rdi_ensemble():
    """
    Generate a valid complete RDI ensemble with default configuration.

    Uses the shared ensemble_builder from parent project.
    Default: 7 datatypes, 4 beams, 30 cells, valid checksums

    Returns:
        bytes: Complete binary RDI ensemble
    """
    return build_ensemble()


@pytest.fixture
def valid_rdi_file(tmp_path, valid_rdi_ensemble):
    """
    Create a temporary file with single valid ensemble.

    Args:
        tmp_path: pytest built-in temp directory fixture
        valid_rdi_ensemble: Generated ensemble bytes

    Returns:
        Path: Pathlib.Path to temp RDI file
    """
    rdi_file = tmp_path / "test_single_ensemble.000"
    rdi_file.write_bytes(valid_rdi_ensemble)
    return rdi_file


@pytest.fixture
def multi_ensemble_rdi_file(tmp_path, valid_rdi_ensemble):
    """
    Create a temporary file with multiple valid ensembles (3 copies).

    Ensures test data spans multiple ensembles for consistency testing.

    Args:
        tmp_path: pytest built-in temp directory fixture
        valid_rdi_ensemble: Generated ensemble bytes

    Returns:
        Path: Pathlib.Path to temp RDI file with 3 ensembles
    """
    rdi_file = tmp_path / "test_multi_ensemble.000"
    rdi_file.write_bytes(valid_rdi_ensemble * 3)
    return rdi_file


# ============================================================================
# FIXTURES: Edge Cases and Error Conditions
# ============================================================================


@pytest.fixture
def truncated_rdi_file(tmp_path, valid_rdi_ensemble):
    """
    Create a file truncated mid-ensemble (incomplete data).

    Simulates corrupted/incomplete file scenarios.

    Args:
        tmp_path: pytest built-in temp directory fixture
        valid_rdi_ensemble: Generated ensemble bytes

    Returns:
        Path: Pathlib.Path to truncated RDI file (first 50 bytes only)
    """
    rdi_file = tmp_path / "test_truncated.000"
    # Write only first 50 bytes (incomplete)
    rdi_file.write_bytes(valid_rdi_ensemble[:50])
    return rdi_file


@pytest.fixture
def empty_file(tmp_path):
    """
    Create an empty temporary file.

    Tests handling of empty files (should raise error or return error status).

    Args:
        tmp_path: pytest built-in temp directory fixture

    Returns:
        Path: Pathlib.Path to empty file
    """
    rdi_file = tmp_path / "test_empty.000"
    rdi_file.write_bytes(b"")
    return rdi_file


@pytest.fixture
def invalid_header_id_file(tmp_path):
    """
    Create a file with invalid header ID (not 0x7F7F).

    Tests file format validation in read_header().

    Args:
        tmp_path: pytest built-in temp directory fixture

    Returns:
        Path: Pathlib.Path to file with invalid header ID
    """
    rdi_file = tmp_path / "test_invalid_header.000"

    # Build invalid header (wrong first byte)
    invalid_header = struct.pack(
        "<BBHBB", 0x6F, 0x7F, 100, 0, 5
    )  # 0x6F instead of 0x7F
    invalid_header += struct.pack("<HHHHH", 0, 1, 256, 512, 1024)
    invalid_header += b"\x00" * (100 - 10)

    rdi_file.write_bytes(invalid_header)
    return rdi_file


# ============================================================================
# FIXTURES: Custom Configurations
# ============================================================================


@pytest.fixture
def multi_beam_ensemble():
    """
    Create ensemble with 5 beams instead of default 4.

    Tests handling of variable beam counts.

    Returns:
        bytes: RDI ensemble with 5-beam configuration
    """
    config = EnsembleConfig(beams=5, cells=30, num_datatypes=7)
    return build_ensemble(config=config)


@pytest.fixture
def multi_beam_rdi_file(tmp_path, multi_beam_ensemble):
    """Create temp file with 5-beam ensemble."""
    rdi_file = tmp_path / "test_multi_beam.000"
    rdi_file.write_bytes(multi_beam_ensemble)
    return rdi_file


@pytest.fixture
def multi_cell_ensemble():
    """
    Create ensemble with 50 cells instead of default 30.

    Tests handling of variable cell counts.

    Returns:
        bytes: RDI ensemble with 50-cell configuration
    """
    config = EnsembleConfig(beams=4, cells=50, num_datatypes=7)
    return build_ensemble(config=config)


@pytest.fixture
def multi_cell_rdi_file(tmp_path, multi_cell_ensemble):
    """Create temp file with 50-cell ensemble."""
    rdi_file = tmp_path / "test_multi_cell.000"
    rdi_file.write_bytes(multi_cell_ensemble)
    return rdi_file


@pytest.fixture
def custom_fixed_leader_ensemble():
    """
    Create ensemble with custom Fixed Leader values.

    Tests specific field handling in fixed leader.

    Returns:
        bytes: RDI ensemble with custom fixed leader data
    """
    fl_data = FixedLeaderData(
        num_beams=5,
        num_cells=40,
        cpu_fw_ver=17,
        cpu_fw_rev=10,
    )
    return build_ensemble(fixed_leader_data=fl_data)


@pytest.fixture
def custom_variable_leader_ensemble():
    """
    Create ensemble with custom Variable Leader values.

    Tests specific timestamp and sensor fields.

    Returns:
        bytes: RDI ensemble with custom variable leader data
    """
    vl_data = VariableLeaderData(
        ensemble_number=42,
        year=24,
        month=12,
        day=25,
        hour=15,
        minute=30,
        second=45,
        heading=18000,  # 180 degrees in 1/100ths
        pitch=-500,  # -5 degrees
        roll=200,  # +2 degrees
    )
    return build_ensemble(variable_leader_data=vl_data)
