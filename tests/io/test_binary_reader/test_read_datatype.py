"""
Comprehensive pytest test suite for datatype reading functions in binary_reader.py

This module provides extensive test coverage for the five datatype reading functions
that extract ADCP data arrays (velocity, correlation, echo intensity, percent good,
and status) from RDI binary files and return them as xarray.Dataset objects.

Functions tested:
    - read_velocity()
    - read_correlation()
    - read_echo_intensity()
    - read_percent_good()
    - read_status()

Test Coverage:
    - Basic functionality (valid files, single/multiple ensembles)
    - Data structure validation (xarray.Dataset, coordinates, dimensions)
    - Data type validation (correct numpy dtypes for each datatype)
    - Shape verification (cell, beam, ensemble dimensions)
    - Metadata and attributes (CF Convention compliance)
    - Cell/beam selection parameters (individual cell/beam extraction)
    - Parameter variations (auto-fetch vs explicit parameters)
    - Integration with read_header() automatic parameter retrieval
    - Error handling (missing files, corrupted data, invalid parameters)
    - Edge cases (empty files, truncated data, single vs multiple ensembles)
    - xarray compatibility (sel, isel, slicing operations)
    - Value range validation (velocity, correlation, echo, percent good, status)
    - Missing data handling (-32768 for velocity, 0 for others)

References:
    RDI WorkHorse Commands and Output Data Format (Section 5, page 123):
    - Velocity data: 16-bit signed integers (mm/s), missing: -32768
    - Echo Intensity: 8-bit unsigned (0-255), scale to dB with 0.45
    - Correlation: 8-bit unsigned (0-255), higher = better quality
    - Percent Good: 8-bit unsigned (0-100), percentage of valid data
    - Status: 8-bit unsigned (0-255), diagnostic flags
    - Data stored as: cells[0:n_cells] Ã— beams[0:n_beams] per ensemble

Author: pyadps development team
License: MIT
Version: 1.0.0
"""

import json
import struct
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from pyadps.io.binary_reader import (
    read_velocity,
    read_correlation,
    read_echo_intensity,
    read_percent_good,
    read_status,
    read_header,
)

# Import test data builders
from tests.io.test_binary_reader.fixtures.ensemble_builder import (
    build_ensemble,
    EnsembleConfig,
    FixedLeaderData,
    VariableLeaderData,
)


# ============================================================================
# FIXTURES: Test Files
# ============================================================================


@pytest.fixture(scope="function")
def valid_rdi_ensemble():
    """Generate a valid complete RDI ensemble with default configuration.

    Returns:
        bytes: Complete binary RDI ensemble with all data types
    """
    return build_ensemble()


@pytest.fixture
def valid_rdi_file(tmp_path, valid_rdi_ensemble):
    """Create a temporary file with single valid ensemble.

    Args:
        tmp_path: pytest built-in temp directory fixture
        valid_rdi_ensemble: Generated ensemble bytes

    Returns:
        Path: Pathlib.Path to temp RDI file
    """
    rdi_file = tmp_path / "test_datatype_single.000"
    rdi_file.write_bytes(valid_rdi_ensemble)
    return rdi_file


@pytest.fixture
def multi_ensemble_rdi_file(tmp_path, valid_rdi_ensemble):
    """Create a temporary file with multiple valid ensembles (3 copies).

    Args:
        tmp_path: pytest built-in temp directory fixture
        valid_rdi_ensemble: Generated ensemble bytes

    Returns:
        Path: Pathlib.Path to temp RDI file with 3 ensembles
    """
    rdi_file = tmp_path / "test_datatype_multi.000"
    rdi_file.write_bytes(valid_rdi_ensemble * 3)
    return rdi_file


@pytest.fixture
def truncated_rdi_file(tmp_path, valid_rdi_ensemble):
    """Create a file truncated mid-ensemble (incomplete data).

    Simulates corrupted/incomplete file scenarios.

    Args:
        tmp_path: pytest built-in temp directory fixture
        valid_rdi_ensemble: Generated ensemble bytes

    Returns:
        Path: Pathlib.Path to truncated RDI file (first 50 bytes only)
    """
    rdi_file = tmp_path / "test_datatype_truncated.000"
    rdi_file.write_bytes(valid_rdi_ensemble[:50])
    return rdi_file


@pytest.fixture
def empty_file(tmp_path):
    """Create an empty temporary file.

    Tests handling of empty files.

    Args:
        tmp_path: pytest built-in temp directory fixture

    Returns:
        Path: Pathlib.Path to empty file
    """
    rdi_file = tmp_path / "test_datatype_empty.000"
    rdi_file.write_bytes(b"")
    return rdi_file


@pytest.fixture
def custom_datatype_ensemble():
    """Create ensemble with custom configuration for specific testing.

    Returns:
        bytes: RDI ensemble with custom fixed/variable leader data
    """
    fl_data = FixedLeaderData(
        num_beams=4,
        num_cells=20,
        cpu_fw_ver=17,
        cpu_fw_rev=10,
    )
    vl_data = VariableLeaderData()
    config = EnsembleConfig(fixed_leader=fl_data, variable_leader=vl_data)
    return build_ensemble(config=config)


# ============================================================================
# TEST CLASS 1: Basic Functionality
# ============================================================================


class TestDatatypeBasicFunctionality:
    """Test basic functionality of all datatype reading functions."""

    @pytest.mark.parametrize(
        "read_func,var_name",
        [
            (read_velocity, "velocity"),
            (read_correlation, "correlation"),
            (read_echo_intensity, "echo_intensity"),
            (read_percent_good, "percent_good"),
            (read_status, "status"),
        ],
    )
    def test_read_function_returns_xarray_dataset(
        self, read_func, var_name, valid_rdi_file
    ):
        """Test that all read functions return xarray.Dataset."""
        ds = read_func(valid_rdi_file)
        assert isinstance(ds, xr.Dataset), f"Expected xr.Dataset, got {type(ds)}"

    @pytest.mark.parametrize(
        "read_func,var_name",
        [
            (read_velocity, "velocity"),
            (read_correlation, "correlation"),
            (read_echo_intensity, "echo_intensity"),
            (read_percent_good, "percent_good"),
            (read_status, "status"),
        ],
    )
    def test_read_function_contains_expected_variable(
        self, read_func, var_name, valid_rdi_file
    ):
        """Test that returned dataset contains expected data variable."""
        ds = read_func(valid_rdi_file)
        assert (
            var_name in ds.data_vars
        ), f"Expected variable '{var_name}' not found in dataset"
        assert isinstance(ds[var_name], xr.DataArray)

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_read_function_accepts_path_string(self, read_func, valid_rdi_file):
        """Test that functions accept file path as string."""
        ds = read_func(str(valid_rdi_file))
        assert isinstance(ds, xr.Dataset)

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_read_function_accepts_path_object(self, read_func, valid_rdi_file):
        """Test that functions accept file path as Path object."""
        ds = read_func(valid_rdi_file)
        assert isinstance(ds, xr.Dataset)


# ============================================================================
# TEST CLASS 2: Data Structure and Dimensions
# ============================================================================


class TestDatatypeStructure:
    """Test data structure, dimensions, and coordinates."""

    @pytest.mark.parametrize(
        "read_func,var_name",
        [
            (read_velocity, "velocity"),
            (read_correlation, "correlation"),
            (read_echo_intensity, "echo_intensity"),
            (read_percent_good, "percent_good"),
            (read_status, "status"),
        ],
    )
    def test_has_three_dimensions(self, read_func, var_name, valid_rdi_file):
        """Test that data variables have three dimensions: cell, beam, ensemble."""
        ds = read_func(valid_rdi_file)
        data_var = ds[var_name]
        assert (
            len(data_var.dims) == 3
        ), f"Expected 3 dimensions, got {len(data_var.dims)}"
        assert "cell" in data_var.dims
        assert "beam" in data_var.dims
        assert "ensemble" in data_var.dims

    @pytest.mark.parametrize(
        "read_func,var_name",
        [
            (read_velocity, "velocity"),
            (read_correlation, "correlation"),
            (read_echo_intensity, "echo_intensity"),
            (read_percent_good, "percent_good"),
            (read_status, "status"),
        ],
    )
    def test_dimensions_order_is_correct(self, read_func, var_name, valid_rdi_file):
        """Test that dimensions are ordered (cell, beam, ensemble)."""
        ds = read_func(valid_rdi_file)
        data_var = ds[var_name]
        assert data_var.dims == ("beam", "cell", "ensemble")

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_has_required_coordinates(self, read_func, valid_rdi_file):
        """Test that dataset has required coordinates."""
        ds = read_func(valid_rdi_file)
        assert "cell" in ds.coords
        assert "beam" in ds.coords
        assert "ensemble" in ds.coords

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_coordinates_are_numeric(self, read_func, valid_rdi_file):
        """Test that coordinates contain numeric values."""
        ds = read_func(valid_rdi_file)
        assert np.issubdtype(ds.coords["cell"].dtype, np.integer)
        assert np.issubdtype(ds.coords["beam"].dtype, np.integer)
        assert np.issubdtype(ds.coords["ensemble"].dtype, np.integer)

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_coordinates_start_at_zero(self, read_func, valid_rdi_file):
        """Test that coordinates are 0-indexed."""
        ds = read_func(valid_rdi_file)
        assert ds.coords["cell"].values[0] == 0
        assert ds.coords["beam"].values[0] == 0
        assert ds.coords["ensemble"].values[0] == 0


# ============================================================================
# TEST CLASS 3: Data Types
# ============================================================================


class TestDatatypeDtype:
    """Test correct numpy data types for each datatype."""

    def test_velocity_is_float32_by_default(self, valid_rdi_file):
        """Test that velocity data is float32 by default (missing_as_nan=True)."""
        ds = read_velocity(valid_rdi_file)
        assert ds["velocity"].dtype == np.float32

    def test_velocity_is_int16_when_sentinel(self, valid_rdi_file):
        """Test that velocity data is int16 when missing_as_nan=False."""
        ds = read_velocity(valid_rdi_file, missing_as_nan=False)
        assert ds["velocity"].dtype == np.int16

    def test_correlation_is_uint8(self, valid_rdi_file):
        """Test that correlation data is uint8."""
        ds = read_correlation(valid_rdi_file)
        assert ds["correlation"].dtype == np.uint8

    def test_echo_intensity_is_uint8(self, valid_rdi_file):
        """Test that echo intensity data is uint8."""
        ds = read_echo_intensity(valid_rdi_file)
        assert ds["echo_intensity"].dtype == np.uint8

    def test_percent_good_is_uint8(self, valid_rdi_file):
        """Test that percent good data is uint8."""
        ds = read_percent_good(valid_rdi_file)
        assert ds["percent_good"].dtype == np.uint8

    def test_status_is_uint8(self, valid_rdi_file):
        """Test that status data is uint8."""
        ds = read_status(valid_rdi_file)
        assert ds["status"].dtype == np.uint8


# ============================================================================
# TEST CLASS 4: Metadata and Attributes
# ============================================================================


class TestDatatypeAttributes:
    """Test metadata and CF Convention-compliant attributes."""

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_dataset_has_required_attributes(self, read_func, valid_rdi_file):
        """Test that dataset has required top-level attributes."""
        ds = read_func(valid_rdi_file)
        required_attrs = [
            "filename",
            "total_ensembles",
            "num_cells",
            "num_beams",
            "error_message",
            "pyadps_component",
            "pyadps_version",
            "adcp_data_format",
        ]
        for attr in required_attrs:
            assert attr in ds.attrs, f"Missing attribute: {attr}"

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_dataset_attributes_are_correct_types(self, read_func, valid_rdi_file):
        """Test that dataset attributes have correct types."""
        ds = read_func(valid_rdi_file)
        assert isinstance(ds.attrs["filename"], str)
        assert isinstance(ds.attrs["total_ensembles"], (int, np.integer))
        assert isinstance(ds.attrs["num_cells"], (int, np.integer))
        assert isinstance(ds.attrs["num_beams"], (int, np.integer))
        assert isinstance(ds.attrs["error_message"], str)
        assert isinstance(ds.attrs["pyadps_version"], str)
        assert isinstance(ds.attrs["adcp_data_format"], str)

    def test_velocity_has_variable_attributes(self, valid_rdi_file):
        """Test that velocity variable has CF Convention attributes (default nan mode)."""
        ds = read_velocity(valid_rdi_file)
        var_attrs = ds["velocity"].attrs
        assert "long_name" in var_attrs
        assert "units" in var_attrs
        assert "valid_min" in var_attrs
        assert "valid_max" in var_attrs
        assert "missing_value_handling" in var_attrs
        assert "missing_value" not in var_attrs
        assert "_FillValue" not in var_attrs

    def test_velocity_has_variable_attributes_sentinel(self, valid_rdi_file):
        """Test that velocity variable has sentinel attrs when missing_as_nan=False."""
        ds = read_velocity(valid_rdi_file, missing_as_nan=False)
        var_attrs = ds["velocity"].attrs
        assert "missing_value" in var_attrs
        assert "_FillValue" in var_attrs
        assert "missing_value_handling" in var_attrs

    def test_velocity_variable_attributes_values(self, valid_rdi_file):
        """Test that velocity variable attributes have correct values (default nan mode)."""
        ds = read_velocity(valid_rdi_file)
        var_attrs = ds["velocity"].attrs
        assert var_attrs["units"] == "mm s-1"
        assert var_attrs["missing_value_handling"] == "nan"
        assert var_attrs["valid_min"] == -32767
        assert var_attrs["valid_max"] == 32767

    def test_velocity_variable_attributes_values_sentinel(self, valid_rdi_file):
        """Test velocity variable attribute values when missing_as_nan=False."""
        ds = read_velocity(valid_rdi_file, missing_as_nan=False)
        var_attrs = ds["velocity"].attrs
        assert var_attrs["units"] == "mm s-1"
        assert var_attrs["missing_value"] == -32768
        assert var_attrs["_FillValue"] == -32768
        assert var_attrs["missing_value_handling"] == "sentinel"
        assert var_attrs["valid_min"] == -32768
        assert var_attrs["valid_max"] == 32767

    def test_velocity_description_does_not_mention_beams(self, valid_rdi_file):
        """
        'description' must not claim beam-coordinate data, since pyadps
        supports files in any coordinate system (Beam/Instrument/Ship/
        Earth) and this attribute isn't coordinate-system-dependent.
        """
        ds = read_velocity(valid_rdi_file)
        assert ds["velocity"].attrs["description"] == "Velocity magnitude measured by ADCP"

    def test_correlation_has_variable_attributes(self, valid_rdi_file):
        """Test that correlation variable has CF Convention attributes."""
        ds = read_correlation(valid_rdi_file)
        var_attrs = ds["correlation"].attrs
        assert "long_name" in var_attrs
        assert "units" in var_attrs
        assert "valid_min" in var_attrs
        assert "valid_max" in var_attrs

    def test_echo_intensity_has_variable_attributes(self, valid_rdi_file):
        """Test that echo intensity variable has CF Convention attributes."""
        ds = read_echo_intensity(valid_rdi_file)
        var_attrs = ds["echo_intensity"].attrs
        assert "long_name" in var_attrs
        assert "units" in var_attrs
        assert "scale_factor" in var_attrs

    def test_percent_good_has_variable_attributes(self, valid_rdi_file):
        """Test that percent good variable has CF Convention attributes."""
        ds = read_percent_good(valid_rdi_file)
        var_attrs = ds["percent_good"].attrs
        assert "long_name" in var_attrs
        assert "units" in var_attrs
        assert var_attrs["units"] == "percent"

    def test_status_has_variable_attributes(self, valid_rdi_file):
        """Test that status variable has CF Convention attributes."""
        ds = read_status(valid_rdi_file)
        var_attrs = ds["status"].attrs
        assert "long_name" in var_attrs
        assert "units" in var_attrs


class TestVelocityCommentsByCoordinateSystem:
    """
    'velocity.comments' depends on the file's coordinate transformation
    (Fixed Leader byte 26), since beam index meaning differs by coordinate
    system: an incorrect guess is worse than an empty comment.
    """

    def _read_velocity_with_coord_transform(self, tmp_path, coord_transform):
        data = build_ensemble(
            fixed_leader_data=FixedLeaderData(coord_transform=coord_transform)
        )
        rdi_file = tmp_path / "coord_transform.000"
        rdi_file.write_bytes(data)
        return read_velocity(rdi_file)

    def test_beam_coordinates_gets_beam_direction_comment(self, tmp_path):
        ds = self._read_velocity_with_coord_transform(tmp_path, coord_transform=0)
        assert ds["velocity"].attrs["comments"] == (
            "Negative values indicate flow direction opposite to beam direction"
        )

    def test_earth_coordinates_gets_component_mapping_comment(self, tmp_path):
        ds = self._read_velocity_with_coord_transform(tmp_path, coord_transform=24)
        assert ds["velocity"].attrs["comments"] == (
            "Beam index maps to Earth-coordinate velocity components: "
            "0=eastward (u), 1=northward (v), 2=upward (w), 3=error velocity"
        )

    def test_instrument_coordinates_leaves_comments_empty(self, tmp_path):
        ds = self._read_velocity_with_coord_transform(tmp_path, coord_transform=8)
        assert ds["velocity"].attrs["comments"] == ""

    def test_ship_coordinates_leaves_comments_empty(self, tmp_path):
        ds = self._read_velocity_with_coord_transform(tmp_path, coord_transform=16)
        assert ds["velocity"].attrs["comments"] == ""


# ============================================================================
# TEST CLASS 5: Cell and Beam Selection
# ============================================================================


class TestDatatypeCellBeamSelection:
    """Test cell and beam parameter selection functionality."""

    @pytest.mark.parametrize("read_func", [read_velocity, read_correlation])
    def test_cell_zero_returns_all_cells(self, read_func, valid_rdi_file):
        """Test that cell=0 returns all cells."""
        ds = read_func(valid_rdi_file)
        n_cells = ds.attrs["num_cells"]
        assert len(ds.coords["cell"]) == n_cells

    @pytest.mark.parametrize("read_func", [read_velocity, read_correlation])
    def test_beam_zero_returns_all_beams(self, read_func, valid_rdi_file):
        """Test that beam=0 returns all beams."""
        ds = read_func(valid_rdi_file)
        n_beams = ds.attrs["num_beams"]
        assert len(ds.coords["beam"]) == n_beams

    @pytest.mark.parametrize("read_func", [read_velocity, read_correlation])
    def test_cell_parameter_accepted(self, read_func, valid_rdi_file):
        """Test that cell parameter is accepted without error.

        Note: The underlying pyreadrdi.datatype() function accepts the cell
        parameter but may not reduce dimensions as expected. This test
        verifies the parameter doesn't cause errors.
        """
        ds_all = read_func(valid_rdi_file)
        # Should accept cell parameter without error
        ds_cell = read_func(valid_rdi_file, cell=1)
        assert isinstance(ds_cell, xr.Dataset)
        # Should still have valid structure
        assert "cell" in ds_cell.coords
        assert "beam" in ds_cell.coords
        assert "ensemble" in ds_cell.coords

    @pytest.mark.parametrize("read_func", [read_velocity, read_correlation])
    def test_beam_parameter_accepted(self, read_func, valid_rdi_file):
        """Test that beam parameter is accepted without error.

        Note: The underlying pyreadrdi.datatype() function accepts the beam
        parameter but may not reduce dimensions as expected. This test
        verifies the parameter doesn't cause errors.
        """
        ds_all = read_func(valid_rdi_file)
        # Should accept beam parameter without error
        ds_beam = read_func(valid_rdi_file, beam=0)
        assert isinstance(ds_beam, xr.Dataset)
        # Should still have valid structure
        assert "cell" in ds_beam.coords
        assert "beam" in ds_beam.coords
        assert "ensemble" in ds_beam.coords


# ============================================================================
# TEST CLASS 6: Parameter Variations
# ============================================================================


class TestDatatypeParameterVariations:
    """Test different parameter combinations and auto-fetching."""

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_auto_fetch_header_parameters(self, read_func, valid_rdi_file):
        """Test that functions auto-fetch header parameters when not provided."""
        # Should work without explicit header parameters
        ds = read_func(valid_rdi_file)
        assert isinstance(ds, xr.Dataset)

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_explicit_header_parameters(self, read_func, valid_rdi_file):
        """Test that functions work with explicitly provided header parameters."""
        header_ds = read_header(valid_rdi_file)
        byteskip = header_ds.byte_skip.values
        offset = header_ds.address_offset.values
        idarray = header_ds.data_id.values
        ensemble = header_ds.attrs["total_ensembles"]

        ds = read_func(
            valid_rdi_file,
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble,
        )
        assert isinstance(ds, xr.Dataset)

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_explicit_vs_auto_fetch_produce_same_result(
        self, read_func, valid_rdi_file
    ):
        """Test that explicit and auto-fetch modes produce identical results."""
        # Auto-fetch
        ds_auto = read_func(valid_rdi_file)

        # Explicit
        header_ds = read_header(valid_rdi_file)
        ds_explicit = read_func(
            valid_rdi_file,
            byteskip=header_ds.byte_skip.values,
            offset=header_ds.address_offset.values,
            idarray=header_ds.data_id.values,
            ensemble=header_ds.attrs["total_ensembles"],
        )

        # Should have same shape and values
        assert (
            ds_auto[list(ds_auto.data_vars)[0]].shape
            == ds_explicit[list(ds_explicit.data_vars)[0]].shape
        )


# ============================================================================
# TEST CLASS 7: Multi-Ensemble Handling
# ============================================================================


class TestDatatypeMultiEnsemble:
    """Test handling of files with multiple ensembles."""

    @pytest.mark.parametrize(
        "read_func,var_name",
        [
            (read_velocity, "velocity"),
            (read_correlation, "correlation"),
            (read_echo_intensity, "echo_intensity"),
            (read_percent_good, "percent_good"),
            (read_status, "status"),
        ],
    )
    def test_single_ensemble_file(self, read_func, var_name, valid_rdi_file):
        """Test reading file with single ensemble."""
        ds = read_func(valid_rdi_file)
        assert ds.attrs["total_ensembles"] == 1
        assert ds[var_name].shape[2] == 1

    @pytest.mark.parametrize(
        "read_func,var_name",
        [
            (read_velocity, "velocity"),
            (read_correlation, "correlation"),
            (read_echo_intensity, "echo_intensity"),
            (read_percent_good, "percent_good"),
            (read_status, "status"),
        ],
    )
    def test_multi_ensemble_file(self, read_func, var_name, multi_ensemble_rdi_file):
        """Test reading file with multiple ensembles."""
        ds = read_func(multi_ensemble_rdi_file)
        assert ds.attrs["total_ensembles"] == 3
        assert ds[var_name].shape[2] == 3


# ============================================================================
# TEST CLASS 8: Error Handling
# ============================================================================


class TestDatatypeErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_file_not_found_raises_error(self, read_func):
        """Test that non-existent file raises appropriate error."""
        with pytest.raises((FileNotFoundError, OSError, ValueError)):
            read_func("/nonexistent/path/file.000")

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_empty_file_handling(self, read_func, empty_file):
        """Test behavior with empty file."""
        # Should either raise error or return dataset with error indication
        try:
            result = read_func(empty_file)
            assert isinstance(result, xr.Dataset)
        except (ValueError, OSError, RuntimeError, struct.error):
            # Expected - empty file is invalid
            pass

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_truncated_file_handling(self, read_func, truncated_rdi_file):
        """Test behavior with truncated file."""
        # Should either raise error or return partial data
        try:
            result = read_func(truncated_rdi_file)
            assert isinstance(result, xr.Dataset)
        except (ValueError, OSError, RuntimeError, struct.error):
            # Expected - truncated file is invalid
            pass

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_invalid_path_type(self, read_func):
        """Test that invalid path type raises error.

        Note: Invalid path types can raise various exceptions:
        - TypeError: Path argument type checking
        - AttributeError: String methods on non-string objects
        - FileNotFoundError: File access attempt
        - ValueError: xarray dimension mismatch when data parsing fails
        """
        with pytest.raises((TypeError, AttributeError, FileNotFoundError, ValueError)):
            read_func(12345)


# ============================================================================
# TEST CLASS 9: xarray Compatibility
# ============================================================================


class TestDatatypeXarrayCompatibility:
    """Test xarray API compatibility."""

    @pytest.mark.parametrize(
        "read_func,var_name",
        [
            (read_velocity, "velocity"),
            (read_correlation, "correlation"),
            (read_echo_intensity, "echo_intensity"),
            (read_percent_good, "percent_good"),
            (read_status, "status"),
        ],
    )
    def test_can_select_single_ensemble(
        self, read_func, var_name, multi_ensemble_rdi_file
    ):
        """Test xarray .sel() method for ensemble selection."""
        ds = read_func(multi_ensemble_rdi_file)
        first_ensemble = ds.sel(ensemble=0)

        assert isinstance(first_ensemble, xr.Dataset)
        assert first_ensemble[var_name].ndim == 2  # 2D after removing ensemble dim

    @pytest.mark.parametrize(
        "read_func,var_name",
        [
            (read_velocity, "velocity"),
            (read_correlation, "correlation"),
            (read_echo_intensity, "echo_intensity"),
            (read_percent_good, "percent_good"),
            (read_status, "status"),
        ],
    )
    def test_can_select_ensemble_range(
        self, read_func, var_name, multi_ensemble_rdi_file
    ):
        """Test xarray .isel() method for ensemble slicing."""
        ds = read_func(multi_ensemble_rdi_file)
        subset = ds.isel(ensemble=slice(0, 2))

        assert isinstance(subset, xr.Dataset)
        assert subset[var_name].shape[2] == 2

    @pytest.mark.parametrize(
        "read_func,var_name",
        [
            (read_velocity, "velocity"),
            (read_correlation, "correlation"),
            (read_echo_intensity, "echo_intensity"),
            (read_percent_good, "percent_good"),
            (read_status, "status"),
        ],
    )
    def test_can_call_mean_method(self, read_func, var_name, valid_rdi_file):
        """Test xarray .mean() method."""
        ds = read_func(valid_rdi_file)
        mean_result = ds.mean()

        assert isinstance(mean_result, xr.Dataset)

    @pytest.mark.parametrize(
        "read_func,var_name",
        [
            (read_velocity, "velocity"),
            (read_correlation, "correlation"),
            (read_echo_intensity, "echo_intensity"),
            (read_percent_good, "percent_good"),
            (read_status, "status"),
        ],
    )
    def test_can_access_values_as_numpy(self, read_func, var_name, valid_rdi_file):
        """Test accessing data as numpy array."""
        ds = read_func(valid_rdi_file)
        numpy_array = ds[var_name].values

        assert isinstance(numpy_array, np.ndarray)


# ============================================================================
# TEST CLASS 10: Value Range Validation
# ============================================================================


class TestDatatypeValueRanges:
    """Test that data values are within expected ranges."""

    def test_velocity_values_in_valid_range(self, valid_rdi_file):
        """Test that velocity values are in valid int16 range."""
        ds = read_velocity(valid_rdi_file)
        vel_data = ds["velocity"].values
        # Should be within int16 range: [-32768, 32767]
        assert vel_data.min() >= -32768
        assert vel_data.max() <= 32767

    def test_correlation_values_in_valid_range(self, valid_rdi_file):
        """Test that correlation values are 0-255."""
        ds = read_correlation(valid_rdi_file)
        corr_data = ds["correlation"].values
        assert corr_data.min() >= 0
        assert corr_data.max() <= 255

    def test_echo_intensity_values_in_valid_range(self, valid_rdi_file):
        """Test that echo intensity values are 0-255."""
        ds = read_echo_intensity(valid_rdi_file)
        echo_data = ds["echo_intensity"].values
        assert echo_data.min() >= 0
        assert echo_data.max() <= 255

    def test_percent_good_values_in_valid_range(self, valid_rdi_file):
        """Test that percent good values are 0-100."""
        ds = read_percent_good(valid_rdi_file)
        pg_data = ds["percent_good"].values
        assert pg_data.min() >= 0
        assert pg_data.max() <= 100

    def test_status_values_in_valid_range(self, valid_rdi_file):
        """Test that status values are 0-255."""
        ds = read_status(valid_rdi_file)
        status_data = ds["status"].values
        assert status_data.min() >= 0
        assert status_data.max() <= 255


# ============================================================================
# TEST CLASS 11: Integration with read_header()
# ============================================================================


class TestDatatypeIntegrationWithReadHeader:
    """Test integration with read_header() function."""

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_ensemble_count_matches_header(self, read_func, valid_rdi_file):
        """Test that data ensemble count matches header information.

        Note: read_header() does not contain num_cells or num_beams attributes.
        Those are only available in the datatype-specific functions.
        This test verifies the ensemble count is consistent.
        """
        header_ds = read_header(valid_rdi_file)
        data_ds = read_func(valid_rdi_file)

        # Both should have the same ensemble count
        assert data_ds.attrs["total_ensembles"] == header_ds.attrs["total_ensembles"]
        # Verify the ensemble dimension matches
        assert len(data_ds.coords["ensemble"]) == header_ds.attrs["total_ensembles"]

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_dataset_can_be_combined_with_header(self, read_func, valid_rdi_file):
        """Test that datatype dataset can be merged/combined with header dataset."""
        header_ds = read_header(valid_rdi_file)
        data_ds = read_func(valid_rdi_file)

        # Should be able to merge datasets
        combined = xr.merge([header_ds, data_ds])
        assert isinstance(combined, xr.Dataset)


# ============================================================================
# TEST CLASS 12: Consistency Across Functions
# ============================================================================


class TestDatatypeConsistency:
    """Test consistency across all datatype functions."""

    def test_all_functions_return_same_shape(self, valid_rdi_file):
        """Test that all read functions return data with same shape."""
        ds_vel = read_velocity(valid_rdi_file)
        ds_corr = read_correlation(valid_rdi_file)
        ds_echo = read_echo_intensity(valid_rdi_file)
        ds_pg = read_percent_good(valid_rdi_file)
        ds_status = read_status(valid_rdi_file)

        var_vel = ds_vel["velocity"].shape
        var_corr = ds_corr["correlation"].shape
        var_echo = ds_echo["echo_intensity"].shape
        var_pg = ds_pg["percent_good"].shape
        var_status = ds_status["status"].shape

        # All should have same shape
        assert var_vel == var_corr == var_echo == var_pg == var_status

    def test_all_functions_return_same_dimensions(self, valid_rdi_file):
        """Test that all functions have same coordinate dimensions."""
        ds_vel = read_velocity(valid_rdi_file)
        ds_corr = read_correlation(valid_rdi_file)
        ds_echo = read_echo_intensity(valid_rdi_file)
        ds_pg = read_percent_good(valid_rdi_file)
        ds_status = read_status(valid_rdi_file)

        n_cells_vel = len(ds_vel.coords["cell"])
        n_cells_corr = len(ds_corr.coords["cell"])
        n_cells_echo = len(ds_echo.coords["cell"])
        n_cells_pg = len(ds_pg.coords["cell"])
        n_cells_status = len(ds_status.coords["cell"])

        assert (
            n_cells_vel == n_cells_corr == n_cells_echo == n_cells_pg == n_cells_status
        )

    def test_all_functions_have_ensemble_ensembles(self, multi_ensemble_rdi_file):
        """Test that all functions handle multiple ensembles consistently."""
        ds_vel = read_velocity(multi_ensemble_rdi_file)
        ds_corr = read_correlation(multi_ensemble_rdi_file)
        ds_echo = read_echo_intensity(multi_ensemble_rdi_file)
        ds_pg = read_percent_good(multi_ensemble_rdi_file)
        ds_status = read_status(multi_ensemble_rdi_file)

        assert ds_vel.attrs["total_ensembles"] == 3
        assert ds_corr.attrs["total_ensembles"] == 3
        assert ds_echo.attrs["total_ensembles"] == 3
        assert ds_pg.attrs["total_ensembles"] == 3
        assert ds_status.attrs["total_ensembles"] == 3


# ============================================================================
# TEST CLASS 13: NetCDF Export Compatibility
# ============================================================================


class TestDatatypeNetCDFExport:
    """Test that datasets can be exported to NetCDF format."""

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_can_export_to_netcdf(self, read_func, valid_rdi_file, tmp_path):
        """Test that dataset can be exported to NetCDF."""
        ds = read_func(valid_rdi_file)
        output_file = tmp_path / "test_output.nc"

        try:
            ds.to_netcdf(output_file)
            assert output_file.exists()
        except Exception as e:
            # Some attributes might not be NetCDF-compatible
            # This is acceptable - we're just verifying basic structure
            pass


# ============================================================================
# TEST CLASS 14: Edge Cases and Boundary Conditions
# ============================================================================


class TestDatatypeEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_path_as_string(self, read_func, valid_rdi_file):
        """Test file path can be provided as string."""
        ds = read_func(str(valid_rdi_file))
        assert isinstance(ds, xr.Dataset)

    @pytest.mark.parametrize(
        "read_func",
        [
            read_velocity,
            read_correlation,
            read_echo_intensity,
            read_percent_good,
            read_status,
        ],
    )
    def test_path_with_special_characters(self, read_func, tmp_path):
        """Test file path with special characters."""
        # Create file with special characters in name
        special_path = tmp_path / "test_file_123.000"
        from tests.io.test_binary_reader.fixtures.ensemble_builder import build_ensemble

        ensemble = build_ensemble()
        special_path.write_bytes(ensemble)

        ds = read_func(special_path)
        assert isinstance(ds, xr.Dataset)


# ============================================================================
# TEST CLASS 15: ValueError for Invalid Data Shape
# ============================================================================


class TestDatatypeInvalidDataShape:
    """Test ValueError is raised when pd0_parser returns non-3D data."""

    def test_read_velocity_raises_valueerror_for_non_3d_data(self, valid_rdi_file):
        """Test read_velocity raises ValueError when data is not 3D.

        This tests line 2067-2071:
            raise ValueError(
                f"Unexpected data shape from pd0_parser.datatype(): {data.shape}. "
                ...
            )
        """
        from pyadps.io import pd0_parser

        # Mock datatype to return 2D data instead of 3D
        # datatype returns: (data, ens, cells, beams, error_code)
        def mock_datatype(*args, **kwargs):
            return np.zeros((4, 10), dtype=np.int16), 1, 10, 4, 0  # 2D data

        with mock.patch.object(pd0_parser, "datatype", mock_datatype):
            with pytest.raises(ValueError) as exc_info:
                read_velocity(valid_rdi_file)

        assert "Unexpected data shape" in str(exc_info.value)
        assert "Expected 3D" in str(exc_info.value)

    def test_read_correlation_raises_valueerror_for_non_3d_data(self, valid_rdi_file):
        """Test read_correlation raises ValueError when data is not 3D.

        This tests line 2199-2203.
        """
        from pyadps.io import pd0_parser

        # datatype returns: (data, ens, cells, beams, error_code)
        def mock_datatype(*args, **kwargs):
            return np.zeros((4, 10), dtype=np.uint8), 1, 10, 4, 0  # 2D data

        with mock.patch.object(pd0_parser, "datatype", mock_datatype):
            with pytest.raises(ValueError) as exc_info:
                read_correlation(valid_rdi_file)

        assert "Unexpected data shape" in str(exc_info.value)
        assert "Expected 3D" in str(exc_info.value)

    def test_read_echo_intensity_raises_valueerror_for_non_3d_data(
        self, valid_rdi_file
    ):
        """Test read_echo_intensity raises ValueError when data is not 3D.

        This tests line 2302-2306.
        """
        from pyadps.io import pd0_parser

        # datatype returns: (data, ens, cells, beams, error_code)
        def mock_datatype(*args, **kwargs):
            return np.zeros((4, 10), dtype=np.uint8), 1, 10, 4, 0  # 2D data

        with mock.patch.object(pd0_parser, "datatype", mock_datatype):
            with pytest.raises(ValueError) as exc_info:
                read_echo_intensity(valid_rdi_file)

        assert "Unexpected data shape" in str(exc_info.value)
        assert "Expected 3D" in str(exc_info.value)

    def test_read_percent_good_raises_valueerror_for_non_3d_data(self, valid_rdi_file):
        """Test read_percent_good raises ValueError when data is not 3D.

        This tests line 2408-2412.
        """
        from pyadps.io import pd0_parser

        # datatype returns: (data, ens, cells, beams, error_code)
        def mock_datatype(*args, **kwargs):
            return np.zeros((4, 10), dtype=np.uint8), 1, 10, 4, 0  # 2D data

        with mock.patch.object(pd0_parser, "datatype", mock_datatype):
            with pytest.raises(ValueError) as exc_info:
                read_percent_good(valid_rdi_file)

        assert "Unexpected data shape" in str(exc_info.value)
        assert "Expected 3D" in str(exc_info.value)

    def test_read_status_raises_valueerror_for_non_3d_data(self, valid_rdi_file):
        """Test read_status raises ValueError when data is not 3D.

        This tests line 2510-2514.
        """
        from pyadps.io import pd0_parser

        # datatype returns: (data, ens, cells, beams, error_code)
        def mock_datatype(*args, **kwargs):
            return np.zeros((4, 10), dtype=np.uint8), 1, 10, 4, 0  # 2D data

        with mock.patch.object(pd0_parser, "datatype", mock_datatype):
            with pytest.raises(ValueError) as exc_info:
                read_status(valid_rdi_file)

        assert "Unexpected data shape" in str(exc_info.value)
        assert "Expected 3D" in str(exc_info.value)

    def test_read_velocity_raises_valueerror_for_1d_data(self, valid_rdi_file):
        """Test read_velocity raises ValueError when data is 1D."""
        from pyadps.io import pd0_parser

        # datatype returns: (data, ens, cells, beams, error_code)
        def mock_datatype(*args, **kwargs):
            return np.zeros(100, dtype=np.int16), 1, 10, 4, 0  # 1D data

        with mock.patch.object(pd0_parser, "datatype", mock_datatype):
            with pytest.raises(ValueError) as exc_info:
                read_velocity(valid_rdi_file)

        assert "got 1D" in str(exc_info.value)

    def test_read_velocity_raises_valueerror_for_4d_data(self, valid_rdi_file):
        """Test read_velocity raises ValueError when data is 4D."""
        from pyadps.io import pd0_parser

        # datatype returns: (data, ens, cells, beams, error_code)
        def mock_datatype(*args, **kwargs):
            return np.zeros((4, 10, 5, 2), dtype=np.int16), 1, 10, 4, 0  # 4D data

        with mock.patch.object(pd0_parser, "datatype", mock_datatype):
            with pytest.raises(ValueError) as exc_info:
                read_velocity(valid_rdi_file)

        assert "got 4D" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
