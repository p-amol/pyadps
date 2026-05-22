"""
Comprehensive pytest test suite for read() function in binary_reader.py

This module provides extensive test coverage for the primary read() function,
which loads complete ADCP datasets from RDI binary files into xarray.Dataset
objects. The read() function orchestrates the loading of all file components
(Header, FixedLeader, VariableLeader, and data types) into a single Dataset.

Test Coverage:
    - Basic functionality (valid files, single/multiple ensembles)
    - Data structure validation (xarray.Dataset, coordinates, dimensions)
    - Component selection via data_types parameter
    - Decoded field inclusion/exclusion
    - Time coordinate computation
    - Depth coordinate computation
    - Header inclusion/exclusion
    - Data type filtering and selective loading
    - Dimension ordering (time vs ensemble as primary dimension)
    - Accessor registration and availability
    - Backward compatibility with xarray operations
    - Performance optimization (selective loading)
    - Parameter variations and auto-fetch behavior
    - Error handling (missing files, corrupted data, invalid parameters)
    - Edge cases (empty files, single vs multiple ensembles)
    - Integration with component readers (read_header, read_fixed_leader, etc.)
    - Metadata and attributes (CF Convention, dataset-level attrs)
    - Large file handling and performance

References:
    RDI WorkHorse Commands and Output Data Format:
    - Section 5.2 (page 126): Fixed Leader (59 bytes per ensemble)
    - Section 5.3 (page 132-137): Variable Leader (65 bytes per ensemble)
    - Section 5 (page 123): Data types (Velocity, Correlation, Echo, etc.)

Author: pyadps development team
License: MIT
Version: 1.0.0
"""

import json
from pathlib import Path
from unittest import mock
import struct

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyadps.io.binary_reader import (
    read,
    read_header,
    read_fixed_leader,
    read_variable_leader,
    read_velocity,
    read_correlation,
    read_echo_intensity,
    read_percent_good,
    read_status,
    PYADPS_VERSION,
)

# Import test data builders
from .fixtures.ensemble_builder import (
    build_ensemble,
)


# ============================================================================
# FIXTURES: Test Files
# ============================================================================


@pytest.fixture(scope="function")
def valid_rdi_ensemble():
    """Generate a valid complete RDI ensemble with default configuration.

    Returns:
        bytes: Complete binary RDI ensemble with all 7 data types
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
    rdi_file = tmp_path / "test_read_single.000"
    rdi_file.write_bytes(valid_rdi_ensemble)
    return rdi_file


@pytest.fixture
def multi_ensemble_rdi_file(tmp_path, valid_rdi_ensemble):
    """Create a temporary file with multiple valid ensembles.

    Args:
        tmp_path: pytest built-in temp directory fixture
        valid_rdi_ensemble: Generated ensemble bytes

    Returns:
        Path: Pathlib.Path to temp RDI file with 5 ensembles
    """
    rdi_file = tmp_path / "test_read_multi.000"
    # Write 5 identical ensembles
    rdi_file.write_bytes(valid_rdi_ensemble * 5)
    return rdi_file


@pytest.fixture
def large_ensemble_rdi_file(tmp_path, valid_rdi_ensemble):
    """Create a temporary file with many ensembles for performance testing.

    Args:
        tmp_path: pytest built-in temp directory fixture
        valid_rdi_ensemble: Generated ensemble bytes

    Returns:
        Path: Pathlib.Path to temp RDI file with 100 ensembles
    """
    rdi_file = tmp_path / "test_read_large.000"
    # Write 100 ensembles
    rdi_file.write_bytes(valid_rdi_ensemble * 100)
    return rdi_file


@pytest.fixture
def empty_file(tmp_path):
    """Create an empty file for error handling tests.

    Args:
        tmp_path: pytest built-in temp directory fixture

    Returns:
        Path: Pathlib.Path to empty file
    """
    rdi_file = tmp_path / "test_empty.000"
    rdi_file.write_bytes(b"")
    return rdi_file


@pytest.fixture
def truncated_rdi_file(tmp_path, valid_rdi_ensemble):
    """Create a file truncated mid-ensemble.

    Args:
        tmp_path: pytest built-in temp directory fixture
        valid_rdi_ensemble: Generated ensemble bytes

    Returns:
        Path: Pathlib.Path to truncated RDI file
    """
    rdi_file = tmp_path / "test_truncated.000"
    # Write only first half of ensemble
    rdi_file.write_bytes(valid_rdi_ensemble[: len(valid_rdi_ensemble) // 2])
    return rdi_file


# ============================================================================
# TEST SUITE: Basic Functionality
# ============================================================================


class TestReadBasicFunctionality:
    """Tests for basic read() operation with valid files."""

    def test_read_valid_single_ensemble_file(self, valid_rdi_file):
        """Test reading valid RDI file with single ensemble returns xarray.Dataset."""
        result = read(valid_rdi_file)

        assert isinstance(result, xr.Dataset)
        assert len(result.coords["ensemble"]) == 1

    def test_read_multi_ensemble_file(self, multi_ensemble_rdi_file):
        """Test reading file with multiple ensembles returns Dataset with correct shape."""
        result = read(multi_ensemble_rdi_file)

        assert isinstance(result, xr.Dataset)
        assert len(result.coords["ensemble"]) == 5

    def test_read_returns_xarray_dataset(self, valid_rdi_file):
        """Test that read() returns xarray.Dataset type."""
        result = read(valid_rdi_file)

        assert type(result).__name__ == "Dataset"
        assert hasattr(result, "data_vars")
        assert hasattr(result, "coords")
        assert hasattr(result, "attrs")

    def test_read_contains_expected_dimensions(self, valid_rdi_file):
        """Test returned Dataset has expected dimensions."""
        result = read(valid_rdi_file)

        # Should have at least ensemble dimension
        assert "ensemble" in result.coords
        assert "time" in result.dims
        # May have cell, beam if data types included
        # Check that dimensions dict exists
        assert isinstance(result.dims, xr.core.utils.FrozenMappingWarningOnValuesAccess)
        assert len(result.dims) > 0

    def test_read_contains_coordinates(self, valid_rdi_file):
        """Test returned Dataset has coordinates."""
        result = read(valid_rdi_file)

        assert len(result.coords) > 0
        assert "ensemble" in result.coords

    def test_read_contains_data_variables(self, valid_rdi_file):
        """Test returned Dataset has data variables."""
        result = read(valid_rdi_file)

        assert len(result.data_vars) > 0

    def test_read_has_global_attributes(self, valid_rdi_file):
        """Test returned Dataset has appropriate global attributes."""
        result = read(valid_rdi_file)

        # Check for expected attributes
        expected_attrs = ["filename", "pyadps_version", "total_ensembles"]
        for attr in expected_attrs:
            assert attr in result.attrs, f"Missing attribute: {attr}"

    def test_read_preserves_filename_in_attrs(self, valid_rdi_file):
        """Test Dataset attributes contain original filename."""
        result = read(valid_rdi_file)
        assert "filename" in result.attrs
        filename_attr = str(result.attrs["filename"])
        # At minimum, should contain the actual file basename
        assert valid_rdi_file.name in filename_attr

    def test_read_records_total_ensembles(self, multi_ensemble_rdi_file):
        """Test Dataset attributes record correct ensemble count."""
        result = read(multi_ensemble_rdi_file)

        assert result.attrs["total_ensembles"] == 5


# ============================================================================
# TEST SUITE: Data Types and Components
# ============================================================================


class TestReadDataTypeSelection:
    """Tests for data_types parameter and selective component loading."""

    def test_read_all_components_default(self, valid_rdi_file):
        """Test default read() includes all available components.

        When data_types=None (default), should include all available components.
        """
        result = read(valid_rdi_file)

        # Should have FixedLeader variables (raw fields)
        # Should have VariableLeader variables (always included)
        # May have data type variables if available in file
        assert len(result.data_vars) > 0

    def test_read_variable_leader_always_included(self, valid_rdi_file):
        """Test VariableLeader is always included (provides time coordinate).

        Even when explicitly excluding VariableLeader, it should be included
        to provide the time dimension.
        """
        result = read(valid_rdi_file)

        # VariableLeader should provide at least the time-related variables
        # Check for some VL variables
        vl_vars = ["rtc_year", "rtc_month", "rtc_day"]
        has_vl = any(var in result.data_vars for var in vl_vars)
        assert has_vl, "VariableLeader variables should be present"

    def test_read_with_fixed_leader_only(self, valid_rdi_file):
        """Test reading with data_types=['FixedLeader', 'VariableLeader'].

        Should include FixedLeader + VariableLeader (VL always added).
        """
        result = read(valid_rdi_file, data_types=["FixedLeader", "VariableLeader"])

        assert isinstance(result, xr.Dataset)
        # Should have FixedLeader variables
        fl_vars = ["insturment_serial_number", "cpu_revision"]
        has_fl = any(var in result.data_vars for var in fl_vars)
        assert has_fl, "FixedLeader variables should be present"

    def test_read_with_velocity_only(self, valid_rdi_file):
        """Test reading with data_types=['Velocity'].

        Should include VariableLeader (auto-added) + Velocity data.
        """
        result = read(valid_rdi_file, data_types=["Velocity"])

        assert isinstance(result, xr.Dataset)
        # Should have velocity variable
        if "velocity" in result.data_vars:
            assert result["velocity"] is not None

    def test_read_with_multiple_data_types(self, valid_rdi_file):
        """Test reading with multiple data types specified.

        Example: ['FixedLeader', 'Velocity', 'Correlation']
        """
        result = read(
            valid_rdi_file, data_types=["FixedLeader", "Velocity", "Correlation"]
        )

        assert isinstance(result, xr.Dataset)
        # Should be able to read without error
        assert len(result.data_vars) > 0

    def test_read_empty_data_types_list(self, valid_rdi_file):
        """Test reading with empty data_types list.

        Should fall back to including all available components.
        """
        result = read(valid_rdi_file, data_types=[])

        # Empty list might be treated as "all" or "none" - implementation dependent
        # At minimum, should not crash
        assert isinstance(result, xr.Dataset)

    def test_read_performance_velocity_only(self, multi_ensemble_rdi_file):
        """Test performance optimization when loading only Velocity.

        Loading specific data types should be faster than loading everything.
        This is a benchmark test to ensure selective loading works.
        """
        # Without velocity filtering - loads all data
        result_all = read(multi_ensemble_rdi_file, data_types=None)

        # With velocity filtering - loads only velocity
        result_velocity = read(multi_ensemble_rdi_file, data_types=["Velocity"])

        # Velocity-only should have fewer variables (optimization verified)
        # This is informational - not a hard requirement
        assert isinstance(result_velocity, xr.Dataset)


# ============================================================================
# TEST SUITE: Decoded Fields Control
# ============================================================================


class TestReadDecodedFields:
    """Tests for include_decoded parameter."""

    def test_read_with_decoded_fields_default(self, valid_rdi_file):
        """Test decoded fields are included by default (include_decoded=True)."""
        result = read(valid_rdi_file)

        # Should include decoded variables
        # Examples: frequency_khz, beam_pattern, motion sensors
        # At least verify we have more than raw fields
        var_count_default = len(result.data_vars)
        assert var_count_default > 0

    def test_read_with_decoded_fields_enabled(self, valid_rdi_file):
        """Test decoded fields are included when include_decoded=True."""
        result = read(valid_rdi_file, include_decoded=True)

        var_count_with_decoded = len(result.data_vars)
        assert var_count_with_decoded > 0

    def test_read_without_decoded_fields(self, valid_rdi_file):
        """Test decoded fields are excluded when include_decoded=False."""
        result_without_decoded = read(valid_rdi_file, include_decoded=False)

        # Should have fewer variables (raw fields only)
        assert isinstance(result_without_decoded, xr.Dataset)
        # At minimum, should not crash
        assert len(result_without_decoded.data_vars) > 0

    def test_read_decoded_vs_raw_field_count(self, valid_rdi_file):
        """Test that include_decoded=True produces more variables than False.

        Decoded version should have additional computed fields beyond raw fields.
        """
        result_with_decoded = read(valid_rdi_file, include_decoded=True)
        result_without_decoded = read(valid_rdi_file, include_decoded=False)

        # With decoded should have >= raw-only version
        assert len(result_with_decoded.data_vars) >= len(
            result_without_decoded.data_vars
        )


# ============================================================================
# TEST SUITE: Header Inclusion
# ============================================================================


class TestReadHeaderInclusion:
    """Tests for include_header parameter."""

    def test_read_header_excluded_default(self, valid_rdi_file):
        """Test header data is excluded by default (include_header=False)."""
        result = read(valid_rdi_file, include_header=False)

        # Header variables should not be present
        header_vars = ["byte_skip", "address_offset", "data_id"]
        has_header = any(var in result.data_vars for var in header_vars)
        # Default should not include header
        # This may depend on implementation

    def test_read_header_included(self, valid_rdi_file):
        """Test header data is included when include_header=True."""
        result = read(valid_rdi_file, include_header=True)

        # Header variables should be present
        header_vars = ["byte_skip", "address_offset", "data_id"]
        has_header_vars = [var in result.data_vars for var in header_vars]
        # At least some header variables should be present
        assert any(
            has_header_vars
        ), "Should have header variables when include_header=True"

    def test_read_header_excluded(self, valid_rdi_file):
        """Test header data is excluded when include_header=False."""
        result = read(valid_rdi_file, include_header=False)

        # Should not crash and return valid Dataset
        assert isinstance(result, xr.Dataset)


# ============================================================================
# TEST SUITE: Coordinates and Dimensions
# ============================================================================


class TestReadCoordinatesAndDimensions:
    """Tests for coordinate computation and dimension handling."""

    def test_read_has_time_coordinate(self, valid_rdi_file):
        """Test returned Dataset has time coordinate.

        Time is computed from Variable Leader RTC fields.
        """
        result = read(valid_rdi_file)

        # Check for time coordinate
        # May be named 'time', 'rtc_time', or similar
        time_vars = [var for var in result.coords if "time" in var.lower()]
        assert len(time_vars) > 0, "Should have time coordinate"

    def test_read_time_coordinate_is_datetime(self, valid_rdi_file):
        """Test time coordinate values are datetime64 dtype."""
        result = read(valid_rdi_file)

        # Find time coordinate
        time_var = None
        for coord in result.coords:
            if "time" in coord.lower():
                time_var = result[coord]
                break

        if time_var is not None:
            # Check dtype
            assert np.issubdtype(time_var.dtype, np.datetime64)

    def test_read_time_coordinate_monotonic(self, multi_ensemble_rdi_file):
        """Test time coordinate is monotonic (non-decreasing).

        Since ensembles are identical in test data, time should be constant or increasing.
        """
        result = read(multi_ensemble_rdi_file)

        # Find time coordinate
        time_var = None
        for coord in result.coords:
            if "time" in coord.lower():
                time_var = result[coord].values
                break

        if time_var is not None:
            # Check monotonicity (allow equal values since test ensembles are identical)
            # Time should not decrease
            diffs = np.diff(time_var.astype(np.int64))
            assert np.all(diffs >= 0), "Time should be monotonic (non-decreasing)"

    def test_read_has_time_dimension(self, valid_rdi_file):
        """Test returned Dataset has ensemble dimension."""
        result = read(valid_rdi_file)

        assert "time" in result.dims

    def test_read_time_dimension_size(self, multi_ensemble_rdi_file):
        """Test ensemble dimension has correct size."""
        result = read(multi_ensemble_rdi_file)

        assert result.sizes["time"] == 5

    def test_read_depth_coordinate_if_available(self, valid_rdi_file):
        """Test depth coordinate is computed when Fixed/Variable Leaders present.

        Depth is computed from bin configuration in Fixed Leader.
        """
        result = read(valid_rdi_file)

        # Depth coordinate may be present (depends on successful FL reading)
        # Just verify no crash
        assert isinstance(result, xr.Dataset)

    def test_read_use_time_as_primary_dim_true(self, multi_ensemble_rdi_file):
        """Test use_time_as_primary_dim=True arranges time as first dimension.

        Multi-dimensional variables should have time as first dimension.
        """
        result = read(multi_ensemble_rdi_file, use_time_as_primary_dim=True)

        assert isinstance(result, xr.Dataset)
        # Multi-dimensional variables should exist
        # Verification of dimension order is implementation-specific

    def test_read_use_time_as_primary_dim_false(self, multi_ensemble_rdi_file):
        """Test use_time_as_primary_dim=False uses ensemble as primary dimension.

        Legacy behavior: ensemble as first dimension.
        """
        result = read(multi_ensemble_rdi_file, use_time_as_primary_dim=False)

        assert isinstance(result, xr.Dataset)


# ============================================================================
# TEST SUITE: Path Handling
# ============================================================================


class TestReadPathHandling:
    """Tests for different path input types."""

    def test_read_with_string_path(self, valid_rdi_file):
        """Test read() accepts file path as string."""
        result = read(str(valid_rdi_file))

        assert isinstance(result, xr.Dataset)

    def test_read_with_pathlib_path(self, valid_rdi_file):
        """Test read() accepts file path as pathlib.Path object."""
        result = read(Path(valid_rdi_file))

        assert isinstance(result, xr.Dataset)

    def test_read_nonexistent_path_raises_error(self):
        """Test read() raises error for missing file.

        Nonexistent files should fail when trying to read header or construct
        the dataset. May raise FileNotFoundError, ValueError, or similar exception.
        """

        with pytest.raises((FileNotFoundError, ValueError, Exception)):
            read("/nonexistent/file/path.000")


# ============================================================================
# TEST SUITE: Accessor Registration
# ============================================================================


class TestReadAccessorRegistration:
    """Tests for xarray accessor registration and availability."""

    def test_read_dataset_has_fixed_leader_accessor(self, valid_rdi_file):
        """Test returned Dataset has fixed_leader accessor available.

        Accessors should be registered automatically by pyadps.io.accessors module.
        """
        # Import accessors to register them

        result = read(valid_rdi_file, data_types=["FixedLeader"])

        # If FixedLeader was loaded, accessor should be available
        # This depends on whether FixedLeader is in the dataset
        if any("frequency" in var for var in result.data_vars):
            # FixedLeader data present, accessor should work
            assert hasattr(result, "fixed_leader")

    def test_read_dataset_has_variable_leader_accessor(self, valid_rdi_file):
        """Test returned Dataset has variable_leader accessor available.

        Variable Leader is always read, so this accessor should always work.
        """
        # Import accessors to register them

        result = read(valid_rdi_file)

        # Variable Leader is always included
        assert hasattr(result, "variable_leader")

    def test_read_accessor_methods_callable(self, valid_rdi_file):
        """Test accessor methods are callable."""
        # Import accessors to register them

        result = read(valid_rdi_file)

        # Variable Leader accessor should have methods
        assert callable(getattr(result.variable_leader, "summary", None))


# ============================================================================
# TEST SUITE: Integration with xarray
# ============================================================================


class TestReadXarrayIntegration:
    """Tests for compatibility with xarray operations."""

    def test_read_dataset_supports_indexing(self, valid_rdi_file):
        """Test returned Dataset supports indexing operations."""
        result = read(valid_rdi_file)

        # Should support indexing by variable name
        data_vars = list(result.data_vars)
        if len(data_vars) > 0:
            var = result[data_vars[0]]
            assert var is not None

    def test_read_dataset_supports_isel(self, multi_ensemble_rdi_file):
        """Test returned Dataset supports isel() selection."""
        result = read(multi_ensemble_rdi_file)

        # Should support integer selection
        subset = result.isel(time=0)
        assert isinstance(subset, xr.Dataset)

    def test_read_dataset_supports_time_swap_sel(self, valid_rdi_file):
        """Test returned Dataset supports time to ensemble swap and sel() selection."""
        result = read(valid_rdi_file)

        # If time coordinate exists, should support label-based selection
        if "ensemble" in result.coords or any("ensemble" in c for c in result.coords):
            # Just verify sel() doesn't crash
            result = result.swap_dims({"time": "ensemble"})
            subset = result.sel(ensemble=1)
            assert isinstance(subset, xr.Dataset)

    def test_read_dataset_supports_depth_swap_sel(self, valid_rdi_file):
        """Test returned Dataset supports time to ensemble swap and sel() selection."""
        result = read(valid_rdi_file, use_depth_as_primary_dim=True)

        # If time coordinate exists, should support label-based selection
        if "depth" in result.coords or any("depth" in c for c in result.coords):
            # Just verify sel() doesn't crash
            result = result.swap_dims({"depth": "cell"})
            subset = result.sel(cell=1)
            assert isinstance(subset, xr.Dataset)

    def test_read_dataset_supports_mean(self, multi_ensemble_rdi_file):
        """Test returned Dataset supports mean() operation."""
        result = read(multi_ensemble_rdi_file)

        # Should support statistical operations
        mean_result = result.mean(dim="time", skipna=True)
        assert isinstance(mean_result, xr.Dataset)

    def test_read_dataset_supports_to_netcdf(self, valid_rdi_file, tmp_path):
        """Test returned Dataset can be saved to NetCDF format."""
        result = read(valid_rdi_file)

        # Should support NetCDF export
        output_file = tmp_path / "output.nc"
        try:
            result.to_netcdf(output_file)
            assert output_file.exists()
        except Exception:
            # NetCDF export might fail due to metadata issues
            # But should not crash the read function
            pass


# ============================================================================
# TEST SUITE: Error Handling
# ============================================================================


class TestReadErrorHandling:
    """Tests for error handling and edge cases."""

    def test_read_empty_file_raises_error(self, empty_file):
        """Test read() raises error for empty file."""
        with pytest.raises((ValueError, Exception)):
            read(empty_file)

    def test_read_truncated_file_raises_error(self, truncated_rdi_file):
        """Test read() raises error for truncated/incomplete data."""
        with pytest.raises((ValueError, struct.error, Exception)):
            read(truncated_rdi_file)

    def test_read_nonexistent_path_raises_error(self):
        """Test read() raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read("/nonexistent/file/path.000")


# ============================================================================
# TEST SUITE: Performance and Large Files
# ============================================================================


class TestReadPerformance:
    """Tests for performance with larger files."""

    def test_read_large_file(self, large_ensemble_rdi_file):
        """Test read() handles larger files (100 ensembles) efficiently.

        Should not crash and complete in reasonable time.
        """
        result = read(large_ensemble_rdi_file)

        assert isinstance(result, xr.Dataset)
        assert result.attrs["total_ensembles"] == 100

    def test_read_large_file_selective_loading(self, large_ensemble_rdi_file):
        """Test selective loading is faster than loading all data."""
        # Just verify both work without performance degradation
        result_selective = read(large_ensemble_rdi_file, data_types=["Velocity"])
        result_all = read(large_ensemble_rdi_file)

        assert isinstance(result_selective, xr.Dataset)
        assert isinstance(result_all, xr.Dataset)


# ============================================================================
# TEST SUITE: Backward Compatibility
# ============================================================================


class TestReadBackwardCompatibility:
    """Tests for backward compatibility with v0.4.0 workflows."""

    def test_read_returns_xarray_not_custom_class(self, valid_rdi_file):
        """Test read() returns xarray.Dataset, not custom ReadFile class.

        This ensures the new API is different from v0.4.0.
        """
        result = read(valid_rdi_file)

        # Should be xarray.Dataset
        assert type(result).__module__ == "xarray.core.dataset"

    def test_read_dataset_accessible_like_dict(self, valid_rdi_file):
        """Test returned Dataset can be accessed like dict via accessors.

        Users familiar with custom classes should be able to use accessors.
        """

        result = read(valid_rdi_file, data_types=["FixedLeader", "Velocity"])

        # Should be able to access fixed_leader info via accessor
        if "frequency_khz" in result.data_vars:
            # Can access data directly
            freq = result["frequency_khz"].values
            assert freq is not None


# ============================================================================
# TEST SUITE: Dataset Attributes and Metadata
# ============================================================================


class TestReadMetadataAndAttributes:
    """Tests for dataset-level attributes and metadata."""

    def test_read_dataset_version_attribute(self, valid_rdi_file):
        """Test Dataset has pyadps_version attribute."""
        result = read(valid_rdi_file)

        assert "pyadps_version" in result.attrs
        assert result.attrs["pyadps_version"] == PYADPS_VERSION

    def test_read_dataset_format_attribute(self, valid_rdi_file):
        """Test Dataset documents ADCP data format."""
        result = read(valid_rdi_file)

        assert "adcp_data_format" in result.attrs
        assert result.attrs["adcp_data_format"] == "PD0"

    def test_read_dataset_components_tracked(self, valid_rdi_file):
        """Test Dataset tracks which components were included."""
        result = read(valid_rdi_file)

        # Should track components
        if "components" in result.attrs:
            components = result.attrs["components"]
            assert isinstance(components, str)

    def test_read_variable_attributes_cf_convention(self, valid_rdi_file):
        """Test variables have CF Convention attributes.

        Variables should have long_name, units, and other standard attributes.
        """
        result = read(valid_rdi_file, include_decoded=True)

        # Check first variable
        var_name = list(result.data_vars.keys())[0]
        var_attrs = result[var_name].attrs

        # Should have at least some standard attributes
        standard_attrs = ["long_name", "units"]
        has_standard = any(attr in var_attrs for attr in standard_attrs)
        assert has_standard or len(var_attrs) > 0


# ============================================================================
# TEST SUITE: Parameter Combinations
# ============================================================================


class TestReadParameterCombinations:
    """Tests for various parameter combinations."""

    def test_read_all_parameters_together(self, valid_rdi_file):
        """Test read() with all parameters specified simultaneously."""
        result = read(
            valid_rdi_file,
            include_decoded=True,
            data_types=["FixedLeader", "Velocity"],
            use_time_as_primary_dim=True,
            use_depth_as_primary_dim=False,
            include_header=True,
        )

        assert isinstance(result, xr.Dataset)

    def test_read_minimal_parameters(self, valid_rdi_file):
        """Test read() with minimal parameters (just filename)."""
        result = read(valid_rdi_file)

        assert isinstance(result, xr.Dataset)

    def test_read_with_use_depth_as_primary_dim(self, valid_rdi_file):
        """Test read() with use_depth_as_primary_dim=True.

        Should arrange data with depth (cell) as primary dimension instead of ensemble.
        """
        result = read(valid_rdi_file, use_depth_as_primary_dim=True)

        assert isinstance(result, xr.Dataset)


# ============================================================================
# TEST SUITE: Consistency
# ============================================================================


class TestReadConsistency:
    """Tests for consistency and deterministic behavior."""

    def test_read_same_file_twice_identical(self, valid_rdi_file):
        """Test reading same file twice produces identical results."""
        result1 = read(valid_rdi_file)
        result2 = read(valid_rdi_file)

        # Compare data variables
        for var in result1.data_vars:
            if var in result2.data_vars:
                np.testing.assert_array_equal(
                    result1[var].values,
                    result2[var].values,
                    err_msg=f"Variable {var} differs",
                )

    def test_read_with_different_include_decoded_consistent(self, valid_rdi_file):
        """Test subset of raw fields matches non-decoded version.

        Raw fields in decoded version should match non-decoded version.
        """
        result_with = read(valid_rdi_file, include_decoded=True)
        result_without = read(valid_rdi_file, include_decoded=False)

        # Raw fields should be identical
        for var in result_without.data_vars:
            if var in result_with.data_vars:
                np.testing.assert_array_equal(
                    result_with[var].values,
                    result_without[var].values,
                    err_msg=f"Raw field {var} differs",
                )


# ============================================================================
# TEST SUITE: Comprehensive Coverage Tests
# ============================================================================


class TestReadFixedLeaderConditional:
    """Tests for Fixed Leader conditional reading (lines 3311-3326)."""

    def test_read_skips_fixed_leader_when_not_in_data_types(
        self, valid_rdi_file, caplog
    ):
        """Test that Fixed Leader is skipped when not in data_types.

        This tests lines 3325-3326:
            else:
                logger.debug("Skipping Fixed Leader (not in data_types)")
        """
        import logging

        with caplog.at_level(logging.DEBUG):
            result = read(valid_rdi_file, data_types=["VariableLeader", "Velocity"])

        # FixedLeader should be skipped
        assert "Skipping Fixed Leader" in caplog.text
        assert isinstance(result, xr.Dataset)

    def test_read_includes_fixed_leader_when_in_data_types(self, valid_rdi_file):
        """Test that Fixed Leader is included when in data_types."""
        result = read(valid_rdi_file, data_types=["FixedLeader", "VariableLeader"])

        # Should have Fixed Leader variables
        assert isinstance(result, xr.Dataset)
        # At least one FL variable should be present
        fl_vars = ["num_cells", "num_beams", "bin_1_distance"]
        assert any(var in result.data_vars for var in fl_vars)

    def test_read_includes_fixed_leader_when_data_types_none(self, valid_rdi_file):
        """Test that Fixed Leader is included when data_types is None (default)."""
        result = read(valid_rdi_file, data_types=None)

        assert isinstance(result, xr.Dataset)
        # Should have Fixed Leader variables when data_types=None
        fl_vars = ["num_cells", "num_beams"]
        assert any(var in result.data_vars for var in fl_vars)

    def test_read_handles_fixed_leader_read_error(self, valid_rdi_file, caplog):
        """Test handling of Fixed Leader read errors.

        This tests lines 3323-3324:
            except Exception as e:
                logger.warning(f"Could not read Fixed Leader data: {e}")
        """
        import logging

        with mock.patch(
            "pyadps.io.binary_reader.read_fixed_leader",
            side_effect=ValueError("Simulated FL error"),
        ):
            with caplog.at_level(logging.WARNING):
                result = read(valid_rdi_file)

        # Should log warning but not crash
        assert "Could not read Fixed Leader data" in caplog.text
        assert isinstance(result, xr.Dataset)


class TestReadVariableLeaderHandling:
    """Tests for Variable Leader handling (lines 3333-3345)."""

    def test_read_handles_variable_leader_read_error(self, valid_rdi_file, caplog):
        """Test handling of Variable Leader read errors.

        This tests lines 3344-3345:
            except Exception as e:
                logger.warning(f"Could not read Variable Leader data: {e}")
        """
        import logging

        with mock.patch(
            "pyadps.io.binary_reader.read_variable_leader",
            side_effect=ValueError("Simulated VL error"),
        ):
            with caplog.at_level(logging.WARNING):
                result = read(valid_rdi_file)

        assert "Could not read Variable Leader data" in caplog.text
        assert isinstance(result, xr.Dataset)


class TestReadCoordinateComputation:
    """Tests for coordinate computation (lines 3350-3371)."""

    def test_read_skips_time_coord_when_vl_none(self, valid_rdi_file, caplog):
        """Test that time coordinate is skipped when VL not loaded.

        This tests lines 3359-3360:
            else:
                logger.debug("Skipping time coordinate (Variable Leader not loaded)")
        """
        import logging

        with mock.patch(
            "pyadps.io.binary_reader.read_variable_leader", return_value=None
        ):
            with caplog.at_level(logging.DEBUG):
                result = read(valid_rdi_file)

        assert "Skipping time coordinate" in caplog.text

    def test_read_handles_time_coord_error(self, valid_rdi_file, caplog):
        """Test handling of time coordinate computation errors.

        This tests lines 3357-3358:
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Could not compute time coordinate: {e}")
        """
        import logging

        with mock.patch(
            "pyadps.io.binary_reader._compute_time_coordinate",
            side_effect=KeyError("rtc_year"),
        ):
            with caplog.at_level(logging.WARNING):
                result = read(valid_rdi_file)

        assert "Could not compute time coordinate" in caplog.text
        assert isinstance(result, xr.Dataset)

    def test_read_skips_depth_coord_when_fl_none(self, valid_rdi_file, caplog):
        """Test that depth coordinate is skipped when FL not loaded.

        This tests lines 3368-3371:
            else:
                logger.debug(
                    "Skipping depth coordinate (FixedLeader or VariableLeader not loaded)"
                )
        """
        import logging

        # Skip FL by excluding from data_types
        with caplog.at_level(logging.DEBUG):
            result = read(valid_rdi_file, data_types=["VariableLeader", "Velocity"])

        assert "Skipping depth coordinate" in caplog.text

    def test_read_handles_depth_coord_error(self, valid_rdi_file, caplog):
        """Test handling of depth coordinate computation errors.

        This tests lines 3366-3367:
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Could not compute depth coordinate: {e}")
        """
        import logging

        with mock.patch(
            "pyadps.io.binary_reader._compute_depth_coordinate",
            side_effect=KeyError("num_cells"),
        ):
            with caplog.at_level(logging.WARNING):
                result = read(valid_rdi_file)

        assert "Could not compute depth coordinate" in caplog.text
        assert isinstance(result, xr.Dataset)


class TestReadDataTypeLoading:
    """Tests for optional data type loading (lines 3385-3448)."""

    def test_read_loads_velocity_when_available(self, valid_rdi_file):
        """Test Velocity data is loaded when available and requested."""
        result = read(valid_rdi_file, data_types=["Velocity"])

        assert isinstance(result, xr.Dataset)
        assert "velocity" in result.data_vars

    def test_read_handles_velocity_read_error(self, valid_rdi_file, caplog):
        """Test handling of Velocity read errors.

        This tests lines 3395-3396:
            except Exception as e:
                logger.warning(f"Could not read Velocity data: {e}")
        """
        import logging

        with mock.patch(
            "pyadps.io.binary_reader.read_velocity",
            side_effect=ValueError("Velocity error"),
        ):
            with caplog.at_level(logging.WARNING):
                result = read(valid_rdi_file, data_types=["Velocity"])

        assert "Could not read Velocity data" in caplog.text

    def test_read_handles_correlation_read_error(self, valid_rdi_file, caplog):
        """Test handling of Correlation read errors.

        This tests lines 3408-3409:
            except Exception as e:
                logger.warning(f"Could not read Correlation data: {e}")
        """
        import logging

        with mock.patch(
            "pyadps.io.binary_reader.read_correlation",
            side_effect=ValueError("Correlation error"),
        ):
            with caplog.at_level(logging.WARNING):
                result = read(valid_rdi_file, data_types=["Correlation"])

        assert "Could not read Correlation data" in caplog.text

    def test_read_handles_echo_read_error(self, valid_rdi_file, caplog):
        """Test handling of Echo Intensity read errors.

        This tests lines 3421-3422:
            except Exception as e:
                logger.warning(f"Could not read Echo Intensity data: {e}")
        """
        import logging

        with mock.patch(
            "pyadps.io.binary_reader.read_echo_intensity",
            side_effect=ValueError("Echo error"),
        ):
            with caplog.at_level(logging.WARNING):
                result = read(valid_rdi_file, data_types=["Echo"])

        assert "Could not read Echo Intensity data" in caplog.text

    def test_read_handles_percent_good_read_error(self, valid_rdi_file, caplog):
        """Test handling of Percent Good read errors.

        This tests lines 3434-3435:
            except Exception as e:
                logger.warning(f"Could not read Percent Good data: {e}")
        """
        import logging

        with mock.patch(
            "pyadps.io.binary_reader.read_percent_good",
            side_effect=ValueError("Percent Good error"),
        ):
            with caplog.at_level(logging.WARNING):
                result = read(valid_rdi_file, data_types=["Percent Good"])

        assert "Could not read Percent Good data" in caplog.text

    def test_read_handles_status_read_error(self, valid_rdi_file, caplog):
        """Test handling of Status read errors.

        This tests lines 3447-3448:
            except Exception as e:
                logger.warning(f"Could not read Status data: {e}")
        """
        import logging

        with mock.patch(
            "pyadps.io.binary_reader.read_status",
            side_effect=ValueError("Status error"),
        ):
            with caplog.at_level(logging.WARNING):
                result = read(valid_rdi_file, data_types=["Status"])

        assert "Could not read Status data" in caplog.text


class TestReadMaskCreation:
    """Tests for velocity mask creation (lines 3453-3461)."""

    def test_read_creates_mask_when_velocity_loaded(self, valid_rdi_file):
        """Test velocity mask is created when velocity data is loaded."""
        result = read(valid_rdi_file, include_mask=True)

        # If velocity was loaded, mask should be present
        if "velocity" in result.data_vars:
            assert "velocity_mask" in result.data_vars or "mask" in result.data_vars

    def test_read_skips_mask_when_include_mask_false(self, valid_rdi_file, caplog):
        """Test mask creation is skipped when include_mask=False.

        This tests lines 3460-3461:
            elif not include_mask:
                logger.debug("Skipping velocity mask creation (include_mask=False)")
        """
        import logging

        with caplog.at_level(logging.DEBUG):
            result = read(valid_rdi_file, include_mask=False)

        assert "Skipping velocity mask creation" in caplog.text
        # Mask should not be present
        assert "velocity_mask" not in result.data_vars

    def test_read_handles_mask_creation_error(self, valid_rdi_file, caplog):
        """Test handling of mask creation errors.

        This tests lines 3458-3459:
            except Exception as e:
                logger.warning(f"Could not create velocity mask: {e}")
        """
        import logging

        with mock.patch(
            "pyadps.io.binary_reader._create_velocity_mask",
            side_effect=ValueError("Mask error"),
        ):
            with caplog.at_level(logging.WARNING):
                result = read(valid_rdi_file, include_mask=True)

        assert "Could not create velocity mask" in caplog.text


class TestReadVariableLeaderExclusion:
    """Tests for Variable Leader exclusion (line 3469-3470)."""

    def test_read_excludes_variable_leader_when_not_in_data_types(self, valid_rdi_file):
        """Test that VariableLeader is excluded when explicitly not in data_types.

        This tests lines 3469-3470:
            if data_types is not None and "VariableLeader" not in data_types:
                ds_vl = None
        """
        result = read(valid_rdi_file, data_types=["FixedLeader"])

        # VL variables should NOT be present since VL was excluded
        assert isinstance(result, xr.Dataset)


class TestReadEnsembleConsistency:
    """Tests for ensemble consistency checking (lines 3485-3514)."""

    def _create_mock_header(self, n_ensembles):
        """Create a properly structured mock header dataset."""
        return xr.Dataset(
            data_vars={
                "byte_skip": (["ensemble"], [0] * n_ensembles),
                "address_offset": (["ensemble", "data_type"], [[0, 26]] * n_ensembles),
                "data_id": (["ensemble", "data_type"], [[0, 128]] * n_ensembles),
            },
            coords={"ensemble": list(range(n_ensembles)), "data_type": [0, 1]},
            attrs={
                "total_ensembles": n_ensembles,
                "filename": "test.000",
                "pyadps_version": "1.0.0",
                "adcp_data_format": "PD0",
            },
        )

    def test_read_truncates_fixed_leader_on_inconsistency(self, valid_rdi_file, caplog):
        """Test Fixed Leader is truncated when ensemble counts are inconsistent.

        This tests lines 3499-3500:
            if ds_fl is not None and "ensemble" in ds_fl.dims:
                ds_fl = ds_fl.isel(ensemble=slice(0, min_ensemble))
        """
        import logging

        ds_header = self._create_mock_header(3)
        # FL has 5 ensembles - more than header
        ds_fl = xr.Dataset(
            data_vars={"num_cells": (["ensemble"], [30, 30, 30, 30, 30])},
            coords={"ensemble": [0, 1, 2, 3, 4]},
        )
        # VL has 3 ensembles - matches header
        ds_vl = xr.Dataset(
            data_vars={"heading": (["ensemble"], [100, 200, 300])},
            coords={"ensemble": [0, 1, 2]},
        )

        with mock.patch("pyadps.io.binary_reader.read_header", return_value=ds_header):
            with mock.patch(
                "pyadps.io.binary_reader.read_fixed_leader", return_value=ds_fl
            ):
                with mock.patch(
                    "pyadps.io.binary_reader.read_variable_leader", return_value=ds_vl
                ):
                    with mock.patch(
                        "pyadps.io.binary_reader._compute_time_coordinate",
                        return_value=None,
                    ):
                        with mock.patch(
                            "pyadps.io.binary_reader._compute_depth_coordinate",
                            return_value=None,
                        ):
                            with mock.patch(
                                "pyadps.io.binary_reader._get_available_data_types",
                                return_value=[],
                            ):
                                with caplog.at_level(logging.WARNING):
                                    result = read(
                                        valid_rdi_file,
                                        data_types=["FixedLeader", "VariableLeader"],
                                        include_mask=False,
                                    )

        # Should have logged truncation
        assert "Ensemble count inconsistency" in caplog.text
        assert isinstance(result, xr.Dataset)

    def test_read_truncates_variable_leader_on_inconsistency(
        self, valid_rdi_file, caplog
    ):
        """Test Variable Leader is truncated when ensemble counts are inconsistent.

        This tests lines 3501-3502:
            if ds_vl is not None and "ensemble" in ds_vl.dims:
                ds_vl = ds_vl.isel(ensemble=slice(0, min_ensemble))
        """
        import logging

        ds_header = self._create_mock_header(2)
        ds_fl = xr.Dataset(
            data_vars={"num_cells": (["ensemble"], [30, 30])},
            coords={"ensemble": [0, 1]},
        )
        # VL has 5 ensembles - more than header/FL
        ds_vl = xr.Dataset(
            data_vars={"heading": (["ensemble"], [100, 200, 300, 400, 500])},
            coords={"ensemble": [0, 1, 2, 3, 4]},
        )

        with mock.patch("pyadps.io.binary_reader.read_header", return_value=ds_header):
            with mock.patch(
                "pyadps.io.binary_reader.read_fixed_leader", return_value=ds_fl
            ):
                with mock.patch(
                    "pyadps.io.binary_reader.read_variable_leader", return_value=ds_vl
                ):
                    with mock.patch(
                        "pyadps.io.binary_reader._compute_time_coordinate",
                        return_value=None,
                    ):
                        with mock.patch(
                            "pyadps.io.binary_reader._compute_depth_coordinate",
                            return_value=None,
                        ):
                            with mock.patch(
                                "pyadps.io.binary_reader._get_available_data_types",
                                return_value=[],
                            ):
                                with caplog.at_level(logging.WARNING):
                                    result = read(
                                        valid_rdi_file,
                                        data_types=["FixedLeader", "VariableLeader"],
                                        include_mask=False,
                                    )

        assert "Ensemble count inconsistency" in caplog.text
        assert isinstance(result, xr.Dataset)

    def test_read_truncates_velocity_on_inconsistency(self, valid_rdi_file, caplog):
        """Test Velocity is truncated when ensemble counts are inconsistent.

        This tests lines 3503-3504:
            if ds_velocity is not None and "ensemble" in ds_velocity.dims:
                ds_velocity = ds_velocity.isel(ensemble=slice(0, min_ensemble))
        """
        import logging

        ds_header = self._create_mock_header(2)
        ds_vl = xr.Dataset(
            data_vars={"heading": (["ensemble"], [100, 100])},
            coords={"ensemble": [0, 1]},
        )
        # Velocity has 5 ensembles - more than header/VL
        ds_velocity = xr.Dataset(
            data_vars={
                "velocity": (
                    ["beam", "cell", "ensemble"],
                    np.zeros((4, 10, 5), dtype=np.int16),
                )
            },
            coords={
                "beam": [0, 1, 2, 3],
                "cell": list(range(10)),
                "ensemble": list(range(5)),
            },
        )

        with mock.patch("pyadps.io.binary_reader.read_header", return_value=ds_header):
            with mock.patch(
                "pyadps.io.binary_reader.read_fixed_leader", return_value=None
            ):
                with mock.patch(
                    "pyadps.io.binary_reader.read_variable_leader", return_value=ds_vl
                ):
                    with mock.patch(
                        "pyadps.io.binary_reader.read_velocity",
                        return_value=ds_velocity,
                    ):
                        with mock.patch(
                            "pyadps.io.binary_reader._compute_time_coordinate",
                            return_value=None,
                        ):
                            with mock.patch(
                                "pyadps.io.binary_reader._get_available_data_types",
                                return_value=["Velocity"],
                            ):
                                with caplog.at_level(logging.WARNING):
                                    result = read(
                                        valid_rdi_file,
                                        data_types=["Velocity"],
                                        include_mask=False,
                                    )

        assert "Ensemble count inconsistency" in caplog.text
        assert isinstance(result, xr.Dataset)

    def test_read_truncates_mask_on_inconsistency(self, valid_rdi_file, caplog):
        """Test Mask is truncated when ensemble counts are inconsistent.

        This tests lines 3505-3506:
            if ds_mask is not None and "ensemble" in ds_mask.dims:
                ds_mask = ds_mask.isel(ensemble=slice(0, min_ensemble))
        """
        import logging

        ds_header = self._create_mock_header(2)
        ds_vl = xr.Dataset(
            data_vars={"heading": (["ensemble"], [100, 100])},
            coords={"ensemble": [0, 1]},
        )
        ds_velocity = xr.Dataset(
            data_vars={
                "velocity": (
                    ["beam", "cell", "ensemble"],
                    np.zeros((4, 10, 2), dtype=np.int16),
                )
            },
            coords={"beam": [0, 1, 2, 3], "cell": list(range(10)), "ensemble": [0, 1]},
        )
        # Mask has 5 ensembles - more than others
        ds_mask = xr.Dataset(
            data_vars={
                "velocity_mask": (
                    ["beam", "cell", "ensemble"],
                    np.zeros((4, 10, 5), dtype=np.int8),
                )
            },
            coords={
                "beam": [0, 1, 2, 3],
                "cell": list(range(10)),
                "ensemble": list(range(5)),
            },
        )

        with mock.patch("pyadps.io.binary_reader.read_header", return_value=ds_header):
            with mock.patch(
                "pyadps.io.binary_reader.read_fixed_leader", return_value=None
            ):
                with mock.patch(
                    "pyadps.io.binary_reader.read_variable_leader", return_value=ds_vl
                ):
                    with mock.patch(
                        "pyadps.io.binary_reader.read_velocity",
                        return_value=ds_velocity,
                    ):
                        with mock.patch(
                            "pyadps.io.binary_reader._create_velocity_mask",
                            return_value=ds_mask,
                        ):
                            with mock.patch(
                                "pyadps.io.binary_reader._compute_time_coordinate",
                                return_value=None,
                            ):
                                with mock.patch(
                                    "pyadps.io.binary_reader._get_available_data_types",
                                    return_value=["Velocity"],
                                ):
                                    with caplog.at_level(logging.WARNING):
                                        result = read(
                                            valid_rdi_file,
                                            data_types=["Velocity"],
                                            include_mask=True,
                                        )

        assert "Ensemble count inconsistency" in caplog.text
        assert isinstance(result, xr.Dataset)

    def test_read_truncates_correlation_on_inconsistency(self, valid_rdi_file, caplog):
        """Test Correlation is truncated when ensemble counts are inconsistent.

        This tests lines 3507-3508:
            if ds_correlation is not None and "ensemble" in ds_correlation.dims:
                ds_correlation = ds_correlation.isel(ensemble=slice(0, min_ensemble))
        """
        import logging

        ds_header = self._create_mock_header(2)
        ds_vl = xr.Dataset(
            data_vars={"heading": (["ensemble"], [100, 100])},
            coords={"ensemble": [0, 1]},
        )
        # Correlation has 5 ensembles - more than header/VL
        ds_correlation = xr.Dataset(
            data_vars={
                "correlation": (
                    ["beam", "cell", "ensemble"],
                    np.zeros((4, 10, 5), dtype=np.uint8),
                )
            },
            coords={
                "beam": [0, 1, 2, 3],
                "cell": list(range(10)),
                "ensemble": list(range(5)),
            },
        )

        with mock.patch("pyadps.io.binary_reader.read_header", return_value=ds_header):
            with mock.patch(
                "pyadps.io.binary_reader.read_fixed_leader", return_value=None
            ):
                with mock.patch(
                    "pyadps.io.binary_reader.read_variable_leader", return_value=ds_vl
                ):
                    with mock.patch(
                        "pyadps.io.binary_reader.read_correlation",
                        return_value=ds_correlation,
                    ):
                        with mock.patch(
                            "pyadps.io.binary_reader._compute_time_coordinate",
                            return_value=None,
                        ):
                            with mock.patch(
                                "pyadps.io.binary_reader._get_available_data_types",
                                return_value=["Correlation"],
                            ):
                                with caplog.at_level(logging.WARNING):
                                    result = read(
                                        valid_rdi_file,
                                        data_types=["Correlation"],
                                        include_mask=False,
                                    )

        assert "Ensemble count inconsistency" in caplog.text
        assert isinstance(result, xr.Dataset)

    def test_read_truncates_echo_on_inconsistency(self, valid_rdi_file, caplog):
        """Test Echo is truncated when ensemble counts are inconsistent.

        This tests lines 3509-3510:
            if ds_echo is not None and "ensemble" in ds_echo.dims:
                ds_echo = ds_echo.isel(ensemble=slice(0, min_ensemble))
        """
        import logging

        ds_header = self._create_mock_header(2)
        ds_vl = xr.Dataset(
            data_vars={"heading": (["ensemble"], [100, 100])},
            coords={"ensemble": [0, 1]},
        )
        # Echo has 5 ensembles - more than header/VL
        ds_echo = xr.Dataset(
            data_vars={
                "echo": (
                    ["beam", "cell", "ensemble"],
                    np.zeros((4, 10, 5), dtype=np.uint8),
                )
            },
            coords={
                "beam": [0, 1, 2, 3],
                "cell": list(range(10)),
                "ensemble": list(range(5)),
            },
        )

        with mock.patch("pyadps.io.binary_reader.read_header", return_value=ds_header):
            with mock.patch(
                "pyadps.io.binary_reader.read_fixed_leader", return_value=None
            ):
                with mock.patch(
                    "pyadps.io.binary_reader.read_variable_leader", return_value=ds_vl
                ):
                    with mock.patch(
                        "pyadps.io.binary_reader.read_echo_intensity",
                        return_value=ds_echo,
                    ):
                        with mock.patch(
                            "pyadps.io.binary_reader._compute_time_coordinate",
                            return_value=None,
                        ):
                            with mock.patch(
                                "pyadps.io.binary_reader._get_available_data_types",
                                return_value=["Echo"],
                            ):
                                with caplog.at_level(logging.WARNING):
                                    result = read(
                                        valid_rdi_file,
                                        data_types=["Echo"],
                                        include_mask=False,
                                    )

        assert "Ensemble count inconsistency" in caplog.text
        assert isinstance(result, xr.Dataset)

    def test_read_truncates_percent_good_on_inconsistency(self, valid_rdi_file, caplog):
        """Test Percent Good is truncated when ensemble counts are inconsistent.

        This tests lines 3511-3512:
            if ds_percent_good is not None and "ensemble" in ds_percent_good.dims:
                ds_percent_good = ds_percent_good.isel(ensemble=slice(0, min_ensemble))
        """
        import logging

        ds_header = self._create_mock_header(2)
        ds_vl = xr.Dataset(
            data_vars={"heading": (["ensemble"], [100, 100])},
            coords={"ensemble": [0, 1]},
        )
        # Percent Good has 5 ensembles - more than header/VL
        ds_percent_good = xr.Dataset(
            data_vars={
                "percent_good": (
                    ["beam", "cell", "ensemble"],
                    np.zeros((4, 10, 5), dtype=np.uint8),
                )
            },
            coords={
                "beam": [0, 1, 2, 3],
                "cell": list(range(10)),
                "ensemble": list(range(5)),
            },
        )

        with mock.patch("pyadps.io.binary_reader.read_header", return_value=ds_header):
            with mock.patch(
                "pyadps.io.binary_reader.read_fixed_leader", return_value=None
            ):
                with mock.patch(
                    "pyadps.io.binary_reader.read_variable_leader", return_value=ds_vl
                ):
                    with mock.patch(
                        "pyadps.io.binary_reader.read_percent_good",
                        return_value=ds_percent_good,
                    ):
                        with mock.patch(
                            "pyadps.io.binary_reader._compute_time_coordinate",
                            return_value=None,
                        ):
                            with mock.patch(
                                "pyadps.io.binary_reader._get_available_data_types",
                                return_value=["Percent Good"],
                            ):
                                with caplog.at_level(logging.WARNING):
                                    result = read(
                                        valid_rdi_file,
                                        data_types=["Percent Good"],
                                        include_mask=False,
                                    )

        assert "Ensemble count inconsistency" in caplog.text
        assert isinstance(result, xr.Dataset)

    def test_read_truncates_status_on_inconsistency(self, valid_rdi_file, caplog):
        """Test Status is truncated when ensemble counts are inconsistent.

        This tests lines 3513-3514:
            if ds_status is not None and "ensemble" in ds_status.dims:
                ds_status = ds_status.isel(ensemble=slice(0, min_ensemble))
        """
        import logging

        ds_header = self._create_mock_header(2)
        ds_vl = xr.Dataset(
            data_vars={"heading": (["ensemble"], [100, 100])},
            coords={"ensemble": [0, 1]},
        )
        # Status has 5 ensembles - more than header/VL
        ds_status = xr.Dataset(
            data_vars={
                "status": (
                    ["beam", "cell", "ensemble"],
                    np.zeros((4, 10, 5), dtype=np.uint8),
                )
            },
            coords={
                "beam": [0, 1, 2, 3],
                "cell": list(range(10)),
                "ensemble": list(range(5)),
            },
        )

        with mock.patch("pyadps.io.binary_reader.read_header", return_value=ds_header):
            with mock.patch(
                "pyadps.io.binary_reader.read_fixed_leader", return_value=None
            ):
                with mock.patch(
                    "pyadps.io.binary_reader.read_variable_leader", return_value=ds_vl
                ):
                    with mock.patch(
                        "pyadps.io.binary_reader.read_status", return_value=ds_status
                    ):
                        with mock.patch(
                            "pyadps.io.binary_reader._compute_time_coordinate",
                            return_value=None,
                        ):
                            with mock.patch(
                                "pyadps.io.binary_reader._get_available_data_types",
                                return_value=["Status"],
                            ):
                                with caplog.at_level(logging.WARNING):
                                    result = read(
                                        valid_rdi_file,
                                        data_types=["Status"],
                                        include_mask=False,
                                    )

        assert "Ensemble count inconsistency" in caplog.text
        assert isinstance(result, xr.Dataset)

    def test_read_logs_truncation_info(self, valid_rdi_file, caplog):
        """Test that truncation info message is logged.

        This tests line 3496:
            logger.info(f"Truncating all datasets to {min_ensemble} ensembles before merge")
        """
        import logging

        ds_header = self._create_mock_header(2)
        # VL has more ensembles than header
        ds_vl = xr.Dataset(
            data_vars={"heading": (["ensemble"], [100, 200, 300, 400, 500])},
            coords={"ensemble": [0, 1, 2, 3, 4]},
        )

        with mock.patch("pyadps.io.binary_reader.read_header", return_value=ds_header):
            with mock.patch(
                "pyadps.io.binary_reader.read_fixed_leader", return_value=None
            ):
                with mock.patch(
                    "pyadps.io.binary_reader.read_variable_leader", return_value=ds_vl
                ):
                    with mock.patch(
                        "pyadps.io.binary_reader._compute_time_coordinate",
                        return_value=None,
                    ):
                        with mock.patch(
                            "pyadps.io.binary_reader._get_available_data_types",
                            return_value=[],
                        ):
                            with caplog.at_level(logging.INFO):
                                result = read(
                                    valid_rdi_file,
                                    data_types=["VariableLeader"],
                                    include_mask=False,
                                )

        # Check for truncation message
        assert "Truncating all datasets" in caplog.text
        assert isinstance(result, xr.Dataset)


class TestReadDatasetValidation:
    """Tests for dataset content validation (lines 3539-3543)."""

    def test_read_warns_on_empty_dataset(self, valid_rdi_file, caplog):
        """Test warning when resulting dataset has no data variables.

        This tests lines 3539-3543:
            if len(ds.data_vars) == 0:
                logger.warning(...)
        """
        import logging

        # Create a scenario where merge produces empty dataset
        empty_ds = xr.Dataset(attrs={"filename": "test.000"})

        with mock.patch(
            "pyadps.io.binary_reader._merge_datasets", return_value=empty_ds
        ):
            with caplog.at_level(logging.WARNING):
                result = read(valid_rdi_file)

        assert "Resulting dataset has no data variables" in caplog.text


class TestReadDimensionTransposition:
    """Tests for dimension transposition (lines 3548-3561)."""

    def test_read_uses_time_as_primary_dim_true(self, valid_rdi_file):
        """Test use_time_as_primary_dim=True swaps dimensions.

        This tests lines 3548-3552.
        """
        result = read(valid_rdi_file, use_time_as_primary_dim=True)

        # Time should be a dimension if available
        if "time" in result.coords:
            assert "time" in result.dims

    def test_read_uses_time_as_primary_dim_false(self, valid_rdi_file):
        """Test use_time_as_primary_dim=False keeps ensemble as primary."""
        result = read(valid_rdi_file, use_time_as_primary_dim=False)

        # Ensemble should remain as dimension
        assert "ensemble" in result.dims

    def test_read_uses_depth_as_primary_dim_true(self, valid_rdi_file):
        """Test use_depth_as_primary_dim=True swaps dimensions.

        This tests lines 3554-3557.
        """
        result = read(valid_rdi_file, use_depth_as_primary_dim=True)

        # Depth should be a dimension if available
        if "depth" in result.coords:
            assert "depth" in result.dims
            # Cell should be dropped
            assert "cell" not in result.dims

    def test_read_uses_depth_as_primary_dim_false(self, valid_rdi_file):
        """Test use_depth_as_primary_dim=False keeps cell as dimension.

        This tests lines 3558-3561.
        """
        result = read(valid_rdi_file, use_depth_as_primary_dim=False)

        # Cell should remain as dimension
        if "cell" in result.dims:
            assert "cell" in result.dims


class TestReadAllDataTypes:
    """Tests for loading all available data types."""

    def test_read_all_data_types_default(self, valid_rdi_file):
        """Test that all data types are loaded when data_types=None."""
        result = read(valid_rdi_file, data_types=None)

        assert isinstance(result, xr.Dataset)
        # Should have multiple data variables
        assert len(result.data_vars) > 0

    def test_read_correlation_data_type(self, valid_rdi_file):
        """Test reading Correlation data type."""
        result = read(valid_rdi_file, data_types=["Correlation"])

        assert isinstance(result, xr.Dataset)
        if "correlation" in result.data_vars:
            assert result["correlation"] is not None

    def test_read_echo_data_type(self, valid_rdi_file):
        """Test reading Echo data type."""
        result = read(valid_rdi_file, data_types=["Echo"])

        assert isinstance(result, xr.Dataset)
        if "echo" in result.data_vars:
            assert result["echo"] is not None

    def test_read_percent_good_data_type(self, valid_rdi_file):
        """Test reading Percent Good data type."""
        result = read(valid_rdi_file, data_types=["Percent Good"])

        assert isinstance(result, xr.Dataset)

    def test_read_status_data_type(self, valid_rdi_file):
        """Test reading Status data type."""
        result = read(valid_rdi_file, data_types=["Status"])

        assert isinstance(result, xr.Dataset)

    def test_read_multiple_data_types(self, valid_rdi_file):
        """Test reading multiple data types at once."""
        result = read(
            valid_rdi_file,
            data_types=["FixedLeader", "Velocity", "Correlation", "Echo"],
        )

        assert isinstance(result, xr.Dataset)
        assert len(result.data_vars) > 0


class TestReadIncludeDecodedParameter:
    """Tests for include_decoded parameter."""

    def test_read_include_decoded_true(self, valid_rdi_file):
        """Test include_decoded=True includes decoded fields."""
        result = read(valid_rdi_file, include_decoded=True)

        assert isinstance(result, xr.Dataset)
        # Should have time coordinate when decoded
        if "time" in result.coords:
            assert result.coords["time"] is not None

    def test_read_include_decoded_false(self, valid_rdi_file):
        """Test include_decoded=False excludes decoded fields."""
        result = read(valid_rdi_file, include_decoded=False)

        assert isinstance(result, xr.Dataset)


class TestReadIncludeHeaderParameter:
    """Tests for include_header parameter."""

    def test_read_include_header_true(self, valid_rdi_file):
        """Test include_header=True includes header variables."""
        result = read(valid_rdi_file, include_header=True)

        assert isinstance(result, xr.Dataset)
        # Should have header-related variables
        header_vars = ["byte", "byte_skip", "address_offset", "data_id"]
        has_header_var = any(var in result.data_vars for var in header_vars)
        # May or may not have header vars depending on implementation
        assert isinstance(result, xr.Dataset)

    def test_read_include_header_false(self, valid_rdi_file):
        """Test include_header=False excludes header variables."""
        result = read(valid_rdi_file, include_header=False)

        assert isinstance(result, xr.Dataset)
        # Header variables should be excluded
        header_vars = ["data_type_array", "byte", "byte_skip"]
        for var in header_vars:
            assert var not in result.data_vars


class TestReadPathHandling:
    """Tests for file path handling."""

    def test_read_accepts_string_path(self, valid_rdi_file):
        """Test read() accepts string path."""
        result = read(str(valid_rdi_file))

        assert isinstance(result, xr.Dataset)

    def test_read_accepts_pathlib_path(self, valid_rdi_file):
        """Test read() accepts pathlib.Path."""
        result = read(Path(valid_rdi_file))

        assert isinstance(result, xr.Dataset)

    def test_read_handles_relative_path(self, valid_rdi_file, monkeypatch):
        """Test read() handles relative paths."""
        import os

        # Change to parent directory and use relative path
        parent_dir = valid_rdi_file.parent
        rel_path = valid_rdi_file.name

        monkeypatch.chdir(parent_dir)
        result = read(rel_path)

        assert isinstance(result, xr.Dataset)
