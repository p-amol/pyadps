"""
Comprehensive pytest tests for filereader.read_header() function.

Tests cover:
- Input validation (valid/invalid file paths)
- Return value structure (xarray.Dataset)
- Data consistency across ensembles
- Error handling (file access, corruption)
- xarray API compatibility

Reference: RDI WorkHorse Commands and Output Data Format (Section 7, page 123)
"""

import pytest
import numpy as np
import xarray as xr
from pathlib import Path
from unittest import mock
import struct
import sys

# Add parent project directory to path (if not already added)
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyadps.io.binary_reader import read_header
# Note: accessor imports would be here if available
# import accessors  # Register accessor


# ============================================================================
# TEST CLASS 1: Input Validation
# ============================================================================


class TestReadHeaderInputValidation:
    """Test read_header() input validation."""

    def test_valid_filepath_string(self, valid_rdi_file):
        """Test read_header() with string path."""
        # Should not raise exception
        ds = read_header(str(valid_rdi_file))
        assert isinstance(ds, xr.Dataset), f"Expected xr.Dataset, got {type(ds)}"

    def test_valid_filepath_pathlib(self, valid_rdi_file):
        """Test read_header() with pathlib.Path object."""
        # Should accept Path objects
        ds = read_header(valid_rdi_file)
        assert isinstance(ds, xr.Dataset), f"Expected xr.Dataset, got {type(ds)}"

    def test_invalid_filename_type_integer(self):
        """Test that integer path raises error."""
        # Integer is not a valid path
        with pytest.raises((FileNotFoundError, TypeError, ValueError)):
            read_header(12345)

    def test_invalid_filename_type_none(self):
        """Test that None raises error."""
        with pytest.raises((FileNotFoundError, TypeError, ValueError)):
            read_header(None)

    def test_invalid_filename_type_list(self):
        """Test that list raises error."""
        with pytest.raises((FileNotFoundError, TypeError, ValueError)):
            read_header(["file.000"])

    def test_file_not_found_error(self):
        """Test handling of non-existent file."""
        # read_header may return a dataset with error, raise FileNotFoundError, or raise ValueError
        try:
            ds = read_header("nonexistent_file_xyz_12345.000")
            # If it returns a dataset, check for error indication
            assert (
                ds.attrs["total_ensembles"] == 0
                or "error" in ds.attrs["error_message"].lower()
            ), "Non-existent file should result in error"
        except (FileNotFoundError, ValueError):
            pass  # Acceptable behavior

    def test_file_not_found_with_path_object(self):
        """Test handling of non-existent file with Path object."""
        try:
            ds = read_header(Path("/nonexistent/path/file.000"))
            # If it returns a dataset, check for error indication
            assert (
                ds.attrs["total_ensembles"] == 0
                or "error" in ds.attrs["error_message"].lower()
            ), "Non-existent file should result in error"
        except (FileNotFoundError, ValueError):
            pass  # Acceptable behavior


# ============================================================================
# TEST CLASS 2: Return Structure Validation
# ============================================================================


class TestReadHeaderReturnStructure:
    """Test read_header() return value structure."""

    def test_returns_xarray_dataset(self, valid_rdi_file):
        """Test that read_header returns xr.Dataset."""
        ds = read_header(str(valid_rdi_file))
        assert isinstance(ds, xr.Dataset), f"Expected xr.Dataset, got {type(ds)}"

    def test_has_ensemble_coordinate(self, valid_rdi_file):
        """Test dataset has ensemble coordinate."""
        ds = read_header(str(valid_rdi_file))
        assert (
            "ensemble" in ds.coords
        ), f"Missing 'ensemble' coordinate. Available: {list(ds.coords)}"
        assert ds.coords["ensemble"].ndim == 1, "ensemble coordinate should be 1D"

    def test_has_data_type_coordinate(self, valid_rdi_file):
        """Test dataset has data_type coordinate."""
        ds = read_header(str(valid_rdi_file))
        assert (
            "data_type" in ds.coords
        ), f"Missing 'data_type' coordinate. Available: {list(ds.coords)}"

    def test_has_required_data_variables(self, valid_rdi_file):
        """Test dataset has all required data variables."""
        ds = read_header(str(valid_rdi_file))
        required_vars = {
            "data_type_array",
            "byte",
            "byte_skip",
            "address_offset",
            "data_id",
        }
        available_vars = set(ds.data_vars)
        missing = required_vars - available_vars
        assert (
            not missing
        ), f"Missing data variables: {missing}. Available: {available_vars}"

    def test_has_required_attributes(self, valid_rdi_file):
        """Test dataset has all required attributes."""
        ds = read_header(str(valid_rdi_file))
        required_attrs = {
            "filename",
            "total_ensembles",
            "num_data_types",
            "error_message",
            "file_size_bytes",
            "adcp_data_format",
            "pyadps_component",
            "pyadps_version",
        }
        available_attrs = set(ds.attrs)
        missing = required_attrs - available_attrs
        assert (
            not missing
        ), f"Missing attributes: {missing}. Available: {available_attrs}"

    def test_data_variable_dtypes(self, valid_rdi_file):
        """Test that data variables have correct dtypes."""
        ds = read_header(str(valid_rdi_file))

        assert (
            ds["byte"].dtype == np.int32
        ), f"byte dtype should be int32, got {ds['byte'].dtype}"
        assert (
            ds["byte_skip"].dtype == np.int32
        ), f"byte_skip dtype should be int32, got {ds['byte_skip'].dtype}"
        assert (
            ds["data_type_array"].dtype == np.int16
        ), f"data_type_array dtype should be int16, got {ds['data_type_array'].dtype}"
        assert (
            ds["address_offset"].dtype == np.int16
        ), f"address_offset dtype should be int16, got {ds['address_offset'].dtype}"
        assert (
            ds["data_id"].dtype == np.int16
        ), f"data_id dtype should be int16, got {ds['data_id'].dtype}"

    def test_address_offset_is_2d(self, valid_rdi_file):
        """Test that address_offset is 2D array."""
        ds = read_header(str(valid_rdi_file))
        assert (
            ds["address_offset"].ndim == 2
        ), f"address_offset should be 2D, got {ds['address_offset'].ndim}D"
        assert (
            ds["address_offset"].shape[0] > 0
        ), "address_offset should have at least 1 ensemble"
        assert (
            ds["address_offset"].shape[1] > 0
        ), "address_offset should have at least 1 datatype"

    def test_data_id_is_2d(self, valid_rdi_file):
        """Test that data_id is 2D array."""
        ds = read_header(str(valid_rdi_file))
        assert (
            ds["data_id"].ndim == 2
        ), f"data_id should be 2D, got {ds['data_id'].ndim}D"

    def test_attribute_types(self, valid_rdi_file):
        """Test that attributes have correct types."""
        ds = read_header(str(valid_rdi_file))

        assert isinstance(ds.attrs["filename"], str)
        assert isinstance(ds.attrs["total_ensembles"], int)
        assert isinstance(ds.attrs["num_data_types"], int)
        assert isinstance(ds.attrs["error_message"], str)
        assert isinstance(ds.attrs["file_size_bytes"], int)
        assert isinstance(ds.attrs["adcp_data_format"], str)
        assert isinstance(ds.attrs["pyadps_component"], str)
        assert isinstance(ds.attrs["pyadps_version"], str)


# ============================================================================
# TEST CLASS 3: Data Consistency
# ============================================================================


class TestReadHeaderDataConsistency:
    """Test read_header() data consistency."""

    def test_ensemble_count_matches_data_shapes(self, multi_ensemble_rdi_file):
        """Test ensemble coordinate matches data variable shapes."""
        ds = read_header(str(multi_ensemble_rdi_file))
        n_ensembles = len(ds.coords["ensemble"])

        assert (
            ds["data_type_array"].shape[0] == n_ensembles
        ), "data_type_array shape doesn't match ensemble count"
        assert (
            ds["byte"].shape[0] == n_ensembles
        ), "byte shape doesn't match ensemble count"
        assert (
            ds["byte_skip"].shape[0] == n_ensembles
        ), "byte_skip shape doesn't match ensemble count"
        assert (
            ds["address_offset"].shape[0] == n_ensembles
        ), "address_offset first dimension doesn't match ensemble count"

    def test_file_size_bytes_is_positive(self, valid_rdi_file):
        """Test that file_size_bytes attribute is positive."""
        ds = read_header(str(valid_rdi_file))
        assert (
            ds.attrs["file_size_bytes"] > 0
        ), f"file_size_bytes should be positive, got {ds.attrs['file_size_bytes']}"

    def test_byte_values_are_positive(self, valid_rdi_file):
        """Test that all byte counts are positive."""
        ds = read_header(str(valid_rdi_file))
        assert np.all(ds["byte"].values > 0), "All byte values should be positive"

    def test_byte_skip_is_cumulative(self, multi_ensemble_rdi_file):
        """Test that byte_skip increases cumulatively."""
        ds = read_header(str(multi_ensemble_rdi_file))
        byte_skip = ds["byte_skip"].values

        # Each subsequent byte_skip should be >= previous (cumulative)
        for i in range(1, len(byte_skip)):
            assert (
                byte_skip[i] >= byte_skip[i - 1]
            ), f"byte_skip not cumulative at index {i}: {byte_skip[i-1]} -> {byte_skip[i]}"

    def test_ensemble_numbers_increment(self, multi_ensemble_rdi_file):
        """Test that ensemble coordinates increment properly."""
        ds = read_header(str(multi_ensemble_rdi_file))
        ensemble_values = ds.coords["ensemble"].values
        expected = np.arange(1, len(ensemble_values) + 1)

        np.testing.assert_array_equal(
            ensemble_values,
            expected,
            err_msg="Ensemble numbers don't increment properly",
        )

    def test_total_ensembles_matches_coordinate(self, multi_ensemble_rdi_file):
        """Test that total_ensembles attribute matches coordinate."""
        ds = read_header(str(multi_ensemble_rdi_file))
        assert ds.attrs["total_ensembles"] == len(
            ds.coords["ensemble"]
        ), "total_ensembles attribute doesn't match ensemble coordinate"

    def test_num_data_types_matches_coordinate(self, valid_rdi_file):
        """Test that num_data_types attribute matches coordinate."""
        ds = read_header(str(valid_rdi_file))
        assert ds.attrs["num_data_types"] == len(
            ds.coords["data_type"]
        ), "num_data_types attribute doesn't match data_type coordinate"

    def test_file_size_bytes_matches_actual(self, valid_rdi_file):
        """Test that file_size_bytes matches actual file size."""
        ds = read_header(str(valid_rdi_file))
        actual_size = valid_rdi_file.stat().st_size
        assert (
            ds.attrs["file_size_bytes"] == actual_size
        ), f"file_size_bytes {ds.attrs['file_size_bytes']} != actual {actual_size}"


# ============================================================================
# TEST CLASS 4: Error Handling
# ============================================================================


class TestReadHeaderErrorHandling:
    """Test read_header() error handling."""

    def test_truncated_file_behavior(self, truncated_rdi_file):
        """Test handling of truncated file."""
        # Either raises error or returns dataset with error message
        try:
            ds = read_header(str(truncated_rdi_file))
            # If no error, should indicate error in attributes
            assert (
                "error" in ds.attrs["error_message"].lower()
                or ds.attrs["error_message"] != "OK"
            ), "Truncated file should have error in message"
        except (FileNotFoundError, ValueError, struct.error):
            pass  # Acceptable: error raised is OK

    def test_empty_file_behavior(self, empty_file):
        """Test handling of empty file."""
        try:
            ds = read_header(str(empty_file))
            # If no error, ensembles should be 0 or minimal
            assert (
                ds.attrs["total_ensembles"] == 0
                or "error" in ds.attrs["error_message"].lower()
            ), "Empty file should have 0 ensembles or error message"
        except (FileNotFoundError, ValueError):
            pass  # Acceptable: error raised is OK

    def test_invalid_header_id_detected(self, invalid_header_id_file):
        """Test detection of invalid header ID."""
        try:
            ds = read_header(str(invalid_header_id_file))
            # Should have error message indicating bad format
            assert (
                "error" in ds.attrs["error_message"].lower()
                or ds.attrs["total_ensembles"] == 0
            ), "Invalid header should result in error or 0 ensembles"
        except (ValueError, struct.error):
            pass  # Acceptable: error raised is OK

    def test_permission_denied_mock(self, valid_rdi_file):
        """Test handling of permission denied via mocking."""
        # Mock the fileheader function call within the filereader module
        with mock.patch(
            "pyadps.io.binary_reader.pd0_parser.fileheader"
        ) as mock_fileheader:
            mock_fileheader.side_effect = PermissionError("Permission denied")

            # Should raise or handle the permission error gracefully
            try:
                ds = read_header(str(valid_rdi_file))
                # If it returns, check for error indication
                assert (
                    "error" in ds.attrs["error_message"].lower()
                    or ds.attrs["total_ensembles"] == 0
                )
            except (PermissionError, OSError):
                pass  # Acceptable behavior

    def test_file_size_read_error(self, valid_rdi_file, caplog):
        """Test handling of file size read error after successful parsing.

        This tests the exception handler at lines 243-244 in read_header():
            except (OSError, IOError) as e:
                logger.warning(f"Could not read file size for {adcp_file}: {e}")

        The file is successfully parsed, but Path.stat() fails when trying
        to read the file size (e.g., file deleted between operations, or
        permission changed).
        """
        import logging

        # Mock Path.stat() to raise OSError after fileheader succeeds
        original_stat = Path.stat

        def mock_stat(self):
            # Only fail for our test file
            if str(valid_rdi_file) in str(self):
                raise OSError("Mocked file stat error")
            return original_stat(self)

        with mock.patch.object(Path, "stat", mock_stat):
            with caplog.at_level(logging.WARNING):
                ds = read_header(str(valid_rdi_file))

        # Should still return a valid dataset
        assert isinstance(ds, xr.Dataset), f"Expected xr.Dataset, got {type(ds)}"

        # file_size_bytes should be -1 (default error value)
        assert (
            ds.attrs["file_size_bytes"] == -1
        ), f"Expected file_size_bytes=-1 on error, got {ds.attrs['file_size_bytes']}"

        # Warning should be logged
        assert "Could not read file size" in caplog.text

    def test_file_size_ioerror(self, valid_rdi_file, caplog):
        """Test handling of IOError when reading file size.

        Similar to test_file_size_read_error but specifically tests IOError.
        """
        import logging

        original_stat = Path.stat

        def mock_stat(self):
            if str(valid_rdi_file) in str(self):
                raise IOError("Mocked IO error reading file")
            return original_stat(self)

        with mock.patch.object(Path, "stat", mock_stat):
            with caplog.at_level(logging.WARNING):
                ds = read_header(str(valid_rdi_file))

        # Should still return a valid dataset
        assert isinstance(ds, xr.Dataset)

        # file_size_bytes should be -1 (default error value)
        assert ds.attrs["file_size_bytes"] == -1

        # Warning should be logged
        assert "Could not read file size" in caplog.text


# ============================================================================
# TEST CLASS 5: xarray Compatibility
# ============================================================================


class TestReadHeaderXarrayCompatibility:
    """Test xarray API compatibility."""

    def test_can_select_single_ensemble(self, multi_ensemble_rdi_file):
        """Test xarray .sel() method for ensemble selection."""
        ds = read_header(str(multi_ensemble_rdi_file))
        first_ensemble = ds.sel(ensemble=1)

        assert isinstance(
            first_ensemble, xr.Dataset
        ), f"Expected Dataset, got {type(first_ensemble)}"
        assert first_ensemble["byte"].ndim == 0, "Selected ensemble should be scalar"

    def test_can_select_ensemble_range(self, multi_ensemble_rdi_file):
        """Test xarray .isel() method for ensemble slicing."""
        ds = read_header(str(multi_ensemble_rdi_file))
        subset = ds.isel(ensemble=slice(0, 2))

        assert isinstance(subset, xr.Dataset), f"Expected Dataset, got {type(subset)}"
        # Use .sizes instead of .dims for future compatibility
        assert (
            subset.sizes["ensemble"] == 2
        ), f"Expected 2 ensembles in subset, got {subset.sizes['ensemble']}"

    def test_dataset_arithmetic(self, valid_rdi_file):
        """Test basic arithmetic operations on dataset."""
        ds = read_header(str(valid_rdi_file))

        # Should allow basic math operations
        result = ds["byte"] * 2
        assert isinstance(
            result, xr.DataArray
        ), f"Expected DataArray, got {type(result)}"
        np.testing.assert_array_equal(
            result.values, ds["byte"].values * 2, err_msg="Arithmetic result incorrect"
        )

    def test_dataset_mean_operation(self, multi_ensemble_rdi_file):
        """Test xarray mean reduction."""
        ds = read_header(str(multi_ensemble_rdi_file))

        # Should be able to compute mean across ensembles
        mean_byte = ds["byte"].mean(dim="ensemble")
        assert isinstance(
            mean_byte, xr.DataArray
        ), f"Expected DataArray, got {type(mean_byte)}"
        assert mean_byte.ndim == 0, f"Mean should be scalar, got {mean_byte.ndim}D"

    def test_dataset_groupby_operation(self, multi_ensemble_rdi_file):
        """Test xarray groupby operations."""
        ds = read_header(str(multi_ensemble_rdi_file))

        # Should support groupby operations
        # Create a simple groupby: ensemble % 2
        groups = ds.groupby(ds["byte"] > 0)
        assert len(groups.groups) > 0, "Groupby should produce groups"

    def test_dataset_attrs_preserved(self, valid_rdi_file):
        """Test that attributes are preserved in subset operations."""
        ds = read_header(str(valid_rdi_file))
        subset = ds.isel(ensemble=slice(0, 1))

        # Attributes should be preserved
        assert "filename" in subset.attrs, "Attributes not preserved in subset"
        assert (
            subset.attrs["filename"] == ds.attrs["filename"]
        ), "Attribute values changed in subset"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
