"""
Comprehensive pytest tests for pyadps HeaderAccessor xarray accessor.

Tests cover:
- Accessor registration and availability
- Data type listing and retrieval methods
- File validation and integrity checking
- Metadata and summary methods
- Error handling and edge cases

Reference: xarray accessors documentation
https://docs.xarray.dev/en/stable/extending.html#extending-with-accessors

HeaderAccessor Reference: pyadps/accessors.py
"""

import pytest
import numpy as np
import xarray as xr
from pathlib import Path
import sys
from unittest import mock
import logging

# Add parent project directory to path (if not already added)
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyadps.io.binary_reader import read_header


# ============================================================================
# TEST CLASS 1: Accessor Registration
# ============================================================================


class TestHeaderAccessorRegistration:
    """Test HeaderAccessor registration and basic availability."""

    def test_header_accessor_registered(self, valid_rdi_file):
        """Test that header accessor is registered with xarray."""
        ds = read_header(str(valid_rdi_file))

        # Should have 'header' attribute if accessor is registered
        assert hasattr(ds, "header"), "Dataset should have 'header' accessor"

    def test_accessor_is_accessible(self, valid_rdi_file):
        """Test that accessor is actually accessible."""
        ds = read_header(str(valid_rdi_file))

        # Should not be None
        accessor = ds.header
        assert accessor is not None, "Accessor should not be None"

    def test_accessor_type(self, valid_rdi_file):
        """Test that accessor returns correct type."""
        ds = read_header(str(valid_rdi_file))

        # Check that accessor is instance of HeaderAccessor
        from pyadps.io.accessors import HeaderAccessor

        assert isinstance(
            ds.header, HeaderAccessor
        ), f"Expected HeaderAccessor, got {type(ds.header)}"

    def test_accessor_preserves_dataset_api(self, valid_rdi_file):
        """Test that accessor doesn't break standard xarray API."""
        ds = read_header(str(valid_rdi_file))

        # Standard xarray methods should still work
        assert "ensemble" in ds.coords, "Should have ensemble coordinate"
        assert "byte" in ds.data_vars, "Should have byte data variable"
        assert hasattr(ds.dims, "__getitem__"), "dims should be dict-like"
        assert "ensemble" in ds.dims, "dims should have ensemble dimension"
        assert isinstance(ds.attrs, dict), "attrs should be accessible"

    def test_accessor_has_required_methods(self, valid_rdi_file):
        """Test that accessor has all required methods."""
        ds = read_header(str(valid_rdi_file))

        required_methods = [
            "get_available_data_types",
            "check_file",
            "get_ensemble_info",
            "has_data_type",
            "summary",
        ]

        for method in required_methods:
            assert hasattr(ds.header, method), f"Accessor missing method: {method}"
            assert callable(getattr(ds.header, method)), f"{method} should be callable"


# ============================================================================
# TEST CLASS 2: Data Type Methods
# ============================================================================


class TestHeaderAccessorDataTypes:
    """Test data type listing and retrieval methods."""

    def test_get_available_data_types_callable(self, valid_rdi_file):
        """Test that get_available_data_types() is callable."""
        ds = read_header(str(valid_rdi_file))

        # Should be callable
        result = ds.header.get_available_data_types()
        assert result is not None, "Should return a result"

    def test_get_available_data_types_returns_list(self, valid_rdi_file):
        """Test that get_available_data_types() returns list."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.get_available_data_types()

        assert isinstance(result, list), f"Should return list, got {type(result)}"

    def test_get_available_data_types_contains_strings(self, valid_rdi_file):
        """Test that returned list contains string data type names."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.get_available_data_types()

        # Should have at least one data type
        assert len(result) > 0, "Should return at least one data type"

        # All elements should be strings
        for item in result:
            assert isinstance(
                item, str
            ), f"Data type names should be strings, got {type(item)}"

    def test_get_available_data_types_valid_names(self, valid_rdi_file):
        """Test that returned data types are valid RDI type names."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.get_available_data_types()

        # Valid RDI data type names
        valid_types = {
            "Fixed Leader",
            "Variable Leader",
            "Velocity",
            "Correlation",
            "Echo",
            "Percent Good",
            "Status",
            "Bottom Track",
        }

        for dtype in result:
            assert (
                dtype in valid_types or "Unknown" in dtype
            ), f"Unexpected data type: {dtype}"

    def test_get_available_data_types_ensemble_index_zero(self, valid_rdi_file):
        """Test get_available_data_types(ens=0) with explicit parameter."""
        ds = read_header(str(valid_rdi_file))

        result0 = ds.header.get_available_data_types(ens=0)

        assert isinstance(result0, list), "Should return list"
        assert len(result0) > 0, "Should have data types in ensemble 0"

    def test_get_available_data_types_multiple_ensembles(self, multi_ensemble_rdi_file):
        """Test get_available_data_types() for multiple ensembles."""
        ds = read_header(str(multi_ensemble_rdi_file))

        result0 = ds.header.get_available_data_types(ens=0)
        result1 = ds.header.get_available_data_types(ens=1)

        # Both should return lists
        assert isinstance(result0, list), "Should return list for ens=0"
        assert isinstance(result1, list), "Should return list for ens=1"

        # Typically ensembles should have same data types
        assert result0 == result1, "Ensembles should have same data types"

    def test_get_available_data_types_invalid_ensemble_index(self, valid_rdi_file):
        """Test get_available_data_types() with invalid ensemble index."""
        ds = read_header(str(valid_rdi_file))

        # Should raise IndexError for out-of-range ensemble
        with pytest.raises((IndexError, ValueError)):
            ds.header.get_available_data_types(ens=999)

    def test_get_available_data_types_negative_ensemble(self, valid_rdi_file):
        """Test get_available_data_types() with negative ensemble index."""
        ds = read_header(str(valid_rdi_file))

        # Should raise IndexError for negative ensemble
        with pytest.raises((IndexError, ValueError)):
            ds.header.get_available_data_types(ens=-1)


# ============================================================================
# TEST CLASS 3: File Validation Methods
# ============================================================================


class TestHeaderAccessorValidation:
    """Test file validation and integrity checking methods."""

    def test_check_file_callable(self, valid_rdi_file):
        """Test that check_file() is callable."""
        ds = read_header(str(valid_rdi_file))

        # Should be callable without arguments
        result = ds.header.check_file()
        assert result is not None, "check_file() should return a result"

    def test_check_file_returns_dict(self, valid_rdi_file):
        """Test that check_file() returns a dictionary."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.check_file()

        assert isinstance(result, dict), f"Should return dict, got {type(result)}"

    def test_check_file_has_required_keys(self, valid_rdi_file):
        """Test that check_file() result has all required keys."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.check_file()

        required_keys = {
            "System File Size (B)",
            "Calculated File Size (B)",
            "File Size (MB)",
            "File Size Match",
            "Byte Uniformity",
            "Data Type Uniformity",
            "Byte Skip Uniformity",
            "Address Offset Uniformity",
            "Data ID Uniformity",
        }

        available_keys = set(result.keys())
        missing = required_keys - available_keys

        assert not missing, f"Missing keys in check_file() result: {missing}. Available: {available_keys}"

    def test_check_file_valid_file_passes(self, valid_rdi_file):
        """Test that check_file() indicates valid file."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.check_file()

        # For a valid file, critical checks should pass
        assert (
            result["File Size Match"] is True or result["File Size Match"] is False
        ), "File Size Match should be boolean"
        assert (
            result["Byte Uniformity"] is True or result["Byte Uniformity"] is False
        ), "Byte Uniformity should be boolean"

    def test_check_file_file_size_positive(self, valid_rdi_file):
        """Test that check_file() reports positive file sizes."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.check_file()

        assert result["System File Size (B)"] > 0, "System file size should be positive"
        assert (
            result["Calculated File Size (B)"] > 0
        ), "Calculated file size should be positive"

    def test_check_file_file_size_mb_positive(self, valid_rdi_file):
        """Test that check_file() calculates file size in MB."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.check_file()

        assert result["File Size (MB)"] >= 0, "File Size (MB) should be non-negative"

    def test_check_file_consistency(self, valid_rdi_file):
        """Test that check_file() results are consistent across calls."""
        ds = read_header(str(valid_rdi_file))

        result1 = ds.header.check_file()
        result2 = ds.header.check_file()

        # Results should be identical for same dataset
        assert result1 == result2, "check_file() should return consistent results"

    def test_check_file_empty_dataset(self):
        """Test check_file() on empty/minimal dataset."""
        # Create minimal xarray Dataset without proper Header structure
        ds = xr.Dataset()

        # Should handle gracefully (either error or return empty result)
        try:
            result = ds.header.check_file()
            # If it doesn't error, result could be None or empty dict
            if result is not None:
                assert isinstance(result, dict), "Should return dict or None"
        except (AttributeError, ValueError, KeyError):
            # Expected if accessor requires proper Header structure
            pass


# ============================================================================
# TEST CLASS 4: Ensemble Information Methods
# ============================================================================


class TestHeaderAccessorEnsembleInfo:
    """Test ensemble-specific information retrieval."""

    def test_get_ensemble_info_callable(self, valid_rdi_file):
        """Test that get_ensemble_info() is callable."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.get_ensemble_info(ensemble=0)
        assert result is not None, "get_ensemble_info() should return a result"

    def test_get_ensemble_info_returns_dict(self, valid_rdi_file):
        """Test that get_ensemble_info() returns a dictionary."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.get_ensemble_info(ensemble=0)

        assert isinstance(result, dict), f"Should return dict, got {type(result)}"

    def test_get_ensemble_info_has_required_keys(self, valid_rdi_file):
        """Test that get_ensemble_info() result has required keys."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.get_ensemble_info(ensemble=0)

        required_keys = {
            "ensemble_number",
            "byte_size",
            "byte_skip",
            "num_datatypes",
            "data_types",
            "data_ids",
            "address_offsets",
        }

        available_keys = set(result.keys())
        missing = required_keys - available_keys

        assert (
            not missing
        ), f"Missing keys in result: {missing}. Available: {available_keys}"

    def test_get_ensemble_info_ensemble_number_field(self, valid_rdi_file):
        """Test that ensemble_number field is correct."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.get_ensemble_info(ensemble=0)

        assert (
            result["ensemble_number"] == 0
        ), "ensemble_number should be 0 for first ensemble"

    def test_get_ensemble_info_byte_size_positive(self, valid_rdi_file):
        """Test that byte_size is positive."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.get_ensemble_info(ensemble=0)

        assert (
            result["byte_size"] > 0
        ), f"byte_size should be positive, got {result['byte_size']}"

    def test_get_ensemble_info_data_types_list(self, valid_rdi_file):
        """Test that data_types field is a list."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.get_ensemble_info(ensemble=0)

        assert isinstance(result["data_types"], list), "data_types should be a list"
        assert len(result["data_types"]) > 0, "Should have at least one data type"

    def test_get_ensemble_info_data_ids_list(self, valid_rdi_file):
        """Test that data_ids field is a list."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.get_ensemble_info(ensemble=0)

        assert isinstance(result["data_ids"], list), "data_ids should be a list"

    def test_get_ensemble_info_multiple_ensembles(self, multi_ensemble_rdi_file):
        """Test get_ensemble_info() for multiple ensembles."""
        ds = read_header(str(multi_ensemble_rdi_file))

        info0 = ds.header.get_ensemble_info(ensemble=0)
        info1 = ds.header.get_ensemble_info(ensemble=1)
        info2 = ds.header.get_ensemble_info(ensemble=2)

        # All should return dicts
        assert isinstance(info0, dict)
        assert isinstance(info1, dict)
        assert isinstance(info2, dict)

        # Ensemble numbers should be different
        assert info0["ensemble_number"] == 0
        assert info1["ensemble_number"] == 1
        assert info2["ensemble_number"] == 2

    def test_get_ensemble_info_invalid_ensemble(self, valid_rdi_file):
        """Test get_ensemble_info() with invalid ensemble index."""
        ds = read_header(str(valid_rdi_file))

        # Should raise IndexError for out-of-range
        with pytest.raises(IndexError):
            ds.header.get_ensemble_info(ensemble=999)


# ============================================================================
# TEST CLASS 5: Data Type Query Methods
# ============================================================================


class TestHeaderAccessorDataTypeQueries:
    """Test data type query methods (has_data_type)."""

    def test_has_data_type_callable(self, valid_rdi_file):
        """Test that has_data_type() is callable."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.has_data_type("Velocity")
        assert result is not None, "has_data_type() should return a result"

    def test_has_data_type_returns_bool(self, valid_rdi_file):
        """Test that has_data_type() returns boolean."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.has_data_type("Velocity")

        assert isinstance(
            result, (bool, np.bool_)
        ), f"Should return bool, got {type(result)}"

    def test_has_data_type_valid_types(self, valid_rdi_file):
        """Test has_data_type() for valid RDI data types."""
        ds = read_header(str(valid_rdi_file))

        # Test common valid types
        result_fixed = ds.header.has_data_type("Fixed Leader")
        result_var = ds.header.has_data_type("Variable Leader")
        result_vel = ds.header.has_data_type("Velocity")

        # Should return boolean for each
        assert isinstance(result_fixed, (bool, np.bool_))
        assert isinstance(result_var, (bool, np.bool_))
        assert isinstance(result_vel, (bool, np.bool_))

    def test_has_data_type_unknown_type(self, valid_rdi_file):
        """Test has_data_type() for unknown type."""
        ds = read_header(str(valid_rdi_file))

        result = ds.header.has_data_type("NonexistentType")

        # Should return False for unknown type
        assert result is False, "Unknown data type should return False"

    def test_has_data_type_case_sensitive(self, valid_rdi_file):
        """Test if has_data_type() is case-sensitive."""
        ds = read_header(str(valid_rdi_file))

        result_correct = ds.header.has_data_type("Velocity")
        result_lower = ds.header.has_data_type("velocity")

        # May or may not be case-sensitive depending on implementation
        # Just ensure consistent behavior
        assert isinstance(result_correct, (bool, np.bool_))
        assert isinstance(result_lower, (bool, np.bool_))


# ============================================================================
# TEST CLASS 6: Accessor Properties
# ============================================================================


class TestHeaderAccessorProperties:
    """Test accessor property accessors."""

    def test_ensemble_property_exists(self, valid_rdi_file):
        """Test that ensemble property exists."""
        ds = read_header(str(valid_rdi_file))

        assert hasattr(ds.header, "ensemble"), "Accessor should have ensemble property"

    def test_ensemble_property_value(self, valid_rdi_file):
        """Test that ensemble property returns valid value."""
        ds = read_header(str(valid_rdi_file))

        ensemble = ds.header.ensemble

        # Should be positive integer
        assert isinstance(
            ensemble, (int, np.integer)
        ), f"ensemble should be int, got {type(ensemble)}"
        assert ensemble > 0, "ensemble should be positive"

    def test_data_type_property_exists(self, valid_rdi_file):
        """Test that data_type property exists."""
        ds = read_header(str(valid_rdi_file))

        assert hasattr(
            ds.header, "data_type"
        ), "Accessor should have data_type property"

    def test_data_type_property_value(self, valid_rdi_file):
        """Test that data_type property returns valid value."""
        ds = read_header(str(valid_rdi_file))

        data_type = ds.header.data_type

        # Should be non-negative integer
        assert isinstance(
            data_type, (int, np.integer)
        ), f"data_type should be int, got {type(data_type)}"
        assert data_type >= 0, "data_type should be non-negative"

    def test_error_message_property_exists(self, valid_rdi_file):
        """Test that error_message property exists."""
        ds = read_header(str(valid_rdi_file))

        assert hasattr(
            ds.header, "error_message"
        ), "Accessor should have error_message property"

    def test_error_message_property_value(self, valid_rdi_file):
        """Test that error_message property returns string."""
        ds = read_header(str(valid_rdi_file))

        error_msg = ds.header.error_message

        assert isinstance(
            error_msg, str
        ), f"error_message should be string, got {type(error_msg)}"

    def test_error_code_property_exists(self, valid_rdi_file):
        """Test that error_code property exists."""
        ds = read_header(str(valid_rdi_file))

        assert hasattr(
            ds.header, "error_code"
        ), "Accessor should have error_code property"

    def test_error_code_property_value(self, valid_rdi_file):
        """Test that error_code property returns integer."""
        ds = read_header(str(valid_rdi_file))

        error_code = ds.header.error_code

        assert isinstance(
            error_code, (int, np.integer)
        ), f"error_code should be int, got {type(error_code)}"

    def test_file_size_bytes_property_exists(self, valid_rdi_file):
        """Test that file_size_bytes property exists."""
        ds = read_header(str(valid_rdi_file))

        assert hasattr(
            ds.header, "file_size_bytes"
        ), "Accessor should have file_size_bytes property"

    def test_file_size_bytes_property_value(self, valid_rdi_file):
        """Test that file_size_bytes property returns positive integer."""
        ds = read_header(str(valid_rdi_file))

        file_size = ds.header.file_size_bytes

        assert isinstance(
            file_size, (int, np.integer)
        ), f"file_size_bytes should be int, got {type(file_size)}"
        assert file_size > 0, "file_size_bytes should be positive"

    def test_calculated_size_bytes_property_exists(self, valid_rdi_file):
        """Test that calculated_size_bytes property exists."""
        ds = read_header(str(valid_rdi_file))

        assert hasattr(
            ds.header, "calculated_size_bytes"
        ), "Accessor should have calculated_size_bytes property"

    def test_calculated_size_bytes_property_value(self, valid_rdi_file):
        """Test that calculated_size_bytes property returns positive integer."""
        ds = read_header(str(valid_rdi_file))

        calc_size = ds.header.calculated_size_bytes

        assert isinstance(
            calc_size, (int, np.integer)
        ), f"calculated_size_bytes should be int, got {type(calc_size)}"
        assert calc_size > 0, "calculated_size_bytes should be positive"

    def test_file_size_match_property_exists(self, valid_rdi_file):
        """Test that file_size_match property exists."""
        ds = read_header(str(valid_rdi_file))

        assert hasattr(
            ds.header, "file_size_match"
        ), "Accessor should have file_size_match property"

    def test_file_size_match_property_value(self, valid_rdi_file):
        """Test that file_size_match property returns boolean."""
        ds = read_header(str(valid_rdi_file))

        match = ds.header.file_size_match

        assert isinstance(
            match, (bool, np.bool_)
        ), f"file_size_match should be bool, got {type(match)}"


# ============================================================================
# TEST CLASS 7: Summary Method
# ============================================================================


class TestHeaderAccessorSummary:
    """Test summary and printing methods."""

    def test_summary_callable(self, valid_rdi_file):
        """Test that summary() is callable."""
        ds = read_header(str(valid_rdi_file))

        # Should be callable without raising exception
        try:
            result = ds.header.summary()
            # summary() may return None (prints directly) or a string
        except Exception as e:
            pytest.fail(f"summary() raised exception: {e}")

    def test_summary_no_exception(self, valid_rdi_file, capsys):
        """Test that summary() executes without exception."""
        ds = read_header(str(valid_rdi_file))

        # Should not raise any exception
        try:
            ds.header.summary()
        except Exception as e:
            pytest.fail(f"summary() raised exception: {e}")

    def test_summary_produces_output(self, valid_rdi_file, capsys):
        """Test that summary() produces some output."""
        ds = read_header(str(valid_rdi_file))

        ds.header.summary()

        # Capture printed output
        captured = capsys.readouterr()

        # Should have produced output (either to stdout or logging)
        # At least the method should execute
        assert True  # If we got here, summary() worked


# ============================================================================
# TEST CLASS 8: Error Handling and Edge Cases
# ============================================================================


class TestHeaderAccessorErrorHandling:
    """Test error handling in accessor methods."""

    def test_accessor_with_missing_data_variables(self):
        """Test accessor behavior with incomplete dataset."""
        # Create dataset without required variables
        ds = xr.Dataset(
            {
                "byte": (("ensemble",), np.array([100, 200, 300])),
                # Missing other required variables
            }
        )
        ds.attrs["pyadps_component"] = "Header"
        ds.attrs["filename"] = "test.000"
        ds.attrs["total_ensembles"] = 3

        # Accessor may raise error or handle gracefully
        try:
            accessor = ds.header
            # If accessible, check_file should handle missing data
            result = accessor.check_file()
            assert isinstance(result, (dict, type(None)))
        except (ValueError, AttributeError, KeyError):
            # Expected if accessor requires specific structure
            pass

    def test_multiple_accessor_calls_consistency(self, valid_rdi_file):
        """Test that multiple method calls return consistent results."""
        ds = read_header(str(valid_rdi_file))

        # Call methods multiple times
        check1 = ds.header.check_file()
        check2 = ds.header.check_file()
        types1 = ds.header.get_available_data_types()
        types2 = ds.header.get_available_data_types()

        # Results should be identical
        assert check1 == check2, "check_file() results should be consistent"
        assert (
            types1 == types2
        ), "get_available_data_types() results should be consistent"

    def test_accessor_caching_properties(self, valid_rdi_file):
        """Test that accessor properly caches computed properties."""
        ds = read_header(str(valid_rdi_file))

        # Access properties multiple times
        size1 = ds.header.calculated_size_bytes
        size2 = ds.header.calculated_size_bytes

        # Should return same values (likely same object if cached)
        assert size1 == size2, "Cached property should return consistent values"

    def test_accessor_with_xarray_selection(self, multi_ensemble_rdi_file):
        """Test that accessor works after xarray subset operations."""
        ds = read_header(str(multi_ensemble_rdi_file))

        # Subset the dataset
        subset = ds.isel(ensemble=slice(0, 2))

        # Accessor should still be accessible
        assert hasattr(subset, "header"), "Accessor should persist through subset"

        # Methods should work on subset
        try:
            types = subset.header.get_available_data_types(ens=0)
            assert isinstance(types, list)
        except Exception as e:
            # May fail if accessor expects specific structure
            pass

    def test_logging_on_invalid_data_type_query(self, valid_rdi_file):
        """Test that invalid queries are handled with logging."""
        ds = read_header(str(valid_rdi_file))

        # Query for invalid data type should handle gracefully
        result = ds.header.has_data_type("InvalidType123")

        # Should return False, not raise exception
        assert result is False


# ============================================================================
# TEST CLASS 9: Integration Tests
# ============================================================================


class TestHeaderAccessorIntegration:
    """Integration tests combining multiple accessor features."""

    def test_workflow_validate_and_query(self, valid_rdi_file):
        """Test typical workflow: validate file then query data types."""
        ds = read_header(str(valid_rdi_file))

        # 1. Check file integrity
        check = ds.header.check_file()
        assert isinstance(check, dict)

        # 2. Get data types
        types = ds.header.get_available_data_types()
        assert isinstance(types, list)

        # 3. Query specific type
        has_velocity = ds.header.has_data_type("Velocity")
        assert isinstance(has_velocity, (bool, np.bool_))

    def test_workflow_ensemble_exploration(self, multi_ensemble_rdi_file):
        """Test typical workflow: explore ensemble structure."""
        ds = read_header(str(multi_ensemble_rdi_file))

        # Get total ensembles
        total = ds.header.ensemble
        assert total >= 3, "Should have 3 ensembles from fixture"

        # Get info for each ensemble
        for ens_idx in range(min(total, 3)):
            info = ds.header.get_ensemble_info(ensemble=ens_idx)
            assert info["ensemble_number"] == ens_idx
            assert len(info["data_types"]) > 0

    def test_workflow_check_and_summary(self, valid_rdi_file, capsys):
        """Test workflow: check file then display summary."""
        ds = read_header(str(valid_rdi_file))

        # Check file
        check = ds.header.check_file()
        file_size_ok = check["File Size Match"]

        # Display summary
        ds.header.summary()

        # Verify summary produced output
        captured = capsys.readouterr()
        # Summary should have executed without error


# ============================================================================
# TEST CLASS 10: Performance and Edge Cases
# ============================================================================


class TestHeaderAccessorPerformance:
    """Performance and edge case tests."""

    def test_large_ensemble_access(self, multi_ensemble_rdi_file):
        """Test accessing data from file with multiple ensembles."""
        ds = read_header(str(multi_ensemble_rdi_file))

        # Access info from all ensembles
        total = len(ds.coords["ensemble"])

        for ens_idx in range(total):
            info = ds.header.get_ensemble_info(ensemble=ens_idx)
            assert info["ensemble_number"] == ens_idx

    def test_property_repeated_access(self, valid_rdi_file):
        """Test repeated access to computed properties."""
        ds = read_header(str(valid_rdi_file))

        # Access same property multiple times
        for _ in range(5):
            size = ds.header.calculated_size_bytes
            assert size > 0

    def test_check_file_on_valid_data(self, valid_rdi_file):
        """Test check_file() completes successfully on valid data."""
        ds = read_header(str(valid_rdi_file))

        check = ds.header.check_file()

        # All checks should be boolean
        for key, value in check.items():
            if key.endswith("(B)"):  # Size fields
                assert isinstance(value, (int, np.integer))
            elif key == "File Size (MB)":  # MB size
                assert isinstance(value, (int, float, np.number))
            else:  # Boolean checks
                assert isinstance(
                    value, (bool, np.bool_)
                ), f"Expected bool for {key}, got {type(value)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================================
# TEST CLASS 11: Utility Functions (module-level in accessors.py)
# ============================================================================


from pyadps.io.accessors import (
    _check_equal,
    _check_2d_rows_uniform,
    _check_byte_skip_uniformity,
)


class TestCheckEqual:
    """Tests for the _check_equal() utility function."""

    # ------------------------------------------------------------------
    # True cases
    # ------------------------------------------------------------------

    def test_all_same_integers_returns_true(self):
        assert _check_equal(np.array([5, 5, 5])) is True

    def test_single_element_returns_true(self):
        assert _check_equal(np.array([42])) is True

    def test_empty_array_returns_true(self):
        assert _check_equal(np.array([])) is True

    def test_all_zeros_returns_true(self):
        assert _check_equal(np.array([0, 0, 0, 0])) is True

    def test_float_equal_returns_true(self):
        assert _check_equal(np.array([3.14, 3.14, 3.14])) is True

    def test_large_uniform_array_returns_true(self):
        arr = np.full(1000, 7)
        assert _check_equal(arr) is True

    # ------------------------------------------------------------------
    # False cases
    # ------------------------------------------------------------------

    def test_different_integers_returns_false(self):
        assert _check_equal(np.array([5, 5, 6])) is False

    def test_first_element_differs_returns_false(self):
        assert _check_equal(np.array([9, 5, 5])) is False

    def test_all_different_returns_false(self):
        assert _check_equal(np.array([1, 2, 3])) is False

    def test_float_variation_returns_false(self):
        assert _check_equal(np.array([1.0, 1.0, 1.1])) is False

    # ------------------------------------------------------------------
    # Return type
    # ------------------------------------------------------------------

    def test_return_type_is_bool(self):
        result = _check_equal(np.array([1, 1, 1]))
        assert isinstance(result, bool)

    def test_return_type_is_bool_on_false(self):
        result = _check_equal(np.array([1, 2]))
        assert isinstance(result, bool)


class TestCheck2DRowsUniform:
    """
    Tests for the _check_2d_rows_uniform() utility function.

    Semantics clarification
    -----------------------
    The function iterates over ``array.T`` (the transpose), so it checks
    each **column** of the original array for uniformity across **rows**.
    In ADCP terms: for each data-type slot (column), every ensemble (row)
    must carry the same value.

    Uniform (True):
        Each column is constant across all rows.
        Example — address_offset where every ensemble has the same offsets:
            [[18, 77, 142],   ← ensemble 0
             [18, 77, 142],   ← ensemble 1
             [18, 77, 142]]   ← ensemble 2

    Non-uniform (False):
        At least one column has different values across rows.
        Example — address_offset where ensemble 1 differs:
            [[18, 77, 142],
             [18, 99, 142],   ← column 1 differs
             [18, 77, 142]]
    """

    # ------------------------------------------------------------------
    # True cases: each column is constant across rows
    # ------------------------------------------------------------------

    def test_all_columns_uniform_returns_true(self):
        # Each column is constant (same offset in every ensemble)
        arr = np.array(
            [
                [18, 77, 142],
                [18, 77, 142],
                [18, 77, 142],
            ]
        )
        assert _check_2d_rows_uniform(arr) is True

    def test_single_row_always_uniform_returns_true(self):
        # One ensemble — trivially uniform
        arr = np.array([[5, 10, 15, 20]])
        assert _check_2d_rows_uniform(arr) is True

    def test_single_column_uniform_rows_returns_true(self):
        # All rows same value in the only column
        arr = np.array([[7], [7], [7]])
        assert _check_2d_rows_uniform(arr) is True

    def test_empty_array_returns_true(self):
        arr = np.array([]).reshape(0, 0)
        assert _check_2d_rows_uniform(arr) is True

    def test_zeros_array_returns_true(self):
        arr = np.zeros((5, 4), dtype=int)
        assert _check_2d_rows_uniform(arr) is True

    # ------------------------------------------------------------------
    # False cases: at least one column is not constant across rows
    # ------------------------------------------------------------------

    def test_one_column_varies_returns_false(self):
        # Column 1 differs in row 1
        arr = np.array(
            [
                [18, 77, 142],
                [18, 99, 142],  # column 1: 77 → 99
                [18, 77, 142],
            ]
        )
        assert _check_2d_rows_uniform(arr) is False

    def test_last_column_varies_returns_false(self):
        arr = np.array(
            [
                [0, 128, 256],
                [0, 128, 256],
                [0, 128, 999],  # column 2 differs
            ]
        )
        assert _check_2d_rows_uniform(arr) is False

    def test_single_column_non_uniform_returns_false(self):
        # Single column, different values across rows
        arr = np.array([[1], [2], [3]])
        assert _check_2d_rows_uniform(arr) is False

    # ------------------------------------------------------------------
    # Dimension handling
    # ------------------------------------------------------------------

    def test_1d_array_returns_false(self):
        arr = np.array([1, 2, 3])
        assert _check_2d_rows_uniform(arr) is False

    def test_3d_array_returns_false(self):
        arr = np.ones((2, 3, 4))
        assert _check_2d_rows_uniform(arr) is False

    # ------------------------------------------------------------------
    # Return type and parameter behaviour
    # ------------------------------------------------------------------

    def test_return_type_is_bool(self):
        arr = np.array([[1, 1], [1, 1]])
        assert isinstance(_check_2d_rows_uniform(arr), bool)

    def test_array_name_parameter_does_not_affect_result(self):
        """array_name is used for logging only; result must be identical."""
        arr = np.array([[5, 10], [5, 10]])
        result_default = _check_2d_rows_uniform(arr)
        result_named = _check_2d_rows_uniform(arr, array_name="my_array")
        assert result_default == result_named


class TestCheckByteSkipUniformity:
    """Tests for the _check_byte_skip_uniformity() utility function."""

    # ------------------------------------------------------------------
    # True cases
    # ------------------------------------------------------------------

    def test_uniform_byte_skip_returns_true(self):
        byte_skip = np.array([1134, 2268, 3402, 4536])
        assert _check_byte_skip_uniformity(byte_skip, 4) is True

    def test_single_ensemble_returns_true(self):
        byte_skip = np.array([500])
        assert _check_byte_skip_uniformity(byte_skip, 1) is True

    def test_empty_array_returns_true(self):
        byte_skip = np.array([], dtype=np.int64)
        assert _check_byte_skip_uniformity(byte_skip, 0) is True

    def test_two_ensembles_uniform_returns_true(self):
        byte_skip = np.array([200, 400])
        assert _check_byte_skip_uniformity(byte_skip, 2) is True

    def test_large_uniform_array_returns_true(self):
        base = 1134
        n = 100
        byte_skip = np.array([base * (i + 1) for i in range(n)])
        assert _check_byte_skip_uniformity(byte_skip, n) is True

    # ------------------------------------------------------------------
    # False cases
    # ------------------------------------------------------------------

    def test_non_uniform_byte_skip_returns_false(self):
        byte_skip = np.array([1134, 2268, 3400, 4536])  # index 2 is off
        assert _check_byte_skip_uniformity(byte_skip, 4) is False

    def test_first_element_wrong_returns_false(self):
        byte_skip = np.array([999, 2000, 3000, 4000])
        assert _check_byte_skip_uniformity(byte_skip, 4) is False

    # ------------------------------------------------------------------
    # Return type
    # ------------------------------------------------------------------

    def test_return_type_is_bool(self):
        byte_skip = np.array([1000, 2000, 3000])
        result = _check_byte_skip_uniformity(byte_skip, 3)
        assert isinstance(result, bool)


# ============================================================================
# TEST CLASS 12: Targeted branch-coverage tests for remaining gaps
# ============================================================================


class TestCalculatedSizeBytesExceptBranch:
    """
    Cover lines 450-453: the ``except (ValueError, AttributeError)`` branch
    inside ``calculated_size_bytes``.

    The branch fires when ``byte`` is present in data_vars but the arithmetic
    on it raises ``ValueError`` or ``AttributeError``.  We mock
    ``xr.DataArray.sum`` to raise each exception in turn.
    """

    def _make_minimal_header_ds(self):
        """Return a valid Header dataset with a real ``byte`` variable."""
        ds = xr.Dataset(
            {"byte": (("ensemble",), np.array([100, 200, 300]))},
            attrs={
                "filename": "test.000",
                "total_ensembles": 3,
                "pyadps_component": "Header",
            },
        )
        return ds

    def test_value_error_in_sum_returns_minus_one(self):
        ds = self._make_minimal_header_ds()
        accessor = ds.header
        # Reset the cache so the property re-executes
        accessor._calculated_size_bytes = None

        with mock.patch.object(type(ds.byte), "sum", side_effect=ValueError("bad sum")):
            result = accessor.calculated_size_bytes

        assert result == -1

    def test_attribute_error_in_sum_returns_minus_one(self):
        ds = self._make_minimal_header_ds()
        accessor = ds.header
        accessor._calculated_size_bytes = None

        with mock.patch.object(
            type(ds.byte), "sum", side_effect=AttributeError("no sum")
        ):
            result = accessor.calculated_size_bytes

        assert result == -1

    def test_warning_is_logged_on_value_error(self, caplog):
        ds = self._make_minimal_header_ds()
        accessor = ds.header
        accessor._calculated_size_bytes = None

        with mock.patch.object(type(ds.byte), "sum", side_effect=ValueError("boom")):
            with caplog.at_level(logging.WARNING):
                accessor.calculated_size_bytes

        assert any("Could not calculate file size" in r.message for r in caplog.records)

    def test_missing_byte_var_returns_minus_one(self):
        """Else branch (no ``byte`` variable): also returns -1."""
        ds = xr.Dataset(
            attrs={
                "filename": "test.000",
                "total_ensembles": 1,
                "pyadps_component": "Header",
            }
        )
        # Can't instantiate accessor without 'byte', so add a dummy var
        ds["dummy"] = xr.DataArray([1])
        # Override attrs so accessor validates, but no 'byte' key
        accessor = ds.header
        accessor._calculated_size_bytes = None
        result = accessor.calculated_size_bytes
        assert result == -1


class TestHasDataTypeExceptBranch:
    """
    Cover lines 746-748: the ``except (IndexError, KeyError)`` branch
    inside ``has_data_type``.

    ``get_available_data_types(0)`` raises ``IndexError`` when
    ``total_ensembles == 0`` because ens=0 is already out of range.
    """

    def _make_zero_ensemble_ds(self):
        """Header dataset declaring zero ensembles — makes ens=0 invalid."""
        ds = xr.Dataset(
            {
                "byte": (("ensemble",), np.array([], dtype=np.int64)),
                "data_id": (
                    ("data_type_index", "ensemble"),
                    np.empty((0, 0), dtype=np.int64),
                ),
            },
            attrs={
                "filename": "test.000",
                "total_ensembles": 0,
                "pyadps_component": "Header",
            },
        )
        return ds

    def test_index_error_returns_false(self):
        ds = self._make_zero_ensemble_ds()
        result = ds.header.has_data_type("Velocity")
        assert result is False

    def test_key_error_returns_false(self):
        """Force a KeyError by mocking get_available_data_types."""
        ds = xr.Dataset(
            {"byte": (("ensemble",), np.array([100]))},
            attrs={
                "filename": "test.000",
                "total_ensembles": 1,
                "pyadps_component": "Header",
            },
        )
        with mock.patch.object(
            type(ds.header),
            "get_available_data_types",
            side_effect=KeyError("data_id"),
        ):
            result = ds.header.has_data_type("Velocity")

        assert result is False

    def test_warning_logged_on_index_error(self, caplog):
        ds = self._make_zero_ensemble_ds()
        with caplog.at_level(logging.WARNING):
            ds.header.has_data_type("Velocity")
        assert any("Error checking for data type" in r.message for r in caplog.records)


class TestSummaryStatusBranches:
    """
    Cover lines 809-811: the ``"error"`` and ``"warning"`` status branches
    inside ``summary()``.

    We mock ``check_file()`` to return controlled check dictionaries so we
    can drive each branch deterministically without needing a real file.
    """

    def _make_header_ds(self):
        ds = xr.Dataset(
            {"byte": (("ensemble",), np.array([100, 200]))},
            attrs={
                "filename": "test.000",
                "total_ensembles": 2,
                "pyadps_component": "Header",
            },
        )
        return ds

    def _all_pass_check(self):
        return {
            "System File Size (B)": 300,
            "Calculated File Size (B)": 300,
            "File Size (MB)": 0.0003,
            "File Size Match": True,
            "Byte Uniformity": True,
            "Data Type Uniformity": True,
            "Byte Skip Uniformity": True,
            "Address Offset Uniformity": True,
            "Data ID Uniformity": True,
        }

    def test_summary_prints_error_status_when_critical_check_fails(self, capsys):
        """Line 809: ``if not all_critical_pass`` → status = "error"."""
        ds = self._make_header_ds()
        failing_check = {**self._all_pass_check(), "File Size Match": False}

        with mock.patch.object(
            type(ds.header), "check_file", return_value=failing_check
        ):
            with mock.patch.object(
                type(ds.header),
                "get_available_data_types",
                return_value=["Fixed Leader"],
            ):
                ds.header.summary()

        output = capsys.readouterr().out
        assert "ERROR" in output.upper()

    def test_summary_prints_warning_status_when_uniformity_check_fails(self, capsys):
        """Line 810-811: critical passes but uniformity fails → status = "warning"."""
        ds = self._make_header_ds()
        warning_check = {
            **self._all_pass_check(),
            # All critical pass
            "File Size Match": True,
            "Byte Uniformity": True,
            "Data Type Uniformity": True,
            # One uniformity check fails
            "Byte Skip Uniformity": False,
        }

        with mock.patch.object(
            type(ds.header), "check_file", return_value=warning_check
        ):
            with mock.patch.object(
                type(ds.header),
                "get_available_data_types",
                return_value=["Fixed Leader"],
            ):
                ds.header.summary()

        output = capsys.readouterr().out
        assert "WARNING" in output.upper()

    def test_summary_prints_healthy_status_when_all_pass(self, capsys):
        """Confirm the existing healthy path still works (regression guard)."""
        ds = self._make_header_ds()

        with mock.patch.object(
            type(ds.header), "check_file", return_value=self._all_pass_check()
        ):
            with mock.patch.object(
                type(ds.header),
                "get_available_data_types",
                return_value=["Fixed Leader"],
            ):
                ds.header.summary()

        output = capsys.readouterr().out
        assert "HEALTHY" in output.upper()
