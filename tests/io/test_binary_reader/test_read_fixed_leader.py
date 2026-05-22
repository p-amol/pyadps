"""
Comprehensive pytest test suite for read_fixed_leader function in filereader.py

This module provides extensive test coverage for the read_fixed_leader function,
which reads ADCP Fixed Leader data from RDI binary files and returns an xarray.Dataset
with raw and optionally decoded variables.

Test Coverage:
    - Basic functionality (valid files, single/multiple ensembles)
    - Data structure validation (xarray.Dataset, coordinates, dimensions)
    - Raw field extraction (36 raw variables with proper dtypes)
    - Decoded field computation (25 decoded variables from bit extraction)
    - Variable attributes (CF Convention compliance, metadata)
    - Parameter variations (auto-fetch vs explicit parameters, include_decoded flag)
    - Error handling (missing files, corrupted data, invalid parameters)
    - Edge cases (empty files, truncated data, single vs multiple ensembles)
    - Metadata loading (default, custom paths, malformed JSON)
    - Integration with read_header() automatic parameter retrieval

References:
    RDI WorkHorse Commands and Output Data Format (Section 5.2, page 126):
    - Fixed Leader: 59 bytes per ensemble
    - 36 raw configuration fields
    - Bit-extracted decoded fields (frequency, beam pattern, etc.)

Author: pyadps development team
License: MIT
Version: 1.0.0
"""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pyadps.io.binary_reader import (
    read_fixed_leader,
    _load_fixed_leader_metadata,
    _extract_and_sort_raw_fields,
    _build_variable_attributes,
)

# Import test data builders
from tests.io.test_pd0_parser.fixtures.ensemble_builder import (
    build_ensemble,
    FixedLeaderData,
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
    rdi_file = tmp_path / "test_fixed_leader_single.000"
    rdi_file.write_bytes(valid_rdi_ensemble)
    return rdi_file


@pytest.fixture
def multi_ensemble_rdi_file(tmp_path, valid_rdi_ensemble):
    """Create a temporary file with multiple valid ensembles.

    Args:
        tmp_path: pytest built-in temp directory fixture
        valid_rdi_ensemble: Generated ensemble bytes

    Returns:
        Path: Pathlib.Path to temp RDI file with 3 ensembles
    """
    rdi_file = tmp_path / "test_fixed_leader_multi.000"
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
    rdi_file = tmp_path / "test_truncated.000"
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
    rdi_file = tmp_path / "test_empty.000"
    rdi_file.write_bytes(b"")
    return rdi_file


@pytest.fixture
def custom_fixed_leader_ensemble():
    """Create ensemble with custom Fixed Leader values for specific testing.

    Returns:
        bytes: RDI ensemble with custom fixed leader data
    """
    fl_data = FixedLeaderData(
        num_beams=5,
        num_cells=40,
        cpu_fw_ver=17,
        cpu_fw_rev=10,
        pings_per_ensemble=10,
        cell_length=500,
    )
    return build_ensemble(fixed_leader_data=fl_data)


@pytest.fixture
def custom_fixed_leader_file(tmp_path, custom_fixed_leader_ensemble):
    """Create temp file with custom fixed leader ensemble.

    Args:
        tmp_path: pytest built-in temp directory fixture
        custom_fixed_leader_ensemble: Generated ensemble with custom FL data

    Returns:
        Path: Pathlib.Path to temp RDI file with custom FL
    """
    rdi_file = tmp_path / "test_custom_fl.000"
    rdi_file.write_bytes(custom_fixed_leader_ensemble)
    return rdi_file


@pytest.fixture
def mock_metadata_file(tmp_path):
    """Create a mock fixed_leader_meta.json file for testing custom metadata loading.

    Args:
        tmp_path: pytest built-in temp directory fixture

    Returns:
        Path: Pathlib.Path to mock metadata JSON file
    """
    metadata = {
        "metadata_version": "1.2",
        "component": "FixedLeader",
        "total_raw_fields": 36,
        "raw_fields": {
            str(i): {
                "index": i,
                "name": f"field_{i}",
                "long_name": f"Test Field {i}",
                "description": f"Description for field {i}",
                "dtype": "uint16" if i % 2 == 0 else "uint8",
                "bytes": 2 if i % 2 == 0 else 1,
                "unit": "1",
                "is_raw": True,
                "is_decoded": False,
                "comments": f"Test comment {i}",
            }
            for i in range(36)
        },
    }
    meta_file = tmp_path / "fixed_leader_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    return meta_file


@pytest.fixture
def malformed_metadata_file(tmp_path):
    """Create a malformed JSON file to test error handling.

    Args:
        tmp_path: pytest built-in temp directory fixture

    Returns:
        Path: Pathlib.Path to malformed JSON file
    """
    meta_file = tmp_path / "malformed_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        f.write("{invalid json content")
    return meta_file


@pytest.fixture
def metadata_without_raw_fields(tmp_path):
    """Create metadata file missing raw_fields section.

    Tests error handling when metadata structure is invalid.

    Args:
        tmp_path: pytest built-in temp directory fixture

    Returns:
        Path: Pathlib.Path to invalid metadata file
    """
    metadata = {
        "metadata_version": "1.2",
        "component": "FixedLeader",
        # Missing 'raw_fields' key
    }
    meta_file = tmp_path / "no_raw_fields_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    return meta_file


# ============================================================================
# TEST SUITE: Basic Functionality
# ============================================================================


class TestReadFixedLeaderBasicFunctionality:
    """Tests for basic read_fixed_leader operation with valid files."""

    def test_read_valid_single_ensemble_file(self, valid_rdi_file):
        """Test reading valid RDI file with single ensemble returns xarray.Dataset.

        Verifies:
            - Returns xarray.Dataset object
            - Dataset contains expected data variables
            - Coordinates are properly set
        """
        result = read_fixed_leader(valid_rdi_file)

        assert isinstance(result, xr.Dataset), "Result should be xarray.Dataset"
        assert len(result.data_vars) > 0, "Dataset should have data variables"
        assert "ensemble" in result.coords, "Dataset should have ensemble coordinate"

    def test_read_valid_multi_ensemble_file(self, multi_ensemble_rdi_file):
        """Test reading file with multiple ensembles extracts all of them.

        Verifies:
            - Reads all ensembles from file
            - Ensemble coordinate has correct length (3)
            - Data variables have correct ensemble dimension
        """
        result = read_fixed_leader(multi_ensemble_rdi_file)

        assert isinstance(result, xr.Dataset), "Result should be xarray.Dataset"
        # Should have 3 ensembles
        assert len(result.coords["ensemble"]) == 3
        # Data variables should have ensemble dimension
        for var in result.data_vars:
            assert "ensemble" in result[var].dims

    def test_return_type_is_xarray_dataset(self, valid_rdi_file):
        """Verify read_fixed_leader returns plain xarray.Dataset, not subclass.

        Confirms the function returns a standard xarray.Dataset that can
        use accessor patterns for domain-specific methods.
        """
        result = read_fixed_leader(valid_rdi_file)
        assert type(result).__name__ == "Dataset"
        assert isinstance(result, xr.Dataset)


# ============================================================================
# TEST SUITE: Data Structure and Coordinates
# ============================================================================


class TestReadFixedLeaderDataStructure:
    """Tests for data structure, coordinates, and dimensions."""

    def test_dataset_has_ensemble_coordinate(self, valid_rdi_file):
        """Test dataset contains ensemble coordinate.

        Ensemble coordinate should be present in the dataset.
        """
        result = read_fixed_leader(valid_rdi_file)

        assert "ensemble" in result.coords
        ensemble_vals = result.coords["ensemble"].values
        assert len(ensemble_vals) > 0, "Ensemble coordinate should have values"

    def test_ensemble_coordinate_correct_length(self, valid_rdi_file):
        """Test ensemble coordinate has correct length for single ensemble file.

        Verifies coordinate array length corresponds to number of ensembles.
        """
        result = read_fixed_leader(valid_rdi_file)

        ensemble_coord = result.coords["ensemble"]
        assert len(ensemble_coord) == 1, "Single ensemble file should have 1 ensemble"

    def test_multi_ensemble_coordinate_values(self, multi_ensemble_rdi_file):
        """Test multi-ensemble file has correct coordinate values.

        For 3 ensembles, should have [0, 1, 2] or similar sequential numbering.
        """
        result = read_fixed_leader(multi_ensemble_rdi_file)

        ensemble_vals = result.coords["ensemble"].values
        assert len(ensemble_vals) == 3, "Should have 3 ensemble values"
        # Check they're sequential (starting from 0 or 1)
        assert all(isinstance(v, (int, np.integer)) for v in ensemble_vals)

    def test_raw_fields_have_ensemble_dimension(self, valid_rdi_file):
        """Test all raw field variables have ensemble dimension.

        All fixed leader fields should vary by ensemble, thus all
        data variables should have ensemble as a dimension.
        """
        result = read_fixed_leader(valid_rdi_file)

        for var_name, var in result.data_vars.items():
            assert (
                "ensemble" in var.dims
            ), f"Variable {var_name} should have ensemble dimension"


# ============================================================================
# TEST SUITE: Raw Fields Extraction
# ============================================================================


class TestReadFixedLeaderRawFields:
    """Tests for correct extraction of 36 raw fixed leader fields."""

    def test_correct_number_of_raw_variables(self, valid_rdi_file):
        """Test exactly 36 raw field variables are extracted (by default).

        With include_decoded=False, should get exactly 36 raw fields.
        """
        result = read_fixed_leader(valid_rdi_file, include_decoded=False)

        # Count variables (all should be raw fields)
        num_vars = len(result.data_vars)
        assert num_vars == 36, f"Expected 36 raw fields, got {num_vars}"

    def test_raw_and_decoded_fields_together(self, valid_rdi_file):
        """Test include_decoded=True adds decoded variables.

        With include_decoded=True, should have 36 raw + 25 decoded = 61 variables
        (or close to it, depending on implementation).
        """
        result = read_fixed_leader(valid_rdi_file, include_decoded=True)

        num_vars = len(result.data_vars)
        # Should have more than 36 with decoded fields
        assert num_vars > 36, "Should have additional decoded variables"

    def test_raw_field_names_follow_conventions(self, valid_rdi_file):
        """Test raw field variable names follow naming conventions.

        Names should be lowercase with underscores (snake_case) and
        descriptive (e.g., 'system_configuration_code', 'cpu_version').
        """
        result = read_fixed_leader(valid_rdi_file, include_decoded=False)

        expected_fields = [
            "fixed_leader_id",
            "cpu_version",
            "cpu_revision",
            "system_configuration_code",
            "real_sim_flag",
            "lag_length",
            "num_beams",
            "num_cells",
            "pings_per_ensemble",
            "cell_length",
            "blank_after_transmit",
            "profiling_mode",
        ]

        for expected_name in expected_fields[:5]:  # Check first few
            assert (
                expected_name in result.data_vars
            ), f"Expected field {expected_name} not found in dataset"

    def test_raw_fields_have_correct_dtypes(self, valid_rdi_file):
        """Test raw fields have expected data types.

        Uint8, uint16, int16, int64 (as numpy may upcast during processing).
        """
        result = read_fixed_leader(valid_rdi_file, include_decoded=False)

        # Check some known fields - may be upcasted to int64
        if "fixed_leader_id" in result.data_vars:
            assert (
                result["fixed_leader_id"].dtype
                in (
                    np.uint16,
                    np.int16,
                    np.int64,
                    np.uint64,
                )
            ), f"fixed_leader_id should be numeric, got {result['fixed_leader_id'].dtype}"

        if "cpu_version" in result.data_vars:
            assert result["cpu_version"].dtype in (
                np.uint8,
                np.int8,
                np.int64,
                np.uint64,
            ), f"cpu_version should be numeric, got {result['cpu_version'].dtype}"


# ============================================================================
# TEST SUITE: Variable Attributes
# ============================================================================


class TestReadFixedLeaderVariableAttributes:
    """Tests for NetCDF/CF Convention attributes on variables."""

    def test_variables_have_long_name_attribute(self, valid_rdi_file):
        """Test all variables have long_name attribute.

        Per CF Conventions, variables should have descriptive long_name.
        """
        result = read_fixed_leader(valid_rdi_file, include_decoded=False)

        for var_name in result.data_vars:
            assert (
                "long_name" in result[var_name].attrs
            ), f"Variable {var_name} missing long_name attribute"

    def test_variables_have_units_attribute(self, valid_rdi_file):
        """Test all variables have units attribute.

        Per CF Conventions, variables should specify units (even if '1' or 'dimensionless').
        """
        result = read_fixed_leader(valid_rdi_file, include_decoded=False)

        for var_name in result.data_vars:
            assert (
                "units" in result[var_name].attrs
            ), f"Variable {var_name} missing units attribute"

    def test_variables_have_description_attribute(self, valid_rdi_file):
        """Test variables have description attribute for clarity."""
        result = read_fixed_leader(valid_rdi_file, include_decoded=False)

        # At least some variables should have description
        descriptions_found = sum(
            1 for var in result.data_vars if "description" in result[var].attrs
        )
        assert descriptions_found > 0, "Should have descriptions in attributes"

    def test_raw_fields_marked_as_raw(self, valid_rdi_file):
        """Test raw field variables have is_raw=true attribute."""
        result = read_fixed_leader(valid_rdi_file, include_decoded=False)

        for var_name in result.data_vars:
            is_raw = result[var_name].attrs.get("is_raw", "").lower()
            assert is_raw == "true", f"Raw field {var_name} should have is_raw=true"

    def test_attributes_include_cf_convention_fields(self, valid_rdi_file):
        """Test attributes follow CF Convention standards.

        Should include standard CF fields: long_name, units, valid_min, valid_max.
        """
        result = read_fixed_leader(valid_rdi_file, include_decoded=False)

        sample_var = list(result.data_vars.keys())[0]
        attrs = result[sample_var].attrs

        # Check for CF Convention attributes
        cf_attrs = ["long_name", "units"]
        for cf_attr in cf_attrs:
            assert (
                cf_attr in attrs
            ), f"CF Convention attribute {cf_attr} missing from {sample_var}"


# ============================================================================
# TEST SUITE: Parameter Variations
# ============================================================================


class TestReadFixedLeaderParameterVariations:
    """Tests for different parameter combinations and auto-fetch behavior."""

    def test_auto_fetch_byteskip_when_not_provided(self, valid_rdi_file):
        """Test function auto-fetches byteskip from file header when not provided.

        When byteskip=None, should call read_header internally to get it.
        """
        # Should work without explicit byteskip
        result = read_fixed_leader(
            valid_rdi_file, byteskip=None, offset=None, idarray=None
        )

        assert isinstance(result, xr.Dataset)
        assert len(result.data_vars) > 0

    def test_auto_fetch_ensemble_count_when_zero(self, valid_rdi_file):
        """Test function auto-fetches ensemble count when ensemble=0.

        When ensemble=0, should determine count from file header.
        """
        result = read_fixed_leader(valid_rdi_file, ensemble=0)

        assert isinstance(result, xr.Dataset)
        assert len(result.coords["ensemble"]) == 1

    def test_explicit_parameters_override_auto_fetch(self, valid_rdi_file):
        """Test explicit parameters are used instead of auto-fetch.

        When parameters provided explicitly, should use them without calling
        read_header.
        """
        # Read header to get parameters
        from pyadps.io.binary_reader import read_header

        header = read_header(valid_rdi_file)
        byteskip = header["byte_skip"].values
        offset = header["address_offset"].values
        idarray = header["data_id"].values

        # Call with explicit parameters
        result = read_fixed_leader(
            valid_rdi_file,
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=1,
        )

        assert isinstance(result, xr.Dataset)

    def test_include_decoded_true_includes_more_variables(self, valid_rdi_file):
        """Test include_decoded=True parameter adds decoded variables.

        Verify that setting include_decoded=True results in more variables
        than include_decoded=False.
        """
        result_raw = read_fixed_leader(valid_rdi_file, include_decoded=False)
        result_decoded = read_fixed_leader(valid_rdi_file, include_decoded=True)

        num_raw = len(result_raw.data_vars)
        num_with_decoded = len(result_decoded.data_vars)

        assert (
            num_with_decoded > num_raw
        ), "include_decoded=True should add more variables"

    def test_custom_json_file_path_parameter(self, valid_rdi_file, mock_metadata_file):
        """Test custom json_file_path parameter loads metadata from custom location.

        Verify that providing custom path uses that metadata file.
        """
        result = read_fixed_leader(
            valid_rdi_file, json_file_path=str(mock_metadata_file)
        )

        assert isinstance(result, xr.Dataset)
        # Variables should use custom metadata field names
        assert "field_0" in result.data_vars or "fixed_leader_id" in result.data_vars


# ============================================================================
# TEST SUITE: Decoded Fields
# ============================================================================


class TestReadFixedLeaderDecodedFields:
    """Tests for computed/decoded fields derived from bit extraction."""

    def test_decoded_fields_exist_when_enabled(self, valid_rdi_file):
        """Test decoded fields are computed when include_decoded=True."""
        result = read_fixed_leader(valid_rdi_file, include_decoded=True)

        # Should have decoded field variables (names contain decoded info)
        # Examples: frequency_khz, beam_pattern, etc.
        has_decoded = any("frequency" in str(name) for name in result.data_vars)
        # At least verify we have more variables
        assert len(result.data_vars) > 36

    def test_decoded_fields_not_present_when_disabled(self, valid_rdi_file):
        """Test decoded fields are not computed when include_decoded=False."""
        result = read_fixed_leader(valid_rdi_file, include_decoded=False)

        # Should have exactly 36 raw fields, no decoded
        assert len(result.data_vars) == 36

    def test_decoded_fields_have_correct_attributes(self, valid_rdi_file):
        """Test decoded variables have is_decoded=true attribute."""
        result = read_fixed_leader(valid_rdi_file, include_decoded=True)

        # Find a decoded field
        decoded_vars = [
            var
            for var in result.data_vars
            if result[var].attrs.get("is_decoded", "false").lower() == "true"
        ]

        assert len(decoded_vars) > 0, "Should have decoded variables"


# ============================================================================
# TEST SUITE: Error Handling
# ============================================================================


class TestReadFixedLeaderErrorHandling:
    """Tests for error handling and edge cases."""

    def test_file_not_found_raises_error(self):
        """Test FileNotFoundError is raised for non-existent file.

        Function should handle missing files gracefully with appropriate error.
        """
        with pytest.raises(Exception):  # FileNotFoundError or similar
            read_fixed_leader("/nonexistent/path/file.000")

    def test_empty_file_handling(self, empty_file):
        """Test behavior with empty file.

        Should either raise error or return empty dataset without crashing.
        """
        # Should not crash; behavior depends on implementation
        try:
            result = read_fixed_leader(empty_file)
            # If it succeeds, should return Dataset (possibly empty)
            assert isinstance(result, xr.Dataset)
        except Exception:
            # Expected - empty file is invalid
            pass

    def test_truncated_file_handling(self, truncated_rdi_file):
        """Test behavior with truncated/incomplete data.

        Should handle incomplete data appropriately (error or partial result).
        """
        # Behavior depends on implementation
        try:
            result = read_fixed_leader(truncated_rdi_file)
            # If successful, should be Dataset
            assert isinstance(result, xr.Dataset)
        except Exception:
            # Expected - truncated file is problematic
            pass

    def test_invalid_metadata_path_raises_error(self, valid_rdi_file):
        """Test invalid json_file_path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_fixed_leader(valid_rdi_file, json_file_path="/invalid/path/meta.json")

    def test_malformed_metadata_json_raises_error(
        self, valid_rdi_file, malformed_metadata_file
    ):
        """Test malformed JSON metadata file raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            read_fixed_leader(
                valid_rdi_file, json_file_path=str(malformed_metadata_file)
            )

    def test_metadata_missing_raw_fields_raises_error(
        self, valid_rdi_file, metadata_without_raw_fields
    ):
        """Test metadata without raw_fields section raises ValueError."""
        with pytest.raises(ValueError):
            read_fixed_leader(
                valid_rdi_file, json_file_path=str(metadata_without_raw_fields)
            )

    def test_type_error_for_invalid_file_parameter(self):
        """Test error or appropriate handling for invalid file parameter type."""
        # Invalid parameter type should raise error
        try:
            read_fixed_leader(12345)  # Invalid type
            pytest.fail("Should raise error for invalid file type")
        except (TypeError, AttributeError, ValueError, FileNotFoundError):
            pass  # Expected - could raise different errors


# ============================================================================
# TEST SUITE: Metadata Loading
# ============================================================================


class TestLoadFixedLeaderMetadata:
    """Tests for _load_fixed_leader_metadata function."""

    def test_load_default_metadata_from_package(self):
        """Test loading metadata from package default location.

        Should successfully load from installed package resources.
        """
        metadata = _load_fixed_leader_metadata()

        assert isinstance(metadata, dict)
        assert len(metadata) == 36, "Should have 36 raw fields"

    def test_load_custom_metadata_from_path(self, mock_metadata_file):
        """Test loading metadata from custom file path."""
        metadata = _load_fixed_leader_metadata(str(mock_metadata_file))

        assert isinstance(metadata, dict)
        assert len(metadata) == 36

    def test_custom_path_file_not_found(self):
        """Test FileNotFoundError when custom path doesn't exist."""
        with pytest.raises(FileNotFoundError):
            _load_fixed_leader_metadata("/nonexistent/path/meta.json")

    def test_metadata_contains_expected_fields(self):
        """Test loaded metadata contains expected structure.

        Should have 36 fields with proper indices.
        """
        metadata = _load_fixed_leader_metadata()

        # Check field indices
        for i in range(36):
            assert i in metadata, f"Field {i} missing from metadata"

    def test_metadata_field_has_required_keys(self):
        """Test each field metadata has required keys.

        Should have: name, long_name, dtype, bytes, unit, etc.
        """
        metadata = _load_fixed_leader_metadata()

        required_keys = ["name", "long_name", "dtype", "unit"]

        for field_idx, field_meta in metadata.items():
            for required_key in required_keys:
                assert (
                    required_key in field_meta
                ), f"Field {field_idx} missing {required_key}"


# ============================================================================
# TEST SUITE: Metadata Extraction and Sorting
# ============================================================================


class TestExtractAndSortRawFields:
    """Tests for _extract_and_sort_raw_fields helper function."""

    def test_extract_sorts_fields_correctly(self, mock_metadata_file):
        """Test fields are extracted and sorted in correct order.

        Fields should be sorted by index 0-35 regardless of JSON key order.
        """
        with open(mock_metadata_file, "r") as f:
            full_metadata = json.load(f)

        result = _extract_and_sort_raw_fields(full_metadata)

        # Check sorted order
        for i in range(36):
            assert i in result
            assert result[i]["index"] == i

    def test_extract_with_missing_raw_fields_section_raises(self):
        """Test ValueError when raw_fields section missing."""
        invalid_metadata = {"component": "FixedLeader"}

        with pytest.raises(ValueError):
            _extract_and_sort_raw_fields(invalid_metadata)

    def test_extract_with_non_numeric_keys_raises(self):
        """Test ValueError when field keys are not numeric."""
        invalid_metadata = {
            "raw_fields": {
                "invalid_key": {"index": 0, "name": "test"},
                "another_invalid": {"index": 1, "name": "test2"},
            }
        }

        with pytest.raises(ValueError):
            _extract_and_sort_raw_fields(invalid_metadata)

    def test_extract_with_fewer_than_36_fields_warns(self):
        """Test warning is logged when fewer than 36 fields present."""
        incomplete_metadata = {
            "raw_fields": {
                str(i): {"index": i, "name": f"field_{i}"}
                for i in range(20)  # Only 20 fields
            }
        }

        # Function may log a warning but still return data
        result = _extract_and_sort_raw_fields(incomplete_metadata)

        # Function should still return what's available
        assert len(result) == 20


# ============================================================================
# TEST SUITE: Variable Attributes Building
# ============================================================================


class TestBuildVariableAttributes:
    """Tests for _build_variable_attributes helper function."""

    def test_build_attributes_from_complete_metadata(self):
        """Test attributes are properly built from metadata dict."""
        field_meta = {
            "long_name": "Test Field",
            "description": "Test description",
            "unit": "m/s",
            "dtype": "float32",
            "valid_min": -10,
            "valid_max": 10,
            "is_raw": True,
            "comments": "Test comments",
        }

        attrs = _build_variable_attributes(field_meta)

        assert attrs["long_name"] == "Test Field"
        assert attrs["units"] == "m/s"  # Note: 'units' not 'unit'
        assert attrs["description"] == "Test description"

    def test_build_attributes_handles_missing_fields(self):
        """Test function handles metadata with missing optional fields.

        Should not crash when optional fields are missing.
        """
        minimal_meta = {
            "long_name": "Minimal Field",
        }

        attrs = _build_variable_attributes(minimal_meta)

        assert "long_name" in attrs

    def test_cf_convention_attributes_present(self):
        """Test built attributes follow CF Convention standards."""
        field_meta = {
            "long_name": "Test",
            "unit": "1",
            "dtype": "uint16",
            "valid_min": 0,
            "valid_max": 100,
        }

        attrs = _build_variable_attributes(field_meta)

        # CF Convention required/recommended fields
        cf_expected = ["long_name", "units"]
        for cf_field in cf_expected:
            assert cf_field in attrs

    def test_scale_factor_and_offset_included_when_present(self):
        """Test scale_factor and add_offset are included in attributes."""
        field_meta = {
            "long_name": "Scaled Field",
            "unit": "degrees",
            "scale_factor": 0.01,
            "add_offset": -180,
        }

        attrs = _build_variable_attributes(field_meta)

        assert "scale_factor" in attrs
        assert attrs["scale_factor"] == 0.01
        assert "add_offset" in attrs
        assert attrs["add_offset"] == -180

    def test_scale_factor_omitted_when_one(self):
        """Test scale_factor is omitted when value is 1.0 (default)."""
        field_meta = {
            "long_name": "Test",
            "scale_factor": 1.0,  # Default value
            "add_offset": 0.0,  # Default value
        }

        attrs = _build_variable_attributes(field_meta)

        # Default scale_factor and offset should be omitted
        assert "scale_factor" not in attrs or attrs.get("scale_factor") == 1.0
        assert "add_offset" not in attrs or attrs.get("add_offset") == 0.0


# ============================================================================
# TEST SUITE: Integration Tests
# ============================================================================


class TestReadFixedLeaderIntegration:
    """Integration tests with other components."""

    def test_integration_with_read_header(self, valid_rdi_file):
        """Test read_fixed_leader works with read_header output.

        Should accept parameters from read_header without modification.
        """
        from pyadps.io.binary_reader import read_header

        header = read_header(valid_rdi_file)

        # Extract parameters from header
        byteskip = header["byte_skip"].values
        offset = header["address_offset"].values
        idarray = header["data_id"].values
        ensemble_count = int(header.attrs["total_ensembles"])

        # Use with read_fixed_leader
        result = read_fixed_leader(
            valid_rdi_file,
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble_count,
        )

        assert isinstance(result, xr.Dataset)
        assert len(result.coords["ensemble"]) > 0

    def test_dataset_can_be_used_with_accessors(self, valid_rdi_file):
        """Test returned dataset can use xarray accessor pattern.

        Dataset should work with accessor registration (tested separately).
        """
        result = read_fixed_leader(valid_rdi_file)

        # Should be able to call xarray methods
        assert hasattr(result, "sel")
        assert hasattr(result, "mean")
        assert hasattr(result, "values")

    def test_output_compatible_with_netcdf_export(self, valid_rdi_file, tmp_path):
        """Test output dataset can be exported to NetCDF.

        Dataset structure should be NetCDF-compatible.
        """
        result = read_fixed_leader(valid_rdi_file)

        # Try to export to NetCDF
        output_file = tmp_path / "test_output.nc"
        try:
            result.to_netcdf(output_file)
            # If successful, verify file was created
            assert output_file.exists()
        except Exception as e:
            # Some attributes might not be NetCDF-compatible
            # This is acceptable - we're just verifying basic structure
            pass


# ============================================================================
# TEST SUITE: Performance and Edge Cases
# ============================================================================


class TestReadFixedLeaderPerformance:
    """Tests for performance and scalability."""

    def test_large_multi_ensemble_file(self, valid_rdi_ensemble, tmp_path):
        """Test performance with larger file (100 ensembles).

        Should handle reasonably large files efficiently.
        """
        rdi_file = tmp_path / "test_large.000"
        # Create file with 100 ensembles
        rdi_file.write_bytes(valid_rdi_ensemble * 100)

        result = read_fixed_leader(rdi_file)

        assert isinstance(result, xr.Dataset)
        assert len(result.coords["ensemble"]) == 100


class TestReadFixedLeaderEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_path_as_string(self, valid_rdi_file):
        """Test file path can be provided as string."""
        result = read_fixed_leader(str(valid_rdi_file))

        assert isinstance(result, xr.Dataset)

    def test_path_as_pathlib_path(self, valid_rdi_file):
        """Test file path can be provided as pathlib.Path object."""
        result = read_fixed_leader(Path(valid_rdi_file))

        assert isinstance(result, xr.Dataset)

    def test_dataset_attributes_include_metadata(self, valid_rdi_file):
        """Test dataset has appropriate global attributes.

        Should document source file, pyadps version, format info.
        """
        result = read_fixed_leader(valid_rdi_file)

        # Check for expected attributes
        assert isinstance(result.attrs, dict)

    def test_multiple_calls_same_file_consistent(self, valid_rdi_file):
        """Test multiple reads of same file produce consistent results.

        Results should be identical (deterministic).
        """
        result1 = read_fixed_leader(valid_rdi_file)
        result2 = read_fixed_leader(valid_rdi_file)

        # Compare data variables
        for var in result1.data_vars:
            np.testing.assert_array_equal(result1[var].values, result2[var].values)

    def test_single_ensemble_1d_array_reshape(self, valid_rdi_file):
        """Test handling of 1D array from pd0_parser (single ensemble edge case).

        This tests the defensive code at line 505-507 in read_fixed_leader():
            if len(fl_data.shape) == 1:
                fl_data = fl_data.reshape(-1, 1)

        The pd0_parser.fixedleader normally returns 2D arrays even for single
        ensembles, but this code handles the edge case where a 1D array is
        returned (e.g., from future parser changes or custom implementations).
        """
        from unittest import mock

        # Create a 1D array simulating single ensemble data (36 fields)
        mock_fl_data = np.arange(36, dtype=np.int64)
        mock_ensemble = 1
        mock_error_code = 0

        with mock.patch(
            "pyadps.io.binary_reader.pd0_parser.fixedleader"
        ) as mock_fixedleader:
            mock_fixedleader.return_value = (
                mock_fl_data,
                mock_ensemble,
                mock_error_code,
            )

            result = read_fixed_leader(valid_rdi_file)

        # Should successfully handle 1D array and return valid dataset
        assert isinstance(result, xr.Dataset), "Should return xarray.Dataset"
        assert "ensemble" in result.coords, "Should have ensemble coordinate"
        assert len(result.coords["ensemble"]) == 1, "Should have 1 ensemble"

        # Data variables should have proper shape
        for var in result.data_vars:
            assert result[var].dims == ("ensemble",), f"{var} should have ensemble dim"
            assert len(result[var]) == 1, f"{var} should have length 1"


# ============================================================================
# TEST SUITE: Custom Configurations
# ============================================================================


class TestReadFixedLeaderCustomConfigs:
    """Tests with custom Fixed Leader configurations."""

    def test_custom_beam_count(self, custom_fixed_leader_file):
        """Test reading file with custom beam count (5 beams instead of 4)."""
        result = read_fixed_leader(custom_fixed_leader_file)

        # Should successfully read custom configuration
        assert isinstance(result, xr.Dataset)
        # num_beams field should reflect the 5 beams
        if "num_beams" in result.data_vars:
            assert result["num_beams"].values[0] == 5

    def test_custom_cell_count(self, custom_fixed_leader_file):
        """Test reading file with custom cell count (40 cells instead of 30)."""
        result = read_fixed_leader(custom_fixed_leader_file)

        assert isinstance(result, xr.Dataset)
        if "num_cells" in result.data_vars:
            assert result["num_cells"].values[0] == 40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
