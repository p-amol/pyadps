"""
Comprehensive pytest test suite for pyreadrdi.fixedleader function.

Tests cover:
- Normal operation with valid RDI files
- Fixed Leader data extraction and parsing
- Error handling (file not found, IO errors, corrupted data)
- Field validation (serial numbers, sensor configuration)
- Multiple ensembles and uniform/non-uniform data
- Struct unpacking errors and edge cases
- Old firmware serial number handling
- Return value types and structure (36, n_ensembles) array
- Data type ID verification (must be 0 or 1 for Fixed Leader)
- Integration with fileheader() for automatic parameter retrieval

References
----------
RDI WorkHorse Commands and Output Data Format (Section 5.2, page 126):
- Fixed Leader ID: bytes 1-4 (always 0x0000)
- CPU FW Version/Revision: bytes 5-8
- System Configuration: bytes 9-10
- Real/Sim Flag: byte 11
- 34 additional fields through byte 59
"""

import struct
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


from pyadps import (
    ErrorCode,
    fileheader,
    fixedleader,
)

from .fixtures.ensemble_builder import build_ensemble


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(scope="function")
def valid_rdi_ensemble_with_fl():
    """Generate a valid complete RDI ensemble using shared builder.

    Uses the imported build_ensemble() function from test_ensemble_builder.py.
    Structure follows RDI WorkHorse spec with 7 data types (default config).
    """
    return build_ensemble()


@pytest.fixture
def valid_rdi_file_with_fl(tmp_path, valid_rdi_ensemble_with_fl):
    """Create a temporary file with single valid ensemble containing Fixed Leader."""
    rdi_file = tmp_path / "test_fl_single.000"
    rdi_file.write_bytes(valid_rdi_ensemble_with_fl)
    return rdi_file


@pytest.fixture
def multi_ensemble_rdi_file_with_fl(tmp_path, valid_rdi_ensemble_with_fl):
    """Create a temporary file with multiple valid ensembles."""
    rdi_file = tmp_path / "test_fl_multi.000"
    # Write 3 identical ensembles
    rdi_file.write_bytes(valid_rdi_ensemble_with_fl * 3)
    return rdi_file


@pytest.fixture
def rdi_file_missing_fl_id(tmp_path, valid_rdi_ensemble_with_fl):
    """Create file where Fixed Leader section has wrong ID.

    Corrupts the Fixed Leader ID (bytes at offset position) to an invalid value
    (9999) while maintaining a valid checksum so that fileheader() passes
    checksum verification. This allows fixedleader() to detect the bad ID.
    """
    data = bytearray(valid_rdi_ensemble_with_fl)

    # Parse header to find Fixed Leader section offset
    # Header structure: BBHBB at bytes 0-5, then offset array follows
    fl_offset_in_header = 6 + (2 * 0)  # Byte position to read first offset from
    fl_position = struct.unpack(
        "<H", data[fl_offset_in_header : fl_offset_in_header + 2]
    )[0]

    # Corrupt the Fixed Leader ID (2 bytes at fl_position)
    data[fl_position : fl_position + 2] = struct.pack("<H", 9999)  # Invalid ID

    # Recalculate and update checksum to keep ensemble valid
    # Per RDI spec Section 7.2, checksum is the sum of all bytes
    # from start of ensemble through (byte_count - 1), excluding the
    # 2-byte checksum itself at the end
    byte_count_offset = 2  # Byte count is at bytes 2-3
    byte_count_val = struct.unpack(
        "<H", data[byte_count_offset : byte_count_offset + 2]
    )[0]

    # Checksum covers bytes [0:byte_count_val] (byte_count_val is exclusive upper bound)
    checksum_payload = data[:byte_count_val]
    calculated_checksum = sum(checksum_payload) & 0xFFFF

    # Checksum is stored at byte_count_val (0-indexed)
    checksum_position = byte_count_val
    data[checksum_position : checksum_position + 2] = struct.pack(
        "<H", calculated_checksum
    )

    rdi_file = tmp_path / "test_fl_missing_id.000"
    rdi_file.write_bytes(bytes(data))
    return rdi_file


@pytest.fixture
def rdi_file_truncated_fl(tmp_path, valid_rdi_ensemble_with_fl):
    """Create file where Fixed Leader section is truncated."""
    rdi_file = tmp_path / "test_fl_truncated.000"
    # Write only 30 bytes of the first ensemble (incomplete FL section)
    rdi_file.write_bytes(valid_rdi_ensemble_with_fl[:30])
    return rdi_file


@pytest.fixture
def rdi_file_multiple_with_corruption(tmp_path, valid_rdi_ensemble_with_fl):
    """Create file with valid ensemble followed by corrupted ensemble."""
    rdi_file = tmp_path / "test_fl_corrupted.000"

    first_ensemble = valid_rdi_ensemble_with_fl

    # Create a corrupted second ensemble (truncated)
    corrupted = first_ensemble[:100]  # Only partial data

    rdi_file.write_bytes(first_ensemble + corrupted)
    return rdi_file


# ============================================================================
# TESTS: Input Validation
# ============================================================================


class TestFixedleaderInputValidation:
    """Test fixedleader function input validation."""

    def test_missing_filename_argument(self):
        """Test that calling fixedleader without arguments raises TypeError."""
        with pytest.raises(TypeError):
            fixedleader()

    def test_invalid_filename_type_integer(self):
        """Test that passing an integer returns FILE_NOT_FOUND error."""
        data, ensemble, error_code = fixedleader(12345)
        assert isinstance(data, np.ndarray)
        assert isinstance(ensemble, int)
        assert error_code == ErrorCode.FILE_NOT_FOUND.code

    def test_invalid_filename_type_none(self):
        """Test that passing None returns FILE_NOT_FOUND error."""
        data, ensemble, error_code = fixedleader(None)
        assert error_code == ErrorCode.FILE_NOT_FOUND.code

    def test_invalid_filename_type_list(self):
        """Test that passing a list returns FILE_NOT_FOUND error."""
        data, ensemble, error_code = fixedleader(["file.000"])
        assert error_code == ErrorCode.FILE_NOT_FOUND.code


# ============================================================================
# TESTS: File Access
# ============================================================================


class TestFixedleaderFileAccess:
    """Test fixedleader error handling for file access issues."""

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        data, ensemble, error_code = fixedleader("nonexistent_file_xyz.000")
        assert error_code == ErrorCode.FILE_NOT_FOUND.code

    def test_file_not_found_with_path_object(self):
        """Test file not found using pathlib.Path."""
        data, ensemble, error_code = fixedleader(Path("/nonexistent/path/file.000"))
        assert error_code == ErrorCode.FILE_NOT_FOUND.code

    def test_permission_denied(self):
        """Test handling of permission denied error via mocking."""
        with mock.patch("builtins.open", side_effect=PermissionError("Access denied")):
            data, ensemble, error_code = fixedleader("somefile.000")
            assert error_code == ErrorCode.PERMISSION_DENIED.code

    def test_io_error_generic(self):
        """Test handling of generic IOError."""
        with mock.patch("builtins.open", side_effect=IOError("IO problem")):
            data, ensemble, error_code = fixedleader("somefile.000")
            assert error_code == ErrorCode.IO_ERROR.code

    def test_oserror_generic(self):
        """Test handling of generic OSError."""
        with mock.patch("builtins.open", side_effect=OSError("OS problem")):
            data, ensemble, error_code = fixedleader("somefile.000")
            assert error_code == ErrorCode.IO_ERROR.code


# ============================================================================
# TESTS: Return Value Structure
# ============================================================================


class TestFixedleaderReturnStructure:
    """Test fixedleader return value types and shapes."""

    def test_return_is_tuple_length_3(self, valid_rdi_file_with_fl):
        """Test that fixedleader returns 3-tuple: (data, ensemble, error_code)."""
        result = fixedleader(str(valid_rdi_file_with_fl))
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_return_data_is_numpy_array(self, valid_rdi_file_with_fl):
        """Test that first return value is numpy array."""
        data, _, _ = fixedleader(str(valid_rdi_file_with_fl))
        assert isinstance(data, np.ndarray)

    def test_return_data_dtype_int64(self, valid_rdi_file_with_fl):
        """Test that data array has int64 dtype."""
        data, _, _ = fixedleader(str(valid_rdi_file_with_fl))
        assert data.dtype == np.int64

    def test_return_data_shape_36_rows(self, valid_rdi_file_with_fl):
        """Test that data array has 36 rows (Fixed Leader fields)."""
        data, _, _ = fixedleader(str(valid_rdi_file_with_fl))
        assert data.shape[0] == 36

    def test_return_ensemble_count_matches_shape(self, valid_rdi_file_with_fl):
        """Test that returned ensemble count matches data array column count."""
        data, ensemble_count, _ = fixedleader(str(valid_rdi_file_with_fl))
        assert data.shape[1] == ensemble_count

    def test_return_error_code_is_integer(self, valid_rdi_file_with_fl):
        """Test that error code is an integer."""
        _, _, error_code = fixedleader(str(valid_rdi_file_with_fl))
        assert isinstance(error_code, (int, np.integer))

    def test_return_error_code_zero_on_success(self, valid_rdi_file_with_fl):
        """Test that error code is 0 on successful parse."""
        _, _, error_code = fixedleader(str(valid_rdi_file_with_fl))
        assert error_code == 0


# ============================================================================
# TESTS: Fixed Leader Data Extraction
# ============================================================================


class TestFixedleaderDataExtraction:
    """Test correct extraction of Fixed Leader fields."""

    def test_fixed_leader_id_field(self, valid_rdi_file_with_fl):
        """Test that Fixed Leader ID (field 0) is extracted correctly."""
        data, _, error_code = fixedleader(str(valid_rdi_file_with_fl))
        assert error_code == 0
        # Fixed Leader ID should be 0x0000 (little-endian)
        assert data[0, 0] == 0

    def test_cpu_fw_version_field(self, valid_rdi_file_with_fl):
        """Test that CPU FW Version (field 1) is extracted correctly."""
        data, _, error_code = fixedleader(str(valid_rdi_file_with_fl))
        assert error_code == 0
        # From fixture: CPU FW Ver = 16
        assert data[1, 0] == 16

    def test_cpu_fw_revision_field(self, valid_rdi_file_with_fl):
        """Test that CPU FW Revision (field 2) is extracted correctly."""
        data, _, error_code = fixedleader(str(valid_rdi_file_with_fl))
        assert error_code == 0
        # From fixture: CPU FW Rev = 7
        assert data[2, 0] == 5

    def test_num_beams_field(self, valid_rdi_file_with_fl):
        """Test that Number of Beams (field 6) is extracted correctly."""
        data, _, error_code = fixedleader(str(valid_rdi_file_with_fl))
        assert error_code == 0
        # From fixture: Num beams = 4
        assert data[6, 0] == 4

    def test_num_cells_field(self, valid_rdi_file_with_fl):
        """Test that Number of Cells (field 7) is extracted correctly."""
        data, _, error_code = fixedleader(str(valid_rdi_file_with_fl))
        assert error_code == 0
        # From fixture: Num cells = 30
        assert data[7, 0] == 30

    def test_beam_angle_field(self, valid_rdi_file_with_fl):
        """Test that Beam Angle (field 35) is extracted correctly."""
        data, _, error_code = fixedleader(str(valid_rdi_file_with_fl))
        assert error_code == 0
        # From fixture: Beam angle = 20
        assert data[35, 0] == 20

    def test_all_36_fields_present(self, valid_rdi_file_with_fl):
        """Test that all 36 Fixed Leader fields are extracted."""
        data, _, error_code = fixedleader(str(valid_rdi_file_with_fl))
        assert error_code == 0
        assert data.shape[0] == 36
        # All fields should have values (at least initialized)
        assert data.shape == (36, 1)


# ============================================================================
# TESTS: Multiple Ensemble Handling
# ============================================================================


class TestFixedleaderMultipleEnsembles:
    """Test fixedleader with multiple ensembles in file."""

    def test_multiple_ensembles_extracted(self, multi_ensemble_rdi_file_with_fl):
        """Test that all ensembles are extracted from multi-ensemble file."""
        data, ensemble_count, error_code = fixedleader(
            str(multi_ensemble_rdi_file_with_fl)
        )
        assert error_code == 0
        assert ensemble_count == 3
        assert data.shape[1] == 3

    def test_multiple_ensembles_have_identical_values(
        self, multi_ensemble_rdi_file_with_fl
    ):
        """Test that identical ensembles have identical Fixed Leader data."""
        data, _, error_code = fixedleader(str(multi_ensemble_rdi_file_with_fl))
        assert error_code == 0

        # Compare all fields across ensembles
        for field in range(36):
            # All ensembles should have identical values since they're identical
            assert np.all(data[field, :] == data[field, 0])

    def test_each_ensemble_has_36_fields(self, multi_ensemble_rdi_file_with_fl):
        """Test that each ensemble has all 36 fields extracted."""
        data, ensemble_count, _ = fixedleader(str(multi_ensemble_rdi_file_with_fl))
        assert data.shape == (36, ensemble_count)


# ============================================================================
# TESTS: Error Handling - Corrupted Data
# ============================================================================


class TestFixedleaderCorruptedData:
    """Test fixedleader error handling with corrupted data."""

    def test_missing_fixed_leader_id(self, rdi_file_missing_fl_id):
        """Test handling when Fixed Leader ID is not 0 or 1."""
        data, ensemble_count, error_code = fixedleader(str(rdi_file_missing_fl_id))
        # Should detect invalid ID and return error
        assert error_code == ErrorCode.ID_NOT_FOUND.code
        # Ensembles should be truncated at 0
        assert ensemble_count == 0

    def test_truncated_fixed_leader_data(self, rdi_file_truncated_fl):
        """Test handling of truncated Fixed Leader section."""
        data, ensemble_count, error_code = fixedleader(str(rdi_file_truncated_fl))
        # Should handle truncation gracefully
        assert error_code in (
            ErrorCode.FILE_CORRUPTED.code,
            ErrorCode.ID_NOT_FOUND.code,
        )

    def test_corrupted_second_ensemble(self, rdi_file_multiple_with_corruption):
        """Test handling of file with valid first ensemble and corrupted second."""
        data, ensemble_count, error_code = fixedleader(
            str(rdi_file_multiple_with_corruption)
        )
        # Should have parsed at least the first ensemble
        assert ensemble_count >= 1
        # Error code should indicate corruption
        assert error_code != 0 or ensemble_count == 1


# ============================================================================
# TESTS: Integration with fileheader
# ============================================================================


class TestFixedleaderIntegration:
    """Test fixedleader integration with fileheader."""

    def test_fixedleader_without_fileheader_params(self, valid_rdi_file_with_fl):
        """Test that fixedleader works without fileheader parameters.

        Should internally call fileheader when parameters not provided.
        """
        # Call with only filename (no fileheader parameters)
        data, ensemble_count, error_code = fixedleader(str(valid_rdi_file_with_fl))
        assert error_code == 0
        assert ensemble_count == 1
        assert data.shape == (36, 1)

    def test_fixedleader_with_fileheader_params(self, valid_rdi_file_with_fl):
        """Test that fixedleader uses provided fileheader parameters."""
        # First call fileheader
        (
            source_id,
            header_id,
            byteskip,
            offset,
            idarray,
            ensemble_count,
            error_code_fh,
        ) = fileheader(str(valid_rdi_file_with_fl))

        assert error_code_fh == 0

        # Now call fixedleader with those parameters
        data, ens_count, error_code = fixedleader(
            str(valid_rdi_file_with_fl),
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble_count,
        )

        assert error_code == 0
        assert ens_count == ensemble_count
        assert data.shape[1] == ensemble_count

    def test_fileheader_params_match_results(self, valid_rdi_file_with_fl):
        """Test that results are consistent whether using fileheader params or not."""
        # Without params
        data1, ens1, err1 = fixedleader(str(valid_rdi_file_with_fl))

        # With params
        (_, _, bs, off, ida, ens_fh, err_fh) = fileheader(str(valid_rdi_file_with_fl))
        data2, ens2, err2 = fixedleader(
            str(valid_rdi_file_with_fl),
            byteskip=bs,
            offset=off,
            idarray=ida,
            ensemble=ens_fh,
        )

        # Results should be identical
        assert err1 == err2
        assert ens1 == ens2
        np.testing.assert_array_equal(data1, data2)


# ============================================================================
# TESTS: Edge Cases and Boundary Conditions
# ============================================================================


class TestFixedleaderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_file(self, tmp_path):
        """Test handling of empty file."""
        empty_file = tmp_path / "empty.000"
        empty_file.write_bytes(b"")

        data, ensemble_count, error_code = fixedleader(str(empty_file))
        # Should return error or empty result
        assert error_code != 0 or ensemble_count == 0

    def test_data_array_dtype_consistency(self, valid_rdi_file_with_fl):
        """Test that all array values are properly cast to int64."""
        data, _, _ = fixedleader(str(valid_rdi_file_with_fl))
        # All values should be int64
        assert data.dtype == np.int64
        # No NaN or inf values
        assert not np.any(np.isnan(data.astype(float)))
        assert not np.any(np.isinf(data.astype(float)))

    def test_consistent_field_meanings_across_ensembles(
        self, multi_ensemble_rdi_file_with_fl
    ):
        """Test that field meanings are consistent across multiple ensembles."""
        data, _, _ = fixedleader(str(multi_ensemble_rdi_file_with_fl))

        # Num beams should be consistent
        assert np.all(data[6, :] == data[6, 0])

        # Num cells should be consistent
        assert np.all(data[7, :] == data[7, 0])


# ============================================================================
# TESTS: Struct Unpacking Errors
# ============================================================================


class TestFixedleaderStructErrors:
    """Test handling of struct unpacking errors."""

    def test_struct_unpack_error_recovery(self, rdi_file_truncated_fl):
        """Test that struct unpacking errors are caught and handled."""
        # Should not raise exception, should return error code
        data, ensemble_count, error_code = fixedleader(str(rdi_file_truncated_fl))

        # Should complete without raising exception
        assert error_code != 0 or ensemble_count == 0


# ============================================================================
# TESTS: Data Validation
# ============================================================================


class TestFixedleaderDataValidation:
    """Test data validation and sanity checks."""

    def test_num_beams_is_reasonable(self, valid_rdi_file_with_fl):
        """Test that number of beams is reasonable (typically 4 or 5)."""
        data, _, error_code = fixedleader(str(valid_rdi_file_with_fl))
        assert error_code == 0
        num_beams = data[6, 0]
        assert 3 <= num_beams <= 8  # Typical ADCP range

    def test_num_cells_is_positive(self, valid_rdi_file_with_fl):
        """Test that number of cells is positive."""
        data, _, error_code = fixedleader(str(valid_rdi_file_with_fl))
        assert error_code == 0
        num_cells = data[7, 0]
        assert num_cells > 0

    def test_system_configuration_is_not_negative(self, valid_rdi_file_with_fl):
        """Test that system configuration value is non-negative."""
        data, _, error_code = fixedleader(str(valid_rdi_file_with_fl))
        assert error_code == 0
        # System config should be a valid bit pattern
        assert data[3, 0] >= 0


# ============================================================================
# TESTS: Pathlib Support
# ============================================================================


class TestFixedleaderPathlibSupport:
    """Test that fixedleader works with pathlib.Path objects."""

    def test_path_object_string_conversion(self, valid_rdi_file_with_fl):
        """Test that fixedleader accepts pathlib.Path objects."""
        data, ensemble_count, error_code = fixedleader(valid_rdi_file_with_fl)
        assert error_code == 0
        assert ensemble_count == 1

    def test_string_path_works_same_as_path_object(self, valid_rdi_file_with_fl):
        """Test that string paths and Path objects give identical results."""
        data1, ens1, err1 = fixedleader(str(valid_rdi_file_with_fl))
        data2, ens2, err2 = fixedleader(valid_rdi_file_with_fl)

        assert err1 == err2
        assert ens1 == ens2
        np.testing.assert_array_equal(data1, data2)


# ============================================================================
# TESTS: Field Extraction Correctness
# ============================================================================


class TestFixedleaderFieldCorrectness:
    """Test correctness of individual field extraction from binary data."""

    def test_cpu_fw_version_is_byte_value(self, valid_rdi_file_with_fl):
        """Test that CPU FW version field is correctly unpacked as single byte."""
        data, _, error_code = fixedleader(str(valid_rdi_file_with_fl))
        assert error_code == 0
        # Should be in reasonable range for version number
        assert 0 <= data[1, 0] <= 255

    def test_lag_length_is_single_byte(self, valid_rdi_file_with_fl):
        """Test that lag length (field 5) is single byte."""
        data, _, error_code = fixedleader(str(valid_rdi_file_with_fl))
        assert error_code == 0
        # From fixture: lag length = 0 (default)
        assert data[5, 0] == 0
        assert 0 <= data[5, 0] <= 255

    def test_pings_per_ensemble_is_16bit(self, valid_rdi_file_with_fl):
        """Test that pings per ensemble (field 8) is 16-bit value."""
        data, _, error_code = fixedleader(str(valid_rdi_file_with_fl))
        assert error_code == 0
        # From fixture: pings per ens = 1
        assert data[8, 0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
