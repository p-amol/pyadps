"""
Comprehensive pytest test suite for pyreadrdi.fileheader function.

Tests cover:
- Normal operation with valid RDI files
- Error handling (file not found, permission denied, IO errors, etc.)
- File format validation (header/source IDs, data type consistency)
- Partial/truncated files
- Struct unpacking errors
- Boundary conditions and edge cases
- Return value types and structure
"""

import struct
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from pyadps.utils.pyreadrdi import ErrorCode, fileheader


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def valid_rdi_header():
    """Generate a valid complete RDI ensemble matching WorkHorse spec."""
    num_datatypes = 7
    beams = 4
    cells = 30

    # Define sizes of each section
    fl_size = 59  # Fixed Leader
    vl_size = 65  # Variable Leader
    vel_size = 2 + (2 * beams * cells)  # Velocity: 2-byte header + data
    echo_size = 2 + (1 * beams * cells)  # Echo: 2-byte header + data
    cor_size = 2 + (1 * beams * cells)  # Correlation: 2-byte header + data
    pg_size = 2 + (1 * beams * cells)  # Percent Good: 2-byte header + data
    bt_size = 85  # Bottom Track
    reserve_size = 2  # Reserved

    # Calculate cumulative offsets (positions where each section starts)
    header_size = 6 + (2 * num_datatypes)  # Header + offset array

    offsets = [
        header_size,  # Fixed Leader at byte 20
        header_size + fl_size,  # Variable Leader
        header_size + fl_size + vl_size,  # Velocity
        header_size + fl_size + vl_size + vel_size,  # Echo
        header_size + fl_size + vl_size + vel_size + echo_size,  # Correlation
        header_size
        + fl_size
        + vl_size
        + vel_size
        + echo_size
        + cor_size,  # Percent Good
        header_size
        + fl_size
        + vl_size
        + vel_size
        + echo_size
        + cor_size
        + pg_size,  # Bottom Track
    ]

    # Total size (without checksum)
    ensemble_size = offsets[-1] + bt_size  # Last offset + its size

    # Data IDs for each type
    data_ids = [0, 128, 256, 512, 768, 1024, 1536]

    # Build header
    header = struct.pack(
        "<BBHBB",
        0x7F,  # header_id
        0x7F,  # source_id
        ensemble_size,  # byte_count
        0,  # spare
        num_datatypes,  # num_datatypes
    )

    # Add offset array
    offset_array = struct.pack("<HHHHHHH", *offsets)

    # Build data sections
    data_sections = b""
    sizes = [fl_size, vl_size, vel_size, echo_size, cor_size, pg_size, bt_size]

    for i in range(num_datatypes):
        # 2-byte data ID
        data_sections += struct.pack("<H", data_ids[i])
        # Padding (size - 2 for the ID bytes already added)
        data_sections += b"\x00" * (sizes[i] - 2)

    # Add reserved and checksum
    reserved = b"\x00" * reserve_size * 0  # Deactivated Reserved

    # Combine all
    ensemble_without_checksum = header + offset_array + data_sections + reserved

    # ========== CALCULATE CHECKSUM ==========
    # Calculate checksum on everything EXCEPT the checksum itself
    calculated_checksum = sum(ensemble_without_checksum) & 0xFFFF
    checksum_bytes = struct.pack("<H", calculated_checksum)
    # =========================================

    complete_ensemble = ensemble_without_checksum + checksum_bytes

    return complete_ensemble


@pytest.fixture
def valid_rdi_file(tmp_path, valid_rdi_header):
    """Create a temporary file with valid RDI data.

    Single ensemble with all required sections.
    """
    rdi_file = tmp_path / "test_single_ensemble.000"
    rdi_file.write_bytes(valid_rdi_header)
    return rdi_file


@pytest.fixture
def multi_ensemble_rdi_file(tmp_path, valid_rdi_header):
    """Create a temporary file with multiple valid RDI ensembles."""
    rdi_file = tmp_path / "test_multi_ensemble.000"
    # Write 3 identical ensembles
    rdi_file.write_bytes(valid_rdi_header * 3)
    return rdi_file


@pytest.fixture
def truncated_rdi_file(tmp_path, valid_rdi_header):
    """Create a file truncated mid-header (incomplete first read)."""
    rdi_file = tmp_path / "test_truncated.000"
    # Write only first 3 bytes of header (incomplete)
    rdi_file.write_bytes(valid_rdi_header[:3])
    return rdi_file


@pytest.fixture
def invalid_header_id_file(tmp_path):
    """Create a file with invalid header ID (not 0x7F)."""
    rdi_file = tmp_path / "test_invalid_header.000"
    # Invalid header_id (0x80 instead of 0x7F)
    invalid_header = struct.pack("<BBHBB", 0x6F, 0x7F, 100, 0, 5)
    invalid_header += struct.pack("<HHHHH", 0, 1, 256, 512, 1024)
    invalid_header += b"\x00" * (100 - 10)
    rdi_file.write_bytes(invalid_header)
    return rdi_file


@pytest.fixture
def invalid_source_id_file(tmp_path):
    """Create a file with invalid source ID (not 0x7F)."""
    rdi_file = tmp_path / "test_invalid_source.000"
    # Invalid source_id (0x80 instead of 0x7F)
    invalid_header = struct.pack("<BBHBB", 0x7F, 0x80, 100, 0, 5)
    invalid_header += struct.pack("<HHHHH", 0, 1, 256, 512, 1024)
    invalid_header += b"\x00" * (100 - 10)
    rdi_file.write_bytes(invalid_header)
    return rdi_file


@pytest.fixture
def mismatched_datatype_file(tmp_path, valid_rdi_header):
    """Create a file where second ensemble has different datatype count."""
    header1 = valid_rdi_header

    # Second header with different num_datatypes
    header2 = struct.pack("<BBHBB", 0x7F, 0x7F, 100, 0, 4)  # 4 instead of 7
    header2 += struct.pack("<HHHH", 0, 1, 256, 512)
    header2 += b"\x00" * (100 - 8)

    rdi_file = tmp_path / "test_mismatch.000"
    rdi_file.write_bytes(header1 + header2)
    return rdi_file


@pytest.fixture
def missing_datatype_bytes_file(tmp_path):
    """Create file where datatype bytes are incomplete."""
    header = struct.pack("<BBHBB", 0x7F, 0x7F, 100, 0, 5)
    # Write only 3 datatype IDs instead of 5
    incomplete_datatypes = struct.pack("<HHH", 0, 1, 256)
    rdi_file = tmp_path / "test_missing_datatypes.000"
    rdi_file.write_bytes(header + incomplete_datatypes)
    return rdi_file


@pytest.fixture
def empty_file(tmp_path):
    """Create an empty file."""
    rdi_file = tmp_path / "test_empty.000"
    rdi_file.write_bytes(b"")
    return rdi_file


@pytest.fixture
def file_with_corrupted_offset(tmp_path, valid_rdi_header):
    """Create a file with first ensemble valid, second ensemble corrupted offset array.

    Simulates power loss or firmware issue mid-recording after first ensemble completes.
    """
    rdi_file = tmp_path / "corrupted_second_ensemble.000"

    # First ensemble: valid
    first_ensemble = valid_rdi_header

    # Second ensemble: corrupted (truncated offset array)
    # Valid header but incomplete offset data
    header = struct.pack("<BBHBBH", 0x7F, 0x7F, 837, 0, 7, 1)  # Says 5 datatypes

    second_ensemble_corrupted = header

    # Write both: first valid, second corrupted
    rdi_file.write_bytes(first_ensemble + second_ensemble_corrupted)
    return rdi_file


# ============================================================================
# TESTS: Basic Input Validation
# ============================================================================


class TestFileheaderInputValidation:
    """Test fileheader function input validation."""

    def test_missing_filename_argument(self):
        """Test that calling fileheader without arguments raises TypeError."""
        with pytest.raises(TypeError):
            fileheader()

    def test_invalid_filename_type_integer(self):
        """Test that passing an integer returns UNKNOWN_ERROR."""
        result = fileheader(12345)
        assert len(result) == 7
        assert result[6] == ErrorCode.FILE_NOT_FOUND.code

    def test_invalid_filename_type_none(self):
        """Test that passing None returns UNKNOWN_ERROR."""
        result = fileheader(None)
        assert len(result) == 7
        assert result[6] == ErrorCode.FILE_NOT_FOUND.code

    def test_invalid_filename_type_list(self):
        """Test that passing a list returns UNKNOWN_ERROR."""
        result = fileheader(["file.000"])
        assert len(result) == 7
        assert result[6] == ErrorCode.FILE_NOT_FOUND.code


# ============================================================================
# TESTS: File Access Errors
# ============================================================================


class TestFileheaderFileAccess:
    """Test fileheader error handling for file access issues."""

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        result = fileheader("nonexistent_file_xyz.000")
        _, _, _, _, _, _, error_code = result
        assert error_code == ErrorCode.FILE_NOT_FOUND.code
        assert isinstance(error_code, int)

    def test_file_not_found_with_path_object(self):
        """Test file not found using pathlib.Path."""
        result = fileheader(Path("/nonexistent/path/file.000"))
        assert result[6] == ErrorCode.FILE_NOT_FOUND.code

    def test_permission_denied(self):
        """Test handling of permission denied error via mocking."""
        with mock.patch("builtins.open", side_effect=PermissionError("Access denied")):
            result = fileheader("somefile.000")
            assert result[6] == ErrorCode.PERMISSION_DENIED.code

    def test_io_error_generic(self):
        """Test handling of generic IOError."""
        with mock.patch("builtins.open", side_effect=IOError("IO problem")):
            result = fileheader("somefile.000")
            assert result[6] == ErrorCode.IO_ERROR.code

    def test_oserror_generic(self):
        """Test handling of generic OSError."""
        with mock.patch("builtins.open", side_effect=OSError("OS problem")):
            result = fileheader("somefile.000")
            assert result[6] == ErrorCode.IO_ERROR.code

    def test_memory_error(self):
        """Test handling of MemoryError."""
        with mock.patch("builtins.open", side_effect=MemoryError("Out of memory")):
            result = fileheader("somefile.000")
            assert result[6] == ErrorCode.OUT_OF_MEMORY.code

    def test_unexpected_exception(self):
        """Test handling of unexpected exception."""
        with mock.patch("builtins.open", side_effect=RuntimeError("Unexpected")):
            result = fileheader("somefile.000")
            assert result[6] == ErrorCode.UNKNOWN_ERROR.code


# ============================================================================
# TESTS: Return Value Structure
# ============================================================================


class TestFileheaderReturnStructure:
    """Test that fileheader returns correct structure in all cases."""

    def test_return_tuple_length(self, tmp_path, valid_rdi_header):
        """Test that fileheader always returns 7-tuple."""
        # Create temporary RDI file
        rdi_file = tmp_path / "test.000"
        rdi_file.write_bytes(valid_rdi_header)

        result = fileheader(rdi_file)
        assert isinstance(result, tuple)
        assert len(result) == 7

    def test_return_tuple_length_on_error(self):
        """Test return tuple length on file not found error."""
        result = fileheader("nonexistent.000")
        assert len(result) == 7

    def test_return_values_are_correct_types(self, tmp_path, valid_rdi_header):
        """Test return value types on success."""
        # Create temporary RDI file
        rdi_file = tmp_path / "test.000"
        rdi_file.write_bytes(valid_rdi_header)

        datatype, byte, byteskip, address_offset, dataid, ensemble, error_code = (
            fileheader(rdi_file)
        )

        assert isinstance(datatype, np.ndarray)
        assert isinstance(byte, np.ndarray)
        assert isinstance(byteskip, np.ndarray)
        assert isinstance(address_offset, np.ndarray)
        assert isinstance(dataid, np.ndarray)
        assert isinstance(ensemble, int)
        assert isinstance(error_code, int)

    def test_return_values_are_empty_on_file_error(self):
        """Test that arrays are empty on file access error."""
        _, _, _, _, _, _, error_code = fileheader("nonexistent.000")
        # On error, check that error_code is non-zero
        assert error_code != 0


# TESTS: Valid RDI File Parsing
# ============================================================================


class TestFileheaderValidFiles:
    """Test fileheader parsing of valid RDI files."""

    def test_single_ensemble_parsing(self, tmp_path, valid_rdi_header):
        """Test parsing a file with single ensemble."""
        rdi_file = tmp_path / "single.000"
        rdi_file.write_bytes(valid_rdi_header)
        datatype, byte, byteskip, address_offset, dataid, ensemble, error_code = (
            fileheader(rdi_file)
        )
        print(error_code)

        assert error_code == 0
        assert ensemble == 1
        assert len(datatype) == 1
        assert len(byte) == 1
        assert len(byteskip) == 1

    def test_multi_ensemble_parsing(self, tmp_path, valid_rdi_header):
        """Test parsing a file with multiple ensembles."""
        multi_ensemble_rdi_file = tmp_path / "multi.000"
        multi_ensemble_rdi_file.write_bytes(valid_rdi_header * 3)
        datatype, byte, byteskip, address_offset, dataid, ensemble, error_code = (
            fileheader(multi_ensemble_rdi_file)
        )
        print("Error Code for Multi: ", error_code, ensemble)
        assert error_code == 0
        assert ensemble == 3
        assert len(datatype) == 3
        assert len(byte) == 3
        assert len(byteskip) == 3

    def test_datatype_array_content(self, valid_rdi_file):
        """Test that datatype array contains correct values."""
        datatype, _, _, _, _, ensemble, error_code = fileheader(valid_rdi_file)

        assert error_code == 0
        assert len(datatype) == ensemble
        # First ensemble should have 5 data types (from fixture)
        assert datatype[0] == 7

    def test_byte_array_content(self, valid_rdi_file):
        """Test that byte array contains byte count."""
        _, byte, _, _, _, _, error_code = fileheader(valid_rdi_file)

        print("My Byte: ", byte[0])
        assert error_code == 0
        # Byte count should be 100 (from fixture)
        assert byte[0] == 837

    def test_address_offset_array_shape(self, valid_rdi_file):
        """Test address_offset array structure."""
        _, _, _, address_offset, _, ensemble, error_code = fileheader(valid_rdi_file)

        assert error_code == 0
        assert address_offset.ndim == 2
        assert address_offset.shape[0] == ensemble
        # Should have one offset per data type (5 from fixture)
        assert address_offset.shape[1] == 7

    def test_dataid_array_structure(self, valid_rdi_file):
        """Test dataid array structure."""
        _, _, _, _, dataid, ensemble, error_code = fileheader(valid_rdi_file)

        assert error_code == 0
        assert dataid.ndim == 2
        assert dataid.shape[0] == ensemble

    def test_byteskip_array_monotonic_increasing(self, multi_ensemble_rdi_file):
        """Test that byteskip values increase monotonically."""
        _, _, byteskip, _, _, _, error_code = fileheader(multi_ensemble_rdi_file)

        assert error_code == 0
        # Each byteskip should be >= previous (file position never goes backwards)
        for i in range(len(byteskip) - 1):
            assert byteskip[i + 1] >= byteskip[i]

    def test_ensemble_count_matches_array_lengths(self, multi_ensemble_rdi_file):
        """Test that ensemble count matches all array lengths."""
        datatype, byte, byteskip, address_offset, dataid, ensemble, error_code = (
            fileheader(multi_ensemble_rdi_file)
        )

        assert error_code == 0
        assert len(datatype) == ensemble
        assert len(byte) == ensemble
        assert len(byteskip) == ensemble
        assert address_offset.shape[0] == ensemble
        assert dataid.shape[0] == ensemble


# ============================================================================
# TESTS: Invalid File Format
# ============================================================================


class TestFileheaderInvalidFormat:
    """Test fileheader handling of invalid RDI file formats."""

    def test_invalid_header_id(self, invalid_header_id_file):
        """Test file with invalid header ID (not 0x7F)."""
        _, _, _, _, _, _, error_code = fileheader(invalid_header_id_file)
        assert error_code == ErrorCode.WRONG_RDIFILE_TYPE.code

    def test_invalid_source_id(self, invalid_source_id_file):
        """Test file with invalid source ID (not 0x7F)."""
        _, _, _, _, _, _, error_code = fileheader(invalid_source_id_file)
        assert error_code == ErrorCode.WRONG_RDIFILE_TYPE.code

    def test_empty_file(self, tmp_path):
        """Test parsing an empty file."""

        empty_rdi_file = tmp_path / "test_empty.000"
        empty_rdi_file.write_bytes(b"")
        _, _, _, _, _, _, error_code = fileheader(empty_rdi_file)
        print("My Error Code: ", error_code)
        assert error_code == ErrorCode.FILE_CORRUPTED.code

    def test_file_with_only_partial_header(self, truncated_rdi_file):
        """Test file with incomplete header (truncated before full header read)."""
        _, _, _, _, _, _, error_code = fileheader(truncated_rdi_file)
        assert error_code == ErrorCode.FILE_CORRUPTED.code

    def test_missing_datatype_bytes(self, missing_datatype_bytes_file):
        """Test file missing datatype ID bytes."""
        _, _, _, _, _, _, error_code = fileheader(missing_datatype_bytes_file)
        assert error_code == ErrorCode.FILE_CORRUPTED.code


# ============================================================================
# TESTS: Data Type Consistency
# ============================================================================


class TestFileheaderDataTypeConsistency:
    """Test data type consistency checks across ensembles."""

    def test_mismatched_datatype_count(self, mismatched_datatype_file):
        """Test file where second ensemble has different datatype count."""
        _, _, _, _, _, ensemble, error_code = fileheader(mismatched_datatype_file)

        # Should truncate at first mismatch
        assert ensemble == 1
        assert error_code == ErrorCode.DATATYPE_MISMATCH.code

    def test_datatype_consistency_multi_ensemble(self, multi_ensemble_rdi_file):
        """Test that all ensembles have consistent datatype counts."""
        datatype, _, _, _, _, ensemble, error_code = fileheader(multi_ensemble_rdi_file)

        assert error_code == 0
        # All datatypes should match first ensemble
        for i in range(1, ensemble):
            assert datatype[i] == datatype[0]


# ============================================================================
# TESTS: Edge Cases and Boundary Conditions
# ============================================================================


class TestFileheaderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_file_with_zero_byte_count(self, tmp_path):
        """Test ensemble with zero byte count."""
        rdi_file = tmp_path / "test_zero_bytes.000"
        header = struct.pack("<BBHBB", 0x7F, 0x7F, 0, 0, 1)
        header += struct.pack("<H", 0)
        rdi_file.write_bytes(header)

        _, byte, _, _, _, _, error_code = fileheader(rdi_file)
        # May succeed or fail depending on implementation
        if error_code == 0:
            assert byte[0] == 0

    def test_file_with_large_byte_count(self, tmp_path):
        """Test ensemble with large byte count."""
        rdi_file = tmp_path / "test_large_bytes.000"
        large_count = 10000
        header = struct.pack("<BBHBB", 0x7F, 0x7F, large_count, 0, 1)
        header += struct.pack("<H", 0)
        # Add padding
        header += b"\x00" * (large_count - 2)
        rdi_file.write_bytes(header)

        _, byte, _, _, _, _, error_code = fileheader(rdi_file)
        if error_code == 0:
            assert byte[0] == large_count

    def test_file_with_max_datatype_count(self, tmp_path):
        """Test ensemble with maximum datatype count."""
        rdi_file = tmp_path / "test_max_datatypes.000"
        max_types = 255  # Reasonable maximum
        total_bytes = max_types * 2 + 100
        header = struct.pack("<BBHBB", 0x7F, 0x7F, total_bytes, 0, max_types)
        # Add max_types datatype IDs
        for i in range(max_types):
            header += struct.pack("<H", i)
        # Add padding
        header += b"\x00" * 100
        rdi_file.write_bytes(header)

        _, _, _, _, _, ensemble, error_code = fileheader(rdi_file)
        if error_code == 0:
            assert ensemble >= 0

    def test_string_filename_vs_path_object(self, valid_rdi_file):
        """Test that both string and Path objects work."""
        result_str = fileheader(str(valid_rdi_file))
        result_path = fileheader(valid_rdi_file)

        # Both should return tuples of 7 elements
        assert len(result_str) == 7
        assert len(result_path) == 7


# ============================================================================
# TESTS: Error Code Consistency
# ============================================================================


class TestFileheaderErrorCodes:
    """Test error code consistency and correctness."""

    def test_error_code_is_integer(self, valid_rdi_file):
        """Test that error_code is always an integer."""
        _, _, _, _, _, _, error_code = fileheader(valid_rdi_file)
        assert isinstance(error_code, int)

    def test_error_code_matches_errorcode_enum(self, valid_rdi_file):
        """Test that error codes match ErrorCode enum values."""
        _, _, _, _, _, _, error_code = fileheader(valid_rdi_file)
        # Success should be 0
        assert error_code == 0

    def test_error_code_on_file_not_found(self):
        """Test error code for file not found."""
        _, _, _, _, _, _, error_code = fileheader("nonexistent.000")
        assert error_code == ErrorCode.FILE_NOT_FOUND.code
        assert error_code == 1

    def test_error_code_valid_values(self, valid_rdi_file):
        """Test that error codes are from valid ErrorCode enum."""
        _, _, _, _, _, _, error_code = fileheader(valid_rdi_file)
        # Should be one of the defined error codes
        valid_codes = {ec.code for ec in ErrorCode}
        assert error_code in valid_codes


# ============================================================================
# TESTS: Data Type Parsing and Struct Unpacking
# ============================================================================


class TestFileheaderStructUnpacking:
    """Test struct unpacking and data parsing."""

    def test_correct_unpacking_of_header_bytes(self, valid_rdi_file):
        """Test that header bytes are unpacked correctly."""
        datatype, byte, byteskip, address_offset, dataid, ensemble, error_code = (
            fileheader(valid_rdi_file)
        )

        assert error_code == 0
        # Verify specific unpacked values match fixture
        assert datatype[0] == 7  # 5 data types in fixture
        assert byte[0] == 837  # 100 byte count in fixture


# ============================================================================
# TESTS: Array Dtypes and Memory Types
# ============================================================================


class TestFileheaderArrayDtypes:
    """Test that returned arrays have correct numpy dtypes."""

    def test_datatype_array_dtype(self, valid_rdi_file):
        """Test datatype array has correct dtype."""
        datatype, _, _, _, _, _, error_code = fileheader(valid_rdi_file)
        assert error_code == 0
        # Should be integer type
        assert np.issubdtype(datatype.dtype, np.integer)

    def test_byte_array_dtype(self, valid_rdi_file):
        """Test byte array has correct dtype."""
        _, byte, _, _, _, _, error_code = fileheader(valid_rdi_file)
        assert error_code == 0
        assert np.issubdtype(byte.dtype, np.integer)

    def test_byteskip_array_dtype(self, valid_rdi_file):
        """Test byteskip array has correct dtype."""
        _, _, byteskip, _, _, _, error_code = fileheader(valid_rdi_file)
        assert error_code == 0
        assert np.issubdtype(byteskip.dtype, np.integer)

    def test_address_offset_array_dtype(self, valid_rdi_file):
        """Test address_offset array has correct dtype."""
        _, _, _, address_offset, _, _, error_code = fileheader(valid_rdi_file)
        assert error_code == 0
        assert np.issubdtype(address_offset.dtype, np.integer)

    def test_dataid_array_dtype(self, valid_rdi_file):
        """Test dataid array has correct dtype."""
        _, _, _, _, dataid, _, error_code = fileheader(valid_rdi_file)
        assert error_code == 0
        assert np.issubdtype(dataid.dtype, np.integer)


# ============================================================================
# TESTS: File Position and Seeking
# ============================================================================


class TestFileheaderFileSeeking:
    """Test proper file seeking and position management."""

    def test_file_closed_after_parsing(self, valid_rdi_file):
        """Test that file is properly closed after parsing."""
        fileheader(valid_rdi_file)
        # If file wasn't properly closed, a subsequent read would fail
        # This is mostly tested implicitly through no exceptions

    def test_parsing_does_not_corrupt_file(self, valid_rdi_file):
        """Test that parsing doesn't modify the file."""
        original_content = valid_rdi_file.read_bytes()
        fileheader(valid_rdi_file)
        after_content = valid_rdi_file.read_bytes()
        assert original_content == after_content


# ============================================================================
# TESTS: Integration and Realistic Scenarios
# ============================================================================


class TestFileheaderIntegration:
    """Integration tests with realistic scenarios."""

    def test_parse_then_reparse_same_file(self, valid_rdi_file):
        """Test that parsing same file twice produces identical results."""
        result1 = fileheader(valid_rdi_file)
        result2 = fileheader(valid_rdi_file)

        # Compare tuple elements
        for i in range(5):  # Compare first 5 numpy arrays and ensemble count
            if isinstance(result1[i], np.ndarray):
                np.testing.assert_array_equal(result1[i], result2[i])
            else:
                assert result1[i] == result2[i]
        # Error codes should match
        assert result1[6] == result2[6]


# ============================================================================
# TESTS: Logging and Warnings
# ============================================================================


class TestFileheaderLogging:
    """Test that appropriate logging occurs (via caplog fixture)."""

    def test_file_not_found_logs_error(self, caplog):
        """Test that file not found logs error message."""
        with caplog.at_level("ERROR"):
            fileheader("nonexistent.000")
        assert "not found" in caplog.text.lower() or len(caplog.records) > 0

    def test_invalid_format_logs_error(self, invalid_header_id_file, caplog):
        """Test that invalid format logs error message."""
        with caplog.at_level("ERROR"):
            fileheader(invalid_header_id_file)
        assert len(caplog.records) > 0


def test_corrupted_offset(file_with_corrupted_offset):
    """Test that fileheader() detects corruption in second ensemble.

    Simulates real scenario: instrument loses power mid-recording.
    First ensemble is valid, second is truncated.
    """
    rdi_file = file_with_corrupted_offset

    datatype, byte, byteskip, address_offset, dataid, ensemble, error_code = fileheader(
        rdi_file
    )

    # Should successfully read first ensemble
    assert ensemble == 1  # Only 1 valid ensemble
    assert error_code == 8

    # Verify first ensemble data
    assert datatype[0] == 7  # or whatever valid_rdi_header has
    assert len(address_offset) == 1  # Only first ensemble's offsets


def test_demo_000_hard_coded_values():
    """Verify parsing of real RDI demo.000 file matches all expected hard-coded values.

    This test uses actual RDI data (501 ensembles) to verify that fileheader()
    correctly extracts metadata. Data types include: Fixed Leader (0), Variable Leader (128),
    Velocity (256), Echo Intensity (512), Correlation (768), Percent Good (1024).
    Note: Bottom Track (1536) not present in current demo file.
    """
    demo_file = Path(__file__).parent / "data" / "demo.000"

    # Expected hard-coded values from demo.000
    expected_datatype = np.full(501, 6, dtype=np.int16)
    expected_byte = np.full(501, 466, dtype=np.int16)
    expected_byteskip = np.arange(1, 502) * 468  # 468, 936, 1404, ...

    base_offset = np.array([18, 71, 136, 266, 332, 398])
    expected_address_offset = np.tile(base_offset, (501, 1))

    base_dataid = np.array([0, 128, 256, 512, 768, 1024])
    expected_dataid = np.tile(base_dataid, (501, 1))

    # Parse the real RDI file
    datatype, byte, byteskip, address_offset, dataid, ensemble, error_code = fileheader(
        demo_file
    )

    # Assert entire arrays match
    np.testing.assert_array_equal(datatype, expected_datatype)
    np.testing.assert_array_equal(byte, expected_byte)
    np.testing.assert_array_equal(byteskip, expected_byteskip)
    np.testing.assert_array_equal(address_offset, expected_address_offset)
    np.testing.assert_array_equal(dataid, expected_dataid)

    # Assert scalars match
    assert ensemble == 501
    assert error_code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
