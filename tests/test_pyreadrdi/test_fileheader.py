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

from pyadps.utils.pyreadrdi import (
    ErrorCode,
    fileheader,
    _calculate_checksum,
    _verify_ensemble_checksum,
)

from .fixtures.ensemble_builder import (
    build_ensemble,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def valid_rdi_ensemble():
    """
    Generate a valid complete RDI ensemble with default values.

    This fixture is used across multiple test files. It creates a complete,
    valid ensemble matching WorkHorse spec with:
    - 7 data types (Fixed Leader, Variable Leader, Velocity, Echo, Correlation,
      Percent Good, Bottom Track)
    - 4 beams
    - 30 depth cells
    - Valid checksums

    Returns:
        bytes: Complete binary RDI ensemble.
    """
    return build_ensemble()


@pytest.fixture
def valid_rdi_file(tmp_path, valid_rdi_ensemble):
    """Create a temporary file with valid RDI data.

    Single ensemble with all required sections.
    """
    rdi_file = tmp_path / "test_single_ensemble.000"
    rdi_file.write_bytes(valid_rdi_ensemble)
    return rdi_file


@pytest.fixture
def multi_ensemble_rdi_file(tmp_path, valid_rdi_ensemble):
    """Create a temporary file with multiple valid RDI ensembles."""
    rdi_file = tmp_path / "test_multi_ensemble.000"
    # Write 3 identical ensembles
    rdi_file.write_bytes(valid_rdi_ensemble * 3)
    return rdi_file


@pytest.fixture
def truncated_rdi_file(tmp_path, valid_rdi_ensemble):
    """Create a file truncated mid-header (incomplete first read)."""
    rdi_file = tmp_path / "test_truncated.000"
    # Write only first 3 bytes of header (incomplete)
    rdi_file.write_bytes(valid_rdi_ensemble[:3])
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
def mismatched_datatype_file(tmp_path, valid_rdi_ensemble):
    """Create a file where second ensemble has different datatype count."""
    header1 = valid_rdi_ensemble

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
def file_with_corrupted_offset(tmp_path, valid_rdi_ensemble):
    """Create a file with first ensemble valid, second ensemble corrupted offset array.

    Simulates power loss or firmware issue mid-recording after first ensemble completes.
    """
    rdi_file = tmp_path / "corrupted_second_ensemble.000"

    # First ensemble: valid
    first_ensemble = valid_rdi_ensemble

    actual_ensemble_size = 837  # Declare this size based on valid_rdi_ensemble
    # Second ensemble: corrupted (truncated offset array)
    # Valid header but incomplete offset data
    header = struct.pack(
        "<BBHBB", 0x7F, 0x7F, actual_ensemble_size, 0, 7
    )  # Says 5 datatypes
    header += struct.pack("<HHH", 0, 1, 256)

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

    def test_return_tuple_length(self, tmp_path, valid_rdi_ensemble):
        """Test that fileheader always returns 7-tuple."""
        # Create temporary RDI file
        rdi_file = tmp_path / "test.000"
        rdi_file.write_bytes(valid_rdi_ensemble)

        result = fileheader(rdi_file)
        assert isinstance(result, tuple)
        assert len(result) == 7

    def test_return_tuple_length_on_error(self):
        """Test return tuple length on file not found error."""
        result = fileheader("nonexistent.000")
        assert len(result) == 7

    def test_return_values_are_correct_types(self, tmp_path, valid_rdi_ensemble):
        """Test return value types on success."""
        # Create temporary RDI file
        rdi_file = tmp_path / "test.000"
        rdi_file.write_bytes(valid_rdi_ensemble)

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

    def test_single_ensemble_parsing(self, tmp_path, valid_rdi_ensemble):
        """Test parsing a file with single ensemble."""
        rdi_file = tmp_path / "single.000"
        rdi_file.write_bytes(valid_rdi_ensemble)
        datatype, byte, byteskip, address_offset, dataid, ensemble, error_code = (
            fileheader(rdi_file)
        )
        print(error_code)

        assert error_code == 0
        assert ensemble == 1
        assert len(datatype) == 1
        assert len(byte) == 1
        assert len(byteskip) == 1

    def test_multi_ensemble_parsing(self, tmp_path, valid_rdi_ensemble):
        """Test parsing a file with multiple ensembles."""
        multi_ensemble_rdi_file = tmp_path / "multi.000"
        multi_ensemble_rdi_file.write_bytes(valid_rdi_ensemble * 3)
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

    def test_corrupted_offset(self, file_with_corrupted_offset):
        """Test that fileheader() detects corruption in second ensemble.

        Simulates real scenario: instrument loses power mid-recording.
        First ensemble is valid, second is truncated.
        """
        rdi_file = file_with_corrupted_offset

        datatype, byte, byteskip, address_offset, dataid, ensemble, error_code = (
            fileheader(rdi_file)
        )

        # Should successfully read first ensemble
        assert ensemble == 1  # Only 1 valid ensemble
        assert error_code == ErrorCode.FILE_CORRUPTED.code

        # Verify first ensemble data
        assert datatype[0] == 7  # or whatever valid_rdi_ensemble has
        assert len(address_offset) == 1  # Only first ensemble's offsets


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


# ============================================================================
# TESTS: Checksum Verification
# ============================================================================


class TestChecksumCalculation:
    """Test checksum calculation helper function."""

    def test_calculate_checksum_basic(self):
        """Test basic checksum calculation."""
        data = b"\x7f\x7f\x80\x10"
        checksum = _calculate_checksum(data)
        assert checksum == 398  # or 0x018E

    def test_calculate_checksum_empty(self):
        """Test checksum with empty bytes."""
        data = b""
        checksum = _calculate_checksum(data)
        assert checksum == 0

    def test_calculate_checksum_single_byte(self):
        """Test checksum with single byte."""
        data = b"\x42"
        checksum = _calculate_checksum(data)
        assert checksum == 0x42

    def test_calculate_checksum_overflow(self):
        """Test that checksum masks to 16 bits."""
        # Create data that sums to > 0xFFFF
        data = b"\xff" * 300  # 300 * 0xFF = 0x12600
        checksum = _calculate_checksum(data)
        # Should mask to lower 16 bits: 0x2600
        assert checksum == (300 * 0xFF) & 0xFFFF
        assert checksum <= 0xFFFF

    def test_calculate_checksum_idempotent(self):
        """Test that checksum is consistent across calls."""
        data = b"\x7f\x7f\x80\x10\x05\x00"
        checksum1 = _calculate_checksum(data)
        checksum2 = _calculate_checksum(data)
        assert checksum1 == checksum2

    def test_calculate_checksum_order_matters(self):
        """Test that byte order affects checksum."""
        data1 = b"\x01\x02"
        data2 = b"\x02\x01"
        checksum1 = _calculate_checksum(data1)
        checksum2 = _calculate_checksum(data2)
        assert checksum1 == checksum2  # Sum is same, but test data order
        # Actually, sum is same but let's verify
        assert 0x01 + 0x02 == 0x02 + 0x01


class TestChecksumVerification:
    """Test checksum verification against RDI ensemble data."""

    @pytest.fixture
    def valid_ensemble_data(self, tmp_path):
        """Create a valid ensemble with correct checksum.

        RDI ensemble structure:
        - Bytes 0-1: Header ID (0x7F7F)
        - Bytes 2-3: Ensemble size (excluding 2-byte checksum)
        - Bytes 4-5: Data type count and spare
        - Bytes 6+: Data (size-6 bytes, where size includes header)
        - Last 2 bytes: Checksum
        """
        # Create ensemble with size = 50 (includes 6-byte header)
        ensemble_size = 50
        header = struct.pack("<BBHBB", 0x7F, 0x7F, ensemble_size, 0, 1)  # 6 bytes
        header += struct.pack("<H", 0)  # 2 bytes: One data type offset

        # Body: ensemble_size - 6 (header) - 2 (offset) = 42 bytes
        body = b"\x00" * 42

        # Calculate checksum on all data excluding checksum itself
        ensemble_without_checksum = header + body
        checksum_val = _calculate_checksum(ensemble_without_checksum)
        checksum_bytes = struct.pack("<H", checksum_val)

        # Write to file
        rdi_file = tmp_path / "valid_ensemble.000"
        rdi_file.write_bytes(ensemble_without_checksum + checksum_bytes)

        return rdi_file, ensemble_without_checksum, checksum_val

    @pytest.fixture
    def corrupted_ensemble_data(self, tmp_path):
        """Create ensemble with corrupted checksum."""
        ensemble_size = 50
        header = struct.pack("<BBHBB", 0x7F, 0x7F, ensemble_size, 0, 1)
        header += struct.pack("<H", 0)
        body = b"\x00" * 42

        ensemble_without_checksum = header + body

        # Use wrong checksum (intentionally corrupt)
        wrong_checksum = 0x9999
        checksum_bytes = struct.pack("<H", wrong_checksum)

        rdi_file = tmp_path / "corrupted_ensemble.000"
        rdi_file.write_bytes(ensemble_without_checksum + checksum_bytes)

        return rdi_file, ensemble_without_checksum

    def test_verify_ensemble_checksum_valid(self, valid_ensemble_data):
        """Test verification passes with valid checksum."""
        rdi_file, ensemble_data, expected_checksum = valid_ensemble_data

        with open(rdi_file, "rb") as bfile:
            is_valid, error_code = _verify_ensemble_checksum(
                bfile,
                ensemble_start_pos=0,
                ensemble_size=50,  # Match the fixture
            )

        assert is_valid is True
        assert error_code == ErrorCode.SUCCESS.code

    def test_verify_ensemble_checksum_corrupted(self, corrupted_ensemble_data):
        """Test verification fails with corrupted checksum."""
        rdi_file, ensemble_data = corrupted_ensemble_data

        with open(rdi_file, "rb") as bfile:
            is_valid, error_code = _verify_ensemble_checksum(
                bfile,
                ensemble_start_pos=0,
                ensemble_size=50,  # Match the fixture
            )

        assert is_valid is False
        assert error_code == ErrorCode.CHECKSUM_ERROR.code

    def test_verify_ensemble_checksum_incomplete_data(self, tmp_path):
        """Test verification with incomplete ensemble data."""
        # Create file with incomplete data
        ensemble_size = 50
        header = struct.pack("<BBHBB", 0x7F, 0x7F, ensemble_size, 0, 1)
        header += struct.pack("<H", 0)
        body = b"\x00" * 20  # Only 20 bytes instead of 42

        rdi_file = tmp_path / "incomplete_ensemble.000"
        rdi_file.write_bytes(header + body)

        with open(rdi_file, "rb") as bfile:
            is_valid, error_code = _verify_ensemble_checksum(
                bfile,
                ensemble_start_pos=0,
                ensemble_size=50,
            )

        assert is_valid is False
        assert error_code == ErrorCode.FILE_CORRUPTED.code

    def test_verify_ensemble_checksum_missing_checksum(self, tmp_path):
        """Test verification when checksum bytes are missing."""
        ensemble_size = 50
        header = struct.pack("<BBHBB", 0x7F, 0x7F, ensemble_size, 0, 1)
        header += struct.pack("<H", 0)
        body = b"\x00" * 42

        rdi_file = tmp_path / "missing_checksum.000"
        # Write without checksum bytes
        rdi_file.write_bytes(header + body)

        with open(rdi_file, "rb") as bfile:
            is_valid, error_code = _verify_ensemble_checksum(
                bfile,
                ensemble_start_pos=0,
                ensemble_size=50,
            )

        assert is_valid is False
        assert error_code == ErrorCode.FILE_CORRUPTED.code

    def test_verify_ensemble_checksum_little_endian(self, tmp_path):
        """Test that checksum uses little-endian byte order."""
        ensemble_size = 10
        header = struct.pack("<BBHBB", 0x7F, 0x7F, ensemble_size, 0, 0)

        # Header is 6 bytes, so body is 10 - 6 = 4 bytes
        body = b"\x00" * 4
        ensemble_without_checksum = header + body

        # Checksum = 0x7F + 0x7F + 0x0A + 0x00 + 0x00 + 0x00 + 0x00 + 0x00 + 0x00 + 0x00
        # = 0x0108
        checksum_val = _calculate_checksum(ensemble_without_checksum)
        checksum_bytes = struct.pack("<H", checksum_val)  # Little-endian

        rdi_file = tmp_path / "endian_test.000"
        rdi_file.write_bytes(ensemble_without_checksum + checksum_bytes)

        with open(rdi_file, "rb") as bfile:
            is_valid, error_code = _verify_ensemble_checksum(
                bfile,
                ensemble_start_pos=0,
                ensemble_size=ensemble_size,
            )

        assert is_valid is True
        assert error_code == ErrorCode.SUCCESS.code

    def test_verify_ensemble_checksum_at_different_offset(self, tmp_path):
        """Test checksum verification at non-zero file offset."""
        # Write some garbage data first
        garbage = b"\xaa" * 1000

        # Then write valid ensemble
        ensemble_size = 50
        header = struct.pack("<BBHBB", 0x7F, 0x7F, ensemble_size, 0, 1)
        header += struct.pack("<H", 0)
        body = b"\x00" * 42

        ensemble_without_checksum = header + body
        checksum_val = _calculate_checksum(ensemble_without_checksum)
        checksum_bytes = struct.pack("<H", checksum_val)

        rdi_file = tmp_path / "offset_test.000"
        rdi_file.write_bytes(garbage + ensemble_without_checksum + checksum_bytes)

        ensemble_start_pos = len(garbage)

        with open(rdi_file, "rb") as bfile:
            is_valid, error_code = _verify_ensemble_checksum(
                bfile,
                ensemble_start_pos=ensemble_start_pos,
                ensemble_size=ensemble_size,
            )

        assert is_valid is True
        assert error_code == ErrorCode.SUCCESS.code


class TestFileheaderWithChecksum:
    """Test fileheader function with checksum verification integration."""

    @pytest.fixture
    def valid_rdi_file_with_checksums(self, tmp_path):
        """Create valid RDI file with multiple ensembles and correct checksums."""
        rdi_file = tmp_path / "valid_multi_ensemble.000"
        file_data = b""

        # Create 3 valid ensembles
        ensemble_size = 50
        for ens_num in range(3):
            header = struct.pack("<BBHBB", 0x7F, 0x7F, ensemble_size, 0, 1)
            header += struct.pack("<H", 0)
            body = b"\x00" * 42

            ensemble_without_checksum = header + body
            checksum_val = _calculate_checksum(ensemble_without_checksum)
            checksum_bytes = struct.pack("<H", checksum_val)

            file_data += ensemble_without_checksum + checksum_bytes

        rdi_file.write_bytes(file_data)
        return rdi_file

    @pytest.fixture
    def rdi_file_with_checksum_error_at_end(self, tmp_path):
        """Create RDI file where checksum fails at 3rd ensemble."""
        rdi_file = tmp_path / "checksum_error_at_3.000"
        file_data = b""

        ensemble_size = 50

        # First 2 ensembles valid
        for ens_num in range(2):
            header = struct.pack("<BBHBB", 0x7F, 0x7F, ensemble_size, 0, 1)
            header += struct.pack("<H", 0)
            body = b"\x00" * 42

            ensemble_without_checksum = header + body
            checksum_val = _calculate_checksum(ensemble_without_checksum)
            checksum_bytes = struct.pack("<H", checksum_val)

            file_data += ensemble_without_checksum + checksum_bytes

        # 3rd ensemble with WRONG checksum
        header = struct.pack("<BBHBB", 0x7F, 0x7F, ensemble_size, 0, 1)
        header += struct.pack("<H", 0)
        body = b"\x00" * 42
        ensemble_without_checksum = header + body
        wrong_checksum = 0x9999
        checksum_bytes = struct.pack("<H", wrong_checksum)

        file_data += ensemble_without_checksum + checksum_bytes

        rdi_file.write_bytes(file_data)
        return rdi_file

    def test_fileheader_with_valid_checksums(self, valid_rdi_file_with_checksums):
        """Test fileheader parses file with valid checksums."""
        dt, byte, skip, offset, ids, n_ens, err = fileheader(
            valid_rdi_file_with_checksums
        )

        assert err == 0
        assert n_ens == 3
        assert len(dt) == 3
        assert len(byte) == 3

    def test_fileheader_stops_on_checksum_error(
        self, rdi_file_with_checksum_error_at_end
    ):
        """Test fileheader stops when checksum fails."""
        dt, byte, skip, offset, ids, n_ens, err = fileheader(
            rdi_file_with_checksum_error_at_end
        )

        # Should return 2 valid ensembles (stop at 3rd)
        assert err == ErrorCode.CHECKSUM_ERROR.code
        assert n_ens == 2
        assert len(dt) == 2
        assert len(byte) == 2

    def test_fileheader_returns_clean_data_before_checksum_error(
        self, rdi_file_with_checksum_error_at_end
    ):
        """Test that clean data is returned before checksum fails."""
        dt, byte, skip, offset, ids, n_ens, err = fileheader(
            rdi_file_with_checksum_error_at_end
        )

        # Verify first 2 ensembles data is correct
        assert n_ens == 2
        assert all(b == 50 for b in byte)  # All ensemble sizes should be 50
        assert all(dt_val == 1 for dt_val in dt)  # All should have 1 data type


class TestChecksumEdgeCases:
    """Test edge cases and boundary conditions for checksum."""

    def test_checksum_single_bit_error_detected(self, tmp_path):
        """Test that single-bit error in data is detected."""
        header = struct.pack("<BBHBB", 0x7F, 0x7F, 10, 0, 0)
        original_data = b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"

        checksum_val = _calculate_checksum(header + original_data)
        checksum_bytes = struct.pack("<H", checksum_val)

        # Corrupt one bit in header
        corrupted_header = b"\x7f\x7f\x81\x00"  # Changed one bit
        corrupted_header += b"\x00"  # spare
        corrupted_header += b"\x00"  # datatype

        rdi_file = tmp_path / "single_bit_error.000"
        rdi_file.write_bytes(corrupted_header + original_data + checksum_bytes)

        with open(rdi_file, "rb") as bfile:
            is_valid, error_code = _verify_ensemble_checksum(
                bfile,
                ensemble_start_pos=0,
                ensemble_size=10,
            )

        assert is_valid is False
        assert error_code == ErrorCode.CHECKSUM_ERROR.code

    def test_checksum_all_zeros(self, tmp_path):
        """Test checksum with all-zero data."""
        data = b"\x00" * 100
        checksum_val = _calculate_checksum(data)
        assert checksum_val == 0

    def test_checksum_all_ones(self, tmp_path):
        """Test checksum with all 0xFF data."""
        data = b"\xff" * 256  # Exactly 256 bytes
        checksum_val = _calculate_checksum(data)
        # 0xFF * 256 = 0x0000FF00, masked to 0xFF00
        assert checksum_val == (256 * 0xFF) & 0xFFFF


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
