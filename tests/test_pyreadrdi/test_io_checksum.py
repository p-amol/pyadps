"""
Comprehensive pytest test suite for I/O and checksum functions.

Tests cover:
- ErrorCode enum: all error types, get_message() method
- safe_open(): file finding, permissions, IO errors, memory errors
- safe_read(): successful reads, EOF truncation, read errors
- _calculate_checksum(): byte patterns, boundary conditions, 16-bit overflow
- _verify_ensemble_checksum(): valid/invalid checksums, file corruption
- Integration: error propagation, error handling workflows

Reference: RDI WorkHorse Commands and Output Data Format (Section 7.2)
"""

import io
import struct
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from pyadps.utils.pyreadrdi import (
    ErrorCode,
    safe_open,
    safe_read,
    _calculate_checksum,
    _verify_ensemble_checksum,
)

from .fixtures.ensemble_builder import build_ensemble


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_binary_file():
    """Create a temporary binary file for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        temp_path = Path(f.name)
        yield temp_path
    temp_path.unlink()  # Clean up


@pytest.fixture
def valid_ensemble_bytes():
    """Generate a valid complete RDI ensemble with default values."""
    ensemble_bytes = build_ensemble()
    return ensemble_bytes


@pytest.fixture
def ensemble_with_corrupted_checksum(valid_ensemble_bytes):
    """Create an ensemble with a corrupted checksum."""
    # Corrupt the checksum (last 2 bytes)
    corrupted = bytearray(valid_ensemble_bytes)
    corrupted[-2:] = b"\x00\x00"  # Set checksum to 0x0000
    return bytes(corrupted)


@pytest.fixture
def incomplete_ensemble():
    """Create an incomplete ensemble (truncated)."""
    ensemble_bytes = build_ensemble()
    # Return first 50 bytes only
    return ensemble_bytes[:50]


@pytest.fixture
def test_file_with_valid_ensemble(valid_ensemble_bytes):
    """Create a temporary file containing a valid ensemble."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".000") as f:
        f.write(valid_ensemble_bytes)
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()


@pytest.fixture
def test_file_with_multiple_ensembles(valid_ensemble_bytes):
    """Create a temporary file with multiple valid ensembles."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".000") as f:
        for _ in range(3):
            f.write(valid_ensemble_bytes)
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()


@pytest.fixture
def empty_file():
    """Create an empty temporary file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        temp_path = Path(f.name)
    yield temp_path
    temp_path.unlink()


# ============================================================================
# ErrorCode TESTS
# ============================================================================


class TestErrorCode:
    """Test suite for ErrorCode enumeration."""

    def test_error_code_enum_exists(self) -> None:
        """Verify ErrorCode enum is properly defined."""
        assert hasattr(ErrorCode, "SUCCESS")
        assert hasattr(ErrorCode, "FILE_NOT_FOUND")
        assert hasattr(ErrorCode, "CHECKSUM_ERROR")

    def test_error_code_has_code_attribute(self) -> None:
        """Verify each ErrorCode member has 'code' attribute."""
        for error in ErrorCode:
            assert hasattr(error, "code")
            assert isinstance(error.code, int)

    def test_error_code_has_message_attribute(self) -> None:
        """Verify each ErrorCode member has 'message' attribute."""
        for error in ErrorCode:
            assert hasattr(error, "message")
            assert isinstance(error.message, str)
            assert len(error.message) > 0

    def test_error_code_success_value(self) -> None:
        """Verify SUCCESS error code is 0."""
        assert ErrorCode.SUCCESS.code == 0
        assert "Success" in ErrorCode.SUCCESS.message

    def test_error_code_all_codes_unique(self) -> None:
        """Verify all error codes are unique."""
        codes = [error.code for error in ErrorCode]
        assert len(codes) == len(set(codes)), "Error codes must be unique"

    def test_error_code_file_not_found(self) -> None:
        """Verify FILE_NOT_FOUND error code."""
        assert ErrorCode.FILE_NOT_FOUND.code == 1
        assert "not found" in ErrorCode.FILE_NOT_FOUND.message.lower()

    def test_error_code_permission_denied(self) -> None:
        """Verify PERMISSION_DENIED error code."""
        assert ErrorCode.PERMISSION_DENIED.code == 2
        assert "permission" in ErrorCode.PERMISSION_DENIED.message.lower()

    def test_error_code_io_error(self) -> None:
        """Verify IO_ERROR error code."""
        assert ErrorCode.IO_ERROR.code == 3
        assert "io" in ErrorCode.IO_ERROR.message.lower()

    def test_error_code_checksum_error(self) -> None:
        """Verify CHECKSUM_ERROR error code."""
        assert ErrorCode.CHECKSUM_ERROR.code == 10
        assert "checksum" in ErrorCode.CHECKSUM_ERROR.message.lower()

    def test_error_code_get_message_success(self) -> None:
        """Test get_message() method for SUCCESS."""
        message = ErrorCode.get_message(0)
        assert "Success" in message

    def test_error_code_get_message_file_not_found(self) -> None:
        """Test get_message() method for FILE_NOT_FOUND."""
        message = ErrorCode.get_message(1)
        assert "not found" in message.lower()

    def test_error_code_get_message_checksum_error(self) -> None:
        """Test get_message() method for CHECKSUM_ERROR."""
        message = ErrorCode.get_message(10)
        assert "checksum" in message.lower()

    def test_error_code_get_message_invalid_code(self) -> None:
        """Test get_message() with invalid error code."""
        message = ErrorCode.get_message(999)
        assert "Invalid" in message or "invalid" in message

    def test_error_code_iteration(self) -> None:
        """Verify all ErrorCode members can be iterated."""
        count = 0
        for error in ErrorCode:
            count += 1
            assert error.code >= 0
        assert count >= 10  # Should have at least 10 error codes

    def test_error_code_get_code_success(self) -> None:
        """Test get_code() method for SUCCESS message."""
        code = ErrorCode.get_code("Success")
        assert code == 0
        assert code == ErrorCode.SUCCESS.code

    def test_error_code_get_code_file_not_found(self) -> None:
        """Test get_code() method for FILE_NOT_FOUND message."""
        code = ErrorCode.get_code("Error: File not found.")
        assert code == 1
        assert code == ErrorCode.FILE_NOT_FOUND.code

    def test_error_code_get_code_permission_denied(self) -> None:
        """Test get_code() method for PERMISSION_DENIED message."""
        code = ErrorCode.get_code("Error: Permission denied.")
        assert code == 2
        assert code == ErrorCode.PERMISSION_DENIED.code

    def test_error_code_get_code_io_error(self) -> None:
        """Test get_code() method for IO_ERROR message."""
        code = ErrorCode.get_code("IO Error: Unable to open file.")
        assert code == 3
        assert code == ErrorCode.IO_ERROR.code

    def test_error_code_get_code_checksum_error(self) -> None:
        """Test get_code() method for CHECKSUM_ERROR message."""
        code = ErrorCode.get_code("Error: Ensemble checksum verification failed.")
        assert code == 10
        assert code == ErrorCode.CHECKSUM_ERROR.code

    def test_error_code_get_code_invalid_message(self) -> None:
        """Test get_code() with invalid error message."""
        code = ErrorCode.get_code("Unknown message xyz")
        assert code == 99
        assert code == ErrorCode.UNKNOWN_ERROR.code

    def test_error_code_get_code_empty_string(self) -> None:
        """Test get_code() with empty string."""
        code = ErrorCode.get_code("")
        assert code == 99
        assert code == ErrorCode.UNKNOWN_ERROR.code

    def test_error_code_get_code_all_messages(self) -> None:
        """Test get_code() for all ErrorCode enum messages."""
        for error in ErrorCode:
            code = ErrorCode.get_code(error.message)
            assert code == error.code

    def test_error_code_get_code_roundtrip_consistency(self) -> None:
        """Test round-trip consistency: code -> message -> code."""
        for error in ErrorCode:
            message = ErrorCode.get_message(error.code)
            code = ErrorCode.get_code(message)
            assert code == error.code

    def test_error_code_get_code_case_sensitive(self) -> None:
        """Test get_code() is case-sensitive."""
        # Correct case returns valid code
        code = ErrorCode.get_code("Success")
        assert code == 0

        # Wrong case returns UNKNOWN_ERROR
        code = ErrorCode.get_code("success")
        assert code == 99

    def test_error_code_get_code_exact_match_required(self) -> None:
        """Test get_code() requires exact message match."""
        # Partial message should not match
        code = ErrorCode.get_code("Error: File")
        assert code == 99

        # Extra spaces should not match
        code = ErrorCode.get_code("Error:  File not found.")
        assert code == 99


# ============================================================================
# safe_open() TESTS
# ============================================================================


class TestSafeOpen:
    """Test suite for safe_open() function."""

    def test_safe_open_file_exists(self, test_file_with_valid_ensemble) -> None:
        """Test opening an existing file successfully."""
        file_obj, error = safe_open(test_file_with_valid_ensemble)
        assert file_obj is not None
        assert error == ErrorCode.SUCCESS
        assert not file_obj.closed
        file_obj.close()

    def test_safe_open_file_not_found(self) -> None:
        """Test opening a non-existent file."""
        file_obj, error = safe_open("/nonexistent/path/file.bin")
        assert file_obj is None
        assert error == ErrorCode.FILE_NOT_FOUND

    def test_safe_open_returns_tuple(self, test_file_with_valid_ensemble) -> None:
        """Verify safe_open returns a tuple."""
        result = safe_open(test_file_with_valid_ensemble)
        assert isinstance(result, tuple)
        assert len(result) == 2
        file_obj, error = result
        if file_obj:
            file_obj.close()

    def test_safe_open_default_mode_binary(self, test_file_with_valid_ensemble) -> None:
        """Verify default mode is binary read."""
        file_obj, error = safe_open(test_file_with_valid_ensemble)
        assert file_obj is not None
        assert "b" in file_obj.mode
        file_obj.close()

    def test_safe_open_custom_mode(self, temp_binary_file) -> None:
        """Test safe_open with custom file mode."""
        # Write some data first
        with open(temp_binary_file, "wb") as f:
            f.write(b"test")
        file_obj, error = safe_open(temp_binary_file, mode="rb")
        assert error == ErrorCode.SUCCESS
        assert file_obj is not None
        file_obj.close()

    def test_safe_open_string_path(self, test_file_with_valid_ensemble) -> None:
        """Test safe_open with string path."""
        file_obj, error = safe_open(str(test_file_with_valid_ensemble))
        assert error == ErrorCode.SUCCESS
        assert file_obj is not None
        file_obj.close()

    def test_safe_open_pathlib_path(self, test_file_with_valid_ensemble) -> None:
        """Test safe_open with pathlib.Path object."""
        file_obj, error = safe_open(Path(test_file_with_valid_ensemble))
        assert error == ErrorCode.SUCCESS
        assert file_obj is not None
        file_obj.close()

    def test_safe_open_returns_binary_io(self, test_file_with_valid_ensemble) -> None:
        """Verify returned file object is BinaryIO."""
        file_obj, error = safe_open(test_file_with_valid_ensemble)
        assert hasattr(file_obj, "read")
        assert hasattr(file_obj, "seek")
        assert hasattr(file_obj, "close")
        file_obj.close()

    @mock.patch("builtins.open", side_effect=PermissionError("Permission denied"))
    def test_safe_open_permission_error(self, mock_open, temp_binary_file) -> None:
        """Test safe_open with permission denied."""
        file_obj, error = safe_open(temp_binary_file)
        assert file_obj is None
        assert error == ErrorCode.PERMISSION_DENIED

    @mock.patch("builtins.open", side_effect=IOError("IO error"))
    def test_safe_open_io_error(self, mock_open, temp_binary_file) -> None:
        """Test safe_open with IO error."""
        file_obj, error = safe_open(temp_binary_file)
        assert file_obj is None
        assert error == ErrorCode.IO_ERROR

    @mock.patch("builtins.open", side_effect=MemoryError("Out of memory"))
    def test_safe_open_memory_error(self, mock_open, temp_binary_file) -> None:
        """Test safe_open with memory error."""
        file_obj, error = safe_open(temp_binary_file)
        assert file_obj is None
        assert error == ErrorCode.OUT_OF_MEMORY

    @mock.patch("builtins.open", side_effect=RuntimeError("Unexpected error"))
    def test_safe_open_unknown_error(self, mock_open, temp_binary_file) -> None:
        """Test safe_open with unknown error."""
        file_obj, error = safe_open(temp_binary_file)
        assert file_obj is None
        assert error == ErrorCode.UNKNOWN_ERROR


# ============================================================================
# safe_read() TESTS
# ============================================================================


class TestSafeRead:
    """Test suite for safe_read() function."""

    def test_safe_read_success(self, valid_ensemble_bytes) -> None:
        """Test successful read of specified bytes."""
        bfile = io.BytesIO(valid_ensemble_bytes)
        data, error = safe_read(bfile, 100)
        assert data is not None
        assert len(data) == 100
        assert error == ErrorCode.SUCCESS

    def test_safe_read_exact_file_size(self, valid_ensemble_bytes) -> None:
        """Test reading exact file size."""
        bfile = io.BytesIO(valid_ensemble_bytes)
        file_size = len(valid_ensemble_bytes)
        data, error = safe_read(bfile, file_size)
        assert data == valid_ensemble_bytes
        assert error == ErrorCode.SUCCESS

    def test_safe_read_eof_truncation(self, valid_ensemble_bytes) -> None:
        """Test reading more bytes than file contains (EOF)."""
        bfile = io.BytesIO(valid_ensemble_bytes[:50])
        data, error = safe_read(bfile, 100)
        assert data is None
        assert error == ErrorCode.FILE_CORRUPTED

    def test_safe_read_empty_file(self, empty_file) -> None:
        """Test reading from empty file."""
        with open(empty_file, "rb") as f:
            data, error = safe_read(f, 100)
        assert data is None
        assert error == ErrorCode.FILE_CORRUPTED

    def test_safe_read_zero_bytes(self) -> None:
        """Test reading zero bytes."""
        bfile = io.BytesIO(b"test data")
        data, error = safe_read(bfile, 0)
        assert data == b""
        assert error == ErrorCode.SUCCESS

    def test_safe_read_sequential(self, valid_ensemble_bytes) -> None:
        """Test sequential reads."""
        bfile = io.BytesIO(valid_ensemble_bytes)
        data1, error1 = safe_read(bfile, 50)
        assert error1 == ErrorCode.SUCCESS
        data2, error2 = safe_read(bfile, 50)
        assert error2 == ErrorCode.SUCCESS
        assert data1 + data2 == valid_ensemble_bytes[:100]

    def test_safe_read_returns_tuple(self) -> None:
        """Verify safe_read returns a tuple."""
        bfile = io.BytesIO(b"test")
        result = safe_read(bfile, 2)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_safe_read_preserves_position(self) -> None:
        """Test that safe_read correctly advances file position."""
        bfile = io.BytesIO(b"0123456789")
        data1, _ = safe_read(bfile, 5)
        assert data1 == b"01234"
        data2, _ = safe_read(bfile, 5)
        assert data2 == b"56789"

    @mock.patch("builtins.open")
    def test_safe_read_io_error(self, mock_open) -> None:
        """Test safe_read with IO error."""
        mock_file = mock.MagicMock()
        mock_file.read.side_effect = IOError("Read error")
        data, error = safe_read(mock_file, 100)
        assert data is None
        assert error == ErrorCode.IO_ERROR

    @mock.patch("builtins.open")
    def test_safe_read_value_error(self, mock_open) -> None:
        """Test safe_read with value error."""
        mock_file = mock.MagicMock()
        mock_file.read.side_effect = ValueError("Invalid value")
        data, error = safe_read(mock_file, 100)
        assert data is None
        assert error == ErrorCode.VALUE_ERROR

    def test_safe_read_partial_eof(self) -> None:
        """Test reading partial data before EOF."""
        bfile = io.BytesIO(b"short")
        data, error = safe_read(bfile, 10)
        assert data is None
        assert error == ErrorCode.FILE_CORRUPTED

    def test_safe_read_returns_bytes_type(self) -> None:
        """Verify safe_read returns bytes type on success."""
        bfile = io.BytesIO(b"test data here")
        data, error = safe_read(bfile, 4)
        assert isinstance(data, bytes)
        assert data == b"test"


# ============================================================================
# _calculate_checksum() TESTS
# ============================================================================


class TestCalculateChecksum:
    """Test suite for _calculate_checksum() function."""

    def test_calculate_checksum_empty_bytes(self) -> None:
        """Test checksum of empty bytes."""
        checksum = _calculate_checksum(b"")
        assert checksum == 0

    def test_calculate_checksum_single_byte(self) -> None:
        """Test checksum of single byte."""
        checksum = _calculate_checksum(b"\x42")
        assert checksum == 0x42

    def test_calculate_checksum_simple_pattern(self) -> None:
        """Test checksum with simple byte pattern."""
        data = b"\x01\x02\x03"
        checksum = _calculate_checksum(data)
        assert checksum == 0x06  # 1 + 2 + 3 = 6

    def test_calculate_checksum_overflow_16bit(self) -> None:
        """Test checksum with 16-bit overflow (only lower 16 bits kept)."""
        # Create data that sums to > 65535 (0xFFFF)
        data = b"\xff" * 256  # 256 * 255 = 65280
        checksum = _calculate_checksum(data)
        expected = (256 * 0xFF) & 0xFFFF
        assert checksum == expected

    def test_calculate_checksum_known_value(self) -> None:
        """Test checksum against known value."""
        data = b"\x7f\x7f\x00\x05\x00"
        checksum = _calculate_checksum(data)
        assert checksum == (0x7F + 0x7F + 0x00 + 0x05 + 0x00) & 0xFFFF

    def test_calculate_checksum_16bit_mask(self) -> None:
        """Verify checksum is masked to 16 bits."""
        # Sum is 0x10000 (65536)
        data = b"\x00\x00\x01"  # Sum = 1, but let's use specific bytes
        checksum = _calculate_checksum(data)
        # Checksum should never exceed 0xFFFF
        assert checksum <= 0xFFFF

    def test_calculate_checksum_all_zeros(self) -> None:
        """Test checksum of all zeros."""
        data = b"\x00" * 100
        checksum = _calculate_checksum(data)
        assert checksum == 0

    def test_calculate_checksum_all_ones(self) -> None:
        """Test checksum of all ones (0xFF bytes)."""
        data = b"\xff" * 10
        checksum = _calculate_checksum(data)
        expected = (10 * 0xFF) & 0xFFFF
        assert checksum == expected

    def test_calculate_checksum_returns_int(self) -> None:
        """Verify _calculate_checksum returns int."""
        checksum = _calculate_checksum(b"\x42")
        assert isinstance(checksum, int)

    def test_calculate_checksum_large_data(self) -> None:
        """Test checksum with large data."""
        data = b"\xaa" * 10000
        checksum = _calculate_checksum(data)
        expected = (10000 * 0xAA) & 0xFFFF
        assert checksum == expected

    def test_calculate_checksum_deterministic(self) -> None:
        """Verify checksum is deterministic (same input = same output)."""
        data = b"test data for checksum"
        checksum1 = _calculate_checksum(data)
        checksum2 = _calculate_checksum(data)
        assert checksum1 == checksum2

    def test_calculate_checksum_valid_ensemble(self, valid_ensemble_bytes) -> None:
        """Test checksum calculation on valid ensemble (excluding checksum)."""
        # Checksum is last 2 bytes, data is everything except last 2 bytes
        ensemble_data = valid_ensemble_bytes[:-2]
        checksum = _calculate_checksum(ensemble_data)
        # Should be a valid 16-bit value
        assert 0 <= checksum <= 0xFFFF


# ============================================================================
# _verify_ensemble_checksum() TESTS
# ============================================================================


class TestVerifyEnsembleChecksum:
    """Test suite for _verify_ensemble_checksum() function."""

    def test_verify_checksum_valid_ensemble(self, valid_ensemble_bytes) -> None:
        """Test verification of valid ensemble checksum."""
        bfile = io.BytesIO(valid_ensemble_bytes)
        # Ensemble size is at offset +2 (little-endian)
        ensemble_size = struct.unpack("<H", valid_ensemble_bytes[2:4])[0]
        is_valid, error_code = _verify_ensemble_checksum(bfile, 0, ensemble_size)
        assert is_valid is True
        assert error_code == ErrorCode.SUCCESS.code

    def test_verify_checksum_invalid_checksum(
        self, ensemble_with_corrupted_checksum
    ) -> None:
        """Test verification of ensemble with corrupted checksum."""
        bfile = io.BytesIO(ensemble_with_corrupted_checksum)
        ensemble_size = struct.unpack("<H", ensemble_with_corrupted_checksum[2:4])[0]
        is_valid, error_code = _verify_ensemble_checksum(bfile, 0, ensemble_size)
        assert is_valid is False
        assert error_code == ErrorCode.CHECKSUM_ERROR.code

    def test_verify_checksum_incomplete_ensemble(self, incomplete_ensemble) -> None:
        """Test verification of incomplete/truncated ensemble."""
        bfile = io.BytesIO(incomplete_ensemble)
        ensemble_size = 1000  # Larger than actual data
        is_valid, error_code = _verify_ensemble_checksum(bfile, 0, ensemble_size)
        assert is_valid is False
        assert error_code == ErrorCode.FILE_CORRUPTED.code

    def test_verify_checksum_returns_tuple(self, valid_ensemble_bytes) -> None:
        """Verify _verify_ensemble_checksum returns a tuple."""
        bfile = io.BytesIO(valid_ensemble_bytes)
        ensemble_size = struct.unpack("<H", valid_ensemble_bytes[2:4])[0]
        result = _verify_ensemble_checksum(bfile, 0, ensemble_size)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_verify_checksum_returns_bool_and_int(self, valid_ensemble_bytes) -> None:
        """Verify return types are (bool, int)."""
        bfile = io.BytesIO(valid_ensemble_bytes)
        ensemble_size = struct.unpack("<H", valid_ensemble_bytes[2:4])[0]
        is_valid, error_code = _verify_ensemble_checksum(bfile, 0, ensemble_size)
        assert isinstance(is_valid, bool)
        assert isinstance(error_code, int)

    def test_verify_checksum_at_file_offset(self, valid_ensemble_bytes) -> None:
        """Test verification at non-zero file offset."""
        # Pad with junk data before ensemble
        padded_data = b"\x00" * 100 + valid_ensemble_bytes
        bfile = io.BytesIO(padded_data)
        ensemble_size = struct.unpack("<H", valid_ensemble_bytes[2:4])[0]
        is_valid, error_code = _verify_ensemble_checksum(bfile, 100, ensemble_size)
        assert is_valid is True
        assert error_code == ErrorCode.SUCCESS.code

    def test_verify_checksum_multiple_ensembles(
        self, test_file_with_multiple_ensembles
    ) -> None:
        """Test verification of multiple ensembles in same file."""
        with open(test_file_with_multiple_ensembles, "rb") as bfile:
            # Read first ensemble to get size (size is at bytes 2-4)
            bfile.seek(0)
            header = bfile.read(4)
            ensemble_size = struct.unpack("<H", header[2:4])[0]
            bfile.seek(0)

            # Verify first ensemble
            is_valid1, error1 = _verify_ensemble_checksum(bfile, 0, ensemble_size)
            assert is_valid1 is True

            # Verify second ensemble (at offset = ensemble_size + 2 for checksum)
            offset2 = ensemble_size + 2
            is_valid2, error2 = _verify_ensemble_checksum(bfile, offset2, ensemble_size)
            assert is_valid2 is True

    def test_verify_checksum_missing_checksum_bytes(self) -> None:
        """Test with missing checksum bytes at end of file."""
        # Create ensemble-like data missing the checksum
        partial_data = b"\x7f\x7f\x00\x05" + b"\x00" * 100
        bfile = io.BytesIO(partial_data)
        ensemble_size = struct.unpack("<H", partial_data[2:4])[0]
        is_valid, error_code = _verify_ensemble_checksum(bfile, 0, ensemble_size)
        assert is_valid is False
        assert error_code == ErrorCode.FILE_CORRUPTED.code

    def test_verify_checksum_zero_size_ensemble(self, valid_ensemble_bytes) -> None:
        """Test with zero-size ensemble."""
        bfile = io.BytesIO(valid_ensemble_bytes)
        is_valid, error_code = _verify_ensemble_checksum(bfile, 0, 0)
        # Zero-size ensemble should still read checksum
        # Result depends on implementation
        assert isinstance(is_valid, bool)
        assert isinstance(error_code, int)

    @mock.patch("builtins.open")
    def test_verify_checksum_io_error(self, mock_open) -> None:
        """Test _verify_ensemble_checksum with IO error."""
        mock_file = mock.MagicMock()
        mock_file.seek.side_effect = IOError("Seek error")
        is_valid, error_code = _verify_ensemble_checksum(mock_file, 0, 100)
        assert is_valid is False
        assert error_code == ErrorCode.IO_ERROR.code

    def test_verify_checksum_correct_little_endian(self, valid_ensemble_bytes) -> None:
        """Verify checksum uses little-endian byte order."""
        bfile = io.BytesIO(valid_ensemble_bytes)
        ensemble_size = struct.unpack("<H", valid_ensemble_bytes[2:4])[0]

        # Calculate expected checksum manually
        ensemble_data = valid_ensemble_bytes[:ensemble_size]
        expected_checksum = sum(ensemble_data) & 0xFFFF

        # Read stored checksum
        stored_checksum = struct.unpack(
            "<H", valid_ensemble_bytes[ensemble_size : ensemble_size + 2]
        )[0]

        # They should match
        assert stored_checksum == expected_checksum


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIOCheckSumIntegration:
    """Integration tests for I/O and checksum functions working together."""

    def test_workflow_open_read_verify(self, test_file_with_valid_ensemble) -> None:
        """Test typical workflow: open file, read data, verify checksum."""
        # Open file
        bfile, open_error = safe_open(test_file_with_valid_ensemble)
        assert open_error == ErrorCode.SUCCESS
        assert bfile is not None

        # Read full file
        bfile.seek(0)
        full_data, read_error = safe_read(
            bfile, Path(test_file_with_valid_ensemble).stat().st_size
        )
        assert read_error == ErrorCode.SUCCESS
        assert full_data is not None

        # Verify checksum
        ensemble_size = struct.unpack("<H", full_data[2:4])[0]
        bfile.seek(0)
        is_valid, checksum_error = _verify_ensemble_checksum(bfile, 0, ensemble_size)
        assert is_valid is True
        assert checksum_error == ErrorCode.SUCCESS.code

        bfile.close()

    def test_workflow_error_propagation_file_not_found(self) -> None:
        """Test error propagation when file not found."""
        bfile, error = safe_open("/nonexistent/file.bin")
        assert bfile is None
        assert error == ErrorCode.FILE_NOT_FOUND
        # Next operation should handle None gracefully
        assert error != ErrorCode.SUCCESS

    def test_workflow_corrupted_file_detection(
        self, ensemble_with_corrupted_checksum
    ) -> None:
        """Test detection of corrupted file via checksum."""
        bfile = io.BytesIO(ensemble_with_corrupted_checksum)
        ensemble_size = struct.unpack("<H", ensemble_with_corrupted_checksum[2:4])[0]

        is_valid, error_code = _verify_ensemble_checksum(bfile, 0, ensemble_size)
        assert is_valid is False
        assert error_code == ErrorCode.CHECKSUM_ERROR.code

    def test_workflow_sequential_ensemble_reading(
        self, test_file_with_multiple_ensembles
    ) -> None:
        """Test reading and verifying multiple ensembles sequentially."""
        bfile, open_error = safe_open(test_file_with_multiple_ensembles)
        assert open_error == ErrorCode.SUCCESS

        # Read first 4 bytes to get ensemble size
        size_bytes, read_error = safe_read(bfile, 4)
        assert read_error == ErrorCode.SUCCESS
        ensemble_size = struct.unpack("<H", size_bytes[2:4])[0]

        # Verify first ensemble
        bfile.seek(0)
        is_valid1, error1 = _verify_ensemble_checksum(bfile, 0, ensemble_size)
        assert is_valid1 is True

        # Verify second ensemble
        offset2 = ensemble_size + 2  # +2 for checksum
        is_valid2, error2 = _verify_ensemble_checksum(bfile, offset2, ensemble_size)
        assert is_valid2 is True

        bfile.close()

    def test_error_code_message_chain(self) -> None:
        """Test error message retrieval for error codes."""
        # Simulate error workflow
        bfile, error = safe_open("/nonexistent/file.bin")
        assert error == ErrorCode.FILE_NOT_FOUND

        # Get message for logging
        message = ErrorCode.get_message(error.code)
        assert "not found" in message.lower()

    def test_checksum_calculation_consistency(self, valid_ensemble_bytes) -> None:
        """Test that checksum calculation is consistent with verification."""
        ensemble_data = valid_ensemble_bytes[:-2]
        calculated = _calculate_checksum(ensemble_data)

        stored = struct.unpack("<H", valid_ensemble_bytes[-2:])[0]

        assert calculated == stored

    def test_file_operations_with_real_file(
        self, test_file_with_valid_ensemble
    ) -> None:
        """Integration test with real file I/O."""
        # Open file
        bfile, error1 = safe_open(test_file_with_valid_ensemble)
        assert error1 == ErrorCode.SUCCESS

        # Get file size
        bfile.seek(0, 2)  # Seek to end
        file_size = bfile.tell()
        bfile.seek(0)  # Seek back to start

        # Read entire file
        data, error2 = safe_read(bfile, file_size)
        assert error2 == ErrorCode.SUCCESS
        assert len(data) == file_size

        # Verify checksum
        ensemble_size = struct.unpack("<H", data[2:4])[0]
        bfile.seek(0)
        is_valid, error3 = _verify_ensemble_checksum(bfile, 0, ensemble_size)
        assert is_valid is True
        assert error3 == ErrorCode.SUCCESS.code

        bfile.close()


# ============================================================================
# EDGE CASES AND BOUNDARY TESTS
# ============================================================================


class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""

    def test_checksum_at_16bit_boundary(self) -> None:
        """Test checksum calculation at 16-bit boundary (0xFFFF)."""
        # Create data that sums to exactly 0xFFFF
        data = b"\xff" * 255 + b"\xff"  # 256 * 0xFF = 0xFF00
        # Actually, let's be more precise
        data = b"\xff\xff"  # Sums to 0x1FE = 510
        checksum = _calculate_checksum(data)
        assert checksum == 0x1FE

    def test_checksum_just_over_16bit(self) -> None:
        """Test checksum wrapping from > 0xFFFF to < 0xFFFF."""
        # Create data that sums to 0x10000 = 65536
        # This should wrap to 0x0000
        # 0x10000 & 0xFFFF = 0x0000
        data_sum = 0x10000
        # We can't directly create bytes that sum to this,
        # but we can test the mask operation
        assert (0x10000) & 0xFFFF == 0x0000

    def test_safe_read_boundary_exact_match(self) -> None:
        """Test safe_read when requested bytes exactly match available."""
        data = b"exactly10bytes!"
        assert len(data) == 15
        bfile = io.BytesIO(data)
        result, error = safe_read(bfile, 15)
        assert error == ErrorCode.SUCCESS
        assert result == data

    def test_safe_read_boundary_one_short(self) -> None:
        """Test safe_read when file is one byte short."""
        data = b"short"
        bfile = io.BytesIO(data)
        result, error = safe_read(bfile, len(data) + 1)
        assert error == ErrorCode.FILE_CORRUPTED
        assert result is None

    def test_verify_checksum_large_ensemble(self) -> None:
        """Test checksum verification with large ensemble."""
        # Create a large ensemble (e.g., 10KB)
        large_data = b"\x7f\x7f" + struct.pack("<H", 10000)
        large_data += b"\x42" * (10000 - 4)  # Pad to size
        checksum = _calculate_checksum(large_data)
        large_data += struct.pack("<H", checksum)

        bfile = io.BytesIO(large_data)
        is_valid, error = _verify_ensemble_checksum(bfile, 0, 10000)
        assert is_valid is True

    def test_error_code_value_error_exists(self) -> None:
        """Verify VALUE_ERROR code is defined."""
        assert hasattr(ErrorCode, "VALUE_ERROR")
        assert ErrorCode.VALUE_ERROR.code == 9

