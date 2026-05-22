"""
Comprehensive pytest test suite for pd0parser.fixedleader function.

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

import io
import struct
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


from pyadps.io.pd0_parser import (
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


@pytest.fixture
def rdi_file_with_invalid_fl_id_after_read(tmp_path, valid_rdi_ensemble_with_fl):
    """Create file where Fixed Leader ID is invalid AFTER bdata is read.

    This tests line 656-660: The FL ID is read from bdata and found to be
    invalid (not 0 or 1).

    The trick: We corrupt the FL ID bytes in the file, but we'll pass a
    manually constructed idarray to fixedleader() that claims the FL section
    exists (contains 0). When fixedleader reads the actual bytes, it finds 999.

    This simulates a scenario where the file metadata is inconsistent with
    the actual data content (e.g., partial file corruption).
    """
    data = bytearray(valid_rdi_ensemble_with_fl)

    # Find Fixed Leader section position from the offset array
    fl_offset_in_header = 6  # First offset points to Fixed Leader
    fl_position = struct.unpack(
        "<H", data[fl_offset_in_header : fl_offset_in_header + 2]
    )[0]

    # Corrupt the FL ID to 999 (not 0 or 1)
    data[fl_position : fl_position + 2] = struct.pack("<H", 999)

    # Recalculate checksum so the file is "valid" structurally
    byte_count = struct.unpack("<H", data[2:4])[0]
    checksum = sum(data[:byte_count]) & 0xFFFF
    data[byte_count : byte_count + 2] = struct.pack("<H", checksum)

    rdi_file = tmp_path / "test_fl_invalid_id_after_read.000"
    rdi_file.write_bytes(bytes(data))
    return rdi_file


@pytest.fixture
def rdi_file_with_invalid_serial_exceeds_int64(tmp_path):
    """Create file with serial number that exceeds INT64_MAX.

    This tests lines 684-690: When the unpacked serial number (big-endian Q)
    is larger than 2^63-1 (INT64_MAX), set is_serial_missing = True.

    The serial number is stored as big-endian unsigned 64-bit integer.
    We set it to 0xFFFFFFFFFFFFFFFF (18446744073709551615) which exceeds INT64_MAX.
    """
    from .fixtures.ensemble_builder import FixedLeaderData, build_ensemble

    # Create ensemble with a valid structure first
    ensemble_bytes = build_ensemble()
    data = bytearray(ensemble_bytes)

    # Find Fixed Leader section position
    fl_offset_in_header = 6
    fl_position = struct.unpack(
        "<H", data[fl_offset_in_header : fl_offset_in_header + 2]
    )[0]

    # Serial number is at bytes 42-49 within Fixed Leader (big-endian Q)
    serial_position = fl_position + 42
    # Set to max uint64 value which exceeds INT64_MAX
    data[serial_position : serial_position + 8] = struct.pack(">Q", 0xFFFFFFFFFFFFFFFF)

    # Recalculate checksum
    byte_count = struct.unpack("<H", data[2:4])[0]
    checksum = sum(data[:byte_count]) & 0xFFFF
    data[byte_count : byte_count + 2] = struct.pack("<H", checksum)

    rdi_file = tmp_path / "test_fl_invalid_serial_int64.000"
    rdi_file.write_bytes(bytes(data))
    return rdi_file


@pytest.fixture
def multi_ensemble_file_with_invalid_serial(tmp_path):
    """Create multi-ensemble file where serial number exceeds INT64_MAX.

    Tests that is_serial_missing flag works across multiple ensembles
    and that the replacement logic (lines 731-740) is triggered.
    """
    from .fixtures.ensemble_builder import build_ensemble

    ensemble_bytes = build_ensemble()
    data = bytearray(ensemble_bytes)

    # Corrupt serial number in the ensemble
    fl_offset_in_header = 6
    fl_position = struct.unpack(
        "<H", data[fl_offset_in_header : fl_offset_in_header + 2]
    )[0]
    serial_position = fl_position + 42
    data[serial_position : serial_position + 8] = struct.pack(">Q", 0xFFFFFFFFFFFFFFFF)

    # Recalculate checksum
    byte_count = struct.unpack("<H", data[2:4])[0]
    checksum = sum(data[:byte_count]) & 0xFFFF
    data[byte_count : byte_count + 2] = struct.pack("<H", checksum)

    corrupted_ensemble = bytes(data)

    # Write 2 ensembles with corrupted serial numbers
    rdi_file = tmp_path / "test_fl_multi_invalid_serial.000"
    rdi_file.write_bytes(corrupted_ensemble * 2)
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
# TESTS: Fixed Leader ID Validation After Read (Line 656-660 Coverage)
# ============================================================================


class TestFixedleaderIDAfterRead:
    """Test fixedleader handling when FL ID is invalid after reading bdata.

    These tests cover lines 656-660 in pd0_parser.py:
        if fid[0][i] not in (0, 1):
            error = ErrorCode.ID_NOT_FOUND
            ensemble = i
            logger.warning(f"{error.message} Ensembles truncated at {i}")
            break

    This is different from the idarray lookup failure (line 645-648) because
    here the FL section is found in idarray but the actual FL ID bytes
    are corrupted when read from the file.
    """

    def test_invalid_fl_id_after_read_returns_id_not_found(
        self, rdi_file_with_invalid_fl_id_after_read, valid_rdi_ensemble_with_fl
    ):
        """Test that invalid FL ID (999) after reading bdata returns ID_NOT_FOUND.

        Scenario: We pass a fake idarray that claims FL section exists (has 0),
        but the actual FL ID bytes in the file are corrupted to 999.
        Expected: Return ID_NOT_FOUND, ensemble count = 0.

        This simulates file corruption where metadata is inconsistent with data.
        """
        # Get valid fileheader data from uncorrupted ensemble to use as template
        from .fixtures.ensemble_builder import build_ensemble

        valid_ensemble = build_ensemble()

        # Parse the valid ensemble to get proper structure
        dt, byte, byteskip, offset, idarray, ens_count, err = fileheader(
            str(rdi_file_with_invalid_fl_id_after_read)
        )

        # The fileheader will have read 999 as the dataid, not 0.
        # We need to manually create an idarray that has 0 so fixedleader
        # will try to read the FL section.

        # Create fake idarray with 0 in first position (FL ID)
        fake_idarray = idarray.copy()
        fake_idarray[0, 0] = 0  # Claim first datatype is Fixed Leader (ID 0)

        # Call fixedleader with the fake idarray
        data, ensemble_count, error_code = fixedleader(
            str(rdi_file_with_invalid_fl_id_after_read),
            byteskip=byteskip,
            offset=offset,
            idarray=fake_idarray,
            ensemble=1,
        )

        # Should detect invalid ID (999 != 0 and != 1) and return error
        assert error_code == ErrorCode.ID_NOT_FOUND.code
        # Ensemble should be truncated at 0 (first ensemble failed)
        assert ensemble_count == 0


# ============================================================================
# TESTS: Serial Number Overflow Handling (Lines 684-698, 731-740 Coverage)
# ============================================================================


class TestFixedleaderSerialNumberOverflow:
    """Test fixedleader handling of invalid/overflow serial numbers.

    These tests cover the reachable code paths for serial number handling:
    - Lines 691-698: ValueError/OverflowError during serial assignment
    - Lines 731-740: Replacing serial fields with MISSING_VALUE_FLAG (0)

    NOTE: Lines 684-690 are DEAD CODE because:
    - The serial number is unpacked as unsigned 64-bit and assigned to int64 array
    - Any value > INT64_MAX causes OverflowError during assignment (line 683)
    - The comparison `fid[30][i] > INT64_MAX` (line 684) is never reached
    - The OverflowError is caught by the inner except block (line 691)
    """

    def test_serial_overflow_triggers_except_block(
        self, rdi_file_with_invalid_serial_exceeds_int64
    ):
        """Test that serial number overflow is caught by except block (line 691-698).

        Scenario: Serial number bytes represent 0xFFFFFFFFFFFFFFFF (max uint64).
        This causes OverflowError during assignment to int64 array.
        The except block at line 691 catches it and sets is_serial_missing = True.
        Expected: Successfully parse, but serial fields (30-35) set to 0.
        """
        data, ensemble_count, error_code = fixedleader(
            str(rdi_file_with_invalid_serial_exceeds_int64)
        )

        # Should successfully parse the ensemble
        assert error_code == 0
        assert ensemble_count == 1

        # Serial number fields (30-35) should be replaced with 0 (MISSING_VALUE_FLAG)
        assert data[30, 0] == 0  # CPU board serial number
        assert data[31, 0] == 0  # System bandwidth
        assert data[32, 0] == 0  # System power
        assert data[33, 0] == 0  # Spare 2
        assert data[34, 0] == 0  # Instrument serial number
        assert data[35, 0] == 0  # Beam angle

    def test_multi_ensemble_serial_overflow_replaces_all(
        self, multi_ensemble_file_with_invalid_serial
    ):
        """Test that serial overflow handling works for multiple ensembles.

        When is_serial_missing becomes True on first ensemble,
        subsequent ensembles won't log the warning again (line 692 check).
        All ensemble serial fields should be replaced with MISSING_VALUE_FLAG.
        """
        data, ensemble_count, error_code = fixedleader(
            str(multi_ensemble_file_with_invalid_serial)
        )

        # Should successfully parse both ensembles
        assert error_code == 0
        assert ensemble_count == 2

        # All serial fields in all ensembles should be 0
        for ens in range(ensemble_count):
            assert data[30, ens] == 0
            assert data[31, ens] == 0
            assert data[32, ens] == 0
            assert data[33, ens] == 0
            assert data[34, ens] == 0
            assert data[35, ens] == 0

    def test_serial_unpack_value_error_sets_missing_flag(
        self, tmp_path, valid_rdi_ensemble_with_fl
    ):
        """Test that ValueError during serial unpack sets is_serial_missing.

        Uses mocking to force unpack(">Q", ...) to raise ValueError.
        This covers lines 691-698.
        """
        rdi_file = tmp_path / "test_serial_value_error.000"
        rdi_file.write_bytes(valid_rdi_ensemble_with_fl)

        original_unpack = struct.unpack
        call_count = [0]

        def mock_unpack(fmt, buffer):
            call_count[0] += 1
            # The ">Q" format is used for big-endian serial number
            if fmt == ">Q":
                raise ValueError("Mocked ValueError for serial number")
            return original_unpack(fmt, buffer)

        with mock.patch("pyadps.io.pd0_parser.unpack", side_effect=mock_unpack):
            data, ensemble_count, error_code = fixedleader(str(rdi_file))

        # Should succeed but with serial fields replaced
        assert error_code == 0
        assert ensemble_count == 1

        # Serial number fields should be replaced with 0
        assert data[30, 0] == 0
        assert data[31, 0] == 0
        assert data[32, 0] == 0
        assert data[33, 0] == 0
        assert data[34, 0] == 0
        assert data[35, 0] == 0

    def test_serial_unpack_overflow_error_sets_missing_flag(
        self, tmp_path, valid_rdi_ensemble_with_fl
    ):
        """Test that OverflowError during serial unpack sets is_serial_missing.

        Uses mocking to force unpack(">Q", ...) to raise OverflowError.
        This covers lines 691-698.
        """
        rdi_file = tmp_path / "test_serial_overflow_error.000"
        rdi_file.write_bytes(valid_rdi_ensemble_with_fl)

        original_unpack = struct.unpack

        def mock_unpack(fmt, buffer):
            if fmt == ">Q":
                raise OverflowError("Mocked OverflowError for serial number")
            return original_unpack(fmt, buffer)

        with mock.patch("pyadps.io.pd0_parser.unpack", side_effect=mock_unpack):
            data, ensemble_count, error_code = fixedleader(str(rdi_file))

        # Should succeed but with serial fields replaced
        assert error_code == 0
        assert ensemble_count == 1
        assert data[30, 0] == 0


# ============================================================================
# TESTS: Struct Parsing Errors (Lines 706-714 Coverage)
# ============================================================================


class TestFixedleaderStructParsingErrors:
    """Test fixedleader handling of struct parsing errors.

    These tests cover lines 706-714 in pd0_parser.py:
        except (ValueError, StructError, OverflowError) as e:
            logger.error(...)
            error = ErrorCode.FILE_CORRUPTED
            ensemble = i
    """

    def test_struct_error_during_fl_parsing(self, tmp_path, valid_rdi_ensemble_with_fl):
        """Test that StructError during FL parsing returns FILE_CORRUPTED.

        Uses mocking to force struct.unpack to raise StructError during
        Fixed Leader field parsing (not serial number).
        """
        rdi_file = tmp_path / "test_fl_struct_error.000"
        rdi_file.write_bytes(valid_rdi_ensemble_with_fl)

        original_unpack = struct.unpack
        call_count = [0]

        def mock_unpack(fmt, buffer):
            call_count[0] += 1
            # Fail on a non-serial format during FL parsing
            # "<HBB" is used for FL ID, CPU version, revision
            if fmt == "<HBB" and call_count[0] > 1:  # Skip fileheader calls
                raise struct.error("Mocked StructError during FL parsing")
            return original_unpack(fmt, buffer)

        with mock.patch("pyadps.io.pd0_parser.unpack", side_effect=mock_unpack):
            data, ensemble_count, error_code = fixedleader(str(rdi_file))

        # Should return FILE_CORRUPTED
        assert error_code == ErrorCode.FILE_CORRUPTED.code
        assert ensemble_count == 0

    def test_value_error_during_fl_parsing(self, tmp_path, valid_rdi_ensemble_with_fl):
        """Test that ValueError during FL parsing returns FILE_CORRUPTED."""
        rdi_file = tmp_path / "test_fl_value_error.000"
        rdi_file.write_bytes(valid_rdi_ensemble_with_fl)

        original_unpack = struct.unpack
        call_count = [0]

        def mock_unpack(fmt, buffer):
            call_count[0] += 1
            # Fail on system config unpack
            if fmt == "<HB" and call_count[0] > 3:
                raise ValueError("Mocked ValueError during FL parsing")
            return original_unpack(fmt, buffer)

        with mock.patch("pyadps.io.pd0_parser.unpack", side_effect=mock_unpack):
            data, ensemble_count, error_code = fixedleader(str(rdi_file))

        assert error_code == ErrorCode.FILE_CORRUPTED.code

    def test_overflow_error_during_fl_parsing(
        self, tmp_path, valid_rdi_ensemble_with_fl
    ):
        """Test that OverflowError during FL parsing returns FILE_CORRUPTED."""
        rdi_file = tmp_path / "test_fl_overflow_error.000"
        rdi_file.write_bytes(valid_rdi_ensemble_with_fl)

        original_unpack = struct.unpack
        call_count = [0]

        def mock_unpack(fmt, buffer):
            call_count[0] += 1
            # Fail on pings per ensemble unpack
            if fmt == "<HHH" and call_count[0] > 3:
                raise OverflowError("Mocked OverflowError during FL parsing")
            return original_unpack(fmt, buffer)

        with mock.patch("pyadps.io.pd0_parser.unpack", side_effect=mock_unpack):
            data, ensemble_count, error_code = fixedleader(str(rdi_file))

        assert error_code == ErrorCode.FILE_CORRUPTED.code


# ============================================================================
# TESTS: File Seeking Errors (Lines 715-721 Coverage)
# ============================================================================


class TestFixedleaderFileSeekingErrors:
    """Test fixedleader handling of file seeking errors.

    These tests cover lines 715-721 in pd0_parser.py:
        except (OSError, io.UnsupportedOperation) as e:
            logger.error(f"File seeking error at ensemble {i + 1}: {e}...")
            error = ErrorCode.FILE_CORRUPTED
            ensemble = i

    The OSError can be raised by:
    - Line 652: bfile.seek(fbyteskip, 1)
    - Line 704: bfile.seek(byteskip[i], 0)
    """

    def test_oserror_during_seek_in_try_block(
        self, tmp_path, valid_rdi_ensemble_with_fl
    ):
        """Test that OSError during bfile.seek returns FILE_CORRUPTED.

        Mock the file's seek method to raise OSError after fileheader completes
        but during fixedleader's FL parsing loop.
        """
        rdi_file = tmp_path / "test_fl_oserror_seek.000"
        rdi_file.write_bytes(valid_rdi_ensemble_with_fl)

        # We need to let fileheader() succeed, then fail during fixedleader's loop
        # The safest way is to patch the seek method of the file object
        # after it's been opened by fixedleader (second open call)

        open_call_count = [0]
        original_open = open

        def counting_open(*args, **kwargs):
            open_call_count[0] += 1
            f = original_open(*args, **kwargs)

            # Second open is for fixedleader (first is fileheader)
            if open_call_count[0] == 2:
                original_seek = f.seek
                seek_call_count = [0]

                def failing_seek(offset, whence=0):
                    seek_call_count[0] += 1
                    # Let initial seek succeed (bfile.seek(0, 0) at start)
                    # Fail on subsequent seeks in the FL parsing loop
                    if seek_call_count[0] > 1:
                        raise OSError("Mocked OSError during seek in FL loop")
                    return original_seek(offset, whence)

                f.seek = failing_seek

            return f

        with mock.patch("builtins.open", side_effect=counting_open):
            data, ensemble_count, error_code = fixedleader(str(rdi_file))

        # Should return FILE_CORRUPTED due to seek error
        assert error_code == ErrorCode.FILE_CORRUPTED.code
        assert ensemble_count == 0

    def test_unsupported_operation_during_seek(
        self, tmp_path, valid_rdi_ensemble_with_fl
    ):
        """Test that io.UnsupportedOperation during seek returns FILE_CORRUPTED."""
        rdi_file = tmp_path / "test_fl_unsupported_seek.000"
        rdi_file.write_bytes(valid_rdi_ensemble_with_fl)

        open_call_count = [0]
        original_open = open

        def counting_open(*args, **kwargs):
            open_call_count[0] += 1
            f = original_open(*args, **kwargs)

            if open_call_count[0] == 2:  # fixedleader's open
                original_seek = f.seek
                seek_call_count = [0]

                def failing_seek(offset, whence=0):
                    seek_call_count[0] += 1
                    if seek_call_count[0] > 1:
                        raise io.UnsupportedOperation("Mocked UnsupportedOperation")
                    return original_seek(offset, whence)

                f.seek = failing_seek

            return f

        with mock.patch("builtins.open", side_effect=counting_open):
            data, ensemble_count, error_code = fixedleader(str(rdi_file))

        assert error_code == ErrorCode.FILE_CORRUPTED.code
        assert ensemble_count == 0

    def test_oserror_on_final_seek_truncates_ensemble(self, tmp_path):
        """Test OSError on line 704's seek (byteskip) truncates to valid ensembles.

        Create multi-ensemble file, let first ensemble parse fully,
        then fail on second ensemble's seek.

        Note: fixedleader calls fileheader internally, which also opens the file.
        So we need to track opens carefully:
        - fileheader opens file once
        - fixedleader opens file once (after fileheader returns)
        """
        from .fixtures.ensemble_builder import build_ensemble

        # Create file with 2 ensembles
        ensemble_bytes = build_ensemble()
        rdi_file = tmp_path / "test_fl_oserror_final_seek.000"
        rdi_file.write_bytes(ensemble_bytes * 2)

        # First, call fileheader to get the parameters, then call fixedleader
        # with those parameters. This way we only have one open in fixedleader.
        dt, byte, byteskip, offset, idarray, ens_count, fh_err = fileheader(
            str(rdi_file)
        )
        assert fh_err == 0
        assert ens_count == 2

        original_open = open

        def mock_open_with_failing_seek(*args, **kwargs):
            f = original_open(*args, **kwargs)
            original_seek = f.seek
            seek_call_count = [0]

            def failing_seek(offset_val, whence=0):
                seek_call_count[0] += 1
                # In fixedleader's loop:
                # - seek(0, 0) at start = call 1
                # - For each ensemble: seek(fbyteskip, 1) then seek(byteskip[i], 0)
                # So for 2 ensembles: 1 + 2*2 = 5 seeks minimum
                # Fail on 4th seek (during second ensemble)
                if seek_call_count[0] >= 4:
                    raise OSError("Mocked OSError on seek during second ensemble")
                return original_seek(offset_val, whence)

            f.seek = failing_seek
            return f

        with mock.patch("builtins.open", side_effect=mock_open_with_failing_seek):
            data, ensemble_count, error_code = fixedleader(
                str(rdi_file),
                byteskip=byteskip,
                offset=offset,
                idarray=idarray,
                ensemble=ens_count,
            )

        # Should return FILE_CORRUPTED
        assert error_code == ErrorCode.FILE_CORRUPTED.code
        # First ensemble should have been parsed before error on second
        assert ensemble_count == 1


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
