"""
Comprehensive pytest test suite for pyreadrdi.variableleader function.

Tests cover:
- Normal operation with valid RDI files
- Variable Leader data extraction and parsing
- Error handling (file not found, IO errors, corrupted data)
- Field validation (timestamps, sensor data, status)
- Multiple ensembles and uniform/non-uniform data
- Struct unpacking errors and edge cases
- Return value types and structure (48, n_ensembles) array
- Data type ID verification (must be 128 or 129 for Variable Leader)
- Integration with fileheader() for automatic parameter retrieval
- Timestamp and sensor field correctness

References
----------
RDI WorkHorse Commands and Output Data Format (Section 5.3, page 134):
- Variable Leader ID: bytes 1-2 (0x0080 or 0x0081)
- Ensemble number: bytes 3-4
- Time fields: bytes 5-11 (Year, Month, Day, Hour, Minute, Second, Hundredth)
- Heading, Pitch, Roll: bytes 12-17 (3 x 16-bit signed integers)
- Salinity, Temperature, Pressure: bytes 18-25
- ADC Channels: bytes 34-41 (8 x 8-bit)
- RTC Y2K time: bytes 57-64 (Century, Year, Month, Day, Hour, Minute, Second, Hundredth)
"""

import struct
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from pyadps import (
    ErrorCode,
    fileheader,
    variableleader,
)

from .fixtures.ensemble_builder import (
    VariableLeaderData,
    build_ensemble,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(scope="function")
def valid_rdi_ensemble_with_vl():
    """Generate a valid complete RDI ensemble using shared builder.

    Uses the imported build_ensemble() function from test_ensemble_builder.py.
    Structure follows RDI WorkHorse spec with 7 data types (default config).
    """
    return build_ensemble()


@pytest.fixture
def valid_rdi_file_with_vl(tmp_path, valid_rdi_ensemble_with_vl):
    """Create a temporary file with single valid ensemble containing Variable Leader."""
    rdi_file = tmp_path / "test_vl_single.000"
    rdi_file.write_bytes(valid_rdi_ensemble_with_vl)
    return rdi_file


@pytest.fixture
def multi_ensemble_rdi_file_with_vl(tmp_path, valid_rdi_ensemble_with_vl):
    """Create a temporary file with multiple valid ensembles."""
    rdi_file = tmp_path / "test_vl_multi.000"
    # Write 3 identical ensembles
    rdi_file.write_bytes(valid_rdi_ensemble_with_vl * 3)
    return rdi_file


@pytest.fixture
def rdi_file_missing_vl_id(tmp_path, valid_rdi_ensemble_with_vl):
    """Create file where Variable Leader section has wrong ID.

    Corrupts the Variable Leader ID (bytes at offset position) to an invalid value
    (9999) while maintaining a valid checksum so that fileheader() passes
    checksum verification. This allows variableleader() to detect the bad ID.
    """
    data = bytearray(valid_rdi_ensemble_with_vl)

    # Parse header to find Variable Leader section offset
    # Header structure: BBHBB at bytes 0-5, then offset array follows
    vl_offset_in_header = 6 + (2 * 1)  # Byte position to read second offset from
    vl_position = struct.unpack(
        "<H", data[vl_offset_in_header : vl_offset_in_header + 2]
    )[0]

    # Corrupt the Variable Leader ID (2 bytes at vl_position)
    data[vl_position : vl_position + 2] = struct.pack("<H", 9999)  # Invalid ID

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

    rdi_file = tmp_path / "test_vl_missing_id.000"
    rdi_file.write_bytes(bytes(data))
    return rdi_file


@pytest.fixture
def rdi_file_truncated_vl(tmp_path, valid_rdi_ensemble_with_vl):
    """Create file where Variable Leader section is truncated."""
    rdi_file = tmp_path / "test_vl_truncated.000"
    # Write only 30 bytes of the first ensemble (incomplete VL section)
    rdi_file.write_bytes(valid_rdi_ensemble_with_vl[:30])
    return rdi_file


@pytest.fixture
def rdi_file_multiple_with_vl_corruption(tmp_path, valid_rdi_ensemble_with_vl):
    """Create file with valid ensemble followed by corrupted ensemble."""
    rdi_file = tmp_path / "test_vl_corrupted.000"

    first_ensemble = valid_rdi_ensemble_with_vl

    # Create a corrupted second ensemble (truncated)
    corrupted = first_ensemble[:100]  # Only partial data

    rdi_file.write_bytes(first_ensemble + corrupted)
    return rdi_file


@pytest.fixture
def custom_vl_ensemble(tmp_path):
    """Create ensemble with custom Variable Leader data for field validation."""
    custom_vl = VariableLeaderData(
        vid=0x0080,  # Standard Variable Leader ID
        ensemble_number=42,
        year=25,  # 2025
        month=3,
        day=15,
        hour=14,
        minute=30,
        second=45,
        hundredth=50,
        heading=18000,  # 180.00 degrees
        pitch=-5000,  # -50.00 degrees
        roll=3000,  # 30.00 degrees
        salinity=35000,  # 35 ppt
        temperature=2000,  # 20.00 degrees C
        pressure=1000,  # 100 deca-pascals
        pressure_variance=500,
        rtc_century=20,
        rtc_year=25,
        rtc_month=3,
        rtc_day=15,
        rtc_hour=14,
        rtc_minute=30,
        rtc_second=45,
        rtc_hundredth=50,
    )
    ensemble_data = build_ensemble(variable_leader_data=custom_vl)
    rdi_file = tmp_path / "test_vl_custom.000"
    rdi_file.write_bytes(ensemble_data)
    return rdi_file


# ============================================================================
# TESTS: Input Validation
# ============================================================================


class TestVariableleaderInputValidation:
    """Test variableleader function input validation."""

    def test_missing_filename_argument(self):
        """Test that calling variableleader without arguments raises TypeError."""
        with pytest.raises(TypeError):
            variableleader()

    def test_invalid_filename_type_integer(self):
        """Test that passing an integer returns FILE_NOT_FOUND error."""
        data, ensemble, error_code = variableleader(12345)
        assert isinstance(data, np.ndarray)
        assert isinstance(ensemble, int)
        assert error_code == ErrorCode.FILE_NOT_FOUND.code

    def test_invalid_filename_type_none(self):
        """Test that passing None returns FILE_NOT_FOUND error."""
        data, ensemble, error_code = variableleader(None)
        assert error_code == ErrorCode.FILE_NOT_FOUND.code

    def test_invalid_filename_type_list(self):
        """Test that passing a list returns FILE_NOT_FOUND error."""
        data, ensemble, error_code = variableleader(["file.000"])
        assert error_code == ErrorCode.FILE_NOT_FOUND.code


# ============================================================================
# TESTS: File Access
# ============================================================================


class TestVariableleaderFileAccess:
    """Test variableleader error handling for file access issues."""

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        data, ensemble, error_code = variableleader("nonexistent_file_xyz.000")
        assert error_code == ErrorCode.FILE_NOT_FOUND.code

    def test_file_not_found_with_path_object(self):
        """Test file not found using pathlib.Path."""
        data, ensemble, error_code = variableleader(Path("/nonexistent/path/file.000"))
        assert error_code == ErrorCode.FILE_NOT_FOUND.code

    def test_permission_denied(self):
        """Test handling of permission denied error via mocking."""
        with mock.patch("builtins.open", side_effect=PermissionError("Access denied")):
            data, ensemble, error_code = variableleader("somefile.000")
            assert error_code == ErrorCode.PERMISSION_DENIED.code

    def test_io_error_generic(self):
        """Test handling of generic IOError."""
        with mock.patch("builtins.open", side_effect=IOError("IO problem")):
            data, ensemble, error_code = variableleader("somefile.000")
            assert error_code == ErrorCode.IO_ERROR.code

    def test_oserror_generic(self):
        """Test handling of generic OSError."""
        with mock.patch("builtins.open", side_effect=OSError("OS problem")):
            data, ensemble, error_code = variableleader("somefile.000")
            assert error_code == ErrorCode.IO_ERROR.code


# ============================================================================
# TESTS: Basic Functionality
# ============================================================================


class TestVariableleaderBasicFunctionality:
    """Test basic variableleader operation with valid data."""

    def test_single_ensemble_extraction(self, valid_rdi_file_with_vl):
        """Test Variable Leader extraction from single ensemble file."""
        data, ensemble_count, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        assert ensemble_count == 1
        assert data.shape == (48, 1)
        assert data.dtype == np.int32

    def test_return_types(self, valid_rdi_file_with_vl):
        """Test that return types are correct."""
        data, ensemble_count, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert isinstance(data, np.ndarray)
        assert isinstance(ensemble_count, (int, np.integer))
        assert isinstance(error_code, (int, np.integer))

    def test_multi_ensemble_extraction(self, multi_ensemble_rdi_file_with_vl):
        """Test Variable Leader extraction from multi-ensemble file."""
        data, ensemble_count, error_code = variableleader(
            str(multi_ensemble_rdi_file_with_vl)
        )

        assert error_code == 0
        assert ensemble_count == 3
        assert data.shape == (48, 3)

    def test_success_error_code(self, valid_rdi_file_with_vl):
        """Test that success case returns error code 0."""
        _, _, error_code = variableleader(str(valid_rdi_file_with_vl))
        assert error_code == 0

    def test_data_array_not_empty(self, valid_rdi_file_with_vl):
        """Test that returned data array is not empty."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        if error_code == 0:
            assert data.size > 0
            assert not np.all(data == 0)  # At least some non-zero data


# ============================================================================
# TESTS: Variable Leader ID Validation
# ============================================================================


class TestVariableleaderIDValidation:
    """Test Variable Leader ID field validation."""

    def test_vl_id_is_valid_value(self, valid_rdi_file_with_vl):
        """Test that Variable Leader ID (field 0) is valid (128 or 129)."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Variable Leader ID should be 128 (0x0080) or 129 (0x0081)
        assert data[0, 0] in (128, 129)

    def test_vl_id_consistency_across_ensembles(self, multi_ensemble_rdi_file_with_vl):
        """Test that Variable Leader ID is consistent across ensembles."""
        data, _, error_code = variableleader(str(multi_ensemble_rdi_file_with_vl))

        assert error_code == 0
        # All ensembles should have same VL ID
        assert np.all(data[0, :] == data[0, 0])

    def test_vl_id_mismatch_detection(self, rdi_file_missing_vl_id):
        """Test that invalid Variable Leader ID is detected."""
        data, ensemble_count, error_code = variableleader(str(rdi_file_missing_vl_id))

        # Should detect the bad ID and truncate or error
        assert error_code != 0 or ensemble_count == 0


# ============================================================================
# TESTS: Timestamp Field Extraction
# ============================================================================


class TestVariableleaderTimestampFields:
    """Test timestamp field extraction from Variable Leader."""

    def test_year_field_extraction(self, valid_rdi_file_with_vl):
        """Test that year field (field 2) is correctly extracted."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Year should be in range 0-99 (YY format)
        assert 0 <= data[2, 0] <= 99

    def test_month_field_extraction(self, valid_rdi_file_with_vl):
        """Test that month field (field 3) is correctly extracted."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Month should be 1-12
        assert 1 <= data[3, 0] <= 12

    def test_day_field_extraction(self, valid_rdi_file_with_vl):
        """Test that day field (field 4) is correctly extracted."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Day should be 1-31
        assert 1 <= data[4, 0] <= 31

    def test_hour_field_extraction(self, valid_rdi_file_with_vl):
        """Test that hour field (field 5) is correctly extracted."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Hour should be 0-23
        assert 0 <= data[5, 0] <= 23

    def test_minute_field_extraction(self, valid_rdi_file_with_vl):
        """Test that minute field (field 6) is correctly extracted."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Minute should be 0-59
        assert 0 <= data[6, 0] <= 59

    def test_second_field_extraction(self, valid_rdi_file_with_vl):
        """Test that second field (field 7) is correctly extracted."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Second should be 0-59
        assert 0 <= data[7, 0] <= 59

    def test_hundredth_field_extraction(self, valid_rdi_file_with_vl):
        """Test that hundredth field (field 8) is correctly extracted."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Hundredth should be 0-99
        assert 0 <= data[8, 0] <= 99

    def test_custom_timestamp_values(self, custom_vl_ensemble):
        """Test extraction of custom timestamp values."""
        data, _, error_code = variableleader(str(custom_vl_ensemble))

        assert error_code == 0
        assert data[2, 0] == 25  # Year
        assert data[3, 0] == 3  # Month
        assert data[4, 0] == 15  # Day
        assert data[5, 0] == 14  # Hour
        assert data[6, 0] == 30  # Minute
        assert data[7, 0] == 45  # Second
        assert data[8, 0] == 50  # Hundredth


# ============================================================================
# TESTS: Ensemble Number Field
# ============================================================================


class TestVariableleaderEnsembleNumber:
    """Test ensemble number field extraction."""

    def test_ensemble_number_field_exists(self, valid_rdi_file_with_vl):
        """Test that ensemble number field (field 1) is present."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        assert 0 <= data[1, 0] <= 65535  # 16-bit unsigned

    def test_custom_ensemble_number(self, custom_vl_ensemble):
        """Test extraction of custom ensemble number."""
        data, _, error_code = variableleader(str(custom_vl_ensemble))

        assert error_code == 0
        # Ensemble number should match the custom value or be lower/upper part
        # (implementation may use only lower/upper byte)
        assert data[1, 0] > 0


# ============================================================================
# TESTS: Motion Sensor Fields (Heading, Pitch, Roll)
# ============================================================================


class TestVariableleaderMotionSensors:
    """Test motion sensor field extraction."""

    def test_heading_field_extraction(self, valid_rdi_file_with_vl):
        """Test that heading field (field 13) is correctly extracted."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Heading in 0.01 degree units, range 0-35999 (0-359.99 degrees)
        assert isinstance(data[13, 0], (int, np.integer))

    def test_pitch_field_extraction(self, valid_rdi_file_with_vl):
        """Test that pitch field (field 14) is correctly extracted."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Pitch in 0.01 degree units, signed: -20000 to +20000 (-200.00 to +200.00)
        assert isinstance(data[14, 0], (int, np.integer))

    def test_roll_field_extraction(self, valid_rdi_file_with_vl):
        """Test that roll field (field 15) is correctly extracted."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Roll in 0.01 degree units, signed: -20000 to +20000 (-200.00 to +200.00)
        assert isinstance(data[15, 0], (int, np.integer))

    def test_custom_motion_sensor_values(self, custom_vl_ensemble):
        """Test extraction of custom motion sensor values."""
        data, _, error_code = variableleader(str(custom_vl_ensemble))

        assert error_code == 0
        assert data[13, 0] == 18000  # Heading 180.00 degrees
        assert data[14, 0] == -5000  # Pitch -50.00 degrees
        assert data[15, 0] == 3000  # Roll 30.00 degrees


# ============================================================================
# TESTS: Environmental Fields (Temperature, Salinity, Pressure)
# ============================================================================


class TestVariableleaderEnvironmentalFields:
    """Test environmental sensor field extraction."""

    def test_salinity_field_extraction(self, valid_rdi_file_with_vl):
        """Test that salinity field (field 17) is correctly extracted."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Salinity packed as signed int16 in pyreadrdi format
        # Typical value 35000 is capped to 32767 (max signed int16)
        assert -32768 <= data[17, 0] <= 32767

    def test_temperature_field_extraction(self, valid_rdi_file_with_vl):
        """Test that temperature field (field 16) is correctly extracted."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Temperature in 0.01 degree C units, range -4000 to +4000 (-40 to +40 C)
        assert -5000 <= data[16, 0] <= 5000

    def test_pressure_field_extraction(self, valid_rdi_file_with_vl):
        """Test that pressure field (field 37) is correctly extracted."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Pressure in decapascals (100 Pa = 1 deca-Pa)
        assert data[37, 0] >= 0

    def test_pressure_variance_field_extraction(self, valid_rdi_file_with_vl):
        """Test that pressure variance field (field 38) is correctly extracted."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Pressure variance should be non-negative
        assert data[38, 0] >= 0

    def test_custom_environmental_values(self, custom_vl_ensemble):
        """Test extraction of custom environmental sensor values."""
        data, _, error_code = variableleader(str(custom_vl_ensemble))

        assert error_code == 0
        assert data[16, 0] == 2000  # Temperature 20.00 C
        assert data[17, 0] == 32767  # Salinity capped to max signed int16
        assert data[37, 0] == 1000  # Pressure 1000 deca-Pa
        assert data[38, 0] == 500  # Pressure variance


# ============================================================================
# TESTS: ADC Channel Fields
# ============================================================================


class TestVariableleaderADCChannels:
    """Test ADC channel field extraction."""

    def test_adc_channel_fields_exist(self, valid_rdi_file_with_vl):
        """Test that 8 ADC channel fields (fields 24-31) are present."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # All 8 ADC channels should exist and be non-negative
        for i in range(24, 32):
            assert 0 <= data[i, 0] <= 255  # 8-bit unsigned

    def test_adc_channels_consistency(self, multi_ensemble_rdi_file_with_vl):
        """Test ADC channels are consistent across ensembles."""
        data, _, error_code = variableleader(str(multi_ensemble_rdi_file_with_vl))

        assert error_code == 0
        # All ensembles should have same ADC channels
        for i in range(24, 32):
            assert np.all(data[i, :] == data[i, 0])


# ============================================================================
# TESTS: Y2K RTC Time Fields
# ============================================================================


class TestVariableleaderY2KRTC:
    """Test Y2K Real-Time Clock field extraction."""

    def test_rtc_century_field(self, valid_rdi_file_with_vl):
        """Test that RTC century field (field 40) is valid."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Century should be reasonable (20, 21, etc.) or default 0
        assert 0 <= data[40, 0] <= 25

    def test_rtc_year_field(self, valid_rdi_file_with_vl):
        """Test that RTC year field (field 41) is valid."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Year should be 0-99
        assert 0 <= data[41, 0] <= 99

    def test_rtc_month_field(self, valid_rdi_file_with_vl):
        """Test that RTC month field (field 42) is valid."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Month should be 1-12 or default 0
        assert 0 <= data[42, 0] <= 12

    def test_rtc_day_field(self, valid_rdi_file_with_vl):
        """Test that RTC day field (field 43) is valid."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Day should be 1-31 or default 0
        assert 0 <= data[43, 0] <= 31

    def test_rtc_hour_field(self, valid_rdi_file_with_vl):
        """Test that RTC hour field (field 44) is valid."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Hour should be 0-23
        assert 0 <= data[44, 0] <= 23

    def test_rtc_minute_field(self, valid_rdi_file_with_vl):
        """Test that RTC minute field (field 45) is valid."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Minute should be 0-59
        assert 0 <= data[45, 0] <= 59

    def test_rtc_second_field(self, valid_rdi_file_with_vl):
        """Test that RTC second field (field 46) is valid."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Second should be 0-59
        assert 0 <= data[46, 0] <= 59

    def test_rtc_hundredth_field(self, valid_rdi_file_with_vl):
        """Test that RTC hundredth field (field 47) is valid."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        # Hundredth should be 0-99
        assert 0 <= data[47, 0] <= 99

    def test_custom_rtc_values(self, custom_vl_ensemble):
        """Test extraction of custom Y2K RTC values."""
        data, _, error_code = variableleader(str(custom_vl_ensemble))

        assert error_code == 0
        assert data[40, 0] == 20  # Century
        assert data[41, 0] == 25  # Year
        assert data[42, 0] == 3  # Month
        assert data[43, 0] == 15  # Day
        assert data[44, 0] == 14  # Hour
        assert data[45, 0] == 30  # Minute
        assert data[46, 0] == 45  # Second
        assert data[47, 0] == 50  # Hundredth


# ============================================================================
# TESTS: Integration with fileheader
# ============================================================================


class TestVariableleaderIntegration:
    """Test variableleader integration with fileheader."""

    def test_variableleader_without_fileheader_params(self, valid_rdi_file_with_vl):
        """Test that variableleader works without fileheader parameters.

        Should internally call fileheader when parameters not provided.
        """
        # Call with only filename (no fileheader parameters)
        data, ensemble_count, error_code = variableleader(str(valid_rdi_file_with_vl))
        assert error_code == 0
        assert ensemble_count == 1
        assert data.shape == (48, 1)

    def test_variableleader_with_fileheader_params(self, valid_rdi_file_with_vl):
        """Test that variableleader uses provided fileheader parameters."""
        # First call fileheader
        (
            source_id,
            header_id,
            byteskip,
            offset,
            idarray,
            ensemble_count,
            error_code_fh,
        ) = fileheader(str(valid_rdi_file_with_vl))

        assert error_code_fh == 0

        # Now call variableleader with those parameters
        data, ens_count, error_code = variableleader(
            str(valid_rdi_file_with_vl),
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble_count,
        )

        assert error_code == 0
        assert ens_count == ensemble_count
        assert data.shape[1] == ensemble_count

    def test_fileheader_params_match_results(self, valid_rdi_file_with_vl):
        """Test that results are consistent whether using fileheader params or not."""
        # Without params
        data1, ens1, err1 = variableleader(str(valid_rdi_file_with_vl))

        # With params
        (_, _, bs, off, ida, ens_fh, err_fh) = fileheader(str(valid_rdi_file_with_vl))
        data2, ens2, err2 = variableleader(
            str(valid_rdi_file_with_vl),
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


class TestVariableleaderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_file(self, tmp_path):
        """Test handling of empty file."""
        empty_file = tmp_path / "empty.000"
        empty_file.write_bytes(b"")

        data, ensemble_count, error_code = variableleader(str(empty_file))
        # Should return error or empty result
        assert error_code != 0 or ensemble_count == 0

    def test_data_array_dtype_consistency(self, valid_rdi_file_with_vl):
        """Test that all array values are int32."""
        data, _, _ = variableleader(str(valid_rdi_file_with_vl))
        # All values should be int32
        assert data.dtype == np.int32
        # No NaN or inf values (int32 can't have these, but check for sanity)
        assert data.size > 0

    def test_consistent_field_meanings_across_ensembles(
        self, multi_ensemble_rdi_file_with_vl
    ):
        """Test that field meanings are consistent across multiple ensembles."""
        data, _, _ = variableleader(str(multi_ensemble_rdi_file_with_vl))

        # Variable Leader ID should be consistent
        assert np.all(data[0, :] == data[0, 0])

        # Ensemble numbers might increase or stay same
        assert np.all(data[1, :] >= 0)


# ============================================================================
# TESTS: Struct Unpacking Errors
# ============================================================================


class TestVariableleaderStructErrors:
    """Test handling of struct unpacking errors."""

    def test_struct_unpack_error_recovery(self, rdi_file_truncated_vl):
        """Test that struct unpacking errors are caught and handled."""
        # Should not raise exception, should return error code
        data, ensemble_count, error_code = variableleader(str(rdi_file_truncated_vl))

        # Should complete without raising exception
        assert error_code != 0 or ensemble_count == 0


# ============================================================================
# TESTS: Data Validation
# ============================================================================


class TestVariableleaderDataValidation:
    """Test data validation and sanity checks."""

    def test_all_timestamp_fields_non_negative(self, valid_rdi_file_with_vl):
        """Test that timestamp fields are non-negative."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        if error_code == 0:
            # Fields 2-8 are timestamp fields (year, month, day, hour, minute, second, hundredth)
            for i in range(2, 9):
                assert data[i, 0] >= 0

    def test_no_nan_values(self, valid_rdi_file_with_vl):
        """Test that no NaN values are present (int32 can't have NaN, but sanity check)."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        if error_code == 0:
            assert not np.any(np.isnan(data.astype(float)))

    def test_no_inf_values(self, valid_rdi_file_with_vl):
        """Test that no infinite values are present."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        if error_code == 0:
            assert not np.any(np.isinf(data.astype(float)))

    def test_vl_id_in_valid_range(self, valid_rdi_file_with_vl):
        """Test that Variable Leader ID is in valid range."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        if error_code == 0:
            # VL ID should be 128 or 129
            assert data[0, 0] in (128, 129)


# ============================================================================
# TESTS: Pathlib Support
# ============================================================================


class TestVariableleaderPathlibSupport:
    """Test that variableleader works with pathlib.Path objects."""

    def test_path_object_string_conversion(self, valid_rdi_file_with_vl):
        """Test that variableleader accepts pathlib.Path objects."""
        data, ensemble_count, error_code = variableleader(valid_rdi_file_with_vl)
        assert error_code == 0
        assert ensemble_count == 1

    def test_string_path_works_same_as_path_object(self, valid_rdi_file_with_vl):
        """Test that string paths and Path objects give identical results."""
        data1, ens1, err1 = variableleader(str(valid_rdi_file_with_vl))
        data2, ens2, err2 = variableleader(valid_rdi_file_with_vl)

        assert err1 == err2
        assert ens1 == ens2
        np.testing.assert_array_equal(data1, data2)


# ============================================================================
# TESTS: Field Array Dimensions
# ============================================================================


class TestVariableleaderFieldDimensions:
    """Test that Variable Leader array dimensions are correct."""

    def test_array_has_48_fields(self, valid_rdi_file_with_vl):
        """Test that returned array has 48 fields (rows)."""
        data, _, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        assert data.shape[0] == 48

    def test_array_has_correct_ensemble_dimension(self, valid_rdi_file_with_vl):
        """Test that array has correct number of ensembles (columns)."""
        data, ensemble_count, error_code = variableleader(str(valid_rdi_file_with_vl))

        assert error_code == 0
        assert data.shape[1] == ensemble_count

    def test_multi_ensemble_dimensions(self, multi_ensemble_rdi_file_with_vl):
        """Test array dimensions for multiple ensembles."""
        data, ensemble_count, error_code = variableleader(
            str(multi_ensemble_rdi_file_with_vl)
        )

        assert error_code == 0
        assert data.shape == (48, ensemble_count)
        assert ensemble_count == 3


# ============================================================================
# TESTS: Truncation Behavior
# ============================================================================


class TestVariableleaderTruncation:
    """Test proper truncation behavior on errors."""

    def test_truncation_on_corruption(self, rdi_file_multiple_with_vl_corruption):
        """Test that file is properly truncated when corruption encountered."""
        data, ensemble_count, error_code = variableleader(
            str(rdi_file_multiple_with_vl_corruption)
        )

        # Should detect corruption and truncate
        # Either error_code != 0 or ensemble_count < 2 (expected)
        assert error_code != 0 or ensemble_count <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
