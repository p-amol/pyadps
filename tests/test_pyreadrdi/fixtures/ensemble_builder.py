"""
Shared test fixture builder for RDI ADCP ensemble construction.

This module provides functions to construct valid RDI ensemble components
matching WorkHorse specification (PD0 format). Fixtures can be composed
to build complete ensembles for testing.

Reference: RDI WorkHorse Commands and Output Data Format (PD0 Section 7)

Key Data Sizes (bytes):
- Fixed Leader: 59 bytes (fixed)
- Variable Leader: 65 bytes (fixed)
- Velocity: 2 + (2 * beams * cells)
- Echo Intensity: 2 + (1 * beams * cells)
- Correlation: 2 + (1 * beams * cells)
- Percent Good: 2 + (1 * beams * cells)
- Bottom Track: 85 bytes (fixed)
"""

import struct
from typing import NamedTuple


class EnsembleConfig(NamedTuple):
    """Configuration parameters for ensemble generation."""

    num_datatypes: int = 7
    beams: int = 4
    cells: int = 30


class FixedLeaderData(NamedTuple):
    """Fixed Leader data fields (59 bytes total)."""

    fid: int = 0x0000  # Fixed Leader ID (bytes 1-2)
    cpu_fw_ver: int = 16  # CPU F/W version (byte 3)
    cpu_fw_rev: int = 5  # CPU F/W revision (byte 4)
    system_config: int = 0x5249  # System configuration (bytes 5-6, little-endian)
    real_sim_flag: int = 0  # Real (0) or simulation (1) data (byte 7)
    lag_length: int = 0  # Lag length (byte 8)
    num_beams: int = 4  # Number of beams (byte 9)
    num_cells: int = 30  # Number of depth cells (byte 10)
    pings_per_ensemble: int = 1  # Pings per ensemble (bytes 11-12)
    cell_length: int = 400  # Depth cell length in 1/100ths mm (bytes 13-14)
    blank_after_transmit: int = 352  # Blank after transmit (bytes 15-16)
    profiling_mode: int = 1  # Profiling mode (byte 17)
    low_corr_thresh: int = 64  # Low correlation threshold (byte 18)
    no_code_reps: int = 0  # Number of code repetitions (byte 19)
    gd_minimum: int = 0  # Percent good minimum (byte 20)
    error_velocity_max: int = 5000  # Error velocity maximum (bytes 21-22)
    tpp_minutes: int = 0  # Time between pings - minutes (byte 23)
    tpp_seconds: int = 0  # Time between pings - seconds (byte 24)
    tpp_hundredths: int = 0  # Time between pings - hundredths (byte 25)
    coord_transform: int = 0  # Coordinate transform (byte 26)
    heading_alignment: int = 0  # Heading alignment in 1/100ths degrees (bytes 27-28)
    heading_bias: int = 0  # Heading bias in 1/100ths degrees (bytes 29-30)
    sensor_source: int = 1  # Sensor source (byte 31)
    sensors_available: int = 255  # Sensors available (byte 32)
    bin_1_distance: int = (
        176  # Distance to first bin in 1/100ths meters (bytes 33-34, 2 bytes)
    )
    xmit_pulse_length: int = 0  # Transmit pulse length (bytes 34-35)
    wp_ref_layer_avg: int = 0  # Water profile reference layer average (bytes 36-37)
    false_target_thresh: int = 0  # False target threshold (byte 38)
    spare_1: int = 0  # Spare (byte 39)
    transmit_lag_distance: int = 0  # Transmit lag distance (bytes 40-41)
    cpu_board_sn: int = 12345  # CPU board serial number (bytes 42-49)
    system_bandwidth: int = 0  # System bandwidth (bytes 50-51)
    system_power: int = 0  # System power (byte 52)
    spare_2: int = 0  # Spare (byte 53)
    instrument_sn: int = 56789  # Instrument serial number (bytes 54-57)
    beam_angle: int = 20  # Beam angle in degrees (byte 58)


class VariableLeaderData(NamedTuple):
    """Variable Leader data fields (65 bytes total)."""

    vid: int = 0x0080  # Variable Leader ID (bytes 1-2)
    ensemble_number: int = 1  # Ensemble number (bytes 3-4)
    year: int = 25  # Year (YY, bytes 5)
    month: int = 1  # Month (MM, bytes 6)
    day: int = 15  # Day (DD, bytes 7)
    hour: int = 10  # Hour (HH, bytes 8)
    minute: int = 30  # Minute (MM, bytes 9)
    second: int = 45  # Second (SS, bytes 10)
    hundredth: int = 50  # Hundredth of second (bytes 11)
    heading: int = 0  # Heading in 1/100ths degrees (bytes 12-13)
    pitch: int = 0  # Pitch in 1/100ths degrees (bytes 14-15)
    roll: int = 0  # Roll in 1/100ths degrees (bytes 16-17)
    salinity: int = 35000  # Salinity in 1/1000ths ppt (bytes 18-19)
    temperature: int = 2000  # Temperature in 1/100ths Â°C (bytes 20-21)
    pressure: int = 0  # Pressure in decapascals (bytes 22-23)
    pressure_variance: int = 0  # Pressure variance (bytes 24-25)
    spare_1: int = 0  # Spare (bytes 26-27)
    spare_2: int = 0  # Spare (bytes 28-29)
    spare_3: int = 0  # Spare (bytes 30-31)
    rtc_century: int = 20  # RTC century (bytes 32)
    rtc_year: int = 25  # RTC year (bytes 33)
    rtc_month: int = 1  # RTC month (bytes 34)
    rtc_day: int = 15  # RTC day (bytes 35)
    rtc_hour: int = 10  # RTC hour (bytes 36)
    rtc_minute: int = 30  # RTC minute (bytes 37)
    rtc_second: int = 45  # RTC second (bytes 38)
    rtc_hundredth: int = 50  # RTC hundredth (bytes 39)


def build_fixed_leader(fixed_leader_data: FixedLeaderData) -> bytes:
    """
    Build Fixed Leader section (59 bytes).

    Args:
        fixed_leader_data: FixedLeaderData namedtuple with all field values.

    Returns:
        59-byte binary Fixed Leader section.

    Reference: WorkHorse spec, Section 5.2 (page 126)
    """
    fl = b""

    # Bytes 1-2: Fixed Leader ID
    fl += struct.pack("<H", fixed_leader_data.fid)

    # Byte 3: CPU F/W Version
    fl += struct.pack("<B", fixed_leader_data.cpu_fw_ver)

    # Byte 4: CPU F/W Revision
    fl += struct.pack("<B", fixed_leader_data.cpu_fw_rev)

    # Bytes 5-6: System Configuration
    fl += struct.pack("<H", fixed_leader_data.system_config)

    # Byte 7: Real/Sim Flag
    fl += struct.pack("<B", fixed_leader_data.real_sim_flag)

    # Byte 8: Lag Length
    fl += struct.pack("<B", fixed_leader_data.lag_length)

    # Byte 9: Number of Beams
    fl += struct.pack("<B", fixed_leader_data.num_beams)

    # Byte 10: Number of Cells
    fl += struct.pack("<B", fixed_leader_data.num_cells)

    # Bytes 11-12: Pings Per Ensemble
    fl += struct.pack("<H", fixed_leader_data.pings_per_ensemble)

    # Bytes 13-14: Depth Cell Length
    fl += struct.pack("<H", fixed_leader_data.cell_length)

    # Bytes 15-16: Blank After Transmit
    fl += struct.pack("<H", fixed_leader_data.blank_after_transmit)

    # Byte 17: Profiling Mode
    fl += struct.pack("<B", fixed_leader_data.profiling_mode)

    # Byte 18: Low Correlation Threshold
    fl += struct.pack("<B", fixed_leader_data.low_corr_thresh)

    # Byte 19: Number of Code Repetitions
    fl += struct.pack("<B", fixed_leader_data.no_code_reps)

    # Byte 20: Percent Good Minimum
    fl += struct.pack("<B", fixed_leader_data.gd_minimum)

    # Bytes 21-22: Error Velocity Maximum
    fl += struct.pack("<H", fixed_leader_data.error_velocity_max)

    # Byte 23: Time Per Ping - Minutes
    fl += struct.pack("<B", fixed_leader_data.tpp_minutes)

    # Byte 24: Time Per Ping - Seconds
    fl += struct.pack("<B", fixed_leader_data.tpp_seconds)

    # Byte 25: Time Per Ping - Hundredths
    fl += struct.pack("<B", fixed_leader_data.tpp_hundredths)

    # Byte 26: Coordinate Transform
    fl += struct.pack("<B", fixed_leader_data.coord_transform)

    # Bytes 27-28: Heading Alignment
    fl += struct.pack("<H", fixed_leader_data.heading_alignment)

    # Bytes 29-30: Heading Bias
    fl += struct.pack("<H", fixed_leader_data.heading_bias)

    # Byte 31: Sensor Source
    fl += struct.pack("<B", fixed_leader_data.sensor_source)

    # Byte 32: Sensors Available
    fl += struct.pack("<B", fixed_leader_data.sensors_available)

    # Bytes 33-34: Bin 1 Distance (2 bytes, not 1!)
    fl += struct.pack("<H", fixed_leader_data.bin_1_distance)

    # Bytes 35-36: Transmit Pulse Length
    fl += struct.pack("<H", fixed_leader_data.xmit_pulse_length)

    # Bytes 36-37: Water Profile Reference Layer Average
    fl += struct.pack("<H", fixed_leader_data.wp_ref_layer_avg)

    # Byte 38: False Target Threshold
    fl += struct.pack("<B", fixed_leader_data.false_target_thresh)

    # Byte 39: Spare
    fl += struct.pack("<B", fixed_leader_data.spare_1)

    # Bytes 40-41: Transmit Lag Distance
    fl += struct.pack("<H", fixed_leader_data.transmit_lag_distance)

    # Bytes 42-49: CPU Board Serial Number (8 bytes)
    fl += struct.pack("<Q", fixed_leader_data.cpu_board_sn)

    # Bytes 50-51: System Bandwidth
    fl += struct.pack("<H", fixed_leader_data.system_bandwidth)

    # Byte 52: System Power
    fl += struct.pack("<B", fixed_leader_data.system_power)

    # Byte 53: Spare
    fl += struct.pack("<B", fixed_leader_data.spare_2)

    # Bytes 54-57: Instrument Serial Number (4 bytes)
    fl += struct.pack("<I", fixed_leader_data.instrument_sn)

    # Byte 58: Beam Angle
    fl += struct.pack("<B", fixed_leader_data.beam_angle)

    assert len(fl) == 59, f"Fixed Leader must be 59 bytes, got {len(fl)}"
    return fl


def build_variable_leader(variable_leader_data: VariableLeaderData) -> bytes:
    """
    Build Variable Leader section (65 bytes).

    Packing order matches pyreadrdi.variableleader() unpacking format exactly.

    Args:
        variable_leader_data: VariableLeaderData namedtuple with all field values.

    Returns:
        65-byte binary Variable Leader section.

    Reference: WorkHorse spec, Section 5.3 (page 134)
    Verified against: pyreadrdi.py variableleader() function (lines 786-848)
    """
    vl = b""

    # Bytes 0-1: Variable Leader ID
    vl += struct.pack("<H", variable_leader_data.vid)

    # Bytes 2-3: Ensemble Number
    vl += struct.pack("<H", variable_leader_data.ensemble_number)

    # Bytes 4-10: UTC Time (Year, Month, Day, Hour, Minute, Second, Hundredth)
    vl += struct.pack(
        "<BBBBBBB",
        variable_leader_data.year,
        variable_leader_data.month,
        variable_leader_data.day,
        variable_leader_data.hour,
        variable_leader_data.minute,
        variable_leader_data.second,
        variable_leader_data.hundredth,
    )

    # Bytes 11-13: Ensemble MSB (B) + BIT Result (H)
    # Note: These fields are not currently in VariableLeaderData, use defaults
    vl += struct.pack("<BH", 0, 0)

    # Bytes 14-27: Sound Speed (H), Transducer Depth (H), Heading (H),
    #             Pitch (h), Roll (h), Temperature (H), Salinity (h)
    # Note: Sound Speed and Transducer Depth not in VariableLeaderData, use defaults
    # NOTE: Salinity is packed as signed (h), so max value is 32767, not 40000
    # Using typical value: 35000 ppt * 0.001 = 35 ppt, but stored as signed
    vl += struct.pack(
        "<HHHhhHh",
        0,  # Sound Speed (m/s * 100)
        0,  # Transducer Depth
        variable_leader_data.heading,
        variable_leader_data.pitch,
        variable_leader_data.roll,
        variable_leader_data.temperature,
        min(variable_leader_data.salinity, 32767),  # Cap to signed int16 max
    )

    # Bytes 28-30: MPT Minutes, Seconds, Hundredth (BBB)
    # Note: These fields are not in VariableLeaderData, use defaults
    vl += struct.pack("<BBB", 0, 0, 0)

    # Bytes 31-33: Heading StdDev, Pitch StdDev, Roll StdDev (BBB)
    # Note: These fields are not in VariableLeaderData, use defaults
    vl += struct.pack("<BBB", 0, 0, 0)

    # Bytes 34-41: ADC Channels 0-7 (8 x B)
    # Note: These fields are not in VariableLeaderData, use defaults
    vl += b"\x00" * 8

    # Bytes 42-45: Error Status Word (4 x B)
    # Note: These fields are not in VariableLeaderData, use defaults
    vl += b"\x00" * 4

    # Bytes 46-56: Reserved (H), Pressure (i), Pressure Variance (i), Spare (B)
    vl += struct.pack(
        "<HiiB",
        0,  # Reserved
        variable_leader_data.pressure,  # Signed integer
        variable_leader_data.pressure_variance,  # Signed integer
        0,  # Spare
    )

    # Bytes 57-64: Y2K RTC Time (Century, Year, Month, Day, Hour, Minute, Second, Hundredth)
    vl += struct.pack(
        "<BBBBBBBB",
        variable_leader_data.rtc_century,
        variable_leader_data.rtc_year,
        variable_leader_data.rtc_month,
        variable_leader_data.rtc_day,
        variable_leader_data.rtc_hour,
        variable_leader_data.rtc_minute,
        variable_leader_data.rtc_second,
        variable_leader_data.rtc_hundredth,
    )

    assert len(vl) == 65, f"Variable Leader must be 65 bytes, got {len(vl)}"
    return vl


def build_data_section(
    data_id: int, payload_size: int, num_beams: int = 4, num_cells: int = 30
) -> bytes:
    """
    Build a generic data section with ID and payload.

    Args:
        data_id: 2-byte data type ID (e.g., 0x0100 for velocity).
        payload_size: Number of bytes in the payload (excluding the 2-byte ID).
        num_beams: Number of beams (used for some payload generation).
        num_cells: Number of depth cells (used for some payload generation).

    Returns:
        Binary data section with ID and payload.
    """
    data_section = struct.pack("<H", data_id)
    data_section += b"\x00" * payload_size
    return data_section


def build_ensemble(
    fixed_leader_data: FixedLeaderData = FixedLeaderData(),
    variable_leader_data: VariableLeaderData = VariableLeaderData(),
    config: EnsembleConfig = EnsembleConfig(),
) -> bytes:
    """
    Build a complete valid RDI ensemble with all standard sections and checksum.

    This is the primary function for test fixture generation. It constructs:
    - Header with offsets
    - Fixed Leader (59 bytes)
    - Variable Leader (65 bytes)
    - Velocity data
    - Echo Intensity data
    - Correlation data
    - Percent Good data
    - Bottom Track data (85 bytes)
    - Checksum

    Args:
        fixed_leader_data: Fixed Leader configuration (defaults provided).
        variable_leader_data: Variable Leader configuration (defaults provided).
        config: Ensemble configuration (datatypes, beams, cells).

    Returns:
        Complete binary RDI ensemble with valid checksum.

    Reference: WorkHorse spec, Section 7 (PD0 format, page 123)
    """
    beams = config.beams
    cells = config.cells
    num_datatypes = config.num_datatypes

    # ========== CALCULATE SECTION SIZES ==========
    fl_size = 59  # Fixed Leader (fixed)
    vl_size = 65  # Variable Leader (fixed)
    vel_size = 2 + (2 * beams * cells)  # Velocity
    echo_size = 2 + (1 * beams * cells)  # Echo Intensity
    cor_size = 2 + (1 * beams * cells)  # Correlation
    pg_size = 2 + (1 * beams * cells)  # Percent Good
    bt_size = 85  # Bottom Track (fixed)

    # Header size: 6 bytes + (2 bytes per datatype for offsets)
    header_size = 6 + (2 * num_datatypes)

    # Calculate cumulative offsets (where each section starts in the ensemble)
    offsets = [
        header_size,  # Fixed Leader
        header_size + fl_size,  # Variable Leader
        header_size + fl_size + vl_size,  # Velocity
        header_size + fl_size + vl_size + vel_size,  # Echo Intensity
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

    # Total size (excluding checksum)
    ensemble_size = offsets[-1] + bt_size

    # ========== BUILD HEADER ==========
    header = struct.pack(
        "<BBHBB",
        0x7F,  # header_id
        0x7F,  # source_id
        ensemble_size,  # byte_count (excludes checksum per RDI spec Section 7.2)
        0,  # spare
        num_datatypes,  # num_datatypes
    )

    # Add offset array
    offset_array = struct.pack("<HHHHHHH", *offsets)

    # ========== BUILD DATA SECTIONS ==========
    # Data IDs matching WorkHorse spec
    data_ids = [0x0000, 0x0080, 0x0100, 0x0200, 0x0300, 0x0400, 0x0600]
    data_sections = b""

    # Build each data section
    data_sections += build_fixed_leader(fixed_leader_data)
    data_sections += build_variable_leader(variable_leader_data)
    data_sections += build_data_section(data_ids[2], vel_size - 2, beams, cells)
    data_sections += build_data_section(data_ids[3], echo_size - 2, beams, cells)
    data_sections += build_data_section(data_ids[4], cor_size - 2, beams, cells)
    data_sections += build_data_section(data_ids[5], pg_size - 2, beams, cells)
    data_sections += build_data_section(data_ids[6], bt_size - 2, beams, cells)

    # ========== CALCULATE CHECKSUM ==========
    ensemble_without_checksum = header + offset_array + data_sections
    calculated_checksum = sum(ensemble_without_checksum) & 0xFFFF
    checksum_bytes = struct.pack("<H", calculated_checksum)

    # ========== ASSEMBLE COMPLETE ENSEMBLE ==========
    complete_ensemble = ensemble_without_checksum + checksum_bytes

    return complete_ensemble


# Convenience defaults for common configurations
DEFAULT_FIXED_LEADER = FixedLeaderData()
DEFAULT_VARIABLE_LEADER = VariableLeaderData()
DEFAULT_CONFIG = EnsembleConfig()
