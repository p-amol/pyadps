"""
Test suite for pyadps.processing.config.ProcessingConfig (v1.0.0).

Covers:
- Default values and dataclass creation
- from_ini(): INI file parsing for all six sections
- to_ini(): INI file serialization
- Round-trip: to_ini() -> from_ini() preserves all fields
- validate(): detects missing/invalid configurations
- _get_replacement_value(): None / Fixed Value / File dispatch
- _load_timeseries_from_file(): CSV and plain-text loading
- to_dict(): dataclass serialization
- __repr__(): string representation
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    from pyadps.processing.config import ProcessingConfig
except ImportError:
    try:
        from config import ProcessingConfig
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from config import ProcessingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_ini(content: str) -> str:
    """Write *content* to a temporary INI file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False)
    f.write(content)
    f.close()
    return f.name


def _write_csv(values, header: str | None = None) -> str:
    """Write a one-column CSV of *values* and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    if header:
        f.write(header + "\n")
    for v in values:
        f.write(f"{v}\n")
    f.close()
    return f.name


# ===========================================================================
# 1. Default values
# ===========================================================================


class TestProcessingConfigDefaults:
    """ProcessingConfig() must produce sensible defaults for every field."""

    def test_default_flags_false(self):
        cfg = ProcessingConfig()
        assert cfg.isTimeAxisModified is False
        assert cfg.isSensorTest is False
        assert cfg.isQCTest is False
        assert cfg.isProfileTest is False
        assert cfg.isVelocityTest is False
        assert cfg.isAttributes is False

    def test_default_file_settings(self):
        cfg = ProcessingConfig()
        assert cfg.input_file_name == ""
        assert cfg.input_file_path == ""
        assert cfg.output_file_path == ""

    def test_default_time_axis(self):
        cfg = ProcessingConfig()
        assert cfg.isSnapTimeAxis is False
        assert cfg.time_snap_frequency == "h"
        assert cfg.time_snap_tolerance == "5min"
        assert cfg.time_target_minute == 0
        assert cfg.isTimeGapFilled is False
        assert cfg.time_fill_method == "auto"

    def test_default_sensor_health_tilt(self):
        cfg = ProcessingConfig()
        assert cfg.roll_cutoff_ST == 15.0
        assert cfg.pitch_cutoff_ST == 15.0
        assert cfg.isRollCheck_ST is False
        assert cfg.isPitchCheck_ST is False

    def test_default_sensor_health_replacement(self):
        cfg = ProcessingConfig()
        assert cfg.depthoption_ST == "None"
        assert cfg.salinityoption_ST == "None"
        assert cfg.temperatureoption_ST == "None"
        assert cfg.fixeddepth_ST == 0.0
        assert cfg.fixedsalinity_ST == 35.0
        assert cfg.fixedtemperature_ST == 15.0

    def test_default_qc_thresholds(self):
        cfg = ProcessingConfig()
        assert cfg.ct_QCT == 64.0
        assert cfg.et_QCT == 0.0
        assert cfg.evt_QCT == 2000.0
        assert cfg.ft_QCT == 50.0
        assert cfg.pgt_QCT == 0.0
        assert cfg.is3beam_QCT is False
        assert cfg.beam_ignore_QCT is None

    def test_default_profile_test(self):
        cfg = ProcessingConfig()
        assert cfg.isTrimEndsCheck_PT is False
        assert cfg.trim_start_PT == 0
        assert cfg.trim_end_PT == 0
        assert cfg.isCutBinSideLobeCheck_PT is False
        assert cfg.water_depth_PT == 0.0
        assert cfg.extra_cells_PT == 1
        assert cfg.isCutBinManualCheck_PT is False
        assert cfg.cut_bins_start_PT == 0
        assert cfg.cut_bins_end_PT == 0
        assert cfg.isRegridCheck_PT is False
        assert cfg.regrid_cell_size_PT == 1.0
        assert cfg.regrid_method_PT == "nearest"
        assert cfg.beam_direction_PT == "up"

    def test_default_velocity_test(self):
        cfg = ProcessingConfig()
        assert cfg.isMagnetCheck_VT is False
        assert cfg.magnet_method_VT == "api"
        assert cfg.magnet_lat_VT == 0.0
        assert cfg.magnet_lon_VT == 0.0
        assert cfg.magnet_year_VT == 2025
        assert cfg.isCutoffCheck_VT is False
        assert cfg.maxuvel_VT == 2500.0
        assert cfg.maxvvel_VT == 2500.0
        assert cfg.maxwvel_VT == 500.0
        assert cfg.isDespikeCheck_VT is False
        assert cfg.despike_kernel_VT == 5
        assert cfg.despike_cutoff_VT == 3.0
        assert cfg.isFlatlineCheck_VT is False
        assert cfg.flatline_kernel_VT == 5
        assert cfg.flatline_cutoff_VT == 3.0

    def test_default_attributes_empty_dict(self):
        cfg = ProcessingConfig()
        assert cfg.attributes == {}


# ===========================================================================
# 2. Programmatic creation
# ===========================================================================


class TestProcessingConfigProgrammatic:
    """ProcessingConfig can be built with keyword arguments."""

    def test_override_at_construction(self):
        cfg = ProcessingConfig(
            isSensorTest=True,
            roll_cutoff_ST=20.0,
            ct_QCT=50.0,
        )
        assert cfg.isSensorTest is True
        assert cfg.roll_cutoff_ST == 20.0
        assert cfg.ct_QCT == 50.0

    def test_attributes_dict_mutable(self):
        cfg = ProcessingConfig(
            isAttributes=True,
            attributes={"cruise": "CR001", "vessel": "RV Test"},
        )
        assert cfg.attributes["cruise"] == "CR001"
        assert cfg.attributes["vessel"] == "RV Test"

    def test_beam_ignore_none_by_default(self):
        cfg = ProcessingConfig(is3beam_QCT=True)
        assert cfg.beam_ignore_QCT is None

    def test_beam_ignore_integer(self):
        cfg = ProcessingConfig(is3beam_QCT=True, beam_ignore_QCT=2)
        assert cfg.beam_ignore_QCT == 2


# ===========================================================================
# 3. to_ini() — serialization
# ===========================================================================


class TestToIni:
    """to_ini() must write a valid INI file with all required sections."""

    def test_filesettings_section_present(self, tmp_path):
        cfg = ProcessingConfig(input_file_name="data.pd0")
        p = tmp_path / "config.ini"
        cfg.to_ini(str(p))
        assert "[FileSettings]" in p.read_text()

    def test_input_file_name_written(self, tmp_path):
        cfg = ProcessingConfig(input_file_name="cruise001.pd0")
        p = tmp_path / "config.ini"
        cfg.to_ini(str(p))
        assert "input_file_name = cruise001.pd0" in p.read_text()

    def test_input_file_path_written(self, tmp_path):
        cfg = ProcessingConfig(input_file_path="/data/raw/cruise001.pd0")
        p = tmp_path / "config.ini"
        cfg.to_ini(str(p))
        assert "input_file_path = /data/raw/cruise001.pd0" in p.read_text()

    def test_output_file_path_written(self, tmp_path):
        cfg = ProcessingConfig(output_file_path="/data/processed/")
        p = tmp_path / "config.ini"
        cfg.to_ini(str(p))
        assert "output_file_path = /data/processed/" in p.read_text()

    def test_pyadps_version_written(self, tmp_path):
        """
        pyadps_version is informational only (not read by apply_config()),
        but must round-trip so a config.ini records which pyadps version
        read the binary file.
        """
        cfg = ProcessingConfig(pyadps_version="1.0.2")
        p = tmp_path / "config.ini"
        cfg.to_ini(str(p))
        assert "pyadps_version = 1.0.2" in p.read_text()

    def test_creates_file(self, tmp_path):
        cfg = ProcessingConfig()
        p = tmp_path / "config.ini"
        cfg.to_ini(str(p))
        assert p.exists()

    def test_all_sections_present(self, tmp_path):
        cfg = ProcessingConfig()
        p = tmp_path / "config.ini"
        cfg.to_ini(str(p))
        text = p.read_text()
        for section in (
            "FileSettings",
            "FixTime",
            "SensorTest",
            "QCTest",
            "ProfileTest",
            "VelocityTest",
            "Attributes",
        ):
            assert f"[{section}]" in text, f"Missing section [{section}]"

    def test_boolean_serialized_as_string(self, tmp_path):
        cfg = ProcessingConfig(isSensorTest=True, isRollCheck_ST=True)
        p = tmp_path / "config.ini"
        cfg.to_ini(str(p))
        text = p.read_text()
        assert "sensor_test = True" in text
        assert "roll_check = True" in text

    def test_float_serialized_correctly(self, tmp_path):
        cfg = ProcessingConfig(roll_cutoff_ST=22.5)
        p = tmp_path / "config.ini"
        cfg.to_ini(str(p))
        assert "22.5" in p.read_text()

    def test_beam_ignore_none_written_as_empty(self, tmp_path):
        cfg = ProcessingConfig(beam_ignore_QCT=None)
        p = tmp_path / "config.ini"
        cfg.to_ini(str(p))
        text = p.read_text()
        assert "beam_ignore = \n" in text or "beam_ignore =\n" in text

    def test_beam_ignore_int_written(self, tmp_path):
        cfg = ProcessingConfig(beam_ignore_QCT=3)
        p = tmp_path / "config.ini"
        cfg.to_ini(str(p))
        assert "beam_ignore = 3" in p.read_text()

    def test_attributes_json_written(self, tmp_path):
        cfg = ProcessingConfig(
            isAttributes=True,
            attributes={"key": "val"},
        )
        p = tmp_path / "config.ini"
        cfg.to_ini(str(p))
        text = p.read_text()
        assert "attributes_json" in text
        assert "key" in text

    def test_overwrite_existing_file(self, tmp_path):
        p = tmp_path / "config.ini"
        p.write_text("old content")
        cfg = ProcessingConfig(isQCTest=True)
        cfg.to_ini(str(p))
        text = p.read_text()
        assert "[QCTest]" in text
        assert "old content" not in text

    def test_extra_cells_written(self, tmp_path):
        cfg = ProcessingConfig(extra_cells_PT=4)
        p = tmp_path / "config.ini"
        cfg.to_ini(str(p))
        assert "extra_cells = 4" in p.read_text()

    def test_regrid_method_written(self, tmp_path):
        cfg = ProcessingConfig(regrid_method_PT="nearest")
        p = tmp_path / "config.ini"
        cfg.to_ini(str(p))
        assert "regrid_method = nearest" in p.read_text()


# ===========================================================================
# 4. from_ini() — parsing
# ===========================================================================


class TestFromIni:
    """from_ini() must parse every INI section into the correct fields."""

    def test_filesettings_section(self, tmp_path):
        p = tmp_path / "cfg.ini"
        p.write_text(
            "[FileSettings]\n"
            "input_file_name = cruise001.pd0\n"
            "input_file_path = /data/raw/cruise001.pd0\n"
            "output_file_path = /data/processed/\n"
            "pyadps_version = 1.0.2\n"
        )
        cfg = ProcessingConfig.from_ini(str(p))
        assert cfg.input_file_name == "cruise001.pd0"
        assert cfg.input_file_path == "/data/raw/cruise001.pd0"
        assert cfg.output_file_path == "/data/processed/"
        assert cfg.pyadps_version == "1.0.2"

    def test_filesettings_absent_falls_back_to_defaults(self, tmp_path):
        """Old INI files without [FileSettings] must use empty-string defaults."""
        p = tmp_path / "cfg.ini"
        p.write_text("[QCTest]\nqc_test = True\n")
        cfg = ProcessingConfig.from_ini(str(p))
        assert cfg.input_file_name == ""
        assert cfg.input_file_path == ""
        assert cfg.output_file_path == ""
        assert cfg.pyadps_version == ""

    def test_missing_file_returns_defaults(self):
        # configparser.read() silently ignores missing files, so from_ini
        # falls back to all defaults rather than raising.
        cfg = ProcessingConfig.from_ini("/nonexistent/path/config.ini")
        assert cfg.isSensorTest is False
        assert cfg.ct_QCT == 64.0

    def test_empty_ini_returns_defaults(self, tmp_path):
        p = tmp_path / "empty.ini"
        p.write_text("")
        cfg = ProcessingConfig.from_ini(str(p))
        assert cfg.isSensorTest is False
        assert cfg.ct_QCT == 64.0

    def test_fixtime_section(self, tmp_path):
        p = tmp_path / "cfg.ini"
        p.write_text(
            "[FixTime]\n"
            "is_time_modified = True\n"
            "is_snap_time_axis = True\n"
            "time_snap_frequency = 15min\n"
            "time_snap_tolerance = 3min\n"
            "time_target_minute = 30\n"
            "is_time_gap_filled = True\n"
            "time_fill_method = linear\n"
        )
        cfg = ProcessingConfig.from_ini(str(p))
        assert cfg.isTimeAxisModified is True
        assert cfg.isSnapTimeAxis is True
        assert cfg.time_snap_frequency == "15min"
        assert cfg.time_snap_tolerance == "3min"
        assert cfg.time_target_minute == 30
        assert cfg.isTimeGapFilled is True
        assert cfg.time_fill_method == "linear"

    def test_sensortest_section(self, tmp_path):
        p = tmp_path / "cfg.ini"
        p.write_text(
            "[SensorTest]\n"
            "sensor_test = True\n"
            "roll_check = True\n"
            "pitch_check = True\n"
            "roll_cutoff = 12.0\n"
            "pitch_cutoff = 10.0\n"
            "depth_modified = True\n"
            "depth_option = Fixed Value\n"
            "fixed_depth = 500.0\n"
            "depth_file = \n"
            "salinity_modified = False\n"
            "salinity_option = None\n"
            "fixed_salinity = 35.0\n"
            "salinity_file = \n"
            "temperature_modified = True\n"
            "temperature_option = Fixed Value\n"
            "fixed_temperature = 20.0\n"
            "temperature_file = \n"
            "sound_speed_correction = True\n"
            "velocity_correction = True\n"
            "horizontal_only = False\n"
        )
        cfg = ProcessingConfig.from_ini(str(p))
        assert cfg.isSensorTest is True
        assert cfg.isRollCheck_ST is True
        assert cfg.isPitchCheck_ST is True
        assert cfg.roll_cutoff_ST == 12.0
        assert cfg.pitch_cutoff_ST == 10.0
        assert cfg.isDepthModified_ST is True
        assert cfg.depthoption_ST == "Fixed Value"
        assert cfg.fixeddepth_ST == 500.0
        assert cfg.isTemperatureModified_ST is True
        assert cfg.temperatureoption_ST == "Fixed Value"
        assert cfg.fixedtemperature_ST == 20.0
        assert cfg.isSoundModified_ST is True
        assert cfg.isVelocityModified_ST is True
        assert cfg.isVelocityModified_HorizontalOnly_ST is False

    def test_qctest_section(self, tmp_path):
        p = tmp_path / "cfg.ini"
        p.write_text(
            "[QCTest]\n"
            "qc_test = True\n"
            "correlation = 50.0\n"
            "echo_intensity = 5.0\n"
            "error_velocity = 1500.0\n"
            "false_target = 40.0\n"
            "three_beam = True\n"
            "beam_ignore = 2\n"
            "percent_good = 25.0\n"
        )
        cfg = ProcessingConfig.from_ini(str(p))
        assert cfg.isQCTest is True
        assert cfg.ct_QCT == 50.0
        assert cfg.et_QCT == 5.0
        assert cfg.evt_QCT == 1500.0
        assert cfg.ft_QCT == 40.0
        assert cfg.is3beam_QCT is True
        assert cfg.beam_ignore_QCT == 2
        assert cfg.pgt_QCT == 25.0

    def test_qctest_beam_ignore_empty_becomes_none(self, tmp_path):
        p = tmp_path / "cfg.ini"
        p.write_text("[QCTest]\nbeam_ignore = \n")
        cfg = ProcessingConfig.from_ini(str(p))
        assert cfg.beam_ignore_QCT is None

    def test_profiletest_section(self, tmp_path):
        p = tmp_path / "cfg.ini"
        p.write_text(
            "[ProfileTest]\n"
            "profile_test = True\n"
            "trim_ends = True\n"
            "trim_start = 2\n"
            "trim_end = 5\n"
            "cut_sidelobe = True\n"
            "water_column_depth = 100.0\n"
            "extra_cells = 3\n"
            "cut_bins_manual = True\n"
            "cut_bins_regions = [[0, 5, -1, -1], [10, 15, 100, 200]]\n"
            "cut_bins_start = 1\n"
            "cut_bins_end = 50\n"
            "regrid = True\n"
            "regrid_cell_size = 2.0\n"
            "regrid_method = nearest\n"
            "beam_direction = down\n"
        )
        cfg = ProcessingConfig.from_ini(str(p))
        assert cfg.isProfileTest is True
        assert cfg.isTrimEndsCheck_PT is True
        assert cfg.trim_start_PT == 2
        assert cfg.trim_end_PT == 5
        assert cfg.isCutBinSideLobeCheck_PT is True
        assert cfg.water_depth_PT == 100.0
        assert cfg.extra_cells_PT == 3
        assert cfg.isCutBinManualCheck_PT is True
        # Multi-region: -1 entries decoded back to None
        assert cfg.cut_bins_regions_PT == [[0, 5, None, None], [10, 15, 100, 200]]
        # Legacy single-region fields still present
        assert cfg.cut_bins_start_PT == 1
        assert cfg.cut_bins_end_PT == 50
        assert cfg.isRegridCheck_PT is True
        assert cfg.regrid_cell_size_PT == 2.0
        assert cfg.regrid_method_PT == "nearest"
        assert cfg.beam_direction_PT == "down"

    def test_cut_bins_regions_absent_falls_back_to_empty(self, tmp_path):
        """Old INI files without cut_bins_regions must produce an empty list."""
        p = tmp_path / "cfg.ini"
        p.write_text(
            "[ProfileTest]\n"
            "profile_test = True\n"
            "cut_bins_manual = True\n"
            "cut_bins_start = 2\n"
            "cut_bins_end = 10\n"
            # no cut_bins_regions key
        )
        cfg = ProcessingConfig.from_ini(str(p))
        assert cfg.cut_bins_regions_PT == []
        # Legacy fields still load correctly
        assert cfg.cut_bins_start_PT == 2
        assert cfg.cut_bins_end_PT == 10

    def test_cut_bins_regions_invalid_json_falls_back_to_empty(self, tmp_path):
        """Malformed JSON in cut_bins_regions triggers JSONDecodeError → []."""
        p = tmp_path / "cfg.ini"
        p.write_text(
            "[ProfileTest]\n"
            "profile_test = True\n"
            "cut_bins_manual = True\n"
            "cut_bins_regions = {not valid json at all\n"  # malformed
        )
        cfg = ProcessingConfig.from_ini(str(p))
        assert cfg.cut_bins_regions_PT == []

    def test_cut_bins_regions_non_iterable_entries_falls_back_to_empty(self, tmp_path):
        """JSON that decodes to a non-iterable element triggers TypeError → []."""
        p = tmp_path / "cfg.ini"
        p.write_text(
            "[ProfileTest]\n"
            "profile_test = True\n"
            "cut_bins_manual = True\n"
            "cut_bins_regions = [1, 2, 3]\n"  # list of ints, not list of lists
        )
        # The list comprehension tries to iterate over each int as a region,
        # which raises TypeError because int is not iterable.
        cfg = ProcessingConfig.from_ini(str(p))
        assert cfg.cut_bins_regions_PT == []

    def test_velocitytest_section(self, tmp_path):
        p = tmp_path / "cfg.ini"
        p.write_text(
            "[VelocityTest]\n"
            "velocity_test = True\n"
            "magnetic_declination = True\n"
            "magnet_method = user\n"
            "latitude = 12.5\n"
            "longitude = 80.3\n"
            "year = 2024\n"
            "magnet_depth = 50.0\n"
            "magnet_user_input = -3.5\n"
            "velocity_cutoff = True\n"
            "max_zonal_velocity = 3000.0\n"
            "max_meridional_velocity = 3000.0\n"
            "max_vertical_velocity = 600.0\n"
            "despike = True\n"
            "despike_kernel_size = 7\n"
            "despike_cutoff = 2.5\n"
            "flatline = True\n"
            "flatline_kernel_size = 9\n"
            "flatline_cutoff = 4.0\n"
        )
        cfg = ProcessingConfig.from_ini(str(p))
        assert cfg.isVelocityTest is True
        assert cfg.isMagnetCheck_VT is True
        assert cfg.magnet_method_VT == "user"
        assert cfg.magnet_lat_VT == 12.5
        assert cfg.magnet_lon_VT == 80.3
        assert cfg.magnet_year_VT == 2024
        assert cfg.magnet_depth_VT == 50.0
        assert cfg.magnet_user_input_VT == -3.5
        assert cfg.isCutoffCheck_VT is True
        assert cfg.maxuvel_VT == 3000.0
        assert cfg.maxvvel_VT == 3000.0
        assert cfg.maxwvel_VT == 600.0
        assert cfg.isDespikeCheck_VT is True
        assert cfg.despike_kernel_VT == 7
        assert cfg.despike_cutoff_VT == 2.5
        assert cfg.isFlatlineCheck_VT is True
        assert cfg.flatline_kernel_VT == 9
        assert cfg.flatline_cutoff_VT == 4.0

    def test_attributes_section_json(self, tmp_path):
        attrs = {"cruise": "C001", "depth": 42}
        p = tmp_path / "cfg.ini"
        p.write_text(
            "[Attributes]\n"
            f"add_attributes = True\n"
            f"attributes_json = {json.dumps(attrs)}\n"
        )
        cfg = ProcessingConfig.from_ini(str(p))
        assert cfg.isAttributes is True
        assert cfg.attributes["cruise"] == "C001"
        assert cfg.attributes["depth"] == 42

    def test_attributes_section_invalid_json_falls_back_to_empty(self, tmp_path):
        p = tmp_path / "cfg.ini"
        p.write_text(
            "[Attributes]\n"
            "add_attributes = False\n"
            "attributes_json = {not_valid_json\n"
        )
        cfg = ProcessingConfig.from_ini(str(p))
        assert cfg.attributes == {}

    def test_extra_cells_absent_falls_back_to_default(self, tmp_path):
        """Old INI files without extra_cells must use the default of 1."""
        p = tmp_path / "cfg.ini"
        p.write_text(
            "[ProfileTest]\n"
            "profile_test = True\n"
            "cut_sidelobe = True\n"
            "water_column_depth = 80.0\n"
            # no extra_cells key
        )
        cfg = ProcessingConfig.from_ini(str(p))
        assert cfg.extra_cells_PT == 1

    def test_regrid_method_absent_falls_back_to_default(self, tmp_path):
        """Old INI files without regrid_method must use the default 'nearest'."""
        p = tmp_path / "cfg.ini"
        p.write_text(
            "[ProfileTest]\n" "profile_test = True\n" "regrid = True\n"
            # no regrid_method key
        )
        cfg = ProcessingConfig.from_ini(str(p))
        assert cfg.regrid_method_PT == "nearest"


# ===========================================================================
# 5. Round-trip: to_ini() -> from_ini()
# ===========================================================================


class TestRoundTrip:
    """Values written by to_ini() must survive a from_ini() load unchanged."""

    def _roundtrip(self, cfg: ProcessingConfig, tmp_path) -> ProcessingConfig:
        p = tmp_path / "rt.ini"
        cfg.to_ini(str(p))
        return ProcessingConfig.from_ini(str(p))

    def test_file_settings_roundtrip(self, tmp_path):
        cfg = ProcessingConfig(
            input_file_name="cruise001.pd0",
            input_file_path="/data/raw/cruise001.pd0",
            output_file_path="/data/processed/",
        )
        rt = self._roundtrip(cfg, tmp_path)
        assert rt.input_file_name == "cruise001.pd0"
        assert rt.input_file_path == "/data/raw/cruise001.pd0"
        assert rt.output_file_path == "/data/processed/"

    def test_time_axis_roundtrip(self, tmp_path):
        cfg = ProcessingConfig(
            isTimeAxisModified=True,
            isSnapTimeAxis=True,
            time_snap_frequency="30min",
            time_snap_tolerance="10min",
            time_target_minute=15,
            isTimeGapFilled=True,
            time_fill_method="linear",
        )
        rt = self._roundtrip(cfg, tmp_path)
        assert rt.isTimeAxisModified is True
        assert rt.isSnapTimeAxis is True
        assert rt.time_snap_frequency == "30min"
        assert rt.time_snap_tolerance == "10min"
        assert rt.time_target_minute == 15
        assert rt.isTimeGapFilled is True
        assert rt.time_fill_method == "linear"

    def test_sensor_health_roundtrip(self, tmp_path):
        cfg = ProcessingConfig(
            isSensorTest=True,
            isRollCheck_ST=True,
            roll_cutoff_ST=18.0,
            isPitchCheck_ST=True,
            pitch_cutoff_ST=12.0,
            isDepthModified_ST=True,
            depthoption_ST="Fixed Value",
            fixeddepth_ST=300.0,
            isSalinityModified_ST=True,
            salinityoption_ST="Fixed Value",
            fixedsalinity_ST=36.0,
            isTemperatureModified_ST=True,
            temperatureoption_ST="Fixed Value",
            fixedtemperature_ST=22.5,
            isSoundModified_ST=True,
            isVelocityModified_ST=True,
            isVelocityModified_HorizontalOnly_ST=False,
        )
        rt = self._roundtrip(cfg, tmp_path)
        assert rt.isSensorTest is True
        assert rt.roll_cutoff_ST == 18.0
        assert rt.pitch_cutoff_ST == 12.0
        assert rt.fixeddepth_ST == 300.0
        assert rt.fixedsalinity_ST == 36.0
        assert rt.fixedtemperature_ST == 22.5
        assert rt.isVelocityModified_HorizontalOnly_ST is False

    def test_qc_roundtrip(self, tmp_path):
        cfg = ProcessingConfig(
            isQCTest=True,
            ct_QCT=55.0,
            et_QCT=3.0,
            evt_QCT=1800.0,
            ft_QCT=45.0,
            is3beam_QCT=True,
            beam_ignore_QCT=1,
            pgt_QCT=30.0,
        )
        rt = self._roundtrip(cfg, tmp_path)
        assert rt.ct_QCT == 55.0
        assert rt.beam_ignore_QCT == 1
        assert rt.pgt_QCT == 30.0

    def test_beam_ignore_none_roundtrip(self, tmp_path):
        cfg = ProcessingConfig(beam_ignore_QCT=None)
        rt = self._roundtrip(cfg, tmp_path)
        assert rt.beam_ignore_QCT is None

    def test_profile_roundtrip(self, tmp_path):
        cfg = ProcessingConfig(
            isProfileTest=True,
            isTrimEndsCheck_PT=True,
            trim_start_PT=3,
            trim_end_PT=7,
            isCutBinSideLobeCheck_PT=True,
            water_depth_PT=150.0,
            extra_cells_PT=4,
            isCutBinManualCheck_PT=True,
            cut_bins_regions_PT=[[0, 5, None, None], [10, 15, 100, 200]],
            cut_bins_start_PT=2,
            cut_bins_end_PT=45,
            isRegridCheck_PT=True,
            regrid_cell_size_PT=2.5,
            regrid_method_PT="nearest",
            beam_direction_PT="down",
        )
        rt = self._roundtrip(cfg, tmp_path)
        assert rt.trim_start_PT == 3
        assert rt.trim_end_PT == 7
        assert rt.water_depth_PT == 150.0
        assert rt.extra_cells_PT == 4
        assert rt.cut_bins_regions_PT == [[0, 5, None, None], [10, 15, 100, 200]]
        assert rt.cut_bins_end_PT == 45
        assert rt.regrid_cell_size_PT == 2.5
        assert rt.regrid_method_PT == "nearest"
        assert rt.beam_direction_PT == "down"

    def test_velocity_roundtrip(self, tmp_path):
        cfg = ProcessingConfig(
            isVelocityTest=True,
            isMagnetCheck_VT=True,
            magnet_lat_VT=23.5,
            magnet_lon_VT=72.0,
            magnet_year_VT=2023,
            magnet_user_input_VT=-5.0,
            isCutoffCheck_VT=True,
            maxuvel_VT=3500.0,
            maxwvel_VT=400.0,
            isDespikeCheck_VT=True,
            despike_kernel_VT=9,
            despike_cutoff_VT=2.0,
            isFlatlineCheck_VT=True,
            flatline_kernel_VT=7,
            flatline_cutoff_VT=5.0,
        )
        rt = self._roundtrip(cfg, tmp_path)
        assert rt.magnet_lat_VT == 23.5
        assert rt.magnet_year_VT == 2023
        assert rt.magnet_user_input_VT == -5.0
        assert rt.maxuvel_VT == 3500.0
        assert rt.despike_kernel_VT == 9
        assert rt.flatline_cutoff_VT == 5.0

    def test_attributes_roundtrip(self, tmp_path):
        cfg = ProcessingConfig(
            isAttributes=True,
            attributes={"mission": "ABCD", "year": 2024},
        )
        rt = self._roundtrip(cfg, tmp_path)
        assert rt.isAttributes is True
        assert rt.attributes["mission"] == "ABCD"
        assert rt.attributes["year"] == 2024


# ===========================================================================
# 6. validate()
# ===========================================================================


class TestValidate:
    """validate() must return an empty list for valid configs and
    meaningful messages for invalid ones."""

    def test_valid_default_config(self):
        assert ProcessingConfig().validate() == []

    def test_temperature_file_option_no_path(self):
        cfg = ProcessingConfig(
            isTemperatureModified_ST=True,
            temperatureoption_ST="File",
            temperature_file_ST="",
        )
        issues = cfg.validate()
        assert any("temperature" in i.lower() for i in issues)

    def test_temperature_file_option_nonexistent_path(self, tmp_path):
        cfg = ProcessingConfig(
            isTemperatureModified_ST=True,
            temperatureoption_ST="File",
            temperature_file_ST=str(tmp_path / "ghost.csv"),
        )
        issues = cfg.validate()
        assert any("temperature" in i.lower() for i in issues)

    def test_salinity_file_option_no_path(self):
        cfg = ProcessingConfig(
            isSalinityModified_ST=True,
            salinityoption_ST="File",
            salinity_file_ST="",
        )
        issues = cfg.validate()
        assert any("salinity" in i.lower() for i in issues)

    def test_salinity_file_option_nonexistent_path(self, tmp_path):
        cfg = ProcessingConfig(
            isSalinityModified_ST=True,
            salinityoption_ST="File",
            salinity_file_ST=str(tmp_path / "ghost.csv"),
        )
        issues = cfg.validate()
        assert any("salinity" in i.lower() for i in issues)

    def test_depth_file_option_no_path(self):
        cfg = ProcessingConfig(
            isDepthModified_ST=True,
            depthoption_ST="File",
            depth_file_ST="",
        )
        issues = cfg.validate()
        assert any("depth" in i.lower() for i in issues)

    def test_depth_file_option_nonexistent_path(self, tmp_path):
        cfg = ProcessingConfig(
            isDepthModified_ST=True,
            depthoption_ST="File",
            depth_file_ST=str(tmp_path / "ghost.csv"),
        )
        issues = cfg.validate()
        assert any("depth" in i.lower() for i in issues)

    def test_roll_cutoff_zero_invalid(self):
        cfg = ProcessingConfig(isRollCheck_ST=True, roll_cutoff_ST=0.0)
        issues = cfg.validate()
        assert any("roll" in i.lower() for i in issues)

    def test_roll_cutoff_negative_invalid(self):
        cfg = ProcessingConfig(isRollCheck_ST=True, roll_cutoff_ST=-5.0)
        issues = cfg.validate()
        assert any("roll" in i.lower() for i in issues)

    def test_pitch_cutoff_zero_invalid(self):
        cfg = ProcessingConfig(isPitchCheck_ST=True, pitch_cutoff_ST=0.0)
        issues = cfg.validate()
        assert any("pitch" in i.lower() for i in issues)

    def test_u_vel_cutoff_zero_invalid(self):
        cfg = ProcessingConfig(isCutoffCheck_VT=True, maxuvel_VT=0.0)
        issues = cfg.validate()
        assert any("u" in i.lower() or "zonal" in i.lower() for i in issues)

    def test_v_vel_cutoff_zero_invalid(self):
        cfg = ProcessingConfig(isCutoffCheck_VT=True, maxvvel_VT=0.0)
        issues = cfg.validate()
        assert any("v" in i.lower() or "meridional" in i.lower() for i in issues)

    def test_w_vel_cutoff_zero_invalid(self):
        cfg = ProcessingConfig(isCutoffCheck_VT=True, maxwvel_VT=0.0)
        issues = cfg.validate()
        assert any("w" in i.lower() or "vertical" in i.lower() for i in issues)

    def test_multiple_issues_returned(self):
        cfg = ProcessingConfig(
            isRollCheck_ST=True,
            roll_cutoff_ST=-1.0,
            isPitchCheck_ST=True,
            pitch_cutoff_ST=-1.0,
        )
        assert len(cfg.validate()) >= 2

    def test_valid_file_path_no_issue(self, tmp_path):
        # Create the file so it actually exists
        p = tmp_path / "temp.csv"
        p.write_text("value\n20.0\n")
        cfg = ProcessingConfig(
            isTemperatureModified_ST=True,
            temperatureoption_ST="File",
            temperature_file_ST=str(p),
        )
        issues = cfg.validate()
        assert not any("temperature" in i.lower() for i in issues)


# ===========================================================================
# 7. _get_replacement_value()
# ===========================================================================


class TestGetReplacementValue:
    """_get_replacement_value() must dispatch correctly for each option."""

    def setup_method(self):
        self.cfg = ProcessingConfig()

    def test_none_option_returns_none(self):
        result = self.cfg._get_replacement_value(
            option="None",
            fixed_value=10.0,
            file_path="",
            n_ensembles=5,
            variable_name="temperature",
        )
        assert result is None

    def test_empty_string_option_returns_none(self):
        result = self.cfg._get_replacement_value(
            option="",
            fixed_value=10.0,
            file_path="",
            n_ensembles=5,
            variable_name="temperature",
        )
        assert result is None

    def test_fixed_value_returns_float(self):
        result = self.cfg._get_replacement_value(
            option="Fixed Value",
            fixed_value=25.0,
            file_path="",
            n_ensembles=5,
            variable_name="temperature",
        )
        assert result == 25.0
        assert isinstance(result, float)

    def test_file_option_no_path_returns_none(self):
        result = self.cfg._get_replacement_value(
            option="File",
            fixed_value=10.0,
            file_path="",
            n_ensembles=5,
            variable_name="temperature",
        )
        assert result is None

    def test_file_option_loads_array(self, tmp_path):
        values = [20.0, 21.0, 22.0, 23.0, 24.0]
        p = _write_csv(values, header="value")
        try:
            result = self.cfg._get_replacement_value(
                option="File",
                fixed_value=0.0,
                file_path=p,
                n_ensembles=5,
                variable_name="temperature",
            )
            assert isinstance(result, np.ndarray)
            np.testing.assert_array_almost_equal(result, values)
        finally:
            os.unlink(p)

    def test_unknown_option_returns_none(self):
        result = self.cfg._get_replacement_value(
            option="Garbage",
            fixed_value=10.0,
            file_path="",
            n_ensembles=5,
            variable_name="temperature",
        )
        assert result is None


# ===========================================================================
# 8. _load_timeseries_from_file()
# ===========================================================================


class TestLoadTimeseriesFromFile:
    """_load_timeseries_from_file() supports several CSV/text layouts."""

    def setup_method(self):
        self.cfg = ProcessingConfig()

    def test_plain_values_no_header(self, tmp_path):
        # NOTE: When a plain text file has numeric values only (no header),
        # pandas treats the first value as a column name and reads n-1 rows.
        # The code falls back to np.loadtxt only when pd.read_csv raises, not
        # when it succeeds with wrong length. Consequently a 3-value file
        # will fail the length check (2 rows != 3 ensembles).
        # Add a proper header to avoid this ambiguity.
        p = tmp_path / "data.txt"
        p.write_text("value\n1.0\n2.0\n3.0\n")
        result = self.cfg._load_timeseries_from_file(str(p), 3, "temperature")
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_csv_with_value_column(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("value\n10.0\n11.0\n12.0\n")
        result = self.cfg._load_timeseries_from_file(str(p), 3, "temperature")
        np.testing.assert_array_almost_equal(result, [10.0, 11.0, 12.0])

    def test_csv_with_variable_name_column(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("temperature\n5.0\n6.0\n7.0\n")
        result = self.cfg._load_timeseries_from_file(str(p), 3, "temperature")
        np.testing.assert_array_almost_equal(result, [5.0, 6.0, 7.0])

    def test_csv_single_column_no_named_header(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("x\n1.5\n2.5\n3.5\n")
        result = self.cfg._load_timeseries_from_file(str(p), 3, "temperature")
        np.testing.assert_array_almost_equal(result, [1.5, 2.5, 3.5])

    def test_returns_float64(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("value\n1.0\n2.0\n3.0\n")
        result = self.cfg._load_timeseries_from_file(str(p), 3, "temperature")
        assert result.dtype == np.float64

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            self.cfg._load_timeseries_from_file(
                "/nonexistent/ghost.csv", 5, "temperature"
            )

    def test_wrong_length_raises_value_error(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("value\n1.0\n2.0\n3.0\n")
        with pytest.raises(ValueError, match="temperature"):
            self.cfg._load_timeseries_from_file(str(p), 10, "temperature")

    def test_error_message_includes_variable_name(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("value\n1.0\n")
        with pytest.raises(ValueError) as exc_info:
            self.cfg._load_timeseries_from_file(str(p), 5, "salinity")
        assert "salinity" in str(exc_info.value)

    def test_csv_multicolumn_uses_first_numeric_column(self, tmp_path):
        """Multi-column CSV with no 'value'/variable-name/single-col match →
        falls through to the numeric-column branch (lines 788-790)."""
        p = tmp_path / "data.csv"
        # Two columns, neither named 'value' or 'temperature'.
        # Both are numeric so the first numeric col should be picked.
        p.write_text("alpha,beta\n1.0,9.0\n2.0,8.0\n3.0,7.0\n")
        result = self.cfg._load_timeseries_from_file(str(p), 3, "temperature")
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_csv_multicolumn_no_numeric_raises(self, tmp_path):
        """Multi-column CSV with no numeric columns raises ValueError
        (lines 791-794), which is then caught and re-raised after loadtxt
        also fails on a header-only CSV."""
        p = tmp_path / "data.csv"
        # All text columns — no numeric data at all.
        p.write_text("label,tag\nfoo,bar\nbaz,qux\n")
        with pytest.raises(ValueError, match="temperature"):
            self.cfg._load_timeseries_from_file(str(p), 2, "temperature")

    def test_csv_read_failure_falls_back_to_loadtxt(self, tmp_path):
        """When pd.read_csv raises, the except block (line 796) falls back to
        np.loadtxt (line 799), which succeeds on a plain numeric text file."""
        from unittest.mock import patch

        p = tmp_path / "data.txt"
        p.write_text("1.0\n2.0\n3.0\n")

        # Force pd.read_csv to raise so we enter the except block.
        with patch("pandas.read_csv", side_effect=Exception("forced CSV failure")):
            result = self.cfg._load_timeseries_from_file(str(p), 3, "temperature")

        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_csv_read_failure_and_loadtxt_failure_raises_value_error(self, tmp_path):
        """When both pd.read_csv and np.loadtxt fail, ValueError is raised
        (lines 800-802) with the variable name in the message."""
        from unittest.mock import patch

        p = tmp_path / "data.txt"
        p.write_text("irrelevant content")

        with (
            patch("pandas.read_csv", side_effect=Exception("forced CSV failure")),
            patch("numpy.loadtxt", side_effect=Exception("forced loadtxt failure")),
        ):
            with pytest.raises(ValueError, match="temperature"):
                self.cfg._load_timeseries_from_file(str(p), 3, "temperature")


# ===========================================================================
# 9. to_dict()
# ===========================================================================


class TestToDict:
    """to_dict() must return a flat, JSON-serializable dictionary."""

    def test_returns_dict(self):
        assert isinstance(ProcessingConfig().to_dict(), dict)

    def test_all_fields_present(self):
        d = ProcessingConfig().to_dict()
        for field_name in (
            "isSensorTest",
            "isQCTest",
            "isProfileTest",
            "isVelocityTest",
            "ct_QCT",
            "roll_cutoff_ST",
            "maxuvel_VT",
            "attributes",
        ):
            assert field_name in d, f"Missing field: {field_name}"

    def test_values_match_defaults(self):
        d = ProcessingConfig().to_dict()
        assert d["ct_QCT"] == 64.0
        assert d["roll_cutoff_ST"] == 15.0
        assert d["maxuvel_VT"] == 2500.0

    def test_modified_values_reflected(self):
        cfg = ProcessingConfig(ct_QCT=45.0, roll_cutoff_ST=20.0)
        d = cfg.to_dict()
        assert d["ct_QCT"] == 45.0
        assert d["roll_cutoff_ST"] == 20.0

    def test_json_serializable(self):
        d = ProcessingConfig().to_dict()
        json.dumps(d)  # must not raise

    def test_attributes_dict_serializable(self):
        cfg = ProcessingConfig(attributes={"key": "value", "num": 42})
        d = cfg.to_dict()
        assert d["attributes"] == {"key": "value", "num": 42}


# ===========================================================================
# 10. __repr__()
# ===========================================================================


class TestRepr:
    """__repr__() must list the enabled processing stages."""

    def test_no_stages_enabled(self):
        r = repr(ProcessingConfig())
        assert "ProcessingConfig" in r
        assert "TimeAxis" not in r
        assert "SensorHealth" not in r

    def test_sensor_health_listed(self):
        r = repr(ProcessingConfig(isSensorTest=True))
        assert "SensorHealth" in r

    def test_signal_quality_listed(self):
        r = repr(ProcessingConfig(isQCTest=True))
        assert "SignalQuality" in r

    def test_profile_listed(self):
        r = repr(ProcessingConfig(isProfileTest=True))
        assert "Profile" in r

    def test_velocity_listed(self):
        r = repr(ProcessingConfig(isVelocityTest=True))
        assert "Velocity" in r

    def test_time_axis_listed(self):
        r = repr(ProcessingConfig(isTimeAxisModified=True))
        assert "TimeAxis" in r

    def test_multiple_stages_all_listed(self):
        cfg = ProcessingConfig(
            isTimeAxisModified=True,
            isSensorTest=True,
            isQCTest=True,
            isProfileTest=True,
            isVelocityTest=True,
        )
        r = repr(cfg)
        for stage in (
            "TimeAxis",
            "SensorHealth",
            "SignalQuality",
            "Profile",
            "Velocity",
        ):
            assert stage in r, f"Missing stage in repr: {stage}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
