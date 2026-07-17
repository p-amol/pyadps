"""
Configuration management for ADCP processing.
Bridges Streamlit-generated INI files and programmatic processing.

This module provides:
- ProcessingConfig: Dataclass for all processing parameters
- INI file parsing and generation for Streamlit integration
- In-memory INI serialisation for Streamlit download buttons

Processing is handled entirely by ProcessedDataset (core.py).
Pass a ProcessingConfig object or an INI file path to
``ProcessedDataset.apply_config()`` to run the pipeline.

Data Replacement Options:
- Temperature, salinity, and depth can be replaced using either:
  1. Fixed Value: A constant value applied to all ensembles
  2. File Path: A CSV/text file containing time series data
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal, Union
import configparser
import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Type alias for data replacement options
DataReplacementOption = Literal["None", "Fixed Value", "File"]


@dataclass
class ProcessingConfig:
    """
    Programmatic configuration object for ADCP processing.

    Can be created from INI files (Streamlit) or directly in Python.
    Provides a bridge between GUI-based configuration and programmatic
    processing via ``ProcessedDataset.apply_config()``.

    Processing is intentionally kept out of this class — it is a pure
    data container.  All execution is delegated to ``ProcessedDataset``
    in ``core.py``, which is the single user-facing entry point.

    Data Replacement:
    -----------------
    Temperature, salinity, and transducer depth can be replaced using
    two methods:

    - ``"Fixed Value"``: Apply a constant value to all ensembles.
    - ``"File"``: Load a time series from a CSV file.

    Select the method via the ``*option_ST`` attributes:

    - ``"None"``: No replacement (use dataset values).
    - ``"Fixed Value"``: Use the corresponding ``fixed*_ST`` value.
    - ``"File"``: Load from the corresponding ``*_file_ST`` path.

    Attributes are organised by processing stage:

    1. File Settings
    2. Time Fixes
    3. Sensor Test (Health)
    4. QC Test (Signal Quality)
    5. Profile Test
    6. Velocity Test
    7. Output Attributes

    Examples
    --------
    Load from an INI file and apply to a dataset::

        import pyadps
        from pyadps.processing import ProcessedDataset

        ds = pyadps.read('file.000')
        proc = ProcessedDataset(ds)
        result = proc.apply_config('config.ini').finalize()

    Build programmatically and apply::

        config = ProcessingConfig(
            isSensorTest=True,
            isRollCheck_ST=True,
            roll_cutoff_ST=15.0,
            isQCTest=True,
            ct_QCT=64.0,
        )
        result = proc.apply_config(config).finalize()

    Use a fixed replacement value for temperature::

        config = ProcessingConfig(
            isSensorTest=True,
            isTemperatureModified_ST=True,
            temperatureoption_ST="Fixed Value",
            fixedtemperature_ST=15.0,
        )

    Load replacement data from a CSV file::

        config = ProcessingConfig(
            isSensorTest=True,
            isTemperatureModified_ST=True,
            temperatureoption_ST="File",
            temperature_file_ST="/path/to/ctd_temperature.csv",
        )

    Save to disk and also obtain the content as a string (e.g. for a
    Streamlit download button)::

        config.to_ini('config.ini')

        import streamlit as st
        st.download_button(
            label="Download config.ini",
            data=config.to_ini_string(),
            file_name="config.ini",
            mime="text/plain",
        )

    Full end-to-end workflow using ``ProcessedDataset``::

        proc = ProcessedDataset.from_file('config.ini')
        proc.apply_config('config.ini')
        output = proc.save_netcdf('config.ini', output_dir='processed/')
    """

    # ========================
    # FILE SETTINGS
    # ========================
    input_file_name: str = ""
    input_file_path: str = ""
    output_file_path: str = ""

    # ========================
    # TIME FIXES
    # ========================
    isTimeAxisModified: bool = False
    isSnapTimeAxis: bool = False
    time_snap_frequency: str = "h"
    time_snap_tolerance: str = "5min"  # full pandas offset string, e.g. "5min", "2h"
    time_target_minute: int = 0
    isTimeGapFilled: bool = False
    time_fill_method: str = "auto"

    # ========================
    # SENSOR TEST (HEALTH)
    # ========================
    isSensorTest: bool = False

    # ----- Transducer Depth Replacement -----
    isDepthModified_ST: bool = False
    depthoption_ST: str = "None"  # "None", "Fixed Value", or "File"
    # Fixed value option (in decimeters to match RDI internal format)
    fixeddepth_ST: float = 0.0
    # File option
    depth_file_ST: str = ""

    # ----- Salinity Replacement -----
    isSalinityModified_ST: bool = False
    salinityoption_ST: str = "None"  # "None", "Fixed Value", or "File"
    # Fixed value option (in PSU)
    fixedsalinity_ST: float = 35.0
    # File option
    salinity_file_ST: str = ""

    # ----- Temperature Replacement -----
    isTemperatureModified_ST: bool = False
    temperatureoption_ST: str = "None"  # "None", "Fixed Value", or "File"
    # Fixed value option (in degrees Celsius)
    fixedtemperature_ST: float = 15.0
    # File option
    temperature_file_ST: str = ""

    # ----- Tilt Checks -----
    isRollCheck_ST: bool = False
    isPitchCheck_ST: bool = False
    roll_cutoff_ST: float = 15.0
    pitch_cutoff_ST: float = 15.0

    # ----- Sound Speed Correction -----
    isSoundModified_ST: bool = False
    isVelocityModified_ST: bool = True
    isVelocityModified_HorizontalOnly_ST: bool = True

    # ========================
    # QC TEST (SIGNAL QUALITY)
    # ========================
    isQCTest: bool = False
    ct_QCT: float = 64.0  # Correlation threshold
    et_QCT: float = 0.0  # Echo intensity threshold
    evt_QCT: float = 2000.0  # Error velocity threshold (mm/s)
    ft_QCT: float = 50.0  # False target threshold
    is3beam_QCT: bool = False
    beam_ignore_QCT: Optional[int] = None  # Beam to ignore in 3-beam mode
    pgt_QCT: float = 0.0  # Percent good threshold

    # ========================
    # PROFILE TEST
    # ========================
    isProfileTest: bool = False
    isTrimEndsCheck_PT: bool = False
    trim_start_PT: int = 0
    trim_end_PT: int = 0
    isCutBinSideLobeCheck_PT: bool = False
    water_depth_PT: float = 0.0
    extra_cells_PT: int = 1  # extra cells masked beyond side-lobe boundary
    isCutBinManualCheck_PT: bool = False
    # All manual cut regions as a list of [min_cell, max_cell, min_ens, max_ens].
    # None entries are stored as -1 in the INI and restored on load.
    cut_bins_regions_PT: list = field(
        default_factory=list
    )  # list of [min_cell, max_cell, min_ens, max_ens], None→-1 in INI
    # Legacy single-region fields kept for Streamlit backward-compatibility.
    cut_bins_start_PT: int = 0
    cut_bins_end_PT: int = 0
    isRegridCheck_PT: bool = False
    regrid_cell_size_PT: float = 1.0
    regrid_method_PT: str = "nearest"  # interpolation method passed to runner.regrid()
    regrid_end_cell_option_PT: str = "cell"  # "cell", "surface", or "manual"
    regrid_boundary_limit_PT: float = 0.0  # used when end_cell_option="manual"
    beam_direction_PT: str = "up"

    # ========================
    # VELOCITY TEST
    # ========================
    isVelocityTest: bool = False

    # Magnetic declination
    isMagnetCheck_VT: bool = False
    magnet_method_VT: str = "api"
    magnet_lat_VT: float = 0.0
    magnet_lon_VT: float = 0.0
    magnet_year_VT: int = 2025
    magnet_depth_VT: float = 0.0
    magnet_user_input_VT: float = 0.0

    # Velocity cutoff
    isCutoffCheck_VT: bool = False
    maxuvel_VT: float = 2500.0  # mm/s (RDI internal units)
    maxvvel_VT: float = 2500.0  # mm/s
    maxwvel_VT: float = 500.0  # mm/s

    # Despike
    isDespikeCheck_VT: bool = False
    despike_kernel_VT: int = 5
    despike_cutoff_VT: float = 3.0

    # Flatline detection
    isFlatlineCheck_VT: bool = False
    flatline_kernel_VT: int = 5
    flatline_cutoff_VT: float = 3.0

    # ========================
    # OUTPUT ATTRIBUTES
    # ========================
    isAttributes: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_ini(cls, filepath: str) -> "ProcessingConfig":
        """
        Load configuration from INI file (generated by Streamlit).

        Parameters
        ----------
        filepath : str
            Path to config.ini file

        Returns
        -------
        ProcessingConfig
            Configuration object with all parameters loaded

        Raises
        ------
        FileNotFoundError
            If the INI file does not exist
        """
        config = configparser.ConfigParser()
        config.read(filepath)

        kwargs: Dict[str, Any] = {}

        # ========================
        # FILE SETTINGS
        # ========================
        if "FileSettings" in config:
            kwargs["input_file_name"] = config.get(
                "FileSettings", "input_file_name", fallback=""
            )
            kwargs["input_file_path"] = config.get(
                "FileSettings", "input_file_path", fallback=""
            )
            kwargs["output_file_path"] = config.get(
                "FileSettings", "output_file_path", fallback=""
            )

        # ========================
        # TIME SETTINGS
        # ========================
        if "FixTime" in config:
            kwargs["isTimeAxisModified"] = config.getboolean(
                "FixTime", "is_time_modified", fallback=False
            )
            kwargs["isSnapTimeAxis"] = config.getboolean(
                "FixTime", "is_snap_time_axis", fallback=False
            )
            kwargs["time_snap_frequency"] = config.get(
                "FixTime", "time_snap_frequency", fallback="h"
            )
            kwargs["time_snap_tolerance"] = config.get(
                "FixTime", "time_snap_tolerance", fallback="5min"
            )
            kwargs["time_target_minute"] = config.getint(
                "FixTime", "time_target_minute", fallback=0
            )
            kwargs["isTimeGapFilled"] = config.getboolean(
                "FixTime", "is_time_gap_filled", fallback=False
            )
            kwargs["time_fill_method"] = config.get(
                "FixTime", "time_fill_method", fallback="auto"
            )

        # ========================
        # SENSOR TEST
        # ========================
        if "SensorTest" in config:
            kwargs["isSensorTest"] = config.getboolean(
                "SensorTest", "sensor_test", fallback=False
            )

            # ----- Transducer Depth -----
            kwargs["isDepthModified_ST"] = config.getboolean(
                "SensorTest", "depth_modified", fallback=False
            )
            kwargs["depthoption_ST"] = config.get(
                "SensorTest", "depth_option", fallback="None"
            )
            kwargs["fixeddepth_ST"] = config.getfloat(
                "SensorTest", "fixed_depth", fallback=0.0
            )
            kwargs["depth_file_ST"] = config.get(
                "SensorTest", "depth_file", fallback=""
            )

            # ----- Salinity -----
            kwargs["isSalinityModified_ST"] = config.getboolean(
                "SensorTest", "salinity_modified", fallback=False
            )
            kwargs["salinityoption_ST"] = config.get(
                "SensorTest", "salinity_option", fallback="None"
            )
            kwargs["fixedsalinity_ST"] = config.getfloat(
                "SensorTest", "fixed_salinity", fallback=35.0
            )
            kwargs["salinity_file_ST"] = config.get(
                "SensorTest", "salinity_file", fallback=""
            )

            # ----- Temperature -----
            kwargs["isTemperatureModified_ST"] = config.getboolean(
                "SensorTest", "temperature_modified", fallback=False
            )
            kwargs["temperatureoption_ST"] = config.get(
                "SensorTest", "temperature_option", fallback="None"
            )
            kwargs["fixedtemperature_ST"] = config.getfloat(
                "SensorTest", "fixed_temperature", fallback=15.0
            )
            kwargs["temperature_file_ST"] = config.get(
                "SensorTest", "temperature_file", fallback=""
            )

            # ----- Tilt Checks -----
            kwargs["isRollCheck_ST"] = config.getboolean(
                "SensorTest", "roll_check", fallback=False
            )
            kwargs["isPitchCheck_ST"] = config.getboolean(
                "SensorTest", "pitch_check", fallback=False
            )
            kwargs["roll_cutoff_ST"] = config.getfloat(
                "SensorTest", "roll_cutoff", fallback=15.0
            )
            kwargs["pitch_cutoff_ST"] = config.getfloat(
                "SensorTest", "pitch_cutoff", fallback=15.0
            )

            # ----- Sound Speed -----
            kwargs["isSoundModified_ST"] = config.getboolean(
                "SensorTest", "sound_speed_correction", fallback=False
            )
            kwargs["isVelocityModified_ST"] = config.getboolean(
                "SensorTest", "velocity_correction", fallback=True
            )
            kwargs["isVelocityModified_HorizontalOnly_ST"] = config.getboolean(
                "SensorTest", "horizontal_only", fallback=True
            )

        # ========================
        # QC TEST
        # ========================
        if "QCTest" in config:
            kwargs["isQCTest"] = config.getboolean("QCTest", "qc_test", fallback=False)
            kwargs["ct_QCT"] = config.getfloat("QCTest", "correlation", fallback=64.0)
            kwargs["et_QCT"] = config.getfloat("QCTest", "echo_intensity", fallback=0.0)
            kwargs["evt_QCT"] = config.getfloat(
                "QCTest", "error_velocity", fallback=2000.0
            )
            kwargs["ft_QCT"] = config.getfloat("QCTest", "false_target", fallback=50.0)
            kwargs["is3beam_QCT"] = config.getboolean(
                "QCTest", "three_beam", fallback=False
            )
            beam_ignore = config.get("QCTest", "beam_ignore", fallback="")
            kwargs["beam_ignore_QCT"] = int(beam_ignore) if beam_ignore else None
            kwargs["pgt_QCT"] = config.getfloat("QCTest", "percent_good", fallback=0.0)

        # ========================
        # PROFILE TEST
        # ========================
        if "ProfileTest" in config:
            kwargs["isProfileTest"] = config.getboolean(
                "ProfileTest", "profile_test", fallback=False
            )
            kwargs["isTrimEndsCheck_PT"] = config.getboolean(
                "ProfileTest", "trim_ends", fallback=False
            )
            kwargs["trim_start_PT"] = config.getint(
                "ProfileTest", "trim_start", fallback=0
            )
            kwargs["trim_end_PT"] = config.getint("ProfileTest", "trim_end", fallback=0)
            kwargs["isCutBinSideLobeCheck_PT"] = config.getboolean(
                "ProfileTest", "cut_sidelobe", fallback=False
            )
            kwargs["water_depth_PT"] = config.getfloat(
                "ProfileTest", "water_column_depth", fallback=0.0
            )
            kwargs["extra_cells_PT"] = config.getint(
                "ProfileTest", "extra_cells", fallback=1
            )
            kwargs["isCutBinManualCheck_PT"] = config.getboolean(
                "ProfileTest", "cut_bins_manual", fallback=False
            )
            # Multi-region JSON list (new format); -1 entries decoded back to None.
            raw_regions = config.get("ProfileTest", "cut_bins_regions", fallback="[]")
            try:
                decoded = json.loads(raw_regions)
                kwargs["cut_bins_regions_PT"] = [
                    [None if v == -1 else v for v in region] for region in decoded
                ]
            except (json.JSONDecodeError, TypeError):
                kwargs["cut_bins_regions_PT"] = []
            # Legacy single-region fields.
            kwargs["cut_bins_start_PT"] = config.getint(
                "ProfileTest", "cut_bins_start", fallback=0
            )
            kwargs["cut_bins_end_PT"] = config.getint(
                "ProfileTest", "cut_bins_end", fallback=0
            )
            kwargs["isRegridCheck_PT"] = config.getboolean(
                "ProfileTest", "regrid", fallback=False
            )
            kwargs["regrid_cell_size_PT"] = config.getfloat(
                "ProfileTest", "regrid_cell_size", fallback=1.0
            )
            kwargs["regrid_method_PT"] = config.get(
                "ProfileTest", "regrid_method", fallback="nearest"
            )
            kwargs["regrid_end_cell_option_PT"] = config.get(
                "ProfileTest", "regrid_end_cell_option", fallback="cell"
            )
            kwargs["regrid_boundary_limit_PT"] = config.getfloat(
                "ProfileTest", "regrid_boundary_limit", fallback=0.0
            )
            kwargs["beam_direction_PT"] = config.get(
                "ProfileTest", "beam_direction", fallback="up"
            )

        # ========================
        # VELOCITY TEST
        # ========================
        if "VelocityTest" in config:
            kwargs["isVelocityTest"] = config.getboolean(
                "VelocityTest", "velocity_test", fallback=False
            )

            # Magnetic declination
            kwargs["isMagnetCheck_VT"] = config.getboolean(
                "VelocityTest", "magnetic_declination", fallback=False
            )
            kwargs["magnet_method_VT"] = config.get(
                "VelocityTest", "magnet_method", fallback="api"
            )
            kwargs["magnet_lat_VT"] = config.getfloat(
                "VelocityTest", "latitude", fallback=0.0
            )
            kwargs["magnet_lon_VT"] = config.getfloat(
                "VelocityTest", "longitude", fallback=0.0
            )
            kwargs["magnet_year_VT"] = config.getint(
                "VelocityTest", "year", fallback=2025
            )
            kwargs["magnet_depth_VT"] = config.getfloat(
                "VelocityTest", "magnet_depth", fallback=0.0
            )
            kwargs["magnet_user_input_VT"] = config.getfloat(
                "VelocityTest", "magnet_user_input", fallback=0.0
            )

            # Velocity cutoff
            kwargs["isCutoffCheck_VT"] = config.getboolean(
                "VelocityTest", "velocity_cutoff", fallback=False
            )
            kwargs["maxuvel_VT"] = config.getfloat(
                "VelocityTest", "max_zonal_velocity", fallback=2500.0
            )
            kwargs["maxvvel_VT"] = config.getfloat(
                "VelocityTest", "max_meridional_velocity", fallback=2500.0
            )
            kwargs["maxwvel_VT"] = config.getfloat(
                "VelocityTest", "max_vertical_velocity", fallback=500.0
            )

            # Despike
            kwargs["isDespikeCheck_VT"] = config.getboolean(
                "VelocityTest", "despike", fallback=False
            )
            kwargs["despike_kernel_VT"] = config.getint(
                "VelocityTest", "despike_kernel_size", fallback=5
            )
            kwargs["despike_cutoff_VT"] = config.getfloat(
                "VelocityTest", "despike_cutoff", fallback=3.0
            )

            # Flatline
            kwargs["isFlatlineCheck_VT"] = config.getboolean(
                "VelocityTest", "flatline", fallback=False
            )
            kwargs["flatline_kernel_VT"] = config.getint(
                "VelocityTest", "flatline_kernel_size", fallback=5
            )
            kwargs["flatline_cutoff_VT"] = config.getfloat(
                "VelocityTest", "flatline_cutoff", fallback=3.0
            )

        # ========================
        # ATTRIBUTES
        # ========================
        if "Attributes" in config:
            kwargs["isAttributes"] = config.getboolean(
                "Attributes", "add_attributes", fallback=False
            )
            # Parse JSON attributes if present
            attrs_json = config.get("Attributes", "attributes_json", fallback="{}")
            try:
                kwargs["attributes"] = json.loads(attrs_json)
            except json.JSONDecodeError:
                kwargs["attributes"] = {}

        return cls(**kwargs)

    def _build_configparser(self) -> configparser.ConfigParser:
        """
        Build a ConfigParser object from the current configuration.

        Single source of truth for INI serialisation, shared by both
        ``to_ini()`` (file output) and ``to_ini_string()`` (in-memory output)
        so the two methods can never drift apart.

        Returns
        -------
        configparser.ConfigParser
            Populated parser ready to be written.
        """
        config = configparser.ConfigParser()

        # ========================
        # FILE SETTINGS
        # ========================
        config["FileSettings"] = {
            "input_file_name": self.input_file_name,
            "input_file_path": self.input_file_path,
            "output_file_path": self.output_file_path,
        }

        # ========================
        # TIME SETTINGS
        # ========================
        config["FixTime"] = {
            "is_time_modified": str(self.isTimeAxisModified),
            "is_snap_time_axis": str(self.isSnapTimeAxis),
            "time_snap_frequency": self.time_snap_frequency,
            "time_snap_tolerance": str(self.time_snap_tolerance),
            "time_target_minute": str(self.time_target_minute),
            "is_time_gap_filled": str(self.isTimeGapFilled),
            "time_fill_method": self.time_fill_method,
        }

        # ========================
        # SENSOR TEST
        # ========================
        config["SensorTest"] = {
            "sensor_test": str(self.isSensorTest),
            # Transducer Depth
            "depth_modified": str(self.isDepthModified_ST),
            "depth_option": self.depthoption_ST,
            "fixed_depth": str(self.fixeddepth_ST),
            "depth_file": self.depth_file_ST,
            # Salinity
            "salinity_modified": str(self.isSalinityModified_ST),
            "salinity_option": self.salinityoption_ST,
            "fixed_salinity": str(self.fixedsalinity_ST),
            "salinity_file": self.salinity_file_ST,
            # Temperature
            "temperature_modified": str(self.isTemperatureModified_ST),
            "temperature_option": self.temperatureoption_ST,
            "fixed_temperature": str(self.fixedtemperature_ST),
            "temperature_file": self.temperature_file_ST,
            # Tilt
            "roll_check": str(self.isRollCheck_ST),
            "pitch_check": str(self.isPitchCheck_ST),
            "roll_cutoff": str(self.roll_cutoff_ST),
            "pitch_cutoff": str(self.pitch_cutoff_ST),
            # Sound speed
            "sound_speed_correction": str(self.isSoundModified_ST),
            "velocity_correction": str(self.isVelocityModified_ST),
            "horizontal_only": str(self.isVelocityModified_HorizontalOnly_ST),
        }

        # ========================
        # QC TEST
        # ========================
        config["QCTest"] = {
            "qc_test": str(self.isQCTest),
            "correlation": str(self.ct_QCT),
            "echo_intensity": str(self.et_QCT),
            "error_velocity": str(self.evt_QCT),
            "false_target": str(self.ft_QCT),
            "three_beam": str(self.is3beam_QCT),
            "beam_ignore": str(self.beam_ignore_QCT)
            if self.beam_ignore_QCT is not None
            else "",
            "percent_good": str(self.pgt_QCT),
        }

        # ========================
        # PROFILE TEST
        # ========================
        config["ProfileTest"] = {
            "profile_test": str(self.isProfileTest),
            "trim_ends": str(self.isTrimEndsCheck_PT),
            "trim_start": str(self.trim_start_PT),
            "trim_end": str(self.trim_end_PT),
            "cut_sidelobe": str(self.isCutBinSideLobeCheck_PT),
            "water_column_depth": str(self.water_depth_PT),
            "extra_cells": str(self.extra_cells_PT),
            "cut_bins_manual": str(self.isCutBinManualCheck_PT),
            # All regions serialised as JSON; None entries encoded as -1.
            "cut_bins_regions": json.dumps(
                [
                    [-1 if v is None else v for v in region]
                    for region in self.cut_bins_regions_PT
                ]
            ),
            # Legacy single-region fields (used by Streamlit UI).
            "cut_bins_start": str(self.cut_bins_start_PT),
            "cut_bins_end": str(self.cut_bins_end_PT),
            "regrid": str(self.isRegridCheck_PT),
            "regrid_cell_size": str(self.regrid_cell_size_PT),
            "regrid_method": self.regrid_method_PT,
            "regrid_end_cell_option": self.regrid_end_cell_option_PT,
            "regrid_boundary_limit": str(self.regrid_boundary_limit_PT),
            "beam_direction": self.beam_direction_PT,
        }

        # ========================
        # VELOCITY TEST
        # ========================
        config["VelocityTest"] = {
            "velocity_test": str(self.isVelocityTest),
            # Magnetic declination
            "magnetic_declination": str(self.isMagnetCheck_VT),
            "magnet_method": self.magnet_method_VT,
            "latitude": str(self.magnet_lat_VT),
            "longitude": str(self.magnet_lon_VT),
            "year": str(self.magnet_year_VT),
            "magnet_depth": str(self.magnet_depth_VT),
            "magnet_user_input": str(self.magnet_user_input_VT),
            # Velocity cutoff
            "velocity_cutoff": str(self.isCutoffCheck_VT),
            "max_zonal_velocity": str(self.maxuvel_VT),
            "max_meridional_velocity": str(self.maxvvel_VT),
            "max_vertical_velocity": str(self.maxwvel_VT),
            # Despike
            "despike": str(self.isDespikeCheck_VT),
            "despike_kernel_size": str(self.despike_kernel_VT),
            "despike_cutoff": str(self.despike_cutoff_VT),
            # Flatline
            "flatline": str(self.isFlatlineCheck_VT),
            "flatline_kernel_size": str(self.flatline_kernel_VT),
            "flatline_cutoff": str(self.flatline_cutoff_VT),
        }

        # ========================
        # ATTRIBUTES
        # ========================
        config["Attributes"] = {
            "add_attributes": str(self.isAttributes),
            "attributes_json": json.dumps(self.attributes),
        }

        return config

    def to_ini_string(self) -> str:
        """
        Serialize configuration to INI format as an in-memory string.

        Useful for Streamlit download buttons or any workflow that needs
        the INI content without writing to disk first.

        Returns
        -------
        str
            INI-formatted configuration string.

        Examples
        --------
        Streamlit download button::

            config = ProcessingConfig(isSensorTest=True, roll_cutoff_ST=15.0)
            st.download_button(
                label="Download config.ini",
                data=config.to_ini_string(),
                file_name="config.ini",
                mime="text/plain",
            )

        In-memory inspection::

            ini_text = config.to_ini_string()
            print(ini_text)
        """
        import io

        buffer = io.StringIO()
        self._build_configparser().write(buffer)
        return buffer.getvalue()

    def to_ini(self, filepath: str) -> None:
        """
        Save configuration to INI file.

        Parameters
        ----------
        filepath : str
            Path where to save config.ini
        """
        with open(filepath, "w") as f:
            f.write(self.to_ini_string())
        logger.info(f"Configuration saved to {filepath}")

    def _load_timeseries_from_file(
        self,
        filepath: str,
        n_ensembles: int,
        variable_name: str,
    ) -> np.ndarray:
        """
        Load time series data from a CSV file.

        Expected file format:
        - Single column of values (one per line), OR
        - CSV with 'value' column, OR
        - CSV with column matching variable_name

        Parameters
        ----------
        filepath : str
            Path to the data file
        n_ensembles : int
            Number of ensembles in the dataset (for validation)
        variable_name : str
            Name of the variable being loaded (for error messages)

        Returns
        -------
        np.ndarray
            1D array of values with shape (n_ensembles,)

        Raises
        ------
        FileNotFoundError
            If the file does not exist
        ValueError
            If the file format is invalid or data length doesn't match
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(
                f"Data file not found for {variable_name}: {filepath}"
            )

        # Try to load as CSV
        try:
            import pandas as pd

            df = pd.read_csv(filepath)

            # Look for the data column
            if "value" in df.columns:
                data = df["value"].values
            elif variable_name in df.columns:
                data = df[variable_name].values
            elif len(df.columns) == 1:
                data = df.iloc[:, 0].values
            else:
                # Try first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    data = df[numeric_cols[0]].values
                else:
                    raise ValueError(
                        f"Could not find numeric data column in {filepath}"
                    )

        except Exception as e:
            # Try simple text file (one value per line)
            try:
                data = np.loadtxt(filepath)
            except Exception:
                raise ValueError(
                    f"Could not parse {variable_name} data file {filepath}: {e}"
                )

        # Validate length
        if len(data) != n_ensembles:
            raise ValueError(
                f"{variable_name} data length ({len(data)}) does not match "
                f"number of ensembles ({n_ensembles}). "
                f"File: {filepath}"
            )

        logger.info(
            f"Loaded {variable_name} data from {filepath}: "
            f"{len(data)} values, range [{data.min():.2f}, {data.max():.2f}]"
        )

        return data.astype(np.float64)

    def _get_replacement_value(
        self,
        option: str,
        fixed_value: float,
        file_path: str,
        n_ensembles: int,
        variable_name: str,
    ) -> Optional[Union[float, np.ndarray]]:
        """
        Get replacement value based on the selected option.

        Returns the appropriate type for ProcessedDataset.apply_sensor_health():
        - float for fixed values (will be expanded by apply_sensor_health)
        - np.ndarray for time series data from file

        Parameters
        ----------
        option : str
            "None", "Fixed Value", or "File"
        fixed_value : float
            Value to use if option is "Fixed Value"
        file_path : str
            Path to file if option is "File"
        n_ensembles : int
            Number of ensembles for file validation
        variable_name : str
            Name of variable (for logging)

        Returns
        -------
        float, np.ndarray, or None
            Replacement value, or None if option is "None"
        """
        if option == "None" or option == "":
            return None

        elif option == "Fixed Value":
            logger.info(f"Using fixed {variable_name} value: {fixed_value}")
            return fixed_value

        elif option == "File":
            if not file_path:
                logger.warning(
                    f"{variable_name} option is 'File' but no file path provided"
                )
                return None
            return self._load_timeseries_from_file(
                file_path, n_ensembles, variable_name
            )

        else:
            logger.warning(f"Unknown {variable_name} option: {option}")
            return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.

        Returns
        -------
        dict
            All configuration parameters as a dictionary
        """
        from dataclasses import asdict

        return asdict(self)

    def validate(self) -> list[str]:
        """
        Validate configuration for common issues.

        Returns
        -------
        list[str]
            List of warning/error messages (empty if valid)
        """
        issues = []

        # Check file paths exist
        if self.isTemperatureModified_ST and self.temperatureoption_ST == "File":
            if not self.temperature_file_ST:
                issues.append(
                    "Temperature file option selected but no file path provided"
                )
            elif not Path(self.temperature_file_ST).exists():
                issues.append(f"Temperature file not found: {self.temperature_file_ST}")

        if self.isSalinityModified_ST and self.salinityoption_ST == "File":
            if not self.salinity_file_ST:
                issues.append("Salinity file option selected but no file path provided")
            elif not Path(self.salinity_file_ST).exists():
                issues.append(f"Salinity file not found: {self.salinity_file_ST}")

        if self.isDepthModified_ST and self.depthoption_ST == "File":
            if not self.depth_file_ST:
                issues.append("Depth file option selected but no file path provided")
            elif not Path(self.depth_file_ST).exists():
                issues.append(f"Depth file not found: {self.depth_file_ST}")

        # Check threshold values
        if self.isRollCheck_ST and self.roll_cutoff_ST <= 0:
            issues.append(f"Roll cutoff must be positive, got {self.roll_cutoff_ST}")
        if self.isPitchCheck_ST and self.pitch_cutoff_ST <= 0:
            issues.append(f"Pitch cutoff must be positive, got {self.pitch_cutoff_ST}")

        # Check velocity cutoffs
        if self.isCutoffCheck_VT:
            if self.maxuvel_VT <= 0:
                issues.append(f"Max U velocity must be positive, got {self.maxuvel_VT}")
            if self.maxvvel_VT <= 0:
                issues.append(f"Max V velocity must be positive, got {self.maxvvel_VT}")
            if self.maxwvel_VT <= 0:
                issues.append(f"Max W velocity must be positive, got {self.maxwvel_VT}")

        return issues

    def __repr__(self) -> str:
        """Return string representation of configuration."""
        enabled = []
        if self.isTimeAxisModified:
            enabled.append("TimeAxis")
        if self.isSensorTest:
            enabled.append("SensorHealth")
        if self.isQCTest:
            enabled.append("SignalQuality")
        if self.isProfileTest:
            enabled.append("Profile")
        if self.isVelocityTest:
            enabled.append("Velocity")

        return f"ProcessingConfig(enabled=[{', '.join(enabled)}])"
