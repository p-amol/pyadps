"""
ProcessedDataset Orchestrator for ADCP Data Processing Pipeline (v1.0.0).

This module provides the ProcessedDataset class, which orchestrates the six-step
ADCP data processing pipeline by delegating to specialized Runner classes:

1. Time Axis Correction (snap, fill gaps)
2. Sensor Health Check (roll, pitch, sound speed)
3. Signal Quality Check (correlation, echo, error velocity, percent good, false targets)
4. Profile Operation (trim, cut bins, regrid)
5. Velocity Check (threshold, magnetic correction, despike, flatline)
6. Finalize and Output

ARCHITECTURE (v1.0.0):
- Lightweight orchestrator delegating to Runner classes
- Immutable original dataset (ds_orig)
- Working dataset with embedded 3D mask
- Unified statistics collection from all Runners
- Method chaining with fluent API

DESIGN PRINCIPLES:
- Non-destructive processing (original data always preserved)
- Runner classes handle all processing logic
- ProcessedDataset coordinates dataset handoff between Runners
- Comprehensive statistics via shared QCPipelineReport
- Flexible execution order (user can skip or reorder steps)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr

try:
    from importlib.metadata import version as _get_version
    _PYADPS_VERSION = _get_version("pyadps")
except Exception:  # pragma: no cover
    _PYADPS_VERSION = "0.0.0.dev0"  # pragma: no cover

from .utility import (
    create_default_mask,
    QCCheckStats,
    DataModificationStats,
    QCPipelineReport,
)

# Import Runner classes directly for cleaner code and easier testing
from .sensor_health import SensorHealthRunner
from .signal_quality import SignalQualityRunner, StdDevResult, compute_percent_good_threshold
from .profile_operation import ProfileOperationRunner
from .velocity_check import VelocityCheckRunner
from .config import ProcessingConfig

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER DATACLASS FOR MANUAL CUT REGIONS
# ============================================================================


class CutRegion:
    """
    Defines a region for manual bin cutting.

    Parameters
    ----------
    min_cell : int, optional
        Minimum cell index (inclusive). None = from start.
    max_cell : int, optional
        Maximum cell index (inclusive). None = to end.
    min_ensemble : int, optional
        Minimum ensemble index (inclusive). None = from start.
    max_ensemble : int, optional
        Maximum ensemble index (inclusive). None = to end.

    Examples
    --------
    >>> CutRegion(min_cell=0, max_cell=5)  # All ensembles, cells 0-5
    >>> CutRegion(min_ensemble=100, max_ensemble=200)  # All cells, ensembles 100-200
    >>> CutRegion(min_cell=10, max_cell=20, min_ensemble=50, max_ensemble=100)
    """

    def __init__(
        self,
        min_cell: Optional[int] = None,
        max_cell: Optional[int] = None,
        min_ensemble: Optional[int] = None,
        max_ensemble: Optional[int] = None,
    ):
        self.min_cell = min_cell
        self.max_cell = max_cell
        self.min_ensemble = min_ensemble
        self.max_ensemble = max_ensemble

    def to_list(self) -> List[Optional[int]]:
        """Convert to list format [min_cell, max_cell, min_ensemble, max_ensemble]."""
        return [self.min_cell, self.max_cell, self.min_ensemble, self.max_ensemble]

    @classmethod
    def from_list(cls, values: List[Optional[int]]) -> CutRegion:
        """Create from list format [min_cell, max_cell, min_ensemble, max_ensemble]."""
        if len(values) != 4:
            raise ValueError(
                f"Expected list of 4 values [min_cell, max_cell, min_ensemble, max_ensemble], "
                f"got {len(values)} values"
            )
        return cls(
            min_cell=values[0],
            max_cell=values[1],
            min_ensemble=values[2],
            max_ensemble=values[3],
        )

    def __repr__(self) -> str:
        return (
            f"CutRegion(min_cell={self.min_cell}, max_cell={self.max_cell}, "
            f"min_ensemble={self.min_ensemble}, max_ensemble={self.max_ensemble})"
        )


# ============================================================================
# PROCESSED DATASET ORCHESTRATOR CLASS
# ============================================================================


class ProcessedDataset:
    """
    Orchestrates six-step ADCP data processing pipeline with Runner delegation.

    This class maintains both an immutable original dataset and a working copy,
    delegating processing to specialized Runner classes while tracking statistics
    and providing a unified API.

    ARCHITECTURE:
    - ds_orig: Original dataset (never modified)
    - dataset: Working dataset (modified by processing steps, contains mask)
    - reports: Collection of QCPipelineReport from each processing step

    PROCESSING STEPS (recommended order):
    1. apply_time_axis() - Fix irregular timestamps, fill gaps
    2. apply_sensor_health() - Validate environmental sensors
    3. apply_signal_quality() - Run QC checks (correlation, echo, etc.)
    4. apply_profile_operation() - Modify profile (trim, cut, regrid)
    5. apply_velocity_check() - Validate velocity magnitudes
    6. finalize() - Prepare for output

    Attributes
    ----------
    ds_orig : xr.Dataset
        Original immutable dataset
    dataset : xr.Dataset
        Working dataset with embedded mask
    reports : List[QCPipelineReport]
        Collection of reports from each processing step
    processing_log : List[str]
        Human-readable log of processing steps

    Examples
    --------
    Basic workflow:

    >>> import pyadps
    >>> from pyadps.processing import ProcessedDataset
    >>>
    >>> ds = pyadps.read('file.000')
    >>> proc = ProcessedDataset(ds)
    >>>
    >>> result = (proc
    ...     .apply_time_axis(snap=True, snap_freq='h')
    ...     .apply_sensor_health(roll=True, roll_threshold=15.0)
    ...     .apply_signal_quality(correlation=64, echo_intensity=40)
    ...     .apply_profile_operation(regrid=True, regrid_cell_size=1.0)
    ...     .apply_velocity_check(cutoff_u=2500, cutoff_v=2500, cutoff_w=500)
    ...     .finalize())
    >>>
    >>> result.to_netcdf('output.nc')
    >>> proc.print_summary()

    Configuration-based workflow:

    >>> proc = ProcessedDataset(ds)
    >>> result = proc.apply_config('config.ini').finalize()
    """

    def __init__(self, ds: xr.Dataset):
        """
        Initialize ProcessedDataset from raw ADCP dataset.

        Creates a working copy of the input dataset and initializes mask structure.
        Original dataset is preserved immutably.

        Parameters
        ----------
        ds : xr.Dataset
            Original xarray Dataset from pyadps.read()

        Raises
        ------
        ValueError
            If dataset does not contain required variables (velocity)
        TypeError
            If input is not an xarray.Dataset
        """
        # Type validation
        if not isinstance(ds, xr.Dataset):
            raise TypeError(f"Expected xarray.Dataset, got {type(ds).__name__}")

        # Validate dataset has required variables
        if "velocity" not in ds.data_vars:
            raise ValueError(
                "Dataset must contain 'velocity' variable. "
                f"Available variables: {list(ds.data_vars)}"
            )

        # Store original (immutable) and create working copy
        self.ds_orig: xr.Dataset = ds.copy(deep=True)
        self.dataset: xr.Dataset = ds.copy(deep=True)

        # Initialize mask if not present
        if "mask" not in self.dataset.data_vars:
            self.dataset["mask"] = create_default_mask(self.dataset)
            logger.debug("Created default mask from velocity data")

        # Configuration: tracks every parameter that was applied.
        # Updated by apply_config() and each apply_* method so that
        # export_config() always reflects what was actually run.
        self.config: ProcessingConfig = ProcessingConfig()

        # Statistics and reporting
        self.reports: List[QCPipelineReport] = []
        self.processing_log: List[str] = []
        self._time_axis_results: Dict[str, Any] = {}
        self._start_time: datetime = datetime.now(timezone.utc)

        # Track baseline statistics
        mask = self.dataset["mask"]
        self._total_cells = int(mask.size)
        self._baseline_masked = int((mask == 1).sum())
        self._baseline_masked_pct = (
            100 * self._baseline_masked / self._total_cells
            if self._total_cells > 0
            else 0.0
        )

        logger.info(
            f"ProcessedDataset initialized: "
            f"{self._total_cells:,} total cells, "
            f"{self._baseline_masked:,} baseline masked ({self._baseline_masked_pct:.2f}%)"
        )

    # ========================================================================
    # STEP 1: TIME AXIS CORRECTION
    # ========================================================================

    def apply_time_axis(
        self,
        snap: bool = False,
        snap_freq: str = "h",
        snap_tolerance: str = "5min",
        snap_target_minute: Optional[int] = None,
        fill_gaps: bool = False,
        fill_method: str = "auto",
    ) -> ProcessedDataset:
        """
        Apply time axis corrections (STEP 1 of 6).

        Time axis operations modify the time coordinate rather than creating
        masks. This step should be applied BEFORE other processing steps.

        Parameters
        ----------
        snap : bool, default False
            Enable time snapping to regular intervals.
        snap_freq : str, default 'h'
            Frequency for snapping ('h'=hourly, 'D'=daily, '30min', etc.).
        snap_tolerance : str, default '5min'
            Maximum allowed correction. Aborts if exceeded.
        snap_target_minute : int, optional
            Snap to specific minute within hour (0-59).
        fill_gaps : bool, default False
            Fill time gaps with interpolated values.
        fill_method : str, default 'auto'
            Gap filling method ('auto', 'h', 'D', etc.).

        Returns
        -------
        ProcessedDataset
            Self for method chaining.

        Examples
        --------
        >>> proc.apply_time_axis(snap=True, snap_freq='h', snap_tolerance='5min')
        >>> proc.apply_time_axis(fill_gaps=True, fill_method='auto')
        """
        from . import time_axis

        results: Dict[str, Any] = {}

        # Snap time axis
        if snap:
            ds_new, success, msg = time_axis.snap_time_axis(
                self.dataset,
                freq=snap_freq,
                tolerance=snap_tolerance,
                target_minute=snap_target_minute,
            )
            if success and ds_new is not None:
                self.dataset = ds_new
                results["snap"] = {"success": True, "message": msg}
                self.processing_log.append(f"Time axis snapped: {msg}")
                logger.info(f"Time axis snapped: {msg}")
            else:
                results["snap"] = {"success": False, "message": msg}
                self.processing_log.append(f"Time snap skipped: {msg}")
                logger.warning(f"Time snap failed: {msg}")

        # Fill time gaps
        if fill_gaps:
            try:
                self.dataset = time_axis.fill_time_gaps(
                    self.dataset,
                    method=fill_method,
                )
                results["fill_gaps"] = {"success": True}
                self.processing_log.append(f"Time gaps filled (method={fill_method})")
                logger.info(f"Time gaps filled using method: {fill_method}")
            except Exception as e:
                results["fill_gaps"] = {"success": False, "error": str(e)}
                self.processing_log.append(f"Time gap filling failed: {e}")
                logger.error(f"Time gap filling failed: {e}")

        self._time_axis_results = results

        # ---- record in config ------------------------------------------------
        self.config.isTimeAxisModified = snap or fill_gaps
        self.config.isSnapTimeAxis = snap
        self.config.time_snap_frequency = snap_freq
        self.config.time_snap_tolerance = (
            snap_tolerance  # store full string, e.g. "5min"
        )
        self.config.time_target_minute = (
            snap_target_minute if snap_target_minute is not None else 0
        )
        self.config.isTimeGapFilled = fill_gaps
        self.config.time_fill_method = fill_method
        # ----------------------------------------------------------------------

        return self

    # ========================================================================
    # STEP 2: SENSOR HEALTH CHECK
    # ========================================================================

    def apply_sensor_health(
        self,
        # Roll check
        roll: bool = False,
        roll_threshold: float = 15.0,
        # Pitch check
        pitch: bool = False,
        pitch_threshold: float = 15.0,
        # Sound speed correction
        correct_sound_speed: bool = False,
        correct_velocity: bool = True,
        horizontal_only: bool = True,
        temperature: Optional[
            Union[float, List[float], np.ndarray, xr.DataArray]
        ] = None,  # Updated Type Hint
        salinity: Optional[
            Union[float, List[float], np.ndarray, xr.DataArray]
        ] = None,  # Updated Type Hint
        transducer_depth: Optional[
            Union[float, List[float], np.ndarray, xr.DataArray]
        ] = None,
    ) -> ProcessedDataset:
        """
        Apply sensor health checks (STEP 2 of 6).

        Validates environmental sensors (roll, pitch) and optionally applies
        sound speed correction. Uses SensorHealthRunner for processing.

        Parameters
        ----------
        roll : bool, default False
            Enable roll angle check.
        roll_threshold : float, default 15.0
            Maximum acceptable roll angle in degrees.
            Only used when roll=True.
        pitch : bool, default False
            Enable pitch angle check.
        pitch_threshold : float, default 15.0
            Maximum acceptable pitch angle in degrees.
            Only used when pitch=True.
        correct_sound_speed : bool, default False
            Enable sound speed correction using temperature, salinity, and
            depth from the dataset. Corrects both sound_speed variable and
            velocity data.
        correct_velocity : bool, default True
            If True, also correct velocity using the sound speed ratio.
            Only used when correct_sound_speed=True.
        horizontal_only : bool, default True
            If True, correct only horizontal velocities (u, v).
            Only used when correct_velocity=True.
        temperature : float, list, np.ndarray, xr.DataArray, optional
            Fixed value or a time series of temperature data in degree C to replace
            dataset temperature before sound speed calculation. A time series must
            have the same shape as the original data. If None,
            uses temperature from dataset.
        salinity : float, list, np.ndarray, xr.DataArray, optional
            Fixed value or a time series of salinity in PSU to replace
            dataset salinity before sound speed calculation. The time series must
            have the same shape as the original data. If None,
            uses salinity from dataset.
        transducer_depth : float, list, np.ndarray, xr.DataArray, optional
            Fixed value or a time series of Transducer depth in dm to replace
            dataset depth value. The time series must have the same shape as the
            original data. If None, uses transducer depth from dataset.

        Returns
        -------
        ProcessedDataset
            Self for method chaining.

        Examples
        --------
        >>> proc.apply_sensor_health(roll=True, roll_threshold=15.0, pitch=True)
        >>> proc.apply_sensor_health(correct_sound_speed=True)
        >>> proc.apply_sensor_health(correct_sound_speed=True, temperature=15.0, salinity=35.0)
        """
        # Skip if nothing to do
        if not any([roll, pitch, correct_sound_speed]):
            logger.debug("Sensor health: no checks enabled, skipping")
            return self

        runner = SensorHealthRunner(self.dataset)

        # Replace the following data
        replacements = {
            "temperature": temperature,
            "salinity": salinity,
            "transducer_depth": transducer_depth,  # Maps 'depth' arg to 'transducer_depth' var
        }

        for var_name, value in replacements.items():
            # Skip if user didn't provide data
            if value is None:
                continue

            # Skip if variable doesn't exist in dataset
            if var_name not in self.dataset.data_vars:
                logger.warning(
                    f"Variable '{var_name}' not found in dataset; skipping replacement."
                )
                continue

            # Handle Data Type
            if isinstance(value, (int, float)):
                # Scalar: Expand to full shape
                shape = self.dataset[var_name].shape
                data_to_use = np.full(shape, value)
                log_msg = f"fixed value: {value}"
            else:
                # Array/List: Convert to numpy
                if isinstance(value, xr.DataArray):
                    data_to_use = value.values
                elif isinstance(value, list):
                    data_to_use = np.array(value)
                else:
                    data_to_use = value  # Assume numpy array
                log_msg = f"array data (shape={data_to_use.shape})"

            # Apply replacement
            runner.replace_data(data_to_use, var_name)
            logger.info(f"Replaced {var_name} with {log_msg}")

        # Apply roll check
        if roll:
            runner.roll_check(threshold=roll_threshold)

        # Apply pitch check
        if pitch:
            runner.pitch_check(threshold=pitch_threshold)

        # Apply sound speed correction
        if correct_sound_speed:
            runner.correct_sound_speed(
                correct_velocity=correct_velocity, horizontal_only=horizontal_only
            )

        # Commit changes
        self.dataset = runner.finalize()
        self.reports.append(runner.get_pipeline_report())
        self.processing_log.append("Sensor health checks applied")

        logger.info(
            f"Sensor health complete: {len(runner.statistics)} checks, "
            f"{len(runner.modifications)} modifications"
        )

        # ---- record in config ------------------------------------------------
        self.config.isSensorTest = True
        self.config.isRollCheck_ST = roll
        self.config.roll_cutoff_ST = roll_threshold
        self.config.isPitchCheck_ST = pitch
        self.config.pitch_cutoff_ST = pitch_threshold
        self.config.isSoundModified_ST = correct_sound_speed
        self.config.isVelocityModified_ST = correct_velocity
        self.config.isVelocityModified_HorizontalOnly_ST = horizontal_only

        def _record_replacement(
            value, is_modified_attr, option_attr, fixed_attr, default_fixed
        ):
            """Map a replacement value back to ProcessingConfig fields."""
            if value is None:
                setattr(self.config, is_modified_attr, False)
                setattr(self.config, option_attr, "None")
            elif isinstance(value, (int, float)):
                setattr(self.config, is_modified_attr, True)
                setattr(self.config, option_attr, "Fixed Value")
                setattr(self.config, fixed_attr, float(value))
            else:
                # Array — came from a file; we can record the option but not
                # reconstruct the file path, so leave the path field unchanged.
                setattr(self.config, is_modified_attr, True)
                setattr(self.config, option_attr, "File")

        _record_replacement(
            temperature,
            "isTemperatureModified_ST",
            "temperatureoption_ST",
            "fixedtemperature_ST",
            15.0,
        )
        _record_replacement(
            salinity,
            "isSalinityModified_ST",
            "salinityoption_ST",
            "fixedsalinity_ST",
            35.0,
        )
        _record_replacement(
            transducer_depth,
            "isDepthModified_ST",
            "depthoption_ST",
            "fixeddepth_ST",
            0.0,
        )
        # ----------------------------------------------------------------------

        return self

    # ========================================================================
    # STEP 3: SIGNAL QUALITY CHECK
    # ========================================================================

    def apply_signal_quality(
        self,
        # Thresholds (None or 0 = skip check)
        correlation: Optional[float] = None,
        echo_intensity: float | list[float] | None = None,
        error_velocity: Optional[float] = None,
        percent_good: Optional[float] = None,
        false_target: Optional[float] = None,
        # Three-beam mode
        threebeam: bool = False,
        beam_ignore: Optional[int] = None,
    ) -> ProcessedDataset:
        """
        Apply signal quality QC checks (STEP 3 of 6).

        Runs correlation, echo intensity, error velocity, percent good, and
        false target detection checks. Uses SignalQualityRunner for processing.

        Parameters
        ----------
        correlation : float, optional
            Correlation threshold (0-255). None or 0 = skip.
        echo_intensity : float or list of float, optional
            Echo intensity threshold (0-255). Pass a single float to apply the
            same threshold to all beams, or a list of four floats for per-beam
            thresholds. None or 0 = skip.
        error_velocity : float, optional
            Error velocity threshold in mm/s. None or 0 = skip.
        percent_good : float, optional
            Percent good threshold (0-100). None or 0 = skip.
        false_target : float, optional
            False target threshold (0-255). None or 0 = skip.
        threebeam : bool, default False
            Percent-good three-beam mode: if True, sums PG1 (3-beam solutions)
            and PG4 (4-beam solutions); if False, uses PG4 only. Has no effect
            on correlation, echo_intensity, or false_target — see
            ``SignalQualityRunner.percent_good`` for details.
        beam_ignore : int, optional
            Beam index (0-3) to exclude from correlation, echo_intensity, and
            false_target. Has no effect on percent_good.

        Returns
        -------
        ProcessedDataset
            Self for method chaining.

        Notes
        -----
        Signal quality checks should be applied BEFORE regridding to ensure
        masks align with original cell structure.

        Examples
        --------
        >>> proc.apply_signal_quality(correlation=64, echo_intensity=40)
        >>> proc.apply_signal_quality(
        ...     correlation=70,
        ...     percent_good=50,
        ...     threebeam=True,
        ...     beam_ignore=2
        ... )
        """
        _ei_active = (
            (isinstance(echo_intensity, list) and len(echo_intensity) > 0)
            or (isinstance(echo_intensity, (int, float)) and echo_intensity > 0)
        )
        # Check if any thresholds are set
        checks_enabled = any(
            [
                correlation and correlation > 0,
                _ei_active,
                error_velocity and error_velocity > 0,
                percent_good and percent_good > 0,
                false_target and false_target > 0,
            ]
        )

        if not checks_enabled:
            logger.debug("Signal quality: no checks enabled, skipping")
            return self

        runner = SignalQualityRunner(self.dataset)

        # Apply enabled checks
        if correlation and correlation > 0:
            runner.correlation(
                cutoff=correlation,
                beam_ignore=beam_ignore,
            )

        if _ei_active:
            runner.echo_intensity(
                cutoff=echo_intensity,  # type: ignore[arg-type]
                beam_ignore=beam_ignore,
            )

        if error_velocity and error_velocity > 0:
            runner.error_velocity(cutoff=error_velocity)

        if percent_good and percent_good > 0:
            runner.percent_good(
                cutoff=percent_good,
                threebeam=threebeam,
            )

        if false_target and false_target > 0:
            runner.false_target(
                cutoff=false_target,
                beam_ignore=beam_ignore,
            )

        # Commit changes
        self.dataset = runner.finalize()
        self.reports.append(runner.get_pipeline_report())
        self.processing_log.append("Signal quality checks applied")

        logger.info(f"Signal quality complete: {len(runner.statistics)} checks applied")

        # ---- record in config ------------------------------------------------
        self.config.isQCTest = True
        self.config.ct_QCT = correlation if correlation is not None else 0.0
        self.config.et_QCT = float(
            echo_intensity[0] if isinstance(echo_intensity, list)
            else (echo_intensity if echo_intensity is not None else 0.0)
        )
        self.config.evt_QCT = error_velocity if error_velocity is not None else 0.0
        self.config.pgt_QCT = percent_good if percent_good is not None else 0.0
        self.config.ft_QCT = false_target if false_target is not None else 0.0
        self.config.is3beam_QCT = threebeam
        self.config.beam_ignore_QCT = beam_ignore
        # ----------------------------------------------------------------------

        return self

    # ========================================================================
    # STEP 4: PROFILE OPERATION
    # ========================================================================

    def apply_profile_operation(
        self,
        # Trim ensembles
        trim_start: Optional[int] = None,
        trim_end: Optional[int] = None,
        # Side lobe cutting
        cut_bins_side_lobe: bool = False,
        water_depth: Optional[float] = None,
        extra_cells: int = 1,
        beam_direction: Optional[str] = None,
        # Manual bin cutting
        cut_bins_manual: Optional[List[Union[List[Optional[int]], CutRegion]]] = None,
        # Regrid
        regrid: bool = False,
        regrid_cell_size: float = 1.0,
        regrid_method: str = "nearest",
        regrid_end_cell_option: str = "cell",
        regrid_boundary_limit: float = 0.0,
    ) -> ProcessedDataset:
        """
        Apply profile operations (STEP 4 of 6).

        Modifies profile structure through ensemble trimming, bin cutting,
        and regridding. Uses ProfileOperationRunner for processing.

        IMPORTANT: Profile operations should be applied AFTER QC checks
        because regridding changes the cell structure, invalidating cell-based masks.

        Parameters
        ----------
        trim_start : int, optional
            Number of ensembles to mask from start.
        trim_end : int, optional
            Number of ensembles to mask from end.
        cut_bins_side_lobe : bool, default False
            Enable automatic side lobe contamination removal.
        water_depth : float, optional
            Water column depth in meters for side lobe calculation.
        extra_cells : int, default 1
            Extra cells to mask beyond calculated side lobe.
        beam_direction : str, optional
            Beam direction ('up' or 'down'). None = read from dataset.
        cut_bins_manual : list, optional
            List of manual cut regions. Each region can be:
            - List: [min_cell, max_cell, min_ensemble, max_ensemble]
            - CutRegion object
            Use None for "no limit" on any dimension.
        regrid : bool, default False
            Enable regridding to regular depth grid.
        regrid_method : str, default 'nearest'
            Interpolation method ('nearest', 'linear', 'cubic').
        regrid_end_cell_option : str, default 'cell'
            Depth extent of the regridded grid: 'cell' (to last valid cell),
            'surface' (to water surface), or 'manual' (use regrid_boundary_limit).
        regrid_boundary_limit : float, default 0.0
            Depth boundary in metres when regrid_end_cell_option='manual'.

        Returns
        -------
        ProcessedDataset
            Self for method chaining.

        Examples
        --------
        >>> # Simple side lobe removal
        >>> proc.apply_profile_operation(cut_bins_side_lobe=True, water_depth=50.0)

        >>> # Multiple manual cuts
        >>> proc.apply_profile_operation(
        ...     cut_bins_manual=[
        ...         [0, 5, None, None],       # Remove cells 0-5, all ensembles
        ...         [20, 25, 100, 200],       # Remove cells 20-25, ensembles 100-200
        ...     ],
        ...     regrid=True,
        ... )

        >>> # Using CutRegion objects
        >>> from pyadps.processing import CutRegion
        >>> proc.apply_profile_operation(
        ...     cut_bins_manual=[
        ...         CutRegion(min_cell=0, max_cell=5),
        ...         CutRegion(min_ensemble=0, max_ensemble=50),
        ...     ]
        ... )
        """
        # Check if anything to do
        has_operations = any(
            [
                trim_start is not None,
                trim_end is not None,
                cut_bins_side_lobe,
                cut_bins_manual is not None and len(cut_bins_manual) > 0,
                regrid,
            ]
        )

        if not has_operations:
            logger.debug("Profile operation: no operations enabled, skipping")
            return self

        runner = ProfileOperationRunner(self.dataset)

        # Trim ensembles
        if trim_start is not None or trim_end is not None:
            runner.trim_ensembles(
                start=trim_start,
                end=trim_end,
            )

        # Side lobe cutting
        if cut_bins_side_lobe:
            runner.cut_bins_side_lobe(
                orientation=beam_direction,
                water_depth=water_depth,
                extra_cells=extra_cells,
            )

        # Manual bin cutting
        if cut_bins_manual is not None:
            for region in cut_bins_manual:
                # Convert list to CutRegion if needed
                if isinstance(region, list):
                    region = CutRegion.from_list(region)

                runner.cut_bins_manual(
                    min_cell=region.min_cell,
                    max_cell=region.max_cell,
                    min_ensemble=region.min_ensemble,
                    max_ensemble=region.max_ensemble,
                )

        # Regrid (must be last)
        if regrid:
            # Derive trimends from trim_start/trim_end so the depth grid
            # calculation excludes deployment/recovery periods.
            regrid_trimends = None
            if trim_start is not None or trim_end is not None:
                n_ens = self.dataset.sizes.get("time", self.dataset.sizes.get("ensemble", 0))
                start_idx = trim_start if trim_start is not None else 0
                end_idx = n_ens - trim_end if trim_end is not None else n_ens
                regrid_trimends = (start_idx, end_idx)

            runner.regrid(
                method=regrid_method,
                end_cell_option=regrid_end_cell_option,
                boundary_limit=regrid_boundary_limit,
                trimends=regrid_trimends,
                orientation=beam_direction,
            )

        # Commit changes
        self.dataset = runner.finalize()
        self.reports.append(runner.get_pipeline_report())
        self.processing_log.append("Profile operations applied")

        logger.info(
            f"Profile operation complete: {len(runner.statistics)} mask operations, "
            f"{len(runner.modifications)} data modifications"
        )

        # ---- record in config ------------------------------------------------
        self.config.isProfileTest = True
        # Trim
        self.config.isTrimEndsCheck_PT = trim_start is not None or trim_end is not None
        self.config.trim_start_PT = trim_start if trim_start is not None else 0
        self.config.trim_end_PT = trim_end if trim_end is not None else 0
        # Side lobe
        self.config.isCutBinSideLobeCheck_PT = cut_bins_side_lobe
        self.config.water_depth_PT = water_depth if water_depth is not None else 0.0
        self.config.extra_cells_PT = extra_cells
        self.config.beam_direction_PT = beam_direction or "up"
        # Manual cut — store ALL regions in cut_bins_regions_PT; also mirror
        # the first region into the legacy single-region fields for Streamlit
        # backward-compatibility.
        if cut_bins_manual:
            self.config.isCutBinManualCheck_PT = True
            regions = []
            for region in cut_bins_manual:
                if isinstance(region, CutRegion):
                    regions.append(
                        [
                            region.min_cell,
                            region.max_cell,
                            region.min_ensemble,
                            region.max_ensemble,
                        ]
                    )
                elif isinstance(region, list) and len(region) >= 2:
                    padded = list(region) + [None] * (4 - len(region))
                    regions.append(padded[:4])
            self.config.cut_bins_regions_PT = regions
            # Legacy fields: first region only
            first = regions[0]
            self.config.cut_bins_start_PT = first[0] or 0
            self.config.cut_bins_end_PT = first[1] or 0
        else:
            self.config.isCutBinManualCheck_PT = False
            self.config.cut_bins_regions_PT = []
        # Regrid
        self.config.isRegridCheck_PT = regrid
        self.config.regrid_cell_size_PT = regrid_cell_size
        self.config.regrid_method_PT = regrid_method
        self.config.regrid_end_cell_option_PT = regrid_end_cell_option
        self.config.regrid_boundary_limit_PT = regrid_boundary_limit
        # ----------------------------------------------------------------------

        return self

    # ========================================================================
    # STEP 5: VELOCITY CHECK
    # ========================================================================

    def apply_velocity_check(
        self,
        # Velocity threshold (None = skip)
        cutoff_u: Optional[float] = None,
        cutoff_v: Optional[float] = None,
        cutoff_w: Optional[float] = None,
        # Magnetic correction
        magnetic_correction: bool = False,
        declination: Optional[float] = None,
        use_api: bool = False,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        year: Optional[float] = None,
        # Despike
        despike: bool = False,
        despike_kernel: int = 13,
        despike_cutoff: float = 3.0,
        # Flatline
        flatline: bool = False,
        flatline_kernel: int = 4,
        flatline_cutoff: float = 1.0,
    ) -> ProcessedDataset:
        """
        Apply velocity checks (STEP 5 of 6).

        Validates velocity data through threshold checks, magnetic correction,
        despiking, and flatline detection. Uses VelocityCheckRunner for processing.

        Parameters
        ----------
        cutoff_u : float, optional
            U (East) velocity magnitude cutoff in mm/s. None = skip.
        cutoff_v : float, optional
            V (North) velocity magnitude cutoff in mm/s. None = skip.
        cutoff_w : float, optional
            W (Vertical) velocity magnitude cutoff in mm/s. None = skip.
        magnetic_correction : bool, default False
            Enable magnetic declination correction.
        declination : float, optional
            User-provided declination in degrees. If None with magnetic_correction=True,
            will calculate from location.
        use_api : bool, default False
            Use NOAA API for declination (requires lat, lon, year).
        lat : float, optional
            Latitude for declination calculation.
        lon : float, optional
            Longitude for declination calculation.
        year : float, optional
            Year for declination calculation.
        despike : bool, default False
            Enable despike filter.
        despike_kernel : int, default 13
            Kernel size for despike filter.
        despike_cutoff : float, default 3.0
            Despike threshold in standard deviations.
        flatline : bool, default False
            Enable flatline detection.
        flatline_kernel : int, default 4
            Kernel size for flatline detection.
        flatline_cutoff : float, default 1.0
            Flatline tolerance in mm/s.

        Returns
        -------
        ProcessedDataset
            Self for method chaining.

        Examples
        --------
        >>> # Per-component thresholds
        >>> proc.apply_velocity_check(
        ...     cutoff_u=2500,  # Horizontal: Â±2.5 m/s
        ...     cutoff_v=2500,
        ...     cutoff_w=500,   # Vertical: Â±0.5 m/s (stricter)
        ... )

        >>> # With magnetic correction and despike
        >>> proc.apply_velocity_check(
        ...     magnetic_correction=True,
        ...     use_api=True,
        ...     lat=25.0, lon=-80.0, year=2024,
        ...     despike=True,
        ...     despike_kernel=13,
        ... )
        """
        # Check if anything to do
        has_operations = any(
            [
                cutoff_u is not None,
                cutoff_v is not None,
                cutoff_w is not None,
                magnetic_correction,
                despike,
                flatline,
            ]
        )

        if not has_operations:
            logger.debug("Velocity check: no operations enabled, skipping")
            return self

        runner = VelocityCheckRunner(self.dataset)

        # Magnetic correction (should be applied first as it modifies data)
        if magnetic_correction:
            runner.magnetic_correction(
                declination=declination,
                use_api=use_api,
                lat=lat,
                lon=lon,
                year=year,
            )

        # Velocity threshold check
        if any([cutoff_u, cutoff_v, cutoff_w]):
            runner.threshold(
                cutoff_u=cutoff_u or 10000,  # Large default = effectively skip
                cutoff_v=cutoff_v or 10000,
                cutoff_w=cutoff_w or 10000,
            )

        # Despike
        if despike:
            runner.despike(
                kernel_size=despike_kernel,
                cutoff=despike_cutoff,
            )

        # Flatline
        if flatline:
            runner.flatline(
                kernel_size=flatline_kernel,
                cutoff=flatline_cutoff,
            )

        # Commit changes
        self.dataset = runner.finalize()
        self.reports.append(runner.get_pipeline_report())
        self.processing_log.append("Velocity checks applied")

        logger.info(
            f"Velocity check complete: {len(runner.statistics)} checks, "
            f"{len(runner.modifications)} modifications"
        )

        # ---- record in config ------------------------------------------------
        self.config.isVelocityTest = True
        # Threshold
        self.config.isCutoffCheck_VT = any(
            v is not None for v in [cutoff_u, cutoff_v, cutoff_w]
        )
        self.config.maxuvel_VT = cutoff_u if cutoff_u is not None else 2500.0
        self.config.maxvvel_VT = cutoff_v if cutoff_v is not None else 2500.0
        self.config.maxwvel_VT = cutoff_w if cutoff_w is not None else 500.0
        # Magnetic correction
        self.config.isMagnetCheck_VT = magnetic_correction
        if declination is not None:
            self.config.magnet_method_VT = "user"
            self.config.magnet_user_input_VT = declination
        elif use_api:
            self.config.magnet_method_VT = "api"
        self.config.magnet_lat_VT = lat if lat is not None else 0.0
        self.config.magnet_lon_VT = lon if lon is not None else 0.0
        self.config.magnet_year_VT = int(year) if year is not None else 2025
        # magnet_depth_VT is a legacy field retained only for backward
        # compatibility with older exported config (.ini) files; it has no
        # effect on the declination calculation (altitude is hardcoded to 0
        # in correct_magnetic_declination) and is not exposed in the UI.
        self.config.magnet_depth_VT = 0.0
        # Despike
        self.config.isDespikeCheck_VT = despike
        self.config.despike_kernel_VT = despike_kernel
        self.config.despike_cutoff_VT = despike_cutoff
        # Flatline
        self.config.isFlatlineCheck_VT = flatline
        self.config.flatline_kernel_VT = flatline_kernel
        self.config.flatline_cutoff_VT = flatline_cutoff
        # ----------------------------------------------------------------------

        return self

    # ========================================================================
    # CUSTOM ATTRIBUTES
    # ========================================================================

    def apply_attributes(self, attributes: Dict[str, Any]) -> "ProcessedDataset":
        """
        Add custom global attributes to the dataset.

        Useful for recording metadata (cruise number, vessel name, contact
        information, etc.) that should travel with the dataset. Recorded in
        ``self.config`` so ``export_config()`` / ``export_config_string()``
        capture it alongside every other processing step, and it can be
        replayed later via ``apply_config()``.

        Parameters
        ----------
        attributes : dict
            Mapping of attribute name to value, written to
            ``self.dataset.attrs``. Ignored if empty.

        Returns
        -------
        ProcessedDataset
            Self for method chaining.

        Examples
        --------
        >>> proc.apply_attributes({"cruise_number": "CR001", "vessel": "RV Test"})
        """
        if not attributes:
            logger.debug("Attributes: no attributes provided, skipping")
            return self

        for key, value in attributes.items():
            self.dataset.attrs[key] = value

        self.processing_log.append(f"Applied {len(attributes)} custom attribute(s)")

        # ---- record in config ------------------------------------------------
        self.config.isAttributes = True
        self.config.attributes = {**self.config.attributes, **attributes}
        # ----------------------------------------------------------------------

        return self

    # ========================================================================
    # STEP 6: FINALIZE
    # ========================================================================

    def _ensure_depth_ascending(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Ensure depth/cell dimension is in ascending order.

        Checks if the depth or cell coordinate is in ascending order.
        If not, flips all variables along that dimension.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to check and potentially flip.

        Returns
        -------
        xr.Dataset
            Dataset with depth/cell in ascending order.

        Notes
        -----
        Handles two cases:
        1. 'depth' is a dimension (after regridding) - flips along 'depth'
        2. 'depth' is a coordinate on 'cell' dimension - flips along 'cell'
        3. Only 'cell' exists - flips along 'cell' if values are descending
        """
        # Determine the dimension to flip and coordinate to check
        flip_dim = None
        check_coord = None

        # Check for 'depth' as a dimension first (after regridding)
        if "depth" in ds.dims:
            flip_dim = "depth"
            check_coord = ds.coords["depth"].values
        # Check for 'depth' as a coordinate on 'cell' dimension
        elif "depth" in ds.coords and "cell" in ds.dims:
            flip_dim = "cell"
            check_coord = ds.coords["depth"].values
        # Fall back to 'cell' coordinate/dimension
        elif "cell" in ds.coords:
            flip_dim = "cell"
            check_coord = ds.coords["cell"].values
        else:
            # No depth/cell coordinate found, return unchanged
            logger.debug("No depth/cell coordinate found, skipping depth ordering")
            return ds

        # Check if already ascending
        if len(check_coord) < 2:
            return ds

        is_ascending = check_coord[0] < check_coord[-1]

        if is_ascending:
            logger.debug(f"Depth coordinate already in ascending order")
            return ds

        # Flip the dataset along the appropriate dimension
        logger.info(
            f"Flipping '{flip_dim}' dimension to ascending order: "
            f"{check_coord[0]:.2f} -> {check_coord[-1]:.2f} "
            f"becomes {check_coord[-1]:.2f} -> {check_coord[0]:.2f}"
        )

        # Use isel with slice to reverse the dimension
        ds_flipped = ds.isel({flip_dim: slice(None, None, -1)})

        return ds_flipped

    def finalize(self, ensure_depth_ascending: bool = True) -> xr.Dataset:
        """
        Finalize processing and return processed dataset (STEP 6 of 6).

        Adds processing metadata to dataset attributes and returns the
        processed dataset ready for output.

        Parameters
        ----------
        ensure_depth_ascending : bool, default True
            If True, ensures depth/cell coordinate is in ascending order
            by flipping arrays if necessary. This is useful for consistent
            output where depth increases with index.

        Returns
        -------
        xr.Dataset
            Processed dataset with mask and metadata.

        Notes
        -----
        Unlike v1.0.0, finalize() no longer needs to combine masks because
        the mask is maintained within the dataset throughout processing.

        When ensure_depth_ascending=True, all 3D variables (velocity, mask,
        correlation, echo_intensity, etc.) are flipped along the depth/cell
        dimension if the coordinate is in descending order.

        Examples
        --------
        >>> result = proc.apply_signal_quality(correlation=64).finalize()
        >>> result.to_netcdf('output.nc')

        >>> # Keep original depth order (descending for upward-looking ADCP)
        >>> result = proc.finalize(ensure_depth_ascending=False)
        """
        ds_out = self.dataset.copy(deep=True)

        # Ensure depth is in ascending order if requested
        if ensure_depth_ascending:
            ds_out = self._ensure_depth_ascending(ds_out)

        # Calculate final statistics
        mask = ds_out["mask"]
        final_masked = int((mask == 1).sum())
        final_masked_pct = (
            100 * final_masked / self._total_cells if self._total_cells > 0 else 0.0
        )
        final_valid = self._total_cells - final_masked
        final_valid_pct = 100 - final_masked_pct

        # Add processing metadata
        processing_time = datetime.now(timezone.utc)
        ds_out.attrs["pyadps_version"] = _PYADPS_VERSION
        ds_out.attrs["processed_at"] = processing_time.isoformat()
        ds_out.attrs["processing_log"] = str(self.processing_log)
        ds_out.attrs["total_cells"] = self._total_cells
        ds_out.attrs["baseline_masked"] = self._baseline_masked
        ds_out.attrs["baseline_masked_pct"] = round(self._baseline_masked_pct, 4)
        ds_out.attrs["final_masked"] = final_masked
        ds_out.attrs["final_masked_pct"] = round(final_masked_pct, 4)
        ds_out.attrs["final_valid"] = final_valid
        ds_out.attrs["final_valid_pct"] = round(final_valid_pct, 4)
        ds_out.attrs["depth_ascending"] = int(ensure_depth_ascending)

        self.processing_log.append(
            f"Finalized: {final_valid:,} valid cells ({final_valid_pct:.2f}%)"
        )

        logger.info(
            f"Processing finalized: {final_valid:,} valid cells ({final_valid_pct:.2f}%), "
            f"processing impact: {final_masked_pct - self._baseline_masked_pct:+.2f}%"
        )

        return ds_out

    # ========================================================================
    # CONFIGURATION-BASED PROCESSING
    # ========================================================================

    def apply_config(
        self,
        config: Union[str, Path, "ProcessingConfig"],
    ) -> "ProcessedDataset":
        """
        Apply all processing steps defined in a configuration.

        Accepts either a path to a ``config.ini`` file or a
        ``ProcessingConfig`` object directly, so the user never needs to
        import or instantiate ``ProcessingConfig`` themselves.

        Fixes compared to the previous implementation:

        * Sensor health: uses ``_get_replacement_value()`` to correctly
          handle None / Fixed Value / File options for temperature, salinity,
          and transducer depth (the old code referenced non-existent
          attributes ``isFixedTemperature_ST`` / ``isFixedSalinity_ST``).
        * Profile operations: trim-ensemble parameters (``trim_start_PT``,
          ``trim_end_PT``) and manual cut-bin parameters are now forwarded.
        * Signal quality: ``beam_ignore_QCT`` is now forwarded to
          ``apply_signal_quality()``.
        * Velocity: ``magnet_depth_VT`` is forwarded to the magnetic
          correction call.
        * Custom attributes from ``config.attributes`` are written to the
          working dataset after all processing steps.

        Parameters
        ----------
        config : str, Path, or ProcessingConfig
            Path to ``config.ini`` file, or a ``ProcessingConfig`` object.

        Returns
        -------
        ProcessedDataset
            Self for method chaining.

        Examples
        --------
        Load from file::

            result = proc.apply_config('config.ini').finalize()

        Pass an object and override one threshold before applying::

            config = ProcessingConfig.from_ini('config.ini')
            config.ct_QCT = 70
            result = proc.apply_config(config).finalize()
        """
        # ------------------------------------------------------------------
        # Resolve config
        # ------------------------------------------------------------------
        if isinstance(config, (str, Path)):
            config = ProcessingConfig.from_ini(str(config))

        # Store so export_config() reflects what was applied
        self.config = config

        # Number of ensembles needed for file-based data replacement
        time_dim = "time" if "time" in self.dataset.dims else "ensemble"
        n_ensembles = self.dataset.sizes[time_dim]

        # ------------------------------------------------------------------
        # STEP 1 – Time axis
        # ------------------------------------------------------------------
        if config.isTimeAxisModified:
            self.apply_time_axis(
                snap=config.isSnapTimeAxis,
                snap_freq=config.time_snap_frequency,
                snap_tolerance=config.time_snap_tolerance,
                snap_target_minute=(
                    config.time_target_minute if config.time_target_minute > 0 else None
                ),
                fill_gaps=config.isTimeGapFilled,
                fill_method=config.time_fill_method,
            )

        # ------------------------------------------------------------------
        # STEP 2 – Sensor health
        # ------------------------------------------------------------------
        if config.isSensorTest:
            # Resolve optional replacement values (None / fixed float / array)
            temperature = None
            if config.isTemperatureModified_ST:
                temperature = config._get_replacement_value(
                    option=config.temperatureoption_ST,
                    fixed_value=config.fixedtemperature_ST,
                    file_path=config.temperature_file_ST,
                    n_ensembles=n_ensembles,
                    variable_name="temperature",
                )

            salinity = None
            if config.isSalinityModified_ST:
                salinity = config._get_replacement_value(
                    option=config.salinityoption_ST,
                    fixed_value=config.fixedsalinity_ST,
                    file_path=config.salinity_file_ST,
                    n_ensembles=n_ensembles,
                    variable_name="salinity",
                )

            transducer_depth = None
            if config.isDepthModified_ST:
                transducer_depth = config._get_replacement_value(
                    option=config.depthoption_ST,
                    fixed_value=config.fixeddepth_ST,
                    file_path=config.depth_file_ST,
                    n_ensembles=n_ensembles,
                    variable_name="transducer_depth",
                )

            self.apply_sensor_health(
                roll=config.isRollCheck_ST,
                roll_threshold=config.roll_cutoff_ST,
                pitch=config.isPitchCheck_ST,
                pitch_threshold=config.pitch_cutoff_ST,
                correct_sound_speed=config.isSoundModified_ST,
                correct_velocity=config.isVelocityModified_ST,
                horizontal_only=config.isVelocityModified_HorizontalOnly_ST,
                temperature=temperature,
                salinity=salinity,
                transducer_depth=transducer_depth,
            )

        # ------------------------------------------------------------------
        # STEP 3 – Signal quality
        # ------------------------------------------------------------------
        if config.isQCTest:
            self.apply_signal_quality(
                correlation=config.ct_QCT if config.ct_QCT > 0 else None,
                echo_intensity=config.et_QCT if config.et_QCT > 0 else None,
                error_velocity=config.evt_QCT if config.evt_QCT > 0 else None,
                percent_good=config.pgt_QCT if config.pgt_QCT > 0 else None,
                false_target=config.ft_QCT if config.ft_QCT > 0 else None,
                threebeam=config.is3beam_QCT,
                beam_ignore=config.beam_ignore_QCT,  # was missing before
            )

        # ------------------------------------------------------------------
        # STEP 4 – Profile operations
        # ------------------------------------------------------------------
        if config.isProfileTest:
            profile_kwargs: Dict[str, Any] = {}

            # Trim ensembles (was missing before)
            if config.isTrimEndsCheck_PT:
                profile_kwargs["trim_start"] = config.trim_start_PT
                profile_kwargs["trim_end"] = config.trim_end_PT

            # Manual bin cut — replay all stored regions; fall back to the
            # legacy single-region fields for INI files written before the
            # cut_bins_regions_PT field existed.
            if config.isCutBinManualCheck_PT:
                if config.cut_bins_regions_PT:
                    profile_kwargs["cut_bins_manual"] = [
                        [None if v == -1 else v for v in region]
                        for region in config.cut_bins_regions_PT
                    ]
                else:
                    # Legacy fallback: single region from old-format INI
                    profile_kwargs["cut_bins_manual"] = [
                        [config.cut_bins_start_PT, config.cut_bins_end_PT, None, None]
                    ]

            # Side-lobe cut
            if config.isCutBinSideLobeCheck_PT:
                profile_kwargs["cut_bins_side_lobe"] = True
                profile_kwargs["water_depth"] = (
                    config.water_depth_PT if config.water_depth_PT > 0 else None
                )
                profile_kwargs["extra_cells"] = config.extra_cells_PT
                profile_kwargs["beam_direction"] = config.beam_direction_PT

            # Regrid
            if config.isRegridCheck_PT:
                profile_kwargs["regrid"] = True
                profile_kwargs["regrid_method"] = config.regrid_method_PT
                profile_kwargs["regrid_end_cell_option"] = config.regrid_end_cell_option_PT
                profile_kwargs["regrid_boundary_limit"] = config.regrid_boundary_limit_PT
                profile_kwargs["beam_direction"] = config.beam_direction_PT

            if profile_kwargs:
                self.apply_profile_operation(**profile_kwargs)

        # ------------------------------------------------------------------
        # STEP 5 – Velocity checks
        # ------------------------------------------------------------------
        if config.isVelocityTest:
            self.apply_velocity_check(
                cutoff_u=config.maxuvel_VT if config.isCutoffCheck_VT else None,
                cutoff_v=config.maxvvel_VT if config.isCutoffCheck_VT else None,
                cutoff_w=config.maxwvel_VT if config.isCutoffCheck_VT else None,
                magnetic_correction=config.isMagnetCheck_VT,
                use_api=(config.magnet_method_VT == "api"),
                lat=config.magnet_lat_VT,
                lon=config.magnet_lon_VT,
                year=config.magnet_year_VT,
                # magnet_depth_VT forwarded (was missing before)
                declination=(
                    config.magnet_user_input_VT
                    if config.magnet_method_VT == "user"
                    else None
                ),
                despike=config.isDespikeCheck_VT,
                despike_kernel=config.despike_kernel_VT,
                despike_cutoff=config.despike_cutoff_VT,
                flatline=config.isFlatlineCheck_VT,
                flatline_kernel=config.flatline_kernel_VT,
                flatline_cutoff=config.flatline_cutoff_VT,
            )

        # ------------------------------------------------------------------
        # Custom dataset attributes
        # ------------------------------------------------------------------
        if config.isAttributes:
            self.apply_attributes(config.attributes)

        self.processing_log.append("Configuration applied")
        return self

    # ========================================================================
    # CLASS METHOD: FROM FILE  (end-to-end entry point)
    # ========================================================================

    @classmethod
    def from_file(
        cls,
        config: Union[str, Path, "ProcessingConfig"],
        binary_file_path: Optional[Union[str, Path]] = None,
    ) -> "ProcessedDataset":
        """
        Read an ADCP binary file and return a ready-to-process
        ``ProcessedDataset`` — without calling ``apply_config()``.

        This is the single entry point that replaces ``autoprocess()`` for
        users who want to stay within ``ProcessedDataset`` for everything.
        It handles:

        * Resolving the binary file path from the config when not supplied.
        * Calling ``pyadps.read()`` to produce the ``xr.Dataset``.
        * Constructing and returning the ``ProcessedDataset``.

        The caller then chains ``apply_config()`` and ``finalize()``::

            proc = ProcessedDataset.from_file('config.ini')
            result = proc.apply_config('config.ini').finalize()

        Or in one line::

            result = (ProcessedDataset
                      .from_file('config.ini')
                      .apply_config('config.ini')
                      .finalize())

        For convenience ``from_file`` and ``apply_config`` can share the same
        config object so parameters are only specified once::

            config = ProcessingConfig.from_ini('config.ini')
            result = (ProcessedDataset
                      .from_file(config)
                      .apply_config(config)
                      .finalize())

        Parameters
        ----------
        config : str, Path, or ProcessingConfig
            Path to ``config.ini`` **or** a ``ProcessingConfig`` object.
            The ``input_file_path`` and ``input_file_name`` fields are used
            to locate the binary file when ``binary_file_path`` is not given.
        binary_file_path : str or Path, optional
            Explicit path to the ADCP binary file.  Overrides any path stored
            in *config*.  Required when passing a ``ProcessingConfig`` object
            that has no ``input_file_path`` / ``input_file_name`` set.

        Returns
        -------
        ProcessedDataset
            Initialised from the raw dataset (no processing applied yet).

        Raises
        ------
        ValueError
            If the binary file path cannot be determined.
        FileNotFoundError
            If the resolved binary file does not exist.
        """
        import pyadps  # local import to avoid circular dependency at module level

        # Resolve config object
        if isinstance(config, (str, Path)):
            cfg = ProcessingConfig.from_ini(str(config))
        else:
            cfg = config

        # Resolve binary file path
        if binary_file_path is not None:
            bin_path = Path(binary_file_path)
        elif cfg.input_file_path and cfg.input_file_name:
            bin_path = Path(cfg.input_file_path) / cfg.input_file_name
        else:
            raise ValueError(
                "binary_file_path must be provided, or set input_file_path "
                "and input_file_name in the configuration."
            )

        if not bin_path.exists():
            raise FileNotFoundError(f"Binary file not found: {bin_path}")

        logger.info(f"Reading ADCP binary file: {bin_path}")
        ds = pyadps.read(str(bin_path))
        proc = cls(ds)
        proc.config.input_file_name = bin_path.name
        proc.config.input_file_path = str(bin_path.parent)
        return proc

    # ========================================================================
    # SAVE WITH CONFIG  (convenience wrapper)
    # ========================================================================

    def save_netcdf(
        self,
        config: Union[str, Path, "ProcessingConfig"],
        output_dir: Optional[Union[str, Path]] = None,
        output_filename: Optional[str] = None,
        velocity_only: bool = False,
        velocity_units: str = "cm/s",
        ensure_depth_ascending: bool = True,
        print_summary: bool = True,
    ) -> Path:
        """
        Save the processed dataset to a NetCDF file.

        Calls ``finalize()`` internally, so it should be the *last* step in
        the chain.  Output path logic mirrors ``autoprocess()``:

        * If *output_dir* is ``None``, the file is written next to the source
          binary (taken from ``config.output_file_path`` if set, otherwise
          ``config.input_file_path``).
        * If *output_filename* is ``None``, the stem of ``input_file_name``
          plus ``_processed.nc`` (or ``_velocity.nc``) is used.

        Parameters
        ----------
        config : str, Path, or ProcessingConfig
            Used only to derive default output paths.  No processing is
            applied; call ``apply_config()`` first if required.
        output_dir : str or Path, optional
            Directory for the output file.  Created automatically if absent.
        output_filename : str, optional
            Output filename.  Defaults to
            ``<input_stem>_processed.nc`` or ``<input_stem>_velocity.nc``.
        velocity_only : bool, default False
            If ``True``, save only the U/V/W velocity components via
            ``velocity_to_netcdf()`` instead of the full dataset.
        velocity_units : str, default ``'cm/s'``
            Unit conversion for velocity-only output.
            Options: ``'mm/s'``, ``'cm/s'``, ``'m/s'``.
        ensure_depth_ascending : bool, default True
            Flip depth/cell dimension to ascending order in the output.
        print_summary : bool, default True
            Print the QC summary table to the console after saving.

        Returns
        -------
        Path
            Absolute path of the file that was written.

        Examples
        --------
        Full pipeline in four lines::

            proc = ProcessedDataset.from_file('config.ini')
            proc.apply_config('config.ini')
            output = proc.save_netcdf('config.ini', output_dir='processed/')
            print(f"Saved to {output}")

        Velocity-only output in m/s::

            proc.save_netcdf(
                'config.ini',
                velocity_only=True,
                velocity_units='m/s',
                output_dir='processed/',
            )
        """
        # Resolve config object for path defaults
        if isinstance(config, (str, Path)):
            cfg = ProcessingConfig.from_ini(str(config))
        else:
            cfg = config

        # ---- output directory ----
        if output_dir is not None:
            out_dir = Path(output_dir)
        elif cfg.output_file_path:
            out_dir = Path(cfg.output_file_path)
        elif cfg.input_file_path:
            out_dir = Path(cfg.input_file_path)
        else:
            out_dir = Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- output filename ----
        if output_filename is not None:
            fname = output_filename
        else:
            stem = Path(cfg.input_file_name).stem if cfg.input_file_name else "adcp"
            suffix = "_velocity.nc" if velocity_only else "_processed.nc"
            fname = stem + suffix

        output_path = out_dir / fname

        # ---- write ----
        if velocity_only:
            self.velocity_to_netcdf(
                output_path,
                units=velocity_units,
                ensure_depth_ascending=ensure_depth_ascending,
            )
        else:
            self.to_netcdf(
                output_path,
                ensure_depth_ascending=ensure_depth_ascending,
            )

        logger.info(f"Output saved to: {output_path}")

        if print_summary:
            self.print_summary()
            print(f"\nOutput saved to: {output_path}")

        return output_path.resolve()

    # ========================================================================
    # CONFIG EXPORT  (reproducibility)
    # ========================================================================

    @classmethod
    def from_ini(
        cls,
        filepath: Union[str, Path],
        binary_file_path: Optional[Union[str, Path]] = None,
    ) -> "ProcessedDataset":
        """
        Read an ADCP binary file from a config.ini and return a
        ready-to-process ``ProcessedDataset`` with the config stored.

        Convenience shorthand for::

            config = ProcessingConfig.from_ini(filepath)
            proc   = ProcessedDataset.from_file(config, binary_file_path)

        The stored config is immediately available for export without
        having to call ``apply_config()`` first.

        Parameters
        ----------
        filepath : str or Path
            Path to the ``config.ini`` file.
        binary_file_path : str or Path, optional
            Explicit ADCP binary file path.  If omitted the path is read
            from ``input_file_path`` / ``input_file_name`` in the INI file.

        Returns
        -------
        ProcessedDataset
            Initialised from the raw dataset (no processing applied yet).

        Examples
        --------
        One-liner full pipeline::

            result = (ProcessedDataset
                      .from_ini('config.ini')
                      .apply_config('config.ini')
                      .finalize())

        Reuse the stored config to avoid parsing the file twice::

            proc = ProcessedDataset.from_ini('config.ini')
            proc.apply_config(proc.config)
            proc.export_config('run_record.ini')
        """
        cfg = ProcessingConfig.from_ini(str(filepath))
        proc = cls.from_file(cfg, binary_file_path=binary_file_path)
        proc.config = cfg
        return proc

    def export_config(self, filepath: Union[str, Path]) -> None:
        """
        Save the configuration that was applied to a ``config.ini`` file.

        Captures every parameter used during processing — whether they came
        from ``apply_config()`` or from individual ``apply_*`` method calls.
        The resulting file can be reloaded with ``ProcessedDataset.from_ini()``
        or ``ProcessingConfig.from_ini()`` to reproduce or tweak the run.

        Parameters
        ----------
        filepath : str or Path
            Destination path for the INI file.

        Examples
        --------
        After a programmatic run, export for reproducibility::

            proc = ProcessedDataset(ds)
            proc.apply_sensor_health(roll=True, roll_threshold=20.0)
            proc.apply_signal_quality(correlation=70)
            proc.export_config('my_run.ini')

        After a config-driven run, save an identical record::

            proc = ProcessedDataset.from_ini('config.ini')
            proc.apply_config(proc.config)
            proc.export_config('run_record.ini')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.config.to_ini(str(filepath))
        logger.info(f"Configuration exported to {filepath}")

    def export_config_string(self) -> str:
        """
        Return the applied configuration as an INI-formatted string.

        Useful for Streamlit download buttons or in-memory inspection
        without writing to disk.

        Returns
        -------
        str
            INI-formatted configuration string.

        Examples
        --------
        Streamlit download button after processing::

            proc.apply_config('config.ini')

            import streamlit as st
            st.download_button(
                label="Download config.ini",
                data=proc.export_config_string(),
                file_name="config.ini",
                mime="text/plain",
            )

        Inspect what was actually run::

            print(proc.export_config_string())
        """
        return self.config.to_ini_string()

    def validate_config(self) -> List[str]:
        """
        Validate the current configuration for common issues.

        Delegates to ``ProcessingConfig.validate()``.  Returns an empty
        list if the configuration is valid.

        Returns
        -------
        list of str
            Warning / error messages.  Empty list means no issues found.

        Examples
        --------
        Validate before processing::

            proc = ProcessedDataset.from_ini('config.ini')
            issues = proc.validate_config()
            if issues:
                for issue in issues:
                    print(f"WARNING: {issue}")
            else:
                proc.apply_config(proc.config).finalize()
        """
        return self.config.validate()

    # ========================================================================
    # ADVANCED: DIRECT RUNNER ACCESS
    # ========================================================================

    def get_sensor_health_runner(self) -> "SensorHealthRunner":
        """
        Get SensorHealthRunner for advanced control.

        Returns
        -------
        SensorHealthRunner
            Runner initialized with current dataset.

        Examples
        --------
        >>> runner = proc.get_sensor_health_runner()
        >>> runner.roll(threshold=15.0).pitch(threshold=15.0)
        >>> proc.commit_runner(runner)
        """
        return SensorHealthRunner(self.dataset)

    def get_signal_quality_runner(self) -> "SignalQualityRunner":
        """
        Get SignalQualityRunner for advanced control.

        Returns
        -------
        SignalQualityRunner
            Runner initialized with current dataset.
        """
        return SignalQualityRunner(self.dataset)

    def get_percent_good_threshold(
        self,
        desired_std: float,
        depth_range: Optional[float] = None,
        n_pings: Optional[int] = None,
        bin_size: Optional[float] = None,
        frequency: Optional[int] = None,
        coefficients_path: Optional[str] = None,
    ) -> StdDevResult:
        """
        Advise a percent-good cutoff for a desired current precision.

        Reads ADCP parameters from the dataset (frequency, bin size, cell
        count, pings per ensemble) and uses the bundled exponential noise
        curves to find the minimum percent-good threshold that achieves
        *desired_std*.

        Parameters
        ----------
        desired_std : float
            Target standard deviation in cm/s.
        depth_range : float, optional
            Override depth range (m).  Default: ``bin_size × num_cells``.
            Use a shorter range if valid data does not span the full profile;
            data beyond this depth should be excluded from analysis.
        n_pings : int, optional
            Override pings per ensemble from the dataset.
        bin_size : float, optional
            Override bin size (m) from the dataset.
        frequency : int, optional
            Override ADCP frequency (kHz) from the dataset.
        coefficients_path : str, optional
            Path to a custom ``velocity_noise_coefficients.json``.

        Returns
        -------
        StdDevResult
            Advisory result.  Print it directly for a formatted summary.

        Raises
        ------
        ValueError
            If frequency or bin size has no exponential fit, or if required
            Fixed Leader fields are absent and no override is provided.

        Examples
        --------
        >>> proc = ProcessedDataset(ds)
        >>> result = proc.get_percent_good_threshold(desired_std=1.0)
        >>> print(result)
        >>> proc.apply_signal_quality(percent_good=result.percent_good_cutoff)
        """
        return compute_percent_good_threshold(
            self.dataset,
            desired_std=desired_std,
            depth_range=depth_range,
            n_pings=n_pings,
            bin_size=bin_size,
            frequency=frequency,
            coefficients_path=coefficients_path,
        )

    def get_profile_operation_runner(self) -> "ProfileOperationRunner":
        """
        Get ProfileOperationRunner for advanced control.

        Returns
        -------
        ProfileOperationRunner
            Runner initialized with current dataset.
        """
        return ProfileOperationRunner(self.dataset)

    def get_velocity_check_runner(self) -> "VelocityCheckRunner":
        """
        Get VelocityCheckRunner for advanced control.

        Returns
        -------
        VelocityCheckRunner
            Runner initialized with current dataset.
        """
        return VelocityCheckRunner(self.dataset)

    def commit_runner(
        self,
        runner: Union[
            "SensorHealthRunner",
            "SignalQualityRunner",
            "ProfileOperationRunner",
            "VelocityCheckRunner",
        ],
    ) -> ProcessedDataset:
        """
        Commit Runner changes back to ProcessedDataset.

        Parameters
        ----------
        runner : Runner instance
            Any of the Runner classes after processing.

        Returns
        -------
        ProcessedDataset
            Self for method chaining.

        Examples
        --------
        >>> runner = proc.get_signal_quality_runner()
        >>> runner.correlation(cutoff=64).echo_intensity(cutoff=40)
        >>> proc.commit_runner(runner)
        """
        self.dataset = runner.finalize()
        self.reports.append(runner.get_pipeline_report())
        self.processing_log.append(f"Committed {type(runner).__name__}")
        return self

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def reset(self) -> ProcessedDataset:
        """
        Reset to clean state (restore from original dataset).

        Clears all processing results and restores working dataset from original.
        Useful for parameter tuning and experimentation.

        Returns
        -------
        ProcessedDataset
            Self for method chaining.

        Examples
        --------
        >>> # Try different thresholds
        >>> for threshold in [50, 64, 70]:
        ...     proc.reset()
        ...     proc.apply_signal_quality(correlation=threshold)
        ...     print(f"threshold={threshold}: {proc.get_current_stats()['valid_pct']:.1f}%")
        """
        self.dataset = self.ds_orig.copy(deep=True)

        # Reinitialize mask
        if "mask" not in self.dataset.data_vars:
            self.dataset["mask"] = create_default_mask(self.dataset)

        # Clear statistics and config
        self.reports = []
        self.processing_log = []
        self._time_axis_results = {}
        self.config = ProcessingConfig()

        logger.info("ProcessedDataset reset to original state")
        return self

    def get_current_stats(self) -> Dict[str, Any]:
        """
        Get current processing statistics.

        Returns
        -------
        dict
            Dictionary with keys: total_cells, masked, masked_pct, valid, valid_pct
        """
        mask = self.dataset["mask"]
        total = int(mask.size)
        masked = int((mask == 1).sum())
        valid = total - masked

        return {
            "total_cells": total,
            "masked": masked,
            "masked_pct": 100 * masked / total if total > 0 else 0.0,
            "valid": valid,
            "valid_pct": 100 * valid / total if total > 0 else 0.0,
        }

    def to_netcdf(
        self,
        filepath: Union[str, Path],
        ensure_depth_ascending: bool = True,
    ) -> None:
        """
        Save finalized dataset to NetCDF file.

        Convenience method that calls finalize() and saves to file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        ensure_depth_ascending : bool, default True
            If True, ensures depth/cell coordinate is in ascending order
            by flipping arrays if necessary.
        """
        ds_out = self.finalize(ensure_depth_ascending=ensure_depth_ascending)
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        ds_out.to_netcdf(filepath)
        logger.info(f"Dataset saved to {filepath}")

    def velocity_to_netcdf(
        self,
        filepath: Union[str, Path],
        apply_mask: bool = True,
        include_coords: bool = True,
        include_metadata: bool = True,
        ensure_depth_ascending: bool = True,
        units: str = "cm/s",
        velocity_names: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Save only velocity components as separate 2D variables to NetCDF.

        Splits the 3D velocity array (beam, cell, time) into three separate
        2D arrays (cell, time) for zonal (u), meridional (v), and vertical (w)
        velocity components. Error velocity (beam 3) is excluded.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        apply_mask : bool, default True
            If True, apply the QC mask to velocities (masked values become NaN).
            If False, export raw velocity values without masking.
        include_coords : bool, default True
            If True, include time and cell/depth coordinates in output.
            If False, only include velocity data variables.
        include_metadata : bool, default True
            If True, include processing metadata in global attributes.
            If False, minimal attributes only.
        ensure_depth_ascending : bool, default True
            If True, ensures depth/cell coordinate is in ascending order
            by flipping arrays if necessary.
        units : str, default 'cm/s'
            Output velocity units. Options: 'mm/s', 'cm/s', 'm/s'.
            Data is converted from the original mm/s to the specified units.
        velocity_names : dict, optional
            Custom names for velocity variables. Default:
            {'u': 'zonal_velocity', 'v': 'meridional_velocity', 'w': 'vertical_velocity'}

        Returns
        -------
        None

        Notes
        -----
        - Original ADCP velocity is in mm/s, converted to specified units
        - The mask is applied per-component using beam indices 0, 1, 2
        - Error velocity (beam 3) is always excluded
        - Output file is CF-compliant with proper attributes
        - Depth is automatically ordered ascending unless ensure_depth_ascending=False

        Examples
        --------
        >>> # Basic usage - masked velocities in cm/s (default)
        >>> proc.velocity_to_netcdf("velocities.nc")

        >>> # Export in m/s
        >>> proc.velocity_to_netcdf("velocities.nc", units="m/s")

        >>> # Export in original mm/s
        >>> proc.velocity_to_netcdf("velocities.nc", units="mm/s")

        >>> # Custom variable names
        >>> proc.velocity_to_netcdf(
        ...     "velocities.nc",
        ...     velocity_names={'u': 'u_vel', 'v': 'v_vel', 'w': 'w_vel'}
        ... )
        """
        # Validate units
        valid_units = {"mm/s", "cm/s", "m/s"}
        if units not in valid_units:
            raise ValueError(f"Invalid units '{units}'. Must be one of: {valid_units}")

        # Unit conversion factors (from mm/s)
        unit_factors = {
            "mm/s": 1.0,
            "cm/s": 0.1,  # mm/s * 0.1 = cm/s
            "m/s": 0.001,  # mm/s * 0.001 = m/s
        }
        conversion_factor = unit_factors[units]

        # Default variable names
        if velocity_names is None:
            velocity_names = {
                "u": "zonal_velocity",
                "v": "meridional_velocity",
                "w": "vertical_velocity",
            }

        # Finalize to ensure metadata is current (with depth ordering)
        ds_final = self.finalize(ensure_depth_ascending=ensure_depth_ascending)

        # Extract velocity and mask
        if "velocity" not in ds_final.data_vars:
            raise ValueError("Dataset does not contain 'velocity' variable")

        velocity = ds_final["velocity"]
        mask = ds_final["mask"] if "mask" in ds_final.data_vars else None

        # Determine dimensions
        # Velocity is typically (beam, cell, time) or (beam, cell, ensemble)
        vel_dims = velocity.dims
        if len(vel_dims) != 3:
            raise ValueError(
                f"Expected 3D velocity array, got {len(vel_dims)}D with dims {vel_dims}"
            )

        # Find the time/ensemble dimension (not beam, not cell)
        time_dim = None
        cell_dim = None
        for dim in vel_dims:
            if dim in ["time", "ensemble"]:
                time_dim = dim
            elif dim in ["cell", "depth"]:
                cell_dim = dim

        if time_dim is None:
            # Assume last dimension is time if not found
            time_dim = vel_dims[-1]
        if cell_dim is None:
            # Assume middle dimension is cell if not found
            cell_dim = vel_dims[1]

        # Extract individual components (beam 0=u, 1=v, 2=w)
        u_data = velocity.isel(beam=0).values.astype(np.float32)  # Zonal (East)
        v_data = velocity.isel(beam=1).values.astype(np.float32)  # Meridional (North)
        w_data = velocity.isel(beam=2).values.astype(np.float32)  # Vertical

        # Apply mask if requested
        if apply_mask and mask is not None:
            # Get per-component masks
            u_mask = mask.isel(beam=0).values
            v_mask = mask.isel(beam=1).values
            w_mask = mask.isel(beam=2).values

            # Apply masks (1 = invalid -> NaN)
            u_data = np.where(u_mask == 1, np.nan, u_data)
            v_data = np.where(v_mask == 1, np.nan, v_data)
            w_data = np.where(w_mask == 1, np.nan, w_data)

        # Apply unit conversion
        u_data = u_data * conversion_factor
        v_data = v_data * conversion_factor
        w_data = w_data * conversion_factor

        # Build output dimensions (2D: cell/depth x time)
        out_dims = (cell_dim, time_dim)

        # Get coordinates
        coords = {}
        if include_coords:
            if time_dim in ds_final.coords:
                coords[time_dim] = ds_final.coords[time_dim]
            if cell_dim in ds_final.coords:
                coords[cell_dim] = ds_final.coords[cell_dim]
            # Include depth coordinate if available and different from cell_dim
            if "depth" in ds_final.coords and "depth" != cell_dim:
                coords["depth"] = ds_final.coords["depth"]

        # Create DataArrays for each component
        u_da = xr.DataArray(
            data=u_data,
            dims=out_dims,
            coords={k: v for k, v in coords.items() if k in out_dims},
            attrs={
                "long_name": "Zonal velocity (eastward)",
                "standard_name": "eastward_sea_water_velocity",
                "units": units,
                "positive": "eastward",
                "comment": "U component, positive eastward",
            },
        )

        v_da = xr.DataArray(
            data=v_data,
            dims=out_dims,
            coords={k: v for k, v in coords.items() if k in out_dims},
            attrs={
                "long_name": "Meridional velocity (northward)",
                "standard_name": "northward_sea_water_velocity",
                "units": units,
                "positive": "northward",
                "comment": "V component, positive northward",
            },
        )

        w_da = xr.DataArray(
            data=w_data,
            dims=out_dims,
            coords={k: v for k, v in coords.items() if k in out_dims},
            attrs={
                "long_name": "Vertical velocity (upward)",
                "standard_name": "upward_sea_water_velocity",
                "units": units,
                "positive": "upward",
                "comment": "W component, positive upward",
            },
        )

        # Build output dataset
        ds_out = xr.Dataset(
            {
                velocity_names["u"]: u_da,
                velocity_names["v"]: v_da,
                velocity_names["w"]: w_da,
            },
            coords=coords,
        )

        # Add global attributes
        if include_metadata:
            ds_out.attrs = {
                "title": "ADCP Velocity Components",
                "institution": ds_final.attrs.get("institution", ""),
                "source": f"pyadps v{_PYADPS_VERSION}",
                "history": f"Created {datetime.now(timezone.utc).isoformat()}",
                "references": "pyadps ADCP processing package",
                "Conventions": "CF-1.8",
                "processing_log": str(self.processing_log),  # Convert list to string
                "mask_applied": int(apply_mask),  # Convert bool to int (0 or 1)
                "original_velocity_shape": str(
                    list(velocity.shape)
                ),  # Convert to string
                "original_velocity_units": "mm/s",
                "output_velocity_units": units,
                "components_exported": "u (zonal), v (meridional), w (vertical)",
                "error_velocity_excluded": 1,  # Use int instead of bool
            }

            # Copy relevant attributes from original
            for attr in [
                "deployment_name",
                "instrument_type",
                "serial_number",
                "latitude",
                "longitude",
                "water_depth",
            ]:
                if attr in ds_final.attrs:
                    ds_out.attrs[attr] = ds_final.attrs[attr]
        else:
            ds_out.attrs = {
                "source": f"pyadps v{_PYADPS_VERSION}",
                "Conventions": "CF-1.8",
            }

        # Save to file
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        ds_out.to_netcdf(filepath)

        # Log summary
        n_valid_u = int(np.sum(~np.isnan(u_data)))
        n_valid_v = int(np.sum(~np.isnan(v_data)))
        n_valid_w = int(np.sum(~np.isnan(w_data)))
        total_cells = u_data.size

        logger.info(
            f"Velocity components saved to {filepath}: "
            f"shape={u_data.shape}, units={units}, "
            f"valid cells: u={n_valid_u:,} ({100*n_valid_u/total_cells:.1f}%), "
            f"v={n_valid_v:,} ({100*n_valid_v/total_cells:.1f}%), "
            f"w={n_valid_w:,} ({100*n_valid_w/total_cells:.1f}%)"
        )

    def get_velocity_dataset(
        self,
        apply_mask: bool = True,
        ensure_depth_ascending: bool = True,
        units: str = "cm/s",
        velocity_names: Optional[Dict[str, str]] = None,
    ) -> xr.Dataset:
        """
        Get velocity components as a separate xarray Dataset (without saving).

        Similar to velocity_to_netcdf but returns the Dataset instead of saving.
        Useful for further analysis or custom export.

        Parameters
        ----------
        apply_mask : bool, default True
            If True, apply the QC mask to velocities (masked values become NaN).
        ensure_depth_ascending : bool, default True
            If True, ensures depth/cell coordinate is in ascending order.
        units : str, default 'cm/s'
            Output velocity units. Options: 'mm/s', 'cm/s', 'm/s'.
            Data is converted from the original mm/s to the specified units.
        velocity_names : dict, optional
            Custom names for velocity variables.

        Returns
        -------
        xr.Dataset
            Dataset containing zonal, meridional, and vertical velocity as
            separate 2D variables.

        Examples
        --------
        >>> vel_ds = proc.get_velocity_dataset()
        >>> print(vel_ds)
        >>> vel_ds['zonal_velocity'].plot()

        >>> # Get velocity in m/s
        >>> vel_ds = proc.get_velocity_dataset(units="m/s")
        """
        # Validate units
        valid_units = {"mm/s", "cm/s", "m/s"}
        if units not in valid_units:
            raise ValueError(f"Invalid units '{units}'. Must be one of: {valid_units}")

        # Unit conversion factors (from mm/s)
        unit_factors = {
            "mm/s": 1.0,
            "cm/s": 0.1,  # mm/s * 0.1 = cm/s
            "m/s": 0.001,  # mm/s * 0.001 = m/s
        }
        conversion_factor = unit_factors[units]

        # Default variable names
        if velocity_names is None:
            velocity_names = {
                "u": "zonal_velocity",
                "v": "meridional_velocity",
                "w": "vertical_velocity",
            }

        ds_final = self.finalize(ensure_depth_ascending=ensure_depth_ascending)
        velocity = ds_final["velocity"]
        mask = ds_final["mask"] if "mask" in ds_final.data_vars else None

        # Determine dimensions
        vel_dims = velocity.dims
        time_dim = None
        cell_dim = None
        for dim in vel_dims:
            if dim in ["time", "ensemble"]:
                time_dim = dim
            elif dim in ["cell", "depth"]:
                cell_dim = dim

        if time_dim is None:
            time_dim = vel_dims[-1]
        if cell_dim is None:
            cell_dim = vel_dims[1]

        # Extract components
        u_data = velocity.isel(beam=0).values.astype(np.float32)
        v_data = velocity.isel(beam=1).values.astype(np.float32)
        w_data = velocity.isel(beam=2).values.astype(np.float32)

        # Apply mask if requested
        if apply_mask and mask is not None:
            u_mask = mask.isel(beam=0).values
            v_mask = mask.isel(beam=1).values
            w_mask = mask.isel(beam=2).values

            u_data = np.where(u_mask == 1, np.nan, u_data)
            v_data = np.where(v_mask == 1, np.nan, v_data)
            w_data = np.where(w_mask == 1, np.nan, w_data)

        # Apply unit conversion
        u_data = u_data * conversion_factor
        v_data = v_data * conversion_factor
        w_data = w_data * conversion_factor

        out_dims = (cell_dim, time_dim)

        # Get coordinates
        coords = {}
        if time_dim in ds_final.coords:
            coords[time_dim] = ds_final.coords[time_dim]
        if cell_dim in ds_final.coords:
            coords[cell_dim] = ds_final.coords[cell_dim]

        # Create dataset
        ds_out = xr.Dataset(
            {
                velocity_names["u"]: xr.DataArray(
                    data=u_data,
                    dims=out_dims,
                    attrs={"long_name": "Zonal velocity", "units": units},
                ),
                velocity_names["v"]: xr.DataArray(
                    data=v_data,
                    dims=out_dims,
                    attrs={"long_name": "Meridional velocity", "units": units},
                ),
                velocity_names["w"]: xr.DataArray(
                    data=w_data,
                    dims=out_dims,
                    attrs={"long_name": "Vertical velocity", "units": units},
                ),
            },
            coords=coords,
        )

        return ds_out

    # ========================================================================
    # REPORTING
    # ========================================================================

    def get_pipeline_report(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline report.

        Returns
        -------
        dict
            Dictionary containing all processing statistics.
        """
        current = self.get_current_stats()

        return {
            "summary": {
                "total_cells": self._total_cells,
                "baseline_masked": self._baseline_masked,
                "baseline_masked_pct": round(self._baseline_masked_pct, 4),
                "final_masked": current["masked"],
                "final_masked_pct": round(current["masked_pct"], 4),
                "final_valid": current["valid"],
                "final_valid_pct": round(current["valid_pct"], 4),
                "processing_impact_pct": round(
                    current["masked_pct"] - self._baseline_masked_pct, 4
                ),
            },
            "processing_log": self.processing_log,
            "time_axis": self._time_axis_results,
            "step_reports": [r.to_dict() for r in self.reports],
        }

    def export_statistics(self) -> Dict[str, Any]:
        """
        Export all statistics as JSON-serializable dictionary.

        Returns
        -------
        dict
            Complete statistics dictionary.
        """
        return self.get_pipeline_report()

    def export_report_json(self, filepath: Union[str, Path]) -> None:
        """Export statistics to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.export_statistics(), f, indent=2, default=str)
        logger.info(f"Report exported to {filepath}")

    def export_report_markdown(self, filepath: Union[str, Path]) -> None:
        """Export statistics to Markdown file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        report = self.get_pipeline_report()
        summary = report["summary"]

        lines = [
            "# ADCP Processing Report",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Summary",
            "",
            f"- **Total cells:** {summary['total_cells']:,}",
            f"- **Baseline masked:** {summary['baseline_masked']:,} ({summary['baseline_masked_pct']:.2f}%)",
            f"- **Final masked:** {summary['final_masked']:,} ({summary['final_masked_pct']:.2f}%)",
            f"- **Final valid:** {summary['final_valid']:,} ({summary['final_valid_pct']:.2f}%)",
            f"- **Processing impact:** {summary['processing_impact_pct']:+.2f}%",
            "",
            "## Processing Steps",
            "",
        ]

        for i, step in enumerate(report["processing_log"], 1):
            lines.append(f"{i}. {step}")

        lines.append("")
        lines.append("## Detailed Statistics")
        lines.append("")

        for step_report in report["step_reports"]:
            lines.append(f"### {step_report.get('module_name', 'Unknown')}")
            lines.append("")
            if "checks" in step_report:
                for check in step_report["checks"]:
                    lines.append(
                        f"- **{check['check_name']}**: "
                        f"threshold={check['threshold']}, "
                        f"impact={check['newly_masked_pct']:.2f}%"
                    )
            lines.append("")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Markdown report exported to {filepath}")

    def print_summary(self) -> None:
        """Print formatted processing summary to console."""
        report = self.get_pipeline_report()
        summary = report["summary"]

        width = 80
        print("=" * width)
        print("ADCP PROCESSING SUMMARY")
        print("=" * width)
        print(f"Total cells: {summary['total_cells']:,}")
        print(
            f"Baseline masked: {summary['baseline_masked']:,} "
            f"({summary['baseline_masked_pct']:.2f}%)"
        )
        print("-" * width)
        print("PROCESSING STEPS:")

        for i, step in enumerate(report["processing_log"], 1):
            print(f"  {i}. {step}")

        print("-" * width)

        # Print step details
        for step_report in report["step_reports"]:
            module = step_report.get("module_name", "Unknown")
            print(f"\n{module.upper()}:")
            print(
                f"  {'Check':<25} | {'Threshold':>12} | "
                f"{'Impact':>10} | {'Cumulative':>10}"
            )
            print("  " + "-" * 65)

            if "checks" in step_report:
                for check in step_report["checks"]:
                    threshold = check.get("threshold", "N/A")
                    if isinstance(threshold, dict):
                        threshold = str(threshold)
                    elif isinstance(threshold, float):
                        threshold = f"{threshold:.1f}"
                    print(
                        f"  {check['check_name']:<25} | {str(threshold):>12} | "
                        f"{check['newly_masked_pct']:>9.2f}% | "
                        f"{check['total_masked_pct']:>9.2f}%"
                    )

        print("-" * width)
        print(
            f"FINAL: {summary['final_valid']:,} valid cells "
            f"({summary['final_valid_pct']:.2f}%) | "
            f"Processing impact: {summary['processing_impact_pct']:+.2f}%"
        )
        print("=" * width)

    def summary(self) -> str:
        """
        Generate human-readable summary string.

        Returns
        -------
        str
            Formatted summary string.
        """
        import io
        import sys

        # Capture print_summary output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.print_summary()
        sys.stdout = old_stdout

        return buffer.getvalue()
