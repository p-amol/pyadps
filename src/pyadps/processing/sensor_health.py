"""
Sensor Health Module for ADCP Data Processing (v1.0.0).

This module provides environmental sensor validation functions for Acoustic Doppler
Current Profiler (ADCP) data processing.

Mask Convention:
- 0 = valid data
- 1 = invalid/flagged data

Functions:
- roll_check: Check roll sensor values against threshold
- pitch_check: Check pitch sensor values against threshold
- correct_sound_speed: Calculate and replace sound speed from T/S/D
- replace_data: Replace variables with user-provided data

Classes:
- SensorHealthRunner: Orchestrator class with method chaining
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .utility import (
    create_default_mask,
    replace_data,
    QCCheckStats,
    DataModificationStats,
    QCPipelineReport,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_ROLL_THRESHOLD: float = 15.0  # degrees
DEFAULT_PITCH_THRESHOLD: float = 15.0  # degrees
VELOCITY_MISSING_VALUE: int = -32768


# ============================================================================
# ROLL CHECK FUNCTION
# ============================================================================


def roll_check(
    ds: xr.Dataset,
    threshold: float = DEFAULT_ROLL_THRESHOLD,
) -> xr.Dataset:
    """
    Check roll sensor values against threshold and update mask.

    This function checks if roll values exceed the specified threshold and
    updates the mask accordingly. Roll values are converted from instrument
    units (0.01 degrees) to degrees before comparison.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset. Must contain 'roll' variable.
        May contain 'mask' variable with dims (beam, cell, time).
        If 'mask' is not present but 'velocity' is, a default mask will
        be created using create_default_mask().
    threshold : float, default 15.0
        Maximum acceptable roll in degrees (absolute value).
        Ensembles where |roll| > threshold will be flagged.

    Returns
    -------
    xr.Dataset
        New dataset with updated mask. Original dataset is not modified.

    Raises
    ------
    ValueError
        If 'roll' variable is not found in the dataset.
    ValueError
        If neither 'mask' nor 'velocity' is found in the dataset.
    """
    # Step 1: Validate roll data exists
    if "roll" not in ds.data_vars:
        raise ValueError(
            "Roll data not found in dataset. "
            "Dataset must contain 'roll' variable to perform roll check."
        )

    # Step 2: Get or create mask
    if "mask" in ds.data_vars:
        mask = ds["mask"]
        logger.debug("Using existing mask from dataset")
    elif "velocity" in ds.data_vars:
        logger.debug("No mask found, creating default mask from velocity data")
        mask = create_default_mask(ds)
    else:
        raise ValueError(
            "Cannot perform roll check: dataset contains neither 'mask' nor "
            "'velocity' variable. Provide a dataset with mask data or velocity "
            "data to create a default mask."
        )

    # Step 3: Get roll data and convert units
    roll_data = ds["roll"]
    roll_degrees = roll_data * 0.01  # Convert from 0.01° to degrees

    # Step 4: Determine time dimension
    time_dim = "time" if "time" in ds.dims else "ensemble"

    # Step 5: Apply roll threshold check
    roll_flag = np.abs(roll_degrees) > threshold

    mask_updated = xr.where(roll_flag, 1, mask).astype(np.int8)
    mask_updated = mask_updated.transpose("beam", "cell", time_dim)
    mask_updated.attrs = mask.attrs.copy()

    # Step 6: Create new dataset with updated mask
    ds_out = ds.copy(deep=True)
    ds_out["mask"] = mask_updated

    # Step 7: Log statistics
    flagged_ensembles = int(roll_flag.sum())
    total_ensembles = roll_data.sizes[time_dim]
    flagged_pct = (
        100 * flagged_ensembles / total_ensembles if total_ensembles > 0 else 0
    )

    logger.info(
        f"Roll check applied: threshold={threshold}°, "
        f"flagged={flagged_ensembles}/{total_ensembles} ensembles ({flagged_pct:.1f}%)"
    )

    return ds_out


# ============================================================================
# PITCH CHECK FUNCTION
# ============================================================================


def pitch_check(
    ds: xr.Dataset,
    threshold: float = DEFAULT_PITCH_THRESHOLD,
) -> xr.Dataset:
    """
    Check pitch sensor values against threshold and update mask.

    This function checks if pitch values exceed the specified threshold and
    updates the mask accordingly. Pitch values are converted from instrument
    units (0.01 degrees) to degrees before comparison.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset. Must contain 'pitch' variable.
        May contain 'mask' variable with dims (beam, cell, time).
        If 'mask' is not present but 'velocity' is, a default mask will
        be created using create_default_mask().
    threshold : float, default 15.0
        Maximum acceptable pitch in degrees (absolute value).
        Ensembles where |pitch| > threshold will be flagged.

    Returns
    -------
    xr.Dataset
        New dataset with updated mask. Original dataset is not modified.

    Raises
    ------
    ValueError
        If 'pitch' variable is not found in the dataset.
    ValueError
        If neither 'mask' nor 'velocity' is found in the dataset.
    """
    # Step 1: Validate pitch data exists
    if "pitch" not in ds.data_vars:
        raise ValueError(
            "Pitch data not found in dataset. "
            "Dataset must contain 'pitch' variable to perform pitch check."
        )

    # Step 2: Get or create mask
    if "mask" in ds.data_vars:
        mask = ds["mask"]
        logger.debug("Using existing mask from dataset")
    elif "velocity" in ds.data_vars:
        logger.debug("No mask found, creating default mask from velocity data")
        mask = create_default_mask(ds)
    else:
        raise ValueError(
            "Cannot perform pitch check: dataset contains neither 'mask' nor "
            "'velocity' variable. Provide a dataset with mask data or velocity "
            "data to create a default mask."
        )

    # Step 3: Get pitch data and convert units
    pitch_data = ds["pitch"]
    pitch_degrees = pitch_data * 0.01  # Convert from 0.01° to degrees

    # Step 4: Determine time dimension
    time_dim = "time" if "time" in ds.dims else "ensemble"

    # Step 5: Apply pitch threshold check
    pitch_flag = np.abs(pitch_degrees) > threshold

    mask_updated = xr.where(pitch_flag, 1, mask).astype(np.int8)
    mask_updated = mask_updated.transpose("beam", "cell", time_dim)
    mask_updated.attrs = mask.attrs.copy()

    # Step 6: Create new dataset with updated mask
    ds_out = ds.copy(deep=True)
    ds_out["mask"] = mask_updated

    # Step 7: Log statistics
    flagged_ensembles = int(pitch_flag.sum())
    total_ensembles = pitch_data.sizes[time_dim]
    flagged_pct = (
        100 * flagged_ensembles / total_ensembles if total_ensembles > 0 else 0
    )

    logger.info(
        f"Pitch check applied: threshold={threshold}°, "
        f"flagged={flagged_ensembles}/{total_ensembles} ensembles ({flagged_pct:.1f}%)"
    )

    return ds_out


# ============================================================================
# SOUND SPEED CORRECTION FUNCTION
# ============================================================================


def correct_sound_speed(
    ds: xr.Dataset,
    correct_velocity: bool = True,
    horizontal_only: bool = True,
) -> xr.Dataset:
    """
    Calculate corrected sound speed and optionally correct velocity.

    This function calculates the theoretical sound speed using the Urick (1983)
    empirical equation based on temperature, salinity, and transducer depth
    from the dataset. The calculated sound speed replaces the original
    instrument-recorded sound speed.

    If correct_velocity=True (default), the velocity is also corrected using
    the ratio of original to corrected sound speed.

    If temperature or salinity need to be updated before this calculation,
    use the replace_data() function first.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset. Must contain:
        - temperature : Temperature in 0.01°C units
        - salinity : Salinity in 0.1 PSU units
        - transducer_depth : Transducer depth in 0.1 m units
        - sound_speed : Original sound speed (will be replaced)
        - velocity : Velocity data (required if correct_velocity=True)
    correct_velocity : bool, default True
        If True, also correct velocity using the sound speed ratio.
        If False, only correct the sound_speed variable.
    horizontal_only : bool, default True
        If True, correct only horizontal velocities (u, v - beams 0, 1).
        If False, also correct vertical velocity (w - beam 2).
        Error velocity (beam 3) is never corrected.
        Only used when correct_velocity=True.

    Returns
    -------
    xr.Dataset
        New dataset with corrected sound_speed (and optionally velocity).
        Original dataset is not modified.

    Raises
    ------
    ValueError
        If required variables are not found in the dataset.

    Notes
    -----
    Sound speed formula (Urick, 1983):
        c = 1449.2 + 4.6T - 0.055T² + 0.00029T³
            + (1.34 - 0.01T)(S - 35) + 0.016D

    Where:
        c = sound speed (m/s)
        T = temperature (°C)
        S = salinity (PSU)
        D = depth (m)

    The velocity correction formula is:
        V_corrected = V_original × (C_original / C_corrected)

    The input variables are expected in raw instrument units:
        - temperature: 0.01°C (divide by 100 to get °C)
        - salinity: 0.1 PSU (divide by 10 to get PSU)
        - transducer_depth: 0.1 m (divide by 10 to get m)

    Missing velocity values (-32768) are preserved.

    Examples
    --------
    >>> ds = read('file.000')
    >>> # Correct sound speed and velocity (default)
    >>> ds_corrected = correct_sound_speed(ds)

    >>> # Optionally update T/S first with CTD data
    >>> ds = replace_data(ds, ctd_temp, "temperature")
    >>> ds = replace_data(ds, ctd_sal, "salinity")
    >>> ds_corrected = correct_sound_speed(ds)

    >>> # Correct sound speed only, not velocity
    >>> ds_corrected = correct_sound_speed(ds, correct_velocity=False)

    >>> # Correct all velocity components including W
    >>> ds_corrected = correct_sound_speed(ds, horizontal_only=False)

    See Also
    --------
    replace_data : Replace variables with external data
    """
    # Step 1: Validate required variables exist
    required_vars = ["temperature", "salinity", "transducer_depth", "sound_speed"]
    if correct_velocity:
        required_vars.append("velocity")

    missing_vars = [var for var in required_vars if var not in ds.data_vars]

    if missing_vars:
        raise ValueError(
            f"Missing required variables: {missing_vars}. "
            f"Dataset must contain: {required_vars}"
        )

    # Step 2: Save original sound speed for velocity correction
    original_sound_speed = ds["sound_speed"].values.copy()

    # Step 3: Extract and convert variables to physical units
    # Temperature: 0.01°C -> °C
    temperature = ds["temperature"].values * 0.01

    # Salinity: 0.1 PSU -> PSU
    salinity = ds["salinity"].values * 0.1

    # Depth: 0.1 m -> m
    depth = ds["transducer_depth"].values * 0.1

    # Step 4: Calculate corrected sound speed using Urick (1983) equation
    sound_speed_corrected = (
        1449.2
        + 4.6 * temperature
        - 0.055 * temperature**2
        + 0.00029 * temperature**3
        + (1.34 - 0.01 * temperature) * (salinity - 35)
        + 0.016 * depth
    )

    # Step 5: Create new dataset
    ds_out = ds.copy(deep=True)

    # Step 6: Update sound_speed
    # Preserve original attributes
    ss_attrs = ds["sound_speed"].attrs.copy()

    # Determine time dimension
    time_dim = "time" if "time" in ds.dims else "ensemble"

    # Create new DataArray for sound_speed
    sound_speed_da = xr.DataArray(
        data=sound_speed_corrected,
        dims=[time_dim],
        coords={time_dim: ds.coords[time_dim]},
        attrs=ss_attrs,
    )

    # Add correction metadata
    sound_speed_da.attrs["corrected"] = 1
    sound_speed_da.attrs["correction_method"] = "Urick (1983)"

    ds_out["sound_speed"] = sound_speed_da

    # Log sound speed statistics
    original_ss_mean = float(original_sound_speed.mean())
    corrected_ss_mean = float(sound_speed_corrected.mean())
    diff_mean = corrected_ss_mean - original_ss_mean

    logger.info(
        f"Sound speed corrected: original_mean={original_ss_mean:.2f} m/s, "
        f"corrected_mean={corrected_ss_mean:.2f} m/s, difference={diff_mean:+.2f} m/s"
    )

    # Step 7: Optionally correct velocity
    if correct_velocity:
        # Calculate correction ratio: original / corrected
        ratio = original_sound_speed / sound_speed_corrected

        # Get velocity data
        velocity = ds["velocity"]
        vel_values = velocity.values.copy()  # Shape: (beam, cell, time)

        # Broadcast ratio to match velocity shape: (time,) -> (1, 1, time)
        ratio_broadcast = ratio[np.newaxis, np.newaxis, :]

        # Correct U component (beam 0)
        vel_values[0, :, :] = np.where(
            vel_values[0, :, :] <= VELOCITY_MISSING_VALUE,
            VELOCITY_MISSING_VALUE,
            vel_values[0, :, :] * ratio_broadcast[0],
        )

        # Correct V component (beam 1)
        vel_values[1, :, :] = np.where(
            vel_values[1, :, :] <= VELOCITY_MISSING_VALUE,
            VELOCITY_MISSING_VALUE,
            vel_values[1, :, :] * ratio_broadcast[0],
        )

        # Optionally correct W component (beam 2)
        if not horizontal_only:
            vel_values[2, :, :] = np.where(
                vel_values[2, :, :] <= VELOCITY_MISSING_VALUE,
                VELOCITY_MISSING_VALUE,
                vel_values[2, :, :] * ratio_broadcast[0],
            )

        # Preserve original velocity attributes
        vel_attrs = velocity.attrs.copy()

        # Create new DataArray for velocity
        velocity_da = xr.DataArray(
            data=vel_values,
            dims=velocity.dims,
            coords=velocity.coords,
            attrs=vel_attrs,
        )

        # Add correction metadata
        velocity_da.attrs["sound_speed_corrected"] = 1
        velocity_da.attrs["horizontal_only"] = horizontal_only

        ds_out["velocity"] = velocity_da

        # Log velocity correction statistics
        mean_ratio = float(ratio.mean())
        logger.info(
            f"Velocity sound speed correction applied: "
            f"mean_ratio={mean_ratio:.4f}, horizontal_only={horizontal_only}"
        )

    return ds_out


# ============================================================================
# SENSOR HEALTH RUNNER CLASS
# ============================================================================


class SensorHealthRunner:
    """
    Orchestrator class for sensor health checks with method chaining.

    This class wraps the sensor health functions and provides a fluent
    interface for applying multiple checks in sequence. It maintains
    processing history, statistics, and reports for reproducibility.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset to process.

    Attributes
    ----------
    dataset : xr.Dataset
        Current working dataset (updated after each operation).
    original : xr.Dataset
        Original unmodified dataset (preserved for reference).
    history : list[dict]
        List of processing steps with parameters and timestamps.
    statistics : list[QCCheckStats]
        Statistics for each QC check applied.
    modifications : list[DataModificationStats]
        Statistics for each data modification applied.
    baseline_masked : int
        Number of cells masked before any sensor health checks.
    total_cells : int
        Total number of cells in the mask.

    Examples
    --------
    >>> ds = read('file.000')
    >>> runner = SensorHealthRunner(ds)
    >>> result = (runner
    ...     .replace_data(ctd_temp, "temperature")
    ...     .replace_data(ctd_sal, "salinity")
    ...     .correct_sound_speed()
    ...     .roll_check(threshold=20.0)
    ...     .pitch_check(threshold=15.0)
    ...     .finalize())

    >>> # Access statistics
    >>> runner.print_statistics()
    >>> report = runner.get_pipeline_report()

    See Also
    --------
    roll_check : Check roll sensor values
    pitch_check : Check pitch sensor values
    correct_sound_speed : Calculate corrected sound speed
    replace_data : Replace variables with external data
    """

    def __init__(self, ds: xr.Dataset) -> None:
        """
        Initialize SensorHealthRunner with dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Input xarray Dataset to process.
        """
        self.original: xr.Dataset = ds.copy(deep=True)
        self.dataset: xr.Dataset = ds.copy(deep=True)
        self.history: list[dict[str, Any]] = []
        self.statistics: list[QCCheckStats] = []
        self.modifications: list[DataModificationStats] = []

        # Calculate baseline statistics
        if "mask" in ds.data_vars:
            mask = ds["mask"].values
            self.total_cells = int(mask.size)
            self.baseline_masked = int((mask == 1).sum())
        elif "velocity" in ds.data_vars:
            vel = ds["velocity"]
            self.total_cells = int(vel.size)
            self.baseline_masked = 0
        else:
            self.total_cells = 0
            self.baseline_masked = 0

        self.baseline_masked_pct = (
            100 * self.baseline_masked / self.total_cells
            if self.total_cells > 0
            else 0.0
        )

        logger.info(
            f"SensorHealthRunner initialized: {self.total_cells:,} cells, "
            f"{self.baseline_masked:,} baseline masked ({self.baseline_masked_pct:.2f}%)"
        )

    def _add_history(
        self,
        operation: str,
        parameters: dict[str, Any],
        stats: dict[str, Any] | None = None,
    ) -> None:
        """Add an entry to the processing history."""
        entry = {
            "operation": operation,
            "parameters": parameters,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if stats:
            entry["stats"] = stats
        self.history.append(entry)

    def _compute_var_stats(self, var: xr.DataArray) -> dict[str, float]:
        """Compute basic statistics for a variable."""
        values = var.values.flatten()
        # Filter out NaN and missing values
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return {
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
            }
        return {
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
        }

    def replace_data(
        self,
        data: NDArray[Any],
        variable_name: str,
        apply_scale_factor: bool = True,
    ) -> SensorHealthRunner:
        """
        Replace a variable in the dataset with user-provided data.

        Parameters
        ----------
        data : NDArray
            New data array to replace the existing variable values.
            Data should be in physical units (e.g., °C for temperature).
        variable_name : str
            Name of the variable to replace.
        apply_scale_factor : bool, default True
            If True and the variable has a 'scale_factor' attribute,
            divide input data by scale factor to convert to RDI format.

        Returns
        -------
        SensorHealthRunner
            Self for method chaining.

        See Also
        --------
        replace_data : Underlying function
        """
        # Capture original stats
        original_stats = self._compute_var_stats(self.dataset[variable_name])

        self.dataset = replace_data(
            self.dataset, data, variable_name, apply_scale_factor=apply_scale_factor
        )

        # Capture modified stats
        modified_stats = self._compute_var_stats(self.dataset[variable_name])

        # Create modification stats
        mod_stat = DataModificationStats(
            operation="replace_data",
            variable_name=variable_name,
            original_stats=original_stats,
            modified_stats=modified_stats,
            metadata={"data_shape": list(data.shape)},
        )
        self.modifications.append(mod_stat)

        self._add_history(
            operation="replace_data",
            parameters={"variable_name": variable_name, "data_shape": list(data.shape)},
            stats={
                "original_mean": round(original_stats["mean"], 4),
                "modified_mean": round(modified_stats["mean"], 4),
                "mean_change": round(
                    modified_stats["mean"] - original_stats["mean"], 4
                ),
            },
        )

        return self

    def correct_sound_speed(
        self,
        correct_velocity: bool = True,
        horizontal_only: bool = True,
    ) -> SensorHealthRunner:
        """
        Calculate corrected sound speed and optionally correct velocity.

        Parameters
        ----------
        correct_velocity : bool, default True
            If True, also correct velocity using the sound speed ratio.
        horizontal_only : bool, default True
            If True, correct only horizontal velocities (u, v).
            Only used when correct_velocity=True.

        Returns
        -------
        SensorHealthRunner
            Self for method chaining.

        See Also
        --------
        correct_sound_speed : Underlying function
        """
        # Capture before stats for sound_speed
        original_ss_stats = self._compute_var_stats(self.dataset["sound_speed"])

        # Capture before stats for velocity if being corrected
        original_vel_stats = None
        if correct_velocity:
            original_vel_stats = self._compute_var_stats(self.dataset["velocity"])

        self.dataset = correct_sound_speed(
            self.dataset,
            correct_velocity=correct_velocity,
            horizontal_only=horizontal_only,
        )

        # Capture after stats
        modified_ss_stats = self._compute_var_stats(self.dataset["sound_speed"])

        # Create modification stats for sound_speed
        ss_mod = DataModificationStats(
            operation="correct_sound_speed",
            variable_name="sound_speed",
            original_stats=original_ss_stats,
            modified_stats=modified_ss_stats,
            metadata={
                "correction_method": "Urick (1983)",
                "correct_velocity": correct_velocity,
                "horizontal_only": horizontal_only,
            },
        )
        self.modifications.append(ss_mod)

        # Create modification stats for velocity if corrected
        if correct_velocity and original_vel_stats:
            modified_vel_stats = self._compute_var_stats(self.dataset["velocity"])
            vel_mod = DataModificationStats(
                operation="correct_sound_speed",
                variable_name="velocity",
                original_stats=original_vel_stats,
                modified_stats=modified_vel_stats,
                metadata={
                    "correction_reason": "sound_speed_ratio",
                    "horizontal_only": horizontal_only,
                },
            )
            self.modifications.append(vel_mod)

        self._add_history(
            operation="correct_sound_speed",
            parameters={
                "correct_velocity": correct_velocity,
                "horizontal_only": horizontal_only,
            },
            stats={
                "original_sound_speed_mean": round(original_ss_stats["mean"], 2),
                "corrected_sound_speed_mean": round(modified_ss_stats["mean"], 2),
                "difference": round(
                    modified_ss_stats["mean"] - original_ss_stats["mean"], 2
                ),
            },
        )

        return self

    def roll_check(
        self,
        threshold: float = DEFAULT_ROLL_THRESHOLD,
    ) -> SensorHealthRunner:
        """
        Check roll sensor values against threshold and update mask.

        Parameters
        ----------
        threshold : float, default 15.0
            Maximum acceptable roll in degrees (absolute value).

        Returns
        -------
        SensorHealthRunner
            Self for method chaining.

        See Also
        --------
        roll_check : Underlying function
        """
        # Capture before stats
        mask_before = self.dataset["mask"].values if "mask" in self.dataset else None
        cells_pre_masked = int(mask_before.sum()) if mask_before is not None else 0

        self.dataset = roll_check(self.dataset, threshold=threshold)

        # Capture after stats
        mask_after = self.dataset["mask"].values
        cells_total_masked = int(mask_after.sum())
        cells_newly_masked = cells_total_masked - cells_pre_masked

        # Calculate roll statistics for history
        roll_degrees = self.dataset["roll"].values * 0.01
        time_dim = "time" if "time" in self.dataset.dims else "ensemble"
        total_ensembles = self.dataset.sizes[time_dim]
        ensembles_exceeding = int(np.sum(np.abs(roll_degrees) > threshold))

        # Create statistics entry
        stat = QCCheckStats(
            check_name="Roll Check",
            threshold=threshold,
            cells_pre_masked=cells_pre_masked,
            cells_newly_masked=cells_newly_masked,
            cells_total_masked=cells_total_masked,
            total_cells=self.total_cells,
            metadata={
                "roll_min": round(float(roll_degrees.min()), 2),
                "roll_max": round(float(roll_degrees.max()), 2),
                "roll_mean": round(float(roll_degrees.mean()), 2),
                "ensembles_exceeding_threshold": ensembles_exceeding,
                "total_ensembles": total_ensembles,
            },
        )
        self.statistics.append(stat)

        self._add_history(
            operation="roll_check",
            parameters={"threshold": threshold},
            stats={
                "roll_min": round(float(roll_degrees.min()), 2),
                "roll_max": round(float(roll_degrees.max()), 2),
                "roll_mean": round(float(roll_degrees.mean()), 2),
                "ensembles_exceeding_threshold": ensembles_exceeding,
                "total_ensembles": total_ensembles,
                "cells_newly_masked": cells_newly_masked,
                "cells_total_masked": cells_total_masked,
            },
        )

        logger.info(
            f"Roll check applied (threshold={threshold}°). "
            f"Pre: {cells_pre_masked:,}, Newly masked: {cells_newly_masked:,}, "
            f"Total: {cells_total_masked:,}"
        )

        return self

    def pitch_check(
        self,
        threshold: float = DEFAULT_PITCH_THRESHOLD,
    ) -> SensorHealthRunner:
        """
        Check pitch sensor values against threshold and update mask.

        Parameters
        ----------
        threshold : float, default 15.0
            Maximum acceptable pitch in degrees (absolute value).

        Returns
        -------
        SensorHealthRunner
            Self for method chaining.

        See Also
        --------
        pitch_check : Underlying function
        """
        # Capture before stats
        mask_before = self.dataset["mask"].values if "mask" in self.dataset else None
        cells_pre_masked = int(mask_before.sum()) if mask_before is not None else 0

        self.dataset = pitch_check(self.dataset, threshold=threshold)

        # Capture after stats
        mask_after = self.dataset["mask"].values
        cells_total_masked = int(mask_after.sum())
        cells_newly_masked = cells_total_masked - cells_pre_masked

        # Calculate pitch statistics for history
        pitch_degrees = self.dataset["pitch"].values * 0.01
        time_dim = "time" if "time" in self.dataset.dims else "ensemble"
        total_ensembles = self.dataset.sizes[time_dim]
        ensembles_exceeding = int(np.sum(np.abs(pitch_degrees) > threshold))

        # Create statistics entry
        stat = QCCheckStats(
            check_name="Pitch Check",
            threshold=threshold,
            cells_pre_masked=cells_pre_masked,
            cells_newly_masked=cells_newly_masked,
            cells_total_masked=cells_total_masked,
            total_cells=self.total_cells,
            metadata={
                "pitch_min": round(float(pitch_degrees.min()), 2),
                "pitch_max": round(float(pitch_degrees.max()), 2),
                "pitch_mean": round(float(pitch_degrees.mean()), 2),
                "ensembles_exceeding_threshold": ensembles_exceeding,
                "total_ensembles": total_ensembles,
            },
        )
        self.statistics.append(stat)

        self._add_history(
            operation="pitch_check",
            parameters={"threshold": threshold},
            stats={
                "pitch_min": round(float(pitch_degrees.min()), 2),
                "pitch_max": round(float(pitch_degrees.max()), 2),
                "pitch_mean": round(float(pitch_degrees.mean()), 2),
                "ensembles_exceeding_threshold": ensembles_exceeding,
                "total_ensembles": total_ensembles,
                "cells_newly_masked": cells_newly_masked,
                "cells_total_masked": cells_total_masked,
            },
        )

        logger.info(
            f"Pitch check applied (threshold={threshold}°). "
            f"Pre: {cells_pre_masked:,}, Newly masked: {cells_newly_masked:,}, "
            f"Total: {cells_total_masked:,}"
        )

        return self

    def reset(self) -> SensorHealthRunner:
        """
        Reset dataset to original state and clear history.

        Returns
        -------
        SensorHealthRunner
            Self for method chaining.
        """
        self.dataset = self.original.copy(deep=True)
        self.history = []
        self.statistics = []
        self.modifications = []
        logger.info("SensorHealthRunner reset to original state")
        return self

    def finalize(self) -> xr.Dataset:
        """
        Finalize processing and return the dataset with history in attributes.

        Returns
        -------
        xr.Dataset
            Processed dataset with processing history in global attributes.
        """
        ds_out = self.dataset.copy(deep=True)

        # Add processing history to global attributes
        ds_out.attrs["sensor_health_processing"] = str(self.history)
        ds_out.attrs["sensor_health_processed_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        logger.info(
            f"SensorHealthRunner finalized with {len(self.history)} processing steps"
        )

        return ds_out

    # ========================================================================
    # STATISTICS AND REPORTING METHODS
    # ========================================================================

    def get_statistics(self) -> dict[str, QCCheckStats]:
        """
        Get statistics dictionary keyed by check name.

        Returns
        -------
        dict[str, QCCheckStats]
            Dictionary mapping check names to statistics objects.
        """
        return {stat.check_name: stat for stat in self.statistics}

    def get_modifications(self) -> dict[str, list[DataModificationStats]]:
        """
        Get modifications dictionary keyed by variable name.

        Returns
        -------
        dict[str, list[DataModificationStats]]
            Dictionary mapping variable names to lists of modification stats.
        """
        result: dict[str, list[DataModificationStats]] = {}
        for mod in self.modifications:
            if mod.variable_name not in result:
                result[mod.variable_name] = []
            result[mod.variable_name].append(mod)
        return result

    def print_statistics(self) -> None:
        """Print formatted statistics table to console."""
        print("\n" + "=" * 100)
        print("SENSOR HEALTH PROCESSING STATISTICS")
        print("=" * 100)
        print(
            f"Baseline masked: {self.baseline_masked:,} ({self.baseline_masked_pct:.2f}%)"
        )
        print(f"Total cells: {self.total_cells:,}")

        # Print data modifications
        if self.modifications:
            print("-" * 100)
            print("DATA MODIFICATIONS:")
            print(
                f"{'Operation':25s} | {'Variable':20s} | "
                f"{'Original Mean':>14s} | {'Modified Mean':>14s} | {'Change':>12s}"
            )
            print("-" * 100)
            for mod in self.modifications:
                orig_mean = mod.original_stats.get("mean", float("nan"))
                mod_mean = mod.modified_stats.get("mean", float("nan"))
                change = (
                    mod.mean_change if mod.mean_change is not None else float("nan")
                )
                print(
                    f"{mod.operation:25s} | {mod.variable_name:20s} | "
                    f"{orig_mean:>14.2f} | {mod_mean:>14.2f} | {change:>+12.2f}"
                )

        # Print QC checks
        print("-" * 100)
        if not self.statistics:
            print("No QC checks applied yet.")
        else:
            print("QC CHECKS:")
            print(
                f"{'Check':25s} | {'Threshold':>12s} | {'Pre-Masked':>12s} | "
                f"{'Impact':>12s} | {'Cumulative':>12s} | {'Valid':>10s}"
            )
            print("-" * 100)

            for stat in self.statistics:
                threshold_str = (
                    f"{stat.threshold:.1f}"
                    if isinstance(stat.threshold, (int, float))
                    else str(stat.threshold)
                )
                print(
                    f"{stat.check_name:25s} | {threshold_str:>12s} | "
                    f"{stat.pre_masked_pct:>11.2f}% | "
                    f"{stat.newly_masked_pct:>11.2f}% | "
                    f"{stat.total_masked_pct:>11.2f}% | "
                    f"{stat.valid_pct:>9.2f}%"
                )

            print("-" * 100)
            final = self.statistics[-1]
            impact = final.total_masked_pct - self.baseline_masked_pct
            print(
                f"FINAL: {final.valid_cells:,} valid cells ({final.valid_pct:.2f}%) | "
                f"Sensor health impact: {impact:+.2f}%"
            )

        print("=" * 100 + "\n")

    def get_report(self) -> str:
        """
        Get formatted report as string.

        Returns
        -------
        str
            Formatted report string suitable for file output.
        """
        report = self.get_pipeline_report()
        return str(report)

    def get_pipeline_report(self) -> QCPipelineReport:
        """
        Get complete pipeline report object.

        Returns
        -------
        QCPipelineReport
            Report object with all statistics and metadata.
        """
        return QCPipelineReport(
            module_name="sensor_health",
            baseline_masked=self.baseline_masked,
            baseline_masked_pct=self.baseline_masked_pct,
            total_cells=self.total_cells,
            checks=self.statistics.copy(),
            modifications=self.modifications.copy(),
        )

    def export_statistics_dict(self) -> dict[str, Any]:
        """
        Export statistics as nested dictionary for JSON/CSV export.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all statistics in serializable format.
        """
        report = self.get_pipeline_report()
        return report.to_dict()

    def summary(self) -> str:
        """
        Generate a human-readable summary of processing history.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = ["=" * 60, "Sensor Health Processing Summary", "=" * 60]

        if not self.history:
            lines.append("No processing steps applied.")
            return "\n".join(lines)

        for i, entry in enumerate(self.history, 1):
            lines.append(f"\nStep {i}: {entry['operation']}")
            lines.append("-" * 40)

            # Parameters
            lines.append("  Parameters:")
            for key, value in entry["parameters"].items():
                lines.append(f"    {key}: {value}")

            # Statistics
            if "stats" in entry:
                lines.append("  Statistics:")
                for key, value in entry["stats"].items():
                    lines.append(f"    {key}: {value}")

            lines.append(f"  Timestamp: {entry['timestamp']}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
