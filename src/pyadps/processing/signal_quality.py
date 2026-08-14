"""
Signal Quality Control Module for ADCP Data Processing (v1.0.0).

This module provides quality control functions for Acoustic Doppler Current Profiler
(ADCP) data, including echo intensity, correlation, error velocity, and percent-good
checks. All functions are designed to work with xarray.Dataset objects.

INTEGRATED FEATURES (v1.0.0):
- Returns modified xarray.Dataset objects (instead of just masks)
- Uses shared utility.py statistics and mask creation
- SignalQualityRunner updated for new dataset-based workflow with full feature parity
- Support for 3D masks (beam-specific flagging)

Key Design Principles:
- **xarray-native**: All functions work with xarray.Dataset
- **Non-destructive**: Returns new Dataset copies, preserving inputs
- **Statistics**: Integrated tracking via SignalQualityRunner
- **3D Masking**: Updates beam-specific masks where applicable
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from importlib import resources as importlib_resources
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import xarray as xr

# Import shared utilities
from .utility import (
    create_default_mask,
    QCCheckStats,
    DataModificationStats,
    QCPipelineReport,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS AND THRESHOLDS
# ============================================================================

DEFAULT_CORRELATION_THRESHOLD = 64
DEFAULT_ECHO_THRESHOLD = 40
DEFAULT_ERROR_VELOCITY_THRESHOLD = 2000  # mm/s
DEFAULT_PERCENT_GOOD_THRESHOLD = 50  # Percentage
DEFAULT_FALSE_TARGET_THRESHOLD = 50  # Echo intensity difference

THRESHOLD_RANGES: dict[str, tuple[int, int]] = {
    "correlation": (0, 255),
    "echo_intensity": (0, 255),
    "error_velocity": (0, 5000),
    "percent_good": (0, 100),
    "false_target": (0, 255),
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _get_mask_or_create(ds: xr.Dataset) -> xr.DataArray:
    """Get existing mask or create default one from utility."""
    if "mask" in ds.data_vars:
        return ds["mask"]
    logger.debug("No mask found, creating default mask from velocity data")
    return create_default_mask(ds)


def _validate_threshold(name: str, value: float) -> None:
    """Validate threshold against defined ranges."""
    if name in THRESHOLD_RANGES:
        min_val, max_val = THRESHOLD_RANGES[name]
        if not (min_val <= value <= max_val):
            logger.warning(
                f"Threshold {name}={value} out of range [{min_val}, {max_val}]"
            )


# ============================================================================
# CORE QC FUNCTIONS
# ============================================================================


def correlation_check(
    ds: xr.Dataset,
    cutoff: float = DEFAULT_CORRELATION_THRESHOLD,
    beam_ignore: int | None = None,
) -> xr.Dataset:
    """
    Perform correlation strength quality control check.

    Flags depth cells where correlation falls below the cutoff. When any beam
    fails the check, the entire depth cell is masked across all beams, because
    post-collection data is in Earth coordinates and individual beams cannot
    be selectively dropped.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing correlation data.
    cutoff : float, default 64
        Minimum acceptable correlation value (0-255 scale).
    beam_ignore : int, optional
        Beam index (0-3) to exclude from the check. Use when a beam is known
        to be permanently faulty (identifiable from correlation, echo
        intensity, or percent-good diagnostics).

    Returns
    -------
    xr.Dataset
        Dataset with updated mask.
    """
    _validate_threshold("correlation", cutoff)

    if "correlation" not in ds.data_vars:
        logger.warning("Correlation data not found in dataset")
        return ds

    mask = _get_mask_or_create(ds)
    correlation = ds["correlation"]

    if "beam" not in correlation.dims:
        logger.warning("Correlation has no beam dimension")
        return ds

    # Flag values < cutoff
    below = correlation < cutoff

    # Exclude known-bad beam from the count
    if beam_ignore is not None:
        if 0 <= beam_ignore < correlation.sizes["beam"]:
            beam_coords = correlation.coords["beam"].values
            keep_beams = [b for b in beam_coords if b != beam_coords[beam_ignore]]
            below = below.sel(beam=keep_beams)
            logger.debug(f"Correlation check: ignoring beam {beam_ignore}")
        else:
            logger.warning(
                f"beam_ignore={beam_ignore} out of range, ignoring parameter"
            )

    # Count beams below threshold at each (cell, time) point
    n_failing = below.sum(dim="beam")

    # Mask if any remaining beam fails
    cell_flag = n_failing >= 1

    # Mask the full depth cell across all beams. Transpose restores (beam, cell, time)
    # order — xr.where with a (cell, time) condition reorders dims to (cell, time, beam).
    mask_updated = xr.where(cell_flag, 1, mask).transpose(*mask.dims).astype(np.int8)
    mask_updated.attrs = mask.attrs.copy()

    ds_out = ds.copy(deep=True)
    ds_out["mask"] = mask_updated

    newly_flagged = int((mask_updated == 1).sum()) - int((mask == 1).sum())
    logger.info(
        f"Correlation check applied: cutoff={cutoff}, beam_ignore={beam_ignore}, "
        f"newly flagged cells: {newly_flagged}"
    )

    return ds_out


def echo_intensity_check(
    ds: xr.Dataset,
    cutoff: float | list[float] = DEFAULT_ECHO_THRESHOLD,
    beam_ignore: int | None = None,
) -> xr.Dataset:
    """
    Perform echo intensity (signal strength) quality control check.

    Flags depth cells where echo intensity falls below the noise floor threshold.
    When any beam fails the check, the entire depth cell is masked across all
    beams, because post-collection data is in Earth coordinates and individual
    beams cannot be selectively dropped.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing echo intensity data.
    cutoff : float or list of float, default 40
        Minimum acceptable echo intensity (0-255 scale). A single value applies
        the same threshold to all beams. A list of four values sets a per-beam
        threshold, useful when beams have different noise floors (e.g. one beam
        has a fouled transducer face). Use the Noise Floor tab to derive these
        values from in-air recordings.
    beam_ignore : int, optional
        Beam index (0-3) to exclude from the check. Use when a beam is known
        to be permanently faulty (identifiable from correlation, echo intensity,
        or percent-good diagnostics).

    Returns
    -------
    xr.Dataset
        Dataset with updated mask.
    """
    var_name = "echo_intensity"
    if var_name not in ds.data_vars:
        if "echo" in ds.data_vars:
            var_name = "echo"
        else:
            logger.warning("Echo intensity data not found")
            return ds

    mask = _get_mask_or_create(ds)
    echo = ds[var_name]

    if "beam" not in echo.dims:
        logger.warning("Echo intensity has no beam dimension")
        return ds

    # Build per-beam threshold and compute below-threshold flag
    if isinstance(cutoff, list):
        for c in cutoff:
            _validate_threshold("echo_intensity", c)
        beam_coords = echo.coords["beam"].values
        if len(cutoff) != len(beam_coords):
            logger.warning(
                f"cutoff list length {len(cutoff)} != n_beams {len(beam_coords)},"
                " using cutoff[0] for all beams"
            )
            below = echo < float(cutoff[0])
        else:
            cutoff_da = xr.DataArray(
                cutoff, dims=["beam"], coords={"beam": beam_coords}
            )
            below = echo < cutoff_da
    else:
        _validate_threshold("echo_intensity", cutoff)
        below = echo < cutoff

    # Exclude known-bad beam from the count
    if beam_ignore is not None:
        if 0 <= beam_ignore < echo.sizes["beam"]:
            beam_coords = echo.coords["beam"].values
            keep_beams = [b for b in beam_coords if b != beam_coords[beam_ignore]]
            below = below.sel(beam=keep_beams)
            logger.debug(f"Echo intensity: ignoring beam {beam_ignore}")
        else:
            logger.warning(
                f"beam_ignore={beam_ignore} out of range, ignoring parameter"
            )

    # Count beams below threshold at each (cell, time) point
    n_failing = below.sum(dim="beam")

    # Mask if any remaining beam fails
    cell_flag = n_failing >= 1

    # Mask the full depth cell across all beams. Transpose restores (beam, cell, time)
    # order — xr.where with a (cell, time) condition reorders dims to (cell, time, beam).
    mask_updated = xr.where(cell_flag, 1, mask).transpose(*mask.dims).astype(np.int8)
    mask_updated.attrs = mask.attrs.copy()

    ds_out = ds.copy(deep=True)
    ds_out["mask"] = mask_updated

    newly_flagged = int((mask_updated == 1).sum()) - int((mask == 1).sum())
    logger.info(
        f"Echo intensity check applied: cutoff={cutoff}, "
        f"beam_ignore={beam_ignore}, newly flagged cells: {newly_flagged}"
    )

    return ds_out


def error_velocity_check(
    ds: xr.Dataset,
    cutoff: float = DEFAULT_ERROR_VELOCITY_THRESHOLD,
) -> xr.Dataset:
    """
    Perform error velocity quality control check.

    Flags depth cells where absolute error velocity (Beam 3) exceeds cutoff.
    When the check fails, the entire depth cell is masked across all beams,
    because post-collection data is in Earth coordinates and a high error
    velocity indicates an unreliable combined u/v/w solution, not an issue
    isolated to one component.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing velocity data.
    cutoff : float, default 2000
        Maximum acceptable error velocity in mm/s.

    Returns
    -------
    xr.Dataset
        Dataset with updated mask.
    """
    _validate_threshold("error_velocity", cutoff)

    if "velocity" not in ds.data_vars:
        logger.error("Velocity data not found")
        return ds

    mask = _get_mask_or_create(ds)
    velocity = ds["velocity"]

    if velocity.sizes.get("beam", 0) < 4:
        logger.warning("Velocity data < 4 beams, skipping Error Velocity check")
        return ds

    # Extract Error Velocity (Beam 3)
    error_vel = np.abs(velocity.isel(beam=3))

    # Flag values > cutoff
    flag = error_vel > cutoff

    # Mask the full depth cell across all beams. Transpose restores (beam, cell, time)
    # order — xr.where with a (cell, time) condition reorders dims to (cell, time, beam).
    mask_updated = xr.where(flag, 1, mask).transpose(*mask.dims).astype(np.int8)
    mask_updated.attrs = mask.attrs.copy()

    ds_out = ds.copy(deep=True)
    ds_out["mask"] = mask_updated

    newly_flagged = int((mask_updated == 1).sum()) - int((mask == 1).sum())
    logger.info(
        f"Error velocity check applied: cutoff={cutoff}, "
        f"newly flagged cells: {newly_flagged}"
    )

    return ds_out


def percent_good_check(
    ds: xr.Dataset,
    cutoff: float = DEFAULT_PERCENT_GOOD_THRESHOLD,
    threebeam: bool = True,
    method: str | None = None,
) -> xr.Dataset:
    """
    Perform percent good quality control check.

    For RDI ADCPs, percent good data has 4 values per cell:
    - PG1 (beam 0): Percentage of 3-beam solutions
    - PG2 (beam 1): Percentage of transformations rejected
    - PG3 (beam 2): Percentage of more than one beam bad
    - PG4 (beam 3): Percentage of 4-beam solutions

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing percent good data.
    cutoff : float, default 50
        Minimum acceptable percent good value (0-100).
    threebeam : bool, default True
        If True, sums PG1 + PG4 (3-beam + 4-beam solutions) for the check.
        If False, uses only PG4 (4-beam solutions).
        This is the recommended RDI approach for percent good validation.
    method : str, optional
        Alternative method to combine beam values: "max", "min", or "mean".
        If provided, overrides the threebeam parameter.
        Use this for non-standard percent good calculations.

    Returns
    -------
    xr.Dataset
        Dataset with updated mask.

    Notes
    -----
    The threebeam=True mode (default) follows RDI's recommendation to accept
    data where either 3 or 4 beams produced a valid solution. This is more
    permissive than requiring all 4 beams (threebeam=False).

    When the combined percent-good value fails the check, the entire depth
    cell is masked across all beams, because post-collection data is in
    Earth coordinates and a low percent-good value indicates an unreliable
    combined u/v/w solution, not an issue isolated to one component.
    """
    _validate_threshold("percent_good", cutoff)

    var_name = "percent_good"
    if var_name not in ds.data_vars:
        if "pg" in ds.data_vars:
            var_name = "pg"
        else:
            logger.warning("Percent good data not found")
            return ds

    mask = _get_mask_or_create(ds)
    pgood = ds[var_name]

    # Determine combination method
    if method is not None:
        # Use explicit method if provided
        if "beam" in pgood.dims:
            if method == "max":
                pgood_combined = pgood.max(dim="beam")
            elif method == "min":
                pgood_combined = pgood.min(dim="beam")
            elif method == "mean":
                pgood_combined = pgood.mean(dim="beam")
            else:
                logger.warning(f"Unknown method '{method}', using threebeam mode")
                method = None  # Fall through to threebeam logic
        else:
            pgood_combined = pgood

    if method is None:
        # Use threebeam logic (RDI standard approach)
        if "beam" in pgood.dims and pgood.sizes["beam"] >= 4:
            if threebeam:
                # Sum PG1 (beam 0) + PG4 (beam 3): 3-beam + 4-beam solutions
                pgood_combined = pgood.isel(beam=0) + pgood.isel(beam=3)
                logger.debug("Percent good: using PG1 + PG4 (threebeam mode)")
            else:
                # Use only PG4 (beam 3): 4-beam solutions only
                pgood_combined = pgood.isel(beam=3)
                logger.debug("Percent good: using PG4 only (4-beam mode)")
        elif "beam" in pgood.dims:
            # Less than 4 beams - use max as fallback
            pgood_combined = pgood.max(dim="beam")
            logger.warning(f"Percent good has {pgood.sizes['beam']} beams, using max")
        else:
            pgood_combined = pgood

    flag = pgood_combined < cutoff

    # Mask the full depth cell across all beams. Transpose restores (beam, cell, time)
    # order — xr.where with a (cell, time) condition reorders dims to (cell, time, beam).
    mask_updated = xr.where(flag, 1, mask).transpose(*mask.dims).astype(np.int8)
    mask_updated.attrs = mask.attrs.copy()

    ds_out = ds.copy(deep=True)
    ds_out["mask"] = mask_updated

    newly_flagged = int((mask_updated == 1).sum()) - int((mask == 1).sum())
    logger.info(
        f"Percent good check applied: cutoff={cutoff}, threebeam={threebeam}, "
        f"newly flagged cells: {newly_flagged}"
    )

    return ds_out


def false_target_detection(
    ds: xr.Dataset,
    cutoff: float = DEFAULT_FALSE_TARGET_THRESHOLD,
    beam_ignore: int | None = None,
) -> xr.Dataset:
    """
    Detect and flag false targets using echo intensity anomaly detection.

    False targets occur when one beam receives a strong echo (e.g., from fish,
    debris, or air bubbles) that is inconsistent with other beams. This check
    identifies cells where the difference between echo intensities exceeds
    a threshold.

    This is a post-collection analogue of the ADCP's WA (false target) command.
    Because data is already in Earth coordinates, individual beams cannot be
    selectively flagged; instead the entire depth cell is rejected. The adjacent
    cell (x+1) is also flagged whenever a false target is detected, because the
    ADCP samples echo intensity near the end of each depth cell.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing echo intensity data.
    cutoff : float, default 50
        Maximum acceptable echo intensity difference between beams (``max -
        min``). Equivalent to the WA command threshold. Use a value lower than
        the pre-deployment WA setting to apply a stricter post-collection check.
    beam_ignore : int, optional
        Beam index (0-3) to exclude from the comparison. Use when a beam is
        known to be permanently faulty (identifiable from correlation, echo
        intensity, or percent-good data). The remaining three beams are checked
        using ``max - min`` with the specified cutoff.

    Returns
    -------
    xr.Dataset
        Dataset with updated mask.

    Notes
    -----
    The pre-deployment WA command runs on the instrument in beam coordinates,
    per ping, before any coordinate transformation. This function extends that
    check to post-collected data, where only whole-ensemble rejection is
    possible. There is no ensemble-data equivalent of the instrument's
    per-ping 3-beam WA leniency: distinguishing "one beam had a transient
    false target this ping" from "one beam is systematically different"
    requires single-ping resolution that ensemble-averaging has already
    discarded by the time this function runs. Use ``beam_ignore`` for a beam
    known to be bad; there is no automatic-detection equivalent.

    The algorithm:

    1. If ``beam_ignore`` is specified, remove that beam from consideration.
    2. Compute ``max - min`` across the remaining beams.
    3. Flag cells where the difference exceeds ``cutoff``.
    4. Also flag the next depth cell (x+1) for each flagged cell, because the
       ADCP samples echo intensity near the end of depth cell x, so the
       velocity measurement at x+1 is also contaminated.
    """
    _validate_threshold("false_target", cutoff)

    var_name = "echo_intensity"
    if var_name not in ds.data_vars:
        if "echo" in ds.data_vars:
            var_name = "echo"
        else:
            logger.warning("Echo intensity data not found")
            return ds

    mask = _get_mask_or_create(ds)
    echo = ds[var_name]

    if "beam" not in echo.dims or echo.sizes["beam"] < 2:
        logger.warning("Echo intensity must have >= 2 beams for false target detection")
        return ds

    # Handle beam_ignore - exclude specified beam from comparison
    if beam_ignore is not None:
        if 0 <= beam_ignore < echo.sizes["beam"]:
            # Get beam coordinate values
            beam_coords = echo.coords["beam"].values
            # Create mask for beams to keep
            keep_beams = [b for b in beam_coords if b != beam_coords[beam_ignore]]
            echo = echo.sel(beam=keep_beams)
            logger.debug(f"False target: ignoring beam {beam_ignore}")
        else:
            logger.warning(
                f"beam_ignore={beam_ignore} out of range, ignoring parameter"
            )

    # Sort echo values along beam dimension for comparison
    # Using numpy for sorted comparison since xarray doesn't have direct sort
    echo_values = echo.values  # Shape: (beam, cell, ensemble) or similar

    # Find axis index for beam dimension
    beam_axis = echo.dims.index("beam")

    # Sort along beam axis
    sorted_echo = np.sort(echo_values, axis=beam_axis)

    # Create DataArray with same coordinates (minus beam for result)
    # Get non-beam dimensions for output
    non_beam_dims = [d for d in echo.dims if d != "beam"]
    non_beam_coords = {d: echo.coords[d] for d in non_beam_dims}

    # max - min: flags whenever any single beam is anomalous.
    echo_max = np.take(sorted_echo, -1, axis=beam_axis)
    echo_min = np.take(sorted_echo, 0, axis=beam_axis)
    difference = echo_max - echo_min

    # Convert difference back to DataArray
    difference_da = xr.DataArray(
        difference,
        dims=non_beam_dims,
        coords=non_beam_coords,
    )

    flag = difference_da > cutoff

    # Also flag the next depth cell: the ADCP samples echo intensity near the
    # end of cell x, so cell x+1 velocity is contaminated by the same target.
    flag = flag | flag.shift({"cell": 1}, fill_value=False)

    # Mask the full depth cell across all beams. Transpose restores (beam, cell, time)
    # order — xr.where with a (cell, time) condition reorders dims to (cell, time, beam).
    mask_updated = xr.where(flag, 1, mask).transpose(*mask.dims).astype(np.int8)
    mask_updated.attrs = mask.attrs.copy()

    ds_out = ds.copy(deep=True)
    ds_out["mask"] = mask_updated

    newly_flagged = int((mask_updated == 1).sum()) - int((mask == 1).sum())
    logger.info(
        f"False target detection applied: cutoff={cutoff}, "
        f"beam_ignore={beam_ignore}, newly flagged cells: {newly_flagged}"
    )

    return ds_out


# ============================================================================
# QC RUNNER CLASS
# ============================================================================


class SignalQualityRunner:
    """
    Orchestrates QC checks with statistics tracking (Dataset-aware).

    Provides a fluent API (method chaining) matching the SensorHealthRunner style.

    Attributes
    ----------
    dataset : xr.Dataset
        Current working dataset.
    original : xr.Dataset
        Original unmodified dataset.
    statistics : list[QCCheckStats]
        Statistics for each QC check applied.
    modifications : list[DataModificationStats]
        Statistics for data modifications (future use).
    history : list[dict]
        Processing history.

    Examples
    --------
    >>> runner = SignalQualityRunner(ds)
    >>> ds_qc = (runner
    ...     .correlation(cutoff=64)
    ...     .echo_intensity(cutoff=40)
    ...     .error_velocity(cutoff=2000)
    ...     .percent_good(cutoff=50)
    ...     .finalize())
    >>> runner.print_statistics()
    """

    def __init__(self, ds: xr.Dataset):
        """
        Initialize runner with dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Input ADCP dataset to process.
        """
        self.original = ds.copy(deep=True)
        self.dataset = ds.copy(deep=True)

        # Ensure mask exists for baseline stats
        if "mask" not in self.dataset.data_vars:
            self.dataset["mask"] = create_default_mask(self.dataset)

        self._update_baseline()
        self.statistics: list[QCCheckStats] = []
        self.modifications: list[DataModificationStats] = []
        self.history: list[dict[str, Any]] = []

        logger.info(
            f"SignalQualityRunner initialized: {self.total_cells:,} cells, "
            f"{self.baseline_masked:,} pre-masked ({self.baseline_masked_pct:.2f}%)"
        )

    def _update_baseline(self) -> None:
        """Calculate baseline statistics from current dataset."""
        mask = self.dataset["mask"]
        self.total_cells = int(mask.size)
        self.baseline_masked = int((mask == 1).sum())
        self.baseline_masked_pct = (
            100 * self.baseline_masked / self.total_cells if self.total_cells > 0 else 0
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

    def _record_check(
        self,
        check_name: str,
        check_func: Any,
        cutoff: float | list[float],
        **kwargs: Any,
    ) -> SignalQualityRunner:
        """Apply check and record statistics."""
        # Stats before check
        mask_pre = self.dataset["mask"].values
        pre_masked = int((mask_pre == 1).sum())

        # Apply Check (updates self.dataset)
        self.dataset = check_func(self.dataset, cutoff=cutoff, **kwargs)

        # Stats after check
        mask_post = self.dataset["mask"].values
        post_masked = int((mask_post == 1).sum())
        newly_masked = post_masked - pre_masked

        # Create Statistics Object
        stat = QCCheckStats(
            check_name=check_name,
            threshold=cutoff,
            cells_pre_masked=pre_masked,
            cells_newly_masked=newly_masked,
            cells_total_masked=post_masked,
            total_cells=self.total_cells,
            check_time=datetime.now(timezone.utc),
            metadata=kwargs.copy(),
        )
        self.statistics.append(stat)

        # Record History
        self._add_history(
            operation=check_name.lower().replace(" ", "_"),
            parameters={"cutoff": cutoff, **kwargs},
            stats={
                "pre_masked": pre_masked,
                "newly_masked": newly_masked,
                "total_masked": post_masked,
            },
        )

        return self

    # ========================================================================
    # CHECK METHODS
    # ========================================================================

    def correlation(
        self,
        cutoff: float = DEFAULT_CORRELATION_THRESHOLD,
        beam_ignore: int | None = None,
    ) -> SignalQualityRunner:
        """
        Apply correlation check.

        Parameters
        ----------
        cutoff : float, default 64
            Minimum acceptable correlation value.
        beam_ignore : int, optional
            Beam to ignore (e.g. a known-faulty beam).

        Returns
        -------
        SignalQualityRunner
            Self for method chaining.
        """
        return self._record_check(
            "Correlation",
            correlation_check,
            cutoff,
            beam_ignore=beam_ignore,
        )

    def echo_intensity(
        self,
        cutoff: float | list[float] = DEFAULT_ECHO_THRESHOLD,
        beam_ignore: int | None = None,
    ) -> SignalQualityRunner:
        """
        Apply echo intensity check.

        Parameters
        ----------
        cutoff : float or list of float, default 40
            Minimum acceptable echo intensity. A single value applies to all
            beams; a list of four values sets a per-beam threshold.
        beam_ignore : int, optional
            Beam index to exclude from the check (0-3).

        Returns
        -------
        SignalQualityRunner
            Self for method chaining.
        """
        return self._record_check(
            "Echo Intensity",
            echo_intensity_check,
            cutoff,
            beam_ignore=beam_ignore,
        )

    def error_velocity(
        self,
        cutoff: float = DEFAULT_ERROR_VELOCITY_THRESHOLD,
    ) -> SignalQualityRunner:
        """
        Apply error velocity check.

        Parameters
        ----------
        cutoff : float, default 2000
            Maximum acceptable error velocity in mm/s.

        Returns
        -------
        SignalQualityRunner
            Self for method chaining.
        """
        return self._record_check(
            "Error Velocity",
            error_velocity_check,
            cutoff,
        )

    def percent_good(
        self,
        cutoff: float = DEFAULT_PERCENT_GOOD_THRESHOLD,
        threebeam: bool = True,
        method: str | None = None,
    ) -> SignalQualityRunner:
        """
        Apply percent good check.

        Parameters
        ----------
        cutoff : float, default 50
            Minimum acceptable percent good value.
        threebeam : bool, default True
            If True, sums PG1 + PG4 (3-beam + 4-beam solutions).
            If False, uses only PG4 (4-beam solutions).
        method : str, optional
            Alternative method to combine beam values ("max", "min", "mean").
            Overrides threebeam if provided.

        Returns
        -------
        SignalQualityRunner
            Self for method chaining.
        """
        return self._record_check(
            "Percent Good",
            percent_good_check,
            cutoff,
            threebeam=threebeam,
            method=method,
        )

    def false_target(
        self,
        cutoff: float = DEFAULT_FALSE_TARGET_THRESHOLD,
        beam_ignore: int | None = None,
    ) -> SignalQualityRunner:
        """
        Apply false target detection.

        Parameters
        ----------
        cutoff : float, default 50
            Maximum acceptable echo intensity difference between beams
            (``max - min``).
        beam_ignore : int, optional
            Beam index to exclude from comparison (0-3).

        Returns
        -------
        SignalQualityRunner
            Self for method chaining.
        """
        return self._record_check(
            "False Target",
            false_target_detection,
            cutoff,
            beam_ignore=beam_ignore,
        )

    # ========================================================================
    # PIPELINE UTILITIES
    # ========================================================================

    def apply_pipeline(
        self,
        checks: dict[str, float] | None = None,
        order: list[str] | None = None,
    ) -> SignalQualityRunner:
        """
        Apply multiple QC checks using a pipeline configuration.

        Parameters
        ----------
        checks : dict, optional
            Dict of {check_name: cutoff_value}. If None, uses defaults.
        order : list, optional
            List of check names in execution order. If None, uses default order.

        Returns
        -------
        SignalQualityRunner
            Self for method chaining.

        Examples
        --------
        >>> runner = SignalQualityRunner(ds)
        >>> runner.apply_pipeline()  # Use all defaults
        >>> runner.apply_pipeline(checks={"correlation": 50, "echo_intensity": 30})
        """
        if checks is None:
            checks = {
                "correlation": DEFAULT_CORRELATION_THRESHOLD,
                "echo_intensity": DEFAULT_ECHO_THRESHOLD,
                "error_velocity": DEFAULT_ERROR_VELOCITY_THRESHOLD,
                "percent_good": DEFAULT_PERCENT_GOOD_THRESHOLD,
                "false_target": DEFAULT_FALSE_TARGET_THRESHOLD,
            }

        if order is None:
            order = [
                "correlation",
                "echo_intensity",
                "error_velocity",
                "percent_good",
                "false_target",
            ]

        for name in order:
            if name not in checks:
                continue

            cutoff = checks[name]
            if name == "correlation":
                self.correlation(cutoff=cutoff)
            elif name == "echo_intensity":
                self.echo_intensity(cutoff=cutoff)
            elif name == "error_velocity":
                self.error_velocity(cutoff=cutoff)
            elif name == "percent_good":
                self.percent_good(cutoff=cutoff)
            elif name == "false_target":
                self.false_target(cutoff=cutoff)

        return self

    def reset(self) -> SignalQualityRunner:
        """
        Reset dataset to original state and clear history.

        Returns
        -------
        SignalQualityRunner
            Self for method chaining.
        """
        self.dataset = self.original.copy(deep=True)
        if "mask" not in self.dataset.data_vars:
            self.dataset["mask"] = create_default_mask(self.dataset)
        self.statistics = []
        self.modifications = []
        self.history = []
        self._update_baseline()
        logger.info("SignalQualityRunner reset to original state")
        return self

    def finalize(self) -> xr.Dataset:
        """
        Finalize processing and return dataset with history attributes.

        Returns
        -------
        xr.Dataset
            Processed dataset with processing history in attributes.
        """
        ds_out = self.dataset.copy(deep=True)
        ds_out.attrs["signal_quality_processing_history"] = str(self.history)
        ds_out.attrs["signal_quality_processed_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        logger.info("SignalQualityRunner finalized")
        return ds_out

    # ========================================================================
    # STATISTICS AND REPORTING
    # ========================================================================

    def get_statistics(self) -> dict[str, QCCheckStats]:
        """
        Get statistics dictionary keyed by check name.

        Returns
        -------
        dict[str, QCCheckStats]
            Dictionary mapping check names to their statistics.
        """
        return {stat.check_name: stat for stat in self.statistics}

    def get_modifications(self) -> dict[str, list[DataModificationStats]]:
        """
        Get modification statistics grouped by variable name.

        Returns
        -------
        dict[str, list[DataModificationStats]]
            Dictionary mapping variable names to their modification statistics.
        """
        result: dict[str, list[DataModificationStats]] = {}
        for mod in self.modifications:
            if mod.variable_name not in result:
                result[mod.variable_name] = []
            result[mod.variable_name].append(mod)
        return result

    def get_pipeline_report(self) -> QCPipelineReport:
        """
        Get complete pipeline report object.

        Returns
        -------
        QCPipelineReport
            Complete report with all statistics.
        """
        return QCPipelineReport(
            module_name="signal_quality",
            baseline_masked=self.baseline_masked,
            baseline_masked_pct=self.baseline_masked_pct,
            total_cells=self.total_cells,
            checks=self.statistics.copy(),
            modifications=self.modifications.copy(),
        )

    def get_report(self) -> str:
        """
        Get formatted report as string.

        Returns
        -------
        str
            Formatted report string.
        """
        return str(self.get_pipeline_report())

    def export_statistics_dict(self) -> dict[str, Any]:
        """
        Export statistics as nested dictionary for JSON/CSV export.

        Returns
        -------
        dict[str, Any]
            JSON-serializable dictionary of all statistics.
        """
        return self.get_pipeline_report().to_dict()

    def print_statistics(self) -> None:
        """Print formatted statistics table to console."""
        width = 100
        print("=" * width)
        print("SIGNAL QUALITY PROCESSING STATISTICS")
        print("=" * width)
        print(
            f"Baseline masked: {self.baseline_masked:,} ({self.baseline_masked_pct:.2f}%)"
        )
        print(f"Total cells: {self.total_cells:,}")
        print("-" * width)

        # Print modifications if any
        if self.modifications:
            print("DATA MODIFICATIONS:")
            header = (
                f"{'Operation':<26} | {'Variable':<20} | "
                f"{'Original Mean':>14} | {'Modified Mean':>14} | {'Change':>12}"
            )
            print(header)
            print("-" * width)
            for mod in self.modifications:
                orig_mean = mod.original_stats.get("mean", float("nan"))
                mod_mean = mod.modified_stats.get("mean", float("nan"))
                change = (
                    mod.mean_change if mod.mean_change is not None else float("nan")
                )
                print(
                    f"{mod.operation:<26} | {mod.variable_name:<20} | "
                    f"{orig_mean:>14.2f} | {mod_mean:>14.2f} | {change:>+12.2f}"
                )
            print("-" * width)

        # Print QC checks
        if not self.statistics:
            print("No QC checks applied yet.")
        else:
            print("QC CHECKS:")
            header = (
                f"{'Check':<25} | {'Threshold':>12} | {'Pre-Masked':>12} | "
                f"{'Impact':>12} | {'Cumulative':>12} | {'Valid':>10}"
            )
            print(header)
            print("-" * width)
            for stat in self.statistics:
                # Format threshold (handle different types)
                if isinstance(stat.threshold, tuple):
                    threshold_str = f"{stat.threshold}"
                elif stat.threshold is None:
                    threshold_str = "N/A"
                else:
                    threshold_str = f"{stat.threshold}"

                print(
                    f"{stat.check_name:<25} | {threshold_str:>12} | "
                    f"{stat.pre_masked_pct:>11.2f}% | {stat.newly_masked_pct:>11.2f}% | "
                    f"{stat.total_masked_pct:>11.2f}% | {stat.valid_pct:>9.2f}%"
                )
            print("-" * width)
            final = self.statistics[-1]
            impact = final.total_masked_pct - self.baseline_masked_pct
            print(
                f"FINAL: {final.valid_cells:,} valid cells ({final.valid_pct:.2f}%) | "
                f"Signal quality impact: {impact:+.2f}%"
            )
        print("=" * width)

    def summary(self) -> str:
        """
        Generate a human-readable summary of processing history.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = [
            "=" * 60,
            "SIGNAL QUALITY PROCESSING SUMMARY",
            "=" * 60,
        ]

        if not self.history:
            lines.append("No processing steps applied yet.")
        else:
            for i, entry in enumerate(self.history, 1):
                op = entry.get("operation", "unknown")
                params = entry.get("parameters", {})
                stats = entry.get("stats", {})
                newly_masked = stats.get("newly_masked", 0)

                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                lines.append(f"{i}. {op}: {param_str}")
                lines.append(f"   Impact: {newly_masked:,} cells newly masked")

        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# PERCENT GOOD THRESHOLD ADVISOR
# ============================================================================

# Frequency bit-code mapping from system_configuration_code (bits 13–15 of
# the 16-bit binary string, i.e. string indices [13:16]).
# Mirrors the table in io/accessors.py; hardware-defined, never changes.
_FREQ_BIT_MAP: dict[str, int] = {
    "000": 75,
    "001": 150,
    "010": 300,
    "011": 600,
    "100": 1200,
    "101": 2400,
    "110": 38,   # not covered by velocity_noise_coefficients.json
}


@dataclass
class StdDevResult:
    """
    Advisory result for percent-good threshold selection.

    All standard deviation values are in cm/s.

    Attributes
    ----------
    frequency : int
        ADCP operating frequency in kHz.
    bin_size : float
        Depth cell length in metres.
    depth_range : float
        Total profiling depth range (bin_size × num_cells) in metres.
    pings_per_ensemble : int
        Number of pings averaged per ensemble.
    single_ping_std : float
        Single-ping standard deviation at *depth_range* (cm/s).
    ensemble_std : float
        Effective standard deviation for the full ensemble (cm/s).
        ``single_ping_std / sqrt(pings_per_ensemble)``
    desired_std : float
        User-requested standard deviation (cm/s).
    valid_pings_required : int
        Minimum valid pings needed to meet *desired_std*.
        Clamped to *pings_per_ensemble* when the target is unachievable.
    percent_good_cutoff : float
        Recommended percent-good threshold (0–100 %).
    achievable : bool
        ``False`` when *desired_std* is tighter than the ensemble average;
        the user would need more pings per ensemble to meet the target.
    """

    frequency: int
    bin_size: float
    depth_range: float
    pings_per_ensemble: int
    single_ping_std: float
    ensemble_std: float
    desired_std: float
    valid_pings_required: int
    percent_good_cutoff: float
    achievable: bool

    def __str__(self) -> str:
        status = "achievable" if self.achievable else "NOT achievable — increase pings"
        return (
            f"StdDevResult:\n"
            f"  Frequency        : {self.frequency} kHz\n"
            f"  Bin size         : {self.bin_size} m\n"
            f"  Depth range      : {self.depth_range} m\n"
            f"  Pings/ensemble   : {self.pings_per_ensemble}\n"
            f"  Single-ping std  : {self.single_ping_std:.4f} cm/s\n"
            f"  Ensemble std     : {self.ensemble_std:.4f} cm/s\n"
            f"  Desired std      : {self.desired_std:.4f} cm/s  [{status}]\n"
            f"  Valid pings req. : {self.valid_pings_required}\n"
            f"  Percent-good cut : {self.percent_good_cutoff:.1f} %"
        )


def _load_adcp_coefficients(path: Optional[str] = None) -> dict:
    """Load velocity_noise_coefficients.json from package or a custom path.

    The coefficients represent the **output velocity noise** (cm/s) after
    coordinate transformation to Earth/instrument coordinates, not raw beam
    velocity noise.  They were derived by digitising the standard-deviation
    vs. depth-range figures in the Teledyne RDI manual *ADCP Coordinate
    Transformation — Formulas and Calculations* and fitting an exponential
    model σ(R) = a·exp(b·R) + c to each (frequency, bin-size) curve.  The
    r_squared field in each entry records the goodness of fit.

    Because the coefficients come from the manual's idealised figures, they
    represent nominal deployment conditions (uniform sound speed, no sidelobe
    contamination, no instrument tilt).  Real-world noise may differ.

    Parameters
    ----------
    path:
        Optional filesystem path to a custom coefficients JSON file.  If
        None, the file bundled with the pyadps package is used.

    Returns
    -------
    dict
        Nested dict keyed by frequency string (kHz) → bin-size string (m)
        → {"a": float, "b": float, "c": float, "r_squared": float}.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Coefficients file not found: {p}")
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        try:
            pkg = importlib_resources.files("pyadps")
            text = pkg.joinpath("velocity_noise_coefficients.json").read_text(encoding="utf-8")
            raw = json.loads(text)
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load velocity_noise_coefficients.json from package: {e}. "
                "Ensure pyadps is properly installed or supply coefficients_path."
            ) from e
    # Strip documentation keys (starting with "_") before returning so
    # callers can iterate over frequency keys without filtering.
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _extract_frequency(ds: xr.Dataset) -> int:
    """
    Return the most-common ADCP frequency (kHz) from the dataset.

    Tries the decoded ``frequency`` field first; falls back to decoding
    ``system_configuration_code`` with a warning.

    Raises
    ------
    ValueError
        If neither field is present, or if the decoded bit-code is unknown.
    """
    if "frequency" in ds.data_vars:
        most_common = Counter(ds["frequency"].values).most_common(1)[0][0]
        # Decoded field stores strings like "300 kHz"
        return int(str(most_common).split()[0])

    if "system_configuration_code" not in ds.data_vars:
        raise ValueError(
            "'frequency' and 'system_configuration_code' are both absent from the "
            "dataset. Provide frequency explicitly or reload with include_decoded=True."
        )

    logger.warning(
        "'frequency' field not found; decoding from 'system_configuration_code'. "
        "Consider reloading with include_decoded=True."
    )
    syscode = Counter(ds["system_configuration_code"].values).most_common(1)[0][0]
    bits = format(int(syscode), "016b")
    freq = _FREQ_BIT_MAP.get(bits[13:16])
    if freq is None:
        raise ValueError(
            f"Unknown frequency bit-code '{bits[13:16]}' in system_configuration_code."
        )
    return freq


def _extract_fl_params(ds: xr.Dataset) -> dict:
    """
    Extract Fixed Leader scalar parameters from the dataset.

    Uses the most-common value across all ensembles (robust for merged files).

    Returns
    -------
    dict
        Keys: ``frequency`` (int, kHz), ``bin_size`` (float, m),
        ``num_cells`` (int), ``pings_per_ensemble`` (int),
        ``depth_range`` (float, m).

    Raises
    ------
    KeyError
        If a required Fixed Leader field is absent.
    """
    def _most_common(ds: xr.Dataset, field: str):
        if field not in ds.data_vars:
            raise KeyError(
                f"Required field '{field}' not found in dataset. "
                "Ensure the dataset was loaded from read_fixed_leader() or pyadps.read()."
            )
        return Counter(ds[field].values).most_common(1)[0][0]

    frequency = _extract_frequency(ds)
    depth_cell_length_cm = int(_most_common(ds, "depth_cell_length"))
    bin_size = depth_cell_length_cm / 100.0  # cm → m
    num_cells = int(_most_common(ds, "num_cells"))
    pings_per_ensemble = int(_most_common(ds, "pings_per_ensemble"))
    depth_range = bin_size * num_cells

    return {
        "frequency": frequency,
        "bin_size": bin_size,
        "num_cells": num_cells,
        "pings_per_ensemble": pings_per_ensemble,
        "depth_range": depth_range,
    }


def compute_percent_good_threshold(
    ds: xr.Dataset,
    desired_std: float,
    depth_range: Optional[float] = None,
    n_pings: Optional[int] = None,
    bin_size: Optional[float] = None,
    frequency: Optional[int] = None,
    coefficients_path: Optional[str] = None,
) -> StdDevResult:
    """
    Compute the recommended percent-good cutoff for a desired current precision.

    Reads instrument parameters from the dataset (Fixed Leader fields), looks up
    the exponential noise curve from ``velocity_noise_coefficients.json``, and returns the
    minimum percent-good threshold that achieves *desired_std*.

    The exponential model is ``σ_single = a·exp(b·R) + c`` (cm/s), where *R* is
    the full profiling depth range (``bin_size × num_cells``).  Ensemble std is
    reduced by averaging: ``σ_ens = σ_single / √N``.  The required number of
    valid pings is therefore ``N_valid = (σ_single / desired_std)²``.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from ``pyadps.read()`` or ``read_fixed_leader()``.
    desired_std : float
        Target standard deviation in cm/s.
    depth_range : float, optional
        Override auto-detected depth range (m).  Use a shorter range if the
        valid data does not span the full profile; note that data beyond this
        depth should be excluded from analysis.
    n_pings : int, optional
        Override pings-per-ensemble from the dataset.
    bin_size : float, optional
        Override bin size (m) from the dataset.
    frequency : int, optional
        Override ADCP frequency (kHz) from the dataset.
    coefficients_path : str, optional
        Path to a custom ``velocity_noise_coefficients.json``.  Defaults to the
        package-bundled file.

    Returns
    -------
    StdDevResult
        Advisory result including single-ping std, ensemble std, valid pings
        required, and percent-good cutoff.

    Raises
    ------
    ValueError
        If *frequency* or *bin_size* has no exponential fit in the coefficients
        file, or if required Fixed Leader fields are missing and no override is
        provided.
    """
    if desired_std <= 0:
        raise ValueError(f"desired_std must be positive, got {desired_std}.")

    # --- Extract parameters from dataset, then apply overrides ---------------
    params = _extract_fl_params(ds)

    if frequency is not None:
        params["frequency"] = int(frequency)
    if bin_size is not None:
        params["bin_size"] = float(bin_size)
    if n_pings is not None:
        params["pings_per_ensemble"] = int(n_pings)
    if depth_range is not None:
        params["depth_range"] = float(depth_range)

    freq = params["frequency"]
    bsize = params["bin_size"]
    n_total = params["pings_per_ensemble"]
    d_range = params["depth_range"]

    # --- Load coefficients and validate freq / bin_size -----------------------
    coeffs = _load_adcp_coefficients(coefficients_path)

    freq_key = next((k for k in coeffs if int(k) == freq), None)
    if freq_key is None:
        valid_freqs = sorted(int(k) for k in coeffs)
        raise ValueError(
            f"Frequency {freq} kHz has no exponential fit in the coefficients file. "
            f"Available frequencies: {valid_freqs} kHz."
        )

    freq_coeffs = coeffs[freq_key]
    bin_key = next((k for k in freq_coeffs if abs(float(k) - bsize) < 1e-9), None)
    if bin_key is None:
        valid_bins = sorted(float(k) for k in freq_coeffs)
        raise ValueError(
            f"Bin size {bsize} m has no exponential fit for {freq} kHz. "
            f"Available bin sizes for {freq} kHz: {valid_bins} m."
        )

    a = freq_coeffs[bin_key]["a"]
    b = freq_coeffs[bin_key]["b"]
    c = freq_coeffs[bin_key]["c"]

    # --- Compute standard deviations ------------------------------------------
    single_ping_std = a * math.exp(b * d_range) + c
    ensemble_std = single_ping_std / math.sqrt(n_total)

    # --- Compute valid pings and percent-good cutoff --------------------------
    # N_valid = (σ_single / σ_desired)²  — pings needed to reach desired_std
    n_valid_float = (single_ping_std / desired_std) ** 2
    n_valid = math.ceil(n_valid_float)
    achievable = n_valid <= n_total

    if not achievable:
        logger.warning(
            f"Desired std {desired_std} cm/s requires {n_valid} valid pings but "
            f"only {n_total} pings are available per ensemble. "
            "The target precision cannot be achieved with the current configuration."
        )
        n_valid = n_total  # clamp — pg cutoff = 100 %

    pg_cutoff = n_valid * 100.0 / n_total

    return StdDevResult(
        frequency=freq,
        bin_size=bsize,
        depth_range=d_range,
        pings_per_ensemble=n_total,
        single_ping_std=single_ping_std,
        ensemble_std=ensemble_std,
        desired_std=desired_std,
        valid_pings_required=n_valid,
        percent_good_cutoff=pg_cutoff,
        achievable=achievable,
    )
