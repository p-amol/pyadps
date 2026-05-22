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

import logging
from datetime import datetime, timezone
from typing import Any

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
    threebeam: bool = False,
    beam_ignore: int | None = None,
) -> xr.Dataset:
    """
    Perform correlation strength quality control check.

    Flags cells where the correlation count is below the cutoff.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing correlation data.
    cutoff : float, default 64
        Minimum acceptable correlation value (0-255 scale).
    threebeam : bool, default False
        Enable three-beam mode (ignore one beam).
    beam_ignore : int, optional
        Beam index to ignore in three-beam mode (0-3).

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

    # Flag values < cutoff
    flag = correlation < cutoff

    # Handle 3-beam mode
    if threebeam and beam_ignore is not None and 0 <= beam_ignore <= 3:
        if "beam" in flag.coords:
            is_ignored = flag["beam"] == beam_ignore
            flag = flag & (~is_ignored)
            logger.info(f"Three-beam mode: ignoring beam {beam_ignore}")

    # Update mask
    mask_updated = xr.where(flag, 1, mask).astype(np.int8)
    mask_updated.attrs = mask.attrs.copy()

    ds_out = ds.copy(deep=True)
    ds_out["mask"] = mask_updated

    newly_flagged = int((mask_updated == 1).sum()) - int((mask == 1).sum())
    logger.info(
        f"Correlation check applied: cutoff={cutoff}, "
        f"newly flagged cells: {newly_flagged}"
    )

    return ds_out


def echo_intensity_check(
    ds: xr.Dataset,
    cutoff: float = DEFAULT_ECHO_THRESHOLD,
    threebeam: bool = False,
    beam_ignore: int | None = None,
) -> xr.Dataset:
    """
    Perform echo intensity (signal strength) quality control check.

    Flags cells where echo intensity is below the cutoff.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing echo intensity data.
    cutoff : float, default 40
        Minimum acceptable echo intensity value (0-255 scale).
    threebeam : bool, default False
        Enable three-beam mode (ignore one beam).
    beam_ignore : int, optional
        Beam index to ignore in three-beam mode (0-3).

    Returns
    -------
    xr.Dataset
        Dataset with updated mask.
    """
    _validate_threshold("echo_intensity", cutoff)

    var_name = "echo_intensity"
    if var_name not in ds.data_vars:
        if "echo" in ds.data_vars:
            var_name = "echo"
        else:
            logger.warning("Echo intensity data not found")
            return ds

    mask = _get_mask_or_create(ds)
    echo = ds[var_name]

    flag = echo < cutoff

    if threebeam and beam_ignore is not None and 0 <= beam_ignore <= 3:
        if "beam" in flag.coords:
            is_ignored = flag["beam"] == beam_ignore
            flag = flag & (~is_ignored)
            logger.info(f"Three-beam mode: ignoring beam {beam_ignore}")

    mask_updated = xr.where(flag, 1, mask).astype(np.int8)
    mask_updated.attrs = mask.attrs.copy()

    ds_out = ds.copy(deep=True)
    ds_out["mask"] = mask_updated

    newly_flagged = int((mask_updated == 1).sum()) - int((mask == 1).sum())
    logger.info(
        f"Echo intensity check applied: cutoff={cutoff}, "
        f"newly flagged cells: {newly_flagged}"
    )

    return ds_out


def error_velocity_check(
    ds: xr.Dataset,
    cutoff: float = DEFAULT_ERROR_VELOCITY_THRESHOLD,
) -> xr.Dataset:
    """
    Perform error velocity quality control check.

    Flags combined signal quality (Beam 3) if error velocity exceeds cutoff.

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

    # Update Combined Mask (Beam 3) only
    flag_3d = xr.zeros_like(mask, dtype=bool)
    if mask.sizes["beam"] > 3:
        flag_3d.loc[dict(beam=mask.coords["beam"].values[3])] = flag

    mask_updated = xr.where(flag_3d, 1, mask).astype(np.int8)
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

    # Update Combined Mask (Beam 3)
    flag_3d = xr.zeros_like(mask, dtype=bool)
    if mask.sizes["beam"] > 3:
        flag_3d.loc[dict(beam=mask.coords["beam"].values[3])] = flag

    mask_updated = xr.where(flag_3d, 1, mask).astype(np.int8)
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
    threebeam: bool = True,
    beam_ignore: int | None = None,
) -> xr.Dataset:
    """
    Detect and flag false targets using echo intensity anomaly detection.

    False targets occur when one beam receives a strong echo (e.g., from fish,
    debris, or air bubbles) that is inconsistent with other beams. This check
    identifies cells where the difference between echo intensities exceeds
    a threshold.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing echo intensity data.
    cutoff : float, default 50
        Maximum acceptable difference between echo intensity values.
    threebeam : bool, default True
        If True and beam_ignore is None, compares highest to second-highest
        echo intensity (more lenient, allows one outlier beam).
        If False, compares highest to lowest (stricter check).
    beam_ignore : int, optional
        Beam index (0-3) to exclude from the comparison. When specified,
        that beam's data is removed before computing the difference.
        If provided with threebeam=True, the remaining beams use max-min.

    Returns
    -------
    xr.Dataset
        Dataset with updated mask.

    Notes
    -----
    The algorithm works as follows:
    1. If beam_ignore is specified, remove that beam from consideration
    2. Sort echo values along beam dimension
    3. If threebeam=True and beam_ignore is None: difference = max - second_highest
       Otherwise: difference = max - min
    4. Flag cells where difference > cutoff

    The threebeam=True mode is more permissive, only flagging when the highest
    value stands out significantly from the second-highest, allowing for normal
    beam-to-beam variation.
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

    if threebeam and beam_ignore is None:
        # Compare highest to second-highest (more lenient)
        # Take last two values along beam axis
        echo_max = np.take(sorted_echo, -1, axis=beam_axis)
        echo_second = np.take(sorted_echo, -2, axis=beam_axis)
        difference = echo_max - echo_second
        logger.debug("False target: using max - second_highest (threebeam mode)")
    else:
        # Compare highest to lowest (stricter)
        echo_max = np.take(sorted_echo, -1, axis=beam_axis)
        echo_min = np.take(sorted_echo, 0, axis=beam_axis)
        difference = echo_max - echo_min
        logger.debug("False target: using max - min")

    # Convert difference back to DataArray
    difference_da = xr.DataArray(
        difference,
        dims=non_beam_dims,
        coords=non_beam_coords,
    )

    flag = difference_da > cutoff

    # Update Combined Mask (Beam 3)
    flag_3d = xr.zeros_like(mask, dtype=bool)
    if mask.sizes["beam"] > 3:
        flag_3d.loc[dict(beam=mask.coords["beam"].values[3])] = flag

    mask_updated = xr.where(flag_3d, 1, mask).astype(np.int8)
    mask_updated.attrs = mask.attrs.copy()

    ds_out = ds.copy(deep=True)
    ds_out["mask"] = mask_updated

    newly_flagged = int((mask_updated == 1).sum()) - int((mask == 1).sum())
    logger.info(
        f"False target detection applied: cutoff={cutoff}, threebeam={threebeam}, "
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
        cutoff: float,
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
        threebeam: bool = False,
        beam_ignore: int | None = None,
    ) -> SignalQualityRunner:
        """
        Apply correlation check.

        Parameters
        ----------
        cutoff : float, default 64
            Minimum acceptable correlation value.
        threebeam : bool, default False
            Enable three-beam mode.
        beam_ignore : int, optional
            Beam to ignore in three-beam mode.

        Returns
        -------
        SignalQualityRunner
            Self for method chaining.
        """
        return self._record_check(
            "Correlation",
            correlation_check,
            cutoff,
            threebeam=threebeam,
            beam_ignore=beam_ignore,
        )

    def echo_intensity(
        self,
        cutoff: float = DEFAULT_ECHO_THRESHOLD,
        threebeam: bool = False,
        beam_ignore: int | None = None,
    ) -> SignalQualityRunner:
        """
        Apply echo intensity check.

        Parameters
        ----------
        cutoff : float, default 40
            Minimum acceptable echo intensity value.
        threebeam : bool, default False
            Enable three-beam mode.
        beam_ignore : int, optional
            Beam to ignore in three-beam mode.

        Returns
        -------
        SignalQualityRunner
            Self for method chaining.
        """
        return self._record_check(
            "Echo Intensity",
            echo_intensity_check,
            cutoff,
            threebeam=threebeam,
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
        threebeam: bool = True,
        beam_ignore: int | None = None,
    ) -> SignalQualityRunner:
        """
        Apply false target detection.

        Parameters
        ----------
        cutoff : float, default 50
            Maximum acceptable echo intensity difference.
        threebeam : bool, default True
            If True and beam_ignore is None, compares highest to second-highest.
            If False, compares highest to lowest.
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
            threebeam=threebeam,
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
