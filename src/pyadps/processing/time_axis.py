# ============================================================================
# UTILITY FUNCTIONS FOR DATA TRANSFORMATION
# ============================================================================
# These functions transform xarray.Dataset objects and return new datasets.
# They follow the principle of immutability - original datasets are never modified.
# ============================================================================

import logging
from typing import Dict, Optional, Union, Any

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def snap_time_axis(
    ds: xr.Dataset,
    freq: str = "h",
    tolerance: Union[str, pd.Timedelta] = "5min",
    target_minute: Optional[int] = None,
) -> tuple[Optional[xr.Dataset], bool, str]:
    """
    Snap (round) time axis to nearest frequency with safety checks.

    This utility function snaps the time axis to the nearest frequency while
    enforcing a tolerance check to prevent excessive time corrections. Unlike
    v0.4.0's in-place modification, this function returns a new dataset,
    following xarray's immutable design philosophy.

    The function provides two modes:
    1. **Simple snapping**: Round to nearest frequency (e.g., nearest hour)
    2. **Targeted snapping**: Round to a specific minute within the hour
       (e.g., snap to HH:30)

    Safety Features:
    - Tolerance validation: Aborts if any correction exceeds tolerance
    - Duplicate detection: Prevents snapping that would create duplicate timestamps
    - Original preservation: Input dataset is never modified

    Parameters
    ----------
    ds : xr.Dataset
        Input ADCP dataset with a 'time' coordinate
    freq : str, default 'h'
        Frequency to round to. Uses pandas frequency strings:
        - 'h' = hour
        - 'D' = day
        - 'T' or 'min' = minute
        - 'S' = second
        - '6h' = 6 hours
        - '30min' = 30 minutes
        See pandas frequency documentation for full list.
    tolerance : str or pd.Timedelta, default '5min'
        Maximum allowed correction. If any timestamp would require a
        correction larger than tolerance, the operation is aborted.
        Examples: '5min', '1h', '30S', '2min'
    target_minute : int, optional
        If specified, snap to nearest hour at a specific minute.
        Must be 0-59. For example:
        - target_minute=0 → snap to HH:00
        - target_minute=30 → snap to HH:30
        When used, freq parameter is ignored.

    Returns
    -------
    tuple of (xr.Dataset or None, bool, str)
        - xr.Dataset: New dataset with snapped time (None if operation failed)
        - bool: Success flag (True = success, False = failed)
        - str: Success message or error message

    Raises
    ------
    ValueError
        If target_minute is not in range 0-59
    TypeError
        If ds is not an xarray.Dataset

    Examples
    --------
    Snap to nearest hour:

    >>> import pyadps
    >>> ds = pyadps.read('file.000')
    >>> ds_snapped, success, msg = pyadps.snap_time_axis(ds, freq='h')
    >>> if success:
    ...     print(f"Snapped: {msg}")
    ...     print(ds_snapped.time.values)
    ... else:
    ...     print(f"Failed: {msg}")

    Snap to nearest 30-minute mark with tighter tolerance:

    >>> ds_snapped, success, msg = pyadps.snap_time_axis(
    ...     ds,
    ...     freq='h',
    ...     target_minute=30,
    ...     tolerance='3min'
    ... )

    Snap to nearest 6-hour boundary:

    >>> ds_snapped, success, msg = pyadps.snap_time_axis(
    ...     ds,
    ...     freq='6h',
    ...     tolerance='30min'
    ... )

    Notes
    -----
    The original dataset is never modified. To use the snapped dataset:

    >>> ds_snapped, success, msg = pyadps.snap_time_axis(ds)
    >>> if success:
    ...     ds = ds_snapped  # Explicitly use the new dataset

    See Also
    --------
    fill_time_gaps : Fill missing time values in a time series
    """

    # Type validation - defensive check with try-catch
    try:
        if not isinstance(ds, xr.Dataset):
            return None, False, "Input must be an xarray.Dataset"

        # Check that time coordinate exists
        if "time" not in ds.coords:
            return None, False, "Dataset does not have a 'time' coordinate"
    except (AttributeError, TypeError) as e:
        return (
            None,
            False,
            f"Input must be an xarray.Dataset (error: {type(e).__name__})",
        )

    # Check that time coordinate exists

    # Validate tolerance
    if isinstance(tolerance, str):
        try:
            tolerance_delta = pd.to_timedelta(tolerance)
        except ValueError as e:
            return None, False, f"Invalid tolerance format: {str(e)}"
    else:
        tolerance_delta = tolerance

    # Validate target_minute if provided
    if target_minute is not None:
        if (
            not isinstance(target_minute, int)
            or target_minute < 0
            or target_minute > 59
        ):
            return (
                None,
                False,
                f"target_minute must be integer 0-59, got {target_minute}",
            )

    # Convert time to pandas Series for manipulation
    time_data = pd.Series(ds.time.values)

    # Perform the snapping
    try:
        if target_minute is not None:
            # Snap to specific minute within the hour
            offset = pd.to_timedelta(target_minute, unit="m")
            snapped_time = (time_data - offset).dt.round("h") + offset
            action = f"nearest hour at minute {target_minute}"
        else:
            # Snap to specified frequency
            snapped_time = time_data.dt.round(freq)
            action = f"nearest {freq}"
    except Exception as e:
        return None, False, f"Error during rounding: {str(e)}"

    # Safety Check 1: Tolerance validation
    max_correction = (time_data - snapped_time).abs().max()
    if max_correction > tolerance_delta:
        return (
            None,
            False,
            f"Aborted: Max correction {max_correction} exceeds tolerance {tolerance_delta}",
        )

    # Safety Check 2: Duplicate detection
    if snapped_time.duplicated().any():
        return (
            None,
            False,
            "Aborted: Snapping would create duplicate timestamps",
        )

    # Create new dataset with updated time coordinate
    try:
        ds_new = ds.assign_coords({"time": snapped_time.values})
    except Exception as e:
        return None, False, f"Error creating new dataset: {str(e)}"

    success_msg = f"Successfully snapped to {action}. Max correction: {max_correction}"
    logger.info(success_msg)
    return ds_new, True, success_msg


def _timedelta_to_freq_string(td: pd.Timedelta) -> str:
    """
    Convert pandas Timedelta to frequency string.

    Helper function that converts a Timedelta object into a frequency string
    compatible with pd.date_range(). Used internally by fill_time_gaps().

    Parameters
    ----------
    td : pd.Timedelta
        Timedelta to convert

    Returns
    -------
    str
        Frequency string compatible with pd.date_range()

    Examples
    --------
    >>> td = pd.Timedelta(hours=1)
    >>> _timedelta_to_freq_string(td)
    'h'

    >>> td = pd.Timedelta(minutes=30)
    >>> _timedelta_to_freq_string(td)
    '30min'

    >>> td = pd.Timedelta(seconds=30)
    >>> _timedelta_to_freq_string(td)
    '30S'
    """
    total_seconds = td.total_seconds()

    if total_seconds % 3600 == 0:  # Hour
        hours = int(total_seconds / 3600)
        return f"{hours}h" if hours > 1 else "h"
    elif total_seconds % 60 == 0:  # Minute
        minutes = int(total_seconds / 60)
        return f"{minutes}min" if minutes > 1 else "min"
    else:  # Second
        return f"{int(total_seconds)}S"


def fill_time_gaps(
    ds: xr.Dataset,
    method: Union[str, pd.Timedelta] = "auto",
    forward_fill_fixed_leader: bool = True,
    forward_fill_variable_leader: bool = True,
    missing_values: Optional[Dict[str, Any]] = None,
) -> xr.Dataset:
    """
    Fill gaps in time axis to create regular time series.

    This utility function reindexes a dataset to a regular time grid, filling
    missing time values. Useful for time series with irregular sampling or
    missing ensembles. The function provides flexible filling strategies for
    both regular data variables and structured metadata.

    The function:
    1. **Detects or validates frequency**: Auto-detects from existing time data
       or accepts explicit frequency string
    2. **Creates regular time grid**: Generates complete time range at specified
       frequency
    3. **Reindexes data**: Aligns all variables to regular grid
    4. **Fills missing values**: Uses sensible defaults for ADCP data types
       or custom values provided by user

    Filling Strategy:
    - **Velocity, Correlation, Echo**: Fill with ADCP missing value codes
      (e.g., -32768 for velocity)
    - **Fixed/Variable Leader**: Forward-fill metadata (propagate last known value)
    - **Custom variables**: Use user-provided missing_values dict

    Parameters
    ----------
    ds : xr.Dataset
        Input ADCP dataset with a 'time' coordinate
    method : str or pd.Timedelta, default 'auto'
        Frequency for regular time grid:
        - 'auto': Auto-detect from median time interval
        - 'h': Hourly
        - 'D': Daily
        - 'T' or '30T': 30-minute
        - '6h': 6-hour intervals
        - pd.Timedelta(hours=1): Explicit timedelta
    forward_fill_fixed_leader : bool, default True
        If True, forward-fill 'fixed_leader' variable with last known values
        (appropriate for quasi-static configuration data)
    forward_fill_variable_leader : bool, default True
        If True, forward-fill 'variable_leader' variable with last known values
        (appropriate for diagnostic variables)
    missing_values : dict, optional
        User-provided mapping of variable names to missing value codes.
        Examples: {'velocity': -32768, 'temperature': np.nan}
        Overrides built-in defaults.

    Returns
    -------
    xr.Dataset
        New dataset with regular time grid. Original dataset is never modified.

    Raises
    ------
    ValueError
        If frequency cannot be auto-detected and method is 'auto'
    TypeError
        If ds is not an xarray.Dataset

    Examples
    --------
    Auto-detect frequency and fill gaps:

    >>> import pyadps
    >>> ds = pyadps.read('file.000')
    >>> ds_filled = pyadps.fill_time_gaps(ds, method='auto')

    Fill with explicit hourly frequency:

    >>> ds_filled = pyadps.fill_time_gaps(ds, method='h')

    Fill with custom missing values:

    >>> ds_filled = pyadps.fill_time_gaps(
    ...     ds,
    ...     method='30T',
    ...     missing_values={'velocity': -32768, 'temperature': -999}
    ... )

    Disable forward-filling of metadata:

    >>> ds_filled = pyadps.fill_time_gaps(
    ...     ds,
    ...     method='auto',
    ...     forward_fill_fixed_leader=False,
    ...     forward_fill_variable_leader=False
    ... )

    Notes
    -----
    Forward-filling is appropriate for:
    - Configuration variables that don't change per ensemble
    - Diagnostic metadata that applies to multiple ensembles

    It is NOT appropriate for:
    - Measured data variables (velocity, temperature, etc.)
    - Time-series that require explicit NaN handling

    The original dataset is never modified:

    >>> ds_filled = pyadps.fill_time_gaps(ds)  # ds unchanged
    >>> # Original ds.time still has gaps, ds_filled.time is regular

    See Also
    --------
    snap_time_axis : Snap time axis to nearest frequency
    """

    try:
        if not isinstance(ds, xr.Dataset):
            raise TypeError("Input must be an xarray.Dataset")

        # Check that time coordinate exists
        if "time" not in ds.coords:
            raise ValueError("Dataset does not have a 'time' coordinate")
    except (AttributeError, TypeError, ValueError) as e:
        if isinstance(e, ValueError):
            raise  # Re-raise ValueError from coordinate check
        raise TypeError(f"Input must be an xarray.Dataset (error: {type(e).__name__})")

    # Determine frequency
    if method == "auto":
        # Auto-detect from time intervals

        # Explicitly provide type for time_diffs as pd.Series to help type checker
        time_diffs: pd.Series = pd.Series(ds.time.values).diff().dropna()

        # time_diffs = pd.Series(ds.time.values).diff().dropna()
        if len(time_diffs) == 0:
            raise ValueError(
                "Cannot auto-detect frequency. Dataset has fewer than 2 time points."
            )

        # Use median interval (more robust than mean for irregular data)
        interval = time_diffs.median()
        if interval is None or not isinstance(interval, pd.Timedelta):
            raise ValueError(
                "Cannot auto-detect frequency. Dataset may have insufficient time points."
            )
        freq_str = _timedelta_to_freq_string(interval)
        logger.info(f"Auto-detected frequency: {freq_str} (interval: {interval})")
    elif isinstance(method, pd.Timedelta):
        freq_str = _timedelta_to_freq_string(method)
        logger.info(f"Using provided timedelta frequency: {freq_str}")
    else:
        freq_str = method
        logger.info(f"Using provided frequency string: {freq_str}")

    # Create regular time grid
    time_min = pd.Timestamp(ds.time.min().values)
    time_max = pd.Timestamp(ds.time.max().values)
    new_time = pd.date_range(start=time_min, end=time_max, freq=freq_str)

    logger.info(
        f"Filling time gaps: {len(ds.time)} existing → {len(new_time)} regular points"
    )

    # Snapshot original dtypes before reindexing.
    # xr.reindex upcasts integer arrays to float64 to accommodate NaN slots, so
    # checking dtype after reindex would misidentify integer vars as float.
    original_dtypes: dict[str, np.dtype] = {
        str(v): ds[v].dtype for v in ds.data_vars
    }

    # Reindex all variables to new time axis
    try:
        ds_reindexed = ds.reindex(time=new_time)
    except Exception as e:
        logger.error(f"Error during reindexing: {str(e)}")
        raise

    # Define default missing values for ADCP data types
    default_missing = {
        "velocity": -32768,
        "correlation": 255,
        "echo_intensity": 255,
        "percent_good": 255,
        "status": 0,
    }

    # Merge with user-provided missing values
    if missing_values is not None:
        default_missing.update(missing_values)

    # Collect per-variable fills, then apply in one shot with assign().
    # Avoids mutating ds_reindexed while iterating over its data_vars,
    # which can silently fail in xarray 2024.x.
    updates: dict = {}
    for var_name in list(ds_reindexed.data_vars):
        var_str = str(var_name)
        try:
            if var_str in default_missing:
                # Float-native variables (missing_as_nan=True, e.g. velocity) use NaN
                # as missing — leave gaps as NaN rather than overwriting with sentinel.
                # Use the pre-reindex dtype; reindex upcasts int → float64 for NaN slots.
                original_dtype = original_dtypes.get(var_str, ds_reindexed[var_name].dtype)
                if np.issubdtype(original_dtype, np.floating):
                    logger.debug(f"Skipped fill for float variable {var_str} (NaN gaps retained)")
                else:
                    missing_val = default_missing[var_str]
                    updates[var_str] = ds_reindexed[var_name].fillna(missing_val)
                    logger.debug(f"Filled {var_str} with missing value: {missing_val}")
            elif forward_fill_fixed_leader and var_str == "fixed_leader":
                updates[var_str] = ds_reindexed[var_name].ffill(dim="time")
                logger.debug(f"Forward-filled {var_str}")
            elif forward_fill_variable_leader and var_str == "variable_leader":
                updates[var_str] = ds_reindexed[var_name].ffill(dim="time")
                logger.debug(f"Forward-filled {var_str}")
        except Exception as e:
            logger.warning(f"Error filling {var_str}: {str(e)}")

    if updates:
        ds_reindexed = ds_reindexed.assign(updates)

    logger.info(
        f"Time gaps filled successfully. New time range: {time_min} to {time_max}"
    )
    return ds_reindexed
