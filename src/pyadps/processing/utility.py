"""
Utility Module for ADCP Data Processing (v1.0.0).

This module provides shared dataclasses and core data functions used by all
Runner classes throughout the processing pipeline.

Dataclasses:
- QCCheckStats: Statistics for individual QC checks (shared across modules)
- DataModificationStats: Statistics for data modification operations
- QCPipelineReport: Complete QC pipeline report

Functions:
- create_default_mask: Create baseline 3D mask from velocity data
- replace_data: Replace a variable in the dataset with user-provided data

Note: Configuration management (ProcessingConfig, INI parsing/writing) lives
in config.py, not here. This module is intentionally limited to runtime data
operations so that Runner classes do not transitively import config machinery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
from numpy.typing import NDArray
import xarray as xr

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

VELOCITY_MISSING_VALUE: int = -32768


# ============================================================================
# SHARED STATISTICS DATACLASSES
# ============================================================================


@dataclass
class QCCheckStats:
    """
    Statistics for a single QC check.

    This dataclass is shared across all processing modules (sensor_health,
    signal_quality, profile_operation, velocity_check) to ensure consistent
    statistics tracking and reporting.

    Attributes
    ----------
    check_name : str
        Name of the check (e.g., "Roll Check", "Correlation Check")
    threshold : float | list[float] | tuple[float, float] | None
        Threshold value(s) used for the check.
        - float: Single threshold (e.g., roll=15.0)
        - list[float]: Per-beam thresholds (e.g., echo intensity noise floor)
        - tuple: Min/max range (e.g., pressure=(0, 1000))
        - None: No threshold (e.g., regrid operation)
    cells_pre_masked : int
        Number of cells already masked before this check
    cells_newly_masked : int
        Number of cells newly masked by this check (impact)
    cells_total_masked : int
        Total number of cells masked after this check
    total_cells : int
        Total number of cells in the mask
    check_time : datetime
        Timestamp when check was applied
    metadata : dict[str, Any]
        Additional check-specific metadata
    """

    check_name: str
    threshold: float | list[float] | tuple[float, float] | None
    cells_pre_masked: int
    cells_newly_masked: int
    cells_total_masked: int
    total_cells: int
    check_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pre_masked_pct(self) -> float:
        """Percentage of cells masked BEFORE this check."""
        if self.total_cells == 0:
            return 0.0
        return 100 * self.cells_pre_masked / self.total_cells

    @property
    def newly_masked_pct(self) -> float:
        """Percentage of cells NEWLY masked BY this check (impact)."""
        if self.total_cells == 0:
            return 0.0
        return 100 * self.cells_newly_masked / self.total_cells

    @property
    def total_masked_pct(self) -> float:
        """Percentage of cells masked AFTER this check (cumulative)."""
        if self.total_cells == 0:
            return 0.0
        return 100 * self.cells_total_masked / self.total_cells

    @property
    def valid_cells(self) -> int:
        """Number of cells that remain valid (not masked)."""
        return self.total_cells - self.cells_total_masked

    @property
    def valid_pct(self) -> float:
        """Percentage of valid cells remaining."""
        if self.total_cells == 0:
            return 0.0
        return 100 * self.valid_cells / self.total_cells

    def __str__(self) -> str:
        """Pretty-print statistics."""
        threshold_str = (
            f"{self.threshold}"
            if isinstance(self.threshold, (int, float))
            else str(self.threshold)
        )
        return (
            f"{self.check_name:25s} | "
            f"Threshold: {threshold_str:>12s} | "
            f"Pre: {self.pre_masked_pct:>5.1f}% | "
            f"Impact: {self.newly_masked_pct:>5.1f}% | "
            f"Valid: {self.valid_pct:>5.1f}%"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "check_name": self.check_name,
            "threshold": self.threshold,
            "cells_pre_masked": self.cells_pre_masked,
            "cells_newly_masked": self.cells_newly_masked,
            "cells_total_masked": self.cells_total_masked,
            "total_cells": self.total_cells,
            "pre_masked_pct": round(self.pre_masked_pct, 4),
            "newly_masked_pct": round(self.newly_masked_pct, 4),
            "total_masked_pct": round(self.total_masked_pct, 4),
            "valid_cells": self.valid_cells,
            "valid_pct": round(self.valid_pct, 4),
            "check_time": self.check_time.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class DataModificationStats:
    """
    Statistics for a data modification operation.

    Tracks modifications made to dataset variables by functions like
    replace_data() and correct_sound_speed().

    Attributes
    ----------
    operation : str
        Name of the operation (e.g., "replace_data", "correct_sound_speed")
    variable_name : str
        Name of the variable that was modified
    modification_time : datetime
        Timestamp when modification was applied
    original_stats : dict[str, float]
        Statistics of the original data (min, max, mean, std)
    modified_stats : dict[str, float]
        Statistics of the modified data (min, max, mean, std)
    metadata : dict[str, Any]
        Additional operation-specific metadata
    """

    operation: str
    variable_name: str
    modification_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    original_stats: dict[str, float] = field(default_factory=dict)
    modified_stats: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_change(self) -> float | None:
        """Change in mean value."""
        if "mean" in self.original_stats and "mean" in self.modified_stats:
            return self.modified_stats["mean"] - self.original_stats["mean"]
        return None

    @property
    def mean_change_pct(self) -> float | None:
        """Percentage change in mean value."""
        if self.mean_change is not None and self.original_stats.get("mean", 0) != 0:
            return 100 * self.mean_change / self.original_stats["mean"]
        return None

    def __str__(self) -> str:
        """Pretty-print modification statistics."""
        change_str = ""
        if self.mean_change is not None:
            change_str = f", change: {self.mean_change:+.2f}"
            if self.mean_change_pct is not None:
                change_str += f" ({self.mean_change_pct:+.2f}%)"

        return (
            f"{self.operation:25s} | "
            f"Variable: {self.variable_name:20s} | "
            f"Original mean: {self.original_stats.get('mean', 'N/A'):>10.2f} | "
            f"Modified mean: {self.modified_stats.get('mean', 'N/A'):>10.2f}"
            f"{change_str}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "operation": self.operation,
            "variable_name": self.variable_name,
            "modification_time": self.modification_time.isoformat(),
            "original_stats": self.original_stats,
            "modified_stats": self.modified_stats,
            "mean_change": self.mean_change,
            "mean_change_pct": self.mean_change_pct,
            "metadata": self.metadata,
        }


@dataclass
class QCPipelineReport:
    """
    Complete QC pipeline report.

    Aggregates statistics from all QC checks and data modifications
    across processing modules.

    Attributes
    ----------
    module_name : str
        Name of the processing module (e.g., "sensor_health", "signal_quality")
    baseline_masked : int
        Number of cells masked before any checks
    baseline_masked_pct : float
        Percentage of cells masked before any checks
    total_cells : int
        Total number of cells in the mask
    checks : list[QCCheckStats]
        List of statistics for each QC check applied
    modifications : list[DataModificationStats]
        List of statistics for data modifications
    timestamp : datetime
        When the report was generated
    """

    module_name: str
    baseline_masked: int
    baseline_masked_pct: float
    total_cells: int
    checks: list[QCCheckStats] = field(default_factory=list)
    modifications: list[DataModificationStats] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def final_valid_pct(self) -> float:
        """Percentage of data valid after all checks."""
        if self.checks:
            return self.checks[-1].valid_pct
        return 100 - self.baseline_masked_pct

    @property
    def pipeline_impact_pct(self) -> float:
        """Percentage of data additionally masked by this pipeline."""
        if self.checks:
            total = self.checks[-1].total_masked_pct
        else:
            total = self.baseline_masked_pct
        return total - self.baseline_masked_pct

    def __str__(self) -> str:
        """Format as human-readable string."""
        lines = [
            "=" * 80,
            f"{self.module_name.upper()} PIPELINE REPORT",
            "=" * 80,
            f"Generated: {self.timestamp.isoformat()}",
            f"Baseline masked: {self.baseline_masked:,} ({self.baseline_masked_pct:.2f}%)",
            f"Total cells: {self.total_cells:,}",
        ]

        if self.modifications:
            lines.append("-" * 80)
            lines.append("DATA MODIFICATIONS:")
            for mod in self.modifications:
                lines.append(f"  {mod}")

        if self.checks:
            lines.append("-" * 80)
            lines.append("QC CHECKS:")
            for check in self.checks:
                lines.append(f"  {check}")
            lines.append("-" * 80)
            lines.append(
                f"FINAL: {self.checks[-1].valid_cells:,} valid cells "
                f"({self.final_valid_pct:.1f}%) | "
                f"Pipeline impact: {self.pipeline_impact_pct:+.1f}%"
            )
        else:
            lines.append("-" * 80)
            lines.append("No QC checks applied.")

        lines.append("=" * 80)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "module_name": self.module_name,
            "baseline": {
                "masked_cells": self.baseline_masked,
                "masked_pct": round(self.baseline_masked_pct, 4),
                "total_cells": self.total_cells,
            },
            "modifications": [mod.to_dict() for mod in self.modifications],
            "checks": [check.to_dict() for check in self.checks],
            "summary": {
                "checks_applied": len(self.checks),
                "modifications_applied": len(self.modifications),
                "final_valid_cells": (
                    self.checks[-1].valid_cells
                    if self.checks
                    else self.total_cells - self.baseline_masked
                ),
                "final_valid_pct": round(self.final_valid_pct, 4),
                "pipeline_impact_pct": round(self.pipeline_impact_pct, 4),
            },
            "timestamp": self.timestamp.isoformat(),
        }


# ============================================================================
# CREATE DEFAULT MASK
# ============================================================================


def create_default_mask(ds: xr.Dataset) -> xr.DataArray:
    """
    Create a 3D baseline mask from velocity data based on missing values.

    This function generates a quality mask where values are marked as invalid (1)
    or valid (0) based on the RDI missing value code (-32768) in the velocity data.

    The mask has 4 beams corresponding to U, V, W velocity components and a
    combined signal quality mask:
    - mask[0, :, :] = U velocity failures
    - mask[1, :, :] = V velocity failures
    - mask[2, :, :] = W velocity failures
    - mask[3, :, :] = Combined mask (OR of U, V, W)

    The 4th beam is NOT based on error velocity (beam 3 of input data).
    Instead, it represents the combined mask of the first three velocity
    components, marking a cell as invalid if ANY of u, v, or w is missing.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing:
        - velocity : DataArray with dims (beam, cell, ensemble)
          Shape is typically (4, n_cells, n_ensembles) where:
          - beam 0 = U (east-west) velocity
          - beam 1 = V (north-south) velocity
          - beam 2 = W (vertical) velocity
          - beam 3 = Error velocity (diagnostic)

    Returns
    -------
    xr.DataArray
        3D mask with dims (beam, cell, time):
        - 1 = invalid data (missing value detected)
        - 0 = valid data
        - beam dimension represents [u_mask, v_mask, w_mask, signal_quality]

    Examples
    --------
    >>> ds = read('file.000')
    >>> mask = create_default_mask(ds)
    >>> print(mask.shape)  # (4, n_cells, n_ensembles)
    """
    velocity = ds["velocity"]

    # Get dimensions
    n_beams = velocity.sizes["beam"]
    n_cells = velocity.sizes["cell"]
    n_ensembles = velocity.sizes["time"]

    # Validate expected shape
    if n_beams < 4:
        logger.warning(
            f"Expected 4 beams in velocity data, got {n_beams}. "
            "Mask will be created for available beams only."
        )

    # Create mask array (same shape as velocity)
    # 1 = invalid (missing), 0 = valid
    mask_data = np.zeros((4, n_cells, n_ensembles), dtype=np.int8)

    # Extract velocity values
    vel_values = velocity.values  # Shape: (beam, cell, ensemble)

    # Create masks for U, V, W components (beams 0, 1, 2)
    # Do NOT mask based on error velocity (beam 3)
    # Use NaN detection for float arrays, sentinel comparison for int arrays
    is_nan_mode = np.issubdtype(vel_values.dtype, np.floating)
    for beam_idx in range(min(3, n_beams)):
        if is_nan_mode:
            mask_data[beam_idx, :, :] = np.isnan(
                vel_values[beam_idx, :, :]
            ).astype(np.int8)
        else:
            mask_data[beam_idx, :, :] = (
                vel_values[beam_idx, :, :] <= VELOCITY_MISSING_VALUE
            ).astype(np.int8)

    # Create combined signal quality mask (4th beam)
    # This is the logical OR of U, V, W masks
    # If ANY of u, v, w is missing, mark as invalid
    mask_data[3, :, :] = (
        mask_data[0, :, :] | mask_data[1, :, :] | mask_data[2, :, :]
    ).astype(np.int8)

    # Coordinates
    coords = {
        "beam": ds.coords["beam"].values,
        "cell": ds.coords["cell"].values,
        "time": ds.coords["time"].values,
    }

    # Mask variable attributes
    mask_attrs = {
        "long_name": "Velocity quality mask",
        "description": (
            "Baseline quality mask based on missing values in velocity data. "
            "1 = invalid (missing), 0 = valid."
        ),
        "units": "dimensionless",
        "missing_value_code": VELOCITY_MISSING_VALUE,
        "flag_values": "0, 1",
        "flag_meanings": "valid invalid",
        "beam_0": "U velocity component mask",
        "beam_1": "V velocity component mask",
        "beam_2": "W velocity component mask",
        "beam_3": "Combined signal quality mask (U OR V OR W)",
        "comment": (
            "The 4th beam (beam_3) represents combined signal quality, not error "
            "velocity. It is the logical OR of U, V, W masks - marking a cell as "
            "invalid if ANY physical velocity component is missing."
        ),
    }

    # Create xarray.DataArray
    ds_mask = xr.DataArray(
        data=mask_data,
        dims=["beam", "cell", "time"],
        coords=coords,
        attrs=mask_attrs,
    )

    # Log statistics
    u_invalid = int((mask_data[0, :, :] == 1).sum())
    v_invalid = int((mask_data[1, :, :] == 1).sum())
    w_invalid = int((mask_data[2, :, :] == 1).sum())
    combined_invalid = int((mask_data[3, :, :] == 1).sum())
    total_cells = n_cells * n_ensembles

    logger.debug(
        f"Created velocity mask: shape={ds_mask.shape}, "
        f"U invalid={u_invalid} ({100*u_invalid/total_cells:.2f}%), "
        f"V invalid={v_invalid} ({100*v_invalid/total_cells:.2f}%), "
        f"W invalid={w_invalid} ({100*w_invalid/total_cells:.2f}%), "
        f"combined invalid={combined_invalid} ({100*combined_invalid/total_cells:.2f}%)"
    )

    return ds_mask


# ============================================================================
# REPLACE DATA
# ============================================================================


def replace_data(
    ds: xr.Dataset,
    data: NDArray,
    variable_name: str,
    apply_scale_factor: bool = True,
) -> xr.Dataset:
    """
    Replace a variable in the dataset with user-provided numpy array data.

    This function validates that the variable exists, checks that the new data
    has matching dimensions and shape, applies scale factor conversion if present,
    preserves the original variable's attributes, and returns a new dataset with
    the replaced variable.

    Parameters
    ----------
    ds : xr.Dataset
        Input xarray Dataset containing the variable to replace.
    data : np.ndarray
        New data to replace the existing variable. Must have the same shape
        as the original variable. Data should be in physical units (e.g.,
        degrees Celsius for temperature, PSU for salinity).
    variable_name : str
        Name of the variable to replace (e.g., "temperature", "salinity").
    apply_scale_factor : bool, default True
        If True and the variable has a 'scale_factor' attribute, divide the
        input data by the scale factor to convert to RDI internal format.
        If False, use the data as-is.

    Returns
    -------
    xr.Dataset
        New dataset with the replaced variable. Original dataset is not modified.
        The replaced variable retains its original attributes.

    Raises
    ------
    ValueError
        If the variable is not found in the dataset.
    ValueError
        If the new data shape does not match the original variable shape.
    ValueError
        If the new data has different number of dimensions than the original.

    Warns
    -----
    UserWarning
        If the variable has no 'scale_factor' attribute and apply_scale_factor
        is True, warning the user to ensure data is in proper RDI format.

    Notes
    -----
    - The original dataset is not modified; a deep copy is returned.
    - All original attributes of the variable are preserved.
    - The new data is converted to the same dtype as the original variable.
    - RDI format uses integer storage with scale factors:
      - temperature: scale_factor=0.01 (stored as centidegrees)
      - salinity: scale_factor=1 (stored as parts per thousand * 1000)
      - pressure: scale_factor=1 (stored as decapascals)

    Examples
    --------
    >>> ds = read('file.000')
    >>> # Replace temperature with CTD data (in Â°C)
    >>> new_temp = np.ones(100) * 15.0  # 15Â°C
    >>> ds_updated = replace_data(ds, new_temp, "temperature")
    >>> # Data is automatically scaled: 15.0 / 0.01 = 1500 (stored value)
    >>> print(ds_updated["temperature"].values.mean())
    1500.0

    >>> # Replace without automatic scaling (data already in RDI format)
    >>> new_temp_rdi = np.ones(100) * 1500  # Already in 0.01Â° units
    >>> ds_updated = replace_data(ds, new_temp_rdi, "temperature",
    ...                           apply_scale_factor=False)

    See Also
    --------
    correct_sound_speed : Uses temperature/salinity data
    """

    # Step 1: Check if variable exists in dataset
    if variable_name not in ds.data_vars:
        raise ValueError(
            f"Variable '{variable_name}' not found in dataset. "
            f"Available variables: {list(ds.data_vars)}"
        )

    # Step 2: Get original variable
    original_var = ds[variable_name]
    original_shape = original_var.shape
    original_dims = original_var.dims
    original_attrs = original_var.attrs.copy()
    original_dtype = original_var.dtype

    # Step 3: Validate new data dimensions
    if data.ndim != len(original_dims):
        raise ValueError(
            f"Dimension mismatch: new data has {data.ndim} dimensions, "
            f"but '{variable_name}' has {len(original_dims)} dimensions {original_dims}."
        )

    # Step 4: Validate new data shape
    if data.shape != original_shape:
        raise ValueError(
            f"Shape mismatch: new data has shape {data.shape}, "
            f"but '{variable_name}' has shape {original_shape}."
        )

    # Step 5: Apply scale factor if present and requested
    scaled_data = data.copy()
    scale_factor = original_attrs.get("scale_factor", None)

    if apply_scale_factor:
        if scale_factor is not None:
            # Divide by scale factor to convert to RDI internal format
            # e.g., 15.0Â°C / 0.01 = 1500 (stored value)
            scaled_data = data / scale_factor
            logger.info(
                f"Applied scale_factor={scale_factor} to '{variable_name}': "
                f"input range [{data.min():.4f}, {data.max():.4f}] -> "
                f"stored range [{scaled_data.min():.1f}, {scaled_data.max():.1f}]"
            )

    # Step 6: Create deep copy of dataset
    ds_out = ds.copy(deep=True)

    # Step 7: Create new DataArray with same structure
    new_var = xr.DataArray(
        data=scaled_data.astype(original_dtype),
        dims=original_dims,
        coords={dim: ds.coords[dim] for dim in original_dims if dim in ds.coords},
        attrs=original_attrs,
    )

    # Step 8: Add metadata about replacement
    new_var.attrs["data_replaced"] = 1
    new_var.attrs["replacement_scale_factor_applied"] = (
        apply_scale_factor and scale_factor is not None
    )

    # Step 9: Replace variable in dataset
    ds_out[variable_name] = new_var

    logger.info(
        f"Replaced variable '{variable_name}': shape={original_shape}, "
        f"dtype={original_dtype}, scale_factor_applied={apply_scale_factor and scale_factor is not None}"
    )

    return ds_out
