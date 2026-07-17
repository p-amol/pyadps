"""
Profile Operation Module for ADCP Data Processing (v1.1.0).

This module provides xarray-compatible functions for profile-level operations including:
- Ensemble/time trimming (mask-based)
- Bin cutting (mask-based, manual and side-lobe)
- Data regridding to regular depth grids (dataset modification)

INTEGRATED FEATURES (v1.1.0):
- ProfileOperationRunner for pipeline orchestration
- Support for both mask-based operations and dataset modifications
- Integration with shared utility module for statistics tracking
- Consistent API with SignalQualityRunner and VelocityCheckRunner

Key Design Principles:
- **xarray-native**: Operates on xr.Dataset objects
- **Immutable**: Returns new datasets without modifying inputs
- **Mask-based trimming/cutting**: Preserves original data, updates mask
- **QC before regridding**: QC checks must be done on original cells before regridding
- **Statistics tracking**: Comprehensive reporting via shared dataclasses

IMPORTANT SEQUENCING:
1. Quality control checks MUST happen BEFORE regridding
2. Regridding CHANGES dataset structure (cell and depth dimension)
3. After regridding, cell-based masks are no longer compatible
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import scipy.interpolate as sp_interp
import xarray as xr

from .utility import (
    create_default_mask,
    QCCheckStats,
    DataModificationStats,
    QCPipelineReport,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_BEAM_ANGLE = 20  # degrees, typical RDI ADCP
DEFAULT_BEAM_DIRECTION = "up"  # up or down
DEFAULT_EXTRA_CELLS = 1  # Extra cells to mask for side lobe
DEFAULT_REGRID_METHOD = "nearest"
DEFAULT_END_CELL_OPTION = "cell"


# ============================================================================
# UTILITY FUNCTIONS - Data extraction and validation
# ============================================================================


def _extract_dataset_params(
    ds: xr.Dataset,
    beam_direction: Optional[str] = None,
    beam_angle: Optional[float] = None,
) -> dict[str, Any]:
    """
    Extract ADCP configuration parameters from xarray.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input ADCP dataset with attributes
    beam_direction : str, optional
        Override beam direction ('up' or 'down')
    beam_angle : float, optional
        Override beam angle in degrees

    Returns
    -------
    dict
        Dictionary with keys:
        - 'beam_direction': 'up' or 'down'
        - 'beam_angle': angle in degrees
        - 'cell_size': cell size in cm
        - 'bin1_distance': distance to first bin in cm
        - 'num_cells': number of cells
        - 'num_ensembles': number of ensembles
    """
    params = {}

    # Get beam direction
    if beam_direction:
        params["beam_direction"] = beam_direction.lower()
    else:
        params["beam_direction"] = ds.fixed_leader.system_configuration()[
            "Beam Direction"
        ].lower()

    # Get beam angle
    if beam_angle:
        params["beam_angle"] = float(beam_angle)
    else:
        params["beam_angle"] = float(
            ds.fixed_leader.system_configuration()["Beam Angle"]
        )

    # Get cell parameters
    params["cell_size"] = float(ds.fixed_leader.field()["depth_cell_length"])  # cm
    params["bin_1_distance"] = float(ds.fixed_leader.field()["bin_1_distance"])  # cm
    params["num_cells"] = ds.sizes.get("cell", 0)
    params["num_ensembles"] = ds.sizes.get("time", 0)

    return params


def _validate_dataset(ds: xr.Dataset, required_vars: Optional[list] = None) -> bool:
    """
    Validate dataset structure and required variables.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate
    required_vars : list, optional
        List of required variable names

    Returns
    -------
    bool
        True if valid, raises ValueError otherwise
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset")

    if required_vars is None:
        required_vars = ["velocity"]

    for var in required_vars:
        if var not in ds.data_vars:
            raise ValueError(f"Dataset missing required variable: {var}")

    if "cell" not in ds.dims or "time" not in ds.dims:
        raise ValueError("Dataset must have 'cell' and 'time' dimensions")

    return True


def _get_mask_or_create(ds: xr.Dataset) -> xr.DataArray:
    """Get existing mask or create default one from utility."""
    if "mask" in ds.data_vars:
        return ds["mask"]
    logger.debug("No mask found, creating default mask from velocity data")
    return create_default_mask(ds)


# ============================================================================
# ENSEMBLE TRIMMING FUNCTIONS (Mask-based)
# ============================================================================


def trim_ensembles(
    ds: xr.Dataset,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> xr.Dataset:
    """
    Trim ensembles (time steps) from beginning/end of deployment by masking.

    Original dataset structure is preserved. Only the mask array is updated
    to mark invalid ensembles. This allows flexible processing where
    deployment/recovery periods can be excluded without data loss.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 'time' dimension and 'mask' variable.
    start : int, optional
        Number of ensembles to mask from start of deployment.
    end : int, optional
        Number of ensembles to mask from end of deployment.

    Returns
    -------
    xr.Dataset
        New dataset with updated mask. Original is not modified.

    Examples
    --------
    >>> # Mask first 10 and last 5 ensembles (deployment/recovery)
    >>> ds_trimmed = trim_ensembles(ds, start=10, end=5)
    """
    _validate_dataset(ds)
    n_ensembles = ds.sizes["time"]

    # Get or create mask
    mask = _get_mask_or_create(ds)
    mask_values = mask.values.copy()

    cells_flagged = 0

    # Mark first ensembles as invalid
    if start is not None and start > 0:
        if start > n_ensembles:
            raise ValueError(f"start ({start}) exceeds n_ensembles ({n_ensembles})")
        # Mark all beams and cells for these time steps
        pre_count = (mask_values[:, :, 0:start] == 1).sum()
        mask_values[:, :, 0:start] = 1
        post_count = (mask_values[:, :, 0:start] == 1).sum()
        cells_flagged += post_count - pre_count
        logger.info(
            f"Marked first {start} ensembles as invalid "
            f"({mask_values.shape[0]} beams Ã— {mask_values.shape[1]} cells)"
        )

    # Mark last ensembles as invalid
    if end is not None and end > 0:
        if end > n_ensembles:
            raise ValueError(f"end ({end}) exceeds n_ensembles ({n_ensembles})")
        pre_count = (mask_values[:, :, -end:] == 1).sum()
        mask_values[:, :, -end:] = 1
        post_count = (mask_values[:, :, -end:] == 1).sum()
        cells_flagged += post_count - pre_count
        logger.info(
            f"Marked last {end} ensembles as invalid "
            f"({mask_values.shape[0]} beams Ã— {mask_values.shape[1]} cells)"
        )

    # Create output dataset
    ds_out = ds.copy(deep=True)
    mask_updated = xr.DataArray(
        data=mask_values,
        dims=mask.dims,
        coords=mask.coords,
        attrs=mask.attrs.copy(),
    )
    ds_out["mask"] = mask_updated

    # Log final state
    invalid_count = int((mask_values == 1).sum())
    pct_invalid = 100 * invalid_count / mask_values.size
    logger.debug(
        f"Trim ensembles complete: {invalid_count:,} invalid values "
        f"({pct_invalid:.1f}% of {mask_values.size:,} total)"
    )

    return ds_out


# ============================================================================
# BIN CUTTING FUNCTIONS (Mask-based)
# ============================================================================


def cut_bins_side_lobe(
    ds: xr.Dataset,
    orientation: Optional[str] = None,
    water_depth: Optional[float] = None,
    extra_cells: int = DEFAULT_EXTRA_CELLS,
) -> xr.Dataset:
    """
    Mask bins contaminated by side-lobe backscatter.

    Based on beam geometry, calculates which cells are affected by side-lobe
    contamination and marks them as invalid in the mask.

    For upward-looking ADCPs: masks cells above the surface reflection point
    For downward-looking ADCPs: masks cells below bottom reflection point

    Parameters
    ----------
    ds : xr.Dataset
        Input ADCP dataset with:
        - 'transducer_depth' coordinate (dm, varies by ensemble/time)
        - Fixed leader attributes: beam_angle, beam_direction, depth_cell_length
    orientation : str, optional
        Beam direction override ('up' or 'down'). If None, uses dataset attributes.
    water_depth : float, optional
        Water column depth in meters (for downward-looking ADCP).
        For upward-looking, this is ignored and set to 0.
        If None and downward-looking, raises ValueError.
    extra_cells : int, default 1
        Additional cells to mark as invalid beyond calculated side-lobe limit.

    Returns
    -------
    xr.Dataset
        New dataset with updated mask. Original is not modified.

    Notes
    -----
    Formula (from v0 profile_test.py):
        depth = transducer_depth / 10  (dm to m, vectorized for all ensembles)
        valid_depth = (water_depth - sgn * depth) * cos(beam_angle) - bin1_distance
        valid_cells = floor(valid_depth * 100 / cell_size) - extra_cells

    References
    ----------
    Implementation based on v0.4.0 profile_test.side_lobe_beam_angle()
    """
    _validate_dataset(ds)

    # Extract parameters from dataset
    params = _extract_dataset_params(ds, beam_direction=orientation)

    # Convert units consistently - work in METERS like legacy code
    beam_angle_rad = np.deg2rad(params["beam_angle"])
    cell_size_cm = params["cell_size"]  # Keep in cm for final calculation
    bin1_dist_m = params["bin_1_distance"] / 100  # cm to m

    # Get dimensions
    num_cells = params["num_cells"]
    num_ensembles = params["num_ensembles"]

    # Determine sign based on orientation
    sgn = -1 if params["beam_direction"] == "up" else 1

    # Determine water depth (in meters)
    if params["beam_direction"] == "up":
        water_depth_m = 0.0
        logger.info("Upward-looking ADCP: water_depth set to 0")
    else:
        if water_depth is None:
            raise ValueError(
                "water_depth required for downward-looking ADCP. "
                "Provide water column depth in meters."
            )
        water_depth_m = water_depth  # Already in meters
        logger.info(f"Downward-looking ADCP: water_depth = {water_depth} m")

    # Get or create mask
    mask = _get_mask_or_create(ds)
    mask_values = mask.values.copy()

    # CALCULATIONS OUTSIDE LOOP (Vectorized for all ensembles)
    # Get transducer depth for all ensembles (array, not scalar)
    # RDI stores transducer depth in decimeters (dm)
    transducer_depth_m = ds["transducer_depth"].values / 10  # dm to m (array)

    # Calculate valid_depth in METERS (following legacy formula)
    # valid_depth = (water_column_depth - sgn * depth) * cos(angle) - bin1dist
    valid_depth_m = (water_depth_m - sgn * transducer_depth_m) * np.cos(
        beam_angle_rad
    ) - bin1_dist_m

    # Calculate valid cells: convert valid_depth from m to cm, divide by cell_size in cm
    # valid_cells = valid_depth * 100 / cell_size_cm
    valid_cells = np.trunc(valid_depth_m * 100 / cell_size_cm) - extra_cells

    # Ensure non-negative
    valid_cells = np.maximum(valid_cells, 0)

    # LOOP ONLY FOR MASKING (apply per-ensemble results)
    total_contaminated = 0
    contaminated_per_ensemble = np.zeros(num_ensembles, dtype=np.int32)
    n_beams = mask_values.shape[0]

    for i in range(num_ensembles):
        c = int(valid_cells[i])  # Get valid cell count for ensemble i

        if num_cells > c:
            # Mark cells beyond valid_cells as contaminated (all beams)
            mask_values[:, c:, i] = 1
            newly_flagged = (num_cells - c) * n_beams
            contaminated_per_ensemble[i] = num_cells - c
            total_contaminated += newly_flagged

    # Create output dataset
    ds_out = ds.copy(deep=True)
    mask_updated = xr.DataArray(
        data=mask_values,
        dims=mask.dims,
        coords=mask.coords,
        attrs=mask.attrs.copy(),
    )
    ds_out["mask"] = mask_updated

    # Log statistics
    num_flagged_ensembles = int((contaminated_per_ensemble > 0).sum())
    if num_flagged_ensembles > 0:
        flagged_cells = contaminated_per_ensemble[contaminated_per_ensemble > 0]
        min_contaminated = int(flagged_cells.min())
        max_contaminated = int(flagged_cells.max())
        mean_contaminated = float(flagged_cells.mean())
        pct_total = 100 * total_contaminated / mask_values.size

        logger.info(
            f"Side-lobe contamination: "
            f"flagged {num_flagged_ensembles}/{num_ensembles} ensembles, "
            f"total {total_contaminated:,} cells ({pct_total:.1f}%) "
            f"(range: {min_contaminated}-{max_contaminated} cells/ensemble, "
            f"mean: {mean_contaminated:.1f} cells/ensemble) "
            f"(beam_direction={params['beam_direction']}, "
            f"water_depth={water_depth} m, extra_cells={extra_cells})"
        )
    else:
        logger.warning("Side-lobe calculation: no ensembles flagged as contaminated")

    return ds_out


def cut_bins_manual(
    ds: xr.Dataset,
    min_cell: Optional[int] = None,
    max_cell: Optional[int] = None,
    min_ensemble: Optional[int] = None,
    max_ensemble: Optional[int] = None,
) -> xr.Dataset:
    """
    Manually mask a rectangular region of cells and ensembles.

    Allows precise user control to mask specific regions (e.g., bad sensor
    regions, problematic time periods, or manually identified contamination).

    Parameters
    ----------
    ds : xr.Dataset
        ADCP dataset with 'mask' variable
    min_cell : int, optional
        Minimum cell index to start masking (inclusive). Default: 0
    max_cell : int, optional
        Maximum cell index to stop masking (exclusive). Default: num_cells
    min_ensemble : int, optional
        Minimum ensemble index to start masking (inclusive). Default: 0
    max_ensemble : int, optional
        Maximum ensemble index to stop masking (exclusive). Default: num_ensembles

    Returns
    -------
    xr.Dataset
        New dataset with updated mask. Original is not modified.

    Examples
    --------
    >>> # Mask cells 10-20 across all ensembles
    >>> ds_masked = cut_bins_manual(ds, min_cell=10, max_cell=20)

    >>> # Mask ensembles 100-200 across all cells
    >>> ds_masked = cut_bins_manual(ds, min_ensemble=100, max_ensemble=200)

    >>> # Mask rectangular region (cells 5-15, ensembles 50-150)
    >>> ds_masked = cut_bins_manual(ds, min_cell=5, max_cell=15,
    ...                              min_ensemble=50, max_ensemble=150)
    """
    _validate_dataset(ds)

    # Get dimensions
    num_cells = ds.sizes["cell"]
    num_ensembles = ds.sizes["time"]

    # Set defaults for None values
    if min_cell is None:
        min_cell = 0
    if max_cell is None:
        max_cell = num_cells
    if min_ensemble is None:
        min_ensemble = 0
    if max_ensemble is None:
        max_ensemble = num_ensembles

    # Validate input indices are integers
    if not all(
        isinstance(idx, (int, np.integer))
        for idx in [min_cell, max_cell, min_ensemble, max_ensemble]
    ):
        raise TypeError("All indices must be integers")

    # Validate that min <= max (before clipping)
    if min_cell > max_cell:
        raise ValueError(f"min_cell ({min_cell}) cannot be > max_cell ({max_cell})")
    if min_ensemble > max_ensemble:
        raise ValueError(
            f"min_ensemble ({min_ensemble}) cannot be > max_ensemble ({max_ensemble})"
        )

    # Clamp indices to valid range
    min_cell_clamped = max(0, min_cell)
    max_cell_clamped = min(num_cells, max_cell)
    min_ensemble_clamped = max(0, min_ensemble)
    max_ensemble_clamped = min(num_ensembles, max_ensemble)

    # Log if clamping occurred
    if (
        min_cell != min_cell_clamped
        or max_cell != max_cell_clamped
        or min_ensemble != min_ensemble_clamped
        or max_ensemble != max_ensemble_clamped
    ):
        logger.debug(
            f"Indices clamped: cell [{min_cell}, {max_cell}] â†’ "
            f"[{min_cell_clamped}, {max_cell_clamped}], "
            f"ensemble [{min_ensemble}, {max_ensemble}] â†’ "
            f"[{min_ensemble_clamped}, {max_ensemble_clamped}]"
        )

    # Get or create mask
    mask = _get_mask_or_create(ds)
    mask_values = mask.values.copy()
    n_beams = mask_values.shape[0]

    # Apply mask to selected rectangular region (all beams)
    mask_values[
        :, min_cell_clamped:max_cell_clamped, min_ensemble_clamped:max_ensemble_clamped
    ] = 1

    # Create output dataset
    ds_out = ds.copy(deep=True)
    mask_updated = xr.DataArray(
        data=mask_values,
        dims=mask.dims,
        coords=mask.coords,
        attrs=mask.attrs.copy(),
    )
    ds_out["mask"] = mask_updated

    # Calculate statistics
    num_cells_masked = max_cell_clamped - min_cell_clamped
    num_ensembles_masked = max_ensemble_clamped - min_ensemble_clamped
    total_masked = num_cells_masked * num_ensembles_masked * n_beams
    pct_masked = 100 * total_masked / mask_values.size

    logger.info(
        f"Manual bin cutting: masked rectangular region "
        f"cells[{min_cell_clamped}:{max_cell_clamped}] Ã— "
        f"ensembles[{min_ensemble_clamped}:{max_ensemble_clamped}] = "
        f"{total_masked:,} cells ({pct_masked:.1f}% of {mask_values.size:,} total)"
    )

    return ds_out


# ============================================================================
# REGRIDDING FUNCTIONS - Transform to regular depth grid
# ============================================================================


def regrid(
    ds: xr.Dataset,
    data_vars: Optional[List[str]] = None,
    fill_value: float = np.nan,
    end_cell_option: str = DEFAULT_END_CELL_OPTION,
    trimends: Optional[Tuple[int, int]] = None,
    method: str = DEFAULT_REGRID_METHOD,
    orientation: Optional[str] = None,
    boundary_limit: float = 0.0,
) -> xr.Dataset:
    """
    Regrid ADCP data from irregular instrument cells to regular depth grid.

    Transforms data from ADCP's native irregular cell spacing (which varies with
    transducer depth) to a uniform regular depth grid for easier analysis and
    visualization.

    WARNING: This operation changes the dataset structure from (beam, cell, time)
    to (beam, depth, time). After regridding, cell-based masks are no longer valid.

    Mask Handling Strategy:
    1. BEFORE regridding: Apply mask to data (masked cells â†’ np.nan)
    2. DURING regridding: Interpolate data (masked regions preserved as np.nan)
    3. AFTER regridding: Recreate binary mask from np.nan locations in velocity

    All invalid/masked data is represented as np.nan after regridding for
    consistency across all variables (velocity, correlation, echo, etc.).

    Parameters
    ----------
    ds : xr.Dataset
        Input ADCP dataset containing:
        - 'transducer_depth' coordinate (dm, varies per ensemble)
        - Fixed leader attributes: beam_angle, beam_direction, depth_cell_length
        - Variables to regrid (velocity, correlation, echo_intensity, etc.)
        - Optional 'mask' variable (will be applied before regridding)
    data_vars : list of str, optional
        Variable names to regrid. If None, regrids all 2D variables
        with shape (cell, time) or 3D with (beam, cell, time).
        Note: 'mask' is never interpolated - it's recreated from velocity.
    fill_value : float, default np.nan
        Value used for extrapolation beyond original data range.
    end_cell_option : str, default "cell"
        Defines the depth extent of regridded grid:
        - "cell": Grid extends to calculated last cell
        - "surface": Grid extends to surface (depth = 0)
        - "manual": User-defined boundary using boundary_limit parameter
    trimends : tuple of (int, int), optional
        (start_ensemble, end_ensemble) for calculating depth range.
        Ignores deployment/recovery periods with bad depth data.
    method : str, default "nearest"
        Interpolation method: "nearest", "linear", "cubic", "quadratic"
    orientation : str, optional
        Beam direction override ('up' or 'down'). If None, uses dataset attributes.
    boundary_limit : float, default 0.0
        Used only when end_cell_option="manual". Specifies depth boundary in meters.

    Returns
    -------
    xr.Dataset
        Regridded dataset with:
        - 'depth': New 1D coordinate (regular grid in meters)
        - Variables regridded to new depth grid
        - 'mask': Recreated binary mask from velocity np.nan locations
        - Dimensions: (beam, depth, time) or (depth, time)

    Notes
    -----
    This function modifies the dataset structure significantly. The 'cell'
    dimension is replaced with a 'depth' dimension.

    The mask is NOT interpolated (which would produce invalid 0.3, 0.7 values).
    Instead, it is recreated from the regridded velocity data:
    - Where velocity is np.nan or VELOCITY_MISSING_VALUE â†’ mask = 1
    - Where velocity is valid â†’ mask = 0

    Examples
    --------
    >>> # Basic regridding to regular depth grid
    >>> ds_regridded = regrid(ds)

    >>> # Linear interpolation, extend to surface
    >>> ds_regridded = regrid(ds, method='linear', end_cell_option='surface')

    >>> # Exclude deployment/recovery from depth calculation
    >>> ds_regridded = regrid(ds, trimends=(100, -100))
    """
    from .utility import VELOCITY_MISSING_VALUE

    _validate_dataset(ds)

    # Extract parameters from dataset
    params = _extract_dataset_params(ds, beam_direction=orientation)

    # Convert units to meters consistently
    # RDI ADCP stores:
    #   - bin_1_distance in cm
    #   - cell_size (depth_cell_length) in cm
    #   - transducer_depth in DECIMETERS (dm), NOT millimeters!
    # This matches the legacy profile_test.py implementation
    bin1_dist = params["bin_1_distance"] / 100  # cm to m
    cell_size = params["cell_size"] / 100  # cm to m
    transducer_depth = ds["transducer_depth"].values / 10  # dm to m (array) - NOT mm!

    num_cells = params["num_cells"]
    num_ensembles = params["num_ensembles"]

    # Debug logging for unit conversion verification
    logger.debug(
        f"Unit conversions: bin_1_distance={params['bin_1_distance']}cm â†’ {bin1_dist}m, "
        f"cell_size={params['cell_size']}cm â†’ {cell_size}m, "
        f"transducer_depth[0]={ds['transducer_depth'].values[0]}dm â†’ {transducer_depth[0]}m"
    )

    # Determine sign based on orientation
    # sgn = -1 for upward-looking (cells go toward surface, shallower = less negative)
    # sgn = 1 for downward-looking (cells go toward bottom, deeper = more positive)
    sgn = -1 if params["beam_direction"] == "up" else 1
    logger.info(
        f"Regridding: {params['beam_direction']}-looking ADCP, "
        f"{num_cells} cells Ã— {num_ensembles} ensembles"
    )

    # Validate end_cell_option
    valid_options = ["cell", "surface", "manual"]
    if end_cell_option.lower() not in valid_options:
        raise ValueError(
            f"end_cell_option '{end_cell_option}' not recognized. "
            f"Must be one of: {valid_options}"
        )

    # Calculate depth range for grid
    if trimends is not None:
        start_ens, end_ens = trimends
        depth_subset = transducer_depth[start_ens:end_ens]
        logger.info(
            f"Using trimends ensembles [{start_ens}, {end_ens}) for depth range"
        )
    else:
        depth_subset = transducer_depth

    # Calculate first cell depths across all ensembles
    # For upward: first_cell = transducer_depth - bin1_dist (shallower than transducer)
    # For downward: first_cell = transducer_depth + bin1_dist (deeper than transducer)
    first_cell_depths = depth_subset + sgn * bin1_dist

    # max_depth = deepest first cell (where grid starts)
    # min_depth = shallowest first cell (where grid may end, depending on option)
    max_depth = abs(np.min(sgn * first_cell_depths))
    min_depth = abs(np.max(sgn * first_cell_depths))

    logger.debug(
        f"First cell depth range: {min_depth:.2f}m to {max_depth:.2f}m "
        f"(transducer: {depth_subset.min():.2f}m to {depth_subset.max():.2f}m)"
    )

    # ========================================================================
    # CALCULATE GRID BOUNDARIES
    # ========================================================================
    # first_grid_depth: Start of grid, rounded down to cell_size boundary
    first_grid_depth = max_depth - (max_depth % cell_size)

    # Handle edge case where modulo gives 0 (max_depth is exact multiple)
    if first_grid_depth == 0 and max_depth > 0:
        first_grid_depth = max_depth
        logger.debug(f"first_grid_depth was 0, set to max_depth={max_depth}")

    # Calculate last_grid_depth based on end_cell_option
    if end_cell_option.lower() == "surface":
        # Grid extends to near-surface (one cell_size from 0)
        last_grid_depth = sgn * cell_size
    elif end_cell_option.lower() == "cell":
        # Grid extends to cover all cells based on min_depth + num_cells
        min_depth_regrid = min_depth - sgn * (min_depth % cell_size)
        last_grid_depth_calc = min_depth_regrid + sgn * (num_cells + 1) * cell_size

        # Clamp based on orientation to prevent grid crossing surface
        if sgn == -1:  # Upward-looking
            # For upward: last_grid_depth should be small positive or slightly negative
            # (shallower depths). If it goes very negative (past surface), clamp.
            if last_grid_depth_calc < 0:
                last_grid_depth = sgn * cell_size  # = -cell_size
                logger.info(
                    f"Grid would extend past surface (calc={last_grid_depth_calc:.2f}m), "
                    f"clamping to {last_grid_depth:.2f}m"
                )
            else:
                last_grid_depth = last_grid_depth_calc
        else:  # Downward-looking (sgn == 1)
            # For downward: grid extends deeper. Usually no clamping needed.
            last_grid_depth = last_grid_depth_calc
    elif end_cell_option.lower() == "manual":
        last_grid_depth = boundary_limit

    logger.info(
        f"Grid extent: {first_grid_depth:.2f}m to {abs(last_grid_depth):.2f}m "
        f"(cell_size={cell_size:.2f}m)"
    )

    # ========================================================================
    # CREATE REGULAR DEPTH GRID
    # ========================================================================
    # For upward ADCP: np.arange(-first, -last, cell_size) goes from deep to shallow
    # For downward ADCP: np.arange(first, last, cell_size) goes from shallow to deep
    grid_start = sgn * first_grid_depth
    grid_stop = sgn * last_grid_depth

    depth_grid = np.arange(grid_start, grid_stop, cell_size)
    num_depth_levels = len(depth_grid)
    depth_grid_abs = np.abs(depth_grid)  # Absolute (positive) depth values

    # ========================================================================
    # VALIDATION: Check for reasonable grid size
    # ========================================================================
    if num_depth_levels == 0:
        raise ValueError(
            f"Empty depth grid! Parameters: first_grid_depth={first_grid_depth:.2f}m, "
            f"last_grid_depth={last_grid_depth:.2f}m, cell_size={cell_size:.2f}m, "
            f"sgn={sgn}. Grid would be np.arange({grid_start:.2f}, {grid_stop:.2f}, {cell_size:.2f}). "
            f"Check unit conversions: bin_1_distance={params['bin_1_distance']}, "
            f"depth_cell_length={params['cell_size']}, "
            f"transducer_depth[0]={ds['transducer_depth'].values[0]}"
        )

    if num_depth_levels < 3:
        logger.warning(
            f"Very small depth grid: only {num_depth_levels} levels. "
            f"Expected approximately {num_cells} levels. "
            f"Grid parameters: start={grid_start:.2f}m, stop={grid_stop:.2f}m, "
            f"step={cell_size:.2f}m. This may indicate unit conversion issues."
        )
        logger.warning(
            f"Raw input values for debugging: "
            f"bin_1_distance={params['bin_1_distance']}(cm), "
            f"depth_cell_length={params['cell_size']}(cm), "
            f"transducer_depth[0]={ds['transducer_depth'].values[0]}(dm), "
            f"num_cells={num_cells}"
        )

    logger.info(f"Created regular grid with {num_depth_levels} depth levels")

    # ========================================================================
    # STEP 1: Get existing mask and prepare for application
    # ========================================================================
    has_mask = "mask" in ds.data_vars
    if has_mask:
        mask_values = ds["mask"].values  # (beam, cell, time)
        mask_count = int((mask_values == 1).sum())
        mask_pct = 100 * mask_count / mask_values.size if mask_values.size > 0 else 0
        logger.info(
            f"Existing mask found - {mask_count:,} cells masked ({mask_pct:.1f}%) "
            f"- will apply to data before regridding"
        )
    else:
        mask_values = None
        logger.info("No mask found - regridding without mask application")

    # ========================================================================
    # STEP 2: Determine which variables to regrid (exclude mask)
    # ========================================================================
    if data_vars is None:
        data_vars = []
        for var_name in ds.data_vars:
            if var_name == "mask":
                continue  # Never interpolate mask
            var = ds[var_name]
            # 2D variables (cell, time)
            if var.ndim == 2 and var.shape == (num_cells, num_ensembles):
                data_vars.append(var_name)
            # 3D variables (beam, cell, time)
            elif var.ndim == 3 and var.shape[1:] == (num_cells, num_ensembles):
                data_vars.append(var_name)
        logger.info(f"Auto-detected variables to regrid: {data_vars}")
    else:
        # Remove 'mask' if user accidentally included it
        data_vars = [v for v in data_vars if v != "mask"]

    # ========================================================================
    # STEP 3: Regrid each variable (applying mask as np.nan)
    # ========================================================================
    regridded_vars = {}

    for var_name in data_vars:
        if var_name not in ds:
            logger.warning(f"Variable '{var_name}' not found, skipping")
            continue

        logger.debug(f"Regridding variable: {var_name}")
        var = ds[var_name]
        var_data = var.values.astype(float)

        # Replace missing values with np.nan
        var_data = np.where(var_data == VELOCITY_MISSING_VALUE, np.nan, var_data)

        # Handle 2D and 3D variables
        if var.ndim == 2:
            # Apply mask if exists (use combined mask beam 3 for 2D data)
            if has_mask and mask_values is not None:
                # Use beam 3 (combined) for 2D variables
                combined_mask = (
                    mask_values[3, :, :]
                    if mask_values.shape[0] > 3
                    else mask_values[0, :, :]
                )
                var_data = np.where(combined_mask == 1, np.nan, var_data)

            regridded_data = _regrid_2d_variable(
                var_data,
                transducer_depth,
                bin1_dist,
                cell_size,
                sgn,
                num_cells,
                num_ensembles,
                depth_grid,
                method,
                fill_value,
            )
            regridded_vars[var_name] = xr.DataArray(
                regridded_data,
                coords={"depth": depth_grid_abs, "time": ds["time"].values},
                dims=["depth", "time"],
                attrs=var.attrs.copy(),
            )
        elif var.ndim == 3:
            n_beams = var.shape[0]
            regridded_data = np.zeros((n_beams, num_depth_levels, num_ensembles))

            for b in range(n_beams):
                beam_data = var_data[b].copy()

                # Count NaNs before mask application
                nan_before = np.isnan(beam_data).sum()

                # Apply beam-specific mask if exists
                if has_mask and mask_values is not None:
                    beam_mask = (
                        mask_values[b, :, :]
                        if b < mask_values.shape[0]
                        else mask_values[0, :, :]
                    )
                    mask_count = int((beam_mask == 1).sum())
                    beam_data = np.where(beam_mask == 1, np.nan, beam_data)
                    nan_after = np.isnan(beam_data).sum()
                    logger.debug(
                        f"  Beam {b}: mask has {mask_count:,} flagged cells, "
                        f"NaNs before={nan_before:,}, after={nan_after:,}"
                    )

                regridded_data[b] = _regrid_2d_variable(
                    beam_data,
                    transducer_depth,
                    bin1_dist,
                    cell_size,
                    sgn,
                    num_cells,
                    num_ensembles,
                    depth_grid,
                    method,
                    fill_value,
                )

            regridded_vars[var_name] = xr.DataArray(
                regridded_data,
                coords={
                    "beam": ds.coords["beam"].values[:n_beams],
                    "depth": depth_grid_abs,
                    "time": ds["time"].values,
                },
                dims=["beam", "depth", "time"],
                attrs=var.attrs.copy(),
            )

    # ========================================================================
    # STEP 4: Recreate mask from velocity np.nan locations
    # ========================================================================
    if "velocity" in regridded_vars:
        velocity_regridded = regridded_vars["velocity"].values
        n_beams = velocity_regridded.shape[0]

        # Create new mask: 1 where velocity is nan, 0 otherwise
        new_mask = np.isnan(velocity_regridded).astype(np.int8)

        # If we have 4 beams, update beam 3 as combined mask (OR of 0, 1, 2)
        if n_beams >= 4:
            new_mask[3, :, :] = (
                (new_mask[0, :, :] == 1)
                | (new_mask[1, :, :] == 1)
                | (new_mask[2, :, :] == 1)
            ).astype(np.int8)

        regridded_vars["mask"] = xr.DataArray(
            new_mask,
            coords={
                "beam": ds.coords["beam"].values[:n_beams],
                "depth": depth_grid_abs,
                "time": ds["time"].values,
            },
            dims=["beam", "depth", "time"],
            attrs={
                "long_name": "Velocity quality mask (regridded)",
                "description": "Binary mask recreated from velocity np.nan after regridding",
                "flag_values": "0, 1",
                "flag_meanings": "valid invalid",
            },
        )

        # Count masked cells
        total_masked = int((new_mask == 1).sum())
        total_cells = new_mask.size
        pct_masked = 100 * total_masked / total_cells if total_cells > 0 else 0
        logger.info(
            f"Recreated mask from regridded velocity: "
            f"{total_masked:,} masked ({pct_masked:.1f}%)"
        )

    # ========================================================================
    # STEP 4.5: Preserve non-cell-dependent variables (e.g. Fixed/Variable
    # Leader fields such as coordinate_transformation_code, heading, etc.)
    # ========================================================================
    # These variables don't vary over 'cell', so they aren't touched by the
    # depth regridding and can be carried over unchanged. Without this, any
    # variable not indexed by 'cell' (like coordinate_transformation_code)
    # would silently disappear from the regridded dataset.
    for var_name in ds.data_vars:
        if var_name in regridded_vars or var_name == "mask":
            continue
        if "cell" not in ds[var_name].dims:
            regridded_vars[var_name] = ds[var_name].copy()

    # ========================================================================
    # STEP 5: Create output dataset
    # ========================================================================
    ds_regridded = xr.Dataset(
        regridded_vars,
        coords={"depth": depth_grid_abs, "time": ds["time"].values},
        attrs=ds.attrs.copy(),
    )

    # Add beam coordinate if we have 3D variables
    if any(regridded_vars[v].ndim == 3 for v in regridded_vars):
        ds_regridded.coords["beam"] = ds.coords["beam"]

    # Add regridding metadata
    ds_regridded.attrs["regridding_method"] = method
    ds_regridded.attrs["regridding_end_cell_option"] = end_cell_option
    ds_regridded.attrs["regridding_fill_value"] = str(fill_value)
    ds_regridded.attrs["regridding_mask_applied"] = int(has_mask)
    ds_regridded.attrs["regridded"] = 1
    if trimends is not None:
        ds_regridded.attrs["regridding_trimends"] = f"{trimends[0]}-{trimends[1]}"

    logger.info(
        f"Regridding complete: {num_depth_levels} depth levels Ã— {num_ensembles} ensembles, "
        f"{len(data_vars)} variables regridded, mask recreated from velocity"
    )

    return ds_regridded


def _regrid_2d_variable(
    data: np.ndarray,
    transducer_depth: np.ndarray,
    bin1_dist: float,
    cell_size: float,
    sgn: int,
    num_cells: int,
    num_ensembles: int,
    depth_grid: np.ndarray,
    method: str,
    fill_value: float,
) -> np.ndarray:
    """
    Regrid a single 2D variable (cell Ã— time) to regular depth grid.

    Handles np.nan values properly during interpolation. Masked regions
    (NaN in input) are preserved in the output - they are NOT interpolated
    over. The mask is mapped from original cell coordinates to the new
    depth grid.

    Parameters
    ----------
    data : np.ndarray
        2D array (cell, time) to regrid. May contain np.nan for masked cells.
    transducer_depth : np.ndarray
        Transducer depth per ensemble (m)
    bin1_dist : float
        Distance to first bin (m)
    cell_size : float
        Cell size (m)
    sgn : int
        Sign for orientation (-1 for up, 1 for down)
    num_cells : int
        Number of cells
    num_ensembles : int
        Number of ensembles
    depth_grid : np.ndarray
        Target depth grid
    method : str
        Interpolation method
    fill_value : float
        Fill value for extrapolation

    Returns
    -------
    np.ndarray
        Regridded data (depth Ã— time). Contains np.nan where input was masked
        or where extrapolation occurred.
    """
    num_depth_levels = len(depth_grid)
    regridded = np.full((num_depth_levels, num_ensembles), fill_value)

    for i in range(num_ensembles):
        transdepth_i = transducer_depth[i]

        # Calculate original cell depths for this ensemble
        first_cell_i = transdepth_i + sgn * bin1_dist
        last_cell_i = first_cell_i + sgn * num_cells * cell_size

        # Create depth array for original cells
        original_depths = np.linspace(sgn * first_cell_i, sgn * last_cell_i, num_cells)

        # Get data for this ensemble
        ensemble_data = data[:, i]

        # Find valid (non-NaN) points for interpolation
        valid_mask = ~np.isnan(ensemble_data)

        if not np.any(valid_mask):
            # All NaN - fill with fill_value (already initialized)
            continue

        if np.sum(valid_mask) < 2:
            # Not enough points for interpolation
            regridded[:, i] = fill_value
            continue

        try:
            # STEP 1: Interpolate ALL data (including masked regions)
            # We do this by first interpolating valid points, then
            # mapping the mask from original cells to new depth grid

            interpolator = sp_interp.interp1d(
                original_depths[valid_mask],
                ensemble_data[valid_mask],
                kind=method,
                fill_value=fill_value,
                bounds_error=False,
            )
            regridded[:, i] = interpolator(depth_grid)

            # STEP 2: Re-apply mask from original coordinates to regridded coordinates
            # For each original masked cell, find which regridded depths fall within
            # that cell's depth range and mask them
            masked_cells = np.where(~valid_mask)[0]

            for cell_idx in masked_cells:
                # Calculate depth range of this masked cell.
                # cell_depth_start < cell_depth_end always holds since cell_size > 0,
                # regardless of whether depth values are positive or negative.
                cell_depth_start = original_depths[cell_idx] - cell_size / 2
                cell_depth_end = original_depths[cell_idx] + cell_size / 2

                in_masked_range = (depth_grid >= cell_depth_start) & (
                    depth_grid <= cell_depth_end
                )

                # Apply fill_value (typically np.nan) to these depths
                regridded[in_masked_range, i] = fill_value

            # STEP 3: For 'nearest' method, also check if points fall outside
            # the valid data range
            if method == "nearest":
                valid_depths = original_depths[valid_mask]
                outside_range = (depth_grid < valid_depths.min()) | (
                    depth_grid > valid_depths.max()
                )
                regridded[outside_range, i] = fill_value

        except ValueError as e:
            logger.warning(f"Interpolation failed for ensemble {i}: {e}")
            regridded[:, i] = fill_value

    return regridded


# ============================================================================
# PROFILE OPERATION RUNNER CLASS
# ============================================================================


class ProfileOperationRunner:
    """
    Orchestrator for profile-level operations on ADCP datasets.

    Provides a fluent API (method chaining) matching the SignalQualityRunner
    and VelocityCheckRunner style. Coordinates ensemble trimming, bin cutting,
    and regridding operations with comprehensive statistics tracking.

    IMPORTANT SEQUENCING:
    1. Quality control checks MUST happen BEFORE regridding
    2. Regridding INVALIDATES cell-based masks
    3. Use this class AFTER all QC steps are complete

    Attributes
    ----------
    dataset : xr.Dataset
        Current working dataset.
    original : xr.Dataset
        Original unmodified dataset.
    statistics : list[QCCheckStats]
        QC check statistics (mask-based operations).
    modifications : list[DataModificationStats]
        Data modification statistics (regridding).
    history : list[dict[str, Any]]
        Processing history.

    Examples
    --------
    >>> runner = ProfileOperationRunner(ds)
    >>> ds_processed = (runner
    ...     .trim_ensembles(start=10, end=5)
    ...     .cut_bins_side_lobe(extra_cells=2)
    ...     .cut_bins_manual(min_cell=0, max_cell=2)
    ...     .regrid(method='linear')
    ...     .finalize())
    >>> runner.print_statistics()
    """

    def __init__(self, ds: xr.Dataset) -> None:
        """
        Initialize ProfileOperationRunner with dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Input ADCP dataset to process.
        """
        _validate_dataset(ds)
        self.original: xr.Dataset = ds.copy(deep=True)
        self.dataset: xr.Dataset = ds.copy(deep=True)

        # Ensure mask exists
        if "mask" not in self.dataset.data_vars:
            self.dataset["mask"] = create_default_mask(self.dataset)

        self._update_baseline()
        self.statistics: list[QCCheckStats] = []
        self.modifications: list[DataModificationStats] = []
        self.history: list[dict[str, Any]] = []
        self._regridded: bool = False

        logger.info(
            f"ProfileOperationRunner initialized: "
            f"{self.dataset.sizes.get('cell', 0)} cells Ã— "
            f"{self.dataset.sizes.get('time', 0)} ensembles, "
            f"{self.baseline_masked:,} pre-masked ({self.baseline_masked_pct:.2f}%)"
        )

    def _update_baseline(self) -> None:
        """Calculate baseline statistics from current dataset."""
        if "mask" in self.dataset.data_vars:
            mask = self.dataset["mask"]
            self.total_cells = int(mask.size)
            self.baseline_masked = int((mask == 1).sum())
            self.baseline_masked_pct = (
                100 * self.baseline_masked / self.total_cells
                if self.total_cells > 0
                else 0.0
            )
        else:
            self.total_cells = 0
            self.baseline_masked = 0
            self.baseline_masked_pct = 0.0

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

    def _record_mask_operation(
        self,
        check_name: str,
        check_func: Callable[..., xr.Dataset],
        threshold: Any,
        **kwargs: Any,
    ) -> ProfileOperationRunner:
        """
        Apply a mask-based operation and record statistics.

        Parameters
        ----------
        check_name : str
            Name of the operation for display.
        check_func : Callable
            Function to apply.
        threshold : Any
            Threshold or parameter value(s).
        **kwargs : Any
            Arguments to pass to check_func.

        Returns
        -------
        ProfileOperationRunner
            Self for method chaining.
        """
        if self._regridded:
            raise RuntimeError(
                f"Cannot apply {check_name} after regridding. "
                "Mask-based operations must be done before regrid()."
            )

        # Stats before operation
        mask_pre = self.dataset["mask"].values
        pre_masked = int((mask_pre == 1).sum())
        logger.debug(
            f"Before {check_name}: {pre_masked:,} cells masked "
            f"(dataset id: {id(self.dataset)})"
        )

        # Apply operation
        self.dataset = check_func(self.dataset, **kwargs)

        # Stats after operation
        mask_post = self.dataset["mask"].values
        post_masked = int((mask_post == 1).sum())
        newly_masked = post_masked - pre_masked
        logger.debug(
            f"After {check_name}: {post_masked:,} cells masked "
            f"(+{newly_masked:,} newly masked, dataset id: {id(self.dataset)})"
        )

        # Create statistics object
        stat = QCCheckStats(
            check_name=check_name,
            threshold=threshold,
            cells_pre_masked=pre_masked,
            cells_newly_masked=newly_masked,
            cells_total_masked=post_masked,
            total_cells=self.total_cells,
            check_time=datetime.now(timezone.utc),
            metadata=kwargs.copy(),
        )
        self.statistics.append(stat)

        # Record history
        self._add_history(
            operation=check_name.lower().replace(" ", "_"),
            parameters=kwargs,
            stats={
                "pre_masked": pre_masked,
                "newly_masked": newly_masked,
                "total_masked": post_masked,
            },
        )

        return self

    # ========================================================================
    # MASK-BASED OPERATIONS
    # ========================================================================

    def trim_ensembles(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> ProfileOperationRunner:
        """
        Trim ensembles from start/end of deployment by masking.

        Parameters
        ----------
        start : int, optional
            Number of ensembles to mask from start.
        end : int, optional
            Number of ensembles to mask from end.

        Returns
        -------
        ProfileOperationRunner
            Self for method chaining.
        """
        threshold_str = f"start={start}, end={end}"
        return self._record_mask_operation(
            "Trim Ensembles",
            trim_ensembles,
            threshold=threshold_str,
            start=start,
            end=end,
        )

    def cut_bins_side_lobe(
        self,
        orientation: Optional[str] = None,
        water_depth: Optional[float] = None,
        extra_cells: int = DEFAULT_EXTRA_CELLS,
    ) -> ProfileOperationRunner:
        """
        Mask side-lobe contaminated bins.

        Parameters
        ----------
        orientation : str, optional
            Beam direction ('up' or 'down'). If None, uses dataset attributes.
        water_depth : float, optional
            Water depth in meters (required for downward-looking).
        extra_cells : int, default 1
            Additional cells to mask beyond calculated limit.

        Returns
        -------
        ProfileOperationRunner
            Self for method chaining.
        """
        threshold_str = f"extra_cells={extra_cells}"
        return self._record_mask_operation(
            "Cut Bins Side Lobe",
            cut_bins_side_lobe,
            threshold=threshold_str,
            orientation=orientation,
            water_depth=water_depth,
            extra_cells=extra_cells,
        )

    def cut_bins_manual(
        self,
        min_cell: Optional[int] = None,
        max_cell: Optional[int] = None,
        min_ensemble: Optional[int] = None,
        max_ensemble: Optional[int] = None,
    ) -> ProfileOperationRunner:
        """
        Manually mask a rectangular region.

        Parameters
        ----------
        min_cell : int, optional
            Minimum cell index (inclusive). Default: 0
        max_cell : int, optional
            Maximum cell index (exclusive). Default: num_cells
        min_ensemble : int, optional
            Minimum ensemble index (inclusive). Default: 0
        max_ensemble : int, optional
            Maximum ensemble index (exclusive). Default: num_ensembles

        Returns
        -------
        ProfileOperationRunner
            Self for method chaining.
        """
        threshold_str = (
            f"cells=[{min_cell}:{max_cell}], "
            f"ensembles=[{min_ensemble}:{max_ensemble}]"
        )
        return self._record_mask_operation(
            "Cut Bins Manual",
            cut_bins_manual,
            threshold=threshold_str,
            min_cell=min_cell,
            max_cell=max_cell,
            min_ensemble=min_ensemble,
            max_ensemble=max_ensemble,
        )

    # ========================================================================
    # REGRIDDING (Dataset Modification)
    # ========================================================================

    def regrid(
        self,
        data_vars: Optional[List[str]] = None,
        fill_value: float = np.nan,
        end_cell_option: str = DEFAULT_END_CELL_OPTION,
        trimends: Optional[Tuple[int, int]] = None,
        method: str = DEFAULT_REGRID_METHOD,
        orientation: Optional[str] = None,
        boundary_limit: float = 0.0,
    ) -> ProfileOperationRunner:
        """
        Regrid data to regular depth grid.

        âš ï¸ WARNING: This operation changes dataset structure.
        After regridding, mask-based operations cannot be applied.

        Parameters
        ----------
        data_vars : list of str, optional
            Variables to regrid. If None, auto-detects.
        fill_value : float, default np.nan
            Fill value for extrapolation.
        end_cell_option : str, default "cell"
            Grid extent option: "cell", "surface", or "manual"
        trimends : tuple, optional
            (start, end) ensembles for depth range calculation.
        method : str, default "nearest"
            Interpolation method.
        orientation : str, optional
            Beam direction override.
        boundary_limit : float, default 0.0
            Manual boundary depth (if end_cell_option="manual").

        Returns
        -------
        ProfileOperationRunner
            Self for method chaining.
        """
        # Capture original stats
        orig_shape = (
            self.dataset.sizes.get("cell", 0),
            self.dataset.sizes.get("time", 0),
        )

        # Apply regridding
        self.dataset = regrid(
            self.dataset,
            data_vars=data_vars,
            fill_value=fill_value,
            end_cell_option=end_cell_option,
            trimends=trimends,
            method=method,
            orientation=orientation,
            boundary_limit=boundary_limit,
        )

        # Capture new stats
        new_shape = (
            self.dataset.sizes.get("depth", 0),
            self.dataset.sizes.get("time", 0),
        )

        # Record modification statistics
        mod_stat = DataModificationStats(
            operation="regrid",
            variable_name="dataset_structure",
            original_stats={
                "cells": orig_shape[0],
                "ensembles": orig_shape[1],
            },
            modified_stats={
                "depth_levels": new_shape[0],
                "ensembles": new_shape[1],
            },
            metadata={
                "method": method,
                "end_cell_option": end_cell_option,
                "fill_value": str(fill_value),
            },
        )
        self.modifications.append(mod_stat)

        # Record history
        self._add_history(
            operation="regrid",
            parameters={
                "method": method,
                "end_cell_option": end_cell_option,
                "fill_value": str(fill_value),
            },
            stats={
                "original_cells": orig_shape[0],
                "new_depth_levels": new_shape[0],
            },
        )

        self._regridded = True
        logger.info(
            f"Regridding applied: {orig_shape[0]} cells â†’ {new_shape[0]} depth levels"
        )

        return self

    # ========================================================================
    # PIPELINE UTILITIES
    # ========================================================================

    def apply_pipeline(
        self,
        operations: Optional[dict[str, dict[str, Any]]] = None,
        order: Optional[list[str]] = None,
    ) -> ProfileOperationRunner:
        """
        Apply multiple operations using a pipeline configuration.

        Parameters
        ----------
        operations : dict, optional
            Dict of {operation_name: {param: value}}. If None, uses defaults.
        order : list, optional
            List of operation names in execution order.

        Returns
        -------
        ProfileOperationRunner
            Self for method chaining.

        Examples
        --------
        >>> runner.apply_pipeline(operations={
        ...     "trim_ensembles": {"start": 10, "end": 5},
        ...     "cut_bins_side_lobe": {"extra_cells": 2},
        ...     "regrid": {"method": "linear"},
        ... })
        """
        if operations is None:
            operations = {}

        if order is None:
            order = [
                "trim_ensembles",
                "cut_bins_side_lobe",
                "cut_bins_manual",
                "regrid",
            ]

        for name in order:
            if name not in operations:
                continue

            params = operations[name]
            if name == "trim_ensembles":
                self.trim_ensembles(**params)
            elif name == "cut_bins_side_lobe":
                self.cut_bins_side_lobe(**params)
            elif name == "cut_bins_manual":
                self.cut_bins_manual(**params)
            elif name == "regrid":
                self.regrid(**params)

        return self

    def reset(self) -> ProfileOperationRunner:
        """
        Reset dataset to original state and clear history.

        Returns
        -------
        ProfileOperationRunner
            Self for method chaining.
        """
        self.dataset = self.original.copy(deep=True)
        if "mask" not in self.dataset.data_vars:
            self.dataset["mask"] = create_default_mask(self.dataset)
        self.statistics = []
        self.modifications = []
        self.history = []
        self._regridded = False
        self._update_baseline()
        logger.info("ProfileOperationRunner reset to original state")
        return self

    def get_dataset(self) -> xr.Dataset:
        """
        Get the current working dataset.

        Returns
        -------
        xr.Dataset
            Current dataset (may be modified from original).
        """
        return self.dataset

    def finalize(self) -> xr.Dataset:
        """
        Finalize processing and return dataset with history attributes.

        Returns
        -------
        xr.Dataset
            Processed dataset with processing history in attributes.
        """
        ds_out = self.dataset.copy(deep=True)
        ds_out.attrs["profile_operation_processing_history"] = str(self.history)
        ds_out.attrs["profile_operation_processed_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        logger.info("ProfileOperationRunner finalized")
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
            module_name="profile_operation",
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
        print("PROFILE OPERATION PROCESSING STATISTICS")
        print("=" * width)
        print(
            f"Baseline masked: {self.baseline_masked:,} ({self.baseline_masked_pct:.2f}%)"
        )
        print(f"Total cells: {self.total_cells:,}")
        print("-" * width)

        # Print modifications if any
        if self.modifications:
            print("STRUCTURAL MODIFICATIONS:")
            for mod in self.modifications:
                orig = mod.original_stats
                modified = mod.modified_stats
                print(f"  {mod.operation}:")
                if "cells" in orig:
                    print(
                        f"    Original: {orig.get('cells', 'N/A')} cells Ã— "
                        f"{orig.get('ensembles', 'N/A')} ensembles"
                    )
                if "depth_levels" in modified:
                    print(
                        f"    Regridded: {modified.get('depth_levels', 'N/A')} depth levels Ã— "
                        f"{modified.get('ensembles', 'N/A')} ensembles"
                    )
                for key, val in mod.metadata.items():
                    print(f"    {key}: {val}")
            print("-" * width)

        # Print mask operations
        if not self.statistics:
            print("No mask operations applied.")
        else:
            print("MASK OPERATIONS:")
            header = (
                f"{'Operation':<25} | {'Parameters':>25} | {'Pre-Masked':>12} | "
                f"{'Impact':>12} | {'Cumulative':>12} | {'Valid':>10}"
            )
            print(header)
            print("-" * width)
            for stat in self.statistics:
                threshold_str = str(stat.threshold)[:25] if stat.threshold else "N/A"
                print(
                    f"{stat.check_name:<25} | {threshold_str:>25} | "
                    f"{stat.pre_masked_pct:>11.2f}% | {stat.newly_masked_pct:>11.2f}% | "
                    f"{stat.total_masked_pct:>11.2f}% | {stat.valid_pct:>9.2f}%"
                )
            print("-" * width)
            if self.statistics:
                final = self.statistics[-1]
                impact = final.total_masked_pct - self.baseline_masked_pct
                print(
                    f"FINAL: {final.valid_cells:,} valid cells ({final.valid_pct:.2f}%) | "
                    f"Profile operation impact: {impact:+.2f}%"
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
            "PROFILE OPERATION PROCESSING SUMMARY",
            "=" * 60,
        ]

        if not self.history:
            lines.append("No processing steps applied yet.")
        else:
            for i, entry in enumerate(self.history, 1):
                op = entry.get("operation", "unknown")
                params = entry.get("parameters", {})
                stats = entry.get("stats", {})

                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                lines.append(f"{i}. {op}: {param_str}")

                if "newly_masked" in stats:
                    lines.append(
                        f"   Impact: {stats['newly_masked']:,} cells newly masked"
                    )
                elif "new_depth_levels" in stats:
                    lines.append(
                        f"   Structure: {stats.get('original_cells', 'N/A')} cells â†’ "
                        f"{stats['new_depth_levels']} depth levels"
                    )

        lines.append("=" * 60)
        return "\n".join(lines)
