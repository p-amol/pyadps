"""
Velocity Quality Control Module for ADCP Data Processing (v1.1.0).

This module provides functions to validate and correct velocity data, including:
- Magnetic declination correction (modifies velocity data)
- Velocity threshold checks (masks data) - per-component thresholds
- Despiking using phase-space/median filtering (masks data)
- Flatline detection for frozen sensors (masks data)

INTEGRATED FEATURES (v1.1.0):
- VelocityCheckRunner for pipeline orchestration
- Support for 3D velocity arrays (Beam x Cell x Time)
- Per-component velocity thresholds (U, V, W)
- Combined mask (beam 3) updated by all checks
- Integration with pygeomag and NOAA API for magnetic declination
- Shared statistics and reporting via utility module

Key Design Principles:
- **xarray-native**: Operates on xr.Dataset
- **Immutable**: Returns new datasets without modifying inputs
- **Component-wise**: Applies QC to individual velocity beams
- **Combined mask**: mask[3,:,:] = OR(mask[0,:,:], mask[1,:,:], mask[2,:,:])
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from itertools import groupby
from typing import Any, Callable

import numpy as np
import pandas as pd
import scipy.signal as sp_signal
import xarray as xr
import requests

# Try importing GeoMag, handle if missing
try:
    from pygeomag import GeoMag

    HAS_PYGEOMAG = True
except ImportError:  # pragma: no cover
    HAS_PYGEOMAG = False  # pragma: no cover

from .utility import (
    create_default_mask,
    QCCheckStats,
    DataModificationStats,
    QCPipelineReport,
    VELOCITY_MISSING_VALUE,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Default thresholds for velocity components (mm/s)
DEFAULT_VELOCITY_THRESHOLD_U = 2500.0  # East component (horizontal)
DEFAULT_VELOCITY_THRESHOLD_V = 2500.0  # North component (horizontal)
DEFAULT_VELOCITY_THRESHOLD_W = 500.0  # Vertical component (typically weaker)

DEFAULT_DESPIKE_KERNEL = 13
DEFAULT_DESPIKE_CUTOFF = 3.0
DEFAULT_FLATLINE_KERNEL = 4
DEFAULT_FLATLINE_CUTOFF = 1.0

# Threshold ranges for validation
THRESHOLD_RANGES: dict[str, tuple[float, float]] = {
    "velocity_u": (0, 10000),  # mm/s
    "velocity_v": (0, 10000),  # mm/s
    "velocity_w": (0, 5000),  # mm/s (typically smaller range)
    "despike_cutoff": (0.5, 10),  # standard deviations
    "flatline_cutoff": (0, 100),  # mm/s tolerance
}


# ============================================================================
# HELPER FUNCTIONS - Magnetic Declination
# ============================================================================


def get_magdec_from_cof(glat: float, glon: float, alt: float, time: float) -> float:
    """
    Calculate magnetic declination using local COF files via pygeomag.

    Parameters
    ----------
    glat : float
        Latitude in degrees.
    glon : float
        Longitude in degrees.
    alt : float
        Altitude in feet (GeoMag default) or km depending on implementation.
    time : float
        Year (float, e.g. 2023.5).

    Returns
    -------
    float
        Declination in degrees.

    Raises
    ------
    ImportError
        If pygeomag module is not installed.
    """
    if not HAS_PYGEOMAG:
        raise ImportError("pygeomag module not found. Install it or use API method.")

    # Selecting COF file According to given year logic from original code
    if 2010 <= time < 2030:
        var = 2010 + (int(time) - 2010) // 5 * 5
        # Assuming wmm folder is in current working directory or path
        file_name = f"wmm/WMM_{var}.COF"
        geo_mag = GeoMag(coefficients_file=file_name)
    else:
        # Fallback or default
        geo_mag = GeoMag("wmm/WMM_2025.COF")

    result = geo_mag.calculate(glat=glat, glon=glon, alt=alt, time=time)
    return result.d


def get_magdec_from_api(lat: float, lon: float, year: float) -> float:
    """
    Retrieve magnetic declination from NOAA WMM2020 API.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    year : float
        Year (float or int).

    Returns
    -------
    float
        Declination in degrees.

    Raises
    ------
    ValueError
        If year is out of supported range.
    requests.RequestException
        If API request fails.
    """
    baseurl_wmm = (
        "https://www.ngdc.noaa.gov/geomag-web/calculators/calculateDeclination?"
    )
    baseurl_igrf = (
        "https://www.ngdc.noaa.gov/geomag-web/calculators/calculateDeclination?"
    )
    baseurl_emm = "https://emmcalc.geomag.info/?magneticcomponent=d&"

    # API Key from original code (Note: Keys may expire/rotate)
    key = "zNEw7"
    result_format = "json"

    if year >= 2025:
        baseurl = baseurl_wmm
        model = "WMM"
    elif year >= 2019:
        baseurl = baseurl_wmm
        model = "IGRF"
    elif year >= 2000:
        baseurl = baseurl_emm
        model = "EMM"
    elif year >= 1590:
        baseurl = baseurl_igrf
        model = "IGRF"
    else:
        raise ValueError(f"Year {year} out of supported range for API")

    url = f"{baseurl}model={model}&lat1={lat}&lon1={lon}&key={key}&startYear={year}&resultFormat={result_format}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # Extract declination from result list
        return data["result"][0]["declination"]
    except Exception as e:
        logger.error(f"Failed to retrieve magnetic declination from API: {e}")
        raise


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
                f"Threshold {name}={value} out of typical range [{min_val}, {max_val}]"
            )


def update_combined_mask(mask: np.ndarray) -> np.ndarray:
    """
    Update the combined signal quality mask (beam 3) from U, V, W masks.

    The combined mask is the logical OR of the three physical velocity
    component masks: if ANY of U, V, or W is flagged, the combined mask
    is also flagged.

    Parameters
    ----------
    mask : np.ndarray
        Mask array with shape (beam, cell, time) where beam >= 4.

    Returns
    -------
    np.ndarray
        Updated mask with beam 3 = OR(beam 0, beam 1, beam 2).
    """
    if mask.shape[0] >= 4:
        # Combined mask = U OR V OR W
        mask[3, :, :] = (
            (mask[0, :, :] == 1) | (mask[1, :, :] == 1) | (mask[2, :, :] == 1)
        ).astype(np.int8)
    return mask


# ============================================================================
# CORE FUNCTIONS
# ============================================================================


def correct_magnetic_declination(
    ds: xr.Dataset,
    declination: float | None = None,
    use_api: bool = False,
    lat: float | None = None,
    lon: float | None = None,
    year: float | None = None,
) -> xr.Dataset:
    """
    Correct horizontal velocity components for magnetic declination.

    Rotates Beam 0 (U/East) and Beam 1 (V/North) by the declination angle.
    Modifies the 'velocity' variable in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing 'velocity'.
    declination : float, optional
        Magnetic declination in degrees. If None, tries to calculate/fetch it.
    use_api : bool, default False
        If True, use NOAA API. If False, use local pygeomag/COF files.
    lat, lon : float, optional
        Location for declination calculation. If None, looks for attrs in dataset.
    year : float, optional
        Year for calculation. If None, infers from dataset time.

    Returns
    -------
    xr.Dataset
        Dataset with rotated velocities.

    Raises
    ------
    ValueError
        If velocity data not found or cannot calculate declination.
    """
    if "velocity" not in ds.data_vars:
        raise ValueError("Velocity data not found in dataset")

    # 1. Determine Declination
    if declination is None:
        # Try to find metadata if arguments missing
        if lat is None:
            lat = ds.attrs.get("latitude", ds.attrs.get("lat"))
        if lon is None:
            lon = ds.attrs.get("longitude", ds.attrs.get("lon"))
        if year is None:
            # Try to get year from middle of time series
            if "time" in ds.coords:
                ts = ds["time"].values
                mid_ts = pd.to_datetime(ts[len(ts) // 2])
                year = mid_ts.year + mid_ts.day_of_year / 366.0

        if lat is None or lon is None or year is None:
            raise ValueError(
                "Cannot calculate declination: missing lat/lon/year. "
                "Provide 'declination' explicitly or ensure metadata exists."
            )

        logger.info(f"Calculating mag dec for Lat:{lat}, Lon:{lon}, Year:{year:.2f}")
        if use_api:
            declination = get_magdec_from_api(lat, lon, year)
        else:
            # Default altitude 0
            declination = get_magdec_from_cof(lat, lon, 0, year)

    # 2. Perform Rotation
    ds_out = ds.copy(deep=True)
    velocity = ds_out["velocity"].values.copy()  # (Beam, Cell, Time)

    # Check shape
    if velocity.shape[0] < 2:
        logger.warning("Velocity has fewer than 2 beams, cannot rotate UV.")
        return ds_out

    # Convert to radians
    mag_rad = np.deg2rad(declination)
    cos_a = np.cos(mag_rad)
    sin_a = np.sin(mag_rad)

    # Extract components (Handle missing values)
    u = velocity[0].astype(float)
    v = velocity[1].astype(float)

    # Mask missing values for calculation
    invalid_mask = (u == VELOCITY_MISSING_VALUE) | (v == VELOCITY_MISSING_VALUE)
    u[invalid_mask] = np.nan
    v[invalid_mask] = np.nan

    # Rotation (Standard vector rotation)
    # New U (East) = U * cos(dec) + V * sin(dec)
    # New V (North) = -U * sin(dec) + V * cos(dec)
    # (Matches provided user logic)
    u_new = u * cos_a + v * sin_a
    v_new = -1 * u * sin_a + v * cos_a

    # Restore missing values (sentinel mode only; NaN propagates correctly in float mode)
    if not np.issubdtype(velocity.dtype, np.floating):
        u_new = np.where(np.isnan(u_new), VELOCITY_MISSING_VALUE, u_new)
        v_new = np.where(np.isnan(v_new), VELOCITY_MISSING_VALUE, v_new)

    # Assign back
    velocity[0] = u_new
    velocity[1] = v_new

    ds_out["velocity"].values = velocity
    ds_out.attrs["magnetic_declination_applied"] = declination

    logger.info(f"Applied magnetic declination correction: {declination:.4f} degrees")
    return ds_out


def velocity_threshold_check(
    ds: xr.Dataset,
    cutoff_u: float = DEFAULT_VELOCITY_THRESHOLD_U,
    cutoff_v: float = DEFAULT_VELOCITY_THRESHOLD_V,
    cutoff_w: float = DEFAULT_VELOCITY_THRESHOLD_W,
) -> xr.Dataset:
    """
    Mask velocities exceeding component-specific magnitude thresholds.

    Applies different thresholds to each velocity component:
    - U (East): typically larger threshold for horizontal currents
    - V (North): typically larger threshold for horizontal currents
    - W (Vertical): typically smaller threshold (vertical velocities are weaker)

    The combined mask (beam 3) is updated as the OR of U, V, W masks.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing 'velocity' with shape (beam, cell, time).
    cutoff_u : float, default 2500.0
        Velocity magnitude cutoff for U (East) component in mm/s.
    cutoff_v : float, default 2500.0
        Velocity magnitude cutoff for V (North) component in mm/s.
    cutoff_w : float, default 500.0
        Velocity magnitude cutoff for W (Vertical) component in mm/s.

    Returns
    -------
    xr.Dataset
        Dataset with updated mask where:
        - mask[0,:,:] flags |U| > cutoff_u
        - mask[1,:,:] flags |V| > cutoff_v
        - mask[2,:,:] flags |W| > cutoff_w
        - mask[3,:,:] = OR(mask[0], mask[1], mask[2])

    Examples
    --------
    >>> ds = velocity_threshold_check(ds, cutoff_u=2500, cutoff_v=2500, cutoff_w=500)
    """
    _validate_threshold("velocity_u", cutoff_u)
    _validate_threshold("velocity_v", cutoff_v)
    _validate_threshold("velocity_w", cutoff_w)

    if "velocity" not in ds.data_vars:
        logger.warning("Velocity data not found")
        return ds

    mask = _get_mask_or_create(ds)
    velocity = ds["velocity"]

    # Get mask as numpy array for modification
    mask_values = mask.values.copy()
    n_beams = velocity.shape[0]

    # Apply per-component thresholds
    cutoffs = [cutoff_u, cutoff_v, cutoff_w]
    newly_flagged_per_beam = []

    for beam_idx in range(min(n_beams, 3)):  # Only U, V, W (beams 0, 1, 2)
        vel_component = velocity.isel(beam=beam_idx).values
        cutoff = cutoffs[beam_idx]

        # Flag where absolute value exceeds threshold
        flag = np.abs(vel_component) > cutoff

        # Count newly flagged before update
        pre_flagged = (mask_values[beam_idx, :, :] == 1).sum()

        # Update mask with OR logic (preserve existing flags)
        mask_values[beam_idx, :, :] = np.where(
            flag, 1, mask_values[beam_idx, :, :]
        ).astype(np.int8)

        post_flagged = (mask_values[beam_idx, :, :] == 1).sum()
        newly_flagged_per_beam.append(int(post_flagged - pre_flagged))

    # Update combined mask (beam 3)
    mask_values = update_combined_mask(mask_values)

    # Create output dataset
    ds_out = ds.copy(deep=True)

    # Preserve mask attributes
    mask_updated = xr.DataArray(
        data=mask_values,
        dims=mask.dims,
        coords=mask.coords,
        attrs=mask.attrs.copy(),
    )
    ds_out["mask"] = mask_updated

    # Calculate total newly flagged
    total_newly_flagged = sum(newly_flagged_per_beam)

    logger.info(
        f"Velocity threshold check applied: "
        f"cutoff_u={cutoff_u}, cutoff_v={cutoff_v}, cutoff_w={cutoff_w}, "
        f"newly flagged: U={newly_flagged_per_beam[0]}, V={newly_flagged_per_beam[1]}, "
        f"W={newly_flagged_per_beam[2]}, total={total_newly_flagged}"
    )

    return ds_out


def despike_check(
    ds: xr.Dataset,
    kernel_size: int = DEFAULT_DESPIKE_KERNEL,
    cutoff: float = DEFAULT_DESPIKE_CUTOFF,
) -> xr.Dataset:
    """
    Remove anomalous spikes using a median filter.

    Iterates over beams (U, V, W) and cells to apply filter along the time
    dimension. The combined mask (beam 3) is updated as the OR of U, V, W masks.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    kernel_size : int, default 13
        Window size for rolling median filter.
    cutoff : float, default 3.0
        Number of standard deviations to identify spikes.

    Returns
    -------
    xr.Dataset
        Dataset with updated mask where:
        - mask[0:3,:,:] updated for each velocity component
        - mask[3,:,:] = OR(mask[0], mask[1], mask[2])
    """
    _validate_threshold("despike_cutoff", cutoff)

    if "velocity" not in ds.data_vars:
        return ds

    ds_out = ds.copy(deep=True)
    mask = _get_mask_or_create(ds_out).values.copy()
    velocity = ds["velocity"].values.astype(float)

    # Handle missing values
    velocity[velocity == VELOCITY_MISSING_VALUE] = np.nan

    n_beams, n_cells, n_time = velocity.shape

    # Only process U, V, W (beams 0, 1, 2), not the combined beam
    beams_to_process = min(n_beams, 3)

    total_spikes = 0
    spikes_per_beam = [0, 0, 0]

    # Iterate over Beams and Cells
    for b in range(beams_to_process):
        for c in range(n_cells):
            ts_data = velocity[b, c, :]

            # Skip if all NaN
            if np.all(np.isnan(ts_data)):
                continue

            # Apply median filter
            filt = sp_signal.medfilt(ts_data, kernel_size=kernel_size)

            # Diff
            diff = np.abs(ts_data - filt)

            # Threshold
            std_dev = np.nanstd(diff)
            spike_thresh = cutoff * std_dev

            # Identify spikes (where diff > threshold)
            # Ensure we don't flag NaNs that were already there
            spikes = (diff >= spike_thresh) & (~np.isnan(diff))

            # Update mask
            if np.any(spikes):
                # OR operation with existing mask
                mask[b, c, spikes] = 1
                spike_count = int(np.sum(spikes))
                spikes_per_beam[b] += spike_count
                total_spikes += spike_count

    # Update combined mask (beam 3)
    mask = update_combined_mask(mask)

    ds_out["mask"].values = mask
    logger.info(
        f"Despike check applied: kernel={kernel_size}, cutoff={cutoff}, "
        f"spikes found: U={spikes_per_beam[0]}, V={spikes_per_beam[1]}, "
        f"W={spikes_per_beam[2]}, total={total_spikes}"
    )
    return ds_out


def flatline_check(
    ds: xr.Dataset,
    kernel_size: int = DEFAULT_FLATLINE_KERNEL,
    cutoff: float = DEFAULT_FLATLINE_CUTOFF,
) -> xr.Dataset:
    """
    Check for velocities that are constant (flatline) over a period of time.
    Often indicates a frozen sensor or repeater error.

    Processes each velocity component (U, V, W) independently.
    The combined mask (beam 3) is updated as the OR of U, V, W masks.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    kernel_size : int, default 4
        Number of consecutive ensembles required to trigger flatline flag.
    cutoff : float, default 1.0
        Permitted deviation (tolerance) to consider values "equal".

    Returns
    -------
    xr.Dataset
        Dataset with updated mask where:
        - mask[0:3,:,:] updated for each velocity component
        - mask[3,:,:] = OR(mask[0], mask[1], mask[2])
    """
    _validate_threshold("flatline_cutoff", cutoff)

    if "velocity" not in ds.data_vars:
        return ds

    ds_out = ds.copy(deep=True)
    mask = _get_mask_or_create(ds_out).values.copy()
    velocity = ds["velocity"].values.astype(float)
    velocity[velocity == VELOCITY_MISSING_VALUE] = np.nan

    n_beams, n_cells, n_time = velocity.shape

    # Only process U, V, W (beams 0, 1, 2), not the combined beam
    beams_to_process = min(n_beams, 3)

    total_flat = 0
    flat_per_beam = [0, 0, 0]

    for b in range(beams_to_process):
        for c in range(n_cells):
            ts_data = velocity[b, c, :]

            # Compute difference between consecutive elements
            # np.diff returns array of size N-1
            diff = np.abs(np.diff(ts_data))

            # Insert 0 at start to maintain shape matching original algorithm
            diff = np.insert(
                diff, 0, np.inf
            )  # Using inf ensures first point isn't accidentally flagged

            # Following user logic: force first element to be '0 diff'
            diff[0] = 0

            # Boolean array where variation is small
            is_flat = diff <= cutoff

            # Group consecutive Trues using itertools.groupby
            index = 0
            for k, g in groupby(is_flat):
                subset_len = len(list(g))
                if k:  # If True (is flat)
                    if subset_len >= kernel_size:
                        # Flag this segment
                        mask[b, c, index : index + subset_len] = 1
                        flat_per_beam[b] += subset_len
                        total_flat += subset_len
                index += subset_len

    # Update combined mask (beam 3)
    mask = update_combined_mask(mask)

    ds_out["mask"].values = mask
    logger.info(
        f"Flatline check applied: kernel={kernel_size}, cutoff={cutoff}, "
        f"flagged: U={flat_per_beam[0]}, V={flat_per_beam[1]}, "
        f"W={flat_per_beam[2]}, total={total_flat}"
    )
    return ds_out


# ============================================================================
# RUNNER CLASS
# ============================================================================


class VelocityCheckRunner:
    """
    Orchestrator for velocity validation and correction.

    Provides a fluent API (method chaining) matching the SignalQualityRunner
    and SensorHealthRunner style.

    Attributes
    ----------
    dataset : xr.Dataset
        Current working dataset.
    original : xr.Dataset
        Original unmodified dataset.
    statistics : list[QCCheckStats]
        QC check statistics.
    modifications : list[DataModificationStats]
        Data modification statistics (e.g. magnetic correction).
    history : list[dict[str, Any]]
        Processing history.

    Examples
    --------
    >>> runner = VelocityCheckRunner(ds)
    >>> ds_qc = (runner
    ...     .magnetic_correction(declination=-5.0)
    ...     .threshold(cutoff_u=2500, cutoff_v=2500, cutoff_w=500)
    ...     .despike(kernel_size=13, cutoff=3.0)
    ...     .flatline(kernel_size=4, cutoff=1.0)
    ...     .finalize())
    >>> runner.print_statistics()
    """

    def __init__(self, ds: xr.Dataset) -> None:
        """
        Initialize VelocityCheckRunner with dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Input ADCP dataset to process.
        """
        self.original: xr.Dataset = ds.copy(deep=True)
        self.dataset: xr.Dataset = ds.copy(deep=True)

        if "mask" not in self.dataset.data_vars:
            self.dataset["mask"] = create_default_mask(self.dataset)

        self._update_baseline()
        self.statistics: list[QCCheckStats] = []
        self.modifications: list[DataModificationStats] = []
        self.history: list[dict[str, Any]] = []

        logger.info(
            f"VelocityCheckRunner initialized: {self.total_cells:,} cells, "
            f"{self.baseline_masked:,} pre-masked ({self.baseline_masked_pct:.2f}%)"
        )

    def _update_baseline(self) -> None:
        """Calculate baseline statistics from current dataset."""
        mask = self.dataset["mask"]
        self.total_cells = int(mask.size)
        self.baseline_masked = int((mask == 1).sum())
        self.baseline_masked_pct = (
            (100 * self.baseline_masked / self.total_cells)
            if self.total_cells > 0
            else 0.0
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
        check_func: Callable[..., xr.Dataset],
        threshold: float | tuple[float, ...] | dict[str, float] | None,
        **kwargs: Any,
    ) -> VelocityCheckRunner:
        """
        Apply a QC check and record statistics.

        Parameters
        ----------
        check_name : str
            Name of the check for display.
        check_func : Callable
            Function to apply.
        threshold : float or tuple or dict or None
            Threshold value(s) for the check.
        **kwargs : Any
            Arguments to pass to check_func.

        Returns
        -------
        VelocityCheckRunner
            Self for method chaining.
        """
        # Stats before check
        mask_pre = self.dataset["mask"].values
        pre_masked = int((mask_pre == 1).sum())

        # Apply Check (updates self.dataset)
        self.dataset = check_func(self.dataset, **kwargs)

        # Stats after check
        mask_post = self.dataset["mask"].values
        post_masked = int((mask_post == 1).sum())
        newly_masked = post_masked - pre_masked

        # Create Statistics Object
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

        # Record History
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
    # CHECK METHODS
    # ========================================================================

    def magnetic_correction(
        self,
        declination: float | None = None,
        use_api: bool = False,
        lat: float | None = None,
        lon: float | None = None,
        year: float | None = None,
    ) -> VelocityCheckRunner:
        """
        Apply magnetic declination correction to velocity data.

        This modifies the data values, not the mask.

        Parameters
        ----------
        declination : float, optional
            Magnetic declination in degrees.
        use_api : bool, default False
            Use NOAA API instead of local COF files.
        lat, lon : float, optional
            Location for declination calculation.
        year : float, optional
            Year for calculation.

        Returns
        -------
        VelocityCheckRunner
            Self for method chaining.
        """
        # Capture stats before
        orig_vel = self.dataset["velocity"]
        orig_mean = float(orig_vel.mean())

        self.dataset = correct_magnetic_declination(
            self.dataset, declination, use_api, lat, lon, year
        )

        # Stats after
        new_vel = self.dataset["velocity"]
        new_mean = float(new_vel.mean())

        mod_stat = DataModificationStats(
            operation="magnetic_correction",
            variable_name="velocity",
            original_stats={"mean": orig_mean},
            modified_stats={"mean": new_mean},
            metadata={
                "declination_applied": self.dataset.attrs.get(
                    "magnetic_declination_applied"
                )
            },
        )
        self.modifications.append(mod_stat)
        self._add_history(
            operation="magnetic_correction",
            parameters={"declination": declination, "use_api": use_api},
        )
        return self

    def threshold(
        self,
        cutoff_u: float = DEFAULT_VELOCITY_THRESHOLD_U,
        cutoff_v: float = DEFAULT_VELOCITY_THRESHOLD_V,
        cutoff_w: float = DEFAULT_VELOCITY_THRESHOLD_W,
    ) -> VelocityCheckRunner:
        """
        Apply velocity magnitude threshold check with per-component thresholds.

        Parameters
        ----------
        cutoff_u : float, default 2500.0
            Velocity magnitude cutoff for U (East) component in mm/s.
        cutoff_v : float, default 2500.0
            Velocity magnitude cutoff for V (North) component in mm/s.
        cutoff_w : float, default 500.0
            Velocity magnitude cutoff for W (Vertical) component in mm/s.

        Returns
        -------
        VelocityCheckRunner
            Self for method chaining.

        Examples
        --------
        >>> runner.threshold(cutoff_u=2500, cutoff_v=2500, cutoff_w=500)
        """
        return self._record_check(
            "Velocity Threshold",
            velocity_threshold_check,
            threshold={"U": cutoff_u, "V": cutoff_v, "W": cutoff_w},
            cutoff_u=cutoff_u,
            cutoff_v=cutoff_v,
            cutoff_w=cutoff_w,
        )

    def despike(
        self,
        kernel_size: int = DEFAULT_DESPIKE_KERNEL,
        cutoff: float = DEFAULT_DESPIKE_CUTOFF,
    ) -> VelocityCheckRunner:
        """
        Apply despike filter check.

        Parameters
        ----------
        kernel_size : int, default 13
            Window size for rolling median filter.
        cutoff : float, default 3.0
            Number of standard deviations to identify spikes.

        Returns
        -------
        VelocityCheckRunner
            Self for method chaining.
        """
        return self._record_check(
            "Despike",
            despike_check,
            threshold=(kernel_size, cutoff),
            kernel_size=kernel_size,
            cutoff=cutoff,
        )

    def flatline(
        self,
        kernel_size: int = DEFAULT_FLATLINE_KERNEL,
        cutoff: float = DEFAULT_FLATLINE_CUTOFF,
    ) -> VelocityCheckRunner:
        """
        Apply flatline detection check.

        Parameters
        ----------
        kernel_size : int, default 4
            Number of consecutive ensembles required to trigger flatline flag.
        cutoff : float, default 1.0
            Permitted deviation (tolerance) to consider values "equal".

        Returns
        -------
        VelocityCheckRunner
            Self for method chaining.
        """
        return self._record_check(
            "Flatline",
            flatline_check,
            threshold=(kernel_size, cutoff),
            kernel_size=kernel_size,
            cutoff=cutoff,
        )

    # ========================================================================
    # PIPELINE UTILITIES
    # ========================================================================

    def apply_pipeline(
        self,
        checks: dict[str, Any] | None = None,
        order: list[str] | None = None,
    ) -> VelocityCheckRunner:
        """
        Apply multiple QC checks using a pipeline configuration.

        Parameters
        ----------
        checks : dict, optional
            Dict of {check_name: cutoff_value or dict of params}.
            If None, uses defaults for all checks.
        order : list, optional
            List of check names in execution order.
            If None, uses default order: threshold, despike, flatline.

        Returns
        -------
        VelocityCheckRunner
            Self for method chaining.

        Examples
        --------
        >>> runner = VelocityCheckRunner(ds)
        >>> runner.apply_pipeline()  # Use all defaults
        >>> runner.apply_pipeline(checks={
        ...     "threshold": {"cutoff_u": 3000, "cutoff_v": 3000, "cutoff_w": 600},
        ...     "despike": {"kernel_size": 11}
        ... })
        """
        if checks is None:
            checks = {
                "threshold": {
                    "cutoff_u": DEFAULT_VELOCITY_THRESHOLD_U,
                    "cutoff_v": DEFAULT_VELOCITY_THRESHOLD_V,
                    "cutoff_w": DEFAULT_VELOCITY_THRESHOLD_W,
                },
                "despike": {
                    "kernel_size": DEFAULT_DESPIKE_KERNEL,
                    "cutoff": DEFAULT_DESPIKE_CUTOFF,
                },
                "flatline": {
                    "kernel_size": DEFAULT_FLATLINE_KERNEL,
                    "cutoff": DEFAULT_FLATLINE_CUTOFF,
                },
            }

        if order is None:
            order = ["threshold", "despike", "flatline"]

        for name in order:
            if name not in checks:
                continue

            params = checks[name]

            if name == "threshold":
                if isinstance(params, dict):
                    self.threshold(**params)
                else:
                    # Legacy: single value applies to U and V, W gets default
                    self.threshold(cutoff_u=params, cutoff_v=params)
            elif name == "despike":
                if isinstance(params, dict):
                    self.despike(**params)
                else:
                    self.despike(cutoff=params)
            elif name == "flatline":
                if isinstance(params, dict):
                    self.flatline(**params)
                else:
                    self.flatline(cutoff=params)

        return self

    def reset(self) -> VelocityCheckRunner:
        """
        Reset dataset to original state and clear history.

        Returns
        -------
        VelocityCheckRunner
            Self for method chaining.
        """
        self.dataset = self.original.copy(deep=True)
        if "mask" not in self.dataset.data_vars:
            self.dataset["mask"] = create_default_mask(self.dataset)
        self.statistics = []
        self.modifications = []
        self.history = []
        self._update_baseline()
        logger.info("VelocityCheckRunner reset to original state")
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
        ds_out.attrs["velocity_check_processing_history"] = str(self.history)
        ds_out.attrs["velocity_check_processed_at"] = datetime.now(
            timezone.utc
        ).isoformat()
        logger.info("VelocityCheckRunner finalized")
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
            module_name="velocity_check",
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
        print("VELOCITY CHECK PROCESSING STATISTICS")
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
                f"{'Check':<25} | {'Threshold':>18} | {'Pre-Masked':>12} | "
                f"{'Impact':>12} | {'Cumulative':>12} | {'Valid':>10}"
            )
            print(header)
            print("-" * width)
            for stat in self.statistics:
                # Format threshold (handle different types)
                if isinstance(stat.threshold, dict):
                    # Per-component thresholds
                    threshold_str = f"U:{stat.threshold.get('U', 'N/A')}"
                elif isinstance(stat.threshold, tuple):
                    threshold_str = f"{stat.threshold}"
                elif stat.threshold is None:
                    threshold_str = "N/A"
                else:
                    threshold_str = f"{stat.threshold}"

                print(
                    f"{stat.check_name:<25} | {threshold_str:>18} | "
                    f"{stat.pre_masked_pct:>11.2f}% | {stat.newly_masked_pct:>11.2f}% | "
                    f"{stat.total_masked_pct:>11.2f}% | {stat.valid_pct:>9.2f}%"
                )
            print("-" * width)
            final = self.statistics[-1]
            impact = final.total_masked_pct - self.baseline_masked_pct
            print(
                f"FINAL: {final.valid_cells:,} valid cells ({final.valid_pct:.2f}%) | "
                f"Velocity QC impact: {impact:+.2f}%"
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
            "VELOCITY CHECK PROCESSING SUMMARY",
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
