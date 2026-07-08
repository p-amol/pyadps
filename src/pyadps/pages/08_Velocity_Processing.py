"""
07_Velocity_Processing.py - Velocity Quality Control Page (Refactored for pyadps v1.0.0)

This page allows users to:
1. Apply magnetic declination correction to velocity data
2. Configure and apply velocity threshold checks:
   - Zonal (U/East) velocity cutoff
   - Meridional (V/North) velocity cutoff
   - Vertical (W) velocity cutoff
3. Apply despike filtering to remove anomalous spikes
4. Apply flatline detection for frozen sensors
5. Preview and compare mask files
6. Save processing results to the central ProcessedDataset

Architecture:
- Uses st.session_state.processor (ProcessedDataset) as central state manager
- Uses preview_velocity_proc (staging ProcessedDataset) for safe preview
- Uses VelocityCheckRunner for interactive preview and threshold application
- All processing is tracked through the processor's reports
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pyadps.processing import ProcessedDataset

try:
    from plotly_resampler import FigureResampler

    HAS_RESAMPLER = True
except ImportError:
    HAS_RESAMPLER = False

# =============================================================================
# PAGE CONFIGURATION AND VALIDATION
# =============================================================================

st.set_page_config(page_title="Velocity Test", page_icon="🌊", layout="wide")

# Check if processor exists
if "processor" not in st.session_state or st.session_state.processor is None:
    st.error("⚠️ No data loaded! Please read a file on the **Read File** page first.")
    st.stop()

# Get processor and dataset
proc = st.session_state.processor
ds = proc.dataset


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_total_ensembles() -> int:
    """Get total number of ensembles from dataset."""
    if "time" in ds.dims:
        return ds.sizes["time"]
    elif "ensemble" in ds.dims:
        return ds.sizes["ensemble"]
    return 0


def get_total_cells() -> int:
    """Get total number of cells from dataset."""
    if "cell" in ds.dims:
        return ds.sizes["cell"]
    elif "depth" in ds.dims:
        return ds.sizes["depth"]
    return 0


def get_total_beams() -> int:
    """Get total number of beams from dataset."""
    if "beam" in ds.dims:
        return ds.sizes["beam"]
    return 4


def get_time_axis():
    """Get time axis for plotting."""
    if "time" in ds.coords:
        return pd.to_datetime(ds["time"].values)
    elif "ensemble" in ds.coords:
        return ds["ensemble"].values
    return np.arange(get_total_ensembles())


def get_time_interval_str() -> str:
    """Get time interval as a formatted string."""
    if "time" in ds.coords:
        time_vals = pd.to_datetime(ds["time"].values)
        if len(time_vals) >= 2:
            interval = time_vals[1] - time_vals[0]
            return str(interval)
    return "N/A"


def _get_dataset_lat_lon() -> tuple[float | None, float | None]:
    """Look up latitude/longitude from dataset attributes, if already present.

    Checks a few common key variants since attributes may have been added
    via different pages (e.g. 'Latitude'/'Longitude' from the standard
    attribute form, or lowercase 'latitude'/'longitude'/'lat'/'lon').
    """
    lat_keys = ("Latitude", "latitude", "lat")
    lon_keys = ("Longitude", "longitude", "lon")

    def _find(keys: tuple[str, ...]) -> float | None:
        for key in keys:
            value = proc.dataset.attrs.get(key)
            if value in (None, ""):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    return _find(lat_keys), _find(lon_keys)


def is_earth_coordinates() -> bool:
    """
    Check if data is in Earth coordinates.
    Returns True if transformed to Earth (U, V, W), False for Beam coordinates.
    """
    try:
        coord_info = ds.fixed_leader.coordinate_transformation(ens=0)
        coord_str = coord_info.get("Coordinates", "")
        return "Earth" in coord_str
    except Exception:
        # Fallback: check attrs
        return ds.attrs.get("coordinate_system", "").lower() == "earth"


def get_velocity_labels() -> tuple[str, str, str]:
    """Get appropriate velocity component labels based on coordinate system."""
    if is_earth_coordinates():
        return ("U (East)", "V (North)", "W (Vertical)")
    else:
        return ("Beam 1", "Beam 2", "Beam 3")


def status_color_map(value: object) -> str:
    """Map status values to colors for dataframe styling."""
    if value == "True":
        return "background-color: green; color: white"
    elif value == "False":
        return "background-color: red; color: white"
    return ""


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def plot_velocity_component(beam_idx: int, title: str) -> None:
    """Plot velocity component as a heatmap."""
    if "velocity" not in ds.data_vars:
        st.warning("No velocity data available.")
        return

    velocity = ds["velocity"].values
    n_ensembles = get_total_ensembles()
    n_cells = get_total_cells()

    # Get velocity component (beam, cell, time)
    vel_data = velocity[beam_idx, :, :].astype(float)
    vel_data[vel_data == -32768] = np.nan

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=vel_data,
            x=np.arange(n_ensembles),
            y=np.arange(n_cells),
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="mm/s"),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Ensemble",
        yaxis_title="Cell",
        height=350,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_mask_comparison(
    mask_original: np.ndarray, mask_preview: np.ndarray, title: str = "Mask Comparison"
) -> None:
    """Plot side-by-side mask comparison using beam 3 (combined mask)."""
    n_ensembles = get_total_ensembles()
    n_cells = get_total_cells()

    # Use beam 3 (combined mask) if available, otherwise beam 0
    beam_idx = 3 if mask_original.shape[0] > 3 else 0

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Original Mask (Combined)", "Preview Mask (Combined)"],
    )

    # Original mask
    fig.add_trace(
        go.Heatmap(
            z=mask_original[beam_idx, :, :],
            x=np.arange(n_ensembles),
            y=np.arange(n_cells),
            colorscale=[[0, "green"], [1, "red"]],
            showscale=False,
            hoverongaps=False,
        ),
        row=1,
        col=1,
    )

    # Preview mask
    fig.add_trace(
        go.Heatmap(
            z=mask_preview[beam_idx, :, :],
            x=np.arange(n_ensembles),
            y=np.arange(n_cells),
            colorscale=[[0, "green"], [1, "red"]],
            showscale=False,
            hoverongaps=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=title,
        height=400,
    )

    fig.update_xaxes(title_text="Ensemble", row=1, col=1)
    fig.update_xaxes(title_text="Ensemble", row=1, col=2)
    fig.update_yaxes(title_text="Cell", row=1, col=1)

    st.plotly_chart(fig, use_container_width=True)


def plot_velocity_histogram(beam_idx: int, title: str) -> None:
    """Plot histogram of velocity component values."""
    if "velocity" not in ds.data_vars:
        st.warning("No velocity data available.")
        return

    velocity = ds["velocity"].values
    vel_data = velocity[beam_idx, :, :].flatten().astype(float)
    vel_data = vel_data[~np.isnan(vel_data)]  # Remove missing values

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=vel_data,
            nbinsx=100,
            marker_color="steelblue",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Velocity (mm/s)",
        yaxis_title="Count",
        height=300,
    )

    st.plotly_chart(fig, use_container_width=True)


def compute_median_filter(data: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Compute rolling median filter for a 1D array.

    Parameters
    ----------
    data : np.ndarray
        1D array of velocity values
    kernel_size : int
        Window size for median filter (must be odd)

    Returns
    -------
    np.ndarray
        Median-filtered values
    """
    import scipy.ndimage as ndimage

    # Handle NaN values by using a masked approach
    data_clean = np.where(np.isnan(data), 0, data)
    median_filtered = ndimage.median_filter(
        data_clean, size=kernel_size, mode="nearest"
    )
    # Restore NaN positions
    median_filtered = np.where(np.isnan(data), np.nan, median_filtered)
    return median_filtered


def detect_spikes(
    data: np.ndarray, kernel_size: int, cutoff: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Detect spikes in velocity data using median filter approach.

    Returns
    -------
    tuple
        (median_filtered, spike_mask, std_dev)
    """
    median_filtered = compute_median_filter(data, kernel_size)
    deviation = np.abs(data - median_filtered)

    # Calculate std dev of deviations (excluding NaN)
    std_dev = np.nanstd(deviation)

    # Flag spikes
    spike_mask = deviation > (cutoff * std_dev)

    return median_filtered, spike_mask, std_dev


def detect_flatlines(data: np.ndarray, kernel_size: int, cutoff: float) -> np.ndarray:
    """
    Detect flatline segments in velocity data.

    Returns
    -------
    np.ndarray
        Boolean mask where True indicates flatline
    """
    n = len(data)
    flatline_mask = np.zeros(n, dtype=bool)

    if n < kernel_size:
        return flatline_mask

    # Calculate consecutive differences
    diffs = np.abs(np.diff(data))

    # Find runs of constant values (diff <= cutoff)
    constant = diffs <= cutoff

    # Find runs of consecutive True values
    run_start = 0
    run_length = 0

    for i, is_const in enumerate(constant):
        if is_const:
            if run_length == 0:
                run_start = i
            run_length += 1
        else:
            # Check if run was long enough
            if run_length >= kernel_size - 1:
                # Mark the entire segment (including endpoints)
                flatline_mask[run_start : run_start + run_length + 1] = True
            run_length = 0

    # Check final run
    if run_length >= kernel_size - 1:
        flatline_mask[run_start : run_start + run_length + 1] = True

    return flatline_mask


def plot_despike_timeseries(
    velocity_data: np.ndarray,
    beam_idx: int,
    cell_idx: int,
    ens_start: int,
    ens_end: int,
    kernel_size: int,
    cutoff: float,
    component_label: str,
) -> None:
    """
    Plot before/after despike visualization for a specific cell.

    Shows:
    - Original velocity time series
    - Median-filtered reference line
    - Detected spikes highlighted
    - ±cutoff×std envelope
    """
    # Extract data for the selected cell and ensemble range
    vel_slice = velocity_data[beam_idx, cell_idx, ens_start:ens_end].astype(float)
    vel_slice[vel_slice == -32768] = np.nan

    x_axis = np.arange(ens_start, ens_end)

    # Detect spikes
    median_filtered, spike_mask, std_dev = detect_spikes(vel_slice, kernel_size, cutoff)

    # Create figure
    if HAS_RESAMPLER and len(x_axis) > 5000:
        fig = FigureResampler(go.Figure())
    else:
        fig = go.Figure()

    # Add envelope (±cutoff×std around median)
    upper_bound = median_filtered + cutoff * std_dev
    lower_bound = median_filtered - cutoff * std_dev

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_axis, x_axis[::-1]]),
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill="toself",
            fillcolor="rgba(135, 206, 250, 0.3)",
            line=dict(color="rgba(135, 206, 250, 0)"),
            name=f"±{cutoff}σ envelope",
            hoverinfo="skip",
        )
    )

    # Add median line
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=median_filtered,
            mode="lines",
            name="Median filtered",
            line=dict(color="blue", width=1.5),
        )
    )

    # Add original velocity (non-spike points)
    valid_points = ~spike_mask & ~np.isnan(vel_slice)
    fig.add_trace(
        go.Scatter(
            x=x_axis[valid_points],
            y=vel_slice[valid_points],
            mode="markers",
            name="Valid data",
            marker=dict(color="green", size=4, opacity=0.6),
        )
    )

    # Add spike points
    if np.any(spike_mask):
        fig.add_trace(
            go.Scatter(
                x=x_axis[spike_mask],
                y=vel_slice[spike_mask],
                mode="markers",
                name=f"Detected spikes ({np.sum(spike_mask)})",
                marker=dict(color="red", size=8, symbol="x"),
            )
        )

    # Count statistics
    n_valid = np.sum(~np.isnan(vel_slice))
    n_spikes = np.sum(spike_mask)
    spike_pct = (n_spikes / n_valid * 100) if n_valid > 0 else 0

    fig.update_layout(
        title=f"{component_label} - Cell {cell_idx} (Ensembles {ens_start}-{ens_end}) | "
        f"Spikes: {n_spikes} ({spike_pct:.1f}%)",
        xaxis_title="Ensemble",
        yaxis_title="Velocity (mm/s)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_flatline_timeseries(
    velocity_data: np.ndarray,
    beam_idx: int,
    cell_idx: int,
    ens_start: int,
    ens_end: int,
    kernel_size: int,
    cutoff: float,
    component_label: str,
) -> None:
    """
    Plot before/after flatline visualization for a specific cell.

    Shows:
    - Original velocity time series
    - Detected flatline segments highlighted
    - Tolerance band visualization
    """
    # Extract data for the selected cell and ensemble range
    vel_slice = velocity_data[beam_idx, cell_idx, ens_start:ens_end].astype(float)
    vel_slice[vel_slice == -32768] = np.nan

    x_axis = np.arange(ens_start, ens_end)

    # Detect flatlines
    flatline_mask = detect_flatlines(vel_slice, kernel_size, cutoff)

    # Create figure
    if HAS_RESAMPLER and len(x_axis) > 5000:
        fig = FigureResampler(go.Figure())
    else:
        fig = go.Figure()

    # Add normal data points
    normal_mask = ~flatline_mask & ~np.isnan(vel_slice)
    fig.add_trace(
        go.Scatter(
            x=x_axis[normal_mask],
            y=vel_slice[normal_mask],
            mode="markers+lines",
            name="Normal data",
            marker=dict(color="green", size=4),
            line=dict(color="green", width=1),
        )
    )

    # Add flatline segments
    if np.any(flatline_mask):
        # Find contiguous flatline segments for highlighting
        flatline_indices = np.where(flatline_mask)[0]

        fig.add_trace(
            go.Scatter(
                x=x_axis[flatline_mask],
                y=vel_slice[flatline_mask],
                mode="markers+lines",
                name=f"Flatline detected ({np.sum(flatline_mask)})",
                marker=dict(color="red", size=6),
                line=dict(color="red", width=2),
            )
        )

        # Add shaded regions for flatline segments
        # Find segment boundaries
        segments = []
        if len(flatline_indices) > 0:
            seg_start = flatline_indices[0]
            for i in range(1, len(flatline_indices)):
                if flatline_indices[i] != flatline_indices[i - 1] + 1:
                    segments.append((seg_start, flatline_indices[i - 1]))
                    seg_start = flatline_indices[i]
            segments.append((seg_start, flatline_indices[-1]))

        # Add shaded rectangles for each segment
        for seg_start, seg_end in segments:
            y_min = np.nanmin(vel_slice)
            y_max = np.nanmax(vel_slice)
            y_range = y_max - y_min if y_max != y_min else 100

            fig.add_vrect(
                x0=x_axis[seg_start] - 0.5,
                x1=x_axis[seg_end] + 0.5,
                fillcolor="rgba(255, 0, 0, 0.1)",
                layer="below",
                line_width=0,
            )

    # Count statistics
    n_valid = np.sum(~np.isnan(vel_slice))
    n_flatline = np.sum(flatline_mask)
    flatline_pct = (n_flatline / n_valid * 100) if n_valid > 0 else 0

    fig.update_layout(
        title=f"{component_label} - Cell {cell_idx} (Ensembles {ens_start}-{ens_end}) | "
        f"Flatlines: {n_flatline} ({flatline_pct:.1f}%)",
        xaxis_title="Ensemble",
        yaxis_title="Velocity (mm/s)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

# Initialize page-specific session state
if "velocity_initialized" not in st.session_state:
    st.session_state.velocity_initialized = False

if not st.session_state.velocity_initialized:
    # Staging processor for safe preview
    st.session_state.preview_velocity_proc = ProcessedDataset(proc.dataset)
    st.session_state.velocity_preview_run = False
    st.session_state.velocity_applied = False

    # Magnetic correction settings
    st.session_state.apply_magnetic = False
    st.session_state.magnetic_method = "pygeomag"
    # Auto-fill from dataset attributes if latitude/longitude were already
    # recorded (e.g. from a previously written file read back in).
    _attr_lat, _attr_lon = _get_dataset_lat_lon()
    st.session_state.magnetic_lat = _attr_lat if _attr_lat is not None else 0.0
    st.session_state.magnetic_lon = _attr_lon if _attr_lon is not None else 0.0
    st.session_state.magnetic_year = 2025
    st.session_state.magnetic_depth = 0
    st.session_state.magnetic_declination = None

    # Threshold settings (in mm/s)
    st.session_state.apply_threshold = True
    st.session_state.cutoff_u = 2500
    st.session_state.cutoff_v = 2500
    st.session_state.cutoff_w = 500

    # Despike settings
    st.session_state.apply_despike = False
    st.session_state.despike_kernel = 13
    st.session_state.despike_cutoff = 3.0

    # Flatline settings
    st.session_state.apply_flatline = False
    st.session_state.flatline_kernel = 4
    st.session_state.flatline_cutoff = 1.0

    st.session_state.velocity_initialized = True

# Ensure staging processor exists
if "preview_velocity_proc" not in st.session_state:
    st.session_state.preview_velocity_proc = ProcessedDataset(proc.dataset)


# =============================================================================
# PAGE HEADER
# =============================================================================

_, _col_fname = st.columns([5, 1])
with _col_fname:
    st.caption(f"📂 {st.session_state.get('fname', 'No file selected')}")
st.header("🌊 Velocity Test", divider="orange")

st.write("""
The processing in this page applies quality control checks specifically to velocity data.
Velocity checks validate the measured current velocities to identify unreliable data.
""")

# Get velocity labels based on coordinate system
u_label, v_label, w_label = get_velocity_labels()

# =============================================================================
# CALLBACKS
# =============================================================================
# Mutating state in on_click callbacks (rather than calling st.rerun() inside
# an if st.button(): block) avoids resetting the active tab back to the first one.


def _reset_magnetic_declination():
    st.session_state.apply_magnetic = False
    st.session_state.magnetic_declination = None
    st.session_state.preview_velocity_proc = ProcessedDataset(proc.dataset)
    st.session_state.velocity_preview_run = False


def _reset_velocity_tests():
    st.session_state.preview_velocity_proc = ProcessedDataset(proc.dataset)
    st.session_state.velocity_preview_run = False
    st.session_state.velocity_preview_stats = None
    st.session_state.apply_magnetic = False
    st.session_state.magnetic_declination = None


# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Magnetic Declination",
        "Velocity Thresholds",
        "Despike Data",
        "Flatline Detection",
        "Preview",
        "Save & Reset",
    ]
)


# =============================================================================
# TAB 1: MAGNETIC DECLINATION
# =============================================================================

with tab1:
    st.header("Magnetic Declination", divider="blue")

    st.write("""
    Magnetic declination correction rotates the U (East) and V (North) velocity 
    components to correct for the difference between magnetic north and true north.
    
    **Methods:**
    - **pygeomag**: Uses the local pygeomag library with WMM coefficient files (2010-2030)
    - **API**: Uses NOAA's online magnetic declination service (requires internet)
    - **Manual**: Enter the declination directly if known
    
    **Note:** If magnetic declination is changed, other velocity tests should be re-run.
    """)

    method = st.radio(
        "Select calculation method",
        ("pygeomag", "API", "Manual"),
        horizontal=True,
        key="magnetic_method_radio",
    )
    st.session_state.magnetic_method = method

    with st.form(key="magnetic_form"):
        if method == "pygeomag":
            col1, col2 = st.columns(2)
            with col1:
                lat = st.number_input(
                    "Latitude (°)", -90.0, 90.0, st.session_state.magnetic_lat, step=0.1
                )
                lon = st.number_input(
                    "Longitude (°)",
                    -180.0,
                    360.0,
                    st.session_state.magnetic_lon,
                    step=0.1,
                )
            with col2:
                depth = st.number_input(
                    "Depth (m)", 0, 10000, st.session_state.magnetic_depth, step=1
                )
                year = st.number_input(
                    "Year", 2010, 2030, st.session_state.magnetic_year, step=1
                )

        elif method == "API":
            col1, col2 = st.columns(2)
            with col1:
                lat = st.number_input(
                    "Latitude (°)", -90.0, 90.0, st.session_state.magnetic_lat, step=0.1
                )
                lon = st.number_input(
                    "Longitude (°)",
                    -180.0,
                    360.0,
                    st.session_state.magnetic_lon,
                    step=0.1,
                )
            with col2:
                year = st.number_input(
                    "Year", 1950, 2030, st.session_state.magnetic_year, step=1
                )
            depth = 0

        else:  # Manual
            declination_input = st.number_input(
                "Declination (°)", -180.0, 180.0, 0.0, step=0.1
            )
            lat, lon, year, depth = 0.0, 0.0, 2025, 0

        button_label = "Accept" if method == "Manual" else "Compute"
        submitted = st.form_submit_button(button_label)

        if submitted:
            st.session_state.magnetic_lat = lat
            st.session_state.magnetic_lon = lon
            st.session_state.magnetic_year = year
            st.session_state.magnetic_depth = depth

            if method == "Manual":
                st.session_state.magnetic_declination = declination_input
                st.session_state.apply_magnetic = True
                st.success(f"Magnetic declination set to {declination_input:.3f}°")
            else:
                # Calculate using the runner's method
                try:
                    preview_proc = st.session_state.preview_velocity_proc
                    runner = preview_proc.get_velocity_check_runner()
                    runner.magnetic_correction(
                        lat=lat,
                        lon=lon,
                        year=float(year),
                        use_api=(method == "API"),
                    )
                    # Get the computed declination from the dataset
                    computed_dec = runner.dataset.attrs.get(
                        "magnetic_declination_applied", 0
                    )
                    st.session_state.magnetic_declination = computed_dec
                    st.session_state.apply_magnetic = True
                    preview_proc.commit_runner(runner)
                    st.success(f"Magnetic declination computed: {computed_dec:.3f}°")
                except Exception as e:
                    st.error(f"Error computing magnetic declination: {e}")
                    if method == "API":
                        st.info("Try using the 'Manual' method if API is unavailable.")

    if (
        st.session_state.apply_magnetic
        and st.session_state.magnetic_declination is not None
    ):
        st.info(
            f"✓ Magnetic declination: **{st.session_state.magnetic_declination:.3f}°**"
        )

    st.button(
        "Reset Magnetic Declination",
        key="reset_magnetic",
        on_click=_reset_magnetic_declination,
    )


# =============================================================================
# TAB 2: VELOCITY THRESHOLDS
# =============================================================================

with tab2:
    st.header("Velocity Thresholds", divider="blue")

    st.write("""
    Flag velocity values whose magnitude exceeds the specified thresholds.
    Different thresholds are applied to each component:
    
    - **U/V (Horizontal)**: Typically allow larger values (2500 mm/s default)
    - **W (Vertical)**: Typically much smaller values (500 mm/s default)
    
    This reflects oceanographic reality where vertical velocities are typically 
    5-10x smaller than horizontal velocities.
    """)

    # Show velocity histograms to help with threshold selection
    with st.expander(
        "📊 Velocity Distribution (help with threshold selection)", expanded=False
    ):
        col1, col2, col3 = st.columns(3)
        with col1:
            plot_velocity_histogram(0, f"{u_label} Distribution")
        with col2:
            plot_velocity_histogram(1, f"{v_label} Distribution")
        with col3:
            plot_velocity_histogram(2, f"{w_label} Distribution")

    st.session_state.apply_threshold = st.checkbox(
        "Apply velocity threshold check",
        value=st.session_state.apply_threshold,
        key="threshold_checkbox",
    )

    if st.session_state.apply_threshold:
        col1, col2, col3 = st.columns(3)

        with col1:
            cutoff_u = st.number_input(
                f"Max {u_label} (mm/s)",
                min_value=0,
                max_value=10000,
                value=st.session_state.cutoff_u,
                step=100,
                key="cutoff_u_input",
            )
            if cutoff_u is not None:
                st.session_state.cutoff_u = cutoff_u

        with col2:
            cutoff_v = st.number_input(
                f"Max {v_label} (mm/s)",
                min_value=0,
                max_value=10000,
                value=st.session_state.cutoff_v,
                step=100,
                key="cutoff_v_input",
            )
            if cutoff_v is not None:
                st.session_state.cutoff_v = cutoff_v

        with col3:
            cutoff_w = st.number_input(
                f"Max {w_label} (mm/s)",
                min_value=0,
                max_value=5000,
                value=st.session_state.cutoff_w,
                step=50,
                key="cutoff_w_input",
            )
            if cutoff_w is not None:
                st.session_state.cutoff_w = cutoff_w


# =============================================================================
# TAB 3: DESPIKE DATA
# =============================================================================

with tab3:
    st.header("Despike Data", divider="blue")

    st.write("""
    Despike filtering identifies and flags anomalous spikes using a rolling median filter.
    
    - **Kernel Size**: Window size for the median filter (number of ensembles)
    - **Cutoff**: Number of standard deviations from median to flag as spike
    
    Points are flagged where `|velocity - median| > cutoff × std_dev`
    """)

    time_interval = get_time_interval_str()
    st.write(f"**Time interval:** {time_interval}")

    st.session_state.apply_despike = st.checkbox(
        "Apply despike filter",
        value=st.session_state.apply_despike,
        key="despike_checkbox",
    )

    if st.session_state.apply_despike:
        col1, col2 = st.columns(2)

        with col1:
            despike_kernel = st.number_input(
                "Kernel Size (ensembles)",
                min_value=3,
                max_value=min(51, get_total_ensembles())
                if get_total_ensembles() > 3
                else 51,
                value=st.session_state.despike_kernel,
                step=2,
                help="Window size for rolling median. Must be odd.",
                key="despike_kernel_input",
            )
            # Ensure odd number
            if despike_kernel is not None:
                if despike_kernel % 2 == 0:
                    despike_kernel += 1
                st.session_state.despike_kernel = despike_kernel

        with col2:
            despike_cutoff = st.number_input(
                "Standard Deviation Cutoff",
                min_value=0.5,
                max_value=10.0,
                value=st.session_state.despike_cutoff,
                step=0.5,
                help="Points beyond this many std devs from median are flagged.",
                key="despike_cutoff_input",
            )
            if despike_cutoff is not None:
                st.session_state.despike_cutoff = despike_cutoff

        # Time-series visualization
        st.divider()
        st.write("**📈 Despike Visualization**")
        st.write("Preview spike detection for a specific cell and ensemble range.")

        n_cells = get_total_cells()
        n_ensembles = get_total_ensembles()

        # Selection controls
        col_cell, col_comp = st.columns(2)

        with col_cell:
            despike_vis_cell = st.slider(
                "Select Cell/Depth",
                min_value=0,
                max_value=max(0, n_cells - 1),
                value=min(n_cells // 2, n_cells - 1) if n_cells > 0 else 0,
                key="despike_vis_cell",
            )

        with col_comp:
            despike_vis_component = st.radio(
                "Velocity Component",
                options=[0, 1, 2],
                format_func=lambda x: [u_label, v_label, w_label][x],
                horizontal=True,
                key="despike_vis_component",
            )

        # Ensemble range slider
        default_end = min(1000, n_ensembles)
        despike_ens_range = st.slider(
            "Ensemble Range",
            min_value=0,
            max_value=n_ensembles,
            value=(0, default_end),
            key="despike_ens_range",
        )

        # Show the visualization
        if "velocity" in ds.data_vars and n_ensembles > 0 and n_cells > 0:
            velocity_data = ds["velocity"].values
            component_labels = [u_label, v_label, w_label]

            plot_despike_timeseries(
                velocity_data=velocity_data,
                beam_idx=despike_vis_component,
                cell_idx=despike_vis_cell,
                ens_start=despike_ens_range[0],
                ens_end=despike_ens_range[1],
                kernel_size=st.session_state.despike_kernel,
                cutoff=st.session_state.despike_cutoff,
                component_label=component_labels[despike_vis_component],
            )

            st.caption(
                f"🔍 Showing {despike_ens_range[1] - despike_ens_range[0]} ensembles. "
                f"Green = valid data, Red X = detected spikes, Blue line = median filter, "
                f"Shaded area = ±{st.session_state.despike_cutoff}σ envelope."
            )
        else:
            st.info("No velocity data available for visualization.")


# =============================================================================
# TAB 4: FLATLINE DETECTION
# =============================================================================

with tab4:
    st.header("Flatline Detection", divider="blue")

    st.write("""
    Flatline detection identifies and flags constant velocity values over time, 
    which typically indicate a frozen sensor or data transmission error.
    
    - **Kernel Size**: Minimum consecutive points to flag as flatline
    - **Cutoff**: Maximum variation (mm/s) to consider values "constant"
    """)

    time_interval = get_time_interval_str()
    st.write(f"**Time interval:** {time_interval}")

    st.session_state.apply_flatline = st.checkbox(
        "Apply flatline detection",
        value=st.session_state.apply_flatline,
        key="flatline_checkbox",
    )

    if st.session_state.apply_flatline:
        col1, col2 = st.columns(2)

        with col1:
            flatline_kernel = st.number_input(
                "Kernel Size (ensembles)",
                min_value=2,
                max_value=100,
                value=st.session_state.flatline_kernel,
                step=1,
                help="Minimum consecutive constant values to trigger flagging.",
                key="flatline_kernel_input",
            )
            if flatline_kernel is not None:
                st.session_state.flatline_kernel = flatline_kernel

        with col2:
            flatline_cutoff = st.number_input(
                "Deviation Tolerance (mm/s)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.flatline_cutoff),
                step=1.0,
                help="Values within this range are considered 'constant'.",
                key="flatline_cutoff_input",
            )
            if flatline_cutoff is not None:
                st.session_state.flatline_cutoff = flatline_cutoff

        # Time-series visualization
        st.divider()
        st.write("**📈 Flatline Visualization**")
        st.write("Preview flatline detection for a specific cell and ensemble range.")

        n_cells = get_total_cells()
        n_ensembles = get_total_ensembles()

        # Selection controls
        col_cell, col_comp = st.columns(2)

        with col_cell:
            flatline_vis_cell = st.slider(
                "Select Cell/Depth",
                min_value=0,
                max_value=max(0, n_cells - 1),
                value=min(n_cells // 2, n_cells - 1) if n_cells > 0 else 0,
                key="flatline_vis_cell",
            )

        with col_comp:
            flatline_vis_component = st.radio(
                "Velocity Component",
                options=[0, 1, 2],
                format_func=lambda x: [u_label, v_label, w_label][x],
                horizontal=True,
                key="flatline_vis_component",
            )

        # Ensemble range slider
        default_end = min(1000, n_ensembles)
        flatline_ens_range = st.slider(
            "Ensemble Range",
            min_value=0,
            max_value=n_ensembles,
            value=(0, default_end),
            key="flatline_ens_range",
        )

        # Show the visualization
        if "velocity" in ds.data_vars and n_ensembles > 0 and n_cells > 0:
            velocity_data = ds["velocity"].values
            component_labels = [u_label, v_label, w_label]

            plot_flatline_timeseries(
                velocity_data=velocity_data,
                beam_idx=flatline_vis_component,
                cell_idx=flatline_vis_cell,
                ens_start=flatline_ens_range[0],
                ens_end=flatline_ens_range[1],
                kernel_size=st.session_state.flatline_kernel,
                cutoff=st.session_state.flatline_cutoff,
                component_label=component_labels[flatline_vis_component],
            )

            st.caption(
                f"🔍 Showing {flatline_ens_range[1] - flatline_ens_range[0]} ensembles. "
                f"Green = normal data, Red = flatline segments (≥{st.session_state.flatline_kernel} consecutive points "
                f"with ≤{st.session_state.flatline_cutoff} mm/s variation)."
            )
        else:
            st.info("No velocity data available for visualization.")


# =============================================================================
# TAB 5: PREVIEW
# =============================================================================

with tab5:
    st.header("Preview Processing", divider="blue")

    st.write("""
    Preview the effect of velocity tests before applying them to the main processor.
    This allows you to verify the impact of your settings without committing changes.
    """)

    # Show current settings summary at the top
    st.write("**📋 Current Settings:**")
    settings_data = [
        ["Magnetic Correction", "True" if st.session_state.apply_magnetic else "False"],
        ["Velocity Threshold", "True" if st.session_state.apply_threshold else "False"],
        ["Despike Filter", "True" if st.session_state.apply_despike else "False"],
        ["Flatline Detection", "True" if st.session_state.apply_flatline else "False"],
    ]
    settings_df = pd.DataFrame(settings_data, columns=["Test", "Enabled"])
    styled_settings = settings_df.style.map(status_color_map, subset=["Enabled"])
    st.write(styled_settings.to_html(), unsafe_allow_html=True)

    # Show threshold details if enabled
    if st.session_state.apply_threshold:
        st.write(
            f"- Threshold U: {st.session_state.cutoff_u} mm/s, V: {st.session_state.cutoff_v} mm/s, W: {st.session_state.cutoff_w} mm/s"
        )
    if st.session_state.apply_despike:
        st.write(
            f"- Despike: kernel={st.session_state.despike_kernel}, cutoff={st.session_state.despike_cutoff}σ"
        )
    if st.session_state.apply_flatline:
        st.write(
            f"- Flatline: kernel={st.session_state.flatline_kernel}, cutoff={st.session_state.flatline_cutoff} mm/s"
        )
    if (
        st.session_state.apply_magnetic
        and st.session_state.magnetic_declination is not None
    ):
        st.write(
            f"- Magnetic declination: {st.session_state.magnetic_declination:.3f}°"
        )

    st.divider()

    # Preview button
    col_btn, col_status = st.columns([1, 2])

    with col_btn:
        if st.button("🔍 Generate Preview", type="primary", key="preview_velocity"):
            # Reset staging processor from main processor's current state
            st.session_state.preview_velocity_proc = ProcessedDataset(proc.dataset)
            preview_proc = st.session_state.preview_velocity_proc

            try:
                runner = preview_proc.get_velocity_check_runner()

                # Apply magnetic correction first (if enabled)
                if (
                    st.session_state.apply_magnetic
                    and st.session_state.magnetic_declination is not None
                ):
                    runner.magnetic_correction(
                        declination=st.session_state.magnetic_declination
                    )

                # Apply threshold check (if enabled)
                if st.session_state.apply_threshold:
                    runner.threshold(
                        cutoff_u=float(st.session_state.cutoff_u),
                        cutoff_v=float(st.session_state.cutoff_v),
                        cutoff_w=float(st.session_state.cutoff_w),
                    )

                # Apply despike (if enabled)
                if st.session_state.apply_despike:
                    runner.despike(
                        kernel_size=st.session_state.despike_kernel,
                        cutoff=st.session_state.despike_cutoff,
                    )

                # Apply flatline detection (if enabled)
                if st.session_state.apply_flatline:
                    runner.flatline(
                        kernel_size=st.session_state.flatline_kernel,
                        cutoff=st.session_state.flatline_cutoff,
                    )

                # Commit to staging processor
                preview_proc.commit_runner(runner)

                # Store preview statistics
                st.session_state.velocity_preview_stats = runner.statistics
                st.session_state.velocity_preview_modifications = runner.modifications
                st.session_state.velocity_preview_run = True

                st.success("✓ Preview generated successfully!")

            except Exception as e:
                st.error(f"Error generating preview: {e}")

    with col_status:
        if st.session_state.velocity_preview_run:
            st.info("✓ Preview is available. See statistics and mask comparison below.")
        else:
            st.warning(
                "⚠️ No preview generated yet. Click 'Generate Preview' to see the impact."
            )

    # --- Preview Statistics ---
    if st.session_state.velocity_preview_run:
        st.divider()
        st.write("**📊 Preview Statistics:**")

        # Display QC check statistics
        if (
            hasattr(st.session_state, "velocity_preview_stats")
            and st.session_state.velocity_preview_stats
        ):
            stats_data = []
            for stat in st.session_state.velocity_preview_stats:
                threshold_str = str(stat.threshold) if stat.threshold else "N/A"
                stats_data.append(
                    {
                        "Check": stat.check_name,
                        "Threshold": threshold_str[:40],
                        "Pre-Masked (%)": f"{stat.pre_masked_pct:.2f}",
                        "Impact (%)": f"{stat.newly_masked_pct:.2f}",
                        "Cumulative (%)": f"{stat.total_masked_pct:.2f}",
                        "Valid (%)": f"{stat.valid_pct:.2f}",
                    }
                )
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

            # Summary metrics
            if stats_data:
                final_stat = st.session_state.velocity_preview_stats[-1]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Total Impact",
                        f"{final_stat.total_masked_pct - stat.pre_masked_pct:.2f}%",
                    )
                with col2:
                    st.metric("Final Valid", f"{final_stat.valid_pct:.2f}%")
                with col3:
                    st.metric("Cells Flagged", f"{final_stat.cells_total_masked:,}")

        # Display modifications (e.g., magnetic correction)
        if (
            hasattr(st.session_state, "velocity_preview_modifications")
            and st.session_state.velocity_preview_modifications
        ):
            st.write("**📝 Data Modifications:**")
            for mod in st.session_state.velocity_preview_modifications:
                dec = mod.metadata.get("declination_applied", "N/A")
                st.write(f"- {mod.operation}: declination = {dec}°")

        # --- Mask Preview ---
        st.divider()
        st.write("**🗺️ Mask Comparison:**")

        preview_mask = st.session_state.preview_velocity_proc.dataset["mask"].values
        original_mask = proc.dataset["mask"].values
        plot_mask_comparison(
            original_mask, preview_mask, "Before vs After Velocity Tests"
        )

        # Calculate and show mask difference statistics
        n_original_masked = (
            np.sum(original_mask[3, :, :] == 1)
            if original_mask.shape[0] > 3
            else np.sum(original_mask[0, :, :] == 1)
        )
        n_preview_masked = (
            np.sum(preview_mask[3, :, :] == 1)
            if preview_mask.shape[0] > 3
            else np.sum(preview_mask[0, :, :] == 1)
        )
        n_newly_masked = n_preview_masked - n_original_masked
        total_cells = original_mask[0, :, :].size

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Original Masked",
                f"{n_original_masked:,}",
                delta=f"{100*n_original_masked/total_cells:.2f}%",
            )
        with col2:
            st.metric(
                "Preview Masked",
                f"{n_preview_masked:,}",
                delta=f"{100*n_preview_masked/total_cells:.2f}%",
            )
        with col3:
            st.metric(
                "Newly Masked",
                f"{n_newly_masked:,}",
                delta=f"+{100*n_newly_masked/total_cells:.2f}%",
            )


# =============================================================================
# TAB 6: SAVE & RESET
# =============================================================================

with tab6:
    st.header("Save & Reset Data", divider="blue")

    col_save, col_reset = st.columns([1, 1])

    # --- Save Column ---
    with col_save:
        st.write("**💾 Apply Processing:**")

        st.write("""
        Apply the configured velocity tests to the main processor. 
        This will commit all changes permanently for this session.
        """)

        # Show current settings summary
        st.write("**Current Settings:**")
        settings_data = [
            [
                "Magnetic Correction",
                "True" if st.session_state.apply_magnetic else "False",
            ],
            [
                "Velocity Threshold",
                "True" if st.session_state.apply_threshold else "False",
            ],
            ["Despike Filter", "True" if st.session_state.apply_despike else "False"],
            [
                "Flatline Detection",
                "True" if st.session_state.apply_flatline else "False",
            ],
        ]
        settings_df = pd.DataFrame(settings_data, columns=["Test", "Enabled"])
        styled_settings = settings_df.style.map(status_color_map, subset=["Enabled"])
        st.write(styled_settings.to_html(), unsafe_allow_html=True)

        st.divider()

        if st.button("🌊 Apply Velocity Tests", type="primary", key="save_velocity"):
            try:
                proc.apply_velocity_check(
                    magnetic_correction=st.session_state.apply_magnetic,
                    declination=st.session_state.magnetic_declination if st.session_state.apply_magnetic else None,
                    lat=st.session_state.magnetic_lat if st.session_state.apply_magnetic else None,
                    lon=st.session_state.magnetic_lon if st.session_state.apply_magnetic else None,
                    year=float(st.session_state.magnetic_year) if st.session_state.apply_magnetic else None,
                    cutoff_u=float(st.session_state.cutoff_u) if st.session_state.apply_threshold else None,
                    cutoff_v=float(st.session_state.cutoff_v) if st.session_state.apply_threshold else None,
                    cutoff_w=float(st.session_state.cutoff_w) if st.session_state.apply_threshold else None,
                    despike=st.session_state.apply_despike,
                    despike_kernel=st.session_state.despike_kernel,
                    despike_cutoff=st.session_state.despike_cutoff,
                    flatline=st.session_state.apply_flatline,
                    flatline_kernel=st.session_state.flatline_kernel,
                    flatline_cutoff=st.session_state.flatline_cutoff,
                )

                st.session_state.velocity_applied = True

                # Reset staging processor to match main processor
                st.session_state.preview_velocity_proc = ProcessedDataset(proc.dataset)
                st.session_state.velocity_preview_run = False

                st.success("✅ Velocity tests applied successfully!")

                # Display final summary
                st.write("**📊 Processing Summary:**")
                summary_data = [
                    [
                        "Magnetic Correction",
                        "True" if st.session_state.apply_magnetic else "False",
                    ],
                    [
                        "Velocity Threshold",
                        "True" if st.session_state.apply_threshold else "False",
                    ],
                    [
                        "Despike Filter",
                        "True" if st.session_state.apply_despike else "False",
                    ],
                    [
                        "Flatline Detection",
                        "True" if st.session_state.apply_flatline else "False",
                    ],
                ]
                summary_df = pd.DataFrame(summary_data, columns=["Test", "Status"])
                styled_summary = summary_df.style.map(
                    status_color_map, subset=["Status"]
                )
                st.write(styled_summary.to_html(), unsafe_allow_html=True)

                # Show statistics
                report = proc.reports[-1] if proc.reports else None
                if report and report.checks:
                    st.write("---")
                    st.write("**📈 QC Test Statistics:**")
                    stats_data = []
                    for stat in report.checks:
                        threshold_str = str(stat.threshold) if stat.threshold else "N/A"
                        stats_data.append(
                            {
                                "Check": stat.check_name,
                                "Threshold": threshold_str[:30],
                                "Impact (%)": f"{stat.newly_masked_pct:.2f}",
                                "Cumulative (%)": f"{stat.total_masked_pct:.2f}",
                                "Valid (%)": f"{stat.valid_pct:.2f}",
                            }
                        )
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, hide_index=True, use_container_width=True)

                # Show overall processor statistics
                st.write("---")
                st.write("**📈 Overall Statistics:**")
                stats = proc.get_current_stats()
                st.write(f"- Total cells: `{stats['total_cells']:,}`")
                st.write(
                    f"- Masked cells: `{stats['masked']:,}` ({stats['masked_pct']:.2f}%)"
                )
                st.write(
                    f"- Valid cells: `{stats['valid']:,}` ({stats['valid_pct']:.2f}%)"
                )

            except Exception as e:
                st.error(f"❌ Error applying velocity tests: {e}")

        if not st.session_state.velocity_applied:
            st.warning("⚠️ Velocity tests not yet applied.")

    # --- Reset Column ---
    with col_reset:
        st.write("**🔄 Reset Processing:**")

        st.write("""
        Reset velocity test settings to their default values.
        This will clear the preview and any unsaved changes.
        """)

        st.divider()

        # Note: This only resets the velocity step, not the entire processor.
        # For a full reset, the main processor would need to be reset.
        st.button(
            "Reset Velocity Tests",
            key="reset_velocity",
            on_click=_reset_velocity_tests,
        )

        st.info("""
            ℹ️ Resetting will:
            - Clear velocity test settings
            - Clear the preview
            - Reset magnetic declination
            
            **Note:** To fully reset processing, 
            use the Reset on the QC Test page.
        """)


# =============================================================================
# SIDEBAR: PROCESSING STATUS
# =============================================================================

with st.sidebar:
    st.header("📊 Processing Status")

    stats = proc.get_current_stats()

    st.metric("Total Cells", f"{stats['total_cells']:,}")
    st.metric("Valid Cells", f"{stats['valid']:,}", delta=f"{stats['valid_pct']:.1f}%")
    st.metric(
        "Masked Cells", f"{stats['masked']:,}", delta=f"-{stats['masked_pct']:.1f}%"
    )

    st.write("---")

    st.write("**Velocity Tests Status:**")
    st.write(f"- Magnetic Corr: {'✅' if st.session_state.apply_magnetic else '❌'}")
    st.write(f"- Threshold: {'✅' if st.session_state.apply_threshold else '❌'}")
    st.write(f"- Despike: {'✅' if st.session_state.apply_despike else '❌'}")
    st.write(f"- Flatline: {'✅' if st.session_state.apply_flatline else '❌'}")

    if st.session_state.apply_threshold:
        st.write("---")
        st.write("**Threshold Values:**")
        st.write(f"- {u_label}: {st.session_state.cutoff_u} mm/s")
        st.write(f"- {v_label}: {st.session_state.cutoff_v} mm/s")
        st.write(f"- {w_label}: {st.session_state.cutoff_w} mm/s")

    if (
        st.session_state.apply_magnetic
        and st.session_state.magnetic_declination is not None
    ):
        st.write("---")
        st.write("**Magnetic Declination:**")
        st.write(f"- Declination: {st.session_state.magnetic_declination:.3f}°")

    st.write("---")
    st.write("**Processing Log:**")
    if hasattr(proc, "processing_log") and proc.processing_log:
        for log_entry in proc.processing_log[-5:]:
            st.write(f"- {log_entry}")
    else:
        st.write("*No processing steps applied yet.*")
