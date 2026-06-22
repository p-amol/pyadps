"""
05_Sensor_Health.py - Sensor Health Check Page (Refactored for pyadps v1.0.0)

This page allows users to:
1. View and replace sensor data (pressure/depth, salinity, temperature)
2. View heading, pitch, and roll sensor data
3. Apply threshold-based masking for roll and pitch
4. Apply sound speed correction to velocity data
5. Save processing results to the central ProcessedDataset

Architecture:
- Uses st.session_state.processor (ProcessedDataset) as central state manager
- Uses SensorHealthRunner for interactive preview and data replacement
- All processing is tracked through the processor's reports
"""

import tempfile
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# PAGE CONFIGURATION AND VALIDATION
# =============================================================================

st.set_page_config(page_title="Sensor Health", page_icon="🔧", layout="wide")

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


def get_time_axis():
    """Get time axis for plotting."""
    if "time" in ds.coords:
        return pd.to_datetime(ds["time"].values)
    elif "ensemble" in ds.coords:
        return ds["ensemble"].values
    return np.arange(get_total_ensembles())


def get_ensemble_axis():
    """Get ensemble axis for plotting."""
    if "rdi_ensemble" in ds.data_vars:
        return ds["rdi_ensemble"].values
    return np.arange(get_total_ensembles())


@st.cache_data
def read_csv_file(uploaded_file) -> Optional[np.ndarray]:
    """Read CSV file and return numpy array."""
    try:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
        df = pd.read_csv(path, header=None)
        return np.squeeze(df.to_numpy())
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


def lineplot(
    data: np.ndarray,
    title: str,
    slope: Optional[np.ndarray] = None,
    xaxis: str = "time",
    y_label: str = "",
) -> None:
    """Create a line plot with optional slope line."""
    if xaxis == "time":
        xdata = get_time_axis()
    else:
        xdata = get_ensemble_axis()

    fig = go.Figure()

    # Main data trace
    fig.add_trace(
        go.Scatter(
            x=xdata,
            y=data,
            mode="lines",
            name=title,
            line=dict(color="blue"),
        )
    )

    # Slope line if provided
    if slope is not None:
        fig.add_trace(
            go.Scatter(
                x=xdata,
                y=slope,
                mode="lines",
                name="Trend Line",
                line=dict(color="red", width=2, dash="dash"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time" if xaxis == "time" else "Ensemble",
        yaxis_title=y_label,
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def compute_circular_mean(data_degrees: np.ndarray) -> float:
    """Compute circular mean for angular data."""
    data_rad = np.radians(data_degrees)
    mean_x = np.nanmean(np.cos(data_rad))
    mean_y = np.nanmean(np.sin(data_rad))
    mean_rad = np.arctan2(mean_y, mean_x)
    return np.degrees(mean_rad)


def compute_drift_analysis(
    data: np.ndarray, ensemble_axis: np.ndarray, std_cutoff: float = 3.0
) -> Tuple[float, float, float, np.ndarray]:
    """
    Compute drift analysis for sensor data.

    Returns: (median, change, slope, fitted_line)
    """
    # Remove outliers
    median_val = float(np.nanmedian(data))
    std = np.nanstd(data)
    mask = np.abs(data - median_val) <= std_cutoff * std
    clean_data = data.copy()
    clean_data[~mask] = np.nan

    # Get valid data for polyfit
    valid_mask = ~np.isnan(clean_data)
    if np.sum(valid_mask) < 2:
        return median_val, 0.0, 0.0, np.full_like(data, median_val, dtype=np.float64)

    x_valid = ensemble_axis[valid_mask]
    y_valid = clean_data[valid_mask]

    slope, intercept = np.polyfit(x_valid, y_valid, 1)
    fitted_line = slope * ensemble_axis + intercept
    change = float(fitted_line[-1] - fitted_line[0])

    return median_val, change, float(slope), fitted_line


def get_scale_factor(var_name: str) -> float:
    """Get scale factor for a variable from dataset attributes."""
    if var_name in ds.data_vars:
        return ds[var_name].attrs.get("scale_factor", 1.0)
    return 1.0


def get_sensor_info(sensor_name: str) -> str:
    """Get sensor availability info from fixed leader."""
    try:
        sensor_info = ds.fixed_leader.sensor_info(ens=0, field="avail")
        return "Available" if sensor_info.get(sensor_name, False) else "Not Available"
    except Exception:
        return "Unknown"


def status_color_map(value: object) -> str:
    """Map status values to colors for dataframe styling."""
    if value == "True":
        return "background-color: green; color: white"
    elif value == "False":
        return "background-color: red; color: white"
    return ""


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

# Initialize sensor health session state variables
if "sensor_health_initialized" not in st.session_state:
    st.session_state.sensor_health_initialized = True
    st.session_state.sensor_health_applied = False

    # Data replacement tracking
    st.session_state.depth_modified = False
    st.session_state.salinity_modified = False
    st.session_state.temperature_modified = False

    # Threshold settings
    st.session_state.roll_threshold = 15.0
    st.session_state.pitch_threshold = 15.0

    # Check selections
    st.session_state.apply_roll_check = False
    st.session_state.apply_pitch_check = False
    st.session_state.apply_sound_speed_correction = False
    st.session_state.correct_velocity = True
    st.session_state.horizontal_only = True

    # Temporary data storage for replacement
    st.session_state.temp_depth_data = None
    st.session_state.temp_salinity_data = None
    st.session_state.temp_temperature_data = None

# =============================================================================
# CALLBACKS
# Mutating state in on_click callbacks (rather than calling st.rerun() inside
# an `if st.button():` block) avoids resetting the active tab back to the
# first one - a known Streamlit limitation made worse when conditional
# content sits before st.tabs(). See: github.com/streamlit/streamlit/issues/6257
# =============================================================================


def _reset_sensor_health_all():
    proc.reset()
    st.session_state.sensor_health_applied = False
    st.session_state.depth_modified = False
    st.session_state.salinity_modified = False
    st.session_state.temperature_modified = False
    st.session_state.temp_depth_data = None
    st.session_state.temp_salinity_data = None
    st.session_state.temp_temperature_data = None


def _reset_depth():
    st.session_state.temp_depth_data = None
    st.session_state.depth_modified = False


def _reset_salinity():
    st.session_state.temp_salinity_data = None
    st.session_state.salinity_modified = False


def _reset_temperature():
    st.session_state.temp_temperature_data = None
    st.session_state.temperature_modified = False


def _reset_sensor_health_full():
    proc.reset()
    st.session_state.sensor_health_applied = False
    st.session_state.depth_modified = False
    st.session_state.salinity_modified = False
    st.session_state.temperature_modified = False
    st.session_state.temp_depth_data = None
    st.session_state.temp_salinity_data = None
    st.session_state.temp_temperature_data = None
    st.session_state.apply_roll_check = False
    st.session_state.apply_pitch_check = False
    st.session_state.apply_sound_speed_correction = False


# =============================================================================
# PAGE HEADER
# =============================================================================

st.header("🔧 Sensor Health Check", divider="blue")
st.write(
    """
    Verify and correct environmental sensor data. This page allows you to:
    - Inspect pressure (depth), salinity, and temperature sensors
    - Replace sensor data with external measurements (e.g., CTD data)
    - Apply tilt sensor (roll/pitch) threshold checks
    - Correct velocity data using updated sound speed calculations
    """
)

# Show current processing status.
# Wrapped in an always-drawn container: an *unconditional* container before
# st.tabs() doesn't break tab state, but conditional content directly in the
# main body (appearing/disappearing across reruns) does.
status_container = st.container()
with status_container:
    if st.session_state.sensor_health_applied:
        st.success("✅ Sensor health checks have been applied to this dataset.")
        st.button(
            "🔄 Reset Sensor Health",
            type="secondary",
            key="reset_sensor_health_top",
            on_click=_reset_sensor_health_all,
        )

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "🌊 Pressure",
        "🧂 Salinity",
        "🌡️ Temperature",
        "🧭 Heading",
        "📐 Pitch",
        "🔄 Roll",
        "⚙️ Apply Checks",
        "💾 Save/Reset",
    ]
)

# =============================================================================
# TAB 1: PRESSURE SENSOR CHECK
# =============================================================================

with tab1:
    st.subheader("Pressure Sensor Check", divider="orange")
    st.write(
        """
        Verify pressure sensor (depth of transducer) data for drift or malfunction.
        The actual deployment depth can be cross-checked using the mooring diagram.
        """
    )

    total_ensembles = get_total_ensembles()
    ensemble_axis = get_ensemble_axis()

    # Get depth data
    if "transducer_depth" in ds.data_vars:
        depth_var = ds["transducer_depth"]
        scale = get_scale_factor("transducer_depth")
        st.write(scale)
        depth_data = depth_var.values * scale * 0.1  # Convert to meters
    else:
        st.warning("Depth of transducer data not found in dataset.")
        depth_data = np.zeros(total_ensembles)

    # Layout
    col_info, col_plot = st.columns([1, 2])

    with col_plot:
        std_cutoff = st.number_input(
            "Standard Deviation Cutoff",
            min_value=0.01,
            max_value=10.0,
            value=3.0,
            step=0.1,
            key="depth_std_cutoff",
        )

        xaxis_option = st.radio(
            "X-axis",
            ["time", "ensemble"],
            horizontal=True,
            key="depth_xaxis",
        )

    # Compute drift analysis
    median_depth, depth_change, _, fitted_line = compute_drift_analysis(
        depth_data, ensemble_axis, std_cutoff
    )

    with col_info:
        st.write("**📊 Sensor Information:**")
        st.write(f"- Depth Sensor: `{get_sensor_info('Depth Sensor')}`")
        st.write(f"- Total ensembles: `{total_ensembles}`")
        st.write(f"- Median depth: `{median_depth:.2f} m`")
        st.write(f"- Change in depth: `{depth_change:.3f} m`")
        st.write(f"- Depth Modified: `{st.session_state.depth_modified}`")

    # Plot
    lineplot(
        depth_data,
        "Depth of Transducer",
        slope=fitted_line,
        xaxis=xaxis_option if xaxis_option else "time",
        y_label="Depth (m)",
    )

    # Data replacement section
    st.info(
        """
        ℹ️ If the pressure sensor is malfunctioning, upload a corrected CSV file 
        or enter a fixed depth value. The CSV file should contain a single column
        without header, with one value per ensemble.
        """,
        icon="ℹ️",
    )

    depth_method = st.radio(
        "Correction Method",
        ["File Upload", "Fixed Value"],
        horizontal=True,
        key="depth_method",
    )

    if depth_method == "Fixed Value":
        fixed_depth = st.number_input(
            "Enter corrected depth (m):",
            min_value=0.0,
            value=None,
            placeholder="Type a number...",
            key="fixed_depth_input",
        )

        if st.button("Apply Fixed Depth", key="apply_fixed_depth"):
            if fixed_depth is not None:
                # Store the replacement data (in dataset units, typically decimeters)
                scale = get_scale_factor("transducer_depth")
                st.session_state.temp_depth_data = np.full(
                    total_ensembles, fixed_depth / scale if scale else fixed_depth
                )
                st.session_state.depth_modified = True
                st.success(f"✅ Depth will be set to {fixed_depth} m when saved.")
            else:
                st.warning("Please enter a depth value.")

    else:  # File Upload
        uploaded_file = st.file_uploader(
            "Upload Corrected Depth File (CSV)",
            type="csv",
            key="depth_file_upload",
        )

        if uploaded_file is not None:
            if st.button("Check & Apply Depth", key="check_depth_file"):
                data = read_csv_file(uploaded_file)
                if data is not None:
                    if len(data) != total_ensembles:
                        st.error(
                            f"❌ Ensemble count mismatch! "
                            f"File has {len(data)} values, expected {total_ensembles}."
                        )
                    else:
                        # Store the replacement data (assuming file is in meters)
                        scale = get_scale_factor("transducer_depth")
                        st.session_state.temp_depth_data = (
                            data / scale if scale else data
                        )
                        st.session_state.depth_modified = True
                        st.success("✅ Depth data will be applied when saved.")

                        # Show preview
                        lineplot(data, "Preview: Modified Depth", y_label="Depth (m)")

    if st.session_state.depth_modified:
        st.button("Reset Depth to Original", key="reset_depth", on_click=_reset_depth)

# =============================================================================
# TAB 2: SALINITY SENSOR CHECK
# =============================================================================

with tab2:
    st.subheader("Salinity Sensor Check", divider="orange")
    st.write(
        """
        Verify salinity sensor data. If a salinity sensor is unavailable or
        malfunctioning, use a constant value based on deployment location.
        """
    )

    total_ensembles = get_total_ensembles()
    ensemble_axis = get_ensemble_axis()

    # Get salinity data
    if "salinity" in ds.data_vars:
        salinity_var = ds["salinity"]
        scale = get_scale_factor("salinity")
        salinity_data = salinity_var.values * scale
    else:
        st.warning("Salinity data not found in dataset.")
        salinity_data = np.zeros(total_ensembles)

    # Layout
    col_info, col_plot = st.columns([1, 2])

    with col_plot:
        std_cutoff = st.number_input(
            "Standard Deviation Cutoff",
            min_value=0.01,
            max_value=10.0,
            value=3.0,
            step=0.1,
            key="salinity_std_cutoff",
        )

        xaxis_option = st.radio(
            "X-axis",
            ["time", "ensemble"],
            horizontal=True,
            key="salinity_xaxis",
        )

    # Compute drift analysis
    median_salinity, salinity_change, _, fitted_line = compute_drift_analysis(
        salinity_data, ensemble_axis, std_cutoff
    )

    with col_info:
        st.write("**📊 Sensor Information:**")
        st.write(f"- Conductivity Sensor: `{get_sensor_info('Conductivity Sensor')}`")
        st.write(f"- Total ensembles: `{total_ensembles}`")
        st.write(f"- Median salinity: `{median_salinity:.2f} PSU`")
        st.write(f"- Change in salinity: `{salinity_change:.3f} PSU`")
        st.write(f"- Salinity Modified: `{st.session_state.salinity_modified}`")

    # Plot
    lineplot(
        salinity_data,
        "Salinity",
        slope=fitted_line,
        xaxis=xaxis_option if xaxis_option else "time",
        y_label="Salinity (PSU)",
    )

    # Data replacement section
    st.info(
        """
        ℹ️ If salinity values need correction, upload a CSV file or enter a fixed value.
        These values will be used for sound speed correction.
        """,
        icon="ℹ️",
    )

    salinity_method = st.radio(
        "Correction Method",
        ["Fixed Value", "File Upload"],
        horizontal=True,
        key="salinity_method",
    )

    if salinity_method == "Fixed Value":
        fixed_salinity = st.number_input(
            "Enter corrected salinity (PSU):",
            min_value=0.0,
            value=None,
            placeholder="Type a number...",
            key="fixed_salinity_input",
        )

        if st.button("Apply Fixed Salinity", key="apply_fixed_salinity"):
            if fixed_salinity is not None:
                st.session_state.temp_salinity_data = np.full(
                    total_ensembles, fixed_salinity
                )
                st.session_state.salinity_modified = True
                st.success(
                    f"✅ Salinity will be set to {fixed_salinity} PSU when saved."
                )
            else:
                st.warning("Please enter a salinity value.")

    else:  # File Upload
        uploaded_file = st.file_uploader(
            "Upload Corrected Salinity File (CSV)",
            type="csv",
            key="salinity_file_upload",
        )

        if uploaded_file is not None:
            if st.button("Check & Apply Salinity", key="check_salinity_file"):
                data = read_csv_file(uploaded_file)
                if data is not None:
                    if len(data) != total_ensembles:
                        st.error(
                            f"❌ Ensemble count mismatch! "
                            f"File has {len(data)} values, expected {total_ensembles}."
                        )
                    else:
                        st.session_state.temp_salinity_data = data
                        st.session_state.salinity_modified = True
                        st.success("✅ Salinity data will be applied when saved.")

                        # Show preview
                        lineplot(
                            data, "Preview: Modified Salinity", y_label="Salinity (PSU)"
                        )

    if st.session_state.salinity_modified:
        st.button(
            "Reset Salinity to Original", key="reset_salinity", on_click=_reset_salinity
        )

# =============================================================================
# TAB 3: TEMPERATURE SENSOR CHECK
# =============================================================================

with tab3:
    st.subheader("Temperature Sensor Check", divider="orange")
    st.write(
        """
        Verify temperature sensor data for drift or malfunction.
        Temperature affects sound speed calculations and velocity accuracy.
        """
    )

    total_ensembles = get_total_ensembles()
    ensemble_axis = get_ensemble_axis()

    # Get temperature data
    if "temperature" in ds.data_vars:
        temp_var = ds["temperature"]
        scale = get_scale_factor("temperature")
        temp_data = temp_var.values * scale  # Convert to degrees C
    else:
        st.warning("Temperature data not found in dataset.")
        temp_data = np.zeros(total_ensembles)

    # Layout
    col_info, col_plot = st.columns([1, 2])

    with col_plot:
        std_cutoff = st.number_input(
            "Standard Deviation Cutoff",
            min_value=0.01,
            max_value=10.0,
            value=3.0,
            step=0.1,
            key="temp_std_cutoff",
        )

        xaxis_option = st.radio(
            "X-axis",
            ["time", "ensemble"],
            horizontal=True,
            key="temp_xaxis",
        )

    # Compute drift analysis
    median_temp, temp_change, _, fitted_line = compute_drift_analysis(
        temp_data, ensemble_axis, std_cutoff
    )

    with col_info:
        st.write("**📊 Sensor Information:**")
        st.write(f"- Temperature Sensor: `{get_sensor_info('Temperature Sensor')}`")
        st.write(f"- Total ensembles: `{total_ensembles}`")
        st.write(f"- Median temperature: `{median_temp:.2f} °C`")
        st.write(f"- Change in temperature: `{temp_change:.3f} °C`")
        st.write(f"- Temperature Modified: `{st.session_state.temperature_modified}`")

    # Plot
    lineplot(
        temp_data,
        "Temperature",
        slope=fitted_line,
        xaxis=xaxis_option if xaxis_option else "time",
        y_label="Temperature (°C)",
    )

    # Data replacement section
    st.info(
        """
        ℹ️ If temperature values need correction (e.g., from CTD data),
        upload a CSV file or enter a fixed value.
        """,
        icon="ℹ️",
    )

    temp_method = st.radio(
        "Correction Method",
        ["File Upload", "Fixed Value"],
        horizontal=True,
        key="temp_method",
    )

    if temp_method == "Fixed Value":
        fixed_temp = st.number_input(
            "Enter corrected temperature (°C):",
            value=None,
            placeholder="Type a number...",
            key="fixed_temp_input",
        )

        if st.button("Apply Fixed Temperature", key="apply_fixed_temp"):
            if fixed_temp is not None:
                st.session_state.temp_temperature_data = np.full(
                    total_ensembles, fixed_temp
                )
                st.session_state.temperature_modified = True
                st.success(f"✅ Temperature will be set to {fixed_temp} °C when saved.")
            else:
                st.warning("Please enter a temperature value.")

    else:  # File Upload
        uploaded_file = st.file_uploader(
            "Upload Corrected Temperature File (CSV)",
            type="csv",
            key="temp_file_upload",
        )

        if uploaded_file is not None:
            if st.button("Check & Apply Temperature", key="check_temp_file"):
                data = read_csv_file(uploaded_file)
                if data is not None:
                    if len(data) != total_ensembles:
                        st.error(
                            f"❌ Ensemble count mismatch! "
                            f"File has {len(data)} values, expected {total_ensembles}."
                        )
                    else:
                        st.session_state.temp_temperature_data = data
                        st.session_state.temperature_modified = True
                        st.success("✅ Temperature data will be applied when saved.")

                        # Show preview
                        lineplot(
                            data,
                            "Preview: Modified Temperature",
                            y_label="Temperature (°C)",
                        )

    if st.session_state.temperature_modified:
        st.button(
            "Reset Temperature to Original",
            key="reset_temp",
            on_click=_reset_temperature,
        )

# =============================================================================
# TAB 4: HEADING SENSOR CHECK
# =============================================================================

with tab4:
    st.subheader("Heading Sensor Check", divider="orange")

    st.warning(
        """
        ⚠️ **Note:** Heading sensor corrections (magnetic declination) are applied
        in the Velocity Test page. This tab is for viewing heading data only.
        """,
        icon="⚠️",
    )

    # Get heading data
    if "heading" in ds.data_vars:
        heading_var = ds["heading"]
        scale = get_scale_factor("heading")
        heading_data = heading_var.values * scale  # Convert to degrees
    else:
        st.warning("Heading data not found in dataset.")
        heading_data = np.zeros(get_total_ensembles())

    # Compute circular mean
    heading_mean = compute_circular_mean(heading_data)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**📊 Statistics:**")
        st.write(f"- Mean heading: `{heading_mean:.2f}°`")
        st.write(f"- Min heading: `{np.nanmin(heading_data):.2f}°`")
        st.write(f"- Max heading: `{np.nanmax(heading_data):.2f}°`")

    with col2:
        xaxis_option = st.radio(
            "X-axis",
            ["time", "ensemble"],
            horizontal=True,
            key="heading_xaxis",
        )

    lineplot(
        heading_data,
        "Heading",
        xaxis=xaxis_option if xaxis_option else "time",
        y_label="Heading (°)",
    )

# =============================================================================
# TAB 5: PITCH SENSOR CHECK
# =============================================================================

with tab5:
    st.subheader("Pitch Sensor Check", divider="orange")

    st.write(
        """
        View pitch sensor data. Excessive pitch indicates instrument tilting
        which can affect measurement accuracy.
        """
    )

    # Get pitch data
    if "pitch" in ds.data_vars:
        pitch_var = ds["pitch"]
        scale = get_scale_factor("pitch")
        pitch_data = pitch_var.values * scale  # Convert to degrees
    else:
        st.warning("Pitch data not found in dataset.")
        pitch_data = np.zeros(get_total_ensembles())

    # Compute circular mean
    pitch_mean = compute_circular_mean(pitch_data)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**📊 Statistics:**")
        st.write(f"- Mean pitch: `{pitch_mean:.2f}°`")
        st.write(f"- Min pitch: `{np.nanmin(pitch_data):.2f}°`")
        st.write(f"- Max pitch: `{np.nanmax(pitch_data):.2f}°`")

    with col2:
        xaxis_option = st.radio(
            "X-axis",
            ["time", "ensemble"],
            horizontal=True,
            key="pitch_xaxis",
        )

    lineplot(
        pitch_data,
        "Pitch",
        xaxis=xaxis_option if xaxis_option else "time",
        y_label="Pitch (°)",
    )

    # Show threshold line
    st.write(f"Current pitch threshold: `{st.session_state.pitch_threshold}°`")

# =============================================================================
# TAB 6: ROLL SENSOR CHECK
# =============================================================================

with tab6:
    st.subheader("Roll Sensor Check", divider="orange")

    st.write(
        """
        View roll sensor data. Excessive roll indicates instrument tilting
        which can affect measurement accuracy.
        """
    )

    # Get roll data
    if "roll" in ds.data_vars:
        roll_var = ds["roll"]
        scale = get_scale_factor("roll")
        roll_data = roll_var.values * scale  # Convert to degrees
    else:
        st.warning("Roll data not found in dataset.")
        roll_data = np.zeros(get_total_ensembles())

    # Compute circular mean
    roll_mean = compute_circular_mean(roll_data)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**📊 Statistics:**")
        st.write(f"- Mean roll: `{roll_mean:.2f}°`")
        st.write(f"- Min roll: `{np.nanmin(roll_data):.2f}°`")
        st.write(f"- Max roll: `{np.nanmax(roll_data):.2f}°`")

    with col2:
        xaxis_option = st.radio(
            "X-axis",
            ["time", "ensemble"],
            horizontal=True,
            key="roll_xaxis",
        )

    lineplot(
        roll_data,
        "Roll",
        xaxis=xaxis_option if xaxis_option else "time",
        y_label="Roll (°)",
    )

    # Show threshold line
    st.write(f"Current roll threshold: `{st.session_state.roll_threshold}°`")

# =============================================================================
# TAB 7: APPLY CHECKS
# =============================================================================

with tab7:
    st.subheader("Apply Sensor Thresholds & Corrections", divider="orange")

    st.write(
        """
        Configure and preview sensor health checks before applying them.
        """
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("**🎚️ Threshold Settings:**")

        st.session_state.roll_threshold = st.number_input(
            "Roll threshold (°)",
            min_value=0.0,
            max_value=90.0,
            value=st.session_state.roll_threshold,
            step=1.0,
            key="roll_threshold_input",
        )

        st.session_state.pitch_threshold = st.number_input(
            "Pitch threshold (°)",
            min_value=0.0,
            max_value=90.0,
            value=st.session_state.pitch_threshold,
            step=1.0,
            key="pitch_threshold_input",
        )

    with col2:
        st.write("**☑️ Select Checks to Apply:**")

        st.session_state.apply_roll_check = st.checkbox(
            "Apply Roll Threshold Check",
            value=st.session_state.apply_roll_check,
            key="roll_check_cb",
        )

        st.session_state.apply_pitch_check = st.checkbox(
            "Apply Pitch Threshold Check",
            value=st.session_state.apply_pitch_check,
            key="pitch_check_cb",
        )

        # Sound speed correction requires modified T or S
        sound_speed_enabled = (
            st.session_state.temperature_modified or st.session_state.salinity_modified
        )

        if sound_speed_enabled:
            st.session_state.apply_sound_speed_correction = st.checkbox(
                "Apply Sound Speed Correction",
                value=st.session_state.apply_sound_speed_correction,
                key="sound_speed_cb",
            )

            # Additional sound speed correction options
            if st.session_state.apply_sound_speed_correction:
                st.write("**Sound Speed Correction Options:**")

                st.session_state.correct_velocity = st.checkbox(
                    "Correct velocity using sound speed ratio",
                    value=st.session_state.get("correct_velocity", True),
                    key="correct_velocity_cb",
                    help="Apply sound speed correction to velocity data",
                )

                st.session_state.horizontal_only = st.checkbox(
                    "Correct horizontal velocities only (U, V)",
                    value=st.session_state.get("horizontal_only", True),
                    key="horizontal_only_cb",
                    help="If checked, only U and V components are corrected. If unchecked, W is also corrected.",
                    disabled=not st.session_state.correct_velocity,
                )
        else:
            st.session_state.apply_sound_speed_correction = False
            st.info(
                "ℹ️ Sound speed correction requires modified temperature or salinity."
            )

    # Preview section
    st.write("---")
    st.write("**📋 Preview of Changes:**")

    preview_items = []

    if st.session_state.depth_modified:
        preview_items.append(f"• Depth data will be replaced")

    if st.session_state.salinity_modified:
        preview_items.append(f"• Salinity data will be replaced")

    if st.session_state.temperature_modified:
        preview_items.append(f"• Temperature data will be replaced")

    if st.session_state.apply_roll_check:
        preview_items.append(
            f"• Roll check: mask ensembles with |roll| > {st.session_state.roll_threshold}°"
        )

    if st.session_state.apply_pitch_check:
        preview_items.append(
            f"• Pitch check: mask ensembles with |pitch| > {st.session_state.pitch_threshold}°"
        )

    if st.session_state.apply_sound_speed_correction:
        correct_vel = st.session_state.get("correct_velocity", True)
        horiz_only = st.session_state.get("horizontal_only", True)
        if correct_vel:
            vel_desc = "U, V only" if horiz_only else "U, V, W"
            preview_items.append(
                f"• Sound speed correction will be applied to velocity ({vel_desc})"
            )
        else:
            preview_items.append(
                "• Sound speed will be recalculated (velocity not corrected)"
            )

    if preview_items:
        for item in preview_items:
            st.write(item)
    else:
        st.write("*No changes configured.*")

# =============================================================================
# TAB 8: SAVE/RESET
# =============================================================================

with tab8:
    st.subheader("Save or Reset Processing", divider="blue")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write("**💾 Save Processing:**")

        if st.button(
            "🔧 Apply Sensor Health Checks", type="primary", key="save_button"
        ):
            try:
                proc.apply_sensor_health(
                    roll=st.session_state.apply_roll_check,
                    roll_threshold=st.session_state.roll_threshold,
                    pitch=st.session_state.apply_pitch_check,
                    pitch_threshold=st.session_state.pitch_threshold,
                    correct_sound_speed=st.session_state.apply_sound_speed_correction,
                    correct_velocity=st.session_state.get("correct_velocity", True),
                    horizontal_only=st.session_state.get("horizontal_only", True),
                    temperature=st.session_state.temp_temperature_data if st.session_state.temperature_modified else None,
                    salinity=st.session_state.temp_salinity_data if st.session_state.salinity_modified else None,
                    transducer_depth=st.session_state.temp_depth_data if st.session_state.depth_modified else None,
                )

                st.session_state.sensor_health_applied = True
                st.success("✅ Sensor health checks applied successfully!")

                # Display summary table
                st.write("**📊 Processing Summary:**")

                summary_data = [
                    [
                        "Depth Modified",
                        "True" if st.session_state.depth_modified else "False",
                    ],
                    [
                        "Salinity Modified",
                        "True" if st.session_state.salinity_modified else "False",
                    ],
                    [
                        "Temperature Modified",
                        "True" if st.session_state.temperature_modified else "False",
                    ],
                    [
                        "Roll Check",
                        "True" if st.session_state.apply_roll_check else "False",
                    ],
                    [
                        "Pitch Check",
                        "True" if st.session_state.apply_pitch_check else "False",
                    ],
                    [
                        "Sound Speed Correction",
                        "True"
                        if st.session_state.apply_sound_speed_correction
                        else "False",
                    ],
                ]

                summary_df = pd.DataFrame(summary_data, columns=["Test", "Status"])
                styled_df = summary_df.style.map(status_color_map, subset=["Status"])
                st.write(styled_df.to_html(), unsafe_allow_html=True)

                # Show statistics from runner
                st.write("---")
                st.write("**📈 Processing Statistics:**")

                stats = proc.get_current_stats()
                st.write(f"- Total cells: `{stats['total_cells']:,}`")
                st.write(
                    f"- Masked cells: `{stats['masked']:,}` ({stats['masked_pct']:.2f}%)"
                )
                st.write(
                    f"- Valid cells: `{stats['valid']:,}` ({stats['valid_pct']:.2f}%)"
                )

            except Exception as e:
                st.error(f"❌ Error applying sensor health checks: {e}")

        if not st.session_state.sensor_health_applied:
            st.warning("⚠️ Sensor health checks not yet applied.")

    with col2:
        st.write("**🔄 Reset Processing:**")

        st.button(
            "Reset Sensor Health",
            key="reset_all_button",
            on_click=_reset_sensor_health_full,
        )

        st.info(
            """
            ℹ️ Resetting will:
            - Restore all sensor data to original values
            - Clear all applied masks
            - Reset the processor to its initial state
            
            **Note:** This will also reset any subsequent processing steps.
            """,
            icon="ℹ️",
        )

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

    st.write("**Current Modifications:**")
    st.write(f"- Depth: {'✅' if st.session_state.depth_modified else '❌'}")
    st.write(f"- Salinity: {'✅' if st.session_state.salinity_modified else '❌'}")
    st.write(
        f"- Temperature: {'✅' if st.session_state.temperature_modified else '❌'}"
    )

    st.write("---")

    st.write("**Processing Log:**")
    if proc.processing_log:
        for log_entry in proc.processing_log[-5:]:  # Show last 5 entries
            st.write(f"- {log_entry}")
    else:
        st.write("*No processing steps applied yet.*")
