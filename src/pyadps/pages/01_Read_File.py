"""
Streamlit Page 01: Read ADCP File

This page loads ADCP binary files and displays comprehensive summary statistics
using the pyadps v1.0.0 API with ProcessedDataset as the central state manager.

Features:
- File upload and parsing
- File header health check
- Fixed Leader summary statistics
- Variable Leader summary statistics
- Time axis diagnostics and correction
- ProcessedDataset initialization for downstream processing
"""

import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# pyadps v1.0.0 imports
import pyadps
from pyadps.processing import ProcessedDataset

# Set page configuration
st.set_page_config(layout="wide", page_title="ADCP Data Analysis Tool")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def color_bool(val: bool) -> str:
    """Returns CSS color formatting based on boolean value."""
    if isinstance(val, bool):
        color = "green" if val else "red"
    else:
        color = "orange"
    return f"color: {color}"


def color_status(val: Any) -> str:
    """Returns CSS color formatting based on status string."""
    val_lower = str(val).lower()
    if val_lower in ("true", "pass", "healthy", "yes"):
        color = "green"
    elif val_lower in ("false", "fail", "error", "no"):
        color = "red"
    else:
        color = "orange"
    return f"color: {color}"


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================


def initialize_minimal_session_state():
    """
    Initialize minimal session state for ProcessedDataset-based workflow.

    This replaces the old 50+ variable initialization with a clean,
    minimal state focused on the ProcessedDataset orchestrator.
    """
    # Core data
    if "processor" not in st.session_state:
        st.session_state.processor = None

    if "ds" not in st.session_state:
        st.session_state.ds = None

    if "ds_header" not in st.session_state:
        st.session_state.ds_header = None

    if "fname" not in st.session_state:
        st.session_state.fname = "No file selected"

    if "fpath" not in st.session_state:
        st.session_state.fpath = None

    # Processing state tracking
    if "processing_step" not in st.session_state:
        st.session_state.processing_step = 0  # 0=not started, 1-5 for each step

    # UI preferences (minimal)
    if "ui_params" not in st.session_state:
        st.session_state.ui_params = {
            "file_prefix": "",
            "axis_option": "time",
            "attributes": {},
        }

    # Time axis state
    if "time_axis_modified" not in st.session_state:
        st.session_state.time_axis_modified = False


# =============================================================================
# FILE READING
# =============================================================================


@st.cache_data(show_spinner="Reading ADCP file...")
def read_adcp_file(file_content: bytes, filename: str):
    """
    Read ADCP binary file and return xarray Dataset.

    Parameters
    ----------
    file_content : bytes
        Binary content of the uploaded file
    filename : str
        Original filename

    Returns
    -------
    xr.Dataset
        Parsed ADCP dataset
    """
    # Create temporary file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)

    with open(temp_path, "wb") as f:
        f.write(file_content)

    # Read using pyadps v1.0.0
    ds = pyadps.read(temp_path)
    ds_head = pyadps.read_header(temp_path)

    return ds, ds_head, temp_path


# =============================================================================
# DISPLAY FUNCTIONS - FILE HEADER
# =============================================================================


def display_file_header(ds_header) -> None:
    """Display file header information with health checks."""
    st.header("File Header", divider="blue")

    st.write("""
    The file header contains metadata about the ADCP binary file structure.
    Use the health check to verify file integrity before processing.
    """)

    col1, col2 = st.columns(2)

    # Both checks take no input and don't mutate state, so they're computed
    # eagerly rather than gated behind a button (consistent with the other
    # tabs on this page).
    with col1:
        st.subheader("File Health Check")
        try:
            check = ds_header.header.check_file()

            # Determine overall health
            critical_pass = (
                check.get("File Size Match", False)
                and check.get("Byte Uniformity", False)
                and check.get("Data Type Uniformity", False)
            )

            if critical_pass:
                st.success("File appears healthy!")
            else:
                st.error("File may be corrupted!")

            # Display check results
            st.write(
                f"**Total Ensembles:** {ds_header.attrs.get('total_ensembles', 'N/A')}"
            )
            st.write(
                f"**File Size:** {format_file_size(check.get('System File Size (B)', 0))}"
            )

            # Create summary table
            check_items = [
                ("File Size Match", check.get("File Size Match", False)),
                ("Byte Uniformity", check.get("Byte Uniformity", False)),
                ("Data Type Uniformity", check.get("Data Type Uniformity", False)),
                ("Byte Skip Uniformity", check.get("Byte Skip Uniformity", False)),
                (
                    "Address Offset Uniformity",
                    check.get("Address Offset Uniformity", False),
                ),
                ("Data ID Uniformity", check.get("Data ID Uniformity", False)),
            ]

            df = pd.DataFrame(check_items, columns=["Check", "Status"])
            df["Status"] = df["Status"].map({True: "PASS", False: "FAIL"})
            st.dataframe(
                df.style.map(color_status, subset=["Status"]),
                use_container_width=True,
                hide_index=True,
            )

        except Exception as e:
            st.error(f"Error checking file: {e}")

    with col2:
        st.subheader("Available Data Types")
        try:
            data_types = ds_header.header.get_available_data_types(0)
            for dt in data_types:
                st.write(f"  - {dt}")
        except Exception as e:
            st.error(f"Error getting data types: {e}")


# =============================================================================
# DISPLAY FUNCTIONS - FIXED LEADER
# =============================================================================


def display_fixed_leader_summary(ds) -> None:
    """Display comprehensive Fixed Leader summary statistics."""
    st.header("Fixed Leader (Static Configuration)", divider="blue")

    st.write("""
    Fixed Leader data contains static ADCP configuration that remains constant
    throughout the deployment. This includes hardware settings, thresholds,
    and coordinate transformation parameters.
    """)

    # Create tabs for different Fixed Leader information
    fl_tabs = st.tabs(
        [
            "System Configuration",
            "Sensor Information",
            "Coordinate Transform",
            "Deployment Thresholds",
            "Uniformity Check",
            "Raw Fields",
        ]
    )

    # Tab 1: System Configuration
    with fl_tabs[0]:
        st.subheader("System Configuration")
        try:
            sys_config = ds.fixed_leader.system_configuration(ens=-1)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Frequency", sys_config.get("Frequency", "N/A"))
                st.metric("Beam Angle", f"{sys_config.get('Beam Angle', 'N/A')} deg")
                st.metric("Beam Direction", sys_config.get("Beam Direction", "N/A"))

            with col2:
                st.metric("Beam Pattern", sys_config.get("Beam Pattern", "N/A"))
                st.metric(
                    "Janus Configuration", sys_config.get("Janus Configuration", "N/A")
                )
                st.metric("XDCR HD", sys_config.get("XDCR HD", "N/A"))

            # Full table
            with st.expander("View All System Configuration"):
                df = pd.DataFrame(
                    list(sys_config.items()), columns=["Parameter", "Value"]
                )
                st.dataframe(df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error reading system configuration: {e}")

    # Tab 2: Sensor Information
    with fl_tabs[1]:
        st.subheader("Sensor Information")
        try:
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Sensor Source (Selected)**")
                sensor_source = ds.fixed_leader.sensor_info(ens=0, field="source")
                df_source = pd.DataFrame(
                    list(sensor_source.items()), columns=["Sensor", "Selected"]
                )
                df_source["Selected"] = df_source["Selected"].map(
                    {True: "Yes", False: "No"}
                )
                st.dataframe(df_source, use_container_width=True, hide_index=True)

            with col2:
                st.write("**Sensor Availability**")
                sensor_avail = ds.fixed_leader.sensor_info(ens=0, field="avail")
                df_avail = pd.DataFrame(
                    list(sensor_avail.items()), columns=["Sensor", "Available"]
                )
                df_avail["Available"] = df_avail["Available"].map(
                    {True: "Yes", False: "No"}
                )
                st.dataframe(df_avail, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error reading sensor information: {e}")

    # Tab 3: Coordinate Transform
    with fl_tabs[2]:
        st.subheader("Coordinate Transformation")
        try:
            coord_trans = ds.fixed_leader.coordinate_transformation(ens=0)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Coordinate System", coord_trans.get("Coordinates", "N/A"))
                st.metric(
                    "Tilt Correction",
                    "Enabled" if coord_trans.get("Tilt Correction") else "Disabled",
                )

            with col2:
                st.metric(
                    "Three-Beam Solution",
                    "Enabled" if coord_trans.get("Three-Beam Solution") else "Disabled",
                )
                st.metric(
                    "Bin Mapping",
                    "Enabled" if coord_trans.get("Bin Mapping") else "Disabled",
                )

        except Exception as e:
            st.error(f"Error reading coordinate transformation: {e}")

    # Tab 4: Deployment Thresholds
    with fl_tabs[3]:
        st.subheader("Deployment Thresholds")
        try:
            # Extract threshold fields from dataset
            thresholds = {}

            threshold_fields = [
                ("number_of_cells", "Number of Cells"),
                ("number_of_beams", "Number of Beams"),
                ("pings_per_ensemble", "Pings per Ensemble"),
                ("depth_cell_length", "Depth Cell Length (cm)"),
                ("bin_1_distance", "Bin 1 Distance (cm)"),
                ("low_correlation_threshold", "Correlation Threshold"),
                ("error_velocity_maximum", "Error Velocity Max (mm/s)"),
                ("percent_good_minimum", "Percent Good Min"),
                ("false_target_threshold", "False Target Threshold"),
            ]

            for var_name, display_name in threshold_fields:
                if var_name in ds.data_vars:
                    thresholds[display_name] = int(ds[var_name].values[0])

            if thresholds:
                df = pd.DataFrame(
                    list(thresholds.items()), columns=["Parameter", "Value"]
                )
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Threshold fields not found in dataset.")

        except Exception as e:
            st.error(f"Error reading thresholds: {e}")

    # Tab 5: Uniformity Check
    # Computed eagerly (no button) since it takes no input and doesn't mutate
    # state - gating it behind a button only added an extra rerun-triggering
    # interaction with no benefit.
    with fl_tabs[4]:
        st.subheader("Fixed Leader Uniformity Check")
        st.write("""
        Fixed Leader values should remain constant throughout the deployment.
        Non-uniform fields may indicate configuration changes or data issues.
        """)

        try:
            uniformity = ds.fixed_leader.is_uniform()

            # Separate uniform and non-uniform fields
            non_uniform_fields = [k for k, v in uniformity.items() if not v]

            if non_uniform_fields:
                st.warning(f"Found {len(non_uniform_fields)} non-uniform field(s)")
                st.write("**Non-uniform fields:**")
                for field in non_uniform_fields:
                    st.write(f"  - :red[{field}]")
            else:
                st.success("All Fixed Leader fields are uniform")

            with st.expander(f"View All Fields ({len(uniformity)} total)"):
                df = pd.DataFrame(
                    list(uniformity.items()), columns=["Field", "Uniform"]
                )
                df["Uniform"] = df["Uniform"].map({True: "Yes", False: "No"})
                st.dataframe(
                    df.style.map(
                        lambda x: "color: green" if x == "Yes" else "color: red",
                        subset=["Uniform"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as e:
            st.error(f"Error checking uniformity: {e}")

    # Tab 6: Raw Fields
    # Also computed eagerly for the same reason as the Uniformity Check tab.
    with fl_tabs[5]:
        st.subheader("Raw Fixed Leader Fields")
        try:
            raw_fields = ds.fixed_leader.field(ens=0)
            df = pd.DataFrame(list(raw_fields.items()), columns=["Field", "Value"])
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error loading raw fields: {e}")


# =============================================================================
# DISPLAY FUNCTIONS - VARIABLE LEADER
# =============================================================================


def display_variable_leader_summary(ds) -> None:
    """Display comprehensive Variable Leader summary statistics."""
    st.header("Variable Leader (Dynamic Measurements)", divider="blue")

    st.write("""
    Variable Leader data contains time-varying measurements including timestamps,
    motion sensors (heading, pitch, roll), environmental sensors (temperature, 
    salinity, pressure), and diagnostic information.
    """)

    # Create tabs for different Variable Leader information
    vl_tabs = st.tabs(
        [
            "Time Analysis",
            "Motion Sensors",
            "Environmental Sensors",
            "Ensemble Continuity",
            "Diagnostics (BIT)",
            "Error Status Words",
        ]
    )

    # Tab 1: Time Analysis
    with vl_tabs[0]:
        st.subheader("Time Analysis")
        try:
            col1, col2 = st.columns(2)

            with col1:
                # Time range
                time_values = pd.Series(ds.time.values)
                st.write("**Time Range:**")
                st.write(f"  - Start: {time_values.iloc[0]}")
                st.write(f"  - End: {time_values.iloc[-1]}")
                st.write(f"  - Duration: {time_values.iloc[-1] - time_values.iloc[0]}")

                # Regular intervals check
                is_regular = ds.variable_leader.is_time_regular()
                if is_regular:
                    st.success("Time intervals are regular")
                else:
                    st.warning("Time intervals are irregular")

            with col2:
                # Common interval
                interval = ds.variable_leader.get_time_interval()
                if interval:
                    st.metric("Most Common Interval", str(interval))

                # Total ensembles
                st.metric("Total Ensembles", len(ds.time))

            # Time interval distribution
            with st.expander("Time Interval Distribution"):
                freq = ds.variable_leader.get_time_interval_frequency()
                if freq:
                    df = pd.DataFrame(
                        list(freq.items()), columns=["Interval (HH:MM:SS)", "Count"]
                    )
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    if len(freq) > 1:
                        st.warning(
                            f"Multiple time intervals detected ({len(freq)} unique)"
                        )

        except Exception as e:
            st.error(f"Error analyzing time: {e}")

    # Tab 2: Motion Sensors
    with vl_tabs[1]:
        st.subheader("Motion Sensor Statistics")
        try:
            motion_stats = []

            # Heading
            if "heading_degrees" in ds.data_vars:
                heading = ds["heading_degrees"].values
            elif "heading" in ds.data_vars:
                heading = ds["heading"].values * ds["heading"].attrs.get(
                    "scale_factor", 0.01
                )
            else:
                heading = None

            if heading is not None:
                motion_stats.append(
                    {
                        "Sensor": "Heading",
                        "Mean": f"{np.nanmean(heading):.2f} deg",
                        "Std Dev": f"{np.nanstd(heading):.2f} deg",
                        "Min": f"{np.nanmin(heading):.2f} deg",
                        "Max": f"{np.nanmax(heading):.2f} deg",
                    }
                )

            # Pitch
            if "pitch_degrees" in ds.data_vars:
                pitch = ds["pitch_degrees"].values
            elif "pitch" in ds.data_vars:
                pitch = ds["pitch"].values * ds["pitch"].attrs.get("scale_factor", 0.01)
            else:
                pitch = None

            if pitch is not None:
                motion_stats.append(
                    {
                        "Sensor": "Pitch",
                        "Mean": f"{np.nanmean(pitch):.2f} deg",
                        "Std Dev": f"{np.nanstd(pitch):.2f} deg",
                        "Min": f"{np.nanmin(pitch):.2f} deg",
                        "Max": f"{np.nanmax(pitch):.2f} deg",
                    }
                )

            # Roll
            if "roll_degrees" in ds.data_vars:
                roll = ds["roll_degrees"].values
            elif "roll" in ds.data_vars:
                roll = ds["roll"].values * ds["roll"].attrs.get("scale_factor", 0.01)
            else:
                roll = None

            if roll is not None:
                motion_stats.append(
                    {
                        "Sensor": "Roll",
                        "Mean": f"{np.nanmean(roll):.2f} deg",
                        "Std Dev": f"{np.nanstd(roll):.2f} deg",
                        "Min": f"{np.nanmin(roll):.2f} deg",
                        "Max": f"{np.nanmax(roll):.2f} deg",
                    }
                )

            if motion_stats:
                df = pd.DataFrame(motion_stats)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Quick tilt assessment
                if pitch is not None and roll is not None:
                    max_tilt = max(np.nanmax(np.abs(pitch)), np.nanmax(np.abs(roll)))
                    if max_tilt > 20:
                        st.error(
                            f"High tilt detected: {max_tilt:.1f} deg - Consider applying tilt corrections"
                        )
                    elif max_tilt > 10:
                        st.warning(
                            f"Moderate tilt: {max_tilt:.1f} deg - Review tilt sensor data"
                        )
                    else:
                        st.success(f"Tilt within normal range: {max_tilt:.1f} deg")
            else:
                st.info("Motion sensor data not found in dataset.")

        except Exception as e:
            st.error(f"Error reading motion sensors: {e}")

    # Tab 3: Environmental Sensors
    with vl_tabs[2]:
        st.subheader("Environmental Sensor Statistics")
        try:
            env_stats = []

            # Temperature
            if "temperature_degrees" in ds.data_vars:
                temp = ds["temperature_degrees"].values
            elif "temperature" in ds.data_vars:
                temp = ds["temperature"].values * ds["temperature"].attrs.get(
                    "scale_factor", 0.01
                )
            else:
                temp = None

            if temp is not None:
                env_stats.append(
                    {
                        "Sensor": "Temperature",
                        "Mean": f"{np.nanmean(temp):.2f} C",
                        "Std Dev": f"{np.nanstd(temp):.2f} C",
                        "Min": f"{np.nanmin(temp):.2f} C",
                        "Max": f"{np.nanmax(temp):.2f} C",
                    }
                )

            # Salinity
            if "salinity" in ds.data_vars:
                sal = ds["salinity"].values * ds["salinity"].attrs.get(
                    "scale_factor", 1.0
                )
                env_stats.append(
                    {
                        "Sensor": "Salinity",
                        "Mean": f"{np.nanmean(sal):.2f} PSU",
                        "Std Dev": f"{np.nanstd(sal):.2f} PSU",
                        "Min": f"{np.nanmin(sal):.2f} PSU",
                        "Max": f"{np.nanmax(sal):.2f} PSU",
                    }
                )

            # Depth/Pressure
            if "depth_meters" in ds.data_vars:
                depth = ds["depth_meters"].values
            elif "depth_of_transducer" in ds.data_vars:
                depth = ds["depth_of_transducer"].values * ds[
                    "depth_of_transducer"
                ].attrs.get("scale_factor", 0.1)
            else:
                depth = None

            if depth is not None:
                env_stats.append(
                    {
                        "Sensor": "Transducer Depth",
                        "Mean": f"{np.nanmean(depth):.2f} m",
                        "Std Dev": f"{np.nanstd(depth):.2f} m",
                        "Min": f"{np.nanmin(depth):.2f} m",
                        "Max": f"{np.nanmax(depth):.2f} m",
                    }
                )

            # Sound Speed
            if "speed_of_sound" in ds.data_vars:
                sos = ds["speed_of_sound"].values
                env_stats.append(
                    {
                        "Sensor": "Sound Speed",
                        "Mean": f"{np.nanmean(sos):.1f} m/s",
                        "Std Dev": f"{np.nanstd(sos):.1f} m/s",
                        "Min": f"{np.nanmin(sos):.1f} m/s",
                        "Max": f"{np.nanmax(sos):.1f} m/s",
                    }
                )

            if env_stats:
                df = pd.DataFrame(env_stats)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("Environmental sensor data not found in dataset.")

        except Exception as e:
            st.error(f"Error reading environmental sensors: {e}")

    # Tab 4: Ensemble Continuity
    with vl_tabs[3]:
        st.subheader("Ensemble Continuity Check")
        try:
            continuity = ds.variable_leader.ensemble_continuity_check()

            if continuity["is_continuous"]:
                st.success("Ensemble numbering is continuous - no gaps detected")
            else:
                st.warning(
                    f"Found {continuity['gap_count']} gap(s) in ensemble numbering"
                )

                if continuity["gap_locations"]:
                    with st.expander("View Gap Details"):
                        gap_data = []
                        for i, (loc, size) in enumerate(
                            zip(
                                continuity["gap_locations"][:20],
                                continuity["gap_sizes"][:20],
                            )
                        ):
                            gap_data.append(
                                {
                                    "Gap #": i + 1,
                                    "Location (ensemble index)": loc,
                                    "Ensembles Missing": size,
                                }
                            )

                        df = pd.DataFrame(gap_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)

                        if len(continuity["gap_locations"]) > 20:
                            st.info(
                                f"Showing first 20 of {len(continuity['gap_locations'])} gaps"
                            )

            # Rollover count
            rollovers = ds.variable_leader.ensemble_rollover_count()
            st.metric("Ensemble Rollovers", rollovers)

        except Exception as e:
            st.error(f"Error checking continuity: {e}")

    # Tab 5: BIT Diagnostics
    with vl_tabs[4]:
        st.subheader("Built-In Test (BIT) Results")
        st.write("""
        The Built-In Test checks hardware and firmware status. 
        All tests should pass (result = 0) for healthy operation.
        """)

        try:
            bit_summary = ds.variable_leader.bit_result_summary()

            if bit_summary["all_passed"]:
                st.success("All BIT tests passed")
            else:
                st.error(
                    f"BIT errors detected in {bit_summary['error_count']} ensemble(s)"
                )

                if bit_summary["unique_error_codes"]:
                    st.write(
                        f"**Error codes found:** {bit_summary['unique_error_codes']}"
                    )

            # Detailed bit checks
            with st.expander("Detailed BIT Checks (8 bits)"):
                bit_data = []
                for bit_name, bit_info in bit_summary["bit_checks"].items():
                    bit_data.append(
                        {
                            "Check": bit_name.replace("_", " ").title(),
                            "Errors": bit_info["error_count"],
                            "Status": "PASS"
                            if bit_info["error_count"] == 0
                            else "FAIL",
                        }
                    )

                df = pd.DataFrame(bit_data)
                st.dataframe(
                    df.style.map(color_status, subset=["Status"]),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as e:
            st.error(f"Error reading BIT results: {e}")

    # Tab 6: Error Status Words
    with vl_tabs[5]:
        st.subheader("Error Status Word (ESW) Summary")
        st.write("""
        Error Status Words record operational events and anomalies.
        Non-zero values indicate events that occurred during data collection.
        """)

        try:
            esw_summary = ds.variable_leader.error_status_word_summary()

            for esw_name in ["ESW1", "ESW2", "ESW3", "ESW4"]:
                if esw_name not in esw_summary:
                    continue

                esw_data = esw_summary[esw_name]

                esw_status = (
                    "OK"
                    if esw_data["all_zeros"]
                    else f"{esw_data['total_events']} events"
                )
                with st.expander(f"{esw_name} - {esw_status}"):
                    if esw_data["all_zeros"]:
                        st.success("No events recorded")
                    else:
                        st.warning(f"Total events: {esw_data['total_events']}")

                        # Show non-zero checks only
                        event_data = []
                        for bit_name, bit_info in esw_data["bit_checks"].items():
                            if bit_info["event_count"] > 0:
                                event_data.append(
                                    {
                                        "Event Type": bit_name.replace(
                                            "_", " "
                                        ).title(),
                                        "Count": bit_info["event_count"],
                                    }
                                )

                        if event_data:
                            df = pd.DataFrame(event_data)
                            st.dataframe(df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error reading ESW: {e}")



# =============================================================================
# DISPLAY FUNCTIONS - DATA OVERVIEW
# =============================================================================


def display_data_overview(ds) -> None:
    """Display overview of velocity and quality data arrays."""
    st.header("Data Overview", divider="blue")

    # Get dimensions
    n_ensembles = ds.attrs.get("total_ensembles", len(ds.time))
    n_cells = ds.sizes.get("cell", 0)
    n_beams = ds.sizes.get("beam", 4)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ensembles", f"{n_ensembles:,}")
    with col2:
        st.metric("Depth Cells", n_cells)
    with col3:
        st.metric("Beams", n_beams)

    # Data availability
    st.subheader("Available Data Arrays")

    data_arrays = [
        ("velocity", "Velocity", "mm/s"),
        ("echo_intensity", "Echo Intensity", "counts"),
        ("correlation", "Correlation", "counts"),
        ("percent_good", "Percent Good", "%"),
    ]

    avail_data = []
    for var_name, display_name, units in data_arrays:
        if var_name in ds.data_vars:
            arr = ds[var_name].values
            shape = arr.shape

            # Count valid data (non-missing)
            if var_name == "velocity":
                if np.issubdtype(arr.dtype, np.floating):
                    valid_pct = (~np.isnan(arr)).mean() * 100
                else:
                    valid_pct = (arr != -32768).mean() * 100
            else:
                valid_pct = (arr != 0).mean() * 100

            avail_data.append(
                {
                    "Data Type": display_name,
                    "Shape": str(shape),
                    "Units": units,
                    "Valid %": f"{valid_pct:.1f}%",
                    "Available": "Yes",
                }
            )
        else:
            avail_data.append(
                {
                    "Data Type": display_name,
                    "Shape": "-",
                    "Units": units,
                    "Valid %": "-",
                    "Available": "No",
                }
            )

    df = pd.DataFrame(avail_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Mask information
    if "mask" in ds.data_vars:
        mask = ds["mask"].values
        masked_pct = (mask == 1).mean() * 100
        st.metric("Initial Masked Data", f"{masked_pct:.2f}%")


# =============================================================================
# MAIN APPLICATION
# =============================================================================


def main():
    """Main application entry point."""

    # Initialize session state
    initialize_minimal_session_state()

    # Title
    st.title("ADCP Data Processing Tool")
    st.write(f"*pyadps v{pyadps.__version__} - ProcessedDataset Workflow*")

    # File upload
    st.sidebar.header("File Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload RDI ADCP Binary File",
        type=["000", "ENS", "ENX", "LTA", "STA"],
        help="Select an RDI ADCP binary file (.000 or similar extension)",
    )

    # Handle file upload - new file uploaded
    if uploaded_file is not None and uploaded_file.name != st.session_state.fname:
        # Read the new file
        try:
            ds, ds_header, fpath = read_adcp_file(
                uploaded_file.getvalue(), uploaded_file.name
            )

            # Store in session state
            st.session_state.ds = ds
            st.session_state.ds_header = ds_header
            st.session_state.fname = uploaded_file.name
            st.session_state.fpath = fpath

            # Create ProcessedDataset
            st.session_state.processor = ProcessedDataset(ds)
            st.session_state.processor.config.input_file_name = uploaded_file.name
            st.session_state.processor.config.input_file_path = fpath
            st.session_state.processor.config.pyadps_version = pyadps.__version__
            st.session_state.processing_step = 0

            # Set file prefix
            st.session_state.ui_params["file_prefix"] = os.path.splitext(
                uploaded_file.name
            )[0]

            # st.toast() rather than st.sidebar.success(): a persistent element
            # shown only on the upload rerun (and gone on every rerun after,
            # once st.session_state.fname is updated above) changes the shape
            # of everything rendered before main_tabs is created, which can
            # reset main_tabs' active tab on the very next interaction.
            st.toast(f"Loaded: {uploaded_file.name}", icon="✅")

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

    # Check if we have data in session state (either just loaded or from previous session)
    if st.session_state.ds is not None:
        # Show current file info
        st.sidebar.info(f"Current file: {st.session_state.fname}")

        # Display sections
        ds = st.session_state.ds
        ds_header = st.session_state.ds_header

        # File info in sidebar
        st.sidebar.divider()
        st.sidebar.subheader("Quick Stats")
        st.sidebar.write(
            f"**Ensembles:** {ds.attrs.get('total_ensembles', len(ds.time)):,}"
        )
        st.sidebar.write(f"**Cells:** {ds.sizes.get('cell', 'N/A')}")

        if st.session_state.processor:
            st.sidebar.write(f"**Processing Step:** {st.session_state.processing_step}")

        # Main content tabs
        main_tabs = st.tabs(
            [
                "File Header",
                "Fixed Leader",
                "Variable Leader",
                "Data Overview",
            ]
        )

        with main_tabs[0]:
            display_file_header(ds_header)

        with main_tabs[1]:
            display_fixed_leader_summary(ds)

        with main_tabs[2]:
            display_variable_leader_summary(ds)

        with main_tabs[3]:
            display_data_overview(ds)

        # Processing status
        st.divider()
        st.subheader("Processing Status")

        if st.session_state.processor:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.success("ProcessedDataset initialized - Ready for processing")
                st.write("Navigate to subsequent pages to apply processing steps:")
                st.write(
                    "  1. **Sensor Health** - Roll/pitch checks, sound speed correction"
                )
                st.write("  2. **QC Tests** - Signal quality checks")
                st.write("  3. **Profile Operations** - Trim, cut bins, regrid")
                st.write(
                    "  4. **Velocity Checks** - Cutoffs, despike, magnetic correction"
                )
                st.write("  5. **Write File** - Export processed data")

            with col2:
                if st.button("Reset Processor"):
                    st.session_state.processor.reset()
                    st.session_state.processing_step = 0
                    st.success("Processor reset to original state")
                    st.rerun()

    else:
        # No file uploaded
        st.info("Please upload an ADCP binary file using the sidebar to begin.")

        st.divider()
        st.subheader("About This Application")
        st.write("""
        This application provides comprehensive tools for processing ADCP 
        (Acoustic Doppler Current Profiler) data using the **pyadps v1.0.0** library.
        
        **Features:**
        - Read RDI ADCP binary files (.000 format)
        - View comprehensive summary statistics
        - Apply sensor health corrections
        - Run quality control tests
        - Profile operations (trim, cut, regrid)
        - Velocity processing (despike, magnetic correction)
        - Export to NetCDF or CSV
        
        **Workflow:**
        The application uses `ProcessedDataset` as a central orchestrator that 
        tracks all processing steps and maintains data integrity throughout the pipeline.
        """)


if __name__ == "__main__":
    main()
