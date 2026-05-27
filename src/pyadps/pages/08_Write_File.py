"""
08_Write_File.py - Write Processed Data Page (Refactored for pyadps v1.0.0)

This page allows users to:
1. Preview processed data (velocity, echo, correlation, percent good)
2. Export processed data to NetCDF or CSV formats
3. Choose between velocity-only export (default) or full dataset export
4. Add custom metadata attributes to exported files
5. Generate configuration files for reproducible processing

Architecture:
- Uses st.session_state.processor (ProcessedDataset) as central state manager
- Uses proc.velocity_to_netcdf() for velocity-only export (default)
- Uses proc.to_netcdf() for full dataset export
- No complex mask variable management - all handled by ProcessedDataset
"""

import os
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pyadps.processing.config import ProcessingConfig

# =============================================================================
# PAGE CONFIGURATION AND VALIDATION
# =============================================================================

st.set_page_config(page_title="Write File", page_icon="💾", layout="wide")

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


def get_depth_axis():
    """Get depth/cell axis for plotting."""
    if "depth" in ds.coords:
        return ds["depth"].values
    elif "cell" in ds.coords:
        return ds["cell"].values
    return np.arange(get_total_cells())


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


def get_velocity_labels() -> tuple[str, str, str, str]:
    """Get appropriate velocity component labels based on coordinate system."""
    if is_earth_coordinates():
        return ("U (East)", "V (North)", "W (Vertical)", "Error")
    else:
        return ("Beam 1", "Beam 2", "Beam 3", "Beam 4")


def get_file_prefix() -> str:
    """Get file prefix from session state or derive from filename."""
    if "file_prefix" in st.session_state and st.session_state.file_prefix:
        return st.session_state.file_prefix
    elif "fname" in st.session_state and st.session_state.fname:
        return os.path.splitext(os.path.basename(st.session_state.fname))[0]
    return "ADCP"


def get_prefixed_filename(base_name: str) -> str:
    """Generate filename with optional prefix."""
    prefix = get_file_prefix()
    if prefix:
        return f"{prefix}_{base_name}"
    return base_name


def get_cell_size() -> float:
    """Get cell size in meters from the dataset."""
    try:
        fl_data = ds.fixed_leader.field(ens=0)
        return fl_data.get("depth_cell_length", 100) / 100.0  # cm to m
    except Exception:
        if "depth_cell_length" in ds.data_vars:
            return float(ds["depth_cell_length"].values[0]) / 100.0
        return 1.0


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def plot_data_heatmap(
    data: np.ndarray,
    title: str,
    apply_mask: bool = False,
    colorscale: str = "balance",
    missing_value: float = -32768,
) -> None:
    """Plot a 2D heatmap of data (cell x time)."""
    time_axis = get_time_axis()
    depth_axis = get_depth_axis()

    # Handle masking
    plot_data = data.copy().astype(float)

    if apply_mask and "mask" in ds.data_vars:
        mask = ds["mask"].values
        # If mask is 3D (beam, cell, time), use beam 0 for non-velocity data
        if mask.ndim == 3 and plot_data.ndim == 2:
            mask_2d = mask[0, :, :]
        elif mask.ndim == 2:
            mask_2d = mask
        else:
            mask_2d = None

        if mask_2d is not None:
            plot_data = np.where(mask_2d == 1, np.nan, plot_data)

    # Replace missing values with NaN
    plot_data = np.where(plot_data == missing_value, np.nan, plot_data)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=plot_data,
            x=time_axis,
            y=depth_axis,
            colorscale=colorscale,
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Depth/Cell",
        height=400,
    )
    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True)


def plot_velocity_component(
    beam_idx: int, title: str, apply_mask: bool = False
) -> None:
    """Plot velocity component as a heatmap."""
    if "velocity" not in ds.data_vars:
        st.warning("No velocity data available.")
        return

    velocity = ds["velocity"].values
    vel_data = velocity[beam_idx, :, :].copy().astype(float)

    # Apply mask if requested
    if apply_mask and "mask" in ds.data_vars:
        mask = ds["mask"].values
        if mask.ndim == 3:
            vel_data = np.where(mask[beam_idx, :, :] == 1, np.nan, vel_data)

    # Replace missing values
    vel_data = np.where(vel_data == -32768, np.nan, vel_data)

    time_axis = get_time_axis()
    depth_axis = get_depth_axis()

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=vel_data,
            x=time_axis,
            y=depth_axis,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="mm/s"),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Depth/Cell",
        height=400,
    )
    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

# Initialize page-specific session state
if "write_initialized" not in st.session_state:
    st.session_state.write_initialized = False

if not st.session_state.write_initialized:
    # File settings
    st.session_state.file_prefix = get_file_prefix()

    # Export options
    st.session_state.export_format = "NetCDF"
    st.session_state.export_type = "Velocity Only"  # Default to velocity-only
    st.session_state.apply_mask_export = True
    st.session_state.velocity_units = "cm/s"

    # Custom attributes
    st.session_state.add_attributes = False
    st.session_state.custom_attributes = {}

    st.session_state.write_initialized = True


# =============================================================================
# PAGE HEADER
# =============================================================================

st.header("💾 Write Processed Data", divider="blue")

st.write("""
Export your processed ADCP data to NetCDF or CSV format. You can choose between:
- **Velocity Only** (recommended): Exports just U, V, W velocity components with QC mask applied
- **Full Dataset**: Exports the complete dataset including all variables and metadata
""")


# =============================================================================
# CONFIG BUILDER
# =============================================================================


def build_config_from_session() -> ProcessingConfig:
    """Build a ProcessingConfig from current session state values."""
    ss = st.session_state
    n_ens = get_total_ensembles()
    trim_start = ss.get("trim_start_ens", 0)
    trim_end = ss.get("trim_end_ens", max(0, n_ens - 1))
    has_trim = trim_start > 0 or trim_end < (n_ens - 1)
    return ProcessingConfig(
        # File
        input_file_name=ss.get("fname", ""),
        # Time
        isTimeAxisModified=ss.get("time_axis_modified", False),
        isSnapTimeAxis=ss.get("snap_time_axis", False),
        time_snap_frequency=ss.get("time_snap_frequency", "h"),
        time_snap_tolerance=ss.get("time_snap_tolerance", "5min"),
        time_target_minute=ss.get("time_target_minute", 0),
        isTimeGapFilled=ss.get("time_gap_filled", False),
        time_fill_method=ss.get("time_fill_method", "auto"),
        # Sensor Health
        isSensorTest=ss.get("sensor_health_applied", False),
        isDepthModified_ST=ss.get("depth_modified", False),
        depthoption_ST=ss.get("depth_option", "None"),
        fixeddepth_ST=float(ss.get("depth_fixed_value", 0.0)),
        isSalinityModified_ST=ss.get("salinity_modified", False),
        salinityoption_ST=ss.get("salinity_option", "None"),
        fixedsalinity_ST=float(ss.get("salinity_fixed_value", 35.0)),
        isTemperatureModified_ST=ss.get("temperature_modified", False),
        temperatureoption_ST=ss.get("temperature_option", "None"),
        fixedtemperature_ST=float(ss.get("temperature_fixed_value", 15.0)),
        isRollCheck_ST=ss.get("apply_roll_check", False),
        isPitchCheck_ST=ss.get("apply_pitch_check", False),
        roll_cutoff_ST=float(ss.get("roll_threshold", 15.0)),
        pitch_cutoff_ST=float(ss.get("pitch_threshold", 15.0)),
        isSoundModified_ST=ss.get("apply_sound_speed_correction", False),
        isVelocityModified_ST=ss.get("correct_velocity", True),
        isVelocityModified_HorizontalOnly_ST=ss.get("horizontal_only", True),
        # QC
        isQCTest=ss.get("qc_applied", False),
        ct_QCT=float(ss.get("correlation_threshold", 64)),
        et_QCT=float(ss.get("echo_intensity_threshold", 0)),
        evt_QCT=float(ss.get("error_velocity_threshold", 2000)),
        ft_QCT=float(ss.get("false_target_threshold", 50)),
        is3beam_QCT=ss.get("threebeam_mode", False),
        beam_ignore_QCT=ss.get("beam_ignore", None),
        pgt_QCT=float(ss.get("percent_good_threshold", 0)),
        # Profile
        isProfileTest=ss.get("profile_applied", False),
        isTrimEndsCheck_PT=has_trim,
        trim_start_PT=trim_start,
        trim_end_PT=trim_end,
        isCutBinSideLobeCheck_PT=ss.get("apply_side_lobe", False),
        water_depth_PT=float(ss.get("water_depth") or 0.0),
        extra_cells_PT=int(ss.get("extra_cells", 1)),
        isCutBinManualCheck_PT=bool(ss.get("cut_regions", [])),
        cut_bins_regions_PT=[
            [r["min_cell"], r["max_cell"], r["min_ensemble"], r["max_ensemble"]]
            for r in ss.get("cut_regions", [])
        ],
        isRegridCheck_PT=ss.get("apply_regrid", False),
        regrid_cell_size_PT=get_cell_size(),
        regrid_method_PT=ss.get("regrid_method", "nearest"),
        regrid_end_cell_option_PT=ss.get("end_cell_option", "cell"),
        regrid_boundary_limit_PT=float(ss.get("boundary_limit", 0.0)),
        beam_direction_PT=(ss.get("beam_direction", "up") or "up").lower(),
        # Velocity
        isVelocityTest=ss.get("velocity_applied", False),
        isMagnetCheck_VT=ss.get("apply_magnetic", False),
        magnet_method_VT=ss.get("magnetic_method", "api"),
        magnet_lat_VT=float(ss.get("magnetic_lat", 0.0)),
        magnet_lon_VT=float(ss.get("magnetic_lon", 0.0)),
        magnet_year_VT=int(ss.get("magnetic_year", 2025)),
        magnet_depth_VT=float(ss.get("magnetic_depth", 0.0)),
        magnet_user_input_VT=float(ss.get("magnetic_declination") or 0.0),
        isCutoffCheck_VT=ss.get("apply_threshold", False),
        maxuvel_VT=float(ss.get("cutoff_u", 2500)),
        maxvvel_VT=float(ss.get("cutoff_v", 2500)),
        maxwvel_VT=float(ss.get("cutoff_w", 500)),
        isDespikeCheck_VT=ss.get("apply_despike", False),
        despike_kernel_VT=int(ss.get("despike_kernel", 5)),
        despike_cutoff_VT=float(ss.get("despike_cutoff", 3.0)),
        isFlatlineCheck_VT=ss.get("apply_flatline", False),
        flatline_kernel_VT=int(ss.get("flatline_kernel", 5)),
        flatline_cutoff_VT=float(ss.get("flatline_cutoff", 3.0)),
        # Attributes
        isAttributes=ss.get("add_attributes", False),
        attributes={k: v for k, v in ss.get("custom_attributes", {}).items() if v},
    )


# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Preview Data", "📝 Attributes", "💾 Export Data", "⚙️ Config File"]
)


# =============================================================================
# TAB 1: PREVIEW DATA
# =============================================================================

with tab1:
    st.header("Preview Processed Data", divider="blue")

    st.write("""
    Preview your processed data before exporting. The mask can be applied to see 
    what the final exported data will look like.
    """)

    # Data type selection
    col1, col2, col3 = st.columns(3)

    with col1:
        var_options = ["Velocity", "Echo Intensity", "Correlation", "Percent Good"]
        var_selection = st.selectbox(
            "Select variable to view", var_options, key="preview_var"
        )

    with col2:
        if var_selection == "Velocity":
            u_label, v_label, w_label, err_label = get_velocity_labels()
            beam_options = [u_label, v_label, w_label, err_label]
            beam_selection = st.selectbox(
                "Select component", beam_options, key="preview_beam"
            )
            beam_idx = beam_options.index(beam_selection)
        else:
            beam_idx = st.selectbox("Select beam", [1, 2, 3, 4], key="preview_beam_num")
            beam_idx = beam_idx - 1  # Convert to 0-indexed

    with col3:
        apply_mask_preview = st.radio(
            "Apply mask?", ["Yes", "No"], horizontal=True, key="preview_mask"
        )
        apply_mask = apply_mask_preview == "Yes"

    # Plot button
    if st.button("📈 Plot Data", key="plot_preview"):
        if var_selection == "Velocity":
            plot_velocity_component(
                beam_idx, f"{var_selection} - {beam_selection}", apply_mask
            )
        else:
            # Get the appropriate data variable
            var_mapping = {
                "Echo Intensity": "echo_intensity",
                "Correlation": "correlation",
                "Percent Good": "percent_good",
            }
            var_name = var_mapping.get(var_selection, "echo_intensity")

            if var_name in ds.data_vars:
                data = ds[var_name].values
                if data.ndim == 3:  # (beam, cell, time)
                    data_2d = data[beam_idx, :, :]
                else:
                    data_2d = data

                colorscale = "Viridis" if var_selection != "Correlation" else "Plasma"
                plot_data_heatmap(
                    data_2d,
                    f"{var_selection} - Beam {beam_idx + 1}",
                    apply_mask=apply_mask,
                    colorscale=colorscale,
                    missing_value=0,
                )
            else:
                st.warning(f"Variable '{var_name}' not found in dataset.")

    # Show processing summary
    with st.expander("📊 Processing Summary", expanded=True):
        stats = proc.get_current_stats()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cells", f"{stats['total_cells']:,}")
        with col2:
            st.metric(
                "Valid Cells", f"{stats['valid']:,}", delta=f"{stats['valid_pct']:.1f}%"
            )
        with col3:
            st.metric(
                "Masked Cells",
                f"{stats['masked']:,}",
                delta=f"-{stats['masked_pct']:.1f}%",
            )

        if hasattr(proc, "processing_log") and proc.processing_log:
            st.write("**Processing Steps Applied:**")
            for step in proc.processing_log:
                st.write(f"- {step}")


# =============================================================================
# TAB 2: CUSTOM ATTRIBUTES
# =============================================================================

with tab2:
    st.header("Custom Attributes", divider="blue")

    st.write("""
    Add custom metadata attributes to your exported NetCDF file. These attributes 
    provide important context about the deployment and data collection.
    """)

    st.session_state.add_attributes = st.checkbox(
        "Add custom attributes to export",
        value=st.session_state.add_attributes,
        key="add_attrs_checkbox",
    )

    if st.session_state.add_attributes:
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Deployment Information:**")

            cruise = st.text_input(
                "Cruise Number",
                value=st.session_state.custom_attributes.get("cruise_number", ""),
                key="attr_cruise",
            )
            ship = st.text_input(
                "Ship Name",
                value=st.session_state.custom_attributes.get("ship_name", ""),
                key="attr_ship",
            )
            project = st.text_input(
                "Project Number",
                value=st.session_state.custom_attributes.get("project_number", ""),
                key="attr_project",
            )
            water_depth = st.text_input(
                "Water Depth (m)",
                value=st.session_state.custom_attributes.get("water_depth", ""),
                key="attr_water_depth",
            )
            deploy_depth = st.text_input(
                "Deployment Depth (m)",
                value=st.session_state.custom_attributes.get("deployment_depth", ""),
                key="attr_deploy_depth",
            )
            deploy_date = st.text_input(
                "Deployment Date",
                value=st.session_state.custom_attributes.get("deployment_date", ""),
                key="attr_deploy_date",
            )
            recovery_date = st.text_input(
                "Recovery Date",
                value=st.session_state.custom_attributes.get("recovery_date", ""),
                key="attr_recovery_date",
            )

        with col2:
            st.write("**Location & Contact:**")

            latitude = st.text_input(
                "Latitude",
                value=st.session_state.custom_attributes.get("latitude", ""),
                key="attr_lat",
            )
            longitude = st.text_input(
                "Longitude",
                value=st.session_state.custom_attributes.get("longitude", ""),
                key="attr_lon",
            )
            platform = st.text_input(
                "Platform Type",
                value=st.session_state.custom_attributes.get("platform_type", ""),
                key="attr_platform",
            )
            participants = st.text_input(
                "Participants",
                value=st.session_state.custom_attributes.get("participants", ""),
                key="attr_participants",
            )
            created_by = st.text_input(
                "File Created By",
                value=st.session_state.custom_attributes.get("file_created_by", ""),
                key="attr_created_by",
            )
            contact = st.text_input(
                "Contact",
                value=st.session_state.custom_attributes.get("contact", ""),
                key="attr_contact",
            )
            comments = st.text_area(
                "Comments",
                value=st.session_state.custom_attributes.get("comments", ""),
                key="attr_comments",
            )

        # Update session state
        st.session_state.custom_attributes = {
            "cruise_number": cruise,
            "ship_name": ship,
            "project_number": project,
            "water_depth": water_depth,
            "deployment_depth": deploy_depth,
            "deployment_date": deploy_date,
            "recovery_date": recovery_date,
            "latitude": latitude,
            "longitude": longitude,
            "platform_type": platform,
            "participants": participants,
            "file_created_by": created_by,
            "contact": contact,
            "comments": comments,
        }

        # Show preview
        with st.expander("Preview Attributes"):
            attrs_df = pd.DataFrame(
                [(k, v) for k, v in st.session_state.custom_attributes.items() if v],
                columns=["Attribute", "Value"],
            )
            if not attrs_df.empty:
                st.dataframe(attrs_df, hide_index=True, use_container_width=True)
            else:
                st.info("No attributes entered yet.")

    # Instruction to proceed to Export tab
    st.info(
        "💡 After configuring attributes, proceed to the **Export Data** tab to download your files."
    )


# =============================================================================
# TAB 3: EXPORT DATA
# =============================================================================

with tab3:
    st.header("Export Processed Data", divider="blue")

    # File prefix
    st.session_state.file_prefix = st.text_input(
        "File prefix",
        value=st.session_state.file_prefix,
        help="Prefix added to all exported filenames",
    )

    st.divider()

    # Export format selection
    col1, col2 = st.columns(2)

    with col1:
        st.session_state.export_format = st.radio(
            "Output format",
            ["NetCDF", "CSV"],
            key="export_format_radio",
            help="NetCDF is recommended for most use cases",
        )

    with col2:
        st.session_state.export_type = st.radio(
            "Export type",
            ["Velocity Only", "Full Dataset"],
            key="export_type_radio",
            help="Velocity Only exports U, V, W components. Full Dataset includes all variables.",
        )

    st.divider()

    # Export options
    col1, col2 = st.columns(2)

    with col1:
        st.session_state.apply_mask_export = st.checkbox(
            "Apply QC mask to exported data",
            value=st.session_state.apply_mask_export,
            help="Masked values will be set to NaN in the exported file",
        )

    with col2:
        if st.session_state.export_type == "Velocity Only":
            st.session_state.velocity_units = st.selectbox(
                "Velocity units",
                ["mm/s", "cm/s", "m/s"],
                index=1,  # Default to cm/s
                key="velocity_units_select",
                help="Original data is in mm/s. Select output units.",
            )

    # Show attributes status
    if st.session_state.add_attributes:
        n_attrs = len([v for v in st.session_state.custom_attributes.values() if v])
        st.success(f"✅ {n_attrs} custom attributes will be included in the export.")
    else:
        st.info(
            "ℹ️ No custom attributes configured. Configure in the **Attributes** tab if needed."
        )

    st.divider()

    # Export button
    st.write("### Generate Export Files")

    if st.button("🚀 Generate Files", type="primary", key="generate_export"):
        try:
            with st.spinner("Generating export files..."):
                # Create temporary directory
                temp_dir = tempfile.mkdtemp()

                if st.session_state.export_format == "NetCDF":
                    if st.session_state.export_type == "Velocity Only":
                        # Use velocity_to_netcdf for velocity-only export
                        filename = get_prefixed_filename("velocity.nc")
                        filepath = os.path.join(temp_dir, filename)

                        # Add custom attributes to the processor dataset before export
                        if (
                            st.session_state.add_attributes
                            and st.session_state.custom_attributes
                        ):
                            for (
                                key,
                                value,
                            ) in st.session_state.custom_attributes.items():
                                if value:  # Only add non-empty attributes
                                    proc.dataset.attrs[key] = value

                        proc.velocity_to_netcdf(
                            filepath,
                            apply_mask=st.session_state.apply_mask_export,
                            units=st.session_state.velocity_units,
                            include_metadata=True,
                        )

                        # Read file for download
                        with open(filepath, "rb") as f:
                            file_data = f.read()

                        st.download_button(
                            label="📥 Download Velocity NetCDF",
                            data=file_data,
                            file_name=filename,
                            mime="application/x-netcdf",
                        )

                        st.success(f"✅ Velocity file generated: {filename}")

                        # Show what was included
                        if st.session_state.add_attributes:
                            n_attrs = len(
                                [
                                    v
                                    for v in st.session_state.custom_attributes.values()
                                    if v
                                ]
                            )
                            st.write(f"📝 Included {n_attrs} custom attributes")

                    else:
                        # Use to_netcdf for full dataset export
                        filename = get_prefixed_filename("processed.nc")
                        filepath = os.path.join(temp_dir, filename)

                        # Add custom attributes to the processor dataset before export
                        if (
                            st.session_state.add_attributes
                            and st.session_state.custom_attributes
                        ):
                            for (
                                key,
                                value,
                            ) in st.session_state.custom_attributes.items():
                                if value:  # Only add non-empty attributes
                                    proc.dataset.attrs[key] = value

                        proc.to_netcdf(filepath)

                        # Read file for download
                        with open(filepath, "rb") as f:
                            file_data = f.read()

                        st.download_button(
                            label="📥 Download Full Dataset NetCDF",
                            data=file_data,
                            file_name=filename,
                            mime="application/x-netcdf",
                        )

                        st.success(f"✅ Full dataset file generated: {filename}")

                        # Show what was included
                        if st.session_state.add_attributes:
                            n_attrs = len(
                                [
                                    v
                                    for v in st.session_state.custom_attributes.values()
                                    if v
                                ]
                            )
                            st.write(f"📝 Included {n_attrs} custom attributes")

                else:  # CSV format
                    st.write("Generating CSV files...")

                    # Get velocity data
                    velocity = ds["velocity"].values
                    time_axis = get_time_axis()
                    depth_axis = get_depth_axis()

                    # Apply mask if requested
                    if st.session_state.apply_mask_export and "mask" in ds.data_vars:
                        mask = ds["mask"].values

                    u_label, v_label, w_label, _ = get_velocity_labels()

                    # Generate CSV for each velocity component
                    for i, (beam_idx, label) in enumerate(
                        [(0, "zonal"), (1, "meridional"), (2, "vertical")]
                    ):
                        vel_data = velocity[beam_idx, :, :].copy().astype(float)
                        vel_data[vel_data == -32768] = np.nan

                        if (
                            st.session_state.apply_mask_export
                            and "mask" in ds.data_vars
                        ):
                            vel_data = np.where(
                                mask[beam_idx, :, :] == 1, np.nan, vel_data
                            )

                        # Create DataFrame
                        df = pd.DataFrame(
                            vel_data.T,
                            index=time_axis,
                            columns=depth_axis,
                        )

                        csv_data = df.to_csv().encode("utf-8")
                        filename = get_prefixed_filename(f"{label}_velocity.csv")

                        st.download_button(
                            label=f"📥 Download {label.title()} Velocity CSV",
                            data=csv_data,
                            file_name=filename,
                            mime="text/csv",
                            key=f"csv_{label}",
                        )

                    # Export mask as CSV
                    if "mask" in ds.data_vars:
                        mask = ds["mask"].values
                        # Use combined mask (beam 3) if available
                        mask_2d = mask[3, :, :] if mask.shape[0] > 3 else mask[0, :, :]
                        mask_df = pd.DataFrame(
                            mask_2d.T, index=time_axis, columns=depth_axis
                        )
                        mask_csv = mask_df.to_csv().encode("utf-8")

                        st.download_button(
                            label="📥 Download Mask CSV",
                            data=mask_csv,
                            file_name=get_prefixed_filename("mask.csv"),
                            mime="text/csv",
                            key="csv_mask",
                        )

                    st.success("✅ CSV files generated successfully!")

        except Exception as e:
            st.error(f"❌ Error generating files: {e}")
            import traceback

            st.code(traceback.format_exc())


# =============================================================================
# TAB 4: CONFIG FILE GENERATOR
# =============================================================================

with tab4:
    st.header("Configuration File Generator", divider="blue")

    st.write("""
    Generate a configuration file (config.ini) that captures all processing settings.
    This file can be used to reproduce the processing using the pyadps autoprocess function.
    """)

    generate_config = st.checkbox(
        "Generate configuration file",
        value=False,
        key="generate_config_checkbox",
    )

    if generate_config:
        st.info("""
        **Note:** Configuration file generation captures the current processing state.
        The generated config.ini can be used with `pyadps.autoprocess()` for batch processing.
        """)

        if st.button("📄 Generate config.ini", key="gen_config_btn"):
            try:
                config = build_config_from_session()
                config_content = config.to_ini_string()

                with st.expander("Preview config.ini", expanded=True):
                    st.code(config_content, language="ini")

                st.download_button(
                    label="📥 Download config.ini",
                    data=config_content.encode("utf-8"),
                    file_name="config.ini",
                    mime="text/plain",
                )

                st.success("✅ Configuration file generated!")

            except Exception as e:
                st.error(f"❌ Error generating config file: {e}")


# =============================================================================
# SIDEBAR: PROCESSING STATUS
# =============================================================================

with st.sidebar:
    st.header("📊 Export Summary")

    # Current statistics
    stats = proc.get_current_stats()

    st.metric("Total Cells", f"{stats['total_cells']:,}")
    st.metric("Valid Cells", f"{stats['valid']:,}", delta=f"{stats['valid_pct']:.1f}%")
    st.metric(
        "Masked Cells", f"{stats['masked']:,}", delta=f"-{stats['masked_pct']:.1f}%"
    )

    st.write("---")

    # Export settings summary
    st.write("**Export Settings:**")
    st.write(f"- Format: {st.session_state.export_format}")
    st.write(f"- Type: {st.session_state.export_type}")
    st.write(f"- Apply Mask: {'✅' if st.session_state.apply_mask_export else '❌'}")
    if st.session_state.export_type == "Velocity Only":
        st.write(f"- Units: {st.session_state.velocity_units}")
    st.write(f"- Custom Attrs: {'✅' if st.session_state.add_attributes else '❌'}")

    st.write("---")

    # Dataset info
    st.write("**Dataset Info:**")
    st.write(f"- Ensembles: {get_total_ensembles():,}")
    st.write(f"- Cells: {get_total_cells()}")
    st.write(f"- Beams: {get_total_beams()}")

    # Check coordinate system
    if is_earth_coordinates():
        st.write("- Coords: Earth (U, V, W)")
    else:
        st.write("- Coords: Beam (1, 2, 3, 4)")

    st.write("---")

    # Processing log
    st.write("**Processing Log:**")
    if hasattr(proc, "processing_log") and proc.processing_log:
        for log_entry in proc.processing_log[-5:]:
            st.write(f"- {log_entry}")
    else:
        st.write("*No processing steps applied yet.*")
