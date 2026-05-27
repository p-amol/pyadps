"""
06_Profile_Operations.py - Profile Operations Page (Refactored for pyadps v1.0.0)

This page allows users to:
1. Trim deployment/recovery ensembles from data
2. Cut bins affected by side lobe contamination
3. Manually cut specific regions (cells/ensembles)
4. Regrid data from instrument cells to regular depth grid
5. Preview changes before committing to the main processor
6. Save processing results to the central ProcessedDataset

Architecture:
- Uses st.session_state.processor (ProcessedDataset) as central state manager
- Uses ProfileOperationRunner for interactive preview and operations
- Staging processor pattern (preview_profile_proc) for safe experimentation
- All processing is tracked through the processor's reports

⚠️ IMPORTANT: Profile operations should be applied AFTER QC checks because
regridding changes the cell structure, invalidating cell-based masks.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pyadps.processing import ProcessedDataset

# =============================================================================
# PAGE CONFIGURATION AND VALIDATION
# =============================================================================

st.set_page_config(page_title="Profile Operations", page_icon="📊", layout="wide")

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
    return 0


def get_total_beams() -> int:
    """Get total number of beams from dataset."""
    if "beam" in ds.dims:
        return ds.sizes["beam"]
    return 4


def get_transducer_depth() -> np.ndarray:
    """Get transducer depth array (in meters)."""
    if "transducer_depth" in ds.data_vars:
        # Convert from decimeters to meters
        scale = ds["transducer_depth"].attrs.get("scale_factor", 0.1)
        return ds["transducer_depth"].values * scale
    return np.zeros(get_total_ensembles())


def get_cell_size() -> float:
    """Get cell size in meters."""
    try:
        fl_data = ds.fixed_leader.field(ens=0)
        return fl_data.get("depth_cell_length", 100) / 100.0  # cm to m
    except Exception:
        if "depth_cell_length" in ds.data_vars:
            return float(ds["depth_cell_length"].values[0]) / 100.0
        return 1.0


def get_bin1_distance() -> float:
    """Get distance to first bin in meters."""
    try:
        fl_data = ds.fixed_leader.field(ens=0)
        return fl_data.get("bin_1_distance", 100) / 100.0  # cm to m
    except Exception:
        if "bin_1_distance" in ds.data_vars:
            return float(ds["bin_1_distance"].values[0]) / 100.0
        return 1.0


def get_beam_angle() -> int:
    """Get beam angle in degrees."""
    try:
        sys_config = ds.fixed_leader.system_configuration(ens=0)
        return int(sys_config.get("Beam Angle", 20))
    except Exception:
        return 20


def get_beam_direction() -> str:
    """
    Get beam direction from dataset.

    Uses the mode of ds['beam_direction'] values since the ADCP might be
    placed in wrong direction on ship deck, but will have correct direction
    during actual deployment.
    """
    try:
        if "beam_direction" in ds.data_vars:
            beam_dir_values = ds["beam_direction"].values
            unique, counts = np.unique(beam_dir_values, return_counts=True)
            mode_value = unique[np.argmax(counts)]
            if mode_value == 0:
                return "Down"
            elif mode_value == 1:
                return "Up"
        # Fallback to fixed_leader accessor
        sys_config = ds.fixed_leader.system_configuration(ens=0)
        return sys_config.get("Beam Direction", "Unknown")
    except Exception:
        if "beam_direction" in ds.attrs:
            return ds.attrs["beam_direction"]
        return "Unknown"


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


def plot_heatmap(
    data: np.ndarray,
    title: str,
    mask_data: np.ndarray = None,
    colorscale: str = "balance",
) -> None:
    """Create a heatmap plot for 2D data (cell x ensemble)."""
    n_ensembles = get_total_ensembles()
    n_cells = get_total_cells()

    # Handle missing values
    plot_data = np.where(data == -32768, np.nan, data)

    # Ensure 2D data (cell x ensemble)
    if plot_data.ndim == 3:
        plot_data = plot_data[0, :, :]

    fig = go.Figure()

    # Add data heatmap
    fig.add_trace(
        go.Heatmap(
            z=plot_data,
            x=np.arange(n_ensembles),
            y=np.arange(n_cells),
            colorscale=colorscale,
            hoverongaps=False,
        )
    )

    # Add mask overlay if provided
    if mask_data is not None:
        # Ensure 2D mask
        if mask_data.ndim == 3:
            mask_2d = mask_data[0, :, :]
        else:
            mask_2d = mask_data

        fig.add_trace(
            go.Heatmap(
                z=mask_2d,
                x=np.arange(n_ensembles),
                y=np.arange(n_cells),
                colorscale="gray",
                hoverongaps=False,
                showscale=False,
                opacity=0.4,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Ensemble",
        yaxis_title="Cell",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_trim_ends(
    start_ens: int = 0, end_ens: int = None, ens_range: int = 20
) -> None:
    """Plot transducer depth for deployment and recovery identification."""
    depth = get_transducer_depth()
    n_ensembles = get_total_ensembles()
    x = np.arange(n_ensembles)

    if end_ens is None:
        end_ens = n_ensembles

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Deployment Ensemble", "Recovery Ensemble"],
    )

    # Deployment (first ens_range points)
    fig.add_trace(
        go.Scatter(
            x=x[:ens_range],
            y=depth[:ens_range],
            name="Deployment",
            mode="markers",
            marker=dict(color="#1f77b4"),
        ),
        row=1,
        col=1,
    )

    # Recovery (last ens_range points)
    fig.add_trace(
        go.Scatter(
            x=x[-ens_range:],
            y=depth[-ens_range:],
            name="Recovery",
            mode="markers",
            marker=dict(color="#17becf"),
        ),
        row=1,
        col=2,
    )

    # Highlight trimmed regions
    if start_ens > 0:
        fig.add_trace(
            go.Scatter(
                x=x[:start_ens],
                y=depth[:start_ens],
                name="To Trim (D)",
                mode="markers",
                marker=dict(color="red"),
            ),
            row=1,
            col=1,
        )

    if end_ens < n_ensembles:
        fig.add_trace(
            go.Scatter(
                x=x[end_ens:],
                y=depth[end_ens:],
                name="To Trim (R)",
                mode="markers",
                marker=dict(color="orange"),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(height=500, title_text="Transducer Depth")
    fig.update_xaxes(title="Ensemble")
    fig.update_yaxes(title="Depth (m)")

    st.plotly_chart(fig, use_container_width=True)


def plot_mask_comparison(original_mask: np.ndarray, preview_mask: np.ndarray) -> None:
    """Plot side-by-side comparison of original and preview masks."""
    n_ensembles = get_total_ensembles()
    n_cells = get_total_cells()

    # Ensure 2D masks for plotting
    if original_mask.ndim == 3:
        orig_2d = original_mask[0, :, :]
    else:
        orig_2d = original_mask

    if preview_mask.ndim == 3:
        prev_2d = preview_mask[0, :, :]
    else:
        prev_2d = preview_mask

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Original Mask", "Preview Mask"],
    )

    # Original mask
    fig.add_trace(
        go.Heatmap(
            z=orig_2d,
            x=np.arange(n_ensembles),
            y=np.arange(n_cells),
            colorscale="greys",
            showscale=False,
        ),
        row=1,
        col=1,
    )

    # Preview mask
    fig.add_trace(
        go.Heatmap(
            z=prev_2d,
            x=np.arange(n_ensembles),
            y=np.arange(n_cells),
            colorscale="greys",
            showscale=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(height=400, title_text="Mask Comparison")
    fig.update_xaxes(title="Ensemble")
    fig.update_yaxes(title="Cell")

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if "profile_initialized" not in st.session_state:
    st.session_state.profile_initialized = True
    st.session_state.profile_applied = False
    st.session_state.profile_preview_run = False

    # Trim settings
    st.session_state.trim_start = 0
    st.session_state.trim_end = 0

    # Side lobe settings
    st.session_state.apply_side_lobe = False
    st.session_state.water_depth = 0.0
    st.session_state.extra_cells = 1

    # Manual cut regions (list of dicts)
    st.session_state.cut_regions = []

    # Regrid settings
    st.session_state.apply_regrid = False
    st.session_state.regrid_method = "nearest"
    st.session_state.end_cell_option = "cell"
    st.session_state.boundary_limit = 0.0

    # Preview statistics
    st.session_state.profile_preview_stats = None

    # Beam selection for visualization
    st.session_state.profile_beam = 0

    # Store orientation
    st.session_state.beam_direction = get_beam_direction()


# =============================================================================
# STAGING PROCESSOR FOR PREVIEW
# =============================================================================
# Create a staging processor (preview_profile_proc) for safe experimentation.
# This allows users to preview profile operation impacts without modifying
# the main processor. Only the final "Save" action applies changes.

if (
    "preview_profile_proc" not in st.session_state
    or st.session_state.preview_profile_proc is None
):
    st.session_state.preview_profile_proc = ProcessedDataset(proc.dataset)

preview_profile_proc = st.session_state.preview_profile_proc


# =============================================================================
# PAGE HEADER
# =============================================================================

st.header("📊 Profile Operations", divider="blue")
st.write(
    """
    Modify the profile structure through ensemble trimming, bin cutting, and
    optional regridding. These operations help remove deployment artifacts and
    side-lobe contamination while preparing data for analysis.
    
    ⚠️ **Note:** Profile operations should be applied AFTER signal quality checks,
    as regridding changes the dataset structure.
    """
)

# Show current processing status
if st.session_state.profile_applied:
    st.success("✅ Profile operations have been applied to this dataset.")

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "✂️ Trim Ends",
        "📡 Side Lobe",
        "🔧 Manual Cut",
        "📐 Regrid",
        "💾 Save/Reset",
    ]
)

# =============================================================================
# TAB 1: TRIM ENDS
# =============================================================================

with tab1:
    st.subheader("Trim Deployment/Recovery Ensembles", divider="orange")
    st.write(
        """
        Remove ensembles at the beginning (deployment) and end (recovery) of the
        deployment period. These often contain unreliable data from when the
        instrument was being deployed or recovered.
        """
    )

    n_ensembles = get_total_ensembles()

    col_left, col_right = st.columns([1, 2])

    with col_left:
        # Range slider for visualization
        ens_range = st.number_input(
            "Display range (ensembles)",
            min_value=10,
            max_value=n_ensembles,
            value=min(50, n_ensembles // 4),
            key="trim_ens_range",
        )

        st.write("---")

        # Trim start
        trim_start = st.slider(
            "Trim from start",
            min_value=0,
            max_value=min(int(ens_range), n_ensembles - 1),
            value=int(st.session_state.trim_start),
            key="trim_start_slider",
        )
        st.session_state.trim_start = trim_start

        # Trim end
        trim_end = st.slider(
            "Trim from end",
            min_value=0,
            max_value=min(int(ens_range), n_ensembles - 1),
            value=int(st.session_state.trim_end),
            key="trim_end_slider",
        )
        st.session_state.trim_end = trim_end

        st.write("---")

        st.write(
            f"**Ensembles to keep:** `{trim_start}` to `{n_ensembles - trim_end - 1}`"
        )
        st.write(
            f"**Total kept:** `{n_ensembles - trim_start - trim_end}` of `{n_ensembles}`"
        )

        # Preview button
        if st.button("👁️ Preview Trim", key="preview_trim"):
            # Reset preview processor and apply trim
            # st.session_state.preview_profile_proc = ProcessedDataset(proc.dataset)
            # preview_profile_proc = st.session_state.preview_profile_proc

            if trim_start > 0 or trim_end > 0:
                runner = preview_profile_proc.get_profile_operation_runner()
                runner.trim_ensembles(
                    start=trim_start if trim_start > 0 else None,
                    end=trim_end if trim_end > 0 else None,
                )
                preview_profile_proc.commit_runner(runner)

            st.session_state.profile_preview_run = True
            st.success("Preview updated!")

    with col_right:
        # Calculate end_ens for plot
        end_ens = n_ensembles - trim_end if trim_end > 0 else n_ensembles
        plot_trim_ends(start_ens=trim_start, end_ens=end_ens, ens_range=int(ens_range))

    if st.session_state.profile_preview_run:
        display_mask = st.session_state.preview_profile_proc.dataset["mask"].values
        plot_heatmap(display_mask, "Revised Mask", colorscale="greys")
# =============================================================================
# TAB 2: SIDE LOBE CONTAMINATION
# =============================================================================

with tab2:
    st.subheader("Cut Bins: Side Lobe Contamination", divider="orange")
    st.write(
        """
        Side-lobe echoes from hard surfaces (sea surface or bottom) can contaminate
        data near these boundaries. This tool calculates the contamination zone
        based on beam angle and transducer depth, masking affected cells.
        """
    )

    orientation = st.session_state.beam_direction

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.write(f"**Beam Direction:** `{orientation}`")
        st.write(f"**Beam Angle:** `{get_beam_angle()}°`")

        st.write("---")

        # Enable side lobe cutting
        apply_side_lobe = st.checkbox(
            "Enable Side Lobe Cutting",
            value=st.session_state.apply_side_lobe,
            key="apply_side_lobe_cb",
        )
        st.session_state.apply_side_lobe = apply_side_lobe

        # Extra cells
        extra_cells = st.number_input(
            "Additional cells to mask",
            min_value=0,
            max_value=10,
            value=int(st.session_state.extra_cells),
            key="extra_cells_input",
        )
        st.session_state.extra_cells = extra_cells

        # Water depth (for downward-looking ADCP)
        if orientation.lower() == "down":
            water_depth = st.number_input(
                "Water column depth (m)",
                min_value=0.0,
                max_value=15000.0,
                value=float(st.session_state.water_depth),
                key="water_depth_input",
            )
            st.session_state.water_depth = water_depth
        else:
            st.session_state.water_depth = None

        st.write("---")

        # Beam selection for visualization
        beam = st.radio(
            "Select beam for visualization",
            options=[1, 2, 3, 4],
            horizontal=True,
            key="sidelobe_beam",
        )
        st.session_state.profile_beam = beam - 1

        # Preview button
        if st.button("👁️ Preview Side Lobe", key="preview_sidelobe"):
            # st.session_state.preview_profile_proc = ProcessedDataset(proc.dataset)
            # preview_profile_proc = st.session_state.preview_profile_proc

            runner = preview_profile_proc.get_profile_operation_runner()

            # Apply trim first if set
            if st.session_state.trim_start > 0 or st.session_state.trim_end > 0:
                runner.trim_ensembles(
                    start=st.session_state.trim_start
                    if st.session_state.trim_start > 0
                    else None,
                    end=st.session_state.trim_end
                    if st.session_state.trim_end > 0
                    else None,
                )

            # Apply side lobe cutting
            if apply_side_lobe:
                runner.cut_bins_side_lobe(
                    orientation=orientation.lower(),
                    water_depth=st.session_state.water_depth,
                    extra_cells=extra_cells,
                )

            preview_profile_proc.commit_runner(runner)
            st.session_state.profile_preview_run = True
            st.success("Preview updated!")

    with col_right:
        # Get data for visualization
        if "echo_intensity" in ds.data_vars:
            echo = ds["echo_intensity"].values
            beam_idx = st.session_state.profile_beam

            # Get mask from preview processor
            preview_ds = st.session_state.preview_profile_proc.dataset
            if "mask" in preview_ds.data_vars:
                preview_mask = preview_ds["mask"].values
            else:
                preview_mask = None

            plot_heatmap(
                echo[beam_idx, :, :],
                title=f"Echo Intensity (Beam {beam_idx + 1})",
                mask_data=preview_mask,
                colorscale="viridis",
            )


# =============================================================================
# TAB 3: MANUAL CUT BINS
# =============================================================================

with tab3:
    st.subheader("Cut Bins: Manual Selection", divider="orange")
    st.write(
        """
        Manually select regions to mask. You can specify ranges of cells and/or
        ensembles to remove. Multiple regions can be added.
        """
    )

    n_cells = get_total_cells()
    n_ensembles = get_total_ensembles()

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.write("**Add Region to Mask:**")

        # Cell range
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            min_cell = st.number_input(
                "Min Cell",
                min_value=0,
                max_value=n_cells - 1,
                value=0,
                key="manual_min_cell",
            )
        with col_c2:
            max_cell = st.number_input(
                "Max Cell",
                min_value=0,
                max_value=n_cells,
                value=n_cells,
                key="manual_max_cell",
            )

        # Ensemble range
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            min_ensemble = st.number_input(
                "Min Ensemble",
                min_value=0,
                max_value=n_ensembles - 1,
                value=0,
                key="manual_min_ens",
            )
        with col_e2:
            max_ensemble = st.number_input(
                "Max Ensemble",
                min_value=0,
                max_value=n_ensembles,
                value=n_ensembles,
                key="manual_max_ens",
            )

        # Add region button
        if st.button("➕ Add Region", key="add_region"):
            region = {
                "min_cell": min_cell,
                "max_cell": max_cell,
                "min_ensemble": min_ensemble,
                "max_ensemble": max_ensemble,
            }
            st.session_state.cut_regions.append(region)
            st.success(
                f"Added region: cells [{min_cell}-{max_cell}], ensembles [{min_ensemble}-{max_ensemble}]"
            )

        st.write("---")

        # Quick actions
        st.write("**Quick Actions:**")

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            delete_cell = st.number_input(
                "Delete entire cell",
                min_value=0,
                max_value=n_cells - 1,
                value=0,
                key="delete_cell",
            )
            if st.button("🗑️ Delete Cell", key="delete_cell_btn"):
                region = {
                    "min_cell": delete_cell,
                    "max_cell": delete_cell + 1,
                    "min_ensemble": 0,
                    "max_ensemble": n_ensembles,
                }
                st.session_state.cut_regions.append(region)
                st.success(f"Added: Delete cell {delete_cell}")

        with col_q2:
            delete_ens = st.number_input(
                "Delete entire ensemble",
                min_value=0,
                max_value=n_ensembles - 1,
                value=0,
                key="delete_ens",
            )
            if st.button("🗑️ Delete Ensemble", key="delete_ens_btn"):
                region = {
                    "min_cell": 0,
                    "max_cell": n_cells,
                    "min_ensemble": delete_ens,
                    "max_ensemble": delete_ens + 1,
                }
                st.session_state.cut_regions.append(region)
                st.success(f"Added: Delete ensemble {delete_ens}")

        st.write("---")

        # Show current regions
        st.write("**Current Regions:**")
        if st.session_state.cut_regions:
            for i, region in enumerate(st.session_state.cut_regions):
                st.write(
                    f"{i + 1}. Cells [{region['min_cell']}-{region['max_cell']}], "
                    f"Ensembles [{region['min_ensemble']}-{region['max_ensemble']}]"
                )

            if st.button("🗑️ Clear All Regions", key="clear_regions"):
                st.session_state.cut_regions = []
                st.success("All regions cleared!")
                st.rerun()
        else:
            st.write("*No regions defined.*")

        st.write("---")

        # Beam selection
        beam = st.radio(
            "Select beam for visualization",
            options=[1, 2, 3, 4],
            horizontal=True,
            key="manual_beam",
        )
        st.session_state.profile_beam = beam - 1

        # Preview button
        if st.button("👁️ Preview Manual Cuts", key="preview_manual"):
            # st.session_state.preview_profile_proc = ProcessedDataset(proc.dataset)
            # preview_profile_proc = st.session_state.preview_profile_proc

            runner = preview_profile_proc.get_profile_operation_runner()

            # Apply trim first if set
            if st.session_state.trim_start > 0 or st.session_state.trim_end > 0:
                runner.trim_ensembles(
                    start=st.session_state.trim_start
                    if st.session_state.trim_start > 0
                    else None,
                    end=st.session_state.trim_end
                    if st.session_state.trim_end > 0
                    else None,
                )

            # Apply side lobe if enabled
            if st.session_state.apply_side_lobe:
                runner.cut_bins_side_lobe(
                    orientation=st.session_state.beam_direction.lower(),
                    water_depth=st.session_state.water_depth,
                    extra_cells=st.session_state.extra_cells,
                )

            # Apply manual cuts
            for region in st.session_state.cut_regions:
                runner.cut_bins_manual(
                    min_cell=region["min_cell"],
                    max_cell=region["max_cell"],
                    min_ensemble=region["min_ensemble"],
                    max_ensemble=region["max_ensemble"],
                )

            preview_profile_proc.commit_runner(runner)
            st.session_state.profile_preview_run = True
            st.success("Preview updated!")

    with col_right:
        # Get data for visualization
        variable = st.selectbox(
            "Select variable to display",
            options=["Velocity", "Echo Intensity", "Correlation", "Percent Good"],
            key="manual_variable",
        )

        # Map variable to dataset field
        var_map = {
            "Velocity": "velocity",
            "Echo Intensity": "echo_intensity",
            "Correlation": "correlation",
            "Percent Good": "percent_good",
        }

        var_name = var_map[variable]
        beam_idx = st.session_state.profile_beam

        if var_name in ds.data_vars:
            data = ds[var_name].values
            if data.ndim == 3:
                plot_data = data[beam_idx, :, :]
            else:
                plot_data = data

            # Get mask from preview processor
            preview_ds = st.session_state.preview_profile_proc.dataset
            if "mask" in preview_ds.data_vars:
                preview_mask = preview_ds["mask"].values
            else:
                preview_mask = None

            plot_heatmap(
                plot_data,
                title=f"{variable} (Beam {beam_idx + 1})",
                mask_data=preview_mask,
                colorscale="balance" if variable == "Velocity" else "viridis",
            )
        else:
            st.warning(f"{variable} data not available.")


# =============================================================================
# TAB 4: REGRID
# =============================================================================

with tab4:
    st.subheader("Regrid Depth Cells", divider="orange")
    st.write(
        """
        When the ADCP buoy has vertical oscillations (greater than cell size),
        the data should be regridded to a regular depth grid using the pressure
        sensor data.
        
        ⚠️ **Warning:** Regridding changes the dataset structure. Quality control
        checks should be applied BEFORE regridding.
        """
    )

    orientation = st.session_state.beam_direction

    col_left, col_right = st.columns([1, 2])

    with col_left:
        # Enable regridding
        apply_regrid = st.checkbox(
            "Enable Regridding",
            value=st.session_state.apply_regrid,
            key="apply_regrid_cb",
        )
        st.session_state.apply_regrid = apply_regrid

        st.write("---")

        # End cell option
        if orientation.lower() == "up":
            end_options = ["cell", "surface", "manual"]
        else:
            end_options = ["cell", "manual"]

        end_cell_option = st.radio(
            "Grid extent option",
            options=end_options,
            key="end_cell_option",
            help="'cell' = to last valid cell, 'surface' = to water surface, 'manual' = to specified depth",
        )

        # Manual boundary
        if end_cell_option == "manual":
            mean_depth = np.mean(get_transducer_depth())
            st.write(f"Mean transducer depth: `{mean_depth:.2f} m`")

            if orientation.lower() == "up":
                boundary_limit = st.number_input(
                    "Boundary depth (m)",
                    min_value=0.0,
                    max_value=float(mean_depth),
                    value=0.0,
                    key="boundary_input",
                )
            else:
                boundary_limit = st.number_input(
                    "Boundary depth (m)",
                    min_value=float(mean_depth),
                    max_value=15000.0,
                    value=float(mean_depth),
                    key="boundary_input",
                )
            st.session_state.boundary_limit = boundary_limit

        st.write("---")

        # Interpolation method
        regrid_method = st.radio(
            "Interpolation method",
            options=["nearest", "linear", "cubic"],
            key="regrid_method",
            help="'nearest' preserves values, 'linear' for simple interpolation, 'cubic' for smoother fit",
        )

        st.write("---")

        # Info box
        st.info(
            """
            **Regridding Notes:**
            - Cell size: `{:.2f} m`
            - Bin 1 distance: `{:.2f} m`
            - Beam angle: `{}°`
            """.format(get_cell_size(), get_bin1_distance(), get_beam_angle())
        )

        # Preview button
        if st.button("👁️ Preview Regrid", key="preview_regrid"):
            if not apply_regrid:
                st.warning("Enable regridding first!")
            else:
                # st.session_state.preview_profile_proc = ProcessedDataset(proc.dataset)
                # preview_profile_proc = st.session_state.preview_profile_proc

                runner = preview_profile_proc.get_profile_operation_runner()

                # Apply trim first if set
                trim_start = st.session_state.trim_start
                trim_end = st.session_state.trim_end
                trimends = None
                n_ens = ds.sizes.get("time", ds.sizes.get("ensemble", 100))
                if trim_start > 0 or trim_end > 0:
                    runner.trim_ensembles(
                        start=trim_start if trim_start > 0 else None,
                        end=trim_end if trim_end > 0 else None,
                    )
                    trimends = (trim_start, trim_end)
                    end_idx = n_ens - trim_end if trim_end > 0 else n_ens
                    start_idx = trim_start if trim_start > 0 else 0
                    trimends = (
                        start_idx,
                        end_idx,
                    )

                # Apply side lobe if enabled
                if st.session_state.apply_side_lobe:
                    runner.cut_bins_side_lobe(
                        orientation=st.session_state.beam_direction.lower(),
                        water_depth=st.session_state.water_depth,
                        extra_cells=st.session_state.extra_cells,
                    )

                # Apply manual cuts
                for region in st.session_state.cut_regions:
                    runner.cut_bins_manual(
                        min_cell=region["min_cell"],
                        max_cell=region["max_cell"],
                        min_ensemble=region["min_ensemble"],
                        max_ensemble=region["max_ensemble"],
                    )

                # Apply regrid
                runner.regrid(
                    method=regrid_method,
                    end_cell_option=end_cell_option,
                    trimends=trimends,
                    orientation=orientation.lower(),
                    boundary_limit=st.session_state.boundary_limit
                    if end_cell_option == "manual"
                    else 0.0,
                )

                preview_profile_proc.commit_runner(runner)
                st.session_state.profile_preview_run = True
                st.success("Preview updated with regridding!")

                # Show dimension changes
                original_cells = get_total_cells()
                new_ds = preview_profile_proc.dataset
                if "depth" in new_ds.dims:
                    new_depths = new_ds.sizes["depth"]
                    st.write(
                        f"**Structure Change:** `{original_cells}` cells → `{new_depths}` depth levels"
                    )

    with col_right:
        # Show preview
        preview_ds = st.session_state.preview_profile_proc.dataset

        if "velocity" in preview_ds.data_vars:
            velocity = preview_ds["velocity"].values
            beam_idx = st.session_state.profile_beam

            if velocity.ndim == 3 and beam_idx < velocity.shape[0]:
                plot_data = velocity[beam_idx, :, :]
            else:
                plot_data = velocity[0, :, :] if velocity.ndim == 3 else velocity

            # Get mask from preview
            if "mask" in preview_ds.data_vars:
                preview_mask = preview_ds["mask"].values
            else:
                preview_mask = None

            title = (
                "Regridded Velocity"
                if st.session_state.apply_regrid
                else "Velocity (Original)"
            )
            plot_heatmap(
                plot_data,
                title=title,
                mask_data=preview_mask,
                colorscale="balance",
            )


# =============================================================================
# TAB 5: SAVE/RESET
# =============================================================================

with tab5:
    st.subheader("Save & Reset", divider="orange")

    col_save, col_reset = st.columns(2)

    with col_save:
        st.write("**💾 Save Profile Operations:**")

        # Summary of operations to apply
        st.write("**Operations to apply:**")
        summary_data = [
            [
                "Trim Start",
                str(st.session_state.trim_start)
                if st.session_state.trim_start > 0
                else "None",
            ],
            [
                "Trim End",
                str(st.session_state.trim_end)
                if st.session_state.trim_end > 0
                else "None",
            ],
            ["Side Lobe Cut", "True" if st.session_state.apply_side_lobe else "False"],
            ["Manual Regions", str(len(st.session_state.cut_regions))],
            ["Regrid", "True" if st.session_state.apply_regrid else "False"],
        ]
        summary_df = pd.DataFrame(summary_data, columns=["Operation", "Value"])
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        st.divider()

        if st.button("📊 Apply Profile Operations", type="primary", key="save_profile"):
            # Get a FRESH runner from the MAIN processor
            runner = proc.get_profile_operation_runner()

            try:
                # Apply trim
                trim_start = st.session_state.trim_start
                trim_end = st.session_state.trim_end
                trimends = None
                n_ens = ds.sizes.get("time", ds.sizes.get("ensemble", 100))
                if trim_start > 0 or trim_end > 0:
                    runner.trim_ensembles(
                        start=trim_start if trim_start > 0 else None,
                        end=trim_end if trim_end > 0 else None,
                    )
                    start_idx = trim_start if trim_start > 0 else 0
                    end_idx = n_ens - trim_end if trim_end > 0 else n_ens
                    trimends = (start_idx, end_idx)

                st.write("Trim Ensembles Complete")

                # Apply side lobe
                if st.session_state.apply_side_lobe:
                    runner.cut_bins_side_lobe(
                        orientation=st.session_state.beam_direction.lower(),
                        water_depth=st.session_state.water_depth,
                        extra_cells=st.session_state.extra_cells,
                    )

                st.write("Side Lobe Complete")
                # Apply manual cuts
                for region in st.session_state.cut_regions:
                    runner.cut_bins_manual(
                        min_cell=region["min_cell"],
                        max_cell=region["max_cell"],
                        min_ensemble=region["min_ensemble"],
                        max_ensemble=region["max_ensemble"],
                    )

                st.write("Cut Region Complete")
                st.write(st.session_state.regrid_method)
                st.write(st.session_state.end_cell_option)
                st.write(trimends)
                st.write(st.session_state.beam_direction.lower())
                # Apply regrid (must be last)
                if st.session_state.apply_regrid:
                    runner.regrid(
                        method=st.session_state.regrid_method,
                        end_cell_option=st.session_state.end_cell_option,
                        trimends=trimends,
                        orientation=st.session_state.beam_direction.lower(),
                        boundary_limit=st.session_state.boundary_limit
                        if st.session_state.end_cell_option == "manual"
                        else 0.0,
                    )

                # Commit the runner to the MAIN processor
                proc.commit_runner(runner)

                st.session_state.profile_applied = True

                # Reset staging processor to match main processor
                st.session_state.preview_profile_proc = ProcessedDataset(proc.dataset)
                st.session_state.profile_preview_run = False

                st.success("✅ Profile operations applied successfully!")

                # Display statistics
                st.write("**📈 Processing Statistics:**")
                if runner.statistics:
                    stats_data = []
                    for stat in runner.statistics:
                        stats_data.append(
                            {
                                "Operation": stat.check_name,
                                "Parameters": str(stat.threshold),
                                "Impact (%)": f"{stat.newly_masked_pct:.2f}",
                                "Cumulative (%)": f"{stat.total_masked_pct:.2f}",
                                "Valid (%)": f"{stat.valid_pct:.2f}",
                            }
                        )
                    stats_df = pd.DataFrame(stats_data)
                    st.dataframe(stats_df, hide_index=True, use_container_width=True)

                # Show modifications (regrid)
                if runner.modifications:
                    st.write("**🔄 Data Modifications:**")
                    for mod in runner.modifications:
                        st.write(
                            f"- {mod.operation}: {mod.original_stats} → {mod.modified_stats}"
                        )

                # Overall statistics
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
                st.error(f"❌ Error applying profile operations: {e}")

        if not st.session_state.profile_applied:
            st.warning("⚠️ Profile operations not yet applied.")

    with col_reset:
        st.write("**🔄 Reset Processing:**")

        if st.button("Reset Profile Operations", key="reset_profile"):
            # Reset the main processor
            proc.reset()

            # Reset staging processor
            st.session_state.preview_profile_proc = ProcessedDataset(proc.dataset)

            # Reset session state
            st.session_state.profile_applied = False
            st.session_state.profile_preview_run = False
            st.session_state.profile_preview_stats = None
            st.session_state.trim_start = 0
            st.session_state.trim_end = 0
            st.session_state.apply_side_lobe = False
            st.session_state.cut_regions = []
            st.session_state.apply_regrid = False

            st.success("✅ Profile operations reset to original values.")
            st.rerun()

        st.info(
            """
            ℹ️ Resetting will:
            - Clear all profile operation masks
            - Reset the main processor to its initial state
            - Clear the staging processor
            
            **Note:** This will also reset any subsequent processing steps.
            """,
            icon="ℹ️",
        )

        st.write("---")

        # Preview local reset
        if st.button("Reset Preview Only", key="reset_preview"):
            st.session_state.preview_profile_proc = ProcessedDataset(proc.dataset)
            st.session_state.profile_preview_run = False
            st.success("Preview reset to current processor state.")
            st.rerun()


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

    st.write("**Profile Operations Status:**")
    st.write(f"- Trim Start: `{st.session_state.trim_start}`")
    st.write(f"- Trim End: `{st.session_state.trim_end}`")
    st.write(f"- Side Lobe: {'✅' if st.session_state.apply_side_lobe else '❌'}")
    st.write(f"- Manual Regions: `{len(st.session_state.cut_regions)}`")
    st.write(f"- Regrid: {'✅' if st.session_state.apply_regrid else '❌'}")

    st.write("---")

    st.write("**Dataset Info:**")
    st.write(f"- Beam Direction: `{st.session_state.beam_direction}`")
    st.write(f"- Ensembles: `{get_total_ensembles()}`")
    st.write(f"- Cells: `{get_total_cells()}`")
    st.write(f"- Beams: `{get_total_beams()}`")

    st.write("---")

    st.write("**Processing Log:**")
    if hasattr(proc, "processing_log") and proc.processing_log:
        for log_entry in proc.processing_log[-5:]:
            st.write(f"- {log_entry}")
    else:
        st.write("*No processing steps applied yet.*")
