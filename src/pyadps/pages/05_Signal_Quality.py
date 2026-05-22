"""
05_Signal_Quality.py - Signal Quality Control Page (Refactored for pyadps v1.0.0)

This page allows users to:
1. Identify noise floor using echo intensity profiles
2. Configure and apply quality control thresholds:
   - Correlation threshold
   - Echo intensity threshold
   - Error velocity threshold
   - Percent good threshold
   - False target detection
3. Enable three-beam mode for problematic beams
4. Preview and compare mask files
5. Fix beam orientation if needed
6. Save processing results to the central ProcessedDataset

Architecture:
- Uses st.session_state.processor (ProcessedDataset) as central state manager
- Uses SignalQualityRunner for interactive preview and threshold application
- All processing is tracked through the processor's reports
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

st.set_page_config(page_title="Signal Quality Tests", page_icon="🔬", layout="wide")

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


def get_default_thresholds() -> dict:
    """Get default threshold values from dataset or use RDI defaults."""
    defaults = {
        "correlation": 64,
        "error_velocity": 2000,
        "echo_intensity": 0,
        "false_target": 50,
        "percent_good": 0,
    }

    # Try to get deployment thresholds from dataset
    if "low_correlation_threshold" in ds.data_vars:
        defaults["correlation"] = int(ds["low_correlation_threshold"].values[0])

    if "error_velocity_maximum" in ds.data_vars:
        defaults["error_velocity"] = int(ds["error_velocity_maximum"].values[0])

    if "false_target_threshold" in ds.data_vars:
        defaults["false_target"] = int(ds["false_target_threshold"].values[0])

    if "percent_good_minimum" in ds.data_vars:
        defaults["percent_good"] = int(ds["percent_good_minimum"].values[0])

    return defaults


def get_fixed_leader_info() -> dict:
    """Get fixed leader information for display."""
    info = {
        "pings": "N/A",
        "beams": get_total_beams(),
        "cells": get_total_cells(),
    }

    # Try accessor method first
    try:
        fl_data = ds.fixed_leader.field(ens=-1)
        info["pings"] = fl_data.get("Pings", info["pings_per_ensemble"])
        info["beams"] = fl_data.get("Beams", info["beams"])
        info["cells"] = fl_data.get("Cells", info["cells"])
    except Exception:
        # Fallback to dataset variables
        if "pings_per_ensemble" in ds.data_vars:
            info["pings"] = int(ds["pings_per_ensemble"].values[0])

    return info


def get_beam_direction() -> str:
    """
    Get beam direction from dataset.

    Uses the mode of ds['beam_direction'] values since the ADCP might be
    placed in wrong direction on ship deck, but will have correct direction
    during actual deployment.

    Returns:
        'Up' if mode is 1, 'Down' if mode is 0, 'Unknown' otherwise.
    """
    try:
        if "beam_direction" in ds.data_vars:
            beam_dir_values = ds["beam_direction"].values
            # Calculate mode (most frequent value)
            unique, counts = np.unique(beam_dir_values, return_counts=True)
            mode_value = unique[np.argmax(counts)]
            if mode_value == 0:
                return "Down"
            elif mode_value == 1:
                return "Up"
        # Fallback to fixed_leader accessor
        sys_config = ds.fixed_leader.system_configuration(ens=-1)
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
    elif value == "Up":
        return "background-color: blue; color: white"
    elif value == "Down":
        return "background-color: orange; color: white"
    return ""


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def plot_heatmap(data: np.ndarray, title: str, colorscale: str = "greys") -> None:
    """Create a heatmap plot for 2D data (cell x ensemble)."""
    n_ensembles = get_total_ensembles()
    n_cells = get_total_cells()

    # Handle missing values
    plot_data = np.where(data == -32768, np.nan, data)

    # Ensure 2D data (cell x ensemble)
    if plot_data.ndim == 3:
        # If 3D (beam, cell, ensemble), take first beam or max across beams
        plot_data = plot_data[0, :, :]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=plot_data,
            x=np.arange(n_ensembles),
            y=np.arange(n_cells),
            colorscale=colorscale,
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Ensemble",
        yaxis_title="Cell",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_noise_floor(dep_ens: int = 0, rec_ens: int = -1) -> None:
    """Plot echo intensity profiles for noise floor identification."""
    if "echo_intensity" not in ds.data_vars:
        st.warning("Echo intensity data not available.")
        return

    echo = ds["echo_intensity"].values  # (beam, cell, time)
    n_beams = echo.shape[0]
    n_cells = echo.shape[1]
    y = np.arange(n_cells)

    # Color schemes
    color_left = [
        "rgb(120, 226, 240)",
        "rgb(57, 168, 290)",
        "rgb(115, 147, 179)",
        "rgb(15, 82, 186)",
    ]
    color_right = [
        "rgb(250, 200, 152)",
        "rgb(255, 165, 0)",
        "rgb(255, 95, 31)",
        "rgb(139, 64, 0)",
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Deployment Ensemble ({dep_ens + 1})",
            f"Recovery Ensemble ({rec_ens + 1 if rec_ens >= 0 else get_total_ensembles() + rec_ens + 1})",
        ],
    )

    # Deployment profiles
    for i in range(min(n_beams, 4)):
        fig.add_trace(
            go.Scatter(
                x=echo[i, :, dep_ens],
                y=y,
                name=f"Beam {i + 1} (D)",
                line=dict(color=color_left[i % len(color_left)]),
            ),
            row=1,
            col=1,
        )

    # Recovery profiles
    for i in range(min(n_beams, 4)):
        fig.add_trace(
            go.Scatter(
                x=echo[i, :, rec_ens],
                y=y,
                name=f"Beam {i + 1} (R)",
                line=dict(color=color_right[i % len(color_right)]),
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        height=600,
        title_text="Echo Intensity Profiles",
    )
    fig.update_xaxes(title="Echo Intensity (count)")
    fig.update_yaxes(title="Cell")

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

# Initialize QC test session state variables
if "qc_initialized" not in st.session_state:
    st.session_state.qc_initialized = True
    st.session_state.qc_applied = False
    st.session_state.qc_preview_run = False

    # Get default thresholds from dataset
    defaults = get_default_thresholds()

    # Threshold values
    st.session_state.correlation_threshold = defaults["correlation"]
    st.session_state.echo_intensity_threshold = defaults["echo_intensity"]
    st.session_state.error_velocity_threshold = defaults["error_velocity"]
    st.session_state.percent_good_threshold = defaults["percent_good"]
    st.session_state.false_target_threshold = defaults["false_target"]

    # Check selections
    st.session_state.apply_correlation = True
    st.session_state.apply_echo_intensity = False
    st.session_state.apply_error_velocity = True
    st.session_state.apply_percent_good = False
    st.session_state.apply_false_target = True

    # Three-beam mode
    st.session_state.threebeam_mode = False
    st.session_state.beam_ignore = None

    # Orientation fix
    st.session_state.beam_direction_current = get_beam_direction()
    st.session_state.beam_direction_modified = False

    # Preview statistics storage
    st.session_state.qc_preview_stats = None

# =============================================================================
# STAGING PROCESSOR FOR PREVIEW
# =============================================================================
# Create a staging processor (preview_qc_proc) for safe experimentation.
# This allows users to preview QC impacts without modifying the main processor.
# Only the final "Save" action applies changes to the main processor.
#
# Pattern: Each page has its own staging processor (preview_qc_proc,
# preview_profile_proc, etc.) to maintain isolation between processing steps.

# Initialize or reset staging processor when needed
if (
    "preview_qc_proc" not in st.session_state
    or st.session_state.preview_qc_proc is None
):
    # Create staging processor from current processed dataset
    st.session_state.preview_qc_proc = ProcessedDataset(proc.dataset)

# Reference to staging processor for convenience
preview_qc_proc = st.session_state.preview_qc_proc


# =============================================================================
# PAGE HEADER
# =============================================================================

st.header("🔬 Signal Quality Control Tests", divider="blue")
st.write(
    """
    Apply quality control tests based on acoustic signal properties. These tests
    follow RDI recommendations for identifying unreliable velocity data due to
    poor acoustic conditions, interference, or instrument issues.
    """
)

# Show current processing status
if st.session_state.qc_applied:
    st.success("✅ Signal quality tests have been applied to this dataset.")
    if st.button("🔄 Reset QC Tests", type="secondary"):
        proc.reset()
        st.session_state.qc_applied = False
        st.session_state.qc_preview_run = False
        st.session_state.qc_preview_stats = None
        st.rerun()

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Noise Floor",
        "⚙️ QC Tests",
        "🗺️ Mask Preview",
        "🔄 Fix Orientation",
        "💾 Save/Reset",
    ]
)

# =============================================================================
# TAB 1: NOISE FLOOR IDENTIFICATION
# =============================================================================

with tab1:
    st.subheader("Noise Floor Identification", divider="orange")
    st.write(
        """
        If the ADCP collected data from the air before deployment or after recovery,
        this data can help estimate the echo intensity threshold. The plots show
        echo intensity profiles from selected ensembles.

        **Noise floor identification:**
        The noise level is typically around 30-40 counts throughout the profile.
        Values significantly above this indicate valid acoustic returns.
        """
    )

    n_ensembles = get_total_ensembles()

    col1, col2 = st.columns(2)

    with col1:
        dep_ens = st.number_input(
            "Deployment Ensemble",
            min_value=1,
            max_value=n_ensembles,
            value=1,
            key="noise_dep_ens",
        )

    with col2:
        rec_ens = st.number_input(
            "Recovery Ensemble",
            min_value=1,
            max_value=n_ensembles,
            value=n_ensembles,
            key="noise_rec_ens",
        )

    # Convert to 0-indexed and cast to int for type safety
    plot_noise_floor(dep_ens=int(dep_ens) - 1, rec_ens=int(rec_ens) - 1)

# =============================================================================
# TAB 2: QC TESTS CONFIGURATION
# =============================================================================

with tab2:
    st.subheader("Quality Control Test Configuration", divider="orange")

    col_info, col_config = st.columns([1, 1])

    with col_info:
        st.write(
            """
            Teledyne RDI recommends these quality control tests, some of which
            can be configured before deployment. The pre-deployment values are
            listed below.

            For more information, refer to *Acoustic Doppler Current Profiler
            Principles of Operation: A Practical Primer* by Teledyne RDI.
            """
        )

        st.divider()

        # Display fixed leader info
        fl_info = get_fixed_leader_info()
        st.write("**📋 Instrument Configuration:**")
        st.write(f"- Number of Pings per Ensemble: `{fl_info['pings']}`")
        st.write(f"- Number of Beams: `{fl_info['beams']}`")
        st.write(f"- Number of Cells: `{fl_info['cells']}`")

        st.divider()

        # Display deployment thresholds
        st.write("**🔧 Deployment Thresholds:**")
        defaults = get_default_thresholds()
        thresh_df = pd.DataFrame(
            [
                ["Correlation", defaults["correlation"]],
                ["Error Velocity", defaults["error_velocity"]],
                ["Echo Intensity", defaults["echo_intensity"]],
                ["False Target", defaults["false_target"]],
                ["Percent Good", defaults["percent_good"]],
            ],
            columns=["Threshold", "Value"],
        )
        st.dataframe(thresh_df, hide_index=True, use_container_width=True)

    with col_config:
        st.write("**Configure Threshold Values:**")

        # Correlation threshold
        st.session_state.apply_correlation = st.checkbox(
            "Apply Correlation Check",
            value=st.session_state.apply_correlation,
            key="cb_correlation",
        )
        st.session_state.correlation_threshold = st.number_input(
            "Correlation Threshold (0-255)",
            min_value=0,
            max_value=255,
            value=st.session_state.correlation_threshold,
            disabled=not st.session_state.apply_correlation,
            key="ni_correlation",
        )

        # Error velocity threshold
        st.session_state.apply_error_velocity = st.checkbox(
            "Apply Error Velocity Check",
            value=st.session_state.apply_error_velocity,
            key="cb_error_velocity",
        )
        st.session_state.error_velocity_threshold = st.number_input(
            "Error Velocity Threshold (mm/s)",
            min_value=0,
            max_value=9999,
            value=st.session_state.error_velocity_threshold,
            disabled=not st.session_state.apply_error_velocity,
            key="ni_error_velocity",
        )

        # Echo intensity threshold
        st.session_state.apply_echo_intensity = st.checkbox(
            "Apply Echo Intensity Check",
            value=st.session_state.apply_echo_intensity,
            key="cb_echo_intensity",
        )
        st.session_state.echo_intensity_threshold = st.number_input(
            "Echo Intensity Threshold (0-255)",
            min_value=0,
            max_value=255,
            value=st.session_state.echo_intensity_threshold,
            disabled=not st.session_state.apply_echo_intensity,
            key="ni_echo_intensity",
        )

        # False target threshold
        st.session_state.apply_false_target = st.checkbox(
            "Apply False Target Check",
            value=st.session_state.apply_false_target,
            key="cb_false_target",
        )
        st.session_state.false_target_threshold = st.number_input(
            "False Target Threshold (0-255)",
            min_value=0,
            max_value=255,
            value=st.session_state.false_target_threshold,
            disabled=not st.session_state.apply_false_target,
            key="ni_false_target",
        )

        # Percent good threshold
        st.session_state.apply_percent_good = st.checkbox(
            "Apply Percent Good Check",
            value=st.session_state.apply_percent_good,
            key="cb_percent_good",
        )
        st.session_state.percent_good_threshold = st.number_input(
            "Percent Good Threshold (0-100)",
            min_value=0,
            max_value=100,
            value=st.session_state.percent_good_threshold,
            disabled=not st.session_state.apply_percent_good,
            key="ni_percent_good",
        )

        st.divider()

        # Three-beam mode
        st.write("**Three-Beam Solution:**")
        st.session_state.threebeam_mode = st.checkbox(
            "Enable Three-Beam Mode",
            value=st.session_state.threebeam_mode,
            help="Use when one beam is known to be problematic",
            key="cb_threebeam",
        )

        if st.session_state.threebeam_mode:
            beam_options: dict[str, int | None] = {
                "None": None,
                "Beam 1": 0,
                "Beam 2": 1,
                "Beam 3": 2,
                "Beam 4": 3,
            }
            selected_beam = st.selectbox(
                "Beam to Ignore",
                options=list(beam_options.keys()),
                index=0,
                key="sb_beam_ignore",
            )
            if selected_beam is not None:
                st.session_state.beam_ignore = beam_options[selected_beam]

    # Preview button - applies to staging processor (preview_qc_proc)
    st.divider()

    if st.button("🔍 Preview QC Impact", type="secondary", key="preview_qc"):
        # Reset staging processor to start fresh from main processor's current state
        # st.session_state.preview_qc_proc = ProcessedDataset(proc.dataset)
        # preview_qc_proc = st.session_state.preview_qc_proc

        # Get a runner from the staging processor
        runner = preview_qc_proc.get_signal_quality_runner()

        try:
            # Apply selected checks
            # Guard against None thresholds (from empty number_input)
            if (
                st.session_state.apply_correlation
                and st.session_state.correlation_threshold is not None
            ):
                runner.correlation(
                    cutoff=int(st.session_state.correlation_threshold),
                    threebeam=st.session_state.threebeam_mode,
                    beam_ignore=st.session_state.beam_ignore,
                )

            if (
                st.session_state.apply_echo_intensity
                and st.session_state.echo_intensity_threshold is not None
            ):
                runner.echo_intensity(
                    cutoff=int(st.session_state.echo_intensity_threshold),
                    threebeam=st.session_state.threebeam_mode,
                    beam_ignore=st.session_state.beam_ignore,
                )

            if (
                st.session_state.apply_error_velocity
                and st.session_state.error_velocity_threshold is not None
            ):
                runner.error_velocity(
                    cutoff=int(st.session_state.error_velocity_threshold),
                )

            if (
                st.session_state.apply_percent_good
                and st.session_state.percent_good_threshold is not None
            ):
                runner.percent_good(
                    cutoff=int(st.session_state.percent_good_threshold),
                    threebeam=st.session_state.threebeam_mode,
                )

            if (
                st.session_state.apply_false_target
                and st.session_state.false_target_threshold is not None
            ):
                runner.false_target(
                    cutoff=int(st.session_state.false_target_threshold),
                    threebeam=st.session_state.threebeam_mode,
                    beam_ignore=st.session_state.beam_ignore,
                )

            # Commit to staging processor (not main processor)
            preview_qc_proc.commit_runner(runner)

            # Store preview statistics
            st.session_state.qc_preview_stats = runner.get_statistics()
            st.session_state.qc_preview_run = True

            st.success(
                "✅ Preview complete! Check the **Mask Preview** tab to see the impact."
            )

            # Display statistics
            if runner.statistics:
                st.write("**📊 QC Impact Preview:**")

                stats_data = []
                for stat in runner.statistics:
                    stats_data.append(
                        {
                            "Check": stat.check_name,
                            "Threshold": stat.threshold,
                            "Pre-Masked (%)": f"{stat.pre_masked_pct:.2f}",
                            "Impact (%)": f"{stat.newly_masked_pct:.2f}",
                            "Cumulative (%)": f"{stat.total_masked_pct:.2f}",
                            "Valid (%)": f"{stat.valid_pct:.2f}",
                        }
                    )

                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, hide_index=True, use_container_width=True)

                # Summary
                final = runner.statistics[-1]
                st.info(
                    f"**Summary:** {final.valid_cells:,} valid cells "
                    f"({final.valid_pct:.2f}%) after QC tests."
                )

        except Exception as e:
            st.error(f"❌ Error during preview: {e}")

    # Local reset button for staging processor
    if st.session_state.qc_preview_run:
        if st.button("🔄 Reset Preview", type="secondary", key="reset_preview"):
            st.session_state.preview_qc_proc = ProcessedDataset(proc.dataset)
            st.session_state.qc_preview_run = False
            st.session_state.qc_preview_stats = None
            st.success(
                "✅ Preview reset. Staging processor restored to main processor state."
            )
            st.rerun()

    # Show current threshold summary
    st.divider()
    st.write("**📋 Current Configuration:**")

    config_data = []
    if st.session_state.apply_correlation:
        config_data.append(
            ["Correlation", str(st.session_state.correlation_threshold), "✅"]
        )
    if st.session_state.apply_error_velocity:
        config_data.append(
            ["Error Velocity", str(st.session_state.error_velocity_threshold), "✅"]
        )
    if st.session_state.apply_echo_intensity:
        config_data.append(
            ["Echo Intensity", str(st.session_state.echo_intensity_threshold), "✅"]
        )
    if st.session_state.apply_false_target:
        config_data.append(
            ["False Target", str(st.session_state.false_target_threshold), "✅"]
        )
    if st.session_state.apply_percent_good:
        config_data.append(
            ["Percent Good", str(st.session_state.percent_good_threshold), "✅"]
        )
    if st.session_state.threebeam_mode:
        beam_str = (
            f"Beam {st.session_state.beam_ignore + 1}"
            if st.session_state.beam_ignore is not None
            else "None"
        )
        config_data.append(["Three-Beam Mode", beam_str, "✅"])

    if config_data:
        config_df = pd.DataFrame(config_data, columns=["Test", "Value", "Enabled"])
        st.dataframe(config_df, hide_index=True, use_container_width=True)
    else:
        st.warning("No QC tests selected.")


# =============================================================================
# TAB 3: MASK PREVIEW
# =============================================================================

with tab3:
    st.subheader("Mask File Preview", divider="orange")
    st.write(
        """
        Compare the original mask (from main processor) with the preview mask 
        (from staging processor after QC tests). Use the **Preview QC Impact** 
        button in the QC Tests tab to generate a preview.
        """
    )

    if not st.session_state.qc_preview_run:
        st.info(
            "💡 **Tip:** Click **Preview QC Impact** in the QC Tests tab first "
            "to see how your threshold settings affect the mask."
        )

    col_left, col_right = st.columns(2)

    if st.button("📊 Display Mask Comparison", key="display_masks"):
        # Get original mask from main processor
        orig_mask = proc.dataset["mask"].values if "mask" in proc.dataset else None

        # Get preview mask from staging processor (preview_qc_proc)
        preview_qc_proc = st.session_state.preview_qc_proc
        preview_mask = (
            preview_qc_proc.dataset["mask"].values
            if "mask" in preview_qc_proc.dataset
            else None
        )

        with col_left:
            st.write("**Original Mask (Main Processor):**")
            st.caption(
                """
                Current mask from the main processor. This includes any 
                previous processing steps (sensor health, etc.) but NOT 
                the QC tests you're previewing.
                """
            )
            if orig_mask is not None:
                # For display, collapse beam dimension if present
                if orig_mask.ndim == 3:
                    display_mask = np.any(orig_mask, axis=0).astype(
                        np.int8
                    )  # Any beam flagged
                else:
                    display_mask = orig_mask
                plot_heatmap(display_mask, "Original Mask", colorscale="greys")

                masked_count = np.sum(orig_mask == 1)
                total_count = orig_mask.size
                st.write(
                    f"Masked: {masked_count:,} / {total_count:,} "
                    f"({100 * masked_count / total_count:.2f}%)"
                )
            else:
                st.warning("Original mask not available.")

        with col_right:
            st.write("**Preview Mask (After QC Tests):**")
            st.caption(
                """
                Preview mask showing the effect of your QC threshold settings.
                This is from the staging processor and has NOT been saved yet.
                """
            )
            if preview_mask is not None and st.session_state.qc_preview_run:
                # For display, collapse beam dimension if present
                if preview_mask.ndim == 3:
                    display_mask = np.any(preview_mask, axis=0).astype(np.int8)
                else:
                    display_mask = preview_mask
                plot_heatmap(
                    display_mask, "Preview Mask (QC Applied)", colorscale="greys"
                )

                masked_count = np.sum(preview_mask == 1)
                total_count = preview_mask.size
                st.write(
                    f"Masked: {masked_count:,} / {total_count:,} "
                    f"({100 * masked_count / total_count:.2f}%)"
                )

                # Show difference
                if orig_mask is not None:
                    orig_masked = np.sum(orig_mask == 1)
                    preview_masked = np.sum(preview_mask == 1)
                    diff = preview_masked - orig_masked
                    diff_pct = 100 * diff / total_count
                    st.metric(
                        "QC Impact",
                        f"{diff:+,} cells",
                        delta=f"{diff_pct:+.2f}%",
                        delta_color="inverse",  # Red for increase (more masked)
                    )
            else:
                st.warning(
                    "Preview mask not available. Click **Preview QC Impact** "
                    "in the QC Tests tab first."
                )


# =============================================================================
# TAB 4: FIX ORIENTATION
# =============================================================================

with tab4:
    st.subheader("Fix Beam Orientation", divider="orange")

    current_direction = st.session_state.beam_direction_current

    st.write(
        f"""
        The current beam orientation is **`{current_direction}`** 


        If the ADCP orientation was incorrectly configured, you can correct it here.
        This affects the calculation of depth for each cell - upward-looking ADCPs 
        measure from the transducer toward the surface, while downward-looking ADCPs 
        measure from the transducer toward the bottom.
        """
    )

    if current_direction == "Up":
        alt_direction = "Down"
    else:
        alt_direction = "Up"

    change_orientation = st.radio(
        f"Change orientation to {alt_direction}?",
        ["No", "Yes"],
        horizontal=True,
        key="orientation_radio",
    )

    if change_orientation == "Yes":
        st.session_state.beam_direction_modified = True
        st.info(f"📝 Orientation will be changed to **{alt_direction}** when saved.")
    else:
        st.session_state.beam_direction_modified = False

    st.info(
        "The orientation is calculated based on mode of recorded values during deployment",
        icon="ℹ️",
    )

# =============================================================================
# TAB 5: SAVE/RESET
# =============================================================================

with tab5:
    st.subheader("Save or Reset Processing", divider="blue")

    st.write(
        """
        **Important:** The Preview in Tab 2 and Mask Preview in Tab 3 use a 
        *staging processor* for safe experimentation. Clicking **Save** below 
        will re-run all configured QC tests on the **main processor** to ensure 
        proper statistics tracking and processing history.
        """
    )

    col_save, col_reset = st.columns([1, 1])

    with col_save:
        st.write("**💾 Save Processing:**")

        # Preview of changes
        st.write("**📋 Changes to Apply:**")

        preview_items = []

        if st.session_state.apply_correlation:
            preview_items.append(
                f"• Correlation check: threshold = {st.session_state.correlation_threshold}"
            )

        if st.session_state.apply_echo_intensity:
            preview_items.append(
                f"• Echo intensity check: threshold = {st.session_state.echo_intensity_threshold}"
            )

        if st.session_state.apply_error_velocity:
            preview_items.append(
                f"• Error velocity check: threshold = {st.session_state.error_velocity_threshold}"
            )

        if st.session_state.apply_percent_good:
            preview_items.append(
                f"• Percent good check: threshold = {st.session_state.percent_good_threshold}"
            )

        if st.session_state.apply_false_target:
            preview_items.append(
                f"• False target check: threshold = {st.session_state.false_target_threshold}"
            )

        if st.session_state.threebeam_mode:
            beam_num = (
                st.session_state.beam_ignore + 1
                if st.session_state.beam_ignore is not None
                else "None"
            )
            preview_items.append(f"• Three-beam mode: ignoring beam {beam_num}")

        if st.session_state.beam_direction_modified:
            current_dir = st.session_state.beam_direction_current
            new_dir = "Down" if current_dir == "Up" else "Up"
            preview_items.append(f"• Orientation change: {current_dir} → {new_dir}")

        if preview_items:
            for item in preview_items:
                st.write(item)
        else:
            st.write("*No QC tests selected.*")

        st.divider()

        if st.button("🔬 Apply Signal Quality Tests", type="primary", key="save_qc"):
            # Get a FRESH runner from the MAIN processor
            # This ensures all statistics and history are properly tracked
            runner = proc.get_signal_quality_runner()

            try:
                # Re-apply all selected checks to the main processor
                # Guard against None thresholds (from empty number_input)
                if (
                    st.session_state.apply_correlation
                    and st.session_state.correlation_threshold is not None
                ):
                    runner.correlation(
                        cutoff=int(st.session_state.correlation_threshold),
                        threebeam=st.session_state.threebeam_mode,
                        beam_ignore=st.session_state.beam_ignore,
                    )

                if (
                    st.session_state.apply_echo_intensity
                    and st.session_state.echo_intensity_threshold is not None
                ):
                    runner.echo_intensity(
                        cutoff=int(st.session_state.echo_intensity_threshold),
                        threebeam=st.session_state.threebeam_mode,
                        beam_ignore=st.session_state.beam_ignore,
                    )

                if (
                    st.session_state.apply_error_velocity
                    and st.session_state.error_velocity_threshold is not None
                ):
                    runner.error_velocity(
                        cutoff=int(st.session_state.error_velocity_threshold),
                    )

                if (
                    st.session_state.apply_percent_good
                    and st.session_state.percent_good_threshold is not None
                ):
                    runner.percent_good(
                        cutoff=int(st.session_state.percent_good_threshold),
                        threebeam=st.session_state.threebeam_mode,
                    )

                if (
                    st.session_state.apply_false_target
                    and st.session_state.false_target_threshold is not None
                ):
                    runner.false_target(
                        cutoff=int(st.session_state.false_target_threshold),
                        threebeam=st.session_state.threebeam_mode,
                        beam_ignore=st.session_state.beam_ignore,
                    )

                # Commit the runner to the MAIN processor
                # This captures all statistics and processing history
                proc.commit_runner(runner)

                # Handle orientation change (if applicable)
                if st.session_state.beam_direction_modified:
                    current_dir = st.session_state.beam_direction_current
                    new_dir = "Down" if current_dir == "Up" else "Up"
                    proc.dataset.attrs["beam_direction"] = new_dir
                    st.session_state.beam_direction_current = new_dir
                    st.session_state.beam_direction_modified = False

                st.session_state.qc_applied = True

                # Reset staging processor to match main processor
                st.session_state.preview_qc_proc = ProcessedDataset(proc.dataset)
                st.session_state.qc_preview_run = False

                st.success("✅ Signal quality tests applied successfully!")

                # Display summary
                st.write("**📊 Processing Summary:**")

                summary_data = [
                    [
                        "Correlation Check",
                        "True" if st.session_state.apply_correlation else "False",
                    ],
                    [
                        "Echo Intensity Check",
                        "True" if st.session_state.apply_echo_intensity else "False",
                    ],
                    [
                        "Error Velocity Check",
                        "True" if st.session_state.apply_error_velocity else "False",
                    ],
                    [
                        "Percent Good Check",
                        "True" if st.session_state.apply_percent_good else "False",
                    ],
                    [
                        "False Target Check",
                        "True" if st.session_state.apply_false_target else "False",
                    ],
                    [
                        "Three-Beam Mode",
                        "True" if st.session_state.threebeam_mode else "False",
                    ],
                ]

                summary_df = pd.DataFrame(summary_data, columns=["Test", "Status"])
                styled_df = summary_df.style.map(status_color_map, subset=["Status"])
                st.write(styled_df.to_html(), unsafe_allow_html=True)

                # Show final statistics from runner
                st.write("---")
                st.write("**📈 QC Test Statistics:**")

                if runner.statistics:
                    stats_data = []
                    for stat in runner.statistics:
                        stats_data.append(
                            {
                                "Check": stat.check_name,
                                "Threshold": stat.threshold,
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
                st.error(f"❌ Error applying signal quality tests: {e}")

        if not st.session_state.qc_applied:
            st.warning("⚠️ Signal quality tests not yet applied.")

    with col_reset:
        st.write("**🔄 Reset Processing:**")

        if st.button("Reset QC Tests", key="reset_qc"):
            # Reset the main processor
            proc.reset()

            # Reset staging processor
            st.session_state.preview_qc_proc = ProcessedDataset(proc.dataset)

            # Reset session state
            st.session_state.qc_applied = False
            st.session_state.qc_preview_run = False
            st.session_state.qc_preview_stats = None
            st.session_state.beam_direction_modified = False

            st.success("✅ Signal quality tests reset to original values.")
            st.rerun()

        st.info(
            """
            ℹ️ Resetting will:
            - Clear all QC masks applied in this step
            - Reset the main processor to its initial state
            - Clear the staging processor
            
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

    st.write("**QC Tests Status:**")
    st.write(f"- Correlation: {'✅' if st.session_state.apply_correlation else '❌'}")
    st.write(
        f"- Echo Intensity: {'✅' if st.session_state.apply_echo_intensity else '❌'}"
    )
    st.write(
        f"- Error Velocity: {'✅' if st.session_state.apply_error_velocity else '❌'}"
    )
    st.write(f"- Percent Good: {'✅' if st.session_state.apply_percent_good else '❌'}")
    st.write(f"- False Target: {'✅' if st.session_state.apply_false_target else '❌'}")
    st.write(f"- Three-Beam: {'✅' if st.session_state.threebeam_mode else '❌'}")

    st.write("---")

    st.write("**Processing Log:**")
    if hasattr(proc, "processing_log") and proc.processing_log:
        for log_entry in proc.processing_log[-5:]:  # Show last 5 entries
            st.write(f"- {log_entry}")
    else:
        st.write("*No processing steps applied yet.*")
