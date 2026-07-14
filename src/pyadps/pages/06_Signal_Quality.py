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
    fig.update_yaxes(autorange="reversed")  # Cell 0 at top

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
    st.session_state.echo_intensity_per_beam_threshold = None
    st.session_state.error_velocity_threshold = defaults["error_velocity"]
    st.session_state.percent_good_threshold = defaults["percent_good"]
    st.session_state.false_target_threshold = defaults["false_target"]

    # Check selections (all off by default — user opts in)
    st.session_state.apply_correlation = False
    st.session_state.apply_echo_intensity = False
    st.session_state.apply_error_velocity = False
    st.session_state.apply_percent_good = False
    st.session_state.apply_false_target = False

    # Three-beam mode (percent good only; active by default when shown)
    st.session_state.threebeam_mode = True
    st.session_state._pg_active_prev = False
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
# CALLBACKS
# Mutating state in on_click callbacks (rather than calling st.rerun() inside
# an `if st.button():` block) avoids resetting the active tab back to the
# first one - a known Streamlit limitation made worse when conditional
# content sits before st.tabs(). See: github.com/streamlit/streamlit/issues/6257
# =============================================================================


def _reset_qc_top():
    proc.reset()
    st.session_state.qc_applied = False
    st.session_state.qc_preview_run = False
    st.session_state.qc_preview_stats = None


def _reset_qc_preview():
    st.session_state.preview_qc_proc = ProcessedDataset(proc.dataset)
    st.session_state.qc_preview_run = False
    st.session_state.qc_preview_stats = None


def _reset_qc_full():
    proc.reset()
    st.session_state.preview_qc_proc = ProcessedDataset(proc.dataset)
    st.session_state.qc_applied = False
    st.session_state.qc_preview_run = False
    st.session_state.qc_preview_stats = None
    st.session_state.beam_direction_modified = False


# =============================================================================
# PAGE HEADER
# =============================================================================

_, _col_fname = st.columns([5, 1])
with _col_fname:
    st.caption(f"📂 {st.session_state.get('fname', 'No file selected')}")
st.header("🔬 Signal Quality Control Tests", divider="blue")
st.write(
    """
    Apply quality control tests based on acoustic signal properties. These tests
    follow RDI recommendations for identifying unreliable velocity data due to
    poor acoustic conditions, interference, or instrument issues.
    """
)

# Show current processing status.
# Wrapped in an always-drawn container: an *unconditional* container before
# st.tabs() doesn't break tab state, but conditional content directly in the
# main body (appearing/disappearing across reruns) does.
status_container = st.container()
with status_container:
    if st.session_state.qc_applied:
        st.success("✅ Signal quality tests have been applied to this dataset.")
        st.button(
            "🔄 Reset QC Tests",
            type="secondary",
            key="reset_qc_top",
            on_click=_reset_qc_top,
        )

# =============================================================================
# TABS
# =============================================================================

tab1, tab_advisor, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Noise Floor",
        "🎯 PG Threshold Advisor",
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
    st.caption(
        "🟡 **Optional** — Only needed when you want to derive an echo intensity "
        "threshold from in-air data. Most deployments do not require this step unless "
        "sensor contamination is suspected."
    )
    st.write(
        """
        If the ADCP collected data in air before deployment and/or after recovery,
        that data can be used to estimate the **echo intensity noise floor** — the
        minimum signal level below which an acoustic return is indistinguishable
        from electronic noise. Thresholds derived here can be sent directly to the
        **Echo Intensity Check** in the QC Tests tab.
        """
    )

    with st.expander("ℹ️ Methodology and how to use this tab"):
        st.markdown(
            """
**Why use in-air data?**

When the ADCP is powered on in air, it transmits acoustic pulses that find no
water to return from. The recorded echo intensities therefore reflect only the
instrument's electronic noise floor. A transducer face contaminated by debris
or biological fouling will show a measurably higher noise floor, making this a
useful diagnostic tool alongside the actual QC check.

---

**Elevated echo in the first cells (exclude these)**

The first several depth cells typically show higher echo intensity than the
flat noise floor further along the profile. The reasons could include near-field
acoustic effects, scattering from air particles, and the speed-of-sound mismatch
between air and water shortening the effective blanking distance.

Exclude these cells using the **Min Cell** input. The default of 10 is a
reasonable starting point; adjust based on where your profile visibly flattens.

---

**Identifying the flat region**

Beyond the elevated near-transducer zone, echo intensity in air becomes
approximately flat with depth. This plateau is the true noise floor. Set
**Max Cell** to the last cell of this plateau (usually the deepest bin
recorded). If the profile is not flat — for example, it still trends downward
— you may be looking at water returns rather than air, and the ensemble is not
suitable for noise floor estimation.

---

**Per-beam thresholds vs. a single threshold**

Each beam has its own transducer and analogue front-end, so noise floors can
differ slightly between beams. Use **Per-beam threshold (4 values)** in the Send
section below when beams differ noticeably. A single threshold is appropriate
when all beams are consistent.

---

**Three-Beam Mode and Beam to Ignore (set in QC Tests tab)**

- **Beam to Ignore**: If one beam shows a consistently higher noise floor than
  the others — suggesting a fouled or damaged transducer — exclude it from
  correlation, echo intensity, and false target checks using the *Beam to
  Ignore* selector in the QC Tests tab.
- **Three-Beam Mode**: Only appears (and only applies) when *Apply Percent
  Good Check* is enabled, and is on by default. It sums PG1 (percent 3-beam
  solutions) and PG4 (percent 4-beam solutions); disabled, it uses PG4 alone.
  It has no effect on correlation, echo intensity, or false target.

---

**Workflow summary**

1. Select a deployment or recovery ensemble where the ADCP was in air.
2. Tick the checkbox below the corresponding plot.
3. Adjust Min Cell (skip ringing) and Max Cell (end of flat region).
4. Review the per-beam maxima in the statistics table.
5. Use the **Send** section to push the derived threshold(s) to the QC Tests tab.
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

    n_cells = get_total_cells()
    default_max_cell = max(n_cells - 1, 0)
    default_min_cell = 10 if n_cells > 9 else default_max_cell

    col_dep_ctrl, col_rec_ctrl = st.columns(2)

    with col_dep_ctrl:
        st.checkbox(
            "Deployment ensemble is in air",
            value=False,
            key="noise_dep_compute",
            help="Tick to compute noise floor statistics from the deployment ensemble.",
        )
        if st.session_state.noise_dep_compute:
            st.number_input(
                "Min cell",
                min_value=0,
                max_value=default_max_cell,
                value=default_min_cell,
                key="noise_dep_min_cell",
                help="Exclude cells below this number (transducer ringing region).",
            )
            st.number_input(
                "Max cell",
                min_value=0,
                max_value=default_max_cell,
                value=default_max_cell,
                key="noise_dep_max_cell",
                help="Last cell to include in the statistics.",
            )

    with col_rec_ctrl:
        st.checkbox(
            "Recovery ensemble is in air",
            value=False,
            key="noise_rec_compute",
            help="Tick to compute noise floor statistics from the recovery ensemble.",
        )
        if st.session_state.noise_rec_compute:
            st.number_input(
                "Min cell",
                min_value=0,
                max_value=default_max_cell,
                value=default_min_cell,
                key="noise_rec_min_cell",
                help="Exclude cells below this number (transducer ringing region).",
            )
            st.number_input(
                "Max cell",
                min_value=0,
                max_value=default_max_cell,
                value=default_max_cell,
                key="noise_rec_max_cell",
                help="Last cell to include in the statistics.",
            )

    dep_active = st.session_state.noise_dep_compute
    rec_active = st.session_state.noise_rec_compute

    if (dep_active or rec_active) and "echo_intensity" not in ds.data_vars:
        st.warning("Echo intensity data not available.")
    elif dep_active or rec_active:
        echo = ds["echo_intensity"].values  # (beam, cell, time)
        n_beams = echo.shape[0]
        dep_idx = int(dep_ens) - 1
        rec_idx = int(rec_ens) - 1

        # Compute per-beam maxima for use in stats table and send section
        dep_max_per_beam: list[float] = []
        rec_max_per_beam: list[float] = []

        rows = []
        for b in range(n_beams):
            row: dict = {"Beam": f"Beam {b + 1}"}

            if dep_active:
                c0_dep = int(st.session_state.noise_dep_min_cell)
                c1_dep = int(st.session_state.noise_dep_max_cell) + 1
                dep_max = int(np.max(echo[b, c0_dep:c1_dep, dep_idx]))
                row["Deployment Max"] = dep_max
                dep_max_per_beam.append(float(dep_max))

            if rec_active:
                c0_rec = int(st.session_state.noise_rec_min_cell)
                c1_rec = int(st.session_state.noise_rec_max_cell) + 1
                rec_max = int(np.max(echo[b, c0_rec:c1_rec, rec_idx]))
                row["Recovery Max"] = rec_max
                rec_max_per_beam.append(float(rec_max))

            if dep_active and rec_active:
                row["Mean"] = round(
                    (row["Deployment Max"] + row["Recovery Max"]) / 2, 1
                )

            rows.append(row)

        # Summary row: mean of all beams' max values per column
        summary: dict = {"Beam": "Mean of all beams"}
        if dep_active:
            summary["Deployment Max"] = round(float(np.mean(dep_max_per_beam)), 1)
        if rec_active:
            summary["Recovery Max"] = round(float(np.mean(rec_max_per_beam)), 1)
        if dep_active and rec_active:
            summary["Mean"] = round(
                (summary["Deployment Max"] + summary["Recovery Max"]) / 2, 1
            )
        rows.append(summary)

        cell_range_note = ""
        if dep_active:
            cell_range_note += (
                f"Deployment: cells {st.session_state.noise_dep_min_cell}"
                f"–{st.session_state.noise_dep_max_cell}"
            )
        if rec_active:
            sep = " | " if dep_active else ""
            cell_range_note += (
                f"{sep}Recovery: cells {st.session_state.noise_rec_min_cell}"
                f"–{st.session_state.noise_rec_max_cell}"
            )

        st.write(f"**Noise floor statistics ({cell_range_note}):**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # ── Send to QC Tests ─────────────────────────────────────────────────
        st.divider()
        st.write("**Send to Echo Intensity Threshold:**")

        send_option = st.radio(
            "Threshold mode",
            options=["Single threshold", "Per-beam threshold (4 values)"],
            index=1,
            horizontal=True,
            key="noise_send_option",
            label_visibility="collapsed",
        )

        if send_option == "Single threshold":
            all_maxima = dep_max_per_beam + rec_max_per_beam
            suggested = int(round(np.mean(all_maxima))) if all_maxima else 0
            val = st.number_input(
                "Threshold value",
                min_value=0,
                max_value=255,
                value=suggested,
                key="noise_single_input",
                help="Mean of all beam maxima across active ensembles.",
            )
            if st.button(
                "Send to Echo Intensity Threshold →",
                key="noise_send_single",
            ):
                st.session_state.echo_intensity_threshold = int(val)
                st.session_state.echo_intensity_per_beam_threshold = None
                st.session_state.apply_echo_intensity = True
                st.session_state.ei_mode_radio = "Single threshold"
                st.success(
                    f"Single threshold **{int(val)}** sent. "
                    "Go to the **QC Tests** tab to preview and apply."
                )

        else:  # Per-beam threshold
            # Initialise session state for the 4 editable inputs
            for _b in range(n_beams):
                if f"noise_pb_input_{_b}" not in st.session_state:
                    st.session_state[f"noise_pb_input_{_b}"] = 0.0

            # Capture loop variables for closures
            _dep = list(dep_max_per_beam)
            _rec = list(rec_max_per_beam)
            _nb = n_beams

            def _fill_dep() -> None:
                for _b in range(_nb):
                    st.session_state[f"noise_pb_input_{_b}"] = float(_dep[_b])

            def _fill_avg() -> None:
                for _b in range(_nb):
                    st.session_state[f"noise_pb_input_{_b}"] = round(
                        (_dep[_b] + _rec[_b]) / 2, 1
                    )

            def _fill_rec() -> None:
                for _b in range(_nb):
                    st.session_state[f"noise_pb_input_{_b}"] = float(_rec[_b])

            col_fd, col_fa, col_fr = st.columns(3)
            with col_fd:
                st.button(
                    "Fill: deployment",
                    on_click=_fill_dep,
                    disabled=not dep_active,
                    key="noise_fill_dep",
                )
            with col_fa:
                st.button(
                    "Fill: avg (dep + rec)",
                    on_click=_fill_avg,
                    disabled=not (dep_active and rec_active),
                    key="noise_fill_avg",
                )
            with col_fr:
                st.button(
                    "Fill: recovery",
                    on_click=_fill_rec,
                    disabled=not rec_active,
                    key="noise_fill_rec",
                )

            beam_cols = st.columns(n_beams)
            for _b, _col in enumerate(beam_cols):
                with _col:
                    st.number_input(
                        f"Beam {_b + 1}",
                        min_value=0.0,
                        max_value=255.0,
                        step=1.0,
                        key=f"noise_pb_input_{_b}",
                    )

            if st.button(
                "Send to Echo Intensity Threshold →",
                key="noise_send_per_beam",
            ):
                pb_values = [
                    float(st.session_state[f"noise_pb_input_{_b}"])
                    for _b in range(n_beams)
                ]
                st.session_state.echo_intensity_per_beam_threshold = pb_values
                st.session_state.apply_echo_intensity = True
                st.session_state.ei_mode_radio = "Per-beam threshold (4 values)"
                for _b, _v in enumerate(pb_values):
                    st.session_state[f"qc_ei_pb_{_b}"] = _v
                st.success(
                    f"Per-beam thresholds "
                    f"**{[int(v) for v in pb_values]}** sent. "
                    "Go to the **QC Tests** tab to preview and apply."
                )

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

        with st.expander("ℹ️ How to use the False Target threshold"):
            st.markdown(
                """
The **False Target** (WA command) threshold set during deployment compares echo
intensities across beams to detect fish or debris within a depth cell. Because
post-collection data is already in Earth coordinates, individual beams cannot be
selectively flagged — the entire depth cell is rejected instead. The check here
always compares `max − min` across beams; any ensemble where `max − min >
threshold` is rejected entirely. The following scenarios guide how to set the
threshold:

**1. Apply a stricter threshold than the deployment setting**
If the deployment WA threshold was high (e.g. 100) and you want a tighter check
(e.g. 30), enter the new value in the *False Target Threshold* field above.

**2. Known faulty beam**
If one beam is permanently faulty (identifiable from correlation, echo
intensity, or percent-good diagnostics), select it in the *Beam to Ignore*
dropdown. The false target check then runs on the three remaining beams using
`max − min`, which is useful for applying a stricter threshold than the
deployment setting to the surviving beams.

Note: the instrument's own onboard WA check runs per ping, in beam
coordinates, before any ensemble averaging. There is no reliable post-collection
equivalent of its 3-beam leniency — by the time ensemble-averaged data reaches
this check, the single-ping resolution that leniency depends on is already
gone. Use *Beam to Ignore* for a beam you already know is bad; there is no
automatic-detection option here.

If a false target is detected at depth cell *x*, the adjacent cell *x+1* is
also flagged, because the ADCP samples echo intensity near the end of each
depth cell.
"""
            )

        with st.expander("ℹ️ How to use Beam to Ignore"):
            st.markdown(
                """
**Beam to Ignore** excludes one specific, known-bad beam from the
**Correlation**, **Echo Intensity**, and **False Target** checks. It has no
effect on **Percent Good** — percent good is a single combined value per
cell (not per beam), so there is no individual beam to drop from it.

Use it when a beam is identifiable as permanently faulty — a fouled or
misaligned transducer, for example — from correlation, echo intensity, or
percent-good diagnostics across the deployment. Once selected, the excluded
beam is dropped from the comparison in each of the three checks above; the
remaining beams are checked as usual.

This is a manual setting: you name the beam. There is no automatic
detection of which beam is bad.
"""
            )

        with st.expander("ℹ️ How to use Three-Beam Solution"):
            st.markdown(
                """
**Three-Beam Solution** only appears, and only applies, when *Apply Percent
Good Check* is enabled — it has no effect on Correlation, Echo Intensity, or
False Target.

It controls how the Percent Good value is combined from the instrument's
four percent-good components:

- **Enabled (default)**: uses PG1 (percent 3-beam solutions) + PG4 (percent
  4-beam solutions). This follows RDI's standard recommendation and accepts
  cells where the instrument used either a 3-beam or 4-beam solution.
- **Disabled**: uses PG4 only, requiring a 4-beam solution. This is
  stricter, and will mask more cells than the default.

Turning off *Apply Percent Good Check* also turns this off; turning it back
on restores the default (enabled).
"""
            )

    with col_config:
        st.write("**Configure Threshold Values:**")

        # Correlation threshold
        st.session_state.apply_correlation = st.checkbox(
            "Apply Correlation Check",
            value=st.session_state.apply_correlation,
            key="cb_correlation",
        )
        st.caption(
            "🟡 Optional — The instrument applies a correlation threshold of 64 in "
            "real time (WC command). The factory default is well-calibrated; "
            "post-processing with the same value is redundant. Only enable if you "
            "need a stricter threshold than the deployment setting."
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
        st.caption(
            "🟢 Recommended — The pre-deployment default (2000 mm/s, EVT command) "
            "is too lenient for most deployments and passes nearly all data. A "
            "tighter threshold (e.g. 50–150 mm/s) removes physically implausible "
            "ensembles that the instrument accepted in real time."
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
        st.caption(
            "🟡 Optional — Requires a noise floor estimate from in-air data "
            "(see the Noise Floor tab). Apply only if sensor contamination is "
            "suspected or if you want to mask returns weaker than the noise floor."
        )
        if st.session_state.apply_echo_intensity:
            ei_mode = st.radio(
                "Echo intensity mode",
                options=["Single threshold", "Per-beam threshold (4 values)"],
                key="ei_mode_radio",
                horizontal=True,
                label_visibility="collapsed",
            )
            if ei_mode == "Single threshold":
                st.session_state.echo_intensity_per_beam_threshold = None
                st.session_state.echo_intensity_threshold = st.number_input(
                    "Echo Intensity Threshold (0-255)",
                    min_value=0,
                    max_value=255,
                    value=st.session_state.echo_intensity_threshold,
                    key="ni_echo_intensity",
                )
            else:
                _pb_init: list[float] = st.session_state.get(
                    "echo_intensity_per_beam_threshold"
                ) or [float(st.session_state.echo_intensity_threshold)] * 4
                for _b in range(4):
                    if f"qc_ei_pb_{_b}" not in st.session_state:
                        st.session_state[f"qc_ei_pb_{_b}"] = _pb_init[_b]
                _ei_cols = st.columns(4)
                for _b, _col in enumerate(_ei_cols):
                    with _col:
                        st.number_input(
                            f"Beam {_b + 1}",
                            min_value=0.0,
                            max_value=255.0,
                            step=1.0,
                            key=f"qc_ei_pb_{_b}",
                        )
                st.session_state.echo_intensity_per_beam_threshold = [
                    float(st.session_state[f"qc_ei_pb_{_b}"]) for _b in range(4)
                ]

        # False target threshold
        st.session_state.apply_false_target = st.checkbox(
            "Apply False Target Check",
            value=st.session_state.apply_false_target,
            key="cb_false_target",
        )
        st.caption(
            "🟡 Optional — The instrument applies a false target filter in real "
            "time (WA command, default 50). Post-processing is only useful to apply "
            "a stricter threshold than the deployment setting or to override lenient "
            "3-beam logic. See the info panel for detailed scenarios."
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
        # Read from a persisted flag, not st.session_state.apply_percent_good
        # directly: the PG Threshold Advisor tab's "Apply to QC Tests" button
        # sets apply_percent_good=True via an on_click callback that runs
        # before this script body, so by the time we'd read it here it would
        # already reflect the new value and the False->True transition would
        # be invisible.
        _percent_good_was_active = st.session_state.get("_pg_active_prev", False)
        st.session_state.apply_percent_good = st.checkbox(
            "Apply Percent Good Check",
            value=st.session_state.apply_percent_good,
            key="cb_percent_good",
        )
        st.caption(
            "🔴 Important — The primary reliability indicator for ensemble-averaged "
            "data. The pre-deployment default varies by configuration and is often "
            "not well-tuned. Use the PG Threshold Advisor tab to derive a threshold "
            "matched to your deployment depth and precision requirements."
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

        # Three-beam mode (percent good only): shown only while Percent Good
        # is active, active by default whenever it (re)appears, and deactivated
        # whenever Percent Good is turned off.
        if st.session_state.apply_percent_good:
            if not _percent_good_was_active:
                # Widget already owns key "cb_threebeam" from a prior render;
                # once that key exists, value= below is ignored on rerun, so
                # the reset must target the widget's own key directly.
                st.session_state.cb_threebeam = True
                st.session_state.threebeam_mode = True
            st.write("**Three-Beam Solution:**")
            st.session_state.threebeam_mode = st.checkbox(
                "Enable Three-Beam Mode",
                value=st.session_state.threebeam_mode,
                help="Percent Good = PG1 (percent 3-beam solutions) + PG4 "
                "(percent 4-beam solutions). Disable to use PG4 (4-beam "
                "solutions) only.",
                key="cb_threebeam",
            )
        else:
            st.session_state.cb_threebeam = False
            st.session_state.threebeam_mode = False
        st.session_state._pg_active_prev = st.session_state.apply_percent_good

        st.divider()

        # Beam to ignore (independent of three-beam mode)
        st.write("**Beam to Ignore:**")
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
            index=0 if st.session_state.beam_ignore is None
            else st.session_state.beam_ignore + 1,
            key="sb_beam_ignore",
            label_visibility="collapsed",
            help="Exclude this beam from all QC checks. "
            "Useful when a beam is permanently faulty.",
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
                    beam_ignore=st.session_state.beam_ignore,
                )

            if st.session_state.apply_echo_intensity:
                _ei_pb = st.session_state.get("echo_intensity_per_beam_threshold")
                if _ei_pb is not None:
                    runner.echo_intensity(
                        cutoff=_ei_pb,
                        beam_ignore=st.session_state.beam_ignore,
                    )
                elif st.session_state.echo_intensity_threshold is not None:
                    runner.echo_intensity(
                        cutoff=int(st.session_state.echo_intensity_threshold),
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
        st.button(
            "🔄 Reset Preview",
            type="secondary",
            key="reset_preview",
            on_click=_reset_qc_preview,
        )

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
        _ei_pb = st.session_state.get("echo_intensity_per_beam_threshold")
        if _ei_pb is not None:
            _ei_val = str([int(v) for v in _ei_pb])
        else:
            _ei_val = str(st.session_state.echo_intensity_threshold)
        config_data.append(["Echo Intensity", _ei_val, "✅"])
    if st.session_state.apply_false_target:
        config_data.append(
            ["False Target", str(st.session_state.false_target_threshold), "✅"]
        )
    if st.session_state.apply_percent_good:
        config_data.append(
            ["Percent Good", str(st.session_state.percent_good_threshold), "✅"]
        )
    if st.session_state.threebeam_mode:
        config_data.append(["Three-Beam Mode", "Enabled", "✅"])
    if st.session_state.beam_ignore is not None:
        config_data.append(
            ["Beam to Ignore", f"Beam {st.session_state.beam_ignore + 1}", "✅"]
        )

    if config_data:
        config_df = pd.DataFrame(config_data, columns=["Test", "Value", "Enabled"])
        st.dataframe(config_df, hide_index=True, use_container_width=True)
    else:
        st.warning("No QC tests selected.")


# =============================================================================
# TAB: PERCENT GOOD THRESHOLD ADVISOR
# =============================================================================


@st.cache_data
def _load_noise_coefficients() -> dict:
    """Load velocity_noise_coefficients.json (cached for the session)."""
    from pyadps.processing.signal_quality import _load_adcp_coefficients

    return _load_adcp_coefficients()


with tab_advisor:
    st.subheader("Percent Good Threshold Advisor", divider="orange")
    st.caption(
        "🟢 **Recommended** — The pre-deployment percent good setting may not reflect "
        "your actual ensemble averaging conditions. Use this tool to derive a "
        "scientifically justified threshold before applying the Percent Good Check."
    )
    st.write(
        """
        This tool estimates the recommended **Percent Good** cutoff threshold
        based on the acoustic noise characteristics of the ADCP. It uses an
        exponential noise model to compute the single-ping velocity standard
        deviation as a function of depth range, then determines how many valid
        pings per ensemble are required to achieve a target current precision.
        """
    )

    st.warning(
        "**Use with caution.** The computed threshold is a guideline, not a strict "
        "rule. The appropriate precision depends entirely on your scientific "
        "requirements — there is no universally correct value. Thresholds computed "
        "here assume the noise model fits your deployment conditions.",
        icon="⚠️",
    )

    st.caption(
        "Noise model source: *ADCP Coordinate Transformation — Formulas and "
        "Calculations*, Teledyne RDI."
    )

    with st.expander("ℹ️ Methodological notes", expanded=False):
        st.markdown(
            "**Depth range**\n\n"
            "The calculation uses the full profiling depth (bin size × number of cells) "
            "by default. This is intentionally conservative — it returns the worst-case "
            "noise estimate for the deepest bin. Shallower bins will have a lower "
            "standard deviation and will therefore meet the same precision target with "
            "fewer valid pings. If a large portion of the profile is unusable due to "
            "surface backscatter (acoustic pings reflecting off the surface before "
            "reaching the full range), consider reducing the depth range to match the "
            "reliable data extent using the **Use custom instrument parameters** option "
            "below.\n\n"
            "**Ping independence**\n\n"
            "The ensemble standard deviation is computed as σ_single / √N, which "
            "assumes that the noise on successive pings is statistically independent. "
            "This holds for typical oceanographic deployments where pings within an "
            "ensemble are separated by at least a few seconds. It may not hold for "
            "very high ping-rate configurations where successive pings sample nearly "
            "the same acoustic volume."
        )

    st.divider()

    noise_coeffs = _load_noise_coefficients()
    _available_freqs = sorted(int(k) for k in noise_coeffs)

    col_inputs, col_results = st.columns([1, 1], gap="large")

    with col_inputs:
        desired_std = st.number_input(
            "Desired Standard Deviation (cm/s)",
            min_value=0.01,
            max_value=50.0,
            value=1.00,
            step=0.10,
            format="%.2f",
            help="Target current measurement precision in cm/s.",
            key="advisor_desired_std",
        )

        if desired_std < 0.2:
            st.warning(
                "Values below **0.2 cm/s** are below the long-term bias of the "
                "ADCP. Instrument bias will dominate the uncertainty at this level, "
                "so there is little scientific benefit in reducing the standard "
                "deviation further.",
                icon="⚠️",
            )

        use_custom = st.checkbox(
            "Use custom instrument parameters",
            value=False,
            key="advisor_use_custom",
            help=(
                "Override the parameters auto-read from the loaded file. "
                "Useful for comparing different configurations or when "
                "Fixed Leader fields are not available in the dataset."
            ),
        )

        # Try to auto-detect instrument parameters from the loaded dataset
        _detected: dict | None = None
        try:
            from pyadps.processing.signal_quality import _extract_fl_params
            _detected = _extract_fl_params(proc.dataset)
        except Exception:
            pass

        freq_override = None
        bin_override = None
        depth_override = None
        pings_override = None

        if use_custom:
            st.write("**Custom Parameters:**")

            # Default frequency: auto-detected, or 300 kHz as fallback
            _def_freq = _detected["frequency"] if _detected else 300
            _def_freq_idx = (
                _available_freqs.index(_def_freq)
                if _def_freq in _available_freqs
                else (_available_freqs.index(300) if 300 in _available_freqs else 0)
            )
            freq_override = st.selectbox(
                "Frequency (kHz)",
                options=_available_freqs,
                index=_def_freq_idx,
                key="advisor_freq",
            )

            _bins_for_freq = sorted(
                float(k) for k in noise_coeffs[str(freq_override)]
            )

            # Default bin size: auto-detected, matched to nearest available bin
            _def_bin = _detected["bin_size"] if _detected else _bins_for_freq[0]
            _def_bin_idx = next(
                (i for i, b in enumerate(_bins_for_freq) if abs(b - _def_bin) < 1e-9),
                0,
            )
            bin_override = st.selectbox(
                "Bin Size (m)",
                options=_bins_for_freq,
                index=_def_bin_idx,
                key="advisor_bin",
            )

            # Default depth range: auto-detected or bin × 30 cells
            _def_depth = (
                _detected["depth_range"] if _detected
                else float(bin_override or 1.0) * 30
            )
            depth_override = st.number_input(
                "Depth Range (m)",
                min_value=0.1,
                max_value=2000.0,
                value=_def_depth,
                step=1.0,
                format="%.1f",
                help=(
                    "Total profiling depth (bin size × number of cells). "
                    "Use a shorter range if valid data does not span the "
                    "full profile — data beyond this depth should be excluded."
                ),
                key="advisor_depth",
            )

            # Default pings: auto-detected or 40
            _def_pings = _detected["pings_per_ensemble"] if _detected else 40
            pings_override = st.number_input(
                "Pings per Ensemble",
                min_value=1,
                max_value=16384,
                value=_def_pings,
                step=1,
                key="advisor_pings",
            )

        if st.button("Compute Threshold", type="primary", key="advisor_compute"):
            try:
                _result = proc.get_percent_good_threshold(
                    desired_std=float(desired_std),
                    frequency=int(freq_override) if freq_override is not None else None,
                    bin_size=float(bin_override) if bin_override is not None else None,
                    depth_range=float(depth_override) if depth_override is not None else None,
                    n_pings=int(pings_override) if pings_override is not None else None,
                )
                st.session_state.advisor_result = _result
            except ValueError as e:
                st.error(f"❌ {e}")
                st.session_state.advisor_result = None
            except Exception as e:
                st.error(
                    f"❌ Could not compute threshold: {e}. "
                    "Try enabling **Use custom instrument parameters** above."
                )
                st.session_state.advisor_result = None

    with col_results:
        _adv_result = st.session_state.get("advisor_result")

        if _adv_result is None:
            st.info(
                "💡 Enter a desired standard deviation and click "
                "**Compute Threshold** to get the recommended percent good cutoff."
            )
        else:
            st.write("**📡 Instrument Parameters:**")
            pc1, pc2 = st.columns(2)
            with pc1:
                st.metric("Frequency", f"{_adv_result.frequency} kHz")
                st.metric("Bin Size", f"{_adv_result.bin_size:g} m")
            with pc2:
                st.metric("Depth Range", f"{_adv_result.depth_range:.1f} m")
                st.metric("Pings / Ensemble", _adv_result.pings_per_ensemble)

            st.divider()

            st.write("**📊 Noise Model Results:**")
            rc1, rc2 = st.columns(2)
            with rc1:
                st.metric(
                    "Single-ping Std Dev",
                    f"{_adv_result.single_ping_std:.3f} cm/s",
                    help="Standard deviation of a single acoustic ping at the full depth range.",
                )
                st.metric(
                    "Ensemble Std Dev",
                    f"{_adv_result.ensemble_std:.3f} cm/s",
                    help="Effective precision when averaging all pings in the ensemble.",
                )
            with rc2:
                st.metric(
                    "Valid Pings Required",
                    _adv_result.valid_pings_required,
                    help="Minimum valid pings needed to reach the desired precision.",
                )
                st.metric(
                    "Percent Good Cutoff",
                    f"{_adv_result.percent_good_cutoff:.1f} %",
                )

            st.divider()

            if _adv_result.achievable:
                st.success(
                    f"✅ A precision of **{_adv_result.desired_std:.2f} cm/s** is "
                    f"achievable. At least **{_adv_result.valid_pings_required}** of "
                    f"{_adv_result.pings_per_ensemble} pings per ensemble must be valid."
                )
            else:
                st.error(
                    f"❌ A precision of **{_adv_result.desired_std:.2f} cm/s** is "
                    f"**not achievable** with {_adv_result.pings_per_ensemble} "
                    f"pings/ensemble. The best achievable precision with all pings "
                    f"valid is **{_adv_result.ensemble_std:.3f} cm/s**. "
                    "Increase the number of pings per ensemble or relax the target."
                )

            st.divider()

            _pg_cutoff = _adv_result.percent_good_cutoff

            def _apply_pg_threshold() -> None:
                st.session_state.percent_good_threshold = int(round(_pg_cutoff))
                st.session_state.apply_percent_good = True

            st.button(
                f"Apply {_adv_result.percent_good_cutoff:.0f} % to QC Tests",
                type="secondary",
                key="advisor_apply",
                on_click=_apply_pg_threshold,
                help="Copies this cutoff to the Percent Good threshold in the QC Tests tab.",
            )
            st.caption(
                "After applying, go to the **QC Tests** tab to preview and save."
            )


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
            _ei_pb = st.session_state.get("echo_intensity_per_beam_threshold")
            if _ei_pb is not None:
                preview_items.append(
                    f"• Echo intensity check: per-beam thresholds "
                    f"{[int(v) for v in _ei_pb]}"
                )
            else:
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
            preview_items.append("• Three-beam mode: enabled")

        if st.session_state.beam_ignore is not None:
            preview_items.append(
                f"• Beam to ignore: Beam {st.session_state.beam_ignore + 1}"
            )

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
            try:
                _ei_pb = st.session_state.get("echo_intensity_per_beam_threshold")
                if st.session_state.apply_echo_intensity and _ei_pb is not None:
                    _ei_arg: float | list[float] | None = _ei_pb
                elif st.session_state.apply_echo_intensity and st.session_state.echo_intensity_threshold is not None:
                    _ei_arg = int(st.session_state.echo_intensity_threshold)
                else:
                    _ei_arg = None
                proc.apply_signal_quality(
                    correlation=int(st.session_state.correlation_threshold) if st.session_state.apply_correlation and st.session_state.correlation_threshold is not None else None,
                    echo_intensity=_ei_arg,
                    error_velocity=int(st.session_state.error_velocity_threshold) if st.session_state.apply_error_velocity and st.session_state.error_velocity_threshold is not None else None,
                    percent_good=int(st.session_state.percent_good_threshold) if st.session_state.apply_percent_good and st.session_state.percent_good_threshold is not None else None,
                    false_target=int(st.session_state.false_target_threshold) if st.session_state.apply_false_target and st.session_state.false_target_threshold is not None else None,
                    threebeam=st.session_state.threebeam_mode,
                    beam_ignore=st.session_state.beam_ignore,
                )

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

                report = proc.reports[-1] if proc.reports else None
                if report and report.checks:
                    stats_data = []
                    for stat in report.checks:
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

        st.button("Reset QC Tests", key="reset_qc", on_click=_reset_qc_full)

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
