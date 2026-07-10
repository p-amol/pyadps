"""
04_Time_Diagnostics.py - Time Axis Diagnostics Page (pyadps v1.0.0)

This page allows users to diagnose and correct the time axis of their ADCP
dataset before any QC processing begins.  Two independent operations are
provided:

1. Snap Time Axis  — round timestamps that drift slightly from the intended
                     recording interval (e.g. 10:00:02 → 10:00:00).
2. Fill Time Gaps  — insert synthetic missing-data ensembles so the time axis
                     is perfectly uniform (no skipped slots).

Both operations call proc.apply_time_axis() and are tracked in ProcessedDataset.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly_resampler import FigureResampler

st.set_page_config(page_title="Time Diagnostics", page_icon="🕐", layout="wide")

# =============================================================================
# COLOR SCALE OPTIONS (Velocity View tab)
# Mirrors the diverging color-scale picker on the Write Processed Data page,
# so users get the same palette/range controls when inspecting velocity here.
# =============================================================================

DIVERGING_COLORSCALE_OPTIONS = [
    "RdBu_r",
    "RdBu",
    "balance",
    "delta",
    "curl",
    "spectral",
    "picnic",
    "portland",
    "tropic",
    "temps",
    "puor",
    "prgn",
]


def render_diverging_color_scale_options(
    data: np.ndarray, key_suffix: str
) -> tuple[str, float, float]:
    """Render a palette selectbox + symmetric range input for a zero-centered
    diverging heatmap (velocity).

    Unlike a plain min/max range, this uses a single "clamp magnitude" so the
    scale stays properly centered on zero (zmin=-N, zmax=+N) — Plotly ignores
    zmid once zmin/zmax are set explicitly, so symmetry has to be enforced
    here rather than relying on zmid.

    Returns
    -------
    tuple of (colorscale, zmin, zmax)
    """
    data_masked = np.where(data == -32768, np.nan, data)
    if np.all(np.isnan(data_masked)):
        default_range = 1.0
    else:
        default_range = float(
            max(abs(np.nanmin(data_masked)), abs(np.nanmax(data_masked)))
        )

    with st.expander("🎨 Color Scale Options", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            colorscale = st.selectbox(
                "Color palette",
                DIVERGING_COLORSCALE_OPTIONS,
                index=0,
                help="Diverging colorscale centered on zero flow.",
                key=f"colorscale_select_{key_suffix}",
            )
        with col_b:
            clamp_range = st.number_input(
                "Clamp range (± mm/s)",
                min_value=0.0,
                value=default_range,
                help="Values beyond ±this are shown with the colorscale's "
                "end colors, so out-of-range data still gets a color. "
                "Kept symmetric around zero to preserve the diverging scale.",
                key=f"clamp_range_input_{key_suffix}",
            )

    colorscale = colorscale or DIVERGING_COLORSCALE_OPTIONS[0]
    return colorscale, -clamp_range, clamp_range


# =============================================================================
# GUARD: require a loaded file
# =============================================================================

if "processor" not in st.session_state or st.session_state.processor is None:
    st.error("⚠️ No data loaded! Please read a file on the **Read File** page first.")
    st.stop()

proc = st.session_state.processor
ds = proc.dataset

# =============================================================================
# RESET CALLBACK
# Defined before any st.button() calls so Streamlit can register it as on_click.
# Using on_click avoids resetting the active tab on the next rerun.
# =============================================================================


def _reset_time_axis() -> None:
    proc.reset()


# =============================================================================
# TIME AXIS SUMMARY (computed once, used across tabs)
# =============================================================================

time_s = pd.Series(pd.to_datetime(ds["time"].values))
n_ens = len(time_s)
t_start = time_s.iloc[0]
t_end = time_s.iloc[-1]
duration = t_end - t_start

# Interval statistics — explicit Series type to satisfy pandas-stubs
time_diffs: pd.Series = pd.to_timedelta(time_s.diff().dropna())
interval_mode = time_diffs.mode().iloc[0] if not time_diffs.empty else None
is_regular = (
    bool((time_diffs == interval_mode).all()) if interval_mode is not None else True
)

# Gap detection (used by Fill tab and for button disabled logic)
if interval_mode is not None and len(time_diffs) > 0:
    gaps = time_diffs[time_diffs > interval_mode]
    n_gaps = len(gaps)
    extra_slots = (
        int((gaps / interval_mode).apply(lambda x: round(x) - 1).sum())
        if n_gaps > 0
        else 0
    )
else:
    gaps = pd.Series([], dtype="timedelta64[ns]")
    n_gaps = 0
    extra_slots = 0

# Previous corrections applied this session
prev = proc._time_axis_results

# =============================================================================
# HEADER
# =============================================================================

_, _col_fname = st.columns([5, 1])
with _col_fname:
    st.caption(f"📂 {st.session_state.get('fname', 'No file selected')}")
st.header("Time Diagnostics", divider="blue")
st.write(
    "Diagnose and correct time-axis irregularities **before** running any QC steps. "
    "Both operations are optional — most datasets need neither."
)
st.info(
    "The correction tools on this page are designed for **minor** time-axis issues "
    "such as small timestamp drifts or isolated missing ensembles. "
    "If the time axis is highly irregular — for example, variable sampling intervals "
    "or large sections of missing data — these tools are not sufficient. "
    "In such cases, the data may be regridded onto a uniform time axis using an "
    "external interpolation toolbox after processing. "
    "Note that time series analysis tools used in the **Sensor Health** and "
    "**Velocity Processing** sections may not perform as expected on an irregular time axis.",
    icon="ℹ️",
)

# --- Summary metrics ---------------------------------------------------------
col1, col2 = st.columns(2)
col1.metric("Total Ensembles", f"{n_ens:,}")
col2.metric("Duration", str(duration))

col3, col4 = st.columns(2)
col3.metric("Start Time", t_start.strftime("%Y-%m-%d %H:%M:%S"))
col4.metric("End Time", t_end.strftime("%Y-%m-%d %H:%M:%S"))

if is_regular:
    st.success(f"Time axis is regular — interval: {interval_mode}")
else:
    st.warning(
        f"Time axis is **irregular**. Most common interval: {interval_mode}. "
        "Use the tools below to diagnose and correct."
    )

if prev:
    st.info(
        "Time axis corrections have already been applied this session. "
        "Use the **Reset** tab to undo them."
    )

st.divider()

# =============================================================================
# TABS
# =============================================================================

tab_diag, tab_snap, tab_fill, tab_vel, tab_reset = st.tabs(
    ["Diagnose", "Snap Time Axis", "Fill Time Gaps", "Velocity View", "Reset"]
)

# ---------------------------------------------------------------------------
# TAB 1: DIAGNOSE
# ---------------------------------------------------------------------------

with tab_diag:
    d1, d2, d3, d4 = st.tabs(
        [
            "Interval Plot",
            "Time Components",
            "Component Frequency",
            "Interval Frequency",
        ]
    )

    with d1:
        st.subheader("Time Interval Between Ensembles")
        if len(time_s) > 1:
            diff_seconds = time_diffs.dt.total_seconds().reset_index(drop=True)
            fig = px.line(
                x=diff_seconds.index,
                y=diff_seconds,
                labels={"x": "Ensemble Number", "y": "Time Difference (s)"},
                title="Time Interval Between Consecutive Ensembles",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.info(
                "A **flat horizontal line** = regular interval. "
                "**Spikes** = gaps in the data. "
                "**Noise or drift** = timestamp jitter that snap can fix."
            )
            st.dataframe(
                diff_seconds.describe().rename("Interval (s)").to_frame(),
                use_container_width=True,
            )
        else:
            st.info("Need at least 2 ensembles to plot intervals.")

    with d2:
        st.subheader("Time Components vs. Ensembles")
        component: str = (
            st.radio(  # type: ignore[assignment]
                "Select time component",
                ("Hour", "Minute", "Second"),
                index=1,
                horizontal=True,
            )
            or "Minute"
        )
        comp_map = {
            "Hour": time_s.dt.hour,
            "Minute": time_s.dt.minute,
            "Second": time_s.dt.second,
        }
        fig = px.line(
            x=np.arange(n_ens),
            y=comp_map[component],
            labels={"x": "Ensemble Number", "y": component},
            title=f"{component} Component vs. Ensembles",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.info(
            "**How to interpret this chart:** For an hourly uniform time interval, "
            "the Hour plot should be a repeating sawtooth wave. "
            "The Minute and Second plots should be flat lines. "
            "For an hourly non-uniform time interval, jumps in the minute and second plots indicate gaps, "
            "while slopes or noise indicate drift."
        )

    with d3:
        st.subheader("Time Component Frequency")
        comp_sel: str = (
            st.selectbox(  # type: ignore[assignment]
                "Component to analyse",
                ("minute", "hour", "second"),
                key="diag_comp_sel",
            )
            or "minute"
        )
        comp_values = getattr(time_s.dt, comp_sel)
        freq_comp = comp_values.value_counts().sort_index()
        fig_comp = px.bar(
            x=freq_comp.index,
            y=freq_comp.values,
            labels={"x": comp_sel.capitalize(), "y": "Count"},
            title=f"Distribution of {comp_sel.capitalize()} values",
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.dataframe(
            freq_comp.reset_index().rename(
                columns={"index": comp_sel.capitalize(), comp_sel: "Count"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    with d4:
        st.subheader("Interval Frequency")
        if len(time_diffs) > 0:
            interval_freq = time_diffs.dt.total_seconds().value_counts().sort_index()
            fig_int = px.bar(
                x=interval_freq.index,
                y=interval_freq.values,
                labels={"x": "Interval (s)", "y": "Count"},
                title="Distribution of Time Intervals",
            )
            st.plotly_chart(fig_int, use_container_width=True)
            st.dataframe(
                interval_freq.reset_index().rename(
                    columns={"index": "Interval (s)", 0: "Count"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Insufficient data to compute intervals.")

# ---------------------------------------------------------------------------
# TAB 2: SNAP TIME AXIS
# ---------------------------------------------------------------------------

with tab_snap:
    st.subheader("Snap Time Axis")
    st.write(
        "Round each timestamp to the nearest specified minute mark. "
        "Use this when timestamps drift slightly from the intended recording time "
        "(e.g. :02 seconds past the hour instead of :00)."
    )

    if is_regular:
        st.info(
            "Time axis is already regular — snap is not needed. "
            "If you believe drift exists, inspect the **Diagnose** tab first."
        )

    # Pre-populate target minute from most common minute in the data
    minute_freq = time_s.dt.minute.value_counts()
    default_minute = int(minute_freq.index[0]) if not minute_freq.empty else 0

    col1, col2 = st.columns(2)
    with col1:
        target_minute = st.slider(
            "Target minute (0–59)",
            min_value=0,
            max_value=59,
            value=default_minute,
            step=1,
            help="Timestamps will be rounded to the nearest occurrence of this minute within each hour.",
        )
    with col2:
        tolerance_min = st.number_input(
            "Max allowed correction (minutes)",
            min_value=1,
            max_value=60,
            value=5,
            step=1,
            help="The operation is aborted if any timestamp would be shifted by more than this amount.",
        )

    st.caption(
        f"Most common minute in dataset: **{default_minute}**. "
        f"Tolerance: **{tolerance_min} min**."
    )

    if st.button("Apply Snap", type="primary", disabled=is_regular):
        with st.spinner("Snapping time axis…"):
            proc.apply_time_axis(
                snap=True,
                snap_freq="h",
                snap_tolerance=f"{tolerance_min}min",
                snap_target_minute=target_minute,
            )
        snap_result = proc._time_axis_results.get("snap", {})
        if snap_result.get("success"):
            st.success(snap_result["message"])
        else:
            st.error(snap_result.get("message", "Snap failed."))

    if prev and "snap" in prev:
        r = prev["snap"]
        if r.get("success"):
            st.info(f"Last snap result: {r['message']}")

# ---------------------------------------------------------------------------
# TAB 3: FILL TIME GAPS
# ---------------------------------------------------------------------------

with tab_fill:
    st.subheader("Fill Time Gaps")
    st.write(
        "Insert synthetic (missing-value) ensembles for any skipped time slots "
        "so the dataset has a perfectly uniform time axis. "
        "Apply **after snapping** (if snapping was needed)."
    )

    if n_gaps == 0:
        st.success("No gaps detected — time axis already has no missing slots.")
    else:
        st.warning(
            f"Found **{n_gaps}** gap(s) in the time axis "
            f"totalling **{extra_slots}** missing ensemble slot(s)."
        )
        with st.expander("Gap details"):
            gap_df = pd.DataFrame(
                {
                    "Ensemble index": gaps.index.tolist(),
                    "Gap duration": gaps.values,
                }
            )
            st.dataframe(gap_df, use_container_width=True, hide_index=True)

    fill_method: str = (
        st.selectbox(  # type: ignore[assignment]
            "Fill frequency method",
            ("auto", "h", "30min", "15min", "10min", "6h", "D"),
            index=0,
            help=(
                "'auto' detects the interval from the median time step. "
                "Choose a fixed string if auto-detection produces the wrong result."
            ),
        )
        or "auto"
    )

    if st.button("Apply Fill", type="primary", disabled=(n_gaps == 0)):
        with st.spinner("Filling time gaps…"):
            proc.apply_time_axis(fill_gaps=True, fill_method=fill_method)
        fill_result = proc._time_axis_results.get("fill_gaps", {})
        if fill_result.get("success"):
            st.success(
                f"Time gaps filled (method={fill_method}). "
                f"New ensemble count: {len(proc.dataset.time):,}."
            )
        else:
            st.error(fill_result.get("error", "Fill failed — check logs for details."))

    if prev and "fill_gaps" in prev:
        r = prev["fill_gaps"]
        if r.get("success"):
            st.info("Time gaps have already been filled this session.")

# ---------------------------------------------------------------------------
# TAB 4: VELOCITY VIEW
# ---------------------------------------------------------------------------

with tab_vel:
    st.subheader("Velocity View")
    st.write(
        "Inspect the velocity field on the **current** time axis, reflecting "
        "any Snap / Fill corrections applied above. Ensembles inserted by "
        "**Fill Time Gaps** show up as blank (missing-value) columns, making "
        "it easy to confirm where gaps were filled."
    )

    if "velocity" not in ds.data_vars:
        st.info("No velocity data available in this dataset.")
    else:
        velocity_vals = ds["velocity"].values
        n_beams_vel = velocity_vals.shape[0]
        n_cells_vel = velocity_vals.shape[1]

        try:
            coord_info = ds.fixed_leader.coordinate_transformation(ens=0)
            is_earth = "Earth" in coord_info.get("Coordinates", "")
        except Exception:
            is_earth = False

        beam_labels = (
            {1: "Zonal (u)", 2: "Meridional (v)", 3: "Vertical (w)", 4: "Error"}
            if is_earth
            else {i: f"Beam {i}" for i in range(1, n_beams_vel + 1)}
        )

        beam_vel: int = (
            st.radio(  # type: ignore[assignment]
                "Select beam/component",
                list(range(1, n_beams_vel + 1)),
                horizontal=True,
                format_func=lambda b: beam_labels.get(b, f"Beam {b}"),
                key="time_diag_vel_beam",
            )
            or 1
        )

        vel_data_raw = velocity_vals[beam_vel - 1, :, :].astype(float)
        colorscale, zmin, zmax = render_diverging_color_scale_options(
            vel_data_raw, key_suffix=f"time_diag_{beam_vel}"
        )

        vel_data = np.where(vel_data_raw == -32768, np.nan, vel_data_raw)
        y_cells_vel = np.arange(1, n_cells_vel + 1)

        fig_vel = FigureResampler(go.Figure())
        fig_vel.add_trace(
            go.Heatmap(
                z=vel_data,
                x=time_s.values,
                y=y_cells_vel,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                hoverongaps=False,
                colorbar=dict(title="Velocity (mm/s)"),
            )
        )
        fig_vel.update_layout(
            xaxis=dict(showline=True, mirror=True, title="Time"),
            yaxis=dict(
                showline=True,
                mirror=True,
                title="Cell Number",
                autorange="reversed",
            ),
            title_text=f"Velocity - {beam_labels.get(beam_vel, f'Beam {beam_vel}')}",
            height=500,
        )
        st.plotly_chart(fig_vel, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 5: RESET
# ---------------------------------------------------------------------------

with tab_reset:
    st.subheader("Applied Corrections")

    if not prev:
        st.info("No time axis corrections have been applied this session.")
    else:
        cfg = proc.config
        summary_rows = [
            ["Time axis modified", str(cfg.isTimeAxisModified)],
            ["Snap applied", str(cfg.isSnapTimeAxis)],
        ]
        if cfg.isSnapTimeAxis:
            summary_rows += [
                ["  Snap frequency", cfg.time_snap_frequency],
                ["  Snap tolerance", cfg.time_snap_tolerance],
                ["  Target minute", str(cfg.time_target_minute)],
            ]
        summary_rows.append(["Fill gaps applied", str(cfg.isTimeGapFilled)])
        if cfg.isTimeGapFilled:
            summary_rows.append(["  Fill method", cfg.time_fill_method])

        st.dataframe(
            pd.DataFrame(summary_rows, columns=["Parameter", "Value"]),
            use_container_width=True,
            hide_index=True,
        )

        if proc.processing_log:
            st.markdown("**Processing log entries:**")
            for entry in proc.processing_log:
                if "time" in entry.lower():
                    st.write(f"- {entry}")

    st.divider()
    st.subheader("Reset")

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        st.button(
            "Reset All Processing",
            on_click=_reset_time_axis,
            disabled=not bool(prev),
            type="primary",
        )
    with col_info:
        st.warning(
            "Resetting undoes **all** processing steps (time axis, sensor health, "
            "signal quality, profile operations, and velocity checks), because time "
            "axis correction is step 1 of the pipeline. Subsequent steps are built "
            "on the corrected time axis and cannot be preserved independently."
        )

