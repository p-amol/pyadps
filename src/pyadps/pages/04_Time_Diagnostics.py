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
import streamlit as st

st.set_page_config(page_title="Time Diagnostics", page_icon="🕐", layout="wide")

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

st.title("Time Diagnostics")
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

tab_diag, tab_snap, tab_fill, tab_reset = st.tabs(
    ["Diagnose", "Snap Time Axis", "Fill Time Gaps", "Reset"]
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
# TAB 4: RESET
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

with st.sidebar:
    st.caption(f"File: {st.session_state.fname}")
