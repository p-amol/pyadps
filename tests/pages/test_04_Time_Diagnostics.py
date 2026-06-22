"""
Test Suite for 04_Time_Diagnostics.py
=====================================
Uses Streamlit's AppTest framework to exercise the time-axis diagnostics page.

Design decisions
----------------
1. The page reads ``st.session_state.processor`` on every run and calls
   ``proc.apply_time_axis()`` when the user clicks Apply Snap / Apply Fill.
   We mock the processor so that apply_time_axis() captures its kwargs and
   stores a fake result in ``proc._time_axis_results``.

2. Regular vs irregular datasets are produced by giving the mock dataset a
   perfectly uniform or deliberately jittered time coordinate.

3. Apply Snap is disabled when the time axis is regular; Apply Fill is disabled
   when no gaps are detected.  AppTest exposes ``button.disabled`` for this.

4. The Reset button is disabled when no corrections have been applied
   (``prev_results`` is empty).

Usage
-----
    pytest tests/pages/test_04_Time_Diagnostics.py -v
    pytest tests/pages/test_04_Time_Diagnostics.py -v --tb=short
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import xarray as xr
from streamlit.testing.v1 import AppTest

# ---------------------------------------------------------------------------
# Script path
# ---------------------------------------------------------------------------
SCRIPT_PATH = str(
    Path(__file__).parent.parent.parent
    / "src"
    / "pyadps"
    / "pages"
    / "04_Time_Diagnostics.py"
)
assert Path(SCRIPT_PATH).exists(), (
    f"Script not found at {SCRIPT_PATH}\n"
    f"Expected layout: pyadps/src/pyadps/pages/04_Time_Diagnostics.py"
)

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _make_regular_ds(n_ens: int = 48) -> xr.Dataset:
    """Dataset with a perfectly regular hourly time axis."""
    times = pd.date_range("2023-01-01", periods=n_ens, freq="h")
    return xr.Dataset(coords={"time": times})


def _make_irregular_ds(n_ens: int = 48) -> xr.Dataset:
    """Dataset with a slightly jittered time axis (timestamp drift)."""
    rng = np.random.default_rng(42)
    times = pd.date_range("2023-01-01", periods=n_ens, freq="h")
    jitter = pd.to_timedelta(rng.integers(-90, 90, size=n_ens), unit="s")
    return xr.Dataset(coords={"time": times + jitter})


def _make_gapped_ds(n_ens: int = 48) -> xr.Dataset:
    """Dataset with a genuine gap (4 missing hours in the middle)."""
    t1 = pd.date_range("2023-01-01", periods=n_ens // 2, freq="h")
    t2 = (
        pd.date_range("2023-01-01 00:00", periods=n_ens // 2, freq="h")
        + pd.Timedelta(hours=n_ens // 2 + 4)
    )
    times = t1.append(t2)
    return xr.Dataset(coords={"time": times})


# ---------------------------------------------------------------------------
# Processor mock
# ---------------------------------------------------------------------------

def _make_proc(ds: xr.Dataset, prev_results: dict | None = None) -> MagicMock:
    """Build a minimal ProcessedDataset mock."""
    proc = MagicMock()
    proc.dataset = ds
    proc._time_axis_results = prev_results or {}
    proc.processing_log = []

    def _apply_time_axis(snap: bool = False, fill_gaps: bool = False, **_kwargs):
        results = {}
        if snap:
            results["snap"] = {"success": True, "message": "Snapped successfully."}
            proc.processing_log.append("Time axis snapped: Snapped successfully.")
        if fill_gaps:
            results["fill_gaps"] = {"success": True}
            proc.processing_log.append("Time gaps filled (method=auto)")
        proc._time_axis_results = results

    proc.apply_time_axis.side_effect = _apply_time_axis

    def _reset():
        proc._time_axis_results = {}
        proc.processing_log = []

    proc.reset.side_effect = _reset
    return proc


# ---------------------------------------------------------------------------
# AppTest factory
# ---------------------------------------------------------------------------

def _make_at(ds: xr.Dataset, prev_results: dict | None = None) -> AppTest:
    at = AppTest.from_file(SCRIPT_PATH, default_timeout=30)
    at.session_state["processor"] = _make_proc(ds, prev_results)
    return at


# ---------------------------------------------------------------------------
# GUARD CLAUSE
# ---------------------------------------------------------------------------

class TestGuardClause:
    def test_no_processor_shows_error(self):
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=30)
        at.run()
        assert not at.exception
        assert any("No data loaded" in e.value for e in at.error)

    def test_none_processor_shows_error(self):
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=30)
        at.session_state["processor"] = None
        at.run()
        assert not at.exception
        assert any("No data loaded" in e.value for e in at.error)


# ---------------------------------------------------------------------------
# SUMMARY SECTION
# ---------------------------------------------------------------------------

class TestSummary:
    def test_regular_shows_success(self):
        at = _make_at(_make_regular_ds())
        at.run()
        assert not at.exception
        assert any("regular" in s.value.lower() for s in at.success)

    def test_irregular_shows_warning(self):
        at = _make_at(_make_irregular_ds())
        at.run()
        assert not at.exception
        assert any("irregular" in w.value.lower() for w in at.warning)

    def test_metrics_present(self):
        at = _make_at(_make_regular_ds(n_ens=48))
        at.run()
        assert not at.exception
        metric_labels = [m.label for m in at.metric]
        assert "Total Ensembles" in metric_labels

    def test_previous_results_banner(self):
        prev = {"snap": {"success": True, "message": "Snapped."}}
        at = _make_at(_make_regular_ds(), prev_results=prev)
        at.run()
        assert not at.exception
        assert any("already been applied" in i.value for i in at.info)


# ---------------------------------------------------------------------------
# DIAGNOSE TAB
# ---------------------------------------------------------------------------

class TestDiagnoseTab:
    def test_renders_without_exception(self):
        at = _make_at(_make_regular_ds())
        at.run()
        assert not at.exception

    def test_component_selectbox_present(self):
        at = _make_at(_make_regular_ds())
        at.run()
        assert not at.exception
        keys = [s.label for s in at.selectbox]
        assert any("component" in k.lower() for k in keys)


# ---------------------------------------------------------------------------
# SNAP TAB — including disabled-when-regular behaviour
# ---------------------------------------------------------------------------

class TestSnapTab:
    def test_snap_button_present(self):
        at = _make_at(_make_irregular_ds())
        at.run()
        assert not at.exception
        labels = [b.label for b in at.button]
        assert any("snap" in l.lower() for l in labels)

    def test_snap_button_disabled_for_regular_time(self):
        at = _make_at(_make_regular_ds())
        at.run()
        assert not at.exception
        snap_btn = next(b for b in at.button if "apply snap" in b.label.lower())
        assert snap_btn.disabled

    def test_snap_button_enabled_for_irregular_time(self):
        at = _make_at(_make_irregular_ds())
        at.run()
        assert not at.exception
        snap_btn = next(b for b in at.button if "apply snap" in b.label.lower())
        assert not snap_btn.disabled

    def test_snap_regular_shows_info(self):
        at = _make_at(_make_regular_ds())
        at.run()
        assert not at.exception
        assert any("not needed" in i.value.lower() for i in at.info)

    def test_snap_apply_calls_apply_time_axis(self):
        ds = _make_irregular_ds()
        proc = _make_proc(ds)
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=30)
        at.session_state["processor"] = proc
        at.run()

        snap_btn = next(b for b in at.button if "apply snap" in b.label.lower())
        snap_btn.click().run()

        assert not at.exception
        proc.apply_time_axis.assert_called_once()
        assert proc.apply_time_axis.call_args.kwargs.get("snap") is True

    def test_snap_success_shows_success_message(self):
        ds = _make_irregular_ds()
        proc = _make_proc(ds)
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=30)
        at.session_state["processor"] = proc
        at.run()

        snap_btn = next(b for b in at.button if "apply snap" in b.label.lower())
        snap_btn.click().run()

        assert not at.exception
        assert any("snapped" in s.value.lower() for s in at.success)


# ---------------------------------------------------------------------------
# FILL TAB — including disabled-when-no-gaps behaviour
# ---------------------------------------------------------------------------

class TestFillTab:
    def test_fill_button_present(self):
        at = _make_at(_make_gapped_ds())
        at.run()
        assert not at.exception
        labels = [b.label for b in at.button]
        assert any("fill" in l.lower() for l in labels)

    def test_fill_button_disabled_when_no_gaps(self):
        at = _make_at(_make_regular_ds())
        at.run()
        assert not at.exception
        fill_btn = next(b for b in at.button if "apply fill" in b.label.lower())
        assert fill_btn.disabled

    def test_fill_button_enabled_when_gaps_exist(self):
        at = _make_at(_make_gapped_ds())
        at.run()
        assert not at.exception
        fill_btn = next(b for b in at.button if "apply fill" in b.label.lower())
        assert not fill_btn.disabled

    def test_fill_shows_gap_warning(self):
        at = _make_at(_make_gapped_ds())
        at.run()
        assert not at.exception
        assert any("gap" in w.value.lower() for w in at.warning)

    def test_fill_apply_calls_apply_time_axis(self):
        ds = _make_gapped_ds()
        proc = _make_proc(ds)
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=30)
        at.session_state["processor"] = proc
        at.run()

        fill_btn = next(b for b in at.button if "apply fill" in b.label.lower())
        fill_btn.click().run()

        assert not at.exception
        proc.apply_time_axis.assert_called_once()
        assert proc.apply_time_axis.call_args.kwargs.get("fill_gaps") is True

    def test_fill_success_shows_message(self):
        ds = _make_gapped_ds()
        proc = _make_proc(ds)
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=30)
        at.session_state["processor"] = proc
        at.run()

        fill_btn = next(b for b in at.button if "apply fill" in b.label.lower())
        fill_btn.click().run()

        assert not at.exception
        assert any("filled" in s.value.lower() for s in at.success)

    def test_no_gaps_shows_success(self):
        at = _make_at(_make_regular_ds())
        at.run()
        assert not at.exception
        assert any("no gaps" in s.value.lower() for s in at.success)


# ---------------------------------------------------------------------------
# RESET TAB
# ---------------------------------------------------------------------------

class TestResetTab:
    def test_reset_button_present(self):
        at = _make_at(_make_regular_ds())
        at.run()
        assert not at.exception
        labels = [b.label for b in at.button]
        assert any("reset" in l.lower() for l in labels)

    def test_reset_button_disabled_when_no_corrections(self):
        at = _make_at(_make_regular_ds(), prev_results={})
        at.run()
        assert not at.exception
        reset_btn = next(b for b in at.button if "reset" in b.label.lower())
        assert reset_btn.disabled

    def test_reset_button_enabled_when_corrections_exist(self):
        prev = {"snap": {"success": True, "message": "Snapped."}}
        at = _make_at(_make_regular_ds(), prev_results=prev)
        at.run()
        assert not at.exception
        reset_btn = next(b for b in at.button if "reset" in b.label.lower())
        assert not reset_btn.disabled

    def test_no_corrections_shows_info(self):
        at = _make_at(_make_regular_ds(), prev_results={})
        at.run()
        assert not at.exception
        assert any("no time axis corrections" in i.value.lower() for i in at.info)

    def test_reset_calls_proc_reset(self):
        prev = {"snap": {"success": True, "message": "Snapped."}}
        ds = _make_regular_ds()
        proc = _make_proc(ds, prev_results=prev)
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=30)
        at.session_state["processor"] = proc
        at.run()

        reset_btn = next(b for b in at.button if "reset" in b.label.lower())
        reset_btn.click().run()

        assert not at.exception
        proc.reset.assert_called_once()

    def test_reset_warning_mentions_all_steps(self):
        at = _make_at(_make_regular_ds())
        at.run()
        assert not at.exception
        warning_texts = " ".join(w.value for w in at.warning)
        assert "all" in warning_texts.lower()
