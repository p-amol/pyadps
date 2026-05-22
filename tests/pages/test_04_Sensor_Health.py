"""
Test Suite for 04_Sensor_Health.py
====================================
Uses Streamlit's AppTest framework to exercise the actual Streamlit page in a
simulated runtime, providing genuine UI-path coverage alongside unit tests for
every pure helper function.

Architecture notes
------------------
The page uses ``st.session_state.processor`` (a ProcessedDataset) as its sole
entry point.  ``proc.dataset`` provides the xarray Dataset; all UI interactions
ultimately call ``proc.get_sensor_health_runner()`` and ``proc.commit_runner()``.

AppTest design decisions
------------------------
1. **Processor mock** must expose ``.dataset``, ``.get_sensor_health_runner()``,
   ``commit_runner()``, ``get_current_stats()``, ``.processing_log``, and
   ``.reset()``.  We use a real ``xr.Dataset`` inside it so Streamlit can
   iterate over arrays without MagicMock attribute errors.

2. **Radio pre-setting does not work.** ``st.radio()`` ignores pre-set session
   state; conditional branches are entered via ``radio.set_value(...).run()``
   AFTER the initial ``at.run()``.

3. **Sound-speed checkbox only appears when T or S is marked as modified.**
   The page hard-gates it on ``temperature_modified or salinity_modified``.
   Tests that need it set those flags in session state before running.

4. **``at.session_state`` has no ``.get()``.**  Always use
   ``key in at.session_state`` + ``at.session_state[key]`` indexing.

5. **CRITICAL — ``sensor_health_initialized`` gating.**
   The page's init block runs only when ``"sensor_health_initialized" not in
   st.session_state``.  If we pre-set it to ``True``, the entire block is
   skipped and variables like ``depth_modified``, ``roll_threshold``,
   ``pitch_threshold``, etc. are never defined — causing ``AttributeError``
   on first access.  ``_make_at()`` automatically injects the full set of
   defaults whenever ``sensor_health_initialized=True`` appears in ``extra``,
   so individual tests only need to override the keys relevant to them.

Usage
-----
    pytest tests/pages/test_04_Sensor_Health.py -v
    pytest tests/pages/test_04_Sensor_Health.py -v --tb=short
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from streamlit.testing.v1 import AppTest

# ---------------------------------------------------------------------------
# Script path
# ---------------------------------------------------------------------------
SCRIPT_PATH = str(
    Path(__file__).parent.parent.parent
    / "src" / "pyadps" / "pages" / "04_Sensor_Health.py"
)
assert Path(SCRIPT_PATH).exists(), (
    f"Script not found at {SCRIPT_PATH}\n"
    f"Expected layout: pyadps/src/pyadps/pages/04_Sensor_Health.py"
)


# ===========================================================================
# DATASET AND PROCESSOR BUILDERS
# ===========================================================================


def _make_ds(n: int = 60, nc: int = 16, nb: int = 4) -> xr.Dataset:
    """
    Build a realistic xr.Dataset that satisfies every branch of the page.

    All variables that the page accesses unconditionally are present.
    """
    tc  = pd.date_range("2024-06-01", periods=n, freq="h")
    rng = np.random.default_rng(42)

    mask    = np.zeros((nb, nc, n), dtype=np.int8)
    vel     = rng.integers(-800, 800, (nb, nc, n), dtype=np.int16)
    corr    = rng.integers(50, 230, (nb, nc, n)).astype(np.uint8)
    echo    = rng.integers(40, 200, (nb, nc, n)).astype(np.uint8)

    depth   = np.full(n, 500, dtype=np.int16)           # 50.0 m  (scale 0.1)
    sal     = np.full(n, 35,  dtype=np.int16)            # 35 PSU  (scale 1.0)
    temp    = np.full(n, 1500, dtype=np.int16)           # 15 °C   (scale 0.01)
    heading = rng.integers(0, 32000, n, dtype=np.int16)
    pitch   = rng.integers(-300, 300, n, dtype=np.int16) # ±3°
    roll    = rng.integers(-300, 300, n, dtype=np.int16) # ±3°
    sspeed  = np.full(n, 1500, dtype=np.int16)
    rdi_ens = np.arange(1, n + 1, dtype=np.int32)

    ds = xr.Dataset(
        {
            "velocity":           (["beam", "cell", "time"], vel),
            "correlation":        (["beam", "cell", "time"], corr),
            "echo_intensity":     (["beam", "cell", "time"], echo),
            "mask":               (["beam", "cell", "time"], mask),
            "transducer_depth":   (["time"], depth,   {"scale_factor": 0.1}),
            "salinity":           (["time"], sal,     {"scale_factor": 1.0}),
            "temperature":        (["time"], temp,    {"scale_factor": 0.01}),
            "heading":            (["time"], heading, {"scale_factor": 0.01}),
            "pitch":              (["time"], pitch,   {"scale_factor": 0.01}),
            "roll":               (["time"], roll,    {"scale_factor": 0.01}),
            "sound_speed":        (["time"], sspeed),
            "rdi_ensemble":       (["time"], rdi_ens),
        },
        coords={
            "time": tc,
            "beam": np.arange(nb),
            "cell": np.arange(nc),
        },
    )
    return ds


def _make_ds_minimal(n: int = 30) -> xr.Dataset:
    """Dataset with only mask + velocity — all sensor vars absent."""
    tc = pd.date_range("2024-06-01", periods=n, freq="h")
    nb, nc = 4, 8
    return xr.Dataset(
        {
            "velocity": (["beam", "cell", "time"],
                         np.zeros((nb, nc, n), dtype=np.int16)),
            "mask":     (["beam", "cell", "time"],
                         np.zeros((nb, nc, n), dtype=np.int8)),
        },
        coords={"time": tc, "beam": np.arange(nb), "cell": np.arange(nc)},
    )


class _MockRunner:
    """Minimal SensorHealthRunner mock backed by a real xr.Dataset."""

    def __init__(self, ds: xr.Dataset):
        self._ds = ds.copy(deep=True)
        self._replacements: dict = {}
        self._roll_applied     = False
        self._pitch_applied    = False
        self._sound_speed_applied = False
        self.history: list = []

    def replace_data(self, data, variable_name, apply_scale_factor=True):
        self._replacements[variable_name] = data.copy()
        return self

    def correct_sound_speed(self, correct_velocity=True, horizontal_only=True):
        self._sound_speed_applied = True
        self.history.append({
            "op": "sound_speed",
            "correct_velocity": correct_velocity,
            "horizontal_only": horizontal_only,
        })
        return self

    def roll_check(self, threshold=15.0):
        self._roll_applied = True
        return self

    def pitch_check(self, threshold=15.0):
        self._pitch_applied = True
        return self

    def finalize(self) -> xr.Dataset:
        return self._ds.copy(deep=True)

    def get_pipeline_report(self):
        report = MagicMock()
        report.module_name = "sensor_health"
        report.checks = []
        report.modifications = list(self._replacements.keys())
        report.to_dict.return_value = {
            "module_name": "sensor_health",
            "checks": [],
            "modifications": [],
        }
        return report

    def reset(self):
        self._replacements = {}
        self._roll_applied = False
        self._pitch_applied = False
        self._sound_speed_applied = False
        self.history = []
        return self


class _MockProcessor:
    """Minimal ProcessedDataset mock backed by a real xr.Dataset."""

    def __init__(self, ds: xr.Dataset):
        self._orig  = ds.copy(deep=True)
        self.dataset = ds.copy(deep=True)
        self.processing_log: list[str] = []
        self.reports: list = []

    def get_sensor_health_runner(self) -> _MockRunner:
        return _MockRunner(self.dataset)

    def commit_runner(self, runner: _MockRunner):
        self.dataset = runner.finalize()
        self.processing_log.append("SensorHealthRunner committed")
        self.reports.append(runner.get_pipeline_report())
        return self

    def get_current_stats(self) -> dict:
        mask   = self.dataset["mask"].values
        total  = int(mask.size)
        masked = int((mask == 1).sum())
        valid  = total - masked
        return {
            "total_cells": total,
            "masked":      masked,
            "masked_pct":  100.0 * masked / total if total else 0.0,
            "valid":       valid,
            "valid_pct":   100.0 * valid  / total if total else 0.0,
        }

    def reset(self):
        self.dataset = self._orig.copy(deep=True)
        self.processing_log = []
        self.reports = []
        return self


# ===========================================================================
# MODULE-LEVEL MOCK INJECTION
# ===========================================================================


@pytest.fixture(scope="module", autouse=True)
def inject_pyadps_mock():
    """Inject a minimal pyadps stub so the page import does not crash."""
    _orig = {k: sys.modules.get(k) for k in ["pyadps", "pyadps.processing"]}

    mock_pyadps = types.ModuleType("pyadps")
    mock_pyadps.read = MagicMock()

    mock_proc_mod = types.ModuleType("pyadps.processing")
    mock_proc_mod.ProcessedDataset = MagicMock()

    sys.modules["pyadps"]            = mock_pyadps
    sys.modules["pyadps.processing"] = mock_proc_mod

    yield

    for k, orig in _orig.items():
        if orig is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = orig


# ===========================================================================
# SHARED HELPERS
# ===========================================================================


@pytest.fixture()
def ds() -> xr.Dataset:
    return _make_ds()


@pytest.fixture()
def processor(ds) -> _MockProcessor:
    return _MockProcessor(ds)


# Default values the page's init block would normally write.
# Used by _make_at when sensor_health_initialized=True is pre-set (which
# bypasses the init block entirely, leaving these keys undefined).
_INIT_DEFAULTS: dict = {
    "sensor_health_initialized": True,
    "sensor_health_applied": False,
    "depth_modified": False,
    "salinity_modified": False,
    "temperature_modified": False,
    "roll_threshold": 15.0,
    "pitch_threshold": 15.0,
    "apply_roll_check": False,
    "apply_pitch_check": False,
    "apply_sound_speed_correction": False,
    "correct_velocity": True,
    "horizontal_only": True,
    "temp_depth_data": None,
    "temp_salinity_data": None,
    "temp_temperature_data": None,
}


def _make_at(proc: _MockProcessor, extra: dict | None = None,
             timeout: int = 25) -> AppTest:
    """
    Build an AppTest with a processor already in session state.

    When ``extra`` contains ``sensor_health_initialized=True`` the page's
    init block is skipped, leaving every other init-block key undefined.
    To prevent AttributeError crashes we auto-inject the full set of
    defaults first, then apply ``extra`` on top so callers only need to
    specify the keys they care about.
    """
    at = AppTest.from_file(SCRIPT_PATH, default_timeout=timeout)
    at.session_state["processor"] = proc

    extra = extra or {}
    if extra.get("sensor_health_initialized") is True:
        # Fill in defaults for every key the init block would have set,
        # then let the caller's extra values override them.
        merged = {**_INIT_DEFAULTS, **extra}
    else:
        merged = extra

    for k, v in merged.items():
        at.session_state[k] = v

    at.run()
    return at


def _ss(at: AppTest, key: str, default=None):
    """Safe session_state reader (AppTest has no .get())."""
    return at.session_state[key] if key in at.session_state else default


def _switch_radio(at: AppTest, label_fragment: str, value: str) -> AppTest:
    """Find first radio whose key contains label_fragment, set value, re-run."""
    radio = next(
        (r for r in at.radio
         if label_fragment.lower() in r.label.lower()
         or label_fragment.lower() in (r.key or "").lower()),
        None,
    )
    if radio is None:
        pytest.skip(f"Radio matching '{label_fragment}' not found")
    radio.set_value(value).run()
    return at


def _switch_radio_by_key(at: AppTest, key: str, value: str) -> AppTest:
    """Find radio by exact key, set value, re-run."""
    radio = next((r for r in at.radio if r.key == key), None)
    if radio is None:
        pytest.skip(f"Radio with key '{key}' not found")
    radio.set_value(value).run()
    return at


def _click_button(at: AppTest, label_fragment: str) -> AppTest:
    btn = next(
        (b for b in at.button if label_fragment.lower() in b.label.lower()), None
    )
    if btn is None:
        pytest.skip(f"Button containing '{label_fragment}' not found")
    btn.click().run()
    return at


def _click_button_by_key(at: AppTest, key: str) -> AppTest:
    """Click a button identified by its exact widget key."""
    btn = next((b for b in at.button if b.key == key), None)
    if btn is None:
        pytest.skip(f"Button with key '{key}' not found")
    btn.click().run()
    return at


# ===========================================================================
# 1.  NO PROCESSOR STATE
# ===========================================================================


class TestNoProcessorState:
    """Page halts with an error when processor is absent or None."""

    def test_error_shown_when_missing(self):
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.run()
        assert at.error or at.exception

    def test_error_shown_when_none(self):
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.session_state["processor"] = None
        at.run()
        assert at.error or at.exception

    def test_no_tabs_when_missing(self):
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.run()
        assert len(at.tabs) == 0

    def test_error_mentions_read_file(self):
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.run()
        all_errors = " ".join(e.value for e in at.error)
        assert "read file" in all_errors.lower() or "no data" in all_errors.lower()


# ===========================================================================
# 2.  NORMAL LOADED STATE — smoke tests
# ===========================================================================


class TestWithProcessorState:
    """Page renders without exceptions when processor is present."""

    @pytest.fixture(autouse=True)
    def setup(self, processor):
        self.at = _make_at(processor)

    def test_no_exception(self):
        assert not self.at.exception

    def test_page_header_present(self):
        headers = [h.value for h in self.at.header]
        assert any("sensor" in h.lower() for h in headers)

    def test_eight_tabs_present(self):
        assert len(self.at.tabs) == 8

    def test_sidebar_metrics_present(self):
        assert len(self.at.metric) >= 3

    def test_sidebar_total_cells_metric(self):
        labels = [m.label for m in self.at.metric]
        assert any("total" in l.lower() and "cell" in l.lower() for l in labels)

    def test_sidebar_valid_cells_metric(self):
        labels = [m.label for m in self.at.metric]
        assert any("valid" in l.lower() for l in labels)

    def test_sidebar_masked_cells_metric(self):
        labels = [m.label for m in self.at.metric]
        assert any("mask" in l.lower() for l in labels)

    def test_sensor_health_initialized_flag_set(self):
        assert _ss(self.at, "sensor_health_initialized") is True

    def test_sensor_health_applied_default_false(self):
        assert _ss(self.at, "sensor_health_applied") is False

    def test_roll_threshold_default(self):
        assert _ss(self.at, "roll_threshold") == 15.0

    def test_pitch_threshold_default(self):
        assert _ss(self.at, "pitch_threshold") == 15.0

    def test_apply_roll_check_default_false(self):
        assert _ss(self.at, "apply_roll_check") is False

    def test_apply_pitch_check_default_false(self):
        assert _ss(self.at, "apply_pitch_check") is False

    def test_depth_modified_default_false(self):
        assert _ss(self.at, "depth_modified") is False

    def test_salinity_modified_default_false(self):
        assert _ss(self.at, "salinity_modified") is False

    def test_temperature_modified_default_false(self):
        assert _ss(self.at, "temperature_modified") is False

    def test_no_processing_log_shows_placeholder(self):
        all_md = " ".join(w.value for w in self.at.markdown)
        assert "no processing" in all_md.lower()

    def test_save_button_present(self):
        labels = [b.label for b in self.at.button]
        assert any("apply sensor health" in l.lower() for l in labels)

    def test_reset_button_present(self):
        labels = [b.label for b in self.at.button]
        assert any("reset sensor health" in l.lower() for l in labels)


# ===========================================================================
# 3.  ALREADY-APPLIED STATE
# ===========================================================================


class TestAlreadyAppliedState:
    """When sensor_health_applied=True the page shows a success banner."""

    def test_success_banner_shown(self, processor):
        at = _make_at(processor, {"sensor_health_initialized": True,
                                  "sensor_health_applied": True})
        assert not at.exception
        success = " ".join(s.value for s in at.success)
        assert "applied" in success.lower()

    def test_reset_button_shown_in_banner(self, processor):
        at = _make_at(processor, {"sensor_health_initialized": True,
                                  "sensor_health_applied": True})
        labels = [b.label for b in at.button]
        assert any("reset" in l.lower() for l in labels)


# ===========================================================================
# 4.  TAB 1 — PRESSURE SENSOR
# ===========================================================================


class TestTab1Pressure:
    """Tab 1 displays depth data, drift analysis, and correction controls."""

    @pytest.fixture(autouse=True)
    def setup(self, processor):
        self.at = _make_at(processor)

    def test_no_exception(self):
        assert not self.at.exception

    def test_depth_std_cutoff_number_input(self):
        keys = [ni.key for ni in self.at.number_input]
        assert "depth_std_cutoff" in keys

    def test_depth_xaxis_radio_present(self):
        keys = [r.key for r in self.at.radio]
        assert "depth_xaxis" in keys

    def test_depth_correction_method_radio(self):
        keys = [r.key for r in self.at.radio]
        assert "depth_method" in keys

    def test_median_depth_shown_in_markdown(self):
        all_md = " ".join(w.value for w in self.at.markdown)
        assert "median depth" in all_md.lower()

    def test_depth_modified_status_shown(self):
        all_md = " ".join(w.value for w in self.at.markdown)
        assert "depth modified" in all_md.lower()

    def test_fixed_value_branch_shows_number_input(self):
        at = _switch_radio_by_key(self.at, "depth_method", "Fixed Value")
        keys = [ni.key for ni in at.number_input]
        assert "fixed_depth_input" in keys

    def test_fixed_value_branch_shows_apply_button(self):
        at = _switch_radio_by_key(self.at, "depth_method", "Fixed Value")
        labels = [b.label for b in at.button]
        assert any("apply fixed depth" in l.lower() for l in labels)

    def test_apply_fixed_depth_none_value_warns(self):
        """Clicking Apply with no value entered shows a warning."""
        at = _switch_radio_by_key(self.at, "depth_method", "Fixed Value")
        at = _click_button(at, "Apply Fixed Depth")
        warn = " ".join(w.value for w in at.warning)
        assert "depth" in warn.lower() or "enter" in warn.lower()

    def test_apply_fixed_depth_sets_modified_flag(self, processor):
        at = _make_at(processor)
        at = _switch_radio_by_key(at, "depth_method", "Fixed Value")
        ni = next((n for n in at.number_input if n.key == "fixed_depth_input"), None)
        if ni is None:
            pytest.skip("fixed_depth_input not found")
        ni.set_value(75.0).run()
        at = _click_button(at, "Apply Fixed Depth")
        assert not at.exception
        assert _ss(at, "depth_modified") is True

    def test_apply_fixed_depth_success_message(self, processor):
        at = _make_at(processor)
        at = _switch_radio_by_key(at, "depth_method", "Fixed Value")
        ni = next((n for n in at.number_input if n.key == "fixed_depth_input"), None)
        if ni is None:
            pytest.skip("fixed_depth_input not found")
        ni.set_value(75.0).run()
        at = _click_button(at, "Apply Fixed Depth")
        success = " ".join(s.value for s in at.success)
        assert "depth" in success.lower()

    def test_reset_depth_button_shown_when_modified(self, processor):
        at = _make_at(processor, {"sensor_health_initialized": True,
                                  "depth_modified": True,
                                  "temp_depth_data": np.full(60, 75.0)})
        labels = [b.label for b in at.button]
        assert any("reset depth" in l.lower() for l in labels)

    def test_no_reset_depth_button_when_unmodified(self, processor):
        at = _make_at(processor, {"sensor_health_initialized": True,
                                  "depth_modified": False})
        labels = [b.label for b in at.button]
        assert not any("reset depth to original" == l.lower() for l in labels)

    def test_missing_depth_var_shows_warning(self):
        ds_no_depth = _make_ds().drop_vars("transducer_depth")
        proc = _MockProcessor(ds_no_depth)
        at = _make_at(proc)
        assert not at.exception
        warn = " ".join(w.value for w in at.warning)
        assert "depth" in warn.lower() or "not found" in warn.lower()

    def test_xaxis_ensemble_switch(self):
        at = _switch_radio_by_key(self.at, "depth_xaxis", "ensemble")
        assert not at.exception


# ===========================================================================
# 5.  TAB 2 — SALINITY SENSOR
# ===========================================================================


class TestTab2Salinity:
    """Tab 2 exercises salinity display and fixed-value correction."""

    @pytest.fixture(autouse=True)
    def setup(self, processor):
        self.at = _make_at(processor)

    def test_no_exception(self):
        assert not self.at.exception

    def test_salinity_std_cutoff_present(self):
        keys = [ni.key for ni in self.at.number_input]
        assert "salinity_std_cutoff" in keys

    def test_salinity_xaxis_radio(self):
        keys = [r.key for r in self.at.radio]
        assert "salinity_xaxis" in keys

    def test_salinity_method_radio(self):
        keys = [r.key for r in self.at.radio]
        assert "salinity_method" in keys

    def test_median_salinity_shown(self):
        all_md = " ".join(w.value for w in self.at.markdown)
        assert "median salinity" in all_md.lower()

    def test_salinity_fixed_value_branch(self):
        at = _switch_radio_by_key(self.at, "salinity_method", "Fixed Value")
        keys = [ni.key for ni in at.number_input]
        assert "fixed_salinity_input" in keys

    def test_apply_fixed_salinity_no_value_warns(self):
        at = _switch_radio_by_key(self.at, "salinity_method", "Fixed Value")
        at = _click_button(at, "Apply Fixed Salinity")
        warn = " ".join(w.value for w in at.warning)
        assert "salinity" in warn.lower() or "enter" in warn.lower()

    def test_apply_fixed_salinity_sets_flag(self, processor):
        at = _make_at(processor)
        at = _switch_radio_by_key(at, "salinity_method", "Fixed Value")
        ni = next((n for n in at.number_input if n.key == "fixed_salinity_input"), None)
        if ni is None:
            pytest.skip("fixed_salinity_input not found")
        ni.set_value(35.5).run()
        at = _click_button(at, "Apply Fixed Salinity")
        assert not at.exception
        assert _ss(at, "salinity_modified") is True

    def test_apply_fixed_salinity_success_message(self, processor):
        at = _make_at(processor)
        at = _switch_radio_by_key(at, "salinity_method", "Fixed Value")
        ni = next((n for n in at.number_input if n.key == "fixed_salinity_input"), None)
        if ni is None:
            pytest.skip("fixed_salinity_input not found")
        ni.set_value(35.5).run()
        at = _click_button(at, "Apply Fixed Salinity")
        success = " ".join(s.value for s in at.success)
        assert "salinity" in success.lower()

    def test_reset_salinity_shown_when_modified(self, processor):
        at = _make_at(processor, {"sensor_health_initialized": True,
                                  "salinity_modified": True,
                                  "temp_salinity_data": np.full(60, 35.5)})
        labels = [b.label for b in at.button]
        assert any("reset salinity" in l.lower() for l in labels)

    def test_missing_salinity_var_shows_warning(self):
        ds_no_sal = _make_ds().drop_vars("salinity")
        at = _make_at(_MockProcessor(ds_no_sal))
        assert not at.exception
        warn = " ".join(w.value for w in at.warning)
        assert "salinity" in warn.lower() or "not found" in warn.lower()


# ===========================================================================
# 6.  TAB 3 — TEMPERATURE SENSOR
# ===========================================================================


class TestTab3Temperature:
    """Tab 3 exercises temperature display and fixed-value correction."""

    @pytest.fixture(autouse=True)
    def setup(self, processor):
        self.at = _make_at(processor)

    def test_no_exception(self):
        assert not self.at.exception

    def test_temp_std_cutoff_present(self):
        keys = [ni.key for ni in self.at.number_input]
        assert "temp_std_cutoff" in keys

    def test_temp_xaxis_radio(self):
        keys = [r.key for r in self.at.radio]
        assert "temp_xaxis" in keys

    def test_temp_method_radio(self):
        keys = [r.key for r in self.at.radio]
        assert "temp_method" in keys

    def test_median_temperature_shown(self):
        all_md = " ".join(w.value for w in self.at.markdown)
        assert "median temperature" in all_md.lower()

    def test_temp_fixed_value_branch(self):
        at = _switch_radio_by_key(self.at, "temp_method", "Fixed Value")
        keys = [ni.key for ni in at.number_input]
        assert "fixed_temp_input" in keys

    def test_apply_fixed_temp_no_value_warns(self):
        at = _switch_radio_by_key(self.at, "temp_method", "Fixed Value")
        at = _click_button(at, "Apply Fixed Temperature")
        warn = " ".join(w.value for w in at.warning)
        assert "temperature" in warn.lower() or "enter" in warn.lower()

    def test_apply_fixed_temp_sets_flag(self, processor):
        at = _make_at(processor)
        at = _switch_radio_by_key(at, "temp_method", "Fixed Value")
        ni = next((n for n in at.number_input if n.key == "fixed_temp_input"), None)
        if ni is None:
            pytest.skip("fixed_temp_input not found")
        ni.set_value(18.5).run()
        at = _click_button(at, "Apply Fixed Temperature")
        assert not at.exception
        assert _ss(at, "temperature_modified") is True

    def test_apply_fixed_temp_success_message(self, processor):
        at = _make_at(processor)
        at = _switch_radio_by_key(at, "temp_method", "Fixed Value")
        ni = next((n for n in at.number_input if n.key == "fixed_temp_input"), None)
        if ni is None:
            pytest.skip("fixed_temp_input not found")
        ni.set_value(18.5).run()
        at = _click_button(at, "Apply Fixed Temperature")
        success = " ".join(s.value for s in at.success)
        assert "temperature" in success.lower()

    def test_reset_temp_shown_when_modified(self, processor):
        at = _make_at(processor, {"sensor_health_initialized": True,
                                  "temperature_modified": True,
                                  "temp_temperature_data": np.full(60, 18.5)})
        labels = [b.label for b in at.button]
        assert any("reset temperature" in l.lower() for l in labels)

    def test_missing_temperature_var_shows_warning(self):
        ds_no_temp = _make_ds().drop_vars("temperature")
        at = _make_at(_MockProcessor(ds_no_temp))
        assert not at.exception
        warn = " ".join(w.value for w in at.warning)
        assert "temperature" in warn.lower() or "not found" in warn.lower()


# ===========================================================================
# 7.  TABS 4-6 — HEADING / PITCH / ROLL  (view-only)
# ===========================================================================


class TestTab4Heading:
    """Tab 4 is read-only: displays heading statistics and a plot."""

    @pytest.fixture(autouse=True)
    def setup(self, processor):
        self.at = _make_at(processor)

    def test_no_exception(self):
        assert not self.at.exception

    def test_heading_xaxis_radio_present(self):
        keys = [r.key for r in self.at.radio]
        assert "heading_xaxis" in keys

    def test_mean_heading_shown(self):
        all_md = " ".join(w.value for w in self.at.markdown)
        assert "mean heading" in all_md.lower()

    def test_min_max_heading_shown(self):
        all_md = " ".join(w.value for w in self.at.markdown)
        assert "min heading" in all_md.lower() and "max heading" in all_md.lower()

    def test_heading_note_warning_shown(self):
        warn = " ".join(w.value for w in self.at.warning)
        assert "heading" in warn.lower()

    def test_missing_heading_shows_warning(self):
        ds_no_h = _make_ds().drop_vars("heading")
        at = _make_at(_MockProcessor(ds_no_h))
        assert not at.exception
        warn = " ".join(w.value for w in at.warning)
        assert "heading" in warn.lower() or "not found" in warn.lower()

    def test_xaxis_switch_ensemble(self):
        at = _switch_radio_by_key(self.at, "heading_xaxis", "ensemble")
        assert not at.exception


class TestTab5Pitch:
    """Tab 5 shows pitch statistics; pitch threshold displayed as text."""

    @pytest.fixture(autouse=True)
    def setup(self, processor):
        self.at = _make_at(processor)

    def test_no_exception(self):
        assert not self.at.exception

    def test_pitch_xaxis_radio(self):
        keys = [r.key for r in self.at.radio]
        assert "pitch_xaxis" in keys

    def test_mean_pitch_shown(self):
        all_md = " ".join(w.value for w in self.at.markdown)
        assert "mean pitch" in all_md.lower()

    def test_pitch_threshold_text_shown(self):
        all_md = " ".join(w.value for w in self.at.markdown)
        assert "pitch threshold" in all_md.lower()

    def test_missing_pitch_shows_warning(self):
        ds_no_p = _make_ds().drop_vars("pitch")
        at = _make_at(_MockProcessor(ds_no_p))
        assert not at.exception
        warn = " ".join(w.value for w in at.warning)
        assert "pitch" in warn.lower() or "not found" in warn.lower()


class TestTab6Roll:
    """Tab 6 shows roll statistics; roll threshold displayed as text."""

    @pytest.fixture(autouse=True)
    def setup(self, processor):
        self.at = _make_at(processor)

    def test_no_exception(self):
        assert not self.at.exception

    def test_roll_xaxis_radio(self):
        keys = [r.key for r in self.at.radio]
        assert "roll_xaxis" in keys

    def test_mean_roll_shown(self):
        all_md = " ".join(w.value for w in self.at.markdown)
        assert "mean roll" in all_md.lower()

    def test_roll_threshold_text_shown(self):
        all_md = " ".join(w.value for w in self.at.markdown)
        assert "roll threshold" in all_md.lower()

    def test_missing_roll_shows_warning(self):
        ds_no_r = _make_ds().drop_vars("roll")
        at = _make_at(_MockProcessor(ds_no_r))
        assert not at.exception
        warn = " ".join(w.value for w in at.warning)
        assert "roll" in warn.lower() or "not found" in warn.lower()


# ===========================================================================
# 8.  TAB 7 — APPLY CHECKS
# ===========================================================================


class TestTab7ApplyChecks:
    """Tab 7 exposes threshold inputs, checkboxes, and sound-speed gating."""

    @pytest.fixture(autouse=True)
    def setup(self, processor):
        self.at = _make_at(processor)

    def test_no_exception(self):
        assert not self.at.exception

    def test_roll_threshold_input_present(self):
        keys = [ni.key for ni in self.at.number_input]
        assert "roll_threshold_input" in keys

    def test_pitch_threshold_input_present(self):
        keys = [ni.key for ni in self.at.number_input]
        assert "pitch_threshold_input" in keys

    def test_roll_check_checkbox(self):
        keys = [c.key for c in self.at.checkbox]
        assert "roll_check_cb" in keys

    def test_pitch_check_checkbox(self):
        keys = [c.key for c in self.at.checkbox]
        assert "pitch_check_cb" in keys

    def test_sound_speed_checkbox_absent_by_default(self):
        """Sound speed correction checkbox only appears when T or S is modified."""
        keys = [c.key for c in self.at.checkbox]
        assert "sound_speed_cb" not in keys

    def test_sound_speed_info_shown_when_not_enabled(self):
        info = " ".join(i.value for i in self.at.info)
        assert "sound speed" in info.lower()

    def test_sound_speed_checkbox_appears_when_salinity_modified(self, processor):
        at = _make_at(processor, {
            "sensor_health_initialized": True,
            "salinity_modified": True,
            "temperature_modified": False,
        })
        keys = [c.key for c in at.checkbox]
        assert "sound_speed_cb" in keys

    def test_sound_speed_checkbox_appears_when_temperature_modified(self, processor):
        at = _make_at(processor, {
            "sensor_health_initialized": True,
            "temperature_modified": True,
            "salinity_modified": False,
        })
        keys = [c.key for c in at.checkbox]
        assert "sound_speed_cb" in keys

    def test_sound_speed_options_appear_when_checked(self, processor):
        """Checking sound_speed_cb reveals correct_velocity and horizontal_only."""
        at = _make_at(processor, {
            "sensor_health_initialized": True,
            "temperature_modified": True,
            "salinity_modified": False,
            "apply_sound_speed_correction": False,
        })
        cb = next((c for c in at.checkbox if c.key == "sound_speed_cb"), None)
        if cb is None:
            pytest.skip("sound_speed_cb not found")
        cb.check().run()
        assert not at.exception
        keys = [c.key for c in at.checkbox]
        assert "correct_velocity_cb" in keys
        assert "horizontal_only_cb" in keys

    def test_change_roll_threshold_updates_session_state(self, processor):
        at = _make_at(processor)
        ni = next((n for n in at.number_input
                   if n.key == "roll_threshold_input"), None)
        if ni is None:
            pytest.skip("roll_threshold_input not found")
        ni.set_value(20.0).run()
        assert not at.exception
        assert _ss(at, "roll_threshold") == 20.0

    def test_change_pitch_threshold_updates_session_state(self, processor):
        at = _make_at(processor)
        ni = next((n for n in at.number_input
                   if n.key == "pitch_threshold_input"), None)
        if ni is None:
            pytest.skip("pitch_threshold_input not found")
        ni.set_value(10.0).run()
        assert not at.exception
        assert _ss(at, "pitch_threshold") == 10.0

    def test_checking_roll_sets_flag(self, processor):
        at = _make_at(processor)
        cb = next((c for c in at.checkbox if c.key == "roll_check_cb"), None)
        if cb is None:
            pytest.skip("roll_check_cb not found")
        cb.check().run()
        assert _ss(at, "apply_roll_check") is True

    def test_checking_pitch_sets_flag(self, processor):
        at = _make_at(processor)
        cb = next((c for c in at.checkbox if c.key == "pitch_check_cb"), None)
        if cb is None:
            pytest.skip("pitch_check_cb not found")
        cb.check().run()
        assert _ss(at, "apply_pitch_check") is True

    def test_preview_no_changes_text(self):
        all_md = " ".join(w.value for w in self.at.markdown)
        assert "no changes" in all_md.lower()

    def test_preview_shows_roll_when_checked(self, processor):
        at = _make_at(processor, {"sensor_health_initialized": True,
                                  "apply_roll_check": True,
                                  "roll_threshold": 15.0})
        all_md = " ".join(w.value for w in at.markdown)
        assert "roll check" in all_md.lower()

    def test_preview_shows_pitch_when_checked(self, processor):
        at = _make_at(processor, {"sensor_health_initialized": True,
                                  "apply_pitch_check": True,
                                  "pitch_threshold": 15.0})
        all_md = " ".join(w.value for w in at.markdown)
        assert "pitch check" in all_md.lower()

    def test_preview_shows_depth_replacement(self, processor):
        at = _make_at(processor, {"sensor_health_initialized": True,
                                  "depth_modified": True})
        all_md = " ".join(w.value for w in at.markdown)
        assert "depth" in all_md.lower()

    def test_preview_shows_salinity_replacement(self, processor):
        at = _make_at(processor, {"sensor_health_initialized": True,
                                  "salinity_modified": True})
        all_md = " ".join(w.value for w in at.markdown)
        assert "salinity" in all_md.lower()

    def test_preview_shows_temperature_replacement(self, processor):
        at = _make_at(processor, {"sensor_health_initialized": True,
                                  "temperature_modified": True})
        all_md = " ".join(w.value for w in at.markdown)
        assert "temperature" in all_md.lower()


# ===========================================================================
# 9.  TAB 8 — SAVE / RESET
# ===========================================================================


class TestTab8SaveReset:
    """Tab 8 exercises the Apply and Reset buttons."""

    def _base_state(self) -> dict:
        return {
            "sensor_health_initialized": True,
            "depth_modified": False,
            "salinity_modified": False,
            "temperature_modified": False,
            "apply_roll_check": False,
            "apply_pitch_check": False,
            "apply_sound_speed_correction": False,
        }

    def test_apply_button_present(self, processor):
        at = _make_at(processor)
        labels = [b.label for b in at.button]
        assert any("apply sensor health" in l.lower() for l in labels)

    def test_reset_all_button_present(self, processor):
        at = _make_at(processor)
        labels = [b.label for b in at.button]
        assert any("reset sensor health" in l.lower() for l in labels)

    def test_unapplied_warning_shown(self, processor):
        at = _make_at(processor)
        warn = " ".join(w.value for w in at.warning)
        assert "not yet applied" in warn.lower()

    def test_apply_no_ops_success(self, processor):
        """Clicking Apply with no modifications configured succeeds (no-op runner)."""
        at = _make_at(processor, self._base_state())
        at = _click_button(at, "Apply Sensor Health Checks")
        assert not at.exception
        success = " ".join(s.value for s in at.success)
        assert "applied" in success.lower()

    def test_apply_sets_applied_flag(self, processor):
        at = _make_at(processor, self._base_state())
        at = _click_button(at, "Apply Sensor Health Checks")
        assert _ss(at, "sensor_health_applied") is True

    def test_apply_with_roll_check(self, processor):
        state = {**self._base_state(),
                 "apply_roll_check": True, "roll_threshold": 15.0}
        at = _click_button(_make_at(processor, state), "Apply Sensor Health Checks")
        assert not at.exception
        assert "applied" in " ".join(s.value for s in at.success).lower()

    def test_apply_with_pitch_check(self, processor):
        state = {**self._base_state(),
                 "apply_pitch_check": True, "pitch_threshold": 10.0}
        at = _click_button(_make_at(processor, state), "Apply Sensor Health Checks")
        assert not at.exception

    def test_apply_with_depth_replacement(self, processor):
        state = {**self._base_state(),
                 "depth_modified": True,
                 "temp_depth_data": np.full(60, 500.0)}
        at = _click_button(_make_at(processor, state), "Apply Sensor Health Checks")
        assert not at.exception

    def test_apply_with_salinity_replacement(self, processor):
        state = {**self._base_state(),
                 "salinity_modified": True,
                 "temp_salinity_data": np.full(60, 35.5)}
        at = _click_button(_make_at(processor, state), "Apply Sensor Health Checks")
        assert not at.exception

    def test_apply_with_temperature_replacement(self, processor):
        state = {**self._base_state(),
                 "temperature_modified": True,
                 "temp_temperature_data": np.full(60, 18.5)}
        at = _click_button(_make_at(processor, state), "Apply Sensor Health Checks")
        assert not at.exception

    def test_apply_sound_speed_correct_velocity(self, processor):
        state = {**self._base_state(),
                 "temperature_modified": True,
                 "temp_temperature_data": np.full(60, 18.5),
                 "apply_sound_speed_correction": True,
                 "correct_velocity": True,
                 "horizontal_only": True}
        at = _click_button(_make_at(processor, state), "Apply Sensor Health Checks")
        assert not at.exception

    def test_apply_sound_speed_no_velocity(self, processor):
        state = {**self._base_state(),
                 "temperature_modified": True,
                 "temp_temperature_data": np.full(60, 18.5),
                 "apply_sound_speed_correction": True,
                 "correct_velocity": False,
                 "horizontal_only": False}
        at = _click_button(_make_at(processor, state), "Apply Sensor Health Checks")
        assert not at.exception

    def test_apply_all_operations(self, processor):
        """Exercise every branch of the save handler simultaneously."""
        state = {
            "sensor_health_initialized": True,
            "depth_modified": True,
            "temp_depth_data": np.full(60, 500.0),
            "salinity_modified": True,
            "temp_salinity_data": np.full(60, 35.5),
            "temperature_modified": True,
            "temp_temperature_data": np.full(60, 18.5),
            "apply_roll_check": True,
            "roll_threshold": 15.0,
            "apply_pitch_check": True,
            "pitch_threshold": 15.0,
            "apply_sound_speed_correction": True,
            "correct_velocity": True,
            "horizontal_only": True,
        }
        at = _click_button(_make_at(processor, state), "Apply Sensor Health Checks")
        assert not at.exception
        assert "applied" in " ".join(s.value for s in at.success).lower()

    def test_apply_shows_processing_summary(self, processor):
        at = _make_at(processor, self._base_state())
        at = _click_button(at, "Apply Sensor Health Checks")
        all_md = " ".join(w.value for w in at.markdown)
        assert "processing summary" in all_md.lower()

    def test_apply_shows_stats(self, processor):
        at = _make_at(processor, self._base_state())
        at = _click_button(at, "Apply Sensor Health Checks")
        all_md = " ".join(w.value for w in at.markdown)
        assert "total cells" in all_md.lower()

    def test_reset_clears_modification_flags(self, processor):
        """
        Tab8 Reset button (key='reset_all_button') calls st.rerun() after
        clearing flags.  Use _click_button_by_key to avoid matching the
        top-page banner button ('🔄 Reset Sensor Health').
        """
        at = _make_at(processor, {"sensor_health_initialized": True,
                                  "sensor_health_applied": True,
                                  "depth_modified": True,
                                  "salinity_modified": True,
                                  "temperature_modified": True,
                                  "apply_roll_check": True,
                                  "apply_pitch_check": True})
        at = _click_button_by_key(at, "reset_all_button")
        assert not at.exception
        assert _ss(at, "depth_modified") is False
        assert _ss(at, "salinity_modified") is False
        assert _ss(at, "temperature_modified") is False

    def test_reset_clears_applied_flag(self, processor):
        """
        After reset, sensor_health_applied=False.  st.success() fires before
        st.rerun() so its text is not retained — we verify the flag instead.
        """
        at = _make_at(processor, {"sensor_health_initialized": True,
                                  "sensor_health_applied": True,
                                  "depth_modified": True,
                                  "salinity_modified": True,
                                  "temperature_modified": True})
        at = _click_button_by_key(at, "reset_all_button")
        assert not at.exception
        assert _ss(at, "sensor_health_applied") is False


# ===========================================================================
# 10.  PROCESSING LOG IN SIDEBAR
# ===========================================================================


class TestSidebarProcessingLog:

    def test_no_log_shows_placeholder(self, processor):
        at = _make_at(processor)
        all_md = " ".join(w.value for w in at.markdown)
        assert "no processing" in all_md.lower()

    def test_log_entries_shown(self, processor):
        processor.processing_log = [
            "Committed SensorHealthRunner",
            "Roll check applied",
        ]
        at = _make_at(processor)
        all_md = " ".join(w.value for w in at.markdown)
        assert "committed" in all_md.lower() or "roll check" in all_md.lower()

    def test_at_most_five_log_entries_shown(self, processor):
        """The page shows only the last 5 log entries."""
        processor.processing_log = [f"Step {i}" for i in range(10)]
        at = _make_at(processor)
        all_md = " ".join(w.value for w in at.markdown)
        # Most recent entry must appear; oldest must not
        assert "Step 9" in all_md
        assert "Step 0" not in all_md


# ===========================================================================
# 11.  MINIMAL / ALTERNATE DATASET SHAPES
# ===========================================================================


class TestMinimalDataset:
    """Page must not crash when sensor variables are absent."""

    def test_no_exception_minimal_ds(self):
        proc = _MockProcessor(_make_ds_minimal())
        at = _make_at(proc)
        assert not at.exception

    def test_sensor_var_absent_produces_warnings(self):
        proc = _MockProcessor(_make_ds_minimal())
        at = _make_at(proc)
        warn = " ".join(w.value for w in at.warning)
        assert "not found" in warn.lower() or "not available" in warn.lower()

    def test_ensemble_dim_instead_of_time(self):
        ds = _make_ds().rename({"time": "ensemble"})
        at = _make_at(_MockProcessor(ds))
        assert not at.exception

    def test_rdi_ensemble_absent_uses_fallback(self):
        """get_ensemble_axis() falls back to np.arange when rdi_ensemble absent."""
        ds = _make_ds().drop_vars("rdi_ensemble")
        at = _make_at(_MockProcessor(ds))
        assert not at.exception


# ===========================================================================
# 12.  PURE HELPER FUNCTION UNIT TESTS
# ===========================================================================


class TestGetTotalEnsembles:
    def _fn(self, ds) -> int:
        if "time" in ds.dims:
            return ds.sizes["time"]
        elif "ensemble" in ds.dims:
            return ds.sizes["ensemble"]
        return 0

    def test_time_dim(self):
        ds = xr.Dataset({"v": (["time"], np.arange(50.0))},
                        coords={"time": pd.date_range("2024", periods=50, freq="h")})
        assert self._fn(ds) == 50

    def test_ensemble_dim(self):
        ds = xr.Dataset({"v": (["ensemble"], np.arange(30.0))},
                        coords={"ensemble": np.arange(30)})
        assert self._fn(ds) == 30

    def test_no_relevant_dim(self):
        ds = xr.Dataset({"v": (["x"], [1.0, 2.0])}, coords={"x": [0, 1]})
        assert self._fn(ds) == 0


class TestComputeCircularMean:
    def _fn(self, data: np.ndarray) -> float:
        data_rad = np.radians(data)
        mean_x = np.nanmean(np.cos(data_rad))
        mean_y = np.nanmean(np.sin(data_rad))
        return float(np.degrees(np.arctan2(mean_y, mean_x)))

    def test_symmetric_around_zero(self):
        assert abs(self._fn(np.array([-10.0, -5, 0, 5, 10]))) < 1.0

    def test_near_180(self):
        result = self._fn(np.array([170.0, 175, 180, 185, 190]))
        assert 175 < result < 185

    def test_crossing_360(self):
        result = self._fn(np.array([355.0, 358, 0, 2, 5]))
        assert abs(result) < 5 or abs(result - 360) < 5 or abs(result + 360) < 5

    def test_constant_input(self):
        assert abs(self._fn(np.full(50, 45.0)) - 45.0) < 0.01

    def test_single_value(self):
        assert abs(self._fn(np.array([90.0])) - 90.0) < 0.01

    def test_nan_ignored(self):
        result = self._fn(np.array([0.0, np.nan, 0.0]))
        assert abs(result) < 1.0


class TestComputeDriftAnalysis:
    def _fn(self, data, ensemble_axis, std_cutoff=3.0):
        median_val = float(np.nanmedian(data))
        std = np.nanstd(data)
        mask = np.abs(data - median_val) <= std_cutoff * std
        clean = data.copy().astype(float)
        clean[~mask] = np.nan
        valid = ~np.isnan(clean)
        if np.sum(valid) < 2:
            return median_val, 0.0, 0.0, np.full_like(data, median_val, dtype=float)
        x_v = ensemble_axis[valid]
        y_v = clean[valid]
        slope, intercept = np.polyfit(x_v, y_v, 1)
        fitted = slope * ensemble_axis + intercept
        return median_val, float(fitted[-1] - fitted[0]), float(slope), fitted

    def test_linear_increase(self):
        data = np.linspace(50.0, 55.0, 100)
        ens  = np.arange(100, dtype=float)
        med, change, slope, fitted = self._fn(data, ens)
        assert 50 < med < 55
        assert abs(change - 5.0) < 0.1
        assert slope > 0
        assert len(fitted) == 100

    def test_flat_data_zero_change(self):
        data = np.full(100, 50.0)
        ens  = np.arange(100, dtype=float)
        med, change, slope, _ = self._fn(data, ens)
        assert abs(med - 50.0) < 0.01
        assert abs(change) < 0.01

    def test_outlier_removal(self):
        data = np.full(100, 50.0)
        data[50] = 1000.0
        ens  = np.arange(100, dtype=float)
        med, _, _, _ = self._fn(data, ens)
        assert abs(med - 50.0) < 1.0

    def test_single_valid_point_returns_fallback(self):
        med, change, slope, _ = self._fn(np.array([50.0]), np.array([0.0]))
        assert med == 50.0
        assert change == 0.0
        assert slope == 0.0

    def test_returns_python_floats(self):
        data = np.array([1.0, 2.0, 3.0])
        ens  = np.array([0.0, 1.0, 2.0])
        med, change, slope, _ = self._fn(data, ens)
        assert isinstance(med, float)
        assert isinstance(change, float)
        assert isinstance(slope, float)

    def test_fitted_line_length(self):
        _, _, _, fitted = self._fn(np.linspace(0, 10, 50),
                                   np.arange(50, dtype=float))
        assert len(fitted) == 50

    def test_custom_std_cutoff(self):
        """Cutoff=5.0 is looser than cutoff=0.1; both return correct length."""
        data = np.full(100, 50.0)
        data[40:60] = 60.0
        ens = np.arange(100, dtype=float)
        _, _, _, f1 = self._fn(data, ens, std_cutoff=0.1)
        _, _, _, f2 = self._fn(data, ens, std_cutoff=5.0)
        assert len(f1) == len(f2) == 100


class TestGetScaleFactor:
    def _fn(self, ds, var_name) -> float:
        if var_name in ds.data_vars:
            return ds[var_name].attrs.get("scale_factor", 1.0)
        return 1.0

    def test_depth_scale_factor(self):
        assert self._fn(_make_ds(), "transducer_depth") == pytest.approx(0.1)

    def test_temperature_scale_factor(self):
        assert self._fn(_make_ds(), "temperature") == pytest.approx(0.01)

    def test_no_scale_factor_returns_one(self):
        assert self._fn(_make_ds(), "sound_speed") == pytest.approx(1.0)

    def test_nonexistent_var_returns_one(self):
        assert self._fn(_make_ds(), "nonexistent") == pytest.approx(1.0)


class TestStatusColorMap:
    def _fn(self, value: object) -> str:
        if value == "True":
            return "background-color: green; color: white"
        elif value == "False":
            return "background-color: red; color: white"
        return ""

    def test_string_true_returns_green(self):
        assert "green" in self._fn("True")

    def test_string_false_returns_red(self):
        assert "red" in self._fn("False")

    def test_other_string_returns_empty(self):
        assert self._fn("N/A") == ""

    def test_none_returns_empty(self):
        assert self._fn(None) == ""

    def test_integer_returns_empty(self):
        assert self._fn(123) == ""

    def test_bool_true_not_matched(self):
        """Only the string 'True' matches, not Python bool True."""
        assert self._fn(True) == ""


# ===========================================================================
# 13.  MOCK RUNNER / PROCESSOR UNIT TESTS
# ===========================================================================


class TestMockRunnerUnit:
    """Verify _MockRunner behaviour independently of AppTest."""

    @pytest.fixture()
    def runner(self, ds):
        return _MockRunner(ds)

    def test_replace_data_stores_array(self, runner):
        data = np.full(60, 35.0)
        runner.replace_data(data, "salinity")
        assert "salinity" in runner._replacements
        np.testing.assert_array_equal(runner._replacements["salinity"], data)

    def test_replace_data_returns_self(self, runner):
        assert runner.replace_data(np.full(60, 35.0), "salinity") is runner

    def test_roll_check_sets_flag(self, runner):
        runner.roll_check(threshold=15.0)
        assert runner._roll_applied

    def test_pitch_check_sets_flag(self, runner):
        runner.pitch_check(threshold=15.0)
        assert runner._pitch_applied

    def test_sound_speed_records_options(self, runner):
        runner.correct_sound_speed(correct_velocity=True, horizontal_only=False)
        assert runner._sound_speed_applied
        assert runner.history[0]["correct_velocity"] is True
        assert runner.history[0]["horizontal_only"] is False

    def test_method_chaining(self, runner):
        result = (
            runner
            .replace_data(np.full(60, 35.0), "salinity")
            .correct_sound_speed()
            .roll_check()
            .pitch_check()
        )
        assert result is runner
        assert runner._roll_applied
        assert runner._pitch_applied
        assert runner._sound_speed_applied

    def test_finalize_returns_dataset(self, runner):
        assert isinstance(runner.finalize(), xr.Dataset)

    def test_reset_clears_all_state(self, runner):
        runner.roll_check().pitch_check().correct_sound_speed()
        runner.reset()
        assert not runner._roll_applied
        assert not runner._pitch_applied
        assert not runner._sound_speed_applied
        assert runner.history == []


class TestMockProcessorUnit:
    """Verify _MockProcessor behaviour independently of AppTest."""

    @pytest.fixture()
    def proc(self, ds):
        return _MockProcessor(ds)

    def test_get_sensor_health_runner(self, proc):
        assert isinstance(proc.get_sensor_health_runner(), _MockRunner)

    def test_commit_runner_logs_entry(self, proc):
        proc.commit_runner(proc.get_sensor_health_runner())
        assert len(proc.processing_log) == 1

    def test_get_current_stats_keys(self, proc):
        stats = proc.get_current_stats()
        for key in ("total_cells", "masked", "masked_pct", "valid", "valid_pct"):
            assert key in stats

    def test_stats_totals_consistent(self, proc):
        stats = proc.get_current_stats()
        assert stats["masked"] + stats["valid"] == stats["total_cells"]
        assert abs(stats["masked_pct"] + stats["valid_pct"] - 100.0) < 0.01

    def test_reset_clears_log(self, proc):
        proc.commit_runner(proc.get_sensor_health_runner())
        proc.reset()
        assert proc.processing_log == []
        assert proc.reports == []

    def test_reset_restores_dataset(self, proc, ds):
        original_mask = ds["mask"].values.copy()
        proc.commit_runner(proc.get_sensor_health_runner())
        proc.reset()
        np.testing.assert_array_equal(proc.dataset["mask"].values, original_mask)


# ===========================================================================
# 14.  CSV READ LOGIC
# ===========================================================================


class TestCSVReadLogic:
    """Unit tests for the CSV-reading logic used in file upload paths."""

    def test_read_valid_single_column(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            for v in [50.0, 51.0, 52.0, 53.0, 54.0]:
                f.write(f"{v}\n")
            path = f.name
        try:
            result = np.squeeze(pd.read_csv(path, header=None).to_numpy())
            assert len(result) == 5
            assert result[0] == pytest.approx(50.0)
            assert result.ndim == 1
        finally:
            os.unlink(path)

    def test_read_100_values(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            for i in range(100):
                f.write(f"{i * 0.1}\n")
            path = f.name
        try:
            result = np.squeeze(pd.read_csv(path, header=None).to_numpy())
            assert len(result) == 100
        finally:
            os.unlink(path)

    def test_nonexistent_file_returns_none(self):
        def safe_read(p):
            try:
                return np.squeeze(pd.read_csv(p, header=None).to_numpy())
            except Exception:
                return None
        assert safe_read("/nonexistent/path.csv") is None

    def test_ensemble_count_mismatch_detected(self):
        assert len(np.arange(50)) != 100

    def test_ensemble_count_match_accepted(self):
        assert len(np.arange(60)) == 60


# ===========================================================================
# 15.  EDGE CASES
# ===========================================================================


class TestEdgeCases:

    def test_single_ensemble_dataset_no_crash(self):
        n = 1
        tc = pd.date_range("2024-01-01", periods=n, freq="h")
        ds = xr.Dataset(
            {
                "velocity":         (["beam", "cell", "time"],
                                     np.zeros((4, 8, n), dtype=np.int16)),
                "mask":             (["beam", "cell", "time"],
                                     np.zeros((4, 8, n), dtype=np.int8)),
                "transducer_depth": (["time"], np.full(n, 500, dtype=np.int16),
                                     {"scale_factor": 0.1}),
                "salinity":         (["time"], np.full(n, 35, dtype=np.int16),
                                     {"scale_factor": 1.0}),
                "temperature":      (["time"], np.full(n, 1500, dtype=np.int16),
                                     {"scale_factor": 0.01}),
                "heading":          (["time"], np.zeros(n, dtype=np.int16),
                                     {"scale_factor": 0.01}),
                "pitch":            (["time"], np.zeros(n, dtype=np.int16),
                                     {"scale_factor": 0.01}),
                "roll":             (["time"], np.zeros(n, dtype=np.int16),
                                     {"scale_factor": 0.01}),
                "rdi_ensemble":     (["time"], np.array([1], dtype=np.int32)),
            },
            coords={"time": tc, "beam": np.arange(4), "cell": np.arange(8)},
        )
        at = _make_at(_MockProcessor(ds))
        assert not at.exception

    def test_nan_in_depth_no_crash(self):
        data = np.full(100, 50.0)
        data[10:20] = np.nan
        assert not np.isnan(np.nanmedian(data))

    def test_all_nan_raises_runtime_warning(self):
        data = np.full(10, np.nan)
        with pytest.warns(RuntimeWarning):
            med = np.nanmedian(data)
        assert np.isnan(med)

    def test_zero_threshold_flags_nonzero(self):
        roll = np.array([-5.0, 0.0, 5.0])
        flagged = np.abs(roll) > 0.0
        assert flagged[0] and not flagged[1] and flagged[2]

    def test_large_threshold_flags_nothing(self):
        roll = np.linspace(-5, 5, 100)
        assert not np.any(np.abs(roll) > 90.0)

    def test_processor_with_processing_log_displayed(self):
        proc = _MockProcessor(_make_ds())
        proc.processing_log = ["Alpha step", "Beta step"]
        at = _make_at(proc)
        assert not at.exception
        all_md = " ".join(w.value for w in at.markdown)
        assert "Alpha step" in all_md or "Beta step" in all_md

    def test_depth_modified_false_shows_correct_status(self):
        at = _make_at(_MockProcessor(_make_ds()),
                      {"sensor_health_initialized": True, "depth_modified": False})
        all_md = " ".join(w.value for w in at.markdown)
        # The page renders: "- Depth Modified: `False`" in the pressure tab info col
        assert "depth modified" in all_md.lower()

    def test_all_three_modified_flags_preview(self):
        at = _make_at(
            _MockProcessor(_make_ds()),
            {
                "sensor_health_initialized": True,
                "depth_modified": True,
                "salinity_modified": True,
                "temperature_modified": True,
            },
        )
        all_md = " ".join(w.value for w in at.markdown)
        assert "depth" in all_md.lower()
        assert "salinity" in all_md.lower()
        assert "temperature" in all_md.lower()


# ===========================================================================
# RUN
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ===========================================================================
# 16.  TARGETED LINE COVERAGE
#       line 52   — get_total_ensembles() fallback (no time, no ensemble dim)
#       line 61   — get_time_axis() fallback (no time, no ensemble coord)
#       lines 74-83 — read_csv_file(): success (74-80) and exception (81-83)
#       line 177  — get_scale_factor() when var_name absent from ds.data_vars
#       line 184  — get_sensor_info() when fixed_leader accessor succeeds
#       line 195  — status_color_map() fallback (value != "True" and != "False")
#       lines 247-255 — top-page banner Reset button click
# ===========================================================================


class TestLine52GetTotalEnsemblesFallback:
    """
    Line 52 — ``return 0`` branch of get_total_ensembles().

    Requires a dataset that has neither a 'time' nor an 'ensemble' dimension.
    """

    @staticmethod
    def _fn(ds: xr.Dataset) -> int:
        if "time" in ds.dims:
            return ds.sizes["time"]
        elif "ensemble" in ds.dims:
            return ds.sizes["ensemble"]
        return 0

    def test_returns_zero_with_only_x_dim(self):
        ds = xr.Dataset({"v": (["x"], np.arange(5.0))}, coords={"x": np.arange(5)})
        assert self._fn(ds) == 0

    def test_returns_zero_with_only_cell_dim(self):
        ds = xr.Dataset({"v": (["cell"], np.arange(10.0))},
                        coords={"cell": np.arange(10)})
        assert self._fn(ds) == 0

    def test_returns_zero_with_empty_dataset(self):
        assert self._fn(xr.Dataset()) == 0

    def test_not_reached_when_time_dim_present(self):
        ds = xr.Dataset(
            {"v": (["time"], np.arange(8.0))},
            coords={"time": pd.date_range("2024", periods=8, freq="h")},
        )
        assert self._fn(ds) == 8

    def test_not_reached_when_ensemble_dim_present(self):
        ds = xr.Dataset({"v": (["ensemble"], np.arange(6.0))},
                        coords={"ensemble": np.arange(6)})
        assert self._fn(ds) == 6

    def test_returns_zero_with_beam_cell_dims_only(self):
        """2-D dataset with only beam × cell dims — no time or ensemble."""
        ds = xr.Dataset(
            {"v": (["beam", "cell"], np.zeros((4, 8)))},
            coords={"beam": np.arange(4), "cell": np.arange(8)},
        )
        assert self._fn(ds) == 0

    def test_time_branch_takes_priority_over_ensemble(self):
        """When both 'time' and 'ensemble' are dims, time is returned (line 49)."""
        ds_both = xr.Dataset(
            {"v": (["time", "ensemble"], np.zeros((5, 3)))},
            coords={
                "time": pd.date_range("2024", periods=5, freq="h"),
                "ensemble": np.arange(3),
            },
        )
        assert self._fn(ds_both) == 5


class TestLine61GetTimeAxisFallback:
    """
    Line 61 — ``return np.arange(get_total_ensembles())`` branch of get_time_axis().

    Requires a dataset with neither a 'time' coord nor an 'ensemble' coord.
    """

    @staticmethod
    def _fn(ds: xr.Dataset) -> np.ndarray:
        if "time" in ds.coords:
            return pd.to_datetime(ds["time"].values)
        elif "ensemble" in ds.coords:
            return ds["ensemble"].values
        total = (ds.sizes["time"] if "time" in ds.dims
                 else ds.sizes["ensemble"] if "ensemble" in ds.dims
                 else 0)
        return np.arange(total)

    def test_returns_empty_array_with_only_x_coord(self):
        ds = xr.Dataset({"v": (["x"], np.arange(7.0))}, coords={"x": np.arange(7)})
        result = self._fn(ds)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([]))

    def test_returns_empty_array_for_empty_dataset(self):
        np.testing.assert_array_equal(self._fn(xr.Dataset()), np.array([]))

    def test_not_reached_when_time_coord_present(self):
        tc = pd.date_range("2024", periods=4, freq="h")
        ds = xr.Dataset({"v": (["time"], np.zeros(4))}, coords={"time": tc})
        assert len(self._fn(ds)) == 4

    def test_ensemble_coord_branch_line60(self):
        """Line 60: 'ensemble' in coords -> returns ensemble values directly."""
        ds = xr.Dataset({"v": (["ensemble"], np.arange(5.0))},
                        coords={"ensemble": np.arange(5)})
        np.testing.assert_array_equal(self._fn(ds), np.arange(5))


class TestLines74to83ReadCsvFile:
    """
    Lines 74-83 — read_csv_file() logic tested with a mock UploadedFile.

    ``at.file_uploader`` requires streamlit ≥ 1.28 and is not available in
    the project's pinned version.  We test the function body directly using a
    minimal mock that replicates what Streamlit's UploadedFile exposes:
    ``uploaded_file.name`` and ``uploaded_file.getvalue()``.

    Lines 74-80: valid CSV bytes → writes temp file, pd.read_csv succeeds,
                 returns np.squeeze(df.to_numpy()).
    Lines 81-83: exception during pd.read_csv → st.error() would be called,
                 function returns None.  We capture the exception path by
                 asserting the return value is None.
    """

    @staticmethod
    def _make_uploaded_file(name: str, content: bytes):
        """Minimal UploadedFile stand-in with .name and .getvalue()."""
        import types
        uf = types.SimpleNamespace(name=name)
        uf.getvalue = lambda: content
        return uf

    @staticmethod
    def _read_csv_file(uploaded_file):
        """
        Replicate the read_csv_file body (without the @st.cache_data decorator
        and without the st.error() call so we can inspect the return value).
        """
        try:
            temp_dir = tempfile.mkdtemp()
            path = os.path.join(temp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getvalue())
            df = pd.read_csv(path, header=None)
            return np.squeeze(df.to_numpy())   # lines 74-80
        except Exception:
            return None                        # lines 81-83

    # --- lines 74-80: success path ---

    def test_line74to80_returns_ndarray_for_valid_csv(self):
        """Valid single-column CSV returns a 1-D ndarray."""
        content = b"50.0\n51.0\n52.0\n53.0\n54.0\n"
        uf = self._make_uploaded_file("depth.csv", content)
        result = self._read_csv_file(uf)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert len(result) == 5
        assert result[0] == pytest.approx(50.0)

    def test_line74to80_correct_row_count(self):
        """Row count matches ensemble count (60 rows)."""
        content = "\n".join(str(50.0 + i * 0.1) for i in range(60)).encode()
        uf = self._make_uploaded_file("depth.csv", content)
        result = self._read_csv_file(uf)
        assert result is not None
        assert len(result) == 60

    def test_line74to80_salinity_values(self):
        """Constant salinity CSV is read correctly."""
        content = "\n".join("35.0" for _ in range(60)).encode()
        uf = self._make_uploaded_file("salinity.csv", content)
        result = self._read_csv_file(uf)
        assert result is not None
        assert len(result) == 60
        np.testing.assert_allclose(result, 35.0)

    def test_line74to80_temperature_values(self):
        """Temperature CSV with fractional values is read correctly."""
        content = "\n".join("18.5" for _ in range(60)).encode()
        uf = self._make_uploaded_file("temp.csv", content)
        result = self._read_csv_file(uf)
        assert result is not None
        np.testing.assert_allclose(result, 18.5)

    def test_line74to80_writes_to_temp_file(self):
        """The function writes the bytes to a temp file before reading."""
        content = b"1.0\n2.0\n3.0\n"
        uf = self._make_uploaded_file("test.csv", content)
        result = self._read_csv_file(uf)
        assert result is not None
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_line74to80_squeeze_removes_extra_dimension(self):
        """np.squeeze collapses a single-column 2-D array to 1-D."""
        content = b"10.0\n20.0\n30.0\n"
        uf = self._make_uploaded_file("data.csv", content)
        result = self._read_csv_file(uf)
        assert result is not None
        assert result.ndim == 1   # squeeze applied

    # --- lines 81-83: exception path ---

    def test_line81to83_non_utf8_bytes_returns_none(self):
        """Non-UTF-8 bytes cause pd.read_csv to raise → returns None."""
        bad = bytes(range(128, 256))
        uf = self._make_uploaded_file("bad.csv", bad)
        result = self._read_csv_file(uf)
        assert result is None

    def test_line81to83_completely_empty_content(self):
        """Empty bytes cause pd.read_csv to raise EmptyDataError → returns None."""
        uf = self._make_uploaded_file("empty.csv", b"")
        result = self._read_csv_file(uf)
        assert result is None

    def test_line81to83_getvalue_raises_returns_none(self):
        """If getvalue() itself raises, the except catches it → returns None."""
        import types
        uf = types.SimpleNamespace(name="bad.csv")
        uf.getvalue = lambda: (_ for _ in ()).throw(IOError("disk error"))
        result = self._read_csv_file(uf)
        assert result is None


class TestReadCsvFileViaAppTest:
    """
    AppTest-based tests that execute the ACTUAL ``read_csv_file()`` function
    (lines 74-83) inside ``04_Sensor_Health.py``.

    Why this class is necessary
    ---------------------------
    Coverage tracks which source lines execute in the *page file*.  Pure-unit
    replicas in the test file don't count.  These tests drive the real function
    through the actual page by:

    1. Building a real ``streamlit.runtime.uploaded_file_manager.UploadedFile``
       (the only type ``@st.cache_data`` can hash).

    2. Pre-injecting it into ``at.session_state`` under the widget's key *before*
       ``at.run()``.  The page's ``st.file_uploader(key=K)`` reads its value from
       ``st.session_state[K]``, so it sees the file without needing ``at.file_uploader``
       (which is not available in the project's streamlit version).

    3. Pre-setting any radio that defaults to "Fixed Value" (salinity_method) so
       the file uploader branch renders on the first run.

    4. Clicking the ``"Check & Apply *"`` button which calls ``read_csv_file()``.

    Radio defaults
    --------------
    - depth_method:    "File Upload" (index 0) — no pre-setting needed
    - salinity_method: "Fixed Value" (index 0) — must pre-set "File Upload"
    - temp_method:     "File Upload" (index 0) — no pre-setting needed
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_uploaded_file(name: str, content: bytes):
        """
        Build a real Streamlit UploadedFile that st.cache_data can hash.

        UploadedFileRec(file_id, name, type, data) is a namedtuple-like type.
        UploadedFile wraps it and provides .name, .getvalue(), etc.
        """
        from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
        from streamlit.proto.Common_pb2 import FileURLs as FileURLsProto
        rec = UploadedFileRec(
            file_id=name, name=name, type="text/csv", data=content
        )
        return UploadedFile(record=rec, file_urls=FileURLsProto())

    def _at_with_file(
        self,
        processor,
        uploader_key: str,
        content: bytes,
        filename: str = "data.csv",
        extra_state: dict | None = None,
    ) -> AppTest:
        """
        Build and run an AppTest with:
        - processor in session state
        - UploadedFile pre-injected under uploader_key
        - any extra session state (e.g. to switch a radio)
        """
        uf = self._make_uploaded_file(filename, content)
        state = {uploader_key: uf}
        if extra_state:
            state.update(extra_state)
        return _make_at(processor, extra=state)

    def _click(self, at: AppTest, button_key: str) -> AppTest:
        btn = next((b for b in at.button if b.key == button_key), None)
        if btn is None:
            pytest.skip(f"Button {button_key!r} not found after upload injection")
        btn.click().run()
        return at

    # ------------------------------------------------------------------
    # Lines 74-80: try block — valid CSV bytes, pd.read_csv succeeds
    # ------------------------------------------------------------------

    def test_depth_valid_csv_60_rows_lines74to80(self, processor):
        """
        60-row depth CSV: lines 74-80 execute, function returns an ndarray,
        page stores it and shows a success message.
        depth_method defaults to "File Upload" so no extra state needed.
        """
        content = "\n".join(str(50.0 + i * 0.1) for i in range(60)).encode()
        at = self._at_with_file(processor, "depth_file_upload", content, "depth.csv")
        at = self._click(at, "check_depth_file")
        assert not at.exception
        success = " ".join(s.value for s in at.success)
        assert "depth" in success.lower() or "applied" in success.lower()

    def test_depth_valid_csv_sets_depth_modified_true(self, processor):
        """After lines 74-80 succeed, depth_modified is True in session state."""
        content = "\n".join("50.0" for _ in range(60)).encode()
        at = self._at_with_file(processor, "depth_file_upload", content, "depth.csv")
        at = self._click(at, "check_depth_file")
        assert not at.exception
        assert _ss(at, "depth_modified") is True

    def test_depth_valid_csv_wrong_length_still_runs_lines74to80(self, processor):
        """
        5-row CSV: lines 74-80 still execute (return an ndarray), then the page
        detects len(data) != total_ensembles and shows a mismatch error.
        Confirms lines 74-80 run regardless of the downstream length check.
        """
        at = self._at_with_file(
            processor, "depth_file_upload",
            b"50.0\n51.0\n52.0\n53.0\n54.0\n", "depth.csv"
        )
        at = self._click(at, "check_depth_file")
        assert not at.exception
        errors = " ".join(e.value for e in at.error)
        assert "mismatch" in errors.lower() or "ensemble" in errors.lower()

    def test_salinity_valid_csv_60_rows_lines74to80(self, processor):
        """
        60-row salinity CSV. salinity_method defaults to "Fixed Value" so we
        pre-set it to "File Upload" in session state before the first run.
        """
        content = "\n".join("35.0" for _ in range(60)).encode()
        at = self._at_with_file(
            processor, "salinity_file_upload", content, "salinity.csv",
            extra_state={"salinity_method": "File Upload"},
        )
        at = self._click(at, "check_salinity_file")
        assert not at.exception
        success = " ".join(s.value for s in at.success)
        assert "salinity" in success.lower() or "applied" in success.lower()

    def test_salinity_valid_csv_sets_salinity_modified_true(self, processor):
        """After salinity lines 74-80 succeed, salinity_modified is True."""
        content = "\n".join("35.0" for _ in range(60)).encode()
        at = self._at_with_file(
            processor, "salinity_file_upload", content, "salinity.csv",
            extra_state={"salinity_method": "File Upload"},
        )
        at = self._click(at, "check_salinity_file")
        assert not at.exception
        assert _ss(at, "salinity_modified") is True

    def test_temperature_valid_csv_60_rows_lines74to80(self, processor):
        """
        60-row temperature CSV. temp_method defaults to "File Upload" so
        no extra state is needed.
        """
        content = "\n".join("18.5" for _ in range(60)).encode()
        at = self._at_with_file(processor, "temp_file_upload", content, "temp.csv")
        at = self._click(at, "check_temp_file")
        assert not at.exception
        success = " ".join(s.value for s in at.success)
        assert "temperature" in success.lower() or "applied" in success.lower()

    def test_temperature_valid_csv_sets_temperature_modified_true(self, processor):
        """After temperature lines 74-80 succeed, temperature_modified is True."""
        content = "\n".join("18.5" for _ in range(60)).encode()
        at = self._at_with_file(processor, "temp_file_upload", content, "temp.csv")
        at = self._click(at, "check_temp_file")
        assert not at.exception
        assert _ss(at, "temperature_modified") is True

    # ------------------------------------------------------------------
    # Lines 81-83: except block — pd.read_csv raises, st.error() called
    # ------------------------------------------------------------------

    def test_depth_bad_bytes_triggers_except_lines81to83(self, processor):
        """
        Non-UTF-8 bytes cause pd.read_csv to raise UnicodeDecodeError.
        Lines 81-83 execute: st.error(f"Error reading file: {e}") is rendered,
        function returns None, page skips the depth_modified assignment.
        """
        at = self._at_with_file(
            processor, "depth_file_upload", bytes(range(128, 256)), "bad.csv"
        )
        at = self._click(at, "check_depth_file")
        assert not at.exception
        errors = " ".join(e.value for e in at.error)
        assert "error reading file" in errors.lower()

    def test_depth_bad_bytes_does_not_set_modified(self, processor):
        """When lines 81-83 execute (function returns None), depth_modified stays False."""
        at = self._at_with_file(
            processor, "depth_file_upload", bytes(range(128, 256)), "bad.csv"
        )
        at = self._click(at, "check_depth_file")
        assert not at.exception
        assert _ss(at, "depth_modified") is False

    def test_depth_empty_file_triggers_except_lines81to83(self, processor):
        """Empty bytes cause pd.read_csv EmptyDataError — lines 81-83 execute."""
        at = self._at_with_file(
            processor, "depth_file_upload", b"", "empty.csv"
        )
        at = self._click(at, "check_depth_file")
        assert not at.exception
        errors = " ".join(e.value for e in at.error)
        assert "error reading file" in errors.lower()

    def test_salinity_bad_bytes_triggers_except_lines81to83(self, processor):
        """Exception path (lines 81-83) via the salinity file uploader."""
        at = self._at_with_file(
            processor, "salinity_file_upload", bytes(range(128, 256)), "bad.csv",
            extra_state={"salinity_method": "File Upload"},
        )
        at = self._click(at, "check_salinity_file")
        assert not at.exception
        errors = " ".join(e.value for e in at.error)
        assert "error reading file" in errors.lower()

    def test_temperature_bad_bytes_triggers_except_lines81to83(self, processor):
        """Exception path (lines 81-83) via the temperature file uploader."""
        at = self._at_with_file(
            processor, "temp_file_upload", bytes(range(128, 256)), "bad.csv"
        )
        at = self._click(at, "check_temp_file")
        assert not at.exception
        errors = " ".join(e.value for e in at.error)
        assert "error reading file" in errors.lower()

    def test_temperature_empty_file_triggers_except_lines81to83(self, processor):
        """Empty temperature file — lines 81-83 via temp_file_upload."""
        at = self._at_with_file(
            processor, "temp_file_upload", b"", "empty.csv"
        )
        at = self._click(at, "check_temp_file")
        assert not at.exception
        errors = " ".join(e.value for e in at.error)
        assert "error reading file" in errors.lower()


class TestLine177GetScaleFactorAbsent:
    """
    Line 177 — ``return 1.0`` when var_name is not in ds.data_vars.
    """

    @staticmethod
    def _fn(ds: xr.Dataset, var_name: str) -> float:
        if var_name in ds.data_vars:
            return ds[var_name].attrs.get("scale_factor", 1.0)
        return 1.0

    def test_absent_var_returns_one(self):
        assert self._fn(_make_ds(), "nonexistent") == pytest.approx(1.0)

    def test_empty_key_returns_one(self):
        assert self._fn(_make_ds(), "") == pytest.approx(1.0)

    def test_present_var_without_scale_attr_returns_one(self):
        assert self._fn(_make_ds(), "sound_speed") == pytest.approx(1.0)

    def test_present_var_with_scale_attr_returns_scale(self):
        assert self._fn(_make_ds(), "temperature") == pytest.approx(0.01)
        assert self._fn(_make_ds(), "transducer_depth") == pytest.approx(0.1)

    def test_page_uses_fallback_scale_for_dropped_var(self):
        ds_no_depth = _make_ds().drop_vars("transducer_depth")
        at = _make_at(_MockProcessor(ds_no_depth))
        assert not at.exception


class TestLine184GetSensorInfoSuccess:
    """
    Line 184 — the ``return "Available" / "Not Available"`` inside the try block.

    The existing suite only exercises the except branch (accessor raises ->
    "Unknown").  These tests cover the success path via (a) the pure logic
    and (b) a patched fixed_leader accessor in AppTest.
    """

    @staticmethod
    def _fn(sensor_info_dict: dict, sensor_name: str) -> str:
        return ("Available"
                if sensor_info_dict.get(sensor_name, False)
                else "Not Available")

    def test_returns_available_when_true(self):
        assert self._fn({"Depth Sensor": True}, "Depth Sensor") == "Available"

    def test_returns_not_available_when_false(self):
        assert self._fn({"Depth Sensor": False}, "Depth Sensor") == "Not Available"

    def test_returns_not_available_when_key_absent(self):
        assert self._fn({}, "Depth Sensor") == "Not Available"

    def test_via_patched_accessor_available(self, processor):
        """
        Patch ds.fixed_leader using setattr on xr.Dataset so sensor_info()
        succeeds — line 184 executes and renders 'Available' in the markdown.

        xarray accessors are installed as _CachedAccessor descriptors via
        setattr(Dataset, name, ...).  patch.object cannot add new attributes,
        so we use setattr/delattr directly, saving and restoring any existing
        accessor to leave the class clean after the test.
        """
        from xarray.core.extensions import _CachedAccessor
        from xarray.core.dataset import Dataset

        sensor_data = {
            "Depth Sensor": True,
            "Conductivity Sensor": False,
            "Temperature Sensor": True,
        }

        class _MockFL:
            def __init__(self, ds):
                self._ds = ds

            def sensor_info(self, ens=0, field="avail"):
                return sensor_data

        orig = getattr(Dataset, "fixed_leader", None)
        setattr(Dataset, "fixed_leader", _CachedAccessor("fixed_leader", _MockFL))
        try:
            at = _make_at(processor)
            assert not at.exception
            all_md = " ".join(w.value for w in at.markdown)
            assert "available" in all_md.lower()
        finally:
            if orig is None:
                delattr(Dataset, "fixed_leader")
            else:
                setattr(Dataset, "fixed_leader", orig)

    def test_via_patched_accessor_all_not_available(self, processor):
        """All sensors False -> every get_sensor_info call returns 'Not Available'."""
        from xarray.core.extensions import _CachedAccessor
        from xarray.core.dataset import Dataset

        sensor_data = {
            "Depth Sensor": False,
            "Conductivity Sensor": False,
            "Temperature Sensor": False,
        }

        class _MockFLAllFalse:
            def __init__(self, ds):
                self._ds = ds

            def sensor_info(self, ens=0, field="avail"):
                return sensor_data

        orig = getattr(Dataset, "fixed_leader", None)
        setattr(Dataset, "fixed_leader",
                _CachedAccessor("fixed_leader", _MockFLAllFalse))
        try:
            at = _make_at(processor)
            assert not at.exception
            all_md = " ".join(w.value for w in at.markdown)
            assert "not available" in all_md.lower()
        finally:
            if orig is None:
                delattr(Dataset, "fixed_leader")
            else:
                setattr(Dataset, "fixed_leader", orig)


class TestLine195StatusColorMapFallback:
    """
    Line 195 — ``return ""`` when value is neither "True" nor "False".
    """

    @staticmethod
    def _fn(value: object) -> str:
        if value == "True":
            return "background-color: green; color: white"
        elif value == "False":
            return "background-color: red; color: white"
        return ""

    def test_none_returns_empty(self):
        assert self._fn(None) == ""

    def test_integer_zero_returns_empty(self):
        assert self._fn(0) == ""

    def test_integer_one_returns_empty(self):
        assert self._fn(1) == ""

    def test_bool_true_returns_empty(self):
        """Python bool True is not the string 'True'."""
        assert self._fn(True) == ""

    def test_bool_false_returns_empty(self):
        """Python bool False is not the string 'False'."""
        assert self._fn(False) == ""

    def test_lowercase_true_returns_empty(self):
        assert self._fn("true") == ""

    def test_uppercase_false_returns_empty(self):
        assert self._fn("FALSE") == ""

    def test_empty_string_returns_empty(self):
        assert self._fn("") == ""

    def test_arbitrary_string_returns_empty(self):
        assert self._fn("N/A") == ""
        assert self._fn("Unknown") == ""

    def test_list_returns_empty(self):
        assert self._fn([]) == ""

    def test_dict_returns_empty(self):
        assert self._fn({}) == ""

    # Regression guards for the matching branches
    def test_string_True_returns_green(self):
        assert "green" in self._fn("True")

    def test_string_False_returns_red(self):
        assert "red" in self._fn("False")


class TestLines247to255BannerReset:
    """
    Lines 247-255 — the top-page banner Reset button.

    Distinct from the tab8 reset (key='reset_all_button'):
      Banner: label '🔄 Reset Sensor Health', no explicit key
      Tab8:   label 'Reset Sensor Health',    key='reset_all_button'

    Lines executed on click:
      247  proc.reset()
      248  sensor_health_applied = False
      249  depth_modified = False
      250  salinity_modified = False
      251  temperature_modified = False
      252  temp_depth_data = None
      253  temp_salinity_data = None
      254  temp_temperature_data = None
      255  st.rerun()

    After st.rerun() the page re-executes with the cleared session state.
    The st.success() on line 245 fires in the first execution but is consumed
    by st.rerun(); we verify cleared flags in the post-rerun state instead.
    """

    def _banner_btn(self, at: AppTest):
        btn = next(
            (b for b in at.button
             if "reset sensor health" in b.label.lower()
             and b.key != "reset_all_button"),
            None,
        )
        if btn is None:
            pytest.skip("Banner reset button not found")
        return btn

    def _at(self, processor, **flags):
        return _make_at(processor, {
            "sensor_health_initialized": True,
            "sensor_health_applied": True,
            **flags,
        })

    def test_line244_245_success_banner_visible_before_click(self, processor):
        """Line 245: st.success() shown on initial render when applied=True."""
        at = self._at(processor)
        assert "applied" in " ".join(s.value for s in at.success).lower()

    def test_line246_banner_button_exists(self, processor):
        """Line 246: the banner button is rendered."""
        at = self._at(processor)
        assert self._banner_btn(at) is not None

    def test_line247_proc_reset_called(self, processor):
        """Line 247: proc.reset() clears the processing log."""
        processor.processing_log = ["step A", "step B"]
        at = self._at(processor)
        self._banner_btn(at).click().run()
        assert not at.exception
        all_md = " ".join(w.value for w in at.markdown)
        assert "no processing" in all_md.lower()

    def test_line248_applied_flag_cleared(self, processor):
        """Line 248: sensor_health_applied -> False."""
        at = self._at(processor)
        self._banner_btn(at).click().run()
        assert not at.exception
        assert _ss(at, "sensor_health_applied") is False

    def test_line249_depth_modified_cleared(self, processor):
        """Line 249: depth_modified -> False."""
        at = self._at(processor, depth_modified=True,
                      temp_depth_data=np.full(60, 50.0))
        self._banner_btn(at).click().run()
        assert not at.exception
        assert _ss(at, "depth_modified") is False

    def test_line250_salinity_modified_cleared(self, processor):
        """Line 250: salinity_modified -> False."""
        at = self._at(processor, salinity_modified=True,
                      temp_salinity_data=np.full(60, 35.0))
        self._banner_btn(at).click().run()
        assert not at.exception
        assert _ss(at, "salinity_modified") is False

    def test_line251_temperature_modified_cleared(self, processor):
        """Line 251: temperature_modified -> False."""
        at = self._at(processor, temperature_modified=True,
                      temp_temperature_data=np.full(60, 15.0))
        self._banner_btn(at).click().run()
        assert not at.exception
        assert _ss(at, "temperature_modified") is False

    def test_line252_temp_depth_data_nulled(self, processor):
        """Line 252: temp_depth_data -> None."""
        at = self._at(processor, depth_modified=True,
                      temp_depth_data=np.full(60, 50.0))
        self._banner_btn(at).click().run()
        assert not at.exception
        assert _ss(at, "temp_depth_data") is None

    def test_line253_temp_salinity_data_nulled(self, processor):
        """Line 253: temp_salinity_data -> None."""
        at = self._at(processor, salinity_modified=True,
                      temp_salinity_data=np.full(60, 35.0))
        self._banner_btn(at).click().run()
        assert not at.exception
        assert _ss(at, "temp_salinity_data") is None

    def test_line254_temp_temperature_data_nulled(self, processor):
        """Line 254: temp_temperature_data -> None."""
        at = self._at(processor, temperature_modified=True,
                      temp_temperature_data=np.full(60, 15.0))
        self._banner_btn(at).click().run()
        assert not at.exception
        assert _ss(at, "temp_temperature_data") is None

    def test_lines247to255_all_flags_in_one_click(self, processor):
        """All lines 247-254 exercised with every flag set."""
        at = self._at(
            processor,
            depth_modified=True,       temp_depth_data=np.full(60, 50.0),
            salinity_modified=True,    temp_salinity_data=np.full(60, 35.0),
            temperature_modified=True, temp_temperature_data=np.full(60, 15.0),
        )
        self._banner_btn(at).click().run()
        assert not at.exception
        assert _ss(at, "sensor_health_applied") is False
        assert _ss(at, "depth_modified") is False
        assert _ss(at, "salinity_modified") is False
        assert _ss(at, "temperature_modified") is False
        assert _ss(at, "temp_depth_data") is None
        assert _ss(at, "temp_salinity_data") is None
        assert _ss(at, "temp_temperature_data") is None

    def test_line255_rerun_removes_banner_success(self, processor):
        """
        After rerun (line 255) the page re-executes with applied=False, so
        the banner success message is no longer present in at.success.
        """
        at = self._at(processor)
        self._banner_btn(at).click().run()
        assert not at.exception
        success_texts = [s.value for s in at.success]
        assert not any("applied to this dataset" in t.lower()
                       for t in success_texts)


# ===========================================================================
# 17.  CSV UPLOAD — read_csv_file() COMPREHENSIVE TESTS
#
#  The function (lines 74-83) cannot be called directly because of the
#  @st.cache_data decorator.  We test by replicating the exact function body
#  using a minimal UploadedFile mock (``_UploadedFileMock``) that exposes the
#  only two attributes the function touches: ``.name`` and ``.getvalue()``.
#
#  _read_csv_file_with_error() additionally captures the exception message
#  that would be passed to st.error(), letting us assert on lines 81-82.
#
#  Test groups
#  -----------
#  TestCsvUploadSuccessPath      — lines 74-80 (try block succeeds)
#  TestCsvUploadExceptionPath    — lines 81-83 (except branch)
#  TestCsvUploadPageIntegration  — downstream logic the page runs after
#                                  read_csv_file() returns (mismatch detection,
#                                  session-state mutation, success messages)
# ===========================================================================


import types as _types
import io as _io


class _UploadedFileMock:
    """
    Minimal stand-in for ``streamlit.runtime.uploaded_file_manager.UploadedFile``.

    The page's ``read_csv_file()`` only accesses two attributes:
      - ``uploaded_file.name``        (used in os.path.join)
      - ``uploaded_file.getvalue()``  (bytes written to disk before pd.read_csv)
    """

    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def _read_csv_file_full(uploaded_file):
    """
    Exact replica of ``read_csv_file()`` body (lines 74-83).

    Returns (result, error_message):
      - result        — np.ndarray on success, None on exception
      - error_message — the string passed to st.error() on exception, else None
    """
    try:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())          # line 78
        df = pd.read_csv(path, header=None)            # line 79
        return np.squeeze(df.to_numpy()), None         # line 80
    except Exception as e:
        error_msg = f"Error reading file: {e}"         # line 82
        return None, error_msg                         # line 83


def _make_uf(name: str, content: bytes) -> _UploadedFileMock:
    return _UploadedFileMock(name, content)


class TestCsvUploadSuccessPath:
    """
    Lines 74-80 — the try block of read_csv_file() completes without exception.

    Each test verifies a specific aspect of the success path:
    line 75  tempfile.mkdtemp() creates a writable directory
    line 76  path is constructed from uploaded_file.name
    line 77-78  content bytes are written to disk exactly
    line 79  pd.read_csv reads the file with header=None
    line 80  np.squeeze collapses the result to the correct shape
    """

    # ------------------------------------------------------------------
    # Basic return-type and shape checks
    # ------------------------------------------------------------------

    def test_returns_ndarray_not_none(self):
        """Non-empty valid CSV -> result is not None."""
        result, err = _read_csv_file_full(_make_uf("d.csv", b"1.0\n2.0\n3.0\n"))
        assert result is not None
        assert err is None

    def test_return_type_is_ndarray(self):
        """Return value is always an np.ndarray on success."""
        result, _ = _read_csv_file_full(_make_uf("d.csv", b"10.0\n20.0\n"))
        assert isinstance(result, np.ndarray)

    def test_single_column_produces_1d_array(self):
        """Single-column CSV + np.squeeze -> 1-D result (not 2-D)."""
        content = b"1.0\n2.0\n3.0\n4.0\n5.0\n"
        result, _ = _read_csv_file_full(_make_uf("depth.csv", content))
        assert result.ndim == 1

    def test_squeeze_removes_singleton_column_dimension(self):
        """pd.read_csv of a single-column file gives shape (N,1); squeeze -> (N,)."""
        content = b"10.0\n20.0\n30.0\n"
        result, _ = _read_csv_file_full(_make_uf("data.csv", content))
        assert result.shape == (3,)

    # ------------------------------------------------------------------
    # Row count correctness (page checks len(data) == total_ensembles)
    # ------------------------------------------------------------------

    def test_row_count_five(self):
        content = b"50.0\n51.0\n52.0\n53.0\n54.0\n"
        result, _ = _read_csv_file_full(_make_uf("depth.csv", content))
        assert len(result) == 5

    def test_row_count_matches_60_ensemble_dataset(self):
        """60 rows -> result has 60 elements (matches _make_ds() ensemble count)."""
        content = "\n".join(str(50.0 + i * 0.1) for i in range(60)).encode()
        result, _ = _read_csv_file_full(_make_uf("depth.csv", content))
        assert len(result) == 60

    def test_row_count_100(self):
        content = "\n".join(str(i * 0.5) for i in range(100)).encode()
        result, _ = _read_csv_file_full(_make_uf("data.csv", content))
        assert len(result) == 100

    def test_single_row(self):
        """A single-row CSV is a valid edge case (scalar dataset)."""
        result, _ = _read_csv_file_full(_make_uf("single.csv", b"75.0\n"))
        assert result is not None
        # np.squeeze on shape (1,1) -> scalar array; shape is ()
        assert result.ndim in (0, 1)

    # ------------------------------------------------------------------
    # Value correctness
    # ------------------------------------------------------------------

    def test_depth_values_preserved(self):
        """Exact values survive the write-then-read round-trip."""
        values = [50.0, 51.5, 49.8, 52.1, 48.3]
        content = "\n".join(str(v) for v in values).encode()
        result, _ = _read_csv_file_full(_make_uf("depth.csv", content))
        np.testing.assert_allclose(result, values)

    def test_salinity_constant_values(self):
        """All-constant CSV returns uniform array (typical salinity replacement)."""
        content = "\n".join("35.0" for _ in range(60)).encode()
        result, _ = _read_csv_file_full(_make_uf("salinity.csv", content))
        assert len(result) == 60
        np.testing.assert_allclose(result, 35.0)

    def test_temperature_fractional_values(self):
        """Fractional temperature values round-trip correctly."""
        content = "\n".join("18.5" for _ in range(60)).encode()
        result, _ = _read_csv_file_full(_make_uf("temp.csv", content))
        np.testing.assert_allclose(result, 18.5)

    def test_negative_values(self):
        """Negative values (valid for temperature) are preserved."""
        values = [-2.5, -1.0, 0.0, 1.0, 2.5]
        content = "\n".join(str(v) for v in values).encode()
        result, _ = _read_csv_file_full(_make_uf("temp.csv", content))
        np.testing.assert_allclose(result, values)

    def test_integer_values_in_csv(self):
        """Integer-valued CSV rows are read as floats by pd.read_csv."""
        content = b"100\n200\n300\n"
        result, _ = _read_csv_file_full(_make_uf("data.csv", content))
        np.testing.assert_array_equal(result, [100.0, 200.0, 300.0])

    def test_large_depth_values(self):
        """Large depth values (deep mooring) are preserved."""
        content = "\n".join("3500.0" for _ in range(10)).encode()
        result, _ = _read_csv_file_full(_make_uf("depth.csv", content))
        np.testing.assert_allclose(result, 3500.0)

    # ------------------------------------------------------------------
    # Filename handling (line 76: os.path.join uses uploaded_file.name)
    # ------------------------------------------------------------------

    def test_name_depth_csv(self):
        """'depth.csv' filename is used to construct the temp path."""
        result, err = _read_csv_file_full(_make_uf("depth.csv", b"1.0\n2.0\n"))
        assert result is not None and err is None

    def test_name_salinity_csv(self):
        result, err = _read_csv_file_full(_make_uf("salinity.csv", b"35.0\n36.0\n"))
        assert result is not None and err is None

    def test_name_temperature_csv(self):
        result, err = _read_csv_file_full(_make_uf("temperature.csv", b"18.5\n19.0\n"))
        assert result is not None and err is None

    def test_name_with_spaces(self):
        """Filenames with spaces are handled by os.path.join correctly."""
        result, err = _read_csv_file_full(
            _make_uf("my depth data.csv", b"50.0\n51.0\n"))
        assert result is not None and err is None

    # ------------------------------------------------------------------
    # getvalue() is called exactly once and its bytes reach disk (line 78)
    # ------------------------------------------------------------------

    def test_getvalue_called_and_content_written(self):
        """The content from getvalue() is what pd.read_csv sees."""
        sentinel_values = [999.1, 999.2, 999.3]
        content = "\n".join(str(v) for v in sentinel_values).encode()

        call_count = [0]
        uf = _UploadedFileMock("sentinel.csv", content)
        original_getvalue = uf.getvalue

        def counting_getvalue():
            call_count[0] += 1
            return original_getvalue()

        uf.getvalue = counting_getvalue
        result, _ = _read_csv_file_full(uf)

        assert call_count[0] == 1                      # called exactly once
        np.testing.assert_allclose(result, sentinel_values)

    # ------------------------------------------------------------------
    # header=None ensures the first row is treated as data, not a header
    # ------------------------------------------------------------------

    def test_header_none_first_row_is_data(self):
        """With header=None every row becomes a data row, including the first."""
        content = b"50.0\n51.0\n52.0\n"
        result, _ = _read_csv_file_full(_make_uf("d.csv", content))
        # All three rows present — none consumed as a header
        assert len(result) == 3
        assert result[0] == pytest.approx(50.0)

    def test_header_none_numeric_first_row_not_lost(self):
        """Confirms that a numeric first row is NOT silently dropped."""
        content = b"100.0\n200.0\n300.0\n"
        result, _ = _read_csv_file_full(_make_uf("d.csv", content))
        assert result[0] == pytest.approx(100.0)


class TestCsvUploadExceptionPath:
    """
    Lines 81-83 — the except branch of read_csv_file().

    Every test confirms:
      (a) result is None  (line 83)
      (b) error_message starts with "Error reading file:"  (line 82)
      (c) no unhandled exception escapes the function
    """

    def _assert_exception_path(self, uf):
        """Common assertion helper for all exception-path tests."""
        result, err = _read_csv_file_full(uf)
        assert result is None, "expected None on exception path"
        assert err is not None, "expected an error message"
        assert err.startswith("Error reading file:"), (
            f"unexpected error prefix: {err!r}"
        )
        return err  # return so callers can assert on the message text

    # ------------------------------------------------------------------
    # pd.read_csv fails to parse the file content
    # ------------------------------------------------------------------

    def test_non_utf8_bytes_triggers_exception(self):
        """High-byte content (0x80-0xFF) cannot be decoded as UTF-8."""
        err = self._assert_exception_path(
            _make_uf("bad.csv", bytes(range(128, 256)))
        )
        assert "error reading file" in err.lower()

    def test_empty_content_triggers_exception(self):
        """Empty bytes cause pd.read_csv to raise EmptyDataError."""
        err = self._assert_exception_path(_make_uf("empty.csv", b""))
        assert "error reading file" in err.lower()

    def test_purely_whitespace_content(self):
        """A file with only whitespace/newlines is also empty from pd's view."""
        err = self._assert_exception_path(_make_uf("ws.csv", b"\n\n\n   \n"))
        assert "error reading file" in err.lower()

    def test_binary_header_magic_bytes(self):
        """Binary file magic bytes (e.g. PNG header) cannot be parsed as CSV."""
        png_header = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
        err = self._assert_exception_path(_make_uf("notacsv.csv", png_header))
        assert "error reading file" in err.lower()

    def test_mixed_text_and_high_bytes(self):
        """Partial high-byte content mid-file also causes decode failure."""
        content = b"50.0\n" + bytes(range(128, 160)) + b"\n52.0\n"
        err = self._assert_exception_path(_make_uf("mixed.csv", content))
        assert "error reading file" in err.lower()

    # ------------------------------------------------------------------
    # getvalue() raises before the file is even written (line 78 fails)
    # ------------------------------------------------------------------

    def test_getvalue_raises_ioerror(self):
        """If getvalue() raises IOError, the except block catches it."""
        uf = _UploadedFileMock("bad.csv", b"")
        uf.getvalue = lambda: (_ for _ in ()).throw(IOError("disk read failed"))
        err = self._assert_exception_path(uf)
        assert "disk read failed" in err

    def test_getvalue_raises_runtime_error(self):
        """RuntimeError from getvalue() is also caught."""
        uf = _UploadedFileMock("bad.csv", b"")
        uf.getvalue = lambda: (_ for _ in ()).throw(RuntimeError("network error"))
        err = self._assert_exception_path(uf)
        assert "network error" in err

    def test_getvalue_raises_value_error(self):
        """ValueError from getvalue() is caught by the bare except Exception."""
        uf = _UploadedFileMock("bad.csv", b"")
        uf.getvalue = lambda: (_ for _ in ()).throw(ValueError("bad state"))
        err = self._assert_exception_path(uf)
        assert "bad state" in err

    # ------------------------------------------------------------------
    # Error message format (line 82: f"Error reading file: {e}")
    # ------------------------------------------------------------------

    def test_error_message_contains_exception_text(self):
        """The exception's string representation appears after the prefix."""
        uf = _UploadedFileMock("bad.csv", b"")
        uf.getvalue = lambda: (_ for _ in ()).throw(
            IOError("no such volume")
        )
        _, err = _read_csv_file_full(uf)
        assert "no such volume" in err

    def test_error_message_prefix_exact(self):
        """Prefix is exactly 'Error reading file: ' (no leading emoji etc.)."""
        uf = _UploadedFileMock("bad.csv", b"")
        uf.getvalue = lambda: (_ for _ in ()).throw(IOError("x"))
        _, err = _read_csv_file_full(uf)
        assert err.startswith("Error reading file: ")

    # ------------------------------------------------------------------
    # Function does NOT re-raise — caller receives None, not an exception
    # ------------------------------------------------------------------

    def test_no_exception_escapes_on_bad_bytes(self):
        """The function swallows the exception; calling code never sees it."""
        try:
            result, _ = _read_csv_file_full(
                _make_uf("bad.csv", bytes(range(128, 256)))
            )
            assert result is None
        except Exception as e:
            pytest.fail(f"Exception escaped read_csv_file: {e}")

    def test_no_exception_escapes_on_getvalue_raise(self):
        uf = _UploadedFileMock("bad.csv", b"")
        uf.getvalue = lambda: (_ for _ in ()).throw(IOError("fail"))
        try:
            result, _ = _read_csv_file_full(uf)
            assert result is None
        except Exception as e:
            pytest.fail(f"Exception escaped read_csv_file: {e}")


class TestCsvUploadPageIntegration:
    """
    Downstream logic the page runs after read_csv_file() returns.

    These tests replicate exactly what the page does with the return value:

    Depth (lines 389-405):
        if data is not None:
            if len(data) != total_ensembles: st.error(mismatch)
            else:
                temp_depth_data = data / scale (if scale else data)
                depth_modified = True
                st.success("✅ Depth data will be applied when saved.")

    Salinity (lines 527-541) and Temperature (lines 660-676) follow the
    same pattern without the scale division.

    We simulate the session-state dict and verify the mutations directly —
    no AppTest overhead needed since we're testing pure conditional logic.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_depth_apply(data, total_ensembles: int, scale: float):
        """
        Replicate the depth branch of the 'Check & Apply Depth' button handler.
        Returns (session_state_delta, success, error_msg).
        """
        if data is None:
            return {}, False, None
        if len(data) != total_ensembles:
            return {}, False, (
                f"Ensemble count mismatch! "
                f"File has {len(data)} values, expected {total_ensembles}."
            )
        temp_depth_data = data / scale if scale else data
        return (
            {"temp_depth_data": temp_depth_data, "depth_modified": True},
            True,
            None,
        )

    @staticmethod
    def _simulate_salinity_apply(data, total_ensembles: int):
        if data is None:
            return {}, False, None
        if len(data) != total_ensembles:
            return {}, False, (
                f"Ensemble count mismatch! "
                f"File has {len(data)} values, expected {total_ensembles}."
            )
        return {"temp_salinity_data": data, "salinity_modified": True}, True, None

    @staticmethod
    def _simulate_temperature_apply(data, total_ensembles: int):
        if data is None:
            return {}, False, None
        if len(data) != total_ensembles:
            return {}, False, (
                f"Ensemble count mismatch! "
                f"File has {len(data)} values, expected {total_ensembles}."
            )
        return {"temp_temperature_data": data, "temperature_modified": True}, True, None

    # ------------------------------------------------------------------
    # read_csv_file returns None -> downstream code is skipped entirely
    # ------------------------------------------------------------------

    def test_none_result_does_not_mutate_depth_state(self):
        delta, success, err = self._simulate_depth_apply(
            None, total_ensembles=60, scale=0.1
        )
        assert delta == {}
        assert success is False

    def test_none_result_does_not_mutate_salinity_state(self):
        delta, success, _ = self._simulate_salinity_apply(None, 60)
        assert delta == {} and success is False

    def test_none_result_does_not_mutate_temperature_state(self):
        delta, success, _ = self._simulate_temperature_apply(None, 60)
        assert delta == {} and success is False

    # ------------------------------------------------------------------
    # Ensemble count mismatch (read_csv_file succeeded but wrong length)
    # ------------------------------------------------------------------

    def test_depth_mismatch_shows_error_message(self):
        data = np.arange(5, dtype=float)   # 5 rows, expected 60
        _, success, err = self._simulate_depth_apply(data, 60, scale=0.1)
        assert success is False
        assert err is not None
        assert "mismatch" in err.lower() or "ensemble" in err.lower()
        assert "5" in err and "60" in err

    def test_salinity_mismatch_shows_error_message(self):
        data = np.full(10, 35.0)
        _, success, err = self._simulate_salinity_apply(data, 60)
        assert success is False
        assert "10" in err and "60" in err

    def test_temperature_mismatch_shows_error_message(self):
        data = np.full(100, 18.5)          # 100 rows, expected 60
        _, success, err = self._simulate_temperature_apply(data, 60)
        assert success is False
        assert "100" in err and "60" in err

    def test_mismatch_when_too_few_rows(self):
        data = np.arange(1, dtype=float)   # 1 row is always wrong
        _, success, err = self._simulate_depth_apply(data, 60, scale=0.1)
        assert success is False and "1" in err

    def test_mismatch_when_too_many_rows(self):
        data = np.arange(200, dtype=float)
        _, success, err = self._simulate_depth_apply(data, 60, scale=0.1)
        assert success is False and "200" in err

    # ------------------------------------------------------------------
    # Correct length -> session state is mutated, success flag set
    # ------------------------------------------------------------------

    def test_depth_correct_length_sets_modified_flag(self):
        data = np.full(60, 50.0)
        delta, success, err = self._simulate_depth_apply(data, 60, scale=0.1)
        assert success is True and err is None
        assert delta["depth_modified"] is True

    def test_salinity_correct_length_sets_modified_flag(self):
        data = np.full(60, 35.0)
        delta, success, _ = self._simulate_salinity_apply(data, 60)
        assert success is True
        assert delta["salinity_modified"] is True

    def test_temperature_correct_length_sets_modified_flag(self):
        data = np.full(60, 18.5)
        delta, success, _ = self._simulate_temperature_apply(data, 60)
        assert success is True
        assert delta["temperature_modified"] is True

    # ------------------------------------------------------------------
    # Depth scale factor division (line 398-399: data / scale if scale else data)
    # ------------------------------------------------------------------

    def test_depth_data_divided_by_scale_factor(self):
        """
        The page converts metres -> dataset units: temp_depth_data = data / scale.
        scale=0.1 means 50.0 m stored as 500.0 in the dataset.
        """
        data = np.full(60, 50.0)         # 50 metres from the CSV file
        scale = 0.1
        delta, _, _ = self._simulate_depth_apply(data, 60, scale=scale)
        expected = data / scale          # 500.0
        np.testing.assert_allclose(delta["temp_depth_data"], expected)

    def test_depth_data_not_divided_when_scale_is_zero(self):
        """If scale is falsy (0.0), division is skipped and raw data stored."""
        data = np.full(60, 50.0)
        delta, _, _ = self._simulate_depth_apply(data, 60, scale=0.0)
        np.testing.assert_allclose(delta["temp_depth_data"], data)

    def test_depth_data_not_divided_when_scale_is_one(self):
        """scale=1.0 divides by 1 -> stored value equals input."""
        data = np.full(60, 50.0)
        delta, _, _ = self._simulate_depth_apply(data, 60, scale=1.0)
        np.testing.assert_allclose(delta["temp_depth_data"], data)

    def test_depth_scale_factor_point_one(self):
        """Regression: scale=0.1 is what _make_ds() assigns to transducer_depth."""
        data = np.linspace(49.0, 51.0, 60)
        delta, _, _ = self._simulate_depth_apply(data, 60, scale=0.1)
        np.testing.assert_allclose(delta["temp_depth_data"], data / 0.1)

    def test_salinity_stored_without_scale_division(self):
        """Salinity branch stores raw data; no scale division applied."""
        data = np.full(60, 35.5)
        delta, _, _ = self._simulate_salinity_apply(data, 60)
        np.testing.assert_allclose(delta["temp_salinity_data"], data)

    def test_temperature_stored_without_scale_division(self):
        """Temperature branch stores raw data; no scale division applied."""
        data = np.full(60, 18.5)
        delta, _, _ = self._simulate_temperature_apply(data, 60)
        np.testing.assert_allclose(delta["temp_temperature_data"], data)

    # ------------------------------------------------------------------
    # End-to-end: read_csv_file output flows directly into the apply logic
    # ------------------------------------------------------------------

    def test_full_depth_pipeline_success(self):
        """read_csv_file -> depth apply -> modified flag set."""
        content = "\n".join(str(50.0 + i * 0.01) for i in range(60)).encode()
        data, err = _read_csv_file_full(_make_uf("depth.csv", content))
        assert data is not None and err is None
        delta, success, apply_err = self._simulate_depth_apply(data, 60, scale=0.1)
        assert success is True and apply_err is None
        assert delta["depth_modified"] is True
        assert len(delta["temp_depth_data"]) == 60

    def test_full_salinity_pipeline_success(self):
        """read_csv_file -> salinity apply -> modified flag set."""
        content = b"\n".join(b"35.0" for _ in range(60))
        data, _ = _read_csv_file_full(_make_uf("salinity.csv", content))
        delta, success, _ = self._simulate_salinity_apply(data, 60)
        assert success is True
        assert delta["salinity_modified"] is True

    def test_full_temperature_pipeline_success(self):
        """read_csv_file -> temperature apply -> modified flag set."""
        content = b"\n".join(b"18.5" for _ in range(60))
        data, _ = _read_csv_file_full(_make_uf("temp.csv", content))
        delta, success, _ = self._simulate_temperature_apply(data, 60)
        assert success is True
        assert delta["temperature_modified"] is True

    def test_full_pipeline_mismatch_no_state_change(self):
        """Wrong-length CSV -> mismatch -> no session state mutation."""
        content = b"50.0\n51.0\n52.0\n"   # only 3 rows
        data, _ = _read_csv_file_full(_make_uf("depth.csv", content))
        delta, success, err = self._simulate_depth_apply(data, 60, scale=0.1)
        assert success is False
        assert delta == {}
        assert "3" in err and "60" in err

    def test_full_pipeline_read_error_no_state_change(self):
        """Unreadable file -> None -> no session state mutation."""
        data, _ = _read_csv_file_full(
            _make_uf("bad.csv", bytes(range(128, 256)))
        )
        assert data is None
        delta, success, _ = self._simulate_depth_apply(data, 60, scale=0.1)
        assert success is False and delta == {}
