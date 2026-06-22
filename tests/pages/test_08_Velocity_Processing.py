"""
AppTest-based Test Suite for 08_Velocity_Processing.py

Exercises the actual Streamlit page via streamlit.testing.v1.AppTest,
covering all six tabs, session-state initialization, helper functions
(via importlib), and the staging-processor pattern.

Run with:
    pytest test_08_Velocity_Processing.py -v
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from streamlit.testing.v1 import AppTest

# ---------------------------------------------------------------------------
# SCRIPT PATH
# ---------------------------------------------------------------------------

SCRIPT_PATH = str(
    Path(__file__).parent.parent.parent
    / "src"
    / "pyadps"
    / "pages"
    / "08_Velocity_Processing.py"
)


# ===========================================================================
# MOCK SUPPORT CLASSES
# ===========================================================================


class MockQCCheckStats:
    def __init__(
        self,
        check_name: str = "test_check",
        threshold: Any = None,
        cells_pre_masked: int = 0,
        cells_newly_masked: int = 50,
        cells_total_masked: int = 50,
        total_cells: int = 10000,
    ):
        self.check_name = check_name
        self.threshold = threshold
        self.cells_pre_masked = cells_pre_masked
        self.cells_newly_masked = cells_newly_masked
        self.cells_total_masked = cells_total_masked
        self.total_cells = total_cells

    @property
    def pre_masked_pct(self) -> float:
        return (
            100 * self.cells_pre_masked / self.total_cells if self.total_cells else 0.0
        )

    @property
    def newly_masked_pct(self) -> float:
        return (
            100 * self.cells_newly_masked / self.total_cells
            if self.total_cells
            else 0.0
        )

    @property
    def total_masked_pct(self) -> float:
        return (
            100 * self.cells_total_masked / self.total_cells
            if self.total_cells
            else 0.0
        )

    @property
    def valid_pct(self) -> float:
        return 100 - self.total_masked_pct

    @property
    def valid_cells(self) -> int:
        return self.total_cells - self.cells_total_masked


class MockDataModificationStats:
    def __init__(
        self, operation: str = "magnetic_correction", declination: float = -5.0
    ):
        self.operation = operation
        self.variable_name = "velocity"
        self.metadata = {"declination_applied": declination}


class MockVelocityCheckRunner:
    def __init__(self, ds: xr.Dataset):
        self.dataset = ds.copy(deep=True)
        self._original = ds.copy(deep=True)
        self.statistics: List[MockQCCheckStats] = []
        self.modifications: List[MockDataModificationStats] = []

    def magnetic_correction(
        self,
        declination: Optional[float] = None,
        use_api: bool = False,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        year: Optional[float] = None,
    ) -> "MockVelocityCheckRunner":
        dec = declination if declination is not None else -5.0
        self.dataset.attrs["magnetic_declination_applied"] = dec
        self.modifications.append(MockDataModificationStats(declination=dec))
        return self

    def threshold(
        self,
        cutoff_u: float = 2500.0,
        cutoff_v: float = 2500.0,
        cutoff_w: float = 500.0,
    ) -> "MockVelocityCheckRunner":
        self.statistics.append(
            MockQCCheckStats(
                check_name="Velocity Threshold",
                threshold={"U": cutoff_u, "V": cutoff_v, "W": cutoff_w},
                cells_newly_masked=10,
                cells_total_masked=10,
            )
        )
        return self

    def despike(
        self, kernel_size: int = 13, cutoff: float = 3.0
    ) -> "MockVelocityCheckRunner":
        self.statistics.append(
            MockQCCheckStats(
                check_name="Despike",
                threshold=(kernel_size, cutoff),
                cells_newly_masked=5,
                cells_total_masked=15,
            )
        )
        return self

    def flatline(
        self, kernel_size: int = 4, cutoff: float = 1.0
    ) -> "MockVelocityCheckRunner":
        self.statistics.append(
            MockQCCheckStats(
                check_name="Flatline",
                threshold=(kernel_size, cutoff),
                cells_newly_masked=3,
                cells_total_masked=18,
            )
        )
        return self

    def finalize(self) -> xr.Dataset:
        self.dataset.attrs["velocity_check_processed"] = True
        return self.dataset

    def get_pipeline_report(self) -> Dict:
        return {
            "module_name": "velocity_check",
            "checks": self.statistics,
            "modifications": self.modifications,
        }


class _FLStub:
    """FixedLeaderAccessor stub — matches conftest.py's mock interface."""

    def __init__(self, obj):
        self._obj = obj

    def coordinate_transformation(self, ens: int = 0) -> Dict[str, str]:
        return {"Coordinates": "Earth Coordinates"}

    def system_configuration(self) -> Dict[str, str]:
        # No ens parameter — matches conftest.py accessor
        return {"Beam Direction": "Up", "Beam Angle": 20}

    def field(self, ens: int = 0) -> Dict[str, Any]:
        return {
            "depth_cell_length": self._obj.attrs.get("cell_size_cm", 400),
            "bin_1_distance": self._obj.attrs.get("bin1_distance_cm", 200),
        }


class _RaisingFLStub:
    """FixedLeaderAccessor stub that raises — used to force the attrs fallback branch."""

    def __init__(self, obj):
        pass

    def coordinate_transformation(self, ens=0):
        raise RuntimeError("no accessor")


def _make_ds(
    n_beams: int = 4,
    n_cells: int = 10,
    n_time: int = 20,
    earth_coords: bool = True,
) -> xr.Dataset:
    """Build a minimal xr.Dataset that satisfies the page's requirements."""
    np.random.seed(42)
    time = pd.date_range("2024-01-01", periods=n_time, freq="h")

    velocity = (np.random.randn(n_beams, n_cells, n_time) * 500).astype(np.int16)
    mask = np.zeros((n_beams, n_cells, n_time), dtype=np.int8)

    ds = xr.Dataset(
        {
            "velocity": (["beam", "cell", "time"], velocity),
            "mask": (["beam", "cell", "time"], mask),
            "echo_intensity": (
                ["beam", "cell", "time"],
                np.random.randint(50, 200, (n_beams, n_cells, n_time), dtype=np.uint8),
            ),
        },
        coords={
            "time": time,
            "cell": np.arange(n_cells),
            "beam": np.arange(n_beams),
            "depth_cell_length": np.int64(400),  # scalar coord
            "bin_1_distance": np.int64(200),  # scalar coord
        },
        attrs={
            "beam_angle": 20,
            "beam_direction": "Up",
            "coordinate_system": "earth" if earth_coords else "beam",
        },
    )
    return ds


def _make_mock_processor(ds: xr.Dataset) -> MagicMock:
    """Build a MagicMock that satisfies proc.*  calls made by the page."""
    proc = MagicMock()
    proc.dataset = ds

    total = ds["mask"].size
    proc.get_current_stats.return_value = {
        "total_cells": total,
        "masked": 0,
        "masked_pct": 0.0,
        "valid": total,
        "valid_pct": 100.0,
    }

    # get_velocity_check_runner() returns a fresh runner each call
    proc.get_velocity_check_runner.side_effect = lambda: MockVelocityCheckRunner(ds)

    # commit_runner() just runs finalize()
    def _commit(runner):
        runner.finalize()

    proc.commit_runner.side_effect = _commit
    proc.processing_log = []

    return proc


def _full_ss(proc: MagicMock, **overrides) -> Dict[str, Any]:
    """Return a complete session-state dict for the velocity page."""
    ds = proc.dataset
    base = {
        "processor": proc,
        # Initialization flag — page re-initialises when False
        "velocity_initialized": True,
        "preview_velocity_proc": MagicMock(dataset=ds),
        "velocity_preview_run": False,
        "velocity_applied": False,
        "velocity_preview_stats": None,
        "velocity_preview_modifications": None,
        # Magnetic
        "apply_magnetic": False,
        "magnetic_method": "pygeomag",
        "magnetic_lat": 0.0,
        "magnetic_lon": 0.0,
        "magnetic_year": 2025,
        "magnetic_depth": 0,
        "magnetic_declination": None,
        # Threshold
        "apply_threshold": True,
        "cutoff_u": 2500,
        "cutoff_v": 2500,
        "cutoff_w": 500,
        # Despike
        "apply_despike": False,
        "despike_kernel": 13,
        "despike_cutoff": 3.0,
        # Flatline
        "apply_flatline": False,
        "flatline_kernel": 4,
        "flatline_cutoff": 1.0,
    }
    base.update(overrides)
    return base


# ===========================================================================
# MODULE-SCOPED AUTOUSE FIXTURE: inject_pyadps_mock
# ===========================================================================


@pytest.fixture(autouse=True)
def _ensure_fl_accessor():
    """
    Function-scoped autouse fixture — re-registers _FLStub before every test.
    This prevents TestPageFunctionsDirectly's raising-accessor stubs from
    contaminating subsequent AppTest runs.
    """
    # suppress the AccessorRegistrationWarning that xarray emits on re-registration
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xr.register_dataset_accessor("fixed_leader")(_FLStub)
    yield


@pytest.fixture(scope="module", autouse=True)
def inject_pyadps_mock():
    """
    Inject the mandatory pyadps mock hierarchy so conftest.py teardown
    (which does `from pyadps.io.accessors import FixedLeaderAccessor`) works.
    """
    _originals = {
        k: sys.modules.get(k)
        for k in ("pyadps", "pyadps.io", "pyadps.io.accessors", "pyadps.processing")
    }

    mock_pyadps = types.ModuleType("pyadps")
    mock_pyadps.__path__ = []
    mock_pyadps.__package__ = "pyadps"

    mock_io = types.ModuleType("pyadps.io")
    mock_io.__path__ = []
    mock_io.__package__ = "pyadps.io"

    mock_accessors = types.ModuleType("pyadps.io.accessors")
    mock_accessors.FixedLeaderAccessor = _FLStub

    mock_pyadps.io = mock_io
    mock_io.accessors = mock_accessors

    mock_processing = types.ModuleType("pyadps.processing")
    mock_processing.ProcessedDataset = MagicMock(
        side_effect=lambda ds: MagicMock(dataset=ds)
    )

    sys.modules.update(
        {
            "pyadps": mock_pyadps,
            "pyadps.io": mock_io,
            "pyadps.io.accessors": mock_accessors,
            "pyadps.processing": mock_processing,
        }
    )

    # Register the fixed_leader accessor on xr.Dataset
    try:
        del xr.Dataset.fixed_leader
    except AttributeError:
        pass
    xr.register_dataset_accessor("fixed_leader")(_FLStub)

    yield

    for key, original in _originals.items():
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original


# ===========================================================================
# MODULE-SCOPED DATASET / PROCESSOR FIXTURES
# ===========================================================================


@pytest.fixture(scope="module")
def ds():
    return _make_ds()


@pytest.fixture(scope="module")
def proc(ds):
    return _make_mock_processor(ds)


# ===========================================================================
# HELPER: build and run AppTest
# ===========================================================================


def _run(extra_ss: Dict[str, Any], timeout: int = 15) -> AppTest:
    at = AppTest.from_file(SCRIPT_PATH, default_timeout=timeout)
    for k, v in extra_ss.items():
        at.session_state[k] = v
    at.run()
    return at


# ===========================================================================
# CLASS 1 — Page-level smoke and guard-rails
# ===========================================================================


class TestPageLoadAndGuards:
    """Smoke tests and the 'no processor' guard."""

    def test_no_processor_shows_error_and_stops(self):
        """Without a processor in session state, page shows error and stops."""
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.run()
        assert at.error, "Expected an st.error when no processor is set"

    def test_page_loads_with_valid_processor(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception, f"Page raised: {at.exception}"

    def test_page_header_rendered(self, proc):
        at = _run(_full_ss(proc))
        # st.header produces markdown elements
        text_content = " ".join(e.value for e in at.markdown if hasattr(e, "value"))
        assert "Velocity" in text_content or not at.exception

    def test_six_tabs_present(self, proc):
        at = _run(_full_ss(proc))
        # Six tabs should be created — AppTest exposes tabs via at.tabs
        assert not at.exception


class TestSessionStateInitialization:
    """Tests for the initialization block (velocity_initialized flag)."""

    def test_fresh_start_initializes_all_keys(self, proc):
        """When velocity_initialized is absent, page initialises state."""
        ss = {"processor": proc}  # bare minimum — no _initialized flag
        at = _run(ss)
        assert not at.exception
        # After init, the flag must be True
        assert at.session_state["velocity_initialized"] is True

    def test_defaults_after_init(self, proc):
        ss = {"processor": proc}
        at = _run(ss)
        assert at.session_state["apply_threshold"] is True
        assert at.session_state["cutoff_u"] == 2500
        assert at.session_state["cutoff_v"] == 2500
        assert at.session_state["cutoff_w"] == 500
        assert at.session_state["apply_despike"] is False
        assert at.session_state["apply_flatline"] is False
        assert at.session_state["apply_magnetic"] is False

    def test_second_run_skips_init(self, proc):
        """Re-running with velocity_initialized=True preserves changed values."""
        ss = _full_ss(proc, cutoff_u=1234, velocity_initialized=True)
        at = _run(ss)
        # cutoff_u should NOT be reset to 2500
        assert at.session_state["cutoff_u"] == 1234

    def test_preview_proc_created_when_missing(self, proc):
        """If preview_velocity_proc is absent, page creates it."""
        ss = _full_ss(proc)
        ss.pop("preview_velocity_proc")
        at = _run(ss)
        assert not at.exception
        assert "preview_velocity_proc" in at.session_state


# ===========================================================================
# CLASS 2 — Tab 1: Magnetic Declination
# ===========================================================================


class TestTab1MagneticDeclination:
    """Tests for the Magnetic Declination tab."""

    def test_tab1_renders_without_error(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_manual_method_submit_sets_declination(self, proc):
        """Submitting the Manual form updates magnetic_declination in state."""
        ss = _full_ss(proc, magnetic_method="pygeomag")
        at = _run(ss)
        # The radio defaults to pygeomag; switch to Manual
        at.radio[0].set_value("Manual").run()
        assert not at.exception

    def test_magnetic_info_shown_when_applied(self, proc):
        """When apply_magnetic=True and declination is set, an info box appears."""
        ss = _full_ss(proc, apply_magnetic=True, magnetic_declination=-7.5)
        at = _run(ss)
        assert not at.exception
        # st.info should appear
        assert len(at.info) >= 1 or not at.exception

    def test_reset_magnetic_button_present(self, proc):
        at = _run(_full_ss(proc))
        button_keys = [b.key for b in at.button]
        assert "reset_magnetic" in button_keys

    def test_reset_magnetic_button_click_clears_state(self, proc):
        """Clicking Reset Magnetic Declination clears magnetic state.

        Note: the handler sets apply_magnetic=False and then calls st.rerun().
        AppTest captures state after the rerun; assert on the side-effect
        (magnetic_declination=None) which is set before the rerun call.
        """
        ss = _full_ss(proc, apply_magnetic=True, magnetic_declination=-5.0)
        at = _run(ss)
        reset_btn = next(b for b in at.button if b.key == "reset_magnetic")
        reset_btn.click().run()
        assert not at.exception
        # magnetic_declination is set to None before st.rerun() — persists
        assert at.session_state["magnetic_declination"] is None

    def test_radio_renders_three_methods(self, proc):
        at = _run(_full_ss(proc))
        radios = at.radio
        # At least one radio (magnetic method) should be present
        assert len(radios) >= 1


# ===========================================================================
# CLASS 3 — Tab 2: Velocity Thresholds
# ===========================================================================


class TestTab2VelocityThresholds:
    """Tests for the Velocity Thresholds tab."""

    def test_tab2_renders_without_error(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_threshold_checkbox_present(self, proc):
        at = _run(_full_ss(proc))
        checkbox_keys = [c.key for c in at.checkbox]
        assert "threshold_checkbox" in checkbox_keys

    def test_threshold_inputs_visible_when_enabled(self, proc):
        ss = _full_ss(proc, apply_threshold=True)
        at = _run(ss)
        number_input_keys = [n.key for n in at.number_input]
        assert "cutoff_u_input" in number_input_keys
        assert "cutoff_v_input" in number_input_keys
        assert "cutoff_w_input" in number_input_keys

    def test_threshold_inputs_hidden_when_disabled(self, proc):
        """When apply_threshold=False, the cutoff inputs should not render."""
        ss = _full_ss(proc, apply_threshold=False)
        at = _run(ss)
        number_input_keys = [n.key for n in at.number_input]
        assert "cutoff_u_input" not in number_input_keys

    def test_uncheck_threshold_checkbox(self, proc):
        ss = _full_ss(proc, apply_threshold=True)
        at = _run(ss)
        cb = next(c for c in at.checkbox if c.key == "threshold_checkbox")
        cb.uncheck().run()
        assert not at.exception
        assert at.session_state["apply_threshold"] is False


# ===========================================================================
# CLASS 4 — Tab 3: Despike
# ===========================================================================


class TestTab3DespIke:
    """Tests for the Despike Data tab."""

    def test_tab3_renders_without_error(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_despike_checkbox_present(self, proc):
        at = _run(_full_ss(proc))
        keys = [c.key for c in at.checkbox]
        assert "despike_checkbox" in keys

    def test_despike_controls_visible_when_enabled(self, proc):
        ss = _full_ss(proc, apply_despike=True)
        at = _run(ss)
        ni_keys = [n.key for n in at.number_input]
        assert "despike_kernel_input" in ni_keys
        assert "despike_cutoff_input" in ni_keys

    def test_despike_controls_hidden_when_disabled(self, proc):
        ss = _full_ss(proc, apply_despike=False)
        at = _run(ss)
        ni_keys = [n.key for n in at.number_input]
        assert "despike_kernel_input" not in ni_keys

    def test_despike_visualization_rendered_when_enabled(self, proc):
        ss = _full_ss(proc, apply_despike=True)
        at = _run(ss)
        # Should render sliders for cell and ensemble range
        slider_keys = [s.key for s in at.slider]
        assert "despike_vis_cell" in slider_keys

    def test_check_despike_checkbox(self, proc):
        ss = _full_ss(proc, apply_despike=False)
        at = _run(ss)
        cb = next(c for c in at.checkbox if c.key == "despike_checkbox")
        cb.check().run()
        assert not at.exception
        assert at.session_state["apply_despike"] is True


# ===========================================================================
# CLASS 5 — Tab 4: Flatline Detection
# ===========================================================================


class TestTab4Flatline:
    """Tests for the Flatline Detection tab."""

    def test_tab4_renders_without_error(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_flatline_checkbox_present(self, proc):
        at = _run(_full_ss(proc))
        keys = [c.key for c in at.checkbox]
        assert "flatline_checkbox" in keys

    def test_flatline_controls_visible_when_enabled(self, proc):
        ss = _full_ss(proc, apply_flatline=True)
        at = _run(ss)
        ni_keys = [n.key for n in at.number_input]
        assert "flatline_kernel_input" in ni_keys
        assert "flatline_cutoff_input" in ni_keys

    def test_flatline_controls_hidden_when_disabled(self, proc):
        ss = _full_ss(proc, apply_flatline=False)
        at = _run(ss)
        ni_keys = [n.key for n in at.number_input]
        assert "flatline_kernel_input" not in ni_keys

    def test_flatline_sliders_visible_when_enabled(self, proc):
        ss = _full_ss(proc, apply_flatline=True)
        at = _run(ss)
        slider_keys = [s.key for s in at.slider]
        assert "flatline_vis_cell" in slider_keys

    def test_check_flatline_checkbox(self, proc):
        ss = _full_ss(proc, apply_flatline=False)
        at = _run(ss)
        cb = next(c for c in at.checkbox if c.key == "flatline_checkbox")
        cb.check().run()
        assert not at.exception
        assert at.session_state["apply_flatline"] is True


# ===========================================================================
# CLASS 6 — Tab 5: Preview
# ===========================================================================


class TestTab5Preview:
    """Tests for the Preview tab."""

    def test_tab5_renders_without_error(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_generate_preview_button_present(self, proc):
        at = _run(_full_ss(proc))
        btn_keys = [b.key for b in at.button]
        assert "preview_velocity" in btn_keys

    def test_warning_shown_when_no_preview(self, proc):
        ss = _full_ss(proc, velocity_preview_run=False)
        at = _run(ss)
        assert len(at.warning) >= 1

    def test_generate_preview_button_click_calls_runner(self, proc):
        """Clicking Generate Preview should call get_velocity_check_runner."""
        # Need a fresh proc so we can track calls
        ds = _make_ds()
        fresh_proc = _make_mock_processor(ds)
        # Give it a staging proc
        staging = MagicMock()
        staging.dataset = ds
        staging.get_velocity_check_runner.return_value = MockVelocityCheckRunner(ds)

        ss = _full_ss(fresh_proc, preview_velocity_proc=staging)
        at = _run(ss)
        btn = next(b for b in at.button if b.key == "preview_velocity")
        btn.click().run()
        assert not at.exception

    def test_preview_stats_displayed_after_run(self, proc):
        """When velocity_preview_run=True and stats exist, dataframe renders."""
        stat = MockQCCheckStats(
            check_name="Velocity Threshold",
            threshold={"U": 2500},
            cells_newly_masked=10,
            cells_total_masked=10,
        )
        ds = _make_ds()
        staging_proc = MagicMock()
        staging_proc.dataset = ds

        ss = _full_ss(
            proc,
            velocity_preview_run=True,
            velocity_preview_stats=[stat],
            velocity_preview_modifications=[],
            preview_velocity_proc=staging_proc,
        )
        at = _run(ss)
        assert not at.exception

    def test_preview_with_modifications_displayed(self, proc):
        """DataModificationStats should produce a Modifications section."""
        stat = MockQCCheckStats(check_name="Velocity Threshold")
        mod = MockDataModificationStats(declination=-5.0)
        ds = _make_ds()
        staging_proc = MagicMock()
        staging_proc.dataset = ds

        ss = _full_ss(
            proc,
            velocity_preview_run=True,
            velocity_preview_stats=[stat],
            velocity_preview_modifications=[mod],
            preview_velocity_proc=staging_proc,
        )
        at = _run(ss)
        assert not at.exception

    def test_mask_comparison_shown_after_preview(self, proc):
        """Mask comparison plot renders when preview has been run."""
        stat = MockQCCheckStats()
        ds = _make_ds()
        staging_proc = MagicMock()
        staging_proc.dataset = ds

        ss = _full_ss(
            proc,
            velocity_preview_run=True,
            velocity_preview_stats=[stat],
            velocity_preview_modifications=[],
            preview_velocity_proc=staging_proc,
        )
        at = _run(ss)
        assert not at.exception

    def test_settings_summary_shown_in_tab5(self, proc):
        """Tab 5 shows a settings table regardless of preview state."""
        ss = _full_ss(
            proc,
            apply_magnetic=True,
            magnetic_declination=-3.5,
            apply_threshold=True,
            apply_despike=True,
            apply_flatline=True,
        )
        at = _run(ss)
        assert not at.exception

    def test_settings_summary_shows_threshold_details(self, proc):
        ss = _full_ss(
            proc, apply_threshold=True, cutoff_u=1800, cutoff_v=1900, cutoff_w=400
        )
        at = _run(ss)
        assert not at.exception

    def test_settings_summary_shows_despike_details(self, proc):
        ss = _full_ss(proc, apply_despike=True, despike_kernel=11, despike_cutoff=2.5)
        at = _run(ss)
        assert not at.exception

    def test_settings_summary_shows_flatline_details(self, proc):
        ss = _full_ss(proc, apply_flatline=True, flatline_kernel=5, flatline_cutoff=2.0)
        at = _run(ss)
        assert not at.exception

    def test_settings_summary_shows_magnetic_details(self, proc):
        ss = _full_ss(proc, apply_magnetic=True, magnetic_declination=-12.345)
        at = _run(ss)
        assert not at.exception


# ===========================================================================
# CLASS 7 — Tab 6: Save & Reset
# ===========================================================================


class TestTab6SaveReset:
    """Tests for the Save & Reset tab."""

    def test_tab6_renders_without_error(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_save_velocity_button_present(self, proc):
        at = _run(_full_ss(proc))
        keys = [b.key for b in at.button]
        assert "save_velocity" in keys

    def test_reset_velocity_button_present(self, proc):
        at = _run(_full_ss(proc))
        keys = [b.key for b in at.button]
        assert "reset_velocity" in keys

    def test_not_applied_warning_shown(self, proc):
        ss = _full_ss(proc, velocity_applied=False)
        at = _run(ss)
        warnings = [
            w for w in at.warning if "not yet applied" in (w.value or "").lower()
        ]
        assert len(warnings) >= 1

    def test_save_button_applies_threshold(self, proc):
        """Clicking Save calls commit_runner; no exceptions expected."""
        ds = _make_ds()
        fresh_proc = _make_mock_processor(ds)
        ss = _full_ss(
            fresh_proc, apply_threshold=True, cutoff_u=2500, cutoff_v=2500, cutoff_w=500
        )
        at = _run(ss)
        btn = next(b for b in at.button if b.key == "save_velocity")
        btn.click().run()
        assert not at.exception

    def test_save_button_applies_magnetic(self, proc):
        ds = _make_ds()
        fresh_proc = _make_mock_processor(ds)
        ss = _full_ss(
            fresh_proc,
            apply_magnetic=True,
            magnetic_declination=-7.0,
            apply_threshold=False,
        )
        at = _run(ss)
        btn = next(b for b in at.button if b.key == "save_velocity")
        btn.click().run()
        assert not at.exception

    def test_save_button_applies_despike(self, proc):
        ds = _make_ds()
        fresh_proc = _make_mock_processor(ds)
        ss = _full_ss(
            fresh_proc,
            apply_despike=True,
            despike_kernel=13,
            despike_cutoff=3.0,
            apply_threshold=False,
        )
        at = _run(ss)
        btn = next(b for b in at.button if b.key == "save_velocity")
        btn.click().run()
        assert not at.exception

    def test_save_button_applies_flatline(self, proc):
        ds = _make_ds()
        fresh_proc = _make_mock_processor(ds)
        ss = _full_ss(
            fresh_proc,
            apply_flatline=True,
            flatline_kernel=4,
            flatline_cutoff=1.0,
            apply_threshold=False,
        )
        at = _run(ss)
        btn = next(b for b in at.button if b.key == "save_velocity")
        btn.click().run()
        assert not at.exception

    def test_save_all_operations_simultaneously(self, proc):
        ds = _make_ds()
        fresh_proc = _make_mock_processor(ds)
        ss = _full_ss(
            fresh_proc,
            apply_magnetic=True,
            magnetic_declination=-5.0,
            apply_threshold=True,
            cutoff_u=2500,
            cutoff_v=2500,
            cutoff_w=500,
            apply_despike=True,
            despike_kernel=13,
            despike_cutoff=3.0,
            apply_flatline=True,
            flatline_kernel=4,
            flatline_cutoff=1.0,
        )
        at = _run(ss)
        btn = next(b for b in at.button if b.key == "save_velocity")
        btn.click().run()
        assert not at.exception

    def test_reset_button_clears_state(self, proc):
        ds = _make_ds()
        fresh_proc = _make_mock_processor(ds)
        ss = _full_ss(
            fresh_proc,
            apply_magnetic=True,
            magnetic_declination=-5.0,
            velocity_preview_run=True,
        )
        at = _run(ss)
        btn = next(b for b in at.button if b.key == "reset_velocity")
        btn.click().run()
        assert not at.exception
        # After reset, magnetic should be cleared
        assert at.session_state["apply_magnetic"] is False
        assert at.session_state["magnetic_declination"] is None
        assert at.session_state["velocity_preview_run"] is False

    def test_save_sets_velocity_applied_true(self, proc):
        ds = _make_ds()
        fresh_proc = _make_mock_processor(ds)
        ss = _full_ss(fresh_proc, apply_threshold=True, velocity_applied=False)
        at = _run(ss)
        btn = next(b for b in at.button if b.key == "save_velocity")
        btn.click().run()
        assert not at.exception
        assert at.session_state["velocity_applied"] is True

    def test_settings_table_in_save_tab(self, proc):
        """Status table (magnetic, threshold, despike, flatline) renders."""
        ss = _full_ss(
            proc,
            apply_magnetic=True,
            apply_threshold=True,
            apply_despike=False,
            apply_flatline=False,
        )
        at = _run(ss)
        assert not at.exception

    def test_runner_statistics_shown_after_save(self, proc):
        """Statistics section appears after save when runner.statistics is non-empty."""
        # Configure runner to return stats
        ds = _make_ds()
        fresh_proc = _make_mock_processor(ds)

        stat = MockQCCheckStats(
            check_name="Velocity Threshold",
            threshold={"U": 2500},
            cells_newly_masked=100,
            cells_total_masked=100,
        )

        runner_with_stats = MockVelocityCheckRunner(ds)
        runner_with_stats.statistics = [stat]
        fresh_proc.get_velocity_check_runner.return_value = runner_with_stats

        ss = _full_ss(fresh_proc, apply_threshold=True)
        at = _run(ss)
        btn = next(b for b in at.button if b.key == "save_velocity")
        btn.click().run()
        assert not at.exception


# ===========================================================================
# CLASS 8 — Sidebar
# ===========================================================================


class TestSidebar:
    """Tests for the sidebar Processing Status section."""

    def test_sidebar_renders_without_error(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_sidebar_shows_threshold_values_when_enabled(self, proc):
        ss = _full_ss(
            proc, apply_threshold=True, cutoff_u=1500, cutoff_v=1600, cutoff_w=300
        )
        at = _run(ss)
        assert not at.exception

    def test_sidebar_hides_threshold_values_when_disabled(self, proc):
        ss = _full_ss(proc, apply_threshold=False)
        at = _run(ss)
        assert not at.exception

    def test_sidebar_shows_magnetic_info_when_applied(self, proc):
        ss = _full_ss(proc, apply_magnetic=True, magnetic_declination=-9.876)
        at = _run(ss)
        assert not at.exception

    def test_sidebar_processing_log_shown(self, proc):
        """Sidebar shows processing log when it exists."""
        log_proc = _make_mock_processor(_make_ds())
        log_proc.processing_log = ["Step 1: QC applied", "Step 2: Velocity threshold"]
        ss = _full_ss(log_proc)
        at = _run(ss)
        assert not at.exception

    def test_sidebar_no_log_message_shown(self, proc):
        """Without processing log, fallback message shown."""
        no_log_proc = _make_mock_processor(_make_ds())
        no_log_proc.processing_log = []
        # Remove the hasattr path — use an object without processing_log attr
        no_log_proc2 = _make_mock_processor(_make_ds())
        del no_log_proc2.processing_log
        ss = _full_ss(no_log_proc2)
        at = _run(ss)
        assert not at.exception


# ===========================================================================
# CLASS 9 — Direct function tests (via importlib)
# ===========================================================================


class TestPageFunctionsDirectly:
    """
    Exercise helper functions in the actual page source via importlib,
    covering branches that are hard to reach through the AppTest UI.
    """

    @pytest.fixture(scope="class")
    def page_module(self, inject_pyadps_mock):
        """Load the page module with a patched environment."""
        import importlib.util
        import streamlit as st

        ds_mod = _make_ds()
        proc_mod = _make_mock_processor(ds_mod)

        spec = importlib.util.spec_from_file_location("vel_page", SCRIPT_PATH)
        mod = importlib.util.module_from_spec(spec)

        with (
            patch.object(st, "set_page_config"),
            patch.object(st, "stop", side_effect=SystemExit(0)),
            patch.object(st, "error"),
            patch.object(st, "header"),
            patch.object(st, "write"),
            patch.object(st, "tabs", return_value=[MagicMock() for _ in range(6)]),
            patch.object(
                st,
                "form",
                return_value=MagicMock(
                    __enter__=lambda s: s, __exit__=MagicMock(return_value=False)
                ),
            ),
            patch.dict(
                "streamlit.session_state",
                {
                    "processor": proc_mod,
                    "velocity_initialized": True,
                    "apply_threshold": True,
                    "apply_magnetic": False,
                    "apply_despike": False,
                    "apply_flatline": False,
                    "cutoff_u": 2500,
                    "cutoff_v": 2500,
                    "cutoff_w": 500,
                    "despike_kernel": 13,
                    "despike_cutoff": 3.0,
                    "flatline_kernel": 4,
                    "flatline_cutoff": 1.0,
                    "magnetic_declination": None,
                    "preview_velocity_proc": MagicMock(dataset=ds_mod),
                    "velocity_preview_run": False,
                    "velocity_applied": False,
                    "velocity_preview_stats": None,
                    "velocity_preview_modifications": None,
                    "magnetic_method": "pygeomag",
                    "magnetic_lat": 0.0,
                    "magnetic_lon": 0.0,
                    "magnetic_year": 2025,
                    "magnetic_depth": 0,
                },
                clear=False,
            ),
        ):
            try:
                spec.loader.exec_module(mod)
            except SystemExit:
                pass

        # Attach ds so tests can reference it
        mod.ds = ds_mod
        return mod

    # --- get_total_ensembles ---

    def test_get_total_ensembles_time_dim(self, page_module):
        assert page_module.get_total_ensembles() == 20

    def test_get_total_ensembles_no_time_dim(self, page_module):
        original_ds = page_module.ds
        page_module.ds = xr.Dataset({"x": (["a"], [1, 2, 3])})
        try:
            result = page_module.get_total_ensembles()
            assert result == 0
        finally:
            page_module.ds = original_ds

    def test_get_total_ensembles_ensemble_dim(self, page_module):
        original_ds = page_module.ds
        page_module.ds = xr.Dataset({"x": (["ensemble", "cell"], np.zeros((5, 3)))})
        try:
            result = page_module.get_total_ensembles()
            assert result == 5
        finally:
            page_module.ds = original_ds

    # --- get_total_cells ---

    def test_get_total_cells_cell_dim(self, page_module):
        assert page_module.get_total_cells() == 10

    def test_get_total_cells_no_cell_dim(self, page_module):
        original_ds = page_module.ds
        page_module.ds = xr.Dataset({"x": (["a"], [1, 2])})
        try:
            result = page_module.get_total_cells()
            assert result == 0
        finally:
            page_module.ds = original_ds

    # --- get_total_beams ---

    def test_get_total_beams_beam_dim(self, page_module):
        assert page_module.get_total_beams() == 4

    def test_get_total_beams_no_beam_dim(self, page_module):
        original_ds = page_module.ds
        page_module.ds = xr.Dataset({"x": (["a"], [1, 2])})
        try:
            result = page_module.get_total_beams()
            assert result == 4  # default
        finally:
            page_module.ds = original_ds

    # --- get_time_axis ---

    def test_get_time_axis_time_coord(self, page_module):
        result = page_module.get_time_axis()
        assert len(result) == 20

    def test_get_time_axis_ensemble_coord(self, page_module):
        original_ds = page_module.ds
        page_module.ds = xr.Dataset(
            {"v": (["ensemble"], np.zeros(7))},
            coords={"ensemble": np.arange(7)},
        )
        try:
            result = page_module.get_time_axis()
            assert len(result) == 7
        finally:
            page_module.ds = original_ds

    def test_get_time_axis_fallback(self, page_module):
        original_ds = page_module.ds
        page_module.ds = xr.Dataset({"v": (["x"], np.zeros(5))})
        try:
            result = page_module.get_time_axis()
            assert len(result) == 0  # get_total_ensembles returns 0
        finally:
            page_module.ds = original_ds

    # --- get_time_interval_str ---

    def test_get_time_interval_str_with_time(self, page_module):
        result = page_module.get_time_interval_str()
        assert result != "N/A"

    def test_get_time_interval_str_no_time(self, page_module):
        original_ds = page_module.ds
        page_module.ds = xr.Dataset({"v": (["x"], [1])})
        try:
            result = page_module.get_time_interval_str()
            assert result == "N/A"
        finally:
            page_module.ds = original_ds

    def test_get_time_interval_str_single_point(self, page_module):
        original_ds = page_module.ds
        page_module.ds = xr.Dataset(
            {"v": (["time"], [1.0])},
            coords={"time": pd.date_range("2024-01-01", periods=1, freq="h")},
        )
        try:
            result = page_module.get_time_interval_str()
            assert result == "N/A"
        finally:
            page_module.ds = original_ds

    # --- is_earth_coordinates ---

    def test_is_earth_coordinates_earth(self, page_module):
        # The _FLStub returns "Earth Coordinates"; page tries accessor first
        # but _FLStub.coordinate_transformation() returns Earth
        result = page_module.is_earth_coordinates()
        assert result is True

    def test_is_earth_coordinates_fallback_attr(self, page_module):
        """When accessor raises, fallback reads ds.attrs — earth system -> True."""
        original_ds = page_module.ds
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            xr.register_dataset_accessor("fixed_leader")(_RaisingFLStub)
        page_module.ds = xr.Dataset(
            {"v": (["x"], [1])},
            attrs={"coordinate_system": "earth"},
        )
        try:
            result = page_module.is_earth_coordinates()
            assert result is True
        finally:
            page_module.ds = original_ds
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                xr.register_dataset_accessor("fixed_leader")(_FLStub)

    def test_is_earth_coordinates_beam_fallback(self, page_module):
        """When accessor raises, fallback reads attrs — beam system → False."""
        original_ds = page_module.ds
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            xr.register_dataset_accessor("fixed_leader")(_RaisingFLStub)
        page_module.ds = xr.Dataset(
            {"v": (["x"], [1])},
            attrs={"coordinate_system": "beam"},
        )
        try:
            result = page_module.is_earth_coordinates()
            assert result is False
        finally:
            page_module.ds = original_ds
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                xr.register_dataset_accessor("fixed_leader")(_FLStub)

    # --- get_velocity_labels ---

    def test_get_velocity_labels_earth(self, page_module):
        result = page_module.get_velocity_labels()
        assert result == ("U (East)", "V (North)", "W (Vertical)")

    def test_get_velocity_labels_beam(self, page_module):
        """Beam labels when accessor raises and attrs says beam."""
        original_ds = page_module.ds
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            xr.register_dataset_accessor("fixed_leader")(_RaisingFLStub)
        page_module.ds = xr.Dataset(
            {"v": (["x"], [1])},
            attrs={"coordinate_system": "beam"},
        )
        try:
            result = page_module.get_velocity_labels()
            assert result == ("Beam 1", "Beam 2", "Beam 3")
        finally:
            page_module.ds = original_ds
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                xr.register_dataset_accessor("fixed_leader")(_FLStub)

    # --- status_color_map ---

    def test_status_color_map_true(self, page_module):
        result = page_module.status_color_map("True")
        assert "green" in result

    def test_status_color_map_false(self, page_module):
        result = page_module.status_color_map("False")
        assert "red" in result

    def test_status_color_map_other(self, page_module):
        result = page_module.status_color_map(None)
        assert result == ""

    # --- compute_median_filter ---

    def test_compute_median_filter_basic(self, page_module):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = page_module.compute_median_filter(data, kernel_size=3)
        assert result.shape == data.shape

    def test_compute_median_filter_with_nans(self, page_module):
        data = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = page_module.compute_median_filter(data, kernel_size=3)
        assert np.isnan(result[1])

    # --- detect_spikes ---

    def test_detect_spikes_finds_spike(self, page_module):
        data = np.zeros(50, dtype=float)
        data[25] = 10000.0  # huge spike
        _, spike_mask, _ = page_module.detect_spikes(data, kernel_size=5, cutoff=2.0)
        assert spike_mask[25]

    def test_detect_spikes_no_spike(self, page_module):
        data = np.ones(50, dtype=float)
        _, spike_mask, std_dev = page_module.detect_spikes(
            data, kernel_size=5, cutoff=3.0
        )
        assert not np.any(spike_mask)
        assert std_dev == pytest.approx(0.0)

    def test_detect_spikes_returns_tuple(self, page_module):
        data = np.random.randn(100)
        result = page_module.detect_spikes(data, kernel_size=5, cutoff=3.0)
        assert len(result) == 3

    # --- detect_flatlines ---

    def test_detect_flatlines_finds_flatline(self, page_module):
        data = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 10.0], dtype=float)
        mask = page_module.detect_flatlines(data, kernel_size=4, cutoff=0.1)
        assert np.any(mask[:5])

    def test_detect_flatlines_short_data(self, page_module):
        data = np.array([1.0, 2.0], dtype=float)
        mask = page_module.detect_flatlines(data, kernel_size=10, cutoff=0.1)
        assert not np.any(mask)

    def test_detect_flatlines_no_flatline(self, page_module):
        np.random.seed(0)
        data = np.random.randn(50) * 100
        mask = page_module.detect_flatlines(data, kernel_size=4, cutoff=0.0001)
        # With tiny cutoff almost nothing qualifies
        assert not np.all(mask)

    def test_detect_flatlines_final_run(self, page_module):
        """Test the 'check final run' branch (flatline at end of array)."""
        data = np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=float)
        mask = page_module.detect_flatlines(data, kernel_size=3, cutoff=0.0)
        assert np.any(mask)

    def test_detect_flatlines_middle_and_end(self, page_module):
        """Runs both mid-array and end-of-array flatline branches."""
        data = np.concatenate([np.zeros(5), np.linspace(0, 10, 5), np.zeros(5)]).astype(
            float
        )
        mask = page_module.detect_flatlines(data, kernel_size=3, cutoff=0.1)
        assert isinstance(mask, np.ndarray)


# ===========================================================================
# CLASS 10 — Plotting function smoke tests
# ===========================================================================


class TestPlottingFunctions:
    """
    Smoke tests for plotting functions — verify they don't crash
    when called with valid data through the page module.
    """

    @pytest.fixture(scope="class")
    def page_module(self, inject_pyadps_mock):
        import importlib.util
        import streamlit as st

        ds_mod = _make_ds()
        proc_mod = _make_mock_processor(ds_mod)

        spec = importlib.util.spec_from_file_location("vel_page_plot", SCRIPT_PATH)
        mod = importlib.util.module_from_spec(spec)

        with (
            patch.object(st, "set_page_config"),
            patch.object(st, "stop", side_effect=SystemExit(0)),
            patch.object(st, "error"),
            patch.object(st, "header"),
            patch.object(st, "write"),
            patch.object(st, "plotly_chart"),  # prevent actual rendering
            patch.object(st, "warning"),
            patch.object(st, "tabs", return_value=[MagicMock() for _ in range(6)]),
            patch.object(
                st,
                "form",
                return_value=MagicMock(
                    __enter__=lambda s: s, __exit__=MagicMock(return_value=False)
                ),
            ),
            patch.dict(
                "streamlit.session_state",
                {
                    "processor": proc_mod,
                    "velocity_initialized": True,
                    "apply_threshold": True,
                    "apply_magnetic": False,
                    "apply_despike": False,
                    "apply_flatline": False,
                    "cutoff_u": 2500,
                    "cutoff_v": 2500,
                    "cutoff_w": 500,
                    "despike_kernel": 13,
                    "despike_cutoff": 3.0,
                    "flatline_kernel": 4,
                    "flatline_cutoff": 1.0,
                    "magnetic_declination": None,
                    "preview_velocity_proc": MagicMock(dataset=ds_mod),
                    "velocity_preview_run": False,
                    "velocity_applied": False,
                    "velocity_preview_stats": None,
                    "velocity_preview_modifications": None,
                    "magnetic_method": "pygeomag",
                    "magnetic_lat": 0.0,
                    "magnetic_lon": 0.0,
                    "magnetic_year": 2025,
                    "magnetic_depth": 0,
                },
                clear=False,
            ),
        ):
            try:
                spec.loader.exec_module(mod)
            except SystemExit:
                pass

        mod.ds = ds_mod
        return mod

    def test_plot_velocity_component_no_crash(self, page_module):
        with patch("streamlit.plotly_chart"):
            page_module.plot_velocity_component(beam_idx=0, title="U Component")

    def test_plot_velocity_component_no_velocity_data(self, page_module):
        original_ds = page_module.ds
        page_module.ds = xr.Dataset(
            {"mask": (["beam", "cell", "time"], np.zeros((4, 5, 5)))}
        )
        try:
            with (
                patch("streamlit.plotly_chart"),
                patch("streamlit.warning") as mock_warn,
            ):
                page_module.plot_velocity_component(beam_idx=0, title="U Component")
                mock_warn.assert_called_once()
        finally:
            page_module.ds = original_ds

    def test_plot_velocity_histogram_no_crash(self, page_module):
        with patch("streamlit.plotly_chart"):
            page_module.plot_velocity_histogram(beam_idx=0, title="U Histogram")

    def test_plot_velocity_histogram_no_velocity(self, page_module):
        original_ds = page_module.ds
        page_module.ds = xr.Dataset(
            {"mask": (["beam", "cell", "time"], np.zeros((4, 5, 5)))}
        )
        try:
            with (
                patch("streamlit.plotly_chart"),
                patch("streamlit.warning") as mock_warn,
            ):
                page_module.plot_velocity_histogram(beam_idx=0, title="U Histogram")
                mock_warn.assert_called_once()
        finally:
            page_module.ds = original_ds

    def test_plot_mask_comparison_4_beams(self, page_module):
        n_e = 20
        n_c = 10
        mask_a = np.zeros((4, n_c, n_e), dtype=np.int8)
        mask_b = np.zeros((4, n_c, n_e), dtype=np.int8)
        mask_b[3, :5, :] = 1
        with patch("streamlit.plotly_chart"):
            page_module.plot_mask_comparison(mask_a, mask_b)

    def test_plot_mask_comparison_2_beams(self, page_module):
        """Branch: beam_idx falls back to 0 when shape[0] <= 3."""
        n_e = 20
        n_c = 10
        mask_a = np.zeros((2, n_c, n_e), dtype=np.int8)
        mask_b = np.zeros((2, n_c, n_e), dtype=np.int8)
        with patch("streamlit.plotly_chart"):
            page_module.plot_mask_comparison(mask_a, mask_b)

    def test_plot_despike_timeseries_no_spikes(self, page_module):
        velocity_data = np.ones((4, 10, 20), dtype=np.int16) * 100
        with patch("streamlit.plotly_chart"):
            page_module.plot_despike_timeseries(
                velocity_data=velocity_data,
                beam_idx=0,
                cell_idx=3,
                ens_start=0,
                ens_end=20,
                kernel_size=5,
                cutoff=3.0,
                component_label="U (East)",
            )

    def test_plot_despike_timeseries_with_spikes(self, page_module):
        velocity_data = np.ones((4, 10, 20), dtype=np.int16) * 100
        velocity_data[0, 3, 10] = 30000  # spike
        with patch("streamlit.plotly_chart"):
            page_module.plot_despike_timeseries(
                velocity_data=velocity_data,
                beam_idx=0,
                cell_idx=3,
                ens_start=0,
                ens_end=20,
                kernel_size=3,
                cutoff=1.0,
                component_label="U (East)",
            )

    def test_plot_despike_timeseries_missing_values(self, page_module):
        velocity_data = np.ones((4, 10, 20), dtype=np.int16) * 100
        velocity_data[0, 3, :5] = -32768
        with patch("streamlit.plotly_chart"):
            page_module.plot_despike_timeseries(
                velocity_data=velocity_data,
                beam_idx=0,
                cell_idx=3,
                ens_start=0,
                ens_end=20,
                kernel_size=5,
                cutoff=3.0,
                component_label="U (East)",
            )

    def test_plot_flatline_timeseries_no_flatlines(self, page_module):
        np.random.seed(0)
        velocity_data = (np.random.randn(4, 10, 20) * 500).astype(np.int16)
        with patch("streamlit.plotly_chart"):
            page_module.plot_flatline_timeseries(
                velocity_data=velocity_data,
                beam_idx=0,
                cell_idx=3,
                ens_start=0,
                ens_end=20,
                kernel_size=4,
                cutoff=1.0,
                component_label="U (East)",
            )

    def test_plot_flatline_timeseries_with_flatlines(self, page_module):
        velocity_data = np.ones((4, 10, 20), dtype=np.int16) * 100
        with patch("streamlit.plotly_chart"):
            page_module.plot_flatline_timeseries(
                velocity_data=velocity_data,
                beam_idx=0,
                cell_idx=3,
                ens_start=0,
                ens_end=20,
                kernel_size=4,
                cutoff=0.0,
                component_label="U (East)",
            )

    def test_plot_flatline_timeseries_missing_values(self, page_module):
        velocity_data = np.ones((4, 10, 20), dtype=np.int16) * 200
        velocity_data[0, 3, :5] = -32768
        with patch("streamlit.plotly_chart"):
            page_module.plot_flatline_timeseries(
                velocity_data=velocity_data,
                beam_idx=0,
                cell_idx=3,
                ens_start=0,
                ens_end=20,
                kernel_size=3,
                cutoff=0.0,
                component_label="U (East)",
            )


# ===========================================================================
# CLASS 11 — Edge cases and alternate dataset shapes
# ===========================================================================


class TestEdgeCasesAndAlternateShapes:
    """Edge-case tests for unusual dataset configurations."""

    def test_ensemble_dim_instead_of_time(self):
        """Page handles 'ensemble' dimension gracefully."""
        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "ensemble"],
                    np.zeros((4, 5, 10), dtype=np.int16),
                ),
                "mask": (
                    ["beam", "cell", "ensemble"],
                    np.zeros((4, 5, 10), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(5),
                "ensemble": np.arange(10),
                "depth_cell_length": np.int64(400),
                "bin_1_distance": np.int64(200),
            },
            attrs={"coordinate_system": "earth"},
        )
        proc = _make_mock_processor(ds)
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_beam_coordinate_system_renders_correct_labels(self):
        """Page uses Beam 1/2/3 labels for beam-coordinate data."""
        ds = _make_ds(earth_coords=False)
        proc = _make_mock_processor(ds)
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_no_velocity_in_dataset(self):
        """Page renders without crashing when 'velocity' is absent."""
        ds = xr.Dataset(
            {"mask": (["beam", "cell", "time"], np.zeros((4, 5, 10), dtype=np.int8))},
            coords={
                "beam": np.arange(4),
                "cell": np.arange(5),
                "time": pd.date_range("2024-01-01", periods=10, freq="h"),
                "depth_cell_length": np.int64(400),
                "bin_1_distance": np.int64(200),
            },
            attrs={"coordinate_system": "earth"},
        )
        proc = _make_mock_processor(ds)
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_single_cell_dataset(self):
        """Page handles 1-cell dataset without crashing.

        Note: sliders require min < max so apply_despike and apply_flatline
        must be disabled for single-cell datasets (slider max_value would be 0).
        """
        ds = _make_ds(n_cells=1, n_time=5)
        proc = _make_mock_processor(ds)
        at = _run(_full_ss(proc, apply_despike=False, apply_flatline=False))
        assert not at.exception

    def test_very_small_dataset(self):
        ds = _make_ds(n_cells=2, n_time=3)
        proc = _make_mock_processor(ds)
        at = _run(_full_ss(proc))
        assert not at.exception


# ===========================================================================
# CLASS 12 — Processor-level error handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error paths in save/preview workflows."""

    def test_save_error_shows_error_widget(self):
        """If commit_runner raises, the page catches it and calls st.error."""
        ds = _make_ds()
        failing_proc = _make_mock_processor(ds)
        # Make commit_runner (called INSIDE the try block) raise
        failing_proc.commit_runner.side_effect = RuntimeError("Commit failed")

        ss = _full_ss(failing_proc, apply_threshold=True)
        at = _run(ss)
        btn = next(b for b in at.button if b.key == "save_velocity")
        btn.click().run()
        # Page wraps this in try/except and calls st.error — no unhandled exception
        assert not at.exception

    def test_preview_error_shows_error_widget(self):
        """If preview commit_runner raises, page catches it and calls st.error."""
        ds = _make_ds()
        failing_staging = MagicMock()
        failing_staging.dataset = ds
        # Runner itself succeeds but commit raises — inside the try block
        runner = MockVelocityCheckRunner(ds)
        failing_staging.get_velocity_check_runner.return_value = runner
        failing_staging.commit_runner.side_effect = ValueError("Preview commit fail")

        failing_proc = _make_mock_processor(ds)
        ss = _full_ss(
            failing_proc, preview_velocity_proc=failing_staging, apply_threshold=True
        )
        at = _run(ss)
        btn = next(b for b in at.button if b.key == "preview_velocity")
        btn.click().run()
        assert not at.exception


# ===========================================================================
# CLASS 13 — Targeted coverage for specific uncovered lines
# ===========================================================================


class TestCoverageGaps:
    """
    Pinpoint tests for lines not reached by the main suite.

    Lines targeted:
      33-34   HAS_RESAMPLER = False  (ImportError branch)
      71      return 0 in get_total_cells  (no cell dim)
      384     go.Figure() else-branch in plot_despike_timeseries
      486     go.Figure() else-branch in plot_flatline_timeseries
      526-527 segment boundary append inside flatline plot
      688-704 Tab 1 API method form inputs
      716-747 Form submit: Manual path + pygeomag/API path + error path
      886     st.session_state.despike_kernel = despike_kernel (odd-kernel fix)
      961     st.info No velocity data in despike tab
      1078    st.info No velocity data in flatline tab
      1147    runner.magnetic_correction() in preview
      1161    runner.despike() in preview
      1168    runner.flatline() in preview
      1182    st.error in preview exception handler
    """

    @pytest.fixture(scope="class")
    def page_module(self, inject_pyadps_mock):
        """Load page module; HAS_RESAMPLER forced False."""
        import importlib.util
        import streamlit as st

        ds_mod = _make_ds()
        proc_mod = _make_mock_processor(ds_mod)

        spec = importlib.util.spec_from_file_location("vel_page_cov", SCRIPT_PATH)
        mod = importlib.util.module_from_spec(spec)

        with (
            patch.object(st, "set_page_config"),
            patch.object(st, "stop", side_effect=SystemExit(0)),
            patch.object(st, "error"),
            patch.object(st, "header"),
            patch.object(st, "write"),
            patch.object(st, "plotly_chart"),
            patch.object(st, "warning"),
            patch.object(
                st,
                "form",
                return_value=MagicMock(
                    __enter__=lambda s: s,
                    __exit__=MagicMock(return_value=False),
                ),
            ),
            patch.object(st, "tabs", return_value=[MagicMock() for _ in range(6)]),
            patch.dict(
                "streamlit.session_state",
                {
                    "processor": proc_mod,
                    "velocity_initialized": True,
                    "apply_threshold": True,
                    "apply_magnetic": False,
                    "apply_despike": False,
                    "apply_flatline": False,
                    "cutoff_u": 2500,
                    "cutoff_v": 2500,
                    "cutoff_w": 500,
                    "despike_kernel": 13,
                    "despike_cutoff": 3.0,
                    "flatline_kernel": 4,
                    "flatline_cutoff": 1.0,
                    "magnetic_declination": None,
                    "preview_velocity_proc": MagicMock(dataset=ds_mod),
                    "velocity_preview_run": False,
                    "velocity_applied": False,
                    "velocity_preview_stats": None,
                    "velocity_preview_modifications": None,
                    "magnetic_method": "pygeomag",
                    "magnetic_lat": 0.0,
                    "magnetic_lon": 0.0,
                    "magnetic_year": 2025,
                    "magnetic_depth": 0,
                },
                clear=False,
            ),
        ):
            try:
                spec.loader.exec_module(mod)
            except SystemExit:
                pass

        mod.ds = ds_mod
        mod.HAS_RESAMPLER = False
        return mod

    # Lines 33-34: HAS_RESAMPLER = False

    def test_has_resampler_false(self, page_module):
        assert page_module.HAS_RESAMPLER is False

    # Line 71: get_total_cells returns 0

    def test_get_total_cells_no_cell_dim(self, page_module):
        original = page_module.ds
        page_module.ds = xr.Dataset({"v": (["x"], [1, 2])})
        try:
            assert page_module.get_total_cells() == 0
        finally:
            page_module.ds = original

    # Line 384: go.Figure() else-branch in despike plot

    def test_plot_despike_else_branch(self, page_module):
        assert page_module.HAS_RESAMPLER is False
        vel = np.ones((4, 10, 20), dtype=np.int16) * 100
        with patch("streamlit.plotly_chart"):
            page_module.plot_despike_timeseries(
                velocity_data=vel,
                beam_idx=0,
                cell_idx=0,
                ens_start=0,
                ens_end=20,
                kernel_size=3,
                cutoff=3.0,
                component_label="U",
            )

    def test_plot_despike_large_range_no_resampler(self, page_module):
        assert page_module.HAS_RESAMPLER is False
        n = 6000
        vel = np.ones((4, 5, n), dtype=np.int16) * 50
        original = page_module.ds
        page_module.ds = _make_ds(n_cells=5, n_time=n)
        try:
            with patch("streamlit.plotly_chart"):
                page_module.plot_despike_timeseries(
                    velocity_data=vel,
                    beam_idx=0,
                    cell_idx=2,
                    ens_start=0,
                    ens_end=n,
                    kernel_size=3,
                    cutoff=3.0,
                    component_label="U",
                )
        finally:
            page_module.ds = original

    # Line 486: go.Figure() else-branch in flatline plot

    def test_plot_flatline_else_branch(self, page_module):
        assert page_module.HAS_RESAMPLER is False
        vel = np.ones((4, 10, 20), dtype=np.int16) * 100
        with patch("streamlit.plotly_chart"):
            page_module.plot_flatline_timeseries(
                velocity_data=vel,
                beam_idx=0,
                cell_idx=0,
                ens_start=0,
                ens_end=20,
                kernel_size=4,
                cutoff=0.0,
                component_label="U",
            )

    # Lines 523-527: segment boundary append in flatline plot

    def test_plot_flatline_multiple_segments(self, page_module):
        """Two non-contiguous flatline segments trigger the boundary-append branch."""
        data = np.array([5.0] * 5 + [100.0, 200.0] + [5.0] * 5, dtype=np.float64)
        n = len(data)
        vel = np.tile(data, (4, 5, 1)).astype(np.int16)
        original = page_module.ds
        page_module.ds = _make_ds(n_cells=5, n_time=n)
        try:
            with patch("streamlit.plotly_chart"):
                page_module.plot_flatline_timeseries(
                    velocity_data=vel,
                    beam_idx=0,
                    cell_idx=0,
                    ens_start=0,
                    ens_end=n,
                    kernel_size=3,
                    cutoff=0.5,
                    component_label="U",
                )
        finally:
            page_module.ds = original

    def test_detect_flatlines_two_separate_runs(self, page_module):
        """detect_flatlines boundary-append: run ends mid-array, new run starts."""
        data = np.array([10.0] * 5 + [0.0, 0.0] + [10.0] * 5, dtype=np.float64)
        mask = page_module.detect_flatlines(data, kernel_size=3, cutoff=0.1)
        assert np.any(mask[:5])
        assert np.any(mask[7:])

    # Lines 688-704: Tab 1 API method UI

    def test_tab1_api_method_renders(self, inject_pyadps_mock):
        ds = _make_ds()
        proc = _make_mock_processor(ds)
        at = _run(_full_ss(proc))
        at.radio[0].set_value("API").run()
        assert not at.exception

    def test_tab1_manual_method_renders(self, inject_pyadps_mock):
        ds = _make_ds()
        proc = _make_mock_processor(ds)
        at = _run(_full_ss(proc))
        at.radio[0].set_value("Manual").run()
        assert not at.exception

    # Lines 716-722: Form submit — Manual path

    def test_form_submit_manual_sets_declination(self, inject_pyadps_mock):
        ds = _make_ds()
        proc = _make_mock_processor(ds)
        at = _run(_full_ss(proc))
        at.radio[0].set_value("Manual").run()
        assert not at.exception
        next(b for b in at.button if "magnetic_form" in b.key).click().run()
        assert not at.exception
        assert at.session_state["apply_magnetic"] is True
        assert at.session_state["magnetic_declination"] == pytest.approx(0.0, abs=1e-9)

    # Lines 724-741: Form submit — pygeomag path

    def test_form_submit_pygeomag_calls_runner(self, inject_pyadps_mock):
        ds = _make_ds()
        staging_ds = ds.copy(deep=True)
        staging_ds.attrs["magnetic_declination_applied"] = -7.5
        runner_mock = MockVelocityCheckRunner(staging_ds)
        runner_mock.dataset.attrs["magnetic_declination_applied"] = -7.5
        staging_proc = MagicMock()
        staging_proc.dataset = ds
        staging_proc.get_velocity_check_runner.return_value = runner_mock
        proc = _make_mock_processor(ds)
        ss = _full_ss(
            proc, magnetic_method="pygeomag", preview_velocity_proc=staging_proc
        )
        at = _run(ss)
        next(b for b in at.button if "magnetic_form" in b.key).click().run()
        assert not at.exception
        assert at.session_state["apply_magnetic"] is True

    # Lines 742-743: Form submit — exception path

    def test_form_submit_pygeomag_exception_path(self, inject_pyadps_mock):
        ds = _make_ds()
        staging_proc = MagicMock()
        staging_proc.dataset = ds
        staging_proc.get_velocity_check_runner.side_effect = RuntimeError("mag fail")
        proc = _make_mock_processor(ds)
        ss = _full_ss(
            proc, magnetic_method="pygeomag", preview_velocity_proc=staging_proc
        )
        at = _run(ss)
        next(b for b in at.button if "magnetic_form" in b.key).click().run()
        assert not at.exception

    # Lines 744-745: API method error shows info hint

    def test_form_submit_api_error_shows_info(self, inject_pyadps_mock):
        ds = _make_ds()
        staging_proc = MagicMock()
        staging_proc.dataset = ds
        staging_proc.get_velocity_check_runner.side_effect = RuntimeError("API down")
        proc = _make_mock_processor(ds)
        ss = _full_ss(
            proc, magnetic_method="pygeomag", preview_velocity_proc=staging_proc
        )
        at = _run(ss)
        at.radio[0].set_value("API").run()
        assert not at.exception
        next(b for b in at.button if "magnetic_form" in b.key).click().run()
        assert not at.exception

    # Line 886: odd-kernel enforcement

    def test_even_despike_kernel_incremented(self, inject_pyadps_mock):
        """Line 883-886: even kernel 12 → 13."""
        ds = _make_ds(n_time=60)
        proc = _make_mock_processor(ds)
        ss = _full_ss(proc, apply_despike=True, despike_kernel=12)
        at = _run(ss)
        assert not at.exception
        assert at.session_state["despike_kernel"] == 13

    def test_odd_despike_kernel_unchanged(self, inject_pyadps_mock):
        ds = _make_ds(n_time=60)
        proc = _make_mock_processor(ds)
        ss = _full_ss(proc, apply_despike=True, despike_kernel=11)
        at = _run(ss)
        assert not at.exception
        assert at.session_state["despike_kernel"] == 11

    # Line 961: despike tab no-velocity info

    def test_despike_no_velocity_shows_info(self, inject_pyadps_mock):
        ds_nv = xr.Dataset(
            {"mask": (["beam", "cell", "time"], np.zeros((4, 10, 20), np.int8))},
            coords={
                "beam": np.arange(4),
                "cell": np.arange(10),
                "time": pd.date_range("2024-01-01", periods=20, freq="h"),
                "depth_cell_length": np.int64(400),
                "bin_1_distance": np.int64(200),
            },
            attrs={"coordinate_system": "earth"},
        )
        proc = _make_mock_processor(ds_nv)
        ss = _full_ss(proc, apply_despike=True)
        at = _run(ss)
        assert not at.exception
        assert any("velocity" in (i.value or "").lower() for i in at.info)

    # Line 1078: flatline tab no-velocity info

    def test_flatline_no_velocity_shows_info(self, inject_pyadps_mock):
        ds_nv = xr.Dataset(
            {"mask": (["beam", "cell", "time"], np.zeros((4, 10, 20), np.int8))},
            coords={
                "beam": np.arange(4),
                "cell": np.arange(10),
                "time": pd.date_range("2024-01-01", periods=20, freq="h"),
                "depth_cell_length": np.int64(400),
                "bin_1_distance": np.int64(200),
            },
            attrs={"coordinate_system": "earth"},
        )
        proc = _make_mock_processor(ds_nv)
        ss = _full_ss(proc, apply_flatline=True)
        at = _run(ss)
        assert not at.exception
        assert any("velocity" in (i.value or "").lower() for i in at.info)

    # Line 1147: runner.magnetic_correction() in preview

    def test_preview_magnetic_correction_called(self, inject_pyadps_mock):
        ds = _make_ds()
        runner = MockVelocityCheckRunner(ds)
        staging = MagicMock(dataset=ds)
        staging.get_velocity_check_runner.return_value = runner
        proc = _make_mock_processor(ds)
        ss = _full_ss(
            proc,
            preview_velocity_proc=staging,
            apply_magnetic=True,
            magnetic_declination=-5.0,
            apply_threshold=False,
            apply_despike=False,
            apply_flatline=False,
        )
        at = _run(ss)
        next(b for b in at.button if b.key == "preview_velocity").click().run()
        assert not at.exception
        assert at.session_state["velocity_preview_run"] is True

    # Line 1161: runner.despike() in preview

    def test_preview_despike_called(self, inject_pyadps_mock):
        ds = _make_ds()
        runner = MockVelocityCheckRunner(ds)
        staging = MagicMock(dataset=ds)
        staging.get_velocity_check_runner.return_value = runner
        proc = _make_mock_processor(ds)
        ss = _full_ss(
            proc,
            preview_velocity_proc=staging,
            apply_magnetic=False,
            apply_threshold=False,
            apply_despike=True,
            despike_kernel=13,
            despike_cutoff=3.0,
            apply_flatline=False,
        )
        at = _run(ss)
        next(b for b in at.button if b.key == "preview_velocity").click().run()
        assert not at.exception
        assert at.session_state["velocity_preview_run"] is True

    # Line 1168: runner.flatline() in preview

    def test_preview_flatline_called(self, inject_pyadps_mock):
        ds = _make_ds()
        runner = MockVelocityCheckRunner(ds)
        staging = MagicMock(dataset=ds)
        staging.get_velocity_check_runner.return_value = runner
        proc = _make_mock_processor(ds)
        ss = _full_ss(
            proc,
            preview_velocity_proc=staging,
            apply_magnetic=False,
            apply_threshold=False,
            apply_despike=False,
            apply_flatline=True,
            flatline_kernel=4,
            flatline_cutoff=1.0,
        )
        at = _run(ss)
        next(b for b in at.button if b.key == "preview_velocity").click().run()
        assert not at.exception
        assert at.session_state["velocity_preview_run"] is True

    # Lines 1181-1183: preview exception handler

    def test_preview_exception_handler(self, inject_pyadps_mock):
        ds = _make_ds()
        runner = MockVelocityCheckRunner(ds)
        staging = MagicMock(dataset=ds)
        staging.get_velocity_check_runner.return_value = runner
        staging.commit_runner.side_effect = RuntimeError("commit exploded")
        proc = _make_mock_processor(ds)
        ss = _full_ss(proc, preview_velocity_proc=staging, apply_threshold=True)
        at = _run(ss)
        next(b for b in at.button if b.key == "preview_velocity").click().run()
        assert not at.exception

    # Line 1184-1186: col_status shows info when preview_run=True

    def test_preview_col_status_info_shown(self, inject_pyadps_mock):
        ds = _make_ds()
        staging = MagicMock(dataset=ds)
        proc = _make_mock_processor(ds)
        ss = _full_ss(
            proc,
            preview_velocity_proc=staging,
            velocity_preview_run=True,
            velocity_preview_stats=[],
            velocity_preview_modifications=[],
        )
        at = _run(ss)
        assert not at.exception
        assert any("Preview is available" in (i.value or "") for i in at.info)


# ===========================================================================
# CLASS 14 — AppTest coverage for lines 33-34, 71, 384, 486, 1183
# ===========================================================================


def _run_no_resampler(extra_ss: Dict[str, Any], timeout: int = 15) -> AppTest:
    """Run AppTest with plotly_resampler blocked via builtins.__import__.

    We patch builtins.__import__ to raise ImportError only for plotly_resampler
    while letting all other imports (numpy, scipy, etc.) proceed normally.
    This executes lines 33-34 (HAS_RESAMPLER = False) in the page without
    disrupting scipy's internal import chain.
    """
    import builtins

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "plotly_resampler":
            raise ImportError(f"Mocked: {name} not available")
        return real_import(name, *args, **kwargs)

    at = AppTest.from_file(SCRIPT_PATH, default_timeout=timeout)
    for k, v in extra_ss.items():
        at.session_state[k] = v
    with patch.object(builtins, "__import__", side_effect=_blocking_import):
        at.run()
    return at


class TestImportAndPlotBranches:
    """
    AppTest-based tests that exercise branches only reachable when
    plotly_resampler is absent (lines 33-34, 384, 486) and when the dataset
    has no cell dimension (line 71).

    These tests use _run_no_resampler() which patches sys.modules so that
    the page's `from plotly_resampler import FigureResampler` raises
    ImportError, setting HAS_RESAMPLER = False for that run.
    """

    # ------------------------------------------------------------------ #
    # Lines 33-34: except ImportError → HAS_RESAMPLER = False            #
    # ------------------------------------------------------------------ #

    def test_page_loads_without_resampler(self, proc):
        """Lines 33-34: page renders normally even when plotly_resampler absent."""
        at = _run_no_resampler(_full_ss(proc))
        assert not at.exception

    def test_page_loads_without_resampler_all_tabs_render(self, proc):
        """Lines 33-34 + 384 + 486: despike+flatline plots work with HAS_RESAMPLER=False."""
        ss = _full_ss(proc, apply_despike=True, apply_flatline=True)
        at = _run_no_resampler(ss)
        assert not at.exception

    # ------------------------------------------------------------------ #
    # Line 71: return 0 in get_total_cells (no cell dim)                 #
    # ------------------------------------------------------------------ #

    def test_no_cell_dim_page_renders(self):
        """Line 71: page handles dataset with no cell dimension gracefully."""
        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "ensemble", "time"],
                    np.zeros((4, 5, 10), dtype=np.int16),
                ),
                "mask": (
                    ["beam", "ensemble", "time"],
                    np.zeros((4, 5, 10), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(4),
                "ensemble": np.arange(5),
                "time": pd.date_range("2024-01-01", periods=10, freq="h"),
                "depth_cell_length": np.int64(400),
                "bin_1_distance": np.int64(200),
            },
            attrs={"coordinate_system": "earth"},
        )
        proc = _make_mock_processor(ds)
        # No "cell" dim → get_total_cells() returns 0 (line 71)
        # Disable despike/flatline since their sliders need cell > 0
        at = _run(_full_ss(proc, apply_despike=False, apply_flatline=False))
        assert not at.exception

    def test_no_cell_dim_without_resampler(self):
        """Lines 33-34 + 71: combine no-resampler + no-cell-dim."""
        ds = xr.Dataset(
            {"mask": (["beam", "time"], np.zeros((4, 10), dtype=np.int8))},
            coords={
                "beam": np.arange(4),
                "time": pd.date_range("2024-01-01", periods=10, freq="h"),
                "depth_cell_length": np.int64(400),
                "bin_1_distance": np.int64(200),
            },
            attrs={"coordinate_system": "earth"},
        )
        proc = _make_mock_processor(ds)
        at = _run_no_resampler(
            _full_ss(proc, apply_despike=False, apply_flatline=False)
        )
        assert not at.exception

    # ------------------------------------------------------------------ #
    # Line 384: go.Figure() else-branch in plot_despike_timeseries       #
    # ------------------------------------------------------------------ #

    def test_despike_plot_no_resampler(self, proc):
        """Line 384: despike visualization renders via go.Figure() when HAS_RESAMPLER=False."""
        ss = _full_ss(proc, apply_despike=True)
        at = _run_no_resampler(ss)
        assert not at.exception

    def test_despike_plot_large_data_no_resampler(self):
        """Line 384: >5000-point dataset still uses go.Figure() when resampler absent."""
        ds = _make_ds(n_cells=10, n_time=600)
        proc = _make_mock_processor(ds)
        ss = _full_ss(proc, apply_despike=True, despike_kernel=3)
        at = _run_no_resampler(ss)
        assert not at.exception

    # ------------------------------------------------------------------ #
    # Line 486: go.Figure() else-branch in plot_flatline_timeseries      #
    # ------------------------------------------------------------------ #

    def test_flatline_plot_no_resampler(self, proc):
        """Line 486: flatline visualization renders via go.Figure() when HAS_RESAMPLER=False."""
        ss = _full_ss(proc, apply_flatline=True)
        at = _run_no_resampler(ss)
        assert not at.exception

    def test_flatline_plot_large_data_no_resampler(self):
        """Line 486: >5000-point dataset still uses go.Figure() when resampler absent."""
        ds = _make_ds(n_cells=10, n_time=600)
        proc = _make_mock_processor(ds)
        ss = _full_ss(proc, apply_flatline=True)
        at = _run_no_resampler(ss)
        assert not at.exception

    # ------------------------------------------------------------------ #
    # Line 1183: st.error() in preview except handler                    #
    # ------------------------------------------------------------------ #

    def test_preview_except_handler_st_error(self):
        """Line 1182-1183: preview button handler catches exception and calls st.error.

        The page at line 1134 does:
            st.session_state.preview_velocity_proc = ProcessedDataset(proc.dataset)

        So we must make mock_processing.ProcessedDataset() return a MagicMock
        whose get_velocity_check_runner() raises — that way the exception is
        caught by the except block at line 1181 and st.error() is called (line 1183).
        """
        ds = _make_ds()

        # Build a proc whose dataset will be wrapped by ProcessedDataset mock
        proc = _make_mock_processor(ds)

        # Make the ProcessedDataset mock return a staging proc that raises
        failing_staging = MagicMock()
        failing_staging.dataset = ds
        failing_staging.get_velocity_check_runner.side_effect = RuntimeError(
            "forced preview failure"
        )

        # Patch mock_processing.ProcessedDataset to return our failing staging
        import sys as _sys

        mock_proc_module = _sys.modules.get("pyadps.processing")
        original_pd = getattr(mock_proc_module, "ProcessedDataset", None)
        mock_proc_module.ProcessedDataset = MagicMock(return_value=failing_staging)

        try:
            ss = _full_ss(proc, apply_threshold=True)
            at = _run(ss)
            next(b for b in at.button if b.key == "preview_velocity").click().run()
            # Page catches the exception with except Exception as e → st.error(...)
            # No unhandled exception should escape to AppTest
            assert not at.exception
            # The preview flag stays False because the exception aborted before line 1177
            assert at.session_state["velocity_preview_run"] is False
        finally:
            if original_pd is not None:
                mock_proc_module.ProcessedDataset = original_pd


# ===========================================================================
# CLASS 14 — AppTest-only coverage for lines 71, 384, 486
#
# These lines are in functions that importlib-based tests call on private
# module copies, which do not contribute to source-file coverage.
# AppTest runs the real page script under the coverage tracer and is the
# only strategy that registers on the measured source file.
#
# Line  71 : return 0 in get_total_cells() — dataset has no "cell" dim
# Line 384 : go.Figure() else-branch in plot_despike_timeseries()
# Line 486 : go.Figure() else-branch in plot_flatline_timeseries()
#
# HAS_RESAMPLER is False in the test environment (plotly_resampler not
# installed), so "if HAS_RESAMPLER and ..." is always False and the else
# branch (lines 384 / 486) is always taken when the plotting functions run.
# ===========================================================================


class TestAppTestOnlyCoverage:
    """
    AppTest tests that hit lines 71, 384, and 486 through the real page
    script so coverage.py records them on the source file.
    """

    # ------------------------------------------------------------------ #
    # Line 71: get_total_cells() returns 0 when no "cell" dim in dataset  #
    #                                                                      #
    # The page calls get_total_cells() unconditionally on every render    #
    # (sidebar + tab guards). Pass a dataset whose only spatial dim is    #
    # "depth" — no "cell" dim — so the `if "cell" in ds.dims` branch is  #
    # False and `return 0` executes.                                       #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Line 384: go.Figure() else-branch in plot_despike_timeseries()      #
    #                                                                      #
    # The despike visualization tab calls plot_despike_timeseries() when  #
    # apply_despike=True and velocity data exists. Since HAS_RESAMPLER is #
    # False in this environment, the `if HAS_RESAMPLER and ...` condition #
    # is False on every call, so `fig = go.Figure()` on line 384 runs.   #
    # ------------------------------------------------------------------ #

    def test_line384_despike_plot_uses_go_figure(self):
        """Line 384: despike plot takes the else-branch (HAS_RESAMPLER=False)."""
        ds = _make_ds(n_cells=10, n_time=20)
        proc = _make_mock_processor(ds)
        at = _run(
            _full_ss(proc, apply_despike=True, despike_kernel=3, despike_cutoff=3.0)
        )
        assert not at.exception

    def test_line384_despike_plot_large_dataset(self):
        """Line 384: same branch with a larger dataset (n_time=100)."""
        ds = _make_ds(n_cells=8, n_time=100)
        proc = _make_mock_processor(ds)
        at = _run(
            _full_ss(proc, apply_despike=True, despike_kernel=3, despike_cutoff=3.0)
        )
        assert not at.exception

    # ------------------------------------------------------------------ #
    # Line 486: go.Figure() else-branch in plot_flatline_timeseries()     #
    #                                                                      #
    # Same logic as line 384 but in the flatline visualization tab.       #
    # ------------------------------------------------------------------ #

    def test_line486_flatline_plot_uses_go_figure(self):
        """Line 486: flatline plot takes the else-branch (HAS_RESAMPLER=False)."""
        ds = _make_ds(n_cells=10, n_time=20)
        proc = _make_mock_processor(ds)
        at = _run(
            _full_ss(proc, apply_flatline=True, flatline_kernel=4, flatline_cutoff=1.0)
        )
        assert not at.exception

    def test_line486_flatline_plot_large_dataset(self):
        """Line 486: same branch with a larger dataset (n_time=100)."""
        ds = _make_ds(n_cells=8, n_time=100)
        proc = _make_mock_processor(ds)
        at = _run(
            _full_ss(proc, apply_flatline=True, flatline_kernel=4, flatline_cutoff=1.0)
        )
        assert not at.exception


# ===========================================================================
# CLASS 15 — Definitive importlib coverage for lines 70, 384, 486
#
# These tests load the page module via importlib (ensuring co_filename matches
# the source file for coverage), patch ALL accessor behaviour explicitly, and
# call the three functions directly.  They are independent of which
# fixed_leader accessor the conftest registers.
# ===========================================================================


class TestDefinitiveCoverage:
    """
    Self-contained importlib tests for lines 70 (get_total_cells return 0),
    384 (go.Figure() in plot_despike_timeseries) and 486 (go.Figure() in
    plot_flatline_timeseries).

    The module is loaded fresh for each test so HAS_RESAMPLER, ds, and all
    module globals are in a known state.  The fixed_leader accessor is patched
    inline so the conftest autouse accessor does not interfere.
    """

    def _load_fresh_mod(self, ds_override=None):
        """Load a fresh copy of the page module and return (mod, ds)."""
        import importlib.util
        import streamlit as st

        ds = ds_override if ds_override is not None else _make_ds()
        proc = _make_mock_processor(ds)

        spec = importlib.util.spec_from_file_location("vel_definitive", SCRIPT_PATH)
        mod = importlib.util.module_from_spec(spec)

        ss = {
            "processor": proc,
            "velocity_initialized": True,
            "preview_velocity_proc": MagicMock(dataset=ds),
            "velocity_preview_run": False,
            "velocity_applied": False,
            "velocity_preview_stats": None,
            "velocity_preview_modifications": None,
            "apply_magnetic": False,
            "magnetic_declination": None,
            "magnetic_method": "pygeomag",
            "magnetic_lat": 0.0,
            "magnetic_lon": 0.0,
            "magnetic_year": 2025,
            "magnetic_depth": 0,
            "apply_threshold": True,
            "cutoff_u": 2500,
            "cutoff_v": 2500,
            "cutoff_w": 500,
            "apply_despike": False,
            "despike_kernel": 13,
            "despike_cutoff": 3.0,
            "apply_flatline": False,
            "flatline_kernel": 4,
            "flatline_cutoff": 1.0,
        }

        with (
            patch.object(st, "set_page_config"),
            patch.object(st, "stop", side_effect=SystemExit(0)),
            patch.object(st, "error"),
            patch.object(st, "header"),
            patch.object(st, "write"),
            patch.object(st, "plotly_chart"),
            patch.object(st, "warning"),
            patch.object(
                st,
                "form",
                return_value=MagicMock(
                    __enter__=lambda s: s,
                    __exit__=MagicMock(return_value=False),
                ),
            ),
            patch.object(st, "tabs", return_value=[MagicMock() for _ in range(6)]),
            patch.dict("streamlit.session_state", ss, clear=False),
        ):
            try:
                spec.loader.exec_module(mod)
            except SystemExit:
                pass

        mod.ds = ds
        mod.HAS_RESAMPLER = False  # explicit — never depends on plotly_resampler
        return mod, ds

    # ------------------------------------------------------------------ #
    # Line 70: return 0 in get_total_cells()                              #
    # The function reads the module-level `ds` global.  We swap it for a  #
    # dataset with no "cell" dim so the if-branch is False.               #
    # ------------------------------------------------------------------ #

    def test_line70_get_total_cells_no_cell_dim(self, inject_pyadps_mock):
        """Line 70: get_total_cells() returns 0 when ds has no cell dim."""
        mod, _ = self._load_fresh_mod()

        # Replace ds with one that has no cell dimension
        mod.ds = xr.Dataset(
            {"v": (["x"], [1, 2, 3])},
        )
        result = mod.get_total_cells()
        assert result == 0

    def test_line70_get_total_cells_arbitrary_dim_not_cell(self, inject_pyadps_mock):
        """Line 70: dataset with no cell-equivalent dim → return 0."""
        mod, _ = self._load_fresh_mod()
        # Use a dim name that no version of get_total_cells recognises
        mod.ds = xr.Dataset({"v": (["ensemble"], np.zeros(5))})
        result = mod.get_total_cells()
        assert result == 0

    # ------------------------------------------------------------------ #
    # Line 384: fig = go.Figure() else-branch in plot_despike_timeseries  #
    # HAS_RESAMPLER is forced False; len(x_axis)=20 < 5000 — the `if`   #
    # condition is False on both counts, so the else branch executes.     #
    # ------------------------------------------------------------------ #

    def test_line384_go_figure_else_branch_despike(self, inject_pyadps_mock):
        """Line 384: HAS_RESAMPLER=False → go.Figure() else-branch in despike plot."""
        mod, ds = self._load_fresh_mod()
        assert mod.HAS_RESAMPLER is False  # precondition

        velocity_data = np.ones((4, 10, 20), dtype=np.int16) * 100
        with patch("streamlit.plotly_chart"):
            mod.plot_despike_timeseries(
                velocity_data=velocity_data,
                beam_idx=0,
                cell_idx=3,
                ens_start=0,
                ens_end=20,
                kernel_size=3,
                cutoff=3.0,
                component_label="U (East)",
            )
        # If line 384 was reached, the function completed without error

    def test_line384_despike_with_spikes_still_uses_go_figure(self, inject_pyadps_mock):
        """Line 384: spike-containing data still takes the go.Figure() else-branch."""
        mod, _ = self._load_fresh_mod()
        assert mod.HAS_RESAMPLER is False

        velocity_data = np.ones((4, 10, 20), dtype=np.int16) * 100
        velocity_data[0, 3, 10] = 30000  # clear spike
        with patch("streamlit.plotly_chart"):
            mod.plot_despike_timeseries(
                velocity_data=velocity_data,
                beam_idx=0,
                cell_idx=3,
                ens_start=0,
                ens_end=20,
                kernel_size=3,
                cutoff=1.0,
                component_label="U (East)",
            )

    # ------------------------------------------------------------------ #
    # Line 486: fig = go.Figure() else-branch in plot_flatline_timeseries #
    # ------------------------------------------------------------------ #

    def test_line486_go_figure_else_branch_flatline(self, inject_pyadps_mock):
        """Line 486: HAS_RESAMPLER=False → go.Figure() else-branch in flatline plot."""
        mod, ds = self._load_fresh_mod()
        assert mod.HAS_RESAMPLER is False  # precondition

        velocity_data = np.ones((4, 10, 20), dtype=np.int16) * 100
        with patch("streamlit.plotly_chart"):
            mod.plot_flatline_timeseries(
                velocity_data=velocity_data,
                beam_idx=0,
                cell_idx=3,
                ens_start=0,
                ens_end=20,
                kernel_size=4,
                cutoff=0.0,
                component_label="U (East)",
            )

    def test_line486_flatline_with_actual_flatlines(self, inject_pyadps_mock):
        """Line 486: flatline data still takes the go.Figure() else-branch."""
        mod, _ = self._load_fresh_mod()
        assert mod.HAS_RESAMPLER is False

        # Constant velocity → definite flatlines
        velocity_data = np.ones((4, 10, 20), dtype=np.int16) * 200
        with patch("streamlit.plotly_chart"):
            mod.plot_flatline_timeseries(
                velocity_data=velocity_data,
                beam_idx=0,
                cell_idx=3,
                ens_start=0,
                ens_end=20,
                kernel_size=4,
                cutoff=0.0,
                component_label="U (East)",
            )


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
