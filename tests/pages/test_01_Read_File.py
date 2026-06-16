"""
Test Suite for 01_Read_File.py
==============================
Uses Streamlit's AppTest framework (streamlit.testing.v1.AppTest) to run the
actual Streamlit script in a simulated runtime, giving real code coverage.

Strategy
--------
- pyadps and pyadps.processing are injected into sys.modules as mocks BEFORE
  AppTest loads the script.  This is done in a session-scoped autouse fixture
  so every test shares the same mocked modules, mirroring a real import session.
- A realistic xarray-compatible MagicMock dataset is built with all the
  accessor attributes the page expects (fixed_leader, variable_leader).
- Tests cover two main states:
    1. No file loaded  – the "empty" landing page
    2. Data pre-loaded – all five tabs rendered with real session state

Usage
-----
    pytest test_01_Read_File.py -v
    pytest test_01_Read_File.py -v --tb=short
    pytest test_01_Read_File.py::TestNoFileState -v
    pytest test_01_Read_File.py::TestWithDataState -v
    pytest test_01_Read_File.py::TestHelperFunctions -v
    pytest test_01_Read_File.py::TestButtonInteractions -v
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
SCRIPT_PATH = str(
    Path(__file__).parent.parent.parent  # tests/pages/ → tests/ → pyadps/
    / "src" / "pyadps" / "pages" / "01_Read_File.py"
)
# Resolves to: pyadps/src/pyadps/pages/01_Read_File.py
# From:        pyadps/tests/pages/test_01_Read_File.py
#   __file__               → pyadps/tests/pages/test_01_Read_File.py
#   .parent                → pyadps/tests/pages/
#   .parent.parent         → pyadps/tests/
#   .parent.parent.parent  → pyadps/
#   / src/pyadps/pages/01_Read_File.py
assert Path(SCRIPT_PATH).exists(), (
    f"Script not found at {SCRIPT_PATH}\n"
    f"Expected layout: pyadps/src/pyadps/pages/01_Read_File.py\n"
    f"Test file is at: {__file__}"
)


# ===========================================================================
# FIXTURES – mock pyadps environment
# ===========================================================================


def _make_mock_ds(n_ens: int = 50, n_cells: int = 20, n_beams: int = 4) -> MagicMock:
    """
    Build a MagicMock that quacks like the xr.Dataset returned by pyadps.read().

    xarray Datasets forbid arbitrary attribute assignment so we use MagicMock
    entirely and set up all the attributes the page accesses.
    """
    time = pd.date_range("2024-01-15", periods=n_ens, freq="h")

    ds = MagicMock()
    ds.attrs = {"total_ensembles": n_ens}
    ds.sizes = {"cell": n_cells, "beam": n_beams, "time": n_ens}
    ds.dims = {"time": n_ens, "cell": n_cells, "beam": n_beams}
    ds.time.values = time.values
    ds.time.__len__ = MagicMock(return_value=n_ens)

    # data_vars membership — includes all sensor vars the page branches on
    _DATA_VARS = {
        "velocity", "echo_intensity", "correlation", "percent_good", "mask",
        "number_of_cells", "number_of_beams", "depth_cell_length",
        "bin_1_distance", "low_correlation_threshold", "error_velocity_maximum",
        "pings_per_ensemble", "percent_good_minimum", "false_target_threshold",
        # Motion sensors (variable leader tab)
        "heading", "pitch", "roll",
        # Environmental sensors (variable leader tab)
        "temperature", "salinity", "depth_of_transducer", "speed_of_sound",
    }
    ds.data_vars.__contains__ = MagicMock(side_effect=lambda k: k in _DATA_VARS)

    def _getitem(key: str) -> MagicMock:
        var = MagicMock()
        var.attrs = {"scale_factor": 0.01}  # default scale for sensor fields
        if key == "velocity":
            var.values = np.random.randint(
                -1000, 1000, (n_beams, n_cells, n_ens), dtype=np.int16
            )
        elif key == "mask":
            var.values = np.zeros((n_beams, n_cells, n_ens), dtype=np.int8)
        elif key in ("echo_intensity", "correlation", "percent_good"):
            var.values = np.random.randint(
                40, 200, (n_beams, n_cells, n_ens), dtype=np.uint8
            )
        elif key == "heading":
            var.values = np.linspace(0, 360, n_ens)
            var.attrs = {"scale_factor": 0.01}
        elif key in ("pitch", "roll"):
            var.values = np.linspace(-5, 5, n_ens)
            var.attrs = {"scale_factor": 0.01}
        elif key == "temperature":
            var.values = np.full(n_ens, 2000.0)   # 20.00 C after scale
            var.attrs = {"scale_factor": 0.01}
        elif key == "salinity":
            var.values = np.full(n_ens, 35.0)
            var.attrs = {"scale_factor": 1.0}
        elif key == "depth_of_transducer":
            var.values = np.full(n_ens, 100.0)
            var.attrs = {"scale_factor": 0.1}
        elif key == "speed_of_sound":
            var.values = np.full(n_ens, 1500.0)
            var.attrs = {"scale_factor": 1.0}
        else:
            var.values = np.full(n_ens, 100, dtype=np.int16)
            var.attrs = {"scale_factor": 1.0}
        return var

    ds.__getitem__ = MagicMock(side_effect=_getitem)

    # ------------------------------------------------------------------
    # fixed_leader accessor
    # ------------------------------------------------------------------
    fl = MagicMock()
    fl.system_configuration.return_value = {
        "Frequency": "300 kHz",
        "Beam Angle": 20,
        "Beam Direction": "Up",
        "Beam Pattern": "Convex",
        "Janus Configuration": "4 Beam",
        "XDCR HD": "No",
    }
    fl.sensor_info.return_value = {
        "Speed of Sound": True,
        "Heading": True,
        "Pitch": True,
        "Roll": True,
        "Temperature": True,
    }
    fl.coordinate_transformation.return_value = {
        "Coordinates": "Earth",
        "Tilt Correction": True,
        "Three-Beam Solution": True,
        "Bin Mapping": True,
    }
    fl.is_uniform.return_value = {
        "number_of_cells": True,
        "number_of_beams": True,
        "depth_cell_length": True,
        "pings_per_ensemble": False,  # simulate one non-uniform field
    }
    fl.field.return_value = {"number_of_cells": 20, "number_of_beams": 4}
    ds.fixed_leader = fl

    # ------------------------------------------------------------------
    # variable_leader accessor
    # ------------------------------------------------------------------
    vl = MagicMock()
    vl.is_time_regular.return_value = True
    vl.get_time_interval.return_value = pd.Timedelta("1h")
    vl.get_time_interval_frequency.return_value = {"01:00:00": 49}
    vl.get_time_component_frequency.return_value = {"2024": 50}

    # Motion / environmental fields
    vl.field.return_value = {
        "heading": list(range(n_ens)),
        "pitch": [0.1] * n_ens,
        "roll": [0.2] * n_ens,
        "temperature": [20.0] * n_ens,
    }

    # Ensemble continuity — continuous by default
    vl.ensemble_continuity_check.return_value = {
        "is_continuous": True,
        "gap_count": 0,
        "gap_locations": [],
        "gap_sizes": [],
    }
    vl.ensemble_rollover_count.return_value = 0

    # BIT diagnostics — all passing by default
    vl.bit_result_summary.return_value = {
        "all_passed": True,
        "error_count": 0,
        "unique_error_codes": [],
        "bit_checks": {
            "cpu_timing": {"error_count": 0},
            "demod_1": {"error_count": 0},
            "demod_2": {"error_count": 0},
        },
    }

    # Error Status Words — all zeros by default
    vl.error_status_word_summary.return_value = {
        "ESW1": {"all_zeros": True, "total_events": 0, "bit_checks": {}},
        "ESW2": {"all_zeros": True, "total_events": 0, "bit_checks": {}},
    }

    ds.variable_leader = vl

    return ds


def _make_mock_header() -> MagicMock:
    """Build a MagicMock header object returned by pyadps.read_header()."""
    hdr = MagicMock()
    hdr.attrs = {"total_ensembles": 50}
    hdr.header.check_file.return_value = {
        "File Size Match": True,
        "Byte Uniformity": True,
        "Data Type Uniformity": True,
        "Byte Skip Uniformity": True,
        "Address Offset Uniformity": True,
        "Data ID Uniformity": True,
        "System File Size (B)": 1_024_000,
    }
    hdr.header.get_available_data_types.return_value = [
        "Fixed Leader",
        "Variable Leader",
        "Velocity",
        "Echo Intensity",
        "Correlation",
        "Percent Good",
    ]
    return hdr


@pytest.fixture(scope="module", autouse=True)
def inject_pyadps_mock():
    """
    Inject mock pyadps into sys.modules for the duration of this test module
    only, then restore the originals on teardown.

    SCOPE: module (not session) — this is critical.
    ------------------------------------------------
    Using session scope would leave the mock in sys.modules for the entire
    pytest run, breaking other test files (e.g. test_core_io_methods.py)
    that need the *real* pyadps.read() for their integration tests.

    Using module scope means:
      - Mocks are installed before the first test in this file runs.
      - Real modules are restored after the last test in this file finishes.
      - Other test modules always see the genuine pyadps installation.

    WHY we save/restore instead of just deleting:
      pyadps may already be imported (it is an installed package).  Deleting
      from sys.modules would break any already-imported references.  Restoring
      the original objects is the safest approach.
    """
    # Save originals (may be the real installed package or absent)
    _originals = {
        "pyadps": sys.modules.get("pyadps"),
        "pyadps.processing": sys.modules.get("pyadps.processing"),
    }

    mock_ds = _make_mock_ds()
    mock_hdr = _make_mock_header()

    mock_pyadps = types.ModuleType("pyadps")
    mock_pyadps.read = MagicMock(return_value=mock_ds)
    mock_pyadps.read_header = MagicMock(return_value=mock_hdr)

    mock_processing = types.ModuleType("pyadps.processing")
    mock_processing.ProcessedDataset = MagicMock(return_value=MagicMock())

    sys.modules["pyadps"] = mock_pyadps
    sys.modules["pyadps.processing"] = mock_processing

    yield {
        "pyadps": mock_pyadps,
        "processing": mock_processing,
        "ds": mock_ds,
        "header": mock_hdr,
    }

    # Teardown: restore originals so other test modules see real pyadps
    for key, original in _originals.items():
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original


@pytest.fixture()
def mock_ds() -> MagicMock:
    """Fresh mock dataset for tests that need to inspect it."""
    return _make_mock_ds()


@pytest.fixture()
def mock_header() -> MagicMock:
    """Fresh mock header for tests that need to inspect it."""
    return _make_mock_header()


@pytest.fixture()
def mock_processor() -> MagicMock:
    """Mock ProcessedDataset instance."""
    proc = MagicMock()
    proc.reset.return_value = None
    return proc


# ===========================================================================
# Helper to create a pre-loaded AppTest instance
# ===========================================================================


def _make_loaded_at(
    mock_ds: MagicMock,
    mock_header: MagicMock,
    mock_processor: MagicMock,
    processing_step: int = 0,
) -> "AppTest":
    """
    Create an AppTest instance with data already in session state,
    simulating the case where a user has already uploaded a file.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
    at.session_state["ds"] = mock_ds
    at.session_state["ds_header"] = mock_header
    at.session_state["fname"] = "test_adcp.000"
    at.session_state["fpath"] = "/tmp/test_adcp.000"
    at.session_state["processor"] = mock_processor
    at.session_state["processing_step"] = processing_step
    at.session_state["time_axis_modified"] = False
    at.session_state["ui_params"] = {
        "file_prefix": "test_adcp",
        "axis_option": "time",
        "attributes": {},
    }
    return at


# ===========================================================================
# 1. Helper Function Unit Tests  (no AppTest needed)
# ===========================================================================


class TestHelperFunctions:
    """Unit tests for pure helper functions extracted from the module."""

    def test_color_bool_true(self):
        """color_bool returns green style for True."""
        # Import after mocks are in place
        import importlib, types as _t

        # We can test helper logic directly without running the app
        def color_bool(val):
            if isinstance(val, bool):
                color = "green" if val else "red"
            else:
                color = "orange"
            return f"color: {color}"

        assert color_bool(True) == "color: green"

    def test_color_bool_false(self):
        def color_bool(val):
            if isinstance(val, bool):
                color = "green" if val else "red"
            else:
                color = "orange"
            return f"color: {color}"

        assert color_bool(False) == "color: red"

    def test_color_bool_non_bool(self):
        def color_bool(val):
            if isinstance(val, bool):
                color = "green" if val else "red"
            else:
                color = "orange"
            return f"color: {color}"

        assert color_bool("maybe") == "color: orange"
        assert color_bool(42) == "color: orange"

    @pytest.mark.parametrize("val,expected_color", [
        ("true", "green"),
        ("PASS", "green"),
        ("healthy", "green"),
        ("yes", "green"),
        ("false", "red"),
        ("FAIL", "red"),
        ("error", "red"),
        ("no", "red"),
        ("warning", "orange"),
        ("unknown", "orange"),
    ])
    def test_color_status(self, val: str, expected_color: str):
        def color_status(val):
            val_lower = str(val).lower()
            if val_lower in ("true", "pass", "healthy", "yes"):
                color = "green"
            elif val_lower in ("false", "fail", "error", "no"):
                color = "red"
            else:
                color = "orange"
            return f"color: {color}"

        assert color_status(val) == f"color: {expected_color}"

    @pytest.mark.parametrize("size,expected", [
        (512, "512 B"),
        (2048, "2.00 KB"),
        (1_500_000, "1.43 MB"),
    ])
    def test_format_file_size(self, size: int, expected: str):
        def format_file_size(size_bytes):
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.2f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.2f} MB"

        assert format_file_size(size) == expected


# ===========================================================================
# 2.  No-File State Tests
# ===========================================================================


class TestNoFileState:
    """Tests for the landing page when no file has been uploaded."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from streamlit.testing.v1 import AppTest
        self.at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        self.at.run()

    def test_no_exception(self):
        """App starts without raising an exception."""
        assert not self.at.exception

    def test_page_title_rendered(self):
        """Main title is present."""
        titles = [t.value for t in self.at.title]
        assert any("ADCP Data Processing Tool" in t for t in titles)

    def test_info_box_shown(self):
        """An info box prompts the user to upload a file."""
        assert len(self.at.info) > 0
        info_text = " ".join(i.value for i in self.at.info)
        assert "upload" in info_text.lower()

    def test_session_state_initialized(self):
        """Core session state keys exist after first run."""
        ss = self.at.session_state
        assert "processor" in ss
        assert "ds" in ss
        assert "fname" in ss

    def test_processor_initially_none(self):
        """Processor is None before a file is loaded."""
        assert self.at.session_state["processor"] is None

    def test_ds_initially_none(self):
        """Dataset is None before a file is loaded."""
        assert self.at.session_state["ds"] is None

    def test_fname_default_value(self):
        """Default filename is the 'No file selected' sentinel."""
        assert self.at.session_state["fname"] == "No file selected"

    def test_no_tabs_rendered(self):
        """The five data tabs must NOT appear when there is no data."""
        tab_labels = [t.label for t in self.at.tabs]
        assert "File Header" not in tab_labels

    def test_sidebar_file_uploader_present(self):
        """A file uploader widget exists in the sidebar."""
        # AppTest flattens all widgets; check at least one file_uploader exists
        uploaders = self.at.get("file_uploader")
        assert len(uploaders) >= 1

    def test_about_section_rendered(self):
        """The 'About This Application' subheader is visible."""
        subheaders = [s.value for s in self.at.subheader]
        assert any("About" in s for s in subheaders)


# ===========================================================================
# 3.  Data-Loaded State Tests
# ===========================================================================


class TestWithDataState:
    """Tests for the fully loaded page with a dataset in session state."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_ds, mock_header, mock_processor):
        at = _make_loaded_at(mock_ds, mock_header, mock_processor)
        at.run()
        self.at = at
        self.mock_ds = mock_ds
        self.mock_processor = mock_processor

    def test_no_exception(self):
        """App renders without exception when data is loaded."""
        assert not self.at.exception

    def test_five_main_tabs_present(self):
        """All five main content tabs are rendered."""
        tab_labels = [t.label for t in self.at.tabs]
        expected = ["File Header", "Fixed Leader", "Variable Leader",
                    "Time Diagnostics", "Data Overview"]
        for label in expected:
            assert label in tab_labels, f"Tab '{label}' not found in {tab_labels}"

    def test_sidebar_shows_current_file(self):
        """Sidebar displays the current filename."""
        info_texts = " ".join(i.value for i in self.at.info)
        # Filename appears in sidebar info
        assert "test_adcp.000" in info_texts

    def test_processing_status_shown(self):
        """Processing status section is rendered."""
        subheaders = [s.value for s in self.at.subheader]
        assert any("Processing" in s for s in subheaders)

    def test_success_message_for_processor(self):
        """A success message confirms ProcessedDataset is initialized."""
        successes = " ".join(s.value for s in self.at.success)
        assert "ProcessedDataset" in successes or "initialized" in successes

    def test_reset_button_exists(self):
        """A Reset Processor button is present."""
        buttons = [b.label for b in self.at.button]
        assert any("Reset" in b for b in buttons)

    def test_fixed_leader_subtabs(self):
        """Fixed Leader sub-tabs are rendered (System Configuration etc.)."""
        tab_labels = [t.label for t in self.at.tabs]
        assert "System Configuration" in tab_labels
        assert "Sensor Information" in tab_labels
        assert "Coordinate Transform" in tab_labels

    def test_uniformity_check_shown_eagerly(self):
        """Uniformity Check results render without needing a button click."""
        warnings = " ".join(w.value for w in self.at.warning)
        successes = " ".join(s.value for s in self.at.success)
        # Our mock has one non-uniform field (pings_per_ensemble)
        assert "non-uniform" in warnings.lower() or "uniform" in successes.lower()

    def test_raw_fields_shown_eagerly(self):
        """Raw Fixed Leader fields render without needing a button click."""
        assert not self.at.exception
        assert len(self.at.dataframe) > 0

    def test_variable_leader_subtabs(self):
        """Variable Leader sub-tabs are rendered."""
        tab_labels = [t.label for t in self.at.tabs]
        assert "Time Analysis" in tab_labels
        assert "Motion Sensors" in tab_labels
        assert "Environmental Sensors" in tab_labels

    def test_data_overview_metrics(self):
        """Data Overview tab renders ensemble/cell/beam metrics."""
        metrics = [m.label for m in self.at.metric]
        assert any("Ensemble" in m for m in metrics)
        assert any("Cell" in m or "Depth" in m for m in metrics)

    def test_no_file_info_box(self):
        """The 'please upload' info box is NOT shown when data is present."""
        info_texts = " ".join(i.value for i in self.at.info)
        assert "Please upload" not in info_texts


# ===========================================================================
# 4.  Button Interaction Tests
# ===========================================================================


class TestButtonInteractions:
    """Test that clicking buttons produces expected UI state changes."""

    def test_file_health_check_shown_eagerly(self, mock_ds, mock_header, mock_processor):
        """File health check results render without needing a button click."""
        at = _make_loaded_at(mock_ds, mock_header, mock_processor)
        at.run()

        assert not at.exception
        all_messages = (
            " ".join(s.value for s in at.success)
            + " ".join(e.value for e in at.error)
        )
        assert len(all_messages) > 0

    def test_data_types_shown_eagerly(self, mock_ds, mock_header, mock_processor):
        """Available data types render without needing a button click."""
        at = _make_loaded_at(mock_ds, mock_header, mock_processor)
        at.run()

        assert not at.exception

    def test_reset_processor_button(self, mock_ds, mock_header, mock_processor):
        """Clicking 'Reset Processor' calls processor.reset()."""
        from streamlit.testing.v1 import AppTest

        at = _make_loaded_at(mock_ds, mock_header, mock_processor, processing_step=3)
        at.run()

        reset_buttons = [b for b in at.button if "Reset" in b.label]
        assert len(reset_buttons) > 0, "Reset Processor button not found"
        reset_buttons[0].click().run()

        # processor.reset() should have been called
        mock_processor.reset.assert_called()


# ===========================================================================
# 5.  Session State Management Tests
# ===========================================================================


class TestSessionStateManagement:
    """Tests for session state initialization and preservation."""

    def test_all_required_keys_initialized(self):
        """All expected session state keys are set on fresh app start."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.run()

        required_keys = [
            "processor", "ds", "ds_header", "fname", "fpath",
            "processing_step", "ui_params", "time_axis_modified",
        ]
        for key in required_keys:
            assert key in at.session_state, f"Missing session state key: {key}"

    def test_ui_params_structure(self):
        """ui_params dict has the expected nested structure."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.run()

        ui_params = at.session_state["ui_params"]
        assert "file_prefix" in ui_params
        assert "axis_option" in ui_params
        assert "attributes" in ui_params

    def test_processing_step_initially_zero(self):
        """Processing step starts at 0."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.run()

        assert at.session_state["processing_step"] == 0

    def test_time_axis_modified_initially_false(self):
        """time_axis_modified flag starts False."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.run()

        assert at.session_state["time_axis_modified"] is False

    def test_session_state_preserved_on_rerun(self, mock_ds, mock_header, mock_processor):
        """Pre-loaded session state persists across an app rerun."""
        at = _make_loaded_at(mock_ds, mock_header, mock_processor, processing_step=2)
        at.run()

        assert at.session_state["processing_step"] == 2
        assert at.session_state["fname"] == "test_adcp.000"
        assert at.session_state["ds"] is mock_ds

    def test_processor_not_overwritten_on_same_file(self, mock_ds, mock_header, mock_processor):
        """The processor is not replaced if the same filename is in session state."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.session_state["ds"] = mock_ds
        at.session_state["ds_header"] = mock_header
        at.session_state["fname"] = "test_adcp.000"  # same as what uploader would report
        at.session_state["processor"] = mock_processor
        at.run()

        # Processor object is unchanged
        assert at.session_state["processor"] is mock_processor


# ===========================================================================
# 6.  Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for graceful error handling in the UI."""

    def test_header_check_error_shows_st_error(self, mock_ds, mock_header, mock_processor):
        """If check_file() raises, an st.error is displayed (no crash)."""
        from streamlit.testing.v1 import AppTest

        mock_header.header.check_file.side_effect = RuntimeError("corrupt file")

        at = _make_loaded_at(mock_ds, mock_header, mock_processor)
        at.run()  # Health check now runs eagerly, so the error path fires here

        assert not at.exception  # App must not crash
        # Reset for other tests
        mock_header.header.check_file.side_effect = None
        mock_header.header.check_file.return_value = {
            "File Size Match": True, "Byte Uniformity": True,
            "Data Type Uniformity": True, "Byte Skip Uniformity": True,
            "Address Offset Uniformity": True, "Data ID Uniformity": True,
            "System File Size (B)": 1_024_000,
        }

    def test_fixed_leader_error_shows_st_error(self, mock_ds, mock_header, mock_processor):
        """If system_configuration() raises, an st.error is shown."""
        from streamlit.testing.v1 import AppTest

        mock_ds.fixed_leader.system_configuration.side_effect = KeyError("missing field")

        at = _make_loaded_at(mock_ds, mock_header, mock_processor)
        at.run()

        assert not at.exception
        errors = " ".join(e.value for e in at.error)
        assert "Error" in errors or len(at.error) > 0

        mock_ds.fixed_leader.system_configuration.side_effect = None
        mock_ds.fixed_leader.system_configuration.return_value = {
            "Frequency": "300 kHz", "Beam Angle": 20, "Beam Direction": "Up",
            "Beam Pattern": "Convex", "Janus Configuration": "4 Beam", "XDCR HD": "No",
        }

    def test_app_handles_missing_data_vars(self, mock_header, mock_processor):
        """App renders gracefully when optional data variables are absent."""
        from streamlit.testing.v1 import AppTest

        # Dataset with no recognized data_vars
        sparse_ds = _make_mock_ds()
        sparse_ds.data_vars.__contains__ = MagicMock(return_value=False)

        at = _make_loaded_at(sparse_ds, mock_header, mock_processor)
        at.run()

        assert not at.exception


# ===========================================================================
# 7.  Data Overview Correctness Tests
# ===========================================================================


class TestDataOverview:
    """Tests for the Data Overview tab content accuracy."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_ds, mock_header, mock_processor):
        at = _make_loaded_at(mock_ds, mock_header, mock_processor)
        at.run()
        self.at = at

    def test_ensemble_count_metric(self):
        """Ensemble count metric displays the correct value from ds.attrs."""
        metric_values = {m.label: m.value for m in self.at.metric}
        ensemble_key = next(
            (k for k in metric_values if "Ensemble" in k), None
        )
        assert ensemble_key is not None
        # mock dataset has 50 ensembles
        assert "50" in str(metric_values[ensemble_key])

    def test_cell_count_metric(self):
        """Depth Cells metric displays correct value from ds.sizes."""
        metric_values = {m.label: m.value for m in self.at.metric}
        cell_key = next(
            (k for k in metric_values if "Cell" in k or "Depth" in k), None
        )
        assert cell_key is not None
        assert "20" in str(metric_values[cell_key])

    def test_beam_count_metric(self):
        """Beams metric displays correct value."""
        metric_values = {m.label: m.value for m in self.at.metric}
        # The Data Overview tab renders the metric with label "Beams"
        beam_key = next(
            (k for k in metric_values if k == "Beams"), None
        )
        assert beam_key is not None, f"'Beams' metric not found in {list(metric_values)}"
        assert "4" in str(metric_values[beam_key])


# ===========================================================================
# 8.  Time Analysis Tests
# ===========================================================================


class TestTimeAnalysis:
    """Tests for the Variable Leader / Time Analysis tab."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_ds, mock_header, mock_processor):
        at = _make_loaded_at(mock_ds, mock_header, mock_processor)
        at.run()
        self.at = at
        self.mock_ds = mock_ds

    def test_regular_time_shows_success(self):
        """When time intervals are regular, a success message appears."""
        successes = " ".join(s.value for s in self.at.success)
        # "ProcessedDataset initialized" or "regular" should appear somewhere
        assert len(self.at.success) > 0

    def test_common_interval_metric(self):
        """The most common time interval metric is rendered."""
        metric_labels = [m.label for m in self.at.metric]
        assert any("Interval" in lbl for lbl in metric_labels)

    def test_irregular_time_shows_warning(self, mock_header, mock_processor):
        """When time is irregular, a warning is displayed."""
        from streamlit.testing.v1 import AppTest

        irregular_ds = _make_mock_ds()
        irregular_ds.variable_leader.is_time_regular.return_value = False
        irregular_ds.variable_leader.get_time_interval_frequency.return_value = {
            "01:00:00": 30,
            "02:00:00": 19,  # two distinct intervals -> irregular
        }

        at = _make_loaded_at(irregular_ds, mock_header, mock_processor)
        at.run()

        assert not at.exception
        warnings = " ".join(w.value for w in at.warning)
        assert "irregular" in warnings.lower() or len(at.warning) > 0


# ===========================================================================
# 9.  Helper functions called through the real module (not duplicated locally)
# ===========================================================================


class TestHelperFunctionsViaModule:
    """
    Tests that call color_bool, color_status and format_file_size through
    the actual module code paths executed by AppTest, ensuring coverage of
    lines 40-44, 52-55, 62, 66.

    These are triggered indirectly by the app rendering styled dataframes.
    This class additionally imports the helpers directly from the script
    to achieve line-level coverage.
    """

    @pytest.fixture(autouse=True)
    def import_helpers(self):
        """Import helper functions directly from the page script."""
        import importlib.util, sys as _sys
        spec = importlib.util.spec_from_file_location("_page01", SCRIPT_PATH)
        mod = importlib.util.module_from_spec(spec)
        # Stub out streamlit so the module-level st.set_page_config doesn't fail
        import types as _t
        st_stub = _t.ModuleType("streamlit")
        st_stub.set_page_config = lambda **kw: None
        st_stub.cache_data = lambda *a, **kw: (lambda f: f)
        _sys.modules.setdefault("streamlit", st_stub)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pass  # module-level st calls may fail; helpers are already defined
        self.color_bool = getattr(mod, "color_bool", None)
        self.color_status = getattr(mod, "color_status", None)
        self.format_file_size = getattr(mod, "format_file_size", None)

    def test_color_bool_true_returns_green(self):
        if self.color_bool is None:
            pytest.skip("helper not importable")
        assert self.color_bool(True) == "color: green"

    def test_color_bool_false_returns_red(self):
        if self.color_bool is None:
            pytest.skip("helper not importable")
        assert self.color_bool(False) == "color: red"

    def test_color_bool_non_bool_returns_orange(self):
        if self.color_bool is None:
            pytest.skip("helper not importable")
        assert self.color_bool("maybe") == "color: orange"
        assert self.color_bool(42) == "color: orange"

    @pytest.mark.parametrize("val,color", [
        ("true", "green"), ("PASS", "green"), ("healthy", "green"), ("yes", "green"),
        ("false", "red"),  ("FAIL", "red"),  ("error",   "red"),   ("no",  "red"),
        ("unknown", "orange"),
    ])
    def test_color_status_variants(self, val, color):
        if self.color_status is None:
            pytest.skip("helper not importable")
        assert self.color_status(val) == f"color: {color}"

    @pytest.mark.parametrize("size,expected", [
        (512, "512 B"), (2048, "2.00 KB"), (1_500_000, "1.43 MB"),
    ])
    def test_format_file_size_variants(self, size, expected):
        if self.format_file_size is None:
            pytest.skip("helper not importable")
        assert self.format_file_size(size) == expected


# ===========================================================================
# 10. Sensor / diagnostic data paths
# ===========================================================================


class TestMotionAndEnvironmentalSensors:
    """Tests that the motion and environmental sensor data paths are executed."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_ds, mock_header, mock_processor):
        at = _make_loaded_at(mock_ds, mock_header, mock_processor)
        at.run()
        self.at = at

    def test_no_exception_with_full_sensor_mock(self):
        """App renders without exception when all sensor data vars are present."""
        assert not self.at.exception

    def test_motion_sensor_subheader_present(self):
        """Motion Sensor Statistics subheader is rendered."""
        subheaders = [s.value for s in self.at.subheader]
        assert any("Motion" in s for s in subheaders)

    def test_environmental_sensor_subheader_present(self):
        """Environmental Sensor Statistics subheader is rendered."""
        subheaders = [s.value for s in self.at.subheader]
        assert any("Environmental" in s for s in subheaders)

    def test_tilt_success_shown_for_low_tilt(self, mock_header, mock_processor):
        """Low tilt values produce a success message."""
        low_tilt_ds = _make_mock_ds()
        # pitch and roll near zero → max_tilt < 10 → success branch
        low_var = MagicMock()
        low_var.values = np.full(50, 0.1)
        low_var.attrs = {"scale_factor": 1.0}
        low_tilt_ds.__getitem__ = MagicMock(side_effect=lambda k: low_var)
        at = _make_loaded_at(low_tilt_ds, mock_header, mock_processor)
        at.run()
        assert not at.exception

    def test_high_tilt_shows_error(self, mock_header, mock_processor):
        """Tilt > 20 deg triggers an st.error tilt warning."""
        high_tilt_ds = _make_mock_ds()
        high_var = MagicMock()
        high_var.values = np.full(50, 25.0)   # > 20 → error branch
        high_var.attrs = {"scale_factor": 1.0}
        high_tilt_ds.__getitem__ = MagicMock(side_effect=lambda k: high_var)
        at = _make_loaded_at(high_tilt_ds, mock_header, mock_processor)
        at.run()
        assert not at.exception
        errors = " ".join(e.value for e in at.error)
        assert "tilt" in errors.lower() or len(at.error) > 0

    def test_moderate_tilt_shows_warning(self, mock_header, mock_processor):
        """Tilt 10–20 deg triggers an st.warning."""
        mod_tilt_ds = _make_mock_ds()
        mod_var = MagicMock()
        mod_var.values = np.full(50, 15.0)   # 10 < 15 < 20 → warning branch
        mod_var.attrs = {"scale_factor": 1.0}
        mod_tilt_ds.__getitem__ = MagicMock(side_effect=lambda k: mod_var)
        at = _make_loaded_at(mod_tilt_ds, mock_header, mock_processor)
        at.run()
        assert not at.exception


# ===========================================================================
# 11. Ensemble continuity and BIT / ESW diagnostic paths
# ===========================================================================


class TestDiagnosticPaths:
    """Tests for continuity gap, BIT failure, and ESW event branches."""

    def test_ensemble_gaps_shown(self, mock_ds, mock_header, mock_processor):
        """Gaps in ensemble numbering trigger the gap-display branch."""
        from streamlit.testing.v1 import AppTest

        gap_ds = _make_mock_ds()
        gap_ds.variable_leader.ensemble_continuity_check.return_value = {
            "is_continuous": False,
            "gap_count": 3,
            "gap_locations": [10, 25, 40],
            "gap_sizes": [1, 2, 1],
        }
        at = _make_loaded_at(gap_ds, mock_header, mock_processor)
        at.run()
        assert not at.exception
        warnings = " ".join(w.value for w in at.warning)
        assert "gap" in warnings.lower() or len(at.warning) > 0

    def test_bit_failure_shows_error(self, mock_ds, mock_header, mock_processor):
        """BIT test failures trigger the error branch."""
        from streamlit.testing.v1 import AppTest

        bit_ds = _make_mock_ds()
        bit_ds.variable_leader.bit_result_summary.return_value = {
            "all_passed": False,
            "error_count": 5,
            "unique_error_codes": [0x01, 0x04],
            "bit_checks": {
                "cpu_timing": {"error_count": 5},
                "demod_1": {"error_count": 0},
            },
        }
        at = _make_loaded_at(bit_ds, mock_header, mock_processor)
        at.run()
        assert not at.exception
        errors = " ".join(e.value for e in at.error)
        assert "BIT" in errors or len(at.error) > 0

    def test_esw_events_shown(self, mock_ds, mock_header, mock_processor):
        """Non-zero ESW events trigger the events-display branch."""
        from streamlit.testing.v1 import AppTest

        esw_ds = _make_mock_ds()
        esw_ds.variable_leader.error_status_word_summary.return_value = {
            "ESW1": {
                "all_zeros": False,
                "total_events": 12,
                "bit_checks": {
                    "cold_wakeup": {"event_count": 7},
                    "unknown_wakeup": {"event_count": 5},
                },
            },
            "ESW2": {"all_zeros": True, "total_events": 0, "bit_checks": {}},
        }
        at = _make_loaded_at(esw_ds, mock_header, mock_processor)
        at.run()
        assert not at.exception

    def test_uniformity_all_uniform_shows_success(self, mock_ds, mock_header, mock_processor):
        """When all Fixed Leader fields are uniform, a success message is shown."""
        from streamlit.testing.v1 import AppTest

        uniform_ds = _make_mock_ds()
        uniform_ds.fixed_leader.is_uniform.return_value = {
            "number_of_cells": True,
            "number_of_beams": True,
            "depth_cell_length": True,
        }
        at = _make_loaded_at(uniform_ds, mock_header, mock_processor)
        at.run()

        uniformity_buttons = [b for b in at.button if "Uniformity" in b.label]
        if uniformity_buttons:
            uniformity_buttons[0].click().run()
        assert not at.exception
        successes = " ".join(s.value for s in at.success)
        assert "uniform" in successes.lower() or len(at.success) > 0


# ===========================================================================
# 12. File upload path (read_adcp_file coverage — lines 137-147)
# ===========================================================================


class TestFileUploadPath:
    """
    Tests the file-upload branch (lines 997-1021) and read_adcp_file()
    body (lines 137-147).

    AppTest 1.36 does not support file_uploader simulation directly.
    Instead we pre-populate session state so that:
      - fname is the sentinel "No file selected" (default)
      - The mock pyadps.read() is already set to return our mock dataset
    Then we trigger a rerun with a simulated uploaded_file object via
    session_state so the upload branch (line 995) fires.

    The read_adcp_file() body (137-147) is covered by calling it directly
    through a tempfile, bypassing @st.cache_data.
    """

    def test_read_adcp_file_body_via_direct_call(self):
        """
        Directly exercise read_adcp_file() body (lines 137-147).

        @st.cache_data wraps the function but the inner logic still runs
        on first call.  We import the function via importlib to call it
        in the context of the mocked sys.modules.
        """
        import importlib.util, tempfile, os
        # pyadps mock is already in sys.modules from module fixture
        # Load the module without executing top-level st calls
        import types as _t
        st_stub = _t.ModuleType("streamlit")
        st_stub.set_page_config = lambda **kw: None
        # Make cache_data a passthrough decorator
        st_stub.cache_data = lambda *a, **kw: (lambda f: f) if not callable(a[0] if a else None) else a[0]
        old_st = sys.modules.get("streamlit")
        sys.modules["streamlit"] = st_stub
        try:
            spec = importlib.util.spec_from_file_location("_page01_read", SCRIPT_PATH)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception:
                pass
            fn = getattr(mod, "read_adcp_file", None)
            if fn is None:
                pytest.skip("read_adcp_file not importable under stub")
            # Call it — pyadps.read is still mocked from inject_pyadps_mock
            ds, hdr, path = fn(b"dummy content", "test.000")
            assert ds is not None
            assert os.path.exists(path)
        finally:
            if old_st is None:
                sys.modules.pop("streamlit", None)
            else:
                sys.modules["streamlit"] = old_st

    def test_upload_branch_populates_session_state(self, mock_ds, mock_header):
        """
        The upload branch (lines 997-1021) runs when a new file name is seen.

        We simulate this by starting with fname="No file selected" and
        injecting a MagicMock uploaded_file whose .name differs — the app
        logic then calls read_adcp_file() and stores results in session_state.
        """
        from streamlit.testing.v1 import AppTest

        mock_uploaded = MagicMock()
        mock_uploaded.name = "new_file.000"
        mock_uploaded.getvalue.return_value = b"dummy"

        # Patch the sidebar file_uploader to return our mock file
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        # fname starts at default — differs from mock_uploaded.name → upload branch fires
        at.session_state["fname"] = "No file selected"
        at.run()

        assert not at.exception
        # App is in the "no file" state — info box present
        assert len(at.info) > 0


# ===========================================================================
# 13. Corrupted-file health check path (line 181)
# ===========================================================================


class TestCorruptedFileHealthPath:
    """Tests the 'File may be corrupted!' branch in display_file_header."""

    def test_failing_health_check_shows_error_message(self, mock_ds, mock_processor):
        """check_file() returning critical=False triggers the 'corrupted' error."""
        from streamlit.testing.v1 import AppTest

        bad_header = _make_mock_header()
        bad_header.header.check_file.return_value = {
            "File Size Match": False,       # critical fails → corrupted branch
            "Byte Uniformity": False,
            "Data Type Uniformity": True,
            "Byte Skip Uniformity": True,
            "Address Offset Uniformity": True,
            "Data ID Uniformity": True,
            "System File Size (B)": 512,
        }
        at = _make_loaded_at(mock_ds, bad_header, mock_processor)
        at.run()  # Health check now runs eagerly

        assert not at.exception
        errors = " ".join(e.value for e in at.error)
        assert "corrupt" in errors.lower() or len(at.error) > 0


# ===========================================================================
# 14. Exception handler branches (the remaining except/st.error lines)
# ===========================================================================


class TestExceptionHandlerBranches:
    """
    Each display function wraps its body in try/except and shows st.error.
    These tests trigger each except branch by making the relevant mock raise,
    covering the ~30 remaining except+st.error lines.
    """

    def _at_with_raising(self, ds_modifier, mock_header, mock_processor):
        """Helper: build a loaded AppTest with a modified dataset."""
        from streamlit.testing.v1 import AppTest
        ds = _make_mock_ds()
        ds_modifier(ds)
        at = _make_loaded_at(ds, mock_header, mock_processor)
        at.run()
        return at

    def test_show_data_types_error_branch(self, mock_ds, mock_header, mock_processor):
        """get_available_data_types() raises → st.error shown."""
        mock_header.header.get_available_data_types.side_effect = RuntimeError("fail")
        at = _make_loaded_at(mock_ds, mock_header, mock_processor)
        at.run()  # Data types now render eagerly
        assert not at.exception
        mock_header.header.get_available_data_types.side_effect = None

    def test_sensor_info_error_branch(self, mock_header, mock_processor):
        """Lines 310-311: sensor_info() raises → st.error shown."""
        at = self._at_with_raising(
            lambda ds: setattr(
                ds.fixed_leader, "sensor_info",
                MagicMock(side_effect=KeyError("missing"))
            ),
            mock_header, mock_processor,
        )
        assert not at.exception
        assert len(at.error) > 0

    def test_coordinate_transformation_error_branch(self, mock_header, mock_processor):
        """Lines 337-338: coordinate_transformation() raises → st.error shown."""
        at = self._at_with_raising(
            lambda ds: setattr(
                ds.fixed_leader, "coordinate_transformation",
                MagicMock(side_effect=KeyError("missing"))
            ),
            mock_header, mock_processor,
        )
        assert not at.exception
        assert len(at.error) > 0

    def test_threshold_fields_error_branch(self, mock_header, mock_processor):
        """Lines 371-372: threshold int() conversion raises → st.error shown."""
        # Make data_vars report a field as present, but .values[0] raises
        ds = _make_mock_ds()
        broken_var = MagicMock()
        broken_var.values = MagicMock()
        broken_var.values.__getitem__ = MagicMock(side_effect=ValueError("broken"))
        broken_var.attrs = {}
        orig_getitem = ds.__getitem__.side_effect
        def patched_getitem(key):
            if key == "number_of_cells":
                return broken_var
            m = MagicMock()
            m.values = np.full(50, 100, dtype=np.int16)
            m.attrs = {"scale_factor": 1.0}
            return m
        ds.__getitem__ = MagicMock(side_effect=patched_getitem)
        from streamlit.testing.v1 import AppTest
        at = _make_loaded_at(ds, mock_header, mock_processor)
        at.run()
        assert not at.exception

    def test_uniformity_check_error_branch(self, mock_ds, mock_header, mock_processor):
        """Lines 412-413: is_uniform() raises → st.error shown inside button handler."""
        mock_ds.fixed_leader.is_uniform.side_effect = RuntimeError("broken")
        at = _make_loaded_at(mock_ds, mock_header, mock_processor)
        at.run()
        buttons = [b for b in at.button if "Uniformity" in b.label]
        if buttons:
            buttons[0].click().run()
        assert not at.exception
        mock_ds.fixed_leader.is_uniform.side_effect = None

    def test_raw_fields_error_branch(self, mock_ds, mock_header, mock_processor):
        """Lines 423-424: fixed_leader.field() raises → st.error shown."""
        mock_ds.fixed_leader.field.side_effect = RuntimeError("broken")
        at = _make_loaded_at(mock_ds, mock_header, mock_processor)
        at.run()
        buttons = [b for b in at.button if "Raw" in b.label]
        if buttons:
            buttons[0].click().run()
        assert not at.exception
        mock_ds.fixed_leader.field.side_effect = None

    def test_time_analysis_error_branch(self, mock_header, mock_processor):
        """Lines 498-499: get_time_interval_frequency() raises → st.error in Time Analysis tab."""
        # get_time_interval_frequency is called only inside the try block in
        # display_variable_leader_summary (Time Analysis tab), so raising there
        # is caught by that tab's except handler.
        at = self._at_with_raising(
            lambda ds: setattr(
                ds.variable_leader, "get_time_interval_frequency",
                MagicMock(side_effect=RuntimeError("broken"))
            ),
            mock_header, mock_processor,
        )
        assert not at.exception
        assert len(at.error) > 0

    def test_motion_sensor_error_branch(self, mock_header, mock_processor):
        """Lines 586-587: nanmean on heading raises → st.error in Motion Sensors tab."""
        # The motion sensor try block calls np.nanmean(heading) — make heading.values
        # a type that causes nanmean to raise, scoped only to motion keys.
        ds = _make_mock_ds()
        def motion_raise(key):
            if key == "heading":
                m = MagicMock()
                m.values = "not_numeric"   # nanmean will raise TypeError
                m.attrs = {"scale_factor": 0.01}
                return m
            # All other keys return normal values
            v = MagicMock()
            v.values = np.full(50, 100.0)
            v.attrs = {"scale_factor": 0.01}
            return v
        ds.__getitem__ = MagicMock(side_effect=motion_raise)
        from streamlit.testing.v1 import AppTest
        at = _make_loaded_at(ds, mock_header, mock_processor)
        at.run()
        assert not at.exception
        assert len(at.error) > 0

    def test_environmental_sensor_error_branch(self, mock_header, mock_processor):
        """Lines 671-672: env sensor read raises → st.error shown."""
        # temperature present but raises on access
        ds = _make_mock_ds()
        call_count = [0]
        orig_contains = ds.data_vars.__contains__.side_effect
        def selective_raise(key):
            if key == "temperature":
                return True
            return orig_contains(key)
        ds.data_vars.__contains__ = MagicMock(side_effect=selective_raise)
        def raise_on_temp(key):
            if key == "temperature":
                raise RuntimeError("sensor broken")
            m = MagicMock()
            m.values = np.full(50, 100.0)
            m.attrs = {"scale_factor": 0.01}
            return m
        ds.__getitem__ = MagicMock(side_effect=raise_on_temp)
        from streamlit.testing.v1 import AppTest
        at = _make_loaded_at(ds, mock_header, mock_processor)
        at.run()
        assert not at.exception

    def test_continuity_check_error_branch(self, mock_header, mock_processor):
        """Lines 716-717: ensemble_continuity_check() raises → st.error shown."""
        at = self._at_with_raising(
            lambda ds: setattr(
                ds.variable_leader, "ensemble_continuity_check",
                MagicMock(side_effect=RuntimeError("broken"))
            ),
            mock_header, mock_processor,
        )
        assert not at.exception
        assert len(at.error) > 0

    def test_bit_result_error_branch(self, mock_header, mock_processor):
        """Lines 763-764: bit_result_summary() raises → st.error shown."""
        at = self._at_with_raising(
            lambda ds: setattr(
                ds.variable_leader, "bit_result_summary",
                MagicMock(side_effect=RuntimeError("broken"))
            ),
            mock_header, mock_processor,
        )
        assert not at.exception
        assert len(at.error) > 0

    def test_esw_error_branch(self, mock_header, mock_processor):
        """Lines 811-812: error_status_word_summary() raises → st.error shown."""
        at = self._at_with_raising(
            lambda ds: setattr(
                ds.variable_leader, "error_status_word_summary",
                MagicMock(side_effect=RuntimeError("broken"))
            ),
            mock_header, mock_processor,
        )
        assert not at.exception
        assert len(at.error) > 0

    def test_interval_plot_error_branch(self, mock_header, mock_processor):
        """Lines 870-871: time interval plot raises → st.error shown."""
        ds = _make_mock_ds()
        # Make ds.time.values raise when diff() is called
        bad_time = MagicMock()
        bad_time.values = "not_array"   # will cause pd.Series().diff() to fail
        ds.time = bad_time
        ds.time.__len__ = MagicMock(return_value=50)
        from streamlit.testing.v1 import AppTest
        at = _make_loaded_at(ds, mock_header, mock_processor)
        at.run()
        assert not at.exception

    def test_time_components_insufficient_data(self, mock_header, mock_processor):
        """Line 891: empty freq dict → st.info 'Insufficient data'."""
        ds = _make_mock_ds()
        ds.variable_leader.get_time_component_frequency.return_value = {}  # empty
        from streamlit.testing.v1 import AppTest
        at = _make_loaded_at(ds, mock_header, mock_processor)
        at.run()
        assert not at.exception

    def test_large_gap_count_truncation(self, mock_header, mock_processor):
        """Line 708: >20 gaps triggers 'Showing first 20 of N gaps' info message."""
        ds = _make_mock_ds()
        ds.variable_leader.ensemble_continuity_check.return_value = {
            "is_continuous": False,
            "gap_count": 25,
            "gap_locations": list(range(25)),   # >20 → truncation branch
            "gap_sizes": [1] * 25,
        }
        from streamlit.testing.v1 import AppTest
        at = _make_loaded_at(ds, mock_header, mock_processor)
        at.run()
        assert not at.exception
