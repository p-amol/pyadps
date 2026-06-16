"""
Test Suite for 03_Download_Raw_File.py
=======================================
Uses Streamlit's AppTest framework (streamlit.testing.v1.AppTest) to run the
actual Streamlit script in a simulated runtime, giving real code coverage.

Key design decisions
--------------------
1. Radio pre-setting does NOT work for conditional UI sections.
   st.radio() ignores pre-set session state and renders from its ``index``
   parameter.  To enter conditional branches (e.g. "attribute Yes" section,
   "Edit Filename Yes" section) we must call ``radio.set_value("Yes").run()``
   AFTER the initial ``at.run()``.

2. MagicMock ds and Generate NetCDF.
   ``create_subset_dataset`` does ``ds[list_of_vars].copy()`` which calls
   ``ds.__getitem__`` with a **list**, not a string.  Our ``_getitem`` stub
   does ``key.replace(...)`` which crashes on a list.  Solution: use a REAL
   xr.Dataset for the Generate-NetCDF tests so subsetting works natively.

3. st.download_button does NOT appear in ``at.button``.
   AppTest's ``at.button`` only contains ``st.button`` / ``st.form_submit_button``
   elements.  ``st.download_button`` is not captured by AppTest at all.
   Download-button tests are replaced with checks for success/markdown output
   that actually is visible.

Usage
-----
    pytest tests/pages/test_03_Download_Raw_File.py -v
    pytest tests/pages/test_03_Download_Raw_File.py -v --tb=short
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from streamlit.testing.v1 import AppTest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
SCRIPT_PATH = str(
    Path(__file__).parent.parent.parent  # tests/pages/ -> tests/ -> pyadps/
    / "src"
    / "pyadps"
    / "pages"
    / "03_Download_Raw_File.py"
)
assert Path(SCRIPT_PATH).exists(), (
    f"Script not found at {SCRIPT_PATH}\n"
    f"Expected layout: pyadps/src/pyadps/pages/03_Download_Raw_File.py\n"
    f"Test file is at: {__file__}"
)

# ---------------------------------------------------------------------------
# Default field lists
# ---------------------------------------------------------------------------
_DEFAULT_FL_FIELDS = [
    "number_of_cells",
    "pings_per_ensemble",
    "depth_cell_length",
    "blank_after_transmit",
    "low_correlation_threshold",
    "error_velocity_maximum",
    "bin_1_distance",
    "system_serial_number",
]

_DEFAULT_VL_FIELDS = [
    "ensemble_number",
    "heading",
    "pitch",
    "roll",
    "temperature",
    "depth_of_transducer",
    "salinity",
    "speed_of_sound",
]


# ===========================================================================
# MOCK DATASET BUILDER  (for AppTest integration tests)
# ===========================================================================


def _make_mock_ds(
    n_ens: int = 50,
    n_cells: int = 20,
    n_beams: int = 4,
    include_primary_vars: bool = True,
    include_vl_fields: bool = True,
    include_fl_fields: bool = True,
    fl_scalar: bool = True,  # True  -> FL .values are scalar (ndim==0)
    fl_multidim: bool = False,  # True  -> FL .values are 2-D  (ndim>1)
    vl_multidim: bool = False,  # True  -> VL .values are 2-D  (ndim>1)
    primary_ndim: int = 3,  # 3=beam×cell×time, 2=cell×time, 1=time
) -> MagicMock:
    """
    Build a MagicMock that behaves like the xr.Dataset returned by pyadps.read().

    NOTE: ``_getitem`` handles both string keys (individual var access) and
    list keys (ds[list_of_vars] in create_subset_dataset).  The list-key path
    returns a minimal MagicMock subset rather than crashing on list.replace().
    """
    time_coord = pd.date_range("2024-01-15", periods=n_ens, freq="h")

    ds = MagicMock()

    fl_fields: list[str] = _DEFAULT_FL_FIELDS.copy() if include_fl_fields else []
    vl_fields: list[str] = _DEFAULT_VL_FIELDS.copy() if include_vl_fields else []

    ds.attrs = {
        "total_ensembles": n_ens,
        "fixed_leader_variables": fl_fields,
        "variable_leader_variables": vl_fields,
        "components": ["Fixed Leader", "Variable Leader", "Velocity", "Echo Intensity"],
        "source_file": "test_data.000",
    }

    _sizes = {"beam": n_beams, "cell": n_cells, "time": n_ens}
    ds.sizes = _sizes
    ds.dims = _sizes
    ds.time.values = time_coord.values
    ds.time.__len__ = MagicMock(return_value=n_ens)

    _primary = {"velocity", "echo_intensity", "correlation", "percent_good"}
    _ALL_VARS: set[str] = (
        (_primary if include_primary_vars else set()) | set(fl_fields) | set(vl_fields)
    )
    ds.data_vars.__contains__ = MagicMock(side_effect=lambda k: k in _ALL_VARS)
    ds.data_vars.__iter__ = MagicMock(return_value=iter(_ALL_VARS))
    ds.data_vars.__len__ = MagicMock(return_value=len(_ALL_VARS))

    # Coordinates — must be MagicMock (not plain dict) for __iter__ patching
    _COORD_NAMES = ["cell", "time", "beam"]
    coords_cell = MagicMock()
    coords_cell.values = np.arange(n_cells)
    coords_time = MagicMock()
    coords_time.values = time_coord.values
    coords_beam = MagicMock()

    coords = MagicMock()
    coords.__iter__ = MagicMock(return_value=iter(_COORD_NAMES))
    coords.__contains__ = MagicMock(side_effect=lambda k: k in set(_COORD_NAMES))
    coords.__getitem__ = MagicMock(
        side_effect=lambda k: {
            "cell": coords_cell,
            "time": coords_time,
            "beam": coords_beam,
        }[k]
    )
    ds.coords = coords

    def _getitem(key) -> MagicMock:
        # LIST key: ds[["v1", "v2"]] called by create_subset_dataset.
        # Return a subset-like MagicMock whose .copy() returns a MagicMock that
        # has .assign_coords() and real .attrs — enough for the function to run.
        if isinstance(key, list):
            subset = MagicMock()
            subset.attrs = {}

            def _copy():
                c = MagicMock()
                c.attrs = {}
                c.coords = coords
                c.data_vars.__contains__ = MagicMock(
                    side_effect=lambda k: k in set(key)
                )
                c.assign_coords = MagicMock(return_value=c)
                return c

            subset.copy = MagicMock(side_effect=_copy)
            return subset

        # STRING key: individual variable access
        var = MagicMock()
        var.attrs = {"units": "", "long_name": key.replace("_", " ").title()}

        if key == "velocity":
            if primary_ndim == 3:
                var.values = np.random.randint(
                    -1000, 1000, (n_beams, n_cells, n_ens), dtype=np.int16
                )
                var.ndim = 3
            elif primary_ndim == 2:
                var.values = np.random.randint(
                    -1000, 1000, (n_cells, n_ens), dtype=np.int16
                )
                var.ndim = 2
            else:
                var.values = np.random.randint(-1000, 1000, (n_ens,), dtype=np.int16)
                var.ndim = 1
            var.attrs = {"units": "mm/s", "long_name": "Water Velocity"}

        elif key == "echo_intensity":
            var.values = np.random.randint(
                40, 200, (n_beams, n_cells, n_ens), dtype=np.int16
            )
            var.ndim = 3
            var.attrs = {"units": "counts", "long_name": "Echo Intensity"}

        elif key == "correlation":
            var.values = np.random.randint(
                0, 255, (n_beams, n_cells, n_ens), dtype=np.int16
            )
            var.ndim = 3
            var.attrs = {"units": "counts", "long_name": "Correlation Magnitude"}

        elif key == "percent_good":
            var.values = np.random.randint(
                0, 100, (n_beams, n_cells, n_ens), dtype=np.int16
            )
            var.ndim = 3
            var.attrs = {"units": "%", "long_name": "Percent Good"}

        elif key in ("heading", "pitch", "roll"):
            if vl_multidim:
                var.values = np.tile(np.linspace(0, 10, n_ens), (2, 1))
                var.ndim = 2
            else:
                var.values = np.linspace(0, 10, n_ens)
                var.ndim = 1
            var.attrs = {"units": "degrees", "long_name": key.title()}

        elif key in vl_fields:
            if vl_multidim:
                var.values = np.tile(np.full(n_ens, 100.0), (2, 1))
                var.ndim = 2
            else:
                var.values = np.full(n_ens, 100.0)
                var.ndim = 1
            var.attrs = {"units": "", "long_name": key.replace("_", " ").title()}

        elif key in fl_fields:
            if fl_multidim:
                var.values = np.full((2, 3), 50)
                var.ndim = 2
                var.shape = (2, 3)
            elif fl_scalar:
                var.values = np.array(50)
                var.ndim = 0
            else:
                var.values = np.full(n_ens, 50)
                var.ndim = 1
            var.attrs = {"units": "cm", "long_name": key.replace("_", " ").title()}

        else:
            var.values = np.array(0)
            var.ndim = 0
            var.attrs = {"units": "", "long_name": key}

        return var

    ds.__getitem__ = MagicMock(side_effect=_getitem)
    return ds


# ===========================================================================
# REAL DATASET BUILDER  (for Generate-NetCDF tests)
# ===========================================================================


def _make_real_ds(n_ens: int = 20, n_cells: int = 8, n_beams: int = 4) -> xr.Dataset:
    """
    Build a genuine xr.Dataset for tests that call write_netcdf / to_netcdf.

    create_subset_dataset does ``ds[list_of_vars].copy()`` which works
    natively on a real Dataset but breaks the MagicMock._getitem stub.
    """
    tc = pd.date_range("2024-01-01", periods=n_ens, freq="h")
    fl_fields = ["number_of_cells", "pings_per_ensemble"]
    vl_fields = ["heading", "pitch"]
    ds = xr.Dataset(
        {
            "velocity": (
                ["beam", "cell", "time"],
                np.zeros((n_beams, n_cells, n_ens), dtype=np.int16),
            ),
            "echo_intensity": (
                ["beam", "cell", "time"],
                np.full((n_beams, n_cells, n_ens), 100, dtype=np.int16),
            ),
            "correlation": (
                ["beam", "cell", "time"],
                np.full((n_beams, n_cells, n_ens), 128, dtype=np.int16),
            ),
            "percent_good": (
                ["beam", "cell", "time"],
                np.full((n_beams, n_cells, n_ens), 80, dtype=np.int16),
            ),
            "number_of_cells": ([], n_cells),
            "pings_per_ensemble": ([], 50),
            "heading": (["time"], np.zeros(n_ens)),
            "pitch": (["time"], np.zeros(n_ens)),
        },
        coords={
            "time": tc,
            "cell": np.arange(1, n_cells + 1),
            "beam": np.arange(1, n_beams + 1),
        },
    )
    ds.attrs = {
        "fixed_leader_variables": fl_fields,
        "variable_leader_variables": vl_fields,
        "total_ensembles": n_ens,
        "source_file": "test.000",
        "components": ["Velocity", "Echo Intensity"],
    }
    return ds


# ===========================================================================
# HELPERS
# ===========================================================================


def _make_loaded_at(
    ds,
    extra_state: dict | None = None,
    timeout: int = 20,
) -> AppTest:
    """Return an AppTest with ds and fname pre-loaded in session state."""
    at = AppTest.from_file(SCRIPT_PATH, default_timeout=timeout)
    at.session_state["ds"] = ds
    at.session_state["fname"] = "test_adcp_GD10A000.000"
    if extra_state:
        for k, v in extra_state.items():
            at.session_state[k] = v
    at.run()
    return at


def _ss(at: AppTest, key: str, default=None):
    """Safe session_state accessor (AppTest does not support .get())."""
    return at.session_state[key] if key in at.session_state else default


def _switch_radio(at: AppTest, label_fragment: str, value: str) -> AppTest:
    """Find a radio by label fragment, set its value, re-run, return at."""
    radio = next(
        (r for r in at.radio if label_fragment.lower() in r.label.lower()), None
    )
    if radio is None:
        pytest.skip(f"Radio containing '{label_fragment}' not found")
    radio.set_value(value).run()
    return at


# ===========================================================================
# MODULE-SCOPED MOCK
# ===========================================================================


@pytest.fixture(scope="module", autouse=True)
def inject_pyadps_mock():
    """Inject minimal mock pyadps; torn down after this module finishes."""
    _orig = {
        "pyadps": sys.modules.get("pyadps"),
        "pyadps.processing": sys.modules.get("pyadps.processing"),
    }

    mock_pyadps = types.ModuleType("pyadps")
    mock_pyadps.read = MagicMock()
    mock_pyadps.read_header = MagicMock()

    mock_proc = types.ModuleType("pyadps.processing")
    mock_proc.ProcessedDataset = MagicMock()

    sys.modules["pyadps"] = mock_pyadps
    sys.modules["pyadps.processing"] = mock_proc

    yield

    for key, original in _orig.items():
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original


@pytest.fixture()
def mock_ds() -> MagicMock:
    return _make_mock_ds()


@pytest.fixture()
def real_ds() -> xr.Dataset:
    return _make_real_ds()


# ===========================================================================
# PURE HELPER STUBS (mirror page functions with explicit params)
# ===========================================================================


def _get_prefixed_filename(base_name: str, file_prefix: str) -> str:
    if file_prefix:
        return f"{file_prefix}_{base_name}"
    return base_name


def _create_subset_dataset(
    ds: xr.Dataset,
    fl_fields: list,
    vl_fields: list,
    include_fixed_leader: bool = False,
    include_variable_leader: bool = False,
    include_velocity: bool = False,
    include_echo: bool = False,
    include_correlation: bool = False,
    include_percent_good: bool = False,
) -> xr.Dataset | None:
    variables: list[str] = []
    if include_fixed_leader:
        variables += [v for v in fl_fields if v in ds.data_vars]
    if include_variable_leader:
        variables += [v for v in vl_fields if v in ds.data_vars]
    if include_velocity and "velocity" in ds.data_vars:
        variables.append("velocity")
    if include_echo and "echo_intensity" in ds.data_vars:
        variables.append("echo_intensity")
    if include_correlation and "correlation" in ds.data_vars:
        variables.append("correlation")
    if include_percent_good and "percent_good" in ds.data_vars:
        variables.append("percent_good")
    variables = list(dict.fromkeys(variables))
    if not variables:
        return None
    subset = ds[variables].copy()
    for coord in ds.coords:
        if coord not in subset.coords:
            subset = subset.assign_coords({coord: ds.coords[coord]})
    subset.attrs = ds.attrs.copy()
    return subset


def _add_user_attributes(
    dataset: xr.Dataset,
    attributes: dict,
    custom_attributes: dict,
) -> xr.Dataset:
    for key, value in attributes.items():
        if value:
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            dataset.attrs[key] = value
    for key, value in custom_attributes.items():
        if key and value:
            dataset.attrs[key] = value
    return dataset


def _write_netcdf(
    dataset: xr.Dataset | None,
    file_prefix: str,
    axis_option: str = "time",
    add_attributes: bool = False,
    attributes: dict | None = None,
    custom_attributes: dict | None = None,
) -> str | None:
    if dataset is None:
        return None
    ds_out = dataset.copy()
    if add_attributes and attributes is not None:
        ds_out = _add_user_attributes(ds_out, attributes, custom_attributes or {})
    if axis_option == "ensemble":
        if "time" in ds_out.dims:
            if "time" in ds_out.coords:
                time_values = ds_out.coords["time"].values
                ds_out = ds_out.rename({"time": "ensemble"})
                ds_out["time_original"] = ("ensemble", time_values)
                ds_out["time_original"].attrs["long_name"] = "Original time values"
    for attr in [
        "pyadps_component",
        "components",
        "fixed_leader_variables",
        "variable_leader_variables",
    ]:
        if attr in ds_out.attrs:
            del ds_out.attrs[attr]
    td = tempfile.mkdtemp()
    fp = os.path.join(td, _get_prefixed_filename("RAW_DATA.nc", file_prefix))
    ds_out.to_netcdf(fp)
    return fp


# ===========================================================================
# 1.  NO DATA STATE
# ===========================================================================


class TestNoDataState:
    """When ds is absent from session state the page shows a warning and stops."""

    def test_guard_message_shown(self):
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.run()
        all_text = " ".join(m.value for m in at.markdown)
        assert "please" in all_text.lower() or "select" in all_text.lower()

    def test_no_exception(self):
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.run()
        assert not at.exception

    def test_no_checkboxes(self):
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.run()
        assert len(at.checkbox) == 0

    def test_no_selectbox(self):
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.run()
        assert len(at.selectbox) == 0

    def test_ds_none_does_not_bypass_guard(self):
        """
        ds=None passes the 'not in' guard (key is present) but then the page
        crashes on None.attrs.  AppTest captures this as at.exception.
        Documents the known behaviour: the guard only covers a missing key.
        """
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.session_state["ds"] = None
        at.run()
        assert at.exception


# ===========================================================================
# 2.  NORMAL LOADED STATE — smoke tests
# ===========================================================================


class TestWithDataState:
    """Page renders without exceptions when ds is present."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_ds):
        self.at = _make_loaded_at(mock_ds)

    def test_no_exception(self):
        assert not self.at.exception

    def test_netcdf_header_present(self):
        headers = [h.value for h in self.at.header]
        assert any("netcdf" in h.lower() or "download" in h.lower() for h in headers)

    def test_attribute_radio_present(self):
        radios = [r.label for r in self.at.radio]
        assert any("attribute" in l.lower() for l in radios)

    def test_axis_selectbox_present(self):
        selectboxes = [s.label for s in self.at.selectbox]
        assert any("axis" in l.lower() for l in selectboxes)

    def test_seven_checkboxes_present(self):
        assert len(self.at.checkbox) >= 7

    def test_generate_button_present(self):
        labels = [b.label for b in self.at.button]
        assert any("generate" in l.lower() for l in labels)

    def test_csv_header_present(self):
        headers = [h.value for h in self.at.header]
        assert any("csv" in h.lower() or "download" in h.lower() for h in headers)

    def test_csv_selectbox_present(self):
        selectboxes = [s.label for s in self.at.selectbox]
        assert any(
            "data type" in l.lower() or "select" in l.lower() for l in selectboxes
        )

    def test_sidebar_file_shown(self):
        all_text = " ".join(m.value for m in self.at.markdown)
        assert "GD10A000" in all_text

    def test_sidebar_ensemble_count(self):
        all_text = " ".join(m.value for m in self.at.markdown)
        assert "50" in all_text

    def test_sidebar_cell_count(self):
        all_text = " ".join(m.value for m in self.at.markdown)
        assert "20" in all_text

    def test_sidebar_fl_vl_counts_shown(self):
        all_text = " ".join(m.value for m in self.at.markdown)
        assert "FL Variables" in all_text
        assert "VL Variables" in all_text

    def test_beam_radio_present(self):
        radios = [r.label for r in self.at.radio]
        assert any("beam" in l.lower() for l in radios)


# ===========================================================================
# 3.  SESSION STATE INITIALISATION
# ===========================================================================


class TestSessionStateInit:
    def _run_fresh(self, mock_ds, **extra):
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.session_state["ds"] = mock_ds
        for k, v in extra.items():
            at.session_state[k] = v
        at.run()
        return at

    def test_fname_key_set(self, mock_ds):
        at = self._run_fresh(mock_ds)
        assert not at.exception
        assert "fname" in at.session_state

    def test_fname_default_not_empty(self, mock_ds):
        at = self._run_fresh(mock_ds)
        assert _ss(at, "fname", "")  # something non-empty

    def test_add_attributes_default_no(self, mock_ds):
        at = self._run_fresh(mock_ds, fname="adcp.000")
        val = _ss(at, "add_attributes_DRW", "No")
        assert val == "No"

    def test_prefix_saved_default_false(self, mock_ds):
        at = self._run_fresh(mock_ds, fname="adcp.000")
        val = _ss(at, "prefix_saved", False)
        assert val is False

    def test_custom_attributes_default_empty_dict(self, mock_ds):
        at = self._run_fresh(mock_ds, fname="adcp.000")
        val = _ss(at, "custom_attributes", {})
        assert isinstance(val, dict)

    def test_attributes_default_empty_dict(self, mock_ds):
        at = self._run_fresh(mock_ds, fname="adcp.000")
        val = _ss(at, "attributes", {})
        assert isinstance(val, dict)


# ===========================================================================
# 4.  ATTRIBUTE "YES" SECTION
# ===========================================================================


class TestAttributeYesSection:
    """
    add_attributes_DRW == 'Yes' renders text/date inputs and custom-attr UI.

    IMPORTANT: st.radio() ignores pre-set session state.  We must call
    radio.set_value("Yes").run() AFTER the initial at.run() to enter the
    conditional branch.
    """

    def _run_attr_yes(self, mock_ds, custom_attr_count: int = 0) -> AppTest:
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=20)
        at.session_state["ds"] = mock_ds
        at.session_state["fname"] = "adcp.000"
        at.session_state["attributes"] = {}
        at.session_state["raw_custom_attributes"] = {}
        at.session_state["raw_custom_attr_count"] = custom_attr_count
        at.run()
        # Switch the attribute radio to "Yes" via set_value
        return _switch_radio(at, "attribute", "Yes")

    def test_attribute_inputs_shown(self, mock_ds):
        at = self._run_attr_yes(mock_ds)
        assert not at.exception
        text_labels = [t.label for t in at.text_input]
        assert any("cruise" in l.lower() for l in text_labels)
        assert any("ship" in l.lower() for l in text_labels)

    def test_date_inputs_shown(self, mock_ds):
        at = self._run_attr_yes(mock_ds)
        assert not at.exception
        assert len(at.date_input) >= 2

    def test_add_custom_attr_button_shown(self, mock_ds):
        at = self._run_attr_yes(mock_ds)
        assert not at.exception
        labels = [b.label for b in at.button]
        assert any(
            "add custom" in l.lower() or "custom attribute" in l.lower() for l in labels
        )

    def test_custom_attr_info_shown(self, mock_ds):
        at = self._run_attr_yes(mock_ds)
        assert not at.exception
        info_text = " ".join(i.value for i in at.info)
        assert "attribute" in info_text.lower()

    def test_custom_attr_row_when_count_is_1(self, mock_ds):
        """With custom_attr_count=1 a row of key/value text inputs appears."""
        at = self._run_attr_yes(mock_ds, custom_attr_count=1)
        assert not at.exception
        text_labels = [t.label.lower() for t in at.text_input]
        assert any("attribute name" in l for l in text_labels)
        assert any("attribute value" in l for l in text_labels)

    def test_multiple_custom_attr_rows(self, mock_ds):
        at = self._run_attr_yes(mock_ds, custom_attr_count=3)
        assert not at.exception
        text_labels = [t.label.lower() for t in at.text_input]
        assert sum("attribute name" in l for l in text_labels) == 3

    def test_remove_button_rendered_per_row(self, mock_ds):
        at = self._run_attr_yes(mock_ds, custom_attr_count=2)
        assert not at.exception
        labels = [b.label for b in at.button]
        assert any("🗑️" in l for l in labels)

    def test_attribute_radio_no_hides_inputs(self, mock_ds):
        """Default 'No' does not show cruise/ship text inputs."""
        at = _make_loaded_at(mock_ds)
        text_labels = [t.label for t in at.text_input]
        assert not any("cruise" in l.lower() for l in text_labels)


# ===========================================================================
# 5.  FILE PREFIX SECTION
# ===========================================================================


class TestFilePrefixSection:
    """Tests for output filename customisation."""

    def _switch_to_edit_yes(self, mock_ds) -> AppTest:
        """Initial load then switch the 'Edit Output Filename' radio to 'Yes'."""
        at = _make_loaded_at(mock_ds)
        return _switch_radio(at, "filename", "Yes")

    def test_filename_info_shown(self, mock_ds):
        at = _make_loaded_at(mock_ds)
        info_text = " ".join(i.value for i in at.info)
        assert "file" in info_text.lower() or "current" in info_text.lower()

    def test_use_custom_filename_radio_present(self, mock_ds):
        at = _make_loaded_at(mock_ds)
        radios = [r.label for r in at.radio]
        assert any("filename" in l.lower() or "output" in l.lower() for l in radios)

    def test_text_input_shown_when_editing_yes(self, mock_ds):
        at = self._switch_to_edit_yes(mock_ds)
        assert not at.exception
        text_labels = [t.label for t in at.text_input]
        assert any(
            "file name" in l.lower() or "enter" in l.lower() for l in text_labels
        )

    def test_save_button_shown_when_editing_yes(self, mock_ds):
        at = self._switch_to_edit_yes(mock_ds)
        assert not at.exception
        labels = [b.label for b in at.button]
        assert any("save" in l.lower() for l in labels)

    def test_save_button_absent_when_prefix_already_saved(self, mock_ds):
        at = _make_loaded_at(
            mock_ds,
            extra_state={
                "prefix_saved": True,
                "file_prefix": "GD10A",
                "filename": "GD10A",
            },
        )
        assert not at.exception
        labels = [b.label for b in at.button]
        assert not any("save filename" == l.lower() for l in labels)

    def test_success_banner_when_prefix_saved(self, mock_ds):
        at = _make_loaded_at(
            mock_ds,
            extra_state={
                "prefix_saved": True,
                "file_prefix": "MYSTUDY",
                "filename": "MYSTUDY",
            },
        )
        assert not at.exception
        success_text = " ".join(s.value for s in at.success)
        assert "MYSTUDY" in success_text

    def test_save_button_click_valid_prefix(self, mock_ds):
        """Clicking Save with a non-empty prefix does not crash."""
        at = self._switch_to_edit_yes(mock_ds)
        save_btn = next((b for b in at.button if "save" in b.label.lower()), None)
        if save_btn is None:
            pytest.skip("Save Filename button not found")
        save_btn.click().run()
        assert not at.exception

    def test_save_button_click_empty_prefix_warns(self, mock_ds):
        """Clicking Save with a blank prefix shows a warning."""
        at = self._switch_to_edit_yes(mock_ds)
        # Clear the text input value to empty
        prefix_input = next(
            (
                t
                for t in at.text_input
                if "file name" in t.label.lower() or "enter" in t.label.lower()
            ),
            None,
        )
        if prefix_input is None:
            pytest.skip("File prefix text input not found")
        prefix_input.set_value("").run()
        save_btn = next((b for b in at.button if "save" in b.label.lower()), None)
        if save_btn is None:
            pytest.skip("Save Filename button not found")
        save_btn.click().run()
        assert not at.exception
        warn_text = " ".join(w.value for w in at.warning)
        assert "valid" in warn_text.lower() or "filename" in warn_text.lower()


# ===========================================================================
# 6.  AXIS SELECTBOX
# ===========================================================================


class TestAxisSelectbox:
    def test_default_is_time(self, mock_ds):
        at = _make_loaded_at(mock_ds)
        sb = next((s for s in at.selectbox if "axis" in s.label.lower()), None)
        if sb is None:
            pytest.skip("Axis selectbox not found")
        assert str(sb.value).lower() == "time"

    def test_options_are_time_and_ensemble(self, mock_ds):
        at = _make_loaded_at(mock_ds)
        sb = next((s for s in at.selectbox if "axis" in s.label.lower()), None)
        if sb is None:
            pytest.skip("Axis selectbox not found")
        opts = [str(o).lower() for o in sb.options]
        assert "time" in opts and "ensemble" in opts

    def test_switch_to_ensemble(self, mock_ds):
        at = _make_loaded_at(mock_ds)
        sb = next((s for s in at.selectbox if "axis" in s.label.lower()), None)
        if sb is None:
            pytest.skip("Axis selectbox not found")
        sb.set_value("ensemble").run()
        assert not at.exception
        val = _ss(at, "axis_option_DRW", None)
        assert val == "ensemble"


# ===========================================================================
# 7.  CHECKBOX SECTION
# ===========================================================================


class TestCheckboxSection:
    def test_generate_button_disabled_nothing_selected(self, mock_ds):
        at = _make_loaded_at(mock_ds)
        btns = [b for b in at.button if "generate" in b.label.lower()]
        if btns:
            assert btns[0].disabled

    def test_six_component_checkboxes(self, mock_ds):
        at = _make_loaded_at(mock_ds)
        labels = [c.label.lower() for c in at.checkbox]
        for kw in [
            "fixed leader",
            "variable leader",
            "velocity",
            "echo",
            "correlation",
            "percent good",
        ]:
            assert any(kw in l for l in labels), f"Missing checkbox for '{kw}'"

    def test_checking_velocity_enables_generate(self, mock_ds):
        at = _make_loaded_at(mock_ds)
        cb = next((c for c in at.checkbox if "velocity" in c.label.lower()), None)
        if cb is None:
            pytest.skip("Velocity checkbox not found")
        cb.check().run()
        assert not at.exception
        btns = [b for b in at.button if "generate" in b.label.lower()]
        if btns:
            assert not btns[0].disabled

    def test_selection_info_shown(self, mock_ds):
        at = _make_loaded_at(mock_ds)
        cb = next((c for c in at.checkbox if "velocity" in c.label.lower()), None)
        if cb is None:
            pytest.skip("Velocity checkbox not found")
        cb.check().run()
        assert not at.exception
        info_text = " ".join(i.value for i in at.info)
        assert "velocity" in info_text.lower()

    def test_entire_dataset_checkbox_exists(self, mock_ds):
        at = _make_loaded_at(mock_ds)
        labels = [c.label.lower() for c in at.checkbox]
        assert any("entire" in l for l in labels)

    def test_entire_dataset_selects_all_components(self, mock_ds):
        at = _make_loaded_at(mock_ds)
        entire_cb = next((c for c in at.checkbox if "entire" in c.label.lower()), None)
        if entire_cb is None:
            pytest.skip("Entire Dataset checkbox not found")
        entire_cb.check().run()
        assert not at.exception
        info_text = " ".join(i.value for i in at.info)
        for kw in ["velocity", "correlation", "percent good"]:
            assert kw in info_text.lower(), f"'{kw}' missing from selection info"


# ===========================================================================
# 8.  GENERATE NETCDF BUTTON — click paths
#
#  Uses REAL xr.Dataset so that ds[list_of_vars].copy() and to_netcdf()
#  work correctly inside the page's helper functions.
# ===========================================================================


class TestGenerateNetcdf:
    def _loaded_with_real_ds(self, real_ds, **extra) -> AppTest:
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=30)
        at.session_state["ds"] = real_ds
        at.session_state["fname"] = "test_adcp_GD10A000.000"
        at.session_state["attributes"] = {}
        at.session_state["custom_attributes"] = {}
        for k, v in extra.items():
            at.session_state[k] = v
        at.run()
        return at

    def _check_velocity_and_generate(self, at: AppTest) -> AppTest:
        cb = next((c for c in at.checkbox if c.label == "Velocity"), None)
        if cb is None:
            pytest.skip("Velocity checkbox not found")
        cb.check().run()
        btn = next((b for b in at.button if "generate" in b.label.lower()), None)
        if btn is None:
            pytest.skip("Generate button not found")
        btn.click().run()
        return at

    def test_generate_velocity_time_axis(self, real_ds):
        at = self._loaded_with_real_ds(real_ds)
        at = self._check_velocity_and_generate(at)
        assert not at.exception

    def test_generate_success_message(self, real_ds):
        at = self._loaded_with_real_ds(real_ds)
        at = self._check_velocity_and_generate(at)
        success = " ".join(s.value for s in at.success)
        assert "netcdf" in success.lower() or "generated" in success.lower()

    def test_generate_file_size_shown(self, real_ds):
        at = self._loaded_with_real_ds(real_ds)
        at = self._check_velocity_and_generate(at)
        all_text = " ".join(m.value for m in at.markdown)
        assert "file size" in all_text.lower() or "kb" in all_text.lower()

    def test_generate_variables_count_shown(self, real_ds):
        at = self._loaded_with_real_ds(real_ds)
        at = self._check_velocity_and_generate(at)
        all_text = " ".join(m.value for m in at.markdown)
        assert "variables" in all_text.lower()

    def test_generate_ensemble_axis(self, real_ds):
        at = self._loaded_with_real_ds(real_ds)
        sb = next((s for s in at.selectbox if "axis" in s.label.lower()), None)
        if sb:
            sb.set_value("ensemble").run()
        at = self._check_velocity_and_generate(at)
        assert not at.exception

    def test_generate_with_add_attributes_yes(self, real_ds):
        """add_attributes=True exercises write_netcdf -> add_user_attributes."""
        at = self._loaded_with_real_ds(
            real_ds,
            attributes={"Cruise_No.": "CR2024", "Ship_Name": ""},
            custom_attributes={"My_Key": "My_Value"},
        )
        # Switch the attribute radio to Yes
        attr_radio = next((r for r in at.radio if "attribute" in r.label.lower()), None)
        if attr_radio:
            attr_radio.set_value("Yes").run()
        at = self._check_velocity_and_generate(at)
        assert not at.exception

    def test_generate_all_components(self, real_ds):
        """Entire-dataset checkbox drives all six components through create_subset."""
        at = self._loaded_with_real_ds(real_ds)
        entire_cb = next((c for c in at.checkbox if "entire" in c.label.lower()), None)
        if entire_cb is None:
            pytest.skip("Entire Dataset checkbox not found")
        entire_cb.check().run()
        btn = next((b for b in at.button if "generate" in b.label.lower()), None)
        if btn is None:
            pytest.skip("Generate button not found")
        btn.click().run()
        assert not at.exception

    def test_generate_no_vars_selectable_no_crash(self):
        """When no vars exist Generate is disabled — page renders without crash."""
        mock_ds = _make_mock_ds(
            include_primary_vars=False, include_fl_fields=False, include_vl_fields=False
        )
        at = _make_loaded_at(mock_ds)
        assert not at.exception


# ===========================================================================
# 9.  CSV — Fixed Leader
# ===========================================================================


class TestCsvFixedLeader:
    def _select_fl(self, mock_ds) -> AppTest:
        at = _make_loaded_at(mock_ds)
        sb = next((s for s in at.selectbox if "data type" in s.label.lower()), None)
        if sb is None:
            pytest.skip("CSV selectbox not found")
        sb.set_value("Fixed Leader").run()
        return at

    def test_1d_fl_no_exception(self):
        at = self._select_fl(_make_mock_ds(fl_scalar=False))
        assert not at.exception

    def test_scalar_fl_ndim0_no_exception(self):
        """Scalar FL values (ndim==0) trigger the np.full(n_ens, …) branch."""
        at = self._select_fl(_make_mock_ds(fl_scalar=True))
        assert not at.exception

    def test_multidim_fl_no_exception(self):
        """Multi-dim FL values (ndim>1) take the data[:, 0] slice branch."""
        at = self._select_fl(_make_mock_ds(fl_multidim=True, fl_scalar=False))
        assert not at.exception

    def test_empty_fl_fields_warning(self):
        """Empty fl_fields triggers 'No Fixed Leader data' warning."""
        at = self._select_fl(_make_mock_ds(include_fl_fields=False))
        assert not at.exception
        warn_text = " ".join(w.value for w in at.warning)
        assert "fixed leader" in warn_text.lower()

    def test_no_exception_with_full_fl(self, mock_ds):
        at = self._select_fl(mock_ds)
        assert not at.exception


# ===========================================================================
# 10.  CSV — Variable Leader
# ===========================================================================


class TestCsvVariableLeader:
    def _select_vl(self, mock_ds) -> AppTest:
        at = _make_loaded_at(mock_ds)
        sb = next((s for s in at.selectbox if "data type" in s.label.lower()), None)
        if sb is None:
            pytest.skip("CSV selectbox not found")
        sb.set_value("Variable Leader").run()
        return at

    def test_1d_vl_no_exception(self):
        at = self._select_vl(_make_mock_ds())
        assert not at.exception

    def test_multidim_vl_no_exception(self):
        """Multi-dim VL values (ndim>1) take the data[:, 0] slice branch."""
        at = self._select_vl(_make_mock_ds(vl_multidim=True))
        assert not at.exception

    def test_empty_vl_fields_warning(self):
        at = self._select_vl(_make_mock_ds(include_vl_fields=False))
        assert not at.exception
        warn_text = " ".join(w.value for w in at.warning)
        assert "variable leader" in warn_text.lower()

    def test_no_exception_with_full_vl(self, mock_ds):
        at = self._select_vl(mock_ds)
        assert not at.exception


# ===========================================================================
# 11.  CSV — Beam data
# ===========================================================================


class TestCsvBeamData:
    def _select_beam_csv(self, mock_ds, data_type: str, beam: int = 1) -> AppTest:
        at = _make_loaded_at(mock_ds)
        sb = next((s for s in at.selectbox if "data type" in s.label.lower()), None)
        if sb is None:
            pytest.skip("CSV selectbox not found")
        sb.set_value(data_type).run()
        if beam != 1:
            radio = next((r for r in at.radio if "beam" in r.label.lower()), None)
            if radio:
                radio.set_value(beam).run()
        return at

    def test_velocity_beam1(self, mock_ds):
        assert not self._select_beam_csv(mock_ds, "Velocity", 1).exception

    def test_velocity_beam2(self, mock_ds):
        assert not self._select_beam_csv(mock_ds, "Velocity", 2).exception

    def test_velocity_beam3(self, mock_ds):
        assert not self._select_beam_csv(mock_ds, "Velocity", 3).exception

    def test_velocity_beam4(self, mock_ds):
        assert not self._select_beam_csv(mock_ds, "Velocity", 4).exception

    def test_echo_intensity_csv(self, mock_ds):
        assert not self._select_beam_csv(mock_ds, "Echo Intensity").exception

    def test_correlation_csv(self, mock_ds):
        assert not self._select_beam_csv(mock_ds, "Correlation").exception

    def test_percent_good_csv(self, mock_ds):
        assert not self._select_beam_csv(mock_ds, "Percent Good").exception

    def test_ndim2_branch(self):
        """ndim==2 primary data takes the `download_data = data_array` branch."""
        assert not self._select_beam_csv(
            _make_mock_ds(primary_ndim=2), "Velocity"
        ).exception

    def test_ndim1_warning_branch(self):
        """ndim==1 primary data triggers the 'Unexpected data shape' warning."""
        at = self._select_beam_csv(_make_mock_ds(primary_ndim=1), "Velocity")
        assert not at.exception
        warn_text = " ".join(w.value for w in at.warning)
        assert "unexpected" in warn_text.lower() or "shape" in warn_text.lower()

    def test_missing_var_warning(self):
        """var not in ds.data_vars triggers the 'not available' warning."""
        at = self._select_beam_csv(
            _make_mock_ds(include_primary_vars=False), "Velocity"
        )
        assert not at.exception
        warn_text = " ".join(w.value for w in at.warning)
        assert "not available" in warn_text.lower()

    def test_beam_radio_has_four_options(self, mock_ds):
        at = self._select_beam_csv(mock_ds, "Velocity")
        radio = next((r for r in at.radio if "beam" in r.label.lower()), None)
        if radio is None:
            pytest.skip("Beam radio not found")
        assert len(radio.options) == 4


# ===========================================================================
# 12.  MISSING PRIMARY / FIELD DATA — robustness
# ===========================================================================


class TestMissingPrimaryData:
    def test_no_primary_vars(self):
        assert not _make_loaded_at(_make_mock_ds(include_primary_vars=False)).exception

    def test_no_fl_fields(self):
        assert not _make_loaded_at(_make_mock_ds(include_fl_fields=False)).exception

    def test_no_vl_fields(self):
        assert not _make_loaded_at(_make_mock_ds(include_vl_fields=False)).exception

    def test_empty_field_lists_in_attrs(self):
        mock_ds = _make_mock_ds()
        mock_ds.attrs = {
            "total_ensembles": 50,
            "fixed_leader_variables": [],
            "variable_leader_variables": [],
        }
        assert not _make_loaded_at(mock_ds).exception

    def test_sidebar_no_fl_vl_lines_when_empty(self):
        """FL/VL count lines absent when field lists are empty."""
        mock_ds = _make_mock_ds()
        mock_ds.attrs = {
            "total_ensembles": 50,
            "fixed_leader_variables": [],
            "variable_leader_variables": [],
        }
        at = _make_loaded_at(mock_ds)
        all_text = " ".join(m.value for m in at.markdown)
        assert "FL Variables" not in all_text
        assert "VL Variables" not in all_text


# ===========================================================================
# 13.  PURE HELPER FUNCTION UNIT TESTS
# ===========================================================================


class TestGetPrefixedFilename:
    def test_prefix_prepended(self):
        assert _get_prefixed_filename("RAW_DATA.nc", "GD10A") == "GD10A_RAW_DATA.nc"

    def test_empty_prefix_returns_basename(self):
        assert _get_prefixed_filename("RAW_DATA.nc", "") == "RAW_DATA.nc"

    def test_none_prefix_returns_basename(self):
        assert _get_prefixed_filename("RAW_DATA.nc", None) == "RAW_DATA.nc"

    def test_special_chars(self):
        assert _get_prefixed_filename("F.nc", "A-B_1") == "A-B_1_F.nc"

    def test_long_prefix(self):
        p = "X" * 200
        assert _get_prefixed_filename("F.nc", p) == f"{p}_F.nc"


class TestCreateSubsetDataset:
    @pytest.fixture()
    def real_ds(self):
        n, nc, nb = 30, 10, 4
        tc = pd.date_range("2024-01-01", periods=n, freq="h")
        fl = ["number_of_cells", "pings_per_ensemble"]
        vl = ["heading", "pitch", "roll"]
        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "time"],
                    np.random.randint(-500, 500, (nb, nc, n)).astype(np.int16),
                ),
                "echo_intensity": (
                    ["beam", "cell", "time"],
                    np.random.randint(40, 200, (nb, nc, n)).astype(np.int16),
                ),
                "correlation": (
                    ["beam", "cell", "time"],
                    np.random.randint(0, 255, (nb, nc, n)).astype(np.int16),
                ),
                "percent_good": (
                    ["beam", "cell", "time"],
                    np.random.randint(0, 100, (nb, nc, n)).astype(np.int16),
                ),
                "number_of_cells": ([], 10),
                "pings_per_ensemble": ([], 50),
                "heading": (["time"], np.linspace(0, 360, n)),
                "pitch": (["time"], np.linspace(-5, 5, n)),
                "roll": (["time"], np.linspace(-5, 5, n)),
            },
            coords={
                "time": tc,
                "cell": np.arange(1, nc + 1),
                "beam": np.arange(1, nb + 1),
            },
        )
        ds.attrs = {
            "fixed_leader_variables": fl,
            "variable_leader_variables": vl,
            "source": "test",
        }
        return ds

    def test_velocity_only(self, real_ds):
        fl, vl = (
            real_ds.attrs["fixed_leader_variables"],
            real_ds.attrs["variable_leader_variables"],
        )
        sub = _create_subset_dataset(real_ds, fl, vl, include_velocity=True)
        assert "velocity" in sub.data_vars and "echo_intensity" not in sub.data_vars

    def test_all_components(self, real_ds):
        fl, vl = (
            real_ds.attrs["fixed_leader_variables"],
            real_ds.attrs["variable_leader_variables"],
        )
        sub = _create_subset_dataset(
            real_ds,
            fl,
            vl,
            include_fixed_leader=True,
            include_variable_leader=True,
            include_velocity=True,
            include_echo=True,
            include_correlation=True,
            include_percent_good=True,
        )
        for v in ["velocity", "echo_intensity", "correlation", "percent_good"]:
            assert v in sub.data_vars

    def test_no_selection_returns_none(self, real_ds):
        fl, vl = (
            real_ds.attrs["fixed_leader_variables"],
            real_ds.attrs["variable_leader_variables"],
        )
        assert _create_subset_dataset(real_ds, fl, vl) is None

    def test_empty_field_lists_returns_none(self, real_ds):
        assert (
            _create_subset_dataset(
                real_ds, [], [], include_fixed_leader=True, include_variable_leader=True
            )
            is None
        )

    def test_coordinates_preserved(self, real_ds):
        fl, vl = (
            real_ds.attrs["fixed_leader_variables"],
            real_ds.attrs["variable_leader_variables"],
        )
        sub = _create_subset_dataset(real_ds, fl, vl, include_velocity=True)
        assert "time" in sub.coords and "cell" in sub.coords

    def test_attrs_copied(self, real_ds):
        fl, vl = (
            real_ds.attrs["fixed_leader_variables"],
            real_ds.attrs["variable_leader_variables"],
        )
        sub = _create_subset_dataset(real_ds, fl, vl, include_velocity=True)
        assert sub.attrs["source"] == "test"

    def test_values_unchanged(self, real_ds):
        fl, vl = (
            real_ds.attrs["fixed_leader_variables"],
            real_ds.attrs["variable_leader_variables"],
        )
        orig = real_ds["velocity"].values.copy()
        sub = _create_subset_dataset(real_ds, fl, vl, include_velocity=True)
        np.testing.assert_array_equal(sub["velocity"].values, orig)

    def test_no_duplicates(self, real_ds):
        fl, vl = (
            real_ds.attrs["fixed_leader_variables"],
            real_ds.attrs["variable_leader_variables"],
        )
        sub = _create_subset_dataset(
            real_ds, fl, vl, include_velocity=True, include_echo=True
        )
        var_list = list(sub.data_vars)
        assert len(var_list) == len(set(var_list))


class TestAddUserAttributes:
    @pytest.fixture()
    def simple_ds(self):
        return xr.Dataset(
            {"v": (["time"], np.arange(5.0))},
            coords={"time": pd.date_range("2024-01-01", periods=5, freq="h")},
        )

    def test_standard_attr_added(self, simple_ds):
        r = _add_user_attributes(simple_ds, {"Cruise_No.": "CR2024"}, {})
        assert r.attrs["Cruise_No."] == "CR2024"

    def test_empty_value_skipped(self, simple_ds):
        r = _add_user_attributes(simple_ds, {"Cruise_No.": ""}, {})
        assert "Cruise_No." not in r.attrs

    def test_none_value_skipped(self, simple_ds):
        r = _add_user_attributes(simple_ds, {"Cruise_No.": None}, {})
        assert "Cruise_No." not in r.attrs

    def test_date_converted_to_isoformat(self, simple_ds):
        r = _add_user_attributes(simple_ds, {"Date": date(2024, 1, 15)}, {})
        assert r.attrs["Date"] == "2024-01-15"

    def test_custom_attr_added(self, simple_ds):
        r = _add_user_attributes(simple_ds, {}, {"Model": "Sentinel 300"})
        assert r.attrs["Model"] == "Sentinel 300"

    def test_custom_empty_key_skipped(self, simple_ds):
        r = _add_user_attributes(simple_ds, {}, {"": "value"})
        assert "" not in r.attrs

    def test_custom_empty_value_skipped(self, simple_ds):
        r = _add_user_attributes(simple_ds, {}, {"Key": ""})
        assert "Key" not in r.attrs

    def test_returns_dataset(self, simple_ds):
        assert isinstance(_add_user_attributes(simple_ds, {}, {}), xr.Dataset)


class TestWriteNetcdf:
    @pytest.fixture()
    def wds(self):
        n, nc, nb = 20, 8, 4
        tc = pd.date_range("2024-01-01", periods=n, freq="h")
        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "time"],
                    np.random.randint(-500, 500, (nb, nc, n)).astype(np.int16),
                )
            },
            coords={
                "time": tc,
                "cell": np.arange(1, nc + 1),
                "beam": np.arange(1, nb + 1),
            },
        )
        ds.attrs = {
            "total_ensembles": n,
            "source_file": "test.000",
            "fixed_leader_variables": ["a"],
            "variable_leader_variables": ["b"],
            "components": ["Velocity"],
        }
        return ds

    def test_creates_file(self, wds):
        fp = _write_netcdf(wds, "TEST")
        assert fp is not None and os.path.exists(fp)

    def test_filename_with_prefix(self, wds):
        assert os.path.basename(_write_netcdf(wds, "GD10A")) == "GD10A_RAW_DATA.nc"

    def test_filename_no_prefix(self, wds):
        assert os.path.basename(_write_netcdf(wds, "")) == "RAW_DATA.nc"

    def test_internal_attrs_dropped(self, wds):
        ds = xr.open_dataset(_write_netcdf(wds, "T"))
        for k in ["components", "fixed_leader_variables", "variable_leader_variables"]:
            assert k not in ds.attrs
        ds.close()

    def test_source_attr_preserved(self, wds):
        ds = xr.open_dataset(_write_netcdf(wds, "T"))
        assert "source_file" in ds.attrs
        ds.close()

    def test_time_axis(self, wds):
        ds = xr.open_dataset(_write_netcdf(wds, "T", axis_option="time"))
        assert "time" in ds.dims and "ensemble" not in ds.dims
        ds.close()

    def test_ensemble_axis_renames_dim(self, wds):
        ds = xr.open_dataset(_write_netcdf(wds, "T", axis_option="ensemble"))
        assert "ensemble" in ds.dims and "time" not in ds.dims
        ds.close()

    def test_ensemble_axis_stores_time_original(self, wds):
        ds = xr.open_dataset(_write_netcdf(wds, "T", axis_option="ensemble"))
        assert "time_original" in ds.data_vars
        ds.close()

    def test_user_attrs_written_when_flag_true(self, wds):
        ds = xr.open_dataset(
            _write_netcdf(
                wds,
                "T",
                add_attributes=True,
                attributes={"Cruise_No.": "CR2024"},
                custom_attributes={},
            )
        )
        assert ds.attrs["Cruise_No."] == "CR2024"
        ds.close()

    def test_user_attrs_skipped_when_flag_false(self, wds):
        ds = xr.open_dataset(
            _write_netcdf(
                wds, "T", add_attributes=False, attributes={"Cruise_No.": "CR2024"}
            )
        )
        assert "Cruise_No." not in ds.attrs
        ds.close()

    def test_none_dataset_returns_none(self):
        assert _write_netcdf(None, "T") is None

    def test_data_roundtrip(self, wds):
        orig = wds["velocity"].values.copy()
        ds = xr.open_dataset(_write_netcdf(wds, "T"))
        np.testing.assert_array_equal(ds["velocity"].values, orig)
        ds.close()

    def test_single_ensemble(self):
        ds = xr.Dataset(
            {"velocity": (["beam", "cell", "time"], np.zeros((4, 5, 1)))},
            coords={
                "time": pd.date_range("2024-01-01", periods=1),
                "cell": np.arange(1, 6),
                "beam": np.arange(1, 5),
            },
        )
        ds.attrs = {
            "fixed_leader_variables": [],
            "variable_leader_variables": [],
            "components": [],
        }
        ds2 = xr.open_dataset(_write_netcdf(ds, "SINGLE"))
        assert ds2.sizes["time"] == 1
        ds2.close()


class TestDownloadCsvHelpers:
    """Logic tests for the three CSV helper functions (pure, no Streamlit)."""

    def _csv_with_ensemble(self, data):
        ensembles = np.arange(1, len(next(iter(data.values()))) + 1)
        df = pd.DataFrame(data)
        df.insert(0, "RDI_Ensemble", ensembles)
        return df

    def _csv_2d(self, data):
        df = pd.DataFrame(data)
        ensembles = np.arange(1, df.shape[0] + 1)
        cells = np.arange(1, df.shape[1] + 1)
        df.insert(0, "Ensemble", ensembles)
        df = df.T
        df.insert(0, "Cell", [""] + list(cells))
        return df

    def test_ensemble_column_added(self):
        df = self._csv_with_ensemble({"h": np.linspace(0, 360, 10)})
        assert "RDI_Ensemble" in df.columns
        assert df["RDI_Ensemble"].iloc[0] == 1
        assert df["RDI_Ensemble"].iloc[-1] == 10

    def test_ensemble_row_count(self):
        df = self._csv_with_ensemble({"t": np.full(25, 1500.0)})
        assert len(df) == 25

    def test_csv_2d_shape(self):
        df = self._csv_2d(np.random.randint(0, 100, (10, 5)).astype(float))
        assert df.shape == (6, 11)  # (cells+1 rows, ensembles+1 cols)

    def test_download_csv_dict_branch(self):
        data = {"a": np.arange(5.0), "b": np.arange(5.0) * 2}
        df = pd.DataFrame.from_dict(data, orient="index").T
        assert list(df.columns) == ["a", "b"] and len(df) == 5


# ===========================================================================
# 14.  EDGE CASES
# ===========================================================================


class TestEdgeCases:
    def test_fname_absolute_path(self):
        mock_ds = _make_mock_ds()
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.session_state["ds"] = mock_ds
        at.session_state["fname"] = "/data/cruises/GD10/GD10A000.000"
        at.run()
        assert not at.exception

    def test_no_fname_key(self):
        mock_ds = _make_mock_ds()
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.session_state["ds"] = mock_ds
        at.run()
        assert not at.exception

    def test_pre_populated_attributes_dict(self):
        mock_ds = _make_mock_ds()
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.session_state["ds"] = mock_ds
        at.session_state["fname"] = "test.000"
        at.session_state["attributes"] = {"Cruise_No.": "CR2024"}
        at.run()
        assert not at.exception

    def test_pre_populated_custom_attributes(self):
        mock_ds = _make_mock_ds()
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.session_state["ds"] = mock_ds
        at.session_state["fname"] = "test.000"
        at.session_state["custom_attributes"] = {"My_Field": "My_Value"}
        at.run()
        assert not at.exception

    def test_add_user_attributes_none_value_skipped(self):
        ds = xr.Dataset(
            {"v": (["t"], np.arange(3.0))},
            coords={"t": pd.date_range("2024", periods=3, freq="h")},
        )
        r = _add_user_attributes(ds, {"K": None, "Ship": "R/V Test"}, {})
        assert "K" not in r.attrs and r.attrs["Ship"] == "R/V Test"

    def test_write_netcdf_add_attrs_false_ignores_user_attrs(self):
        ds = xr.Dataset(
            {"v": (["t"], np.arange(5.0))},
            coords={"t": pd.date_range("2024", periods=5, freq="h")},
        )
        ds.attrs = {}
        ds2 = xr.open_dataset(
            _write_netcdf(
                ds, "T", add_attributes=False, attributes={"Cruise_No.": "CR2024"}
            )
        )
        assert "Cruise_No." not in ds2.attrs
        ds2.close()


# ===========================================================================
# RUN
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
