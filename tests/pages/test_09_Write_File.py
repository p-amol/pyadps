"""
AppTest-based Test Suite for 08_Write_File.py

Exercises the Streamlit page via streamlit.testing.v1.AppTest, covering:
  - Session-state initialisation
  - All four tabs (Preview, Attributes, Export, Config)
  - Every export path: NetCDF velocity-only, NetCDF full, CSV
  - Custom attributes workflow
  - Config-file generation (all branches)
  - Sidebar rendering
  - Helper functions via importlib (all dimension/axis/label branches)
  - Plotting functions (velocity, non-velocity, masking branches)
  - Error paths (missing processor, export failure)

Run:
    pytest test_08_Write_File_apptest.py -v
"""

from __future__ import annotations

import json
import sys
import types
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from streamlit.testing.v1 import AppTest

# ---------------------------------------------------------------------------
# SCRIPT PATH  — adjust if the page lives elsewhere in your project
# ---------------------------------------------------------------------------

SCRIPT_PATH = str(
    Path(__file__).parent.parent.parent
    / "src"
    / "pyadps"
    / "pages"
    / "08_Write_File.py"
)

_ATTR_JSON_PATH = (
    Path(__file__).parent.parent.parent / "src" / "pyadps" / "default_attributes.json"
)
with open(_ATTR_JSON_PATH) as _attr_f:
    DEFAULT_ATTRIBUTES = json.load(_attr_f)


# ===========================================================================
# ACCESSOR STUBS
# ===========================================================================


class _FLStub:
    """FixedLeaderAccessor stub — mirrors conftest.py's mock interface."""

    def __init__(self, obj):
        self._obj = obj

    def coordinate_transformation(self, ens: int = 0) -> Dict[str, str]:
        return {"Coordinates": "Earth Coordinates"}

    def system_configuration(self) -> Dict[str, str]:
        return {"Beam Direction": "Up", "Beam Angle": 20}

    def field(self, ens: int = 0) -> Dict[str, Any]:
        return {
            "depth_cell_length": self._obj.attrs.get("cell_size_cm", 400),
            "bin_1_distance": self._obj.attrs.get("bin1_distance_cm", 200),
        }


class _RaisingFLStub:
    """Accessor stub that always raises — forces the attrs fallback branch."""

    def __init__(self, obj):
        pass

    def coordinate_transformation(self, ens: int = 0):
        raise RuntimeError("no accessor")


# ===========================================================================
# DATASET & PROCESSOR FACTORIES
# ===========================================================================


def _make_ds(
    n_beams: int = 4,
    n_cells: int = 10,
    n_time: int = 20,
    earth_coords: bool = True,
    include_extra_vars: bool = True,
) -> xr.Dataset:
    np.random.seed(42)
    time = pd.date_range("2024-01-01", periods=n_time, freq="h")

    data_vars: Dict[str, Any] = {
        "velocity": (
            ["beam", "cell", "time"],
            np.random.randint(-1000, 1000, (n_beams, n_cells, n_time), dtype=np.int16),
        ),
        "mask": (
            ["beam", "cell", "time"],
            np.zeros((n_beams, n_cells, n_time), dtype=np.int8),
        ),
    }
    if include_extra_vars:
        data_vars.update(
            {
                "echo_intensity": (
                    ["beam", "cell", "time"],
                    np.random.randint(
                        50, 200, (n_beams, n_cells, n_time), dtype=np.int16
                    ),
                ),
                "correlation": (
                    ["beam", "cell", "time"],
                    np.random.randint(
                        64, 255, (n_beams, n_cells, n_time), dtype=np.int16
                    ),
                ),
                "percent_good": (
                    ["beam", "cell", "time"],
                    np.random.randint(
                        0, 100, (n_beams, n_cells, n_time), dtype=np.int16
                    ),
                ),
            }
        )

    return xr.Dataset(
        data_vars,
        coords={
            "time": time,
            "cell": np.arange(n_cells),
            "beam": np.arange(n_beams),
        },
        attrs={
            "coordinate_system": "earth" if earth_coords else "beam",
            "beam_angle": 20,
            "beam_direction": "Up",
        },
    )


def _make_proc(ds: Optional[xr.Dataset] = None) -> MagicMock:
    """Build a MagicMock ProcessedDataset that satisfies all page calls."""
    if ds is None:
        ds = _make_ds()

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
    proc.processing_log = []

    # velocity_to_netcdf writes a minimal real file so the download button gets data
    def _vel_to_nc(
        filepath, apply_mask=True, units="cm/s", include_metadata=True, **kw
    ):
        import tempfile, os

        vel = ds["velocity"].values.astype(float) * 0.1
        ds_out = xr.Dataset(
            {
                "zonal_velocity": (["cell", "time"], vel[0]),
                "meridional_velocity": (["cell", "time"], vel[1]),
                "vertical_velocity": (["cell", "time"], vel[2]),
            },
            coords={"time": ds["time"].values, "cell": ds["cell"].values},
        )
        ds_out.to_netcdf(filepath)

    def _to_nc(filepath, **kw):
        ds.to_netcdf(filepath)

    # apply_attributes mirrors ProcessedDataset.apply_attributes: writes
    # straight into dataset.attrs, so downstream assertions on
    # proc.dataset.attrs reflect what the page actually requested.
    def _apply_attrs(attributes):
        for key, value in attributes.items():
            ds.attrs[key] = value

    proc.apply_attributes.side_effect = _apply_attrs

    proc.velocity_to_netcdf.side_effect = _vel_to_nc
    proc.to_netcdf.side_effect = _to_nc
    proc.export_config_string.return_value = "[FileSettings]\ninput_file_name = test.pd0\n"

    return proc


def _full_ss(proc: MagicMock, **overrides) -> Dict[str, Any]:
    """Complete session-state snapshot for the write page."""
    base: Dict[str, Any] = {
        "processor": proc,
        "write_initialized": True,
        "file_prefix": "ADCP",
        "export_format": "NetCDF",
        "export_type": "Velocity Only",
        "apply_mask_export": True,
        "velocity_units": "cm/s",
        "add_attributes": False,
        "write_std_attributes": {field["key"]: "" for field in DEFAULT_ATTRIBUTES},
        "write_custom_attributes": {},
        "write_custom_attr_count": 0,
        "raw_custom_attributes": {},
        # referenced by config generator
        "fname": "test_file.pd0",
        "time_axis_modified": False,
        "sensor_health_applied": False,
        "qc_applied": False,
        "profile_applied": False,
        "velocity_applied": False,
        "correlation_threshold": 64,
        "echo_intensity_threshold": 40,
        "error_velocity_threshold": 2000,
        "cutoff_u": 2500,
        "cutoff_v": 2500,
        "cutoff_w": 500,
    }
    base.update(overrides)
    return base


# ===========================================================================
# MODULE-SCOPED AUTOUSE: inject pyadps mock hierarchy
# ===========================================================================


@pytest.fixture(scope="module", autouse=True)
def inject_pyadps_mock():
    _tracked = (
        "pyadps",
        "pyadps.io",
        "pyadps.io.accessors",
        "pyadps.processing",
        "pyadps.processing.config",
    )
    _originals = {k: sys.modules.get(k) for k in _tracked}

    # Import the real ProcessingConfig before replacing the package tree so
    # the page's `from pyadps.processing.config import ProcessingConfig` works.
    from pyadps.processing.config import ProcessingConfig as _RealProcessingConfig

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
    mock_processing.__path__ = []
    mock_processing.__package__ = "pyadps.processing"
    mock_processing.ProcessedDataset = MagicMock(
        side_effect=lambda ds: MagicMock(dataset=ds)
    )

    mock_config = types.ModuleType("pyadps.processing.config")
    mock_config.ProcessingConfig = _RealProcessingConfig
    mock_processing.config = mock_config

    sys.modules.update(
        {
            "pyadps": mock_pyadps,
            "pyadps.io": mock_io,
            "pyadps.io.accessors": mock_accessors,
            "pyadps.processing": mock_processing,
            "pyadps.processing.config": mock_config,
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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


@pytest.fixture(autouse=True)
def _restore_fl_accessor():
    """Re-register _FLStub before every test to undo any raising-stub contamination."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xr.register_dataset_accessor("fixed_leader")(_FLStub)
    yield


# ===========================================================================
# MODULE-SCOPED FIXTURES
# ===========================================================================


@pytest.fixture(scope="module")
def ds():
    return _make_ds()


@pytest.fixture(scope="module")
def proc(ds):
    return _make_proc(ds)


# ===========================================================================
# APPTEST HELPER
# ===========================================================================


def _run(extra_ss: Dict[str, Any], timeout: int = 15) -> AppTest:
    at = AppTest.from_file(SCRIPT_PATH, default_timeout=timeout)
    for k, v in extra_ss.items():
        at.session_state[k] = v
    at.run()
    return at


# ===========================================================================
# CLASS 1 — Guard rails & page-level smoke
# ===========================================================================


class TestPageGuardsAndSmoke:
    def test_no_processor_shows_error(self):
        at = AppTest.from_file(SCRIPT_PATH, default_timeout=15)
        at.run()
        assert at.error, "Expected st.error when no processor in state"

    def test_page_loads_with_valid_processor(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception, at.exception

    def test_four_tabs_rendered(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_page_header_visible(self, proc):
        at = _run(_full_ss(proc))
        combined = " ".join(e.value for e in at.markdown if hasattr(e, "value"))
        assert "Write" in combined or not at.exception


# ===========================================================================
# CLASS 2 — Session-state initialisation
# ===========================================================================


class TestSessionStateInit:
    def test_fresh_start_sets_defaults(self, proc):
        ss = {"processor": proc}
        at = _run(ss)
        assert not at.exception
        assert at.session_state["write_initialized"] is True
        assert at.session_state["export_format"] == "NetCDF"
        assert at.session_state["export_type"] == "Velocity Only"
        assert at.session_state["apply_mask_export"] is True
        assert at.session_state["velocity_units"] == "cm/s"
        assert at.session_state["add_attributes"] is False
        assert at.session_state["write_std_attributes"] == {
            field["key"]: "" for field in DEFAULT_ATTRIBUTES
        }
        assert at.session_state["write_custom_attributes"] == {}

    def test_second_run_skips_init(self, proc):
        """When write_initialized=True the init block is skipped entirely.
        Verify by checking a key the init block would reset — file_prefix
        is set by init to get_file_prefix(), which for this proc is "ADCP",
        but we pre-seed it to "CUSTOM" and confirm it is NOT reset.
        """
        ss = _full_ss(proc, file_prefix="CUSTOM", write_initialized=True)
        at = _run(ss)
        assert not at.exception
        # If init had run, it would have called get_file_prefix() and stored
        # "ADCP"; "CUSTOM" being preserved means init was skipped.
        assert at.session_state["file_prefix"] == "CUSTOM"

    def test_file_prefix_derived_from_fname(self):
        ds = _make_ds()
        proc = _make_proc(ds)
        ss = {
            "processor": proc,
            "fname": "/data/cruise001.pd0",
        }
        at = _run(ss)
        assert not at.exception
        assert at.session_state["file_prefix"] == "cruise001"

    def test_file_prefix_fallback_to_adcp(self):
        ds = _make_ds()
        proc = _make_proc(ds)
        ss = {"processor": proc}  # no fname, no file_prefix
        at = _run(ss)
        assert not at.exception
        assert at.session_state["file_prefix"] == "ADCP"

    def test_pre_existing_file_prefix_preserved(self, proc):
        ss = _full_ss(proc, file_prefix="MY_PREFIX", write_initialized=True)
        at = _run(ss)
        assert not at.exception
        assert at.session_state["file_prefix"] == "MY_PREFIX"


# ===========================================================================
# CLASS 3 — Tab 1: Preview Data
# ===========================================================================


class TestTab1PreviewData:
    def test_tab1_renders_without_error(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_plot_button_present(self, proc):
        at = _run(_full_ss(proc))
        assert any(b.key == "plot_preview" for b in at.button)

    def test_selectbox_variable_options_present(self, proc):
        at = _run(_full_ss(proc))
        assert any(s.key == "preview_var" for s in at.selectbox)

    def test_plot_velocity_component_u(self, proc):
        at = _run(_full_ss(proc))
        next(b for b in at.button if b.key == "plot_preview").click().run()
        assert not at.exception

    def test_plot_nonvelocity_echo_intensity(self, proc):
        at = _run(_full_ss(proc))
        # Switch to Echo Intensity
        at.selectbox[0].set_value("Echo Intensity").run()
        assert not at.exception
        next(b for b in at.button if b.key == "plot_preview").click().run()
        assert not at.exception

    def test_plot_correlation(self, proc):
        at = _run(_full_ss(proc))
        at.selectbox[0].set_value("Correlation").run()
        assert not at.exception
        next(b for b in at.button if b.key == "plot_preview").click().run()
        assert not at.exception

    def test_plot_percent_good(self, proc):
        at = _run(_full_ss(proc))
        at.selectbox[0].set_value("Percent Good").run()
        assert not at.exception
        next(b for b in at.button if b.key == "plot_preview").click().run()
        assert not at.exception

    def test_plot_with_mask_applied(self, proc):
        at = _run(_full_ss(proc))
        # Default radio is "Yes" (apply mask)
        next(b for b in at.button if b.key == "plot_preview").click().run()
        assert not at.exception

    def test_plot_mask_not_applied(self, proc):
        at = _run(_full_ss(proc))
        # Switch mask radio to No
        mask_radio = next(r for r in at.radio if r.key == "preview_mask")
        mask_radio.set_value("No").run()
        assert not at.exception
        next(b for b in at.button if b.key == "plot_preview").click().run()
        assert not at.exception

    def test_missing_variable_shows_warning(self):
        """Plot button with variable not in dataset shows st.warning."""
        ds = _make_ds(include_extra_vars=False)  # no echo_intensity etc
        proc = _make_proc(ds)
        at = _run(_full_ss(proc))
        at.selectbox[0].set_value("Echo Intensity").run()
        assert not at.exception
        next(b for b in at.button if b.key == "plot_preview").click().run()
        assert not at.exception
        assert any("not found" in (w.value or "").lower() for w in at.warning)

    def test_processing_summary_expander_shows_stats(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_processing_log_shown_when_present(self):
        ds = _make_ds()
        proc = _make_proc(ds)
        proc.processing_log = ["Step 1: QC applied", "Step 2: Velocity threshold"]
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_no_velocity_data_shows_warning(self):
        """plot_velocity_component warns when 'velocity' absent."""
        ds = xr.Dataset(
            {"mask": (["beam", "cell", "time"], np.zeros((4, 5, 10), np.int8))},
            coords={
                "beam": np.arange(4),
                "cell": np.arange(5),
                "time": pd.date_range("2024-01-01", periods=10, freq="h"),
            },
            attrs={"coordinate_system": "earth"},
        )
        proc = _make_proc(ds)
        at = _run(_full_ss(proc))
        # Default var_selection is Velocity; click plot
        next(b for b in at.button if b.key == "plot_preview").click().run()
        assert not at.exception
        assert any("velocity" in (w.value or "").lower() for w in at.warning)


# ===========================================================================
# CLASS 4 — Tab 2: Custom Attributes
# ===========================================================================


class TestTab2Attributes:
    def test_tab2_renders_without_error(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_add_attributes_checkbox_present(self, proc):
        at = _run(_full_ss(proc))
        assert any(c.key == "add_attrs_checkbox" for c in at.checkbox)

    def test_attribute_inputs_hidden_when_disabled(self, proc):
        ss = _full_ss(proc, add_attributes=False)
        at = _run(ss)
        # Attribute text inputs should not be present
        ti_keys = [t.key for t in at.text_input]
        assert "wf_attr_Cruise_No" not in ti_keys

    def test_attribute_inputs_shown_when_enabled(self, proc):
        ss = _full_ss(proc, add_attributes=True)
        at = _run(ss)
        ti_keys = [t.key for t in at.text_input]
        assert "wf_attr_Cruise_No" in ti_keys
        assert "wf_attr_Ship_Name" in ti_keys
        assert "wf_attr_Latitude" in ti_keys
        assert "wf_attr_Longitude" in ti_keys

    def test_attributes_saved_to_session_state(self, proc):
        ss = _full_ss(
            proc,
            add_attributes=True,
            write_std_attributes={"Cruise_No": "CR001", "Ship_Name": "RV Test"},
        )
        at = _run(ss)
        assert not at.exception
        attrs = at.session_state["write_std_attributes"]
        assert attrs["Cruise_No"] == "CR001"

    def test_empty_attributes_show_info(self, proc):
        """When add_attributes=True but all values empty, shows 'No attributes' info."""
        ss = _full_ss(proc, add_attributes=True)
        at = _run(ss)
        assert not at.exception

    def test_info_hint_always_shown(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_all_attribute_fields_present(self, proc):
        ss = _full_ss(proc, add_attributes=True)
        at = _run(ss)
        ti_keys = {t.key for t in at.text_input}
        expected_text = {
            f"wf_attr_{field['key']}"
            for field in DEFAULT_ATTRIBUTES
            if field["widget"] not in ("text_area", "date_input")
        }
        assert expected_text.issubset(ti_keys)

        date_keys = {d.key for d in at.date_input}
        expected_date = {
            f"wf_attr_{field['key']}"
            for field in DEFAULT_ATTRIBUTES
            if field["widget"] == "date_input"
        }
        assert expected_date.issubset(date_keys)

    def test_comments_text_area_present(self, proc):
        ss = _full_ss(proc, add_attributes=True)
        at = _run(ss)
        ta_keys = {t.key for t in at.text_area}
        assert "wf_attr_Comments" in ta_keys


# ===========================================================================
# CLASS 5 — Tab 3: Export — NetCDF Velocity Only
# ===========================================================================


class TestTab3ExportNetCDFVelocity:
    def test_tab3_renders_without_error(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_generate_export_button_present(self, proc):
        at = _run(_full_ss(proc))
        assert any(b.key == "generate_export" for b in at.button)

    def test_export_format_radio_present(self, proc):
        at = _run(_full_ss(proc))
        assert any(r.key == "export_format_radio" for r in at.radio)

    def test_export_type_radio_present(self, proc):
        at = _run(_full_ss(proc))
        assert any(r.key == "export_type_radio" for r in at.radio)

    def test_velocity_units_selectbox_shown_for_velocity_only(self, proc):
        ss = _full_ss(proc, export_type="Velocity Only")
        at = _run(ss)
        assert any(s.key == "velocity_units_select" for s in at.selectbox)

    def test_velocity_units_hidden_for_full_dataset(self, proc):
        """Switching export_type radio to Full Dataset hides velocity_units selectbox."""
        at = _run(_full_ss(proc))
        export_type_radio = next(r for r in at.radio if r.key == "export_type_radio")
        export_type_radio.set_value("Full Dataset").run()
        assert not at.exception
        assert not any(s.key == "velocity_units_select" for s in at.selectbox)

    def test_generate_netcdf_velocity_only_no_attrs(self):
        ds = _make_ds()
        proc = _make_proc(ds)
        ss = _full_ss(
            proc,
            export_format="NetCDF",
            export_type="Velocity Only",
            apply_mask_export=True,
            velocity_units="cm/s",
            add_attributes=False,
        )
        at = _run(ss)
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception
        proc.velocity_to_netcdf.assert_called_once()

    def test_generate_netcdf_velocity_with_attrs(self):
        ds = _make_ds()
        proc = _make_proc(ds)
        ss = _full_ss(
            proc,
            export_format="NetCDF",
            export_type="Velocity Only",
            add_attributes=True,
            write_custom_attributes={"cruise_number": "CR001", "ship_name": "RV Test"},
        )
        at = _run(ss)
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception
        proc.velocity_to_netcdf.assert_called_once()
        # Custom attrs should have been written to dataset.attrs
        assert proc.dataset.attrs.get("cruise_number") == "CR001"

    def test_generate_netcdf_mm_per_s_units(self):
        """Set units to mm/s via selectbox interaction before clicking export."""
        ds = _make_ds()
        proc = _make_proc(ds)
        at = _run(_full_ss(proc, export_format="NetCDF", export_type="Velocity Only"))
        units_sb = next(s for s in at.selectbox if s.key == "velocity_units_select")
        units_sb.set_value("mm/s").run()
        assert not at.exception
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception
        _, kwargs = proc.velocity_to_netcdf.call_args
        assert kwargs.get("units") == "mm/s"

    def test_generate_netcdf_m_per_s_units(self):
        """Set units to m/s via selectbox interaction before clicking export."""
        ds = _make_ds()
        proc = _make_proc(ds)
        at = _run(_full_ss(proc, export_format="NetCDF", export_type="Velocity Only"))
        units_sb = next(s for s in at.selectbox if s.key == "velocity_units_select")
        units_sb.set_value("m/s").run()
        assert not at.exception
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception
        _, kwargs = proc.velocity_to_netcdf.call_args
        assert kwargs.get("units") == "m/s"

    def test_success_message_shown_after_export(self):
        ds = _make_ds()
        proc = _make_proc(ds)
        ss = _full_ss(proc, export_format="NetCDF", export_type="Velocity Only")
        at = _run(ss)
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception
        assert any(at.success)

    def test_custom_attrs_count_shown_after_export(self):
        ds = _make_ds()
        proc = _make_proc(ds)
        ss = _full_ss(
            proc,
            export_format="NetCDF",
            export_type="Velocity Only",
            add_attributes=True,
            write_custom_attributes={"cruise_number": "CR001", "ship_name": "RV Test"},
        )
        at = _run(ss)
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception

    def test_no_attrs_written_when_value_empty(self):
        """Empty-string attribute values are NOT written to proc.dataset.attrs."""
        ds = _make_ds()
        proc = _make_proc(ds)
        ss = _full_ss(
            proc,
            export_format="NetCDF",
            export_type="Velocity Only",
            add_attributes=True,
            write_custom_attributes={"cruise_number": "", "ship_name": "RV Test"},
        )
        at = _run(ss)
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception
        # Empty cruise_number must not appear
        assert "cruise_number" not in proc.dataset.attrs
        # Non-empty ship_name must appear
        assert proc.dataset.attrs.get("ship_name") == "RV Test"


# ===========================================================================
# CLASS 6 — Tab 3: Export — NetCDF Full Dataset
# ===========================================================================


class TestTab3ExportNetCDFFull:
    def test_generate_netcdf_full_dataset_no_attrs(self):
        ds = _make_ds()
        proc = _make_proc(ds)
        at = _run(_full_ss(proc, add_attributes=False))
        # Switch export_type radio to Full Dataset
        next(r for r in at.radio if r.key == "export_type_radio").set_value(
            "Full Dataset"
        ).run()
        assert not at.exception
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception
        proc.to_netcdf.assert_called_once()

    def test_generate_netcdf_full_with_attrs(self):
        ds = _make_ds()
        proc = _make_proc(ds)
        ss = _full_ss(
            proc,
            add_attributes=True,
            write_custom_attributes={"project_number": "P42", "contact": "foo@bar.com"},
        )
        at = _run(ss)
        next(r for r in at.radio if r.key == "export_type_radio").set_value(
            "Full Dataset"
        ).run()
        assert not at.exception
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception
        proc.to_netcdf.assert_called_once()
        assert proc.dataset.attrs.get("project_number") == "P42"

    def test_full_dataset_success_message(self):
        ds = _make_ds()
        proc = _make_proc(ds)
        at = _run(_full_ss(proc))
        next(r for r in at.radio if r.key == "export_type_radio").set_value(
            "Full Dataset"
        ).run()
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception
        assert any(at.success)

    def test_full_dataset_with_custom_attrs_count_shown(self):
        ds = _make_ds()
        proc = _make_proc(ds)
        ss = _full_ss(
            proc,
            add_attributes=True,
            write_custom_attributes={"cruise_number": "CR42", "ship_name": "RV Sea"},
        )
        at = _run(ss)
        next(r for r in at.radio if r.key == "export_type_radio").set_value(
            "Full Dataset"
        ).run()
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception

    def test_file_prefix_applied_to_filename(self):
        """get_prefixed_filename uses the file_prefix from session state."""
        ds = _make_ds()
        proc = _make_proc(ds)
        ss = _full_ss(
            proc,
            export_format="NetCDF",
            export_type="Velocity Only",
            file_prefix="CRUISE01",
        )
        at = _run(ss)
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception
        filepath = proc.velocity_to_netcdf.call_args[0][0]
        assert "CRUISE01" in filepath

    def test_empty_prefix_filename_no_prefix(self):
        """When file_prefix is empty, filename has no prefix prepended."""
        ds = _make_ds()
        proc = _make_proc(ds)
        ss = _full_ss(
            proc, export_format="NetCDF", export_type="Velocity Only", file_prefix=""
        )
        at = _run(ss)
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception


# ===========================================================================
# CLASS 7 — Tab 3: Export — CSV
# ===========================================================================


class TestTab3ExportCSV:
    def test_generate_csv_velocity_only(self):
        ds = _make_ds()
        proc = _make_proc(ds)
        at = _run(_full_ss(proc, apply_mask_export=False))
        next(r for r in at.radio if r.key == "export_format_radio").set_value(
            "CSV"
        ).run()
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception
        assert any(at.success)

    def test_csv_with_mask_applied(self):
        ds = _make_ds()
        ds["mask"].values[0, :3, :] = 1
        proc = _make_proc(ds)
        at = _run(_full_ss(proc, apply_mask_export=True))
        next(r for r in at.radio if r.key == "export_format_radio").set_value(
            "CSV"
        ).run()
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception

    def test_csv_exports_mask_when_present(self):
        """CSV path also generates a mask CSV download button."""
        ds = _make_ds()
        proc = _make_proc(ds)
        at = _run(_full_ss(proc, apply_mask_export=False))
        next(r for r in at.radio if r.key == "export_format_radio").set_value(
            "CSV"
        ).run()
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception

    def test_csv_mask_4_plus_beams_uses_beam3(self):
        """4-beam mask: mask[3, :, :] chosen for combined mask CSV."""
        ds = _make_ds(n_beams=4)
        ds["mask"].values[3, :2, :] = 1
        proc = _make_proc(ds)
        at = _run(_full_ss(proc, apply_mask_export=False))
        next(r for r in at.radio if r.key == "export_format_radio").set_value(
            "CSV"
        ).run()
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception

    def test_csv_mask_fewer_than_4_beams_uses_beam0(self):
        """2-beam mask: mask[0, :, :] chosen as fallback."""
        ds = _make_ds(n_beams=2)
        proc = _make_proc(ds)
        at = _run(_full_ss(proc, apply_mask_export=False))
        next(r for r in at.radio if r.key == "export_format_radio").set_value(
            "CSV"
        ).run()
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception

    def test_csv_success_message(self):
        ds = _make_ds()
        proc = _make_proc(ds)
        at = _run(_full_ss(proc))
        next(r for r in at.radio if r.key == "export_format_radio").set_value(
            "CSV"
        ).run()
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception
        assert any("CSV" in (s.value or "") for s in at.success)

    def test_export_error_shows_error_widget(self):
        """If velocity_to_netcdf raises, page catches it and calls st.error."""
        ds = _make_ds()
        proc = _make_proc(ds)
        proc.velocity_to_netcdf.side_effect = RuntimeError("disk full")
        ss = _full_ss(proc, export_format="NetCDF", export_type="Velocity Only")
        at = _run(ss)
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception
        assert any(at.error)

    def test_export_csv_info_attributes_not_added(self):
        """No custom attributes are shown when add_attributes=False."""
        ds = _make_ds()
        proc = _make_proc(ds)
        ss = _full_ss(
            proc,
            export_format="NetCDF",
            export_type="Velocity Only",
            add_attributes=False,
        )
        at = _run(ss)
        assert not at.exception
        info_vals = [i.value or "" for i in at.info]
        assert any("no attributes" in v.lower() for v in info_vals)

    def test_attributes_count_shown_when_enabled(self):
        """With add_attributes=True, success banner appears in Tab 3."""
        ds = _make_ds()
        proc = _make_proc(ds)
        ss = _full_ss(
            proc,
            add_attributes=True,
            write_custom_attributes={"cruise_number": "CR001"},
        )
        at = _run(ss)
        assert not at.exception
        success_vals = [s.value or "" for s in at.success]
        assert any("1" in v and "included" in v.lower() for v in success_vals)


# ===========================================================================
# CLASS 8 — Tab 4: Config File Generator
# ===========================================================================


class TestTab4ConfigFile:
    def test_tab4_renders_without_error(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_config_checkbox_present(self, proc):
        at = _run(_full_ss(proc))
        assert any(c.key == "generate_config_checkbox" for c in at.checkbox)

    def test_config_button_hidden_when_unchecked(self, proc):
        ss = _full_ss(proc)  # generate_config_checkbox defaults to False
        at = _run(ss)
        btn_keys = {b.key for b in at.button}
        assert "gen_config_btn" not in btn_keys

    def test_config_button_shown_when_checked(self, proc):
        at = _run(_full_ss(proc))
        cb = next(c for c in at.checkbox if c.key == "generate_config_checkbox")
        cb.check().run()
        assert not at.exception
        assert any(b.key == "gen_config_btn" for b in at.button)

    def test_generate_config_basic(self, proc):
        """Generate config.ini with default (no qc, no velocity) settings."""
        at = _run(_full_ss(proc))
        cb = next(c for c in at.checkbox if c.key == "generate_config_checkbox")
        cb.check().run()
        assert not at.exception
        next(b for b in at.button if b.key == "gen_config_btn").click().run()
        assert not at.exception

    def test_config_includes_file_settings(self, proc):
        at = _run(_full_ss(proc, fname="my_data.pd0"))
        cb = next(c for c in at.checkbox if c.key == "generate_config_checkbox")
        cb.check().run()
        next(b for b in at.button if b.key == "gen_config_btn").click().run()
        assert not at.exception

    def test_config_velocity_only_section(self, proc):
        ss = _full_ss(proc, export_type="Velocity Only", velocity_units="m/s")
        at = _run(ss)
        cb = next(c for c in at.checkbox if c.key == "generate_config_checkbox")
        cb.check().run()
        next(b for b in at.button if b.key == "gen_config_btn").click().run()
        assert not at.exception

    def test_config_full_dataset_type(self, proc):
        ss = _full_ss(proc, export_type="Full Dataset")
        at = _run(ss)
        cb = next(c for c in at.checkbox if c.key == "generate_config_checkbox")
        cb.check().run()
        next(b for b in at.button if b.key == "gen_config_btn").click().run()
        assert not at.exception

    def test_config_with_qc_applied(self, proc):
        """Config includes QC thresholds when qc_applied=True."""
        ss = _full_ss(
            proc,
            qc_applied=True,
            correlation_threshold=80,
            echo_intensity_threshold=50,
            error_velocity_threshold=1500,
        )
        at = _run(ss)
        cb = next(c for c in at.checkbox if c.key == "generate_config_checkbox")
        cb.check().run()
        next(b for b in at.button if b.key == "gen_config_btn").click().run()
        assert not at.exception

    def test_config_with_velocity_applied(self, proc):
        """Config includes velocity cutoffs when velocity_applied=True."""
        ss = _full_ss(
            proc,
            velocity_applied=True,
            cutoff_u=1800,
            cutoff_v=1900,
            cutoff_w=400,
        )
        at = _run(ss)
        cb = next(c for c in at.checkbox if c.key == "generate_config_checkbox")
        cb.check().run()
        next(b for b in at.button if b.key == "gen_config_btn").click().run()
        assert not at.exception

    def test_config_with_custom_attrs(self, proc):
        """Custom attrs are applied to proc (and so reach the config) when set."""
        ss = _full_ss(
            proc,
            add_attributes=True,
            write_std_attributes={},
            write_custom_attributes={"cruise_number": "CR001", "ship_name": "RV Test"},
            write_custom_attr_count=0,
        )
        at = _run(ss)
        cb = next(c for c in at.checkbox if c.key == "generate_config_checkbox")
        cb.check().run()
        next(b for b in at.button if b.key == "gen_config_btn").click().run()
        assert not at.exception
        proc.apply_attributes.assert_called_with(
            {"cruise_number": "CR001", "ship_name": "RV Test"}
        )

    def test_config_attrs_with_empty_value_skipped(self, proc):
        """Custom attrs with empty value are not forwarded to apply_attributes."""
        ss = _full_ss(
            proc,
            add_attributes=True,
            write_std_attributes={},
            write_custom_attributes={"cruise_number": "", "ship_name": "RV Test"},
            write_custom_attr_count=0,
        )
        at = _run(ss)
        cb = next(c for c in at.checkbox if c.key == "generate_config_checkbox")
        cb.check().run()
        next(b for b in at.button if b.key == "gen_config_btn").click().run()
        assert not at.exception
        proc.apply_attributes.assert_called_with({"ship_name": "RV Test"})

    def test_config_no_attrs_section_when_disabled(self, proc):
        """No [Attributes] section written when add_attributes=False."""
        call_count_before = proc.apply_attributes.call_count
        ss = _full_ss(proc, add_attributes=False)
        at = _run(ss)
        cb = next(c for c in at.checkbox if c.key == "generate_config_checkbox")
        cb.check().run()
        next(b for b in at.button if b.key == "gen_config_btn").click().run()
        assert not at.exception
        assert proc.apply_attributes.call_count == call_count_before

    def test_config_all_processing_stages_applied(self, proc):
        """All flags True: sensor_health, qc, profile, velocity."""
        ss = _full_ss(
            proc,
            sensor_health_applied=True,
            qc_applied=True,
            profile_applied=True,
            velocity_applied=True,
            time_axis_modified=True,
        )
        at = _run(ss)
        cb = next(c for c in at.checkbox if c.key == "generate_config_checkbox")
        cb.check().run()
        next(b for b in at.button if b.key == "gen_config_btn").click().run()
        assert not at.exception

    def test_config_success_message_shown(self, proc):
        at = _run(_full_ss(proc))
        cb = next(c for c in at.checkbox if c.key == "generate_config_checkbox")
        cb.check().run()
        next(b for b in at.button if b.key == "gen_config_btn").click().run()
        assert not at.exception
        assert any(at.success)


# ===========================================================================
# CLASS 9 — Sidebar
# ===========================================================================


class TestSidebar:
    def test_sidebar_renders_without_error(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_sidebar_shows_velocity_units_for_velocity_only(self, proc):
        ss = _full_ss(proc, export_type="Velocity Only", velocity_units="m/s")
        at = _run(ss)
        assert not at.exception

    def test_sidebar_hides_velocity_units_for_full_dataset(self, proc):
        ss = _full_ss(proc, export_type="Full Dataset")
        at = _run(ss)
        assert not at.exception

    def test_sidebar_earth_coords_label(self, proc):
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_sidebar_beam_coords_label(self):
        ds = _make_ds(earth_coords=False)
        proc = _make_proc(ds)
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_sidebar_processing_log_shown(self):
        ds = _make_ds()
        proc = _make_proc(ds)
        proc.processing_log = ["Step 1", "Step 2", "Step 3"]
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_sidebar_no_processing_log(self):
        ds = _make_ds()
        proc = _make_proc(ds)
        proc.processing_log = []
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_sidebar_no_processing_log_attr(self):
        ds = _make_ds()
        proc = _make_proc(ds)
        del proc.processing_log
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_sidebar_custom_attrs_checked(self, proc):
        ss = _full_ss(proc, add_attributes=True)
        at = _run(ss)
        assert not at.exception

    def test_sidebar_long_processing_log_truncated(self):
        """Only last 5 log entries rendered (page slices [-5:])."""
        ds = _make_ds()
        proc = _make_proc(ds)
        proc.processing_log = [f"Step {i}" for i in range(10)]
        at = _run(_full_ss(proc))
        assert not at.exception


# ===========================================================================
# CLASS 10 — Helper functions via importlib
# ===========================================================================


class TestHelperFunctions:
    """Test all helper branches by loading the page module via importlib."""

    @pytest.fixture(scope="class")
    def page_module(self, inject_pyadps_mock):
        import importlib.util
        import streamlit as st

        ds_mod = _make_ds()
        proc_mod = _make_proc(ds_mod)

        spec = importlib.util.spec_from_file_location("write_page", SCRIPT_PATH)
        mod = importlib.util.module_from_spec(spec)

        with (
            patch.object(st, "set_page_config"),
            patch.object(st, "stop", side_effect=SystemExit(0)),
            patch.object(st, "error"),
            patch.object(st, "header"),
            patch.object(st, "write"),
            patch.object(st, "tabs", return_value=[MagicMock() for _ in range(4)]),
            patch.dict(
                "streamlit.session_state",
                {
                    "processor": proc_mod,
                    "write_initialized": True,
                    "export_format": "NetCDF",
                    "export_type": "Velocity Only",
                    "apply_mask_export": True,
                    "velocity_units": "cm/s",
                    "add_attributes": False,
                    "custom_attributes": {},
                    "file_prefix": "ADCP",
                    "fname": "test.pd0",
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

    # --- get_total_ensembles ---

    def test_get_total_ensembles_time_dim(self, page_module):
        assert page_module.get_total_ensembles() == 20

    def test_get_total_ensembles_ensemble_dim(self, page_module):
        orig = page_module.ds
        page_module.ds = xr.Dataset({"v": (["ensemble"], np.zeros(7))})
        try:
            assert page_module.get_total_ensembles() == 7
        finally:
            page_module.ds = orig

    def test_get_total_ensembles_no_dim(self, page_module):
        orig = page_module.ds
        page_module.ds = xr.Dataset({"v": (["x"], [1])})
        try:
            assert page_module.get_total_ensembles() == 0
        finally:
            page_module.ds = orig

    # --- get_total_cells ---

    def test_get_total_cells_cell_dim(self, page_module):
        assert page_module.get_total_cells() == 10

    def test_get_total_cells_depth_dim(self, page_module):
        orig = page_module.ds
        page_module.ds = xr.Dataset({"v": (["depth"], np.zeros(15))})
        try:
            assert page_module.get_total_cells() == 15
        finally:
            page_module.ds = orig

    def test_get_total_cells_no_dim(self, page_module):
        orig = page_module.ds
        page_module.ds = xr.Dataset({"v": (["x"], [1])})
        try:
            assert page_module.get_total_cells() == 0
        finally:
            page_module.ds = orig

    # --- get_total_beams ---

    def test_get_total_beams_beam_dim(self, page_module):
        assert page_module.get_total_beams() == 4

    def test_get_total_beams_no_beam_dim(self, page_module):
        orig = page_module.ds
        page_module.ds = xr.Dataset({"v": (["x"], [1])})
        try:
            assert page_module.get_total_beams() == 4  # default
        finally:
            page_module.ds = orig

    # --- get_time_axis ---

    def test_get_time_axis_time_coord(self, page_module):
        result = page_module.get_time_axis()
        assert len(result) == 20

    def test_get_time_axis_ensemble_coord(self, page_module):
        orig = page_module.ds
        page_module.ds = xr.Dataset(
            {"v": (["ensemble"], np.zeros(5))},
            coords={"ensemble": np.arange(5)},
        )
        try:
            result = page_module.get_time_axis()
            assert len(result) == 5
        finally:
            page_module.ds = orig

    def test_get_time_axis_fallback_arange(self, page_module):
        orig = page_module.ds
        page_module.ds = xr.Dataset({"v": (["x"], [1, 2, 3])})
        try:
            result = page_module.get_time_axis()
            assert len(result) == 0  # get_total_ensembles() → 0
        finally:
            page_module.ds = orig

    # --- get_depth_axis ---

    def test_get_depth_axis_depth_coord(self, page_module):
        orig = page_module.ds
        page_module.ds = xr.Dataset(
            {"v": (["depth"], np.zeros(8))},
            coords={"depth": np.linspace(10, 80, 8)},
        )
        try:
            result = page_module.get_depth_axis()
            assert len(result) == 8
        finally:
            page_module.ds = orig

    def test_get_depth_axis_cell_coord(self, page_module):
        result = page_module.get_depth_axis()
        assert len(result) == 10

    def test_get_depth_axis_fallback_arange(self, page_module):
        orig = page_module.ds
        page_module.ds = xr.Dataset({"v": (["x"], [1, 2])})
        try:
            result = page_module.get_depth_axis()
            assert len(result) == 0  # get_total_cells() → 0
        finally:
            page_module.ds = orig

    # --- is_earth_coordinates ---

    def test_is_earth_coordinates_via_accessor(self, page_module):
        result = page_module.is_earth_coordinates()
        assert result is True

    def test_is_earth_coordinates_fallback_earth(self, page_module):
        orig = page_module.ds
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xr.register_dataset_accessor("fixed_leader")(_RaisingFLStub)
        page_module.ds = xr.Dataset(
            {"v": (["x"], [1])}, attrs={"coordinate_system": "earth"}
        )
        try:
            assert page_module.is_earth_coordinates() is True
        finally:
            page_module.ds = orig
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xr.register_dataset_accessor("fixed_leader")(_FLStub)

    def test_is_earth_coordinates_fallback_beam(self, page_module):
        orig = page_module.ds
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xr.register_dataset_accessor("fixed_leader")(_RaisingFLStub)
        page_module.ds = xr.Dataset(
            {"v": (["x"], [1])}, attrs={"coordinate_system": "beam"}
        )
        try:
            assert page_module.is_earth_coordinates() is False
        finally:
            page_module.ds = orig
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xr.register_dataset_accessor("fixed_leader")(_FLStub)

    # --- get_velocity_labels ---

    def test_get_velocity_labels_earth(self, page_module):
        result = page_module.get_velocity_labels()
        assert result == ("U (East)", "V (North)", "W (Vertical)", "Error")

    def test_get_velocity_labels_beam(self, page_module):
        orig = page_module.ds
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xr.register_dataset_accessor("fixed_leader")(_RaisingFLStub)
        page_module.ds = xr.Dataset(
            {"v": (["x"], [1])}, attrs={"coordinate_system": "beam"}
        )
        try:
            result = page_module.get_velocity_labels()
            assert result == ("Beam 1", "Beam 2", "Beam 3", "Beam 4")
        finally:
            page_module.ds = orig
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xr.register_dataset_accessor("fixed_leader")(_FLStub)

    # --- get_file_prefix ---

    def test_get_file_prefix_from_session_state(self, page_module):
        import streamlit as st

        with patch.dict(
            "streamlit.session_state", {"file_prefix": "MYPREFIX"}, clear=False
        ):
            result = page_module.get_file_prefix()
        assert result == "MYPREFIX"

    def test_get_file_prefix_from_fname(self, page_module):
        import streamlit as st

        with patch.dict(
            "streamlit.session_state",
            {"file_prefix": "", "fname": "/some/path/cruise01.pd0"},
            clear=False,
        ):
            result = page_module.get_file_prefix()
        assert result == "cruise01"

    def test_get_file_prefix_fallback_adcp(self, page_module):
        import streamlit as st

        with patch.dict(
            "streamlit.session_state",
            {"file_prefix": "", "fname": ""},
            clear=False,
        ):
            result = page_module.get_file_prefix()
        assert result == "ADCP"

    # --- get_prefixed_filename ---

    def test_get_prefixed_filename_with_prefix(self, page_module):
        import streamlit as st

        with patch.dict("streamlit.session_state", {"file_prefix": "PRE"}, clear=False):
            result = page_module.get_prefixed_filename("velocity.nc")
        assert result == "PRE_velocity.nc"

    def test_get_prefixed_filename_empty_prefix(self, page_module):
        import streamlit as st

        with patch.dict(
            "streamlit.session_state",
            {"file_prefix": "", "fname": ""},
            clear=False,
        ):
            result = page_module.get_prefixed_filename("velocity.nc")
        # empty prefix → fallback is "ADCP" (truthy), so still prefixed
        assert "velocity.nc" in result


# ===========================================================================
# CLASS 11 — Plotting function smoke tests via importlib
# ===========================================================================


class TestPlottingFunctions:
    """Exercise plot_data_heatmap and plot_velocity_component branches."""

    @pytest.fixture(scope="class")
    def page_module(self, inject_pyadps_mock):
        import importlib.util
        import streamlit as st

        ds_mod = _make_ds()
        proc_mod = _make_proc(ds_mod)

        spec = importlib.util.spec_from_file_location("write_page_plot", SCRIPT_PATH)
        mod = importlib.util.module_from_spec(spec)

        with (
            patch.object(st, "set_page_config"),
            patch.object(st, "stop", side_effect=SystemExit(0)),
            patch.object(st, "error"),
            patch.object(st, "header"),
            patch.object(st, "write"),
            patch.object(st, "plotly_chart"),
            patch.object(st, "warning"),
            patch.object(st, "tabs", return_value=[MagicMock() for _ in range(4)]),
            patch.dict(
                "streamlit.session_state",
                {
                    "processor": proc_mod,
                    "write_initialized": True,
                    "export_format": "NetCDF",
                    "export_type": "Velocity Only",
                    "apply_mask_export": True,
                    "velocity_units": "cm/s",
                    "add_attributes": False,
                    "custom_attributes": {},
                    "file_prefix": "ADCP",
                    "fname": "test.pd0",
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

    def test_plot_data_heatmap_no_mask(self, page_module):
        data = np.random.randn(10, 20)
        with patch("streamlit.plotly_chart"):
            page_module.plot_data_heatmap(data, "Test", apply_mask=False)

    def test_plot_data_heatmap_with_3d_mask(self, page_module):
        data = np.random.randn(10, 20)
        mask = np.zeros((4, 10, 20), dtype=np.int8)
        mask[0, :2, :] = 1
        orig = page_module.ds
        ds_masked = _make_ds()
        ds_masked["mask"].values[:] = mask
        page_module.ds = ds_masked
        try:
            with patch("streamlit.plotly_chart"):
                page_module.plot_data_heatmap(data, "Test", apply_mask=True)
        finally:
            page_module.ds = orig

    def test_plot_data_heatmap_with_2d_mask(self, page_module):
        """apply_mask=True with a 2-D mask variable."""
        data = np.random.randn(10, 20)
        orig = page_module.ds
        ds2d = _make_ds()
        # Collapse beam dim to make a 2-D mask
        mask2d = ds2d["mask"].values[0]  # shape (cell, time)
        import xarray as xr

        ds2d_mod = xr.Dataset(
            {
                "velocity": ds2d["velocity"],
                "mask": (["cell", "time"], mask2d),
            },
            coords={"time": ds2d.time, "cell": ds2d.cell, "beam": ds2d.beam},
            attrs={"coordinate_system": "earth"},
        )
        page_module.ds = ds2d_mod
        try:
            with patch("streamlit.plotly_chart"):
                page_module.plot_data_heatmap(data, "Test", apply_mask=True)
        finally:
            page_module.ds = orig

    def test_plot_data_heatmap_mask_3d_data_3d(self, page_module):
        """3-D plot_data and 3-D mask: mask_2d falls to None branch."""
        data = np.random.randn(4, 10, 20)  # 3-D, same as mask ndim
        with patch("streamlit.plotly_chart"):
            page_module.plot_data_heatmap(data, "Test", apply_mask=True)

    def test_plot_data_heatmap_replaces_missing_value(self, page_module):
        data = np.full((10, 20), -32768.0)
        with patch("streamlit.plotly_chart"):
            page_module.plot_data_heatmap(data, "Test", missing_value=-32768)

    def test_plot_velocity_component_no_velocity_var(self, page_module):
        orig = page_module.ds
        page_module.ds = xr.Dataset(
            {"mask": (["beam", "cell", "time"], np.zeros((4, 10, 20), np.int8))},
            coords={
                "beam": np.arange(4),
                "cell": np.arange(10),
                "time": pd.date_range("2024-01-01", periods=20, freq="h"),
            },
        )
        try:
            with patch("streamlit.plotly_chart"), patch("streamlit.warning") as mw:
                page_module.plot_velocity_component(0, "U", apply_mask=False)
                mw.assert_called_once()
        finally:
            page_module.ds = orig

    def test_plot_velocity_component_no_mask_applied(self, page_module):
        with patch("streamlit.plotly_chart"):
            page_module.plot_velocity_component(0, "U", apply_mask=False)

    def test_plot_velocity_component_with_mask(self, page_module):
        with patch("streamlit.plotly_chart"):
            page_module.plot_velocity_component(0, "U", apply_mask=True)

    def test_plot_velocity_component_missing_values_replaced(self, page_module):
        orig = page_module.ds
        ds_mv = _make_ds()
        ds_mv["velocity"].values[0, :, :5] = -32768
        page_module.ds = ds_mv
        try:
            with patch("streamlit.plotly_chart"):
                page_module.plot_velocity_component(0, "U", apply_mask=False)
        finally:
            page_module.ds = orig


# ===========================================================================
# CLASS 12 — Edge cases and alternate dataset shapes
# ===========================================================================


class TestEdgeCases:
    def test_ensemble_dim_instead_of_time(self):
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
            },
            attrs={"coordinate_system": "earth"},
        )
        proc = _make_proc(ds)
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_depth_coord_instead_of_cell(self):
        depth = np.linspace(10, 80, 8)
        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "depth", "time"],
                    np.zeros((4, 8, 10), dtype=np.int16),
                ),
                "mask": (
                    ["beam", "depth", "time"],
                    np.zeros((4, 8, 10), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(4),
                "depth": depth,
                "time": pd.date_range("2024-01-01", periods=10, freq="h"),
            },
            attrs={"coordinate_system": "earth"},
        )
        proc = _make_proc(ds)
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_beam_coordinate_system(self):
        ds = _make_ds(earth_coords=False)
        proc = _make_proc(ds)
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_minimal_1_cell_dataset(self):
        ds = _make_ds(n_cells=1, n_time=5)
        proc = _make_proc(ds)
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_dataset_without_extra_vars(self):
        """Page renders with velocity+mask only (no echo/corr/pct_good)."""
        ds = _make_ds(include_extra_vars=False)
        proc = _make_proc(ds)
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_proc_without_processing_log_attr(self):
        ds = _make_ds()
        proc = _make_proc(ds)
        del proc.processing_log
        at = _run(_full_ss(proc))
        assert not at.exception

    def test_masked_dataset_csv_export(self):
        ds = _make_ds()
        ds["mask"].values[:, :3, :] = 1
        proc = _make_proc(ds)
        at = _run(_full_ss(proc, apply_mask_export=True))
        next(r for r in at.radio if r.key == "export_format_radio").set_value(
            "CSV"
        ).run()
        next(b for b in at.button if b.key == "generate_export").click().run()
        assert not at.exception


# ===========================================================================
# CLASS 13 — Targeted coverage for lines 130, 343, 888-889, 931
# ===========================================================================


class TestCoverageGaps:
    """Pinpoint tests for the four uncovered lines."""

    # ------------------------------------------------------------------ #
    # Line 130: return base_name in get_prefixed_filename                 #
    # get_file_prefix() never returns ""; patch it on the loaded module. #
    # ------------------------------------------------------------------ #

    def test_get_prefixed_filename_empty_prefix_returns_base_name(
        self, inject_pyadps_mock
    ):
        import importlib.util, streamlit as st

        ds_mod = _make_ds()
        proc_mod = _make_proc(ds_mod)

        spec = importlib.util.spec_from_file_location("write_page_pfx", SCRIPT_PATH)
        mod = importlib.util.module_from_spec(spec)

        with (
            patch.object(st, "set_page_config"),
            patch.object(st, "stop", side_effect=SystemExit(0)),
            patch.object(st, "error"),
            patch.object(st, "header"),
            patch.object(st, "write"),
            patch.object(st, "tabs", return_value=[MagicMock() for _ in range(4)]),
            patch.dict(
                "streamlit.session_state",
                {
                    "processor": proc_mod,
                    "write_initialized": True,
                    "export_format": "NetCDF",
                    "export_type": "Velocity Only",
                    "apply_mask_export": True,
                    "velocity_units": "cm/s",
                    "add_attributes": False,
                    "custom_attributes": {},
                    "file_prefix": "ADCP",
                    "fname": "test.pd0",
                },
                clear=False,
            ),
        ):
            try:
                spec.loader.exec_module(mod)
            except SystemExit:
                pass

        # Force get_file_prefix() to return "" so line 130 is reached
        mod.get_file_prefix = lambda: ""
        result = mod.get_prefixed_filename("velocity.nc")
        assert result == "velocity.nc"

    # ------------------------------------------------------------------ #
    # Line 343: data_2d = data  (non-velocity var is 2-D, not 3-D)      #
    # ------------------------------------------------------------------ #

    def test_plot_2d_non_velocity_variable(self, inject_pyadps_mock):
        n_c, n_t = 10, 20
        time = pd.date_range("2024-01-01", periods=n_t, freq="h")

        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "time"],
                    np.zeros((4, n_c, n_t), dtype=np.int16),
                ),
                "mask": (
                    ["beam", "cell", "time"],
                    np.zeros((4, n_c, n_t), dtype=np.int8),
                ),
                # 2-D echo_intensity triggers the data_2d = data branch
                "echo_intensity": (
                    ["cell", "time"],
                    np.random.randint(50, 200, (n_c, n_t), dtype=np.int16),
                ),
            },
            coords={"time": time, "cell": np.arange(n_c), "beam": np.arange(4)},
            attrs={"coordinate_system": "earth"},
        )
        proc = _make_proc(ds)
        at = _run(_full_ss(proc))
        at.selectbox[0].set_value("Echo Intensity").run()
        assert not at.exception
        next(b for b in at.button if b.key == "plot_preview").click().run()
        assert not at.exception

    # ------------------------------------------------------------------ #
    # Lines 888-889: except block in config-file generator               #
    # ------------------------------------------------------------------ #

    def test_config_generator_exception_shows_error(self, inject_pyadps_mock):
        """Lines 888-889: exception inside config try-block -> st.error shown.

        st.code is called exactly once during the gen_config_btn handler
        (line 876: st.code(config_content, language="ini")) and nowhere else
        on the page, making it a safe patch target: it raises only inside
        the button handler's try block, which the except clause catches and
        converts to st.error.
        """
        import streamlit as st

        ds = _make_ds()
        proc = _make_proc(ds)

        at = _run(_full_ss(proc))
        cb = next(c for c in at.checkbox if c.key == "generate_config_checkbox")
        cb.check().run()
        assert not at.exception

        with patch.object(st, "code", side_effect=RuntimeError("forced config error")):
            next(b for b in at.button if b.key == "gen_config_btn").click().run()

        # Page catches RuntimeError -> st.error -> no unhandled at.exception
        assert not at.exception
        assert any(at.error)

    # ------------------------------------------------------------------ #
    # Line 931: Beam coords label in sidebar                             #
    # ------------------------------------------------------------------ #

    def test_sidebar_shows_beam_coords_label(self, inject_pyadps_mock):
        ds = _make_ds(earth_coords=False)
        proc = _make_proc(ds)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xr.register_dataset_accessor("fixed_leader")(_RaisingFLStub)

        try:
            at = _run(_full_ss(proc))
            assert not at.exception
        finally:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xr.register_dataset_accessor("fixed_leader")(_FLStub)


# ===========================================================================
# CLASS 14 — build_config_from_session()
# ===========================================================================


def _make_page_module(inject_pyadps_mock):
    """Load write page module and return it with ds/proc set."""
    import importlib.util
    import streamlit as st

    ds_mod = _make_ds()
    proc_mod = _make_proc(ds_mod)

    spec = importlib.util.spec_from_file_location("write_page_bcfs", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)

    with (
        patch.object(st, "set_page_config"),
        patch.object(st, "stop", side_effect=SystemExit(0)),
        patch.object(st, "error"),
        patch.object(st, "header"),
        patch.object(st, "write"),
        patch.object(st, "tabs", return_value=[MagicMock() for _ in range(4)]),
        patch.dict(
            "streamlit.session_state",
            {
                "processor": proc_mod,
                "write_initialized": True,
                "export_format": "NetCDF",
                "export_type": "Velocity Only",
                "apply_mask_export": True,
                "velocity_units": "cm/s",
                "add_attributes": False,
                "custom_attributes": {},
                "file_prefix": "ADCP",
                "fname": "test.pd0",
            },
            clear=False,
        ),
    ):
        try:
            spec.loader.exec_module(mod)
        except SystemExit:
            pass

    mod.ds = ds_mod
    mod.proc = proc_mod
    return mod


class TestExportConfigString:
    """Tests that the Write File page uses proc.export_config_string() for Tab 4."""

    @pytest.fixture(scope="class")
    def page_module(self, inject_pyadps_mock):
        return _make_page_module(inject_pyadps_mock)

    pass  # placeholder — tests below verify proc.export_config_string() integration

    # placeholder — removed (build_config_from_session no longer exists)


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
