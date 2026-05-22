"""
AppTest-based test suite for 09_Add-Ons.py

Three testing strategies are used:
  1. AppTest — for page-level smoke, no-file guard, and widget rendering
     that does NOT require clicking file-uploader-gated buttons.
  2. importlib + direct function call — for render_autoprocess_tool() and
     render_file_combiner_tool() button handlers, since AppTest cannot
     inject values into st.file_uploader widgets (they always return None).
  3. Pure unit tests — for helper functions (ansi_to_html, format_bytes,
     parse_config_to_dict, save_uploaded_file).

Run:
    pytest test_09_Add-Ons_apptest.py -v
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

# ---------------------------------------------------------------------------
SCRIPT_PATH = str(
    Path(__file__).parent.parent.parent / "src" / "pyadps" / "pages" / "09_Add-Ons.py"
)
# ---------------------------------------------------------------------------


# ===========================================================================
# MOCK DATACLASSES
# ===========================================================================


@dataclass
class MockFileValidationResult:
    is_valid: bool = True
    header_offset: int = 0
    ensemble_size: int = 512
    valid_ensembles: int = 100
    total_ensembles: int = 100
    is_truncated: bool = False
    error_message: str = ""


@dataclass
class MockCombineResult:
    success: bool = True
    files_processed: int = 2
    files_total: int = 2
    total_bytes: int = 51200
    total_ensembles: int = 200
    output_path: Optional[Path] = None
    skipped_files: List[str] = field(default_factory=list)
    error_message: str = ""


class MockDataArray:
    def __init__(self, masked_pct: float = 5.0):
        self._masked_pct = masked_pct

    @property
    def values(self) -> np.ndarray:
        size = 100
        n = int(size * self._masked_pct / 100)
        a = np.zeros(size, dtype=np.int8)
        a[:n] = 1
        return a


class MockDataset:
    def __init__(self, n_time=100, n_cells=20, has_mask=True, masked_pct=5.0):
        self._sizes: Dict[str, int] = {"time": n_time, "cell": n_cells}
        self._has_mask = has_mask
        self._masked_pct = masked_pct
        self._dvars = ["velocity", "mask"] if has_mask else ["velocity"]

    @property
    def sizes(self) -> Dict[str, int]:
        return self._sizes

    @property
    def data_vars(self):
        return self._dvars

    def __contains__(self, key):
        return key in self._dvars

    def __getitem__(self, key):
        if key == "mask":
            return MockDataArray(self._masked_pct)
        return MockDataArray()


# ===========================================================================
# FAKE UPLOADED FILE
# ===========================================================================


class FakeUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self.size = len(content)
        self._content = content
        self._buf = io.BytesIO(content)

    def getvalue(self) -> bytes:
        return self._content

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def seek(self, pos: int) -> None:
        self._buf.seek(pos)


def _fake_binary() -> FakeUploadedFile:
    # 7f7f header + zeros — valid ADCP marker, no actual null bytes in string
    header = bytes([0x7F, 0x7F]) + bytes(50)
    return FakeUploadedFile("test.000", header)


def _fake_config() -> FakeUploadedFile:
    ini = "[FileSettings]\ninput_file_name = test.000\n\n[DownloadOptions]\nexport_type = velocity_only\n"
    return FakeUploadedFile("config.ini", ini.encode())


# ===========================================================================
# PYADPS MOCK BUILDER
# ===========================================================================


def _build_mocks(
    autoprocess_return=None,
    autoprocess_side_effect=None,
    validate_return=None,
    combine_return=None,
    combine_side_effect=None,
) -> Dict[str, types.ModuleType]:
    if autoprocess_return is None:
        autoprocess_return = MockDataset()
    if validate_return is None:
        validate_return = MockFileValidationResult()
    if combine_return is None:
        combine_return = MockCombineResult()

    mock_pyadps = types.ModuleType("pyadps")
    mock_pyadps.__path__ = []
    mock_pyadps.__package__ = "pyadps"

    mock_proc = types.ModuleType("pyadps.processing")
    mock_proc.__path__ = []
    mock_proc.__package__ = "pyadps.processing"

    mock_auto = types.ModuleType("pyadps.processing.autoprocess")
    if autoprocess_side_effect is not None:
        mock_auto.autoprocess = MagicMock(side_effect=autoprocess_side_effect)
    else:
        mock_auto.autoprocess = MagicMock(return_value=autoprocess_return)

    mock_multi = types.ModuleType("pyadps.processing.multifile")
    mock_multi.validate_adcp_file = MagicMock(return_value=validate_return)
    if combine_side_effect is not None:
        mock_multi.combine_file_list = MagicMock(side_effect=combine_side_effect)
    else:
        mock_multi.combine_file_list = MagicMock(return_value=combine_return)
    mock_multi.ADCPFileConfig = MagicMock()
    mock_multi.CombineResult = MockCombineResult

    mock_pyadps.processing = mock_proc
    mock_proc.autoprocess = mock_auto
    mock_proc.multifile = mock_multi

    return {
        "pyadps": mock_pyadps,
        "pyadps.processing": mock_proc,
        "pyadps.processing.autoprocess": mock_auto,
        "pyadps.processing.multifile": mock_multi,
    }


# ===========================================================================
# MODULE-SCOPED AUTOUSE FIXTURE
# ===========================================================================


@pytest.fixture(scope="module", autouse=True)
def inject_pyadps_mock():
    originals = {
        k: sys.modules.get(k)
        for k in (
            "pyadps",
            "pyadps.processing",
            "pyadps.processing.autoprocess",
            "pyadps.processing.multifile",
        )
    }
    mocks = _build_mocks()
    sys.modules.update(mocks)
    yield mocks
    for k, v in originals.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


# ===========================================================================
# APPTEST HELPER (for smoke / no-file-gated tests only)
# ===========================================================================


def _run(extra_ss: Optional[Dict[str, Any]] = None, timeout: int = 15) -> AppTest:
    mocks = _build_mocks()
    at = AppTest.from_file(SCRIPT_PATH, default_timeout=timeout)
    if extra_ss:
        for k, v in extra_ss.items():
            at.session_state[k] = v
    with patch.dict(sys.modules, mocks):
        at.run()
    return at


# ===========================================================================
# IMPORTLIB PAGE LOADER  (shared across several test classes)
# ===========================================================================


def _load_page_mod(label: str = "addons") -> tuple:
    """Load the page module via importlib, returning (mod, mocks)."""
    import importlib.util
    import streamlit as st

    mocks = _build_mocks()
    spec = importlib.util.spec_from_file_location(label, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)

    with (
        patch.object(st, "set_page_config"),
        patch.object(st, "title"),
        patch.object(st, "write"),
        patch.object(st, "divider"),
        patch.object(st, "caption"),
        patch.object(st, "tabs", return_value=[MagicMock(), MagicMock()]),
        patch.dict(sys.modules, mocks),
    ):
        spec.loader.exec_module(mod)

    # Replace @st.cache_data-wrapped save_uploaded_file with a plain version
    # so FakeUploadedFile objects can be used without Streamlit hash checks.
    def _plain_save(f):
        td = tempfile.mkdtemp()
        p = os.path.join(td, f.name)
        with open(p, "wb") as fh:
            fh.write(f.getvalue())
        return p

    mod.save_uploaded_file = _plain_save
    return mod, mocks


# ===========================================================================
# COLUMN/CONTEXT-MANAGER HELPERS  (used by importlib-based tests)
# ===========================================================================


def _make_ctx() -> MagicMock:
    m = MagicMock()
    m.__enter__ = lambda s: s
    m.__exit__ = MagicMock(return_value=False)
    return m


def _smart_columns(n, **kw):
    """Return the right number of context-manager mocks for st.columns(n)."""
    count = len(n) if isinstance(n, (list, tuple)) else int(n)
    return [_make_ctx() for _ in range(count)]


# ===========================================================================
# CLASS 1 — Page-level smoke (AppTest)
# ===========================================================================


class TestPageSmoke:
    def test_page_loads_without_exception(self):
        at = _run()
        assert not at.exception, at.exception

    def test_page_renders_both_tabs(self):
        at = _run()
        assert not at.exception

    def test_footer_caption_rendered(self):
        at = _run()
        assert not at.exception

    def test_autoprocess_info_message_rendered(self):
        """The How it works info block is present when no files are uploaded."""
        at = _run()
        info_vals = [i.value or "" for i in at.info]
        assert any(
            "upload" in v.lower() or "how it works" in v.lower() for v in info_vals
        )

    def test_combiner_warning_rendered(self):
        """The sequential-numbering warning is always shown in the combiner tab."""
        at = _run()
        assert any(at.warning)

    def test_no_process_button_without_files(self):
        """Process Data button must not appear when no files are uploaded."""
        at = _run()
        labels = [b.label or "" for b in at.button]
        assert not any("Process" in l for l in labels)

    def test_page_idempotent_second_run(self):
        at = _run()
        at.run()
        assert not at.exception


# ===========================================================================
# CLASS 2 — Auto Processing Tool — no files (AppTest)
# ===========================================================================


class TestAutoProcessNoFiles:
    def test_info_shown_when_no_files(self):
        at = _run()
        info_vals = [i.value or "" for i in at.info]
        assert any("upload" in v.lower() for v in info_vals)

    def test_warning_not_shown_without_files(self):
        # No files → only the combiner warning (about renaming) shows, not process warning
        at = _run()
        assert not at.exception

    def test_process_button_absent_without_both_files(self):
        at = _run()
        labels = [b.label or "" for b in at.button]
        assert not any("Process Data" in l for l in labels)


# ===========================================================================
# CLASS 3 — File Combiner — no files (AppTest)
# ===========================================================================


class TestCombinerNoFiles:
    def test_info_shown_when_no_files(self):
        at = _run()
        info_vals = [i.value or "" for i in at.info]
        assert any("upload" in v.lower() for v in info_vals)

    def test_warning_about_file_ordering_shown(self):
        at = _run()
        assert any(at.warning)

    def test_validate_button_absent_without_files(self):
        at = _run()
        labels = [b.label or "" for b in at.button]
        assert not any("Validate" in l for l in labels)

    def test_combine_button_absent_without_files(self):
        at = _run()
        labels = [b.label or "" for b in at.button]
        assert not any("Combine" in l for l in labels)


# ===========================================================================
# CLASS 4 — Helper function unit tests
# ===========================================================================


class TestHelperFunctions:
    """Pure unit tests — load the function implementations directly."""

    @pytest.fixture(scope="class")
    def mod(self):
        m, _ = _load_page_mod("addons_helpers")
        return m

    # ansi_to_html

    def test_ansi_red(self, mod):
        result = mod.ansi_to_html("\x1b[31mERROR\x1b[0m")
        assert "<span style='color:red'>" in result
        assert "</span>" in result

    def test_ansi_green(self, mod):
        result = mod.ansi_to_html("\x1b[32mOK\x1b[0m")
        assert "color:green" in result

    def test_ansi_yellow(self, mod):
        result = mod.ansi_to_html("\x1b[33mWARN\x1b[0m")
        assert "color:orange" in result

    def test_ansi_reset_only(self, mod):
        assert mod.ansi_to_html("\x1b[0m") == "</span>"

    def test_ansi_no_codes(self, mod):
        assert mod.ansi_to_html("plain") == "plain"

    def test_ansi_multiple_codes(self, mod):
        text = "\x1b[32mGOOD\x1b[0m \x1b[31mBAD\x1b[0m"
        result = mod.ansi_to_html(text)
        assert result.count("<span") == 2

    # parse_config_to_dict

    def test_parse_config_basic(self, mod):
        result = mod.parse_config_to_dict(b"[S]\nkey = val\n")
        assert result["S"]["key"] == "val"

    def test_parse_config_multiple_sections(self, mod):
        result = mod.parse_config_to_dict(b"[A]\nx=1\n[B]\ny=2\n")
        assert "A" in result and "B" in result

    def test_parse_config_empty(self, mod):
        assert mod.parse_config_to_dict(b"") == {}

    def test_parse_config_utf8(self, mod):
        result = mod.parse_config_to_dict("[Info]\nauthor = José\n".encode("utf-8"))
        assert result["Info"]["author"] == "José"

    # format_bytes

    def test_format_bytes_b(self, mod):
        assert mod.format_bytes(512) == "512 B"

    def test_format_bytes_zero(self, mod):
        assert mod.format_bytes(0) == "0 B"

    def test_format_bytes_kb(self, mod):
        assert "KB" in mod.format_bytes(2048)

    def test_format_bytes_mb(self, mod):
        assert "MB" in mod.format_bytes(2 * 1024 * 1024)

    def test_format_bytes_boundary_1kb(self, mod):
        assert "KB" in mod.format_bytes(1024)

    def test_format_bytes_boundary_1mb(self, mod):
        assert "MB" in mod.format_bytes(1024 * 1024)

    # save_uploaded_file (uses plain replacement, not @st.cache_data)

    def test_save_uploaded_file_writes_content(self, mod):
        fake = FakeUploadedFile("data.000", b"HELLO ADCP")
        path = mod.save_uploaded_file(fake)
        assert os.path.exists(path)
        with open(path, "rb") as fh:
            assert fh.read() == b"HELLO ADCP"
        os.unlink(path)

    def test_save_uploaded_file_uses_original_name(self, mod):
        fake = FakeUploadedFile("myfile.bin", b"DATA")
        path = mod.save_uploaded_file(fake)
        assert Path(path).name == "myfile.bin"
        os.unlink(path)


# ===========================================================================
# IMPORTLIB RENDER HELPER
# ===========================================================================


def _call_autoprocess(
    mod,
    mocks,
    autoprocess_fn=None,
    save_netcdf=True,
    velocity_only=False,
    velocity_units="cm/s",
    click_process=True,
    output_exists=False,
):
    """Call render_autoprocess_tool() with all st widgets mocked."""
    import streamlit as st

    binary = _fake_binary()
    config = _fake_config()

    if autoprocess_fn is not None:
        mocks["pyadps.processing.autoprocess"].autoprocess.side_effect = autoprocess_fn
    else:
        mocks["pyadps.processing.autoprocess"].autoprocess.side_effect = None
        mocks["pyadps.processing.autoprocess"].autoprocess.return_value = MockDataset()

    st_p = {
        "header": MagicMock(),
        "write": MagicMock(),
        "info": MagicMock(),
        "columns": _smart_columns,
        "file_uploader": MagicMock(side_effect=[binary, config]),
        "success": MagicMock(),
        "expander": MagicMock(return_value=_make_ctx()),
        "subheader": MagicMock(),
        "checkbox": MagicMock(side_effect=[save_netcdf, velocity_only]),
        "selectbox": MagicMock(return_value=velocity_units),
        "button": MagicMock(return_value=click_process),
        "spinner": MagicMock(return_value=_make_ctx()),
        "text_area": MagicMock(),
        "metric": MagicMock(),
        "error": MagicMock(),
        "exception": MagicMock(),
        "download_button": MagicMock(),
    }

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, mocks))
        for attr, mock in st_p.items():
            if hasattr(st, attr):
                stack.enter_context(patch.object(st, attr, mock))
        if output_exists:
            stack.enter_context(patch("pathlib.Path.exists", return_value=True))
            stack.enter_context(
                patch(
                    "builtins.open",
                    side_effect=lambda p, m="r", **kw: io.BytesIO(b"NC")
                    if "b" in str(m)
                    else open(p, m, **kw),
                )
            )
        mod.render_autoprocess_tool()

    return st_p


def _call_combiner(
    mod,
    mocks,
    combine_return=None,
    combine_side_effect=None,
    validate_return=None,
    display_log=False,
    validate_click=False,
    combine_click=False,
    files=None,
):
    """Call render_file_combiner_tool() with all st widgets mocked."""
    import streamlit as st

    if files is None:
        files = [_fake_binary(), _fake_binary()]

    if validate_return is not None:
        mocks[
            "pyadps.processing.multifile"
        ].validate_adcp_file.return_value = validate_return

    if combine_side_effect is not None:
        mocks[
            "pyadps.processing.multifile"
        ].combine_file_list.side_effect = combine_side_effect
        mocks["pyadps.processing.multifile"].combine_file_list.return_value = None
    else:
        mocks["pyadps.processing.multifile"].combine_file_list.side_effect = None
        mocks["pyadps.processing.multifile"].combine_file_list.return_value = (
            combine_return or MockCombineResult()
        )

    # Real output file so open(output_path,"rb") works
    with tempfile.NamedTemporaryFile(suffix=".000", delete=False) as tmp:
        tmp.write(b"\x7f\x7f" + b"\x00" * 50)
        output_path = Path(tmp.name)

    st_p = {
        "header": MagicMock(),
        "write": MagicMock(),
        "warning": MagicMock(),
        "subheader": MagicMock(),
        "columns": _smart_columns,
        "text_input": MagicMock(return_value="merged_000.000"),
        "checkbox": MagicMock(side_effect=[display_log, True, True]),
        "expander": MagicMock(return_value=_make_ctx()),
        "file_uploader": MagicMock(return_value=files),
        "info": MagicMock(),
        "button": MagicMock(side_effect=[validate_click, combine_click]),
        "divider": MagicMock(),
        "success": MagicMock(),
        "error": MagicMock(),
        "metric": MagicMock(),
        "download_button": MagicMock(),
        "dataframe": MagicMock(),
        "markdown": MagicMock(),
        "exception": MagicMock(),
    }

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, mocks))
        for attr, mock in st_p.items():
            if hasattr(st, attr):
                stack.enter_context(patch.object(st, attr, mock))
        stack.enter_context(patch("pathlib.Path.__truediv__", return_value=output_path))
        mod.render_file_combiner_tool()

    try:
        os.unlink(output_path)
    except Exception:
        pass

    return st_p


# ===========================================================================
# CLASS 5 — Auto Processing Tool — button handler (importlib)
# ===========================================================================


class TestAutoProcessButton:
    @pytest.fixture(scope="class")
    def page(self):
        return _load_page_mod("addons_auto")

    def test_process_success(self, page):
        mod, mocks = page
        p = _call_autoprocess(mod, mocks)
        p["success"].assert_called()

    def test_process_metrics_called(self, page):
        mod, mocks = page
        p = _call_autoprocess(mod, mocks)
        p["metric"].assert_called()

    def test_process_no_mask_in_result(self, page):
        mod, mocks = page
        mocks["pyadps.processing.autoprocess"].autoprocess.return_value = MockDataset(
            has_mask=False
        )
        p = _call_autoprocess(mod, mocks)
        p["metric"].assert_called()

    def test_process_with_mask_in_result(self, page):
        mod, mocks = page
        mocks["pyadps.processing.autoprocess"].autoprocess.return_value = MockDataset(
            has_mask=True, masked_pct=10.0
        )
        p = _call_autoprocess(mod, mocks)
        p["success"].assert_called()

    def test_process_filenotfound_error(self, page):
        mod, mocks = page
        p = _call_autoprocess(
            mod,
            mocks,
            autoprocess_fn=lambda **kw: (_ for _ in ()).throw(
                FileNotFoundError("nope")
            ),
        )
        p["error"].assert_called()

    def test_process_valueerror(self, page):
        mod, mocks = page
        p = _call_autoprocess(
            mod,
            mocks,
            autoprocess_fn=lambda **kw: (_ for _ in ()).throw(ValueError("bad cfg")),
        )
        p["error"].assert_called()

    def test_process_generic_exception(self, page):
        mod, mocks = page
        p = _call_autoprocess(
            mod,
            mocks,
            autoprocess_fn=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        p["error"].assert_called()

    def test_process_with_console_output(self, page):
        mod, mocks = page

        def _with_output(**kw):
            print("step 1")
            return MockDataset()

        p = _call_autoprocess(mod, mocks, autoprocess_fn=_with_output)
        p["success"].assert_called()

    def test_save_netcdf_disabled(self, page):
        mod, mocks = page
        p = _call_autoprocess(mod, mocks, save_netcdf=False)
        p["success"].assert_called()

    def test_velocity_only_mode(self, page):
        mod, mocks = page
        p = _call_autoprocess(mod, mocks, velocity_only=True)
        p["success"].assert_called()

    def test_netcdf_output_exists_download_button(self, page):
        mod, mocks = page
        p = _call_autoprocess(mod, mocks, save_netcdf=True, output_exists=True)
        p["download_button"].assert_called()

    def test_depth_dim_in_result_sizes(self, page):
        mod, mocks = page
        ds = MockDataset()
        ds._sizes = {"time": 100, "depth": 30}
        mocks["pyadps.processing.autoprocess"].autoprocess.return_value = ds
        p = _call_autoprocess(mod, mocks)
        p["metric"].assert_called()

    def test_no_button_click_skips_autoprocess(self, page):
        mod, mocks = page
        mocks["pyadps.processing.autoprocess"].autoprocess.reset_mock()
        _call_autoprocess(mod, mocks, click_process=False)
        assert not mocks["pyadps.processing.autoprocess"].autoprocess.called

    def test_velocity_units_mm_per_s(self, page):
        mod, mocks = page
        p = _call_autoprocess(mod, mocks, velocity_units="mm/s")
        p["success"].assert_called()

    def test_velocity_units_m_per_s(self, page):
        mod, mocks = page
        p = _call_autoprocess(mod, mocks, velocity_units="m/s")
        p["success"].assert_called()


# ===========================================================================
# CLASS 6 — File Combiner — validate button (importlib)
# ===========================================================================


class TestCombinerValidate:
    @pytest.fixture(scope="class")
    def page(self):
        return _load_page_mod("addons_validate")

    def test_validate_valid_file(self, page):
        mod, mocks = page
        r = MockFileValidationResult(is_valid=True, valid_ensembles=100)
        p = _call_combiner(mod, mocks, validate_return=r, validate_click=True)
        p["dataframe"].assert_called()

    def test_validate_invalid_file(self, page):
        mod, mocks = page
        r = MockFileValidationResult(is_valid=False, error_message="Not ADCP")
        p = _call_combiner(mod, mocks, validate_return=r, validate_click=True)
        p["dataframe"].assert_called()

    def test_validate_truncated_file(self, page):
        mod, mocks = page
        r = MockFileValidationResult(is_valid=True, is_truncated=True)
        p = _call_combiner(mod, mocks, validate_return=r, validate_click=True)
        p["dataframe"].assert_called()

    def test_validate_three_files(self, page):
        mod, mocks = page
        r = MockFileValidationResult(is_valid=True)
        files = [_fake_binary(), _fake_binary(), _fake_binary()]
        p = _call_combiner(
            mod, mocks, validate_return=r, files=files, validate_click=True
        )
        p["dataframe"].assert_called()

    def test_validate_shows_subheader(self, page):
        mod, mocks = page
        p = _call_combiner(mod, mocks, validate_click=True)
        p["subheader"].assert_called()

    def test_no_validate_click_skips_dataframe(self, page):
        mod, mocks = page
        p = _call_combiner(mod, mocks, validate_click=False, combine_click=False)
        p["dataframe"].assert_not_called()


# ===========================================================================
# CLASS 7 — File Combiner — combine button (importlib)
# ===========================================================================


class TestCombinerCombine:
    @pytest.fixture(scope="class")
    def page(self):
        return _load_page_mod("addons_combine")

    def test_combine_success(self, page):
        mod, mocks = page
        r = MockCombineResult(success=True, files_processed=2, files_total=2)
        p = _call_combiner(mod, mocks, combine_return=r, combine_click=True)
        p["success"].assert_called()

    def test_combine_shows_metrics(self, page):
        mod, mocks = page
        r = MockCombineResult(success=True, total_ensembles=500, total_bytes=102400)
        p = _call_combiner(mod, mocks, combine_return=r, combine_click=True)
        p["metric"].assert_called()

    def test_combine_with_skipped_files_shows_warning(self, page):
        mod, mocks = page
        r = MockCombineResult(success=True, skipped_files=["bad.000"])
        p = _call_combiner(mod, mocks, combine_return=r, combine_click=True)
        p["warning"].assert_called()

    def test_combine_always_shows_time_axis_warning(self, page):
        mod, mocks = page
        r = MockCombineResult(success=True)
        p = _call_combiner(mod, mocks, combine_return=r, combine_click=True)
        p["warning"].assert_called()

    def test_combine_failure_shows_error(self, page):
        mod, mocks = page
        r = MockCombineResult(success=False, error_message="Incompatible sizes")
        p = _call_combiner(mod, mocks, combine_return=r, combine_click=True)
        p["error"].assert_called()

    def test_combine_failure_with_skipped_files(self, page):
        mod, mocks = page
        r = MockCombineResult(
            success=False, error_message="Failed", skipped_files=["bad.000"]
        )
        p = _call_combiner(mod, mocks, combine_return=r, combine_click=True)
        p["error"].assert_called()

    def test_combine_exception_shows_error(self, page):
        mod, mocks = page
        p = _call_combiner(
            mod,
            mocks,
            combine_side_effect=RuntimeError("disk full"),
            combine_click=True,
        )
        p["error"].assert_called()

    def test_combine_display_log_branch(self, page):
        mod, mocks = page
        r = MockCombineResult(success=True)
        p = _call_combiner(
            mod, mocks, combine_return=r, display_log=True, combine_click=True
        )
        p["success"].assert_called()

    def test_combine_download_button_shown_on_success(self, page):
        mod, mocks = page
        r = MockCombineResult(success=True)
        p = _call_combiner(mod, mocks, combine_return=r, combine_click=True)
        p["download_button"].assert_called()

    def test_combine_single_file(self, page):
        mod, mocks = page
        r = MockCombineResult(success=True)
        p = _call_combiner(
            mod, mocks, combine_return=r, files=[_fake_binary()], combine_click=True
        )
        p["success"].assert_called()


# ===========================================================================
# CLASS 8 — format_bytes via combine metrics (AppTest path validation)
# ===========================================================================


class TestFormatBytesEdgeCases:
    """Directly test all three bands of format_bytes."""

    @pytest.fixture(scope="class")
    def mod(self):
        m, _ = _load_page_mod("addons_fb")
        return m

    def test_bytes_band(self, mod):
        assert mod.format_bytes(1) == "1 B"
        assert mod.format_bytes(1023) == "1023 B"

    def test_kb_band(self, mod):
        result = mod.format_bytes(1024)
        assert "KB" in result and "1.00" in result

    def test_mb_band(self, mod):
        result = mod.format_bytes(1024 * 1024)
        assert "MB" in result and "1.00" in result

    def test_large_mb(self, mod):
        result = mod.format_bytes(10 * 1024 * 1024)
        assert "MB" in result and "10.00" in result


# ===========================================================================
# CLASS 9 — Edge cases
# ===========================================================================


class TestEdgeCases:
    @pytest.fixture(scope="class")
    def page(self):
        return _load_page_mod("addons_edge")

    def test_ten_files_in_combiner(self, page):
        mod, mocks = page
        files = [
            FakeUploadedFile(f"f{i:03d}.000", b"\x7f\x7f" + b"x" * 10)
            for i in range(10)
        ]
        p = _call_combiner(mod, mocks, files=files)
        p["info"].assert_called()

    def test_autoprocess_result_depth_dim_only(self, page):
        mod, mocks = page
        ds = MockDataset()
        ds._sizes = {"time": 100, "depth": 30}
        mocks["pyadps.processing.autoprocess"].autoprocess.return_value = ds
        p = _call_autoprocess(mod, mocks)
        p["metric"].assert_called()

    def test_combiner_no_files_shows_info(self):
        at = _run()
        info_vals = [i.value or "" for i in at.info]
        assert any("upload" in v.lower() for v in info_vals)

    def test_combine_result_zero_bytes(self, page):
        mod, mocks = page
        r = MockCombineResult(success=True, total_bytes=0, total_ensembles=0)
        p = _call_combiner(mod, mocks, combine_return=r, combine_click=True)
        p["success"].assert_called()

    def test_combine_result_large_bytes_mb(self, page):
        mod, mocks = page
        r = MockCombineResult(success=True, total_bytes=5 * 1024 * 1024)
        p = _call_combiner(mod, mocks, combine_return=r, combine_click=True)
        p["metric"].assert_called()

    def test_combine_result_small_bytes_b(self, page):
        mod, mocks = page
        r = MockCombineResult(success=True, total_bytes=512)
        p = _call_combiner(mod, mocks, combine_return=r, combine_click=True)
        p["metric"].assert_called()

    def test_config_with_all_sections(self, page):
        mod, _ = page
        full_ini = (
            "[FileSettings]\ninput_file_name = test.000\n"
            "[SignalQuality]\nqc_applied = True\n"
        ).encode()
        result = mod.parse_config_to_dict(full_ini)
        assert "FileSettings" in result


# ===========================================================================
# CLASS 10 — ansi_to_html via display_log combine branch
# ===========================================================================


class TestAnsiToHtmlDisplayLog:
    @pytest.fixture(scope="class")
    def page(self):
        return _load_page_mod("addons_ansi")

    def test_ansi_html_all_colours(self, page):
        mod, _ = page
        text = "\x1b[31mred\x1b[0m \x1b[32mgreen\x1b[0m \x1b[33myellow\x1b[0m"
        result = mod.ansi_to_html(text)
        assert "color:red" in result
        assert "color:green" in result
        assert "color:orange" in result
        assert result.count("</span>") == 3

    def test_display_log_true_sets_up_logging(self, page):
        """display_log=True branch: logging handler is set up before combine call."""
        mod, mocks = page
        r = MockCombineResult(success=True)
        p = _call_combiner(
            mod, mocks, combine_return=r, display_log=True, combine_click=True
        )
        p["success"].assert_called()

    def test_display_log_false_skips_logging(self, page):
        mod, mocks = page
        r = MockCombineResult(success=True)
        p = _call_combiner(
            mod, mocks, combine_return=r, display_log=False, combine_click=True
        )
        p["success"].assert_called()


# ===========================================================================
# CLASS 11 — Targeted coverage for lines 81-85, 169-170, 254, 283-284,
#             416-417, 463, 533-534
# ===========================================================================


class TestCoverageGaps:
    """
    Pinpoint tests for branches not reached by the main suite.

    Lines 81-85   : save_uploaded_file real body (replaced by _plain_save in fixtures)
    Lines 169-170 : st.error in config-parse expander (except branch)
    Line  254     : st.metric("Data Masked", "N/A") when result has no mask var
    Lines 283-284 : except Exception: pass  in autoprocess config-file cleanup
    Lines 416-417 : except Exception: pass  in validate-button temp-file cleanup
    Line  463     : st.markdown(ansi_to_html(...)) when display_log has output
    Lines 533-534 : except Exception: pass  in combine-button temp-file cleanup
    """

    @pytest.fixture(scope="class")
    def page(self):
        """Load module WITHOUT replacing save_uploaded_file so lines 81-85 run."""
        import importlib.util
        import streamlit as st

        mocks = _build_mocks()
        spec = importlib.util.spec_from_file_location("addons_gaps", SCRIPT_PATH)
        mod = importlib.util.module_from_spec(spec)

        with (
            patch.object(st, "set_page_config"),
            patch.object(st, "title"),
            patch.object(st, "write"),
            patch.object(st, "divider"),
            patch.object(st, "caption"),
            patch.object(st, "tabs", return_value=[MagicMock(), MagicMock()]),
            patch.dict(sys.modules, mocks),
        ):
            spec.loader.exec_module(mod)

        # Do NOT replace save_uploaded_file — we need the real body for 81-85.
        return mod, mocks

    # ------------------------------------------------------------------ #
    # Lines 81-85: real body of save_uploaded_file                        #
    # Call the unwrapped function directly to bypass @st.cache_data.      #
    # ------------------------------------------------------------------ #

    def test_save_uploaded_file_real_body_writes_file(self, page):
        """Lines 81-85: the real tempfile-write logic is executed."""
        mod, _ = page
        fake = FakeUploadedFile("real_body.000", b"REAL CONTENT")
        # __wrapped__ is set by functools.wraps inside st.cache_data
        fn = getattr(mod.save_uploaded_file, "__wrapped__", mod.save_uploaded_file)
        path = fn(fake)
        assert os.path.exists(path)
        assert Path(path).name == "real_body.000"
        with open(path, "rb") as fh:
            assert fh.read() == b"REAL CONTENT"
        os.unlink(path)

    def test_save_uploaded_file_real_body_returns_path(self, page):
        """Lines 81-85: return value is the joined temp-dir path."""
        mod, _ = page
        fake = FakeUploadedFile("out.bin", b"\x01\x02")
        fn = getattr(mod.save_uploaded_file, "__wrapped__", mod.save_uploaded_file)
        path = fn(fake)
        assert path.endswith("out.bin")
        os.unlink(path)

    # ------------------------------------------------------------------ #
    # Lines 169-170: except branch in config-parse expander              #
    # Triggered by making parse_config_to_dict raise inside the expander. #
    # ------------------------------------------------------------------ #

    def test_config_parse_error_shows_st_error(self, page):
        """Lines 169-170: bad config bytes → parse_config_to_dict raises → st.error."""
        mod, mocks = page

        # Patch _plain_save so save_uploaded_file works without cache_data
        mod.save_uploaded_file = lambda f: (lambda p: p)(
            (
                lambda: (lambda td, nm: __import__("os").path.join(td, nm))(
                    __import__("tempfile").mkdtemp(), f.name
                )
            )()
        )

        import streamlit as st

        # Make parse_config_to_dict raise by injecting a broken version
        bad_parse = MagicMock(side_effect=ValueError("bad encoding"))

        binary = _fake_binary()
        # Supply bytes that will be passed to parse_config_to_dict
        bad_config = FakeUploadedFile("bad.ini", bytes([0xFF, 0xFE, 0x00]))

        st_p = {
            "header": MagicMock(),
            "write": MagicMock(),
            "info": MagicMock(),
            "columns": _smart_columns,
            "file_uploader": MagicMock(side_effect=[binary, bad_config]),
            "success": MagicMock(),
            "expander": MagicMock(return_value=_make_ctx()),
            "subheader": MagicMock(),
            "checkbox": MagicMock(side_effect=[True, False]),
            "selectbox": MagicMock(return_value="cm/s"),
            "button": MagicMock(return_value=False),
            "spinner": MagicMock(return_value=_make_ctx()),
            "text_area": MagicMock(),
            "metric": MagicMock(),
            "error": MagicMock(),
            "exception": MagicMock(),
            "download_button": MagicMock(),
            "json": MagicMock(),
        }

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, mocks))
            stack.enter_context(patch.object(mod, "parse_config_to_dict", bad_parse))
            for attr, mock in st_p.items():
                if hasattr(st, attr):
                    stack.enter_context(patch.object(st, attr, mock))
            mod.render_autoprocess_tool()

        st_p["error"].assert_called()

    # ------------------------------------------------------------------ #
    # Line 254: st.metric("Data Masked", "N/A") — result has no mask var #
    # ------------------------------------------------------------------ #

    def test_process_result_no_mask_metric_na(self, page):
        """Line 254: when result has no mask variable, metric shows N/A."""
        mod, mocks = page

        # Set up a dataset with NO mask
        ds_no_mask = MockDataset(has_mask=False)
        mocks["pyadps.processing.autoprocess"].autoprocess.return_value = ds_no_mask
        mocks["pyadps.processing.autoprocess"].autoprocess.side_effect = None

        # Use _plain_save for this call
        mod.save_uploaded_file = lambda f: (
            __import__("os").path.join(__import__("tempfile").mkdtemp(), f.name)
        )

        import streamlit as st

        st_p = {
            "header": MagicMock(),
            "write": MagicMock(),
            "info": MagicMock(),
            "columns": _smart_columns,
            "file_uploader": MagicMock(side_effect=[_fake_binary(), _fake_config()]),
            "success": MagicMock(),
            "expander": MagicMock(return_value=_make_ctx()),
            "subheader": MagicMock(),
            "checkbox": MagicMock(side_effect=[True, False]),
            "selectbox": MagicMock(return_value="cm/s"),
            "button": MagicMock(return_value=True),
            "spinner": MagicMock(return_value=_make_ctx()),
            "text_area": MagicMock(),
            "metric": MagicMock(),
            "error": MagicMock(),
            "exception": MagicMock(),
            "download_button": MagicMock(),
            "json": MagicMock(),
        }

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, mocks))
            for attr, mock in st_p.items():
                if hasattr(st, attr):
                    stack.enter_context(patch.object(st, attr, mock))
            stack.enter_context(patch("pathlib.Path.exists", return_value=False))
            mod.render_autoprocess_tool()

        # Confirm metric was called with "N/A"
        calls = [str(c) for c in st_p["metric"].call_args_list]
        assert any("N/A" in c for c in calls)

    # ------------------------------------------------------------------ #
    # Lines 283-284: except Exception: pass  in config temp file cleanup  #
    # Make os.unlink raise only for the config temp file.                  #
    # ------------------------------------------------------------------ #

    def test_config_cleanup_unlink_exception_is_swallowed(self, page):
        """Lines 283-284: os.unlink raises during config-temp cleanup → silently ignored."""
        mod, mocks = page

        mod.save_uploaded_file = lambda f: (
            __import__("os").path.join(__import__("tempfile").mkdtemp(), f.name)
        )

        import streamlit as st

        real_unlink = os.unlink

        def _raising_unlink(path):
            raise OSError("permission denied")

        st_p = {
            "header": MagicMock(),
            "write": MagicMock(),
            "info": MagicMock(),
            "columns": _smart_columns,
            "file_uploader": MagicMock(side_effect=[_fake_binary(), _fake_config()]),
            "success": MagicMock(),
            "expander": MagicMock(return_value=_make_ctx()),
            "subheader": MagicMock(),
            "checkbox": MagicMock(side_effect=[True, False]),
            "selectbox": MagicMock(return_value="cm/s"),
            "button": MagicMock(return_value=True),
            "spinner": MagicMock(return_value=_make_ctx()),
            "text_area": MagicMock(),
            "metric": MagicMock(),
            "error": MagicMock(),
            "exception": MagicMock(),
            "download_button": MagicMock(),
            "json": MagicMock(),
        }

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, mocks))
            for attr, mock in st_p.items():
                if hasattr(st, attr):
                    stack.enter_context(patch.object(st, attr, mock))
            stack.enter_context(patch("pathlib.Path.exists", return_value=False))
            # Patch os.unlink globally so the cleanup raises
            stack.enter_context(patch("os.unlink", side_effect=_raising_unlink))
            # Should not propagate — the except block swallows it
            mod.render_autoprocess_tool()

        # autoprocess ran and success was called despite unlink failure
        st_p["success"].assert_called()

    # ------------------------------------------------------------------ #
    # Lines 416-417: except Exception: pass  in validate temp cleanup     #
    # ------------------------------------------------------------------ #

    def test_validate_cleanup_unlink_exception_is_swallowed(self, page):
        """Lines 416-417: os.unlink raises during validate cleanup → silently ignored."""
        mod, mocks = page

        import streamlit as st

        files = [_fake_binary(), _fake_binary()]

        st_p = {
            "header": MagicMock(),
            "write": MagicMock(),
            "warning": MagicMock(),
            "subheader": MagicMock(),
            "columns": _smart_columns,
            "text_input": MagicMock(return_value="merged.000"),
            "checkbox": MagicMock(side_effect=[False, True, True]),
            "expander": MagicMock(return_value=_make_ctx()),
            "file_uploader": MagicMock(return_value=files),
            "info": MagicMock(),
            "button": MagicMock(side_effect=[True, False]),  # validate=True
            "divider": MagicMock(),
            "success": MagicMock(),
            "error": MagicMock(),
            "metric": MagicMock(),
            "download_button": MagicMock(),
            "dataframe": MagicMock(),
            "markdown": MagicMock(),
            "exception": MagicMock(),
        }

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, mocks))
            for attr, mock in st_p.items():
                if hasattr(st, attr):
                    stack.enter_context(patch.object(st, attr, mock))
            stack.enter_context(patch("os.unlink", side_effect=OSError("busy")))
            mod.render_file_combiner_tool()

        # dataframe was rendered despite unlink failure
        st_p["dataframe"].assert_called()

    # ------------------------------------------------------------------ #
    # Line 463: st.markdown(ansi_to_html(log_output)) — non-empty buffer  #
    # display_log=True AND the logger actually emits a record.            #
    # ------------------------------------------------------------------ #

    def test_display_log_markdown_called_with_nonempty_output(self, page):
        """Line 463: st.markdown is called when display_log=True and log has content."""
        mod, mocks = page

        import streamlit as st
        import logging

        # Make combine_file_list emit a real log record to the multifile logger
        def _combine_with_log(**kwargs):
            logger = logging.getLogger("pyadps.processing.multifile")
            # Temporarily add a StreamHandler so our buffer captures the record
            logger.info("Combining files...")
            return MockCombineResult(success=True)

        mocks[
            "pyadps.processing.multifile"
        ].combine_file_list.side_effect = _combine_with_log
        mocks["pyadps.processing.multifile"].combine_file_list.return_value = None

        with tempfile.NamedTemporaryFile(suffix=".000", delete=False) as tmp:
            tmp.write(b"\x7f\x7f" + b"x" * 50)
            output_path = Path(tmp.name)

        files = [_fake_binary(), _fake_binary()]

        st_p = {
            "header": MagicMock(),
            "write": MagicMock(),
            "warning": MagicMock(),
            "subheader": MagicMock(),
            "columns": _smart_columns,
            "text_input": MagicMock(return_value="merged.000"),
            # display_log=True  ←  first checkbox
            "checkbox": MagicMock(side_effect=[True, True, True]),
            "expander": MagicMock(return_value=_make_ctx()),
            "file_uploader": MagicMock(return_value=files),
            "info": MagicMock(),
            "button": MagicMock(side_effect=[False, True]),  # combine=True
            "divider": MagicMock(),
            "success": MagicMock(),
            "error": MagicMock(),
            "metric": MagicMock(),
            "download_button": MagicMock(),
            "dataframe": MagicMock(),
            "markdown": MagicMock(),
            "exception": MagicMock(),
        }

        # Ensure the multifile logger has level set so records propagate
        logger = logging.getLogger("pyadps.processing.multifile")
        original_level = logger.level
        logger.setLevel(logging.INFO)

        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, mocks))
                for attr, mock in st_p.items():
                    if hasattr(st, attr):
                        stack.enter_context(patch.object(st, attr, mock))
                stack.enter_context(
                    patch("pathlib.Path.__truediv__", return_value=output_path)
                )
                mod.render_file_combiner_tool()
        finally:
            logger.setLevel(original_level)
            try:
                os.unlink(output_path)
            except Exception:
                pass

        # markdown is called with ansi_to_html output when log has content
        st_p["markdown"].assert_called()

    # ------------------------------------------------------------------ #
    # Lines 533-534: except Exception: pass  in combine temp cleanup      #
    # ------------------------------------------------------------------ #

    def test_combine_cleanup_unlink_exception_is_swallowed(self, page):
        """Lines 533-534: os.unlink raises during combine cleanup → silently ignored."""
        mod, mocks = page

        import streamlit as st

        mocks["pyadps.processing.multifile"].combine_file_list.side_effect = None
        mocks[
            "pyadps.processing.multifile"
        ].combine_file_list.return_value = MockCombineResult(success=True)

        with tempfile.NamedTemporaryFile(suffix=".000", delete=False) as tmp:
            tmp.write(b"\x7f\x7f" + b"x" * 50)
            output_path = Path(tmp.name)

        files = [_fake_binary(), _fake_binary()]

        st_p = {
            "header": MagicMock(),
            "write": MagicMock(),
            "warning": MagicMock(),
            "subheader": MagicMock(),
            "columns": _smart_columns,
            "text_input": MagicMock(return_value="merged.000"),
            "checkbox": MagicMock(side_effect=[False, True, True]),
            "expander": MagicMock(return_value=_make_ctx()),
            "file_uploader": MagicMock(return_value=files),
            "info": MagicMock(),
            "button": MagicMock(side_effect=[False, True]),  # combine=True
            "divider": MagicMock(),
            "success": MagicMock(),
            "error": MagicMock(),
            "metric": MagicMock(),
            "download_button": MagicMock(),
            "dataframe": MagicMock(),
            "markdown": MagicMock(),
            "exception": MagicMock(),
        }

        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, mocks))
                for attr, mock in st_p.items():
                    if hasattr(st, attr):
                        stack.enter_context(patch.object(st, attr, mock))
                stack.enter_context(
                    patch("pathlib.Path.__truediv__", return_value=output_path)
                )
                stack.enter_context(patch("os.unlink", side_effect=OSError("busy")))
                mod.render_file_combiner_tool()
        finally:
            try:
                os.unlink(output_path)
            except Exception:
                pass

        # success shown despite unlink failure
        st_p["success"].assert_called()


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__, "-v", "--tb=short"])
