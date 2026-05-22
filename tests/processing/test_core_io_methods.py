"""
Tests for ProcessedDataset I/O class-methods.

Covers the three methods that depend on pyadps.read():

    ProcessedDataset.from_file(config, binary_file_path=...)
    ProcessedDataset.save_netcdf(config, ...)
    ProcessedDataset.from_ini(filepath, binary_file_path=...)

TWO TEST LAYERS
---------------
Unit tests (TestFromFileUnit, TestSaveNetcdfUnit, TestFromIniUnit)
    Patch pyadps.read so no real binary I/O is needed.  Every
    control-flow branch in the three methods is covered.

Integration tests (TestFromFileIntegration, TestSaveNetcdfIntegration,
                   TestFromIniIntegration)
    Write a real, parseable PD0 binary file with ensemble_builder and
    call the real pyadps.read() through the public API.  Tests that
    require pyadps to be installed are skipped automatically when the
    package is unavailable.

    If you place demo.000 in a data/ sub-directory next to this file
    the integration tests also exercise the real instrument file:

        processing/
            test_io_methods.py
            ensemble_builder.py
            data/
                demo.000

HOW PATCHING WORKS
------------------
from_file() contains a lazy import:

    import pyadps           # resolves sys.modules['pyadps'] at call time
    ds = pyadps.read(...)

patch("pyadps.read") only works when pyadps is a fully installed package.
In the flat-directory test runner the stub pyadps module has no .read
attribute, so patch() raises AttributeError.

Fix: we use patch.object(pyadps_mod, "read", ...) where pyadps_mod is the
actual module object from sys.modules.  _get_pyadps() ensures the attribute
exists (adding a stub lambda if absent) so patch.object never fails.

Run:  pytest test_io_methods.py -v
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# IMPORT CONFIGURATION
# ---------------------------------------------------------------------------

try:
    from pyadps.processing.core import ProcessedDataset
    from pyadps.processing.config import ProcessingConfig
except ImportError:
    from core import ProcessedDataset  # type: ignore[no-redef]
    from config import ProcessingConfig  # type: ignore[no-redef]

try:
    from tests.io.test_pd0_parser.fixtures.ensemble_builder import (
        build_ensemble,
        EnsembleConfig,
        FixedLeaderData,
        VariableLeaderData,
    )
except ImportError:
    from ensemble_builder import (  # type: ignore[no-redef]
        build_ensemble,
        EnsembleConfig,
        FixedLeaderData,
        VariableLeaderData,
    )

# ---------------------------------------------------------------------------
# PYADPS MODULE HANDLE – used by unit tests for patching
# ---------------------------------------------------------------------------


def _get_pyadps():
    """
    Return the pyadps module from sys.modules, inserting a stub if absent.

    from_file() executes `import pyadps; pyadps.read(...)`.  This helper
    guarantees sys.modules['pyadps'] always has a `.read` attribute so
    patch.object can safely replace it in unit tests.
    """
    import types

    if "pyadps" not in sys.modules:
        stub = types.ModuleType("pyadps")
        sys.modules["pyadps"] = stub

    mod = sys.modules["pyadps"]

    if not hasattr(mod, "read"):
        # Placeholder – always overridden inside `with patch.object(...)` blocks
        mod.read = lambda path: None

    return mod


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

N_BEAMS = 4
N_CELLS = 10  # small for speed
N_ENSEMBLES = 5

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _make_minimal_dataset() -> xr.Dataset:
    """Minimal valid xr.Dataset that ProcessedDataset.__init__ accepts."""
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "velocity": (
                ["beam", "cell", "time"],
                rng.integers(
                    -2000, 2000, (N_BEAMS, N_CELLS, N_ENSEMBLES), dtype=np.int16
                ).astype(np.float32),
                {"units": "mm/s"},
            ),
        },
        coords={
            "beam": np.arange(N_BEAMS),
            "cell": np.arange(N_CELLS),
            "time": np.arange(N_ENSEMBLES),
        },
    )


def _write_synthetic_binary(path: Path, n_ensembles: int = N_ENSEMBLES) -> Path:
    """
    Write n_ensembles valid PD0 ensembles to path using ensemble_builder.

    Each ensemble has a unique ensemble_number so readers can identify
    them as separate records.
    """
    cfg = EnsembleConfig(beams=N_BEAMS, cells=N_CELLS)
    data = b""
    for i in range(n_ensembles):
        vl = VariableLeaderData(
            ensemble_number=i + 1,
            year=25,
            month=1,
            day=15,
            hour=10,
            minute=i,
            second=0,
            hundredth=0,
            rtc_century=20,
            rtc_year=25,
            rtc_month=1,
            rtc_day=15,
            rtc_hour=10,
            rtc_minute=i,
            rtc_second=0,
            rtc_hundredth=0,
        )
        data += build_ensemble(
            fixed_leader_data=FixedLeaderData(num_cells=N_CELLS),
            variable_leader_data=vl,
            config=cfg,
        )
    path.write_bytes(data)
    return path


def _demo_binary() -> "Path | None":
    """Return path to data/demo.000 if it sits next to this test file."""
    candidate = Path(__file__).parent / "data" / "demo.000"
    return candidate if candidate.exists() else None


def _write_minimal_ini(path: Path) -> Path:
    """Write a canonical minimal config.ini using ProcessingConfig."""
    try:
        ProcessingConfig().to_ini(str(path))
    except Exception:
        # Fallback: bare-minimum sections that from_ini() can parse
        path.write_text(
            "[FixTime]\nis_time_modified = False\n"
            "[SensorTest]\nsensor_test = False\n"
            "[QCTest]\nqc_test = False\n"
            "[ProfileTest]\nprofile_test = False\n"
            "[VelocityTest]\nvelocity_test = False\n"
            "[Attributes]\nadd_attributes = False\nattributes_json = {}\n"
        )
    return path


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_dataset():
    return _make_minimal_dataset()


@pytest.fixture
def pyadps_mod():
    """The pyadps module object, guaranteed to have a .read attribute."""
    return _get_pyadps()


@pytest.fixture
def synthetic_binary(temp_dir):
    """A real, parseable PD0 binary produced by ensemble_builder."""
    return _write_synthetic_binary(temp_dir / "synthetic.000")


@pytest.fixture
def demo_binary_path():
    """Path to data/demo.000 when present, else None."""
    return _demo_binary()


@pytest.fixture
def ini_file(temp_dir):
    """A minimal config.ini in a temp directory."""
    return _write_minimal_ini(temp_dir / "config.ini")


@pytest.fixture
def placeholder_binary(temp_dir):
    """A file that exists on disk but contains no real PD0 data (unit tests only)."""
    p = temp_dir / "placeholder.000"
    p.write_bytes(b"placeholder")
    return p


# ===========================================================================
# UNIT TESTS – from_file()
# ===========================================================================


class TestFromFileUnit:
    """
    Unit tests for ProcessedDataset.from_file().

    pyadps.read is replaced by a mock so no real binary parsing occurs.
    Every control-flow branch in from_file() is exercised.
    """

    # ---- path resolution --------------------------------------------------

    def test_explicit_binary_path_bypasses_config_fields(
        self, pyadps_mod, sample_dataset, placeholder_binary
    ):
        """binary_file_path argument takes priority over config.input_file_* fields."""
        cfg = ProcessingConfig()  # input_file_path / input_file_name both empty

        with patch.object(pyadps_mod, "read", return_value=sample_dataset) as m:
            proc = ProcessedDataset.from_file(cfg, binary_file_path=placeholder_binary)

        m.assert_called_once_with(str(placeholder_binary))
        assert isinstance(proc, ProcessedDataset)

    def test_path_resolved_from_config_fields(
        self, pyadps_mod, sample_dataset, temp_dir, placeholder_binary
    ):
        """When binary_file_path is None, the path comes from config.input_file_path/name."""
        cfg = ProcessingConfig()
        cfg.input_file_path = str(temp_dir)
        cfg.input_file_name = placeholder_binary.name

        with patch.object(pyadps_mod, "read", return_value=sample_dataset) as m:
            proc = ProcessedDataset.from_file(cfg)

        m.assert_called_once_with(str(placeholder_binary))
        assert isinstance(proc, ProcessedDataset)

    def test_ini_string_path_accepted(
        self, pyadps_mod, sample_dataset, ini_file, placeholder_binary
    ):
        """config may be a string path to a config.ini file."""
        with patch.object(pyadps_mod, "read", return_value=sample_dataset):
            proc = ProcessedDataset.from_file(
                str(ini_file), binary_file_path=placeholder_binary
            )
        assert isinstance(proc, ProcessedDataset)

    def test_ini_path_object_accepted(
        self, pyadps_mod, sample_dataset, ini_file, placeholder_binary
    ):
        """config may be a pathlib.Path to a config.ini file."""
        with patch.object(pyadps_mod, "read", return_value=sample_dataset):
            proc = ProcessedDataset.from_file(
                ini_file, binary_file_path=placeholder_binary
            )
        assert isinstance(proc, ProcessedDataset)

    def test_binary_path_object_accepted(
        self, pyadps_mod, sample_dataset, placeholder_binary
    ):
        """binary_file_path accepts pathlib.Path, not just str."""
        cfg = ProcessingConfig()
        with patch.object(pyadps_mod, "read", return_value=sample_dataset) as m:
            ProcessedDataset.from_file(cfg, binary_file_path=Path(placeholder_binary))
        m.assert_called_once_with(str(placeholder_binary))

    # ---- return value and state -------------------------------------------

    def test_returns_processed_dataset_instance(
        self, pyadps_mod, sample_dataset, placeholder_binary
    ):
        cfg = ProcessingConfig()
        with patch.object(pyadps_mod, "read", return_value=sample_dataset):
            result = ProcessedDataset.from_file(
                cfg, binary_file_path=placeholder_binary
            )
        assert isinstance(result, ProcessedDataset)

    def test_no_processing_applied(
        self, pyadps_mod, sample_dataset, placeholder_binary
    ):
        """from_file() must not run any QC — reports and log stay empty."""
        cfg = ProcessingConfig()
        with patch.object(pyadps_mod, "read", return_value=sample_dataset):
            proc = ProcessedDataset.from_file(cfg, binary_file_path=placeholder_binary)
        assert proc.reports == []
        assert proc.processing_log == []

    def test_config_not_stored_on_instance(
        self, pyadps_mod, sample_dataset, placeholder_binary
    ):
        """
        from_file() does NOT store the supplied config object on the instance —
        that is from_ini()'s responsibility.  proc.config remains a blank default.
        """
        cfg = ProcessingConfig(isSensorTest=True)  # non-default flag
        with patch.object(pyadps_mod, "read", return_value=sample_dataset):
            proc = ProcessedDataset.from_file(cfg, binary_file_path=placeholder_binary)
        # The stored config must be a blank default, not the cfg we passed in
        assert proc.config.isSensorTest is False

    # ---- error conditions -------------------------------------------------

    def test_raises_value_error_when_path_unresolvable(self):
        """Raises ValueError when neither binary_file_path nor config fields provide a path."""
        cfg = ProcessingConfig()  # input_file_path = "", input_file_name = ""
        with pytest.raises(ValueError, match="binary_file_path"):
            ProcessedDataset.from_file(cfg)

    def test_raises_value_error_only_input_file_path_set(self, temp_dir):
        """Raises ValueError when only input_file_path is set (name is empty)."""
        cfg = ProcessingConfig()
        cfg.input_file_path = str(temp_dir)
        # input_file_name intentionally left ""
        with pytest.raises(ValueError, match="binary_file_path"):
            ProcessedDataset.from_file(cfg)

    def test_raises_value_error_only_input_file_name_set(self):
        """Raises ValueError when only input_file_name is set (path is empty)."""
        cfg = ProcessingConfig()
        cfg.input_file_name = "data.000"
        # input_file_path intentionally left ""
        with pytest.raises(ValueError, match="binary_file_path"):
            ProcessedDataset.from_file(cfg)

    def test_raises_file_not_found_when_file_absent(self, temp_dir):
        """Raises FileNotFoundError when the resolved path does not exist."""
        cfg = ProcessingConfig()
        with pytest.raises(FileNotFoundError):
            ProcessedDataset.from_file(
                cfg, binary_file_path=temp_dir / "nonexistent.000"
            )


# ===========================================================================
# UNIT TESTS – save_netcdf()
# ===========================================================================


class TestSaveNetcdfUnit:
    """
    Unit tests for ProcessedDataset.save_netcdf().

    to_netcdf / velocity_to_netcdf are patched so nothing is written to
    disk (except in the few tests that explicitly need an output file).
    """

    def _make_proc(self, sample_dataset) -> ProcessedDataset:
        return ProcessedDataset(sample_dataset)

    # ---- output directory resolution -------------------------------------

    def test_explicit_output_dir_used(self, sample_dataset, temp_dir):
        """output_dir kwarg takes priority over any config path."""
        proc = self._make_proc(sample_dataset)
        cfg = ProcessingConfig()
        cfg.input_file_name = "demo.000"

        with patch.object(proc, "to_netcdf"):
            result = proc.save_netcdf(
                cfg,
                output_dir=temp_dir,
                output_filename="out.nc",
                print_summary=False,
            )
        assert result.parent.resolve() == temp_dir.resolve()

    def test_falls_back_to_config_output_file_path(self, sample_dataset, temp_dir):
        """Uses cfg.output_file_path when output_dir is not given."""
        out_sub = temp_dir / "results"
        proc = self._make_proc(sample_dataset)
        cfg = ProcessingConfig()
        cfg.output_file_path = str(out_sub)
        cfg.input_file_name = "demo.000"

        with patch.object(proc, "to_netcdf"):
            result = proc.save_netcdf(cfg, print_summary=False)

        assert result.parent.resolve() == out_sub.resolve()

    def test_falls_back_to_config_input_file_path(self, sample_dataset, temp_dir):
        """Falls back to cfg.input_file_path when output_file_path is empty."""
        proc = self._make_proc(sample_dataset)
        cfg = ProcessingConfig()
        cfg.input_file_path = str(temp_dir)
        cfg.input_file_name = "myfile.000"
        # output_file_path left ""

        with patch.object(proc, "to_netcdf"):
            result = proc.save_netcdf(cfg, print_summary=False)

        assert result.parent.resolve() == temp_dir.resolve()

    def test_output_dir_created_when_absent(self, sample_dataset, temp_dir):
        """The output directory is created automatically when it does not exist."""
        proc = self._make_proc(sample_dataset)
        cfg = ProcessingConfig()
        new_dir = temp_dir / "nested" / "subdir"
        assert not new_dir.exists()

        with patch.object(proc, "to_netcdf"):
            proc.save_netcdf(
                cfg,
                output_dir=new_dir,
                output_filename="out.nc",
                print_summary=False,
            )

        assert new_dir.exists()

    # ---- filename derivation ---------------------------------------------

    def test_default_filename_processed_suffix(self, sample_dataset, temp_dir):
        """Default filename is '<stem>_processed.nc' when velocity_only=False."""
        proc = self._make_proc(sample_dataset)
        cfg = ProcessingConfig()
        cfg.input_file_name = "demo.000"

        with patch.object(proc, "to_netcdf"):
            result = proc.save_netcdf(cfg, output_dir=temp_dir, print_summary=False)

        assert result.name == "demo_processed.nc"

    def test_default_filename_velocity_suffix(self, sample_dataset, temp_dir):
        """Default filename is '<stem>_velocity.nc' when velocity_only=True."""
        proc = self._make_proc(sample_dataset)
        cfg = ProcessingConfig()
        cfg.input_file_name = "demo.000"

        with patch.object(proc, "velocity_to_netcdf"):
            result = proc.save_netcdf(
                cfg, output_dir=temp_dir, velocity_only=True, print_summary=False
            )

        assert result.name == "demo_velocity.nc"

    def test_default_stem_adcp_when_no_input_name(self, sample_dataset, temp_dir):
        """Uses 'adcp' as stem when input_file_name is empty."""
        proc = self._make_proc(sample_dataset)
        cfg = ProcessingConfig()  # input_file_name left ""

        with patch.object(proc, "to_netcdf"):
            result = proc.save_netcdf(cfg, output_dir=temp_dir, print_summary=False)

        assert result.name == "adcp_processed.nc"

    def test_explicit_output_filename_overrides_default(self, sample_dataset, temp_dir):
        """output_filename kwarg overrides the derived filename."""
        proc = self._make_proc(sample_dataset)
        cfg = ProcessingConfig()
        cfg.input_file_name = "demo.000"

        with patch.object(proc, "to_netcdf"):
            result = proc.save_netcdf(
                cfg,
                output_dir=temp_dir,
                output_filename="custom.nc",
                print_summary=False,
            )

        assert result.name == "custom.nc"

    # ---- writer dispatch --------------------------------------------------

    def test_calls_to_netcdf_when_velocity_only_false(self, sample_dataset, temp_dir):
        """to_netcdf() is called and velocity_to_netcdf() is not when velocity_only=False."""
        proc = self._make_proc(sample_dataset)
        cfg = ProcessingConfig()

        with (
            patch.object(proc, "to_netcdf") as mock_full,
            patch.object(proc, "velocity_to_netcdf") as mock_vel,
        ):
            proc.save_netcdf(cfg, output_dir=temp_dir, print_summary=False)

        mock_full.assert_called_once()
        mock_vel.assert_not_called()

    def test_calls_velocity_to_netcdf_when_velocity_only_true(
        self, sample_dataset, temp_dir
    ):
        """velocity_to_netcdf() is called and to_netcdf() is not when velocity_only=True."""
        proc = self._make_proc(sample_dataset)
        cfg = ProcessingConfig()

        with (
            patch.object(proc, "to_netcdf") as mock_full,
            patch.object(proc, "velocity_to_netcdf") as mock_vel,
        ):
            proc.save_netcdf(
                cfg, output_dir=temp_dir, velocity_only=True, print_summary=False
            )

        mock_vel.assert_called_once()
        mock_full.assert_not_called()

    def test_velocity_units_forwarded(self, sample_dataset, temp_dir):
        """velocity_units is forwarded as the `units` kwarg to velocity_to_netcdf()."""
        proc = self._make_proc(sample_dataset)
        cfg = ProcessingConfig()

        with patch.object(proc, "velocity_to_netcdf") as mock_vel:
            proc.save_netcdf(
                cfg,
                output_dir=temp_dir,
                velocity_only=True,
                velocity_units="m/s",
                print_summary=False,
            )

        _, call_kwargs = mock_vel.call_args
        assert call_kwargs.get("units") == "m/s"

    def test_ensure_depth_ascending_forwarded_to_to_netcdf(
        self, sample_dataset, temp_dir
    ):
        """ensure_depth_ascending is forwarded to to_netcdf()."""
        proc = self._make_proc(sample_dataset)
        cfg = ProcessingConfig()

        with patch.object(proc, "to_netcdf") as mock_nc:
            proc.save_netcdf(
                cfg,
                output_dir=temp_dir,
                ensure_depth_ascending=False,
                print_summary=False,
            )

        _, call_kwargs = mock_nc.call_args
        assert call_kwargs.get("ensure_depth_ascending") is False

    # ---- summary printing ------------------------------------------------

    def test_print_summary_true_calls_print_summary_method(
        self, sample_dataset, temp_dir
    ):
        """print_summary=True triggers self.print_summary()."""
        proc = self._make_proc(sample_dataset)
        cfg = ProcessingConfig()

        with (
            patch.object(proc, "to_netcdf"),
            patch.object(proc, "print_summary") as mock_ps,
        ):
            proc.save_netcdf(cfg, output_dir=temp_dir, print_summary=True)

        mock_ps.assert_called_once()

    def test_print_summary_false_suppresses_print_summary_method(
        self, sample_dataset, temp_dir
    ):
        """print_summary=False must not call self.print_summary()."""
        proc = self._make_proc(sample_dataset)
        cfg = ProcessingConfig()

        with (
            patch.object(proc, "to_netcdf"),
            patch.object(proc, "print_summary") as mock_ps,
        ):
            proc.save_netcdf(cfg, output_dir=temp_dir, print_summary=False)

        mock_ps.assert_not_called()

    # ---- return value ----------------------------------------------------

    def test_returns_absolute_path(self, sample_dataset, temp_dir):
        """Return value is an absolute Path."""
        proc = self._make_proc(sample_dataset)
        cfg = ProcessingConfig()

        with patch.object(proc, "to_netcdf"):
            result = proc.save_netcdf(
                cfg,
                output_dir=temp_dir,
                output_filename="out.nc",
                print_summary=False,
            )

        assert result.is_absolute()

    def test_returned_path_has_correct_name(self, sample_dataset, temp_dir):
        """Returned Path has the expected filename."""
        proc = self._make_proc(sample_dataset)
        cfg = ProcessingConfig()

        with patch.object(proc, "to_netcdf"):
            result = proc.save_netcdf(
                cfg,
                output_dir=temp_dir,
                output_filename="specific.nc",
                print_summary=False,
            )

        assert result.name == "specific.nc"

    # ---- config argument forms -------------------------------------------

    def test_accepts_processing_config_object(self, sample_dataset, temp_dir):
        """config may be a ProcessingConfig instance."""
        proc = self._make_proc(sample_dataset)

        with patch.object(proc, "to_netcdf"):
            result = proc.save_netcdf(
                ProcessingConfig(), output_dir=temp_dir, print_summary=False
            )

        assert result.suffix == ".nc"

    def test_accepts_ini_string_path(self, sample_dataset, temp_dir, ini_file):
        """config may be a string path to a config.ini file."""
        proc = self._make_proc(sample_dataset)

        with patch.object(proc, "to_netcdf"):
            result = proc.save_netcdf(
                str(ini_file),
                output_dir=temp_dir,
                output_filename="from_str.nc",
                print_summary=False,
            )

        assert result.name == "from_str.nc"

    def test_accepts_ini_path_object(self, sample_dataset, temp_dir, ini_file):
        """config may be a pathlib.Path pointing to a config.ini file."""
        proc = self._make_proc(sample_dataset)

        with patch.object(proc, "to_netcdf"):
            result = proc.save_netcdf(
                Path(ini_file),
                output_dir=temp_dir,
                output_filename="from_path.nc",
                print_summary=False,
            )

        assert result.name == "from_path.nc"


# ===========================================================================
# UNIT TESTS – from_ini()
# ===========================================================================


class TestFromIniUnit:
    """
    Unit tests for ProcessedDataset.from_ini().

    pyadps.read is patched.  Tests focus on config parsing and storage.
    """

    # ---- basic behaviour -------------------------------------------------

    def test_returns_processed_dataset(
        self, pyadps_mod, sample_dataset, ini_file, placeholder_binary
    ):
        with patch.object(pyadps_mod, "read", return_value=sample_dataset):
            result = ProcessedDataset.from_ini(
                ini_file, binary_file_path=placeholder_binary
            )
        assert isinstance(result, ProcessedDataset)

    def test_config_stored_on_instance(
        self, pyadps_mod, sample_dataset, ini_file, placeholder_binary
    ):
        """from_ini() stores the parsed config on proc.config."""
        with patch.object(pyadps_mod, "read", return_value=sample_dataset):
            proc = ProcessedDataset.from_ini(
                ini_file, binary_file_path=placeholder_binary
            )
        assert type(proc.config).__name__ == "ProcessingConfig"

    def test_no_processing_applied(
        self, pyadps_mod, sample_dataset, ini_file, placeholder_binary
    ):
        """from_ini() does not apply any QC — reports and log are empty."""
        with patch.object(pyadps_mod, "read", return_value=sample_dataset):
            proc = ProcessedDataset.from_ini(
                ini_file, binary_file_path=placeholder_binary
            )
        assert proc.reports == []
        assert proc.processing_log == []

    # ---- config flag round-trip ------------------------------------------

    def test_boolean_true_flag_preserved(
        self, pyadps_mod, sample_dataset, temp_dir, placeholder_binary
    ):
        """A True flag written to INI survives the parse round-trip."""
        ini_path = temp_dir / "true_flag.ini"
        ProcessingConfig(isSensorTest=True).to_ini(str(ini_path))

        with patch.object(pyadps_mod, "read", return_value=sample_dataset):
            proc = ProcessedDataset.from_ini(
                ini_path, binary_file_path=placeholder_binary
            )

        assert proc.config.isSensorTest is True

    def test_boolean_false_flags_preserved(
        self, pyadps_mod, sample_dataset, temp_dir, placeholder_binary
    ):
        """False flags written to INI survive the parse round-trip."""
        ini_path = temp_dir / "false_flags.ini"
        ProcessingConfig(isSensorTest=False, isQCTest=False).to_ini(str(ini_path))

        with patch.object(pyadps_mod, "read", return_value=sample_dataset):
            proc = ProcessedDataset.from_ini(
                ini_path, binary_file_path=placeholder_binary
            )

        assert proc.config.isSensorTest is False
        assert proc.config.isQCTest is False

    def test_numeric_field_preserved(
        self, pyadps_mod, sample_dataset, temp_dir, placeholder_binary
    ):
        """A float config field survives the INI round-trip."""
        ini_path = temp_dir / "numeric.ini"
        ProcessingConfig(roll_cutoff_ST=12.5).to_ini(str(ini_path))

        with patch.object(pyadps_mod, "read", return_value=sample_dataset):
            proc = ProcessedDataset.from_ini(
                ini_path, binary_file_path=placeholder_binary
            )

        assert proc.config.roll_cutoff_ST == pytest.approx(12.5)

    def test_config_not_blank_when_flags_differ_from_default(
        self, pyadps_mod, sample_dataset, temp_dir, placeholder_binary
    ):
        """
        proc.config reflects the parsed INI, not the blank default that
        __init__ creates — verified via a non-default flag.
        """
        ini_path = temp_dir / "nondefault.ini"
        ProcessingConfig(isVelocityTest=True).to_ini(str(ini_path))

        with patch.object(pyadps_mod, "read", return_value=sample_dataset):
            proc = ProcessedDataset.from_ini(
                ini_path, binary_file_path=placeholder_binary
            )

        assert proc.config.isVelocityTest is True

    # ---- path argument forms ---------------------------------------------

    def test_accepts_string_filepath(
        self, pyadps_mod, sample_dataset, ini_file, placeholder_binary
    ):
        """filepath may be a plain string."""
        with patch.object(pyadps_mod, "read", return_value=sample_dataset):
            proc = ProcessedDataset.from_ini(
                str(ini_file), binary_file_path=placeholder_binary
            )
        assert isinstance(proc, ProcessedDataset)

    def test_accepts_path_object(
        self, pyadps_mod, sample_dataset, ini_file, placeholder_binary
    ):
        """filepath may be a pathlib.Path object."""
        with patch.object(pyadps_mod, "read", return_value=sample_dataset):
            proc = ProcessedDataset.from_ini(
                Path(ini_file), binary_file_path=placeholder_binary
            )
        assert isinstance(proc, ProcessedDataset)

    def test_binary_path_forwarded_to_pyadps_read(
        self, pyadps_mod, sample_dataset, ini_file, placeholder_binary
    ):
        """binary_file_path is passed through to pyadps.read()."""
        with patch.object(pyadps_mod, "read", return_value=sample_dataset) as m:
            ProcessedDataset.from_ini(ini_file, binary_file_path=placeholder_binary)
        m.assert_called_once_with(str(placeholder_binary))


# ===========================================================================
# INTEGRATION TESTS – real pyadps.read() with a real binary file
# ===========================================================================


def _pyadps_read_available() -> bool:
    """
    Return True only when pyadps is a real installed package.

    A real installation has __file__ set and pyadps.read comes from
    pyadps.io.binary_reader (not a stub lambda we injected into the shim).
    Checking __file__ reliably distinguishes the two cases.
    """
    try:
        import pyadps

        # A stub module created by the flat-directory test runner has no
        # __file__ attribute (types.ModuleType() does not set one).
        if not getattr(pyadps, "__file__", None):
            return False
        return callable(getattr(pyadps, "read", None))
    except ImportError:
        return False


requires_pyadps = pytest.mark.skipif(
    not _pyadps_read_available(),
    reason="pyadps not installed — integration tests skipped",
)


class TestFromFileIntegration:
    """
    Integration tests: from_file() calls the real pyadps.read() on a PD0
    binary produced by ensemble_builder.

    All tests in this class are skipped when pyadps is not installed.
    """

    @requires_pyadps
    def test_reads_synthetic_binary(self, synthetic_binary):
        """from_file() succeeds with a PD0 binary from ensemble_builder."""
        cfg = ProcessingConfig()
        proc = ProcessedDataset.from_file(cfg, binary_file_path=synthetic_binary)
        assert isinstance(proc, ProcessedDataset)

    @requires_pyadps
    def test_dataset_has_velocity(self, synthetic_binary):
        """Parsed dataset contains a 'velocity' variable."""
        cfg = ProcessingConfig()
        proc = ProcessedDataset.from_file(cfg, binary_file_path=synthetic_binary)
        assert "velocity" in proc.dataset.data_vars

    @requires_pyadps
    def test_dataset_has_mask(self, synthetic_binary):
        """A mask variable is initialised after reading."""
        cfg = ProcessingConfig()
        proc = ProcessedDataset.from_file(cfg, binary_file_path=synthetic_binary)
        assert "mask" in proc.dataset.data_vars

    @requires_pyadps
    def test_velocity_is_3d(self, synthetic_binary):
        """Velocity has 3 dimensions: beam × cell × time."""
        cfg = ProcessingConfig()
        proc = ProcessedDataset.from_file(cfg, binary_file_path=synthetic_binary)
        assert proc.dataset["velocity"].ndim == 3

    @requires_pyadps
    def test_correct_beam_count(self, synthetic_binary):
        """Beam dimension size matches EnsembleConfig.beams."""
        cfg = ProcessingConfig()
        proc = ProcessedDataset.from_file(cfg, binary_file_path=synthetic_binary)
        beam_size = proc.dataset.sizes[proc.dataset["velocity"].dims[0]]
        assert beam_size == N_BEAMS

    @requires_pyadps
    def test_correct_cell_count(self, synthetic_binary):
        """Cell dimension size matches EnsembleConfig.cells."""
        cfg = ProcessingConfig()
        proc = ProcessedDataset.from_file(cfg, binary_file_path=synthetic_binary)
        cell_size = proc.dataset.sizes[proc.dataset["velocity"].dims[1]]
        assert cell_size == N_CELLS

    @requires_pyadps
    def test_correct_ensemble_count(self, synthetic_binary):
        """Time/ensemble dimension matches the number of ensembles written."""
        cfg = ProcessingConfig()
        proc = ProcessedDataset.from_file(cfg, binary_file_path=synthetic_binary)
        time_size = proc.dataset.sizes[proc.dataset["velocity"].dims[2]]
        assert time_size == N_ENSEMBLES

    @requires_pyadps
    def test_path_resolved_from_config_fields(self, temp_dir):
        """from_file() resolves the binary path from config.input_file_path/name."""
        bin_path = _write_synthetic_binary(temp_dir / "cfg_fields.000")
        cfg = ProcessingConfig()
        cfg.input_file_path = str(temp_dir)
        cfg.input_file_name = "cfg_fields.000"

        proc = ProcessedDataset.from_file(cfg)
        assert isinstance(proc, ProcessedDataset)
        assert "velocity" in proc.dataset.data_vars

    @requires_pyadps
    def test_no_processing_applied(self, synthetic_binary):
        """from_file() alone applies zero processing steps."""
        cfg = ProcessingConfig()
        proc = ProcessedDataset.from_file(cfg, binary_file_path=synthetic_binary)
        assert proc.reports == []

    @requires_pyadps
    def test_reads_demo_binary_if_present(self, demo_binary_path):
        """demo.000 in data/ is parsed without error when present."""
        if demo_binary_path is None:
            pytest.skip(
                "data/demo.000 not found — place file there to enable this test"
            )
        cfg = ProcessingConfig()
        proc = ProcessedDataset.from_file(cfg, binary_file_path=demo_binary_path)
        assert isinstance(proc, ProcessedDataset)
        assert "velocity" in proc.dataset.data_vars


class TestSaveNetcdfIntegration:
    """Integration tests: save_netcdf() writes real NetCDF files."""

    @requires_pyadps
    def test_creates_output_file(self, synthetic_binary, temp_dir):
        """save_netcdf() creates a file at the returned path."""
        cfg = ProcessingConfig()
        proc = ProcessedDataset.from_file(cfg, binary_file_path=synthetic_binary)
        result = proc.save_netcdf(
            cfg,
            output_dir=temp_dir,
            output_filename="out.nc",
            print_summary=False,
        )
        assert result.exists()

    @requires_pyadps
    def test_output_readable_by_xarray(self, synthetic_binary, temp_dir):
        """Written NetCDF can be re-opened with xarray."""
        cfg = ProcessingConfig()
        proc = ProcessedDataset.from_file(cfg, binary_file_path=synthetic_binary)
        result = proc.save_netcdf(
            cfg,
            output_dir=temp_dir,
            output_filename="readable.nc",
            print_summary=False,
        )
        with xr.open_dataset(result) as ds:
            assert "velocity" in ds.data_vars

    @requires_pyadps
    def test_full_pipeline_round_trip(self, synthetic_binary, temp_dir):
        """from_file → apply_config (no-op) → save_netcdf end-to-end."""
        cfg = ProcessingConfig()  # all checks disabled → identity transform
        proc = ProcessedDataset.from_file(cfg, binary_file_path=synthetic_binary)
        proc.apply_config(cfg)

        result = proc.save_netcdf(
            cfg,
            output_dir=temp_dir,
            output_filename="pipeline.nc",
            print_summary=False,
        )
        assert result.exists()
        with xr.open_dataset(result) as ds:
            assert "velocity" in ds.data_vars

    @requires_pyadps
    def test_velocity_only_has_component_variables(self, synthetic_binary, temp_dir):
        """velocity_only=True writes separate component velocity variables."""
        cfg = ProcessingConfig()
        proc = ProcessedDataset.from_file(cfg, binary_file_path=synthetic_binary)
        result = proc.save_netcdf(
            cfg,
            output_dir=temp_dir,
            velocity_only=True,
            velocity_units="cm/s",
            output_filename="vel_only.nc",
            print_summary=False,
        )
        assert result.exists()
        with xr.open_dataset(result) as ds:
            component_keywords = (
                "zonal",
                "meridional",
                "vertical",
                "velocity_u",
                "velocity_v",
                "u_vel",
                "v_vel",
                "w_vel",
            )
            found = any(any(kw in v for kw in component_keywords) for v in ds.data_vars)
            assert (
                found
            ), f"No velocity component variables found. Variables: {list(ds.data_vars)}"

    @requires_pyadps
    def test_with_demo_binary(self, demo_binary_path, temp_dir):
        """demo.000 → save_netcdf() produces a valid NetCDF file."""
        if demo_binary_path is None:
            pytest.skip(
                "data/demo.000 not found — place file there to enable this test"
            )
        cfg = ProcessingConfig()
        proc = ProcessedDataset.from_file(cfg, binary_file_path=demo_binary_path)
        result = proc.save_netcdf(
            cfg,
            output_dir=temp_dir,
            output_filename="demo_out.nc",
            print_summary=False,
        )
        assert result.exists()
        with xr.open_dataset(result) as ds:
            assert "velocity" in ds.data_vars


class TestFromIniIntegration:
    """Integration tests: from_ini() with a real INI file and real binary."""

    @requires_pyadps
    def test_reads_binary_returns_proc(self, synthetic_binary, temp_dir):
        """from_ini() parses INI + reads binary, returning a ProcessedDataset."""
        ini_path = _write_minimal_ini(temp_dir / "basic.ini")
        proc = ProcessedDataset.from_ini(ini_path, binary_file_path=synthetic_binary)
        assert isinstance(proc, ProcessedDataset)

    @requires_pyadps
    def test_config_stored_with_correct_values(self, synthetic_binary, temp_dir):
        """proc.config reflects the non-default flag values written to INI."""
        ini_path = temp_dir / "flagged.ini"
        ProcessingConfig(isSensorTest=True, roll_cutoff_ST=8.0).to_ini(str(ini_path))

        proc = ProcessedDataset.from_ini(ini_path, binary_file_path=synthetic_binary)
        assert proc.config.isSensorTest is True
        assert proc.config.roll_cutoff_ST == pytest.approx(8.0)

    @requires_pyadps
    def test_velocity_present_in_dataset(self, synthetic_binary, temp_dir):
        """Dataset read by from_ini() contains a velocity variable."""
        ini_path = _write_minimal_ini(temp_dir / "v.ini")
        proc = ProcessedDataset.from_ini(ini_path, binary_file_path=synthetic_binary)
        assert "velocity" in proc.dataset.data_vars

    @requires_pyadps
    def test_no_processing_applied(self, synthetic_binary, temp_dir):
        """from_ini() alone applies zero processing steps."""
        ini_path = _write_minimal_ini(temp_dir / "noproc.ini")
        proc = ProcessedDataset.from_ini(ini_path, binary_file_path=synthetic_binary)
        assert proc.reports == []
        assert proc.processing_log == []

    @requires_pyadps
    def test_full_pipeline_end_to_end(self, synthetic_binary, temp_dir):
        """from_ini → apply_config → save_netcdf end-to-end round-trip."""
        ini_path = _write_minimal_ini(temp_dir / "full.ini")
        proc = ProcessedDataset.from_ini(ini_path, binary_file_path=synthetic_binary)
        proc.apply_config(proc.config)

        out = proc.save_netcdf(
            proc.config,
            output_dir=temp_dir,
            output_filename="from_ini_pipeline.nc",
            print_summary=False,
        )
        assert out.exists()
        with xr.open_dataset(out) as ds:
            assert "velocity" in ds.data_vars

    @requires_pyadps
    def test_exported_config_round_trips_correctly(self, synthetic_binary, temp_dir):
        """Config exported after from_ini() re-parses with identical field values."""
        ini_path = temp_dir / "orig.ini"
        ProcessingConfig(isSensorTest=True, roll_cutoff_ST=8.0).to_ini(str(ini_path))

        proc = ProcessedDataset.from_ini(ini_path, binary_file_path=synthetic_binary)
        export_path = temp_dir / "exported.ini"
        proc.export_config(str(export_path))

        reloaded = ProcessingConfig.from_ini(str(export_path))
        assert reloaded.isSensorTest == proc.config.isSensorTest
        assert reloaded.roll_cutoff_ST == pytest.approx(proc.config.roll_cutoff_ST)

    @requires_pyadps
    def test_with_demo_binary(self, demo_binary_path, temp_dir):
        """from_ini() works with the real instrument demo.000 file."""
        if demo_binary_path is None:
            pytest.skip(
                "data/demo.000 not found — place file there to enable this test"
            )
        ini_path = _write_minimal_ini(temp_dir / "demo.ini")
        proc = ProcessedDataset.from_ini(ini_path, binary_file_path=demo_binary_path)
        assert isinstance(proc, ProcessedDataset)
        assert "velocity" in proc.dataset.data_vars


if __name__ == "__main__":
    import pytest as _p

    _p.main([__file__, "-v"])
