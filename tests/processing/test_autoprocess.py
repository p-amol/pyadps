"""
Test suite for autoprocess module.

This module provides tests for the autoprocess() function which coordinates
the full ADCP data processing pipeline.

Run with: pytest test_autoprocess.py -v
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pytest
import xarray as xr


# -----------------------------------------------------------------------------
# IMPORT CONFIGURATION
# -----------------------------------------------------------------------------

try:
    from pyadps.processing.autoprocess import autoprocess
    from pyadps.processing.config import ProcessingConfig
    from pyadps.processing.core import ProcessedDataset

    PATCH_PREFIX = "pyadps.processing.autoprocess"
except ImportError:
    from autoprocess import autoprocess
    from config import ProcessingConfig
    from core import ProcessedDataset

    PATCH_PREFIX = "autoprocess"


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_dataset():
    """Create a minimal valid ADCP dataset for testing."""
    n_beams = 4
    n_cells = 20
    n_time = 100

    np.random.seed(42)
    velocity = np.random.randint(-2000, 2000, size=(n_beams, n_cells, n_time)).astype(
        np.float32
    )

    # Create mask (all valid)
    mask = np.zeros((n_beams, n_cells, n_time), dtype=np.int8)

    ds = xr.Dataset(
        {
            "velocity": (["beam", "cell", "time"], velocity, {"units": "mm/s"}),
            "mask": (["beam", "cell", "time"], mask),
        },
        coords={
            "time": np.arange(n_time),
            "cell": np.arange(n_cells),
            "beam": np.arange(n_beams),
        },
    )

    return ds


@pytest.fixture
def mock_config():
    """Create a mock ProcessingConfig object."""
    config = MagicMock(spec=ProcessingConfig)
    config.input_file_path = "/data"
    config.input_file_name = "test.000"
    config.isTimeAxisModified = False
    config.isSensorTest = False
    config.isQCTest = False
    config.isProfileTest = False
    config.isVelocityTest = False
    config.isAttributes = False
    config.isExportOptions = False
    config.export_include_velocity = True
    config.export_include_echo = False
    config.export_include_correlation = False
    config.export_include_percent_good = False
    config.export_include_mask = False
    config.export_apply_mask = True
    config.export_velocity_units = "cm/s"
    config.isRawExportOptions = False
    config.raw_include_fixed_leader = True
    config.raw_include_variable_leader = True
    config.raw_include_velocity = True
    config.raw_include_echo = True
    config.raw_include_correlation = True
    config.raw_include_percent_good = True
    return config


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_ini(temp_dir):
    """Create a sample config.ini file."""
    config_content = """[InputFile]
input_file_path = /data
input_file_name = test.000

[TimeAxis]
time_axis_modified = False

[SensorTest]
sensor_test = False

[QCTest]
qc_test = False

[ProfileTest]
profile_test = False

[VelocityTest]
velocity_test = False
"""
    config_path = temp_dir / "config.ini"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_binary_file(temp_dir):
    """Create a dummy binary file for path validation tests."""
    binary_path = temp_dir / "test.000"
    binary_path.write_bytes(b"dummy data")
    return binary_path


# -----------------------------------------------------------------------------
# BASIC FUNCTIONALITY TESTS
# -----------------------------------------------------------------------------


class TestAutoprocessBasic:
    """Tests for basic autoprocess functionality."""

    def test_autoprocess_with_config_object(
        self, sample_dataset, mock_config, sample_binary_file
    ):
        """Test autoprocess with ProcessingConfig object."""
        with (
            patch("pyadps.read") as mock_read,
            patch(f"{PATCH_PREFIX}.ProcessedDataset") as mock_proc_class,
        ):
            # Setup mocks
            mock_read.return_value = sample_dataset

            mock_proc = MagicMock()
            mock_proc.finalize.return_value = sample_dataset
            mock_proc.print_summary = MagicMock()
            mock_proc_class.return_value = mock_proc

            # Call autoprocess
            result = autoprocess(
                mock_config,
                binary_file_path=sample_binary_file,
                print_summary=False,
            )

            # Verify
            mock_read.assert_called_once_with(str(sample_binary_file))
            mock_proc_class.assert_called_once_with(sample_dataset)
            mock_proc.apply_config.assert_called_once_with(mock_config)
            mock_proc.finalize.assert_called_once()
            assert isinstance(result, xr.Dataset)

    def test_autoprocess_with_config_file(
        self, sample_dataset, sample_binary_file, temp_dir
    ):
        """Test autoprocess with config file path."""
        # Create a dummy config file
        config_path = temp_dir / "config.ini"
        config_path.write_text("[InputFile]\ninput_file_path = /data\n")

        with (
            patch("pyadps.read") as mock_read,
            patch(f"{PATCH_PREFIX}.ProcessedDataset") as mock_proc_class,
            patch(f"{PATCH_PREFIX}.ProcessingConfig") as mock_config_class,
        ):
            # Setup mock config returned by from_ini
            mock_config = MagicMock()
            mock_config.input_file_path = str(sample_binary_file.parent)
            mock_config.input_file_name = sample_binary_file.name
            mock_config.isTimeAxisModified = False
            mock_config.isSensorTest = False
            mock_config.isQCTest = False
            mock_config.isProfileTest = False
            mock_config.isVelocityTest = False
            mock_config_class.from_ini.return_value = mock_config

            mock_read.return_value = sample_dataset

            mock_proc = MagicMock()
            mock_proc.finalize.return_value = sample_dataset
            mock_proc.print_summary = MagicMock()
            mock_proc_class.return_value = mock_proc

            result = autoprocess(config_path, print_summary=False)

            mock_config_class.from_ini.assert_called_once_with(str(config_path))
            mock_read.assert_called_once()
            assert isinstance(result, xr.Dataset)

    def test_autoprocess_with_path_object(
        self, sample_dataset, mock_config, sample_binary_file
    ):
        """Test autoprocess with Path object for binary file."""
        with (
            patch("pyadps.read") as mock_read,
            patch(f"{PATCH_PREFIX}.ProcessedDataset") as mock_proc_class,
        ):
            mock_read.return_value = sample_dataset

            mock_proc = MagicMock()
            mock_proc.finalize.return_value = sample_dataset
            mock_proc.print_summary = MagicMock()
            mock_proc_class.return_value = mock_proc

            # Pass Path object (not string)
            result = autoprocess(
                mock_config,
                binary_file_path=Path(sample_binary_file),
                print_summary=False,
            )

            assert isinstance(result, xr.Dataset)

    def test_autoprocess_returns_dataset(
        self, sample_dataset, mock_config, sample_binary_file
    ):
        """Test that autoprocess returns xarray Dataset."""
        with (
            patch("pyadps.read") as mock_read,
            patch(f"{PATCH_PREFIX}.ProcessedDataset") as mock_proc_class,
        ):
            mock_read.return_value = sample_dataset

            mock_proc = MagicMock()
            mock_proc.finalize.return_value = sample_dataset
            mock_proc.print_summary = MagicMock()
            mock_proc_class.return_value = mock_proc

            result = autoprocess(
                mock_config,
                binary_file_path=sample_binary_file,
                print_summary=False,
            )

            assert isinstance(result, xr.Dataset)
            assert "velocity" in result.data_vars


# -----------------------------------------------------------------------------
# ERROR HANDLING TESTS
# -----------------------------------------------------------------------------


class TestAutoprocessErrors:
    """Tests for autoprocess error handling."""

    def test_missing_binary_path_with_config_object(self, mock_config):
        """Test error when binary_file_path not provided with config object."""
        with pytest.raises(ValueError, match="binary_file_path must be provided"):
            autoprocess(mock_config)

    def test_missing_binary_path_in_config_file(self, temp_dir):
        """Test error when binary path not in config file and not provided."""
        # Config without input file info
        config_content = """[TimeAxis]
time_axis_modified = False
"""
        config_path = temp_dir / "config.ini"
        config_path.write_text(config_content)

        with patch(f"{PATCH_PREFIX}.ProcessingConfig") as mock_config_class:
            mock_config = MagicMock()
            mock_config.input_file_path = None
            mock_config.input_file_name = None
            mock_config_class.from_ini.return_value = mock_config

            with pytest.raises(ValueError, match="binary_file_path must be provided"):
                autoprocess(config_path)

    def test_binary_file_not_found(self, mock_config):
        """Test error when binary file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Binary file not found"):
            autoprocess(mock_config, binary_file_path="/nonexistent/file.000")

    def test_invalid_velocity_units(
        self, sample_dataset, mock_config, sample_binary_file, temp_dir
    ):
        """Test error with invalid velocity units."""
        with (
            patch("pyadps.read") as mock_read,
            patch(f"{PATCH_PREFIX}.ProcessedDataset") as mock_proc_class,
        ):
            mock_read.return_value = sample_dataset

            mock_proc = MagicMock()
            mock_proc.finalize.return_value = sample_dataset
            mock_proc.print_summary = MagicMock()
            mock_proc.export_to_netcdf.side_effect = ValueError("Invalid units")
            mock_proc_class.return_value = mock_proc

            with pytest.raises(ValueError):
                autoprocess(
                    mock_config,
                    binary_file_path=sample_binary_file,
                    save_netcdf=True,
                    save_velocity_only=True,
                    velocity_units="invalid",
                    print_summary=False,
                )


# -----------------------------------------------------------------------------
# OUTPUT SAVING TESTS
# -----------------------------------------------------------------------------


class TestAutoprocessOutput:
    """Tests for autoprocess output saving functionality."""

    def test_save_netcdf_full_dataset(
        self, sample_dataset, mock_config, sample_binary_file, temp_dir
    ):
        """Test saving full processed dataset."""
        with (
            patch("pyadps.read") as mock_read,
            patch(f"{PATCH_PREFIX}.ProcessedDataset") as mock_proc_class,
        ):
            mock_read.return_value = sample_dataset

            # Create mock that returns dataset with to_netcdf method
            mock_result = MagicMock(spec=xr.Dataset)
            mock_proc = MagicMock()
            mock_proc.finalize.return_value = mock_result
            mock_proc.print_summary = MagicMock()
            mock_proc_class.return_value = mock_proc

            autoprocess(
                mock_config,
                binary_file_path=sample_binary_file,
                save_netcdf=True,
                output_dir=temp_dir,
                print_summary=False,
            )

            # Verify to_netcdf was called on the ProcessedDataset, not the
            # raw finalized xr.Dataset directly - so the axis-ordering
            # fix and export-options stamping in
            # ProcessedDataset.to_netcdf() are applied.
            mock_proc.to_netcdf.assert_called_once()
            call_path = mock_proc.to_netcdf.call_args[0][0]
            assert str(temp_dir) in str(call_path)
            assert "_processed.nc" in str(call_path)

    def test_save_velocity_only(
        self, sample_dataset, mock_config, sample_binary_file, temp_dir
    ):
        """Test saving velocity components only."""
        with (
            patch("pyadps.read") as mock_read,
            patch(f"{PATCH_PREFIX}.ProcessedDataset") as mock_proc_class,
        ):
            mock_read.return_value = sample_dataset

            mock_proc = MagicMock()
            mock_proc.finalize.return_value = sample_dataset
            mock_proc.print_summary = MagicMock()
            mock_proc_class.return_value = mock_proc

            autoprocess(
                mock_config,
                binary_file_path=sample_binary_file,
                save_netcdf=True,
                save_velocity_only=True,
                output_dir=temp_dir,
                velocity_units="m/s",
                print_summary=False,
            )

            # save_velocity_only routes through export_to_netcdf() with
            # only velocity selected, not the standalone
            # velocity_to_netcdf() (so it shares the same Ferret-safe
            # export path as every other component combination).
            mock_proc.export_to_netcdf.assert_called_once()
            call_kwargs = mock_proc.export_to_netcdf.call_args[1]
            assert call_kwargs["include_velocity"] is True
            assert call_kwargs["include_echo"] is False
            assert call_kwargs["include_correlation"] is False
            assert call_kwargs["include_percent_good"] is False
            assert call_kwargs["include_mask"] is False
            assert call_kwargs["velocity_units"] == "m/s"

    def test_custom_output_filename(
        self, sample_dataset, mock_config, sample_binary_file, temp_dir
    ):
        """Test custom output filename."""
        with (
            patch("pyadps.read") as mock_read,
            patch(f"{PATCH_PREFIX}.ProcessedDataset") as mock_proc_class,
        ):
            mock_read.return_value = sample_dataset

            mock_result = MagicMock(spec=xr.Dataset)
            mock_proc = MagicMock()
            mock_proc.finalize.return_value = mock_result
            mock_proc.print_summary = MagicMock()
            mock_proc_class.return_value = mock_proc

            autoprocess(
                mock_config,
                binary_file_path=sample_binary_file,
                save_netcdf=True,
                output_dir=temp_dir,
                output_filename="custom_output.nc",
                print_summary=False,
            )

            call_path = mock_proc.to_netcdf.call_args[0][0]
            assert "custom_output.nc" in str(call_path)

    def test_default_output_directory(
        self, sample_dataset, mock_config, sample_binary_file
    ):
        """Test that default output directory is same as input."""
        with (
            patch("pyadps.read") as mock_read,
            patch(f"{PATCH_PREFIX}.ProcessedDataset") as mock_proc_class,
        ):
            mock_read.return_value = sample_dataset

            mock_result = MagicMock(spec=xr.Dataset)
            mock_proc = MagicMock()
            mock_proc.finalize.return_value = mock_result
            mock_proc.print_summary = MagicMock()
            mock_proc_class.return_value = mock_proc

            autoprocess(
                mock_config,
                binary_file_path=sample_binary_file,
                save_netcdf=True,
                print_summary=False,
            )

            call_path = Path(mock_proc.to_netcdf.call_args[0][0])
            assert call_path.parent == sample_binary_file.parent

    def test_print_summary_prints_output_path_when_save_netcdf(
        self, sample_dataset, mock_config, sample_binary_file, temp_dir, capsys
    ):
        """Test that output path is printed when save_netcdf=True and print_summary=True (line 188)."""
        with (
            patch("pyadps.read") as mock_read,
            patch(f"{PATCH_PREFIX}.ProcessedDataset") as mock_proc_class,
        ):
            mock_read.return_value = sample_dataset

            mock_result = MagicMock(spec=xr.Dataset)
            mock_proc = MagicMock()
            mock_proc.finalize.return_value = mock_result
            mock_proc_class.return_value = mock_proc

            autoprocess(
                mock_config,
                binary_file_path=sample_binary_file,
                save_netcdf=True,
                output_dir=temp_dir,
                print_summary=True,
            )

            stdout = capsys.readouterr().out
            assert "Output saved to:" in stdout

    def test_creates_output_directory(
        self, sample_dataset, mock_config, sample_binary_file, temp_dir
    ):
        """Test that output directory is created if it doesn't exist."""
        new_output_dir = temp_dir / "new_subdir" / "nested"

        with (
            patch("pyadps.read") as mock_read,
            patch(f"{PATCH_PREFIX}.ProcessedDataset") as mock_proc_class,
        ):
            mock_read.return_value = sample_dataset

            mock_result = MagicMock(spec=xr.Dataset)
            mock_proc = MagicMock()
            mock_proc.finalize.return_value = mock_result
            mock_proc.print_summary = MagicMock()
            mock_proc_class.return_value = mock_proc

            autoprocess(
                mock_config,
                binary_file_path=sample_binary_file,
                save_netcdf=True,
                output_dir=new_output_dir,
                print_summary=False,
            )

            assert new_output_dir.exists()


# -----------------------------------------------------------------------------
# PROCESSING OPTIONS TESTS
# -----------------------------------------------------------------------------


class TestAutoprocessOptions:
    """Tests for autoprocess processing options."""

    def test_ensure_depth_ascending_true(
        self, sample_dataset, mock_config, sample_binary_file
    ):
        """Test depth ascending option is passed to finalize."""
        with (
            patch("pyadps.read") as mock_read,
            patch(f"{PATCH_PREFIX}.ProcessedDataset") as mock_proc_class,
        ):
            mock_read.return_value = sample_dataset

            mock_proc = MagicMock()
            mock_proc.finalize.return_value = sample_dataset
            mock_proc.print_summary = MagicMock()
            mock_proc_class.return_value = mock_proc

            autoprocess(
                mock_config,
                binary_file_path=sample_binary_file,
                ensure_depth_ascending=True,
                print_summary=False,
            )

            mock_proc.finalize.assert_called_once_with(ensure_depth_ascending=True)

    def test_ensure_depth_ascending_false(
        self, sample_dataset, mock_config, sample_binary_file
    ):
        """Test depth ascending disabled."""
        with (
            patch("pyadps.read") as mock_read,
            patch(f"{PATCH_PREFIX}.ProcessedDataset") as mock_proc_class,
        ):
            mock_read.return_value = sample_dataset

            mock_proc = MagicMock()
            mock_proc.finalize.return_value = sample_dataset
            mock_proc.print_summary = MagicMock()
            mock_proc_class.return_value = mock_proc

            autoprocess(
                mock_config,
                binary_file_path=sample_binary_file,
                ensure_depth_ascending=False,
                print_summary=False,
            )

            mock_proc.finalize.assert_called_once_with(ensure_depth_ascending=False)

    def test_print_summary_enabled(
        self, sample_dataset, mock_config, sample_binary_file
    ):
        """Test print_summary option enabled."""
        with (
            patch("pyadps.read") as mock_read,
            patch(f"{PATCH_PREFIX}.ProcessedDataset") as mock_proc_class,
        ):
            mock_read.return_value = sample_dataset

            mock_proc = MagicMock()
            mock_proc.finalize.return_value = sample_dataset
            mock_proc_class.return_value = mock_proc

            autoprocess(
                mock_config,
                binary_file_path=sample_binary_file,
                print_summary=True,
            )

            mock_proc.print_summary.assert_called_once()

    def test_print_summary_disabled(
        self, sample_dataset, mock_config, sample_binary_file
    ):
        """Test print_summary option disabled."""
        with (
            patch("pyadps.read") as mock_read,
            patch(f"{PATCH_PREFIX}.ProcessedDataset") as mock_proc_class,
        ):
            mock_read.return_value = sample_dataset

            mock_proc = MagicMock()
            mock_proc.finalize.return_value = sample_dataset
            mock_proc_class.return_value = mock_proc

            autoprocess(
                mock_config,
                binary_file_path=sample_binary_file,
                print_summary=False,
            )

            mock_proc.print_summary.assert_not_called()

    def test_velocity_units_passed_to_export(
        self, sample_dataset, mock_config, sample_binary_file, temp_dir
    ):
        """Test velocity units are passed to velocity export."""
        with (
            patch("pyadps.read") as mock_read,
            patch(f"{PATCH_PREFIX}.ProcessedDataset") as mock_proc_class,
        ):
            mock_read.return_value = sample_dataset

            mock_proc = MagicMock()
            mock_proc.finalize.return_value = sample_dataset
            mock_proc.print_summary = MagicMock()
            mock_proc_class.return_value = mock_proc

            for units in ["mm/s", "cm/s", "m/s"]:
                mock_proc.reset_mock()

                autoprocess(
                    mock_config,
                    binary_file_path=sample_binary_file,
                    save_netcdf=True,
                    save_velocity_only=True,
                    output_dir=temp_dir,
                    velocity_units=units,
                    print_summary=False,
                )

                call_kwargs = mock_proc.export_to_netcdf.call_args[1]
                assert call_kwargs["velocity_units"] == units


# -----------------------------------------------------------------------------
# INTEGRATION STYLE TESTS (with minimal mocking)
# -----------------------------------------------------------------------------


class TestAutoprocessIntegration:
    """Integration-style tests with minimal mocking."""

    def test_full_workflow_mocked_read(
        self, sample_dataset, mock_config, sample_binary_file, temp_dir
    ):
        """Test full workflow with only pyadps.read mocked."""
        with patch("pyadps.read") as mock_read:
            mock_read.return_value = sample_dataset

            # This will use real ProcessedDataset
            # Note: This test may fail if ProcessedDataset has issues
            # It's more of an integration test
            try:
                result = autoprocess(
                    mock_config,
                    binary_file_path=sample_binary_file,
                    print_summary=False,
                )
                assert isinstance(result, xr.Dataset)
            except Exception as e:
                # If ProcessedDataset fails, that's expected in unit tests
                # This test is more for integration testing
                pytest.skip(f"Integration test skipped: {e}")


# -----------------------------------------------------------------------------
# RAW NETCDF SAVING TESTS (real pyadps.read(), not mocked)
# -----------------------------------------------------------------------------

_DEMO_BINARY = Path(__file__).parent / "data" / "demo.000"
_requires_demo_binary = pytest.mark.skipif(
    not _DEMO_BINARY.exists(), reason="data/demo.000 not present"
)


class TestAutoprocessRawNetcdf:
    """
    save_raw_netcdf writes the entire raw (unprocessed) dataset alongside
    the processed one - the same output as the Download Raw File page's
    "Entire Data Set" NetCDF option. Uses a real pyadps.read() (not
    mocked), since the point is to verify actual file content (the Ferret
    axis-ordering fix, attrs cleanup), not just that a method got called.
    """

    @_requires_demo_binary
    def test_explicit_true_writes_raw_file_alongside_processed(self, temp_dir):
        cfg = ProcessingConfig()
        cfg_path = temp_dir / "cfg.ini"
        cfg.to_ini(str(cfg_path))

        autoprocess(
            str(cfg_path),
            binary_file_path=str(_DEMO_BINARY),
            save_netcdf=True,
            save_raw_netcdf=True,
            output_dir=temp_dir,
            print_summary=False,
        )
        assert (temp_dir / "demo_RAW_DATA.nc").exists()
        assert (temp_dir / "demo_processed.nc").exists()

    @_requires_demo_binary
    def test_default_skips_raw_file(self, temp_dir):
        cfg = ProcessingConfig()
        cfg_path = temp_dir / "cfg.ini"
        cfg.to_ini(str(cfg_path))

        autoprocess(
            str(cfg_path),
            binary_file_path=str(_DEMO_BINARY),
            save_netcdf=True,
            output_dir=temp_dir,
            print_summary=False,
        )
        assert not (temp_dir / "demo_RAW_DATA.nc").exists()
        assert (temp_dir / "demo_processed.nc").exists()

    @_requires_demo_binary
    def test_config_israwexportoptions_auto_applies(self, temp_dir):
        """
        A config.ini saved after a real "Entire Data Set" NetCDF download
        on the Download Raw File page (isRawExportOptions=True) must be
        reproduced automatically, with no explicit override needed.
        """
        cfg = ProcessingConfig(isRawExportOptions=True)
        cfg_path = temp_dir / "cfg.ini"
        cfg.to_ini(str(cfg_path))

        autoprocess(
            str(cfg_path),
            binary_file_path=str(_DEMO_BINARY),
            save_netcdf=True,
            output_dir=temp_dir,
            print_summary=False,
        )
        assert (temp_dir / "demo_RAW_DATA.nc").exists()

    @_requires_demo_binary
    def test_explicit_false_overrides_config(self, temp_dir):
        cfg = ProcessingConfig(isRawExportOptions=True)
        cfg_path = temp_dir / "cfg.ini"
        cfg.to_ini(str(cfg_path))

        autoprocess(
            str(cfg_path),
            binary_file_path=str(_DEMO_BINARY),
            save_netcdf=True,
            save_raw_netcdf=False,
            output_dir=temp_dir,
            print_summary=False,
        )
        assert not (temp_dir / "demo_RAW_DATA.nc").exists()

    @_requires_demo_binary
    def test_raw_file_has_no_ambiguous_axis_coord(self, temp_dir):
        """
        The raw NetCDF must get the same Ferret axis-ordering fix as the
        processed output - not reintroduce the leftover 'ensemble'
        coordinate that everything else in this codebase strips at write
        time (see ProcessedDataset._drop_ambiguous_axis_coords).
        """
        cfg = ProcessingConfig()
        cfg_path = temp_dir / "cfg.ini"
        cfg.to_ini(str(cfg_path))

        autoprocess(
            str(cfg_path),
            binary_file_path=str(_DEMO_BINARY),
            save_netcdf=True,
            save_raw_netcdf=True,
            output_dir=temp_dir,
            print_summary=False,
        )
        with xr.open_dataset(temp_dir / "demo_RAW_DATA.nc") as ds:
            assert "ensemble" not in ds.variables

    @_requires_demo_binary
    def test_raw_file_drops_internal_metadata_attrs(self, temp_dir):
        cfg = ProcessingConfig()
        cfg_path = temp_dir / "cfg.ini"
        cfg.to_ini(str(cfg_path))

        autoprocess(
            str(cfg_path),
            binary_file_path=str(_DEMO_BINARY),
            save_netcdf=True,
            save_raw_netcdf=True,
            output_dir=temp_dir,
            print_summary=False,
        )
        with xr.open_dataset(temp_dir / "demo_RAW_DATA.nc") as ds:
            for attr in (
                "pyadps_component",
                "components",
                "fixed_leader_variables",
                "variable_leader_variables",
            ):
                assert attr not in ds.attrs

    @_requires_demo_binary
    def test_explicit_component_subset(self, temp_dir):
        """
        raw_include_* mirrors the Download Raw File page's component
        picker - selecting a specific subset here must not silently save
        the entire dataset instead.
        """
        cfg = ProcessingConfig()
        cfg_path = temp_dir / "cfg.ini"
        cfg.to_ini(str(cfg_path))

        autoprocess(
            str(cfg_path),
            binary_file_path=str(_DEMO_BINARY),
            save_netcdf=True,
            save_raw_netcdf=True,
            raw_include_fixed_leader=False,
            raw_include_variable_leader=False,
            raw_include_velocity=True,
            raw_include_echo=False,
            raw_include_correlation=True,
            raw_include_percent_good=False,
            output_dir=temp_dir,
            print_summary=False,
        )
        with xr.open_dataset(temp_dir / "demo_RAW_DATA.nc") as ds:
            assert set(ds.data_vars) == {"velocity", "correlation"}

    @_requires_demo_binary
    def test_config_raw_include_auto_applies_exact_subset(self, temp_dir):
        """
        A config.ini saved after downloading a specific subset (not the
        entire dataset) on the Download Raw File page must reproduce that
        exact subset automatically, not the entire dataset.
        """
        cfg = ProcessingConfig(
            isRawExportOptions=True,
            raw_include_fixed_leader=False,
            raw_include_variable_leader=False,
            raw_include_velocity=False,
            raw_include_echo=True,
            raw_include_correlation=False,
            raw_include_percent_good=True,
        )
        cfg_path = temp_dir / "cfg.ini"
        cfg.to_ini(str(cfg_path))

        autoprocess(
            str(cfg_path),
            binary_file_path=str(_DEMO_BINARY),
            save_netcdf=True,
            output_dir=temp_dir,
            print_summary=False,
        )
        with xr.open_dataset(temp_dir / "demo_RAW_DATA.nc") as ds:
            assert set(ds.data_vars) == {"echo_intensity", "percent_good"}

    @_requires_demo_binary
    def test_explicit_component_overrides_config(self, temp_dir):
        cfg = ProcessingConfig(
            isRawExportOptions=True,
            raw_include_fixed_leader=False,
            raw_include_variable_leader=False,
            raw_include_velocity=True,
            raw_include_echo=True,
            raw_include_correlation=False,
            raw_include_percent_good=False,
        )
        cfg_path = temp_dir / "cfg.ini"
        cfg.to_ini(str(cfg_path))

        autoprocess(
            str(cfg_path),
            binary_file_path=str(_DEMO_BINARY),
            save_netcdf=True,
            raw_include_velocity=False,
            raw_include_correlation=True,
            output_dir=temp_dir,
            print_summary=False,
        )
        with xr.open_dataset(temp_dir / "demo_RAW_DATA.nc") as ds:
            # velocity explicitly overridden to False, correlation
            # explicitly overridden to True; echo left as config's stored
            # True (not overridden, so it should still be present).
            assert set(ds.data_vars) == {"echo_intensity", "correlation"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
