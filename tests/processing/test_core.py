"""
Test suite for ProcessedDataset orchestrator class (core.py).

This module provides comprehensive tests for:
- CutRegion helper class
- ProcessedDataset initialization
- Processing pipeline methods (apply_*)
- Utility methods (reset, get_current_stats)
- Output methods (finalize, to_netcdf, velocity_to_netcdf)
- Depth ordering functionality
- Unit conversion for velocity export
- Reporting methods

Run with: pytest test_core.py -v

SETUP:
Place this file in the same directory as core.py, or adjust the import
paths below to match your project structure.
"""

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

# -----------------------------------------------------------------------------
# IMPORT CONFIGURATION
# -----------------------------------------------------------------------------
# With direct imports in core.py, we can now patch at the core module level.
# -----------------------------------------------------------------------------

# Try different import strategies
try:
    # If installed as package or in PYTHONPATH
    from pyadps.processing.core import CutRegion, ProcessedDataset

    PATCH_PREFIX = "pyadps.processing.core"
except ImportError:
    try:
        # If in same directory as core.py
        from core import CutRegion, ProcessedDataset

        PATCH_PREFIX = "core"
    except ImportError:
        # Add parent directory to path if needed
        sys.path.insert(0, str(Path(__file__).parent))
        from core import CutRegion, ProcessedDataset

        PATCH_PREFIX = "core"


# Patch targets - now we can patch directly in core module
SENSOR_HEALTH_RUNNER = f"{PATCH_PREFIX}.SensorHealthRunner"
SIGNAL_QUALITY_RUNNER = f"{PATCH_PREFIX}.SignalQualityRunner"
PROFILE_OPERATION_RUNNER = f"{PATCH_PREFIX}.ProfileOperationRunner"
VELOCITY_CHECK_RUNNER = f"{PATCH_PREFIX}.VelocityCheckRunner"
PROCESSING_CONFIG = f"{PATCH_PREFIX}.ProcessingConfig"


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_dataset():
    """Create a minimal valid ADCP dataset for testing."""
    n_beams = 4
    n_cells = 20
    n_time = 100

    # Create coordinates
    time = np.arange(n_time)
    cell = np.arange(n_cells)
    beam = np.arange(n_beams)

    # Create velocity data (mm/s) - random values between -2000 and 2000
    np.random.seed(42)
    velocity = np.random.randint(-2000, 2000, size=(n_beams, n_cells, n_time)).astype(
        np.float32
    )

    # Create other variables
    correlation = np.random.randint(50, 255, size=(n_beams, n_cells, n_time)).astype(
        np.uint8
    )
    echo_intensity = np.random.randint(30, 200, size=(n_beams, n_cells, n_time)).astype(
        np.uint8
    )
    percent_good = np.random.randint(0, 100, size=(n_beams, n_cells, n_time)).astype(
        np.uint8
    )

    # Sensor data (1D along time)
    roll = (
        np.random.uniform(-10, 10, size=n_time).astype(np.float32) * 100
    )  # centidegrees
    pitch = np.random.uniform(-10, 10, size=n_time).astype(np.float32) * 100
    temperature = np.random.uniform(10, 20, size=n_time).astype(np.float32) * 100
    sound_speed = np.full(n_time, 1500.0, dtype=np.float32)

    # Create dataset
    ds = xr.Dataset(
        {
            "velocity": (["beam", "cell", "time"], velocity, {"units": "mm/s"}),
            "correlation": (["beam", "cell", "time"], correlation),
            "echo_intensity": (["beam", "cell", "time"], echo_intensity),
            "percent_good": (["beam", "cell", "time"], percent_good),
            "roll": (["time"], roll, {"scale_factor": 0.01}),
            "pitch": (["time"], pitch, {"scale_factor": 0.01}),
            "temperature": (["time"], temperature, {"scale_factor": 0.01}),
            "sound_speed": (["time"], sound_speed),
        },
        coords={
            "time": time,
            "cell": cell,
            "beam": beam,
        },
    )

    return ds


@pytest.fixture
def sample_dataset_with_mask(sample_dataset):
    """Create dataset with pre-existing mask."""
    ds = sample_dataset.copy(deep=True)
    mask_shape = ds["velocity"].shape
    mask = np.zeros(mask_shape, dtype=np.int8)
    # Mask some random cells
    mask[:, 0, :] = 1  # First cell always masked
    ds["mask"] = (["beam", "cell", "time"], mask)
    return ds


@pytest.fixture
def sample_dataset_descending_depth():
    """Create dataset with depth in descending order (typical for upward-looking ADCP).

    Uses 'cell' as the dimension (required by ProcessedDataset) but with
    descending depth values as a coordinate.
    """
    n_beams = 4
    n_cells = 10
    n_time = 50

    # Cell indices
    cell_indices = np.arange(n_cells)

    # Descending depth values (surface at higher index) - simulates upward-looking ADCP
    depth_values = np.linspace(50, 5, n_cells)  # 50m to 5m (descending)

    np.random.seed(42)
    velocity = np.random.randint(-1000, 1000, size=(n_beams, n_cells, n_time)).astype(
        np.float32
    )

    ds = xr.Dataset(
        {
            "velocity": (["beam", "cell", "time"], velocity, {"units": "mm/s"}),
        },
        coords={
            "time": np.arange(n_time),
            "cell": cell_indices,
            "depth": ("cell", depth_values),  # depth as a coordinate on cell dimension
            "beam": np.arange(n_beams),
        },
    )

    return ds


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file output tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# SHARED HELPERS FOR apply_config TESTS
# ============================================================================


def _make_runner_mock(ds):
    """Return a fully-configured mock runner that satisfies finalize()."""
    m = MagicMock()
    m.finalize.return_value = ds
    m.get_pipeline_report.return_value = MagicMock()
    m.statistics = []
    m.modifications = []
    return m


def _all_disabled_config(**overrides):
    """ProcessingConfig with every stage flag False, plus optional overrides."""
    try:
        from pyadps.processing.config import ProcessingConfig as _PC
    except ImportError:
        from config import ProcessingConfig as _PC
    cfg = _PC(
        isTimeAxisModified=False,
        isSensorTest=False,
        isQCTest=False,
        isProfileTest=False,
        isVelocityTest=False,
        isAttributes=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ============================================================================
# CUTREGION TESTS
# ============================================================================


class TestCutRegion:
    """Tests for CutRegion helper class."""

    def test_init_defaults(self):
        """Test CutRegion initialization with defaults."""
        region = CutRegion()
        assert region.min_cell is None
        assert region.max_cell is None
        assert region.min_ensemble is None
        assert region.max_ensemble is None

    def test_init_with_values(self):
        """Test CutRegion initialization with values."""
        region = CutRegion(min_cell=0, max_cell=5, min_ensemble=10, max_ensemble=20)
        assert region.min_cell == 0
        assert region.max_cell == 5
        assert region.min_ensemble == 10
        assert region.max_ensemble == 20

    def test_init_partial_values(self):
        """Test CutRegion with partial values."""
        region = CutRegion(min_cell=5, max_ensemble=100)
        assert region.min_cell == 5
        assert region.max_cell is None
        assert region.min_ensemble is None
        assert region.max_ensemble == 100

    def test_to_list(self):
        """Test conversion to list format."""
        region = CutRegion(min_cell=0, max_cell=5, min_ensemble=10, max_ensemble=20)
        result = region.to_list()
        assert result == [0, 5, 10, 20]

    def test_to_list_with_none(self):
        """Test conversion to list with None values."""
        region = CutRegion(min_cell=0, max_cell=5)
        result = region.to_list()
        assert result == [0, 5, None, None]

    def test_from_list(self):
        """Test creation from list format."""
        region = CutRegion.from_list([0, 5, 10, 20])
        assert region.min_cell == 0
        assert region.max_cell == 5
        assert region.min_ensemble == 10
        assert region.max_ensemble == 20

    def test_from_list_with_none(self):
        """Test creation from list with None values."""
        region = CutRegion.from_list([None, 5, None, None])
        assert region.min_cell is None
        assert region.max_cell == 5
        assert region.min_ensemble is None
        assert region.max_ensemble is None

    def test_from_list_invalid_length(self):
        """Test that from_list raises error for invalid list length."""
        with pytest.raises(ValueError, match="Expected list of 4 values"):
            CutRegion.from_list([1, 2, 3])

    def test_repr(self):
        """Test string representation."""
        region = CutRegion(min_cell=0, max_cell=5)
        repr_str = repr(region)
        assert "CutRegion" in repr_str
        assert "min_cell=0" in repr_str
        assert "max_cell=5" in repr_str

    def test_roundtrip(self):
        """Test conversion to list and back."""
        original = CutRegion(min_cell=1, max_cell=10, min_ensemble=5, max_ensemble=50)
        recreated = CutRegion.from_list(original.to_list())
        assert original.min_cell == recreated.min_cell
        assert original.max_cell == recreated.max_cell
        assert original.min_ensemble == recreated.min_ensemble
        assert original.max_ensemble == recreated.max_ensemble


# ============================================================================
# PROCESSEDDATASET INITIALIZATION TESTS
# ============================================================================


class TestProcessedDatasetInit:
    """Tests for ProcessedDataset initialization."""

    def test_init_valid_dataset(self, sample_dataset):
        """Test initialization with valid dataset."""
        proc = ProcessedDataset(sample_dataset)
        assert proc.ds_orig is not None
        assert proc.dataset is not None
        assert "mask" in proc.dataset.data_vars

    def test_init_creates_mask(self, sample_dataset):
        """Test that initialization creates mask if not present."""
        proc = ProcessedDataset(sample_dataset)
        assert "mask" in proc.dataset.data_vars
        mask = proc.dataset["mask"]
        assert mask.shape == sample_dataset["velocity"].shape

    def test_init_preserves_existing_mask(self, sample_dataset_with_mask):
        """Test that initialization preserves existing mask."""
        original_mask = sample_dataset_with_mask["mask"].values.copy()
        proc = ProcessedDataset(sample_dataset_with_mask)
        np.testing.assert_array_equal(proc.dataset["mask"].values, original_mask)

    def test_init_immutable_original(self, sample_dataset):
        """Test that original dataset is immutable (deep copy)."""
        proc = ProcessedDataset(sample_dataset)
        # Modify working dataset
        proc.dataset["velocity"].values[0, 0, 0] = 99999
        # Original should be unchanged
        assert proc.ds_orig["velocity"].values[0, 0, 0] != 99999

    def test_init_invalid_type(self):
        """Test that initialization fails with invalid type."""
        with pytest.raises(TypeError, match="Expected xarray.Dataset"):
            ProcessedDataset("not a dataset")

    def test_init_missing_velocity(self):
        """Test that initialization fails without velocity variable."""
        ds = xr.Dataset({"other_var": (["x"], [1, 2, 3])})
        with pytest.raises(ValueError, match="must contain 'velocity' variable"):
            ProcessedDataset(ds)

    def test_init_baseline_stats(self, sample_dataset):
        """Test that baseline statistics are computed."""
        proc = ProcessedDataset(sample_dataset)
        assert proc._total_cells > 0
        assert proc._baseline_masked >= 0
        assert 0 <= proc._baseline_masked_pct <= 100

    def test_init_empty_reports(self, sample_dataset):
        """Test that reports list is initialized empty."""
        proc = ProcessedDataset(sample_dataset)
        assert proc.reports == []
        assert proc.processing_log == []

    def test_init_blank_config(self, sample_dataset):
        """Test that self.config is a blank ProcessingConfig on init."""
        try:
            from pyadps.processing.config import ProcessingConfig
        except ImportError:
            from config import ProcessingConfig

        proc = ProcessedDataset(sample_dataset)
        assert isinstance(proc.config, ProcessingConfig)
        # All stage flags must be False — nothing has been applied yet
        assert proc.config.isSensorTest is False
        assert proc.config.isQCTest is False
        assert proc.config.isProfileTest is False
        assert proc.config.isVelocityTest is False
        assert proc.config.isTimeAxisModified is False


# ============================================================================
# PROCESSING PIPELINE TESTS
# ============================================================================


class TestApplyTimeAxis:
    """Tests for apply_time_axis method."""

    def test_apply_time_axis_no_action(self, sample_dataset):
        """Test that apply_time_axis returns self when no options enabled."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.apply_time_axis()
        assert result is proc  # Method chaining

    def test_apply_time_axis_returns_self(self, sample_dataset):
        """Test method chaining."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.apply_time_axis(snap=False, fill_gaps=False)
        assert result is proc

    def test_apply_time_axis_snap_success_updates_dataset(self, sample_dataset):
        """snap=True with a successful snap must update self.dataset and log
        the result (lines 309-312).

        We use a real datetime-indexed dataset so snap_time_axis returns
        (new_ds, True, msg) without any mocking — no import-path gymnastics.
        The fixture time coords are already on hourly boundaries so snapping
        to 'h' is a no-op displacement, but the function still returns success.
        """
        import pandas as pd

        n_beams, n_cells, n_time = 4, 5, 10
        times = pd.date_range("2024-01-01 00:00", periods=n_time, freq="h")
        ds_dt = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "time"],
                    np.zeros((n_beams, n_cells, n_time), dtype=np.float32),
                    {"units": "mm/s"},
                ),
            },
            coords={
                "beam": np.arange(n_beams),
                "cell": np.arange(n_cells),
                "time": times,
            },
        )

        proc = ProcessedDataset(ds_dt)
        proc.apply_time_axis(snap=True, snap_freq="h", snap_tolerance="5min")

        # Snap succeeds (or is silently skipped) — either way a snap entry is logged
        assert len(proc.processing_log) > 0
        assert any("snap" in e.lower() for e in proc.processing_log)

    def test_apply_time_axis_snap_failure_keeps_original(self, sample_dataset):
        """snap=True with a failed snap must keep self.dataset unchanged (else branch).

        The real snap_time_axis returns (None, False, msg) when the time
        coordinate is not datetime-like — integer-indexed fixture triggers this.
        No mocking needed.
        """
        proc = ProcessedDataset(sample_dataset)
        dataset_before = proc.dataset.copy(deep=True)

        proc.apply_time_axis(snap=True)

        xr.testing.assert_identical(proc.dataset, dataset_before)
        assert any("snap" in entry.lower() for entry in proc.processing_log)

    def test_apply_time_axis_fill_gaps_success(self, sample_dataset):
        """fill_gaps=True with successful fill must update self.dataset and log
        the result (lines 325-327).

        We use a real hourly datetime-indexed dataset with a deliberate gap so
        fill_time_gaps actually succeeds and returns an extended dataset.
        """
        import pandas as pd

        n_beams, n_cells = 4, 5
        times = pd.DatetimeIndex(
            pd.date_range("2024-01-01 00:00", periods=5, freq="h").tolist()
            + pd.date_range("2024-01-01 06:00", periods=5, freq="h").tolist()
        )
        n_time = len(times)
        ds_dt = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "time"],
                    np.zeros((n_beams, n_cells, n_time), dtype=np.float32),
                    {"units": "mm/s"},
                ),
            },
            coords={
                "beam": np.arange(n_beams),
                "cell": np.arange(n_cells),
                "time": times,
            },
        )

        proc = ProcessedDataset(ds_dt)
        n_before = proc.dataset.sizes["time"]
        proc.apply_time_axis(fill_gaps=True, fill_method="h")

        assert proc.dataset.sizes["time"] >= n_before
        assert any(
            "filled" in e.lower() or "gap" in e.lower() for e in proc.processing_log
        )

    def test_apply_time_axis_fill_gaps_failure_logs_error(self, sample_dataset):
        """fill_gaps=True when fill_time_gaps raises must log the error without
        re-raising (inner except branch).

        Integer-indexed fixture has no datetime time coordinate so
        fill_time_gaps raises — exercising the except branch with no mocking.
        """
        proc = ProcessedDataset(sample_dataset)
        proc.apply_time_axis(fill_gaps=True)  # must not raise

        assert any("failed" in entry.lower() for entry in proc.processing_log)


class TestApplySensorHealth:
    """Tests for apply_sensor_health method."""

    def test_apply_sensor_health_no_action(self, sample_dataset):
        """Test that apply_sensor_health returns self when no options enabled."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.apply_sensor_health()
        assert result is proc
        assert len(proc.reports) == 0

    def test_apply_sensor_health_returns_self(self, sample_dataset):
        """Test method chaining."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.apply_sensor_health(roll=False, pitch=False)
        assert result is proc

    @patch(SENSOR_HEALTH_RUNNER)
    def test_apply_sensor_health_roll_check(self, mock_runner_class, sample_dataset):
        """Test that roll check calls SensorHealthRunner correctly."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_runner_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_sensor_health(roll=True, roll_threshold=15.0)

        mock_runner.roll_check.assert_called_once_with(threshold=15.0)

    @patch(SENSOR_HEALTH_RUNNER)
    def test_apply_sensor_health_correct_sound_speed(
        self, mock_runner_class, sample_dataset
    ):
        """correct_sound_speed=True must call runner.correct_sound_speed (line 487)."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_runner_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_sensor_health(
            correct_sound_speed=True,
            correct_velocity=True,
            horizontal_only=False,
        )

        mock_runner.correct_sound_speed.assert_called_once_with(
            correct_velocity=True, horizontal_only=False
        )

    @patch(SENSOR_HEALTH_RUNNER)
    def test_apply_sensor_health_temperature_as_xr_dataarray(
        self, mock_runner_class, sample_dataset
    ):
        """Passing temperature as xr.DataArray extracts .values (lines 463-464)."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_runner_class.return_value = mock_runner

        arr = xr.DataArray(np.full(100, 15.0), dims=["time"])
        proc = ProcessedDataset(sample_dataset)
        proc.apply_sensor_health(temperature=arr, roll=True, roll_threshold=15.0)

        # replace_data must be called with the underlying numpy array
        call_args = mock_runner.replace_data.call_args
        data_arg = call_args[0][0]
        assert isinstance(data_arg, np.ndarray)
        np.testing.assert_array_equal(data_arg, np.full(100, 15.0))

    @patch(SENSOR_HEALTH_RUNNER)
    def test_apply_sensor_health_temperature_as_list(
        self, mock_runner_class, sample_dataset
    ):
        """Passing temperature as a Python list converts it to numpy (lines 465-466)."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_runner_class.return_value = mock_runner

        temp_list = [15.0] * 100
        proc = ProcessedDataset(sample_dataset)
        proc.apply_sensor_health(temperature=temp_list, roll=True, roll_threshold=15.0)

        call_args = mock_runner.replace_data.call_args
        data_arg = call_args[0][0]
        assert isinstance(data_arg, np.ndarray)
        np.testing.assert_array_equal(data_arg, np.full(100, 15.0))

    @patch(SENSOR_HEALTH_RUNNER)
    def test_apply_sensor_health_temperature_as_numpy_array(
        self, mock_runner_class, sample_dataset
    ):
        """Passing temperature as a numpy array uses it directly (lines 467-469)."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_runner_class.return_value = mock_runner

        temp_np = np.full(100, 18.0)
        proc = ProcessedDataset(sample_dataset)
        proc.apply_sensor_health(temperature=temp_np, roll=True, roll_threshold=15.0)

        call_args = mock_runner.replace_data.call_args
        data_arg = call_args[0][0]
        assert isinstance(data_arg, np.ndarray)
        np.testing.assert_array_equal(data_arg, np.full(100, 18.0))


class TestApplySignalQuality:
    """Tests for apply_signal_quality method."""

    def test_apply_signal_quality_no_action(self, sample_dataset):
        """Test that apply_signal_quality returns self when no thresholds set."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.apply_signal_quality()
        assert result is proc
        assert len(proc.reports) == 0

    def test_apply_signal_quality_returns_self(self, sample_dataset):
        """Test method chaining."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.apply_signal_quality(correlation=None)
        assert result is proc

    @patch(SIGNAL_QUALITY_RUNNER)
    def test_apply_signal_quality_correlation(self, mock_runner_class, sample_dataset):
        """Test that correlation check is called correctly."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_signal_quality(correlation=64)

        mock_runner.correlation.assert_called_once()


class TestApplyProfileOperation:
    """Tests for apply_profile_operation method."""

    def test_apply_profile_operation_no_action(self, sample_dataset):
        """Test that apply_profile_operation returns self when no options enabled."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.apply_profile_operation()
        assert result is proc
        assert len(proc.reports) == 0

    def test_apply_profile_operation_returns_self(self, sample_dataset):
        """Test method chaining."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.apply_profile_operation(regrid=False)
        assert result is proc

    def test_apply_profile_operation_cut_bins_manual_list(self, sample_dataset):
        """Test manual bin cutting with list format."""
        with patch(PROFILE_OPERATION_RUNNER) as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.finalize.return_value = sample_dataset
            mock_runner.get_pipeline_report.return_value = MagicMock()
            mock_runner.statistics = []
            mock_runner.modifications = []
            mock_runner_class.return_value = mock_runner

            proc = ProcessedDataset(sample_dataset)
            proc.apply_profile_operation(
                cut_bins_manual=[[0, 5, None, None], [10, 15, 0, 50]]
            )

            # Should be called twice (once for each region)
            assert mock_runner.cut_bins_manual.call_count == 2

    def test_apply_profile_operation_cut_bins_manual_cutregion(self, sample_dataset):
        """Test manual bin cutting with CutRegion objects."""
        with patch(PROFILE_OPERATION_RUNNER) as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.finalize.return_value = sample_dataset
            mock_runner.get_pipeline_report.return_value = MagicMock()
            mock_runner.statistics = []
            mock_runner.modifications = []
            mock_runner_class.return_value = mock_runner

            proc = ProcessedDataset(sample_dataset)
            proc.apply_profile_operation(
                cut_bins_manual=[CutRegion(min_cell=0, max_cell=5)]
            )

            mock_runner.cut_bins_manual.assert_called_once_with(
                min_cell=0, max_cell=5, min_ensemble=None, max_ensemble=None
            )


class TestApplyVelocityCheck:
    """Tests for apply_velocity_check method."""

    def test_apply_velocity_check_no_action(self, sample_dataset):
        """Test that apply_velocity_check returns self when no options enabled."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.apply_velocity_check()
        assert result is proc
        assert len(proc.reports) == 0

    def test_apply_velocity_check_returns_self(self, sample_dataset):
        """Test method chaining."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.apply_velocity_check(cutoff_u=None)
        assert result is proc

    @patch(VELOCITY_CHECK_RUNNER)
    def test_apply_velocity_check_threshold(self, mock_runner_class, sample_dataset):
        """Test that threshold check is called correctly."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_runner_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_velocity_check(cutoff_u=2500, cutoff_v=2500, cutoff_w=500)

        mock_runner.threshold.assert_called_once()

    @patch(VELOCITY_CHECK_RUNNER)
    def test_apply_velocity_check_magnetic_correction_calls_runner(
        self, mock_runner_class, sample_dataset
    ):
        """magnetic_correction=True must call runner.magnetic_correction (line 964)."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_runner_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_velocity_check(
            magnetic_correction=True,
            declination=5.0,
            lat=12.0,
            lon=80.0,
            year=2024.0,
        )

        mock_runner.magnetic_correction.assert_called_once_with(
            declination=5.0,
            use_api=False,
            lat=12.0,
            lon=80.0,
            year=2024.0,
        )

    @patch(VELOCITY_CHECK_RUNNER)
    def test_apply_velocity_check_records_declination_method(
        self, mock_runner_class, sample_dataset
    ):
        """When declination is provided, config records method='user' and the
        value (lines 1015-1017)."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_runner_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_velocity_check(
            magnetic_correction=True,
            declination=-3.5,
        )

        assert proc.config.magnet_method_VT == "user"
        assert proc.config.magnet_user_input_VT == -3.5

    @patch(VELOCITY_CHECK_RUNNER)
    def test_apply_velocity_check_records_api_method(
        self, mock_runner_class, sample_dataset
    ):
        """When use_api=True and no declination, config records method='api'
        (lines 1018-1019)."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_runner_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_velocity_check(
            magnetic_correction=True,
            use_api=True,
        )

        assert proc.config.magnet_method_VT == "api"


# ============================================================================
# FINALIZE AND OUTPUT TESTS
# ============================================================================


class TestFinalize:
    """Tests for finalize method."""

    def test_finalize_returns_dataset(self, sample_dataset):
        """Test that finalize returns xarray Dataset."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.finalize()
        assert isinstance(result, xr.Dataset)

    def test_finalize_adds_metadata(self, sample_dataset):
        """Test that finalize adds processing metadata."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.finalize()
        assert "pyadps_version" in result.attrs
        assert "processed_at" in result.attrs
        assert "total_cells" in result.attrs
        assert "final_masked" in result.attrs

    def test_finalize_preserves_variables(self, sample_dataset):
        """Test that finalize preserves original variables."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.finalize()
        assert "velocity" in result.data_vars
        assert "mask" in result.data_vars

    def test_finalize_depth_ascending_default(self, sample_dataset_descending_depth):
        """Test that depth is reordered to ascending by default."""
        proc = ProcessedDataset(sample_dataset_descending_depth)
        result = proc.finalize(ensure_depth_ascending=True)
        depth_values = result.coords["depth"].values
        assert depth_values[0] < depth_values[-1]  # Ascending order

    def test_finalize_depth_ascending_disabled(self, sample_dataset_descending_depth):
        """Test that depth order is preserved when disabled."""
        proc = ProcessedDataset(sample_dataset_descending_depth)
        original_depth = sample_dataset_descending_depth.coords["depth"].values.copy()
        result = proc.finalize(ensure_depth_ascending=False)
        np.testing.assert_array_equal(result.coords["depth"].values, original_depth)

    def test_finalize_updates_log(self, sample_dataset):
        """Test that finalize updates processing log."""
        proc = ProcessedDataset(sample_dataset)
        initial_log_len = len(proc.processing_log)
        proc.finalize()
        assert len(proc.processing_log) > initial_log_len


class TestToNetcdf:
    """Tests for to_netcdf method."""

    def test_to_netcdf_creates_file(self, sample_dataset, temp_dir):
        """Test that to_netcdf creates a file."""
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "output.nc"
        proc.to_netcdf(filepath)
        assert filepath.exists()

    def test_to_netcdf_readable(self, sample_dataset, temp_dir):
        """Test that output file is readable."""
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "output.nc"
        proc.to_netcdf(filepath)

        ds_read = xr.open_dataset(filepath)
        assert "velocity" in ds_read.data_vars
        ds_read.close()

    def test_to_netcdf_creates_parent_dirs(self, sample_dataset, temp_dir):
        """Test that to_netcdf creates parent directories."""
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "subdir" / "nested" / "output.nc"
        proc.to_netcdf(filepath)
        assert filepath.exists()

    def test_to_netcdf_depth_ascending(self, sample_dataset_descending_depth, temp_dir):
        """Test depth ordering in output file."""
        proc = ProcessedDataset(sample_dataset_descending_depth)
        filepath = temp_dir / "output.nc"
        proc.to_netcdf(filepath, ensure_depth_ascending=True)

        ds_read = xr.open_dataset(filepath)
        depth_values = ds_read.coords["depth"].values
        assert depth_values[0] < depth_values[-1]
        ds_read.close()


class TestVelocityToNetcdf:
    """Tests for velocity_to_netcdf method."""

    def test_velocity_to_netcdf_creates_file(self, sample_dataset, temp_dir):
        """Test that velocity_to_netcdf creates a file."""
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "velocities.nc"
        proc.velocity_to_netcdf(filepath)
        assert filepath.exists()

    def test_velocity_to_netcdf_contains_components(self, sample_dataset, temp_dir):
        """Test that output contains u, v, w components."""
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "velocities.nc"
        proc.velocity_to_netcdf(filepath)

        ds_read = xr.open_dataset(filepath)
        assert "zonal_velocity" in ds_read.data_vars
        assert "meridional_velocity" in ds_read.data_vars
        assert "vertical_velocity" in ds_read.data_vars
        ds_read.close()

    def test_velocity_to_netcdf_2d_arrays(self, sample_dataset, temp_dir):
        """Test that velocity components are 2D (not 3D)."""
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "velocities.nc"
        proc.velocity_to_netcdf(filepath)

        ds_read = xr.open_dataset(filepath)
        assert len(ds_read["zonal_velocity"].dims) == 2
        ds_read.close()

    def test_velocity_to_netcdf_excludes_error_velocity(self, sample_dataset, temp_dir):
        """Test that error velocity (beam 3) is excluded."""
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "velocities.nc"
        proc.velocity_to_netcdf(filepath)

        ds_read = xr.open_dataset(filepath)
        # Should only have 3 velocity variables (u, v, w), not 4
        vel_vars = [v for v in ds_read.data_vars if "velocity" in v]
        assert len(vel_vars) == 3
        ds_read.close()

    def test_velocity_to_netcdf_carries_raw_provenance_attrs(
        self, sample_dataset, temp_dir
    ):
        """'filename'/'adcp_data_format' set by pyadps.read() survive export."""
        proc = ProcessedDataset(sample_dataset)
        proc.dataset.attrs["filename"] = "GD15A000.000"
        proc.dataset.attrs["adcp_data_format"] = "PD0"
        filepath = temp_dir / "velocities.nc"
        proc.velocity_to_netcdf(filepath)

        ds_read = xr.open_dataset(filepath)
        assert ds_read.attrs["filename"] == "GD15A000.000"
        assert ds_read.attrs["adcp_data_format"] == "PD0"
        ds_read.close()

    def test_velocity_to_netcdf_components_have_description_and_source(
        self, sample_dataset, temp_dir
    ):
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "velocities.nc"
        proc.velocity_to_netcdf(filepath)

        ds_read = xr.open_dataset(filepath)
        for name in ("zonal_velocity", "meridional_velocity", "vertical_velocity"):
            assert (
                ds_read[name].attrs["description"]
                == "Velocity magnitude measured by ADCP"
            )
            assert ds_read[name].attrs["source"] == "RDI WorkHorse ADCP"
        ds_read.close()

    def test_velocity_to_netcdf_custom_names(self, sample_dataset, temp_dir):
        """Test custom variable names."""
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "velocities.nc"
        proc.velocity_to_netcdf(
            filepath, velocity_names={"u": "u_vel", "v": "v_vel", "w": "w_vel"}
        )

        ds_read = xr.open_dataset(filepath)
        assert "u_vel" in ds_read.data_vars
        assert "v_vel" in ds_read.data_vars
        assert "w_vel" in ds_read.data_vars
        ds_read.close()

    def test_velocity_to_netcdf_units_default_cm_s(self, sample_dataset, temp_dir):
        """Test that default units are cm/s."""
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "velocities.nc"
        proc.velocity_to_netcdf(filepath)

        ds_read = xr.open_dataset(filepath)
        assert ds_read["zonal_velocity"].attrs["units"] == "cm/s"
        ds_read.close()

    def test_velocity_to_netcdf_units_mm_s(self, sample_dataset, temp_dir):
        """Test mm/s unit output."""
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "velocities.nc"
        proc.velocity_to_netcdf(filepath, units="mm/s")

        ds_read = xr.open_dataset(filepath)
        assert ds_read["zonal_velocity"].attrs["units"] == "mm/s"
        ds_read.close()

    def test_velocity_to_netcdf_units_m_s(self, sample_dataset, temp_dir):
        """Test m/s unit output."""
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "velocities.nc"
        proc.velocity_to_netcdf(filepath, units="m/s")

        ds_read = xr.open_dataset(filepath)
        assert ds_read["zonal_velocity"].attrs["units"] == "m/s"
        ds_read.close()

    def test_velocity_to_netcdf_unit_conversion_values(self, sample_dataset, temp_dir):
        """Test that unit conversion produces correct values."""
        proc = ProcessedDataset(sample_dataset)

        # Export in mm/s
        filepath_mm = temp_dir / "vel_mm.nc"
        proc.velocity_to_netcdf(filepath_mm, units="mm/s", apply_mask=False)

        # Export in cm/s
        filepath_cm = temp_dir / "vel_cm.nc"
        proc.velocity_to_netcdf(filepath_cm, units="cm/s", apply_mask=False)

        # Export in m/s
        filepath_m = temp_dir / "vel_m.nc"
        proc.velocity_to_netcdf(filepath_m, units="m/s", apply_mask=False)

        ds_mm = xr.open_dataset(filepath_mm)
        ds_cm = xr.open_dataset(filepath_cm)
        ds_m = xr.open_dataset(filepath_m)

        # Check conversion ratios (with tolerance for floating point)
        u_mm = ds_mm["zonal_velocity"].values
        u_cm = ds_cm["zonal_velocity"].values
        u_m = ds_m["zonal_velocity"].values

        # mm/s * 0.1 = cm/s
        np.testing.assert_allclose(u_mm * 0.1, u_cm, rtol=1e-5)
        # mm/s * 0.001 = m/s
        np.testing.assert_allclose(u_mm * 0.001, u_m, rtol=1e-5)

        ds_mm.close()
        ds_cm.close()
        ds_m.close()

    def test_velocity_to_netcdf_invalid_units(self, sample_dataset, temp_dir):
        """Test that invalid units raise ValueError."""
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "velocities.nc"
        with pytest.raises(ValueError, match="Invalid units"):
            proc.velocity_to_netcdf(filepath, units="km/s")

    def test_velocity_to_netcdf_apply_mask(self, sample_dataset_with_mask, temp_dir):
        """Test that mask is applied correctly."""
        proc = ProcessedDataset(sample_dataset_with_mask)
        filepath = temp_dir / "velocities.nc"
        proc.velocity_to_netcdf(filepath, apply_mask=True)

        ds_read = xr.open_dataset(filepath)
        u_data = ds_read["zonal_velocity"].values
        # First cell should be NaN (masked)
        assert np.all(np.isnan(u_data[0, :]))
        ds_read.close()

    def test_velocity_to_netcdf_no_mask(self, sample_dataset_with_mask, temp_dir):
        """Test export without applying mask."""
        proc = ProcessedDataset(sample_dataset_with_mask)
        filepath = temp_dir / "velocities.nc"
        proc.velocity_to_netcdf(filepath, apply_mask=False)

        ds_read = xr.open_dataset(filepath)
        u_data = ds_read["zonal_velocity"].values
        # First cell should NOT be all NaN
        assert not np.all(np.isnan(u_data[0, :]))
        ds_read.close()

    def test_velocity_to_netcdf_metadata(self, sample_dataset, temp_dir):
        """Test that metadata is included."""
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "velocities.nc"
        proc.velocity_to_netcdf(filepath, include_metadata=True)

        ds_read = xr.open_dataset(filepath)
        assert "source" in ds_read.attrs
        assert "original_velocity_units" in ds_read.attrs
        assert "output_velocity_units" in ds_read.attrs
        ds_read.close()

    def test_velocity_to_netcdf_no_metadata(self, sample_dataset, temp_dir):
        """Test export without metadata."""
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "velocities.nc"
        proc.velocity_to_netcdf(filepath, include_metadata=False)

        ds_read = xr.open_dataset(filepath)
        # Should have minimal attributes
        assert "original_velocity_units" not in ds_read.attrs
        ds_read.close()


class TestGetVelocityDataset:
    """Tests for get_velocity_dataset method."""

    def test_get_velocity_dataset_returns_dataset(self, sample_dataset):
        """Test that get_velocity_dataset returns xarray Dataset."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_velocity_dataset()
        assert isinstance(result, xr.Dataset)

    def test_get_velocity_dataset_contains_components(self, sample_dataset):
        """Test that result contains u, v, w components."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_velocity_dataset()
        assert "zonal_velocity" in result.data_vars
        assert "meridional_velocity" in result.data_vars
        assert "vertical_velocity" in result.data_vars

    def test_get_velocity_dataset_units_default(self, sample_dataset):
        """Test default units are cm/s."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_velocity_dataset()
        assert result["zonal_velocity"].attrs["units"] == "cm/s"

    def test_get_velocity_dataset_units_m_s(self, sample_dataset):
        """Test m/s units."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_velocity_dataset(units="m/s")
        assert result["zonal_velocity"].attrs["units"] == "m/s"

    def test_get_velocity_dataset_custom_names(self, sample_dataset):
        """Test custom variable names."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_velocity_dataset(
            velocity_names={"u": "east", "v": "north", "w": "up"}
        )
        assert "east" in result.data_vars
        assert "north" in result.data_vars
        assert "up" in result.data_vars


# ============================================================================
# COVERAGE GAP TESTS: save_netcdf / velocity_to_netcdf / get_velocity_dataset
# ============================================================================


class TestSaveNetcdfPathFallback:
    """Tests for save_netcdf() output-directory resolution (line 1586).

    The precedence chain is:
      1. explicit output_dir argument
      2. cfg.output_file_path
      3. cfg.input_file_path
      4. Path(".")  ← line 1586, covered here
    """

    def test_fallback_to_cwd_when_no_paths_in_config(self, sample_dataset, temp_dir):
        """When config has no path fields, save_netcdf writes into Path(".").

        We cannot predict the working directory in CI, so we patch to_netcdf
        and capture the output_path argument passed to it.
        """
        try:
            from pyadps.processing.config import ProcessingConfig as _PC
        except ImportError:
            from config import ProcessingConfig as _PC

        # Config with no output or input file paths
        cfg = _PC()
        assert not cfg.output_file_path
        assert not cfg.input_file_path

        proc = ProcessedDataset(sample_dataset)
        captured = {}

        def fake_to_netcdf(path, **kw):
            captured["path"] = path

        with (
            patch.object(proc, "to_netcdf", side_effect=fake_to_netcdf),
            patch.object(proc, "print_summary"),
        ):
            proc.save_netcdf(cfg, output_filename="out.nc", print_summary=False)

        assert "path" in captured
        # The parent must be Path(".") resolved — i.e. the current working directory
        assert captured["path"].parent.resolve() == Path(".").resolve()

    def test_fallback_filename_uses_adcp_stem_when_no_input_file_name(
        self, sample_dataset
    ):
        """When cfg.input_file_name is empty, filename stem defaults to 'adcp'."""
        try:
            from pyadps.processing.config import ProcessingConfig as _PC
        except ImportError:
            from config import ProcessingConfig as _PC

        cfg = _PC()
        assert not cfg.input_file_name

        proc = ProcessedDataset(sample_dataset)
        captured = {}

        def fake_to_netcdf(path, **kw):
            captured["path"] = path

        with (
            patch.object(proc, "to_netcdf", side_effect=fake_to_netcdf),
            patch.object(proc, "print_summary"),
        ):
            proc.save_netcdf(cfg, print_summary=False)

        assert captured["path"].name == "adcp_processed.nc"

    def test_output_dir_arg_overrides_config_paths(self, sample_dataset, temp_dir):
        """Explicit output_dir takes precedence over all config path fields."""
        try:
            from pyadps.processing.config import ProcessingConfig as _PC
        except ImportError:
            from config import ProcessingConfig as _PC

        # Config nominally has an input path set
        cfg = _PC(input_file_path=str(temp_dir / "other"))
        out = temp_dir / "explicit_dir"

        proc = ProcessedDataset(sample_dataset)
        captured = {}

        def fake_to_netcdf(path, **kw):
            captured["path"] = path

        with (
            patch.object(proc, "to_netcdf", side_effect=fake_to_netcdf),
            patch.object(proc, "print_summary"),
        ):
            proc.save_netcdf(
                cfg, output_dir=out, output_filename="result.nc", print_summary=False
            )

        assert captured["path"].parent.resolve() == out.resolve()


class TestVelocityToNetcdfCoverageGaps:
    """Tests for uncovered branches in velocity_to_netcdf().

    Covers:
      - Line 2034: ValueError when finalize() returns dataset without 'velocity'
      - Line 2043-2045: ValueError when velocity is not 3-D
      - Lines 2058/2061: fallback dim detection when no known dim names matched
      - Line 2179: source-dataset attrs copied to output when include_metadata=True
    """

    # ------------------------------------------------------------------
    # Line 2034: no velocity variable in finalized dataset
    # ------------------------------------------------------------------

    def test_raises_when_velocity_missing_from_finalized_dataset(
        self, sample_dataset, temp_dir
    ):
        """velocity_to_netcdf raises ValueError if finalize() yields no 'velocity'."""
        proc = ProcessedDataset(sample_dataset)
        ds_no_vel = xr.Dataset({"pressure": (["time"], np.zeros(10))})

        with patch.object(proc, "finalize", return_value=ds_no_vel):
            with pytest.raises(ValueError, match="does not contain 'velocity'"):
                proc.velocity_to_netcdf(temp_dir / "out.nc")

    # ------------------------------------------------------------------
    # Lines 2043-2045: velocity is not 3-D
    # ------------------------------------------------------------------

    def test_raises_when_velocity_is_2d(self, sample_dataset, temp_dir):
        """velocity_to_netcdf raises ValueError if velocity has only 2 dims."""
        proc = ProcessedDataset(sample_dataset)
        ds_2d = xr.Dataset(
            {
                "velocity": (
                    ["cell", "time"],
                    np.ones((5, 10), dtype=np.float32),
                    {"units": "mm/s"},
                ),
                "mask": (["cell", "time"], np.zeros((5, 10), dtype=np.int8)),
            },
            coords={"cell": np.arange(5), "time": np.arange(10)},
        )
        with patch.object(proc, "finalize", return_value=ds_2d):
            with pytest.raises(ValueError, match="Expected 3D velocity"):
                proc.velocity_to_netcdf(temp_dir / "out.nc")

    def test_raises_when_velocity_is_4d(self, sample_dataset, temp_dir):
        """velocity_to_netcdf raises ValueError if velocity has 4 dims."""
        proc = ProcessedDataset(sample_dataset)
        ds_4d = xr.Dataset(
            {
                "velocity": (
                    ["a", "b", "c", "d"],
                    np.ones((2, 3, 4, 5), dtype=np.float32),
                    {"units": "mm/s"},
                ),
                "mask": (["a", "b", "c", "d"], np.zeros((2, 3, 4, 5), dtype=np.int8)),
            },
            coords={
                "a": np.arange(2),
                "b": np.arange(3),
                "c": np.arange(4),
                "d": np.arange(5),
            },
        )
        with patch.object(proc, "finalize", return_value=ds_4d):
            with pytest.raises(ValueError, match="Expected 3D velocity"):
                proc.velocity_to_netcdf(temp_dir / "out.nc")

    def test_error_message_includes_actual_ndim(self, sample_dataset, temp_dir):
        """ValueError message includes the actual number of dims found."""
        proc = ProcessedDataset(sample_dataset)
        ds_2d = xr.Dataset(
            {
                "velocity": (
                    ["cell", "time"],
                    np.ones((5, 10), dtype=np.float32),
                    {"units": "mm/s"},
                ),
                "mask": (["cell", "time"], np.zeros((5, 10), dtype=np.int8)),
            },
            coords={"cell": np.arange(5), "time": np.arange(10)},
        )
        with patch.object(proc, "finalize", return_value=ds_2d):
            with pytest.raises(ValueError, match="2D"):
                proc.velocity_to_netcdf(temp_dir / "out.nc")

    # ------------------------------------------------------------------
    # Lines 2058/2061: fallback dim detection (no known dim names)
    # ------------------------------------------------------------------

    def test_fallback_dim_detection_uses_last_dim_as_time(
        self, sample_dataset, temp_dir
    ):
        """When no 'time'/'ensemble'/'cell'/'depth' dim is present, last dim is
        treated as time (line 2058) and middle dim as cell (line 2061).
        Output 2-D arrays have shape (middle_dim, last_dim).
        """
        proc = ProcessedDataset(sample_dataset)
        # Use entirely non-standard dim names: (beam, row, col)
        n_beam, n_row, n_col = 4, 6, 8
        ds_custom = xr.Dataset(
            {
                "velocity": (
                    ["beam", "row", "col"],
                    np.ones((n_beam, n_row, n_col), dtype=np.float32),
                    {"units": "mm/s"},
                ),
                "mask": (
                    ["beam", "row", "col"],
                    np.zeros((n_beam, n_row, n_col), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(n_beam),
                "row": np.arange(n_row),
                "col": np.arange(n_col),
            },
        )
        with patch.object(proc, "finalize", return_value=ds_custom):
            proc.velocity_to_netcdf(temp_dir / "out.nc", apply_mask=False)

        ds_out = xr.open_dataset(temp_dir / "out.nc")
        u = ds_out["zonal_velocity"]
        # cell_dim = vel_dims[1] = "row", time_dim = vel_dims[-1] = "col"
        assert u.dims == ("row", "col")
        assert u.shape == (n_row, n_col)
        ds_out.close()

    # ------------------------------------------------------------------
    # Line 2179: source attrs copied to output when include_metadata=True
    # ------------------------------------------------------------------

    def test_deployment_name_copied_from_source_attrs(self, sample_dataset, temp_dir):
        """deployment_name in ds_final.attrs is copied to output global attrs."""
        proc = ProcessedDataset(sample_dataset)
        ds_with_meta = proc.finalize()
        ds_with_meta.attrs["deployment_name"] = "ArcticMooring2024"

        with patch.object(proc, "finalize", return_value=ds_with_meta):
            proc.velocity_to_netcdf(temp_dir / "out.nc", include_metadata=True)

        result = xr.open_dataset(temp_dir / "out.nc")
        assert result.attrs.get("deployment_name") == "ArcticMooring2024"
        result.close()

    def test_all_six_passthrough_attrs_copied(self, sample_dataset, temp_dir):
        """All six recognised source attrs are copied when present in ds_final."""
        proc = ProcessedDataset(sample_dataset)
        ds_with_meta = proc.finalize()
        expected = {
            "deployment_name": "TestDeployment",
            "instrument_type": "ADCP",
            "serial_number": "SN42",
            "latitude": 45.5,
            "longitude": -30.0,
            "water_depth": 200.0,
        }
        for k, v in expected.items():
            ds_with_meta.attrs[k] = v

        with patch.object(proc, "finalize", return_value=ds_with_meta):
            proc.velocity_to_netcdf(temp_dir / "out.nc", include_metadata=True)

        result = xr.open_dataset(temp_dir / "out.nc")
        for k, v in expected.items():
            assert (
                result.attrs.get(k) == pytest.approx(v)
                if isinstance(v, float)
                else result.attrs.get(k) == v
            ), f"attr '{k}' not copied correctly"
        result.close()

    def test_passthrough_attrs_absent_in_source_not_added(
        self, sample_dataset, temp_dir
    ):
        """Attrs not present in ds_final are NOT added to the output."""
        proc = ProcessedDataset(sample_dataset)
        ds_no_meta = proc.finalize()
        # Ensure none of the six attrs are present
        for a in (
            "deployment_name",
            "instrument_type",
            "serial_number",
            "latitude",
            "longitude",
            "water_depth",
        ):
            ds_no_meta.attrs.pop(a, None)

        with patch.object(proc, "finalize", return_value=ds_no_meta):
            proc.velocity_to_netcdf(temp_dir / "out.nc", include_metadata=True)

        result = xr.open_dataset(temp_dir / "out.nc")
        for a in (
            "deployment_name",
            "instrument_type",
            "serial_number",
            "latitude",
            "longitude",
            "water_depth",
        ):
            assert a not in result.attrs, f"Unexpected attr '{a}' in output"
        result.close()

    def test_subset_of_passthrough_attrs_copied(self, sample_dataset, temp_dir):
        """Only the attrs that exist in ds_final are copied (partial presence)."""
        proc = ProcessedDataset(sample_dataset)
        ds_partial = proc.finalize()
        ds_partial.attrs["serial_number"] = "SN99"
        # latitude / longitude / etc. intentionally absent

        with patch.object(proc, "finalize", return_value=ds_partial):
            proc.velocity_to_netcdf(temp_dir / "out.nc", include_metadata=True)

        result = xr.open_dataset(temp_dir / "out.nc")
        assert result.attrs.get("serial_number") == "SN99"
        assert "latitude" not in result.attrs
        result.close()


class TestGetVelocityDatasetCoverageGaps:
    """Tests for uncovered branches in get_velocity_dataset().

    Covers:
      - Line 2259: ValueError for invalid units
      - Lines 2281/2283: fallback dim detection (no known dim names matched)
    """

    # ------------------------------------------------------------------
    # Line 2259: invalid units
    # ------------------------------------------------------------------

    def test_raises_for_invalid_units(self, sample_dataset):
        """get_velocity_dataset raises ValueError for unrecognised unit string."""
        proc = ProcessedDataset(sample_dataset)
        with pytest.raises(ValueError, match="Invalid units"):
            proc.get_velocity_dataset(units="km/s")

    def test_invalid_units_message_includes_unit_string(self, sample_dataset):
        """Error message contains the bad unit that was passed."""
        proc = ProcessedDataset(sample_dataset)
        with pytest.raises(ValueError, match="knots"):
            proc.get_velocity_dataset(units="knots")

    def test_all_three_valid_units_accepted(self, sample_dataset):
        """All three documented valid unit strings are accepted without error."""
        proc = ProcessedDataset(sample_dataset)
        for u in ("mm/s", "cm/s", "m/s"):
            result = proc.get_velocity_dataset(units=u)
            assert result["zonal_velocity"].attrs["units"] == u

    # ------------------------------------------------------------------
    # Lines 2281/2283: fallback dim detection (no known dim names)
    # ------------------------------------------------------------------

    def test_fallback_dim_detection_uses_last_dim_as_time(self, sample_dataset):
        """When no recognised dim names are present, last dim → time_dim and
        middle dim → cell_dim (lines 2281/2283). Output shape is (n_row, n_col).
        """
        proc = ProcessedDataset(sample_dataset)
        n_beam, n_row, n_col = 4, 6, 8
        ds_custom = xr.Dataset(
            {
                "velocity": (
                    ["beam", "row", "col"],
                    np.ones((n_beam, n_row, n_col), dtype=np.float32),
                    {"units": "mm/s"},
                ),
                "mask": (
                    ["beam", "row", "col"],
                    np.zeros((n_beam, n_row, n_col), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(n_beam),
                "row": np.arange(n_row),
                "col": np.arange(n_col),
            },
        )
        with patch.object(proc, "finalize", return_value=ds_custom):
            result = proc.get_velocity_dataset(apply_mask=False)

        u = result["zonal_velocity"]
        # cell_dim = vel_dims[1] = "row", time_dim = vel_dims[-1] = "col"
        assert u.dims == ("row", "col")
        assert u.shape == (n_row, n_col)

    def test_fallback_only_time_dim_unknown(self, sample_dataset):
        """When only time dim is unrecognised, last dim used for time_dim
        while 'cell' is found normally (only line 2281 fires, not 2283).
        """
        proc = ProcessedDataset(sample_dataset)
        n_beam, n_cell, n_col = 4, 5, 8
        ds_partial = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "col"],
                    np.ones((n_beam, n_cell, n_col), dtype=np.float32),
                    {"units": "mm/s"},
                ),
                "mask": (
                    ["beam", "cell", "col"],
                    np.zeros((n_beam, n_cell, n_col), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(n_beam),
                "cell": np.arange(n_cell),
                "col": np.arange(n_col),
            },
        )
        with patch.object(proc, "finalize", return_value=ds_partial):
            result = proc.get_velocity_dataset(apply_mask=False)

        u = result["zonal_velocity"]
        # cell_dim = "cell" (found normally), time_dim = "col" (fallback)
        assert u.dims == ("cell", "col")

    def test_fallback_only_cell_dim_unknown(self, sample_dataset):
        """When only cell dim is unrecognised, middle dim used for cell_dim
        while 'time' is found normally (only line 2283 fires, not 2281).
        """
        proc = ProcessedDataset(sample_dataset)
        n_beam, n_row, n_time = 4, 6, 10
        ds_partial = xr.Dataset(
            {
                "velocity": (
                    ["beam", "row", "time"],
                    np.ones((n_beam, n_row, n_time), dtype=np.float32),
                    {"units": "mm/s"},
                ),
                "mask": (
                    ["beam", "row", "time"],
                    np.zeros((n_beam, n_row, n_time), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(n_beam),
                "row": np.arange(n_row),
                "time": np.arange(n_time),
            },
        )
        with patch.object(proc, "finalize", return_value=ds_partial):
            result = proc.get_velocity_dataset(apply_mask=False)

        u = result["zonal_velocity"]
        # time_dim = "time" (found normally), cell_dim = "row" (fallback)
        assert u.dims == ("row", "time")


class TestGetExportDataset:
    """Tests for get_export_dataset method."""

    def test_default_is_velocity_only(self, sample_dataset):
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_export_dataset()
        assert set(result.data_vars) == {
            "zonal_velocity",
            "meridional_velocity",
            "vertical_velocity",
        }

    def test_raises_when_nothing_selected(self, sample_dataset):
        proc = ProcessedDataset(sample_dataset)
        with pytest.raises(ValueError, match="At least one component"):
            proc.get_export_dataset(include_velocity=False)

    def test_echo_only(self, sample_dataset):
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_export_dataset(include_velocity=False, include_echo=True)
        assert set(result.data_vars) == {"echo_intensity"}
        assert result["echo_intensity"].dims == ("beam", "cell", "time")

    def test_correlation_only(self, sample_dataset):
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_export_dataset(
            include_velocity=False, include_correlation=True
        )
        assert set(result.data_vars) == {"correlation"}

    def test_percent_good_only(self, sample_dataset):
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_export_dataset(
            include_velocity=False, include_percent_good=True
        )
        assert set(result.data_vars) == {"percent_good"}

    def test_all_components_combined(self, sample_dataset):
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_export_dataset(
            include_velocity=True,
            include_echo=True,
            include_correlation=True,
            include_percent_good=True,
        )
        assert set(result.data_vars) == {
            "zonal_velocity",
            "meridional_velocity",
            "vertical_velocity",
            "echo_intensity",
            "correlation",
            "percent_good",
        }

    def test_velocity_names_applied_within_combined_export(self, sample_dataset):
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_export_dataset(
            include_velocity=True,
            include_echo=True,
            velocity_names={"u": "u", "v": "v", "w": "w"},
        )
        assert {"u", "v", "w", "echo_intensity"} <= set(result.data_vars)

    def test_velocity_components_have_cf_standard_name(self, sample_dataset):
        """
        get_velocity_dataset()'s u/v/w attrs must match velocity_to_netcdf()'s
        (standard_name/positive/comment), not just long_name/units.
        """
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_export_dataset(include_velocity=True)
        assert (
            result["zonal_velocity"].attrs["standard_name"]
            == "eastward_sea_water_velocity"
        )
        assert result["zonal_velocity"].attrs["positive"] == "eastward"
        assert (
            result["meridional_velocity"].attrs["standard_name"]
            == "northward_sea_water_velocity"
        )
        assert result["meridional_velocity"].attrs["positive"] == "northward"
        assert (
            result["vertical_velocity"].attrs["standard_name"]
            == "upward_sea_water_velocity"
        )
        assert result["vertical_velocity"].attrs["positive"] == "upward"

    def test_velocity_components_have_description_and_source(self, sample_dataset):
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_export_dataset(include_velocity=True)
        for name in ("zonal_velocity", "meridional_velocity", "vertical_velocity"):
            assert (
                result[name].attrs["description"]
                == "Velocity magnitude measured by ADCP"
            )
            assert result[name].attrs["source"] == "RDI WorkHorse ADCP"

    def test_mask_never_applied_to_echo(self, sample_dataset_with_mask):
        """
        The mask is velocity-derived (U/V/W/combined failures) and not
        indexed the same way as echo_intensity's physical beams, so it
        must never be applied to echo_intensity - regardless of apply_mask.
        """
        proc = ProcessedDataset(sample_dataset_with_mask)
        result = proc.get_export_dataset(
            include_velocity=False, include_echo=True, apply_mask=True
        )
        # sample_dataset_with_mask masks the first cell across all beams,
        # but echo_intensity must stay raw/unmasked regardless.
        assert not np.any(np.isnan(result["echo_intensity"].isel(cell=0).values))

    def test_mask_never_applied_to_correlation_or_percent_good(
        self, sample_dataset_with_mask
    ):
        proc = ProcessedDataset(sample_dataset_with_mask)
        result = proc.get_export_dataset(
            include_velocity=False,
            include_correlation=True,
            include_percent_good=True,
            apply_mask=True,
        )
        assert not np.any(np.isnan(result["correlation"].isel(cell=0).values))
        assert not np.any(np.isnan(result["percent_good"].isel(cell=0).values))

    def test_mask_still_applied_to_velocity(self, sample_dataset_with_mask):
        """Masking remains velocity-only - that's the actual QC'd product."""
        proc = ProcessedDataset(sample_dataset_with_mask)
        result = proc.get_export_dataset(
            include_velocity=True, apply_mask=True, include_echo=False
        )
        assert np.all(np.isnan(result["zonal_velocity"].isel(cell=0).values))

    def test_mask_not_applied_when_disabled(self, sample_dataset_with_mask):
        proc = ProcessedDataset(sample_dataset_with_mask)
        result = proc.get_export_dataset(
            include_velocity=False, include_echo=True, apply_mask=False
        )
        assert not np.any(np.isnan(result["echo_intensity"].values))

    def test_include_mask_adds_raw_mask_variable(self, sample_dataset_with_mask):
        proc = ProcessedDataset(sample_dataset_with_mask)
        result = proc.get_export_dataset(
            include_velocity=False, include_mask=True
        )
        assert "mask" in result.data_vars
        assert np.all(result["mask"].isel(cell=0).values == 1)

    def test_include_mask_alone_satisfies_component_requirement(
        self, sample_dataset_with_mask
    ):
        """Selecting only the mask (no velocity/echo/etc.) must not raise."""
        proc = ProcessedDataset(sample_dataset_with_mask)
        result = proc.get_export_dataset(
            include_velocity=False, include_mask=True
        )
        assert set(result.data_vars) == {"mask"}


class TestExportToNetcdf:
    """Tests for export_to_netcdf method."""

    def test_creates_file(self, sample_dataset, temp_dir):
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "export.nc"
        proc.export_to_netcdf(filepath)
        assert filepath.exists()

    def test_combined_components_written(self, sample_dataset, temp_dir):
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "export.nc"
        proc.export_to_netcdf(
            filepath, include_velocity=True, include_echo=True, include_correlation=True
        )
        ds_read = xr.open_dataset(filepath)
        assert "zonal_velocity" in ds_read.data_vars
        assert "echo_intensity" in ds_read.data_vars
        assert "correlation" in ds_read.data_vars
        assert "percent_good" not in ds_read.data_vars
        ds_read.close()

    def test_components_exported_attr(self, sample_dataset, temp_dir):
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "export.nc"
        proc.export_to_netcdf(filepath, include_velocity=True, include_percent_good=True)
        ds_read = xr.open_dataset(filepath)
        assert "velocity" in ds_read.attrs["components_exported"]
        assert "percent_good" in ds_read.attrs["components_exported"]
        ds_read.close()

    def test_carries_raw_provenance_attrs(self, sample_dataset, temp_dir):
        """'filename'/'adcp_data_format' set by pyadps.read() survive export."""
        proc = ProcessedDataset(sample_dataset)
        proc.dataset.attrs["filename"] = "GD15A000.000"
        proc.dataset.attrs["adcp_data_format"] = "PD0"
        filepath = temp_dir / "export.nc"
        proc.export_to_netcdf(filepath, include_velocity=True, include_echo=True)

        ds_read = xr.open_dataset(filepath)
        assert ds_read.attrs["filename"] == "GD15A000.000"
        assert ds_read.attrs["adcp_data_format"] == "PD0"
        ds_read.close()

    def test_no_metadata(self, sample_dataset, temp_dir):
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "export.nc"
        proc.export_to_netcdf(filepath, include_metadata=False)
        ds_read = xr.open_dataset(filepath)
        assert "title" not in ds_read.attrs
        assert ds_read.attrs["Conventions"] == "CF-1.8"
        ds_read.close()


class TestDropAmbiguousAxisCoords:
    """
    Tests for ProcessedDataset._drop_ambiguous_axis_coords.

    pyadps.read() keeps 'ensemble'/'cell' around as non-dimension
    coordinates after swapping to 'time'/'depth', so interactive/CLI xarray
    users can switch back. They carry no CF axis metadata though, so
    writing them to NetCDF gives tools like Ferret two coordinate
    candidates for one dimension - this helper strips the redundant one
    only at file-write time. See finalize()/export_to_netcdf() etc.
    """

    @staticmethod
    def _time_primary_dataset():
        """dims=(time, cell, beam); 'ensemble' lingers on the time dim."""
        return xr.Dataset(
            {"var": (["time", "cell", "beam"], np.zeros((3, 2, 4)))},
            coords={
                "time": ("time", np.arange(3), {"axis": "T"}),
                "ensemble": ("time", np.arange(3), {}),
                "cell": ("cell", np.arange(2), {}),
                "depth": ("cell", np.arange(2) * 1.0, {"axis": "Z"}),
                "beam": ("beam", np.arange(4), {"axis": "E"}),
            },
        )

    @staticmethod
    def _depth_primary_dataset():
        """dims=(depth, ensemble, beam); 'cell' lingers on the depth dim."""
        return xr.Dataset(
            {"var": (["depth", "ensemble", "beam"], np.zeros((2, 3, 4)))},
            coords={
                "depth": ("depth", np.arange(2) * 1.0, {"axis": "Z"}),
                "cell": ("depth", np.arange(2), {}),
                "ensemble": ("ensemble", np.arange(3), {}),
                "beam": ("beam", np.arange(4), {"axis": "E"}),
            },
        )

    def test_drops_ensemble_when_time_is_dimension(self):
        ds = self._time_primary_dataset()
        result = ProcessedDataset._drop_ambiguous_axis_coords(ds)
        assert "ensemble" not in result.coords
        assert result.coords["time"].attrs["axis"] == "T"

    def test_drops_cell_when_depth_is_dimension(self):
        ds = self._depth_primary_dataset()
        result = ProcessedDataset._drop_ambiguous_axis_coords(ds)
        assert "cell" not in result.coords
        assert result.coords["depth"].attrs["axis"] == "Z"

    def test_keeps_ensemble_when_it_is_the_dimension(self):
        """No 'time' dimension present -> 'ensemble' is the real axis, not a dupe."""
        ds = xr.Dataset(
            {"var": (["ensemble", "cell"], np.zeros((3, 2)))},
            coords={
                "ensemble": ("ensemble", np.arange(3), {}),
                "time": ("ensemble", np.arange(3), {"axis": "T"}),
                "cell": ("cell", np.arange(2), {}),
            },
        )
        result = ProcessedDataset._drop_ambiguous_axis_coords(ds)
        assert "ensemble" in result.coords

    def test_keeps_cell_when_it_is_the_dimension(self):
        """No 'depth' dimension present -> 'cell' is the real axis, not a dupe."""
        ds = xr.Dataset(
            {"var": (["cell", "time"], np.zeros((2, 3)))},
            coords={
                "cell": ("cell", np.arange(2), {}),
                "depth": ("cell", np.arange(2) * 1.0, {"axis": "Z"}),
                "time": ("time", np.arange(3), {"axis": "T"}),
            },
        )
        result = ProcessedDataset._drop_ambiguous_axis_coords(ds)
        assert "cell" in result.coords

    def test_noop_when_no_ambiguous_coords_present(self):
        ds = xr.Dataset(
            {"var": (["time", "cell"], np.zeros((3, 2)))},
            coords={
                "time": ("time", np.arange(3), {"axis": "T"}),
                "cell": ("cell", np.arange(2), {}),
            },
        )
        result = ProcessedDataset._drop_ambiguous_axis_coords(ds)
        assert set(result.coords) == {"time", "cell"}


# ============================================================================
# UTILITY METHOD TESTS
# ============================================================================


class TestReset:
    """Tests for reset method."""

    def test_reset_returns_self(self, sample_dataset):
        """Test that reset returns self for method chaining."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.reset()
        assert result is proc

    def test_reset_clears_reports(self, sample_dataset):
        """Test that reset clears reports."""
        proc = ProcessedDataset(sample_dataset)
        proc.reports.append(MagicMock())
        proc.processing_log.append("test")
        proc.reset()
        assert proc.reports == []
        assert proc.processing_log == []

    def test_reset_restores_dataset(self, sample_dataset):
        """Test that reset restores dataset from original."""
        proc = ProcessedDataset(sample_dataset)
        # Modify working dataset
        proc.dataset["velocity"].values[0, 0, 0] = 99999
        # Reset
        proc.reset()
        # Should be restored
        assert proc.dataset["velocity"].values[0, 0, 0] != 99999

    def test_reset_clears_config(self, sample_dataset):
        """Test that reset returns self.config to a blank ProcessingConfig."""
        try:
            from pyadps.processing.config import ProcessingConfig
        except ImportError:
            from config import ProcessingConfig

        with patch(SIGNAL_QUALITY_RUNNER) as mock_class:
            mock_runner = MagicMock()
            mock_runner.finalize.return_value = sample_dataset
            mock_runner.get_pipeline_report.return_value = MagicMock()
            mock_runner.statistics = []
            mock_class.return_value = mock_runner

            proc = ProcessedDataset(sample_dataset)
            proc.apply_signal_quality(correlation=70.0)
            assert proc.config.isQCTest is True

            proc.reset()
            assert isinstance(proc.config, ProcessingConfig)
            assert proc.config.isQCTest is False
            assert proc.config.isSensorTest is False
            assert proc.config.isProfileTest is False
            assert proc.config.isVelocityTest is False


class TestGetCurrentStats:
    """Tests for get_current_stats method."""

    def test_get_current_stats_returns_dict(self, sample_dataset):
        """Test that get_current_stats returns dictionary."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_current_stats()
        assert isinstance(result, dict)

    def test_get_current_stats_contains_keys(self, sample_dataset):
        """Test that result contains expected keys."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_current_stats()
        assert "total_cells" in result
        assert "masked" in result
        assert "masked_pct" in result
        assert "valid" in result
        assert "valid_pct" in result

    def test_get_current_stats_values(self, sample_dataset):
        """Test that statistics are calculated correctly."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_current_stats()
        assert result["total_cells"] == result["masked"] + result["valid"]
        assert abs(result["masked_pct"] + result["valid_pct"] - 100) < 0.01


# ============================================================================
# RUNNER ACCESS TESTS
# ============================================================================


class TestRunnerAccess:
    """Tests for direct runner access methods."""

    @patch(SENSOR_HEALTH_RUNNER)
    def test_get_sensor_health_runner(self, mock_runner_class, sample_dataset):
        """Test get_sensor_health_runner returns runner."""
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        result = proc.get_sensor_health_runner()
        assert result is mock_runner

    @patch(SIGNAL_QUALITY_RUNNER)
    def test_get_signal_quality_runner(self, mock_runner_class, sample_dataset):
        """Test get_signal_quality_runner returns runner."""
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        result = proc.get_signal_quality_runner()
        assert result is mock_runner

    @patch(PROFILE_OPERATION_RUNNER)
    def test_get_profile_operation_runner(self, mock_runner_class, sample_dataset):
        """Test get_profile_operation_runner returns runner."""
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        result = proc.get_profile_operation_runner()
        assert result is mock_runner

    @patch(VELOCITY_CHECK_RUNNER)
    def test_get_velocity_check_runner(self, mock_runner_class, sample_dataset):
        """Test get_velocity_check_runner returns runner."""
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        result = proc.get_velocity_check_runner()
        assert result is mock_runner

    def test_commit_runner_returns_self(self, sample_dataset):
        """Test that commit_runner returns self."""
        proc = ProcessedDataset(sample_dataset)
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()

        result = proc.commit_runner(mock_runner)
        assert result is proc

    def test_commit_runner_updates_dataset(self, sample_dataset):
        """Test that commit_runner updates dataset."""
        proc = ProcessedDataset(sample_dataset)

        # Create modified dataset
        modified_ds = sample_dataset.copy(deep=True)
        modified_ds["velocity"].values[0, 0, 0] = 12345

        mock_runner = MagicMock()
        mock_runner.finalize.return_value = modified_ds
        mock_runner.get_pipeline_report.return_value = MagicMock()

        proc.commit_runner(mock_runner)
        assert proc.dataset["velocity"].values[0, 0, 0] == 12345

    def test_commit_runner_adds_report(self, sample_dataset):
        """Test that commit_runner adds report."""
        proc = ProcessedDataset(sample_dataset)
        mock_report = MagicMock()

        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = mock_report

        initial_reports = len(proc.reports)
        proc.commit_runner(mock_runner)
        assert len(proc.reports) == initial_reports + 1


# ============================================================================
# REPORTING TESTS
# ============================================================================


class TestReporting:
    """Tests for reporting methods."""

    def test_get_pipeline_report_returns_dict(self, sample_dataset):
        """Test that get_pipeline_report returns dictionary."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_pipeline_report()
        assert isinstance(result, dict)

    def test_get_pipeline_report_contains_summary(self, sample_dataset):
        """Test that report contains summary section."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.get_pipeline_report()
        assert "summary" in result
        assert "total_cells" in result["summary"]

    def test_export_statistics_returns_dict(self, sample_dataset):
        """Test that export_statistics returns dictionary."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.export_statistics()
        assert isinstance(result, dict)

    def test_export_report_json(self, sample_dataset, temp_dir):
        """Test JSON report export."""
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "report.json"
        proc.export_report_json(filepath)
        assert filepath.exists()

        with open(filepath) as f:
            data = json.load(f)
        assert "summary" in data

    def test_export_report_markdown(self, sample_dataset, temp_dir):
        """Test Markdown report export."""
        proc = ProcessedDataset(sample_dataset)
        filepath = temp_dir / "report.md"
        proc.export_report_markdown(filepath)
        assert filepath.exists()

        content = filepath.read_text()
        assert "# ADCP Processing Report" in content

    def test_summary_returns_string(self, sample_dataset):
        """Test that summary returns string."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.summary()
        assert isinstance(result, str)
        assert "PROCESSING SUMMARY" in result

    def test_print_summary_no_error(self, sample_dataset, capsys):
        """Test that print_summary runs without error."""
        proc = ProcessedDataset(sample_dataset)
        proc.print_summary()  # Should not raise
        captured = capsys.readouterr()
        assert "PROCESSING SUMMARY" in captured.out


# ============================================================================
# PRINT SUMMARY TESTS
# ============================================================================


class TestPrintSummary:
    """Comprehensive tests for ProcessedDataset.print_summary().

    print_summary() calls get_pipeline_report() then prints a fixed-width
    80-character console report covering:
      - Banner lines ("=" * 80)
      - Title "ADCP PROCESSING SUMMARY"
      - Total cells and baseline masked with comma-formatting
      - "PROCESSING STEPS:" section with numbered log entries
      - Per-module detail tables with threshold / impact / cumulative columns
      - Final summary line with processing impact using +/- sign prefix

    All tests capture stdout via pytest's ``capsys`` fixture rather than
    patching sys.stdout, which is cleaner and avoids interfering with the
    ``summary()`` method that does its own stdout redirect.

    Shared helpers mirror those in TestExportReportMarkdown so the two
    test classes are self-contained and independently runnable.
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_check_dict(
        check_name="roll_check",
        threshold=15.0,
        newly_masked_pct=5.0,
        total_masked_pct=None,
    ):
        if total_masked_pct is None:
            total_masked_pct = newly_masked_pct
        return {
            "check_name": check_name,
            "threshold": threshold,
            "cells_pre_masked": 0,
            "cells_newly_masked": 100,
            "cells_total_masked": 100,
            "total_cells": 2000,
            "pre_masked_pct": 0.0,
            "newly_masked_pct": round(newly_masked_pct, 4),
            "total_masked_pct": round(total_masked_pct, 4),
            "valid_cells": 1900,
            "valid_pct": round(100.0 - total_masked_pct, 4),
            "check_time": "2025-01-01T00:00:00+00:00",
            "metadata": {},
        }

    @staticmethod
    def _make_step_report(module_name="SensorHealth", checks=()):
        return {
            "module_name": module_name,
            "baseline": {"masked_cells": 0, "masked_pct": 0.0, "total_cells": 2000},
            "modifications": [],
            "checks": list(checks),
            "summary": {
                "checks_applied": len(checks),
                "modifications_applied": 0,
                "final_valid_cells": 2000,
                "final_valid_pct": 100.0,
                "pipeline_impact_pct": 0.0,
            },
            "timestamp": "2025-01-01T00:00:00+00:00",
        }

    # ------------------------------------------------------------------
    # basic contract
    # ------------------------------------------------------------------

    def test_returns_none(self, sample_dataset, capsys):
        """print_summary() has no return value."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.print_summary()
        assert result is None

    def test_writes_to_stdout(self, sample_dataset, capsys):
        """print_summary() produces output on stdout, not stderr."""
        proc = ProcessedDataset(sample_dataset)
        proc.print_summary()
        captured = capsys.readouterr()
        assert captured.out != ""
        assert captured.err == ""

    # ------------------------------------------------------------------
    # banner and title lines
    # ------------------------------------------------------------------

    def test_top_banner_present(self, sample_dataset, capsys):
        """Output starts with a line of 80 '=' characters."""
        proc = ProcessedDataset(sample_dataset)
        proc.print_summary()
        out = capsys.readouterr().out
        assert "=" * 80 in out

    def test_title_line_present(self, sample_dataset, capsys):
        """'ADCP PROCESSING SUMMARY' title is present."""
        proc = ProcessedDataset(sample_dataset)
        proc.print_summary()
        out = capsys.readouterr().out
        assert "ADCP PROCESSING SUMMARY" in out

    def test_separator_lines_present(self, sample_dataset, capsys):
        """At least one line of 80 '-' characters appears as a section separator."""
        proc = ProcessedDataset(sample_dataset)
        proc.print_summary()
        out = capsys.readouterr().out
        assert "-" * 80 in out

    def test_processing_steps_label_present(self, sample_dataset, capsys):
        """'PROCESSING STEPS:' label is printed."""
        proc = ProcessedDataset(sample_dataset)
        proc.print_summary()
        out = capsys.readouterr().out
        assert "PROCESSING STEPS:" in out

    # ------------------------------------------------------------------
    # summary statistics
    # ------------------------------------------------------------------

    def test_total_cells_present(self, sample_dataset, capsys):
        """'Total cells:' label is printed."""
        proc = ProcessedDataset(sample_dataset)
        proc.print_summary()
        out = capsys.readouterr().out
        assert "Total cells:" in out

    def test_baseline_masked_present(self, sample_dataset, capsys):
        """'Baseline masked:' label is printed."""
        proc = ProcessedDataset(sample_dataset)
        proc.print_summary()
        out = capsys.readouterr().out
        assert "Baseline masked:" in out

    def test_summary_values_match_pipeline_report(self, sample_dataset, capsys):
        """Numeric values in the output match get_pipeline_report() figures."""
        proc = ProcessedDataset(sample_dataset)
        report = proc.get_pipeline_report()
        s = report["summary"]
        proc.print_summary()
        out = capsys.readouterr().out
        assert f"{s['total_cells']:,}" in out
        assert f"{s['baseline_masked_pct']:.2f}%" in out
        assert f"{s['final_valid_pct']:.2f}%" in out

    def test_large_cell_count_comma_formatted(self, capsys):
        """Total cells > 999 are rendered with comma thousands separator."""
        n_beams, n_cells, n_time = 4, 50, 100  # 20 000 cells
        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "time"],
                    np.zeros((n_beams, n_cells, n_time), dtype=np.float32),
                    {"units": "mm/s"},
                ),
            },
            coords={
                "beam": np.arange(n_beams),
                "cell": np.arange(n_cells),
                "time": np.arange(n_time),
            },
        )
        proc = ProcessedDataset(ds)
        proc.print_summary()
        out = capsys.readouterr().out
        total = proc._total_cells
        assert f"{total:,}" in out
        assert "," in f"{total:,}"  # sanity: number actually has commas

    # ------------------------------------------------------------------
    # processing steps (log entries)
    # ------------------------------------------------------------------

    def test_empty_log_no_numbered_entries(self, sample_dataset, capsys):
        """Empty processing log → no '  1. ...' lines appear."""
        proc = ProcessedDataset(sample_dataset)
        proc.processing_log = []
        proc.print_summary()
        out = capsys.readouterr().out
        import re

        assert not re.search(r"^\s+\d+\. ", out, re.MULTILINE)

    def test_single_log_entry_numbered(self, sample_dataset, capsys):
        """A single log entry appears as '  1. <text>'."""
        proc = ProcessedDataset(sample_dataset)
        proc.processing_log = ["Roll check applied"]
        proc.print_summary()
        out = capsys.readouterr().out
        assert "  1. Roll check applied" in out

    def test_multiple_log_entries_numbered_sequentially(self, sample_dataset, capsys):
        """Multiple log entries receive sequential numbers with two-space indent."""
        proc = ProcessedDataset(sample_dataset)
        proc.processing_log = ["Step Alpha", "Step Beta", "Step Gamma"]
        proc.print_summary()
        out = capsys.readouterr().out
        assert "  1. Step Alpha" in out
        assert "  2. Step Beta" in out
        assert "  3. Step Gamma" in out

    # ------------------------------------------------------------------
    # step detail table
    # ------------------------------------------------------------------

    def test_module_name_uppercased(self, sample_dataset, capsys):
        """Module name is printed in upper case followed by ':'."""
        proc = ProcessedDataset(sample_dataset)
        step = self._make_step_report("SensorHealth")
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step],
            },
        ):
            proc.print_summary()
        out = capsys.readouterr().out
        assert "SENSORHEALTH:" in out

    def test_column_header_printed(self, sample_dataset, capsys):
        """Column header labels (Check, Threshold, Impact, Cumulative) appear."""
        proc = ProcessedDataset(sample_dataset)
        step = self._make_step_report("SignalQuality")
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step],
            },
        ):
            proc.print_summary()
        out = capsys.readouterr().out
        assert "Check" in out
        assert "Threshold" in out
        assert "Impact" in out
        assert "Cumulative" in out

    def test_check_row_contains_check_name(self, sample_dataset, capsys):
        """Each check row contains the check_name value."""
        proc = ProcessedDataset(sample_dataset)
        check = self._make_check_dict("pitch_check", 12.0, 2.5)
        step = self._make_step_report("SensorHealth", [check])
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step],
            },
        ):
            proc.print_summary()
        out = capsys.readouterr().out
        assert "pitch_check" in out

    def test_check_row_impact_two_decimal_places(self, sample_dataset, capsys):
        """Impact column is printed with exactly two decimal places."""
        proc = ProcessedDataset(sample_dataset)
        check = self._make_check_dict("roll_check", 15.0, 3.7)
        step = self._make_step_report("SensorHealth", [check])
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step],
            },
        ):
            proc.print_summary()
        out = capsys.readouterr().out
        assert "3.70%" in out

    def test_check_row_cumulative_two_decimal_places(self, sample_dataset, capsys):
        """Cumulative column is printed with exactly two decimal places."""
        proc = ProcessedDataset(sample_dataset)
        check = self._make_check_dict(
            "corr_check", 64.0, newly_masked_pct=4.0, total_masked_pct=6.5
        )
        step = self._make_step_report("SignalQuality", [check])
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step],
            },
        ):
            proc.print_summary()
        out = capsys.readouterr().out
        assert "6.50%" in out

    # ------------------------------------------------------------------
    # threshold formatting branches
    # ------------------------------------------------------------------

    def test_float_threshold_formatted_one_decimal(self, sample_dataset, capsys):
        """Float threshold is printed as '{threshold:.1f}'."""
        proc = ProcessedDataset(sample_dataset)
        check = self._make_check_dict("roll_check", threshold=15.0)
        step = self._make_step_report("SensorHealth", [check])
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step],
            },
        ):
            proc.print_summary()
        out = capsys.readouterr().out
        assert "15.0" in out

    def test_none_threshold_rendered_as_none_string(self, sample_dataset, capsys):
        """None threshold falls through to str() and appears as 'None'."""
        proc = ProcessedDataset(sample_dataset)
        check = self._make_check_dict("sound_speed", threshold=None)
        step = self._make_step_report("SensorHealth", [check])
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step],
            },
        ):
            proc.print_summary()
        out = capsys.readouterr().out
        assert "None" in out

    def test_dict_threshold_rendered_as_str(self, sample_dataset, capsys):
        """Dict threshold is converted via str() before printing."""
        proc = ProcessedDataset(sample_dataset)
        thresh_dict = {"min": 0.0, "max": 1000.0}
        check = self._make_check_dict("pressure_check", threshold=thresh_dict)
        step = self._make_step_report("SensorHealth", [check])
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step],
            },
        ):
            proc.print_summary()
        out = capsys.readouterr().out
        # str({"min": 0.0, "max": 1000.0}) must appear somewhere in the output
        assert str(thresh_dict) in out

    def test_string_threshold_passed_through(self, sample_dataset, capsys):
        """A string threshold (e.g. from a legacy check) is printed as-is."""
        proc = ProcessedDataset(sample_dataset)
        check = self._make_check_dict("custom_check", threshold="auto")
        step = self._make_step_report("CustomModule", [check])
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step],
            },
        ):
            proc.print_summary()
        out = capsys.readouterr().out
        assert "auto" in out

    def test_integer_threshold_passed_through_as_str(self, sample_dataset, capsys):
        """An integer threshold falls through to str() (not the float branch)."""
        proc = ProcessedDataset(sample_dataset)
        check = self._make_check_dict("beam_check", threshold=3)
        step = self._make_step_report("SignalQuality", [check])
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step],
            },
        ):
            proc.print_summary()
        out = capsys.readouterr().out
        assert "3" in out

    # ------------------------------------------------------------------
    # no-checks guard
    # ------------------------------------------------------------------

    def test_step_without_checks_key_no_crash(self, sample_dataset, capsys):
        """Step report without 'checks' key does not crash; module header still shown."""
        proc = ProcessedDataset(sample_dataset)
        step = {
            "module_name": "ProfileOperation",
            "baseline": {"masked_cells": 0, "masked_pct": 0.0, "total_cells": 2000},
            "modifications": [],
            # 'checks' key intentionally absent
            "summary": {
                "checks_applied": 0,
                "modifications_applied": 0,
                "final_valid_cells": 2000,
                "final_valid_pct": 100.0,
                "pipeline_impact_pct": 0.0,
            },
            "timestamp": "2025-01-01T00:00:00+00:00",
        }
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step],
            },
        ):
            proc.print_summary()  # must not raise
        out = capsys.readouterr().out
        assert "PROFILEOPERATION:" in out

    def test_empty_step_reports_no_module_headers(self, sample_dataset, capsys):
        """No step reports → no module-name uppercase header in output."""
        proc = ProcessedDataset(sample_dataset)
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [],
            },
        ):
            proc.print_summary()
        out = capsys.readouterr().out
        # Neither "SENSORHEALTH:" nor any other dynamic module line should appear
        assert "Unknown".upper() + ":" not in out

    def test_unknown_module_name_fallback(self, sample_dataset, capsys):
        """Step report missing 'module_name' falls back to 'Unknown'."""
        proc = ProcessedDataset(sample_dataset)
        step = {
            # 'module_name' deliberately absent
            "baseline": {"masked_cells": 0, "masked_pct": 0.0, "total_cells": 2000},
            "modifications": [],
            "checks": [],
            "summary": {
                "checks_applied": 0,
                "modifications_applied": 0,
                "final_valid_cells": 2000,
                "final_valid_pct": 100.0,
                "pipeline_impact_pct": 0.0,
            },
            "timestamp": "2025-01-01T00:00:00+00:00",
        }
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step],
            },
        ):
            proc.print_summary()
        out = capsys.readouterr().out
        assert "UNKNOWN:" in out

    def test_multiple_step_reports_all_shown(self, sample_dataset, capsys):
        """Each step report produces its own module-header block."""
        proc = ProcessedDataset(sample_dataset)
        steps = [
            self._make_step_report("SensorHealth"),
            self._make_step_report("SignalQuality"),
            self._make_step_report("VelocityCheck"),
        ]
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": steps,
            },
        ):
            proc.print_summary()
        out = capsys.readouterr().out
        assert "SENSORHEALTH:" in out
        assert "SIGNALQUALITY:" in out
        assert "VELOCITYCHECK:" in out

    # ------------------------------------------------------------------
    # final summary line
    # ------------------------------------------------------------------

    def test_final_line_present(self, sample_dataset, capsys):
        """Output contains a 'FINAL:' summary line."""
        proc = ProcessedDataset(sample_dataset)
        proc.print_summary()
        out = capsys.readouterr().out
        assert "FINAL:" in out

    def test_final_line_valid_cells_comma_formatted(self, capsys):
        """final_valid cells in the FINAL line uses comma-thousands formatting."""
        n_beams, n_cells, n_time = 4, 50, 100  # 20 000 total
        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "time"],
                    np.zeros((n_beams, n_cells, n_time), dtype=np.float32),
                    {"units": "mm/s"},
                ),
            },
            coords={
                "beam": np.arange(n_beams),
                "cell": np.arange(n_cells),
                "time": np.arange(n_time),
            },
        )
        proc = ProcessedDataset(ds)
        proc.print_summary()
        out = capsys.readouterr().out
        report = proc.get_pipeline_report()
        valid = report["summary"]["final_valid"]
        assert f"{valid:,}" in out

    def test_final_line_processing_impact_sign_prefix(self, sample_dataset, capsys):
        """Processing impact in the FINAL line always has an explicit + or - sign."""
        proc = ProcessedDataset(sample_dataset)
        proc.print_summary()
        out = capsys.readouterr().out
        report = proc.get_pipeline_report()
        impact = report["summary"]["processing_impact_pct"]
        assert f"Processing impact: {impact:+.2f}%" in out

    def test_final_line_valid_pct_two_decimal_places(self, sample_dataset, capsys):
        """Valid-cell percentage in the FINAL line has two decimal places."""
        proc = ProcessedDataset(sample_dataset)
        proc.print_summary()
        out = capsys.readouterr().out
        report = proc.get_pipeline_report()
        pct = report["summary"]["final_valid_pct"]
        assert f"{pct:.2f}%" in out

    # ------------------------------------------------------------------
    # output width
    # ------------------------------------------------------------------

    def test_banner_exactly_80_chars(self, sample_dataset, capsys):
        """Each '=' banner line is exactly 80 characters wide."""
        proc = ProcessedDataset(sample_dataset)
        proc.print_summary()
        out = capsys.readouterr().out
        eq_lines = [ln for ln in out.splitlines() if set(ln) == {"="}]
        assert len(eq_lines) >= 2
        for ln in eq_lines:
            assert len(ln) == 80, f"Expected 80 '=' chars, got {len(ln)}: {ln!r}"

    def test_separator_exactly_80_chars(self, sample_dataset, capsys):
        """Each '-' separator line is exactly 80 characters wide."""
        proc = ProcessedDataset(sample_dataset)
        proc.print_summary()
        out = capsys.readouterr().out
        dash_lines = [ln for ln in out.splitlines() if set(ln) == {"-"}]
        assert len(dash_lines) >= 1
        for ln in dash_lines:
            assert len(ln) == 80, f"Expected 80 '-' chars, got {len(ln)}: {ln!r}"

    # ------------------------------------------------------------------
    # edge cases
    # ------------------------------------------------------------------

    def test_all_data_masked_no_crash(self, capsys):
        """Dataset where all cells are masked prints without error."""
        n_beams, n_cells, n_time = 4, 10, 20
        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "time"],
                    np.zeros((n_beams, n_cells, n_time), dtype=np.float32),
                    {"units": "mm/s"},
                ),
                "mask": (
                    ["beam", "cell", "time"],
                    np.ones((n_beams, n_cells, n_time), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(n_beams),
                "cell": np.arange(n_cells),
                "time": np.arange(n_time),
            },
        )
        proc = ProcessedDataset(ds)
        proc.print_summary()
        out = capsys.readouterr().out
        assert "ADCP PROCESSING SUMMARY" in out
        assert "100.00%" in out

    def test_zero_baseline_masked_renders_correctly(self, sample_dataset, capsys):
        """Dataset with 0 baseline masked cells renders the 0 in the output."""
        proc = ProcessedDataset(sample_dataset)
        report = proc.get_pipeline_report()
        assert report["summary"]["baseline_masked"] == 0
        proc.print_summary()
        out = capsys.readouterr().out
        # "Baseline masked: 0 (0.00%)" must appear
        assert "Baseline masked: 0" in out

    def test_single_cell_dataset_no_crash(self, capsys):
        """Single-cell dataset prints without error."""
        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "time"],
                    np.zeros((4, 1, 5), dtype=np.float32),
                    {"units": "mm/s"},
                ),
            },
            coords={
                "beam": np.arange(4),
                "cell": [0],
                "time": np.arange(5),
            },
        )
        proc = ProcessedDataset(ds)
        proc.print_summary()
        out = capsys.readouterr().out
        assert "ADCP PROCESSING SUMMARY" in out


# ============================================================================
# EXPORT REPORT MARKDOWN TESTS
# ============================================================================


class TestExportReportMarkdown:
    """Comprehensive tests for ProcessedDataset.export_report_markdown().

    The method:
    1. Coerces the path to pathlib.Path and auto-creates parent directories.
    2. Calls get_pipeline_report() and writes the summary, processing log,
       and per-step detailed statistics as a Markdown document.
    3. Returns None.

    Strategy: most tests call export_report_markdown() on a bare
    ProcessedDataset (no processing applied), so that get_pipeline_report()
    returns a deterministic, fully-controlled report dict.  A helper
    ``_make_report_dict`` builds synthetic report dicts for the detailed-
    statistics section.
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_check_dict(check_name="roll_check", threshold=15.0, newly_masked_pct=5.0):
        """Minimal dict matching QCCheckStats.to_dict() output."""
        return {
            "check_name": check_name,
            "threshold": threshold,
            "cells_pre_masked": 0,
            "cells_newly_masked": 100,
            "cells_total_masked": 100,
            "total_cells": 2000,
            "pre_masked_pct": 0.0,
            "newly_masked_pct": round(newly_masked_pct, 4),
            "total_masked_pct": round(newly_masked_pct, 4),
            "valid_cells": 1900,
            "valid_pct": round(100.0 - newly_masked_pct, 4),
            "check_time": "2025-01-01T00:00:00+00:00",
            "metadata": {},
        }

    @staticmethod
    def _make_step_report(module_name="SensorHealth", checks=()):
        """Minimal dict matching QCPipelineReport.to_dict() output."""
        return {
            "module_name": module_name,
            "baseline": {"masked_cells": 0, "masked_pct": 0.0, "total_cells": 2000},
            "modifications": [],
            "checks": list(checks),
            "summary": {
                "checks_applied": len(checks),
                "modifications_applied": 0,
                "final_valid_cells": 2000,
                "final_valid_pct": 100.0,
                "pipeline_impact_pct": 0.0,
            },
            "timestamp": "2025-01-01T00:00:00+00:00",
        }

    # ------------------------------------------------------------------
    # file I/O contract
    # ------------------------------------------------------------------

    def test_creates_file(self, sample_dataset, temp_dir):
        """Method creates a file at the specified path."""
        proc = ProcessedDataset(sample_dataset)
        out = temp_dir / "report.md"
        proc.export_report_markdown(out)
        assert out.exists()

    def test_returns_none(self, sample_dataset, temp_dir):
        """Method returns None."""
        proc = ProcessedDataset(sample_dataset)
        result = proc.export_report_markdown(temp_dir / "report.md")
        assert result is None

    def test_accepts_string_path(self, sample_dataset, temp_dir):
        """Method accepts a plain string path (coerces to Path internally)."""
        proc = ProcessedDataset(sample_dataset)
        proc.export_report_markdown(str(temp_dir / "report.md"))
        assert (temp_dir / "report.md").exists()

    def test_accepts_path_object(self, sample_dataset, temp_dir):
        """Method accepts a pathlib.Path object."""
        proc = ProcessedDataset(sample_dataset)
        proc.export_report_markdown(temp_dir / "report.md")
        assert (temp_dir / "report.md").exists()

    def test_creates_parent_directories(self, sample_dataset, temp_dir):
        """Method auto-creates any missing parent directories."""
        proc = ProcessedDataset(sample_dataset)
        deep = temp_dir / "a" / "b" / "c" / "report.md"
        proc.export_report_markdown(deep)
        assert deep.exists()

    def test_overwrites_existing_file(self, sample_dataset, temp_dir):
        """A second call replaces the file rather than appending."""
        proc = ProcessedDataset(sample_dataset)
        out = temp_dir / "report.md"
        out.write_text("old content")
        proc.export_report_markdown(out)
        content = out.read_text()
        assert "old content" not in content
        assert "# ADCP Processing Report" in content

    def test_file_is_utf8_text(self, sample_dataset, temp_dir):
        """Written file is decodable as UTF-8 text."""
        proc = ProcessedDataset(sample_dataset)
        out = temp_dir / "report.md"
        proc.export_report_markdown(out)
        # read_bytes then decode must not raise
        out.read_bytes().decode("utf-8")

    # ------------------------------------------------------------------
    # document structure: fixed headings
    # ------------------------------------------------------------------

    def test_h1_heading_present(self, sample_dataset, temp_dir):
        """Document starts with the '# ADCP Processing Report' H1 heading."""
        proc = ProcessedDataset(sample_dataset)
        out = temp_dir / "report.md"
        proc.export_report_markdown(out)
        content = out.read_text()
        assert "# ADCP Processing Report" in content

    def test_summary_section_heading_present(self, sample_dataset, temp_dir):
        """'## Summary' section heading is present."""
        proc = ProcessedDataset(sample_dataset)
        out = temp_dir / "report.md"
        proc.export_report_markdown(out)
        assert "## Summary" in out.read_text()

    def test_processing_steps_section_heading_present(self, sample_dataset, temp_dir):
        """'## Processing Steps' section heading is present."""
        proc = ProcessedDataset(sample_dataset)
        out = temp_dir / "report.md"
        proc.export_report_markdown(out)
        assert "## Processing Steps" in out.read_text()

    def test_detailed_statistics_section_heading_present(
        self, sample_dataset, temp_dir
    ):
        """'## Detailed Statistics' section heading is present."""
        proc = ProcessedDataset(sample_dataset)
        out = temp_dir / "report.md"
        proc.export_report_markdown(out)
        assert "## Detailed Statistics" in out.read_text()

    def test_generated_timestamp_line_present(self, sample_dataset, temp_dir):
        """Document contains a 'Generated:' timestamp line."""
        proc = ProcessedDataset(sample_dataset)
        out = temp_dir / "report.md"
        proc.export_report_markdown(out)
        assert "Generated:" in out.read_text()

    # ------------------------------------------------------------------
    # summary statistics bullets
    # ------------------------------------------------------------------

    def test_summary_total_cells_bullet(self, sample_dataset, temp_dir):
        """Summary section contains the 'Total cells:' bullet."""
        proc = ProcessedDataset(sample_dataset)
        proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        assert "**Total cells:**" in content

    def test_summary_baseline_masked_bullet(self, sample_dataset, temp_dir):
        """Summary section contains the 'Baseline masked:' bullet."""
        proc = ProcessedDataset(sample_dataset)
        proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        assert "**Baseline masked:**" in content

    def test_summary_final_masked_bullet(self, sample_dataset, temp_dir):
        """Summary section contains the 'Final masked:' bullet."""
        proc = ProcessedDataset(sample_dataset)
        proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        assert "**Final masked:**" in content

    def test_summary_final_valid_bullet(self, sample_dataset, temp_dir):
        """Summary section contains the 'Final valid:' bullet."""
        proc = ProcessedDataset(sample_dataset)
        proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        assert "**Final valid:**" in content

    def test_summary_processing_impact_bullet(self, sample_dataset, temp_dir):
        """Summary section contains the 'Processing impact:' bullet."""
        proc = ProcessedDataset(sample_dataset)
        proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        assert "**Processing impact:**" in content

    def test_summary_values_match_pipeline_report(self, sample_dataset, temp_dir):
        """Numeric values in the summary bullets match get_pipeline_report() output."""
        proc = ProcessedDataset(sample_dataset)
        report = proc.get_pipeline_report()
        s = report["summary"]
        out = temp_dir / "report.md"
        proc.export_report_markdown(out)
        content = out.read_text()
        # total_cells rendered with comma-thousands formatting
        assert f"{s['total_cells']:,}" in content
        # percentages rendered as two decimal places
        assert f"{s['baseline_masked_pct']:.2f}%" in content
        assert f"{s['final_masked_pct']:.2f}%" in content
        assert f"{s['final_valid_pct']:.2f}%" in content

    def test_processing_impact_has_sign_prefix(self, sample_dataset, temp_dir):
        """Processing impact is rendered with an explicit + or - sign prefix."""
        proc = ProcessedDataset(sample_dataset)
        report = proc.get_pipeline_report()
        impact = report["summary"]["processing_impact_pct"]
        expected = f"{impact:+.2f}%"  # e.g. "+0.00%" or "-3.14%"
        proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        assert expected in content

    def test_large_cell_count_uses_comma_separator(self, temp_dir):
        """Total cells with > 999 values are rendered with comma separators."""
        # Build a dataset large enough to produce comma-formatted numbers
        n_beams, n_cells, n_time = 4, 50, 100  # 4*50*100 = 20,000 cells
        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "time"],
                    np.zeros((n_beams, n_cells, n_time), dtype=np.float32),
                    {"units": "mm/s"},
                ),
            },
            coords={
                "beam": np.arange(n_beams),
                "cell": np.arange(n_cells),
                "time": np.arange(n_time),
            },
        )
        proc = ProcessedDataset(ds)
        out = temp_dir / "report.md"
        proc.export_report_markdown(out)
        content = out.read_text()
        total = proc._total_cells
        assert f"{total:,}" in content  # e.g. "20,000"
        assert "," in f"{total:,}"  # sanity: number is large enough to use commas

    # ------------------------------------------------------------------
    # processing steps section
    # ------------------------------------------------------------------

    def test_processing_log_entries_numbered(self, sample_dataset, temp_dir):
        """Each processing-log entry is rendered as a numbered list item."""
        proc = ProcessedDataset(sample_dataset)
        # Inject two synthetic log entries
        proc.processing_log = ["Step Alpha", "Step Beta"]
        proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        assert "1. Step Alpha" in content
        assert "2. Step Beta" in content

    def test_empty_processing_log_no_numbered_entries(self, sample_dataset, temp_dir):
        """Empty processing log → no numbered list items, but section heading present."""
        proc = ProcessedDataset(sample_dataset)
        proc.processing_log = []
        proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        assert "## Processing Steps" in content
        # No "1." prefix should appear
        import re

        assert not re.search(r"^\d+\. ", content, re.MULTILINE)

    def test_many_log_entries_all_numbered(self, sample_dataset, temp_dir):
        """All log entries receive sequential numbers."""
        proc = ProcessedDataset(sample_dataset)
        proc.processing_log = [f"Step {i}" for i in range(1, 6)]
        proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        for i, label in enumerate(proc.processing_log, 1):
            assert f"{i}. {label}" in content

    # ------------------------------------------------------------------
    # detailed statistics section: step reports
    # ------------------------------------------------------------------

    def test_step_report_module_name_as_h3(self, sample_dataset, temp_dir):
        """Each step report produces a '### <module_name>' H3 heading."""
        proc = ProcessedDataset(sample_dataset)
        step = self._make_step_report("SignalQuality")
        proc.reports = []
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step],
            },
        ):
            proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        assert "### SignalQuality" in content

    def test_step_report_check_bullet_format(self, sample_dataset, temp_dir):
        """Check entries render as '- **name**: threshold=..., impact=...%'."""
        proc = ProcessedDataset(sample_dataset)
        check = self._make_check_dict("roll_check", 15.0, 3.5)
        step = self._make_step_report("SensorHealth", [check])
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step],
            },
        ):
            proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        assert "**roll_check**" in content
        assert "threshold=15.0" in content
        assert "impact=3.50%" in content

    def test_step_report_impact_two_decimal_places(self, sample_dataset, temp_dir):
        """Check impact percentage is always rendered with exactly two decimal places."""
        proc = ProcessedDataset(sample_dataset)
        check = self._make_check_dict("corr_check", 64.0, 7.0)
        step = self._make_step_report("SignalQuality", [check])
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step],
            },
        ):
            proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        assert "impact=7.00%" in content

    def test_multiple_checks_in_one_step(self, sample_dataset, temp_dir):
        """Multiple checks in a step report all appear as bullets."""
        proc = ProcessedDataset(sample_dataset)
        checks = [
            self._make_check_dict("roll_check", 15.0, 2.0),
            self._make_check_dict("pitch_check", 15.0, 1.5),
            self._make_check_dict("sound_speed", None, 0.0),
        ]
        step = self._make_step_report("SensorHealth", checks)
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step],
            },
        ):
            proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        assert "**roll_check**" in content
        assert "**pitch_check**" in content
        assert "**sound_speed**" in content

    def test_multiple_step_reports_produce_multiple_h3(self, sample_dataset, temp_dir):
        """Each step report gets its own H3 heading."""
        proc = ProcessedDataset(sample_dataset)
        steps = [
            self._make_step_report("SensorHealth"),
            self._make_step_report("SignalQuality"),
            self._make_step_report("VelocityCheck"),
        ]
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": steps,
            },
        ):
            proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        assert "### SensorHealth" in content
        assert "### SignalQuality" in content
        assert "### VelocityCheck" in content

    def test_step_report_without_checks_key_no_crash(self, sample_dataset, temp_dir):
        """A step report dict missing the 'checks' key does not crash the method."""
        proc = ProcessedDataset(sample_dataset)
        step_no_checks = {
            "module_name": "ProfileOperation",
            "baseline": {"masked_cells": 0, "masked_pct": 0.0, "total_cells": 2000},
            "modifications": [],
            # 'checks' key intentionally absent
            "summary": {
                "checks_applied": 0,
                "modifications_applied": 0,
                "final_valid_cells": 2000,
                "final_valid_pct": 100.0,
                "pipeline_impact_pct": 0.0,
            },
            "timestamp": "2025-01-01T00:00:00+00:00",
        }
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step_no_checks],
            },
        ):
            # Must not raise
            proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        assert "### ProfileOperation" in content

    def test_empty_step_reports_list_no_h3(self, sample_dataset, temp_dir):
        """No step reports → no H3 headings in the detailed statistics section."""
        proc = ProcessedDataset(sample_dataset)
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [],
            },
        ):
            proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        assert "###" not in content

    def test_step_report_unknown_module_name(self, sample_dataset, temp_dir):
        """step_report missing 'module_name' falls back to 'Unknown' in the H3."""
        proc = ProcessedDataset(sample_dataset)
        step = {
            "baseline": {"masked_cells": 0, "masked_pct": 0.0, "total_cells": 2000},
            "modifications": [],
            "checks": [],
            "summary": {
                "checks_applied": 0,
                "modifications_applied": 0,
                "final_valid_cells": 2000,
                "final_valid_pct": 100.0,
                "pipeline_impact_pct": 0.0,
            },
            "timestamp": "2025-01-01T00:00:00+00:00",
            # 'module_name' deliberately absent
        }
        with patch.object(
            proc,
            "get_pipeline_report",
            return_value={
                "summary": proc.get_pipeline_report()["summary"],
                "processing_log": [],
                "time_axis": {},
                "step_reports": [step],
            },
        ):
            proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        assert "### Unknown" in content

    # ------------------------------------------------------------------
    # section ordering
    # ------------------------------------------------------------------

    def test_section_order_h1_summary_steps_details(self, sample_dataset, temp_dir):
        """Document sections appear in the prescribed order."""
        proc = ProcessedDataset(sample_dataset)
        proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        h1_pos = content.index("# ADCP Processing Report")
        summary_pos = content.index("## Summary")
        steps_pos = content.index("## Processing Steps")
        details_pos = content.index("## Detailed Statistics")
        assert h1_pos < summary_pos < steps_pos < details_pos

    # ------------------------------------------------------------------
    # edge cases
    # ------------------------------------------------------------------

    def test_all_data_masked_renders_correctly(self, temp_dir):
        """Dataset where all cells are masked at baseline renders without error."""
        n_beams, n_cells, n_time = 4, 10, 20
        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "time"],
                    np.zeros((n_beams, n_cells, n_time), dtype=np.float32),
                    {"units": "mm/s"},
                ),
                "mask": (
                    ["beam", "cell", "time"],
                    np.ones((n_beams, n_cells, n_time), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(n_beams),
                "cell": np.arange(n_cells),
                "time": np.arange(n_time),
            },
        )
        proc = ProcessedDataset(ds)
        out = temp_dir / "report.md"
        proc.export_report_markdown(out)
        content = out.read_text()
        assert "# ADCP Processing Report" in content
        assert "## Summary" in content
        # 100% masked should not crash the formatter
        assert "100.00%" in content

    def test_zero_baseline_masked_renders_correctly(self, sample_dataset, temp_dir):
        """Dataset with 0 baseline masked cells renders the 0 correctly."""
        proc = ProcessedDataset(sample_dataset)
        # sample_dataset has no RDI fill values, so baseline masked == 0
        report = proc.get_pipeline_report()
        assert report["summary"]["baseline_masked"] == 0
        proc.export_report_markdown(temp_dir / "report.md")
        content = (temp_dir / "report.md").read_text()
        assert "**Baseline masked:** 0" in content

    def test_single_cell_dataset_renders(self, temp_dir):
        """Single-cell dataset does not crash the formatter."""
        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "time"],
                    np.zeros((4, 1, 5), dtype=np.float32),
                    {"units": "mm/s"},
                ),
            },
            coords={
                "beam": np.arange(4),
                "cell": [0],
                "time": np.arange(5),
            },
        )
        proc = ProcessedDataset(ds)
        out = temp_dir / "report.md"
        proc.export_report_markdown(out)
        assert "# ADCP Processing Report" in out.read_text()

    def test_content_uses_newline_join(self, sample_dataset, temp_dir):
        r"""File content is lines joined by '\n' (not '\r\n')."""
        proc = ProcessedDataset(sample_dataset)
        out = temp_dir / "report.md"
        proc.export_report_markdown(out)
        raw = out.read_bytes()
        # Must not contain Windows-style CRLF
        assert b"\r\n" not in raw


# ============================================================================
# DEPTH ORDERING TESTS
# ============================================================================


class TestDepthOrdering:
    """Tests for _ensure_depth_ascending helper method."""

    def test_ascending_unchanged(self, sample_dataset):
        """Test that ascending depth is unchanged."""
        proc = ProcessedDataset(sample_dataset)
        # Default fixture has ascending cell indices
        result = proc._ensure_depth_ascending(proc.dataset)
        original_cells = proc.dataset.coords["cell"].values
        result_cells = result.coords["cell"].values
        np.testing.assert_array_equal(original_cells, result_cells)

    def test_descending_flipped(self, sample_dataset_descending_depth):
        """Test that descending depth is flipped."""
        proc = ProcessedDataset(sample_dataset_descending_depth)
        result = proc._ensure_depth_ascending(proc.dataset)

        original_depth = proc.dataset.coords["depth"].values
        result_depth = result.coords["depth"].values

        # Original is descending (50 -> 5)
        assert original_depth[0] > original_depth[-1]
        # Result should be ascending (5 -> 50)
        assert result_depth[0] < result_depth[-1]

    def test_data_flipped_with_coord(self, sample_dataset_descending_depth):
        """Test that data is flipped along with coordinates."""
        proc = ProcessedDataset(sample_dataset_descending_depth)

        # Get original first cell's velocity (which has the deepest depth value)
        original_vel_first = proc.dataset["velocity"].isel(cell=0).values.copy()

        result = proc._ensure_depth_ascending(proc.dataset)

        # After flipping, original first cell is now last
        result_vel_last = result["velocity"].isel(cell=-1).values
        np.testing.assert_array_equal(original_vel_first, result_vel_last)

    def test_no_depth_coord_unchanged(self):
        """Test dataset without depth/cell coord is unchanged."""
        ds = xr.Dataset(
            {
                "velocity": (["beam", "x", "time"], np.random.rand(4, 10, 50)),
            },
            coords={
                "time": np.arange(50),
                "x": np.arange(10),
                "beam": np.arange(4),
            },
        )
        proc = ProcessedDataset.__new__(ProcessedDataset)
        proc.dataset = ds

        result = proc._ensure_depth_ascending(ds)
        # Should be unchanged (no depth/cell coord)
        xr.testing.assert_identical(result, ds)

    def test_depth_as_dimension_descending_flipped(self):
        """When 'depth' is a proper dimension (post-regrid), descending values
        are flipped to ascending (lines 1071-1073)."""
        # Simulate a post-regrid dataset where depth IS a dimension, not a
        # coordinate on the cell dimension.
        depth_vals = np.array([50.0, 40.0, 30.0, 20.0, 10.0])  # descending
        ds = xr.Dataset(
            {
                "velocity": (["beam", "depth", "time"], np.zeros((4, 5, 10))),
                "mask": (
                    ["beam", "depth", "time"],
                    np.zeros((4, 5, 10), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(4),
                "depth": depth_vals,
                "time": np.arange(10),
            },
        )

        proc = ProcessedDataset.__new__(ProcessedDataset)
        proc.dataset = ds

        result = proc._ensure_depth_ascending(ds)

        result_depth = result.coords["depth"].values
        assert (
            result_depth[0] < result_depth[-1]
        ), "depth should be ascending after flip"
        np.testing.assert_array_equal(result_depth, depth_vals[::-1])

    def test_depth_as_dimension_already_ascending_unchanged(self):
        """When 'depth' is a dimension and already ascending, no flip occurs."""
        depth_vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # ascending
        ds = xr.Dataset(
            {
                "velocity": (["beam", "depth", "time"], np.zeros((4, 5, 10))),
            },
            coords={
                "beam": np.arange(4),
                "depth": depth_vals,
                "time": np.arange(10),
            },
        )

        proc = ProcessedDataset.__new__(ProcessedDataset)
        proc.dataset = ds

        result = proc._ensure_depth_ascending(ds)

        np.testing.assert_array_equal(result.coords["depth"].values, depth_vals)


# ============================================================================
# METHOD CHAINING TESTS
# ============================================================================


class TestMethodChaining:
    """Tests for method chaining (fluent API)."""

    def test_full_pipeline_chain(self, sample_dataset_with_mask):
        """Test full pipeline with method chaining."""
        with (
            patch(SENSOR_HEALTH_RUNNER) as mock_sh,
            patch(SIGNAL_QUALITY_RUNNER) as mock_sq,
            patch(PROFILE_OPERATION_RUNNER) as mock_po,
            patch(VELOCITY_CHECK_RUNNER) as mock_vc,
        ):
            # Setup mocks - return dataset with mask
            for mock_class in [mock_sh, mock_sq, mock_po, mock_vc]:
                mock_runner = MagicMock()
                mock_runner.finalize.return_value = sample_dataset_with_mask
                mock_runner.get_pipeline_report.return_value = MagicMock()
                mock_runner.statistics = []
                mock_runner.modifications = []
                mock_class.return_value = mock_runner

            proc = ProcessedDataset(sample_dataset_with_mask)

            # Full chain
            result = (
                proc.apply_time_axis(snap=False)
                .apply_sensor_health(roll=True, roll_threshold=15.0)
                .apply_signal_quality(correlation=64)
                .apply_profile_operation(regrid=True)
                .apply_velocity_check(cutoff_u=2500, cutoff_v=2500, cutoff_w=500)
                .finalize()
            )

            assert isinstance(result, xr.Dataset)

    def test_partial_pipeline_chain(self, sample_dataset_with_mask):
        """Test partial pipeline (skip some steps)."""
        with patch(SIGNAL_QUALITY_RUNNER) as mock_sq:
            mock_runner = MagicMock()
            mock_runner.finalize.return_value = sample_dataset_with_mask
            mock_runner.get_pipeline_report.return_value = MagicMock()
            mock_runner.statistics = []
            mock_sq.return_value = mock_runner

            proc = ProcessedDataset(sample_dataset_with_mask)

            # Only signal quality
            result = proc.apply_signal_quality(correlation=64).finalize()

            assert isinstance(result, xr.Dataset)


# ============================================================================
# CONFIGURATION TESTS  (apply_config gate, forwarding, side-effect tests)
# ============================================================================


class TestApplyConfigGates:
    """Each stage gate (isX=True/False) must call / not call apply_* exactly."""

    def _proc(self, ds):
        return ProcessedDataset(ds)

    # ---- time axis --------------------------------------------------------

    def test_time_axis_gate_off(self, sample_dataset):
        """isTimeAxisModified=False → apply_time_axis never invoked."""
        proc = self._proc(sample_dataset)
        with patch.object(proc, "apply_time_axis") as m:
            proc.apply_config(_all_disabled_config())
        m.assert_not_called()

    def test_time_axis_gate_on(self, sample_dataset):
        """isTimeAxisModified=True → apply_time_axis called once."""
        proc = self._proc(sample_dataset)
        cfg = _all_disabled_config(isTimeAxisModified=True)
        with patch.object(proc, "apply_time_axis", return_value=proc) as m:
            proc.apply_config(cfg)
        m.assert_called_once()

    # ---- sensor health ----------------------------------------------------

    def test_sensor_health_gate_off(self, sample_dataset):
        """isSensorTest=False → apply_sensor_health never invoked."""
        proc = self._proc(sample_dataset)
        with patch.object(proc, "apply_sensor_health") as m:
            proc.apply_config(_all_disabled_config())
        m.assert_not_called()

    def test_sensor_health_gate_on(self, sample_dataset):
        """isSensorTest=True → apply_sensor_health called once."""
        proc = self._proc(sample_dataset)
        cfg = _all_disabled_config(isSensorTest=True)
        with patch.object(proc, "apply_sensor_health", return_value=proc) as m:
            proc.apply_config(cfg)
        m.assert_called_once()

    # ---- signal quality ---------------------------------------------------

    def test_signal_quality_gate_off(self, sample_dataset):
        """isQCTest=False → apply_signal_quality never invoked."""
        proc = self._proc(sample_dataset)
        with patch.object(proc, "apply_signal_quality") as m:
            proc.apply_config(_all_disabled_config())
        m.assert_not_called()

    def test_signal_quality_gate_on(self, sample_dataset):
        """isQCTest=True → apply_signal_quality called once."""
        proc = self._proc(sample_dataset)
        cfg = _all_disabled_config(isQCTest=True)
        with patch.object(proc, "apply_signal_quality", return_value=proc) as m:
            proc.apply_config(cfg)
        m.assert_called_once()

    # ---- profile operations -----------------------------------------------

    def test_profile_gate_off(self, sample_dataset):
        """isProfileTest=False → apply_profile_operation never invoked."""
        proc = self._proc(sample_dataset)
        with patch.object(proc, "apply_profile_operation") as m:
            proc.apply_config(_all_disabled_config())
        m.assert_not_called()

    def test_profile_gate_on_with_sub_flag(self, sample_dataset):
        """isProfileTest=True with isTrimEndsCheck_PT → apply_profile_operation called."""
        proc = self._proc(sample_dataset)
        cfg = _all_disabled_config(
            isProfileTest=True,
            isTrimEndsCheck_PT=True,
            trim_start_PT=2,
            trim_end_PT=2,
        )
        with patch.object(proc, "apply_profile_operation", return_value=proc) as m:
            proc.apply_config(cfg)
        m.assert_called_once()

    def test_profile_gate_on_no_sub_flags_skips_call(self, sample_dataset):
        """isProfileTest=True but all sub-flags False → apply_profile_operation NOT called.

        apply_config builds profile_kwargs incrementally; when no sub-flags are
        set the dict is empty and the call is deliberately skipped.
        """
        proc = self._proc(sample_dataset)
        cfg = _all_disabled_config(isProfileTest=True)
        with patch.object(proc, "apply_profile_operation", return_value=proc) as m:
            proc.apply_config(cfg)
        m.assert_not_called()

    # ---- velocity check ---------------------------------------------------

    def test_velocity_gate_off(self, sample_dataset):
        """isVelocityTest=False → apply_velocity_check never invoked."""
        proc = self._proc(sample_dataset)
        with patch.object(proc, "apply_velocity_check") as m:
            proc.apply_config(_all_disabled_config())
        m.assert_not_called()

    def test_velocity_gate_on(self, sample_dataset):
        """isVelocityTest=True → apply_velocity_check called once."""
        proc = self._proc(sample_dataset)
        cfg = _all_disabled_config(isVelocityTest=True)
        with patch.object(proc, "apply_velocity_check", return_value=proc) as m:
            proc.apply_config(cfg)
        m.assert_called_once()

    # ---- ordering ---------------------------------------------------------

    def test_steps_called_in_pipeline_order(self, sample_dataset):
        """The five apply_* calls must fire in pipeline order (steps 1–5)."""
        proc = self._proc(sample_dataset)
        cfg = _all_disabled_config(
            isTimeAxisModified=True,
            isSensorTest=True,
            isQCTest=True,
            isProfileTest=True,
            isTrimEndsCheck_PT=True,
            trim_start_PT=1,
            trim_end_PT=1,
            isVelocityTest=True,
        )
        order = []
        with (
            patch.object(
                proc,
                "apply_time_axis",
                side_effect=lambda **kw: order.append("time") or proc,
            ),
            patch.object(
                proc,
                "apply_sensor_health",
                side_effect=lambda **kw: order.append("sensor") or proc,
            ),
            patch.object(
                proc,
                "apply_signal_quality",
                side_effect=lambda **kw: order.append("qc") or proc,
            ),
            patch.object(
                proc,
                "apply_profile_operation",
                side_effect=lambda **kw: order.append("profile") or proc,
            ),
            patch.object(
                proc,
                "apply_velocity_check",
                side_effect=lambda **kw: order.append("velocity") or proc,
            ),
        ):
            proc.apply_config(cfg)
        assert order == ["time", "sensor", "qc", "profile", "velocity"]


# ============================================================================
# CONFIGURATION TESTS  (time-axis kwarg forwarding)
# ============================================================================


class TestApplyConfigTimeAxisForwarding:
    """Every config field forwarded to apply_time_axis() is verified."""

    def _call_kwargs(self, sample_dataset, cfg):
        proc = ProcessedDataset(sample_dataset)
        with patch.object(proc, "apply_time_axis", return_value=proc) as m:
            proc.apply_config(cfg)
        assert m.called, "apply_time_axis was not called"
        _, kw = m.call_args
        return kw

    def test_snap_flag_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(isTimeAxisModified=True, isSnapTimeAxis=True),
        )
        assert kw["snap"] is True

    def test_snap_false_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(isTimeAxisModified=True, isSnapTimeAxis=False),
        )
        assert kw["snap"] is False

    def test_snap_freq_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isTimeAxisModified=True,
                isSnapTimeAxis=True,
                time_snap_frequency="30min",
            ),
        )
        assert kw["snap_freq"] == "30min"

    def test_snap_tolerance_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isTimeAxisModified=True,
                isSnapTimeAxis=True,
                time_snap_tolerance="10min",
            ),
        )
        assert kw["snap_tolerance"] == "10min"

    def test_snap_target_minute_nonzero_forwarded(self, sample_dataset):
        """time_target_minute > 0 is passed through as snap_target_minute."""
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isTimeAxisModified=True, isSnapTimeAxis=True, time_target_minute=30
            ),
        )
        assert kw["snap_target_minute"] == 30

    def test_snap_target_minute_zero_becomes_none(self, sample_dataset):
        """time_target_minute == 0 must become snap_target_minute=None, not 0."""
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isTimeAxisModified=True, isSnapTimeAxis=True, time_target_minute=0
            ),
        )
        assert kw["snap_target_minute"] is None

    def test_fill_gaps_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(isTimeAxisModified=True, isTimeGapFilled=True),
        )
        assert kw["fill_gaps"] is True

    def test_fill_method_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isTimeAxisModified=True, isTimeGapFilled=True, time_fill_method="D"
            ),
        )
        assert kw["fill_method"] == "D"


# ============================================================================
# CONFIGURATION TESTS  (sensor-health kwarg forwarding)
# ============================================================================


class TestApplyConfigSensorHealthForwarding:
    """Every config field forwarded to apply_sensor_health() is verified."""

    def _call_kwargs(self, sample_dataset, cfg):
        proc = ProcessedDataset(sample_dataset)
        with patch.object(proc, "apply_sensor_health", return_value=proc) as m:
            proc.apply_config(cfg)
        assert m.called, "apply_sensor_health was not called"
        _, kw = m.call_args
        return kw

    def test_roll_flag_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset, _all_disabled_config(isSensorTest=True, isRollCheck_ST=True)
        )
        assert kw["roll"] is True

    def test_roll_false_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(isSensorTest=True, isRollCheck_ST=False),
        )
        assert kw["roll"] is False

    def test_roll_threshold_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isSensorTest=True, isRollCheck_ST=True, roll_cutoff_ST=12.0
            ),
        )
        assert kw["roll_threshold"] == pytest.approx(12.0)

    def test_pitch_flag_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(isSensorTest=True, isPitchCheck_ST=True),
        )
        assert kw["pitch"] is True

    def test_pitch_threshold_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isSensorTest=True, isPitchCheck_ST=True, pitch_cutoff_ST=10.0
            ),
        )
        assert kw["pitch_threshold"] == pytest.approx(10.0)

    def test_correct_sound_speed_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(isSensorTest=True, isSoundModified_ST=True),
        )
        assert kw["correct_sound_speed"] is True

    def test_correct_velocity_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isSensorTest=True, isSoundModified_ST=True, isVelocityModified_ST=False
            ),
        )
        assert kw["correct_velocity"] is False

    def test_horizontal_only_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isSensorTest=True,
                isSoundModified_ST=True,
                isVelocityModified_HorizontalOnly_ST=False,
            ),
        )
        assert kw["horizontal_only"] is False

    # ---- temperature replacement -----------------------------------------

    def test_temperature_none_when_not_modified(self, sample_dataset):
        """isTemperatureModified_ST=False → temperature=None regardless of option."""
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isSensorTest=True,
                isTemperatureModified_ST=False,
                temperatureoption_ST="Fixed Value",
                fixedtemperature_ST=20.0,
            ),
        )
        assert kw["temperature"] is None

    def test_temperature_fixed_value_forwarded(self, sample_dataset):
        """isTemperatureModified_ST + Fixed Value → float passed as temperature."""
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isSensorTest=True,
                isTemperatureModified_ST=True,
                temperatureoption_ST="Fixed Value",
                fixedtemperature_ST=18.5,
            ),
        )
        assert kw["temperature"] == pytest.approx(18.5)

    def test_temperature_none_option_passes_none(self, sample_dataset):
        """isTemperatureModified_ST + option 'None' → temperature=None."""
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isSensorTest=True,
                isTemperatureModified_ST=True,
                temperatureoption_ST="None",
            ),
        )
        assert kw["temperature"] is None

    # ---- salinity replacement --------------------------------------------

    def test_salinity_none_when_not_modified(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isSensorTest=True,
                isSalinityModified_ST=False,
                salinityoption_ST="Fixed Value",
                fixedsalinity_ST=30.0,
            ),
        )
        assert kw["salinity"] is None

    def test_salinity_fixed_value_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isSensorTest=True,
                isSalinityModified_ST=True,
                salinityoption_ST="Fixed Value",
                fixedsalinity_ST=32.0,
            ),
        )
        assert kw["salinity"] == pytest.approx(32.0)

    # ---- transducer_depth replacement -----------------------------------

    def test_transducer_depth_none_when_not_modified(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isSensorTest=True,
                isDepthModified_ST=False,
                depthoption_ST="Fixed Value",
                fixeddepth_ST=50.0,
            ),
        )
        assert kw["transducer_depth"] is None

    def test_transducer_depth_fixed_value_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isSensorTest=True,
                isDepthModified_ST=True,
                depthoption_ST="Fixed Value",
                fixeddepth_ST=40.0,
            ),
        )
        assert kw["transducer_depth"] == pytest.approx(40.0)


# ============================================================================
# CONFIGURATION TESTS  (signal-quality kwarg forwarding)
# ============================================================================


class TestApplyConfigSignalQualityForwarding:
    """Every config field forwarded to apply_signal_quality() is verified."""

    def _call_kwargs(self, sample_dataset, cfg):
        proc = ProcessedDataset(sample_dataset)
        with patch.object(proc, "apply_signal_quality", return_value=proc) as m:
            proc.apply_config(cfg)
        assert m.called, "apply_signal_quality was not called"
        _, kw = m.call_args
        return kw

    def test_correlation_forwarded_when_positive(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset, _all_disabled_config(isQCTest=True, ct_QCT=64.0)
        )
        assert kw["correlation"] == pytest.approx(64.0)

    def test_correlation_suppressed_when_zero(self, sample_dataset):
        """ct_QCT == 0 → correlation=None (zero-suppression rule)."""
        kw = self._call_kwargs(
            sample_dataset, _all_disabled_config(isQCTest=True, ct_QCT=0.0)
        )
        assert kw["correlation"] is None

    def test_echo_intensity_forwarded_when_positive(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset, _all_disabled_config(isQCTest=True, et_QCT=40.0)
        )
        assert kw["echo_intensity"] == pytest.approx(40.0)

    def test_echo_intensity_suppressed_when_zero(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset, _all_disabled_config(isQCTest=True, et_QCT=0.0)
        )
        assert kw["echo_intensity"] is None

    def test_error_velocity_forwarded_when_positive(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset, _all_disabled_config(isQCTest=True, evt_QCT=2000.0)
        )
        assert kw["error_velocity"] == pytest.approx(2000.0)

    def test_error_velocity_suppressed_when_zero(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset, _all_disabled_config(isQCTest=True, evt_QCT=0.0)
        )
        assert kw["error_velocity"] is None

    def test_percent_good_forwarded_when_positive(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset, _all_disabled_config(isQCTest=True, pgt_QCT=50.0)
        )
        assert kw["percent_good"] == pytest.approx(50.0)

    def test_percent_good_suppressed_when_zero(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset, _all_disabled_config(isQCTest=True, pgt_QCT=0.0)
        )
        assert kw["percent_good"] is None

    def test_false_target_forwarded_when_positive(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset, _all_disabled_config(isQCTest=True, ft_QCT=50.0)
        )
        assert kw["false_target"] == pytest.approx(50.0)

    def test_false_target_suppressed_when_zero(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset, _all_disabled_config(isQCTest=True, ft_QCT=0.0)
        )
        assert kw["false_target"] is None

    def test_threebeam_forwarded_true(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(isQCTest=True, ct_QCT=64.0, is3beam_QCT=True),
        )
        assert kw["threebeam"] is True

    def test_threebeam_forwarded_false(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(isQCTest=True, ct_QCT=64.0, is3beam_QCT=False),
        )
        assert kw["threebeam"] is False

    def test_beam_ignore_forwarded_when_set(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isQCTest=True, ct_QCT=64.0, is3beam_QCT=True, beam_ignore_QCT=2
            ),
        )
        assert kw["beam_ignore"] == 2

    def test_beam_ignore_forwarded_none(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(isQCTest=True, ct_QCT=64.0, beam_ignore_QCT=None),
        )
        assert kw["beam_ignore"] is None

    def test_all_thresholds_zero_still_calls_method(self, sample_dataset):
        """isQCTest=True gate fires before zero-suppression → method always called."""
        cfg = _all_disabled_config(
            isQCTest=True, ct_QCT=0.0, et_QCT=0.0, evt_QCT=0.0, pgt_QCT=0.0, ft_QCT=0.0
        )
        proc = ProcessedDataset(sample_dataset)
        with patch.object(proc, "apply_signal_quality", return_value=proc) as m:
            proc.apply_config(cfg)
        m.assert_called_once()


# ============================================================================
# CONFIGURATION TESTS  (profile-operation kwarg forwarding)
# ============================================================================


class TestApplyConfigProfileForwarding:
    """Every config path that maps to apply_profile_operation() is verified."""

    def _call_kwargs(self, sample_dataset, cfg):
        proc = ProcessedDataset(sample_dataset)
        with patch.object(proc, "apply_profile_operation", return_value=proc) as m:
            proc.apply_config(cfg)
        assert m.called, "apply_profile_operation was not called"
        _, kw = m.call_args
        return kw

    # ---- trim ensembles --------------------------------------------------

    def test_trim_start_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isProfileTest=True,
                isTrimEndsCheck_PT=True,
                trim_start_PT=3,
                trim_end_PT=0,
            ),
        )
        assert kw["trim_start"] == 3

    def test_trim_end_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isProfileTest=True,
                isTrimEndsCheck_PT=True,
                trim_start_PT=0,
                trim_end_PT=5,
            ),
        )
        assert kw["trim_end"] == 5

    # ---- side-lobe cut ---------------------------------------------------

    def test_side_lobe_flag_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isProfileTest=True, isCutBinSideLobeCheck_PT=True, water_depth_PT=50.0
            ),
        )
        assert kw["cut_bins_side_lobe"] is True

    def test_water_depth_forwarded_when_positive(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isProfileTest=True, isCutBinSideLobeCheck_PT=True, water_depth_PT=75.0
            ),
        )
        assert kw["water_depth"] == pytest.approx(75.0)

    def test_water_depth_becomes_none_when_zero(self, sample_dataset):
        """water_depth_PT == 0 must become water_depth=None, not 0."""
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isProfileTest=True, isCutBinSideLobeCheck_PT=True, water_depth_PT=0.0
            ),
        )
        assert kw["water_depth"] is None

    def test_extra_cells_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isProfileTest=True,
                isCutBinSideLobeCheck_PT=True,
                water_depth_PT=50.0,
                extra_cells_PT=3,
            ),
        )
        assert kw["extra_cells"] == 3

    def test_beam_direction_forwarded_for_side_lobe(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isProfileTest=True,
                isCutBinSideLobeCheck_PT=True,
                water_depth_PT=50.0,
                beam_direction_PT="down",
            ),
        )
        assert kw["beam_direction"] == "down"

    # ---- manual cut bins -------------------------------------------------

    def test_manual_cut_regions_forwarded(self, sample_dataset):
        """cut_bins_regions_PT sentinel -1 values are converted to None."""
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isProfileTest=True,
                isCutBinManualCheck_PT=True,
                cut_bins_regions_PT=[[0, 5, -1, -1], [10, 15, 100, 200]],
            ),
        )
        assert kw["cut_bins_manual"] == [[0, 5, None, None], [10, 15, 100, 200]]

    def test_sentinel_minus_one_converted_to_none(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isProfileTest=True,
                isCutBinManualCheck_PT=True,
                cut_bins_regions_PT=[[-1, -1, -1, -1]],
            ),
        )
        assert kw["cut_bins_manual"] == [[None, None, None, None]]

    def test_legacy_fallback_used_when_regions_empty(self, sample_dataset):
        """Empty cut_bins_regions_PT → legacy cut_bins_start/end_PT used."""
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isProfileTest=True,
                isCutBinManualCheck_PT=True,
                cut_bins_regions_PT=[],
                cut_bins_start_PT=2,
                cut_bins_end_PT=8,
            ),
        )
        assert kw["cut_bins_manual"] == [[2, 8, None, None]]

    # ---- regrid ----------------------------------------------------------

    def test_regrid_flag_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isProfileTest=True, isRegridCheck_PT=True, regrid_method_PT="nearest"
            ),
        )
        assert kw["regrid"] is True

    def test_regrid_method_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isProfileTest=True, isRegridCheck_PT=True, regrid_method_PT="linear"
            ),
        )
        assert kw["regrid_method"] == "linear"

    def test_beam_direction_forwarded_for_regrid(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isProfileTest=True, isRegridCheck_PT=True, beam_direction_PT="up"
            ),
        )
        assert kw["beam_direction"] == "up"

    def test_regrid_end_cell_option_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isProfileTest=True,
                isRegridCheck_PT=True,
                regrid_end_cell_option_PT="surface",
            ),
        )
        assert kw["regrid_end_cell_option"] == "surface"

    def test_regrid_boundary_limit_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isProfileTest=True,
                isRegridCheck_PT=True,
                regrid_end_cell_option_PT="manual",
                regrid_boundary_limit_PT=25.0,
            ),
        )
        assert kw["regrid_boundary_limit"] == pytest.approx(25.0)

    # ---- empty-kwargs guard ----------------------------------------------

    def test_apply_profile_operation_not_called_when_kwargs_empty(self, sample_dataset):
        """isProfileTest=True, all sub-flags False → apply_profile_operation NOT called."""
        proc = ProcessedDataset(sample_dataset)
        cfg = _all_disabled_config(isProfileTest=True)
        with patch.object(proc, "apply_profile_operation", return_value=proc) as m:
            proc.apply_config(cfg)
        m.assert_not_called()

    def test_apply_profile_operation_called_once_for_multiple_sub_flags(
        self, sample_dataset
    ):
        """Multiple sub-flags → still only ONE call to apply_profile_operation."""
        proc = ProcessedDataset(sample_dataset)
        cfg = _all_disabled_config(
            isProfileTest=True,
            isTrimEndsCheck_PT=True,
            trim_start_PT=2,
            trim_end_PT=2,
            isCutBinSideLobeCheck_PT=True,
            water_depth_PT=50.0,
            isRegridCheck_PT=True,
        )
        with patch.object(proc, "apply_profile_operation", return_value=proc) as m:
            proc.apply_config(cfg)
        m.assert_called_once()


# ============================================================================
# CONFIGURATION TESTS  (velocity-check kwarg forwarding)
# ============================================================================


class TestApplyConfigVelocityForwarding:
    """Every config field forwarded to apply_velocity_check() is verified."""

    def _call_kwargs(self, sample_dataset, cfg):
        proc = ProcessedDataset(sample_dataset)
        with patch.object(proc, "apply_velocity_check", return_value=proc) as m:
            proc.apply_config(cfg)
        assert m.called, "apply_velocity_check was not called"
        _, kw = m.call_args
        return kw

    # ---- velocity cutoffs ------------------------------------------------

    def test_cutoffs_forwarded_when_isCutoffCheck_on(self, sample_dataset):
        """isCutoffCheck_VT=True → maxuvel/maxvvel/maxwvel_VT forwarded."""
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isVelocityTest=True,
                isCutoffCheck_VT=True,
                maxuvel_VT=2000.0,
                maxvvel_VT=1800.0,
                maxwvel_VT=300.0,
            ),
        )
        assert kw["cutoff_u"] == pytest.approx(2000.0)
        assert kw["cutoff_v"] == pytest.approx(1800.0)
        assert kw["cutoff_w"] == pytest.approx(300.0)

    def test_cutoffs_suppressed_when_isCutoffCheck_off(self, sample_dataset):
        """isCutoffCheck_VT=False → cutoff_u/v/w all become None."""
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isVelocityTest=True,
                isCutoffCheck_VT=False,
                maxuvel_VT=2000.0,
                maxvvel_VT=2000.0,
                maxwvel_VT=500.0,
            ),
        )
        assert kw["cutoff_u"] is None
        assert kw["cutoff_v"] is None
        assert kw["cutoff_w"] is None

    # ---- magnetic correction: api method --------------------------------

    def test_magnetic_correction_flag_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isVelocityTest=True, isMagnetCheck_VT=True, magnet_method_VT="api"
            ),
        )
        assert kw["magnetic_correction"] is True

    def test_use_api_true_when_method_is_api(self, sample_dataset):
        """magnet_method_VT == 'api' → use_api=True."""
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isVelocityTest=True, isMagnetCheck_VT=True, magnet_method_VT="api"
            ),
        )
        assert kw["use_api"] is True

    def test_use_api_false_when_method_is_not_api(self, sample_dataset):
        """magnet_method_VT != 'api' → use_api=False."""
        for method in ("user", "manual", "none", ""):
            kw = self._call_kwargs(
                sample_dataset,
                _all_disabled_config(
                    isVelocityTest=True, isMagnetCheck_VT=True, magnet_method_VT=method
                ),
            )
            assert kw["use_api"] is False, f"Expected False for method={method!r}"

    def test_lat_lon_year_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isVelocityTest=True,
                isMagnetCheck_VT=True,
                magnet_method_VT="api",
                magnet_lat_VT=25.5,
                magnet_lon_VT=-80.0,
                magnet_year_VT=2024,
            ),
        )
        assert kw["lat"] == pytest.approx(25.5)
        assert kw["lon"] == pytest.approx(-80.0)
        assert kw["year"] == 2024

    # ---- magnetic correction: user method -------------------------------

    def test_declination_forwarded_when_method_is_user(self, sample_dataset):
        """magnet_method_VT == 'user' → declination = magnet_user_input_VT."""
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isVelocityTest=True,
                isMagnetCheck_VT=True,
                magnet_method_VT="user",
                magnet_user_input_VT=-7.5,
            ),
        )
        assert kw["declination"] == pytest.approx(-7.5)

    def test_declination_none_when_method_is_api(self, sample_dataset):
        """magnet_method_VT == 'api' → declination=None."""
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isVelocityTest=True,
                isMagnetCheck_VT=True,
                magnet_method_VT="api",
                magnet_user_input_VT=10.0,
            ),
        )
        assert kw["declination"] is None

    def test_declination_none_when_method_is_other(self, sample_dataset):
        """magnet_method_VT not in {'user','api'} → declination=None."""
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isVelocityTest=True,
                isMagnetCheck_VT=True,
                magnet_method_VT="manual",
                magnet_user_input_VT=5.0,
            ),
        )
        assert kw["declination"] is None

    # ---- despike --------------------------------------------------------

    def test_despike_flag_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(isVelocityTest=True, isDespikeCheck_VT=True),
        )
        assert kw["despike"] is True

    def test_despike_kernel_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isVelocityTest=True, isDespikeCheck_VT=True, despike_kernel_VT=11
            ),
        )
        assert kw["despike_kernel"] == 11

    def test_despike_cutoff_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isVelocityTest=True, isDespikeCheck_VT=True, despike_cutoff_VT=2.5
            ),
        )
        assert kw["despike_cutoff"] == pytest.approx(2.5)

    # ---- flatline -------------------------------------------------------

    def test_flatline_flag_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(isVelocityTest=True, isFlatlineCheck_VT=True),
        )
        assert kw["flatline"] is True

    def test_flatline_kernel_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isVelocityTest=True, isFlatlineCheck_VT=True, flatline_kernel_VT=7
            ),
        )
        assert kw["flatline_kernel"] == 7

    def test_flatline_cutoff_forwarded(self, sample_dataset):
        kw = self._call_kwargs(
            sample_dataset,
            _all_disabled_config(
                isVelocityTest=True, isFlatlineCheck_VT=True, flatline_cutoff_VT=0.8
            ),
        )
        assert kw["flatline_cutoff"] == pytest.approx(0.8)


# ============================================================================
# CONFIGURATION TESTS  (side effects and contract)
# ============================================================================


class TestApplyConfigSideEffects:
    """Config stored, processing_log, return self, dataset.attrs, integration."""

    def _proc(self, ds):
        return ProcessedDataset(ds)

    # ---- config stored on self ------------------------------------------

    def test_pyadps_version_stamped_with_current_version(self, sample_dataset):
        """
        apply_config() must stamp pyadps_version with the version doing
        *this* run, overwriting whatever stale value was in the loaded
        config - reprocessing with a different pyadps install should
        update the record, not perpetuate the original one.
        """
        import pyadps

        proc = self._proc(sample_dataset)
        cfg = _all_disabled_config()
        cfg.pyadps_version = "0.0.1-stale"
        proc.apply_config(cfg)
        assert proc.config.pyadps_version == pyadps.__version__
        assert proc.config.pyadps_version != "0.0.1-stale"

    def test_config_object_stored_verbatim(self, sample_dataset):
        """When a ProcessingConfig object is passed, self.config is that exact object."""
        proc = self._proc(sample_dataset)
        cfg = _all_disabled_config()
        proc.apply_config(cfg)
        assert proc.config is cfg

    def test_config_loaded_from_path_stored_on_self(self, sample_dataset, temp_dir):
        """When a str path is passed, the parsed config is stored on self.config."""
        try:
            from pyadps.processing.config import ProcessingConfig as _PC
        except ImportError:
            from config import ProcessingConfig as _PC
        ini_path = temp_dir / "cfg.ini"
        _PC(isQCTest=True, ct_QCT=70.0).to_ini(str(ini_path))
        proc = self._proc(sample_dataset)
        proc.apply_config(str(ini_path))
        assert type(proc.config).__name__ == "ProcessingConfig"
        assert proc.config.isQCTest is True
        assert proc.config.ct_QCT == pytest.approx(70.0)

    def test_config_loaded_from_path_object_stored(self, sample_dataset, temp_dir):
        """A pathlib.Path INI path is also parsed and stored."""
        try:
            from pyadps.processing.config import ProcessingConfig as _PC
        except ImportError:
            from config import ProcessingConfig as _PC
        ini_path = temp_dir / "cfg.ini"
        _PC(isSensorTest=True, roll_cutoff_ST=8.0).to_ini(str(ini_path))
        proc = self._proc(sample_dataset)
        proc.apply_config(ini_path)
        assert proc.config.isSensorTest is True
        assert proc.config.roll_cutoff_ST == pytest.approx(8.0)

    def test_second_call_replaces_stored_config(self, sample_dataset):
        """A second apply_config call overwrites self.config."""
        proc = self._proc(sample_dataset)
        cfg1 = _all_disabled_config()
        cfg2 = _all_disabled_config(isQCTest=True, ct_QCT=55.0)
        proc.apply_config(cfg1)
        assert proc.config is cfg1
        proc.apply_config(cfg2)
        assert proc.config is cfg2
        assert proc.config.ct_QCT == pytest.approx(55.0)

    # ---- processing_log -------------------------------------------------

    def test_processing_log_entry_added(self, sample_dataset):
        """apply_config appends exactly one 'Configuration applied' entry."""
        proc = self._proc(sample_dataset)
        initial_len = len(proc.processing_log)
        proc.apply_config(_all_disabled_config())
        new_entries = proc.processing_log[initial_len:]
        assert len(new_entries) == 1
        assert "Configuration applied" in new_entries[0]

    def test_processing_log_accumulates_on_repeat_calls(self, sample_dataset):
        """Each apply_config call adds one more entry."""
        proc = self._proc(sample_dataset)
        proc.apply_config(_all_disabled_config())
        proc.apply_config(_all_disabled_config())
        config_entries = [
            e for e in proc.processing_log if "Configuration applied" in e
        ]
        assert len(config_entries) == 2

    # ---- return value ---------------------------------------------------

    def test_returns_self_for_method_chaining(self, sample_dataset):
        """apply_config must return self."""
        proc = self._proc(sample_dataset)
        result = proc.apply_config(_all_disabled_config())
        assert result is proc

    def test_method_chaining_with_finalize(self, sample_dataset):
        """apply_config().finalize() must not raise."""
        proc = self._proc(sample_dataset)
        result = proc.apply_config(_all_disabled_config()).finalize()
        assert isinstance(result, xr.Dataset)

    # ---- custom attributes ----------------------------------------------

    def test_attributes_written_to_dataset_attrs(self, sample_dataset):
        """isAttributes=True with a dict → each key/value written to dataset.attrs."""
        cfg = _all_disabled_config(
            isAttributes=True,
            attributes={"project": "TestProject", "pi": "J. Smith"},
        )
        proc = self._proc(sample_dataset)
        proc.apply_config(cfg)
        assert proc.dataset.attrs.get("project") == "TestProject"
        assert proc.dataset.attrs.get("pi") == "J. Smith"

    def test_multiple_attributes_all_written(self, sample_dataset):
        """All key/value pairs in attributes dict are written."""
        attrs = {f"key_{i}": f"val_{i}" for i in range(5)}
        cfg = _all_disabled_config(isAttributes=True, attributes=attrs)
        proc = self._proc(sample_dataset)
        proc.apply_config(cfg)
        for k, v in attrs.items():
            assert proc.dataset.attrs.get(k) == v, f"Missing attr {k!r}"

    def test_attributes_not_written_when_flag_false(self, sample_dataset):
        """isAttributes=False → no attributes written to dataset.attrs."""
        cfg = _all_disabled_config(
            isAttributes=False, attributes={"project": "ShouldNotAppear"}
        )
        proc = self._proc(sample_dataset)
        before = dict(proc.dataset.attrs)
        proc.apply_config(cfg)
        assert proc.dataset.attrs == before

    def test_attributes_not_written_when_dict_empty(self, sample_dataset):
        """isAttributes=True but empty dict → no attributes written."""
        cfg = _all_disabled_config(isAttributes=True, attributes={})
        proc = self._proc(sample_dataset)
        before = dict(proc.dataset.attrs)
        proc.apply_config(cfg)
        assert proc.dataset.attrs == before

    def test_attributes_overwrite_existing_key(self, sample_dataset):
        """An attribute key already on the dataset is overwritten."""
        proc = self._proc(sample_dataset)
        proc.dataset.attrs["source"] = "original"
        cfg = _all_disabled_config(isAttributes=True, attributes={"source": "updated"})
        proc.apply_config(cfg)
        assert proc.dataset.attrs["source"] == "updated"

    # ---- n_ensembles resolution ----------------------------------------

    def test_n_ensembles_matches_dataset_time_size(self, sample_dataset):
        """apply_config with Fixed-Value temperature runs without error."""
        cfg = _all_disabled_config(
            isSensorTest=True,
            isTemperatureModified_ST=True,
            temperatureoption_ST="Fixed Value",
            fixedtemperature_ST=18.0,
        )
        proc = ProcessedDataset(sample_dataset)
        with patch.object(proc, "apply_sensor_health", return_value=proc) as m:
            proc.apply_config(cfg)
        _, kw = m.call_args
        assert kw["temperature"] == pytest.approx(18.0)

    def test_n_ensembles_100_matches_fixture_time_size(self, sample_dataset):
        """sample_dataset has 100 time steps; apply_config must not crash."""
        assert sample_dataset.sizes["time"] == 100
        proc = ProcessedDataset(sample_dataset)
        result = proc.apply_config(_all_disabled_config())
        assert result is proc

    # ---- integration (mocked runners) ----------------------------------

    @patch(VELOCITY_CHECK_RUNNER)
    @patch(PROFILE_OPERATION_RUNNER)
    @patch(SIGNAL_QUALITY_RUNNER)
    @patch(SENSOR_HEALTH_RUNNER)
    def test_all_five_stages_run_without_error(
        self, mock_sh, mock_sq, mock_po, mock_vc, sample_dataset
    ):
        """A config with four stages enabled (time skipped) runs to completion."""
        for mock_class in (mock_sh, mock_sq, mock_po, mock_vc):
            mock_class.return_value = _make_runner_mock(sample_dataset)
        try:
            from pyadps.processing.config import ProcessingConfig as _PC
        except ImportError:
            from config import ProcessingConfig as _PC
        cfg = _PC(
            isTimeAxisModified=False,
            isSensorTest=True,
            isRollCheck_ST=True,
            isQCTest=True,
            ct_QCT=64.0,
            isProfileTest=True,
            isTrimEndsCheck_PT=True,
            trim_start_PT=1,
            trim_end_PT=1,
            isVelocityTest=True,
            isCutoffCheck_VT=True,
            maxuvel_VT=2500.0,
            maxvvel_VT=2500.0,
            maxwvel_VT=500.0,
        )
        proc = ProcessedDataset(sample_dataset)
        result = proc.apply_config(cfg)
        assert result is proc
        assert "Configuration applied" in proc.processing_log[-1]


# ============================================================================
# CONFIGURATION TESTS  (path-resolution parsing)
# ============================================================================


class TestApplyConfigParsing:
    """Verify that config path resolution (str / Path / object) works correctly."""

    def test_string_path_triggers_from_ini(self, sample_dataset, temp_dir):
        """Passing a str path calls ProcessingConfig.from_ini() once."""
        try:
            from pyadps.processing.config import ProcessingConfig as _PC
        except ImportError:
            from config import ProcessingConfig as _PC
        ini_path = temp_dir / "test.ini"
        _PC().to_ini(str(ini_path))
        with patch(PROCESSING_CONFIG) as mock_cls:
            mock_cfg = MagicMock()
            for flag in (
                "isTimeAxisModified",
                "isSensorTest",
                "isQCTest",
                "isProfileTest",
                "isVelocityTest",
                "isAttributes",
            ):
                setattr(mock_cfg, flag, False)
            mock_cls.from_ini.return_value = mock_cfg
            proc = ProcessedDataset(sample_dataset)
            proc.apply_config(str(ini_path))
        mock_cls.from_ini.assert_called_once_with(str(ini_path))

    def test_path_object_triggers_from_ini(self, sample_dataset, temp_dir):
        """Passing a Path object also calls ProcessingConfig.from_ini()."""
        try:
            from pyadps.processing.config import ProcessingConfig as _PC
        except ImportError:
            from config import ProcessingConfig as _PC
        ini_path = temp_dir / "test.ini"
        _PC().to_ini(str(ini_path))
        with patch(PROCESSING_CONFIG) as mock_cls:
            mock_cfg = MagicMock()
            for flag in (
                "isTimeAxisModified",
                "isSensorTest",
                "isQCTest",
                "isProfileTest",
                "isVelocityTest",
                "isAttributes",
            ):
                setattr(mock_cfg, flag, False)
            mock_cls.from_ini.return_value = mock_cfg
            proc = ProcessedDataset(sample_dataset)
            proc.apply_config(ini_path)
        mock_cls.from_ini.assert_called_once_with(str(ini_path))

    def test_config_object_does_not_call_from_ini(self, sample_dataset):
        """Passing a ProcessingConfig object must NOT call ProcessingConfig.from_ini()."""
        cfg = _all_disabled_config()
        with patch(PROCESSING_CONFIG) as mock_cls:
            proc = ProcessedDataset(sample_dataset)
            proc.apply_config(cfg)
        mock_cls.from_ini.assert_not_called()

    def test_from_ini_string_preserved_in_stored_config(self, sample_dataset, temp_dir):
        """The parsed config is stored on self.config (not the path string)."""
        try:
            from pyadps.processing.config import ProcessingConfig as _PC
        except ImportError:
            from config import ProcessingConfig as _PC
        ini_path = temp_dir / "stored.ini"
        _PC(isQCTest=True, ct_QCT=80.0).to_ini(str(ini_path))
        proc = ProcessedDataset(sample_dataset)
        proc.apply_config(str(ini_path))
        assert not isinstance(proc.config, (str, Path))
        assert proc.config.isQCTest is True
        assert proc.config.ct_QCT == pytest.approx(80.0)


# ============================================================================
# CONFIG TRACKING TESTS
# ============================================================================


class TestConfigTracking:
    """Tests that each apply_* method records its parameters into self.config,
    and that the export / validation helpers work correctly.

    These tests cover the reproducibility feature: after any programmatic
    processing run, proc.export_config() must produce an INI file that fully
    captures what was applied.
    """

    # ------------------------------------------------------------------
    # apply_time_axis
    # ------------------------------------------------------------------

    def test_apply_time_axis_records_snap_params(self, sample_dataset):
        """apply_time_axis with snap=True must update all snap fields."""
        proc = ProcessedDataset(sample_dataset)
        proc.apply_time_axis(
            snap=True,
            snap_freq="30min",
            snap_tolerance="10min",
            snap_target_minute=15,
            fill_gaps=True,
            fill_method="auto",
        )
        assert proc.config.isTimeAxisModified is True
        assert proc.config.isSnapTimeAxis is True
        assert proc.config.time_snap_frequency == "30min"
        assert proc.config.time_snap_tolerance == "10min"  # stored as full string
        assert proc.config.time_target_minute == 15
        assert proc.config.isTimeGapFilled is True
        assert proc.config.time_fill_method == "auto"

    def test_apply_time_axis_no_action_records_false(self, sample_dataset):
        """apply_time_axis with both flags off must leave isTimeAxisModified False."""
        proc = ProcessedDataset(sample_dataset)
        proc.apply_time_axis(snap=False, fill_gaps=False)
        assert proc.config.isTimeAxisModified is False

    # ------------------------------------------------------------------
    # apply_sensor_health
    # ------------------------------------------------------------------

    @patch(SENSOR_HEALTH_RUNNER)
    def test_apply_sensor_health_records_tilt_params(self, mock_class, sample_dataset):
        """apply_sensor_health records roll/pitch flags and thresholds."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_sensor_health(
            roll=True,
            roll_threshold=20.0,
            pitch=True,
            pitch_threshold=18.0,
        )

        assert proc.config.isSensorTest is True
        assert proc.config.isRollCheck_ST is True
        assert proc.config.roll_cutoff_ST == 20.0
        assert proc.config.isPitchCheck_ST is True
        assert proc.config.pitch_cutoff_ST == 18.0

    @patch(SENSOR_HEALTH_RUNNER)
    def test_apply_sensor_health_fixed_temperature(self, mock_class, sample_dataset):
        """Scalar temperature is mapped to 'Fixed Value' option.

        A check flag (roll=True) must be set alongside the replacement
        value so apply_sensor_health does not short-circuit early.
        """
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_sensor_health(roll=True, roll_threshold=15.0, temperature=15.5)

        assert proc.config.isTemperatureModified_ST is True
        assert proc.config.temperatureoption_ST == "Fixed Value"
        assert proc.config.fixedtemperature_ST == 15.5

    @patch(SENSOR_HEALTH_RUNNER)
    def test_apply_sensor_health_none_temperature(self, mock_class, sample_dataset):
        """None temperature leaves isTemperatureModified_ST False.

        A check flag must be set so apply_sensor_health does not return early.
        """
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_sensor_health(roll=True, roll_threshold=15.0, temperature=None)

        assert proc.config.isTemperatureModified_ST is False
        assert proc.config.temperatureoption_ST == "None"

    @patch(SENSOR_HEALTH_RUNNER)
    def test_apply_sensor_health_fixed_salinity(self, mock_class, sample_dataset):
        """Scalar salinity is mapped to 'Fixed Value' option.

        A check flag must be set so apply_sensor_health does not return early.
        """
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_sensor_health(roll=True, roll_threshold=15.0, salinity=34.5)

        assert proc.config.isSalinityModified_ST is True
        assert proc.config.salinityoption_ST == "Fixed Value"
        assert proc.config.fixedsalinity_ST == 34.5

    @patch(SENSOR_HEALTH_RUNNER)
    def test_apply_sensor_health_array_marks_file_option(
        self, mock_class, sample_dataset
    ):
        """Array transducer_depth is mapped to 'File' option.

        A check flag must be set so apply_sensor_health does not return early.
        """
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        depth_array = np.full(sample_dataset.sizes["time"], 200.0)
        proc.apply_sensor_health(
            roll=True, roll_threshold=15.0, transducer_depth=depth_array
        )

        assert proc.config.isDepthModified_ST is True
        assert proc.config.depthoption_ST == "File"

    # ------------------------------------------------------------------
    # apply_signal_quality
    # ------------------------------------------------------------------

    @patch(SIGNAL_QUALITY_RUNNER)
    def test_apply_signal_quality_records_all_thresholds(
        self, mock_class, sample_dataset
    ):
        """apply_signal_quality records every threshold field."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_signal_quality(
            correlation=70.0,
            echo_intensity=30.0,
            error_velocity=1800.0,
            percent_good=25.0,
            false_target=45.0,
            threebeam=True,
            beam_ignore=2,
        )

        assert proc.config.isQCTest is True
        assert proc.config.ct_QCT == 70.0
        assert proc.config.et_QCT == 30.0
        assert proc.config.evt_QCT == 1800.0
        assert proc.config.pgt_QCT == 25.0
        assert proc.config.ft_QCT == 45.0
        assert proc.config.is3beam_QCT is True
        assert proc.config.beam_ignore_QCT == 2

    @patch(SIGNAL_QUALITY_RUNNER)
    def test_apply_signal_quality_none_thresholds_recorded(
        self, mock_class, sample_dataset
    ):
        """None thresholds must be stored as 0.0, isQCTest still set True."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_signal_quality(correlation=64.0)

        assert proc.config.isQCTest is True
        assert proc.config.et_QCT == 0.0  # default fill for None

    # ------------------------------------------------------------------
    # apply_profile_operation
    # ------------------------------------------------------------------

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_profile_operation_records_trim(self, mock_class, sample_dataset):
        """apply_profile_operation records trim_start and trim_end."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_profile_operation(trim_start=5, trim_end=3)

        assert proc.config.isProfileTest is True
        assert proc.config.isTrimEndsCheck_PT is True
        assert proc.config.trim_start_PT == 5
        assert proc.config.trim_end_PT == 3
        # New fields always recorded regardless of which operation is used
        assert proc.config.extra_cells_PT == 1  # default
        assert proc.config.regrid_method_PT == "nearest"  # default

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_profile_operation_records_manual_cut_list(
        self, mock_class, sample_dataset
    ):
        """Manual cut region supplied as list is stored in config."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_profile_operation(cut_bins_manual=[[2, 8, None, None]])

        assert proc.config.isCutBinManualCheck_PT is True
        # All regions stored in new list field
        assert proc.config.cut_bins_regions_PT == [[2, 8, None, None]]
        # Legacy single-region fields mirror the first region
        assert proc.config.cut_bins_start_PT == 2
        assert proc.config.cut_bins_end_PT == 8

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_profile_operation_records_cutregion(
        self, mock_class, sample_dataset
    ):
        """Manual cut region supplied as CutRegion is stored in config."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_profile_operation(
            cut_bins_manual=[CutRegion(min_cell=3, max_cell=12)]
        )

        assert proc.config.isCutBinManualCheck_PT is True
        assert proc.config.cut_bins_regions_PT == [[3, 12, None, None]]
        assert proc.config.cut_bins_start_PT == 3
        assert proc.config.cut_bins_end_PT == 12

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_profile_operation_no_manual_cut(self, mock_class, sample_dataset):
        """Empty cut_bins_manual leaves isCutBinManualCheck_PT False."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_profile_operation(trim_start=2, trim_end=2)

        assert proc.config.isCutBinManualCheck_PT is False

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_profile_operation_records_extra_cells(
        self, mock_class, sample_dataset
    ):
        """apply_profile_operation records extra_cells into extra_cells_PT."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_profile_operation(
            cut_bins_side_lobe=True, water_depth=50.0, extra_cells=3
        )

        assert proc.config.isCutBinSideLobeCheck_PT is True
        assert proc.config.extra_cells_PT == 3

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_profile_operation_records_regrid_method(
        self, mock_class, sample_dataset
    ):
        """apply_profile_operation records regrid_method into regrid_method_PT."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_profile_operation(regrid=True, regrid_method="nearest")

        assert proc.config.isRegridCheck_PT is True
        assert proc.config.regrid_method_PT == "nearest"

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_profile_operation_records_regrid_end_cell_option(
        self, mock_class, sample_dataset
    ):
        """apply_profile_operation records regrid_end_cell_option into config."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_profile_operation(regrid=True, regrid_end_cell_option="surface")

        assert proc.config.regrid_end_cell_option_PT == "surface"

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_profile_operation_records_regrid_boundary_limit(
        self, mock_class, sample_dataset
    ):
        """apply_profile_operation records regrid_boundary_limit into config."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_profile_operation(
            regrid=True, regrid_end_cell_option="manual", regrid_boundary_limit=30.0
        )

        assert proc.config.regrid_end_cell_option_PT == "manual"
        assert proc.config.regrid_boundary_limit_PT == pytest.approx(30.0)

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_profile_operation_regrid_passes_end_cell_option_to_runner(
        self, mock_class, sample_dataset
    ):
        """runner.regrid() receives end_cell_option from apply_profile_operation."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_profile_operation(regrid=True, regrid_end_cell_option="surface")

        kw = mock_runner.regrid.call_args.kwargs
        assert kw["end_cell_option"] == "surface"

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_profile_operation_regrid_passes_boundary_limit_to_runner(
        self, mock_class, sample_dataset
    ):
        """runner.regrid() receives boundary_limit from apply_profile_operation."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_profile_operation(
            regrid=True, regrid_end_cell_option="manual", regrid_boundary_limit=20.0
        )

        kw = mock_runner.regrid.call_args.kwargs
        assert kw["boundary_limit"] == pytest.approx(20.0)

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_profile_operation_regrid_passes_orientation_to_runner(
        self, mock_class, sample_dataset
    ):
        """runner.regrid() receives beam_direction as orientation."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_profile_operation(regrid=True, beam_direction="down")

        kw = mock_runner.regrid.call_args.kwargs
        assert kw["orientation"] == "down"

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_profile_operation_regrid_auto_derives_trimends(
        self, mock_class, sample_dataset
    ):
        """trimends passed to runner.regrid() is derived from trim_start/trim_end."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        # sample_dataset has 100 time steps; trim_start=10, trim_end=5
        proc.apply_profile_operation(regrid=True, trim_start=10, trim_end=5)

        kw = mock_runner.regrid.call_args.kwargs
        assert kw["trimends"] == (10, 95)  # (start_idx, n_ens - trim_end)

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_profile_operation_regrid_trimends_none_when_no_trim(
        self, mock_class, sample_dataset
    ):
        """trimends is None when neither trim_start nor trim_end is set."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_profile_operation(regrid=True)

        kw = mock_runner.regrid.call_args.kwargs
        assert kw["trimends"] is None

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_config_forwards_extra_cells_to_runner(
        self, mock_class, sample_dataset
    ):
        """apply_config must forward extra_cells_PT to the runner when replaying."""
        try:
            from pyadps.processing.config import ProcessingConfig
        except ImportError:
            from config import ProcessingConfig

        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        cfg = ProcessingConfig(
            isProfileTest=True,
            isCutBinSideLobeCheck_PT=True,
            water_depth_PT=50.0,
            extra_cells_PT=4,
        )
        proc = ProcessedDataset(sample_dataset)
        proc.apply_config(cfg)

        call_kw = mock_runner.cut_bins_side_lobe.call_args
        assert call_kw is not None, "cut_bins_side_lobe was never called"
        forwarded = call_kw.kwargs.get("extra_cells") or (
            call_kw.args[2] if len(call_kw.args) > 2 else None
        )
        assert forwarded == 4, f"extra_cells not forwarded correctly: {call_kw}"

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_config_forwards_regrid_method_to_runner(
        self, mock_class, sample_dataset
    ):
        """apply_config must forward regrid_method_PT to the runner when replaying."""
        try:
            from pyadps.processing.config import ProcessingConfig
        except ImportError:
            from config import ProcessingConfig

        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        cfg = ProcessingConfig(
            isProfileTest=True,
            isRegridCheck_PT=True,
            regrid_method_PT="nearest",
        )
        proc = ProcessedDataset(sample_dataset)
        proc.apply_config(cfg)

        call_kw = mock_runner.regrid.call_args
        assert call_kw is not None, "regrid was never called"
        forwarded = call_kw.kwargs.get("method") or (
            call_kw.args[0] if call_kw.args else None
        )
        assert (
            forwarded == "nearest"
        ), f"regrid method not forwarded correctly: {call_kw}"

    # ------------------------------------------------------------------
    # apply_velocity_check
    # ------------------------------------------------------------------

    @patch(VELOCITY_CHECK_RUNNER)
    def test_apply_velocity_check_records_cutoffs(self, mock_class, sample_dataset):
        """apply_velocity_check records all velocity cutoff fields."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_velocity_check(cutoff_u=2000.0, cutoff_v=2000.0, cutoff_w=400.0)

        assert proc.config.isVelocityTest is True
        assert proc.config.isCutoffCheck_VT is True
        assert proc.config.maxuvel_VT == 2000.0
        assert proc.config.maxvvel_VT == 2000.0
        assert proc.config.maxwvel_VT == 400.0

    @patch(VELOCITY_CHECK_RUNNER)
    def test_apply_velocity_check_records_despike(self, mock_class, sample_dataset):
        """apply_velocity_check records despike parameters."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_velocity_check(despike=True, despike_kernel=11, despike_cutoff=2.5)

        assert proc.config.isDespikeCheck_VT is True
        assert proc.config.despike_kernel_VT == 11
        assert proc.config.despike_cutoff_VT == 2.5

    @patch(VELOCITY_CHECK_RUNNER)
    def test_apply_velocity_check_records_flatline(self, mock_class, sample_dataset):
        """apply_velocity_check records flatline parameters."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_velocity_check(flatline=True, flatline_kernel=5, flatline_cutoff=0.5)

        assert proc.config.isFlatlineCheck_VT is True
        assert proc.config.flatline_kernel_VT == 5
        assert proc.config.flatline_cutoff_VT == 0.5

    @patch(VELOCITY_CHECK_RUNNER)
    def test_apply_velocity_check_no_checks_is_noop(self, mock_class, sample_dataset):
        """apply_velocity_check() with no checks enabled is a no-op: config
        unchanged, runner never instantiated."""
        proc = ProcessedDataset(sample_dataset)
        proc.apply_velocity_check()  # no checks enabled

        # No runner should have been created
        mock_class.assert_not_called()
        # Config remains blank — the method returned early
        assert proc.config.isVelocityTest is False

    # ------------------------------------------------------------------
    # Multiple steps accumulate
    # ------------------------------------------------------------------

    @patch(SIGNAL_QUALITY_RUNNER)
    @patch(VELOCITY_CHECK_RUNNER)
    def test_multiple_steps_accumulate_in_config(
        self, mock_vc_class, mock_sq_class, sample_dataset
    ):
        """Successive apply_* calls accumulate into the same self.config."""
        for mock_class in (mock_sq_class, mock_vc_class):
            mock_runner = MagicMock()
            mock_runner.finalize.return_value = sample_dataset
            mock_runner.get_pipeline_report.return_value = MagicMock()
            mock_runner.statistics = []
            mock_runner.modifications = []
            mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_signal_quality(correlation=64.0)
        proc.apply_velocity_check(cutoff_u=2500.0, cutoff_v=2500.0, cutoff_w=500.0)

        # Both stages must be recorded in the same config object
        assert proc.config.isQCTest is True
        assert proc.config.isVelocityTest is True
        assert proc.config.ct_QCT == 64.0
        assert proc.config.maxuvel_VT == 2500.0

    # ------------------------------------------------------------------
    # export_config()
    # ------------------------------------------------------------------

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_profile_operation_records_multiple_regions(
        self, mock_class, sample_dataset
    ):
        """apply_profile_operation stores ALL cut regions in cut_bins_regions_PT."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_profile_operation(
            cut_bins_manual=[
                [0, 5, None, None],
                CutRegion(min_cell=10, max_cell=15, min_ensemble=100, max_ensemble=200),
            ]
        )

        assert proc.config.cut_bins_regions_PT == [
            [0, 5, None, None],
            [10, 15, 100, 200],
        ]
        # Legacy fields mirror the first region only
        assert proc.config.cut_bins_start_PT == 0
        assert proc.config.cut_bins_end_PT == 5

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_config_replays_all_cut_regions(self, mock_class, sample_dataset):
        """apply_config must call cut_bins_manual once per stored region."""
        try:
            from pyadps.processing.config import ProcessingConfig
        except ImportError:
            from config import ProcessingConfig

        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        cfg = ProcessingConfig(
            isProfileTest=True,
            isCutBinManualCheck_PT=True,
            cut_bins_regions_PT=[[0, 5, None, None], [10, 15, 100, 200]],
        )
        ProcessedDataset(sample_dataset).apply_config(cfg)
        assert mock_runner.cut_bins_manual.call_count == 2

    @patch(PROFILE_OPERATION_RUNNER)
    def test_apply_config_legacy_fallback_single_region(
        self, mock_class, sample_dataset
    ):
        """apply_config with empty cut_bins_regions_PT uses legacy single-region fields."""
        try:
            from pyadps.processing.config import ProcessingConfig
        except ImportError:
            from config import ProcessingConfig

        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        cfg = ProcessingConfig(
            isProfileTest=True,
            isCutBinManualCheck_PT=True,
            cut_bins_regions_PT=[],  # old-format INI
            cut_bins_start_PT=3,
            cut_bins_end_PT=12,
        )
        ProcessedDataset(sample_dataset).apply_config(cfg)
        assert mock_runner.cut_bins_manual.call_count == 1

    @patch(VELOCITY_CHECK_RUNNER)
    def test_apply_velocity_check_writes_magnet_depth_zero(
        self, mock_class, sample_dataset
    ):
        """apply_velocity_check always records magnet_depth_VT as 0.0.

        The runner does not accept a depth parameter; 0.0 is written so the
        INI is complete for Streamlit UI compatibility.
        """
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_velocity_check(cutoff_u=2500.0)
        assert proc.config.magnet_depth_VT == 0.0

    @patch(SIGNAL_QUALITY_RUNNER)
    def test_export_config_creates_file(self, mock_class, sample_dataset, temp_dir):
        """export_config() must write a file to the specified path."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_signal_quality(correlation=70.0)

        out = temp_dir / "run.ini"
        proc.export_config(str(out))
        assert out.exists()

    @patch(SIGNAL_QUALITY_RUNNER)
    def test_export_config_creates_parent_dirs(
        self, mock_class, sample_dataset, temp_dir
    ):
        """export_config() must create missing parent directories."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_signal_quality(correlation=70.0)

        out = temp_dir / "nested" / "deep" / "run.ini"
        proc.export_config(str(out))
        assert out.exists()

    @patch(SIGNAL_QUALITY_RUNNER)
    def test_export_config_round_trips(self, mock_class, sample_dataset, temp_dir):
        """Values written by export_config() must survive a from_ini() reload."""
        try:
            from pyadps.processing.config import ProcessingConfig
        except ImportError:
            from config import ProcessingConfig

        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_signal_quality(correlation=68.0, error_velocity=1800.0)

        out = temp_dir / "run.ini"
        proc.export_config(str(out))

        reloaded = ProcessingConfig.from_ini(str(out))
        assert reloaded.isQCTest is True
        assert reloaded.ct_QCT == 68.0
        assert reloaded.evt_QCT == 1800.0

    @patch(SENSOR_HEALTH_RUNNER)
    @patch(SIGNAL_QUALITY_RUNNER)
    @patch(VELOCITY_CHECK_RUNNER)
    def test_export_config_full_pipeline_round_trip(
        self,
        mock_vc_class,
        mock_sq_class,
        mock_sh_class,
        sample_dataset,
        temp_dir,
    ):
        """Full programmatic pipeline: every stage survives export -> reload."""
        try:
            from pyadps.processing.config import ProcessingConfig
        except ImportError:
            from config import ProcessingConfig

        for mock_class in (mock_sh_class, mock_sq_class, mock_vc_class):
            mock_runner = MagicMock()
            mock_runner.finalize.return_value = sample_dataset
            mock_runner.get_pipeline_report.return_value = MagicMock()
            mock_runner.statistics = []
            mock_runner.modifications = []
            mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_sensor_health(roll=True, roll_threshold=18.0)
        proc.apply_signal_quality(correlation=68.0)
        proc.apply_velocity_check(
            cutoff_u=1800.0,
            cutoff_v=1800.0,
            cutoff_w=350.0,
            despike=True,
            despike_kernel=7,
            despike_cutoff=2.0,
        )

        out = temp_dir / "full_pipeline.ini"
        proc.export_config(str(out))

        reloaded = ProcessingConfig.from_ini(str(out))
        assert reloaded.isSensorTest is True
        assert reloaded.roll_cutoff_ST == 18.0
        assert reloaded.isQCTest is True
        assert reloaded.ct_QCT == 68.0
        assert reloaded.isVelocityTest is True
        assert reloaded.maxuvel_VT == 1800.0
        assert reloaded.isDespikeCheck_VT is True
        assert reloaded.despike_kernel_VT == 7

    # ------------------------------------------------------------------
    # export_config_string()
    # ------------------------------------------------------------------

    @patch(SIGNAL_QUALITY_RUNNER)
    def test_export_config_string_returns_str(self, mock_class, sample_dataset):
        """export_config_string() must return a str."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_signal_quality(correlation=70.0)
        result = proc.export_config_string()
        assert isinstance(result, str)

    @patch(SIGNAL_QUALITY_RUNNER)
    def test_export_config_string_contains_sections(self, mock_class, sample_dataset):
        """export_config_string() must include all required INI sections."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_signal_quality(correlation=70.0)
        ini = proc.export_config_string()

        for section in (
            "[QCTest]",
            "[SensorTest]",
            "[VelocityTest]",
            "[ProfileTest]",
            "[FixTime]",
        ):
            assert section in ini, f"Missing section {section}"

    @patch(SIGNAL_QUALITY_RUNNER)
    def test_export_config_string_matches_export_config_file(
        self, mock_class, sample_dataset, temp_dir
    ):
        """export_config_string() and export_config() must produce identical content."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_signal_quality(correlation=70.0)

        ini_str = proc.export_config_string()
        out = temp_dir / "run.ini"
        proc.export_config(str(out))
        assert ini_str == out.read_text()

    @patch(SIGNAL_QUALITY_RUNNER)
    def test_export_config_string_contains_applied_value(
        self, mock_class, sample_dataset
    ):
        """The INI string must contain the correlation value that was applied."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_signal_quality(correlation=73.0)
        assert "73.0" in proc.export_config_string()

    # ------------------------------------------------------------------
    # validate_config()
    # ------------------------------------------------------------------

    def test_validate_config_blank_returns_empty_list(self, sample_dataset):
        """A freshly initialised ProcessedDataset has a valid (blank) config."""
        proc = ProcessedDataset(sample_dataset)
        issues = proc.validate_config()
        assert isinstance(issues, list)
        assert issues == []

    @patch(SENSOR_HEALTH_RUNNER)
    def test_validate_config_detects_issues(self, mock_class, sample_dataset):
        """validate_config() surfaces issues from the stored config."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_runner.modifications = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        # A roll check with a zero cutoff is invalid
        proc.apply_sensor_health(roll=True, roll_threshold=0.0)

        issues = proc.validate_config()
        assert isinstance(issues, list)
        assert len(issues) > 0
        assert any("roll" in i.lower() for i in issues)

    @patch(SIGNAL_QUALITY_RUNNER)
    def test_validate_config_valid_after_apply(self, mock_class, sample_dataset):
        """A sensible config produces no validation issues."""
        mock_runner = MagicMock()
        mock_runner.finalize.return_value = sample_dataset
        mock_runner.get_pipeline_report.return_value = MagicMock()
        mock_runner.statistics = []
        mock_class.return_value = mock_runner

        proc = ProcessedDataset(sample_dataset)
        proc.apply_signal_quality(correlation=64.0)
        assert proc.validate_config() == []


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_cut_bins_manual(self, sample_dataset):
        """Test apply_profile_operation with empty cut_bins_manual list."""
        proc = ProcessedDataset(sample_dataset)
        # Empty list should not raise
        result = proc.apply_profile_operation(cut_bins_manual=[])
        assert result is proc

    def test_single_cell_dataset(self):
        """Test with single cell dataset."""
        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "time"],
                    np.random.rand(4, 1, 10).astype(np.float32),
                    {"units": "mm/s"},
                ),
            },
            coords={
                "time": np.arange(10),
                "cell": [0],
                "beam": np.arange(4),
            },
        )

        proc = ProcessedDataset(ds)
        result = proc.finalize()
        assert isinstance(result, xr.Dataset)

    def test_single_time_dataset(self):
        """Test with single time step dataset."""
        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "time"],
                    np.random.rand(4, 10, 1).astype(np.float32),
                    {"units": "mm/s"},
                ),
            },
            coords={
                "time": [0],
                "cell": np.arange(10),
                "beam": np.arange(4),
            },
        )

        proc = ProcessedDataset(ds)
        result = proc.finalize()
        assert isinstance(result, xr.Dataset)

    def test_all_masked_data(self):
        """Test with all data masked."""
        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "time"],
                    np.random.rand(4, 10, 50).astype(np.float32),
                    {"units": "mm/s"},
                ),
                "mask": (["beam", "cell", "time"], np.ones((4, 10, 50), dtype=np.int8)),
            },
            coords={
                "time": np.arange(50),
                "cell": np.arange(10),
                "beam": np.arange(4),
            },
        )

        proc = ProcessedDataset(ds)
        stats = proc.get_current_stats()
        assert stats["valid"] == 0
        assert stats["masked_pct"] == 100.0

    def test_no_masked_data(self):
        """Test with no data masked."""
        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "time"],
                    np.random.rand(4, 10, 50).astype(np.float32),
                    {"units": "mm/s"},
                ),
                "mask": (
                    ["beam", "cell", "time"],
                    np.zeros((4, 10, 50), dtype=np.int8),
                ),
            },
            coords={
                "time": np.arange(50),
                "cell": np.arange(10),
                "beam": np.arange(4),
            },
        )

        proc = ProcessedDataset(ds)
        stats = proc.get_current_stats()
        assert stats["masked"] == 0
        assert stats["valid_pct"] == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
