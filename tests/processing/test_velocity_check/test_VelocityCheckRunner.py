"""
Tests for velocity_runner.py - VelocityCheckRunner Class.

This module contains comprehensive tests for the VelocityCheckRunner orchestrator:
- Initialization and baseline statistics
- Method chaining API
- Individual check methods (threshold, despike, flatline, magnetic_correction)
- Pipeline execution
- Statistics and reporting
- Reset and finalize functionality

Test Categories:
1. Initialization tests
2. Check method tests
3. Pipeline tests
4. Statistics/reporting tests
5. State management tests
"""

import numpy as np
import pytest
import xarray as xr
from unittest.mock import patch, MagicMock

from pyadps.processing.velocity_check import (
    VelocityCheckRunner,
    DEFAULT_VELOCITY_THRESHOLD_U,
    DEFAULT_VELOCITY_THRESHOLD_V,
    DEFAULT_VELOCITY_THRESHOLD_W,
    DEFAULT_DESPIKE_KERNEL,
    DEFAULT_DESPIKE_CUTOFF,
    DEFAULT_FLATLINE_KERNEL,
    DEFAULT_FLATLINE_CUTOFF,
    correct_magnetic_declination,
    VELOCITY_MISSING_VALUE,
)
from pyadps.processing.utility import (
    QCCheckStats,
    DataModificationStats,
    QCPipelineReport,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_dataset():
    """Create a basic sample dataset for testing."""
    n_beams = 4
    n_cells = 10
    n_time = 100

    np.random.seed(42)  # Reproducible random data
    velocity_data = np.random.randn(n_beams, n_cells, n_time) * 100

    coords = {
        "beam": np.arange(n_beams),
        "cell": np.arange(n_cells),
        "time": np.arange(n_time),
    }

    velocity = xr.DataArray(
        data=velocity_data.astype(np.float32),
        dims=["beam", "cell", "time"],
        coords=coords,
        attrs={"units": "mm/s"},
    )

    mask = xr.DataArray(
        data=np.zeros((n_beams, n_cells, n_time), dtype=np.int8),
        dims=["beam", "cell", "time"],
        coords=coords,
    )

    return xr.Dataset({"velocity": velocity, "mask": mask})


@pytest.fixture
def dataset_with_issues():
    """Create dataset with known issues for testing."""
    n_beams = 4
    n_cells = 5
    n_time = 50

    np.random.seed(42)
    velocity_data = np.random.randn(n_beams, n_cells, n_time) * 100
    velocity_data = velocity_data.astype(np.float32)

    # Add extreme values for threshold test
    velocity_data[0, 0, 10] = 3000.0  # U extreme
    velocity_data[1, 1, 20] = 3000.0  # V extreme
    velocity_data[2, 2, 30] = 600.0  # W extreme

    # Add spike for despike test
    velocity_data[0, 3, 25] = 5000.0

    # Add flatline for flatline test
    velocity_data[1, 4, 35:45] = 150.0

    coords = {
        "beam": np.arange(n_beams),
        "cell": np.arange(n_cells),
        "time": np.arange(n_time),
    }

    velocity = xr.DataArray(
        data=velocity_data,
        dims=["beam", "cell", "time"],
        coords=coords,
    )

    mask = xr.DataArray(
        data=np.zeros((n_beams, n_cells, n_time), dtype=np.int8),
        dims=["beam", "cell", "time"],
        coords=coords,
    )

    return xr.Dataset({"velocity": velocity, "mask": mask})


@pytest.fixture
def dataset_with_pre_masked():
    """Create dataset with pre-existing mask."""
    n_beams = 4
    n_cells = 5
    n_time = 20

    velocity_data = np.random.randn(n_beams, n_cells, n_time) * 100

    mask_data = np.zeros((n_beams, n_cells, n_time), dtype=np.int8)
    # Pre-mask some cells
    mask_data[:, 0, :] = 1  # Mask first cell across all beams
    mask_data[:, :, 0] = 1  # Mask first time step

    coords = {
        "beam": np.arange(n_beams),
        "cell": np.arange(n_cells),
        "time": np.arange(n_time),
    }

    velocity = xr.DataArray(
        data=velocity_data.astype(np.float32),
        dims=["beam", "cell", "time"],
        coords=coords,
    )

    mask = xr.DataArray(
        data=mask_data,
        dims=["beam", "cell", "time"],
        coords=coords,
    )

    return xr.Dataset({"velocity": velocity, "mask": mask})


@pytest.fixture
def regridded_dataset():
    """Create a regridded (depth-indexed) dataset for trim_depths() tests."""
    n_beams = 4
    depths = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0])
    n_time = 20

    coords = {"beam": np.arange(n_beams), "depth": depths, "time": np.arange(n_time)}

    velocity = xr.DataArray(
        data=np.full((n_beams, len(depths), n_time), 100.0, dtype=np.float32),
        dims=["beam", "depth", "time"],
        coords=coords,
    )
    echo = xr.DataArray(
        data=np.full((n_beams, len(depths), n_time), 80.0, dtype=np.float32),
        dims=["beam", "depth", "time"],
        coords=coords,
    )
    mask = xr.DataArray(
        data=np.zeros((n_beams, len(depths), n_time), dtype=np.int8),
        dims=["beam", "depth", "time"],
        coords=coords,
    )

    return xr.Dataset({"velocity": velocity, "echo_intensity": echo, "mask": mask})


# ============================================================================
# TEST: Initialization
# ============================================================================


class TestVelocityCheckRunnerInit:
    """Tests for VelocityCheckRunner initialization."""

    def test_initialization_basic(self, sample_dataset):
        """Should initialize with dataset."""
        runner = VelocityCheckRunner(sample_dataset)

        assert runner.dataset is not None
        assert runner.original is not None
        assert runner.statistics == []
        assert runner.modifications == []
        assert runner.history == []

    def test_stores_original_copy(self, sample_dataset):
        """Should store a deep copy of original dataset."""
        runner = VelocityCheckRunner(sample_dataset)

        # Modify runner dataset
        runner.dataset["velocity"].values[0, 0, 0] = 99999

        # Original should be unchanged
        assert runner.original["velocity"].values[0, 0, 0] != 99999

    def test_calculates_baseline_statistics(self, sample_dataset):
        """Should calculate baseline mask statistics."""
        runner = VelocityCheckRunner(sample_dataset)

        assert runner.total_cells == 4 * 10 * 100  # 4000 cells
        assert runner.baseline_masked == 0  # No pre-masked cells
        assert runner.baseline_masked_pct == 0.0

    def test_baseline_with_pre_masked(self, dataset_with_pre_masked):
        """Should correctly calculate baseline with pre-masked data."""
        runner = VelocityCheckRunner(dataset_with_pre_masked)

        # First cell (all beams, all times) + first time (all beams, remaining cells)
        # = 4*20 + 4*4 = 80 + 16 = 96 cells (with overlap at [0,0])
        # Actually: first cell = 4*20=80, first time step for remaining 4 cells = 4*4=16
        # But [0,0] is counted in both, so total unique = 80 + 16 = 96
        # Wait, let's recalculate: mask[:, 0, :] = 1 -> 4 beams * 20 times = 80
        # mask[:, :, 0] = 1 -> 4 beams * 5 cells = 20, but [:, 0, 0] already counted
        # So additional = 4 * 4 = 16 (cells 1-4 at time 0)
        # Total = 80 + 16 = 96

        assert runner.baseline_masked > 0
        assert runner.baseline_masked_pct > 0

    def test_creates_mask_if_missing(self):
        """Should create mask if not present in dataset."""
        velocity = xr.DataArray(
            np.zeros((4, 5, 10)),
            dims=["beam", "cell", "time"],
            coords={
                "beam": np.arange(4),
                "cell": np.arange(5),
                "time": np.arange(10),
            },
        )
        ds = xr.Dataset({"velocity": velocity})

        runner = VelocityCheckRunner(ds)

        assert "mask" in runner.dataset.data_vars


# ============================================================================
# TEST: Check Methods
# ============================================================================


class TestThresholdMethod:
    """Tests for threshold() method."""

    def test_threshold_returns_self(self, sample_dataset):
        """Should return self for method chaining."""
        runner = VelocityCheckRunner(sample_dataset)
        result = runner.threshold()

        assert result is runner

    def test_threshold_uses_defaults(self, sample_dataset):
        """Should use default thresholds when not specified."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.threshold()

        # Check that statistics were recorded with default values
        assert len(runner.statistics) == 1
        stat = runner.statistics[0]
        assert stat.threshold == {
            "U": DEFAULT_VELOCITY_THRESHOLD_U,
            "V": DEFAULT_VELOCITY_THRESHOLD_V,
            "W": DEFAULT_VELOCITY_THRESHOLD_W,
        }

    def test_threshold_custom_values(self, sample_dataset):
        """Should use custom thresholds when specified."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.threshold(cutoff_u=1000, cutoff_v=1500, cutoff_w=200)

        stat = runner.statistics[0]
        assert stat.threshold == {"U": 1000, "V": 1500, "W": 200}

    def test_threshold_flags_extremes(self, dataset_with_issues):
        """Should flag values exceeding thresholds."""
        runner = VelocityCheckRunner(dataset_with_issues)
        runner.threshold(cutoff_u=2500, cutoff_v=2500, cutoff_w=500)

        mask = runner.dataset["mask"].values

        # Check extreme values are flagged
        assert mask[0, 0, 10] == 1  # U extreme
        assert mask[1, 1, 20] == 1  # V extreme
        assert mask[2, 2, 30] == 1  # W extreme

    def test_threshold_records_statistics(self, dataset_with_issues):
        """Should record statistics for threshold check."""
        runner = VelocityCheckRunner(dataset_with_issues)
        runner.threshold()

        assert len(runner.statistics) == 1
        stat = runner.statistics[0]

        assert stat.check_name == "Velocity Threshold"
        assert stat.cells_newly_masked >= 0
        assert stat.cells_total_masked >= stat.cells_newly_masked


class TestDespikeMethod:
    """Tests for despike() method."""

    def test_despike_returns_self(self, sample_dataset):
        """Should return self for method chaining."""
        runner = VelocityCheckRunner(sample_dataset)
        result = runner.despike()

        assert result is runner

    def test_despike_uses_defaults(self, sample_dataset):
        """Should use default parameters when not specified."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.despike()

        stat = runner.statistics[0]
        assert stat.threshold == (DEFAULT_DESPIKE_KERNEL, DEFAULT_DESPIKE_CUTOFF)

    def test_despike_custom_values(self, sample_dataset):
        """Should use custom parameters when specified."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.despike(kernel_size=7, cutoff=2.5)

        stat = runner.statistics[0]
        assert stat.threshold == (7, 2.5)

    def test_despike_detects_spike(self, dataset_with_issues):
        """Should flag obvious spikes."""
        runner = VelocityCheckRunner(dataset_with_issues)
        runner.despike(kernel_size=13, cutoff=3.0)

        mask = runner.dataset["mask"].values

        # Spike at (0, 3, 25) should be flagged
        assert mask[0, 3, 25] == 1


class TestFlatlineMethod:
    """Tests for flatline() method."""

    def test_flatline_returns_self(self, sample_dataset):
        """Should return self for method chaining."""
        runner = VelocityCheckRunner(sample_dataset)
        result = runner.flatline()

        assert result is runner

    def test_flatline_uses_defaults(self, sample_dataset):
        """Should use default parameters when not specified."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.flatline()

        stat = runner.statistics[0]
        assert stat.threshold == (DEFAULT_FLATLINE_KERNEL, DEFAULT_FLATLINE_CUTOFF)

    def test_flatline_detects_constant(self, dataset_with_issues):
        """Should flag constant value segments."""
        runner = VelocityCheckRunner(dataset_with_issues)
        runner.flatline(kernel_size=4, cutoff=1.0)

        mask = runner.dataset["mask"].values

        # Flatline at (1, 4, 35:45) should be flagged
        assert mask[1, 4, 35:45].sum() > 0


class TestTrimSurfaceMethod:
    """Tests for trim_depths() method."""

    def test_trim_depths_returns_self(self, regridded_dataset):
        runner = VelocityCheckRunner(regridded_dataset)
        result = runner.trim_depths(depths=[12.0])

        assert result is runner

    def test_trim_depths_masks_selected_depth(self, regridded_dataset):
        runner = VelocityCheckRunner(regridded_dataset)
        runner.trim_depths(depths=[12.0])

        mask = runner.dataset["mask"]
        assert bool((mask.sel(depth=12.0) == 1).all())
        assert bool((mask.sel(depth=16.0) == 0).all())

    def test_trim_depths_records_statistics(self, regridded_dataset):
        runner = VelocityCheckRunner(regridded_dataset)
        runner.trim_depths(depths=[12.0])

        stat = runner.statistics[0]
        assert stat.check_name == "Depth Trim"
        assert stat.cells_newly_masked > 0

    def test_trim_depths_default_leaves_echo_intensity(self, regridded_dataset):
        runner = VelocityCheckRunner(regridded_dataset)
        runner.trim_depths(depths=[12.0])

        echo = runner.dataset["echo_intensity"].sel(depth=12.0).values
        assert not np.any(np.isnan(echo))

    def test_trim_depths_apply_to_all_variables_masks_echo_intensity(
        self, regridded_dataset
    ):
        runner = VelocityCheckRunner(regridded_dataset)
        runner.trim_depths(depths=[12.0], apply_to_all_variables=True)

        echo = runner.dataset["echo_intensity"].sel(depth=12.0).values
        assert np.all(np.isnan(echo))

    def test_trim_depths_chains_with_other_checks(self, regridded_dataset):
        runner = VelocityCheckRunner(regridded_dataset)
        result = runner.threshold().trim_depths(depths=[12.0]).flatline()

        assert result is runner
        assert len(runner.statistics) == 3


class TestMagneticCorrectionMethod:
    """Tests for magnetic_correction() method."""

    def test_magnetic_correction_returns_self(self, sample_dataset):
        """Should return self for method chaining."""
        runner = VelocityCheckRunner(sample_dataset)

        # Mock the correction function to avoid external dependencies
        with patch(
            "pyadps.processing.velocity_check.correct_magnetic_declination"
        ) as mock_correct:
            mock_correct.return_value = runner.dataset.copy()
            mock_correct.return_value.attrs["magnetic_declination_applied"] = -5.0

            result = runner.magnetic_correction(declination=-5.0)

        assert result is runner

    def test_magnetic_correction_records_modification(self, sample_dataset):
        """Should record modification statistics."""
        runner = VelocityCheckRunner(sample_dataset)

        with patch(
            "pyadps.processing.velocity_check.correct_magnetic_declination"
        ) as mock_correct:
            mock_correct.return_value = runner.dataset.copy()
            mock_correct.return_value.attrs["magnetic_declination_applied"] = -5.0

            runner.magnetic_correction(declination=-5.0)

        assert len(runner.modifications) == 1
        mod = runner.modifications[0]
        assert mod.operation == "magnetic_correction"
        assert mod.variable_name == "velocity"


# ============================================================================
# TEST: Method Chaining
# ============================================================================


class TestMethodChaining:
    """Tests for method chaining functionality."""

    def test_chain_all_checks(self, sample_dataset):
        """Should support chaining all check methods."""
        runner = VelocityCheckRunner(sample_dataset)

        result = (
            runner.threshold(cutoff_u=2500, cutoff_v=2500, cutoff_w=500)
            .despike(kernel_size=13, cutoff=3.0)
            .flatline(kernel_size=4, cutoff=1.0)
        )

        assert result is runner
        assert len(runner.statistics) == 3

    def test_chain_with_finalize(self, sample_dataset):
        """Should support chaining with finalize."""
        runner = VelocityCheckRunner(sample_dataset)

        ds_out = runner.threshold().despike().finalize()

        assert isinstance(ds_out, xr.Dataset)

    def test_chain_preserves_state(self, dataset_with_issues):
        """Chained operations should accumulate state."""
        runner = VelocityCheckRunner(dataset_with_issues)

        runner.threshold()
        threshold_masked = (runner.dataset["mask"].values == 1).sum()

        runner.despike()
        despike_masked = (runner.dataset["mask"].values == 1).sum()

        # Despike should only add to mask, not replace
        assert despike_masked >= threshold_masked


# ============================================================================
# TEST: Pipeline
# ============================================================================


class TestApplyPipeline:
    """Tests for apply_pipeline() method."""

    def test_pipeline_default(self, sample_dataset):
        """Should apply all checks with defaults."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.apply_pipeline()

        assert len(runner.statistics) == 3  # threshold, despike, flatline

    def test_pipeline_custom_checks(self, sample_dataset):
        """Should apply custom check configuration."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.apply_pipeline(
            checks={
                "threshold": {"cutoff_u": 1000, "cutoff_v": 1000, "cutoff_w": 200},
                "despike": {"kernel_size": 7, "cutoff": 2.0},
            }
        )

        assert len(runner.statistics) == 2  # Only threshold and despike

    def test_pipeline_custom_order(self, sample_dataset):
        """Should respect custom order."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.apply_pipeline(
            order=["flatline", "threshold"]  # Reversed order, no despike
        )

        assert len(runner.statistics) == 2
        assert runner.statistics[0].check_name == "Flatline"
        assert runner.statistics[1].check_name == "Velocity Threshold"

    def test_pipeline_skip_checks(self, sample_dataset):
        """Should skip checks not in config."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.apply_pipeline(
            checks={"threshold": 2500},  # Only threshold
            order=["threshold", "despike", "flatline"],
        )

        assert len(runner.statistics) == 1
        assert runner.statistics[0].check_name == "Velocity Threshold"


# ============================================================================
# TEST: Reset
# ============================================================================


class TestReset:
    """Tests for reset() method."""

    def test_reset_restores_dataset(self, sample_dataset):
        """Should restore dataset to original state."""
        runner = VelocityCheckRunner(sample_dataset)

        # Modify dataset
        runner.threshold()
        modified_mask = runner.dataset["mask"].values.copy()

        # Reset
        runner.reset()

        # Check restored
        np.testing.assert_array_equal(
            runner.dataset["mask"].values,
            sample_dataset["mask"].values,
        )

    def test_reset_clears_statistics(self, sample_dataset):
        """Should clear statistics."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.threshold()
        runner.despike()

        assert len(runner.statistics) > 0

        runner.reset()

        assert len(runner.statistics) == 0

    def test_reset_clears_history(self, sample_dataset):
        """Should clear history."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.threshold()

        assert len(runner.history) > 0

        runner.reset()

        assert len(runner.history) == 0

    def test_reset_returns_self(self, sample_dataset):
        """Should return self for chaining."""
        runner = VelocityCheckRunner(sample_dataset)
        result = runner.reset()

        assert result is runner


# ============================================================================
# TEST: Finalize
# ============================================================================


class TestFinalize:
    """Tests for finalize() method."""

    def test_finalize_returns_dataset(self, sample_dataset):
        """Should return xr.Dataset."""
        runner = VelocityCheckRunner(sample_dataset)
        result = runner.finalize()

        assert isinstance(result, xr.Dataset)

    def test_finalize_adds_history_attr(self, sample_dataset):
        """Should add processing history attribute."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.threshold()
        result = runner.finalize()

        assert "velocity_check_processing_history" in result.attrs

    def test_finalize_adds_timestamp_attr(self, sample_dataset):
        """Should add timestamp attribute."""
        runner = VelocityCheckRunner(sample_dataset)
        result = runner.finalize()

        assert "velocity_check_processed_at" in result.attrs

    def test_finalize_returns_copy(self, sample_dataset):
        """Should return a copy, not modify runner dataset."""
        runner = VelocityCheckRunner(sample_dataset)
        result = runner.finalize()

        # Modify result
        result.attrs["test"] = "modified"

        # Runner dataset should not have this attr
        assert "test" not in runner.dataset.attrs


# ============================================================================
# TEST: Statistics and Reporting
# ============================================================================


class TestStatisticsReporting:
    """Tests for statistics and reporting methods."""

    def test_get_statistics(self, sample_dataset):
        """Should return statistics dict keyed by check name."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.threshold()
        runner.despike()

        stats = runner.get_statistics()

        assert "Velocity Threshold" in stats
        assert "Despike" in stats
        assert isinstance(stats["Velocity Threshold"], QCCheckStats)

    def test_get_modifications(self, sample_dataset):
        """Should return modifications dict keyed by variable."""
        runner = VelocityCheckRunner(sample_dataset)

        with patch(
            "pyadps.processing.velocity_check.correct_magnetic_declination"
        ) as mock_correct:
            mock_correct.return_value = runner.dataset.copy()
            mock_correct.return_value.attrs["magnetic_declination_applied"] = -5.0
            runner.magnetic_correction(declination=-5.0)

        mods = runner.get_modifications()

        assert "velocity" in mods
        assert len(mods["velocity"]) == 1

    def test_get_pipeline_report(self, sample_dataset):
        """Should return QCPipelineReport object."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.threshold()

        report = runner.get_pipeline_report()

        assert isinstance(report, QCPipelineReport)
        assert report.module_name == "velocity_check"

    def test_export_statistics_dict(self, sample_dataset):
        """Should return serializable dict."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.threshold()

        export = runner.export_statistics_dict()

        assert isinstance(export, dict)
        # Should be JSON-serializable
        import json

        json.dumps(export)  # Should not raise

    def test_get_report(self, sample_dataset):
        """Should return string report."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.threshold()

        report = runner.get_report()

        assert isinstance(report, str)

    def test_summary(self, sample_dataset):
        """Should return summary string."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.threshold()
        runner.despike()

        summary = runner.summary()

        assert isinstance(summary, str)
        assert "threshold" in summary.lower()
        assert "despike" in summary.lower()

    def test_print_statistics(self, sample_dataset, capsys):
        """Should print statistics table."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.threshold()

        runner.print_statistics()

        captured = capsys.readouterr()
        assert "VELOCITY CHECK PROCESSING STATISTICS" in captured.out
        assert "Velocity Threshold" in captured.out


# ============================================================================
# TEST: History Tracking
# ============================================================================


class TestHistoryTracking:
    """Tests for processing history tracking."""

    def test_history_records_operations(self, sample_dataset):
        """Should record each operation in history."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.threshold()
        runner.despike()

        assert len(runner.history) == 2

    def test_history_contains_parameters(self, sample_dataset):
        """Should record parameters in history."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.threshold(cutoff_u=1000, cutoff_v=1500, cutoff_w=200)

        entry = runner.history[0]

        assert "parameters" in entry
        assert entry["parameters"]["cutoff_u"] == 1000
        assert entry["parameters"]["cutoff_v"] == 1500
        assert entry["parameters"]["cutoff_w"] == 200

    def test_history_contains_timestamp(self, sample_dataset):
        """Should record timestamp in history."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.threshold()

        entry = runner.history[0]

        assert "timestamp" in entry

    def test_history_contains_stats(self, sample_dataset):
        """Should record stats in history."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.threshold()

        entry = runner.history[0]

        assert "stats" in entry
        assert "newly_masked" in entry["stats"]


# ============================================================================
# TEST: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataset(self):
        """Should handle minimal dataset."""
        ds = xr.Dataset(
            {
                "velocity": xr.DataArray(
                    np.zeros((4, 1, 1)),
                    dims=["beam", "cell", "time"],
                ),
            }
        )

        runner = VelocityCheckRunner(ds)
        runner.threshold()

        assert len(runner.statistics) == 1

    def test_no_checks_applied(self, sample_dataset):
        """Should handle case with no checks applied."""
        runner = VelocityCheckRunner(sample_dataset)

        assert len(runner.statistics) == 0
        assert runner.summary() is not None

    def test_multiple_same_check(self, sample_dataset):
        """Should handle applying same check multiple times."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.threshold(cutoff_u=3000)
        runner.threshold(cutoff_u=2000)  # Tighter threshold

        assert len(runner.statistics) == 2

    def test_get_dataset(self, sample_dataset):
        """Should return current working dataset."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.threshold()

        ds = runner.get_dataset()

        assert isinstance(ds, xr.Dataset)
        assert ds is runner.dataset


# ============================================================================
# TESTS: Coverage for specific uncovered branches
# ============================================================================


class TestVelocityCheckRunnerUncoveredBranches:
    """Tests for branches not covered by existing test classes."""

    # ---- Line 1016: scalar params for despike in apply_pipeline ----

    def test_apply_pipeline_despike_scalar_param(self, sample_dataset):
        """Line 1016: passing a scalar (not dict) for 'despike' calls
        self.despike(cutoff=params) via the legacy scalar branch."""
        runner = VelocityCheckRunner(sample_dataset)

        runner.apply_pipeline(
            checks={"despike": 2.5},  # scalar, not a dict
            order=["despike"],
        )

        ops = [h["operation"] for h in runner.history]
        assert "despike" in ops

    # ---- Line 1021: scalar params for flatline in apply_pipeline ----

    def test_apply_pipeline_flatline_scalar_param(self, sample_dataset):
        """Line 1021: passing a scalar for 'flatline' calls
        self.flatline(cutoff=params) via the legacy scalar branch."""
        runner = VelocityCheckRunner(sample_dataset)

        runner.apply_pipeline(
            checks={"flatline": 0.5},  # scalar, not a dict
            order=["flatline"],
        )

        ops = [h["operation"] for h in runner.history]
        assert "flatline" in ops

    # ---- Line 1036: reset() recreates mask when original has none ----

    def test_reset_recreates_mask_when_original_has_none(self):
        """Line 1036: reset() calls create_default_mask when original lacks mask."""
        n_beams, n_cells, n_time = 4, 5, 20
        ds_no_mask = xr.Dataset(
            {
                "velocity": (
                    ("beam", "cell", "time"),
                    np.random.randn(n_beams, n_cells, n_time).astype(np.float32),
                )
            },
            coords={
                "beam": np.arange(n_beams),
                "cell": np.arange(n_cells),
                "time": np.arange(n_time),
            },
        )

        runner = VelocityCheckRunner(ds_no_mask)
        assert "mask" not in runner.original.data_vars  # original stays mask-free

        runner.threshold(cutoff_u=2500)
        runner.reset()  # triggers line 1036

        assert "mask" in runner.dataset.data_vars

    # ---- Lines 1157-1174: DATA MODIFICATIONS block in print_statistics ----

    def test_print_statistics_data_modifications_block(self, sample_dataset, capsys):
        """Lines 1157-1174: DATA MODIFICATIONS block is printed when
        self.modifications is non-empty (e.g. after magnetic correction)."""
        runner = VelocityCheckRunner(sample_dataset)

        # Inject a DataModificationStats directly to trigger the block without
        # needing a real magnetic correction call.
        runner.modifications.append(
            DataModificationStats(
                operation="magnetic_correction",
                variable_name="velocity",
                original_stats={"mean": 10.0},
                modified_stats={"mean": 12.5},
            )
        )
        runner.print_statistics()

        out = capsys.readouterr().out
        assert "DATA MODIFICATIONS" in out
        assert "magnetic_correction" in out

    # ---- Line 1178: "No QC checks applied yet." ----

    def test_print_statistics_no_qc_checks_message(self, sample_dataset, capsys):
        """Line 1178: 'No QC checks applied yet.' printed when statistics is empty."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.print_statistics()

        out = capsys.readouterr().out
        assert "No QC checks applied yet." in out

    # ---- Lines 1192-1197: threshold formatting branches in print_statistics ----

    def _make_stat(self, threshold):
        """Helper: build a QCCheckStats with the given threshold."""
        return QCCheckStats(
            check_name="test_check",
            threshold=threshold,
            cells_pre_masked=0,
            cells_newly_masked=5,
            cells_total_masked=5,
            total_cells=1000,
        )

    def test_print_statistics_tuple_threshold_formatted(self, sample_dataset, capsys):
        """Lines 1192-1193: tuple threshold is repr'd as a string."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.statistics.append(self._make_stat((100, 200)))

        runner.print_statistics()

        out = capsys.readouterr().out
        assert "100" in out
        assert "200" in out

    def test_print_statistics_none_threshold_shown_as_na(self, sample_dataset, capsys):
        """Lines 1194-1195: None threshold is shown as 'N/A'."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.statistics.append(self._make_stat(None))

        runner.print_statistics()

        out = capsys.readouterr().out
        assert "N/A" in out

    def test_print_statistics_numeric_threshold_formatted(self, sample_dataset, capsys):
        """Lines 1196-1197: plain numeric threshold is formatted as its string value."""
        runner = VelocityCheckRunner(sample_dataset)
        runner.statistics.append(self._make_stat(3.14))

        runner.print_statistics()

        out = capsys.readouterr().out
        assert "3.14" in out
