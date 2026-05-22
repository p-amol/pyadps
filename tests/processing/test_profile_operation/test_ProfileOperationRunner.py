"""
Test suite for ProfileOperationRunner class.

Tests the orchestrator class for profile-level operations including
method chaining, statistics tracking, and pipeline execution.

Note: Tests use a MockFixedLeaderAccessor that reads configuration from
dataset attributes, avoiding issues with dataset copy operations.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

try:
    from pyadps.processing.profile_operation import ProfileOperationRunner
    from pyadps.processing.utility import (
        QCCheckStats,
        QCPipelineReport,
        DataModificationStats,
    )
except ImportError:
    import sys

    sys.path.insert(0, "/mnt/user-data/outputs")
    from profile_operation import ProfileOperationRunner


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def adcp_dataset():
    """Create ADCP dataset with all required variables.

    Configuration:
    - 50m transducer depth (500 decimeters)
    - 4m cell size (400 cm)
    - 2m bin1 distance (200 cm)
    - 20Â° beam angle, upward-looking
    """
    np.random.seed(42)
    n_time = 100
    n_cell = 30
    n_beam = 4

    times = pd.date_range("2024-01-01", periods=n_time, freq="h")
    cells = np.arange(n_cell)
    beams = np.arange(n_beam)

    # Transducer at 50m depth (500 decimeters)
    transducer_depth = np.full(n_time, 500)

    velocity = np.random.randint(-500, 500, size=(n_beam, n_cell, n_time)).astype(
        np.int16
    )
    correlation = np.random.randint(50, 200, size=(n_beam, n_cell, n_time)).astype(
        np.int16
    )
    echo_intensity = np.random.randint(30, 150, size=(n_beam, n_cell, n_time)).astype(
        np.int16
    )

    data_vars = {
        "velocity": (("beam", "cell", "time"), velocity),
        "correlation": (("beam", "cell", "time"), correlation),
        "echo_intensity": (("beam", "cell", "time"), echo_intensity),
        "transducer_depth": (("time",), transducer_depth),
        "mask": (
            ("beam", "cell", "time"),
            np.zeros((n_beam, n_cell, n_time), dtype=np.int8),
        ),
    }

    coords = {"time": times, "cell": cells, "beam": beams}
    ds = xr.Dataset(data_vars, coords=coords)

    # Store configuration in attrs
    ds.attrs["cell_size_cm"] = 400
    ds.attrs["bin1_distance_cm"] = 200
    ds.attrs["beam_angle"] = 20
    ds.attrs["beam_direction"] = "up"

    return ds


@pytest.fixture
def dataset_with_pre_masked():
    """Create dataset with some pre-masked values.

    Configuration:
    - 50m transducer depth (500 decimeters)
    - 4m cell size (400 cm)
    - 2m bin1 distance (200 cm)
    - 20Â° beam angle, upward-looking
    - Pre-masked: first 2 cells, first 5 ensembles
    """
    np.random.seed(42)
    n_time = 50
    n_cell = 20
    n_beam = 4

    times = pd.date_range("2024-01-01", periods=n_time, freq="h")

    mask = np.zeros((n_beam, n_cell, n_time), dtype=np.int8)
    mask[:, :2, :5] = 1  # Pre-mask first 2 cells, first 5 ensembles

    data_vars = {
        "velocity": (
            ("beam", "cell", "time"),
            np.random.randint(-500, 500, size=(n_beam, n_cell, n_time)),
        ),
        "transducer_depth": (("time",), np.full(n_time, 500)),
        "mask": (("beam", "cell", "time"), mask),
    }

    coords = {
        "time": times,
        "cell": np.arange(n_cell),
        "beam": np.arange(n_beam),
    }

    ds = xr.Dataset(data_vars, coords=coords)

    ds.attrs["cell_size_cm"] = 400
    ds.attrs["bin1_distance_cm"] = 200
    ds.attrs["beam_angle"] = 20
    ds.attrs["beam_direction"] = "up"

    return ds


# ============================================================================
# TESTS: Initialization
# ============================================================================


class TestProfileOperationRunnerInit:
    """Tests for ProfileOperationRunner initialization."""

    def test_initialization(self, adcp_dataset):
        """Test basic initialization."""
        runner = ProfileOperationRunner(adcp_dataset)

        assert isinstance(runner.dataset, xr.Dataset)
        assert isinstance(runner.original, xr.Dataset)
        assert isinstance(runner.history, list)
        assert isinstance(runner.statistics, list)
        assert isinstance(runner.modifications, list)
        assert len(runner.history) == 0
        assert len(runner.statistics) == 0
        assert len(runner.modifications) == 0

    def test_initialization_creates_mask(self):
        """Test that mask is created if not present."""
        ds = xr.Dataset(
            {
                "velocity": (("beam", "cell", "time"), np.random.randn(4, 20, 30)),
                "transducer_depth": (("time",), np.full(30, 500)),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(20),
                "time": pd.date_range("2024-01-01", periods=30, freq="h"),
            },
        )
        ds.attrs["cell_size_cm"] = 400
        ds.attrs["bin1_distance_cm"] = 200
        ds.attrs["beam_angle"] = 20
        ds.attrs["beam_direction"] = "up"

        runner = ProfileOperationRunner(ds)

        assert "mask" in runner.dataset.data_vars

    def test_initialization_preserves_original(self, adcp_dataset):
        """Test that original dataset is preserved."""
        runner = ProfileOperationRunner(adcp_dataset)

        # Modify working dataset
        runner.dataset["velocity"].values[0, 0, 0] = 9999

        # Original should be unchanged
        assert runner.original["velocity"].values[0, 0, 0] != 9999

    def test_initialization_calculates_baseline(self, dataset_with_pre_masked):
        """Test baseline statistics calculation."""
        runner = ProfileOperationRunner(dataset_with_pre_masked)

        # Should have baseline masked count
        assert runner.baseline_masked > 0
        assert runner.baseline_masked_pct > 0

    def test_regridded_flag_initial(self, adcp_dataset):
        """Test that regridded flag is False initially."""
        runner = ProfileOperationRunner(adcp_dataset)

        assert runner._regridded is False


# ============================================================================
# TESTS: Trim Ensembles
# ============================================================================


class TestProfileOperationRunnerTrim:
    """Tests for trim_ensembles method."""

    def test_trim_ensembles_method(self, adcp_dataset):
        """Test trim_ensembles method."""
        runner = ProfileOperationRunner(adcp_dataset)
        result = runner.trim_ensembles(start=10, end=5)

        # Should return self for chaining
        assert result is runner

        # Mask should be updated
        assert np.all(runner.dataset["mask"].values[:, :, :10] == 1)
        assert np.all(runner.dataset["mask"].values[:, :, -5:] == 1)

    def test_trim_ensembles_records_statistics(self, adcp_dataset):
        """Test that trim_ensembles records statistics."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.trim_ensembles(start=10)

        assert len(runner.statistics) == 1
        assert runner.statistics[0].check_name == "Trim Ensembles"

    def test_trim_ensembles_records_history(self, adcp_dataset):
        """Test that trim_ensembles records history."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.trim_ensembles(start=10)

        assert len(runner.history) == 1
        assert runner.history[0]["operation"] == "trim_ensembles"


# ============================================================================
# TESTS: Cut Bins Side Lobe
# ============================================================================


class TestProfileOperationRunnerSideLobe:
    """Tests for cut_bins_side_lobe method."""

    def test_cut_bins_side_lobe_method(self, adcp_dataset):
        """Test cut_bins_side_lobe method."""
        runner = ProfileOperationRunner(adcp_dataset)
        result = runner.cut_bins_side_lobe(orientation="up", extra_cells=2)

        # Should return self for chaining
        assert result is runner

        # Some cells should be masked
        assert np.any(runner.dataset["mask"].values == 1)

    def test_cut_bins_side_lobe_records_statistics(self, adcp_dataset):
        """Test that cut_bins_side_lobe records statistics."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.cut_bins_side_lobe(orientation="up")

        assert len(runner.statistics) == 1
        assert runner.statistics[0].check_name == "Cut Bins Side Lobe"

    def test_cut_bins_side_lobe_records_history(self, adcp_dataset):
        """Test that cut_bins_side_lobe records history."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.cut_bins_side_lobe(orientation="up")

        assert len(runner.history) == 1
        assert runner.history[0]["operation"] == "cut_bins_side_lobe"


# ============================================================================
# TESTS: Cut Bins Manual
# ============================================================================


class TestProfileOperationRunnerManual:
    """Tests for cut_bins_manual method."""

    def test_cut_bins_manual_method(self, adcp_dataset):
        """Test cut_bins_manual method."""
        runner = ProfileOperationRunner(adcp_dataset)
        result = runner.cut_bins_manual(min_cell=5, max_cell=10)

        # Should return self for chaining
        assert result is runner

        # Specified cells should be masked
        assert np.all(runner.dataset["mask"].values[:, 5:10, :] == 1)

    def test_cut_bins_manual_records_statistics(self, adcp_dataset):
        """Test that cut_bins_manual records statistics."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.cut_bins_manual(min_cell=5, max_cell=10)

        assert len(runner.statistics) == 1
        assert runner.statistics[0].check_name == "Cut Bins Manual"

    def test_cut_bins_manual_records_history(self, adcp_dataset):
        """Test that cut_bins_manual records history."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.cut_bins_manual(min_cell=5, max_cell=10)

        assert len(runner.history) == 1
        assert runner.history[0]["operation"] == "cut_bins_manual"


# ============================================================================
# TESTS: Regrid
# ============================================================================


class TestProfileOperationRunnerRegrid:
    """Tests for regrid method."""

    def test_regrid_method(self, adcp_dataset):
        """Test regrid method."""
        runner = ProfileOperationRunner(adcp_dataset)
        result = runner.regrid(method="nearest")

        # Should return self for chaining
        assert result is runner

        # Dataset should have depth dimension
        assert "depth" in runner.dataset.dims
        assert "cell" not in runner.dataset.dims

    def test_regrid_sets_flag(self, adcp_dataset):
        """Test that regrid sets the regridded flag."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.regrid()

        assert runner._regridded is True

    def test_regrid_records_modification(self, adcp_dataset):
        """Test that regrid records modification statistics."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.regrid()

        assert len(runner.modifications) == 1
        assert runner.modifications[0].operation == "regrid"

    def test_regrid_records_history(self, adcp_dataset):
        """Test that regrid records history."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.regrid()

        assert len(runner.history) == 1
        assert runner.history[0]["operation"] == "regrid"


# ============================================================================
# TESTS: Method Chaining
# ============================================================================


class TestProfileOperationRunnerChaining:
    """Tests for method chaining."""

    def test_full_chain(self, adcp_dataset):
        """Test full method chain."""
        runner = ProfileOperationRunner(adcp_dataset)
        result = (
            runner.trim_ensembles(start=5, end=3)
            .cut_bins_side_lobe(extra_cells=2)
            .cut_bins_manual(min_cell=0, max_cell=2)
            .regrid(method="nearest")
            .finalize()
        )

        # Should return dataset
        assert isinstance(result, xr.Dataset)

        # All operations should be recorded
        assert len(runner.history) == 4  # 3 mask ops + 1 regrid

    def test_chain_accumulates_statistics(self, adcp_dataset):
        """Test that chaining accumulates statistics."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.trim_ensembles(start=5).cut_bins_manual(min_cell=5, max_cell=10)

        assert len(runner.statistics) == 2

    def test_chain_order_matters(self, adcp_dataset):
        """Test that operation order affects results."""
        runner1 = ProfileOperationRunner(adcp_dataset)
        runner1.trim_ensembles(start=10).cut_bins_manual(min_cell=5, max_cell=15)

        runner2 = ProfileOperationRunner(adcp_dataset)
        runner2.cut_bins_manual(min_cell=5, max_cell=15).trim_ensembles(start=10)

        # Both should have same final mask (operations are cumulative)
        np.testing.assert_array_equal(
            runner1.dataset["mask"].values, runner2.dataset["mask"].values
        )


# ============================================================================
# TESTS: Reset
# ============================================================================


class TestProfileOperationRunnerReset:
    """Tests for reset method."""

    def test_reset_restores_original(self, adcp_dataset):
        """Test that reset restores original dataset."""
        runner = ProfileOperationRunner(adcp_dataset)

        # Apply some operations
        runner.trim_ensembles(start=10)

        # Reset
        runner.reset()

        # Dataset should be back to original
        np.testing.assert_array_equal(
            runner.dataset["mask"].values, runner.original["mask"].values
        )

    def test_reset_clears_statistics(self, adcp_dataset):
        """Test that reset clears statistics."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.trim_ensembles(start=10)

        runner.reset()

        assert len(runner.statistics) == 0
        assert len(runner.modifications) == 0
        assert len(runner.history) == 0

    def test_reset_clears_regridded_flag(self, adcp_dataset):
        """Test that reset clears regridded flag."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.regrid()

        assert runner._regridded is True

        runner.reset()

        assert runner._regridded is False

    def test_reset_returns_self(self, adcp_dataset):
        """Test that reset returns self for chaining."""
        runner = ProfileOperationRunner(adcp_dataset)
        result = runner.reset()

        assert result is runner


# ============================================================================
# TESTS: Finalize
# ============================================================================


class TestProfileOperationRunnerFinalize:
    """Tests for finalize method."""

    def test_finalize_returns_dataset(self, adcp_dataset):
        """Test that finalize returns dataset."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.trim_ensembles(start=5)

        result = runner.finalize()

        assert isinstance(result, xr.Dataset)

    def test_finalize_adds_attributes(self, adcp_dataset):
        """Test that finalize adds processing attributes."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.trim_ensembles(start=5)

        result = runner.finalize()

        assert "profile_operation_processing_history" in result.attrs
        assert "profile_operation_processed_at" in result.attrs


# ============================================================================
# TESTS: Statistics Methods
# ============================================================================


class TestProfileOperationRunnerStatistics:
    """Tests for statistics methods."""

    def test_get_statistics(self, adcp_dataset):
        """Test get_statistics method."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.trim_ensembles(start=10)

        stats = runner.get_statistics()

        assert isinstance(stats, dict)
        assert "Trim Ensembles" in stats

    def test_get_modifications(self, adcp_dataset):
        """Test get_modifications method."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.regrid()

        mods = runner.get_modifications()

        assert isinstance(mods, dict)
        assert "dataset_structure" in mods

    def test_get_pipeline_report(self, adcp_dataset):
        """Test get_pipeline_report method."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.trim_ensembles(start=10).cut_bins_manual(min_cell=5, max_cell=10)

        report = runner.get_pipeline_report()

        assert report.module_name == "profile_operation"
        assert len(report.checks) == 2

    def test_get_report(self, adcp_dataset):
        """Test get_report method."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.trim_ensembles(start=10)

        report = runner.get_report()

        assert isinstance(report, str)

    def test_export_statistics_dict(self, adcp_dataset):
        """Test export_statistics_dict method."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.trim_ensembles(start=10)

        export = runner.export_statistics_dict()

        assert isinstance(export, dict)
        assert "module_name" in export


# ============================================================================
# TESTS: Print Methods
# ============================================================================


class TestProfileOperationRunnerPrint:
    """Tests for print methods."""

    def test_print_statistics(self, adcp_dataset, capsys):
        """Test print_statistics method."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.trim_ensembles(start=10)

        runner.print_statistics()

        captured = capsys.readouterr()
        assert "PROFILE OPERATION PROCESSING STATISTICS" in captured.out

    def test_summary(self, adcp_dataset):
        """Test summary method."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.trim_ensembles(start=10)

        summary = runner.summary()

        assert isinstance(summary, str)
        assert "PROFILE OPERATION PROCESSING SUMMARY" in summary
        assert "trim_ensembles" in summary


# ============================================================================
# TESTS: Apply Pipeline
# ============================================================================


class TestProfileOperationRunnerPipeline:
    """Tests for apply_pipeline method."""

    def test_apply_pipeline_basic(self, adcp_dataset):
        """Test apply_pipeline with configuration."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.apply_pipeline(
            operations={
                "trim_ensembles": {"start": 5, "end": 3},
                "cut_bins_manual": {"min_cell": 0, "max_cell": 2},
            },
            order=["trim_ensembles", "cut_bins_manual"],
        )

        assert len(runner.statistics) == 2

    def test_apply_pipeline_order_respected(self, adcp_dataset):
        """Test that pipeline respects operation order."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.apply_pipeline(
            operations={
                "trim_ensembles": {"start": 5},
                "cut_bins_manual": {"min_cell": 5, "max_cell": 10},
            },
            order=["cut_bins_manual", "trim_ensembles"],
        )

        # First operation should be cut_bins_manual
        assert runner.history[0]["operation"] == "cut_bins_manual"

    def test_apply_pipeline_returns_self(self, adcp_dataset):
        """Test that apply_pipeline returns self."""
        runner = ProfileOperationRunner(adcp_dataset)
        result = runner.apply_pipeline(
            operations={"trim_ensembles": {"start": 5}},
            order=["trim_ensembles"],
        )

        assert result is runner


# ============================================================================
# TESTS: Get Dataset
# ============================================================================


class TestProfileOperationRunnerGetDataset:
    """Tests for get_dataset method."""

    def test_get_dataset_returns_current(self, adcp_dataset):
        """Test get_dataset returns current working dataset."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.trim_ensembles(start=10)

        ds = runner.get_dataset()

        # Should be the modified dataset
        assert np.all(ds["mask"].values[:, :, :10] == 1)

    def test_get_dataset_same_as_dataset_attr(self, adcp_dataset):
        """Test get_dataset returns same as dataset attribute."""
        runner = ProfileOperationRunner(adcp_dataset)

        ds = runner.get_dataset()

        assert ds is runner.dataset


# ============================================================================
# TESTS: Edge Cases
# ============================================================================


class TestProfileOperationRunnerEdgeCases:
    """Tests for edge cases."""

    def test_empty_pipeline(self, adcp_dataset):
        """Test finalize without any operations."""
        runner = ProfileOperationRunner(adcp_dataset)
        result = runner.finalize()

        # Should work and return dataset
        assert isinstance(result, xr.Dataset)

    def test_multiple_resets(self, adcp_dataset):
        """Test multiple reset calls."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.trim_ensembles(start=10)
        runner.reset()
        runner.trim_ensembles(start=5)
        runner.reset()

        # Dataset should be at original state
        np.testing.assert_array_equal(
            runner.dataset["mask"].values, runner.original["mask"].values
        )

    def test_regrid_then_reset(self, adcp_dataset):
        """Test reset after regridding."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.regrid()

        # Dataset now has depth dimension
        assert "depth" in runner.dataset.dims

        runner.reset()

        # Should be back to cell dimension
        assert "cell" in runner.dataset.dims
        assert "depth" not in runner.dataset.dims

    def test_statistics_cumulative(self, adcp_dataset):
        """Test that statistics are cumulative."""
        runner = ProfileOperationRunner(adcp_dataset)

        runner.trim_ensembles(start=5)
        stat1 = runner.statistics[-1]

        runner.cut_bins_manual(min_cell=5, max_cell=10)
        stat2 = runner.statistics[-1]

        # Second stat should show cumulative masking
        assert stat2.cells_pre_masked >= stat1.cells_newly_masked


# ============================================================================
# TESTS: Input Validation
# ============================================================================


class TestProfileOperationRunnerValidation:
    """Tests for input validation."""

    def test_invalid_dataset(self):
        """Test error for invalid dataset."""
        with pytest.raises(TypeError):
            ProfileOperationRunner("not a dataset")

    def test_missing_velocity(self):
        """Test error when velocity is missing."""
        ds = xr.Dataset(
            {
                "correlation": (("beam", "cell", "time"), np.random.randn(4, 10, 20)),
                "transducer_depth": (("time",), np.full(20, 500)),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(10),
                "time": pd.date_range("2024-01-01", periods=20, freq="h"),
            },
        )
        ds.attrs["cell_size_cm"] = 400
        ds.attrs["bin1_distance_cm"] = 200
        ds.attrs["beam_angle"] = 20
        ds.attrs["beam_direction"] = "up"

        with pytest.raises(ValueError, match="missing required variable"):
            ProfileOperationRunner(ds)


# ============================================================================
# TESTS: Integration
# ============================================================================


class TestProfileOperationRunnerIntegration:
    """Integration tests for ProfileOperationRunner."""

    def test_full_workflow(self, adcp_dataset):
        """Test complete workflow from start to finish."""
        runner = ProfileOperationRunner(adcp_dataset)

        # Apply operations
        result = (
            runner.trim_ensembles(start=10, end=5)
            .cut_bins_side_lobe(extra_cells=2)
            .cut_bins_manual(min_cell=0, max_cell=2)
            .regrid(method="linear")
            .finalize()
        )

        # Verify result
        assert isinstance(result, xr.Dataset)
        assert "depth" in result.dims
        assert "velocity" in result.data_vars
        assert "mask" in result.data_vars

        # Verify statistics recorded
        assert len(runner.statistics) == 3  # 3 mask operations
        assert len(runner.modifications) == 1  # 1 regrid
        assert len(runner.history) == 4  # All operations

    def test_workflow_statistics_accurate(self, adcp_dataset):
        """Test that statistics accurately reflect operations."""
        runner = ProfileOperationRunner(adcp_dataset)

        n_beam = adcp_dataset.sizes["beam"]
        n_cell = adcp_dataset.sizes["cell"]
        trim_count = 10

        runner.trim_ensembles(start=trim_count)

        stat = runner.statistics[-1]

        # Check exact count
        expected_masked = n_beam * n_cell * trim_count
        assert stat.cells_newly_masked == expected_masked


# ============================================================================
# TESTS: Coverage for specific uncovered branches
# ============================================================================


class TestProfileOperationRunnerUncoveredBranches:
    """Tests for branches not covered by existing test classes."""

    # ---- Lines 1202-1204: _update_baseline with no mask ----

    def test_update_baseline_no_mask(self):
        """Lines 1202-1204: _update_baseline sets zeros when dataset has no mask."""
        n_time, n_cell, n_beam = 20, 10, 4
        times = pd.date_range("2024-01-01", periods=n_time, freq="h")

        ds_no_mask = xr.Dataset(
            {
                "velocity": (
                    ("beam", "cell", "time"),
                    np.random.randint(-500, 500, (n_beam, n_cell, n_time)).astype(
                        np.int16
                    ),
                ),
                "transducer_depth": (("time",), np.full(n_time, 500)),
            },
            coords={
                "beam": np.arange(n_beam),
                "cell": np.arange(n_cell),
                "time": times,
            },
        )
        ds_no_mask.attrs.update(
            {
                "cell_size_cm": 400,
                "bin1_distance_cm": 200,
                "beam_angle": 20,
                "beam_direction": "up",
            }
        )

        runner = ProfileOperationRunner(ds_no_mask)
        # After init the runner creates a mask; manually remove and call _update_baseline
        del runner.dataset["mask"]
        runner._update_baseline()

        assert runner.total_cells == 0
        assert runner.baseline_masked == 0
        assert runner.baseline_masked_pct == 0.0

    # ---- Line 1249: raise RuntimeError after regrid ----

    def test_mask_operation_after_regrid_raises(self, adcp_dataset):
        """Line 1249: mask-based operation after regrid() raises RuntimeError."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.regrid(method="linear")

        with pytest.raises(RuntimeError, match="after regridding"):
            runner.trim_ensembles(start=5)

    # ---- apply_pipeline (lines 1546, 1549, 1558, 1564, 1567-1568) ----

    def test_apply_pipeline_with_none_uses_defaults(self, adcp_dataset):
        """Lines 1546, 1549: apply_pipeline(None, None) fills both defaults."""
        runner = ProfileOperationRunner(adcp_dataset)
        # With both None, operations={} and default order -> all names hit
        # 'continue' because none are in {}. Should return self without error.
        result = runner.apply_pipeline(operations=None, order=None)
        assert result is runner
        assert len(runner.history) == 0  # nothing was executed

    def test_apply_pipeline_skips_names_not_in_operations(self, adcp_dataset):
        """Line 1558: names in order that are absent from operations are skipped."""
        runner = ProfileOperationRunner(adcp_dataset)
        # Provide only trim_ensembles; cut_bins_side_lobe and others are skipped
        runner.apply_pipeline(
            operations={"trim_ensembles": {"start": 5}},
            order=["trim_ensembles", "cut_bins_side_lobe", "regrid"],
        )
        ops = [h["operation"] for h in runner.history]
        assert "trim_ensembles" in ops
        assert "cut_bins_side_lobe" not in ops
        assert "regrid" not in ops

    def test_apply_pipeline_cut_bins_side_lobe(self, adcp_dataset):
        """Line 1564: cut_bins_side_lobe branch in apply_pipeline."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.apply_pipeline(
            operations={"cut_bins_side_lobe": {"extra_cells": 1}},
            order=["cut_bins_side_lobe"],
        )
        ops = [h["operation"] for h in runner.history]
        assert "cut_bins_side_lobe" in ops

    def test_apply_pipeline_regrid(self, adcp_dataset):
        """Lines 1567-1568: regrid branch in apply_pipeline."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.apply_pipeline(
            operations={"regrid": {"method": "linear"}},
            order=["regrid"],
        )
        assert runner._regridded is True

    # ---- Line 1583: reset() recreates mask when original has none ----

    def test_reset_recreates_mask_when_original_has_none(self):
        """Line 1583: reset() calls create_default_mask when original lacks mask."""
        n_time, n_cell, n_beam = 20, 10, 4
        times = pd.date_range("2024-01-01", periods=n_time, freq="h")

        ds_no_mask = xr.Dataset(
            {
                "velocity": (
                    ("beam", "cell", "time"),
                    np.random.randint(-500, 500, (n_beam, n_cell, n_time)).astype(
                        np.int16
                    ),
                ),
                "transducer_depth": (("time",), np.full(n_time, 500)),
            },
            coords={
                "beam": np.arange(n_beam),
                "cell": np.arange(n_cell),
                "time": times,
            },
        )
        ds_no_mask.attrs.update(
            {
                "cell_size_cm": 400,
                "bin1_distance_cm": 200,
                "beam_angle": 20,
                "beam_direction": "up",
            }
        )

        runner = ProfileOperationRunner(ds_no_mask)
        assert "mask" not in runner.original.data_vars  # original still mask-free

        runner.trim_ensembles(start=2)
        runner.reset()  # hits line 1583

        assert "mask" in runner.dataset.data_vars

    # ---- Lines 1705-1722: print_statistics STRUCTURAL MODIFICATIONS block ----

    def test_print_statistics_with_modifications(self, adcp_dataset, capsys):
        """Lines 1705-1722: print_statistics prints STRUCTURAL MODIFICATIONS
        when self.modifications is non-empty (i.e. after regrid)."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.regrid(method="linear")
        runner.print_statistics()

        captured = capsys.readouterr()
        assert "STRUCTURAL MODIFICATIONS" in captured.out
        assert "regrid" in captured.out

    # ---- Line 1726: "No mask operations applied." ----

    def test_print_statistics_no_mask_operations(self, adcp_dataset, capsys):
        """Line 1726: when no mask operations applied, appropriate message printed."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.print_statistics()

        captured = capsys.readouterr()
        assert "No mask operations applied." in captured.out

    # ---- Line 1768: summary() with empty history ----

    def test_summary_empty_history(self, adcp_dataset):
        """Line 1768: summary() with no history returns 'No processing steps' message."""
        runner = ProfileOperationRunner(adcp_dataset)
        result = runner.summary()

        assert "No processing steps applied yet." in result

    # ---- Lines 1782-1783: summary() regrid entry with new_depth_levels ----

    def test_summary_regrid_shows_depth_levels(self, adcp_dataset):
        """Lines 1782-1783: summary() shows structure change line for regrid entries."""
        runner = ProfileOperationRunner(adcp_dataset)
        runner.regrid(method="linear")
        result = runner.summary()

        assert "depth levels" in result
        assert "Structure" in result
