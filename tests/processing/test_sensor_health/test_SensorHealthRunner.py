"""
Test suite for SensorHealthRunner class.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr

from pyadps.processing.sensor_health import SensorHealthRunner
from pyadps.processing.utility import (
    QCCheckStats,
    QCPipelineReport,
    DataModificationStats,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def adcp_dataset():
    """Create ADCP dataset with all required variables."""
    n_time = 50
    n_cell = 20
    n_beam = 4

    times = pd.date_range("2024-01-01 00:00:00", periods=n_time, freq="h")
    cells = np.arange(n_cell)
    beams = np.arange(n_beam)

    # Create velocity data with some missing values
    velocity = np.random.randn(n_beam, n_cell, n_time) * 100
    velocity[0, 5, 10] = -32768  # Missing value

    data_vars = {
        "velocity": (
            ("beam", "cell", "time"),
            velocity,
        ),
        # Roll in 0.01Â° units: values from -5Â° to +5Â° (normal range)
        "roll": (("time",), np.random.uniform(-500, 500, n_time)),
        # Pitch in 0.01Â° units: values from -5Â° to +5Â° (normal range)
        "pitch": (("time",), np.random.uniform(-500, 500, n_time)),
        # Temperature in 0.01Â°C units (15Â°C = 1500)
        "temperature": (("time",), np.ones(n_time) * 1500),
        # Salinity in 0.001 PSU units (35 PSU = 35000)
        "salinity": (("time",), np.ones(n_time) * 35000),
        # Depth in 0.1 m units (100 m = 1000)
        "transducer_depth": (("time",), np.ones(n_time) * 1000),
        # Sound speed in m/s (typical ~1500 m/s)
        "sound_speed": (("time",), np.ones(n_time) * 1500),
        "mask": (
            ("beam", "cell", "time"),
            np.zeros((n_beam, n_cell, n_time), dtype=np.int8),
        ),
    }

    coords = {
        "time": times,
        "cell": cells,
        "beam": beams,
    }

    ds = xr.Dataset(data_vars, coords=coords)

    # Add attributes with scale_factor for proper RDI format handling
    ds["sound_speed"].attrs = {"long_name": "Speed of sound", "units": "m/s"}
    ds["velocity"].attrs = {"long_name": "Velocity", "units": "mm/s"}
    ds["temperature"].attrs = {
        "long_name": "Temperature",
        "units": "0.01 degC",
        "scale_factor": 0.01,
    }
    ds["salinity"].attrs = {
        "long_name": "Salinity",
        "units": "0.001 PSU",
        "scale_factor": 0.001,
    }
    ds["roll"].attrs = {
        "long_name": "Roll",
        "units": "0.01 degrees",
        "scale_factor": 0.01,
    }
    ds["pitch"].attrs = {
        "long_name": "Pitch",
        "units": "0.01 degrees",
        "scale_factor": 0.01,
    }
    ds["transducer_depth"].attrs = {
        "long_name": "Transducer Depth",
        "units": "0.1 m",
        "scale_factor": 0.1,
    }

    return ds


@pytest.fixture
def adcp_dataset_with_high_roll():
    """Create dataset with some high roll values."""
    n_time = 50
    n_cell = 20
    n_beam = 4

    times = pd.date_range("2024-01-01 00:00:00", periods=n_time, freq="h")

    # Roll: first 10 ensembles exceed 20Â° threshold
    roll = np.random.uniform(-500, 500, n_time)
    roll[:10] = 2500  # 25Â° (exceeds 20Â° threshold)

    data_vars = {
        "velocity": (
            ("beam", "cell", "time"),
            np.random.randn(n_beam, n_cell, n_time) * 100,
        ),
        "roll": (("time",), roll),
        "pitch": (("time",), np.random.uniform(-500, 500, n_time)),
        "temperature": (("time",), np.ones(n_time) * 1500),
        "salinity": (("time",), np.ones(n_time) * 35000),
        "transducer_depth": (("time",), np.ones(n_time) * 1000),
        "sound_speed": (("time",), np.ones(n_time) * 1500),
        "mask": (
            ("beam", "cell", "time"),
            np.zeros((n_beam, n_cell, n_time), dtype=np.int8),
        ),
    }

    coords = {
        "time": times,
        "cell": np.arange(n_cell),
        "beam": np.arange(n_beam),
    }

    ds = xr.Dataset(data_vars, coords=coords)
    ds["sound_speed"].attrs = {"long_name": "Speed of sound", "units": "m/s"}
    ds["velocity"].attrs = {"long_name": "Velocity", "units": "mm/s"}
    ds["temperature"].attrs = {
        "long_name": "Temperature",
        "units": "0.01 degC",
        "scale_factor": 0.01,
    }
    ds["salinity"].attrs = {
        "long_name": "Salinity",
        "units": "0.001 PSU",
        "scale_factor": 0.001,
    }
    ds["roll"].attrs = {
        "long_name": "Roll",
        "units": "0.01 degrees",
        "scale_factor": 0.01,
    }
    ds["pitch"].attrs = {
        "long_name": "Pitch",
        "units": "0.01 degrees",
        "scale_factor": 0.01,
    }

    return ds


# ============================================================================
# TESTS: Initialization
# ============================================================================


class TestSensorHealthRunnerInit:
    """Test SensorHealthRunner initialization."""

    def test_initialization(self, adcp_dataset):
        """Test basic initialization."""
        runner = SensorHealthRunner(adcp_dataset)
        assert isinstance(runner.dataset, xr.Dataset)
        assert isinstance(runner.original, xr.Dataset)
        assert isinstance(runner.history, list)
        assert len(runner.history) == 0

    def test_original_preserved(self, adcp_dataset):
        """Test that original dataset is preserved as a copy."""
        runner = SensorHealthRunner(adcp_dataset)
        # Modify the working dataset
        runner.dataset["roll"].values[:] = 0
        # Original should be unchanged
        assert not np.allclose(runner.original["roll"].values, 0)

    def test_dataset_is_copy(self, adcp_dataset):
        """Test that dataset is a deep copy."""
        runner = SensorHealthRunner(adcp_dataset)
        # Modify original input
        adcp_dataset["roll"].values[:] = 9999
        # Runner's dataset should be unchanged
        assert not np.allclose(runner.dataset["roll"].values, 9999)


# ============================================================================
# TESTS: Method Chaining
# ============================================================================


class TestMethodChaining:
    """Test method chaining functionality."""

    def test_roll_check_returns_self(self, adcp_dataset):
        """Test that roll_check returns self."""
        runner = SensorHealthRunner(adcp_dataset)
        result = runner.roll_check()
        assert result is runner

    def test_pitch_check_returns_self(self, adcp_dataset):
        """Test that pitch_check returns self."""
        runner = SensorHealthRunner(adcp_dataset)
        result = runner.pitch_check()
        assert result is runner

    def test_correct_sound_speed_returns_self(self, adcp_dataset):
        """Test that correct_sound_speed returns self."""
        runner = SensorHealthRunner(adcp_dataset)
        result = runner.correct_sound_speed()
        assert result is runner

    def test_replace_data_returns_self(self, adcp_dataset):
        """Test that replace_data returns self."""
        runner = SensorHealthRunner(adcp_dataset)
        new_temp = np.ones(50) * 20.0  # 20Â°C in physical units
        result = runner.replace_data(new_temp, "temperature")
        assert result is runner

    def test_reset_returns_self(self, adcp_dataset):
        """Test that reset returns self."""
        runner = SensorHealthRunner(adcp_dataset)
        result = runner.reset()
        assert result is runner

    def test_chained_operations(self, adcp_dataset):
        """Test multiple chained operations."""
        runner = SensorHealthRunner(adcp_dataset)
        result = (
            runner.roll_check(threshold=20.0)
            .pitch_check(threshold=15.0)
            .correct_sound_speed()
            .finalize()
        )
        assert isinstance(result, xr.Dataset)

    def test_full_workflow(self, adcp_dataset):
        """Test complete workflow with all operations."""
        runner = SensorHealthRunner(adcp_dataset)
        new_temp = np.ones(50) * 20.0  # 20Â°C in physical units
        new_sal = np.ones(50) * 36.0  # 36 PSU in physical units

        result = (
            runner.replace_data(new_temp, "temperature")
            .replace_data(new_sal, "salinity")
            .correct_sound_speed()
            .roll_check()
            .pitch_check()
            .finalize()
        )

        assert isinstance(result, xr.Dataset)
        assert len(runner.history) == 5


# ============================================================================
# TESTS: Processing History
# ============================================================================


class TestProcessingHistory:
    """Test processing history tracking."""

    def test_history_empty_initially(self, adcp_dataset):
        """Test that history is empty on initialization."""
        runner = SensorHealthRunner(adcp_dataset)
        assert len(runner.history) == 0

    def test_history_after_roll_check(self, adcp_dataset):
        """Test history entry after roll_check."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.roll_check(threshold=20.0)

        assert len(runner.history) == 1
        entry = runner.history[0]
        assert entry["operation"] == "roll_check"
        assert entry["parameters"]["threshold"] == 20.0
        assert "timestamp" in entry
        assert "stats" in entry

    def test_history_after_pitch_check(self, adcp_dataset):
        """Test history entry after pitch_check."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.pitch_check(threshold=10.0)

        assert len(runner.history) == 1
        entry = runner.history[0]
        assert entry["operation"] == "pitch_check"
        assert entry["parameters"]["threshold"] == 10.0

    def test_history_after_correct_sound_speed(self, adcp_dataset):
        """Test history entry after correct_sound_speed."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.correct_sound_speed(correct_velocity=True, horizontal_only=False)

        assert len(runner.history) == 1
        entry = runner.history[0]
        assert entry["operation"] == "correct_sound_speed"
        assert entry["parameters"]["correct_velocity"] is True
        assert entry["parameters"]["horizontal_only"] is False
        assert "stats" in entry

    def test_history_after_replace_data(self, adcp_dataset):
        """Test history entry after replace_data."""
        runner = SensorHealthRunner(adcp_dataset)
        new_temp = np.ones(50) * 20.0  # 20Â°C in physical units
        runner.replace_data(new_temp, "temperature")

        assert len(runner.history) == 1
        entry = runner.history[0]
        assert entry["operation"] == "replace_data"
        assert entry["parameters"]["variable_name"] == "temperature"
        assert entry["parameters"]["data_shape"] == [50]

    def test_history_accumulates(self, adcp_dataset):
        """Test that history accumulates across operations."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.roll_check().pitch_check().correct_sound_speed()

        assert len(runner.history) == 3
        assert runner.history[0]["operation"] == "roll_check"
        assert runner.history[1]["operation"] == "pitch_check"
        assert runner.history[2]["operation"] == "correct_sound_speed"

    def test_history_cleared_on_reset(self, adcp_dataset):
        """Test that history is cleared on reset."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.roll_check().pitch_check()
        assert len(runner.history) == 2

        runner.reset()
        assert len(runner.history) == 0


# ============================================================================
# TESTS: Statistics Tracking
# ============================================================================


class TestStatisticsTracking:
    """Test statistics tracking in history."""

    def test_roll_check_stats(self, adcp_dataset_with_high_roll):
        """Test roll_check captures correct statistics."""
        runner = SensorHealthRunner(adcp_dataset_with_high_roll)
        runner.roll_check(threshold=20.0)

        stats = runner.history[0]["stats"]
        assert "roll_min" in stats
        assert "roll_max" in stats
        assert "roll_mean" in stats
        assert "ensembles_exceeding_threshold" in stats
        assert "total_ensembles" in stats
        assert "cells_newly_masked" in stats

        # 10 ensembles exceed 20Â° threshold
        assert stats["ensembles_exceeding_threshold"] == 10
        assert stats["total_ensembles"] == 50

    def test_sound_speed_stats(self, adcp_dataset):
        """Test correct_sound_speed captures statistics."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.correct_sound_speed()

        stats = runner.history[0]["stats"]
        assert "original_sound_speed_mean" in stats
        assert "corrected_sound_speed_mean" in stats
        assert "difference" in stats

        # Original was 1500
        assert stats["original_sound_speed_mean"] == 1500.0

    def test_new_flags_tracking(self, adcp_dataset_with_high_roll):
        """Test that cells_newly_masked is correctly calculated."""
        runner = SensorHealthRunner(adcp_dataset_with_high_roll)

        # First check - should add flags
        runner.roll_check(threshold=20.0)
        first_flags = runner.history[0]["stats"]["cells_newly_masked"]
        assert first_flags > 0

        # Second check with same threshold - should add 0 new flags
        # (already flagged)
        runner.roll_check(threshold=20.0)
        second_flags = runner.history[1]["stats"]["cells_newly_masked"]
        assert second_flags == 0


# ============================================================================
# TESTS: Reset Functionality
# ============================================================================


class TestReset:
    """Test reset functionality."""

    def test_reset_restores_dataset(self, adcp_dataset):
        """Test that reset restores dataset to original state."""
        runner = SensorHealthRunner(adcp_dataset)
        original_ss = runner.dataset["sound_speed"].values.copy()

        # Modify dataset
        runner.correct_sound_speed()
        assert not np.allclose(runner.dataset["sound_speed"].values, original_ss)

        # Reset
        runner.reset()
        np.testing.assert_array_equal(runner.dataset["sound_speed"].values, original_ss)

    def test_reset_clears_history(self, adcp_dataset):
        """Test that reset clears history."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.roll_check().pitch_check()
        assert len(runner.history) == 2

        runner.reset()
        assert len(runner.history) == 0

    def test_reset_preserves_original(self, adcp_dataset):
        """Test that reset doesn't modify original."""
        runner = SensorHealthRunner(adcp_dataset)
        original_copy = runner.original.copy(deep=True)

        runner.correct_sound_speed()
        runner.reset()

        xr.testing.assert_identical(runner.original, original_copy)


# ============================================================================
# TESTS: Finalize
# ============================================================================


class TestFinalize:
    """Test finalize functionality."""

    def test_finalize_returns_dataset(self, adcp_dataset):
        """Test that finalize returns xr.Dataset."""
        runner = SensorHealthRunner(adcp_dataset)
        result = runner.roll_check().finalize()
        assert isinstance(result, xr.Dataset)

    def test_finalize_adds_history_attribute(self, adcp_dataset):
        """Test that finalize adds processing history to attributes."""
        runner = SensorHealthRunner(adcp_dataset)
        result = runner.roll_check().pitch_check().finalize()

        assert "sensor_health_processing" in result.attrs
        assert "sensor_health_processed_at" in result.attrs

    def test_finalize_preserves_data(self, adcp_dataset):
        """Test that finalize doesn't modify data."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.roll_check()
        expected_mask = runner.dataset["mask"].values.copy()

        result = runner.finalize()
        np.testing.assert_array_equal(result["mask"].values, expected_mask)

    def test_finalize_doesnt_modify_runner(self, adcp_dataset):
        """Test that finalize doesn't modify runner state."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.roll_check()
        history_len_before = len(runner.history)

        runner.finalize()

        assert len(runner.history) == history_len_before


# ============================================================================
# TESTS: Summary
# ============================================================================


class TestSummary:
    """Test summary functionality."""

    def test_summary_empty_history(self, adcp_dataset):
        """Test summary with no operations."""
        runner = SensorHealthRunner(adcp_dataset)
        summary = runner.summary()
        assert "No processing steps applied" in summary

    def test_summary_with_operations(self, adcp_dataset):
        """Test summary with operations."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.roll_check(threshold=20.0).pitch_check(threshold=15.0)

        summary = runner.summary()
        assert "roll_check" in summary
        assert "pitch_check" in summary
        assert "threshold" in summary
        assert "Step 1" in summary
        assert "Step 2" in summary

    def test_summary_returns_string(self, adcp_dataset):
        """Test that summary returns a string."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.roll_check()
        summary = runner.summary()
        assert isinstance(summary, str)


# ============================================================================
# TESTS: Data Modification
# ============================================================================


class TestDataModification:
    """Test that data is correctly modified."""

    def test_roll_check_modifies_mask(self, adcp_dataset_with_high_roll):
        """Test that roll_check modifies the mask."""
        runner = SensorHealthRunner(adcp_dataset_with_high_roll)
        initial_flags = int(runner.dataset["mask"].sum())

        runner.roll_check(threshold=20.0)
        final_flags = int(runner.dataset["mask"].sum())

        assert final_flags > initial_flags

    def test_correct_sound_speed_modifies_data(self, adcp_dataset):
        """Test that correct_sound_speed modifies sound_speed and velocity."""
        runner = SensorHealthRunner(adcp_dataset)
        original_ss = runner.dataset["sound_speed"].values.copy()
        original_vel = runner.dataset["velocity"].values.copy()

        runner.correct_sound_speed()

        assert not np.allclose(runner.dataset["sound_speed"].values, original_ss)
        assert not np.allclose(runner.dataset["velocity"].values, original_vel)

    def test_replace_data_modifies_variable(self, adcp_dataset):
        """Test that replace_data modifies the specified variable."""
        runner = SensorHealthRunner(adcp_dataset)
        # Provide data in physical units (25Â°C)
        new_temp_physical = np.ones(50) * 25.0
        # With scale_factor=0.01, this becomes 2500 in stored format
        expected_stored = np.ones(50) * 2500

        runner.replace_data(new_temp_physical, "temperature")

        np.testing.assert_array_almost_equal(
            runner.dataset["temperature"].values, expected_stored
        )


# ============================================================================
# TESTS: Original Dataset Preservation
# ============================================================================


class TestOriginalPreservation:
    """Test that original dataset is never modified."""

    def test_roll_check_preserves_original(self, adcp_dataset):
        """Test that roll_check doesn't modify original."""
        runner = SensorHealthRunner(adcp_dataset)
        original_mask = runner.original["mask"].values.copy()

        runner.roll_check()

        np.testing.assert_array_equal(runner.original["mask"].values, original_mask)

    def test_correct_sound_speed_preserves_original(self, adcp_dataset):
        """Test that correct_sound_speed doesn't modify original."""
        runner = SensorHealthRunner(adcp_dataset)
        original_ss = runner.original["sound_speed"].values.copy()

        runner.correct_sound_speed()

        np.testing.assert_array_equal(
            runner.original["sound_speed"].values, original_ss
        )

    def test_replace_data_preserves_original(self, adcp_dataset):
        """Test that replace_data doesn't modify original."""
        runner = SensorHealthRunner(adcp_dataset)
        original_temp = runner.original["temperature"].values.copy()

        new_temp = np.ones(50) * 9999
        # Use apply_scale_factor=False to provide raw RDI format data
        runner.replace_data(new_temp, "temperature", apply_scale_factor=False)

        np.testing.assert_array_equal(
            runner.original["temperature"].values, original_temp
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================================
# TESTS: Statistics Dataclasses
# ============================================================================


class TestQCCheckStats:
    """Test QCCheckStats dataclass."""

    def test_basic_creation(self):
        """Test basic dataclass creation."""
        stat = QCCheckStats(
            check_name="Roll Check",
            threshold=15.0,
            cells_pre_masked=100,
            cells_newly_masked=50,
            cells_total_masked=150,
            total_cells=1000,
        )
        assert stat.check_name == "Roll Check"
        assert stat.threshold == 15.0
        assert stat.cells_pre_masked == 100
        assert stat.cells_newly_masked == 50
        assert stat.cells_total_masked == 150
        assert stat.total_cells == 1000

    def test_pre_masked_pct(self):
        """Test pre_masked_pct property."""
        stat = QCCheckStats(
            check_name="Roll Check",
            threshold=15.0,
            cells_pre_masked=100,
            cells_newly_masked=50,
            cells_total_masked=150,
            total_cells=1000,
        )
        assert stat.pre_masked_pct == 10.0

    def test_newly_masked_pct(self):
        """Test newly_masked_pct property."""
        stat = QCCheckStats(
            check_name="Roll Check",
            threshold=15.0,
            cells_pre_masked=100,
            cells_newly_masked=50,
            cells_total_masked=150,
            total_cells=1000,
        )
        assert stat.newly_masked_pct == 5.0

    def test_total_masked_pct(self):
        """Test total_masked_pct property."""
        stat = QCCheckStats(
            check_name="Roll Check",
            threshold=15.0,
            cells_pre_masked=100,
            cells_newly_masked=50,
            cells_total_masked=150,
            total_cells=1000,
        )
        assert stat.total_masked_pct == 15.0

    def test_valid_cells(self):
        """Test valid_cells property."""
        stat = QCCheckStats(
            check_name="Roll Check",
            threshold=15.0,
            cells_pre_masked=100,
            cells_newly_masked=50,
            cells_total_masked=150,
            total_cells=1000,
        )
        assert stat.valid_cells == 850

    def test_valid_pct(self):
        """Test valid_pct property."""
        stat = QCCheckStats(
            check_name="Roll Check",
            threshold=15.0,
            cells_pre_masked=100,
            cells_newly_masked=50,
            cells_total_masked=150,
            total_cells=1000,
        )
        assert stat.valid_pct == 85.0

    def test_zero_total_cells(self):
        """Test edge case with zero total cells."""
        stat = QCCheckStats(
            check_name="Roll Check",
            threshold=15.0,
            cells_pre_masked=0,
            cells_newly_masked=0,
            cells_total_masked=0,
            total_cells=0,
        )
        assert stat.pre_masked_pct == 0.0
        assert stat.newly_masked_pct == 0.0
        assert stat.total_masked_pct == 0.0
        assert stat.valid_pct == 0.0

    def test_str_representation(self):
        """Test string representation."""
        stat = QCCheckStats(
            check_name="Roll Check",
            threshold=15.0,
            cells_pre_masked=100,
            cells_newly_masked=50,
            cells_total_masked=150,
            total_cells=1000,
        )
        s = str(stat)
        assert "Roll Check" in s
        assert "15.0" in s


class TestQCPipelineReport:
    """Test QCPipelineReport dataclass."""

    def test_basic_creation(self):
        """Test basic report creation."""
        report = QCPipelineReport(
            module_name="sensor_health",
            baseline_masked=100,
            baseline_masked_pct=10.0,
            total_cells=1000,
            checks=[],
        )
        assert report.baseline_masked == 100
        assert report.baseline_masked_pct == 10.0
        assert report.total_cells == 1000
        assert len(report.checks) == 0

    def test_final_valid_pct_no_checks(self):
        """Test final_valid_pct with no checks."""
        report = QCPipelineReport(
            module_name="sensor_health",
            baseline_masked=100,
            baseline_masked_pct=10.0,
            total_cells=1000,
            checks=[],
        )
        assert report.final_valid_pct == 90.0

    def test_final_valid_pct_with_checks(self):
        """Test final_valid_pct with checks."""
        stat = QCCheckStats(
            check_name="Roll Check",
            threshold=15.0,
            cells_pre_masked=100,
            cells_newly_masked=50,
            cells_total_masked=150,
            total_cells=1000,
        )
        report = QCPipelineReport(
            module_name="sensor_health",
            baseline_masked=100,
            baseline_masked_pct=10.0,
            total_cells=1000,
            checks=[stat],
        )
        assert report.final_valid_pct == 85.0

    def test_pipeline_impact_pct(self):
        """Test pipeline_impact_pct."""
        stat = QCCheckStats(
            check_name="Roll Check",
            threshold=15.0,
            cells_pre_masked=100,
            cells_newly_masked=50,
            cells_total_masked=150,
            total_cells=1000,
        )
        report = QCPipelineReport(
            module_name="sensor_health",
            baseline_masked=100,
            baseline_masked_pct=10.0,
            total_cells=1000,
            checks=[stat],
        )
        # Impact = 15% - 10% = 5%
        assert report.pipeline_impact_pct == 5.0

    def test_str_representation(self):
        """Test string representation."""
        report = QCPipelineReport(
            module_name="sensor_health",
            baseline_masked=100,
            baseline_masked_pct=10.0,
            total_cells=1000,
            checks=[],
        )
        s = str(report)
        assert "SENSOR_HEALTH" in s
        assert "Baseline masked" in s


# ============================================================================
# TESTS: Statistics Methods in Runner
# ============================================================================


class TestRunnerStatisticsMethods:
    """Test statistics methods in SensorHealthRunner."""

    def test_get_statistics_empty(self, adcp_dataset):
        """Test get_statistics with no checks."""
        runner = SensorHealthRunner(adcp_dataset)
        stats = runner.get_statistics()
        assert len(stats) == 0

    def test_get_statistics_after_checks(self, adcp_dataset_with_high_roll):
        """Test get_statistics after applying checks."""
        runner = SensorHealthRunner(adcp_dataset_with_high_roll)
        runner.roll_check(threshold=20.0).pitch_check(threshold=15.0)

        stats = runner.get_statistics()
        assert len(stats) == 2
        assert "Roll Check" in stats
        assert "Pitch Check" in stats
        assert isinstance(stats["Roll Check"], QCCheckStats)

    def test_print_statistics(self, adcp_dataset, capsys):
        """Test print_statistics output."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.roll_check().pitch_check()
        runner.print_statistics()

        captured = capsys.readouterr()
        assert "SENSOR HEALTH PROCESSING STATISTICS" in captured.out
        assert "Roll Check" in captured.out
        assert "Pitch Check" in captured.out
        assert "FINAL" in captured.out

    def test_print_statistics_empty(self, adcp_dataset, capsys):
        """Test print_statistics with no checks."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.print_statistics()

        captured = capsys.readouterr()
        assert "No QC checks applied yet" in captured.out

    def test_get_report(self, adcp_dataset):
        """Test get_report returns string."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.roll_check()

        report = runner.get_report()
        assert isinstance(report, str)
        assert "PIPELINE REPORT" in report

    def test_get_pipeline_report(self, adcp_dataset):
        """Test get_pipeline_report returns QCPipelineReport."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.roll_check().pitch_check()

        report = runner.get_pipeline_report()
        assert isinstance(report, QCPipelineReport)
        assert len(report.checks) == 2
        assert report.total_cells == runner.total_cells
        assert report.baseline_masked == runner.baseline_masked

    def test_export_statistics_dict(self, adcp_dataset):
        """Test export_statistics_dict returns serializable dict."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.roll_check(threshold=20.0).pitch_check(threshold=15.0)

        export = runner.export_statistics_dict()

        # Check structure
        assert "baseline" in export
        assert "checks" in export
        assert "summary" in export
        assert "timestamp" in export

        # Check baseline
        assert export["baseline"]["total_cells"] == runner.total_cells
        assert export["baseline"]["masked_cells"] == runner.baseline_masked

        # Check checks list
        assert len(export["checks"]) == 2
        assert export["checks"][0]["check_name"] == "Roll Check"
        assert export["checks"][0]["threshold"] == 20.0
        assert export["checks"][1]["check_name"] == "Pitch Check"

        # Check summary
        assert export["summary"]["checks_applied"] == 2

    def test_export_statistics_dict_empty(self, adcp_dataset):
        """Test export_statistics_dict with no checks."""
        runner = SensorHealthRunner(adcp_dataset)
        export = runner.export_statistics_dict()

        assert len(export["checks"]) == 0
        assert export["summary"]["checks_applied"] == 0


# ============================================================================
# TESTS: Statistics Tracking Accuracy
# ============================================================================


class TestStatisticsAccuracy:
    """Test that statistics are accurately tracked."""

    def test_baseline_calculated_correctly(self, adcp_dataset):
        """Test baseline is calculated correctly on init."""
        # Dataset has all zeros mask
        runner = SensorHealthRunner(adcp_dataset)
        assert runner.baseline_masked == 0
        assert runner.baseline_masked_pct == 0.0
        assert runner.total_cells == 4 * 20 * 50  # beam * cell * time

    def test_statistics_track_incremental_masking(self, adcp_dataset_with_high_roll):
        """Test that statistics correctly track incremental masking."""
        runner = SensorHealthRunner(adcp_dataset_with_high_roll)

        # First check
        runner.roll_check(threshold=20.0)
        first_stat = runner.statistics[0]
        assert first_stat.cells_pre_masked == 0
        assert first_stat.cells_newly_masked > 0

        # Second check (should have cells_pre_masked > 0)
        runner.pitch_check(threshold=15.0)
        second_stat = runner.statistics[1]
        assert second_stat.cells_pre_masked == first_stat.cells_total_masked

    def test_statistics_cleared_on_reset(self, adcp_dataset):
        """Test that statistics are cleared on reset."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.roll_check().pitch_check()
        assert len(runner.statistics) == 2

        runner.reset()
        assert len(runner.statistics) == 0


# ============================================================================
# TESTS: Init Branches (lines 542-548)
# ============================================================================


class TestInitBranches:
    """Test SensorHealthRunner __init__ baseline branches."""

    def test_init_with_velocity_only_no_mask(self):
        """Init with velocity but no mask uses velocity size as total_cells (line 542-545)."""
        times = pd.date_range("2024-01-01", periods=10, freq="h")
        ds = xr.Dataset(
            {
                "velocity": (
                    ("beam", "cell", "time"),
                    np.random.randn(4, 5, 10),
                ),
                "roll": (("time",), np.zeros(10)),
                "pitch": (("time",), np.zeros(10)),
            },
            coords={"time": times, "beam": np.arange(4), "cell": np.arange(5)},
        )

        runner = SensorHealthRunner(ds)

        assert runner.total_cells == 4 * 5 * 10
        assert runner.baseline_masked == 0
        assert runner.baseline_masked_pct == 0.0

    def test_init_with_neither_mask_nor_velocity(self):
        """Init with no mask and no velocity sets totals to zero (line 546-548)."""
        times = pd.date_range("2024-01-01", periods=10, freq="h")
        ds = xr.Dataset(
            {
                "roll": (("time",), np.zeros(10)),
                "pitch": (("time",), np.zeros(10)),
            },
            coords={"time": times},
        )

        runner = SensorHealthRunner(ds)

        assert runner.total_cells == 0
        assert runner.baseline_masked == 0
        assert runner.baseline_masked_pct == 0.0


# ============================================================================
# TESTS: _compute_var_stats all-NaN branch (line 583)
# ============================================================================


class TestComputeVarStats:
    """Test _compute_var_stats helper."""

    def test_all_nan_returns_nan_dict(self, adcp_dataset):
        """All-NaN input triggers the early-return nan dict (line 583)."""
        runner = SensorHealthRunner(adcp_dataset)
        all_nan = xr.DataArray(np.full((4, 5), np.nan))

        result = runner._compute_var_stats(all_nan)

        assert np.isnan(result["min"])
        assert np.isnan(result["max"])
        assert np.isnan(result["mean"])
        assert np.isnan(result["std"])

    def test_valid_values_returns_stats(self, adcp_dataset):
        """Normal input returns correct statistics."""
        runner = SensorHealthRunner(adcp_dataset)
        da = xr.DataArray(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        result = runner._compute_var_stats(da)

        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["mean"] == 3.0


# ============================================================================
# TESTS: get_modifications method
# ============================================================================


class TestGetModifications:
    """Test get_modifications method."""

    def test_get_modifications_empty(self, adcp_dataset):
        """get_modifications returns empty dict when no replacements made."""
        runner = SensorHealthRunner(adcp_dataset)
        mods = runner.get_modifications()
        assert mods == {}

    def test_get_modifications_single_variable(self, adcp_dataset):
        """get_modifications returns correct structure after one replacement."""
        runner = SensorHealthRunner(adcp_dataset)
        new_temp = np.ones(50) * 20.0
        runner.replace_data(new_temp, "temperature")

        mods = runner.get_modifications()

        assert "temperature" in mods
        assert isinstance(mods["temperature"], list)
        assert len(mods["temperature"]) == 1
        assert isinstance(mods["temperature"][0], DataModificationStats)

    def test_get_modifications_multiple_variables(self, adcp_dataset):
        """get_modifications groups by variable name correctly."""
        runner = SensorHealthRunner(adcp_dataset)
        new_temp = np.ones(50) * 20.0
        new_sal = np.ones(50) * 36.0
        runner.replace_data(new_temp, "temperature")
        runner.replace_data(new_sal, "salinity")

        mods = runner.get_modifications()

        assert "temperature" in mods
        assert "salinity" in mods
        assert len(mods["temperature"]) == 1
        assert len(mods["salinity"]) == 1

    def test_get_modifications_same_variable_twice(self, adcp_dataset):
        """get_modifications lists multiple replacements of the same variable."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.replace_data(np.ones(50) * 20.0, "temperature")
        runner.replace_data(np.ones(50) * 22.0, "temperature")

        mods = runner.get_modifications()

        assert "temperature" in mods
        assert len(mods["temperature"]) == 2

    def test_get_modifications_cleared_on_reset(self, adcp_dataset):
        """get_modifications returns empty after reset."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.replace_data(np.ones(50) * 20.0, "temperature")
        assert len(runner.get_modifications()) == 1

        runner.reset()
        assert runner.get_modifications() == {}


# ============================================================================
# TESTS: print_statistics with modifications (line 984 branch)
# ============================================================================


class TestPrintStatisticsWithModifications:
    """Test print_statistics DATA MODIFICATIONS branch."""

    def test_print_statistics_with_modifications(self, adcp_dataset, capsys):
        """print_statistics prints DATA MODIFICATIONS block when replacements exist."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.replace_data(np.ones(50) * 20.0, "temperature")
        runner.roll_check()
        runner.print_statistics()

        captured = capsys.readouterr()
        assert "DATA MODIFICATIONS" in captured.out
        assert "replace_data" in captured.out
        assert "temperature" in captured.out

    def test_print_statistics_modifications_no_checks(self, adcp_dataset, capsys):
        """print_statistics shows modifications and 'No QC checks' when only replace called."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.replace_data(np.ones(50) * 20.0, "temperature")
        runner.print_statistics()

        captured = capsys.readouterr()
        assert "DATA MODIFICATIONS" in captured.out
        assert "No QC checks applied yet" in captured.out

    def test_print_statistics_threshold_non_numeric(self, adcp_dataset, capsys):
        """print_statistics handles non-numeric threshold in QCCheckStats (str branch)."""
        runner = SensorHealthRunner(adcp_dataset)
        runner.roll_check()

        # Inject a stat with a string threshold to exercise the else branch
        runner.statistics[0] = QCCheckStats(
            check_name="Roll Check",
            threshold="custom",
            cells_pre_masked=runner.statistics[0].cells_pre_masked,
            cells_newly_masked=runner.statistics[0].cells_newly_masked,
            cells_total_masked=runner.statistics[0].cells_total_masked,
            total_cells=runner.statistics[0].total_cells,
        )
        runner.print_statistics()

        captured = capsys.readouterr()
        assert "custom" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
