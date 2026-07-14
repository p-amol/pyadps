"""
Test suite for SignalQualityRunner class.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyadps.processing.signal_quality import (
    SignalQualityRunner,
)
from pyadps.processing.utility import QCCheckStats, QCPipelineReport


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def adcp_dataset():
    """Create ADCP dataset with all required variables for signal quality tests."""
    np.random.seed(42)
    n_time = 50
    n_cell = 20
    n_beam = 4

    times = pd.date_range("2024-01-01 00:00:00", periods=n_time, freq="h")
    cells = np.arange(n_cell)
    beams = np.arange(n_beam)

    # Create velocity data (beam 3 is error velocity)
    velocity = np.random.randint(-500, 500, size=(n_beam, n_cell, n_time))
    velocity[0, 5, 10] = -32768  # Missing value

    # Create correlation data (0-255 scale)
    correlation = np.random.randint(50, 200, size=(n_beam, n_cell, n_time))
    # Add some low correlation values
    correlation[0, :5, :10] = 30  # Below default threshold of 64

    # Create echo intensity data (0-255 scale)
    echo_intensity = np.random.randint(30, 150, size=(n_beam, n_cell, n_time))
    # Add some low echo values
    echo_intensity[1, :3, :5] = 20  # Below default threshold of 40

    # Create percent good data (0-100 scale)
    percent_good = np.random.randint(40, 100, size=(n_beam, n_cell, n_time))
    # Add some low percent good values
    # With threebeam=True (default), PG1 + PG4 is used
    # Set PG1 (beam 0) = 20 and PG4 (beam 3) = 20, sum = 40 < 50
    percent_good[0, :2, :3] = 20  # PG1
    percent_good[3, :2, :3] = 20  # PG4

    data_vars = {
        "velocity": (("beam", "cell", "time"), velocity),
        "correlation": (("beam", "cell", "time"), correlation),
        "echo_intensity": (("beam", "cell", "time"), echo_intensity),
        "percent_good": (("beam", "cell", "time"), percent_good),
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

    # Add attributes
    ds["velocity"].attrs = {"long_name": "Velocity", "units": "mm/s"}
    ds["correlation"].attrs = {"long_name": "Correlation", "units": "counts"}
    ds["echo_intensity"].attrs = {"long_name": "Echo Intensity", "units": "counts"}
    ds["percent_good"].attrs = {"long_name": "Percent Good", "units": "%"}

    return ds


@pytest.fixture
def adcp_dataset_high_error_velocity():
    """Create dataset with high error velocity values."""
    np.random.seed(42)
    n_time = 50
    n_cell = 20
    n_beam = 4

    times = pd.date_range("2024-01-01 00:00:00", periods=n_time, freq="h")

    velocity = np.random.randint(-500, 500, size=(n_beam, n_cell, n_time))
    # Set high error velocity (beam 3) for first 10 ensembles
    velocity[3, :, :10] = 3000  # Above default threshold of 2000

    data_vars = {
        "velocity": (("beam", "cell", "time"), velocity),
        "correlation": (
            ("beam", "cell", "time"),
            np.random.randint(100, 200, size=(n_beam, n_cell, n_time)),
        ),
        "echo_intensity": (
            ("beam", "cell", "time"),
            np.random.randint(80, 150, size=(n_beam, n_cell, n_time)),
        ),
        "percent_good": (
            ("beam", "cell", "time"),
            np.random.randint(70, 100, size=(n_beam, n_cell, n_time)),
        ),
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

    return xr.Dataset(data_vars, coords=coords)


# ============================================================================
# TESTS: Initialization
# ============================================================================


class TestSignalQualityRunnerInit:
    """Test SignalQualityRunner initialization."""

    def test_initialization(self, adcp_dataset):
        """Test basic initialization."""
        runner = SignalQualityRunner(adcp_dataset)
        assert isinstance(runner.dataset, xr.Dataset)
        assert isinstance(runner.original, xr.Dataset)
        assert isinstance(runner.history, list)
        assert isinstance(runner.statistics, list)
        assert isinstance(runner.modifications, list)
        assert len(runner.history) == 0
        assert len(runner.statistics) == 0
        assert len(runner.modifications) == 0

    def test_original_preserved(self, adcp_dataset):
        """Test that original dataset is preserved as a copy."""
        runner = SignalQualityRunner(adcp_dataset)
        # Modify the working dataset
        runner.dataset["correlation"].values[:] = 0
        # Original should be unchanged
        assert runner.original["correlation"].values.mean() != 0

    def test_baseline_calculated(self, adcp_dataset):
        """Test that baseline statistics are calculated."""
        runner = SignalQualityRunner(adcp_dataset)
        assert runner.total_cells > 0
        assert runner.baseline_masked >= 0
        assert 0 <= runner.baseline_masked_pct <= 100

    def test_mask_created_if_missing(self):
        """Test that mask is created if not present."""
        ds = xr.Dataset(
            {
                "velocity": (
                    ["beam", "cell", "time"],
                    np.random.randn(4, 10, 20),
                ),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(10),
                "time": pd.date_range("2024-01-01", periods=20, freq="h"),
            },
        )
        runner = SignalQualityRunner(ds)
        assert "mask" in runner.dataset.data_vars


# ============================================================================
# TESTS: Method Chaining
# ============================================================================


class TestMethodChaining:
    """Test method chaining functionality."""

    def test_correlation_returns_self(self, adcp_dataset):
        """Test that correlation returns self."""
        runner = SignalQualityRunner(adcp_dataset)
        result = runner.correlation(cutoff=64)
        assert result is runner

    def test_echo_intensity_returns_self(self, adcp_dataset):
        """Test that echo_intensity returns self."""
        runner = SignalQualityRunner(adcp_dataset)
        result = runner.echo_intensity(cutoff=40)
        assert result is runner

    def test_error_velocity_returns_self(self, adcp_dataset):
        """Test that error_velocity returns self."""
        runner = SignalQualityRunner(adcp_dataset)
        result = runner.error_velocity(cutoff=2000)
        assert result is runner

    def test_percent_good_returns_self(self, adcp_dataset):
        """Test that percent_good returns self."""
        runner = SignalQualityRunner(adcp_dataset)
        result = runner.percent_good(cutoff=50)
        assert result is runner

    def test_false_target_returns_self(self, adcp_dataset):
        """Test that false_target returns self."""
        runner = SignalQualityRunner(adcp_dataset)
        result = runner.false_target(cutoff=50)
        assert result is runner

    def test_percent_good_with_threebeam(self, adcp_dataset):
        """Test percent_good with threebeam parameter."""
        runner = SignalQualityRunner(adcp_dataset)
        result = runner.percent_good(cutoff=50, threebeam=True)
        assert result is runner

    def test_percent_good_with_method(self, adcp_dataset):
        """Test percent_good with method parameter."""
        runner = SignalQualityRunner(adcp_dataset)
        result = runner.percent_good(cutoff=50, method="max")
        assert result is runner

    def test_false_target_with_beam_ignore(self, adcp_dataset):
        """Test false_target with beam_ignore parameter."""
        runner = SignalQualityRunner(adcp_dataset)
        result = runner.false_target(cutoff=50, beam_ignore=0)
        assert result is runner

    def test_reset_returns_self(self, adcp_dataset):
        """Test that reset returns self."""
        runner = SignalQualityRunner(adcp_dataset)
        result = runner.reset()
        assert result is runner

    def test_chained_operations(self, adcp_dataset):
        """Test multiple chained operations."""
        runner = SignalQualityRunner(adcp_dataset)
        result = (
            runner.correlation(cutoff=64)
            .echo_intensity(cutoff=40)
            .error_velocity(cutoff=2000)
            .percent_good(cutoff=50)
            .finalize()
        )
        assert isinstance(result, xr.Dataset)

    def test_full_workflow(self, adcp_dataset):
        """Test complete workflow with all operations."""
        runner = SignalQualityRunner(adcp_dataset)
        result = (
            runner.correlation(cutoff=64)
            .echo_intensity(cutoff=40)
            .error_velocity(cutoff=2000)
            .percent_good(cutoff=50)
            .false_target(cutoff=50)
            .finalize()
        )
        assert isinstance(result, xr.Dataset)
        assert len(runner.statistics) == 5


# ============================================================================
# TESTS: Processing History
# ============================================================================


class TestProcessingHistory:
    """Test processing history tracking."""

    def test_history_after_correlation(self, adcp_dataset):
        """Test history entry after correlation check."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.correlation(cutoff=64)

        assert len(runner.history) == 1
        entry = runner.history[0]
        assert entry["operation"] == "correlation"
        assert entry["parameters"]["cutoff"] == 64
        assert "timestamp" in entry

    def test_history_accumulates(self, adcp_dataset):
        """Test that history accumulates across operations."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.correlation().echo_intensity().error_velocity()

        assert len(runner.history) == 3
        assert runner.history[0]["operation"] == "correlation"
        assert runner.history[1]["operation"] == "echo_intensity"
        assert runner.history[2]["operation"] == "error_velocity"

    def test_history_cleared_on_reset(self, adcp_dataset):
        """Test that history is cleared on reset."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.correlation().echo_intensity()
        assert len(runner.history) == 2

        runner.reset()
        assert len(runner.history) == 0

    def test_history_records_threebeam_params(self, adcp_dataset):
        """Test that threebeam parameters are recorded in history."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.percent_good(cutoff=50, threebeam=False)

        assert len(runner.history) == 1
        entry = runner.history[0]
        assert entry["parameters"]["threebeam"] is False

    def test_history_records_beam_ignore(self, adcp_dataset):
        """Test that beam_ignore is recorded in history."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.false_target(cutoff=50, beam_ignore=2)

        assert len(runner.history) == 1
        entry = runner.history[0]
        assert entry["parameters"]["beam_ignore"] == 2

    def test_history_records_method_param(self, adcp_dataset):
        """Test that method parameter is recorded in history."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.percent_good(cutoff=50, method="max")

        assert len(runner.history) == 1
        entry = runner.history[0]
        assert entry["parameters"]["method"] == "max"


# ============================================================================
# TESTS: Statistics Tracking
# ============================================================================


class TestStatisticsTracking:
    """Test statistics tracking."""

    def test_statistics_recorded(self, adcp_dataset):
        """Test that statistics are recorded for each check."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.correlation(cutoff=64)

        assert len(runner.statistics) == 1
        stat = runner.statistics[0]
        assert isinstance(stat, QCCheckStats)
        assert stat.check_name == "Correlation"
        assert stat.threshold == 64

    def test_statistics_accumulate(self, adcp_dataset):
        """Test that statistics accumulate."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.correlation().echo_intensity().error_velocity()

        assert len(runner.statistics) == 3
        assert runner.statistics[0].check_name == "Correlation"
        assert runner.statistics[1].check_name == "Echo Intensity"
        assert runner.statistics[2].check_name == "Error Velocity"

    def test_statistics_cleared_on_reset(self, adcp_dataset):
        """Test that statistics are cleared on reset."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.correlation().echo_intensity()
        assert len(runner.statistics) == 2

        runner.reset()
        assert len(runner.statistics) == 0

    def test_get_statistics_returns_dict(self, adcp_dataset):
        """Test get_statistics returns dictionary."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.correlation().echo_intensity()

        stats = runner.get_statistics()
        assert isinstance(stats, dict)
        assert "Correlation" in stats
        assert "Echo Intensity" in stats


# ============================================================================
# TESTS: Pipeline
# ============================================================================


class TestPipeline:
    """Test apply_pipeline functionality."""

    def test_apply_pipeline_defaults(self, adcp_dataset):
        """Test apply_pipeline with default settings."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.apply_pipeline()

        assert len(runner.statistics) == 5
        check_names = [s.check_name for s in runner.statistics]
        assert "Correlation" in check_names
        assert "Echo Intensity" in check_names
        assert "Error Velocity" in check_names
        assert "Percent Good" in check_names
        assert "False Target" in check_names

    def test_apply_pipeline_custom_thresholds(self, adcp_dataset):
        """Test apply_pipeline with custom thresholds."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.apply_pipeline(
            checks={"correlation": 50, "echo_intensity": 30},
            order=["correlation", "echo_intensity"],
        )

        assert len(runner.statistics) == 2
        assert runner.statistics[0].threshold == 50
        assert runner.statistics[1].threshold == 30

    def test_apply_pipeline_custom_order(self, adcp_dataset):
        """Test apply_pipeline with custom order."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.apply_pipeline(
            checks={
                "echo_intensity": 40,
                "correlation": 64,
            },
            order=["echo_intensity", "correlation"],
        )

        assert runner.statistics[0].check_name == "Echo Intensity"
        assert runner.statistics[1].check_name == "Correlation"


# ============================================================================
# TESTS: Reset
# ============================================================================


class TestReset:
    """Test reset functionality."""

    def test_reset_restores_dataset(self, adcp_dataset):
        """Test that reset restores original dataset."""
        runner = SignalQualityRunner(adcp_dataset)
        original_mask_sum = runner.dataset["mask"].sum().values

        runner.correlation(cutoff=64)
        modified_mask_sum = runner.dataset["mask"].sum().values
        assert modified_mask_sum >= original_mask_sum

        runner.reset()
        reset_mask_sum = runner.dataset["mask"].sum().values
        assert reset_mask_sum == original_mask_sum

    def test_reset_clears_all_state(self, adcp_dataset):
        """Test that reset clears all state."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.correlation().echo_intensity()

        runner.reset()
        assert len(runner.history) == 0
        assert len(runner.statistics) == 0
        assert len(runner.modifications) == 0


# ============================================================================
# TESTS: Finalize
# ============================================================================


class TestFinalize:
    """Test finalize functionality."""

    def test_finalize_returns_dataset(self, adcp_dataset):
        """Test that finalize returns a dataset."""
        runner = SignalQualityRunner(adcp_dataset)
        result = runner.correlation().finalize()
        assert isinstance(result, xr.Dataset)

    def test_finalize_adds_history_attribute(self, adcp_dataset):
        """Test that finalize adds history to attributes."""
        runner = SignalQualityRunner(adcp_dataset)
        result = runner.correlation().finalize()
        assert "signal_quality_processing_history" in result.attrs
        assert "signal_quality_processed_at" in result.attrs

    def test_finalize_preserves_data(self, adcp_dataset):
        """Test that finalize preserves processed data."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.correlation(cutoff=64)
        mask_before = runner.dataset["mask"].values.copy()

        result = runner.finalize()
        np.testing.assert_array_equal(result["mask"].values, mask_before)


# ============================================================================
# TESTS: Reporting
# ============================================================================


class TestReporting:
    """Test reporting functionality."""

    def test_get_pipeline_report(self, adcp_dataset):
        """Test get_pipeline_report returns QCPipelineReport."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.correlation()

        report = runner.get_pipeline_report()
        assert isinstance(report, QCPipelineReport)
        assert report.module_name == "signal_quality"
        assert len(report.checks) == 1

    def test_get_report_returns_string(self, adcp_dataset):
        """Test get_report returns string."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.correlation()

        report = runner.get_report()
        assert isinstance(report, str)
        assert "signal_quality" in report.lower()

    def test_export_statistics_dict(self, adcp_dataset):
        """Test export_statistics_dict returns dict."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.correlation()

        export = runner.export_statistics_dict()
        assert isinstance(export, dict)
        assert "module_name" in export
        assert "checks" in export

    def test_print_statistics_no_error(self, adcp_dataset, capsys):
        """Test print_statistics runs without error."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.correlation().echo_intensity()

        runner.print_statistics()
        captured = capsys.readouterr()
        assert "SIGNAL QUALITY PROCESSING STATISTICS" in captured.out
        assert "Correlation" in captured.out
        assert "Echo Intensity" in captured.out

    def test_print_statistics_empty(self, adcp_dataset, capsys):
        """Test print_statistics with no checks."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.print_statistics()

        captured = capsys.readouterr()
        assert "No QC checks applied yet" in captured.out

    def test_summary_returns_string(self, adcp_dataset):
        """Test summary returns string."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.correlation()

        summary = runner.summary()
        assert isinstance(summary, str)
        assert "correlation" in summary.lower()


# ============================================================================
# TESTS: Original Preservation
# ============================================================================


class TestOriginalPreservation:
    """Test that original dataset is never modified."""

    def test_correlation_preserves_original(self, adcp_dataset):
        """Test that correlation doesn't modify original."""
        runner = SignalQualityRunner(adcp_dataset)
        original_mask = runner.original["mask"].values.copy()

        runner.correlation(cutoff=64)

        np.testing.assert_array_equal(runner.original["mask"].values, original_mask)

    def test_all_checks_preserve_original(self, adcp_dataset):
        """Test that all checks preserve original."""
        runner = SignalQualityRunner(adcp_dataset)
        original_mask = runner.original["mask"].values.copy()

        runner.correlation().echo_intensity().error_velocity().percent_good()

        np.testing.assert_array_equal(runner.original["mask"].values, original_mask)


# ============================================================================
# TESTS: Threebeam and Beam Ignore Functionality
# ============================================================================


class TestThreebeamAndBeamIgnore:
    """Test threebeam and beam_ignore parameter functionality."""

    def test_percent_good_threebeam_default(self, adcp_dataset):
        """Test percent_good uses threebeam=True by default."""
        runner = SignalQualityRunner(adcp_dataset)

        # Set specific values for PG1 and PG4
        runner.dataset["percent_good"].values[0, 0, 0] = 30  # PG1
        runner.dataset["percent_good"].values[3, 0, 0] = 30  # PG4
        # Sum = 60 > 50, should NOT flag with default threebeam=True

        runner.percent_good(cutoff=50)

        assert runner.dataset["mask"].isel(beam=3, cell=0, time=0).values == 0

    def test_percent_good_threebeam_false(self, adcp_dataset):
        """Test percent_good with threebeam=False uses only PG4."""
        runner = SignalQualityRunner(adcp_dataset)

        # Set PG4 low, PG1 high
        runner.dataset["percent_good"].values[0, 0, 0] = 80  # PG1 (high)
        runner.dataset["percent_good"].values[3, 0, 0] = 30  # PG4 (low)

        runner.percent_good(cutoff=50, threebeam=False)

        # Uses only PG4 = 30 < 50, should flag
        assert runner.dataset["mask"].isel(beam=3, cell=0, time=0).values == 1

    def test_false_target_uses_max_minus_min(self, adcp_dataset):
        """Test false_target always uses max-min."""
        runner = SignalQualityRunner(adcp_dataset)

        # Set values
        runner.dataset["echo_intensity"].values[:, 0, 0] = [150, 100, 95, 90]
        # Max=150, min=90, diff=60

        runner.false_target(cutoff=55)

        # diff=60 > 55, should flag
        assert runner.dataset["mask"].isel(beam=3, cell=0, time=0).values == 1

    def test_false_target_beam_ignore(self, adcp_dataset):
        """Test false_target with beam_ignore excludes beam."""
        runner = SignalQualityRunner(adcp_dataset)

        # Set outlier on beam 0
        runner.dataset["echo_intensity"].values[:, 0, 0] = [200, 80, 85, 90]
        # Without ignore: max=200, min=80, diff=120
        # With beam 0 ignored: max=90, min=80, diff=10

        runner.false_target(cutoff=50, beam_ignore=0)

        # With beam_ignore=0, diff=10 < 50, should NOT flag
        assert runner.dataset["mask"].isel(beam=3, cell=0, time=0).values == 0

    def test_statistics_record_threebeam_metadata(self, adcp_dataset):
        """Test that statistics record threebeam in metadata."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.percent_good(cutoff=50, threebeam=False)

        stat = runner.statistics[0]
        assert "threebeam" in stat.metadata
        assert stat.metadata["threebeam"] is False

    def test_statistics_record_beam_ignore_metadata(self, adcp_dataset):
        """Test that statistics record beam_ignore in metadata."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.false_target(cutoff=50, beam_ignore=2)

        stat = runner.statistics[0]
        assert "beam_ignore" in stat.metadata
        assert stat.metadata["beam_ignore"] == 2

    def test_chained_with_threebeam_params(self, adcp_dataset):
        """Test method chaining with threebeam parameters."""
        runner = SignalQualityRunner(adcp_dataset)
        result = (
            runner.correlation(cutoff=64, beam_ignore=1)
            .echo_intensity(cutoff=40, beam_ignore=1)
            .percent_good(cutoff=50, threebeam=True)
            .false_target(cutoff=50, beam_ignore=0)
            .finalize()
        )
        assert isinstance(result, xr.Dataset)
        assert len(runner.statistics) == 4


# ============================================================================
# TESTS: Coverage for specific uncovered lines
# ============================================================================


class TestResetWithoutMaskInOriginal:
    """Line 882: reset() calls create_default_mask when original has no mask."""

    def test_reset_creates_mask_when_original_has_none(self):
        """Line 882: when original dataset has no mask, reset creates one."""
        times = pd.date_range("2024-01-01", periods=10, freq="h")
        ds_no_mask = xr.Dataset(
            {
                "velocity": (
                    ("beam", "cell", "time"),
                    np.random.randn(4, 5, 10),
                ),
                "correlation": (
                    ("beam", "cell", "time"),
                    np.random.randint(50, 200, (4, 5, 10)),
                ),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(5),
                "time": times,
            },
        )

        # __init__ creates mask automatically; original still has no mask
        runner = SignalQualityRunner(ds_no_mask)
        assert "mask" not in runner.original.data_vars  # original untouched

        runner.correlation(cutoff=64)

        # reset() should hit line 882 and recreate the mask
        runner.reset()
        assert "mask" in runner.dataset.data_vars


class TestGetModifications:
    """get_modifications method - never called in existing tests."""

    def test_get_modifications_empty(self, adcp_dataset):
        """get_modifications returns empty dict when no modifications made."""
        runner = SignalQualityRunner(adcp_dataset)
        mods = runner.get_modifications()
        assert mods == {}

    def test_get_modifications_after_manual_append(self, adcp_dataset):
        """get_modifications groups DataModificationStats by variable name."""
        from pyadps.processing.utility import DataModificationStats

        runner = SignalQualityRunner(adcp_dataset)

        # Manually inject modification stats (the runner doesn't expose a
        # public modify method, so we inject directly as the class allows)
        mod1 = DataModificationStats(
            operation="test_op",
            variable_name="correlation",
            original_stats={"mean": 100.0},
            modified_stats={"mean": 80.0},
        )
        mod2 = DataModificationStats(
            operation="test_op2",
            variable_name="correlation",
            original_stats={"mean": 80.0},
            modified_stats={"mean": 60.0},
        )
        runner.modifications.append(mod1)
        runner.modifications.append(mod2)

        mods = runner.get_modifications()

        assert "correlation" in mods
        assert len(mods["correlation"]) == 2

    def test_get_modifications_multiple_variables(self, adcp_dataset):
        """get_modifications groups correctly for multiple variables."""
        from pyadps.processing.utility import DataModificationStats

        runner = SignalQualityRunner(adcp_dataset)
        runner.modifications.append(
            DataModificationStats(
                operation="op1",
                variable_name="correlation",
                original_stats={"mean": 100.0},
                modified_stats={"mean": 80.0},
            )
        )
        runner.modifications.append(
            DataModificationStats(
                operation="op2",
                variable_name="echo_intensity",
                original_stats={"mean": 90.0},
                modified_stats={"mean": 70.0},
            )
        )

        mods = runner.get_modifications()
        assert "correlation" in mods
        assert "echo_intensity" in mods
        assert len(mods["correlation"]) == 1
        assert len(mods["echo_intensity"]) == 1


class TestPrintStatisticsWithModifications:
    """print_statistics DATA MODIFICATIONS branch (line 991)."""

    def test_print_statistics_with_modifications(self, adcp_dataset, capsys):
        """Modifications block is printed when self.modifications is non-empty."""
        from pyadps.processing.utility import DataModificationStats

        runner = SignalQualityRunner(adcp_dataset)
        runner.modifications.append(
            DataModificationStats(
                operation="replace_data",
                variable_name="correlation",
                original_stats={"mean": 100.0},
                modified_stats={"mean": 80.0},
            )
        )
        runner.correlation()
        runner.print_statistics()

        captured = capsys.readouterr()
        assert "DATA MODIFICATIONS" in captured.out
        assert "replace_data" in captured.out
        assert "correlation" in captured.out

    def test_print_statistics_modifications_no_checks(self, adcp_dataset, capsys):
        """Modifications block printed alongside 'No QC checks' message."""
        from pyadps.processing.utility import DataModificationStats

        runner = SignalQualityRunner(adcp_dataset)
        runner.modifications.append(
            DataModificationStats(
                operation="replace_data",
                variable_name="echo_intensity",
                original_stats={"mean": 90.0},
                modified_stats={"mean": 70.0},
            )
        )
        runner.print_statistics()

        captured = capsys.readouterr()
        assert "DATA MODIFICATIONS" in captured.out
        assert "No QC checks applied yet" in captured.out


class TestSummaryNoHistory:
    """Line 1061: summary() with no processing history."""

    def test_summary_empty_history(self, adcp_dataset):
        """summary() returns 'No processing steps applied yet.' when history is empty."""
        runner = SignalQualityRunner(adcp_dataset)
        result = runner.summary()

        assert "No processing steps applied yet." in result

    def test_summary_after_checks_not_empty(self, adcp_dataset):
        """summary() after checks does NOT include the empty message."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.correlation()
        result = runner.summary()

        assert "No processing steps applied yet." not in result
        assert "correlation" in result.lower()


class TestApplyPipelineContinue:
    """Line 855: apply_pipeline skips names in order that are not in checks dict."""

    def test_apply_pipeline_skips_missing_checks(self, adcp_dataset):
        """Order list contains a name absent from checks -> continue (line 855)."""
        runner = SignalQualityRunner(adcp_dataset)

        # Only provide correlation, but order includes echo_intensity too
        # echo_intensity is not in checks -> hits 'continue' on line 855
        runner.apply_pipeline(
            checks={"correlation": 64},
            order=["correlation", "echo_intensity"],
        )

        # Only correlation should have been applied
        stats = runner.get_statistics()
        assert "Correlation" in stats
        assert "Echo Intensity" not in stats


class TestPrintStatisticsThresholdFormatting:
    """Lines 1025 and 1027: threshold formatting branches in print_statistics."""

    def test_print_statistics_tuple_threshold(self, adcp_dataset, capsys):
        """Line 1025: threshold formatted as tuple string when stat.threshold is a tuple."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.correlation()

        # Inject a stat with a tuple threshold
        runner.statistics[0] = QCCheckStats(
            check_name="Correlation",
            threshold=(50, 100),
            cells_pre_masked=runner.statistics[0].cells_pre_masked,
            cells_newly_masked=runner.statistics[0].cells_newly_masked,
            cells_total_masked=runner.statistics[0].cells_total_masked,
            total_cells=runner.statistics[0].total_cells,
        )
        runner.print_statistics()

        captured = capsys.readouterr()
        assert "(50, 100)" in captured.out

    def test_print_statistics_none_threshold(self, adcp_dataset, capsys):
        """Line 1027: threshold formatted as 'N/A' when stat.threshold is None."""
        runner = SignalQualityRunner(adcp_dataset)
        runner.correlation()

        # Inject a stat with None threshold
        runner.statistics[0] = QCCheckStats(
            check_name="Correlation",
            threshold=None,
            cells_pre_masked=runner.statistics[0].cells_pre_masked,
            cells_newly_masked=runner.statistics[0].cells_newly_masked,
            cells_total_masked=runner.statistics[0].cells_total_masked,
            total_cells=runner.statistics[0].total_cells,
        )
        runner.print_statistics()

        captured = capsys.readouterr()
        assert "N/A" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
