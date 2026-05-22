"""
Test suite for QCPipelineReport dataclass.
"""

import pytest
from datetime import datetime, timezone

from pyadps.processing.utility import (
    QCCheckStats,
    DataModificationStats,
    QCPipelineReport,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def qc_check_stat():
    """Create a sample QCCheckStats object."""
    return QCCheckStats(
        check_name="Roll Check",
        threshold=15.0,
        cells_pre_masked=100,
        cells_newly_masked=50,
        cells_total_masked=150,
        total_cells=1000,
    )


@pytest.fixture
def data_mod_stat():
    """Create a sample DataModificationStats object."""
    return DataModificationStats(
        operation="replace_data",
        variable_name="temperature",
        original_stats={"min": 10.0, "max": 30.0, "mean": 20.0, "std": 5.0},
        modified_stats={"min": 12.0, "max": 28.0, "mean": 22.0, "std": 4.0},
    )


@pytest.fixture
def empty_report():
    """Create an empty report with no checks or modifications."""
    return QCPipelineReport(
        module_name="sensor_health",
        baseline_masked=100,
        baseline_masked_pct=10.0,
        total_cells=1000,
    )


@pytest.fixture
def report_with_checks(qc_check_stat):
    """Create a report with checks."""
    return QCPipelineReport(
        module_name="sensor_health",
        baseline_masked=100,
        baseline_masked_pct=10.0,
        total_cells=1000,
        checks=[qc_check_stat],
    )


@pytest.fixture
def full_report(qc_check_stat, data_mod_stat):
    """Create a report with checks and modifications."""
    return QCPipelineReport(
        module_name="sensor_health",
        baseline_masked=100,
        baseline_masked_pct=10.0,
        total_cells=1000,
        checks=[qc_check_stat],
        modifications=[data_mod_stat],
    )


# ============================================================================
# TESTS: QCPipelineReport
# ============================================================================


class TestQCPipelineReportCreation:
    """Test QCPipelineReport creation and basic attributes."""

    def test_basic_creation(self, empty_report):
        """Test basic report creation."""
        assert empty_report.module_name == "sensor_health"
        assert empty_report.baseline_masked == 100
        assert empty_report.baseline_masked_pct == 10.0
        assert empty_report.total_cells == 1000
        assert len(empty_report.checks) == 0
        assert len(empty_report.modifications) == 0

    def test_timestamp_auto_generated(self, empty_report):
        """Test that timestamp is automatically generated."""
        assert isinstance(empty_report.timestamp, datetime)
        assert empty_report.timestamp.tzinfo == timezone.utc

    def test_with_checks(self, report_with_checks):
        """Test creation with checks."""
        assert len(report_with_checks.checks) == 1
        assert report_with_checks.checks[0].check_name == "Roll Check"

    def test_with_modifications(self, full_report):
        """Test creation with modifications."""
        assert len(full_report.modifications) == 1
        assert full_report.modifications[0].variable_name == "temperature"

    def test_multiple_checks(self):
        """Test report with multiple checks."""
        checks = [
            QCCheckStats(
                check_name="Roll Check",
                threshold=15.0,
                cells_pre_masked=0,
                cells_newly_masked=50,
                cells_total_masked=50,
                total_cells=1000,
            ),
            QCCheckStats(
                check_name="Pitch Check",
                threshold=15.0,
                cells_pre_masked=50,
                cells_newly_masked=30,
                cells_total_masked=80,
                total_cells=1000,
            ),
        ]
        report = QCPipelineReport(
            module_name="sensor_health",
            baseline_masked=0,
            baseline_masked_pct=0.0,
            total_cells=1000,
            checks=checks,
        )
        assert len(report.checks) == 2


class TestQCPipelineReportProperties:
    """Test QCPipelineReport computed properties."""

    def test_final_valid_pct_no_checks(self, empty_report):
        """Test final_valid_pct with no checks."""
        assert empty_report.final_valid_pct == 90.0

    def test_final_valid_pct_with_checks(self, report_with_checks):
        """Test final_valid_pct with checks."""
        # Last check has valid_pct of 85%
        assert report_with_checks.final_valid_pct == 85.0

    def test_pipeline_impact_pct_no_checks(self, empty_report):
        """Test pipeline_impact_pct with no checks."""
        assert empty_report.pipeline_impact_pct == 0.0

    def test_pipeline_impact_pct_with_checks(self, report_with_checks):
        """Test pipeline_impact_pct with checks."""
        # total_masked_pct (15%) - baseline_masked_pct (10%) = 5%
        assert report_with_checks.pipeline_impact_pct == 5.0

    def test_final_valid_pct_multiple_checks(self):
        """Test final_valid_pct uses last check."""
        checks = [
            QCCheckStats(
                check_name="Check 1",
                threshold=15.0,
                cells_pre_masked=0,
                cells_newly_masked=100,
                cells_total_masked=100,
                total_cells=1000,
            ),
            QCCheckStats(
                check_name="Check 2",
                threshold=15.0,
                cells_pre_masked=100,
                cells_newly_masked=50,
                cells_total_masked=150,
                total_cells=1000,
            ),
        ]
        report = QCPipelineReport(
            module_name="test",
            baseline_masked=0,
            baseline_masked_pct=0.0,
            total_cells=1000,
            checks=checks,
        )
        # Last check has 85% valid
        assert report.final_valid_pct == 85.0


class TestQCPipelineReportStringRepresentation:
    """Test QCPipelineReport string representation."""

    def test_str_representation_empty(self, empty_report):
        """Test string representation with no checks."""
        s = str(empty_report)
        assert "SENSOR_HEALTH" in s
        assert "PIPELINE REPORT" in s
        assert "No QC checks applied" in s

    def test_str_representation_with_checks(self, report_with_checks):
        """Test string representation with checks."""
        s = str(report_with_checks)
        assert "SENSOR_HEALTH" in s
        assert "Roll Check" in s
        assert "QC CHECKS" in s
        assert "FINAL" in s

    def test_str_representation_with_modifications(self, full_report):
        """Test string representation with modifications."""
        s = str(full_report)
        assert "DATA MODIFICATIONS" in s
        assert "replace_data" in s
        assert "temperature" in s

    def test_str_includes_baseline(self, empty_report):
        """Test string includes baseline info."""
        s = str(empty_report)
        assert "Baseline" in s
        assert "100" in s
        assert "10.00%" in s

    def test_str_includes_timestamp(self, empty_report):
        """Test string includes timestamp."""
        s = str(empty_report)
        assert "Generated" in s


class TestQCPipelineReportSerialization:
    """Test QCPipelineReport serialization."""

    def test_to_dict_basic(self, empty_report):
        """Test to_dict with empty report."""
        d = empty_report.to_dict()
        assert d["module_name"] == "sensor_health"
        assert d["baseline"]["masked_cells"] == 100
        assert d["baseline"]["masked_pct"] == 10.0
        assert d["baseline"]["total_cells"] == 1000
        assert d["checks"] == []
        assert d["modifications"] == []
        assert "summary" in d
        assert "timestamp" in d

    def test_to_dict_with_checks(self, report_with_checks):
        """Test to_dict with checks."""
        d = report_with_checks.to_dict()
        assert len(d["checks"]) == 1
        assert d["checks"][0]["check_name"] == "Roll Check"

    def test_to_dict_with_modifications(self, full_report):
        """Test to_dict with modifications."""
        d = full_report.to_dict()
        assert len(d["modifications"]) == 1
        assert d["modifications"][0]["variable_name"] == "temperature"

    def test_to_dict_summary(self, report_with_checks):
        """Test to_dict summary section."""
        d = report_with_checks.to_dict()
        assert "final_valid_pct" in d["summary"]
        assert "pipeline_impact_pct" in d["summary"]
        assert d["summary"]["final_valid_pct"] == 85.0

    def test_to_dict_json_serializable(self, full_report):
        """Test that to_dict output is JSON serializable."""
        import json

        d = full_report.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


class TestQCPipelineReportModuleNames:
    """Test different module names."""

    def test_signal_quality_module(self):
        """Test report with signal_quality module name."""
        report = QCPipelineReport(
            module_name="signal_quality",
            baseline_masked=0,
            baseline_masked_pct=0.0,
            total_cells=1000,
        )
        s = str(report)
        assert "SIGNAL_QUALITY" in s

    def test_profile_operation_module(self):
        """Test report with profile_operation module name."""
        report = QCPipelineReport(
            module_name="profile_operation",
            baseline_masked=0,
            baseline_masked_pct=0.0,
            total_cells=1000,
        )
        s = str(report)
        assert "PROFILE_OPERATION" in s

    def test_velocity_check_module(self):
        """Test report with velocity_check module name."""
        report = QCPipelineReport(
            module_name="velocity_check",
            baseline_masked=0,
            baseline_masked_pct=0.0,
            total_cells=1000,
        )
        s = str(report)
        assert "VELOCITY_CHECK" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
