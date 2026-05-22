"""
Test suite for QCCheckStats dataclass.
"""

import pytest
from datetime import datetime, timezone

from pyadps.processing.utility import QCCheckStats


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


# ============================================================================
# TESTS: QCCheckStats
# ============================================================================


class TestQCCheckStatsCreation:
    """Test QCCheckStats creation and basic attributes."""

    def test_basic_creation(self, qc_check_stat):
        """Test basic dataclass creation."""
        assert qc_check_stat.check_name == "Roll Check"
        assert qc_check_stat.threshold == 15.0
        assert qc_check_stat.cells_pre_masked == 100
        assert qc_check_stat.cells_newly_masked == 50
        assert qc_check_stat.cells_total_masked == 150
        assert qc_check_stat.total_cells == 1000

    def test_check_time_auto_generated(self, qc_check_stat):
        """Test that check_time is automatically generated."""
        assert isinstance(qc_check_stat.check_time, datetime)
        assert qc_check_stat.check_time.tzinfo == timezone.utc

    def test_metadata_default_empty(self, qc_check_stat):
        """Test that metadata defaults to empty dict."""
        assert qc_check_stat.metadata == {}

    def test_tuple_threshold(self):
        """Test with tuple threshold (for range checks)."""
        stat = QCCheckStats(
            check_name="Pressure Check",
            threshold=(0.0, 1000.0),
            cells_pre_masked=0,
            cells_newly_masked=10,
            cells_total_masked=10,
            total_cells=100,
        )
        assert stat.threshold == (0.0, 1000.0)

    def test_none_threshold(self):
        """Test with None threshold (for operations without threshold)."""
        stat = QCCheckStats(
            check_name="Regrid",
            threshold=None,
            cells_pre_masked=0,
            cells_newly_masked=0,
            cells_total_masked=0,
            total_cells=100,
        )
        assert stat.threshold is None

    def test_metadata_with_values(self):
        """Test metadata field with values."""
        stat = QCCheckStats(
            check_name="Roll Check",
            threshold=15.0,
            cells_pre_masked=0,
            cells_newly_masked=10,
            cells_total_masked=10,
            total_cells=100,
            metadata={"roll_min": -5.0, "roll_max": 25.0, "roll_mean": 10.0},
        )
        assert stat.metadata["roll_min"] == -5.0
        assert stat.metadata["roll_max"] == 25.0
        assert stat.metadata["roll_mean"] == 10.0


class TestQCCheckStatsProperties:
    """Test QCCheckStats computed properties."""

    def test_pre_masked_pct(self, qc_check_stat):
        """Test pre_masked_pct property."""
        assert qc_check_stat.pre_masked_pct == 10.0

    def test_newly_masked_pct(self, qc_check_stat):
        """Test newly_masked_pct property."""
        assert qc_check_stat.newly_masked_pct == 5.0

    def test_total_masked_pct(self, qc_check_stat):
        """Test total_masked_pct property."""
        assert qc_check_stat.total_masked_pct == 15.0

    def test_valid_cells(self, qc_check_stat):
        """Test valid_cells property."""
        assert qc_check_stat.valid_cells == 850

    def test_valid_pct(self, qc_check_stat):
        """Test valid_pct property."""
        assert qc_check_stat.valid_pct == 85.0

    def test_zero_total_cells(self):
        """Test edge case with zero total cells."""
        stat = QCCheckStats(
            check_name="Empty Check",
            threshold=10.0,
            cells_pre_masked=0,
            cells_newly_masked=0,
            cells_total_masked=0,
            total_cells=0,
        )
        assert stat.pre_masked_pct == 0.0
        assert stat.newly_masked_pct == 0.0
        assert stat.total_masked_pct == 0.0
        assert stat.valid_pct == 0.0
        assert stat.valid_cells == 0

    def test_all_masked(self):
        """Test when all cells are masked."""
        stat = QCCheckStats(
            check_name="All Masked",
            threshold=1.0,
            cells_pre_masked=0,
            cells_newly_masked=100,
            cells_total_masked=100,
            total_cells=100,
        )
        assert stat.valid_pct == 0.0
        assert stat.valid_cells == 0
        assert stat.total_masked_pct == 100.0


class TestQCCheckStatsStringRepresentation:
    """Test QCCheckStats string representation."""

    def test_str_representation(self, qc_check_stat):
        """Test string representation."""
        s = str(qc_check_stat)
        assert "Roll Check" in s
        assert "15.0" in s
        assert "Pre" in s
        assert "Impact" in s

    def test_str_with_tuple_threshold(self):
        """Test string representation with tuple threshold."""
        stat = QCCheckStats(
            check_name="Range Check",
            threshold=(0.0, 100.0),
            cells_pre_masked=0,
            cells_newly_masked=10,
            cells_total_masked=10,
            total_cells=100,
        )
        s = str(stat)
        assert "Range Check" in s
        assert "(0.0, 100.0)" in s

    def test_str_with_none_threshold(self):
        """Test string representation with None threshold."""
        stat = QCCheckStats(
            check_name="No Threshold Op",
            threshold=None,
            cells_pre_masked=0,
            cells_newly_masked=0,
            cells_total_masked=0,
            total_cells=100,
        )
        s = str(stat)
        assert "No Threshold Op" in s
        assert "None" in s


class TestQCCheckStatsSerialization:
    """Test QCCheckStats serialization."""

    def test_to_dict(self, qc_check_stat):
        """Test to_dict method."""
        d = qc_check_stat.to_dict()
        assert d["check_name"] == "Roll Check"
        assert d["threshold"] == 15.0
        assert d["cells_pre_masked"] == 100
        assert d["cells_newly_masked"] == 50
        assert d["cells_total_masked"] == 150
        assert d["total_cells"] == 1000
        assert d["pre_masked_pct"] == 10.0
        assert d["newly_masked_pct"] == 5.0
        assert d["total_masked_pct"] == 15.0
        assert d["valid_cells"] == 850
        assert d["valid_pct"] == 85.0
        assert "check_time" in d
        assert "metadata" in d

    def test_to_dict_json_serializable(self, qc_check_stat):
        """Test that to_dict output is JSON serializable."""
        import json

        d = qc_check_stat.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_to_dict_with_metadata(self):
        """Test to_dict includes metadata."""
        stat = QCCheckStats(
            check_name="Test",
            threshold=10.0,
            cells_pre_masked=0,
            cells_newly_masked=5,
            cells_total_masked=5,
            total_cells=100,
            metadata={"custom_field": "value"},
        )
        d = stat.to_dict()
        assert d["metadata"]["custom_field"] == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
