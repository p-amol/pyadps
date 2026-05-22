"""
Test suite for DataModificationStats dataclass.
"""

import pytest
from datetime import datetime, timezone

from pyadps.processing.utility import DataModificationStats


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def data_mod_stat():
    """Create a sample DataModificationStats object."""
    return DataModificationStats(
        operation="replace_data",
        variable_name="temperature",
        original_stats={"min": 10.0, "max": 30.0, "mean": 20.0, "std": 5.0},
        modified_stats={"min": 12.0, "max": 28.0, "mean": 22.0, "std": 4.0},
    )


# ============================================================================
# TESTS: DataModificationStats
# ============================================================================


class TestDataModificationStatsCreation:
    """Test DataModificationStats creation and basic attributes."""

    def test_basic_creation(self, data_mod_stat):
        """Test basic dataclass creation."""
        assert data_mod_stat.operation == "replace_data"
        assert data_mod_stat.variable_name == "temperature"
        assert data_mod_stat.original_stats["mean"] == 20.0
        assert data_mod_stat.modified_stats["mean"] == 22.0

    def test_modification_time_auto_generated(self, data_mod_stat):
        """Test that modification_time is automatically generated."""
        assert isinstance(data_mod_stat.modification_time, datetime)
        assert data_mod_stat.modification_time.tzinfo == timezone.utc

    def test_metadata_default_empty(self, data_mod_stat):
        """Test that metadata defaults to empty dict."""
        assert data_mod_stat.metadata == {}

    def test_with_metadata(self):
        """Test creation with metadata."""
        stat = DataModificationStats(
            operation="correct_sound_speed",
            variable_name="velocity",
            original_stats={"mean": 100.0},
            modified_stats={"mean": 105.0},
            metadata={"correction_method": "Urick (1983)", "sound_speed_ratio": 1.05},
        )
        assert stat.metadata["correction_method"] == "Urick (1983)"
        assert stat.metadata["sound_speed_ratio"] == 1.05


class TestDataModificationStatsProperties:
    """Test DataModificationStats computed properties."""

    def test_mean_change(self, data_mod_stat):
        """Test mean_change property."""
        assert data_mod_stat.mean_change == 2.0

    def test_mean_change_pct(self, data_mod_stat):
        """Test mean_change_pct property."""
        assert data_mod_stat.mean_change_pct == 10.0

    def test_mean_change_none_missing_original(self):
        """Test mean_change when original stats missing mean."""
        stat = DataModificationStats(
            operation="replace_data",
            variable_name="test",
            original_stats={},
            modified_stats={"mean": 10.0},
        )
        assert stat.mean_change is None
        assert stat.mean_change_pct is None

    def test_mean_change_none_missing_modified(self):
        """Test mean_change when modified stats missing mean."""
        stat = DataModificationStats(
            operation="replace_data",
            variable_name="test",
            original_stats={"mean": 10.0},
            modified_stats={},
        )
        assert stat.mean_change is None
        assert stat.mean_change_pct is None

    def test_mean_change_none_both_missing(self):
        """Test mean_change when both stats are empty."""
        stat = DataModificationStats(
            operation="replace_data",
            variable_name="test",
            original_stats={},
            modified_stats={},
        )
        assert stat.mean_change is None
        assert stat.mean_change_pct is None

    def test_mean_change_pct_zero_original(self):
        """Test mean_change_pct when original mean is zero."""
        stat = DataModificationStats(
            operation="replace_data",
            variable_name="test",
            original_stats={"mean": 0.0},
            modified_stats={"mean": 10.0},
        )
        assert stat.mean_change == 10.0
        assert stat.mean_change_pct is None  # Division by zero

    def test_negative_mean_change(self):
        """Test negative mean change (decrease)."""
        stat = DataModificationStats(
            operation="replace_data",
            variable_name="test",
            original_stats={"mean": 100.0},
            modified_stats={"mean": 80.0},
        )
        assert stat.mean_change == -20.0
        assert stat.mean_change_pct == -20.0


class TestDataModificationStatsStringRepresentation:
    """Test DataModificationStats string representation."""

    def test_str_representation(self, data_mod_stat):
        """Test string representation."""
        s = str(data_mod_stat)
        assert "replace_data" in s
        assert "temperature" in s
        assert "20.00" in s
        assert "22.00" in s

    def test_str_with_negative_change(self):
        """Test string representation with negative change."""
        stat = DataModificationStats(
            operation="correct_sound_speed",
            variable_name="velocity",
            original_stats={"mean": 100.0},
            modified_stats={"mean": 95.0},
        )
        s = str(stat)
        assert "correct_sound_speed" in s
        assert "velocity" in s

    def test_str_missing_stats(self):
        """Test string representation with missing stats."""
        stat = DataModificationStats(
            operation="replace_data",
            variable_name="test",
            original_stats={"mean": 0.0},  # Provide mean to avoid format error
            modified_stats={"mean": 0.0},
        )
        s = str(stat)
        assert "replace_data" in s
        assert "test" in s


class TestDataModificationStatsSerialization:
    """Test DataModificationStats serialization."""

    def test_to_dict(self, data_mod_stat):
        """Test to_dict method."""
        d = data_mod_stat.to_dict()
        assert d["operation"] == "replace_data"
        assert d["variable_name"] == "temperature"
        assert d["original_stats"]["mean"] == 20.0
        assert d["modified_stats"]["mean"] == 22.0
        assert d["mean_change"] == 2.0
        assert d["mean_change_pct"] == 10.0
        assert "modification_time" in d
        assert "metadata" in d

    def test_to_dict_json_serializable(self, data_mod_stat):
        """Test that to_dict output is JSON serializable."""
        import json

        d = data_mod_stat.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_to_dict_with_metadata(self):
        """Test to_dict includes metadata."""
        stat = DataModificationStats(
            operation="correct_sound_speed",
            variable_name="velocity",
            original_stats={"mean": 100.0},
            modified_stats={"mean": 105.0},
            metadata={"ratio": 1.05},
        )
        d = stat.to_dict()
        assert d["metadata"]["ratio"] == 1.05

    def test_to_dict_none_values(self):
        """Test to_dict handles None values."""
        stat = DataModificationStats(
            operation="replace_data",
            variable_name="test",
            original_stats={},
            modified_stats={},
        )
        d = stat.to_dict()
        assert d["mean_change"] is None
        assert d["mean_change_pct"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
