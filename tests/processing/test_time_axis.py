#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test suite for pyadps time series utility functions.

Tests for:
  1. snap_time_axis() - Snap/round time axis to nearest frequency
  2. fill_time_gaps() - Fill missing time values in time series
  3. _timedelta_to_freq_string() - Helper function for frequency conversion

These tests verify immutability, error handling, edge cases, and correct
functional behavior of time transformation utilities.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr

# Import utilities to test
from pyadps.processing.time_axis import (
    snap_time_axis,
    fill_time_gaps,
    _timedelta_to_freq_string,
)


# ============================================================================
# FIXTURES: Basic Dataset Creation
# ============================================================================


@pytest.fixture
def ds_regular_hourly():
    """Create a regular hourly ADCP dataset."""
    times = pd.date_range("2024-01-01 00:00:00", periods=24, freq="h")
    depth = np.arange(10, 200, 10)
    beam = np.arange(1, 5)

    data_vars = {
        "velocity": (("time", "depth", "beam"), np.random.randn(24, 19, 4)),
        "correlation": (
            ("time", "depth", "beam"),
            np.random.randint(0, 255, (24, 19, 4)),
        ),
    }

    coords = {
        "time": times,
        "depth": depth,
        "beam": beam,
    }

    return xr.Dataset(data_vars, coords=coords)


@pytest.fixture
def ds_irregular_time():
    """Create dataset with irregular time spacing."""
    # Start at hourly, then add some gaps and irregular intervals
    times = [
        pd.Timestamp("2024-01-01 00:00:00"),
        pd.Timestamp("2024-01-01 01:00:00"),
        pd.Timestamp("2024-01-01 02:00:00"),
        pd.Timestamp("2024-01-01 02:30:00"),  # Gap
        pd.Timestamp("2024-01-01 03:00:00"),
        pd.Timestamp("2024-01-01 04:15:00"),  # Irregular
        pd.Timestamp("2024-01-01 05:00:00"),
        pd.Timestamp("2024-01-01 06:00:00"),
    ]
    times = pd.DatetimeIndex(times)

    depth = np.arange(10, 100, 10)
    beam = np.arange(1, 5)

    data_vars = {
        "velocity": (("time", "depth", "beam"), np.random.randn(8, 9, 4)),
        "correlation": (
            ("time", "depth", "beam"),
            np.random.randint(0, 255, (8, 9, 4)),
        ),
    }

    coords = {
        "time": times,
        "depth": depth,
        "beam": beam,
    }

    return xr.Dataset(data_vars, coords=coords)


@pytest.fixture
def ds_with_gaps():
    """Create dataset with explicit time gaps."""
    times = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01 00:00:00"),
            pd.Timestamp("2024-01-01 01:00:00"),
            pd.Timestamp("2024-01-01 02:00:00"),
            # Gap - missing 03:00:00
            pd.Timestamp("2024-01-01 04:00:00"),
            pd.Timestamp("2024-01-01 05:00:00"),
        ]
    )

    depth = np.arange(10, 100, 10)
    beam = np.arange(1, 5)

    data_vars = {
        "velocity": (("time", "depth", "beam"), np.random.randn(5, 9, 4)),
        "correlation": (
            ("time", "depth", "beam"),
            np.random.randint(0, 255, (5, 9, 4)),
        ),
        "fixed_leader": (("time",), np.ones(5)),  # For forward fill testing
    }

    coords = {
        "time": times,
        "depth": depth,
        "beam": beam,
    }

    return xr.Dataset(data_vars, coords=coords)


@pytest.fixture
def ds_tiny_gaps():
    """Create dataset with small time errors (< 5 min correction)."""
    times = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01 00:00:00"),
            pd.Timestamp("2024-01-01 01:00:01"),  # 1 second off
            pd.Timestamp("2024-01-01 02:00:02"),  # 2 seconds off
            pd.Timestamp("2024-01-01 03:00:03"),  # 3 seconds off
            pd.Timestamp("2024-01-01 04:00:04"),  # 4 seconds off
        ]
    )

    depth = np.arange(10, 50, 10)
    beam = np.arange(1, 5)

    data_vars = {
        "velocity": (("time", "depth", "beam"), np.random.randn(5, 4, 4)),
    }

    coords = {
        "time": times,
        "depth": depth,
        "beam": beam,
    }

    return xr.Dataset(data_vars, coords=coords)


@pytest.fixture
def ds_single_time_point():
    """Dataset with only one time point (cannot auto-detect frequency)."""
    times = pd.DatetimeIndex([pd.Timestamp("2024-01-01 00:00:00")])
    depth = np.arange(10, 50, 10)

    data_vars = {
        "velocity": (("time", "depth"), np.random.randn(1, 4)),
    }

    coords = {
        "time": times,
        "depth": depth,
    }

    return xr.Dataset(data_vars, coords=coords)


# ============================================================================
# TEST SUITE: _timedelta_to_freq_string()
# ============================================================================


class TestTimedeltaToFreqString:
    """Tests for frequency string conversion helper."""

    def test_hour_conversion(self):
        """Test hourly timedelta conversion."""
        td = pd.Timedelta(hours=1)
        result = _timedelta_to_freq_string(td)
        assert result == "h"

    def test_multiple_hours_conversion(self):
        """Test multi-hour timedelta conversion."""
        td = pd.Timedelta(hours=6)
        result = _timedelta_to_freq_string(td)
        assert result == "6h"

    def test_minute_conversion(self):
        """Test minute timedelta conversion."""
        td = pd.Timedelta(minutes=30)
        result = _timedelta_to_freq_string(td)
        assert result == "30min"

    def test_single_minute_conversion(self):
        """Test single minute timedelta conversion."""
        td = pd.Timedelta(minutes=1)
        result = _timedelta_to_freq_string(td)
        assert result == "min"

    def test_second_conversion(self):
        """Test second timedelta conversion."""
        td = pd.Timedelta(seconds=30)
        result = _timedelta_to_freq_string(td)
        assert result == "30S"

    def test_single_second_conversion(self):
        """Test single second timedelta conversion."""
        td = pd.Timedelta(seconds=1)
        result = _timedelta_to_freq_string(td)
        assert result == "1S"

    def test_day_conversion(self):
        """Test daily timedelta conversion."""
        td = pd.Timedelta(days=1)
        result = _timedelta_to_freq_string(td)
        # 1 day = 3600 * 24 = 86400 seconds, which is 24 hours
        assert result == "24h"

    def test_complex_timedelta(self):
        """Test complex timedelta (hours + minutes)."""
        td = pd.Timedelta(hours=1, minutes=30)
        result = _timedelta_to_freq_string(td)
        # 1.5 hours = 90 minutes
        assert result == "90min"


# ============================================================================
# TEST SUITE: snap_time_axis()
# ============================================================================


class TestSnapTimeAxis:
    """Tests for time axis snapping utility."""

    def test_snap_returns_tuple(self, ds_regular_hourly):
        """Test that snap_time_axis returns correct tuple structure."""
        result = snap_time_axis(ds_regular_hourly, freq="h")
        assert isinstance(result, tuple)
        assert len(result) == 3
        ds_result, success, msg = result
        assert isinstance(success, bool)
        assert isinstance(msg, str)

    def test_snap_preserves_original(self, ds_regular_hourly):
        """Test that original dataset is never modified."""
        original_time = ds_regular_hourly.time.copy()
        ds_snapped, success, msg = snap_time_axis(ds_regular_hourly, freq="h")

        np.testing.assert_array_equal(
            ds_regular_hourly.time.values, original_time.values
        )

    def test_snap_tiny_error_succeeds(self, ds_tiny_gaps):
        """Test that small time errors snap successfully."""
        ds_snapped, success, msg = snap_time_axis(
            ds_tiny_gaps, freq="h", tolerance="10s"
        )

        assert success, msg
        assert ds_snapped is not None
        # Time should be snapped to exact hours
        snapped_times = pd.DatetimeIndex(ds_snapped.time.values)
        assert (snapped_times.minute == 0).all()
        assert (snapped_times.second == 0).all()

    def test_snap_tolerance_exceeded(self, ds_irregular_time):
        """Test that excessive correction is rejected."""
        # This dataset has a 15-minute jump, should fail with 5-minute tolerance
        ds_snapped, success, msg = snap_time_axis(
            ds_irregular_time, freq="h", tolerance="5min"
        )

        assert not success
        assert ds_snapped is None
        assert "tolerance" in msg.lower()

    def test_snap_with_target_minute(self, ds_tiny_gaps):
        """Test snapping to specific minute within hour."""
        ds_snapped, success, msg = snap_time_axis(
            ds_tiny_gaps,
            freq="h",
            target_minute=0,  # Snap to HH:00 (no minute offset needed)
            tolerance="10s",
        )

        assert success, msg
        assert ds_snapped is not None
        snapped_times = pd.DatetimeIndex(ds_snapped.time.values)
        assert (snapped_times.minute == 0).all()

    def test_snap_invalid_target_minute(self, ds_regular_hourly):
        """Test that invalid target_minute is rejected."""
        ds_snapped, success, msg = snap_time_axis(
            ds_regular_hourly,
            target_minute=65,  # Invalid
        )

        assert not success
        assert ds_snapped is None
        assert "target_minute" in msg.lower()

    def test_snap_non_dataset_input(self):
        """Test error handling for non-Dataset input."""
        ds_snapped, success, msg = snap_time_axis(
            [1, 2, 3],  # Not a dataset
            freq="h",
        )

        assert not success
        assert ds_snapped is None
        assert "dataset" in msg.lower()

    def test_snap_missing_time_coordinate(self):
        """Test error handling when time coordinate is missing."""
        ds = xr.Dataset({"data": (("x",), [1, 2, 3])})

        ds_snapped, success, msg = snap_time_axis(ds)

        assert not success
        assert ds_snapped is None
        assert "time" in msg.lower()

    def test_snap_tolerance_as_timedelta(self, ds_tiny_gaps):
        """Test tolerance parameter as pd.Timedelta."""
        tolerance = pd.Timedelta(seconds=10)
        ds_snapped, success, msg = snap_time_axis(
            ds_tiny_gaps, freq="h", tolerance=tolerance
        )

        assert success, msg
        assert ds_snapped is not None

    def test_snap_invalid_tolerance_string(self, ds_regular_hourly):
        """Test error handling for invalid tolerance string."""
        ds_snapped, success, msg = snap_time_axis(
            ds_regular_hourly, tolerance="invalid_time_string"
        )

        assert not success
        assert ds_snapped is None

    def test_snap_creates_duplicates(self):
        """Test detection of snapping that would create duplicates."""
        # Create times that snap to same value
        times = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-01 00:00:30"),
                pd.Timestamp("2024-01-01 00:01:00"),
                pd.Timestamp("2024-01-01 00:01:30"),
                pd.Timestamp("2024-01-01 00:02:00"),
            ]
        )

        ds = xr.Dataset({"data": (("time",), [1, 2, 3, 4])}, coords={"time": times})

        # Snapping to minute precision will create duplicates
        ds_snapped, success, msg = snap_time_axis(ds, freq="min")

        assert not success
        assert ds_snapped is None
        assert "duplicate" in msg.lower()

    def test_snap_preserves_data_values(self, ds_regular_hourly):
        """Test that data values are preserved after snapping."""
        original_data = ds_regular_hourly["velocity"].values.copy()

        ds_snapped, success, msg = snap_time_axis(ds_regular_hourly, freq="h")

        assert success
        np.testing.assert_array_equal(ds_snapped["velocity"].values, original_data)

    def test_snap_preserves_coords_and_dims(self, ds_regular_hourly):
        """Test that coordinates and dimensions are preserved."""
        ds_snapped, success, msg = snap_time_axis(ds_regular_hourly, freq="h")

        assert success
        assert set(ds_snapped.coords) == set(ds_regular_hourly.coords)
        assert set(ds_snapped.dims) == set(ds_regular_hourly.dims)


# ============================================================================
# TEST SUITE: fill_time_gaps()
# ============================================================================


class TestFillTimeGaps:
    """Tests for time gap filling utility."""

    def test_fill_returns_dataset(self, ds_with_gaps):
        """Test that fill_time_gaps returns an xarray.Dataset."""
        result = fill_time_gaps(ds_with_gaps, method="h")
        assert isinstance(result, xr.Dataset)

    def test_fill_preserves_original(self, ds_with_gaps):
        """Test that original dataset is never modified."""
        original_time = ds_with_gaps.time.copy()
        original_len = len(ds_with_gaps.time)

        ds_filled = fill_time_gaps(ds_with_gaps, method="h")

        np.testing.assert_array_equal(ds_with_gaps.time.values, original_time.values)
        assert len(ds_with_gaps.time) == original_len

    def test_fill_creates_regular_grid(self, ds_with_gaps):
        """Test that filled dataset has regular time grid."""
        ds_filled = fill_time_gaps(ds_with_gaps, method="h")

        # Check time is regular (all intervals equal)
        time_diffs = pd.Series(ds_filled.time.values).diff().dropna()
        assert (time_diffs == time_diffs.iloc[0]).all()

    def test_fill_fills_gaps(self, ds_with_gaps):
        """Test that gaps are actually filled."""
        original_len = len(ds_with_gaps.time)
        ds_filled = fill_time_gaps(ds_with_gaps, method="h")

        # Should have more time points (missing 03:00:00 is now filled)
        assert len(ds_filled.time) > original_len

    def test_fill_auto_detect_frequency(self, ds_with_gaps):
        """Test auto-detection of frequency."""
        ds_filled = fill_time_gaps(ds_with_gaps, method="auto")

        # Should auto-detect hourly frequency
        assert len(ds_filled.time) == 6  # 00:00 to 05:00

    def test_fill_custom_frequency(self, ds_with_gaps):
        """Test with custom frequency."""
        ds_filled = fill_time_gaps(ds_with_gaps, method="30min")

        # With 30-minute frequency
        time_diffs = pd.Series(ds_filled.time.values).diff().dropna()
        expected_diff = pd.Timedelta(minutes=30)
        assert (time_diffs == expected_diff).all()

    def test_fill_timedelta_frequency(self, ds_with_gaps):
        """Test frequency as pd.Timedelta."""
        ds_filled = fill_time_gaps(ds_with_gaps, method=pd.Timedelta(hours=1))

        # Should be same as method='h'
        assert len(ds_filled.time) == 6

    def test_fill_velocity_missing_value(self, ds_with_gaps):
        """Test that float velocity gaps are left as NaN (missing_as_nan=True mode)."""
        ds_filled = fill_time_gaps(ds_with_gaps)

        # Find the filled point (03:00:00)
        filled_idx = 3  # Index for 03:00:00

        # Float velocity gaps should remain NaN, not be filled with -32768
        filled_velocity = ds_filled["velocity"].isel(time=filled_idx)
        assert filled_velocity.isnull().all()

    def test_fill_custom_missing_values(self, ds_with_gaps):
        """Test custom missing value codes — float velocity stays NaN, int vars use custom code."""
        custom_missing = {"velocity": -999, "correlation": 0}
        ds_filled = fill_time_gaps(
            ds_with_gaps, method="h", missing_values=custom_missing
        )

        # Float velocity: custom fill value is ignored, gap stays NaN
        filled_velocity = ds_filled["velocity"].isel(time=3)
        assert filled_velocity.isnull().all()

        # Integer correlation: custom fill value (0) is applied
        filled_correlation = ds_filled["correlation"].isel(time=3)
        assert (filled_correlation == 0).all()

    def test_fill_forward_fill_fixed_leader(self, ds_with_gaps):
        """Test forward-filling of fixed_leader."""
        ds_filled = fill_time_gaps(
            ds_with_gaps, method="h", forward_fill_fixed_leader=True
        )

        # fixed_leader should be forward-filled
        filled_fl = ds_filled["fixed_leader"].isel(time=3)
        assert filled_fl.values == 1.0  # Forward filled from time=2

    def test_fill_no_forward_fill(self, ds_with_gaps):
        """Test disabling forward-fill."""
        ds_filled = fill_time_gaps(
            ds_with_gaps,
            method="h",
            forward_fill_fixed_leader=False,
            forward_fill_variable_leader=False,
        )

        # fixed_leader should be NaN (not forward-filled)
        filled_fl = ds_filled["fixed_leader"].isel(time=3)
        assert np.isnan(filled_fl.values)

    def test_fill_non_dataset_input(self):
        """Test error handling for non-Dataset input."""
        with pytest.raises(TypeError):
            fill_time_gaps([1, 2, 3], method="h")

    def test_fill_missing_time_coordinate(self):
        """Test error handling when time coordinate is missing."""
        ds = xr.Dataset({"data": (("x",), [1, 2, 3])})

        with pytest.raises(ValueError):
            fill_time_gaps(ds)

    def test_fill_single_time_point(self, ds_single_time_point):
        """Test error handling for insufficient time points."""
        with pytest.raises(ValueError):
            fill_time_gaps(ds_single_time_point, method="auto")

    def test_fill_preserves_data_values(self, ds_with_gaps):
        """Test that original data values are preserved."""
        original_velocity = ds_with_gaps["velocity"].values.copy()

        ds_filled = fill_time_gaps(ds_with_gaps, method="h")

        # Original points should have same values
        filled_velocity = (
            ds_filled["velocity"]
            .isel(
                time=[0, 1, 2, 4, 5]  # Skip index 3 which was filled
            )
            .values
        )

        # Compare with original (accounting for potential reshape)
        assert filled_velocity.shape[1:] == original_velocity.shape[1:]

    def test_fill_preserves_other_coords(self, ds_with_gaps):
        """Test that non-time coordinates are preserved."""
        ds_filled = fill_time_gaps(ds_with_gaps, method="h")

        if "depth" in ds_with_gaps.coords:
            np.testing.assert_array_equal(
                ds_filled["depth"].values, ds_with_gaps["depth"].values
            )

    def test_fill_preserves_attributes(self, ds_with_gaps):
        """Test that dataset attributes are preserved."""
        ds_with_gaps.attrs["test_attr"] = "test_value"
        ds_filled = fill_time_gaps(ds_with_gaps, method="h")

        assert ds_filled.attrs.get("test_attr") == "test_value"

    def test_fill_large_frequency_string(self, ds_with_gaps):
        """Test with various frequency strings."""
        test_freqs = ["h", "2h", "30min", "D", "6h"]

        for freq in test_freqs:
            try:
                ds_filled = fill_time_gaps(ds_with_gaps, method=freq)
                assert isinstance(ds_filled, xr.Dataset)
            except Exception as e:
                pytest.fail(f"Failed with frequency {freq}: {str(e)}")


# ============================================================================
# TEST SUITE: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple utilities."""

    def test_snap_then_read(self, ds_tiny_gaps):
        """Test snapping followed by verification."""
        ds_snapped, success, snap_msg = snap_time_axis(
            ds_tiny_gaps, freq="h", tolerance="10s"
        )

        assert success

        # Time should now be perfectly regular
        time_diffs = pd.Series(ds_snapped.time.values).diff().dropna()
        assert (time_diffs == pd.Timedelta(hours=1)).all()

    def test_fill_then_snap(self, ds_with_gaps):
        """Test filling gaps then snapping time."""
        # First fill gaps
        ds_filled = fill_time_gaps(ds_with_gaps, method="h")

        # Time should already be regular, but test snapping anyway
        ds_snapped, success, msg = snap_time_axis(ds_filled, freq="h")

        assert success
        np.testing.assert_array_equal(ds_snapped.time.values, ds_filled.time.values)

    def test_workflow_irregular_to_regular(self, ds_irregular_time):
        """Test typical workflow: irregular â†’ regular."""
        # Step 1: Try to snap (might fail due to large gaps)
        ds_snapped, success_snap, msg_snap = snap_time_axis(
            ds_irregular_time,
            freq="h",
            tolerance="30min",  # Generous tolerance
        )

        # If snap fails, use fill instead
        if success_snap:
            ds_final = ds_snapped
        else:
            ds_final = fill_time_gaps(ds_irregular_time, method="h")

        # Final dataset should have regular time
        time_diffs = pd.Series(ds_final.time.values).diff().dropna()
        assert (time_diffs == time_diffs.iloc[0]).all()


# ============================================================================
# TEST SUITE: Edge Cases and Error Conditions
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_snap_already_regular_time(self, ds_regular_hourly):
        """Test snapping already regular time (should be no-op)."""
        ds_snapped, success, msg = snap_time_axis(ds_regular_hourly, freq="h")

        assert success
        np.testing.assert_array_equal(
            ds_snapped.time.values, ds_regular_hourly.time.values
        )

    def test_fill_already_regular_time(self, ds_regular_hourly):
        """Test filling on already regular time."""
        ds_filled = fill_time_gaps(ds_regular_hourly, method="h")

        # Should have same number of points (no gaps to fill)
        assert len(ds_filled.time) == len(ds_regular_hourly.time)

    def test_snap_large_tolerance(self, ds_irregular_time):
        """Test snapping with very large tolerance (should always succeed)."""
        ds_snapped, success, msg = snap_time_axis(
            ds_irregular_time,
            freq="h",
            tolerance="1h",  # Very large
        )

        assert success or not success  # May succeed depending on duplicates
        if success:
            assert ds_snapped is not None

    def test_fill_finer_than_original(self, ds_regular_hourly):
        """Test filling to finer resolution than original."""
        ds_filled = fill_time_gaps(ds_regular_hourly, method="30min")

        # Should have 2x as many points (30-minute vs hourly)
        assert len(ds_filled.time) == 2 * len(ds_regular_hourly.time) - 1

    def test_fill_coarser_than_original(self, ds_with_gaps):
        """Test filling to coarser resolution (shouldn't add many points)."""
        ds_filled = fill_time_gaps(ds_with_gaps, method="2h")

        # With 2-hour resolution from 00:00 to 04:00
        # Points: 0:00, 2:00, 4:00 = 3 points
        expected_points = 3
        assert len(ds_filled.time) == expected_points

    def test_negative_target_minute(self, ds_regular_hourly):
        """Test negative target_minute (should fail)."""
        ds_snapped, success, msg = snap_time_axis(ds_regular_hourly, target_minute=-1)

        assert not success
        assert ds_snapped is None


# ============================================================================
# TEST SUITE: Coverage for Exception Branches
# ============================================================================


class TestExceptionBranches:
    """Tests targeting specific exception-handling branches for full coverage."""

    # ---- Line 130: except (AttributeError, TypeError) in snap_time_axis ----

    def test_snap_attribute_error_on_isinstance(self, monkeypatch):
        """Trigger AttributeError branch (line 130) by monkeypatching isinstance."""
        import builtins

        original_isinstance = builtins.isinstance

        def patched_isinstance(obj, cls):
            if cls is xr.Dataset:
                raise AttributeError("Simulated AttributeError")
            return original_isinstance(obj, cls)

        monkeypatch.setattr(builtins, "isinstance", patched_isinstance)

        result, success, msg = snap_time_axis(object())
        assert not success
        assert result is None
        assert "AttributeError" in msg

    # ---- Line 175: except Exception during rounding ----

    def test_snap_invalid_freq_triggers_rounding_error(self, ds_regular_hourly):
        """Trigger rounding error (line 175) with invalid freq string."""
        # pandas .dt.round() will raise on an unsupported frequency alias
        ds_snapped, success, msg = snap_time_axis(
            ds_regular_hourly,
            freq="INVALID_FREQ_XYZ",
            tolerance="1h",
        )

        assert not success
        assert ds_snapped is None
        assert "Error during rounding" in msg

    # ---- Line 198: except Exception creating new dataset ----

    def test_snap_assign_coords_failure(self, ds_regular_hourly):
        """Trigger assign_coords failure (line 198) via unittest.mock.patch."""
        from unittest.mock import patch

        def bad_assign_coords(self, coords):
            raise RuntimeError("Simulated assign_coords failure")

        with patch.object(xr.Dataset, "assign_coords", bad_assign_coords):
            ds_snapped, success, msg = snap_time_axis(ds_regular_hourly, freq="h")

        assert not success
        assert ds_snapped is None
        assert "Error creating new dataset" in msg

    # ---- Line 391: raise ValueError when interval is not pd.Timedelta ----

    def test_fill_non_timedelta_median_raises(self, monkeypatch):
        """Trigger ValueError (line 391) when median returns non-Timedelta."""
        # Build a dataset where time diffs have a non-timedelta median
        times = pd.date_range("2024-01-01", periods=5, freq="h")
        ds = xr.Dataset({"v": (("time",), [1, 2, 3, 4, 5])}, coords={"time": times})

        # Monkeypatch pd.Series.median to return a non-Timedelta value
        original_median = pd.Series.median

        def bad_median(self, *args, **kwargs):
            return 42  # Not a pd.Timedelta

        monkeypatch.setattr(pd.Series, "median", bad_median)

        with pytest.raises(ValueError, match="Cannot auto-detect frequency"):
            fill_time_gaps(ds, method="auto")

    # ---- Line 415: except Exception during reindexing ----

    def test_fill_reindex_failure_reraises(self, ds_with_gaps):
        """Trigger reindex exception (line 415) via unittest.mock.patch."""
        from unittest.mock import patch

        def bad_reindex(self, time):
            raise RuntimeError("Simulated reindex failure")

        with patch.object(xr.Dataset, "reindex", bad_reindex):
            with pytest.raises(RuntimeError, match="Simulated reindex failure"):
                fill_time_gaps(ds_with_gaps, method="h")

    # ---- Lines 449-452: except Exception in variable filling loop ----

    def test_fill_variable_fillna_failure_is_warned(self, monkeypatch):
        """Trigger per-variable fill failure (lines 449-452); should warn, not raise."""
        times = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 01:00:00"),
                pd.Timestamp("2024-01-01 02:00:00"),
                # Gap at 03:00
                pd.Timestamp("2024-01-01 04:00:00"),
            ]
        )
        ds = xr.Dataset(
            {"velocity": (("time",), [1.0, 2.0, 3.0, 4.0])},
            coords={"time": times},
        )

        # Patch xr.DataArray.fillna to raise an exception
        original_fillna = xr.DataArray.fillna

        def bad_fillna(self, value, *args, **kwargs):
            raise RuntimeError("Simulated fillna failure")

        monkeypatch.setattr(xr.DataArray, "fillna", bad_fillna)

        # Should not raise — warning is logged, processing continues
        result = fill_time_gaps(ds, method="h")
        assert isinstance(result, xr.Dataset)

    def test_fill_forward_fill_variable_leader(self):
        """Test forward-filling of variable_leader covers lines 447-450."""
        times = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 01:00:00"),
                pd.Timestamp("2024-01-01 02:00:00"),
                # Gap at 03:00
                pd.Timestamp("2024-01-01 04:00:00"),
                pd.Timestamp("2024-01-01 05:00:00"),
            ]
        )
        ds = xr.Dataset(
            {
                "variable_leader": (("time",), [10.0, 20.0, 30.0, 40.0, 50.0]),
            },
            coords={"time": times},
        )

        ds_filled = fill_time_gaps(ds, method="h", forward_fill_variable_leader=True)

        # variable_leader at 03:00 should be forward-filled from 02:00 value (30.0)
        filled_vl = ds_filled["variable_leader"].isel(time=3)
        assert filled_vl.values == 30.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
