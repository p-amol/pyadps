"""
Test suite for signal quality standalone QC functions.

Tests for:
- correlation_check
- echo_intensity_check
- error_velocity_check
- percent_good_check
- false_target_detection
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyadps.processing.signal_quality import (
    correlation_check,
    echo_intensity_check,
    error_velocity_check,
    percent_good_check,
    false_target_detection,
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_ECHO_THRESHOLD,
    DEFAULT_ERROR_VELOCITY_THRESHOLD,
    DEFAULT_PERCENT_GOOD_THRESHOLD,
    DEFAULT_FALSE_TARGET_THRESHOLD,
    THRESHOLD_RANGES,
    _validate_threshold,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def basic_dataset():
    """Create basic ADCP dataset with all required variables."""
    np.random.seed(42)
    n_time = 20
    n_cell = 10
    n_beam = 4

    times = pd.date_range("2024-01-01", periods=n_time, freq="h")
    cells = np.arange(n_cell)
    beams = np.arange(n_beam)

    # All values pass default thresholds
    data_vars = {
        "velocity": (
            ("beam", "cell", "time"),
            np.random.randint(-500, 500, size=(n_beam, n_cell, n_time)),
        ),
        "correlation": (
            ("beam", "cell", "time"),
            np.full((n_beam, n_cell, n_time), 100, dtype=np.int16),  # Above 64
        ),
        "echo_intensity": (
            ("beam", "cell", "time"),
            np.full((n_beam, n_cell, n_time), 80, dtype=np.int16),  # Above 40
        ),
        "percent_good": (
            ("beam", "cell", "time"),
            np.full((n_beam, n_cell, n_time), 75, dtype=np.int16),  # Above 50
        ),
        "mask": (
            ("beam", "cell", "time"),
            np.zeros((n_beam, n_cell, n_time), dtype=np.int8),
        ),
    }

    coords = {"time": times, "cell": cells, "beam": beams}
    ds = xr.Dataset(data_vars, coords=coords)

    # Add attributes
    ds["correlation"].attrs = {"long_name": "Correlation", "units": "counts"}
    ds["echo_intensity"].attrs = {"long_name": "Echo Intensity", "units": "counts"}
    ds["percent_good"].attrs = {"long_name": "Percent Good", "units": "%"}

    return ds


@pytest.fixture
def dataset_no_mask():
    """Create dataset without mask variable."""
    np.random.seed(42)
    n_time = 20
    n_cell = 10
    n_beam = 4

    times = pd.date_range("2024-01-01", periods=n_time, freq="h")

    data_vars = {
        "velocity": (
            ("beam", "cell", "time"),
            np.random.randint(-500, 500, size=(n_beam, n_cell, n_time)),
        ),
        "correlation": (
            ("beam", "cell", "time"),
            np.full((n_beam, n_cell, n_time), 100, dtype=np.int16),
        ),
        "echo_intensity": (
            ("beam", "cell", "time"),
            np.full((n_beam, n_cell, n_time), 80, dtype=np.int16),
        ),
        "percent_good": (
            ("beam", "cell", "time"),
            np.full((n_beam, n_cell, n_time), 75, dtype=np.int16),
        ),
    }

    coords = {
        "time": times,
        "cell": np.arange(n_cell),
        "beam": np.arange(n_beam),
    }

    return xr.Dataset(data_vars, coords=coords)


@pytest.fixture
def dataset_with_pre_masked():
    """Create dataset with some pre-masked values."""
    np.random.seed(42)
    n_time = 20
    n_cell = 10
    n_beam = 4

    times = pd.date_range("2024-01-01", periods=n_time, freq="h")

    mask = np.zeros((n_beam, n_cell, n_time), dtype=np.int8)
    mask[:, :2, :5] = 1  # Pre-mask some cells

    data_vars = {
        "velocity": (
            ("beam", "cell", "time"),
            np.random.randint(-500, 500, size=(n_beam, n_cell, n_time)),
        ),
        "correlation": (
            ("beam", "cell", "time"),
            np.full((n_beam, n_cell, n_time), 100, dtype=np.int16),
        ),
        "echo_intensity": (
            ("beam", "cell", "time"),
            np.full((n_beam, n_cell, n_time), 80, dtype=np.int16),
        ),
        "percent_good": (
            ("beam", "cell", "time"),
            np.full((n_beam, n_cell, n_time), 75, dtype=np.int16),
        ),
        "mask": (("beam", "cell", "time"), mask),
    }

    coords = {
        "time": times,
        "cell": np.arange(n_cell),
        "beam": np.arange(n_beam),
    }

    return xr.Dataset(data_vars, coords=coords)


# ============================================================================
# TESTS: correlation_check
# ============================================================================


class TestCorrelationCheck:
    """Tests for correlation_check function."""

    def test_returns_dataset(self, basic_dataset):
        """Test that function returns xarray Dataset."""
        result = correlation_check(basic_dataset)
        assert isinstance(result, xr.Dataset)

    def test_preserves_original(self, basic_dataset):
        """Test that original dataset is not modified."""
        original_mask = basic_dataset["mask"].values.copy()
        correlation_check(basic_dataset, cutoff=64)
        np.testing.assert_array_equal(basic_dataset["mask"].values, original_mask)

    def test_preserves_mask_attributes(self, basic_dataset):
        """Test that mask attributes are preserved."""
        basic_dataset["mask"].attrs = {"long_name": "QC Mask", "flag_values": [0, 1]}
        result = correlation_check(basic_dataset, cutoff=64)
        assert result["mask"].attrs["long_name"] == "QC Mask"

    def test_no_flagging_above_threshold(self, basic_dataset):
        """Test no flagging when all values above threshold."""
        # Dataset has correlation=100, threshold=64
        result = correlation_check(basic_dataset, cutoff=64)
        assert result["mask"].sum() == 0

    def test_flags_below_threshold(self, basic_dataset):
        """Test flagging when any beam is below threshold."""
        basic_dataset["correlation"].values[0, :3, :5] = 30
        result = correlation_check(basic_dataset, cutoff=64)
        assert result["mask"].sum() > 0
        # Entire cell is masked across all beams when any beam fails
        assert result["mask"].isel(cell=0, time=0).values.all()

    def test_flags_at_threshold_boundary(self, basic_dataset):
        """Test behavior at exact threshold value (not flagged) vs one below."""
        basic_dataset["correlation"].values[0, 0, 0] = 64  # At threshold — not flagged
        basic_dataset["correlation"].values[0, 0, 1] = 63  # Below — flagged

        result = correlation_check(basic_dataset, cutoff=64)

        assert result["mask"].isel(cell=0, time=0).values.sum() == 0
        assert result["mask"].isel(cell=0, time=1).values.all()

    def test_custom_threshold(self, basic_dataset):
        """Test with custom threshold."""
        basic_dataset["correlation"].values[:] = 90
        basic_dataset["correlation"].values[0, 0, 0] = 85

        # With cutoff=80, value 85 should pass
        result = correlation_check(basic_dataset, cutoff=80)
        assert result["mask"].isel(cell=0, time=0).values.sum() == 0

        # With cutoff=90, value 85 should fail — whole cell masked
        result = correlation_check(basic_dataset, cutoff=90)
        assert result["mask"].isel(cell=0, time=0).values.all()

    def test_creates_mask_if_missing(self, dataset_no_mask):
        """Test that mask is created if not present."""
        result = correlation_check(dataset_no_mask, cutoff=64)
        assert "mask" in result.data_vars

    def test_preserves_existing_mask(self, dataset_with_pre_masked):
        """Test that existing masked values remain masked."""
        pre_masked = dataset_with_pre_masked["mask"].sum().values
        result = correlation_check(dataset_with_pre_masked, cutoff=64)

        # Should have at least as many masked as before
        assert result["mask"].sum() >= pre_masked

    def test_missing_correlation_returns_unchanged(self, basic_dataset):
        """Test that missing correlation data returns unchanged dataset."""
        del basic_dataset["correlation"]
        result = correlation_check(basic_dataset, cutoff=64)
        # Should return dataset unchanged
        assert "mask" in result.data_vars

    def test_beam_ignore_excludes_from_count(self, basic_dataset):
        """beam_ignore removes that beam; remaining beams determine flag."""
        # Only beam 1 fails — with beam_ignore=1 it should not be flagged
        basic_dataset["correlation"].values[1, :, :] = 30

        result_no_ignore = correlation_check(basic_dataset, cutoff=64)
        result_ignored = correlation_check(basic_dataset, cutoff=64, beam_ignore=1)

        assert result_no_ignore["mask"].sum() > 0
        assert result_ignored["mask"].sum() == 0

    def test_invalid_beam_ignore_is_ignored(self, basic_dataset):
        """Test an out-of-range beam_ignore is ignored."""
        basic_dataset["correlation"].values[0, 0, 0] = 30

        # beam_ignore=5 is invalid (only 0-3 valid)
        result = correlation_check(basic_dataset, cutoff=64, beam_ignore=5)
        # Should still flag the cell (all beams) at that position
        assert result["mask"].isel(cell=0, time=0).values.all()

    def test_all_beams_masked_when_any_fails(self, basic_dataset):
        """When one beam fails, all beams at that cell/time are masked."""
        # Only beam 2 is below threshold at (cell=1, time=2)
        basic_dataset["correlation"].values[2, 1, 2] = 30
        result = correlation_check(basic_dataset, cutoff=64)
        # All four beams at (cell=1, time=2) must be masked
        assert result["mask"].isel(cell=1, time=2).values.all()
        # Other cells untouched
        assert result["mask"].isel(cell=0, time=0).values.sum() == 0


# ============================================================================
# TESTS: echo_intensity_check
# ============================================================================


class TestEchoIntensityCheck:
    """Tests for echo_intensity_check function."""

    def test_returns_dataset(self, basic_dataset):
        """Test that function returns xarray Dataset."""
        result = echo_intensity_check(basic_dataset)
        assert isinstance(result, xr.Dataset)

    def test_preserves_original(self, basic_dataset):
        """Test that original dataset is not modified."""
        original_mask = basic_dataset["mask"].values.copy()
        echo_intensity_check(basic_dataset, cutoff=40)
        np.testing.assert_array_equal(basic_dataset["mask"].values, original_mask)

    def test_no_flagging_above_threshold(self, basic_dataset):
        """Test no flagging when all values above threshold."""
        # Dataset has echo_intensity=80, threshold=40
        result = echo_intensity_check(basic_dataset, cutoff=40)
        assert result["mask"].sum() == 0

    def test_flags_below_threshold(self, basic_dataset):
        """Test flagging when any beam is below threshold."""
        basic_dataset["echo_intensity"].values[0, :3, :5] = 20
        result = echo_intensity_check(basic_dataset, cutoff=40)
        assert result["mask"].sum() > 0
        # Entire cell is masked across all beams when any beam fails
        assert result["mask"].isel(cell=0, time=0).values.all()

    def test_flags_at_threshold_boundary(self, basic_dataset):
        """Test behavior at exact threshold value (not flagged) vs one below."""
        basic_dataset["echo_intensity"].values[0, 0, 0] = 40  # At threshold — not flagged
        basic_dataset["echo_intensity"].values[0, 0, 1] = 39  # Below — flagged

        result = echo_intensity_check(basic_dataset, cutoff=40)

        assert result["mask"].isel(cell=0, time=0).values.sum() == 0
        assert result["mask"].isel(cell=0, time=1).values.all()

    def test_custom_threshold(self, basic_dataset):
        """Test with custom threshold."""
        basic_dataset["echo_intensity"].values[:] = 50
        basic_dataset["echo_intensity"].values[0, 0, 0] = 45

        result = echo_intensity_check(basic_dataset, cutoff=60)
        # beam 0 fails at (cell=0, time=0) → all beams at that cell/time masked
        assert result["mask"].isel(cell=0, time=0).values.all()

    def test_creates_mask_if_missing(self, dataset_no_mask):
        """Test that mask is created if not present."""
        result = echo_intensity_check(dataset_no_mask, cutoff=40)
        assert "mask" in result.data_vars

    def test_alternative_variable_name_echo(self, basic_dataset):
        """Test that 'echo' variable name is also accepted."""
        basic_dataset["echo"] = basic_dataset["echo_intensity"]
        del basic_dataset["echo_intensity"]

        basic_dataset["echo"].values[0, 0, 0] = 20
        result = echo_intensity_check(basic_dataset, cutoff=40)
        assert result["mask"].isel(cell=0, time=0).values.all()

    def test_missing_echo_returns_unchanged(self, basic_dataset):
        """Test that missing echo data returns unchanged dataset."""
        del basic_dataset["echo_intensity"]
        result = echo_intensity_check(basic_dataset, cutoff=40)
        assert "mask" in result.data_vars

    def test_all_beams_masked_when_any_fails(self, basic_dataset):
        """When one beam fails, all beams at that cell/time are masked."""
        # Only beam 2 is below threshold at (cell=1, time=2)
        basic_dataset["echo_intensity"].values[2, 1, 2] = 10
        result = echo_intensity_check(basic_dataset, cutoff=40)
        # All four beams at (cell=1, time=2) must be masked
        assert result["mask"].isel(cell=1, time=2).values.all()
        # Other cells untouched
        assert result["mask"].isel(cell=0, time=0).values.sum() == 0

    def test_beam_ignore_excludes_from_count(self, basic_dataset):
        """beam_ignore removes that beam; remaining beams determine flag."""
        # Only beam 1 fails — with beam_ignore=1 it should not be flagged
        basic_dataset["echo_intensity"].values[1, :, :] = 20

        result_no_ignore = echo_intensity_check(basic_dataset, cutoff=40)
        result_ignored = echo_intensity_check(
            basic_dataset, cutoff=40, beam_ignore=1
        )

        assert result_no_ignore["mask"].sum() > 0
        assert result_ignored["mask"].sum() == 0

    def test_per_beam_cutoff_list(self, basic_dataset):
        """Per-beam cutoff list applies different thresholds per beam."""
        # All echo = 80; set beam 2 to 50
        basic_dataset["echo_intensity"].values[2, 0, 0] = 50
        # Cutoff list: beam 2 has threshold 60, others 40
        cutoffs = [40.0, 40.0, 60.0, 40.0]
        result = echo_intensity_check(basic_dataset, cutoff=cutoffs)
        # beam 2 < 60 → all beams at (cell=0, time=0) masked
        assert result["mask"].isel(cell=0, time=0).values.all()
        # Other beams at other cells: beam 2 = 80 > 60, no fail
        assert result["mask"].isel(cell=1, time=0).values.sum() == 0

    def test_per_beam_cutoff_wrong_length_falls_back(self, basic_dataset):
        """Per-beam cutoff with wrong length falls back to cutoff[0]."""
        basic_dataset["echo_intensity"].values[0, 0, 0] = 10
        result = echo_intensity_check(basic_dataset, cutoff=[40, 40])  # wrong length
        # cutoff[0]=40 used for all beams; beam 0 fails → cell masked
        assert result["mask"].isel(cell=0, time=0).values.all()

    def test_preserves_attributes(self, basic_dataset):
        """Test that other data variables retain attributes."""
        result = echo_intensity_check(basic_dataset, cutoff=40)
        assert "long_name" in result["echo_intensity"].attrs


# ============================================================================
# TESTS: error_velocity_check
# ============================================================================


class TestErrorVelocityCheck:
    """Tests for error_velocity_check function."""

    def test_returns_dataset(self, basic_dataset):
        """Test that function returns xarray Dataset."""
        result = error_velocity_check(basic_dataset)
        assert isinstance(result, xr.Dataset)

    def test_preserves_original(self, basic_dataset):
        """Test that original dataset is not modified."""
        original_mask = basic_dataset["mask"].values.copy()
        error_velocity_check(basic_dataset, cutoff=2000)
        np.testing.assert_array_equal(basic_dataset["mask"].values, original_mask)

    def test_no_flagging_below_threshold(self, basic_dataset):
        """Test no flagging when error velocity below threshold."""
        # Velocity values are -500 to 500, well below 2000
        result = error_velocity_check(basic_dataset, cutoff=2000)
        assert result["mask"].sum() == 0

    def test_flags_above_threshold(self, basic_dataset):
        """Test flagging values above threshold."""
        # Set high error velocity on beam 3
        basic_dataset["velocity"].values[3, :, :5] = 3000
        result = error_velocity_check(basic_dataset, cutoff=2000)

        # Should flag beam 3 only
        assert result["mask"].isel(beam=3).sum() > 0
        # Other beams should not be affected
        assert result["mask"].isel(beam=0).sum() == 0

    def test_uses_absolute_value(self, basic_dataset):
        """Test that absolute value is used for comparison."""
        # Set negative error velocity
        basic_dataset["velocity"].values[3, 0, 0] = -3000
        result = error_velocity_check(basic_dataset, cutoff=2000)

        # Should be flagged because |âˆ’3000| > 2000
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1

    def test_flags_at_threshold_boundary(self, basic_dataset):
        """Test behavior at exact threshold value."""
        basic_dataset["velocity"].values[3, 0, 0] = 2000  # At threshold
        basic_dataset["velocity"].values[3, 0, 1] = 2001  # Above threshold

        result = error_velocity_check(basic_dataset, cutoff=2000)

        # 2000 should NOT be flagged (> cutoff, not >=)
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 0
        # 2001 should be flagged
        assert result["mask"].isel(beam=3, cell=0, time=1).values == 1

    def test_custom_threshold(self, basic_dataset):
        """Test with custom threshold."""
        basic_dataset["velocity"].values[3, :, :] = 1500

        # With cutoff=2000, should not flag
        result = error_velocity_check(basic_dataset, cutoff=2000)
        assert result["mask"].sum() == 0

        # With cutoff=1000, should flag
        result = error_velocity_check(basic_dataset, cutoff=1000)
        assert result["mask"].sum() > 0

    def test_only_flags_beam_3(self, basic_dataset):
        """Test that only beam 3 (combined mask) is flagged."""
        # Set high values on all beams
        basic_dataset["velocity"].values[:, 0, 0] = 5000

        result = error_velocity_check(basic_dataset, cutoff=2000)

        # Only beam 3 should be flagged
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1
        assert result["mask"].isel(beam=0, cell=0, time=0).values == 0
        assert result["mask"].isel(beam=1, cell=0, time=0).values == 0
        assert result["mask"].isel(beam=2, cell=0, time=0).values == 0

    def test_creates_mask_if_missing(self, dataset_no_mask):
        """Test that mask is created if not present."""
        result = error_velocity_check(dataset_no_mask, cutoff=2000)
        assert "mask" in result.data_vars

    def test_missing_velocity_returns_unchanged(self, basic_dataset):
        """Test that missing velocity data returns unchanged dataset."""
        del basic_dataset["velocity"]
        result = error_velocity_check(basic_dataset, cutoff=2000)
        assert "mask" in result.data_vars

    def test_insufficient_beams_returns_unchanged(self):
        """Test that dataset with < 4 beams returns unchanged."""
        ds = xr.Dataset(
            {
                "velocity": (
                    ("beam", "cell", "time"),
                    np.random.randn(3, 10, 20),  # Only 3 beams
                ),
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((3, 10, 20), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(3),
                "cell": np.arange(10),
                "time": pd.date_range("2024-01-01", periods=20, freq="h"),
            },
        )
        result = error_velocity_check(ds, cutoff=2000)
        # Should return unchanged
        assert result["mask"].sum() == 0


# ============================================================================
# TESTS: percent_good_check
# ============================================================================


class TestPercentGoodCheck:
    """Tests for percent_good_check function."""

    def test_returns_dataset(self, basic_dataset):
        """Test that function returns xarray Dataset."""
        result = percent_good_check(basic_dataset)
        assert isinstance(result, xr.Dataset)

    def test_preserves_original(self, basic_dataset):
        """Test that original dataset is not modified."""
        original_mask = basic_dataset["mask"].values.copy()
        percent_good_check(basic_dataset, cutoff=50)
        np.testing.assert_array_equal(basic_dataset["mask"].values, original_mask)

    def test_no_flagging_above_threshold(self, basic_dataset):
        """Test no flagging when all values above threshold."""
        # Dataset has percent_good=75, threshold=50
        result = percent_good_check(basic_dataset, cutoff=50)
        assert result["mask"].sum() == 0

    def test_flags_below_threshold(self, basic_dataset):
        """Test flagging values below threshold with threebeam mode."""
        # With threebeam=True (default), PG1 + PG4 is used
        # Set PG1=15, PG4=15, sum=30 < 50
        basic_dataset["percent_good"].values[0, 0, 0] = 15  # PG1
        basic_dataset["percent_good"].values[3, 0, 0] = 15  # PG4

        result = percent_good_check(basic_dataset, cutoff=50)

        # Should flag beam 3 (combined mask) for this location
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1

    def test_flags_at_threshold_boundary(self, basic_dataset):
        """Test behavior at exact threshold value with threebeam mode."""
        # With threebeam=True (default), PG1 + PG4 is used
        # Test case 1: PG1=25, PG4=25, sum=50 (at threshold)
        basic_dataset["percent_good"].values[0, 0, 0] = 25  # PG1
        basic_dataset["percent_good"].values[3, 0, 0] = 25  # PG4

        # Test case 2: PG1=24, PG4=25, sum=49 (below threshold)
        basic_dataset["percent_good"].values[0, 0, 1] = 24  # PG1
        basic_dataset["percent_good"].values[3, 0, 1] = 25  # PG4

        result = percent_good_check(basic_dataset, cutoff=50)

        assert result["mask"].isel(beam=3, cell=0, time=0).values == 0
        assert result["mask"].isel(beam=3, cell=0, time=1).values == 1

    def test_custom_threshold(self, basic_dataset):
        """Test with custom threshold using threebeam mode."""
        # Set PG1=30, PG4=30, sum=60
        basic_dataset["percent_good"].values[:] = 75
        basic_dataset["percent_good"].values[0, :, :] = 30  # PG1
        basic_dataset["percent_good"].values[3, :, :] = 30  # PG4

        # With cutoff=50, sum=60 > 50, should not flag
        result = percent_good_check(basic_dataset, cutoff=50)
        assert result["mask"].sum() == 0

        # With cutoff=70, sum=60 < 70, should flag
        result = percent_good_check(basic_dataset, cutoff=70)
        assert result["mask"].sum() > 0

    def test_method_max(self, basic_dataset):
        """Test 'max' method combines beams correctly."""
        # Set one beam high, others low
        basic_dataset["percent_good"].values[:, 0, 0] = 30
        basic_dataset["percent_good"].values[0, 0, 0] = 80  # One beam high

        result = percent_good_check(basic_dataset, cutoff=50, method="max")

        # Max is 80, above 50, should NOT flag
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 0

    def test_method_min(self, basic_dataset):
        """Test 'min' method combines beams correctly."""
        # Set one beam low, others high
        basic_dataset["percent_good"].values[:, 0, 0] = 80
        basic_dataset["percent_good"].values[0, 0, 0] = 30  # One beam low

        result = percent_good_check(basic_dataset, cutoff=50, method="min")

        # Min is 30, below 50, should flag
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1

    def test_method_mean(self, basic_dataset):
        """Test 'mean' method combines beams correctly."""
        # Set values so mean is 50
        basic_dataset["percent_good"].values[:, 0, 0] = [40, 50, 60, 50]  # Mean = 50

        result = percent_good_check(basic_dataset, cutoff=50, method="mean")
        # Mean is exactly 50, should NOT flag
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 0

        # Change to have mean below 50
        basic_dataset["percent_good"].values[:, 0, 0] = [40, 40, 50, 50]  # Mean = 45
        result = percent_good_check(basic_dataset, cutoff=50, method="mean")
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1

    def test_invalid_method_defaults_to_threebeam(self, basic_dataset):
        """Test that invalid method defaults to threebeam mode."""
        # Set PG1=80, PG4=0, so threebeam sum = 80
        basic_dataset["percent_good"].values[0, 0, 0] = 80  # PG1
        basic_dataset["percent_good"].values[1, 0, 0] = 30
        basic_dataset["percent_good"].values[2, 0, 0] = 30
        basic_dataset["percent_good"].values[3, 0, 0] = 0  # PG4

        result = percent_good_check(basic_dataset, cutoff=50, method="invalid")

        # Should use threebeam (PG1+PG4 = 80), so 80 > 50, not flagged
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 0

    def test_threebeam_true_sums_pg1_pg4(self, basic_dataset):
        """Test threebeam=True sums PG1 and PG4."""
        # PG1 (beam 0) = 30, PG4 (beam 3) = 30, sum = 60
        basic_dataset["percent_good"].values[:, 0, 0] = 10
        basic_dataset["percent_good"].values[0, 0, 0] = 30  # PG1
        basic_dataset["percent_good"].values[3, 0, 0] = 30  # PG4

        result = percent_good_check(basic_dataset, cutoff=50, threebeam=True)

        # Sum = 60 > 50, should NOT flag
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 0

    def test_threebeam_true_flags_when_sum_below_cutoff(self, basic_dataset):
        """Test threebeam=True flags when PG1+PG4 sum is below cutoff."""
        # PG1 (beam 0) = 20, PG4 (beam 3) = 20, sum = 40
        basic_dataset["percent_good"].values[:, 0, 0] = 10
        basic_dataset["percent_good"].values[0, 0, 0] = 20  # PG1
        basic_dataset["percent_good"].values[3, 0, 0] = 20  # PG4

        result = percent_good_check(basic_dataset, cutoff=50, threebeam=True)

        # Sum = 40 < 50, should flag
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1

    def test_threebeam_false_uses_only_pg4(self, basic_dataset):
        """Test threebeam=False uses only PG4."""
        # PG1 = 80, PG4 = 30
        basic_dataset["percent_good"].values[:, 0, 0] = 10
        basic_dataset["percent_good"].values[0, 0, 0] = 80  # PG1 (high)
        basic_dataset["percent_good"].values[3, 0, 0] = 30  # PG4 (low)

        result = percent_good_check(basic_dataset, cutoff=50, threebeam=False)

        # Uses only PG4 = 30 < 50, should flag
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1

    def test_threebeam_false_passes_when_pg4_high(self, basic_dataset):
        """Test threebeam=False passes when PG4 is high."""
        # PG1 = 10, PG4 = 80
        basic_dataset["percent_good"].values[:, 0, 0] = 10
        basic_dataset["percent_good"].values[0, 0, 0] = 10  # PG1 (low)
        basic_dataset["percent_good"].values[3, 0, 0] = 80  # PG4 (high)

        result = percent_good_check(basic_dataset, cutoff=50, threebeam=False)

        # Uses only PG4 = 80 > 50, should NOT flag
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 0

    def test_method_overrides_threebeam(self, basic_dataset):
        """Test that method parameter overrides threebeam parameter."""
        # Set all beams to 30 except beam 2 which is 80
        basic_dataset["percent_good"].values[:, 0, 0] = 30
        basic_dataset["percent_good"].values[2, 0, 0] = 80

        # With method="max", max=80 > 50, should not flag
        result = percent_good_check(
            basic_dataset, cutoff=50, threebeam=True, method="max"
        )
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 0

        # With method="min", min=30 < 50, should flag
        result = percent_good_check(
            basic_dataset, cutoff=50, threebeam=True, method="min"
        )
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1

    def test_only_flags_beam_3(self, basic_dataset):
        """Test that only beam 3 (combined mask) is flagged."""
        # Set PG1=10, PG4=10, sum=20 < 50
        basic_dataset["percent_good"].values[0, 0, 0] = 10  # PG1
        basic_dataset["percent_good"].values[3, 0, 0] = 10  # PG4

        result = percent_good_check(basic_dataset, cutoff=50)

        # Only beam 3 should be flagged
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1
        assert result["mask"].isel(beam=0, cell=0, time=0).values == 0
        assert result["mask"].isel(beam=1, cell=0, time=0).values == 0
        assert result["mask"].isel(beam=2, cell=0, time=0).values == 0

    def test_creates_mask_if_missing(self, dataset_no_mask):
        """Test that mask is created if not present."""
        result = percent_good_check(dataset_no_mask, cutoff=50)
        assert "mask" in result.data_vars

    def test_alternative_variable_name_pg(self, basic_dataset):
        """Test that 'pg' variable name is also accepted."""
        basic_dataset["pg"] = basic_dataset["percent_good"]
        del basic_dataset["percent_good"]

        # Set PG1=10, PG4=10, sum=20 < 50
        basic_dataset["pg"].values[0, 0, 0] = 10  # PG1
        basic_dataset["pg"].values[3, 0, 0] = 10  # PG4
        result = percent_good_check(basic_dataset, cutoff=50)
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1

    def test_missing_percent_good_returns_unchanged(self, basic_dataset):
        """Test that missing percent_good data returns unchanged dataset."""
        del basic_dataset["percent_good"]
        result = percent_good_check(basic_dataset, cutoff=50)
        assert "mask" in result.data_vars

    def test_no_beam_dimension(self):
        """Test handling of data without beam dimension."""
        ds = xr.Dataset(
            {
                "percent_good": (
                    ("cell", "time"),
                    np.full((10, 20), 75, dtype=np.int16),
                ),
                "velocity": (
                    ("beam", "cell", "time"),
                    np.random.randn(4, 10, 20),
                ),
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((4, 10, 20), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(10),
                "time": pd.date_range("2024-01-01", periods=20, freq="h"),
            },
        )
        ds["percent_good"].values[0, 0] = 30

        result = percent_good_check(ds, cutoff=50)
        # Should flag beam 3 at cell 0, time 0
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1


# ============================================================================
# TESTS: false_target_detection
# ============================================================================


class TestFalseTargetDetection:
    """Tests for false_target_detection function."""

    def test_returns_dataset(self, basic_dataset):
        """Test that function returns xarray Dataset."""
        result = false_target_detection(basic_dataset)
        assert isinstance(result, xr.Dataset)

    def test_preserves_original(self, basic_dataset):
        """Test that original dataset is not modified."""
        original_mask = basic_dataset["mask"].values.copy()
        false_target_detection(basic_dataset, cutoff=50)
        np.testing.assert_array_equal(basic_dataset["mask"].values, original_mask)

    def test_no_flagging_uniform_echo(self, basic_dataset):
        """Test no flagging when echo is uniform across beams."""
        # Dataset has uniform echo_intensity=80
        result = false_target_detection(basic_dataset, cutoff=50)
        # Difference is 0, below 50, should not flag
        assert result["mask"].sum() == 0

    def test_flags_large_difference(self, basic_dataset):
        """Test flagging when difference exceeds threshold."""
        # All beams at 80 except beam 0 at 150
        # max - min = 150 - 80 = 70
        basic_dataset["echo_intensity"].values[:, 0, 0] = 80
        basic_dataset["echo_intensity"].values[0, 0, 0] = 150

        result = false_target_detection(basic_dataset, cutoff=50)

        # Difference 70 > 50, should flag beam 3
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1

    def test_flags_at_threshold_boundary(self, basic_dataset):
        """Test behavior at exact threshold value."""
        # All beams at 80, except beam 0 at 130
        basic_dataset["echo_intensity"].values[:, 0, 0] = 80
        basic_dataset["echo_intensity"].values[0, 0, 0] = 130  # Difference = 50

        result = false_target_detection(basic_dataset, cutoff=50)

        # Difference = 50 is NOT > 50, should not flag
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 0

        # Set difference just above threshold
        basic_dataset["echo_intensity"].values[0, 0, 0] = 131  # Difference = 51
        result = false_target_detection(basic_dataset, cutoff=50)
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1

    def test_custom_threshold(self, basic_dataset):
        """Test with custom threshold."""
        # Set values: [100, 90, 80, 70]
        # max - min = 100 - 70 = 30
        basic_dataset["echo_intensity"].values[:, 0, 0] = [100, 90, 80, 70]

        # With cutoff=50, diff=30 < 50, should not flag
        result = false_target_detection(basic_dataset, cutoff=50)
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 0

        # With cutoff=5, diff=30 > 5, should flag
        result = false_target_detection(basic_dataset, cutoff=5)
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1

    def test_only_flags_beam_3(self, basic_dataset):
        """Test that only beam 3 (combined mask) is flagged."""
        # Set one beam very high, others at 80
        # max - min = 200 - 80 = 120
        basic_dataset["echo_intensity"].values[:, 0, 0] = 80
        basic_dataset["echo_intensity"].values[0, 0, 0] = 200

        result = false_target_detection(basic_dataset, cutoff=50)

        # Only beam 3 should be flagged
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1
        assert result["mask"].isel(beam=0, cell=0, time=0).values == 0
        assert result["mask"].isel(beam=1, cell=0, time=0).values == 0
        assert result["mask"].isel(beam=2, cell=0, time=0).values == 0

    def test_creates_mask_if_missing(self, dataset_no_mask):
        """Test that mask is created if not present."""
        result = false_target_detection(dataset_no_mask, cutoff=50)
        assert "mask" in result.data_vars

    def test_alternative_variable_name_echo(self, basic_dataset):
        """Test that 'echo' variable name is also accepted."""
        basic_dataset["echo"] = basic_dataset["echo_intensity"]
        del basic_dataset["echo_intensity"]

        # Set one beam very high, others at default 80
        # max - min = 200 - 80 = 120
        basic_dataset["echo"].values[:, 0, 0] = 80
        basic_dataset["echo"].values[0, 0, 0] = 200

        result = false_target_detection(basic_dataset, cutoff=50)
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1

    def test_missing_echo_returns_unchanged(self, basic_dataset):
        """Test that missing echo data returns unchanged dataset."""
        del basic_dataset["echo_intensity"]
        result = false_target_detection(basic_dataset, cutoff=50)
        assert "mask" in result.data_vars

    def test_insufficient_beams_returns_unchanged(self):
        """Test that dataset with < 2 beams in echo returns unchanged."""
        # Create dataset with echo_intensity that has only 1 beam
        ds = xr.Dataset(
            {
                "echo_intensity": (
                    ("echo_beam", "cell", "time"),
                    np.full((1, 10, 20), 80, dtype=np.int16),  # Only 1 beam
                ),
                "velocity": (
                    ("beam", "cell", "time"),
                    np.random.randn(4, 10, 20),
                ),
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((4, 10, 20), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(4),
                "echo_beam": [0],
                "cell": np.arange(10),
                "time": pd.date_range("2024-01-01", periods=20, freq="h"),
            },
        )
        # echo_intensity has no "beam" dimension, so function should return unchanged
        result = false_target_detection(ds, cutoff=50)
        # Function should handle this gracefully
        assert "mask" in result.data_vars
        # Mask should be unchanged since echo doesn't have beam dim
        assert result["mask"].sum() == 0

    def test_beam_ignore_excludes_beam(self, basic_dataset):
        """Test that beam_ignore excludes specified beam from comparison."""
        # Set values with one extreme outlier on beam 0
        basic_dataset["echo_intensity"].values[:, 0, 0] = [200, 80, 85, 90]
        # Without ignore: max=200, min=80, diff=120
        # With beam 0 ignored: max=90, min=80, diff=10

        # Without beam_ignore, diff=120 > 50, should flag
        result_normal = false_target_detection(basic_dataset, cutoff=50)
        assert result_normal["mask"].isel(beam=3, cell=0, time=0).values == 1

        # With beam_ignore=0, diff=10 < 50, should NOT flag
        result_ignore = false_target_detection(
            basic_dataset, cutoff=50, beam_ignore=0
        )
        assert result_ignore["mask"].isel(beam=3, cell=0, time=0).values == 0

    def test_beam_ignore_invalid_index(self, basic_dataset):
        """Test that invalid beam_ignore index is ignored."""
        basic_dataset["echo_intensity"].values[:, 0, 0] = [150, 100, 90, 50]

        # beam_ignore=5 is invalid (only 0-3 valid), should be ignored
        result = false_target_detection(
            basic_dataset, cutoff=60, beam_ignore=5
        )
        # Should still use all beams, diff = 150-50 = 100 > 60, flagged
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1

    def test_beam_ignore_with_different_beams(self, basic_dataset):
        """Test beam_ignore works correctly for each beam index."""
        # Set distinct values on each beam
        basic_dataset["echo_intensity"].values[:, 0, 0] = [100, 80, 60, 40]

        # Ignore beam 0: remaining = [80, 60, 40], diff = 40
        result = false_target_detection(
            basic_dataset, cutoff=30, beam_ignore=0
        )
        # 40 > 30, should flag
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1

        # Ignore beam 3: remaining = [100, 80, 60], diff = 40
        result = false_target_detection(
            basic_dataset, cutoff=50, beam_ignore=3
        )
        # 40 < 50, should NOT flag
        assert result["mask"].isel(beam=3, cell=0, time=0).values == 0


# ============================================================================
# TESTS: Default Constants
# ============================================================================


class TestDefaultConstants:
    """Test that default constants have expected values."""

    def test_default_correlation_threshold_value(self):
        """Test DEFAULT_CORRELATION_THRESHOLD is 64."""
        assert DEFAULT_CORRELATION_THRESHOLD == 64

    def test_default_echo_threshold_value(self):
        """Test DEFAULT_ECHO_THRESHOLD is 40."""
        assert DEFAULT_ECHO_THRESHOLD == 40

    def test_default_error_velocity_threshold_value(self):
        """Test DEFAULT_ERROR_VELOCITY_THRESHOLD is 2000 mm/s."""
        assert DEFAULT_ERROR_VELOCITY_THRESHOLD == 2000

    def test_default_percent_good_threshold_value(self):
        """Test DEFAULT_PERCENT_GOOD_THRESHOLD is 50."""
        assert DEFAULT_PERCENT_GOOD_THRESHOLD == 50

    def test_default_false_target_threshold_value(self):
        """Test DEFAULT_FALSE_TARGET_THRESHOLD is 50."""
        assert DEFAULT_FALSE_TARGET_THRESHOLD == 50

    def test_threshold_ranges_correlation(self):
        """Test correlation threshold range is 0-255."""
        assert "correlation" in THRESHOLD_RANGES
        assert THRESHOLD_RANGES["correlation"] == (0, 255)

    def test_threshold_ranges_echo_intensity(self):
        """Test echo intensity threshold range is 0-255."""
        assert "echo_intensity" in THRESHOLD_RANGES
        assert THRESHOLD_RANGES["echo_intensity"] == (0, 255)

    def test_threshold_ranges_error_velocity(self):
        """Test error velocity threshold range is 0-5000."""
        assert "error_velocity" in THRESHOLD_RANGES
        assert THRESHOLD_RANGES["error_velocity"] == (0, 5000)

    def test_threshold_ranges_percent_good(self):
        """Test percent good threshold range is 0-100."""
        assert "percent_good" in THRESHOLD_RANGES
        assert THRESHOLD_RANGES["percent_good"] == (0, 100)

    def test_threshold_ranges_false_target(self):
        """Test false target threshold range is 0-255."""
        assert "false_target" in THRESHOLD_RANGES
        assert THRESHOLD_RANGES["false_target"] == (0, 255)

    def test_all_threshold_ranges_are_tuples(self):
        """Test all threshold ranges are tuples of two integers."""
        for name, range_tuple in THRESHOLD_RANGES.items():
            assert isinstance(range_tuple, tuple), f"{name} range is not a tuple"
            assert len(range_tuple) == 2, f"{name} range doesn't have 2 elements"
            assert isinstance(range_tuple[0], int), f"{name} min is not int"
            assert isinstance(range_tuple[1], int), f"{name} max is not int"
            assert range_tuple[0] < range_tuple[1], f"{name} min >= max"


# ============================================================================
# TESTS: Default Thresholds
# ============================================================================


class TestDefaultThresholds:
    """Test that default thresholds are applied correctly."""

    def test_correlation_default_threshold(self, basic_dataset):
        """Test correlation uses default threshold of 64."""
        basic_dataset["correlation"].values[0, 0, 0] = 63

        result = correlation_check(basic_dataset)  # No cutoff specified

        assert result["mask"].isel(beam=0, cell=0, time=0).values == 1

    def test_echo_intensity_default_threshold(self, basic_dataset):
        """Test echo_intensity uses default threshold of 40."""
        basic_dataset["echo_intensity"].values[0, 0, 0] = 39

        result = echo_intensity_check(basic_dataset)

        assert result["mask"].isel(beam=0, cell=0, time=0).values == 1

    def test_error_velocity_default_threshold(self, basic_dataset):
        """Test error_velocity uses default threshold of 2000."""
        basic_dataset["velocity"].values[3, 0, 0] = 2001

        result = error_velocity_check(basic_dataset)

        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1

    def test_percent_good_default_threshold(self, basic_dataset):
        """Test percent_good uses default threshold of 50 with threebeam mode."""
        # With threebeam=True (default), PG1 + PG4 is used
        # Set PG1=20, PG4=20, sum=40 < 50
        basic_dataset["percent_good"].values[0, 0, 0] = 20  # PG1
        basic_dataset["percent_good"].values[3, 0, 0] = 20  # PG4

        result = percent_good_check(basic_dataset)

        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1

    def test_false_target_default_threshold(self, basic_dataset):
        """Test false_target uses default threshold of 50."""
        # Set values: beam0=131, beam1=80, beam2=80, beam3=80
        # max - min = 131 - 80 = 51
        basic_dataset["echo_intensity"].values[0, 0, 0] = 131
        basic_dataset["echo_intensity"].values[1, 0, 0] = 80
        basic_dataset["echo_intensity"].values[2, 0, 0] = 80
        basic_dataset["echo_intensity"].values[3, 0, 0] = 80

        result = false_target_detection(basic_dataset)

        assert result["mask"].isel(beam=3, cell=0, time=0).values == 1


# ============================================================================
# TESTS: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        ds = xr.Dataset(
            {
                "velocity": (("beam", "cell", "time"), np.array([]).reshape(4, 0, 0)),
                "correlation": (
                    ("beam", "cell", "time"),
                    np.array([]).reshape(4, 0, 0),
                ),
                "mask": (("beam", "cell", "time"), np.array([]).reshape(4, 0, 0)),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.array([]),
                "time": pd.DatetimeIndex([]),
            },
        )
        # Should not raise error
        result = correlation_check(ds, cutoff=64)
        assert isinstance(result, xr.Dataset)

    def test_single_cell_dataset(self):
        """Test handling of single-cell dataset."""
        ds = xr.Dataset(
            {
                "velocity": (("beam", "cell", "time"), np.zeros((4, 1, 1))),
                "correlation": (("beam", "cell", "time"), np.full((4, 1, 1), 100)),
                "echo_intensity": (("beam", "cell", "time"), np.full((4, 1, 1), 80)),
                "percent_good": (("beam", "cell", "time"), np.full((4, 1, 1), 75)),
                "mask": (("beam", "cell", "time"), np.zeros((4, 1, 1), dtype=np.int8)),
            },
            coords={
                "beam": np.arange(4),
                "cell": [0],
                "time": pd.date_range("2024-01-01", periods=1, freq="h"),
            },
        )
        result = correlation_check(ds, cutoff=64)
        assert isinstance(result, xr.Dataset)
        assert result["mask"].shape == (4, 1, 1)

    def test_all_values_flagged(self, basic_dataset):
        """Test when all values should be flagged."""
        basic_dataset["correlation"].values[:] = 10  # All below 64

        result = correlation_check(basic_dataset, cutoff=64)

        # All cells should be flagged
        assert result["mask"].sum() == result["mask"].size

    def test_preserves_coordinates(self, basic_dataset):
        """Test that coordinates are preserved."""
        result = correlation_check(basic_dataset, cutoff=64)

        assert "time" in result.coords
        assert "cell" in result.coords
        assert "beam" in result.coords
        np.testing.assert_array_equal(
            result.coords["time"], basic_dataset.coords["time"]
        )

    def test_preserves_other_variables(self, basic_dataset):
        """Test that other data variables are preserved."""
        result = correlation_check(basic_dataset, cutoff=64)

        assert "velocity" in result.data_vars
        assert "echo_intensity" in result.data_vars
        assert "percent_good" in result.data_vars

    def test_large_dataset_performance(self):
        """Test that functions handle larger datasets."""
        np.random.seed(42)
        n_time = 1000
        n_cell = 100
        n_beam = 4

        ds = xr.Dataset(
            {
                "velocity": (
                    ("beam", "cell", "time"),
                    np.random.randint(-500, 500, size=(n_beam, n_cell, n_time)),
                ),
                "correlation": (
                    ("beam", "cell", "time"),
                    np.random.randint(50, 200, size=(n_beam, n_cell, n_time)),
                ),
                "echo_intensity": (
                    ("beam", "cell", "time"),
                    np.random.randint(30, 150, size=(n_beam, n_cell, n_time)),
                ),
                "percent_good": (
                    ("beam", "cell", "time"),
                    np.random.randint(40, 100, size=(n_beam, n_cell, n_time)),
                ),
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((n_beam, n_cell, n_time), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(n_beam),
                "cell": np.arange(n_cell),
                "time": pd.date_range("2024-01-01", periods=n_time, freq="h"),
            },
        )

        # Should complete without error
        result = correlation_check(ds, cutoff=64)
        assert result["mask"].shape == (n_beam, n_cell, n_time)


# ============================================================================
# TESTS: Validate threshold helper (line 77)
# ============================================================================


class TestValidateThresholdWarning:
    """Test _validate_threshold helper function."""

    def test_out_of_range_threshold_logs_warning(self, caplog):
        """Out-of-range threshold triggers logger.warning (line 77)."""
        import logging

        with caplog.at_level(
            logging.WARNING, logger="pyadps.processing.signal_quality"
        ):
            _validate_threshold("percent_good", 150)  # max is 100

        assert any("out of range" in msg for msg in caplog.messages)

    def test_in_range_threshold_no_warning(self, caplog):
        """Threshold within range emits no warning."""
        import logging

        with caplog.at_level(
            logging.WARNING, logger="pyadps.processing.signal_quality"
        ):
            _validate_threshold("percent_good", 50)

        assert not any("out of range" in msg for msg in caplog.messages)

    def test_unknown_threshold_name_no_warning(self, caplog):
        """Threshold name not in THRESHOLD_RANGES emits no warning."""
        import logging

        with caplog.at_level(
            logging.WARNING, logger="pyadps.processing.signal_quality"
        ):
            _validate_threshold("unknown_check", 999)

        assert not any("out of range" in msg for msg in caplog.messages)


# ============================================================================
# TESTS: percent_good_check edge-case branches (lines 340 and 355)
# ============================================================================


class TestPercentGoodEdgeCases:
    """Test percent_good_check branches for no-beam-dim and <4-beam inputs."""

    def test_method_provided_but_no_beam_dim_line_340(self):
        """Line 340: method given but percent_good has no beam dim -> pgood_combined = pgood."""
        times = pd.date_range("2024-01-01", periods=10, freq="h")
        cells = np.arange(5)

        # percent_good has no beam dimension; mask and velocity do
        ds = xr.Dataset(
            {
                "percent_good": (("cell", "time"), np.ones((5, 10)) * 80),
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((4, 5, 10), dtype=np.int8),
                ),
                "velocity": (
                    ("beam", "cell", "time"),
                    np.random.randn(4, 5, 10),
                ),
            },
            coords={"beam": np.arange(4), "cell": cells, "time": times},
        )

        # method="max" provided but no beam dim in percent_good -> line 340
        result = percent_good_check(ds, cutoff=50, method="max")

        assert isinstance(result, xr.Dataset)
        assert int(result["mask"].sum()) == 0  # all values 80 > 50

    def test_fewer_than_4_beams_uses_max_fallback_line_355(self):
        """Line 355: percent_good with <4 beams and method=None falls back to max."""
        times = pd.date_range("2024-01-01", periods=10, freq="h")
        cells = np.arange(5)

        # Both mask and percent_good share the same 2-beam coord so xarray
        # doesn't upcast percent_good's beam size via a shared dataset dimension.
        ds = xr.Dataset(
            {
                "percent_good": (
                    ("beam", "cell", "time"),
                    np.ones((2, 5, 10)) * 80,
                ),
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((2, 5, 10), dtype=np.int8),
                ),
            },
            coords={"beam": np.arange(2), "cell": cells, "time": times},
        )

        # method=None and pgood.sizes["beam"]==2 < 4 -> line 355 (max fallback)
        result = percent_good_check(ds, cutoff=50, method=None)

        assert isinstance(result, xr.Dataset)


# ============================================================================
# TESTS: Smoke tests confirming standalone functions work end-to-end
# ============================================================================


class TestCoreFunctionsSmoke:
    """Smoke tests for standalone QC functions using the basic_dataset fixture.

    These complement the detailed per-function tests above by verifying that
    each function flags bad data and preserves the original dataset.
    """

    def test_correlation_check_flags_low_values(self, basic_dataset):
        """correlation_check flags cells below the cutoff."""
        basic_dataset["correlation"].values[0, :5, :10] = 30  # below 64
        ds_out = correlation_check(basic_dataset, cutoff=64)
        assert ds_out["mask"].sum() > 0

    def test_echo_intensity_check_flags_low_values(self, basic_dataset):
        """echo_intensity_check flags cells below the cutoff."""
        basic_dataset["echo_intensity"].values[1, :3, :5] = 20  # below 40
        ds_out = echo_intensity_check(basic_dataset, cutoff=40)
        assert ds_out["mask"].sum() > 0

    def test_error_velocity_check_flags_high_values(self, basic_dataset):
        """error_velocity_check flags cells above the cutoff."""
        basic_dataset["velocity"].values[3, :, :10] = 3000  # above 2000
        ds_out = error_velocity_check(basic_dataset, cutoff=2000)
        assert ds_out["mask"].sum() > 0

    def test_percent_good_check_flags_low_values(self, basic_dataset):
        """percent_good_check flags cells below the cutoff."""
        basic_dataset["percent_good"].values[0, :2, :3] = 20  # PG1
        basic_dataset["percent_good"].values[3, :2, :3] = 20  # PG4, sum=40 < 50
        ds_out = percent_good_check(basic_dataset, cutoff=50)
        assert ds_out["mask"].sum() > 0

    def test_all_functions_return_dataset(self, basic_dataset):
        """All standalone functions return xr.Dataset."""
        assert isinstance(correlation_check(basic_dataset), xr.Dataset)
        assert isinstance(echo_intensity_check(basic_dataset), xr.Dataset)
        assert isinstance(error_velocity_check(basic_dataset), xr.Dataset)
        assert isinstance(percent_good_check(basic_dataset), xr.Dataset)
        assert isinstance(false_target_detection(basic_dataset), xr.Dataset)

    def test_functions_do_not_modify_input(self, basic_dataset):
        """Standalone functions leave the input dataset unchanged."""
        original_mask = basic_dataset["mask"].values.copy()

        correlation_check(basic_dataset, cutoff=64)
        echo_intensity_check(basic_dataset, cutoff=40)

        np.testing.assert_array_equal(basic_dataset["mask"].values, original_mask)


# ===========================================================================
# Dimension-order preservation in echo_intensity_check
#
# xr.where(cell_flag, 1, mask) — where cell_flag has dims (cell, time) and
# mask has dims (beam, cell, time) — used to produce output with dims
# (cell, time, beam). The fix adds .transpose(*mask.dims) to restore the
# original (beam, cell, time) order.
# ===========================================================================


class TestEchoIntensityCheckDimOrder:
    """Output mask must preserve (beam, cell, time) dim order after the fix."""

    def _make_ds(self, n_beam=4, n_cell=10, n_time=20, cutoff=40):
        """Dataset where all echo values exceed cutoff so the cell-flag path fires."""
        times = pd.date_range("2024-01-01", periods=n_time, freq="h")
        cells = np.arange(n_cell)
        beams = np.arange(n_beam)
        echo = np.full((n_beam, n_cell, n_time), cutoff + 10, dtype=np.int16)
        mask = np.zeros((n_beam, n_cell, n_time), dtype=np.int8)
        return xr.Dataset(
            {
                "echo_intensity": (("beam", "cell", "time"), echo),
                "mask": (("beam", "cell", "time"), mask),
            },
            coords={"time": times, "cell": cells, "beam": beams},
        )

    def _make_ds_with_failing_cells(self, n_beam=4, n_cell=10, n_time=20):
        """Dataset where first 3 cells fail the echo threshold (trigger cell_flag)."""
        times = pd.date_range("2024-01-01", periods=n_time, freq="h")
        cells = np.arange(n_cell)
        beams = np.arange(n_beam)
        echo = np.full((n_beam, n_cell, n_time), 80, dtype=np.int16)
        echo[:, :3, :] = 20  # below threshold=40 → cell_flag fires
        mask = np.zeros((n_beam, n_cell, n_time), dtype=np.int8)
        return xr.Dataset(
            {
                "echo_intensity": (("beam", "cell", "time"), echo),
                "mask": (("beam", "cell", "time"), mask),
            },
            coords={"time": times, "cell": cells, "beam": beams},
        )

    def test_output_mask_dims_match_input_all_pass(self, basic_dataset):
        """All values pass — mask dims unchanged."""
        result = echo_intensity_check(basic_dataset, cutoff=40)
        assert tuple(result["mask"].dims) == ("beam", "cell", "time")

    def test_output_mask_dims_match_input_with_cell_flag(self):
        """Some cells fail → cell_flag fires the xr.where path; dims must stay correct."""
        ds = self._make_ds_with_failing_cells()
        result = echo_intensity_check(ds, cutoff=40)
        assert tuple(result["mask"].dims) == ("beam", "cell", "time")

    def test_output_shape_correct_with_cell_flag(self):
        """Shape must be (beam, cell, time) — was (cell, time, beam) before fix."""
        n_beam, n_cell, n_time = 4, 10, 20
        ds = self._make_ds_with_failing_cells(n_beam=n_beam, n_cell=n_cell, n_time=n_time)
        result = echo_intensity_check(ds, cutoff=40)
        assert result["mask"].shape == (n_beam, n_cell, n_time)

    def test_cell_flag_masks_all_beams_in_flagged_cells(self):
        """Cells that fail must be masked across all beams (cell-level flag)."""
        ds = self._make_ds_with_failing_cells()
        result = echo_intensity_check(ds, cutoff=40)
        mask = result["mask"].values  # (beam, cell, time)
        # First 3 cells are below threshold — all 4 beams should be masked
        assert np.all(mask[:, :3, :] == 1), (
            "All beams in failing cells must be masked"
        )
        # Cells 3+ are above threshold — should remain 0
        assert np.all(mask[:, 3:, :] == 0), (
            "Passing cells must not be masked"
        )

    def test_per_beam_threshold_list_dims_preserved(self):
        """Per-beam list threshold also preserves (beam, cell, time) dim order."""
        ds = self._make_ds_with_failing_cells()
        per_beam_cutoffs = [40.0, 40.0, 40.0, 40.0]
        result = echo_intensity_check(ds, cutoff=per_beam_cutoffs)
        assert tuple(result["mask"].dims) == ("beam", "cell", "time")
        assert result["mask"].shape == (4, 10, 20)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
