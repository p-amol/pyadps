"""
Test suite for roll_check function.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr

from pyadps.processing.sensor_health import roll_check, DEFAULT_ROLL_THRESHOLD
from pyadps.processing.utility import create_default_mask


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def adcp_dummy_data(
    n_time=10,
    n_cell=20,
    n_beam=4,
):
    """Create a simple ADCP xarray.Dataset for testing with sensor data."""

    times = pd.date_range("2024-01-01 00:00:00", periods=n_time, freq="h")
    cells = np.arange(n_cell)
    beams = np.arange(n_beam)

    # Create realistic ADCP data with sensor variables
    data_vars = {
        "velocity": (
            ("beam", "cell", "time"),
            np.random.randn(n_beam, n_cell, n_time) * 100,  # mm/s
        ),
        "roll": (
            ("time",),
            np.random.uniform(0, 1500, n_time),  # 0.01 degree units
        ),
        "pitch": (
            ("time",),
            np.random.uniform(0, 1500, n_time),  # 0.01 degree units
        ),
        "pressure": (
            ("time",),
            np.random.uniform(500, 50000, n_time),  # 0.1 dbar units
        ),
        "temperature": (
            ("time",),
            np.random.uniform(-200, 3500, n_time),  # 0.01°C units
        ),
        "sound_speed": (
            ("time",),
            np.random.uniform(1400, 1550, n_time),  # m/s
        ),
        "salinity": (
            ("time",),
            np.random.uniform(340, 360, n_time),  # 0.1 PSU units
        ),
        "depth_of_transducer": (
            ("time",),
            np.random.uniform(500, 50000, n_time),  # 0.1 m units
        ),
    }

    coords = {
        "time": times,
        "cell": cells,
        "beam": beams,
    }

    ds = xr.Dataset(data_vars, coords=coords)
    ds.coords["time"].attrs.update({"axis": "T"})
    ds.coords["cell"].attrs.update({"axis": "Z"})
    ds.coords["beam"].attrs.update({"axis": "X"})

    return ds


@pytest.fixture
def adcp_dataset_with_mask():
    """Create ADCP dataset with velocity, roll, and mask."""
    n_beam = 4
    n_cell = 20
    n_time = 50

    times = pd.date_range("2024-01-01 00:00:00", periods=n_time, freq="h")
    # Create velocity data (no missing values)
    velocity = np.random.randn(n_beam, n_cell, n_time) * 100

    # Create roll data in 0.01 degree units (±5 degrees = ±500 raw)
    roll = np.random.uniform(-500, 500, n_time)

    # Create initial mask (all zeros = valid)
    mask = np.zeros((n_beam, n_cell, n_time), dtype=np.int8)

    ds = xr.Dataset(
        {
            "velocity": (("beam", "cell", "time"), velocity),
            "roll": (("time",), roll),
            "mask": (("beam", "cell", "time"), mask),
        },
        coords={
            "beam": np.arange(n_beam),
            "cell": np.arange(n_cell),
            "time": times,
        },
    )
    return ds


@pytest.fixture
def adcp_dataset_without_mask():
    """Create ADCP dataset with velocity and roll, but no mask."""
    n_beam = 4
    n_cell = 20
    n_time = 50

    times = pd.date_range("2024-01-01 00:00:00", periods=n_time, freq="h")

    velocity = np.random.randn(n_beam, n_cell, n_time) * 100
    roll = np.random.uniform(-500, 500, n_time)

    ds = xr.Dataset(
        {
            "velocity": (("beam", "cell", "time"), velocity),
            "roll": (("time",), roll),
        },
        coords={
            "beam": np.arange(n_beam),
            "cell": np.arange(n_cell),
            "time": times,
        },
    )
    return ds


@pytest.fixture
def adcp_dataset_with_extreme_roll():
    """Create dataset with some extreme roll values."""
    n_beam = 4
    n_cell = 20
    n_time = 50

    times = pd.date_range("2024-01-01 00:00:00", periods=n_time, freq="h")

    velocity = np.random.randn(n_beam, n_cell, n_time) * 100

    # Roll with specific extreme values
    roll = np.ones(n_time) * 500  # 5 degrees (normal)
    roll[10:15] = 2000  # 20 degrees (should fail 15° threshold)
    roll[30:35] = 2500  # 25 degrees (should fail)

    mask = np.zeros((n_beam, n_cell, n_time), dtype=np.int8)

    ds = xr.Dataset(
        {
            "velocity": (("beam", "cell", "time"), velocity),
            "roll": (("time",), roll),
            "mask": (("beam", "cell", "time"), mask),
        },
        coords={
            "beam": np.arange(n_beam),
            "cell": np.arange(n_cell),
            "time": times,
        },
    )
    return ds


@pytest.fixture
def adcp_dataset_no_roll():
    """Create dataset without roll data."""
    n_beam = 4
    n_cell = 20
    n_time = 50

    times = pd.date_range("2024-01-01 00:00:00", periods=n_time, freq="h")

    velocity = np.random.randn(n_beam, n_cell, n_time) * 100
    mask = np.zeros((n_beam, n_cell, n_time), dtype=np.int8)

    ds = xr.Dataset(
        {
            "velocity": (("beam", "cell", "time"), velocity),
            "mask": (("beam", "cell", "time"), mask),
        },
        coords={
            "beam": np.arange(n_beam),
            "cell": np.arange(n_cell),
            "time": times,
        },
    )
    return ds


@pytest.fixture
def adcp_dataset_no_mask_no_velocity():
    """Create dataset with roll but no mask or velocity."""
    n_time = 50
    times = pd.date_range("2024-01-01 00:00:00", periods=n_time, freq="h")
    roll = np.random.uniform(-500, 500, n_time)

    ds = xr.Dataset(
        {"roll": (("time",), roll)},
        coords={"time": times},
    )
    return ds


# ============================================================================
# TESTS: Basic Functionality
# ============================================================================


class TestRollCheckBasic:
    """Test basic roll_check functionality."""

    def test_returns_dataset(self, adcp_dataset_with_mask):
        """Test that roll_check returns xr.Dataset."""
        result = roll_check(adcp_dataset_with_mask)
        assert isinstance(result, xr.Dataset)

    def test_returns_correct_shape(self, adcp_dataset_with_mask):
        """Test mask inside dataset has correct 3D shape."""
        result = roll_check(adcp_dataset_with_mask)
        assert result["mask"].shape == (4, 20, 50)  # (beam, cell, ensemble)

    def test_returns_correct_dims(self, adcp_dataset_with_mask):
        """Test mask inside dataset has correct dimensions."""
        result = roll_check(adcp_dataset_with_mask)
        assert result["mask"].dims == ("beam", "cell", "time")

    def test_returns_int8_dtype(self, adcp_dataset_with_mask):
        """Test mask inside dataset uses int8 dtype."""
        result = roll_check(adcp_dataset_with_mask)
        assert result["mask"].dtype == np.int8

    def test_mask_values_are_0_or_1(self, adcp_dataset_with_mask):
        """Test mask inside dataset contains only 0 and 1."""
        result = roll_check(adcp_dataset_with_mask)
        unique_values = np.unique(result["mask"].values)
        assert all(v in [0, 1] for v in unique_values)


# ============================================================================
# TESTS: Roll Check Logic
# ============================================================================


class TestRollCheckLogic:
    """Test roll_check threshold logic."""

    def test_flags_high_roll_values(self, adcp_dataset_with_extreme_roll):
        """Test that high roll values are flagged."""
        result = roll_check(adcp_dataset_with_extreme_roll, threshold=15.0)

        # Ensembles 10-14 have roll=20° (>15°), should be flagged
        # Ensembles 30-34 have roll=25° (>15°), should be flagged
        for ens in [10, 11, 12, 13, 14, 30, 31, 32, 33, 34]:
            # All cells in flagged ensemble should be 1
            assert result["mask"].isel(beam=3, time=ens).values.all() == 1

    def test_does_not_flag_normal_roll(self, adcp_dataset_with_extreme_roll):
        """Test that normal roll values are not flagged."""
        result = roll_check(adcp_dataset_with_extreme_roll, threshold=15.0)

        # Ensemble 0 has roll=5° (<15°), should NOT be flagged
        assert result["mask"].isel(beam=3, time=0).values.sum() == 0

    def test_threshold_boundary(self, adcp_dataset_with_extreme_roll):
        """Test threshold exactly at boundary."""
        # Ensembles 10-14 have exactly 20° roll
        # With threshold=20°, they should NOT be flagged (> not >=)
        result = roll_check(adcp_dataset_with_extreme_roll, threshold=20.0)

        # 20° is NOT > 20°, so should not be flagged
        assert result["mask"].isel(beam=3, time=10).values.sum() == 0

    def test_all_beams_flagged_together(self, adcp_dataset_with_extreme_roll):
        """Test that all 4 beams are flagged for failed ensembles."""
        result = roll_check(adcp_dataset_with_extreme_roll, threshold=15.0)

        # For ensemble 10 (flagged), all beams should be 1
        for beam in range(4):
            assert result["mask"].isel(beam=beam, time=10).values.all() == 1

    def test_all_cells_flagged_in_ensemble(self, adcp_dataset_with_extreme_roll):
        """Test that all cells are flagged when ensemble fails."""
        result = roll_check(adcp_dataset_with_extreme_roll, threshold=15.0)

        # All 20 cells in ensemble 10 should be flagged
        assert result["mask"].isel(beam=3, time=10).sum() == 20


# ============================================================================
# TESTS: Mask Handling
# ============================================================================


class TestRollCheckMaskHandling:
    """Test mask creation and handling."""

    def test_uses_existing_mask(self, adcp_dataset_with_mask):
        """Test that existing mask is used."""
        # Pre-flag some cells
        adcp_dataset_with_mask["mask"].values[3, 0:5, 0] = 1

        result = roll_check(adcp_dataset_with_mask, threshold=50.0)

        # With high threshold, no new flags, but existing should remain
        assert result["mask"].isel(beam=3, cell=slice(0, 5), time=0).sum() == 5

    def test_creates_mask_from_velocity(self, adcp_dataset_without_mask):
        """Test mask is created from velocity when not present."""
        result = roll_check(adcp_dataset_without_mask, threshold=50.0)

        # Should return a dataset with a mask variable
        assert isinstance(result, xr.Dataset)
        assert "mask" in result.data_vars
        assert result["mask"].shape == (4, 20, 50)

    def test_original_mask_not_modified(self, adcp_dataset_with_mask):
        """Test that original dataset mask is not modified."""
        original_mask_sum = adcp_dataset_with_mask["mask"].sum().item()

        _ = roll_check(adcp_dataset_with_mask, threshold=5.0)  # Flag many

        # Original should be unchanged
        assert adcp_dataset_with_mask["mask"].sum().item() == original_mask_sum


# ============================================================================
# TESTS: Error Handling
# ============================================================================


class TestRollCheckErrors:
    """Test error handling."""

    def test_raises_without_roll(self, adcp_dataset_no_roll):
        """Test ValueError when roll data is missing."""
        with pytest.raises(ValueError, match="Roll data not found"):
            roll_check(adcp_dataset_no_roll)

    def test_raises_without_mask_or_velocity(self, adcp_dataset_no_mask_no_velocity):
        """Test ValueError when neither mask nor velocity exists."""
        with pytest.raises(ValueError, match="neither 'mask' nor 'velocity'"):
            roll_check(adcp_dataset_no_mask_no_velocity)


# ============================================================================
# TESTS: Unit Conversion
# ============================================================================


class TestRollCheckUnitConversion:
    """Test roll unit conversion (always scales by 0.01)."""

    def test_converts_raw_units(self):
        """Test that raw 0.01 degree units are converted."""
        n_beam, n_cell, n_ensemble = 4, 10, 20

        # Roll in raw units: 1500 = 15 degrees
        roll_raw = np.ones(n_ensemble) * 1500

        ds = xr.Dataset(
            {
                "velocity": (
                    ("beam", "cell", "time"),
                    np.zeros((n_beam, n_cell, n_ensemble)),
                ),
                "roll": (("time",), roll_raw),
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((n_beam, n_cell, n_ensemble), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(n_beam),
                "cell": np.arange(n_cell),
                "time": np.arange(n_ensemble),
            },
        )

        # Threshold 20° should NOT flag 15° roll
        result = roll_check(ds, threshold=20.0)
        assert result["mask"].sum() == 0

        # Threshold 10° SHOULD flag 15° roll
        result = roll_check(ds, threshold=10.0)
        assert result["mask"].sum() > 0

    def test_scaling_always_applied(self):
        """Test that 0.01 scaling is always applied to raw values."""
        n_beam, n_cell, n_ensemble = 4, 10, 20

        # Roll = 500 raw units = 5 degrees after scaling
        roll_raw = np.ones(n_ensemble) * 500

        ds = xr.Dataset(
            {
                "velocity": (
                    ("beam", "cell", "time"),
                    np.zeros((n_beam, n_cell, n_ensemble)),
                ),
                "roll": (("time",), roll_raw),
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((n_beam, n_cell, n_ensemble), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(n_beam),
                "cell": np.arange(n_cell),
                "time": np.arange(n_ensemble),
            },
        )

        # 5° < 10° threshold, should NOT flag
        result = roll_check(ds, threshold=10.0)
        assert result["mask"].sum() == 0

        # 5° > 3° threshold, SHOULD flag
        result = roll_check(ds, threshold=3.0)
        assert result["mask"].sum() > 0


# ============================================================================
# TESTS: Edge Cases
# ============================================================================


class TestRollCheckEdgeCases:
    """Test edge cases."""

    def test_empty_dataset(self):
        """Test with minimal dataset."""
        ds = xr.Dataset(
            {
                "velocity": (("beam", "cell", "time"), np.zeros((4, 5, 10))),
                "roll": (("time",), np.zeros(10)),
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((4, 5, 10), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(5),
                "time": np.arange(10),
            },
        )
        result = roll_check(ds)
        assert result["mask"].shape == (4, 5, 10)

    def test_single_ensemble(self):
        """Test with single ensemble."""
        ds = xr.Dataset(
            {
                "velocity": (("beam", "cell", "time"), np.zeros((4, 10, 1))),
                "roll": (("time",), np.array([500])),  # 5 degrees
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((4, 10, 1), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(10),
                "time": np.arange(1),
            },
        )
        result = roll_check(ds, threshold=10.0)
        assert result["mask"].shape == (4, 10, 1)
        assert result["mask"].sum() == 0  # 5° < 10°

    def test_very_strict_threshold(self, adcp_dataset_with_mask):
        """Test very strict threshold flags most data."""
        result = roll_check(adcp_dataset_with_mask, threshold=0.1)
        # Most should be flagged
        flagged_pct = result["mask"].sum() / result["mask"].size * 100
        assert flagged_pct > 50

    def test_very_lenient_threshold(self, adcp_dataset_with_mask):
        """Test lenient threshold flags nothing."""
        result = roll_check(adcp_dataset_with_mask, threshold=180.0)
        assert result["mask"].sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
