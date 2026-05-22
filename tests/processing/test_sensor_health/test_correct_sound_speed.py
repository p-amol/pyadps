"""
Test suite for sound speed correction function.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr

from pyadps.processing.sensor_health import correct_sound_speed


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def adcp_dataset():
    """Create ADCP dataset with all required variables for sound speed correction."""
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
        "roll": (("time",), np.random.uniform(-500, 500, n_time)),
        "pitch": (("time",), np.random.uniform(-500, 500, n_time)),
        # Temperature in 0.01Â°C units (15Â°C = 1500)
        "temperature": (("time",), np.ones(n_time) * 1500),
        # Salinity in 0.1 PSU units (35 PSU = 350)
        "salinity": (("time",), np.ones(n_time) * 350),
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

    # Add attributes
    ds["sound_speed"].attrs = {
        "long_name": "Speed of sound",
        "units": "m/s",
    }
    ds["velocity"].attrs = {
        "long_name": "Velocity",
        "units": "mm/s",
    }

    return ds


@pytest.fixture
def adcp_dataset_varying_ts():
    """Create dataset with varying temperature and salinity."""
    n_time = 50
    n_cell = 20
    n_beam = 4

    times = pd.date_range("2024-01-01 00:00:00", periods=n_time, freq="h")

    # Varying temperature: 10Â°C to 20Â°C (1000 to 2000 in 0.01Â°C units)
    temperature = np.linspace(1000, 2000, n_time)

    # Varying salinity: 33 to 37 PSU (330 to 370 in 0.1 PSU units)
    salinity = np.linspace(330, 370, n_time)

    # Varying depth: 50 to 150 m (500 to 1500 in 0.1 m units)
    depth = np.linspace(500, 1500, n_time)

    data_vars = {
        "velocity": (
            ("beam", "cell", "time"),
            np.random.randn(n_beam, n_cell, n_time) * 100,
        ),
        "temperature": (("time",), temperature),
        "salinity": (("time",), salinity),
        "transducer_depth": (("time",), depth),
        "sound_speed": (("time",), np.ones(n_time) * 1500),
    }

    coords = {
        "time": times,
        "cell": np.arange(n_cell),
        "beam": np.arange(n_beam),
    }

    ds = xr.Dataset(data_vars, coords=coords)
    ds["sound_speed"].attrs = {"long_name": "Speed of sound", "units": "m/s"}
    ds["velocity"].attrs = {"long_name": "Velocity", "units": "mm/s"}

    return ds


# ============================================================================
# TESTS: Basic Functionality
# ============================================================================


class TestCorrectSoundSpeedBasic:
    """Test basic correct_sound_speed functionality."""

    def test_returns_dataset(self, adcp_dataset):
        """Test that correct_sound_speed returns xr.Dataset."""
        result = correct_sound_speed(adcp_dataset)
        assert isinstance(result, xr.Dataset)

    def test_original_not_modified(self, adcp_dataset):
        """Test that original dataset is not modified."""
        original_ss = adcp_dataset["sound_speed"].values.copy()
        original_vel = adcp_dataset["velocity"].values.copy()
        _ = correct_sound_speed(adcp_dataset)
        np.testing.assert_array_equal(adcp_dataset["sound_speed"].values, original_ss)
        np.testing.assert_array_equal(adcp_dataset["velocity"].values, original_vel)

    def test_sound_speed_replaced(self, adcp_dataset):
        """Test that sound_speed is actually replaced."""
        result = correct_sound_speed(adcp_dataset)
        # With T=15Â°C, S=35 PSU, D=100m, the Urick formula gives ~1508 m/s
        # Original was 1500, so values should be different
        assert not np.allclose(
            result["sound_speed"].values, adcp_dataset["sound_speed"].values
        )

    def test_velocity_corrected_by_default(self, adcp_dataset):
        """Test that velocity is corrected by default."""
        result = correct_sound_speed(adcp_dataset)
        # Velocity should be different after correction
        assert not np.allclose(
            result["velocity"].values, adcp_dataset["velocity"].values
        )

    def test_preserves_sound_speed_attributes(self, adcp_dataset):
        """Test that original sound_speed attributes are preserved."""
        result = correct_sound_speed(adcp_dataset)
        assert result["sound_speed"].attrs["long_name"] == "Speed of sound"
        assert result["sound_speed"].attrs["units"] == "m/s"

    def test_preserves_velocity_attributes(self, adcp_dataset):
        """Test that original velocity attributes are preserved."""
        result = correct_sound_speed(adcp_dataset)
        assert result["velocity"].attrs["long_name"] == "Velocity"
        assert result["velocity"].attrs["units"] == "mm/s"

    def test_adds_sound_speed_metadata(self, adcp_dataset):
        """Test that correction metadata is added to sound_speed."""
        result = correct_sound_speed(adcp_dataset)
        assert result["sound_speed"].attrs["corrected"] == 1
        assert result["sound_speed"].attrs["correction_method"] == "Urick (1983)"

    def test_adds_velocity_metadata(self, adcp_dataset):
        """Test that correction metadata is added to velocity."""
        result = correct_sound_speed(adcp_dataset)
        assert result["velocity"].attrs["sound_speed_corrected"] == 1
        assert "horizontal_only" in result["velocity"].attrs

    def test_preserves_other_variables(self, adcp_dataset):
        """Test that other variables are unchanged."""
        original_temp = adcp_dataset["temperature"].values.copy()
        result = correct_sound_speed(adcp_dataset)
        np.testing.assert_array_equal(result["temperature"].values, original_temp)


# ============================================================================
# TESTS: correct_velocity Parameter
# ============================================================================


class TestCorrectVelocityParameter:
    """Test correct_velocity parameter."""

    def test_correct_velocity_true(self, adcp_dataset):
        """Test correct_velocity=True corrects velocity."""
        result = correct_sound_speed(adcp_dataset, correct_velocity=True)
        assert result["velocity"].attrs.get("sound_speed_corrected") == 1

    def test_correct_velocity_false(self, adcp_dataset):
        """Test correct_velocity=False does not correct velocity."""
        original_vel = adcp_dataset["velocity"].values.copy()
        result = correct_sound_speed(adcp_dataset, correct_velocity=False)

        # Velocity should be unchanged
        np.testing.assert_array_equal(result["velocity"].values, original_vel)

        # Sound speed should still be corrected
        assert result["sound_speed"].attrs["corrected"] == 1

    def test_correct_velocity_false_no_metadata(self, adcp_dataset):
        """Test correct_velocity=False doesn't add velocity metadata."""
        result = correct_sound_speed(adcp_dataset, correct_velocity=False)
        assert "sound_speed_corrected" not in result["velocity"].attrs


# ============================================================================
# TESTS: horizontal_only Parameter
# ============================================================================


class TestHorizontalOnlyParameter:
    """Test horizontal_only parameter."""

    def test_horizontal_only_true_default(self, adcp_dataset):
        """Test that horizontal_only=True by default."""
        # Get original W velocity
        original_w = adcp_dataset["velocity"].isel(beam=2).values.copy()

        result = correct_sound_speed(adcp_dataset)

        # W velocity should be unchanged
        np.testing.assert_array_equal(
            result["velocity"].isel(beam=2).values, original_w
        )

    def test_horizontal_only_false(self, adcp_dataset):
        """Test that W is corrected when horizontal_only=False."""
        # Get original W velocity
        original_w = adcp_dataset["velocity"].isel(beam=2).values.copy()

        result = correct_sound_speed(adcp_dataset, horizontal_only=False)

        # W velocity should be different
        assert not np.allclose(result["velocity"].isel(beam=2).values, original_w)

    def test_error_velocity_never_corrected(self, adcp_dataset):
        """Test that error velocity (beam 3) is never corrected."""
        # Get original error velocity
        original_e = adcp_dataset["velocity"].isel(beam=3).values.copy()

        result = correct_sound_speed(adcp_dataset, horizontal_only=False)

        # Error velocity should be unchanged
        np.testing.assert_array_equal(
            result["velocity"].isel(beam=3).values, original_e
        )

    def test_horizontal_only_attribute_true(self, adcp_dataset):
        """Test horizontal_only=True is recorded in attributes."""
        result = correct_sound_speed(adcp_dataset, horizontal_only=True)
        assert result["velocity"].attrs["horizontal_only"] is True

    def test_horizontal_only_attribute_false(self, adcp_dataset):
        """Test horizontal_only=False is recorded in attributes."""
        result = correct_sound_speed(adcp_dataset, horizontal_only=False)
        assert result["velocity"].attrs["horizontal_only"] is False


# ============================================================================
# TESTS: Formula Validation
# ============================================================================


class TestSoundSpeedFormula:
    """Test the Urick (1983) formula calculation."""

    def test_known_values(self):
        """Test with known temperature, salinity, depth values."""
        n_time = 10
        times = pd.date_range("2024-01-01", periods=n_time, freq="h")

        # T=15Â°C (1500), S=35 PSU (350), D=100m (1000)
        ds = xr.Dataset(
            {
                "temperature": (("time",), np.ones(n_time) * 1500),
                "salinity": (("time",), np.ones(n_time) * 350),
                "transducer_depth": (("time",), np.ones(n_time) * 1000),
                "sound_speed": (("time",), np.ones(n_time) * 1500),
                "velocity": (
                    ("beam", "cell", "time"),
                    np.ones((4, 10, n_time)) * 100,
                ),
            },
            coords={"time": times, "beam": np.arange(4), "cell": np.arange(10)},
        )

        result = correct_sound_speed(ds)

        # Manual calculation:
        # c = 1449.2 + 4.6*15 - 0.055*15^2 + 0.00029*15^3 + (1.34-0.01*15)*(35-35) + 0.016*100
        expected = 1449.2 + 4.6 * 15 - 0.055 * 225 + 0.00029 * 3375 + 0 + 0.016 * 100
        np.testing.assert_array_almost_equal(
            result["sound_speed"].values, expected, decimal=1
        )

    def test_temperature_effect(self):
        """Test that temperature affects sound speed correctly."""
        n_time = 10
        times = pd.date_range("2024-01-01", periods=n_time, freq="h")

        base_vars = {
            "salinity": (("time",), np.ones(n_time) * 350),
            "transducer_depth": (("time",), np.ones(n_time) * 1000),
            "sound_speed": (("time",), np.ones(n_time) * 1500),
            "velocity": (("beam", "cell", "time"), np.ones((4, 10, n_time)) * 100),
        }
        coords = {"time": times, "beam": np.arange(4), "cell": np.arange(10)}

        # Low temperature: 5Â°C
        ds_cold = xr.Dataset(
            {"temperature": (("time",), np.ones(n_time) * 500), **base_vars},
            coords=coords,
        )

        # High temperature: 25Â°C
        ds_warm = xr.Dataset(
            {"temperature": (("time",), np.ones(n_time) * 2500), **base_vars},
            coords=coords,
        )

        result_cold = correct_sound_speed(ds_cold, correct_velocity=False)
        result_warm = correct_sound_speed(ds_warm, correct_velocity=False)

        # Warmer water should have higher sound speed
        assert result_warm["sound_speed"].mean() > result_cold["sound_speed"].mean()

    def test_salinity_effect(self):
        """Test that salinity affects sound speed correctly."""
        n_time = 10
        times = pd.date_range("2024-01-01", periods=n_time, freq="h")

        base_vars = {
            "temperature": (("time",), np.ones(n_time) * 1500),
            "transducer_depth": (("time",), np.ones(n_time) * 1000),
            "sound_speed": (("time",), np.ones(n_time) * 1500),
            "velocity": (("beam", "cell", "time"), np.ones((4, 10, n_time)) * 100),
        }
        coords = {"time": times, "beam": np.arange(4), "cell": np.arange(10)}

        # Low salinity: 30 PSU
        ds_fresh = xr.Dataset(
            {"salinity": (("time",), np.ones(n_time) * 300), **base_vars},
            coords=coords,
        )

        # High salinity: 40 PSU
        ds_salty = xr.Dataset(
            {"salinity": (("time",), np.ones(n_time) * 400), **base_vars},
            coords=coords,
        )

        result_fresh = correct_sound_speed(ds_fresh, correct_velocity=False)
        result_salty = correct_sound_speed(ds_salty, correct_velocity=False)

        # Saltier water should have higher sound speed
        assert result_salty["sound_speed"].mean() > result_fresh["sound_speed"].mean()


# ============================================================================
# TESTS: Velocity Correction Logic
# ============================================================================


class TestVelocityCorrectionLogic:
    """Test velocity correction logic."""

    def test_missing_values_preserved(self, adcp_dataset):
        """Test that missing velocity values (-32768) are preserved."""
        result = correct_sound_speed(adcp_dataset)

        # Check that missing value at [0, 5, 10] is preserved
        assert result["velocity"].values[0, 5, 10] == -32768

    def test_correction_ratio(self):
        """Test that correction uses correct ratio."""
        n_time = 10
        times = pd.date_range("2024-01-01", periods=n_time, freq="h")

        # Create dataset where we know the correction ratio
        # T=15Â°C, S=35 PSU, D=100m gives ~1508 m/s
        # Original SS = 1500 m/s
        # Ratio = 1500 / 1508 â‰ˆ 0.9947
        ds = xr.Dataset(
            {
                "temperature": (("time",), np.ones(n_time) * 1500),
                "salinity": (("time",), np.ones(n_time) * 350),
                "transducer_depth": (("time",), np.ones(n_time) * 1000),
                "sound_speed": (("time",), np.ones(n_time) * 1500),
                "velocity": (
                    ("beam", "cell", "time"),
                    np.ones((4, 10, n_time)) * 1000,  # 1000 mm/s
                ),
            },
            coords={"time": times, "beam": np.arange(4), "cell": np.arange(10)},
        )

        result = correct_sound_speed(ds)

        # Corrected velocity should be slightly less than original
        # because corrected SS > original SS
        assert result["velocity"].isel(beam=0, cell=0, time=0).values < 1000


# ============================================================================
# TESTS: Error Handling
# ============================================================================


class TestCorrectSoundSpeedErrors:
    """Test error handling."""

    def test_raises_missing_temperature(self, adcp_dataset):
        """Test ValueError when temperature is missing."""
        ds = adcp_dataset.drop_vars("temperature")
        with pytest.raises(ValueError, match="Missing required variables"):
            correct_sound_speed(ds)

    def test_raises_missing_salinity(self, adcp_dataset):
        """Test ValueError when salinity is missing."""
        ds = adcp_dataset.drop_vars("salinity")
        with pytest.raises(ValueError, match="Missing required variables"):
            correct_sound_speed(ds)

    def test_raises_missing_depth(self, adcp_dataset):
        """Test ValueError when depth is missing."""
        ds = adcp_dataset.drop_vars("transducer_depth")
        with pytest.raises(ValueError, match="Missing required variables"):
            correct_sound_speed(ds)

    def test_raises_missing_sound_speed(self, adcp_dataset):
        """Test ValueError when sound_speed is missing."""
        ds = adcp_dataset.drop_vars("sound_speed")
        with pytest.raises(ValueError, match="Missing required variables"):
            correct_sound_speed(ds)

    def test_raises_missing_velocity_when_needed(self, adcp_dataset):
        """Test ValueError when velocity missing and correct_velocity=True."""
        ds = adcp_dataset.drop_vars("velocity")
        with pytest.raises(ValueError, match="Missing required variables"):
            correct_sound_speed(ds, correct_velocity=True)

    def test_no_error_missing_velocity_when_not_needed(self, adcp_dataset):
        """Test no error when velocity missing but correct_velocity=False."""
        ds = adcp_dataset.drop_vars("velocity")
        # Should not raise
        result = correct_sound_speed(ds, correct_velocity=False)
        assert result["sound_speed"].attrs["corrected"] == 1


# ============================================================================
# TESTS: Integration
# ============================================================================


class TestSoundSpeedIntegration:
    """Integration tests for sound speed correction workflow."""

    def test_simple_workflow(self, adcp_dataset):
        """Test simple one-call workflow."""
        # Just one call does everything
        ds_corrected = correct_sound_speed(adcp_dataset)

        # Verify both corrections applied
        assert ds_corrected["sound_speed"].attrs.get("corrected") == 1
        assert ds_corrected["velocity"].attrs.get("sound_speed_corrected") == 1

    def test_workflow_with_varying_ts(self, adcp_dataset_varying_ts):
        """Test workflow with varying temperature and salinity."""
        ds = correct_sound_speed(adcp_dataset_varying_ts)

        # Sound speed should vary with T/S/D
        ss_values = ds["sound_speed"].values
        assert ss_values.std() > 0  # Should have variation

    def test_sound_speed_only_workflow(self, adcp_dataset):
        """Test workflow correcting only sound speed."""
        ds_corrected = correct_sound_speed(adcp_dataset, correct_velocity=False)

        # Sound speed corrected
        assert ds_corrected["sound_speed"].attrs.get("corrected") == 1

        # Velocity unchanged
        assert "sound_speed_corrected" not in ds_corrected["velocity"].attrs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
