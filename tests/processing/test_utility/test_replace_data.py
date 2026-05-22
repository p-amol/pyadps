"""
Test suite for replace_data function.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyadps.processing.utility import replace_data


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_dataset():
    """Create a sample xarray Dataset for testing."""
    np.random.seed(42)
    n_ensembles = 50

    # Create temperature with scale_factor
    temperature = np.random.randint(1000, 3000, size=(n_ensembles,))

    # Create salinity with scale_factor
    salinity = np.random.randint(30000, 40000, size=(n_ensembles,))

    # Create pressure without scale_factor
    pressure = np.random.randint(1000, 5000, size=(n_ensembles,))

    # Create time coordinate
    time = pd.date_range("2024-01-01", periods=n_ensembles, freq="h")

    ds = xr.Dataset(
        {
            "temperature": (
                ["time"],
                temperature,
                {
                    "units": "0.01 degC",
                    "scale_factor": 0.01,
                    "long_name": "Temperature",
                },
            ),
            "salinity": (
                ["time"],
                salinity,
                {"units": "0.001 PSU", "scale_factor": 0.001, "long_name": "Salinity"},
            ),
            "pressure": (
                ["time"],
                pressure,
                {"units": "0.001 dbar", "long_name": "Pressure"},
            ),
        },
        coords={
            "time": time,
        },
    )
    return ds


@pytest.fixture
def dataset_2d():
    """Create a 2D dataset for testing dimension validation."""
    ds = xr.Dataset(
        {
            "velocity": (
                ["cell", "time"],
                np.random.randn(20, 50),
                {"units": "m/s", "scale_factor": 0.001},
            ),
        },
        coords={
            "cell": np.arange(20),
            "time": pd.date_range("2024-01-01", periods=50, freq="h"),
        },
    )
    return ds


# ============================================================================
# TESTS: Basic Replacement
# ============================================================================


class TestReplaceDataBasic:
    """Test basic replace_data functionality."""

    def test_basic_replacement(self, sample_dataset):
        """Test basic variable replacement."""
        new_temp = np.ones(50) * 25.0  # 25Â°C in physical units
        ds_out = replace_data(sample_dataset, new_temp, "temperature")

        # Check that replacement happened (scaled by 0.01)
        expected = 25.0 / 0.01  # 2500
        assert ds_out["temperature"].values.mean() == expected

    def test_returns_dataset(self, sample_dataset):
        """Test that function returns a Dataset."""
        new_temp = np.ones(50) * 25.0
        ds_out = replace_data(sample_dataset, new_temp, "temperature")
        assert isinstance(ds_out, xr.Dataset)

    def test_original_not_modified(self, sample_dataset):
        """Test that original dataset is not modified."""
        original_mean = sample_dataset["temperature"].values.mean()
        new_temp = np.ones(50) * 25.0
        replace_data(sample_dataset, new_temp, "temperature")
        assert sample_dataset["temperature"].values.mean() == original_mean


class TestReplaceDataScaleFactor:
    """Test scale factor handling."""

    def test_scale_factor_applied(self, sample_dataset):
        """Test that scale factor is applied correctly."""
        # Input 25Â°C, scale_factor=0.01, stored as 2500
        new_temp = np.ones(50) * 25.0
        ds_out = replace_data(sample_dataset, new_temp, "temperature")
        assert ds_out["temperature"].values[0] == 2500

    def test_scale_factor_disabled(self, sample_dataset):
        """Test with apply_scale_factor=False."""
        new_temp = np.ones(50) * 2500.0  # Already in RDI format
        ds_out = replace_data(
            sample_dataset, new_temp, "temperature", apply_scale_factor=False
        )
        assert ds_out["temperature"].values[0] == 2500

    def test_scale_factor_salinity(self, sample_dataset):
        """Test scale factor for salinity."""
        # Input 35 PSU, scale_factor=0.001, stored as 35000
        new_sal = np.ones(50) * 35.0
        ds_out = replace_data(sample_dataset, new_sal, "salinity")
        assert ds_out["salinity"].values[0] == 35000

    def test_no_warning_when_disabled(self, sample_dataset):
        """Test no warning when apply_scale_factor=False."""
        new_pressure = np.ones(50) * 100.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            replace_data(
                sample_dataset, new_pressure, "pressure", apply_scale_factor=False
            )
            assert len(w) == 0


class TestReplaceDataAttributes:
    """Test attribute preservation and addition."""

    def test_preserves_original_attributes(self, sample_dataset):
        """Test that original attributes are preserved."""
        new_temp = np.ones(50) * 25.0
        ds_out = replace_data(sample_dataset, new_temp, "temperature")
        assert ds_out["temperature"].attrs["units"] == "0.01 degC"
        assert ds_out["temperature"].attrs["long_name"] == "Temperature"
        assert ds_out["temperature"].attrs["scale_factor"] == 0.01

    def test_adds_replacement_metadata(self, sample_dataset):
        """Test that replacement metadata is added."""
        new_temp = np.ones(50) * 25.0
        ds_out = replace_data(sample_dataset, new_temp, "temperature")
        assert ds_out["temperature"].attrs["data_replaced"] == 1
        assert ds_out["temperature"].attrs["replacement_scale_factor_applied"] == 1

    def test_replacement_metadata_false_when_disabled(self, sample_dataset):
        """Test replacement metadata when scale_factor disabled."""
        new_temp = np.ones(50) * 2500.0
        ds_out = replace_data(
            sample_dataset, new_temp, "temperature", apply_scale_factor=False
        )
        assert ds_out["temperature"].attrs["data_replaced"] == 1
        assert ds_out["temperature"].attrs["replacement_scale_factor_applied"] == 0

    def test_replacement_metadata_false_no_scale_factor(self, sample_dataset):
        """Test replacement metadata when variable has no scale_factor."""
        new_pressure = np.ones(50) * 100.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds_out = replace_data(sample_dataset, new_pressure, "pressure")
        assert ds_out["pressure"].attrs["data_replaced"] == 1
        assert ds_out["pressure"].attrs["replacement_scale_factor_applied"] == 0


class TestReplaceDataCoordinates:
    """Test coordinate preservation."""

    def test_preserves_coordinates(self, sample_dataset):
        """Test that replacement preserves coordinates."""
        new_temp = np.ones(50) * 25.0
        ds_out = replace_data(sample_dataset, new_temp, "temperature")
        np.testing.assert_array_equal(
            ds_out["temperature"].coords["time"].values,
            sample_dataset["temperature"].coords["time"].values,
        )

    def test_preserves_other_variables(self, sample_dataset):
        """Test that other variables are unchanged."""
        original_salinity = sample_dataset["salinity"].values.copy()
        new_temp = np.ones(50) * 25.0
        ds_out = replace_data(sample_dataset, new_temp, "temperature")
        np.testing.assert_array_equal(ds_out["salinity"].values, original_salinity)


class TestReplaceDataErrors:
    """Test error handling."""

    def test_error_variable_not_found(self, sample_dataset):
        """Test error when variable not found."""
        with pytest.raises(ValueError, match="not found"):
            replace_data(sample_dataset, np.ones(50), "nonexistent")

    def test_error_dimension_mismatch(self, sample_dataset):
        """Test error when dimensions don't match."""
        # temperature is 1D, provide 2D data
        with pytest.raises(ValueError, match="Dimension mismatch"):
            replace_data(sample_dataset, np.ones((50, 10)), "temperature")

    def test_error_shape_mismatch(self, sample_dataset):
        """Test error when shape doesn't match."""
        # temperature has 50 elements
        with pytest.raises(ValueError, match="Shape mismatch"):
            replace_data(sample_dataset, np.ones(100), "temperature")

    def test_error_message_shows_available_variables(self, sample_dataset):
        """Test error message shows available variables."""
        with pytest.raises(ValueError) as exc_info:
            replace_data(sample_dataset, np.ones(50), "nonexistent")
        assert "temperature" in str(exc_info.value)
        assert "salinity" in str(exc_info.value)
        assert "pressure" in str(exc_info.value)


class TestReplaceData2D:
    """Test replace_data with 2D variables."""

    def test_2d_replacement(self, dataset_2d):
        """Test replacement of 2D variable."""
        new_velocity = np.ones((20, 50)) * 1.5  # 1.5 m/s
        ds_out = replace_data(dataset_2d, new_velocity, "velocity")
        # scale_factor=0.001, so 1.5 / 0.001 = 1500
        assert ds_out["velocity"].values[0, 0] == 1500

    def test_2d_shape_mismatch(self, dataset_2d):
        """Test error with wrong 2D shape."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            replace_data(dataset_2d, np.ones((10, 50)), "velocity")


class TestReplaceDataDtype:
    """Test dtype handling."""

    def test_preserves_dtype(self, sample_dataset):
        """Test that original dtype is preserved."""
        original_dtype = sample_dataset["temperature"].dtype
        new_temp = np.ones(50) * 25.0
        ds_out = replace_data(sample_dataset, new_temp, "temperature")
        assert ds_out["temperature"].dtype == original_dtype

    def test_converts_float_to_int(self):
        """Test conversion from float input to int dtype."""
        ds = xr.Dataset(
            {
                "count": (
                    ["time"],
                    np.array([1, 2, 3], dtype=np.int32),
                    {"scale_factor": 1.0},
                ),
            },
            coords={"time": pd.date_range("2024-01-01", periods=3)},
        )
        new_count = np.array([10.5, 20.5, 30.5])  # Float input
        ds_out = replace_data(ds, new_count, "count")
        assert ds_out["count"].dtype == np.int32
        # Values should be truncated
        np.testing.assert_array_equal(ds_out["count"].values, [10, 20, 30])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
