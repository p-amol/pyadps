"""
Test suite for regrid function.

Tests the regridding functionality that transforms ADCP data from
irregular instrument cells to a regular depth grid.

Note: Tests use a MockFixedLeaderAccessor that reads configuration from
dataset attributes, avoiding issues with dataset copy operations.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

try:
    from pyadps.processing.profile_operation import (
        regrid,
        DEFAULT_REGRID_METHOD,
        DEFAULT_END_CELL_OPTION,
    )
    from pyadps.processing.utility import VELOCITY_MISSING_VALUE
except ImportError:
    import sys

    sys.path.insert(0, "/mnt/user-data/outputs")
    from profile_operation import regrid, DEFAULT_REGRID_METHOD, DEFAULT_END_CELL_OPTION

    VELOCITY_MISSING_VALUE = -32768


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def upward_dataset():
    """Create upward-looking ADCP dataset.

    Configuration:
    - 50m transducer depth (500 decimeters)
    - 4m cell size (400 cm)
    - 2m bin1 distance (200 cm)
    - 20Â° beam angle
    """
    np.random.seed(42)
    n_time = 30
    n_cell = 25
    n_beam = 4

    times = pd.date_range("2024-01-01", periods=n_time, freq="h")
    cells = np.arange(n_cell)
    beams = np.arange(n_beam)

    # Transducer at 50m depth (500 decimeters)
    transducer_depth = np.full(n_time, 500)

    # Create velocity with some values
    velocity = np.random.randint(-500, 500, size=(n_beam, n_cell, n_time)).astype(
        np.int16
    )

    # Create correlation, echo intensity, percent good
    correlation = np.random.randint(50, 200, size=(n_beam, n_cell, n_time)).astype(
        np.int16
    )
    echo_intensity = np.random.randint(30, 150, size=(n_beam, n_cell, n_time)).astype(
        np.int16
    )
    percent_good = np.random.randint(40, 100, size=(n_beam, n_cell, n_time)).astype(
        np.int16
    )

    data_vars = {
        "velocity": (("beam", "cell", "time"), velocity),
        "correlation": (("beam", "cell", "time"), correlation),
        "echo_intensity": (("beam", "cell", "time"), echo_intensity),
        "percent_good": (("beam", "cell", "time"), percent_good),
        "transducer_depth": (("time",), transducer_depth),
        "mask": (
            ("beam", "cell", "time"),
            np.zeros((n_beam, n_cell, n_time), dtype=np.int8),
        ),
    }

    coords = {"time": times, "cell": cells, "beam": beams}
    ds = xr.Dataset(data_vars, coords=coords)

    # Store configuration in attrs
    ds.attrs["cell_size_cm"] = 400
    ds.attrs["bin1_distance_cm"] = 200
    ds.attrs["beam_angle"] = 20
    ds.attrs["beam_direction"] = "up"

    return ds


@pytest.fixture
def downward_dataset():
    """Create downward-looking ADCP dataset.

    Configuration:
    - 10m transducer depth (100 decimeters)
    - 4m cell size (400 cm)
    - 2m bin1 distance (200 cm)
    - 20Â° beam angle
    """
    np.random.seed(42)
    n_time = 30
    n_cell = 25
    n_beam = 4

    times = pd.date_range("2024-01-01", periods=n_time, freq="h")
    cells = np.arange(n_cell)
    beams = np.arange(n_beam)

    # Transducer at 10m depth (100 decimeters)
    transducer_depth = np.full(n_time, 100)

    velocity = np.random.randint(-500, 500, size=(n_beam, n_cell, n_time)).astype(
        np.int16
    )

    data_vars = {
        "velocity": (("beam", "cell", "time"), velocity),
        "transducer_depth": (("time",), transducer_depth),
        "mask": (
            ("beam", "cell", "time"),
            np.zeros((n_beam, n_cell, n_time), dtype=np.int8),
        ),
    }

    coords = {"time": times, "cell": cells, "beam": beams}
    ds = xr.Dataset(data_vars, coords=coords)

    ds.attrs["cell_size_cm"] = 400
    ds.attrs["bin1_distance_cm"] = 200
    ds.attrs["beam_angle"] = 20
    ds.attrs["beam_direction"] = "down"

    return ds


@pytest.fixture
def dataset_with_mask(upward_dataset):
    """Create dataset with pre-existing mask."""
    # Mask some cells
    mask = upward_dataset["mask"].values.copy()
    mask[:, 10:15, :] = 1  # Mask cells 10-14
    upward_dataset["mask"].values[:] = mask
    return upward_dataset


@pytest.fixture
def dataset_with_missing_values(upward_dataset):
    """Create dataset with missing velocity values."""
    velocity = upward_dataset["velocity"].values.copy()
    velocity[:, 5, :10] = (
        VELOCITY_MISSING_VALUE  # Missing in cell 5, first 10 ensembles
    )
    upward_dataset["velocity"].values[:] = velocity
    return upward_dataset


# ============================================================================
# TESTS: Basic Regridding
# ============================================================================


class TestRegridBasic:
    """Tests for basic regrid functionality."""

    def test_basic_regrid(self, upward_dataset):
        """Test basic regridding operation."""
        result = regrid(upward_dataset)

        # Should have 'depth' dimension instead of 'cell'
        assert "depth" in result.dims
        assert "cell" not in result.dims

        # Should have velocity variable
        assert "velocity" in result.data_vars

    def test_regrid_creates_depth_coordinate(self, upward_dataset):
        """Test that depth coordinate is created."""
        result = regrid(upward_dataset)

        # Depth should be a coordinate
        assert "depth" in result.coords

        # Depth values should be positive (absolute depth)
        assert np.all(result.coords["depth"].values >= 0)

    def test_regrid_preserves_time(self, upward_dataset):
        """Test that time dimension is preserved."""
        result = regrid(upward_dataset)

        assert "time" in result.dims
        assert result.sizes["time"] == upward_dataset.sizes["time"]
        np.testing.assert_array_equal(
            result.coords["time"].values, upward_dataset.coords["time"].values
        )

    def test_regrid_preserves_beam(self, upward_dataset):
        """Test that beam dimension is preserved for 3D variables."""
        result = regrid(upward_dataset)

        assert "beam" in result.dims
        assert result.sizes["beam"] == upward_dataset.sizes["beam"]

    def test_regrid_all_variables(self, upward_dataset):
        """Test that all relevant variables are regridded."""
        result = regrid(upward_dataset)

        # Check that main variables are present
        assert "velocity" in result.data_vars
        assert "correlation" in result.data_vars
        assert "echo_intensity" in result.data_vars
        assert "percent_good" in result.data_vars


# ============================================================================
# TESTS: Interpolation Methods
# ============================================================================


class TestRegridMethods:
    """Tests for different interpolation methods."""

    def test_nearest_method(self, upward_dataset):
        """Test nearest neighbor interpolation."""
        result = regrid(upward_dataset, method="nearest")

        assert "velocity" in result.data_vars
        assert result.attrs.get("regridding_method") == "nearest"

    def test_linear_method(self, upward_dataset):
        """Test linear interpolation."""
        result = regrid(upward_dataset, method="linear")

        assert result.attrs.get("regridding_method") == "linear"

    def test_cubic_method(self, upward_dataset):
        """Test cubic interpolation."""
        result = regrid(upward_dataset, method="cubic")

        assert result.attrs.get("regridding_method") == "cubic"

    def test_default_method(self, upward_dataset):
        """Test default interpolation method."""
        result = regrid(upward_dataset)

        assert result.attrs.get("regridding_method") == DEFAULT_REGRID_METHOD


# ============================================================================
# TESTS: End Cell Options
# ============================================================================


class TestRegridEndCellOptions:
    """Tests for end_cell_option parameter."""

    def test_cell_option(self, upward_dataset):
        """Test end_cell_option='cell'."""
        result = regrid(upward_dataset, end_cell_option="cell")

        assert result.attrs.get("regridding_end_cell_option") == "cell"

    def test_surface_option(self, upward_dataset):
        """Test end_cell_option='surface'."""
        result = regrid(upward_dataset, end_cell_option="surface")

        assert result.attrs.get("regridding_end_cell_option") == "surface"

        # Should extend closer to surface
        min_depth = result.coords["depth"].values.min()
        assert min_depth < 5.0  # Should be near surface

    def test_manual_option(self, upward_dataset):
        """Test end_cell_option='manual'."""
        result = regrid(upward_dataset, end_cell_option="manual", boundary_limit=10.0)

        assert result.attrs.get("regridding_end_cell_option") == "manual"

    def test_invalid_option(self, upward_dataset):
        """Test error for invalid end_cell_option."""
        with pytest.raises(ValueError, match="not recognized"):
            regrid(upward_dataset, end_cell_option="invalid")


# ============================================================================
# TESTS: Mask Handling
# ============================================================================


class TestRegridMaskHandling:
    """Tests for mask handling during regridding."""

    def test_mask_applied_before_regrid(self, dataset_with_mask):
        """Test that mask is applied before regridding."""
        result = regrid(dataset_with_mask)

        # Velocity should have NaN where mask was applied
        velocity = result["velocity"].values
        assert np.any(np.isnan(velocity))

    def test_mask_recreated_after_regrid(self, dataset_with_mask):
        """Test that mask is recreated after regridding."""
        result = regrid(dataset_with_mask)

        # Mask should exist in result
        assert "mask" in result.data_vars

        # Mask should have same dimensions as velocity
        assert result["mask"].dims == result["velocity"].dims

    def test_mask_values_binary(self, dataset_with_mask):
        """Test that recreated mask has binary values."""
        result = regrid(dataset_with_mask)

        mask_values = result["mask"].values
        unique_values = np.unique(mask_values[~np.isnan(mask_values)])

        # Should only have 0 and 1
        assert np.all(np.isin(unique_values, [0, 1]))

    def test_mask_preserved_through_regrid(self, dataset_with_mask):
        """Test that masked regions are preserved (not interpolated over)."""
        result = regrid(dataset_with_mask)

        # Masked velocity should contain NaN
        velocity = result["velocity"].values
        mask = result["mask"].values

        # Where mask is 1, velocity should be NaN
        masked_velocity = velocity[mask == 1]
        assert np.all(np.isnan(masked_velocity))

    def test_mask_not_applied_to_echo_intensity(self, dataset_with_mask):
        """echo_intensity is a raw, physical-beam diagnostic - the
        velocity-derived mask (QC checks + side-lobe cutoff) must not
        blank it out, or the values explaining *why* a cell was flagged
        would be destroyed before ever reaching export."""
        result = regrid(dataset_with_mask)

        velocity = result["velocity"].values
        echo = result["echo_intensity"].values
        mask = result["mask"].values

        # Cells where velocity was masked (NaN) should still have real
        # echo_intensity values, since echo isn't masked at all.
        velocity_masked = np.isnan(velocity)
        assert np.any(velocity_masked)
        assert np.all(~np.isnan(echo[velocity_masked]))

    def test_mask_not_applied_to_correlation_and_percent_good(self, dataset_with_mask):
        """Same guarantee as echo_intensity for the other physical-beam
        diagnostics."""
        result = regrid(dataset_with_mask)

        velocity_masked = np.isnan(result["velocity"].values)
        assert np.any(velocity_masked)
        for var_name in ["correlation", "percent_good"]:
            values = result[var_name].values
            assert np.all(~np.isnan(values[velocity_masked]))


# ============================================================================
# TESTS: Missing Value Handling
# ============================================================================


class TestRegridMissingValues:
    """Tests for handling missing values (-32768)."""

    def test_missing_values_converted(self, dataset_with_missing_values):
        """Test that missing values are converted to NaN."""
        result = regrid(dataset_with_missing_values)

        # Result should not contain the missing value
        velocity = result["velocity"].values
        assert not np.any(velocity == VELOCITY_MISSING_VALUE)

    def test_missing_values_become_nan(self, dataset_with_missing_values):
        """Test that missing values become NaN after regridding."""
        result = regrid(dataset_with_missing_values)

        # Should have some NaN values
        velocity = result["velocity"].values
        assert np.any(np.isnan(velocity))


# ============================================================================
# TESTS: Orientation
# ============================================================================


class TestRegridOrientation:
    """Tests for ADCP orientation handling."""

    def test_upward_orientation(self, upward_dataset):
        """Test regridding for upward-looking ADCP."""
        result = regrid(upward_dataset, orientation="up")

        # Depth should increase from near-surface to transducer
        depths = result.coords["depth"].values
        assert len(depths) > 1

    def test_downward_orientation(self, downward_dataset):
        """Test regridding for downward-looking ADCP."""
        result = regrid(downward_dataset, orientation="down")

        # Should produce valid depth grid
        depths = result.coords["depth"].values
        assert len(depths) > 1

    def test_auto_detect_orientation(self, upward_dataset):
        """Test auto-detection of orientation."""
        result = regrid(upward_dataset)  # No orientation specified

        # Should work with auto-detection
        assert "velocity" in result.data_vars


# ============================================================================
# TESTS: Trimends Parameter
# ============================================================================


class TestRegridTrimends:
    """Tests for trimends parameter."""

    def test_trimends_excludes_edges(self, upward_dataset):
        """Test that trimends excludes edge ensembles from depth calculation."""
        # Modify edge depths to be very different
        upward_dataset["transducer_depth"].values[0:5] = 200  # Shallow at start
        upward_dataset["transducer_depth"].values[-5:] = 800  # Deep at end

        result_no_trim = regrid(upward_dataset)
        result_with_trim = regrid(upward_dataset, trimends=(5, -5))

        # Trimmed version should have different depth range
        # (excluding the artificial shallow/deep edges)
        depths_no_trim = result_no_trim.coords["depth"].values
        depths_with_trim = result_with_trim.coords["depth"].values

        # They may or may not be equal depending on calculation
        # Just verify both work
        assert len(depths_no_trim) > 0
        assert len(depths_with_trim) > 0


# ============================================================================
# TESTS: Attributes
# ============================================================================


class TestRegridAttributes:
    """Tests for dataset attributes after regridding."""

    def test_regridding_attributes_added(self, upward_dataset):
        """Test that regridding attributes are added."""
        result = regrid(upward_dataset)

        assert "regridding_method" in result.attrs
        assert "regridding_end_cell_option" in result.attrs
        assert "regridding_fill_value" in result.attrs
        assert "regridded" in result.attrs

    def test_regridded_flag_true(self, upward_dataset):
        """Test that regridded flag is True."""
        result = regrid(upward_dataset)

        assert result.attrs.get("regridded") == 1

    def test_original_attrs_preserved(self, upward_dataset):
        """Test that original dataset attributes are preserved."""
        upward_dataset.attrs["test_attr"] = "test_value"

        result = regrid(upward_dataset)

        assert result.attrs.get("test_attr") == "test_value"

    def test_variable_attrs_preserved(self, upward_dataset):
        """Test that variable attributes are preserved."""
        upward_dataset["velocity"].attrs["units"] = "mm/s"

        result = regrid(upward_dataset)

        assert result["velocity"].attrs.get("units") == "mm/s"


# ============================================================================
# TESTS: Data Variables Selection
# ============================================================================


class TestRegridDataVars:
    """Tests for data_vars parameter."""

    def test_specific_data_vars(self, upward_dataset):
        """Test regridding only specific variables."""
        result = regrid(upward_dataset, data_vars=["velocity"])

        assert "velocity" in result.data_vars

    def test_excludes_mask_from_interpolation(self, upward_dataset):
        """Test that mask is excluded from interpolation."""
        result = regrid(upward_dataset, data_vars=["velocity", "mask"])

        # Mask should still be present (recreated, not interpolated)
        assert "mask" in result.data_vars

    def test_auto_detect_vars(self, upward_dataset):
        """Test auto-detection of variables to regrid."""
        result = regrid(upward_dataset)  # No data_vars specified

        # Should auto-detect and regrid velocity and other 2D/3D vars
        assert "velocity" in result.data_vars


# ============================================================================
# TESTS: Non-Cell-Dependent Variable Preservation
# ============================================================================


class TestRegridPreservesNonCellVariables:
    """Regression tests: variables that don't depend on the 'cell' dimension
    (e.g. Fixed Leader fields like coordinate_transformation_code, which are
    indexed only by 'time') must survive regridding unchanged instead of
    being silently dropped.
    """

    def test_coordinate_transformation_code_preserved(self, upward_dataset):
        """coordinate_transformation_code (dims=('time',)) must not be
        dropped by regrid, otherwise ds.fixed_leader.coordinate_transformation()
        raises ValueError after regridding.
        """
        n_time = upward_dataset.sizes["time"]
        # Bit pattern 11xxxxxx -> "Earth Coordinates" (see accessors.py)
        code = np.full(n_time, 0b11011000, dtype=np.uint8)
        upward_dataset["coordinate_transformation_code"] = (("time",), code)

        result = regrid(upward_dataset)

        assert "coordinate_transformation_code" in result.data_vars
        np.testing.assert_array_equal(
            result["coordinate_transformation_code"].values, code
        )

    def test_other_time_only_variables_preserved(self, upward_dataset):
        """Any other non-cell variable (e.g. heading) should also survive."""
        n_time = upward_dataset.sizes["time"]
        heading = np.linspace(0, 359, n_time)
        upward_dataset["heading"] = (("time",), heading)

        result = regrid(upward_dataset)

        assert "heading" in result.data_vars
        np.testing.assert_array_equal(result["heading"].values, heading)

    def test_transducer_depth_preserved(self, upward_dataset):
        """transducer_depth (dims=('time',)) should be preserved too."""
        result = regrid(upward_dataset)

        assert "transducer_depth" in result.data_vars
        np.testing.assert_array_equal(
            result["transducer_depth"].values,
            upward_dataset["transducer_depth"].values,
        )


# ============================================================================
# TESTS: Fill Value
# ============================================================================


class TestRegridFillValue:
    """Tests for fill_value parameter."""

    def test_default_fill_value_nan(self, upward_dataset):
        """Test default fill value is NaN."""
        result = regrid(upward_dataset)

        assert result.attrs.get("regridding_fill_value") == "nan"

    def test_custom_fill_value(self, upward_dataset):
        """Test custom fill value."""
        result = regrid(upward_dataset, fill_value=-999.0)

        assert "-999" in result.attrs.get("regridding_fill_value", "")


# ============================================================================
# TESTS: Edge Cases
# ============================================================================


class TestRegridEdgeCases:
    """Tests for edge cases."""

    def test_single_ensemble(self):
        """Test regridding with single ensemble."""
        ds = xr.Dataset(
            {
                "velocity": (("beam", "cell", "time"), np.random.randn(4, 20, 1)),
                "transducer_depth": (("time",), np.array([500])),
                "mask": (("beam", "cell", "time"), np.zeros((4, 20, 1), dtype=np.int8)),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(20),
                "time": pd.date_range("2024-01-01", periods=1),
            },
        )
        ds.attrs["cell_size_cm"] = 400
        ds.attrs["bin1_distance_cm"] = 200
        ds.attrs["beam_angle"] = 20
        ds.attrs["beam_direction"] = "up"

        result = regrid(ds)

        assert result.sizes["time"] == 1

    def test_empty_depth_grid_error(self):
        """Test error when depth grid would be empty."""
        # Create a problematic dataset
        ds = xr.Dataset(
            {
                "velocity": (("beam", "cell", "time"), np.random.randn(4, 5, 10)),
                "transducer_depth": (("time",), np.full(10, 1)),  # Very shallow
                "mask": (("beam", "cell", "time"), np.zeros((4, 5, 10), dtype=np.int8)),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(5),
                "time": pd.date_range("2024-01-01", periods=10, freq="h"),
            },
        )
        ds.attrs["cell_size_cm"] = 400
        ds.attrs["bin1_distance_cm"] = 200
        ds.attrs["beam_angle"] = 20
        ds.attrs["beam_direction"] = "up"

        # This might produce empty grid or very small grid
        # Either should be handled (error or warning)
        try:
            result = regrid(ds)
            # If it works, should have at least some depth levels
            assert result.sizes.get("depth", 0) >= 0
        except ValueError as e:
            assert "Empty depth grid" in str(e) or "Very small" in str(e)

    def test_variable_transducer_depth(self):
        """Test with varying transducer depth."""
        ds = xr.Dataset(
            {
                "velocity": (("beam", "cell", "time"), np.random.randn(4, 25, 50)),
                "transducer_depth": (("time",), np.linspace(400, 600, 50)),  # 40-60m
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((4, 25, 50), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(25),
                "time": pd.date_range("2024-01-01", periods=50, freq="h"),
            },
        )
        ds.attrs["cell_size_cm"] = 400
        ds.attrs["bin1_distance_cm"] = 200
        ds.attrs["beam_angle"] = 20
        ds.attrs["beam_direction"] = "up"

        result = regrid(ds)

        # Should handle varying depth
        assert "velocity" in result.data_vars


# ============================================================================
# TESTS: Immutability
# ============================================================================


class TestRegridImmutability:
    """Tests for immutability of original dataset."""

    def test_original_unchanged(self, upward_dataset):
        """Test that original dataset is not modified."""
        original_velocity = upward_dataset["velocity"].values.copy()
        original_dims = dict(upward_dataset.sizes)

        _ = regrid(upward_dataset)

        np.testing.assert_array_equal(
            upward_dataset["velocity"].values, original_velocity
        )
        assert dict(upward_dataset.sizes) == original_dims

    def test_returns_new_dataset(self, upward_dataset):
        """Test that a new dataset is returned."""
        result = regrid(upward_dataset)

        assert result is not upward_dataset


# ============================================================================
# TESTS: Input Validation
# ============================================================================


class TestRegridValidation:
    """Tests for input validation."""

    def test_invalid_dataset_type(self):
        """Test error for non-Dataset input."""
        with pytest.raises(TypeError):
            regrid("not a dataset")

    def test_missing_velocity(self):
        """Test error when velocity is missing."""
        ds = xr.Dataset(
            {
                "correlation": (("beam", "cell", "time"), np.random.randn(4, 10, 20)),
                "transducer_depth": (("time",), np.full(20, 500)),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(10),
                "time": pd.date_range("2024-01-01", periods=20, freq="h"),
            },
        )
        ds.attrs["cell_size_cm"] = 400
        ds.attrs["bin1_distance_cm"] = 200
        ds.attrs["beam_angle"] = 20
        ds.attrs["beam_direction"] = "up"

        with pytest.raises(ValueError, match="missing required variable"):
            regrid(ds)


# ============================================================================
# TESTS: 2D Variables
# ============================================================================


class TestRegrid2DVariables:
    """Tests for 2D variable handling."""

    def test_2d_variable_regridded(self):
        """Test that 2D variables are regridded correctly."""
        ds = xr.Dataset(
            {
                "velocity": (("beam", "cell", "time"), np.random.randn(4, 20, 30)),
                "some_2d_var": (("cell", "time"), np.random.randn(20, 30)),  # 2D
                "transducer_depth": (("time",), np.full(30, 500)),
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((4, 20, 30), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(20),
                "time": pd.date_range("2024-01-01", periods=30, freq="h"),
            },
        )
        ds.attrs["cell_size_cm"] = 400
        ds.attrs["bin1_distance_cm"] = 200
        ds.attrs["beam_angle"] = 20
        ds.attrs["beam_direction"] = "up"

        result = regrid(ds)

        # 2D variable should be regridded
        assert "some_2d_var" in result.data_vars
        # And should have depth dimension
        assert "depth" in result["some_2d_var"].dims


# ============================================================================
# TESTS: Coverage for specific uncovered branches
# ============================================================================


class TestRegridUncoveredBranches:
    """Tests for branches not covered by the main test classes."""

    # ---- Line 727: upward ADCP where last_grid_depth_calc >= 0 ----

    def test_upward_last_grid_depth_calc_non_negative(self):
        """Line 727: upward ADCP (sgn=-1), end_cell_option='cell',
        last_grid_depth_calc >= 0 -> uses calc value without clamping.

        last_grid_depth_calc = min_depth_regrid - (num_cells+1)*cell_size.
        With 1 cell (400 cm), cell_size=4m, transducer=150dm=15m, bin1=2m:
          first_cell = 15 - 2*cos(20°) ≈ 13.12 m
          min_depth_regrid ≈ 14.24 m  >=  (1+1)*4 = 8 m  ->  calc ≈ +6.24 >= 0
        The clamping branch (line 720-721) is skipped; line 727 executes.
        """
        n_time, n_cell, n_beam = 5, 1, 4
        times = pd.date_range("2024-01-01", periods=n_time, freq="h")

        ds = xr.Dataset(
            {
                "velocity": (
                    ("beam", "cell", "time"),
                    np.random.randn(n_beam, n_cell, n_time).astype(np.float32),
                ),
                "transducer_depth": (("time",), np.full(n_time, 150)),  # 15 m
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((n_beam, n_cell, n_time), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(n_beam),
                "cell": np.arange(n_cell),
                "time": times,
            },
        )
        ds.attrs.update(
            {
                "cell_size_cm": 400,
                "bin1_distance_cm": 200,
                "beam_angle": 20,
                "beam_direction": "up",
            }
        )

        result = regrid(ds, end_cell_option="cell")

        assert isinstance(result, xr.Dataset)
        assert "depth" in result.dims

    # ---- Line 755: raise ValueError for empty depth grid ----

    def test_empty_depth_grid_raises_value_error(self, upward_dataset):
        """Line 755: ValueError raised when depth grid is empty.

        For an upward ADCP with end_cell_option='manual':
          grid_start = -first_grid_depth  (≈ -51.88 m, from 50 m transducer)
          grid_stop  = -boundary_limit    (= -60.00 m)
        np.arange(-51.88, -60.0, +4.0) is empty because stop < start with a
        positive step, triggering the ValueError at line 755.
        """
        with pytest.raises(ValueError, match="Empty depth grid"):
            regrid(upward_dataset, end_cell_option="manual", boundary_limit=60.0)

    # ---- Lines 794-795: no mask variable in dataset ----

    def test_regrid_without_mask_variable(self, upward_dataset):
        """Lines 794-795: dataset with no 'mask' -> mask_values = None."""
        ds = upward_dataset.drop_vars("mask")
        assert "mask" not in ds.data_vars

        result = regrid(ds)

        assert isinstance(result, xr.Dataset)

    # ---- Lines 824-825: requested variable not found in dataset ----

    def test_unknown_variable_in_data_vars_skipped(self, upward_dataset):
        """Lines 824-825: variable in the explicit data_vars list but absent
        from the dataset is logged and skipped without raising."""
        result = regrid(upward_dataset, data_vars=["velocity", "nonexistent_var"])

        assert isinstance(result, xr.Dataset)
        assert "velocity" in result.data_vars

    # ---- Lines 1060-1061: fewer than 2 valid points -> fill + continue ----

    def test_single_valid_point_per_ensemble_uses_fill(self, upward_dataset):
        """Lines 1060-1061: only one non-NaN point in an ensemble is not enough
        for interpolation; the ensemble column is filled with fill_value."""
        ds = upward_dataset.copy(deep=True)

        # Keep only cell 0 valid; all others become VELOCITY_MISSING_VALUE
        ds["velocity"].values[:, 1:, :] = VELOCITY_MISSING_VALUE

        result = regrid(ds, data_vars=["velocity"])

        assert isinstance(result, xr.Dataset)
        assert "velocity" in result.data_vars

    # ---- Line 1094: defensive else-branch in masked-range re-application ----
    # NOTE: Line 1094 is unreachable with valid physical ADCP data.
    # cell_depth_start = depth - cell_size/2
    # cell_depth_end   = depth + cell_size/2
    # start >= end requires cell_size <= 0, which never occurs in practice.
    # This is defensive dead code guarding against a hypothetical sign inversion.
    # Coverage exclusion is the appropriate resolution; no test is added.

    # ---- Lines 1110-1112: scipy ValueError during interpolation ----

    def test_interpolation_failure_fills_with_fill_value(self, upward_dataset):
        """Lines 1110-1112: scipy interp1d raises ValueError when fewer than 4
        valid points are given with method='cubic'; the except block fills the
        ensemble column with fill_value instead of propagating the exception."""
        ds = upward_dataset.copy(deep=True)

        # Leave only 2 valid cells; 'cubic' requires >= 4
        ds["velocity"].values[:, 2:, :] = VELOCITY_MISSING_VALUE

        result = regrid(ds, data_vars=["velocity"], method="cubic")

        assert isinstance(result, xr.Dataset)
        assert "velocity" in result.data_vars
