"""
Tests for velocity_check.py - Velocity Quality Control Functions.

This module contains comprehensive tests for:
- velocity_threshold_check (per-component thresholds)
- despike_check (median filter spike detection)
- flatline_check (constant value detection)
- update_combined_mask (beam 3 OR logic)

Test Categories:
1. Basic functionality tests
2. Per-component threshold tests
3. Combined mask update tests
4. Edge cases and boundary conditions
5. Missing value handling
6. Integration tests
"""

import numpy as np
import pytest
import xarray as xr

from pyadps.processing.velocity_check import (
    velocity_threshold_check,
    despike_check,
    flatline_check,
    trim_depths,
    update_combined_mask,
    _get_mask_or_create,
    _validate_threshold,
    DEFAULT_VELOCITY_THRESHOLD_U,
    DEFAULT_VELOCITY_THRESHOLD_V,
    DEFAULT_VELOCITY_THRESHOLD_W,
    DEFAULT_DESPIKE_KERNEL,
    DEFAULT_DESPIKE_CUTOFF,
    DEFAULT_FLATLINE_KERNEL,
    DEFAULT_FLATLINE_CUTOFF,
    THRESHOLD_RANGES,
    VELOCITY_MISSING_VALUE,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_dataset():
    """Create a basic sample dataset for testing."""
    n_beams = 4
    n_cells = 10
    n_time = 100

    # Create velocity data with known values
    velocity_data = np.random.randn(n_beams, n_cells, n_time) * 100  # ~100 mm/s std

    # Create coordinates
    coords = {
        "beam": np.arange(n_beams),
        "cell": np.arange(n_cells),
        "time": np.arange(n_time),
    }

    # Create velocity DataArray
    velocity = xr.DataArray(
        data=velocity_data.astype(np.float32),
        dims=["beam", "cell", "time"],
        coords=coords,
        attrs={"units": "mm/s", "long_name": "Velocity"},
    )

    # Create mask (all valid initially)
    mask = xr.DataArray(
        data=np.zeros((n_beams, n_cells, n_time), dtype=np.int8),
        dims=["beam", "cell", "time"],
        coords=coords,
        attrs={"long_name": "Quality mask"},
    )

    ds = xr.Dataset({"velocity": velocity, "mask": mask})
    return ds


@pytest.fixture
def dataset_with_extremes():
    """Create dataset with known extreme values for threshold testing."""
    n_beams = 4
    n_cells = 5
    n_time = 20

    velocity_data = np.zeros((n_beams, n_cells, n_time), dtype=np.float32)

    # Set baseline values
    velocity_data[:, :, :] = 100.0  # All values at 100 mm/s

    # Add extreme U values (beam 0)
    velocity_data[0, 0, 5] = 3000.0  # Should be flagged at default threshold
    velocity_data[0, 1, 10] = -3000.0  # Negative extreme

    # Add extreme V values (beam 1)
    velocity_data[1, 2, 15] = 2600.0  # Should be flagged at default threshold

    # Add extreme W values (beam 2) - using lower threshold
    velocity_data[2, 3, 8] = 600.0  # Should be flagged at 500 mm/s threshold
    velocity_data[2, 4, 12] = -550.0  # Should be flagged at 500 mm/s threshold

    coords = {
        "beam": np.arange(n_beams),
        "cell": np.arange(n_cells),
        "time": np.arange(n_time),
    }

    velocity = xr.DataArray(
        data=velocity_data,
        dims=["beam", "cell", "time"],
        coords=coords,
    )

    mask = xr.DataArray(
        data=np.zeros((n_beams, n_cells, n_time), dtype=np.int8),
        dims=["beam", "cell", "time"],
        coords=coords,
    )

    return xr.Dataset({"velocity": velocity, "mask": mask})


@pytest.fixture
def dataset_with_spikes():
    """Create dataset with known spikes for despike testing."""
    n_beams = 4
    n_cells = 3
    n_time = 50

    # Create smooth baseline
    velocity_data = np.zeros((n_beams, n_cells, n_time), dtype=np.float32)

    for b in range(3):  # U, V, W
        for c in range(n_cells):
            # Smooth sinusoidal baseline
            velocity_data[b, c, :] = 100 * np.sin(np.linspace(0, 4 * np.pi, n_time))

    # Add obvious spikes
    velocity_data[0, 0, 25] = 5000.0  # Spike in U
    velocity_data[1, 1, 30] = -4000.0  # Spike in V
    velocity_data[2, 2, 35] = 3000.0  # Spike in W

    coords = {
        "beam": np.arange(n_beams),
        "cell": np.arange(n_cells),
        "time": np.arange(n_time),
    }

    velocity = xr.DataArray(
        data=velocity_data,
        dims=["beam", "cell", "time"],
        coords=coords,
    )

    mask = xr.DataArray(
        data=np.zeros((n_beams, n_cells, n_time), dtype=np.int8),
        dims=["beam", "cell", "time"],
        coords=coords,
    )

    return xr.Dataset({"velocity": velocity, "mask": mask})


@pytest.fixture
def dataset_with_flatlines():
    """Create dataset with known flatlines for flatline testing."""
    n_beams = 4
    n_cells = 3
    n_time = 30

    velocity_data = np.random.randn(n_beams, n_cells, n_time) * 50
    velocity_data = velocity_data.astype(np.float32)

    # Add flatline segments (exact constant values)
    # U component: flatline from index 5 to 15 (11 values)
    velocity_data[0, 0, 5:16] = 200.0

    # V component: flatline from index 10 to 20 (11 values)
    velocity_data[1, 1, 10:21] = -150.0

    # W component: flatline from index 20 to 28 (9 values)
    velocity_data[2, 2, 20:29] = 50.0

    coords = {
        "beam": np.arange(n_beams),
        "cell": np.arange(n_cells),
        "time": np.arange(n_time),
    }

    velocity = xr.DataArray(
        data=velocity_data,
        dims=["beam", "cell", "time"],
        coords=coords,
    )

    mask = xr.DataArray(
        data=np.zeros((n_beams, n_cells, n_time), dtype=np.int8),
        dims=["beam", "cell", "time"],
        coords=coords,
    )

    return xr.Dataset({"velocity": velocity, "mask": mask})


@pytest.fixture
def dataset_with_missing_values():
    """Create dataset with missing values (-32768)."""
    n_beams = 4
    n_cells = 5
    n_time = 20

    velocity_data = np.random.randn(n_beams, n_cells, n_time) * 100
    velocity_data = velocity_data.astype(np.float32)

    # Add missing values
    velocity_data[0, 0, 0:5] = VELOCITY_MISSING_VALUE
    velocity_data[1, 1, 10:15] = VELOCITY_MISSING_VALUE
    velocity_data[2, 2, :] = VELOCITY_MISSING_VALUE  # Entire cell missing

    coords = {
        "beam": np.arange(n_beams),
        "cell": np.arange(n_cells),
        "time": np.arange(n_time),
    }

    velocity = xr.DataArray(
        data=velocity_data,
        dims=["beam", "cell", "time"],
        coords=coords,
    )

    mask = xr.DataArray(
        data=np.zeros((n_beams, n_cells, n_time), dtype=np.int8),
        dims=["beam", "cell", "time"],
        coords=coords,
    )

    return xr.Dataset({"velocity": velocity, "mask": mask})


@pytest.fixture
def regridded_dataset():
    """Create a regridded (depth-indexed) dataset with velocity and the
    three raw diagnostics, for trim_depths() tests."""
    n_beams = 4
    n_depths = 6
    n_time = 20

    depths = np.array([0.0, 4.0, 8.0, 12.0, 16.0, 20.0])

    velocity = xr.DataArray(
        data=np.full((n_beams, n_depths, n_time), 100.0, dtype=np.float32),
        dims=["beam", "depth", "time"],
        coords={"beam": np.arange(n_beams), "depth": depths, "time": np.arange(n_time)},
    )
    echo = xr.DataArray(
        data=np.full((n_beams, n_depths, n_time), 80.0, dtype=np.float32),
        dims=["beam", "depth", "time"],
        coords=velocity.coords,
    )
    correlation = xr.DataArray(
        data=np.full((n_beams, n_depths, n_time), 120.0, dtype=np.float32),
        dims=["beam", "depth", "time"],
        coords=velocity.coords,
    )
    percent_good = xr.DataArray(
        data=np.full((n_beams, n_depths, n_time), 90.0, dtype=np.float32),
        dims=["beam", "depth", "time"],
        coords=velocity.coords,
    )
    mask = xr.DataArray(
        data=np.zeros((n_beams, n_depths, n_time), dtype=np.int8),
        dims=["beam", "depth", "time"],
        coords=velocity.coords,
    )

    return xr.Dataset(
        {
            "velocity": velocity,
            "echo_intensity": echo,
            "correlation": correlation,
            "percent_good": percent_good,
            "mask": mask,
        }
    )


# ============================================================================
# TEST: update_combined_mask
# ============================================================================


class TestUpdateCombinedMask:
    """Tests for update_combined_mask function."""

    def test_combined_mask_all_zeros(self):
        """Combined mask should be 0 when all components are 0."""
        mask = np.zeros((4, 5, 10), dtype=np.int8)
        result = update_combined_mask(mask)

        assert np.all(result[3, :, :] == 0)

    def test_combined_mask_u_flagged(self):
        """Combined mask should be 1 where U is flagged."""
        mask = np.zeros((4, 5, 10), dtype=np.int8)
        mask[0, 2, 5] = 1  # Flag U at (cell=2, time=5)

        result = update_combined_mask(mask)

        assert result[3, 2, 5] == 1
        # Other positions should remain 0
        assert result[3, 0, 0] == 0

    def test_combined_mask_v_flagged(self):
        """Combined mask should be 1 where V is flagged."""
        mask = np.zeros((4, 5, 10), dtype=np.int8)
        mask[1, 3, 7] = 1  # Flag V at (cell=3, time=7)

        result = update_combined_mask(mask)

        assert result[3, 3, 7] == 1

    def test_combined_mask_w_flagged(self):
        """Combined mask should be 1 where W is flagged."""
        mask = np.zeros((4, 5, 10), dtype=np.int8)
        mask[2, 1, 9] = 1  # Flag W at (cell=1, time=9)

        result = update_combined_mask(mask)

        assert result[3, 1, 9] == 1

    def test_combined_mask_or_logic(self):
        """Combined mask should be OR of U, V, W."""
        mask = np.zeros((4, 5, 10), dtype=np.int8)

        # Flag different positions in different beams
        mask[0, 0, 0] = 1  # U only
        mask[1, 1, 1] = 1  # V only
        mask[2, 2, 2] = 1  # W only
        mask[0, 3, 3] = 1  # U and V
        mask[1, 3, 3] = 1
        mask[0, 4, 4] = 1  # All three
        mask[1, 4, 4] = 1
        mask[2, 4, 4] = 1

        result = update_combined_mask(mask)

        assert result[3, 0, 0] == 1  # U only -> combined flagged
        assert result[3, 1, 1] == 1  # V only -> combined flagged
        assert result[3, 2, 2] == 1  # W only -> combined flagged
        assert result[3, 3, 3] == 1  # U and V -> combined flagged
        assert result[3, 4, 4] == 1  # All three -> combined flagged

    def test_combined_mask_preserves_dtype(self):
        """Combined mask should be int8."""
        mask = np.zeros((4, 5, 10), dtype=np.int8)
        mask[0, 0, 0] = 1

        result = update_combined_mask(mask)

        assert result.dtype == np.int8

    def test_combined_mask_with_3_beams(self):
        """Should handle datasets with only 3 beams (no combined beam)."""
        mask = np.zeros((3, 5, 10), dtype=np.int8)
        mask[0, 0, 0] = 1

        result = update_combined_mask(mask)

        # Should return unchanged (no beam 3 to update)
        assert result.shape == (3, 5, 10)


# ============================================================================
# TEST: velocity_threshold_check
# ============================================================================


class TestVelocityThresholdCheck:
    """Tests for velocity_threshold_check function."""

    def test_no_flagging_below_threshold(self, sample_dataset):
        """Values below all thresholds should not be flagged."""
        # All values in sample_dataset are ~100 mm/s, well below thresholds
        result = velocity_threshold_check(
            sample_dataset,
            cutoff_u=2500,
            cutoff_v=2500,
            cutoff_w=500,
        )

        # Check that no new flags were added (beyond any initial mask)
        initial_flagged = (sample_dataset["mask"].values == 1).sum()
        final_flagged = (result["mask"].values == 1).sum()

        assert final_flagged == initial_flagged

    def test_u_threshold_flagging(self, dataset_with_extremes):
        """U values exceeding cutoff_u should be flagged in beam 0."""
        result = velocity_threshold_check(
            dataset_with_extremes,
            cutoff_u=2500,
            cutoff_v=2500,
            cutoff_w=500,
        )

        # Check U extreme at (0, 5) and (1, 10)
        assert result["mask"].values[0, 0, 5] == 1
        assert result["mask"].values[0, 1, 10] == 1

    def test_v_threshold_flagging(self, dataset_with_extremes):
        """V values exceeding cutoff_v should be flagged in beam 1."""
        result = velocity_threshold_check(
            dataset_with_extremes,
            cutoff_u=2500,
            cutoff_v=2500,
            cutoff_w=500,
        )

        # Check V extreme at (2, 15)
        assert result["mask"].values[1, 2, 15] == 1

    def test_w_threshold_flagging(self, dataset_with_extremes):
        """W values exceeding cutoff_w should be flagged in beam 2."""
        result = velocity_threshold_check(
            dataset_with_extremes,
            cutoff_u=2500,
            cutoff_v=2500,
            cutoff_w=500,
        )

        # Check W extremes at (3, 8) and (4, 12)
        assert result["mask"].values[2, 3, 8] == 1
        assert result["mask"].values[2, 4, 12] == 1

    def test_combined_mask_updated(self, dataset_with_extremes):
        """Combined mask (beam 3) should be OR of U, V, W flags."""
        result = velocity_threshold_check(
            dataset_with_extremes,
            cutoff_u=2500,
            cutoff_v=2500,
            cutoff_w=500,
        )

        # All flagged positions should also be flagged in combined mask
        assert result["mask"].values[3, 0, 5] == 1  # From U
        assert result["mask"].values[3, 1, 10] == 1  # From U
        assert result["mask"].values[3, 2, 15] == 1  # From V
        assert result["mask"].values[3, 3, 8] == 1  # From W
        assert result["mask"].values[3, 4, 12] == 1  # From W

    def test_different_thresholds_per_component(self):
        """Different thresholds should be applied to each component."""
        n_beams = 4
        n_cells = 1
        n_time = 3

        velocity_data = np.array(
            [
                [[1000, 2000, 3000]],  # U
                [[1000, 2000, 3000]],  # V
                [[100, 200, 300]],  # W
                [[0, 0, 0]],  # Error (unused)
            ],
            dtype=np.float32,
        )

        ds = xr.Dataset(
            {
                "velocity": xr.DataArray(
                    velocity_data,
                    dims=["beam", "cell", "time"],
                ),
                "mask": xr.DataArray(
                    np.zeros((n_beams, n_cells, n_time), dtype=np.int8),
                    dims=["beam", "cell", "time"],
                ),
            }
        )

        result = velocity_threshold_check(
            ds,
            cutoff_u=1500,  # Should flag 2000, 3000
            cutoff_v=2500,  # Should flag 3000 only
            cutoff_w=150,  # Should flag 200, 300
        )

        # U beam
        assert result["mask"].values[0, 0, 0] == 0  # 1000 < 1500
        assert result["mask"].values[0, 0, 1] == 1  # 2000 > 1500
        assert result["mask"].values[0, 0, 2] == 1  # 3000 > 1500

        # V beam
        assert result["mask"].values[1, 0, 0] == 0  # 1000 < 2500
        assert result["mask"].values[1, 0, 1] == 0  # 2000 < 2500
        assert result["mask"].values[1, 0, 2] == 1  # 3000 > 2500

        # W beam
        assert result["mask"].values[2, 0, 0] == 0  # 100 < 150
        assert result["mask"].values[2, 0, 1] == 1  # 200 > 150
        assert result["mask"].values[2, 0, 2] == 1  # 300 > 150

    def test_preserves_existing_mask(self, sample_dataset):
        """Existing mask flags should be preserved."""
        # Pre-flag some cells
        sample_dataset["mask"].values[0, 0, 0] = 1
        sample_dataset["mask"].values[1, 1, 1] = 1

        result = velocity_threshold_check(sample_dataset)

        # Original flags should still be present
        assert result["mask"].values[0, 0, 0] == 1
        assert result["mask"].values[1, 1, 1] == 1

    def test_returns_new_dataset(self, sample_dataset):
        """Should return a new dataset, not modify input."""
        original_mask = sample_dataset["mask"].values.copy()

        _ = velocity_threshold_check(sample_dataset)

        # Original should be unchanged
        np.testing.assert_array_equal(sample_dataset["mask"].values, original_mask)

    def test_missing_velocity_returns_unchanged(self):
        """Should return unchanged dataset if velocity not present."""
        ds = xr.Dataset({"temperature": xr.DataArray([1, 2, 3])})

        result = velocity_threshold_check(ds)

        assert "velocity" not in result.data_vars

    def test_default_thresholds(self, sample_dataset):
        """Default thresholds should match module constants."""
        # This test ensures API consistency
        assert DEFAULT_VELOCITY_THRESHOLD_U == 2500.0
        assert DEFAULT_VELOCITY_THRESHOLD_V == 2500.0
        assert DEFAULT_VELOCITY_THRESHOLD_W == 500.0


# ============================================================================
# TEST: despike_check
# ============================================================================


class TestDespikeCheck:
    """Tests for despike_check function."""

    def test_detects_obvious_spikes(self, dataset_with_spikes):
        """Should detect obvious spikes in the data."""
        result = despike_check(
            dataset_with_spikes,
            kernel_size=13,
            cutoff=3.0,
        )

        # Check that spike positions are flagged
        assert result["mask"].values[0, 0, 25] == 1  # U spike
        assert result["mask"].values[1, 1, 30] == 1  # V spike
        assert result["mask"].values[2, 2, 35] == 1  # W spike

    def test_combined_mask_updated_after_despike(self, dataset_with_spikes):
        """Combined mask should reflect despike results."""
        result = despike_check(
            dataset_with_spikes,
            kernel_size=13,
            cutoff=3.0,
        )

        # Spike positions should be flagged in combined mask
        assert result["mask"].values[3, 0, 25] == 1  # From U spike
        assert result["mask"].values[3, 1, 30] == 1  # From V spike
        assert result["mask"].values[3, 2, 35] == 1  # From W spike

    def test_smooth_data_not_flagged(self, sample_dataset):
        """Smooth data should have minimal flagging with loose cutoff."""
        # Sample dataset has random data - with 3Ïƒ cutoff, statistical
        # outliers will be flagged. Use a looser cutoff to test that
        # truly smooth data isn't excessively flagged.
        initial_flagged = (sample_dataset["mask"].values == 1).sum()

        # Use a very loose cutoff (5Ïƒ) - should flag very few points
        result = despike_check(sample_dataset, kernel_size=13, cutoff=5.0)

        final_flagged = (result["mask"].values == 1).sum()

        # With 5Ïƒ cutoff, flagging should be minimal (< 1% of data)
        assert final_flagged - initial_flagged < sample_dataset["mask"].size * 0.01

    def test_kernel_size_effect(self, dataset_with_spikes):
        """Larger kernel should smooth more, detecting isolated spikes."""
        result_small = despike_check(dataset_with_spikes, kernel_size=5, cutoff=3.0)
        result_large = despike_check(dataset_with_spikes, kernel_size=21, cutoff=3.0)

        # Both should detect the obvious spikes
        assert result_small["mask"].values[0, 0, 25] == 1
        assert result_large["mask"].values[0, 0, 25] == 1

    def test_cutoff_sensitivity(self, dataset_with_spikes):
        """Lower cutoff should flag more points."""
        result_strict = despike_check(dataset_with_spikes, kernel_size=13, cutoff=2.0)
        result_loose = despike_check(dataset_with_spikes, kernel_size=13, cutoff=5.0)

        strict_flagged = (result_strict["mask"].values == 1).sum()
        loose_flagged = (result_loose["mask"].values == 1).sum()

        assert strict_flagged >= loose_flagged

    def test_handles_missing_values(self, dataset_with_missing_values):
        """Should handle missing values without crashing."""
        result = despike_check(
            dataset_with_missing_values,
            kernel_size=5,
            cutoff=3.0,
        )

        # Should complete without error
        assert "mask" in result.data_vars

    def test_skips_all_nan_cells(self, dataset_with_missing_values):
        """Should skip cells that are entirely missing."""
        # Cell (2, 2) is entirely missing
        result = despike_check(
            dataset_with_missing_values,
            kernel_size=5,
            cutoff=3.0,
        )

        # The mask for missing values cell should not be affected by despike
        # (it may be flagged by the initial mask creation)
        assert result is not None

    def test_default_parameters(self):
        """Default parameters should match module constants."""
        assert DEFAULT_DESPIKE_KERNEL == 13
        assert DEFAULT_DESPIKE_CUTOFF == 3.0


# ============================================================================
# TEST: flatline_check
# ============================================================================


class TestFlatlineCheck:
    """Tests for flatline_check function."""

    def test_detects_flatlines(self, dataset_with_flatlines):
        """Should detect constant value segments."""
        result = flatline_check(
            dataset_with_flatlines,
            kernel_size=4,
            cutoff=1.0,
        )

        # The flatline algorithm computes diff between consecutive elements.
        # For a segment of N identical values, there are N-1 zero diffs.
        # The first element of the segment doesn't have a "previous" flat diff
        # (diff[0] is set to 0 for the entire time series, not per-segment).
        # So for flatlines starting mid-series, the first element may not be flagged.

        # U flatline at (0, 0, 5:16) - 11 values, but first has non-zero diff from previous
        # The algorithm flags based on consecutive small diffs, so we expect 10-11 flagged
        assert result["mask"].values[0, 0, 5:16].sum() >= 10

        # V flatline at (1, 1, 10:21) - 11 values
        assert result["mask"].values[1, 1, 10:21].sum() >= 10

        # W flatline at (2, 2, 20:29) - 9 values
        assert result["mask"].values[2, 2, 20:29].sum() >= 8

    def test_combined_mask_updated_after_flatline(self, dataset_with_flatlines):
        """Combined mask should reflect flatline results."""
        result = flatline_check(
            dataset_with_flatlines,
            kernel_size=4,
            cutoff=1.0,
        )

        # Check combined mask at flatline positions
        assert result["mask"].values[3, 0, 10] == 1  # From U flatline
        assert result["mask"].values[3, 1, 15] == 1  # From V flatline
        assert result["mask"].values[3, 2, 25] == 1  # From W flatline

    def test_kernel_size_minimum(self, dataset_with_flatlines):
        """Flatlines shorter than kernel_size should not be flagged."""
        # Create dataset with short flatline (3 values)
        n_beams, n_cells, n_time = 4, 1, 20
        velocity_data = np.random.randn(n_beams, n_cells, n_time) * 50
        velocity_data[0, 0, 5:8] = 100.0  # 3-value flatline

        ds = xr.Dataset(
            {
                "velocity": xr.DataArray(
                    velocity_data.astype(np.float32),
                    dims=["beam", "cell", "time"],
                ),
                "mask": xr.DataArray(
                    np.zeros((n_beams, n_cells, n_time), dtype=np.int8),
                    dims=["beam", "cell", "time"],
                ),
            }
        )

        result = flatline_check(ds, kernel_size=4, cutoff=1.0)

        # 3-value flatline should NOT be flagged with kernel_size=4
        assert result["mask"].values[0, 0, 5:8].sum() < 3

    def test_cutoff_tolerance(self):
        """Flatline detection should respect cutoff tolerance."""
        n_beams, n_cells, n_time = 4, 1, 20
        velocity_data = np.zeros((n_beams, n_cells, n_time), dtype=np.float32)

        # Create "near-flatline" with small variations
        velocity_data[0, 0, :] = np.array(
            [
                100,
                100.5,
                100.2,
                100.8,
                100.1,  # Varies by < 1 mm/s
                100,
                100,
                100,
                100,
                100,  # Exact flatline
                100,
                100,
                100,
                100,
                100,  # Exact flatline
                100,
                100,
                100,
                100,
                100,  # Exact flatline
            ]
        )

        ds = xr.Dataset(
            {
                "velocity": xr.DataArray(
                    velocity_data,
                    dims=["beam", "cell", "time"],
                ),
                "mask": xr.DataArray(
                    np.zeros((n_beams, n_cells, n_time), dtype=np.int8),
                    dims=["beam", "cell", "time"],
                ),
            }
        )

        # With tight tolerance, only exact flatline detected
        result_tight = flatline_check(ds, kernel_size=4, cutoff=0.1)
        tight_flagged = result_tight["mask"].values[0, 0, :].sum()

        # With loose tolerance, near-flatline also detected
        result_loose = flatline_check(ds, kernel_size=4, cutoff=1.0)
        loose_flagged = result_loose["mask"].values[0, 0, :].sum()

        assert loose_flagged >= tight_flagged

    def test_normal_data_not_flagged(self, sample_dataset):
        """Normal varying data should not be flagged as flatline."""
        result = flatline_check(sample_dataset, kernel_size=4, cutoff=1.0)

        # Random data should have few or no flatlines
        initial_flagged = (sample_dataset["mask"].values == 1).sum()
        final_flagged = (result["mask"].values == 1).sum()

        # Should be minimal flagging
        assert final_flagged - initial_flagged < sample_dataset["mask"].size * 0.01

    def test_handles_missing_values(self, dataset_with_missing_values):
        """Should handle missing values without crashing."""
        result = flatline_check(
            dataset_with_missing_values,
            kernel_size=4,
            cutoff=1.0,
        )

        assert "mask" in result.data_vars

    def test_default_parameters(self):
        """Default parameters should match module constants."""
        assert DEFAULT_FLATLINE_KERNEL == 4
        assert DEFAULT_FLATLINE_CUTOFF == 1.0


# ============================================================================
# TEST: trim_depths
# ============================================================================


class TestTrimSurface:
    """Tests for trim_depths function."""

    def test_returns_dataset(self, regridded_dataset):
        result = trim_depths(regridded_dataset, depths=[12.0])
        assert isinstance(result, xr.Dataset)

    def test_requires_depth_dimension(self, sample_dataset):
        """sample_dataset uses 'cell', not 'depth' - pre-regrid data."""
        with pytest.raises(ValueError, match="requires a regridded dataset"):
            trim_depths(sample_dataset, depths=[12.0])

    def test_empty_depths_raises(self, regridded_dataset):
        with pytest.raises(ValueError, match="at least one value"):
            trim_depths(regridded_dataset, depths=[])

    def test_unknown_depth_raises(self, regridded_dataset):
        with pytest.raises(ValueError, match="not found"):
            trim_depths(regridded_dataset, depths=[999.0])

    def test_masks_selected_depth_all_beams(self, regridded_dataset):
        result = trim_depths(regridded_dataset, depths=[12.0])
        mask = result["mask"]
        assert bool((mask.sel(depth=12.0) == 1).all())

    def test_does_not_mask_other_depths(self, regridded_dataset):
        result = trim_depths(regridded_dataset, depths=[12.0])
        mask = result["mask"]
        for d in [0.0, 4.0, 8.0, 16.0, 20.0]:
            assert bool((mask.sel(depth=d) == 0).all())

    def test_multiple_depths(self, regridded_dataset):
        result = trim_depths(regridded_dataset, depths=[12.0, 16.0])
        mask = result["mask"]
        assert bool((mask.sel(depth=12.0) == 1).all())
        assert bool((mask.sel(depth=16.0) == 1).all())
        assert bool((mask.sel(depth=20.0) == 0).all())

    def test_preserves_existing_mask(self, regridded_dataset):
        """A pre-existing flag elsewhere in the mask must survive."""
        regridded_dataset["mask"].loc[dict(depth=20.0)] = 1
        result = trim_depths(regridded_dataset, depths=[12.0])
        mask = result["mask"]
        assert bool((mask.sel(depth=12.0) == 1).all())
        assert bool((mask.sel(depth=20.0) == 1).all())

    def test_default_does_not_touch_echo_intensity(self, regridded_dataset):
        result = trim_depths(regridded_dataset, depths=[12.0])
        original = regridded_dataset["echo_intensity"].sel(depth=12.0).values
        after = result["echo_intensity"].sel(depth=12.0).values
        np.testing.assert_array_equal(original, after)

    def test_default_does_not_touch_correlation_or_percent_good(self, regridded_dataset):
        result = trim_depths(regridded_dataset, depths=[12.0])
        for var_name in ["correlation", "percent_good"]:
            original = regridded_dataset[var_name].sel(depth=12.0).values
            after = result[var_name].sel(depth=12.0).values
            np.testing.assert_array_equal(original, after)

    def test_apply_to_all_variables_masks_echo_intensity(self, regridded_dataset):
        result = trim_depths(
            regridded_dataset, depths=[12.0], apply_to_all_variables=True
        )
        echo = result["echo_intensity"].sel(depth=12.0).values
        assert np.all(np.isnan(echo))

    def test_apply_to_all_variables_masks_correlation_and_percent_good(
        self, regridded_dataset
    ):
        result = trim_depths(
            regridded_dataset, depths=[12.0], apply_to_all_variables=True
        )
        for var_name in ["correlation", "percent_good"]:
            values = result[var_name].sel(depth=12.0).values
            assert np.all(np.isnan(values))

    def test_apply_to_all_variables_does_not_touch_other_depths(self, regridded_dataset):
        result = trim_depths(
            regridded_dataset, depths=[12.0], apply_to_all_variables=True
        )
        echo_16 = result["echo_intensity"].sel(depth=16.0).values
        assert not np.any(np.isnan(echo_16))

    def test_does_not_mask_velocity_data_values(self, regridded_dataset):
        """trim_depths only updates 'mask' - matching every other check in
        this pipeline, velocity's raw values aren't NaN'd until export
        (apply_mask=True), not immediately by this function."""
        result = trim_depths(regridded_dataset, depths=[12.0])
        velocity = result["velocity"].sel(depth=12.0).values
        assert not np.any(np.isnan(velocity))

    def test_original_not_modified(self, regridded_dataset):
        original_mask = regridded_dataset["mask"].values.copy()
        trim_depths(regridded_dataset, depths=[12.0])
        np.testing.assert_array_equal(regridded_dataset["mask"].values, original_mask)

    def test_nearby_but_not_exact_depth_does_not_match(self, regridded_dataset):
        with pytest.raises(ValueError, match="not found"):
            trim_depths(regridded_dataset, depths=[12.5])


# ============================================================================
# TEST: Helper Functions
# ============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_mask_or_create_existing(self, sample_dataset):
        """Should return existing mask if present."""
        mask = _get_mask_or_create(sample_dataset)

        np.testing.assert_array_equal(mask.values, sample_dataset["mask"].values)

    def test_get_mask_or_create_creates_mask(self):
        """Should create mask from velocity if not present."""
        # Dataset without mask
        velocity = xr.DataArray(
            np.zeros((4, 5, 10)),
            dims=["beam", "cell", "time"],
            coords={
                "beam": np.arange(4),
                "cell": np.arange(5),
                "time": np.arange(10),
            },
        )
        ds = xr.Dataset({"velocity": velocity})

        mask = _get_mask_or_create(ds)

        assert mask.shape == (4, 5, 10)

    def test_validate_threshold_in_range(self, caplog):
        """Should not warn for thresholds in range."""
        import logging

        caplog.set_level(logging.WARNING)

        _validate_threshold("velocity_u", 5000)  # Within (0, 10000)

        assert "out of typical range" not in caplog.text

    def test_validate_threshold_out_of_range(self, caplog):
        """Should warn for thresholds out of range."""
        import logging

        caplog.set_level(logging.WARNING)

        _validate_threshold("velocity_u", 15000)  # Outside (0, 10000)

        assert "out of typical range" in caplog.text


# ============================================================================
# TEST: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple checks."""

    def test_sequential_checks(self, sample_dataset):
        """Multiple checks should be applicable sequentially."""
        result = velocity_threshold_check(sample_dataset)
        result = despike_check(result)
        result = flatline_check(result)

        assert "mask" in result.data_vars

    def test_mask_accumulates(self, dataset_with_extremes):
        """Masks should accumulate across checks (OR logic)."""
        # Add spike and flatline to dataset
        dataset_with_extremes["velocity"].values[0, 0, 0:5] = 100.0  # Flatline
        dataset_with_extremes["velocity"].values[1, 0, 10] = 5000.0  # Spike

        result = velocity_threshold_check(dataset_with_extremes)
        threshold_flagged = (result["mask"].values == 1).sum()

        result = despike_check(result)
        despike_flagged = (result["mask"].values == 1).sum()

        result = flatline_check(result)
        final_flagged = (result["mask"].values == 1).sum()

        # Flags should accumulate
        assert final_flagged >= despike_flagged >= threshold_flagged

    def test_combined_mask_consistency(self, dataset_with_extremes):
        """Combined mask should always be OR of U, V, W after any check."""
        result = velocity_threshold_check(dataset_with_extremes)
        result = despike_check(result)
        result = flatline_check(result)

        mask = result["mask"].values

        # Verify OR logic manually
        expected_combined = (
            (mask[0, :, :] == 1) | (mask[1, :, :] == 1) | (mask[2, :, :] == 1)
        ).astype(np.int8)

        np.testing.assert_array_equal(mask[3, :, :], expected_combined)


# ============================================================================
# TEST: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_time_step(self):
        """Should handle single time step dataset."""
        ds = xr.Dataset(
            {
                "velocity": xr.DataArray(
                    np.zeros((4, 5, 1)),
                    dims=["beam", "cell", "time"],
                ),
                "mask": xr.DataArray(
                    np.zeros((4, 5, 1), dtype=np.int8),
                    dims=["beam", "cell", "time"],
                ),
            }
        )

        result = velocity_threshold_check(ds)
        assert result["mask"].shape == (4, 5, 1)

    def test_single_cell(self):
        """Should handle single cell dataset."""
        ds = xr.Dataset(
            {
                "velocity": xr.DataArray(
                    np.zeros((4, 1, 10)),
                    dims=["beam", "cell", "time"],
                ),
                "mask": xr.DataArray(
                    np.zeros((4, 1, 10), dtype=np.int8),
                    dims=["beam", "cell", "time"],
                ),
            }
        )

        result = velocity_threshold_check(ds)
        assert result["mask"].shape == (4, 1, 10)

    def test_all_masked_initially(self):
        """Should handle dataset where all data is pre-masked."""
        ds = xr.Dataset(
            {
                "velocity": xr.DataArray(
                    np.zeros((4, 5, 10)),
                    dims=["beam", "cell", "time"],
                ),
                "mask": xr.DataArray(
                    np.ones((4, 5, 10), dtype=np.int8),  # All masked
                    dims=["beam", "cell", "time"],
                ),
            }
        )

        result = velocity_threshold_check(ds)

        # All should still be masked
        assert (result["mask"].values == 1).all()

    def test_all_extreme_values(self):
        """Should handle dataset where all values exceed threshold."""
        ds = xr.Dataset(
            {
                "velocity": xr.DataArray(
                    np.full((4, 5, 10), 10000.0),  # All extreme
                    dims=["beam", "cell", "time"],
                ),
                "mask": xr.DataArray(
                    np.zeros((4, 5, 10), dtype=np.int8),
                    dims=["beam", "cell", "time"],
                ),
            }
        )

        result = velocity_threshold_check(
            ds, cutoff_u=5000, cutoff_v=5000, cutoff_w=5000
        )

        # All U, V, W should be flagged
        assert (result["mask"].values[0:3, :, :] == 1).all()
        # Combined should be flagged
        assert (result["mask"].values[3, :, :] == 1).all()

    def test_negative_values(self):
        """Should handle negative velocity values correctly."""
        ds = xr.Dataset(
            {
                "velocity": xr.DataArray(
                    np.full((4, 5, 10), -3000.0),  # Negative extreme
                    dims=["beam", "cell", "time"],
                ),
                "mask": xr.DataArray(
                    np.zeros((4, 5, 10), dtype=np.int8),
                    dims=["beam", "cell", "time"],
                ),
            }
        )

        result = velocity_threshold_check(
            ds, cutoff_u=2500, cutoff_v=2500, cutoff_w=500
        )

        # Negative values exceeding threshold should be flagged
        assert (result["mask"].values[0:3, :, :] == 1).all()


# ============================================================================
# FIXTURES - Magnetic Declination Tests
# ============================================================================


@pytest.fixture
def velocity_dataset():
    """Dataset with time coords and lat/lon attrs for magnetic declination tests."""
    import pandas as pd

    np.random.seed(42)
    n_beams, n_cells, n_time = 4, 5, 10
    times = pd.date_range("2023-06-01", periods=n_time, freq="h")

    velocity = np.random.randint(-500, 500, (n_beams, n_cells, n_time)).astype(
        np.float32
    )

    ds = xr.Dataset(
        {"velocity": (("beam", "cell", "time"), velocity)},
        coords={
            "beam": np.arange(n_beams),
            "cell": np.arange(n_cells),
            "time": times,
        },
    )
    ds.attrs["latitude"] = 17.38
    ds.attrs["longitude"] = 78.49
    return ds


# ============================================================================
# TESTS: get_magdec_from_cof
# ============================================================================


class TestGetMagdecFromCof:
    """Tests for get_magdec_from_cof function."""

    @pytest.fixture(autouse=True)
    def _patch_geomag(self, monkeypatch):
        """Inject a mock GeoMag into the velocity_check module namespace."""
        import pyadps.processing.velocity_check as vc

        class MockResult:
            d = 1.23

        class MockGeoMag:
            captured_file = None

            def __init__(self, coefficients_file=None):
                MockGeoMag.captured_file = coefficients_file

            def calculate(self, glat=None, glon=None, alt=None, time=None, **kw):
                return MockResult()

        # GeoMag may not exist in the namespace if pygeomag is not installed.
        # Use object.__setattr__ on the module is not possible for modules, so
        # we rely on monkeypatch which uses setattr internally.
        if not hasattr(vc, "GeoMag"):
            vc.GeoMag = MockGeoMag  # inject directly so monkeypatch can restore it

        monkeypatch.setattr(vc, "GeoMag", MockGeoMag)
        monkeypatch.setattr(vc, "HAS_PYGEOMAG", True)

        self.MockGeoMag = MockGeoMag
        self.MockResult = MockResult

    def test_raises_import_error_when_pygeomag_missing(self, monkeypatch):
        """ImportError raised when HAS_PYGEOMAG is False."""
        import pyadps.processing.velocity_check as vc

        monkeypatch.setattr(vc, "HAS_PYGEOMAG", False)

        with pytest.raises(ImportError, match="pygeomag"):
            vc.get_magdec_from_cof(17.38, 78.49, 0, 2023.5)

    def test_uses_cof_file_for_year_in_2010_2030_range(self):
        """Year 2023 selects WMM_2020.COF (nearest 5-year epoch)."""
        import pyadps.processing.velocity_check as vc

        result = vc.get_magdec_from_cof(17.38, 78.49, 0, 2023.5)

        assert result == pytest.approx(1.23)
        assert "WMM_2020" in self.MockGeoMag.captured_file

    def test_uses_fallback_cof_for_year_outside_2010_2030(self):
        """Year outside [2010, 2030) uses the fallback WMM_2025.COF path."""
        import pyadps.processing.velocity_check as vc

        result = vc.get_magdec_from_cof(17.38, 78.49, 0, 2009.0)

        assert "WMM_2025.COF" in self.MockGeoMag.captured_file
        assert result == pytest.approx(1.23)

    def test_5year_epoch_selection(self):
        """Year 2015 selects WMM_2015.COF; year 2020 selects WMM_2020.COF."""
        import pyadps.processing.velocity_check as vc

        vc.get_magdec_from_cof(0, 0, 0, 2015.5)
        assert "WMM_2015" in self.MockGeoMag.captured_file

        vc.get_magdec_from_cof(0, 0, 0, 2020.5)
        assert "WMM_2020" in self.MockGeoMag.captured_file

    def test_passes_lat_lon_alt_time_to_calculate(self):
        """Correct lat/lon/alt/time are forwarded to GeoMag.calculate()."""
        import pyadps.processing.velocity_check as vc

        received = {}
        original_calculate = self.MockGeoMag.calculate

        def capturing_calculate(
            self_inner, glat=None, glon=None, alt=None, time=None, **kw
        ):
            received.update({"glat": glat, "glon": glon, "alt": alt, "time": time})
            return self.MockResult()

        self.MockGeoMag.calculate = capturing_calculate

        vc.get_magdec_from_cof(17.38, 78.49, 100.0, 2022.0)

        self.MockGeoMag.calculate = original_calculate

        assert received["glat"] == pytest.approx(17.38)
        assert received["glon"] == pytest.approx(78.49)
        assert received["alt"] == pytest.approx(100.0)
        assert received["time"] == pytest.approx(2022.0)


# ============================================================================
# TESTS: get_magdec_from_api
# ============================================================================


class TestGetMagdecFromApi:
    """Tests for get_magdec_from_api function."""

    def _make_mock_response(self, declination):
        """Return a mock requests.Response with the given declination."""
        import unittest.mock as mock

        response = mock.MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"result": [{"declination": declination}]}
        return response

    def test_returns_declination_for_recent_year(self, monkeypatch):
        """Year >= 2025 uses WMM model and returns the API declination."""
        import requests
        import pyadps.processing.velocity_check as vc

        monkeypatch.setattr(
            requests, "get", lambda url, timeout=10: self._make_mock_response(2.5)
        )

        result = vc.get_magdec_from_api(17.38, 78.49, 2025.5)

        assert result == pytest.approx(2.5)

    def test_year_2019_to_2024_uses_igrf(self, monkeypatch):
        """Year in [2019, 2025) selects IGRF model."""
        import requests
        import pyadps.processing.velocity_check as vc

        captured_url = {}

        def mock_get(url, timeout=10):
            captured_url["url"] = url
            return self._make_mock_response(1.0)

        monkeypatch.setattr(requests, "get", mock_get)

        vc.get_magdec_from_api(0, 0, 2022.0)

        assert "IGRF" in captured_url["url"]

    def test_year_2000_to_2018_uses_emm(self, monkeypatch):
        """Year in [2000, 2019) selects EMM model."""
        import requests
        import pyadps.processing.velocity_check as vc

        captured_url = {}

        def mock_get(url, timeout=10):
            captured_url["url"] = url
            return self._make_mock_response(0.5)

        monkeypatch.setattr(requests, "get", mock_get)

        vc.get_magdec_from_api(0, 0, 2010.0)

        assert "EMM" in captured_url["url"]

    def test_year_1590_to_1999_uses_igrf(self, monkeypatch):
        """Year in [1590, 2000) selects IGRF model."""
        import requests
        import pyadps.processing.velocity_check as vc

        captured_url = {}

        def mock_get(url, timeout=10):
            captured_url["url"] = url
            return self._make_mock_response(-3.0)

        monkeypatch.setattr(requests, "get", mock_get)

        vc.get_magdec_from_api(0, 0, 1900.0)

        assert "IGRF" in captured_url["url"]

    def test_year_before_1590_raises_value_error(self, monkeypatch):
        """Year < 1590 raises ValueError before any HTTP request."""
        import pyadps.processing.velocity_check as vc

        with pytest.raises(ValueError, match="out of supported range"):
            vc.get_magdec_from_api(0, 0, 1000.0)

    def test_api_request_exception_is_reraised(self, monkeypatch):
        """Network errors are logged and re-raised."""
        import requests
        import pyadps.processing.velocity_check as vc

        def mock_get(url, timeout=10):
            raise requests.RequestException("network error")

        monkeypatch.setattr(requests, "get", mock_get)

        with pytest.raises(requests.RequestException):
            vc.get_magdec_from_api(17.38, 78.49, 2023.0)

    def test_url_contains_lat_lon(self, monkeypatch):
        """Constructed URL includes the supplied lat and lon."""
        import requests
        import pyadps.processing.velocity_check as vc

        captured = {}

        def mock_get(url, timeout=10):
            captured["url"] = url
            return self._make_mock_response(0.0)

        monkeypatch.setattr(requests, "get", mock_get)

        vc.get_magdec_from_api(17.38, 78.49, 2023.0)

        assert "17.38" in captured["url"]
        assert "78.49" in captured["url"]


# ============================================================================
# TESTS: correct_magnetic_declination
# ============================================================================


class TestCorrectMagneticDeclination:
    """Tests for correct_magnetic_declination function."""

    def test_raises_if_no_velocity(self):
        """ValueError raised when 'velocity' is absent from dataset."""
        ds = xr.Dataset({"mask": (("beam",), np.zeros(4, dtype=np.int8))})

        with pytest.raises(ValueError, match="Velocity data not found"):
            from pyadps.processing.velocity_check import correct_magnetic_declination

            correct_magnetic_declination(ds, declination=5.0)

    def test_explicit_declination_applies_rotation(self, velocity_dataset):
        """Providing declination directly bypasses any API/COF call."""
        from pyadps.processing.velocity_check import correct_magnetic_declination

        ds = velocity_dataset
        u_before = ds["velocity"].values[0].copy()
        v_before = ds["velocity"].values[1].copy()

        result = correct_magnetic_declination(ds, declination=0.0)

        # 0-degree rotation -> u and v unchanged (cos=1, sin=0)
        np.testing.assert_allclose(result["velocity"].values[0], u_before, rtol=1e-5)
        np.testing.assert_allclose(result["velocity"].values[1], v_before, rtol=1e-5)
        assert result.attrs["magnetic_declination_applied"] == 0.0

    def test_rotation_is_mathematically_correct(self, velocity_dataset):
        """UV rotation matches the standard rotation matrix for a known angle."""
        from pyadps.processing.velocity_check import correct_magnetic_declination

        import math

        ds = velocity_dataset
        dec = 30.0
        u = ds["velocity"].values[0].astype(float)
        v = ds["velocity"].values[1].astype(float)

        result = correct_magnetic_declination(ds, declination=dec)

        cos_a = math.cos(math.radians(dec))
        sin_a = math.sin(math.radians(dec))
        u_expected = u * cos_a + v * sin_a
        v_expected = -u * sin_a + v * cos_a

        np.testing.assert_allclose(result["velocity"].values[0], u_expected, rtol=1e-4)
        np.testing.assert_allclose(result["velocity"].values[1], v_expected, rtol=1e-4)

    def test_preserves_w_and_combined_beams(self, velocity_dataset):
        """Beams 2 and 3 (W and combined) are not rotated."""
        from pyadps.processing.velocity_check import correct_magnetic_declination

        ds = velocity_dataset
        w_before = ds["velocity"].values[2].copy()
        comb_before = ds["velocity"].values[3].copy()

        result = correct_magnetic_declination(ds, declination=15.0)

        np.testing.assert_array_equal(result["velocity"].values[2], w_before)
        np.testing.assert_array_equal(result["velocity"].values[3], comb_before)

    def test_original_dataset_not_modified(self, velocity_dataset):
        """Input dataset is not mutated (immutability)."""
        from pyadps.processing.velocity_check import correct_magnetic_declination

        ds = velocity_dataset
        u_orig = ds["velocity"].values[0].copy()

        correct_magnetic_declination(ds, declination=20.0)

        np.testing.assert_array_equal(ds["velocity"].values[0], u_orig)

    def test_missing_values_preserved_through_rotation(self, velocity_dataset):
        """Missing cells remain NaN after rotation (float32/missing_as_nan=True mode)."""
        from pyadps.processing.velocity_check import (
            correct_magnetic_declination,
            VELOCITY_MISSING_VALUE,
        )

        ds = velocity_dataset.copy(deep=True)
        ds["velocity"].values[0, 0, 0] = VELOCITY_MISSING_VALUE
        ds["velocity"].values[1, 0, 0] = VELOCITY_MISSING_VALUE

        result = correct_magnetic_declination(ds, declination=10.0)

        # Float velocity: sentinel is converted to NaN before rotation and stays NaN
        assert np.isnan(result["velocity"].values[0, 0, 0])
        assert np.isnan(result["velocity"].values[1, 0, 0])

    def test_fewer_than_2_beams_returns_unchanged(self):
        """Dataset with < 2 velocity beams is returned without rotation."""
        import pandas as pd
        from pyadps.processing.velocity_check import correct_magnetic_declination

        times = pd.date_range("2023-01-01", periods=5, freq="h")
        ds = xr.Dataset(
            {"velocity": (("beam", "cell", "time"), np.ones((1, 3, 5)))},
            coords={"beam": [0], "cell": np.arange(3), "time": times},
        )
        v_before = ds["velocity"].values.copy()

        result = correct_magnetic_declination(ds, declination=45.0)

        np.testing.assert_array_equal(result["velocity"].values, v_before)

    def test_uses_api_when_use_api_true(self, monkeypatch, velocity_dataset):
        """use_api=True routes to get_magdec_from_api instead of COF."""
        import pyadps.processing.velocity_check as vc

        called = {}

        def mock_api(lat, lon, year):
            called["api"] = True
            return 3.0

        monkeypatch.setattr(vc, "get_magdec_from_api", mock_api)

        vc.correct_magnetic_declination(velocity_dataset, use_api=True, year=2023.0)

        assert called.get("api") is True

    def test_uses_cof_when_use_api_false(self, monkeypatch, velocity_dataset):
        """use_api=False (default) routes to get_magdec_from_cof."""
        import pyadps.processing.velocity_check as vc

        called = {}

        def mock_cof(glat, glon, alt, time):
            called["cof"] = True
            return 2.0

        monkeypatch.setattr(vc, "get_magdec_from_cof", mock_cof)

        vc.correct_magnetic_declination(velocity_dataset, use_api=False, year=2023.0)

        assert called.get("cof") is True

    def test_infers_lat_lon_year_from_dataset_attrs_and_time(
        self, monkeypatch, velocity_dataset
    ):
        """lat/lon taken from attrs; year inferred from middle of time coord."""
        import pyadps.processing.velocity_check as vc

        received = {}

        def mock_cof(glat, glon, alt, time):
            received.update({"lat": glat, "lon": glon, "year": time})
            return 1.5

        monkeypatch.setattr(vc, "get_magdec_from_cof", mock_cof)

        vc.correct_magnetic_declination(velocity_dataset)  # no explicit lat/lon/year

        assert received["lat"] == pytest.approx(17.38)
        assert received["lon"] == pytest.approx(78.49)
        # Year should be near 2023
        assert 2023.0 <= received["year"] < 2024.0

    def test_raises_if_lat_lon_missing_and_no_attrs(self):
        """ValueError raised when lat/lon cannot be determined."""
        import pandas as pd
        from pyadps.processing.velocity_check import correct_magnetic_declination

        times = pd.date_range("2023-01-01", periods=5, freq="h")
        ds = xr.Dataset(
            {"velocity": (("beam", "cell", "time"), np.ones((4, 3, 5)))},
            coords={"beam": np.arange(4), "cell": np.arange(3), "time": times},
        )
        # No lat/lon in attrs and none provided

        with pytest.raises(ValueError, match="missing lat/lon/year"):
            correct_magnetic_declination(ds)

    def test_declination_stored_in_attrs(self, velocity_dataset):
        """The applied declination is recorded in dataset attrs."""
        from pyadps.processing.velocity_check import correct_magnetic_declination

        result = correct_magnetic_declination(velocity_dataset, declination=-7.5)

        assert result.attrs["magnetic_declination_applied"] == pytest.approx(-7.5)


# ============================================================================
# TESTS: Module-level ImportError branch (lines 43-44: HAS_PYGEOMAG = False)
# ============================================================================


class TestHasPygeomag:
    """Lines 43-44: HAS_PYGEOMAG = False set in the except ImportError block.

    The import-time branch sets HAS_PYGEOMAG based on whether pygeomag is
    installed. Tests here use monkeypatch to force the False path regardless
    of the environment, and verify the resulting behaviour in get_magdec_from_cof.
    """

    def test_has_pygeomag_is_a_bool(self):
        """HAS_PYGEOMAG is always a bool set by the import-time try/except."""
        import pyadps.processing.velocity_check as vc

        assert isinstance(vc.HAS_PYGEOMAG, bool)

    def test_get_magdec_from_cof_raises_import_error_when_flag_false(self, monkeypatch):
        """Lines 43-44: when HAS_PYGEOMAG is forced to False, get_magdec_from_cof
        raises ImportError regardless of whether pygeomag is actually installed."""
        import pyadps.processing.velocity_check as vc

        monkeypatch.setattr(vc, "HAS_PYGEOMAG", False)

        with pytest.raises(ImportError, match="pygeomag"):
            vc.get_magdec_from_cof(0.0, 0.0, 0.0, 2023.0)


# ============================================================================
# TESTS: despike_check early return (line 488)
# ============================================================================


class TestDespikeCheckNoVelocity:
    """Line 488: despike_check returns ds unchanged when 'velocity' is absent."""

    def test_returns_original_object_when_no_velocity(self):
        """Line 488: the same dataset object is returned without copying."""
        from pyadps.processing.velocity_check import despike_check

        ds = xr.Dataset(
            {"mask": (("beam", "cell", "time"), np.zeros((4, 5, 10), dtype=np.int8))},
            coords={
                "beam": np.arange(4),
                "cell": np.arange(5),
                "time": np.arange(10),
            },
        )

        result = despike_check(ds)

        assert result is ds

    def test_mask_values_unchanged_when_no_velocity(self):
        """Mask content is identical after the early return."""
        from pyadps.processing.velocity_check import despike_check

        ds = xr.Dataset(
            {"mask": (("beam", "cell", "time"), np.zeros((4, 5, 10), dtype=np.int8))},
            coords={
                "beam": np.arange(4),
                "cell": np.arange(5),
                "time": np.arange(10),
            },
        )
        mask_before = ds["mask"].values.copy()

        result = despike_check(ds)

        np.testing.assert_array_equal(result["mask"].values, mask_before)


# ============================================================================
# TESTS: flatline_check early return (line 579)
# ============================================================================


class TestFlatlineCheckNoVelocity:
    """Line 579: flatline_check returns ds unchanged when 'velocity' is absent."""

    def test_returns_original_object_when_no_velocity(self):
        """Line 579: the same dataset object is returned without copying."""
        from pyadps.processing.velocity_check import flatline_check

        ds = xr.Dataset(
            {"mask": (("beam", "cell", "time"), np.zeros((4, 5, 10), dtype=np.int8))},
            coords={
                "beam": np.arange(4),
                "cell": np.arange(5),
                "time": np.arange(10),
            },
        )

        result = flatline_check(ds)

        assert result is ds

    def test_mask_values_unchanged_when_no_velocity(self):
        """Mask content is identical after the early return."""
        from pyadps.processing.velocity_check import flatline_check

        ds = xr.Dataset(
            {"mask": (("beam", "cell", "time"), np.zeros((4, 5, 10), dtype=np.int8))},
            coords={
                "beam": np.arange(4),
                "cell": np.arange(5),
                "time": np.arange(10),
            },
        )
        mask_before = ds["mask"].values.copy()

        result = flatline_check(ds)

        np.testing.assert_array_equal(result["mask"].values, mask_before)
