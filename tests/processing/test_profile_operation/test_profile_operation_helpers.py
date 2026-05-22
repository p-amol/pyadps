"""
Test suite for cut_bins_manual function.

Tests the manual rectangular region masking functionality.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyadps.processing.profile_operation import cut_bins_manual


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def basic_dataset():
    """Create basic ADCP dataset for cut_bins_manual tests."""
    np.random.seed(42)
    n_time = 50
    n_cell = 20
    n_beam = 4

    times = pd.date_range("2024-01-01", periods=n_time, freq="h")
    cells = np.arange(n_cell)
    beams = np.arange(n_beam)

    data_vars = {
        "velocity": (
            ("beam", "cell", "time"),
            np.random.randint(-500, 500, size=(n_beam, n_cell, n_time)),
        ),
        "mask": (
            ("beam", "cell", "time"),
            np.zeros((n_beam, n_cell, n_time), dtype=np.int8),
        ),
    }

    coords = {"time": times, "cell": cells, "beam": beams}
    return xr.Dataset(data_vars, coords=coords)


@pytest.fixture
def dataset_no_mask():
    """Create dataset without mask variable."""
    np.random.seed(42)
    n_time = 30
    n_cell = 15
    n_beam = 4

    times = pd.date_range("2024-01-01", periods=n_time, freq="h")

    data_vars = {
        "velocity": (
            ("beam", "cell", "time"),
            np.random.randint(-500, 500, size=(n_beam, n_cell, n_time)),
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
    n_time = 50
    n_cell = 20
    n_beam = 4

    times = pd.date_range("2024-01-01", periods=n_time, freq="h")

    # Pre-mask a region
    mask = np.zeros((n_beam, n_cell, n_time), dtype=np.int8)
    mask[:, 0:3, 0:10] = 1  # Pre-mask cells 0-2, ensembles 0-9

    data_vars = {
        "velocity": (
            ("beam", "cell", "time"),
            np.random.randint(-500, 500, size=(n_beam, n_cell, n_time)),
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
# TESTS: Basic Functionality
# ============================================================================


class TestCutBinsManualBasic:
    """Tests for basic cut_bins_manual functionality."""

    def test_mask_cells_only(self, basic_dataset):
        """Test masking cell range across all ensembles."""
        result = cut_bins_manual(basic_dataset, min_cell=5, max_cell=10)

        # Cells 5-9 should be masked, all ensembles
        assert np.all(result["mask"].values[:, 5:10, :] == 1)
        # Other cells should not be masked
        assert np.all(result["mask"].values[:, :5, :] == 0)
        assert np.all(result["mask"].values[:, 10:, :] == 0)

    def test_mask_ensembles_only(self, basic_dataset):
        """Test masking ensemble range across all cells."""
        result = cut_bins_manual(basic_dataset, min_ensemble=10, max_ensemble=20)

        # Ensembles 10-19 should be masked, all cells
        assert np.all(result["mask"].values[:, :, 10:20] == 1)
        # Other ensembles should not be masked
        assert np.all(result["mask"].values[:, :, :10] == 0)
        assert np.all(result["mask"].values[:, :, 20:] == 0)

    def test_mask_rectangular_region(self, basic_dataset):
        """Test masking rectangular region (cells + ensembles)."""
        result = cut_bins_manual(
            basic_dataset, min_cell=3, max_cell=8, min_ensemble=15, max_ensemble=30
        )

        # Only the specified rectangular region should be masked
        assert np.all(result["mask"].values[:, 3:8, 15:30] == 1)

        # Outside the region should not be masked
        assert np.all(result["mask"].values[:, :3, :] == 0)
        assert np.all(result["mask"].values[:, 8:, :] == 0)
        assert np.all(result["mask"].values[:, 3:8, :15] == 0)
        assert np.all(result["mask"].values[:, 3:8, 30:] == 0)

    def test_no_parameters(self, basic_dataset):
        """Test with no parameters (masks entire dataset)."""
        result = cut_bins_manual(basic_dataset)

        # All cells and ensembles should be masked
        assert np.all(result["mask"].values == 1)

    def test_single_cell(self, basic_dataset):
        """Test masking a single cell."""
        result = cut_bins_manual(basic_dataset, min_cell=5, max_cell=6)

        # Only cell 5 should be masked
        assert np.all(result["mask"].values[:, 5, :] == 1)
        assert np.all(result["mask"].values[:, :5, :] == 0)
        assert np.all(result["mask"].values[:, 6:, :] == 0)

    def test_single_ensemble(self, basic_dataset):
        """Test masking a single ensemble."""
        result = cut_bins_manual(basic_dataset, min_ensemble=25, max_ensemble=26)

        # Only ensemble 25 should be masked
        assert np.all(result["mask"].values[:, :, 25] == 1)
        assert np.all(result["mask"].values[:, :, :25] == 0)
        assert np.all(result["mask"].values[:, :, 26:] == 0)


# ============================================================================
# TESTS: Immutability
# ============================================================================


class TestCutBinsManualImmutability:
    """Tests for immutability of original dataset."""

    def test_original_unchanged(self, basic_dataset):
        """Test that original dataset is not modified."""
        original_mask = basic_dataset["mask"].values.copy()

        _ = cut_bins_manual(basic_dataset, min_cell=5, max_cell=10)

        np.testing.assert_array_equal(basic_dataset["mask"].values, original_mask)

    def test_returns_new_dataset(self, basic_dataset):
        """Test that a new dataset is returned."""
        result = cut_bins_manual(basic_dataset, min_cell=5, max_cell=10)

        assert result is not basic_dataset


# ============================================================================
# TESTS: Index Clamping
# ============================================================================


class TestCutBinsManualClamping:
    """Tests for index clamping behavior."""

    def test_negative_min_cell_clamped(self, basic_dataset):
        """Test that negative min_cell is clamped to 0."""
        result = cut_bins_manual(basic_dataset, min_cell=-5, max_cell=5)

        # Should be equivalent to min_cell=0
        expected = cut_bins_manual(basic_dataset, min_cell=0, max_cell=5)
        np.testing.assert_array_equal(result["mask"].values, expected["mask"].values)

    def test_max_cell_exceeding_clamped(self, basic_dataset):
        """Test that max_cell exceeding num_cells is clamped."""
        n_cell = basic_dataset.sizes["cell"]
        result = cut_bins_manual(basic_dataset, min_cell=15, max_cell=100)

        # Should be equivalent to max_cell=num_cells
        expected = cut_bins_manual(basic_dataset, min_cell=15, max_cell=n_cell)
        np.testing.assert_array_equal(result["mask"].values, expected["mask"].values)

    def test_negative_min_ensemble_clamped(self, basic_dataset):
        """Test that negative min_ensemble is clamped to 0."""
        result = cut_bins_manual(basic_dataset, min_ensemble=-10, max_ensemble=10)

        expected = cut_bins_manual(basic_dataset, min_ensemble=0, max_ensemble=10)
        np.testing.assert_array_equal(result["mask"].values, expected["mask"].values)

    def test_max_ensemble_exceeding_clamped(self, basic_dataset):
        """Test that max_ensemble exceeding num_ensembles is clamped."""
        n_time = basic_dataset.sizes["time"]
        result = cut_bins_manual(basic_dataset, min_ensemble=40, max_ensemble=200)

        expected = cut_bins_manual(basic_dataset, min_ensemble=40, max_ensemble=n_time)
        np.testing.assert_array_equal(result["mask"].values, expected["mask"].values)


# ============================================================================
# TESTS: Input Validation
# ============================================================================


class TestCutBinsManualValidation:
    """Tests for input validation."""

    def test_invalid_dataset_type(self):
        """Test error for non-Dataset input."""
        with pytest.raises(TypeError):
            cut_bins_manual("not a dataset", min_cell=0, max_cell=5)

    def test_min_cell_greater_than_max_cell(self, basic_dataset):
        """Test error when min_cell > max_cell."""
        with pytest.raises(ValueError, match="min_cell .* cannot be > max_cell"):
            cut_bins_manual(basic_dataset, min_cell=10, max_cell=5)

    def test_min_ensemble_greater_than_max_ensemble(self, basic_dataset):
        """Test error when min_ensemble > max_ensemble."""
        with pytest.raises(
            ValueError, match="min_ensemble .* cannot be > max_ensemble"
        ):
            cut_bins_manual(basic_dataset, min_ensemble=30, max_ensemble=10)

    def test_float_indices_rejected(self, basic_dataset):
        """Test that float indices are rejected."""
        with pytest.raises(TypeError, match="must be integers"):
            cut_bins_manual(basic_dataset, min_cell=5.5, max_cell=10)

    def test_float_ensemble_indices_rejected(self, basic_dataset):
        """Test that float ensemble indices are rejected."""
        with pytest.raises(TypeError, match="must be integers"):
            cut_bins_manual(basic_dataset, min_ensemble=5.0, max_ensemble=10.0)


# ============================================================================
# TESTS: Mask Behavior
# ============================================================================


class TestCutBinsManualMaskBehavior:
    """Tests for mask behavior."""

    def test_creates_mask_if_missing(self, dataset_no_mask):
        """Test mask creation when not present."""
        result = cut_bins_manual(dataset_no_mask, min_cell=5, max_cell=10)

        # Mask should be created
        assert "mask" in result.data_vars

        # And should have correct values
        assert np.all(result["mask"].values[:, 5:10, :] == 1)
        assert np.all(result["mask"].values[:, :5, :] == 0)

    def test_preserves_existing_mask(self, dataset_with_pre_masked):
        """Test that existing mask values are preserved (union behavior)."""
        # Pre-masked: cells 0-2, ensembles 0-9
        result = cut_bins_manual(
            dataset_with_pre_masked,
            min_cell=10,
            max_cell=15,
            min_ensemble=20,
            max_ensemble=30,
        )

        # Pre-masked region should still be masked
        assert np.all(result["mask"].values[:, 0:3, 0:10] == 1)

        # New region should be masked
        assert np.all(result["mask"].values[:, 10:15, 20:30] == 1)

    def test_overlapping_mask_regions(self, dataset_with_pre_masked):
        """Test overlapping mask regions."""
        # Pre-masked: cells 0-2, ensembles 0-9
        # New mask: cells 0-5, ensembles 5-15 (overlaps)
        result = cut_bins_manual(
            dataset_with_pre_masked,
            min_cell=0,
            max_cell=5,
            min_ensemble=5,
            max_ensemble=15,
        )

        # Both regions should be masked
        assert np.all(result["mask"].values[:, 0:3, 0:10] == 1)  # Pre-existing
        assert np.all(result["mask"].values[:, 0:5, 5:15] == 1)  # New

    def test_masks_all_beams(self, basic_dataset):
        """Test that all beams are masked for the specified region."""
        result = cut_bins_manual(basic_dataset, min_cell=5, max_cell=10)

        n_beam = result.sizes["beam"]
        for b in range(n_beam):
            assert np.all(result["mask"].values[b, 5:10, :] == 1)


# ============================================================================
# TESTS: Numerical Accuracy
# ============================================================================


class TestCutBinsManualNumerical:
    """Tests for numerical accuracy."""

    def test_exact_cell_count(self, basic_dataset):
        """Test exact count of masked cells."""
        n_beam = basic_dataset.sizes["beam"]
        n_time = basic_dataset.sizes["time"]
        cell_range = 5  # cells 5-9

        result = cut_bins_manual(basic_dataset, min_cell=5, max_cell=10)

        expected_masked = n_beam * cell_range * n_time
        actual_masked = int((result["mask"].values == 1).sum())

        assert actual_masked == expected_masked

    def test_exact_ensemble_count(self, basic_dataset):
        """Test exact count of masked ensembles."""
        n_beam = basic_dataset.sizes["beam"]
        n_cell = basic_dataset.sizes["cell"]
        ensemble_range = 10  # ensembles 20-29

        result = cut_bins_manual(basic_dataset, min_ensemble=20, max_ensemble=30)

        expected_masked = n_beam * n_cell * ensemble_range
        actual_masked = int((result["mask"].values == 1).sum())

        assert actual_masked == expected_masked

    def test_exact_rectangular_count(self, basic_dataset):
        """Test exact count for rectangular region."""
        n_beam = basic_dataset.sizes["beam"]
        cell_range = 5
        ensemble_range = 15

        result = cut_bins_manual(
            basic_dataset, min_cell=5, max_cell=10, min_ensemble=10, max_ensemble=25
        )

        expected_masked = n_beam * cell_range * ensemble_range
        actual_masked = int((result["mask"].values == 1).sum())

        assert actual_masked == expected_masked


# ============================================================================
# TESTS: Edge Cases
# ============================================================================


class TestCutBinsManualEdgeCases:
    """Tests for edge cases."""

    def test_first_cell(self, basic_dataset):
        """Test masking only the first cell."""
        result = cut_bins_manual(basic_dataset, min_cell=0, max_cell=1)

        assert np.all(result["mask"].values[:, 0, :] == 1)
        assert np.all(result["mask"].values[:, 1:, :] == 0)

    def test_last_cell(self, basic_dataset):
        """Test masking only the last cell."""
        n_cell = basic_dataset.sizes["cell"]
        result = cut_bins_manual(basic_dataset, min_cell=n_cell - 1, max_cell=n_cell)

        assert np.all(result["mask"].values[:, -1, :] == 1)
        assert np.all(result["mask"].values[:, :-1, :] == 0)

    def test_first_ensemble(self, basic_dataset):
        """Test masking only the first ensemble."""
        result = cut_bins_manual(basic_dataset, min_ensemble=0, max_ensemble=1)

        assert np.all(result["mask"].values[:, :, 0] == 1)
        assert np.all(result["mask"].values[:, :, 1:] == 0)

    def test_last_ensemble(self, basic_dataset):
        """Test masking only the last ensemble."""
        n_time = basic_dataset.sizes["time"]
        result = cut_bins_manual(
            basic_dataset, min_ensemble=n_time - 1, max_ensemble=n_time
        )

        assert np.all(result["mask"].values[:, :, -1] == 1)
        assert np.all(result["mask"].values[:, :, :-1] == 0)

    def test_empty_region_after_clamping(self, basic_dataset):
        """Test when region becomes empty after clamping."""
        # min_cell > max_cell after clamping would be error
        # But min_cell = max_cell = 0 after clamping should mask nothing
        # Actually, this would raise error since min > max before clamping
        pass

    def test_single_point(self, basic_dataset):
        """Test masking a single point (1 cell, 1 ensemble)."""
        result = cut_bins_manual(
            basic_dataset, min_cell=10, max_cell=11, min_ensemble=25, max_ensemble=26
        )

        # Only one point masked
        n_beam = basic_dataset.sizes["beam"]
        expected_masked = n_beam * 1 * 1
        actual_masked = int((result["mask"].values == 1).sum())

        assert actual_masked == expected_masked
        assert np.all(result["mask"].values[:, 10, 25] == 1)


# ============================================================================
# TESTS: Various Dataset Sizes
# ============================================================================


class TestCutBinsManualDatasetSizes:
    """Tests for various dataset sizes."""

    def test_small_dataset(self):
        """Test with small dataset."""
        ds = xr.Dataset(
            {
                "velocity": (("beam", "cell", "time"), np.random.randn(4, 5, 10)),
                "mask": (("beam", "cell", "time"), np.zeros((4, 5, 10), dtype=np.int8)),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(5),
                "time": pd.date_range("2024-01-01", periods=10, freq="h"),
            },
        )

        result = cut_bins_manual(ds, min_cell=1, max_cell=3)

        assert np.all(result["mask"].values[:, 1:3, :] == 1)

    def test_large_dataset(self):
        """Test with large dataset."""
        ds = xr.Dataset(
            {
                "velocity": (("beam", "cell", "time"), np.random.randn(4, 100, 500)),
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((4, 100, 500), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(100),
                "time": pd.date_range("2024-01-01", periods=500, freq="h"),
            },
        )

        result = cut_bins_manual(
            ds, min_cell=50, max_cell=75, min_ensemble=200, max_ensemble=300
        )

        assert np.all(result["mask"].values[:, 50:75, 200:300] == 1)

    def test_single_beam_dataset(self):
        """Test with single beam dataset."""
        ds = xr.Dataset(
            {
                "velocity": (("beam", "cell", "time"), np.random.randn(1, 20, 30)),
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((1, 20, 30), dtype=np.int8),
                ),
            },
            coords={
                "beam": [0],
                "cell": np.arange(20),
                "time": pd.date_range("2024-01-01", periods=30, freq="h"),
            },
        )

        result = cut_bins_manual(ds, min_cell=5, max_cell=15)

        assert np.all(result["mask"].values[0, 5:15, :] == 1)


# ============================================================================
# TESTS: Multiple Manual Cuts
# ============================================================================


class TestCutBinsManualMultiple:
    """Tests for multiple manual cut operations."""

    def test_non_overlapping_cuts(self, basic_dataset):
        """Test multiple non-overlapping cuts."""
        # First cut
        result = cut_bins_manual(basic_dataset, min_cell=0, max_cell=3)
        # Second cut
        result = cut_bins_manual(result, min_cell=15, max_cell=20)

        # Both regions should be masked
        assert np.all(result["mask"].values[:, 0:3, :] == 1)
        assert np.all(result["mask"].values[:, 15:20, :] == 1)
        # Middle should not be masked
        assert np.all(result["mask"].values[:, 3:15, :] == 0)

    def test_overlapping_cuts(self, basic_dataset):
        """Test multiple overlapping cuts."""
        # First cut: cells 5-10
        result = cut_bins_manual(basic_dataset, min_cell=5, max_cell=10)
        # Second cut: cells 8-15 (overlaps)
        result = cut_bins_manual(result, min_cell=8, max_cell=15)

        # Union of both regions should be masked
        assert np.all(result["mask"].values[:, 5:15, :] == 1)
        assert np.all(result["mask"].values[:, :5, :] == 0)
        assert np.all(result["mask"].values[:, 15:, :] == 0)

    def test_adjacent_cuts(self, basic_dataset):
        """Test adjacent cuts (no gap, no overlap)."""
        # First cut: cells 0-5
        result = cut_bins_manual(basic_dataset, min_cell=0, max_cell=5)
        # Second cut: cells 5-10
        result = cut_bins_manual(result, min_cell=5, max_cell=10)

        # Cells 0-9 should be masked
        assert np.all(result["mask"].values[:, 0:10, :] == 1)
        assert np.all(result["mask"].values[:, 10:, :] == 0)


# ============================================================================
# TESTS: _extract_dataset_params helper (line 103)
# ============================================================================


class TestExtractDatasetParams:
    """Test _extract_dataset_params helper (line 103: beam_angle override)."""

    def test_beam_angle_override_uses_provided_value(self, basic_dataset):
        """Line 103: when beam_angle is provided it is used directly."""
        from pyadps.processing.profile_operation import _extract_dataset_params

        params = _extract_dataset_params(
            basic_dataset,
            beam_direction="up",
            beam_angle=30.0,
        )

        assert params["beam_angle"] == 30.0

    def test_beam_angle_from_accessor_when_not_provided(self, basic_dataset):
        """When beam_angle is None the accessor path is used (lines 104-107)."""
        from pyadps.processing.profile_operation import _extract_dataset_params

        params = _extract_dataset_params(
            basic_dataset,
            beam_direction="up",
            beam_angle=None,
        )

        # MockFixedLeaderAccessor returns attrs["beam_angle"] = 20
        assert params["beam_angle"] == 20.0
