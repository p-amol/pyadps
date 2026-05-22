"""
Test suite for trim_ensembles function.

Tests the ensemble trimming functionality which masks ensembles
at the beginning and/or end of deployment.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyadps.processing.profile_operation import trim_ensembles
from pyadps.processing.utility import create_default_mask


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def basic_dataset():
    """Create basic ADCP dataset for trim_ensembles tests."""
    np.random.seed(42)
    n_time = 100
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
    n_time = 50
    n_cell = 10
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
    n_cell = 10
    n_beam = 4

    times = pd.date_range("2024-01-01", periods=n_time, freq="h")

    # Pre-mask some cells
    mask = np.zeros((n_beam, n_cell, n_time), dtype=np.int8)
    mask[:, :2, :5] = 1  # Pre-mask first 2 cells, first 5 ensembles

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


class TestTrimEnsemblesBasic:
    """Tests for basic trim_ensembles functionality."""

    def test_trim_start_only(self, basic_dataset):
        """Test trimming only from start."""
        result = trim_ensembles(basic_dataset, start=10)

        # Check mask shape unchanged
        assert result["mask"].shape == basic_dataset["mask"].shape

        # First 10 ensembles should be masked
        assert np.all(result["mask"].values[:, :, :10] == 1)
        # Rest should be unchanged (0)
        assert np.all(result["mask"].values[:, :, 10:] == 0)

    def test_trim_end_only(self, basic_dataset):
        """Test trimming only from end."""
        result = trim_ensembles(basic_dataset, end=15)

        # Last 15 ensembles should be masked
        assert np.all(result["mask"].values[:, :, -15:] == 1)
        # Rest should be unchanged (0)
        assert np.all(result["mask"].values[:, :, :-15] == 0)

    def test_trim_both_ends(self, basic_dataset):
        """Test trimming from both ends."""
        result = trim_ensembles(basic_dataset, start=10, end=20)

        # First 10 ensembles should be masked
        assert np.all(result["mask"].values[:, :, :10] == 1)
        # Last 20 ensembles should be masked
        assert np.all(result["mask"].values[:, :, -20:] == 1)
        # Middle should be unchanged
        middle = result["mask"].values[:, :, 10:-20]
        assert np.all(middle == 0)

    def test_no_trim(self, basic_dataset):
        """Test with no trimming parameters."""
        result = trim_ensembles(basic_dataset)

        # Mask should be unchanged
        assert np.all(result["mask"].values == basic_dataset["mask"].values)

    def test_trim_zero_values(self, basic_dataset):
        """Test with zero trim values (should not modify)."""
        result = trim_ensembles(basic_dataset, start=0, end=0)

        # Mask should be unchanged
        assert np.all(result["mask"].values == basic_dataset["mask"].values)


# ============================================================================
# TESTS: Immutability
# ============================================================================


class TestTrimEnsemblesImmutability:
    """Tests for immutability of original dataset."""

    def test_original_unchanged(self, basic_dataset):
        """Test that original dataset is not modified."""
        original_mask = basic_dataset["mask"].values.copy()

        _ = trim_ensembles(basic_dataset, start=10, end=10)

        # Original should be unchanged
        assert np.array_equal(basic_dataset["mask"].values, original_mask)

    def test_returns_new_dataset(self, basic_dataset):
        """Test that a new dataset is returned."""
        result = trim_ensembles(basic_dataset, start=5)

        # Should be different objects
        assert result is not basic_dataset

        # But same structure
        assert result.sizes == basic_dataset.sizes
        assert set(result.data_vars) == set(basic_dataset.data_vars)


# ============================================================================
# TESTS: Edge Cases
# ============================================================================


class TestTrimEnsemblesEdgeCases:
    """Tests for edge cases in trim_ensembles."""

    def test_trim_all_ensembles(self, basic_dataset):
        """Test trimming when start + end equals total ensembles."""
        n_ens = basic_dataset.sizes["time"]
        result = trim_ensembles(basic_dataset, start=50, end=50)

        # All ensembles should be masked
        assert np.all(result["mask"].values == 1)

    def test_trim_exceeds_start(self, basic_dataset):
        """Test error when start exceeds total ensembles."""
        with pytest.raises(ValueError, match="start .* exceeds n_ensembles"):
            trim_ensembles(basic_dataset, start=200)

    def test_trim_exceeds_end(self, basic_dataset):
        """Test error when end exceeds total ensembles."""
        with pytest.raises(ValueError, match="end .* exceeds n_ensembles"):
            trim_ensembles(basic_dataset, end=200)

    def test_trim_overlapping(self, basic_dataset):
        """Test when start and end overlap (trim more than total)."""
        # Should work but result in fully masked data
        result = trim_ensembles(basic_dataset, start=60, end=60)

        # Overlapping regions get masked from both operations
        # All ensembles should be masked
        assert np.all(result["mask"].values == 1)

    def test_trim_single_ensemble_start(self, basic_dataset):
        """Test trimming single ensemble from start."""
        result = trim_ensembles(basic_dataset, start=1)

        # Only first ensemble masked
        assert np.all(result["mask"].values[:, :, 0] == 1)
        assert np.all(result["mask"].values[:, :, 1:] == 0)

    def test_trim_single_ensemble_end(self, basic_dataset):
        """Test trimming single ensemble from end."""
        result = trim_ensembles(basic_dataset, end=1)

        # Only last ensemble masked
        assert np.all(result["mask"].values[:, :, -1] == 1)
        assert np.all(result["mask"].values[:, :, :-1] == 0)


# ============================================================================
# TESTS: Mask Behavior
# ============================================================================


class TestTrimEnsemblesMaskBehavior:
    """Tests for mask behavior in trim_ensembles."""

    def test_creates_mask_if_missing(self, dataset_no_mask):
        """Test mask creation when not present."""
        result = trim_ensembles(dataset_no_mask, start=5)

        # Mask should be created
        assert "mask" in result.data_vars

        # And should have correct flagging
        assert np.all(result["mask"].values[:, :, :5] == 1)
        assert np.all(result["mask"].values[:, :, 5:] == 0)

    def test_preserves_existing_mask(self, dataset_with_pre_masked):
        """Test that existing mask values are preserved."""
        result = trim_ensembles(dataset_with_pre_masked, start=10)

        # Original pre-masked values should still be masked
        # (within the first 10 which are now all masked anyway)
        assert np.all(result["mask"].values[:, :, :10] == 1)

    def test_adds_to_existing_mask(self, dataset_with_pre_masked):
        """Test that trimming adds to existing mask."""
        # Pre-masked: first 2 cells, first 5 ensembles
        # Trim: last 10 ensembles
        result = trim_ensembles(dataset_with_pre_masked, end=10)

        # Pre-masked region should still be masked
        assert np.all(result["mask"].values[:, :2, :5] == 1)

        # Trimmed region should be masked
        assert np.all(result["mask"].values[:, :, -10:] == 1)

        # Non-overlapping middle region should be unchanged
        # cells 2+ for ensembles 5 to -10
        middle = result["mask"].values[:, 2:, 5:-10]
        assert np.all(middle == 0)

    def test_all_beams_masked(self, basic_dataset):
        """Test that all beams are masked for trimmed ensembles."""
        result = trim_ensembles(basic_dataset, start=5)

        n_beam = result.sizes["beam"]
        for b in range(n_beam):
            assert np.all(result["mask"].values[b, :, :5] == 1)

    def test_all_cells_masked(self, basic_dataset):
        """Test that all cells are masked for trimmed ensembles."""
        result = trim_ensembles(basic_dataset, end=3)

        n_cell = result.sizes["cell"]
        for c in range(n_cell):
            assert np.all(result["mask"].values[:, c, -3:] == 1)


# ============================================================================
# TESTS: Input Validation
# ============================================================================


class TestTrimEnsemblesValidation:
    """Tests for input validation in trim_ensembles."""

    def test_invalid_dataset_type(self):
        """Test error for non-Dataset input."""
        with pytest.raises(TypeError):
            trim_ensembles("not a dataset", start=5)

    def test_missing_velocity(self):
        """Test error when velocity is missing."""
        ds = xr.Dataset(
            {"other_var": (("cell", "time"), np.zeros((10, 20)))},
            coords={"cell": np.arange(10), "time": np.arange(20)},
        )
        with pytest.raises(ValueError, match="missing required variable"):
            trim_ensembles(ds, start=5)

    def test_missing_time_dimension(self):
        """Test error when time dimension is missing."""
        ds = xr.Dataset(
            {"velocity": (("beam", "cell"), np.zeros((4, 10)))},
            coords={"beam": np.arange(4), "cell": np.arange(10)},
        )
        with pytest.raises(ValueError, match="must have 'cell' and 'time' dimensions"):
            trim_ensembles(ds, start=5)


# ============================================================================
# TESTS: Numerical Accuracy
# ============================================================================


class TestTrimEnsemblesNumerical:
    """Tests for numerical accuracy in trim_ensembles."""

    def test_exact_count_start(self, basic_dataset):
        """Test exact count of masked cells from start."""
        n_beam = basic_dataset.sizes["beam"]
        n_cell = basic_dataset.sizes["cell"]
        trim_count = 10

        result = trim_ensembles(basic_dataset, start=trim_count)

        # Expected masked: n_beam * n_cell * trim_count
        expected_masked = n_beam * n_cell * trim_count
        actual_masked = int((result["mask"].values == 1).sum())

        assert actual_masked == expected_masked

    def test_exact_count_end(self, basic_dataset):
        """Test exact count of masked cells from end."""
        n_beam = basic_dataset.sizes["beam"]
        n_cell = basic_dataset.sizes["cell"]
        trim_count = 15

        result = trim_ensembles(basic_dataset, end=trim_count)

        # Expected masked: n_beam * n_cell * trim_count
        expected_masked = n_beam * n_cell * trim_count
        actual_masked = int((result["mask"].values == 1).sum())

        assert actual_masked == expected_masked

    def test_exact_count_both(self, basic_dataset):
        """Test exact count of masked cells from both ends."""
        n_beam = basic_dataset.sizes["beam"]
        n_cell = basic_dataset.sizes["cell"]
        start_trim = 10
        end_trim = 20

        result = trim_ensembles(basic_dataset, start=start_trim, end=end_trim)

        # Expected masked: n_beam * n_cell * (start_trim + end_trim)
        expected_masked = n_beam * n_cell * (start_trim + end_trim)
        actual_masked = int((result["mask"].values == 1).sum())

        assert actual_masked == expected_masked

    def test_percentage_masked(self, basic_dataset):
        """Test percentage of data masked."""
        n_beam = basic_dataset.sizes["beam"]
        n_cell = basic_dataset.sizes["cell"]
        n_time = basic_dataset.sizes["time"]
        total = n_beam * n_cell * n_time

        result = trim_ensembles(basic_dataset, start=10, end=10)

        masked = int((result["mask"].values == 1).sum())
        expected_pct = (10 + 10) / n_time * 100
        actual_pct = masked / total * 100

        assert np.isclose(actual_pct, expected_pct)


# ============================================================================
# TESTS: Various Dataset Sizes
# ============================================================================


class TestTrimEnsemblesDatasetSizes:
    """Tests for various dataset sizes."""

    @pytest.fixture
    def small_dataset(self):
        """Create small dataset."""
        return xr.Dataset(
            {
                "velocity": (
                    ("beam", "cell", "time"),
                    np.random.randn(4, 5, 10),
                ),
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((4, 5, 10), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(5),
                "time": pd.date_range("2024-01-01", periods=10, freq="h"),
            },
        )

    @pytest.fixture
    def large_dataset(self):
        """Create larger dataset."""
        return xr.Dataset(
            {
                "velocity": (
                    ("beam", "cell", "time"),
                    np.random.randn(4, 50, 1000),
                ),
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((4, 50, 1000), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(50),
                "time": pd.date_range("2024-01-01", periods=1000, freq="h"),
            },
        )

    def test_small_dataset(self, small_dataset):
        """Test with small dataset."""
        result = trim_ensembles(small_dataset, start=2, end=2)

        assert np.all(result["mask"].values[:, :, :2] == 1)
        assert np.all(result["mask"].values[:, :, -2:] == 1)
        assert np.all(result["mask"].values[:, :, 2:-2] == 0)

    def test_large_dataset(self, large_dataset):
        """Test with large dataset."""
        result = trim_ensembles(large_dataset, start=100, end=50)

        assert np.all(result["mask"].values[:, :, :100] == 1)
        assert np.all(result["mask"].values[:, :, -50:] == 1)

    def test_single_beam(self):
        """Test with single beam."""
        ds = xr.Dataset(
            {
                "velocity": (("beam", "cell", "time"), np.random.randn(1, 10, 20)),
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((1, 10, 20), dtype=np.int8),
                ),
            },
            coords={
                "beam": [0],
                "cell": np.arange(10),
                "time": pd.date_range("2024-01-01", periods=20, freq="h"),
            },
        )

        result = trim_ensembles(ds, start=5)

        assert np.all(result["mask"].values[0, :, :5] == 1)
        assert np.all(result["mask"].values[0, :, 5:] == 0)
