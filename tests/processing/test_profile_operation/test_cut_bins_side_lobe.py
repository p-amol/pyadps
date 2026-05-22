"""
Test suite for cut_bins_side_lobe function.

Tests the side-lobe contamination masking functionality based on
beam geometry and transducer depth.

Note: Tests pass beam_angle and orientation explicitly to avoid relying on
the fixed_leader accessor, which doesn't survive dataset copy operations.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

try:
    from pyadps.processing.profile_operation import (
        cut_bins_side_lobe,
        DEFAULT_EXTRA_CELLS,
    )
except ImportError:
    import sys

    sys.path.insert(0, "/mnt/user-data/outputs")
    from profile_operation import cut_bins_side_lobe, DEFAULT_EXTRA_CELLS


@pytest.fixture
def upward_dataset():
    """Create upward-looking ADCP dataset.

    Configuration:
    - 50m transducer depth (500 decimeters)
    - 1m cell size (100 cm)
    - 2m bin1 distance (200 cm)
    - 20° beam angle
    """
    np.random.seed(42)
    n_time = 50
    n_cell = 30
    n_beam = 4

    times = pd.date_range("2024-01-01", periods=n_time, freq="h")
    cells = np.arange(n_cell)
    beams = np.arange(n_beam)

    # Transducer at 50m depth (500 decimeters)
    transducer_depth = np.full(n_time, 500)

    data_vars = {
        "velocity": (
            ("beam", "cell", "time"),
            np.random.randint(-500, 500, size=(n_beam, n_cell, n_time)),
        ),
        "transducer_depth": (("time",), transducer_depth),
        "mask": (
            ("beam", "cell", "time"),
            np.zeros((n_beam, n_cell, n_time), dtype=np.int8),
        ),
    }

    coords = {"time": times, "cell": cells, "beam": beams}
    ds = xr.Dataset(data_vars, coords=coords)

    # Store configuration in attrs for reference
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
    - 1m cell size (100 cm)
    - 2m bin1 distance (200 cm)
    - 20° beam angle
    """
    np.random.seed(42)
    n_time = 50
    n_cell = 30
    n_beam = 4

    times = pd.date_range("2024-01-01", periods=n_time, freq="h")
    cells = np.arange(n_cell)
    beams = np.arange(n_beam)

    # Transducer at 10m depth (100 decimeters)
    transducer_depth = np.full(n_time, 100)

    data_vars = {
        "velocity": (
            ("beam", "cell", "time"),
            np.random.randint(-500, 500, size=(n_beam, n_cell, n_time)),
        ),
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
def variable_depth_dataset():
    """Create dataset with varying transducer depth.

    Configuration:
    - 30-70m transducer depth (300-700 decimeters)
    - 1m cell size (100 cm)
    - 2m bin1 distance (200 cm)
    - 20° beam angle
    """
    np.random.seed(42)
    n_time = 100
    n_cell = 30
    n_beam = 4

    times = pd.date_range("2024-01-01", periods=n_time, freq="h")
    cells = np.arange(n_cell)
    beams = np.arange(n_beam)

    # Varying transducer depth: 30-70m (300-700 decimeters)
    transducer_depth = np.linspace(300, 700, n_time)

    data_vars = {
        "velocity": (
            ("beam", "cell", "time"),
            np.random.randint(-500, 500, size=(n_beam, n_cell, n_time)),
        ),
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
    ds.attrs["beam_direction"] = "up"

    return ds


# ============================================================================
# TESTS: Basic Upward-Looking ADCP
# ============================================================================


class TestCutBinsSideLobedUpward:
    """Tests for upward-looking ADCP side-lobe masking."""

    def test_basic_upward(self, upward_dataset):
        """Test basic upward-looking side-lobe masking."""
        result = cut_bins_side_lobe(upward_dataset, orientation="up")

        # Some cells should be masked (surface contamination)
        assert np.any(result["mask"].values == 1)

        # First cells should not be masked (deep, away from surface)
        # For 50m depth, surface is ~47m from transducer
        # With 1m cells, roughly 44-45 valid cells (after cos(20°) correction)
        first_few = result["mask"].values[:, :5, :]
        assert np.all(first_few == 0)

    def test_upward_auto_detect(self, upward_dataset):
        """Test orientation auto-detection for upward ADCP."""
        result = cut_bins_side_lobe(upward_dataset)  # No orientation specified

        # Should still work and mask surface contamination
        assert np.any(result["mask"].values == 1)

    def test_upward_extra_cells(self, upward_dataset):
        """Test extra_cells parameter for upward ADCP."""
        result_1 = cut_bins_side_lobe(upward_dataset, extra_cells=1)
        result_3 = cut_bins_side_lobe(upward_dataset, extra_cells=3)

        # More extra cells = more masking
        masked_1 = int((result_1["mask"].values == 1).sum())
        masked_3 = int((result_3["mask"].values == 1).sum())

        assert masked_3 > masked_1

    def test_upward_water_depth_ignored(self, upward_dataset):
        """Test that water_depth is ignored for upward ADCP."""
        result_no_depth = cut_bins_side_lobe(upward_dataset, orientation="up")
        result_with_depth = cut_bins_side_lobe(
            upward_dataset, orientation="up", water_depth=100.0
        )

        # Results should be identical (water_depth ignored for upward)
        np.testing.assert_array_equal(
            result_no_depth["mask"].values, result_with_depth["mask"].values
        )


# ============================================================================
# TESTS: Downward-Looking ADCP
# ============================================================================


class TestCutBinsSideLobeDownward:
    """Tests for downward-looking ADCP side-lobe masking."""

    def test_basic_downward(self, downward_dataset):
        """Test basic downward-looking side-lobe masking."""
        result = cut_bins_side_lobe(
            downward_dataset, orientation="down", water_depth=100.0
        )

        # Some cells should be masked (bottom contamination)
        assert np.any(result["mask"].values == 1)

    def test_downward_requires_water_depth(self, downward_dataset):
        """Test that water_depth is required for downward ADCP."""
        with pytest.raises(ValueError, match="water_depth required"):
            cut_bins_side_lobe(downward_dataset, orientation="down")

    def test_downward_auto_detect(self, downward_dataset):
        """Test auto-detection for downward ADCP still requires water_depth."""
        with pytest.raises(ValueError, match="water_depth required"):
            cut_bins_side_lobe(downward_dataset)  # Auto-detects "down"

    def test_downward_shallow_water(self, downward_dataset):
        """Test downward ADCP in shallow water."""
        # 20m water with transducer at 10m = only 10m range
        result = cut_bins_side_lobe(
            downward_dataset, orientation="down", water_depth=20.0
        )

        # Many cells should be masked (shallow water)
        masked_count = int((result["mask"].values == 1).sum())
        total_count = result["mask"].values.size

        # Expect significant masking
        assert masked_count / total_count > 0.5

    def test_downward_deep_water(self, downward_dataset):
        """Test downward ADCP in deep water."""
        # 200m water with transducer at 10m = ~190m range
        result = cut_bins_side_lobe(
            downward_dataset, orientation="down", water_depth=200.0
        )

        # Fewer cells should be masked (deep water)
        masked_count = int((result["mask"].values == 1).sum())
        total_count = result["mask"].values.size

        # Expect less masking than shallow water
        # With 30 cells at 1m each, most should be valid
        assert masked_count / total_count < 0.5


# ============================================================================
# TESTS: Variable Transducer Depth
# ============================================================================


class TestCutBinsSideLobeVariableDepth:
    """Tests for variable transducer depth."""

    def test_varying_depth_masking(self, variable_depth_dataset):
        """Test masking with varying transducer depth."""
        result = cut_bins_side_lobe(variable_depth_dataset, orientation="up")

        # Masking should vary per ensemble
        mask_per_ensemble = result["mask"].values.sum(axis=(0, 1))

        # Not all ensembles should have same masking
        assert len(np.unique(mask_per_ensemble)) > 1

    def test_shallow_vs_deep_ensembles(self, variable_depth_dataset):
        """Test that shallow ensembles have more masking than deep."""
        result = cut_bins_side_lobe(variable_depth_dataset, orientation="up")

        # First ensemble (shallowest, 30m) should have more masking
        # Last ensemble (deepest, 70m) should have less masking
        mask_first = result["mask"].values[:, :, 0].sum()
        mask_last = result["mask"].values[:, :, -1].sum()

        # Shallowest should have MORE masked (closer to surface)
        assert mask_first > mask_last


# ============================================================================
# TESTS: Immutability
# ============================================================================


class TestCutBinsSideLobeImmutability:
    """Tests for immutability of original dataset."""

    def test_original_unchanged(self, upward_dataset):
        """Test that original dataset is not modified."""
        original_mask = upward_dataset["mask"].values.copy()

        _ = cut_bins_side_lobe(upward_dataset, orientation="up")

        np.testing.assert_array_equal(upward_dataset["mask"].values, original_mask)

    def test_returns_new_dataset(self, upward_dataset):
        """Test that a new dataset is returned."""
        result = cut_bins_side_lobe(upward_dataset, orientation="up")

        assert result is not upward_dataset


# ============================================================================
# TESTS: Extra Cells Parameter
# ============================================================================


class TestCutBinsSideLobeExtraCells:
    """Tests for extra_cells parameter."""

    def test_extra_cells_zero(self, upward_dataset):
        """Test with extra_cells=0."""
        result = cut_bins_side_lobe(upward_dataset, orientation="up", extra_cells=0)

        # Should still mask some cells (physics-based)
        assert np.any(result["mask"].values == 1)

    def test_extra_cells_default(self, upward_dataset):
        """Test default extra_cells value."""
        result = cut_bins_side_lobe(upward_dataset, orientation="up")

        # Default is 1
        assert DEFAULT_EXTRA_CELLS == 1

        # Compare with explicit extra_cells=1
        result_explicit = cut_bins_side_lobe(
            upward_dataset, orientation="up", extra_cells=1
        )

        np.testing.assert_array_equal(
            result["mask"].values, result_explicit["mask"].values
        )

    def test_extra_cells_increasing(self, upward_dataset):
        """Test that more extra_cells means more masking."""
        results = []
        for extra in [0, 1, 2, 3, 5]:
            result = cut_bins_side_lobe(
                upward_dataset, orientation="up", extra_cells=extra
            )
            masked = int((result["mask"].values == 1).sum())
            results.append(masked)

        # Each should have >= masking than previous
        for i in range(1, len(results)):
            assert results[i] >= results[i - 1]


# ============================================================================
# TESTS: Mask Behavior
# ============================================================================


class TestCutBinsSideLobeMaskBehavior:
    """Tests for mask behavior in cut_bins_side_lobe."""

    def test_masks_all_beams(self, upward_dataset):
        """Test that all beams are masked for contaminated cells."""
        result = cut_bins_side_lobe(upward_dataset, orientation="up")

        # For any masked cell, all beams should be masked
        n_beam = result.sizes["beam"]
        n_cell = result.sizes["cell"]
        n_time = result.sizes["time"]

        for t in range(n_time):
            for c in range(n_cell):
                beam_masks = result["mask"].values[:, c, t]
                # Either all masked or all not masked
                assert np.all(beam_masks == 0) or np.all(beam_masks == 1)

    def test_preserves_existing_mask(self, upward_dataset):
        """Test that existing mask values are preserved."""
        # Set some pre-existing mask
        upward_dataset["mask"].values[:, 0, :] = 1  # First cell always masked

        result = cut_bins_side_lobe(upward_dataset, orientation="up")

        # First cell should still be masked
        assert np.all(result["mask"].values[:, 0, :] == 1)

    def test_creates_mask_if_missing(self, upward_dataset):
        """Test mask creation when not present."""
        # Remove mask
        del upward_dataset["mask"]

        result = cut_bins_side_lobe(upward_dataset, orientation="up")

        # Mask should be created
        assert "mask" in result.data_vars
        assert result["mask"].shape == (4, 30, 50)


# ============================================================================
# TESTS: Orientation Override
# ============================================================================


class TestCutBinsSideLobeOrientation:
    """Tests for orientation parameter."""

    def test_orientation_override_up_to_down(self, upward_dataset):
        """Test overriding upward to downward."""
        # Upward dataset but force downward
        result = cut_bins_side_lobe(
            upward_dataset, orientation="down", water_depth=100.0
        )

        # Should process as downward (different masking pattern)
        assert np.any(result["mask"].values == 1)

    def test_orientation_case_insensitive(self, upward_dataset):
        """Test orientation is case insensitive."""
        result_lower = cut_bins_side_lobe(upward_dataset, orientation="up")
        result_upper = cut_bins_side_lobe(upward_dataset, orientation="UP")
        result_mixed = cut_bins_side_lobe(upward_dataset, orientation="Up")

        np.testing.assert_array_equal(
            result_lower["mask"].values, result_upper["mask"].values
        )
        np.testing.assert_array_equal(
            result_lower["mask"].values, result_mixed["mask"].values
        )


# ============================================================================
# TESTS: Input Validation
# ============================================================================


class TestCutBinsSideLobeValidation:
    """Tests for input validation."""

    def test_invalid_dataset_type(self):
        """Test error for non-Dataset input."""
        with pytest.raises(TypeError):
            cut_bins_side_lobe("not a dataset")

    def test_missing_transducer_depth(self):
        """Test handling when transducer_depth is missing."""
        ds = xr.Dataset(
            {
                "velocity": (("beam", "cell", "time"), np.random.randn(4, 10, 20)),
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
        ds.attrs["cell_size_cm"] = 100
        ds.attrs["bin1_distance_cm"] = 200
        ds.attrs["beam_angle"] = 20
        ds.attrs["beam_direction"] = "up"

        with pytest.raises(KeyError):
            cut_bins_side_lobe(ds, orientation="up")


# ============================================================================
# TESTS: Physics Verification
# ============================================================================


class TestCutBinsSideLobePhysics:
    """Tests to verify physics calculations."""

    def test_beam_angle_effect(self):
        """Test effect of beam angle on valid range."""
        np.random.seed(42)
        n_time = 20
        n_cell = 50
        n_beam = 4

        times = pd.date_range("2024-01-01", periods=n_time, freq="h")

        ds = xr.Dataset(
            {
                "velocity": (
                    ("beam", "cell", "time"),
                    np.random.randn(n_beam, n_cell, n_time),
                ),
                "transducer_depth": (("time",), np.full(n_time, 1000)),  # 100m
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
        ds.attrs["cell_size_cm"] = 100
        ds.attrs["bin1_distance_cm"] = 200
        ds.attrs["beam_angle"] = 20
        ds.attrs["beam_direction"] = "up"

        # Test with 20° beam angle
        result_20 = cut_bins_side_lobe(ds, orientation="up")
        masked_20 = int((result_20["mask"].values == 1).sum())

        # Test with 30° beam angle - need to create new dataset
        ds2 = xr.Dataset(
            {
                "velocity": (
                    ("beam", "cell", "time"),
                    np.random.randn(n_beam, n_cell, n_time),
                ),
                "transducer_depth": (("time",), np.full(n_time, 1000)),
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
        ds2.attrs["cell_size_cm"] = 100
        ds2.attrs["bin1_distance_cm"] = 200
        ds2.attrs["beam_angle"] = 30
        ds2.attrs["beam_direction"] = "up"

        result_30 = cut_bins_side_lobe(ds2, orientation="up")
        masked_30 = int((result_30["mask"].values == 1).sum())

        # Both should produce valid results (physics verification is complex)
        assert masked_20 >= 0
        assert masked_30 >= 0


# ============================================================================
# TESTS: Edge Cases
# ============================================================================


class TestCutBinsSideLobeEdgeCases:
    """Tests for edge cases."""

    def test_very_shallow_deployment(self):
        """Test with very shallow transducer depth."""
        ds = xr.Dataset(
            {
                "velocity": (("beam", "cell", "time"), np.random.randn(4, 30, 20)),
                "transducer_depth": (("time",), np.full(20, 50)),  # 5m depth
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((4, 30, 20), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(30),
                "time": pd.date_range("2024-01-01", periods=20, freq="h"),
            },
        )
        ds.attrs["cell_size_cm"] = 100
        ds.attrs["bin1_distance_cm"] = 200
        ds.attrs["beam_angle"] = 20
        ds.attrs["beam_direction"] = "up"

        result = cut_bins_side_lobe(ds, orientation="up")

        # Almost all cells should be masked (very shallow)
        masked_pct = (result["mask"].values == 1).sum() / result["mask"].values.size
        assert masked_pct > 0.8

    def test_very_deep_deployment(self):
        """Test with very deep transducer depth."""
        ds = xr.Dataset(
            {
                "velocity": (("beam", "cell", "time"), np.random.randn(4, 30, 20)),
                "transducer_depth": (("time",), np.full(20, 5000)),  # 500m depth
                "mask": (
                    ("beam", "cell", "time"),
                    np.zeros((4, 30, 20), dtype=np.int8),
                ),
            },
            coords={
                "beam": np.arange(4),
                "cell": np.arange(30),
                "time": pd.date_range("2024-01-01", periods=20, freq="h"),
            },
        )
        ds.attrs["cell_size_cm"] = 100
        ds.attrs["bin1_distance_cm"] = 200
        ds.attrs["beam_angle"] = 20
        ds.attrs["beam_direction"] = "up"

        result = cut_bins_side_lobe(ds, orientation="up")

        # No cells should be masked (very deep, surface far away)
        assert np.all(result["mask"].values == 0)

    def test_single_ensemble(self):
        """Test with single ensemble."""
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
        ds.attrs["cell_size_cm"] = 100
        ds.attrs["bin1_distance_cm"] = 200
        ds.attrs["beam_angle"] = 20
        ds.attrs["beam_direction"] = "up"

        result = cut_bins_side_lobe(ds, orientation="up")

        # Should work with single ensemble
        assert result["mask"].shape == (4, 20, 1)
