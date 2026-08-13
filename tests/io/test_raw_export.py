"""
Tests for pyadps.io.raw_export - the reusable raw-dataset subsetting/attrs
helpers shared by the Download Raw File page and autoprocess().
"""

import numpy as np
import pytest
import xarray as xr

from pyadps.io.raw_export import INTERNAL_ATTRS, drop_internal_attrs, subset_raw_dataset


@pytest.fixture
def raw_ds() -> xr.Dataset:
    n_beams, n_cells, n_time = 4, 5, 10
    return xr.Dataset(
        {
            "fixed_leader_id": (["time"], np.zeros(n_time)),
            "cpu_version": (["time"], np.zeros(n_time)),
            "ensemble_number": (["time"], np.arange(n_time)),
            "velocity": (["beam", "cell", "time"], np.zeros((n_beams, n_cells, n_time))),
            "echo_intensity": (
                ["beam", "cell", "time"],
                np.zeros((n_beams, n_cells, n_time)),
            ),
            "correlation": (
                ["beam", "cell", "time"],
                np.zeros((n_beams, n_cells, n_time)),
            ),
            "percent_good": (
                ["beam", "cell", "time"],
                np.zeros((n_beams, n_cells, n_time)),
            ),
        },
        coords={
            "beam": np.arange(n_beams),
            "cell": np.arange(n_cells),
            "time": np.arange(n_time),
        },
        attrs={
            "fixed_leader_variables": ["fixed_leader_id", "cpu_version"],
            "variable_leader_variables": ["ensemble_number"],
            "pyadps_component": "Complete",
            "components": "{'header': True}",
            "some_other_attr": "kept",
        },
    )


class TestSubsetRawDataset:
    def test_raises_when_nothing_selected(self, raw_ds):
        with pytest.raises(ValueError, match="No variables selected"):
            subset_raw_dataset(raw_ds)

    def test_velocity_only(self, raw_ds):
        result = subset_raw_dataset(raw_ds, include_velocity=True)
        assert set(result.data_vars) == {"velocity"}

    def test_entire_dataset_equivalent(self, raw_ds):
        """Checking all six matches the page's 'Entire Data Set' option."""
        result = subset_raw_dataset(
            raw_ds,
            include_fixed_leader=True,
            include_variable_leader=True,
            include_velocity=True,
            include_echo=True,
            include_correlation=True,
            include_percent_good=True,
        )
        assert set(result.data_vars) == {
            "fixed_leader_id",
            "cpu_version",
            "ensemble_number",
            "velocity",
            "echo_intensity",
            "correlation",
            "percent_good",
        }

    def test_arbitrary_subset(self, raw_ds):
        result = subset_raw_dataset(
            raw_ds, include_echo=True, include_correlation=True
        )
        assert set(result.data_vars) == {"echo_intensity", "correlation"}

    def test_fixed_leader_uses_ds_attrs_field_list(self, raw_ds):
        result = subset_raw_dataset(raw_ds, include_fixed_leader=True)
        assert set(result.data_vars) == {"fixed_leader_id", "cpu_version"}

    def test_missing_component_silently_skipped(self, raw_ds):
        """velocity requested but dataset lacks it -> not an error by itself."""
        ds = raw_ds.drop_vars("velocity")
        result = subset_raw_dataset(ds, include_velocity=True, include_echo=True)
        assert set(result.data_vars) == {"echo_intensity"}

    def test_coordinates_preserved(self, raw_ds):
        result = subset_raw_dataset(raw_ds, include_velocity=True)
        assert set(result.coords) == {"beam", "cell", "time"}

    def test_global_attrs_copied(self, raw_ds):
        result = subset_raw_dataset(raw_ds, include_velocity=True)
        assert result.attrs["some_other_attr"] == "kept"

    def test_original_dataset_not_mutated(self, raw_ds):
        original_vars = set(raw_ds.data_vars)
        subset_raw_dataset(raw_ds, include_velocity=True)
        assert set(raw_ds.data_vars) == original_vars


class TestDropInternalAttrs:
    def test_drops_all_internal_attrs(self, raw_ds):
        result = drop_internal_attrs(raw_ds)
        for attr in INTERNAL_ATTRS:
            assert attr not in result.attrs

    def test_keeps_other_attrs(self, raw_ds):
        result = drop_internal_attrs(raw_ds)
        assert result.attrs["some_other_attr"] == "kept"

    def test_original_dataset_not_mutated(self, raw_ds):
        drop_internal_attrs(raw_ds)
        assert "pyadps_component" in raw_ds.attrs

    def test_missing_attrs_do_not_raise(self):
        ds = xr.Dataset({"x": (["a"], [1])}, attrs={})
        result = drop_internal_attrs(ds)
        assert result.attrs == {}
