import pytest
import xarray as xr


@pytest.fixture(autouse=True)
def mock_fixed_leader_accessor(monkeypatch):
    """Register MockFixedLeaderAccessor for processing tests only.

    Temporarily replaces the real FixedLeaderAccessor with a mock that
    reads configuration from dataset attributes, then restores the real
    accessor after each test.
    """

    class MockFixedLeaderAccessor:
        def __init__(self, xarray_obj):
            self._obj = xarray_obj

        def system_configuration(self):
            return {
                "Beam Direction": self._obj.attrs.get("beam_direction", "up"),
                "Beam Angle": self._obj.attrs.get("beam_angle", 20),
            }

        def field(self, ens=0):
            return {
                "depth_cell_length": self._obj.attrs.get("cell_size_cm", 400),
                "bin_1_distance": self._obj.attrs.get("bin1_distance_cm", 200),
                "num_cells": self._obj.sizes.get("cell", 30),
                "num_ensembles": self._obj.sizes.get("time", 50),
            }

    # Swap in the mock
    try:
        del xr.Dataset.fixed_leader
    except AttributeError:
        pass
    xr.register_dataset_accessor("fixed_leader")(MockFixedLeaderAccessor)

    yield

    # Restore the real accessor
    try:
        del xr.Dataset.fixed_leader
    except AttributeError:
        pass
    from pyadps.io.accessors import FixedLeaderAccessor

    xr.register_dataset_accessor("fixed_leader")(FixedLeaderAccessor)
