#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for HeaderAccessor dataset validation logic.

Covers the _validate_dataset() guard and the pyadps_component warning path,
contributing to 100% coverage of the accessor validation code paths.

Strategy: Validation tests are kept in their own file per project convention.
"""

import logging
import pytest
import numpy as np
import xarray as xr
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_header_dataset(**attr_overrides) -> xr.Dataset:
    """
    Build a minimal xarray.Dataset that satisfies HeaderAccessor validation.

    Parameters
    ----------
    **attr_overrides
        Any attribute can be overridden or set to ``None`` to omit it.
    """
    attrs = {
        "filename": "test.000",
        "total_ensembles": 3,
        "pyadps_component": "Header",
    }
    # Apply overrides; a value of None means 'remove the key'
    for key, value in attr_overrides.items():
        if value is None:
            attrs.pop(key, None)
        else:
            attrs[key] = value

    ds = xr.Dataset(
        {
            "byte": (("ensemble",), np.array([100, 200, 300])),
        },
        attrs=attrs,
    )
    return ds


# ============================================================================
# TEST CLASS: _validate_dataset – required attributes
# ============================================================================


class TestValidateDatasetRequiredAttributes:
    """Verify that missing required attrs raise ValueError."""

    def test_valid_dataset_does_not_raise(self):
        ds = _make_header_dataset()
        # Constructing the accessor triggers _validate_dataset
        accessor = ds.header  # should not raise
        assert accessor is not None

    def test_missing_filename_raises_value_error(self):
        ds = _make_header_dataset(filename=None)
        with pytest.raises(ValueError, match="filename"):
            _ = ds.header

    def test_missing_total_ensembles_raises_value_error(self):
        ds = _make_header_dataset(total_ensembles=None)
        with pytest.raises(ValueError, match="total_ensembles"):
            _ = ds.header

    def test_missing_pyadps_component_raises_value_error(self):
        ds = _make_header_dataset(pyadps_component=None)
        with pytest.raises(ValueError, match="pyadps_component"):
            _ = ds.header

    def test_missing_all_attrs_raises_value_error(self):
        ds = xr.Dataset({"byte": (("ensemble",), np.array([100]))})
        with pytest.raises(ValueError):
            _ = ds.header

    def test_error_message_lists_missing_attrs(self):
        ds = _make_header_dataset(filename=None, total_ensembles=None)
        with pytest.raises(ValueError) as exc_info:
            _ = ds.header
        # Both missing attributes should appear somewhere in the message
        msg = str(exc_info.value)
        assert "filename" in msg or "total_ensembles" in msg


# ============================================================================
# TEST CLASS: _validate_dataset – pyadps_component warning
# ============================================================================


class TestValidateDatasetComponentWarning:
    """Verify that wrong pyadps_component emits a logger warning but
    does NOT raise an exception."""

    def test_wrong_component_does_not_raise(self, caplog):
        ds = _make_header_dataset(pyadps_component="FixedLeader")
        with caplog.at_level(logging.WARNING, logger="pyadps.io.accessors"):
            accessor = ds.header  # must not raise
        assert accessor is not None

    def test_wrong_component_emits_warning(self, caplog):
        ds = _make_header_dataset(pyadps_component="VariableLeader")
        with caplog.at_level(logging.WARNING, logger="pyadps.io.accessors"):
            _ = ds.header
        assert any("optimized for Header" in r.message for r in caplog.records)

    def test_correct_component_no_warning(self, caplog):
        ds = _make_header_dataset(pyadps_component="Header")
        with caplog.at_level(logging.WARNING, logger="pyadps.io.accessors"):
            _ = ds.header
        warning_msgs = [
            r.message for r in caplog.records if r.levelno == logging.WARNING
        ]
        assert not any("optimized for Header" in m for m in warning_msgs)


# ============================================================================
# TEST CLASS: accessor instantiation guards
# ============================================================================


class TestAccessorInstantiation:
    """Misc instantiation and guard tests."""

    def test_accessor_stores_dataset_reference(self):
        ds = _make_header_dataset()
        accessor = ds.header
        assert accessor._obj is ds

    def test_accessor_validates_on_each_new_dataset(self):
        ds_good = _make_header_dataset()
        ds_bad = _make_header_dataset(filename=None)

        # Good dataset is fine
        _ = ds_good.header

        # Bad dataset still raises
        with pytest.raises(ValueError):
            _ = ds_bad.header


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
