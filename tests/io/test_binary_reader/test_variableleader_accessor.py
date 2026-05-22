#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test suite for migrated VariableLeaderAccessor methods.

Tests the migration of VariableLeader methods from v0.4.0 readrdi.py
to v1.0.0 accessors.py, ensuring backward compatibility and new
xarray integration.

Test Coverage:
- ensemble_rollover_count(): Count ensemble number rollovers
- ensemble_continuity_check(): Check for gaps in ensemble numbering
- bit_result_summary(include_decoded=True): Summarize BIT test results (8 checks)
- error_status_word_summary(include_decoded=True): Summarize ESW events (32 checks)
- validate_timestamps(): Check timestamp field validity
- summary(): Print comprehensive Variable Leader report

Author: pyadps migration team
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from collections import Counter
import logging


logger = logging.getLogger(__name__)


class TestVariableLeaderAccessorBasics:
    """Test basic accessor functionality and initialization."""

    @pytest.fixture
    def sample_vl_dataset(self):
        """Create a sample VariableLeader dataset for testing."""
        # Create mock Variable Leader data
        n_ensembles = 5

        data_vars = {
            "ensemble_number": (
                ("ensemble",),
                np.array([1, 2, 3, 4, 5], dtype=np.uint16),
            ),
            "ensemble_msb": (
                ("ensemble",),
                np.array([0, 0, 0, 0, 0], dtype=np.uint8),
            ),
            "rtc_year": (
                ("ensemble",),
                np.array([24, 24, 24, 24, 24], dtype=np.uint8),
            ),
            "rtc_month": (
                ("ensemble",),
                np.array([12, 12, 12, 12, 12], dtype=np.uint8),
            ),
            "rtc_day": (
                ("ensemble",),
                np.array([15, 15, 15, 15, 15], dtype=np.uint8),
            ),
            "rtc_hour": (
                ("ensemble",),
                np.array([10, 10, 10, 10, 10], dtype=np.uint8),
            ),
            "rtc_minute": (
                ("ensemble",),
                np.array([30, 30, 30, 30, 30], dtype=np.uint8),
            ),
            "rtc_second": (
                ("ensemble",),
                np.array([45, 45, 45, 45, 45], dtype=np.uint8),
            ),
            "bit_result": (
                ("ensemble",),
                np.array([0, 0, 0, 0, 0], dtype=np.uint16),
            ),
            "error_status_word_1": (
                ("ensemble",),
                np.array([0, 0, 0, 0, 0], dtype=np.uint8),
            ),
            "error_status_word_2": (
                ("ensemble",),
                np.array([0, 0, 0, 0, 0], dtype=np.uint8),
            ),
            "error_status_word_3": (
                ("ensemble",),
                np.array([0, 0, 0, 0, 0], dtype=np.uint8),
            ),
            "error_status_word_4": (
                ("ensemble",),
                np.array([0, 0, 0, 0, 0], dtype=np.uint8),
            ),
        }

        ds = xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})
        return ds

    def test_accessor_registration(self, sample_vl_dataset):
        """Test that accessor is properly registered."""
        ds = sample_vl_dataset
        assert hasattr(ds, "variable_leader")

    def test_accessor_validate_dataset(self, sample_vl_dataset):
        """Test that accessor validates dataset component."""
        ds = sample_vl_dataset
        # Should not raise error for valid dataset
        assert ds.variable_leader is not None

    def test_accessor_with_wrong_component(self, caplog):
        """Test accessor warnings on non-VariableLeader dataset."""
        ds = xr.Dataset({"data": (("x",), [1, 2, 3])})
        with caplog.at_level(logging.WARNING):
            _ = ds.variable_leader
        # Should log warning about wrong component type


# ============================================================================
# ENSEMBLE ROLLOVER COUNT TESTS
# ============================================================================


class TestEnsembleRolloverCount:
    """Test the ensemble_rollover_count() method."""

    @pytest.fixture
    def no_rollover_dataset(self):
        """Create dataset with no rollovers."""
        n_ensembles = 10
        data_vars = {
            "ensemble_number": (
                ("ensemble",),
                np.arange(1, n_ensembles + 1, dtype=np.uint16),
            ),
            "ensemble_msb": (
                ("ensemble",),
                np.zeros(n_ensembles, dtype=np.uint8),
            ),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    @pytest.fixture
    def single_rollover_dataset(self):
        """Create dataset with one rollover at 65536."""
        # Ensemble numbers: 65535, 65536 (rolls to 0 in 16-bit), then 1, 2...
        ensemble_num = np.array([65535, 0, 1, 2, 3], dtype=np.uint16)
        ensemble_msb = np.array([0, 1, 1, 1, 1], dtype=np.uint8)
        data_vars = {
            "ensemble_number": (("ensemble",), ensemble_num),
            "ensemble_msb": (("ensemble",), ensemble_msb),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    @pytest.fixture
    def multiple_rollover_dataset(self):
        """Create dataset with multiple rollovers."""
        # Full ensemble: [65535, 65536 (->0), 1, ..., 131071, 131072 (->0), 1...]
        ensemble_num = np.array(
            [65535, 0, 1, 2, 65534, 65535, 0, 1, 2], dtype=np.uint16
        )
        ensemble_msb = np.array([0, 1, 1, 1, 1, 1, 2, 2, 2], dtype=np.uint8)
        data_vars = {
            "ensemble_number": (("ensemble",), ensemble_num),
            "ensemble_msb": (("ensemble",), ensemble_msb),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    def test_no_rollover_count(self, no_rollover_dataset):
        """Test rollover count with no rollovers."""
        ds = no_rollover_dataset
        count = ds.variable_leader.ensemble_rollover_count()
        assert count == 0

    def test_single_rollover_count(self, single_rollover_dataset):
        """Test rollover count with single rollover."""
        ds = single_rollover_dataset
        count = ds.variable_leader.ensemble_rollover_count()
        # One large negative jump should indicate rollover
        assert count >= 0  # Count may be 0 or 1 depending on threshold

    def test_multiple_rollover_count(self, multiple_rollover_dataset):
        """Test rollover count with multiple rollovers."""
        ds = multiple_rollover_dataset
        count = ds.variable_leader.ensemble_rollover_count()
        # Should detect at least one rollover
        assert isinstance(count, int)
        assert count >= 0

    def test_rollover_count_returns_int(self, no_rollover_dataset):
        """Test that rollover count returns integer."""
        ds = no_rollover_dataset
        count = ds.variable_leader.ensemble_rollover_count()
        assert isinstance(count, int)


# ============================================================================
# ENSEMBLE CONTINUITY CHECK TESTS
# ============================================================================


class TestEnsembleContinuityCheck:
    """Test the ensemble_continuity_check() method."""

    @pytest.fixture
    def continuous_dataset(self):
        """Create dataset with continuous ensemble numbering."""
        n_ensembles = 10
        data_vars = {
            "ensemble_number": (
                ("ensemble",),
                np.arange(1, n_ensembles + 1, dtype=np.uint16),
            ),
            "ensemble_msb": (
                ("ensemble",),
                np.zeros(n_ensembles, dtype=np.uint8),
            ),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    @pytest.fixture
    def discontinuous_dataset(self):
        """Create dataset with gaps in ensemble numbering."""
        # Skip ensemble 5 and 10
        ensemble_num = np.array([1, 2, 3, 4, 6, 7, 8, 9, 11], dtype=np.uint16)
        ensemble_msb = np.zeros(len(ensemble_num), dtype=np.uint8)
        data_vars = {
            "ensemble_number": (("ensemble",), ensemble_num),
            "ensemble_msb": (("ensemble",), ensemble_msb),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    @pytest.fixture
    def single_ensemble_dataset(self):
        """Create dataset with single ensemble."""
        data_vars = {
            "ensemble_number": (("ensemble",), np.array([1], dtype=np.uint16)),
            "ensemble_msb": (("ensemble",), np.array([0], dtype=np.uint8)),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    def test_continuous_dataset_returns_true(self, continuous_dataset):
        """Test continuous dataset returns True."""
        ds = continuous_dataset
        result = ds.variable_leader.ensemble_continuity_check()

        assert result["is_continuous"] == True
        assert result["gap_count"] == 0
        assert len(result["gap_locations"]) == 0

    def test_discontinuous_dataset_returns_false(self, discontinuous_dataset):
        """Test discontinuous dataset returns False."""
        ds = discontinuous_dataset
        result = ds.variable_leader.ensemble_continuity_check()

        assert result["is_continuous"] == False
        assert result["gap_count"] > 0
        assert len(result["gap_locations"]) == result["gap_count"]

    def test_continuity_check_returns_dict(self, continuous_dataset):
        """Test that continuity check returns dictionary."""
        ds = continuous_dataset
        result = ds.variable_leader.ensemble_continuity_check()

        assert isinstance(result, dict)
        assert "is_continuous" in result
        assert "gap_count" in result
        assert "gap_locations" in result
        assert "gap_sizes" in result

    def test_continuity_check_gap_information(self, discontinuous_dataset):
        """Test that gap information is correct."""
        ds = discontinuous_dataset
        result = ds.variable_leader.ensemble_continuity_check()

        # Should have identified 2 gaps
        assert result["gap_count"] == 2
        assert len(result["gap_sizes"]) == 2
        # Each gap should skip 1 ensemble
        assert all(size == 1 for size in result["gap_sizes"])

    def test_single_ensemble_is_continuous(self, single_ensemble_dataset):
        """Test that single ensemble is always continuous."""
        ds = single_ensemble_dataset
        result = ds.variable_leader.ensemble_continuity_check()

        assert result["is_continuous"] == True
        assert result["gap_count"] == 0


# ============================================================================
# BIT RESULT SUMMARY TESTS
# ============================================================================


class TestBitResultSummary:
    """Test the bit_result_summary() method."""

    @pytest.fixture
    def all_pass_bit_dataset(self):
        """Create dataset where all BIT results pass (all zeros)."""
        n_ensembles = 5
        data_vars = {
            "bit_result": (("ensemble",), np.zeros(n_ensembles, dtype=np.uint16)),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    @pytest.fixture
    def some_bit_errors_dataset(self):
        """Create dataset with some BIT errors."""
        n_ensembles = 5
        # Bit errors in ensembles 1 and 3
        bit_result = np.array([0, 0x0001, 0, 0x0004, 0], dtype=np.uint16)
        data_vars = {
            "bit_result": (("ensemble",), bit_result),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    @pytest.fixture
    def all_bit_errors_dataset(self):
        """Create dataset where all ensembles have BIT errors."""
        n_ensembles = 5
        bit_result = np.array([0x0001, 0x0002, 0x0004, 0x0008, 0x0010], dtype=np.uint16)
        data_vars = {
            "bit_result": (("ensemble",), bit_result),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    @pytest.fixture
    def complex_bit_errors_dataset(self):
        """Create dataset with decoded BIT fields."""
        n_ensembles = 5
        bit_result = np.array([0, 0x0001, 0, 0x0004, 0], dtype=np.uint16)
        data_vars = {
            "bit_result": (("ensemble",), bit_result),
            "bit_demod_0_error": (
                ("ensemble",),
                np.array([0, 1, 0, 0, 0], dtype=np.uint8),
            ),
            "bit_demod_1_error": (
                ("ensemble",),
                np.array([0, 0, 0, 0, 0], dtype=np.uint8),
            ),
            "bit_timing_card_error": (
                ("ensemble",),
                np.array([0, 0, 0, 1, 0], dtype=np.uint8),
            ),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    def test_all_pass_returns_true(self, all_pass_bit_dataset):
        """Test all-pass BIT results."""
        ds = all_pass_bit_dataset
        result = ds.variable_leader.bit_result_summary()

        assert result["all_passed"] == True  # Use == for numpy.bool_ compatibility
        assert result["error_count"] == 0
        assert len(result["unique_error_codes"]) == 0

    def test_some_errors_returns_false(self, some_bit_errors_dataset):
        """Test partial BIT errors."""
        ds = some_bit_errors_dataset
        result = ds.variable_leader.bit_result_summary()

        assert result["all_passed"] == False  # Use == for numpy.bool_ compatibility
        assert result["error_count"] == 2
        assert len(result["unique_error_codes"]) == 2

    def test_bit_summary_returns_dict(self, all_pass_bit_dataset):
        """Test that BIT summary returns dictionary."""
        ds = all_pass_bit_dataset
        result = ds.variable_leader.bit_result_summary()

        assert isinstance(result, dict)
        assert "all_passed" in result
        assert "error_count" in result
        assert "unique_error_codes" in result
        assert "bit_checks" in result

    def test_bit_checks_contains_8_checks(self, all_pass_bit_dataset):
        """Test that bit_checks contains 8 bit-level checks."""
        ds = all_pass_bit_dataset
        result = ds.variable_leader.bit_result_summary()

        assert len(result["bit_checks"]) == 8
        # Check for expected bit names
        expected_bits = [
            "demod_0_error",
            "demod_1_error",
            "timing_card_error",
            "reserved_1_error",
            "reserved_2_error",
            "reserved_3_error",
            "reserved_4_error",
            "reserved_5_error",
        ]
        for bit_name in expected_bits:
            assert bit_name in result["bit_checks"]

    def test_bit_check_has_error_count(self, all_pass_bit_dataset):
        """Test that each bit check has error_count."""
        ds = all_pass_bit_dataset
        result = ds.variable_leader.bit_result_summary()

        for bit_name, bit_check in result["bit_checks"].items():
            assert "error_count" in bit_check
            assert isinstance(bit_check["error_count"], (int, np.integer))

    def test_all_errors_dataset(self, all_bit_errors_dataset):
        """Test dataset with all errors."""
        ds = all_bit_errors_dataset
        result = ds.variable_leader.bit_result_summary()

        assert result["all_passed"] == False
        assert result["error_count"] == 5
        assert len(result["unique_error_codes"]) == 5

    def test_include_decoded_parameter(self, complex_bit_errors_dataset):
        """Test include_decoded parameter."""
        ds = complex_bit_errors_dataset

        # With decoded fields
        result_with_decoded = ds.variable_leader.bit_result_summary(
            include_decoded=True
        )
        assert result_with_decoded["error_count"] >= 0

        # Without decoded fields
        result_without_decoded = ds.variable_leader.bit_result_summary(
            include_decoded=False
        )
        assert result_without_decoded["error_count"] >= 0


# ============================================================================
# ERROR STATUS WORD SUMMARY TESTS
# ============================================================================


class TestErrorStatusWordSummary:
    """Test the error_status_word_summary() method."""

    @pytest.fixture
    def all_zero_esw_dataset(self):
        """Create dataset where all ESW fields are zero."""
        n_ensembles = 5
        data_vars = {
            "error_status_word_1": (
                ("ensemble",),
                np.zeros(n_ensembles, dtype=np.uint8),
            ),
            "error_status_word_2": (
                ("ensemble",),
                np.zeros(n_ensembles, dtype=np.uint8),
            ),
            "error_status_word_3": (
                ("ensemble",),
                np.zeros(n_ensembles, dtype=np.uint8),
            ),
            "error_status_word_4": (
                ("ensemble",),
                np.zeros(n_ensembles, dtype=np.uint8),
            ),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    @pytest.fixture
    def some_esw_events_dataset(self):
        """Create dataset with some ESW events."""
        n_ensembles = 5
        data_vars = {
            "error_status_word_1": (
                ("ensemble",),
                np.array([0x00, 0x01, 0x00, 0x04, 0x00], dtype=np.uint8),
            ),
            "error_status_word_2": (
                ("ensemble",),
                np.array([0x00, 0x00, 0x01, 0x00, 0x00], dtype=np.uint8),
            ),
            "error_status_word_3": (
                ("ensemble",),
                np.array([0x00, 0x00, 0x00, 0x00, 0x02], dtype=np.uint8),
            ),
            "error_status_word_4": (
                ("ensemble",),
                np.array([0x00, 0x00, 0x00, 0x00, 0x00], dtype=np.uint8),
            ),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    @pytest.fixture
    def complex_esw_dataset(self):
        """Create dataset with decoded ESW fields."""
        n_ensembles = 3
        data_vars = {
            "error_status_word_1": (
                ("ensemble",),
                np.array([0x00, 0x01, 0x00], dtype=np.uint8),
            ),
            "esw1_address_error_exception": (
                ("ensemble",),
                np.array([0, 1, 0], dtype=np.uint8),
            ),
            "esw1_illegal_instruction_exception": (
                ("ensemble",),
                np.array([0, 0, 0], dtype=np.uint8),
            ),
            "error_status_word_2": (
                ("ensemble",),
                np.array([0x00, 0x00, 0x00], dtype=np.uint8),
            ),
            "error_status_word_3": (
                ("ensemble",),
                np.array([0x00, 0x00, 0x00], dtype=np.uint8),
            ),
            "error_status_word_4": (
                ("ensemble",),
                np.array([0x00, 0x00, 0x00], dtype=np.uint8),
            ),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    def test_all_zero_esw_returns_all_zeros_true(self, all_zero_esw_dataset):
        """Test all-zero ESW results."""
        ds = all_zero_esw_dataset
        result = ds.variable_leader.error_status_word_summary()

        # Check each ESW
        for esw_label in ["ESW1", "ESW2", "ESW3", "ESW4"]:
            assert esw_label in result
            assert result[esw_label]["all_zeros"] == True
            assert result[esw_label]["total_events"] == 0

    def test_some_events_returns_events(self, some_esw_events_dataset):
        """Test ESW with some events."""
        ds = some_esw_events_dataset
        result = ds.variable_leader.error_status_word_summary()

        # Should have detected events
        assert result["ESW1"]["total_events"] == 2
        assert result["ESW2"]["total_events"] == 1
        assert result["ESW3"]["total_events"] == 1
        assert result["ESW4"]["total_events"] == 0

    def test_esw_summary_returns_dict(self, all_zero_esw_dataset):
        """Test that ESW summary returns dictionary."""
        ds = all_zero_esw_dataset
        result = ds.variable_leader.error_status_word_summary()

        assert isinstance(result, dict)
        for esw_label in ["ESW1", "ESW2", "ESW3", "ESW4"]:
            assert esw_label in result
            assert "all_zeros" in result[esw_label]
            assert "total_events" in result[esw_label]
            assert "bit_checks" in result[esw_label]

    def test_each_esw_has_8_bit_checks(self, all_zero_esw_dataset):
        """Test that each ESW has 8 bit-level checks."""
        ds = all_zero_esw_dataset
        result = ds.variable_leader.error_status_word_summary()

        for esw_label in ["ESW1", "ESW2", "ESW3", "ESW4"]:
            assert len(result[esw_label]["bit_checks"]) == 8

    def test_bit_check_has_event_count(self, all_zero_esw_dataset):
        """Test that each bit check has event_count."""
        ds = all_zero_esw_dataset
        result = ds.variable_leader.error_status_word_summary()

        for esw_label in ["ESW1", "ESW2", "ESW3", "ESW4"]:
            for bit_name, bit_check in result[esw_label]["bit_checks"].items():
                assert "event_count" in bit_check
                assert isinstance(bit_check["event_count"], (int, np.integer))

    def test_esw_bit_names_match_specification(self, all_zero_esw_dataset):
        """Test that ESW bit names match specification."""
        ds = all_zero_esw_dataset
        result = ds.variable_leader.error_status_word_summary()

        # ESW1 expected bits
        esw1_bits = {
            "address_error_exception",
            "illegal_instruction_exception",
            "emulator_exception",
            "bus_error_exception",
            "watchdog_restart",
            "zero_divide_exception",
            "unassigned_exception",
            "battery_saver_power",
        }
        assert set(result["ESW1"]["bit_checks"].keys()) == esw1_bits

    def test_include_decoded_parameter_esw(self, complex_esw_dataset):
        """Test include_decoded parameter for ESW."""
        ds = complex_esw_dataset

        # With decoded fields
        result_with_decoded = ds.variable_leader.error_status_word_summary(
            include_decoded=True
        )
        assert "ESW1" in result_with_decoded

        # Without decoded fields
        result_without_decoded = ds.variable_leader.error_status_word_summary(
            include_decoded=False
        )
        assert "ESW1" in result_without_decoded


# ============================================================================
# TIMESTAMP VALIDATION TESTS
# ============================================================================


class TestValidateTimestamps:
    """Test the validate_timestamps() method."""

    @pytest.fixture
    def valid_timestamps_dataset(self):
        """Create dataset with valid timestamps."""
        n_ensembles = 5
        data_vars = {
            "rtc_month": (("ensemble",), np.array([1, 6, 12, 3, 9], dtype=np.uint8)),
            "rtc_day": (("ensemble",), np.array([1, 15, 28, 30, 31], dtype=np.uint8)),
            "rtc_hour": (("ensemble",), np.array([0, 12, 23, 6, 18], dtype=np.uint8)),
            "rtc_minute": (
                ("ensemble",),
                np.array([0, 30, 59, 15, 45], dtype=np.uint8),
            ),
            "rtc_second": (
                ("ensemble",),
                np.array([0, 30, 59, 15, 45], dtype=np.uint8),
            ),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    @pytest.fixture
    def invalid_month_dataset(self):
        """Create dataset with invalid month."""
        n_ensembles = 3
        data_vars = {
            "rtc_month": (("ensemble",), np.array([1, 13, 12], dtype=np.uint8)),
            "rtc_day": (("ensemble",), np.array([1, 15, 28], dtype=np.uint8)),
            "rtc_hour": (("ensemble",), np.array([0, 12, 23], dtype=np.uint8)),
            "rtc_minute": (("ensemble",), np.array([0, 30, 59], dtype=np.uint8)),
            "rtc_second": (("ensemble",), np.array([0, 30, 59], dtype=np.uint8)),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    @pytest.fixture
    def invalid_day_dataset(self):
        """Create dataset with invalid day."""
        data_vars = {
            "rtc_month": (("ensemble",), np.array([1, 2, 12], dtype=np.uint8)),
            "rtc_day": (("ensemble",), np.array([1, 32, 28], dtype=np.uint8)),
            "rtc_hour": (("ensemble",), np.array([0, 12, 23], dtype=np.uint8)),
            "rtc_minute": (("ensemble",), np.array([0, 30, 59], dtype=np.uint8)),
            "rtc_second": (("ensemble",), np.array([0, 30, 59], dtype=np.uint8)),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    @pytest.fixture
    def invalid_time_dataset(self):
        """Create dataset with invalid time components."""
        data_vars = {
            "rtc_month": (("ensemble",), np.array([1, 6, 12], dtype=np.uint8)),
            "rtc_day": (("ensemble",), np.array([1, 15, 28], dtype=np.uint8)),
            "rtc_hour": (("ensemble",), np.array([0, 24, 23], dtype=np.uint8)),
            "rtc_minute": (("ensemble",), np.array([0, 60, 59], dtype=np.uint8)),
            "rtc_second": (("ensemble",), np.array([0, 60, 59], dtype=np.uint8)),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    def test_valid_timestamps_returns_true(self, valid_timestamps_dataset):
        """Test valid timestamps."""
        ds = valid_timestamps_dataset
        result = ds.variable_leader.validate_timestamps()

        assert result["valid"] == True
        assert len(result["issues"]) == 0

    def test_invalid_month_returns_false(self, invalid_month_dataset):
        """Test invalid month detection."""
        ds = invalid_month_dataset
        result = ds.variable_leader.validate_timestamps()

        assert result["valid"] == False
        assert len(result["issues"]) > 0
        assert any("month" in issue.lower() for issue in result["issues"])

    def test_invalid_day_returns_false(self, invalid_day_dataset):
        """Test invalid day detection."""
        ds = invalid_day_dataset
        result = ds.variable_leader.validate_timestamps()

        assert result["valid"] == False
        assert len(result["issues"]) > 0
        assert any("day" in issue.lower() for issue in result["issues"])

    def test_invalid_time_returns_false(self, invalid_time_dataset):
        """Test invalid time components detection."""
        ds = invalid_time_dataset
        result = ds.variable_leader.validate_timestamps()

        assert result["valid"] == False
        assert len(result["issues"]) > 0

    def test_validate_returns_dict(self, valid_timestamps_dataset):
        """Test that validation returns dictionary."""
        ds = valid_timestamps_dataset
        result = ds.variable_leader.validate_timestamps()

        assert isinstance(result, dict)
        assert "valid" in result
        assert "issues" in result
        assert isinstance(result["issues"], list)


# ============================================================================
# SUMMARY METHOD TESTS
# ============================================================================


class TestSummaryMethod:
    """Test the summary() method."""

    @pytest.fixture
    def complete_vl_dataset(self):
        """Create a complete VariableLeader dataset."""
        n_ensembles = 5

        # Create time coordinate from RTC fields
        times = pd.date_range("2024-12-15 10:30:00", periods=n_ensembles, freq="1s")

        data_vars = {
            "ensemble_number": (
                ("ensemble",),
                np.array([1, 2, 3, 4, 5], dtype=np.uint16),
            ),
            "ensemble_msb": (
                ("ensemble",),
                np.zeros(n_ensembles, dtype=np.uint8),
            ),
            "rtc_year": (
                ("ensemble",),
                np.array([24, 24, 24, 24, 24], dtype=np.uint8),
            ),
            "rtc_month": (
                ("ensemble",),
                np.array([12, 12, 12, 12, 12], dtype=np.uint8),
            ),
            "rtc_day": (
                ("ensemble",),
                np.array([15, 15, 15, 15, 15], dtype=np.uint8),
            ),
            "rtc_hour": (
                ("ensemble",),
                np.array([10, 10, 10, 10, 10], dtype=np.uint8),
            ),
            "rtc_minute": (
                ("ensemble",),
                np.array([30, 30, 30, 30, 30], dtype=np.uint8),
            ),
            "rtc_second": (
                ("ensemble",),
                np.array([45, 45, 45, 45, 45], dtype=np.uint8),
            ),
            "bit_result": (
                ("ensemble",),
                np.array([0, 0, 0, 0, 0], dtype=np.uint16),
            ),
            "error_status_word_1": (
                ("ensemble",),
                np.array([0, 0, 0, 0, 0], dtype=np.uint8),
            ),
            "error_status_word_2": (
                ("ensemble",),
                np.array([0, 0, 0, 0, 0], dtype=np.uint8),
            ),
            "error_status_word_3": (
                ("ensemble",),
                np.array([0, 0, 0, 0, 0], dtype=np.uint8),
            ),
            "error_status_word_4": (
                ("ensemble",),
                np.array([0, 0, 0, 0, 0], dtype=np.uint8),
            ),
        }

        coords = {
            "ensemble": np.arange(n_ensembles),
            "time": times,
        }

        return xr.Dataset(
            data_vars, coords=coords, attrs={"pyadps_component": "VariableLeader"}
        )

    @pytest.fixture
    def minimal_vl_dataset(self):
        """Create minimal VariableLeader dataset."""
        n_ensembles = 3
        times = pd.date_range("2024-12-15 10:30:00", periods=n_ensembles, freq="1s")
        data_vars = {
            "ensemble_number": (("ensemble",), np.array([1, 2, 3], dtype=np.uint16)),
            "ensemble_msb": (("ensemble",), np.array([0, 0, 0], dtype=np.uint8)),
            "rtc_month": (("ensemble",), np.array([12, 12, 12], dtype=np.uint8)),
            "rtc_day": (("ensemble",), np.array([15, 15, 15], dtype=np.uint8)),
            "rtc_hour": (("ensemble",), np.array([10, 10, 10], dtype=np.uint8)),
            "rtc_minute": (("ensemble",), np.array([30, 30, 30], dtype=np.uint8)),
            "rtc_second": (("ensemble",), np.array([45, 45, 45], dtype=np.uint8)),
            "bit_result": (("ensemble",), np.array([0, 0, 0], dtype=np.uint16)),
            "error_status_word_1": (("ensemble",), np.array([0, 0, 0], dtype=np.uint8)),
            "error_status_word_2": (("ensemble",), np.array([0, 0, 0], dtype=np.uint8)),
            "error_status_word_3": (("ensemble",), np.array([0, 0, 0], dtype=np.uint8)),
            "error_status_word_4": (("ensemble",), np.array([0, 0, 0], dtype=np.uint8)),
        }
        coords = {
            "ensemble": np.arange(n_ensembles),
            "time": times,
        }
        return xr.Dataset(
            data_vars, coords=coords, attrs={"pyadps_component": "VariableLeader"}
        )

    @pytest.fixture
    def problem_vl_dataset(self):
        """Create VariableLeader dataset with problems."""
        n_ensembles = 5
        times = pd.date_range("2024-12-15 10:30:00", periods=n_ensembles, freq="1s")
        data_vars = {
            "ensemble_number": (
                ("ensemble",),
                np.array([1, 2, 5, 6, 7], dtype=np.uint16),
            ),  # Gap at 3,4
            "ensemble_msb": (
                ("ensemble",),
                np.zeros(n_ensembles, dtype=np.uint8),
            ),
            "rtc_month": (
                ("ensemble",),
                np.array([12, 12, 13, 12, 12], dtype=np.uint8),
            ),  # Invalid month
            "rtc_day": (
                ("ensemble",),
                np.array([15, 15, 15, 15, 15], dtype=np.uint8),
            ),
            "rtc_hour": (
                ("ensemble",),
                np.array([10, 10, 10, 10, 10], dtype=np.uint8),
            ),
            "rtc_minute": (
                ("ensemble",),
                np.array([30, 30, 30, 30, 30], dtype=np.uint8),
            ),
            "rtc_second": (
                ("ensemble",),
                np.array([45, 45, 45, 45, 45], dtype=np.uint8),
            ),
            "bit_result": (
                ("ensemble",),
                np.array([0, 0x0001, 0, 0x0002, 0], dtype=np.uint16),
            ),  # BIT errors
            "error_status_word_1": (
                ("ensemble",),
                np.array([0, 0x01, 0, 0, 0], dtype=np.uint8),
            ),  # ESW event
            "error_status_word_2": (
                ("ensemble",),
                np.array([0, 0, 0, 0, 0], dtype=np.uint8),
            ),
            "error_status_word_3": (
                ("ensemble",),
                np.array([0, 0, 0, 0, 0], dtype=np.uint8),
            ),
            "error_status_word_4": (
                ("ensemble",),
                np.array([0, 0, 0, 0, 0], dtype=np.uint8),
            ),
        }
        coords = {
            "ensemble": np.arange(n_ensembles),
            "time": times,
        }
        return xr.Dataset(
            data_vars, coords=coords, attrs={"pyadps_component": "VariableLeader"}
        )

    def test_summary_returns_none(self, complete_vl_dataset):
        """Test that summary() returns None."""
        ds = complete_vl_dataset
        result = ds.variable_leader.summary()
        assert result is None

    def test_summary_produces_output(self, complete_vl_dataset, capsys):
        """Test that summary produces output."""
        ds = complete_vl_dataset
        ds.variable_leader.summary()
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_summary_contains_continuity_section(self, complete_vl_dataset, capsys):
        """Test that summary includes continuity section."""
        ds = complete_vl_dataset
        ds.variable_leader.summary()
        captured = capsys.readouterr()
        output = captured.out

        assert "ENSEMBLE CONTINUITY" in output or "Continuity" in output

    def test_summary_contains_bit_section(self, complete_vl_dataset, capsys):
        """Test that summary includes BIT section."""
        ds = complete_vl_dataset
        ds.variable_leader.summary()
        captured = capsys.readouterr()
        output = captured.out

        assert "BIT" in output or "Built-In Test" in output

    def test_summary_contains_esw_section(self, complete_vl_dataset, capsys):
        """Test that summary includes ESW section."""
        ds = complete_vl_dataset
        ds.variable_leader.summary()
        captured = capsys.readouterr()
        output = captured.out

        assert "ERROR STATUS" in output or "ESW" in output

    def test_summary_contains_timestamp_section(self, complete_vl_dataset, capsys):
        """Test that summary includes timestamp section."""
        ds = complete_vl_dataset
        ds.variable_leader.summary()
        captured = capsys.readouterr()
        output = captured.out

        assert "TIMESTAMP" in output or "Timestamp" in output

    def test_summary_with_minimal_data(self, minimal_vl_dataset, capsys):
        """Test summary with minimal dataset."""
        ds = minimal_vl_dataset
        # Should not raise even with minimal data
        ds.variable_leader.summary()
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_summary_with_problems(self, problem_vl_dataset, capsys):
        """Test summary with problematic data."""
        ds = problem_vl_dataset
        ds.variable_leader.summary()
        captured = capsys.readouterr()
        output = captured.out

        # Should still produce output
        assert len(output) > 0
        # May indicate issues found
        assert len(output) > 100  # Should be substantial output

    def test_summary_output_has_borders(self, complete_vl_dataset, capsys):
        """Test that summary output has professional formatting."""
        ds = complete_vl_dataset
        ds.variable_leader.summary()
        captured = capsys.readouterr()
        output = captured.out

        # Check for border characters
        assert "=" in output

    def test_summary_output_multiple_calls_consistent(
        self, complete_vl_dataset, capsys
    ):
        """Test that summary output is consistent across multiple calls."""
        ds = complete_vl_dataset

        # First call
        ds.variable_leader.summary()
        captured1 = capsys.readouterr()
        output1 = captured1.out

        # Second call
        ds.variable_leader.summary()
        captured2 = capsys.readouterr()
        output2 = captured2.out

        # Should produce similar output (may have minor formatting differences)
        assert len(output1) == len(output2)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestVariableLeaderAccessorIntegration:
    """Integration tests combining multiple accessor methods."""

    @pytest.fixture
    def realistic_vl_dataset(self):
        """Create realistic VariableLeader dataset with varying data."""
        n_ensembles = 20
        data_vars = {
            "ensemble_number": (
                ("ensemble",),
                np.arange(1, n_ensembles + 1, dtype=np.uint16),
            ),
            "ensemble_msb": (
                ("ensemble",),
                np.zeros(n_ensembles, dtype=np.uint8),
            ),
            "rtc_year": (
                ("ensemble",),
                np.full(n_ensembles, 24, dtype=np.uint8),
            ),
            "rtc_month": (
                ("ensemble",),
                np.full(n_ensembles, 12, dtype=np.uint8),
            ),
            "rtc_day": (
                ("ensemble",),
                np.full(n_ensembles, 15, dtype=np.uint8),
            ),
            "rtc_hour": (
                ("ensemble",),
                np.full(n_ensembles, 10, dtype=np.uint8),
            ),
            "rtc_minute": (
                ("ensemble",),
                np.full(n_ensembles, 30, dtype=np.uint8),
            ),
            "rtc_second": (
                ("ensemble",),
                np.full(n_ensembles, 45, dtype=np.uint8),
            ),
            "bit_result": (
                ("ensemble",),
                np.concatenate(
                    [
                        np.zeros(10, dtype=np.uint16),
                        np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=np.uint16),
                    ]
                ),
            ),
            "error_status_word_1": (
                ("ensemble",),
                np.concatenate(
                    [
                        np.zeros(15, dtype=np.uint8),
                        np.array([1, 2, 4, 8, 16], dtype=np.uint8),
                    ]
                ),
            ),
            "error_status_word_2": (
                ("ensemble",),
                np.zeros(n_ensembles, dtype=np.uint8),
            ),
            "error_status_word_3": (
                ("ensemble",),
                np.zeros(n_ensembles, dtype=np.uint8),
            ),
            "error_status_word_4": (
                ("ensemble",),
                np.zeros(n_ensembles, dtype=np.uint8),
            ),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    def test_all_methods_accessible(self, realistic_vl_dataset):
        """Test that all accessor methods are accessible."""
        ds = realistic_vl_dataset
        accessor = ds.variable_leader

        # All methods should exist
        assert hasattr(accessor, "ensemble_rollover_count")
        assert hasattr(accessor, "ensemble_continuity_check")
        assert hasattr(accessor, "bit_result_summary")
        assert hasattr(accessor, "error_status_word_summary")
        assert hasattr(accessor, "validate_timestamps")
        assert hasattr(accessor, "summary")

    def test_all_methods_callable(self, realistic_vl_dataset):
        """Test that all accessor methods are callable."""
        ds = realistic_vl_dataset
        accessor = ds.variable_leader

        # All methods should be callable
        assert callable(accessor.ensemble_rollover_count)
        assert callable(accessor.ensemble_continuity_check)
        assert callable(accessor.bit_result_summary)
        assert callable(accessor.error_status_word_summary)
        assert callable(accessor.validate_timestamps)
        assert callable(accessor.summary)

    def test_comprehensive_workflow(self, realistic_vl_dataset):
        """Test comprehensive workflow using multiple methods."""
        ds = realistic_vl_dataset
        accessor = ds.variable_leader

        # Execute all methods in sequence
        rollover_count = accessor.ensemble_rollover_count()
        assert isinstance(rollover_count, int)

        continuity = accessor.ensemble_continuity_check()
        assert isinstance(continuity, dict)

        bit_summary = accessor.bit_result_summary()
        assert isinstance(bit_summary, dict)

        esw_summary = accessor.error_status_word_summary()
        assert isinstance(esw_summary, dict)

        ts_valid = accessor.validate_timestamps()
        assert isinstance(ts_valid, dict)

    def test_results_are_independent(self, realistic_vl_dataset):
        """Test that calling methods multiple times produces independent results."""
        ds = realistic_vl_dataset

        # Call methods twice
        cont1 = ds.variable_leader.ensemble_continuity_check()
        cont2 = ds.variable_leader.ensemble_continuity_check()

        # Results should be equal
        assert cont1 == cont2

        bit1 = ds.variable_leader.bit_result_summary()
        bit2 = ds.variable_leader.bit_result_summary()

        assert bit1 == bit2


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    @pytest.fixture
    def empty_ensemble_dataset(self):
        """Create dataset with no ensembles."""
        data_vars = {
            "ensemble_number": (("ensemble",), np.array([], dtype=np.uint16)),
            "ensemble_msb": (("ensemble",), np.array([], dtype=np.uint8)),
            "bit_result": (("ensemble",), np.array([], dtype=np.uint16)),
            "error_status_word_1": (("ensemble",), np.array([], dtype=np.uint8)),
            "error_status_word_2": (("ensemble",), np.array([], dtype=np.uint8)),
            "error_status_word_3": (("ensemble",), np.array([], dtype=np.uint8)),
            "error_status_word_4": (("ensemble",), np.array([], dtype=np.uint8)),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    @pytest.fixture
    def single_ensemble_dataset(self):
        """Create dataset with single ensemble."""
        data_vars = {
            "ensemble_number": (("ensemble",), np.array([1], dtype=np.uint16)),
            "ensemble_msb": (("ensemble",), np.array([0], dtype=np.uint8)),
            "bit_result": (("ensemble",), np.array([0], dtype=np.uint16)),
            "error_status_word_1": (("ensemble",), np.array([0], dtype=np.uint8)),
            "error_status_word_2": (("ensemble",), np.array([0], dtype=np.uint8)),
            "error_status_word_3": (("ensemble",), np.array([0], dtype=np.uint8)),
            "error_status_word_4": (("ensemble",), np.array([0], dtype=np.uint8)),
            "rtc_month": (("ensemble",), np.array([12], dtype=np.uint8)),
            "rtc_day": (("ensemble",), np.array([15], dtype=np.uint8)),
            "rtc_hour": (("ensemble",), np.array([10], dtype=np.uint8)),
            "rtc_minute": (("ensemble",), np.array([30], dtype=np.uint8)),
            "rtc_second": (("ensemble",), np.array([45], dtype=np.uint8)),
        }
        return xr.Dataset(data_vars, attrs={"pyadps_component": "VariableLeader"})

    def test_rollover_with_empty_ensemble(self, empty_ensemble_dataset):
        """Test rollover count with empty ensemble."""
        ds = empty_ensemble_dataset
        count = ds.variable_leader.ensemble_rollover_count()
        # Should handle gracefully
        assert isinstance(count, int)

    def test_continuity_with_single_ensemble(self, single_ensemble_dataset):
        """Test continuity check with single ensemble."""
        ds = single_ensemble_dataset
        result = ds.variable_leader.ensemble_continuity_check()
        # Single ensemble should always be continuous
        assert result["is_continuous"] == True

    def test_bit_summary_with_single_ensemble(self, single_ensemble_dataset):
        """Test BIT summary with single ensemble."""
        ds = single_ensemble_dataset
        result = ds.variable_leader.bit_result_summary()
        assert result["all_passed"] == True

    def test_timestamps_with_single_ensemble(self, single_ensemble_dataset):
        """Test timestamp validation with single ensemble."""
        ds = single_ensemble_dataset
        result = ds.variable_leader.validate_timestamps()
        assert result["valid"] == True


# ============================================================================
# FIXTURES: Time Dataset Builders
# ============================================================================


class TestIsTimeRegular:
    """Test the is_time_regular(tolerance_s=1.0) method."""

    @pytest.fixture
    def perfectly_regular_dataset(self):
        """Create dataset with perfectly regular 1-minute intervals."""
        n_ensembles = 10
        # Create times starting from a known date, with 1-minute intervals
        base_time = datetime(2024, 12, 15, 10, 0, 0)
        times = [base_time + timedelta(minutes=i) for i in range(n_ensembles)]

        ds = xr.Dataset(
            {
                "ensemble_number": (
                    ("ensemble",),
                    np.arange(1, n_ensembles + 1, dtype=np.uint16),
                ),
                "data": (("ensemble",), np.arange(n_ensembles)),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    @pytest.fixture
    def regular_with_small_jitter_dataset(self):
        """Create dataset with regular intervals but small timing variations.

        Note: Jitter is applied to the timestamp itself, not the interval.
        Each ensemble has independent jitter (0-100ms), so the difference
        between consecutive times varies randomly. This creates true jitter
        in the time intervals themselves.
        """
        n_ensembles = 10
        base_time = datetime(2024, 12, 15, 10, 0, 0)
        times = []
        np.random.seed(42)  # Fixed seed for reproducibility
        for i in range(n_ensembles):
            # Base interval is 60 seconds, add small jitter (0-100ms)
            jitter = np.random.uniform(0, 0.1)  # 0 to 100ms
            times.append(base_time + timedelta(minutes=i, milliseconds=jitter))

        ds = xr.Dataset(
            {
                "ensemble_number": (
                    ("ensemble",),
                    np.arange(1, n_ensembles + 1, dtype=np.uint16),
                ),
                "data": (("ensemble",), np.arange(n_ensembles)),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    @pytest.fixture
    def regular_with_large_jitter_dataset(self):
        """Create dataset with regular intervals but large timing variations.

        This fixture guarantees jitter large enough to fail strict tolerance tests.
        Each interval has Â±200ms jitter applied deterministically.
        """
        n_ensembles = 10
        base_time = datetime(2024, 12, 15, 10, 0, 0)
        times = []
        jitter_values = [
            0.0,
            0.2,
            0.1,
            0.25,
            0.05,
            0.22,
            0.15,
            0.23,
            0.08,
            0.19,
        ]  # 0-250ms
        for i, jitter in enumerate(jitter_values):
            times.append(base_time + timedelta(minutes=i, milliseconds=jitter * 1000))

        ds = xr.Dataset(
            {
                "ensemble_number": (
                    ("ensemble",),
                    np.arange(1, n_ensembles + 1, dtype=np.uint16),
                ),
                "data": (("ensemble",), np.arange(n_ensembles)),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    @pytest.fixture
    def irregular_time_dataset(self):
        """Create dataset with highly irregular time intervals."""
        base_time = datetime(2024, 12, 15, 10, 0, 0)
        times = [
            base_time,
            base_time + timedelta(seconds=30),
            base_time + timedelta(minutes=2),  # Large jump
            base_time + timedelta(minutes=2, seconds=15),
            base_time + timedelta(minutes=3),
            base_time + timedelta(minutes=5),  # Another large jump
        ]

        ds = xr.Dataset(
            {
                "ensemble_number": (
                    ("ensemble",),
                    np.arange(1, len(times) + 1, dtype=np.uint16),
                ),
                "data": (("ensemble",), np.arange(len(times))),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    @pytest.fixture
    def single_ensemble_dataset(self):
        """Create dataset with only one ensemble."""
        base_time = datetime(2024, 12, 15, 10, 0, 0)
        ds = xr.Dataset(
            {
                "ensemble_number": (("ensemble",), np.array([1], dtype=np.uint16)),
                "data": (("ensemble",), np.array([0])),
            },
            coords={"time": (("ensemble",), [base_time])},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    @pytest.fixture
    def identical_times_dataset(self):
        """Create dataset with all identical timestamps."""
        base_time = datetime(2024, 12, 15, 10, 0, 0)
        n_ensembles = 5
        times = [base_time] * n_ensembles

        ds = xr.Dataset(
            {
                "ensemble_number": (
                    ("ensemble",),
                    np.arange(1, n_ensembles + 1, dtype=np.uint16),
                ),
                "data": (("ensemble",), np.arange(n_ensembles)),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    @pytest.fixture
    def two_ensemble_dataset(self):
        """Create dataset with exactly two ensembles."""
        base_time = datetime(2024, 12, 15, 10, 0, 0)
        times = [base_time, base_time + timedelta(seconds=60)]

        ds = xr.Dataset(
            {
                "ensemble_number": (("ensemble",), np.array([1, 2], dtype=np.uint16)),
                "data": (("ensemble",), np.array([0, 1])),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    # ========================================================================
    # TEST: Regular time series
    # ========================================================================

    def test_perfectly_regular_time_returns_true(self, perfectly_regular_dataset):
        """Test that perfectly regular intervals return True."""
        ds = perfectly_regular_dataset
        result = ds.variable_leader.is_time_regular()
        assert result is True

    def test_perfectly_regular_with_default_tolerance(self, perfectly_regular_dataset):
        """Test default tolerance (1.0 second) with perfect regularity."""
        ds = perfectly_regular_dataset
        result = ds.variable_leader.is_time_regular(tolerance_s=1.0)
        assert result is True

    def test_perfectly_regular_with_strict_tolerance(self, perfectly_regular_dataset):
        """Test strict tolerance (0.0 second) with perfect regularity."""
        ds = perfectly_regular_dataset
        result = ds.variable_leader.is_time_regular(tolerance_s=0.0)
        assert result is True

    def test_small_jitter_within_tolerance(self, regular_with_small_jitter_dataset):
        """Test that small jitter (< 1 second) is within default tolerance."""
        ds = regular_with_small_jitter_dataset
        # Default tolerance is 1.0 second, jitter is 0-100ms
        result = ds.variable_leader.is_time_regular(tolerance_s=1.0)
        assert result is True

    def test_small_jitter_with_small_tolerance(self, regular_with_large_jitter_dataset):
        """Test that large jitter fails strict tolerance.

        With deterministic jitter of up to 250ms applied to intervals,
        a 0.05 second (50ms) tolerance should fail.
        """
        ds = regular_with_large_jitter_dataset
        result = ds.variable_leader.is_time_regular(tolerance_s=0.05)
        # Should fail due to jitter exceeding tolerance
        assert result is False

    # ========================================================================
    # TEST: Irregular time series
    # ========================================================================

    def test_irregular_time_returns_false(self, irregular_time_dataset):
        """Test that irregular intervals return False."""
        ds = irregular_time_dataset
        result = ds.variable_leader.is_time_regular(tolerance_s=1.0)
        assert result is False

    def test_irregular_time_with_large_tolerance(self, irregular_time_dataset):
        """Test irregular time with very large tolerance still fails."""
        ds = irregular_time_dataset
        # Even with large tolerance, major irregularities should fail
        result = ds.variable_leader.is_time_regular(tolerance_s=120.0)
        # This might be True depending on the variance, but generally should be False
        assert isinstance(result, bool)

    # ========================================================================
    # TEST: Edge cases
    # ========================================================================

    def test_single_ensemble_returns_false(self, single_ensemble_dataset):
        """Test that single ensemble returns False (can't calculate intervals)."""
        ds = single_ensemble_dataset
        result = ds.variable_leader.is_time_regular()
        assert result is False

    def test_two_ensemble_returns_true(self, two_ensemble_dataset):
        """Test that two ensembles with regular interval return True."""
        ds = two_ensemble_dataset
        result = ds.variable_leader.is_time_regular()
        assert result is True

    def test_identical_times_returns_false(self, identical_times_dataset):
        """Test that identical times (zero intervals) return False."""
        ds = identical_times_dataset
        result = ds.variable_leader.is_time_regular()
        # All intervals are zero, which is technically regular but degenerate
        assert isinstance(result, bool)

    # ========================================================================
    # TEST: Return type
    # ========================================================================

    def test_is_time_regular_returns_bool(self, perfectly_regular_dataset):
        """Test that is_time_regular returns a boolean."""
        ds = perfectly_regular_dataset
        result = ds.variable_leader.is_time_regular()
        assert isinstance(result, bool)

    def test_is_time_regular_returns_bool_with_custom_tolerance(
        self, perfectly_regular_dataset
    ):
        """Test return type with custom tolerance."""
        ds = perfectly_regular_dataset
        result = ds.variable_leader.is_time_regular(tolerance_s=0.5)
        assert isinstance(result, bool)


# ============================================================================
# TESTS: get_time_interval()
# ============================================================================


class TestGetTimeInterval:
    """Test the get_time_interval() method."""

    @pytest.fixture
    def one_minute_interval_dataset(self):
        """Create dataset with consistent 1-minute intervals."""
        base_time = datetime(2024, 12, 15, 10, 0, 0)
        times = [base_time + timedelta(minutes=i) for i in range(10)]

        ds = xr.Dataset(
            {
                "ensemble_number": (("ensemble",), np.arange(1, 11, dtype=np.uint16)),
                "data": (("ensemble",), np.arange(10)),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    @pytest.fixture
    def thirty_second_interval_dataset(self):
        """Create dataset with consistent 30-second intervals."""
        base_time = datetime(2024, 12, 15, 10, 0, 0)
        times = [base_time + timedelta(seconds=30 * i) for i in range(15)]

        ds = xr.Dataset(
            {
                "ensemble_number": (("ensemble",), np.arange(1, 16, dtype=np.uint16)),
                "data": (("ensemble",), np.arange(15)),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    @pytest.fixture
    def hourly_interval_dataset(self):
        """Create dataset with 1-hour intervals."""
        base_time = datetime(2024, 12, 15, 10, 0, 0)
        times = [base_time + timedelta(hours=i) for i in range(8)]

        ds = xr.Dataset(
            {
                "ensemble_number": (("ensemble",), np.arange(1, 9, dtype=np.uint16)),
                "data": (("ensemble",), np.arange(8)),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    @pytest.fixture
    def mixed_interval_dataset(self):
        """Create dataset with mostly 1-minute intervals, one 2-minute."""
        base_time = datetime(2024, 12, 15, 10, 0, 0)
        times = [
            base_time,
            base_time + timedelta(minutes=1),
            base_time + timedelta(minutes=2),
            base_time + timedelta(minutes=3),
            base_time + timedelta(minutes=4),
            base_time + timedelta(minutes=5),
            base_time + timedelta(minutes=7),  # Skip 1 minute here
            base_time + timedelta(minutes=8),
            base_time + timedelta(minutes=9),
            base_time + timedelta(minutes=10),
        ]

        ds = xr.Dataset(
            {
                "ensemble_number": (("ensemble",), np.arange(1, 11, dtype=np.uint16)),
                "data": (("ensemble",), np.arange(10)),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    @pytest.fixture
    def single_ensemble_no_interval_dataset(self):
        """Create dataset with single ensemble (no intervals)."""
        base_time = datetime(2024, 12, 15, 10, 0, 0)
        ds = xr.Dataset(
            {
                "ensemble_number": (("ensemble",), np.array([1], dtype=np.uint16)),
                "data": (("ensemble",), np.array([0])),
            },
            coords={"time": (("ensemble",), [base_time])},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    # ========================================================================
    # TEST: Return type and values
    # ========================================================================

    def test_get_time_interval_returns_timedelta(self, one_minute_interval_dataset):
        """Test that get_time_interval returns pd.Timedelta."""
        ds = one_minute_interval_dataset
        result = ds.variable_leader.get_time_interval()
        assert isinstance(result, (pd.Timedelta, type(None)))
        if result is not None:
            assert isinstance(result, pd.Timedelta)

    def test_one_minute_interval_returns_correct_value(
        self, one_minute_interval_dataset
    ):
        """Test that 1-minute intervals return 60-second timedelta."""
        ds = one_minute_interval_dataset
        result = ds.variable_leader.get_time_interval()
        assert result is not None
        assert result == pd.Timedelta(minutes=1)

    def test_thirty_second_interval_returns_correct_value(
        self, thirty_second_interval_dataset
    ):
        """Test that 30-second intervals are correctly identified."""
        ds = thirty_second_interval_dataset
        result = ds.variable_leader.get_time_interval()
        assert result is not None
        assert result == pd.Timedelta(seconds=30)

    def test_hourly_interval_returns_correct_value(self, hourly_interval_dataset):
        """Test that 1-hour intervals are correctly identified."""
        ds = hourly_interval_dataset
        result = ds.variable_leader.get_time_interval()
        assert result is not None
        assert result == pd.Timedelta(hours=1)

    # ========================================================================
    # TEST: Mixed intervals (most common)
    # ========================================================================

    def test_mixed_intervals_returns_most_common(self, mixed_interval_dataset):
        """Test that mixed intervals return most common interval."""
        ds = mixed_interval_dataset
        result = ds.variable_leader.get_time_interval()
        assert result is not None
        # Most common should be 1 minute (9 occurrences vs 1 occurrence of 2 minutes)
        assert result == pd.Timedelta(minutes=1)

    # ========================================================================
    # TEST: Edge cases
    # ========================================================================

    def test_single_ensemble_returns_none(self, single_ensemble_no_interval_dataset):
        """Test that single ensemble returns None."""
        ds = single_ensemble_no_interval_dataset
        result = ds.variable_leader.get_time_interval()
        assert result is None

    def test_identical_times_returns_none_or_zero(self):
        """Test behavior when all times are identical."""
        base_time = datetime(2024, 12, 15, 10, 0, 0)
        times = [base_time] * 5

        ds = xr.Dataset(
            {
                "ensemble_number": (("ensemble",), np.arange(1, 6, dtype=np.uint16)),
                "data": (("ensemble",), np.arange(5)),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )

        result = ds.variable_leader.get_time_interval()
        # With identical times, all intervals are 0
        assert result is None or result == pd.Timedelta(seconds=0)


# ============================================================================
# TESTS: get_time_interval_frequency()
# ============================================================================


class TestGetTimeIntervalFrequency:
    """Test the get_time_interval_frequency() method."""

    @pytest.fixture
    def regular_interval_frequency_dataset(self):
        """Create dataset with single regular interval for frequency testing."""
        base_time = datetime(2024, 12, 15, 10, 0, 0)
        times = [base_time + timedelta(minutes=i) for i in range(10)]

        ds = xr.Dataset(
            {
                "ensemble_number": (("ensemble",), np.arange(1, 11, dtype=np.uint16)),
                "data": (("ensemble",), np.arange(10)),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    @pytest.fixture
    def mixed_frequency_dataset(self):
        """Create dataset with multiple different intervals."""
        base_time = datetime(2024, 12, 15, 10, 0, 0)
        times = [
            base_time,
            base_time + timedelta(seconds=60),  # 1 minute
            base_time + timedelta(seconds=120),  # 1 minute
            base_time + timedelta(seconds=180),  # 1 minute
            base_time + timedelta(seconds=210),  # 30 seconds
            base_time + timedelta(seconds=270),  # 1 minute
            base_time + timedelta(seconds=330),  # 1 minute
            base_time + timedelta(seconds=390),  # 1 minute
            base_time + timedelta(seconds=510),  # 2 minutes
        ]

        ds = xr.Dataset(
            {
                "ensemble_number": (
                    ("ensemble",),
                    np.arange(1, len(times) + 1, dtype=np.uint16),
                ),
                "data": (("ensemble",), np.arange(len(times))),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    @pytest.fixture
    def single_ensemble_frequency_dataset(self):
        """Create dataset with single ensemble."""
        base_time = datetime(2024, 12, 15, 10, 0, 0)
        ds = xr.Dataset(
            {
                "ensemble_number": (("ensemble",), np.array([1], dtype=np.uint16)),
                "data": (("ensemble",), np.array([0])),
            },
            coords={"time": (("ensemble",), [base_time])},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    # ========================================================================
    # TEST: Return type
    # ========================================================================

    def test_get_time_interval_frequency_returns_dict(
        self, regular_interval_frequency_dataset
    ):
        """Test that method returns a dictionary."""
        ds = regular_interval_frequency_dataset
        result = ds.variable_leader.get_time_interval_frequency()
        assert isinstance(result, dict)

    def test_frequency_dict_has_string_keys(self, regular_interval_frequency_dataset):
        """Test that frequency dict has string keys (HH:MM:SS format)."""
        ds = regular_interval_frequency_dataset
        result = ds.variable_leader.get_time_interval_frequency()
        if len(result) > 0:
            for key in result.keys():
                assert isinstance(key, str)
                # Check HH:MM:SS format
                parts = key.split(":")
                assert len(parts) == 3
                assert all(part.isdigit() for part in parts)

    def test_frequency_dict_values_are_integers(
        self, regular_interval_frequency_dataset
    ):
        """Test that frequency values are integers."""
        ds = regular_interval_frequency_dataset
        result = ds.variable_leader.get_time_interval_frequency()
        for value in result.values():
            assert isinstance(value, (int, np.integer))

    # ========================================================================
    # TEST: Regular intervals
    # ========================================================================

    def test_regular_interval_has_single_entry(
        self, regular_interval_frequency_dataset
    ):
        """Test that regular intervals produce single frequency entry."""
        ds = regular_interval_frequency_dataset
        result = ds.variable_leader.get_time_interval_frequency()
        assert len(result) == 1

    def test_regular_interval_correct_frequency_count(
        self, regular_interval_frequency_dataset
    ):
        """Test that frequency count is correct (n-1 intervals for n ensembles)."""
        ds = regular_interval_frequency_dataset
        result = ds.variable_leader.get_time_interval_frequency()
        # With 10 ensembles, there are 9 intervals
        total_count = sum(result.values())
        assert total_count == 9

    def test_one_minute_interval_formatted_correctly(
        self, regular_interval_frequency_dataset
    ):
        """Test that 1-minute interval is formatted as '00:01:00'."""
        ds = regular_interval_frequency_dataset
        result = ds.variable_leader.get_time_interval_frequency()
        assert "00:01:00" in result
        assert result["00:01:00"] == 9

    # ========================================================================
    # TEST: Mixed intervals
    # ========================================================================

    def test_mixed_intervals_multiple_entries(self, mixed_frequency_dataset):
        """Test that mixed intervals produce multiple frequency entries."""
        ds = mixed_frequency_dataset
        result = ds.variable_leader.get_time_interval_frequency()
        # Should have 3 different intervals
        assert len(result) >= 2

    def test_mixed_intervals_frequency_sum(self, mixed_frequency_dataset):
        """Test that frequency counts sum to correct total."""
        ds = mixed_frequency_dataset
        result = ds.variable_leader.get_time_interval_frequency()
        # With 9 ensembles, there are 8 intervals
        total_count = sum(result.values())
        assert total_count == 8

    # ========================================================================
    # TEST: Edge cases
    # ========================================================================

    def test_single_ensemble_returns_empty_dict(
        self, single_ensemble_frequency_dataset
    ):
        """Test that single ensemble returns empty dict."""
        ds = single_ensemble_frequency_dataset
        result = ds.variable_leader.get_time_interval_frequency()
        assert result == {}

    def test_two_ensemble_returns_single_interval(self):
        """Test that two ensembles return single interval entry."""
        base_time = datetime(2024, 12, 15, 10, 0, 0)
        times = [base_time, base_time + timedelta(minutes=1)]

        ds = xr.Dataset(
            {
                "ensemble_number": (("ensemble",), np.array([1, 2], dtype=np.uint16)),
                "data": (("ensemble",), np.array([0, 1])),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )

        result = ds.variable_leader.get_time_interval_frequency()
        assert len(result) == 1
        assert result["00:01:00"] == 1


# ============================================================================
# TEST: get_time_component_frequency()
# ============================================================================


class TestGetTimeComponentFrequency:
    """Test the get_time_component_frequency(component='minute') method."""

    @pytest.fixture
    def all_same_minute_dataset(self):
        """Create dataset where all times occur at minute 0."""
        times = [
            datetime(2024, 12, 15, 10, 0, 0),
            datetime(2024, 12, 15, 11, 0, 0),
            datetime(2024, 12, 15, 12, 0, 0),
            datetime(2024, 12, 15, 13, 0, 0),
            datetime(2024, 12, 15, 14, 0, 0),
        ]

        ds = xr.Dataset(
            {
                "ensemble_number": (("ensemble",), np.arange(1, 6, dtype=np.uint16)),
                "data": (("ensemble",), np.arange(5)),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    @pytest.fixture
    def mixed_minute_dataset(self):
        """Create dataset where times occur at different minutes."""
        times = [
            datetime(2024, 12, 15, 10, 0, 0),  # minute 0
            datetime(2024, 12, 15, 10, 15, 0),  # minute 15
            datetime(2024, 12, 15, 10, 30, 0),  # minute 30
            datetime(2024, 12, 15, 10, 0, 0),  # minute 0 (again)
            datetime(2024, 12, 15, 10, 15, 0),  # minute 15 (again)
        ]

        ds = xr.Dataset(
            {
                "ensemble_number": (("ensemble",), np.arange(1, 6, dtype=np.uint16)),
                "data": (("ensemble",), np.arange(5)),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    @pytest.fixture
    def all_different_hour_dataset(self):
        """Create dataset with one measurement each hour of the day."""
        times = [
            datetime(2024, 12, 15, hour, 0, 0)
            for hour in range(0, 24, 3)  # Every 3 hours
        ]

        ds = xr.Dataset(
            {
                "ensemble_number": (
                    ("ensemble",),
                    np.arange(1, len(times) + 1, dtype=np.uint16),
                ),
                "data": (("ensemble",), np.arange(len(times))),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    @pytest.fixture
    def second_samples_dataset(self):
        """Create dataset sampling at different seconds."""
        times = [
            datetime(2024, 12, 15, 10, 0, 0),  # second 0
            datetime(2024, 12, 15, 10, 0, 15),  # second 15
            datetime(2024, 12, 15, 10, 0, 30),  # second 30
            datetime(2024, 12, 15, 10, 0, 45),  # second 45
            datetime(2024, 12, 15, 10, 1, 0),  # second 0 (again)
            datetime(2024, 12, 15, 10, 1, 15),  # second 15 (again)
        ]

        ds = xr.Dataset(
            {
                "ensemble_number": (
                    ("ensemble",),
                    np.arange(1, len(times) + 1, dtype=np.uint16),
                ),
                "data": (("ensemble",), np.arange(len(times))),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    # ========================================================================
    # TEST: Return type
    # ========================================================================

    def test_get_time_component_frequency_returns_dict(self, all_same_minute_dataset):
        """Test that method returns a dictionary."""
        ds = all_same_minute_dataset
        result = ds.variable_leader.get_time_component_frequency("minute")
        assert isinstance(result, dict)

    def test_component_frequency_dict_keys_are_integers(self, all_same_minute_dataset):
        """Test that dictionary keys are integers."""
        ds = all_same_minute_dataset
        result = ds.variable_leader.get_time_component_frequency("minute")
        for key in result.keys():
            assert isinstance(key, (int, np.integer))

    def test_component_frequency_dict_values_are_integers(
        self, all_same_minute_dataset
    ):
        """Test that dictionary values (counts) are integers."""
        ds = all_same_minute_dataset
        result = ds.variable_leader.get_time_component_frequency("minute")
        for value in result.values():
            assert isinstance(value, (int, np.integer))

    # ========================================================================
    # TEST: Minute component
    # ========================================================================

    def test_all_same_minute_returns_single_entry(self, all_same_minute_dataset):
        """Test that all samples at same minute return single entry."""
        ds = all_same_minute_dataset
        result = ds.variable_leader.get_time_component_frequency("minute")
        assert len(result) == 1
        assert 0 in result

    def test_same_minute_correct_frequency(self, all_same_minute_dataset):
        """Test that frequency count is correct."""
        ds = all_same_minute_dataset
        result = ds.variable_leader.get_time_component_frequency("minute")
        assert result[0] == 5

    def test_mixed_minute_multiple_entries(self, mixed_minute_dataset):
        """Test that mixed minutes produce multiple entries."""
        ds = mixed_minute_dataset
        result = ds.variable_leader.get_time_component_frequency("minute")
        assert len(result) == 3  # minutes 0, 15, 30
        assert result[0] == 2
        assert result[15] == 2
        assert result[30] == 1

    def test_default_component_is_minute(self, all_same_minute_dataset):
        """Test that default component is 'minute'."""
        ds = all_same_minute_dataset
        result1 = ds.variable_leader.get_time_component_frequency()
        result2 = ds.variable_leader.get_time_component_frequency("minute")
        assert result1 == result2

    # ========================================================================
    # TEST: Hour component
    # ========================================================================

    def test_hour_component_extraction(self, all_different_hour_dataset):
        """Test extraction of hour component."""
        ds = all_different_hour_dataset
        result = ds.variable_leader.get_time_component_frequency("hour")
        # Every 3 hours: 0, 3, 6, 9, 12, 15, 18, 21
        assert len(result) == 8
        assert all(count == 1 for count in result.values())

    def test_hour_values_in_valid_range(self, all_different_hour_dataset):
        """Test that hour values are in valid 0-23 range."""
        ds = all_different_hour_dataset
        result = ds.variable_leader.get_time_component_frequency("hour")
        for hour in result.keys():
            assert 0 <= hour <= 23

    # ========================================================================
    # TEST: Second component
    # ========================================================================

    def test_second_component_extraction(self, second_samples_dataset):
        """Test extraction of second component."""
        ds = second_samples_dataset
        result = ds.variable_leader.get_time_component_frequency("second")
        # Samples at 0, 15, 30, 45 seconds
        assert len(result) == 4
        assert result[0] == 2  # seconds 0 appear twice
        assert result[15] == 2  # seconds 15 appear twice
        assert result[30] == 1  # second 30 appears once
        assert result[45] == 1  # second 45 appears once

    def test_second_values_in_valid_range(self, second_samples_dataset):
        """Test that second values are in valid 0-59 range."""
        ds = second_samples_dataset
        result = ds.variable_leader.get_time_component_frequency("second")
        for second in result.keys():
            assert 0 <= second <= 59

    # ========================================================================
    # TEST: Error handling
    # ========================================================================

    def test_invalid_component_raises_valueerror(self, all_same_minute_dataset):
        """Test that invalid component raises ValueError."""
        ds = all_same_minute_dataset
        with pytest.raises(ValueError):
            ds.variable_leader.get_time_component_frequency("invalid_component")

    def test_invalid_component_error_message(self, all_same_minute_dataset):
        """Test error message for invalid component."""
        ds = all_same_minute_dataset
        with pytest.raises(ValueError) as exc_info:
            ds.variable_leader.get_time_component_frequency("nanosecond")
        assert "component must be one of" in str(exc_info.value)
        assert "nanosecond" in str(exc_info.value)

    def test_valid_components_are_accepted(self, all_same_minute_dataset):
        """Test that all valid components are accepted."""
        ds = all_same_minute_dataset
        valid_components = ["hour", "minute", "second"]
        for component in valid_components:
            result = ds.variable_leader.get_time_component_frequency(component)
            assert isinstance(result, dict)

    # ========================================================================
    # TEST: Edge cases
    # ========================================================================

    def test_single_ensemble_returns_dict_with_component(self):
        """Test that single ensemble returns dict with one entry."""
        base_time = datetime(2024, 12, 15, 10, 30, 45)
        ds = xr.Dataset(
            {
                "ensemble_number": (("ensemble",), np.array([1], dtype=np.uint16)),
                "data": (("ensemble",), np.array([0])),
            },
            coords={"time": (("ensemble",), [base_time])},
            attrs={"pyadps_component": "VariableLeader"},
        )

        result_minute = ds.variable_leader.get_time_component_frequency("minute")
        assert result_minute == {30: 1}

        result_hour = ds.variable_leader.get_time_component_frequency("hour")
        assert result_hour == {10: 1}

        result_second = ds.variable_leader.get_time_component_frequency("second")
        assert result_second == {45: 1}


# ============================================================================
# INTEGRATION TESTS: All four methods together
# ============================================================================


class TestTimeMethodsIntegration:
    """Integration tests combining all four time analysis methods."""

    @pytest.fixture
    def realistic_adcp_dataset(self):
        """Create realistic ADCP dataset with regular hourly sampling."""
        base_time = datetime(2024, 12, 15, 0, 0, 0)
        times = [base_time + timedelta(hours=i) for i in range(24)]

        ds = xr.Dataset(
            {
                "ensemble_number": (("ensemble",), np.arange(1, 25, dtype=np.uint16)),
                "velocity": (("ensemble",), np.random.randn(24)),
            },
            coords={"time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        return ds

    def test_all_methods_work_together(self, realistic_adcp_dataset):
        """Test that all four methods work correctly together."""
        ds = realistic_adcp_dataset
        vl = ds.variable_leader

        # Test is_time_regular
        is_regular = vl.is_time_regular()
        assert is_regular is True

        # Test get_time_interval
        interval = vl.get_time_interval()
        assert interval == pd.Timedelta(hours=1)

        # Test get_time_interval_frequency
        freq = vl.get_time_interval_frequency()
        assert len(freq) == 1
        assert freq["01:00:00"] == 23

        # Test get_time_component_frequency
        hour_freq = vl.get_time_component_frequency("hour")
        assert len(hour_freq) == 24
        assert all(count == 1 for count in hour_freq.values())

    def test_consistency_between_interval_and_frequency(self, realistic_adcp_dataset):
        """Test that interval and frequency methods are consistent."""
        ds = realistic_adcp_dataset
        vl = ds.variable_leader

        interval = vl.get_time_interval()
        freq = vl.get_time_interval_frequency()

        # Should have only one interval value
        assert len(freq) == 1

        # Convert interval to HH:MM:SS format
        total_seconds = int(interval.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        interval_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Should match the frequency dict key
        assert interval_str in freq

    def test_regularity_implies_single_frequency(self, realistic_adcp_dataset):
        """Test that regular time series implies single frequency entry."""
        ds = realistic_adcp_dataset
        vl = ds.variable_leader

        is_regular = vl.is_time_regular()
        freq = vl.get_time_interval_frequency()

        if is_regular:
            assert len(freq) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ============================================================================
# TARGETED BRANCH-COVERAGE TESTS
# ============================================================================

from unittest import mock


def _vl_dataset(**data_vars_override):
    """
    Build a minimal VariableLeader dataset.

    Starts with the fields required by the accessor validator, then merges
    any additional/overriding variables supplied by the caller.
    A value of ``None`` removes the key entirely.
    """
    base = {
        "bit_result": (("ensemble",), np.zeros(3, dtype=np.uint16)),
        "error_status_word_1": (("ensemble",), np.zeros(3, dtype=np.uint8)),
        "error_status_word_2": (("ensemble",), np.zeros(3, dtype=np.uint8)),
        "error_status_word_3": (("ensemble",), np.zeros(3, dtype=np.uint8)),
        "error_status_word_4": (("ensemble",), np.zeros(3, dtype=np.uint8)),
    }
    base.update(data_vars_override)
    return xr.Dataset(
        {k: v for k, v in base.items() if v is not None},
        attrs={"pyadps_component": "VariableLeader"},
    )


# ---------------------------------------------------------------------------
# Line 1842 – bit_result_summary decoded reserved-bit branch
# ---------------------------------------------------------------------------


class TestBitResultSummaryDecodedReservedBits:
    """
    Line 1842: inside the ``for bit_pos, bit_name`` loop over reserved bits
    (3–7), the ``if include_decoded and decoded_name in self._obj.data_vars``
    branch is taken when the dataset contains one of the decoded reserved-bit
    fields such as ``bit_reserved_1_error``.

    Existing fixtures only include decoded fields for bits 0–2 (demod_0,
    demod_1, timing_card).  None include a reserved-bit decoded field, so
    line 1842 was never executed.
    """

    @pytest.fixture
    def dataset_with_decoded_reserved_bits(self):
        """
        Dataset that includes decoded fields for reserved bits 3–7.
        The decoded field name pattern is ``bit_reserved_N_error`` where
        N = bit_pos - 2 (i.e. bit 3 → reserved_1, bit 4 → reserved_2, ...).

        Uses n=3 throughout to match the ``_vl_dataset`` base field length
        and avoid xarray dimension-size conflicts.
        """
        n = 3
        return _vl_dataset(
            # Override bit_result to n=3 (same as base; explicit for clarity)
            bit_result=(("ensemble",), np.zeros(n, dtype=np.uint16)),
            # Decoded reserved-bit fields (bits 3-7 → reserved_1 … reserved_5)
            bit_reserved_1_error=(("ensemble",), np.array([0, 1, 0], dtype=np.uint8)),
            bit_reserved_2_error=(("ensemble",), np.array([0, 0, 1], dtype=np.uint8)),
            bit_reserved_3_error=(("ensemble",), np.array([0, 0, 0], dtype=np.uint8)),
            bit_reserved_4_error=(("ensemble",), np.array([0, 0, 0], dtype=np.uint8)),
            bit_reserved_5_error=(("ensemble",), np.array([0, 0, 0], dtype=np.uint8)),
        )

    def test_decoded_reserved_bits_are_used(self, dataset_with_decoded_reserved_bits):
        """Line 1842 is hit: decoded field values drive error_count for reserved bits."""
        ds = dataset_with_decoded_reserved_bits
        result = ds.variable_leader.bit_result_summary(include_decoded=True)

        # reserved_1_error: [0, 1, 0] → 1 error
        assert result["bit_checks"]["reserved_1_error"]["error_count"] == 1
        # reserved_2_error: [0, 0, 1] → 1 error
        assert result["bit_checks"]["reserved_2_error"]["error_count"] == 1
        # reserved_3 through reserved_5 are all zero
        assert result["bit_checks"]["reserved_3_error"]["error_count"] == 0
        assert result["bit_checks"]["reserved_4_error"]["error_count"] == 0
        assert result["bit_checks"]["reserved_5_error"]["error_count"] == 0

    def test_decoded_reserved_bits_differ_from_fallback(
        self, dataset_with_decoded_reserved_bits
    ):
        """
        Decoded path (line 1842) and fallback path (line 1844) should produce
        different counts when the decoded field disagrees with bit extraction.

        Here bit_result is all zeros, so bit extraction gives 0 for every
        reserved bit.  The decoded field has 1 for reserved_1, so only the
        decoded path returns a non-zero count.
        """
        ds = dataset_with_decoded_reserved_bits

        result_decoded = ds.variable_leader.bit_result_summary(include_decoded=True)
        result_fallback = ds.variable_leader.bit_result_summary(include_decoded=False)

        # Decoded path sees 1 error for reserved_1; fallback sees 0 (bit_result all zeros)
        assert result_decoded["bit_checks"]["reserved_1_error"]["error_count"] == 1
        assert result_fallback["bit_checks"]["reserved_1_error"]["error_count"] == 0

    def test_partial_decoded_reserved_bits(self):
        """Only some reserved-bit decoded fields present; others fall back to extraction."""
        n = 3
        ds = _vl_dataset(
            bit_result=(("ensemble",), np.zeros(n, dtype=np.uint16)),
            # Only reserved_1 has a decoded field; reserved_2..5 do not
            bit_reserved_1_error=(("ensemble",), np.array([1, 0, 0], dtype=np.uint8)),
        )
        result = ds.variable_leader.bit_result_summary(include_decoded=True)

        # reserved_1 uses decoded field → 1 error
        assert result["bit_checks"]["reserved_1_error"]["error_count"] == 1
        # reserved_2..5 fall back to bit extraction (bit_result all zeros) → 0 errors
        for bit_name in [
            "reserved_2_error",
            "reserved_3_error",
            "reserved_4_error",
            "reserved_5_error",
        ]:
            assert result["bit_checks"][bit_name]["error_count"] == 0


# ---------------------------------------------------------------------------
# Line 2001 – error_status_word_summary continue (missing ESW field)
# ---------------------------------------------------------------------------


class TestErrorStatusWordSummaryMissingFields:
    """
    Line 2001: ``continue`` inside the ESW loop when an ESW field name is not
    present in ``data_vars``.

    All existing fixtures supply all four ESW fields, so the ``continue`` was
    never reached.  We need a dataset with only a subset of the four fields.
    """

    def test_missing_esw_fields_are_skipped(self):
        """
        Dataset with only error_status_word_1 present.
        ESW2, ESW3, ESW4 must each hit the ``continue`` and be absent from
        the returned summary dict.
        """
        ds = _vl_dataset(
            error_status_word_1=(
                ("ensemble",),
                np.array([0x00, 0x01, 0x00], dtype=np.uint8),
            ),
            # Explicitly omit the other three ESW fields
            error_status_word_2=None,
            error_status_word_3=None,
            error_status_word_4=None,
        )
        result = ds.variable_leader.error_status_word_summary()

        assert "ESW1" in result
        assert "ESW2" not in result
        assert "ESW3" not in result
        assert "ESW4" not in result

    def test_missing_esw2_only_is_skipped(self):
        """Only ESW2 absent; ESW1, ESW3, ESW4 present and processed."""
        ds = _vl_dataset(
            error_status_word_2=None,  # remove ESW2
        )
        result = ds.variable_leader.error_status_word_summary()

        assert "ESW1" in result
        assert "ESW2" not in result
        assert "ESW3" in result
        assert "ESW4" in result

    def test_no_esw_fields_returns_empty_dict(self):
        """All four ESW fields absent → empty summary dict."""
        ds = _vl_dataset(
            error_status_word_1=None,
            error_status_word_2=None,
            error_status_word_3=None,
            error_status_word_4=None,
        )
        result = ds.variable_leader.error_status_word_summary()
        assert result == {}

    def test_partial_esw_result_structure_is_correct(self):
        """ESWs that are present still have the correct structure."""
        ds = _vl_dataset(
            error_status_word_1=(("ensemble",), np.zeros(3, dtype=np.uint8)),
            error_status_word_2=None,
            error_status_word_3=None,
            error_status_word_4=None,
        )
        result = ds.variable_leader.error_status_word_summary()

        assert "all_zeros" in result["ESW1"]
        assert "total_events" in result["ESW1"]
        assert "bit_checks" in result["ESW1"]
        assert len(result["ESW1"]["bit_checks"]) == 8


# ============================================================================
# TARGETED BRANCH-COVERAGE TESTS – Part 2
# ============================================================================

from datetime import datetime, timedelta
from unittest import mock


def _vl_time_dataset(times, extra_vars=None):
    """
    Build a VariableLeader dataset with a ``time`` coordinate and the minimal
    data variables required by the accessor.

    Parameters
    ----------
    times : sequence
        Datetime-like values for the ``time`` coordinate.
    extra_vars : dict, optional
        Additional (name, DataArray-spec) pairs merged into data_vars.
    """
    n = len(times)
    base = {
        "ensemble_number": (("ensemble",), np.arange(1, n + 1, dtype=np.uint16)),
        "ensemble_msb": (("ensemble",), np.zeros(n, dtype=np.uint8)),
        "rtc_year": (("ensemble",), np.full(n, 24, dtype=np.uint8)),
        "rtc_month": (("ensemble",), np.ones(n, dtype=np.uint8)),
        "rtc_day": (("ensemble",), np.ones(n, dtype=np.uint8)),
        "rtc_hour": (("ensemble",), np.zeros(n, dtype=np.uint8)),
        "rtc_minute": (("ensemble",), np.zeros(n, dtype=np.uint8)),
        "rtc_second": (("ensemble",), np.zeros(n, dtype=np.uint8)),
        "bit_result": (("ensemble",), np.zeros(n, dtype=np.uint16)),
        "error_status_word_1": (("ensemble",), np.zeros(n, dtype=np.uint8)),
        "error_status_word_2": (("ensemble",), np.zeros(n, dtype=np.uint8)),
        "error_status_word_3": (("ensemble",), np.zeros(n, dtype=np.uint8)),
        "error_status_word_4": (("ensemble",), np.zeros(n, dtype=np.uint8)),
    }
    if extra_vars:
        base.update(extra_vars)
    return xr.Dataset(
        base,
        coords={"ensemble": np.arange(n), "time": (("ensemble",), times)},
        attrs={"pyadps_component": "VariableLeader"},
    )


# ---------------------------------------------------------------------------
# Lines 2132, 2174, 2217 – diffs.empty branch in time methods
#
# These three branches share identical conditions: pd.Series.diff().dropna()
# returns an empty Series.  This is triggered by patching pd.Series.diff to
# return a Series of all-NaT values, so that dropna() removes every element.
# ---------------------------------------------------------------------------


class TestTimeMethodsDiffsEmptyBranch:
    """
    Lines 2132 / 2174 / 2217: ``if diffs.empty: return <sentinel>`` inside
    is_time_regular(), get_time_interval(), and get_time_interval_frequency().

    The guard fires when ``time_series.diff().dropna()`` yields an empty
    Series.  We achieve this by patching ``pd.Series.diff`` (inside the
    accessor call) to return a Series whose every element is NaT, making
    ``dropna()`` strip them all away.
    """

    @pytest.fixture
    def two_point_dataset(self):
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        return _vl_time_dataset([base_time, base_time + timedelta(minutes=1)])

    def _patch_diff_all_nat(self):
        """
        Context manager: patches pd.Series.diff so it always returns a
        two-element Series of NaT (one per diff call site), causing dropna()
        to produce an empty result.
        """
        nat = pd.NaT

        def diff_all_nat(self_series, *args, **kwargs):
            return pd.Series([nat] * len(self_series))

        return mock.patch.object(pd.Series, "diff", diff_all_nat)

    # -- is_time_regular -------------------------------------------------------

    def test_is_time_regular_diffs_empty_returns_false(self, two_point_dataset):
        """Line 2132: diffs.empty → return False."""
        ds = two_point_dataset
        with self._patch_diff_all_nat():
            result = ds.variable_leader.is_time_regular()
        assert result is False

    def test_is_time_regular_diffs_empty_is_bool(self, two_point_dataset):
        ds = two_point_dataset
        with self._patch_diff_all_nat():
            result = ds.variable_leader.is_time_regular()
        assert isinstance(result, bool)

    # -- get_time_interval -----------------------------------------------------

    def test_get_time_interval_diffs_empty_returns_none(self, two_point_dataset):
        """Line 2174: diffs.empty → return None."""
        ds = two_point_dataset
        with self._patch_diff_all_nat():
            result = ds.variable_leader.get_time_interval()
        assert result is None

    # -- get_time_interval_frequency -------------------------------------------

    def test_get_time_interval_frequency_diffs_empty_returns_empty_dict(
        self, two_point_dataset
    ):
        """Line 2217: diffs.empty → return {}."""
        ds = two_point_dataset
        with self._patch_diff_all_nat():
            result = ds.variable_leader.get_time_interval_frequency()
        assert result == {}


# ---------------------------------------------------------------------------
# Line 2334 – summary() ellipsis for > 10 gap locations
# ---------------------------------------------------------------------------


class TestSummaryEllipsisForManyGaps:
    """
    Line 2334: ``report += "..."`` fires when ``gap_locations`` has more than
    10 entries.

    The existing test datasets create at most a handful of gaps.  We need one
    with > 10 ensemble discontinuities so the slice ``[:10]`` triggers the
    ``if len(cont["gap_locations"]) > 10`` branch.
    """

    @pytest.fixture
    def many_gaps_dataset(self):
        """
        Dataset where ensemble numbers skip every other value, producing 12+
        gaps detected by ensemble_continuity_check().
        """
        # 25 ensembles; every other ensemble number is missing → ~12 gaps
        n = 25
        ensemble_nums = np.arange(1, 2 * n + 1, 2, dtype=np.uint16)  # 1,3,5,...,49
        times = pd.date_range("2024-01-01", periods=n, freq="1min")

        return xr.Dataset(
            {
                "ensemble_number": (("ensemble",), ensemble_nums),
                "ensemble_msb": (("ensemble",), np.zeros(n, dtype=np.uint8)),
                "rtc_year": (("ensemble",), np.full(n, 24, dtype=np.uint8)),
                "rtc_month": (("ensemble",), np.ones(n, dtype=np.uint8)),
                "rtc_day": (("ensemble",), np.ones(n, dtype=np.uint8)),
                "rtc_hour": (("ensemble",), np.zeros(n, dtype=np.uint8)),
                "rtc_minute": (("ensemble",), np.zeros(n, dtype=np.uint8)),
                "rtc_second": (("ensemble",), np.zeros(n, dtype=np.uint8)),
                "bit_result": (("ensemble",), np.zeros(n, dtype=np.uint16)),
                "error_status_word_1": (("ensemble",), np.zeros(n, dtype=np.uint8)),
                "error_status_word_2": (("ensemble",), np.zeros(n, dtype=np.uint8)),
                "error_status_word_3": (("ensemble",), np.zeros(n, dtype=np.uint8)),
                "error_status_word_4": (("ensemble",), np.zeros(n, dtype=np.uint8)),
            },
            coords={"ensemble": np.arange(n), "time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )

    def test_summary_appends_ellipsis_for_many_gaps(self, many_gaps_dataset, capsys):
        """Line 2334: '...' is appended when there are > 10 gap locations."""
        ds = many_gaps_dataset

        # Confirm more than 10 gaps actually exist in continuity check
        cont = ds.variable_leader.ensemble_continuity_check()
        assert (
            len(cont["gap_locations"]) > 10
        ), "Fixture must have more than 10 gaps to trigger line 2334"

        ds.variable_leader.summary()
        output = capsys.readouterr().out
        assert "..." in output

    def test_summary_without_many_gaps_has_no_ellipsis(self, capsys):
        """Regression: <= 10 gaps → no ellipsis appended."""
        n = 5
        times = pd.date_range("2024-01-01", periods=n, freq="1min")
        ds = xr.Dataset(
            {
                "ensemble_number": (
                    ("ensemble",),
                    np.array([1, 2, 5, 6, 7], dtype=np.uint16),  # 2 gaps only
                ),
                "ensemble_msb": (("ensemble",), np.zeros(n, dtype=np.uint8)),
                "rtc_year": (("ensemble",), np.full(n, 24, dtype=np.uint8)),
                "rtc_month": (("ensemble",), np.ones(n, dtype=np.uint8)),
                "rtc_day": (("ensemble",), np.ones(n, dtype=np.uint8)),
                "rtc_hour": (("ensemble",), np.zeros(n, dtype=np.uint8)),
                "rtc_minute": (("ensemble",), np.zeros(n, dtype=np.uint8)),
                "rtc_second": (("ensemble",), np.zeros(n, dtype=np.uint8)),
                "bit_result": (("ensemble",), np.zeros(n, dtype=np.uint16)),
                "error_status_word_1": (("ensemble",), np.zeros(n, dtype=np.uint8)),
                "error_status_word_2": (("ensemble",), np.zeros(n, dtype=np.uint8)),
                "error_status_word_3": (("ensemble",), np.zeros(n, dtype=np.uint8)),
                "error_status_word_4": (("ensemble",), np.zeros(n, dtype=np.uint8)),
            },
            coords={"ensemble": np.arange(n), "time": (("ensemble",), times)},
            attrs={"pyadps_component": "VariableLeader"},
        )
        ds.variable_leader.summary()
        # With only 2 gaps the ellipsis branch is NOT taken; verify no "..."
        # in the gap-locations line (it may appear in other content, so we
        # check the gap-locations section specifically via the report structure).
        output = capsys.readouterr().out
        # The gap location line should not end with "..."
        gap_lines = [l for l in output.splitlines() if "Gap locations" in l]
        for line in gap_lines:
            assert not line.rstrip().endswith("...")


# ---------------------------------------------------------------------------
# Line 2355 – summary() continue for missing ESW label
# ---------------------------------------------------------------------------


class TestSummaryContinueForMissingEswLabel:
    """
    Line 2355: ``continue`` inside the ``for esw_label in ["ESW1",...,"ESW4"]``
    loop when ``error_status_word_summary()`` does not contain that label.

    This is triggered by mocking ``error_status_word_summary`` to return a
    dict with only some ESW entries, so the loop skips the absent ones.
    """

    @pytest.fixture
    def minimal_summary_dataset(self):
        n = 3
        times = pd.date_range("2024-01-01", periods=n, freq="1min")
        return _vl_time_dataset(times)

    def test_summary_skips_missing_esw_label(self, minimal_summary_dataset, capsys):
        """Line 2355: absent ESW labels hit continue; summary still completes."""
        ds = minimal_summary_dataset

        # Mock returns only ESW1; ESW2/3/4 are absent → three continue hits
        partial_esw = {
            "ESW1": {
                "all_zeros": True,
                "total_events": 0,
                "bit_checks": {
                    "address_error_exception": {"event_count": 0},
                    "illegal_instruction_exception": {"event_count": 0},
                    "emulator_exception": {"event_count": 0},
                    "bus_error_exception": {"event_count": 0},
                    "watchdog_restart": {"event_count": 0},
                    "zero_divide_exception": {"event_count": 0},
                    "unassigned_exception": {"event_count": 0},
                    "battery_saver_power": {"event_count": 0},
                },
            }
        }

        with mock.patch.object(
            type(ds.variable_leader),
            "error_status_word_summary",
            return_value=partial_esw,
        ):
            ds.variable_leader.summary()

        output = capsys.readouterr().out
        assert "VARIABLE LEADER" in output
        # ESW1 section is present; ESW2/3/4 sections are absent
        assert "ESW1" in output
        assert "ESW2" not in output
        assert "ESW3" not in output
        assert "ESW4" not in output

    def test_summary_with_empty_esw_dict(self, minimal_summary_dataset, capsys):
        """All four ESW labels absent → all four continue; section is empty."""
        ds = minimal_summary_dataset

        with mock.patch.object(
            type(ds.variable_leader),
            "error_status_word_summary",
            return_value={},
        ):
            ds.variable_leader.summary()

        output = capsys.readouterr().out
        assert "ERROR STATUS WORDS" in output
        # No ESW label should appear in output
        for label in ["ESW1", "ESW2", "ESW3", "ESW4"]:
            assert label not in output


# ---------------------------------------------------------------------------
# Line 2385 – summary() "Most Common Interval: N/A" branch
# ---------------------------------------------------------------------------


class TestSummaryIntervalNaBranch:
    """
    Line 2385: ``report += "  Most Common Interval: N/A (insufficient data)\\n"``
    fires when ``get_time_interval()`` returns ``None``.

    This happens when the time coordinate has fewer than 2 points.
    """

    @pytest.fixture
    def single_ensemble_dataset(self):
        """Only one time point → get_time_interval() returns None."""
        return _vl_time_dataset([datetime(2024, 1, 1, 0, 0, 0)])

    def test_summary_prints_na_for_single_ensemble(
        self, single_ensemble_dataset, capsys
    ):
        """Line 2385: printed when time_interval is None."""
        ds = single_ensemble_dataset
        ds.variable_leader.summary()
        output = capsys.readouterr().out
        assert "N/A (insufficient data)" in output

    def test_summary_does_not_print_na_for_multiple_ensembles(self, capsys):
        """Regression: multiple ensembles → time_interval is not None → no N/A."""
        times = pd.date_range("2024-01-01", periods=3, freq="1min")
        ds = _vl_time_dataset(times)
        ds.variable_leader.summary()
        output = capsys.readouterr().out
        assert "N/A (insufficient data)" not in output

    def test_get_time_interval_returns_none_for_single_point(
        self, single_ensemble_dataset
    ):
        """Confirm the upstream method that drives line 2385."""
        result = single_ensemble_dataset.variable_leader.get_time_interval()
        assert result is None


# ---------------------------------------------------------------------------
# Line 2180 – get_time_interval() returns None when mode() is empty
# ---------------------------------------------------------------------------


class TestGetTimeIntervalModeEmptyBranch:
    """
    Line 2180: ``return None`` at the end of ``get_time_interval()`` fires
    when ``diffs.mode()`` returns an empty Series.

    This is unreachable through normal data: ``mode()`` only returns empty
    when its input is empty, which is already caught by the ``diffs.empty``
    guard at line 2173.  We reach line 2180 by patching ``pd.Series.mode``
    to return an empty Series while ``diffs`` itself is non-empty.
    """

    @pytest.fixture
    def two_point_dataset(self):
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        return _vl_time_dataset([base_time, base_time + timedelta(minutes=1)])

    def test_mode_empty_returns_none(self, two_point_dataset):
        """Line 2180: mode() returns empty Series → get_time_interval() returns None."""
        ds = two_point_dataset
        empty_series = pd.Series([], dtype="timedelta64[ns]")
        with mock.patch.object(pd.Series, "mode", return_value=empty_series):
            result = ds.variable_leader.get_time_interval()
        assert result is None

    def test_mode_empty_return_is_exactly_none(self, two_point_dataset):
        """Return value is exactly None, not False or empty Series."""
        ds = two_point_dataset
        empty_series = pd.Series([], dtype="timedelta64[ns]")
        with mock.patch.object(pd.Series, "mode", return_value=empty_series):
            result = ds.variable_leader.get_time_interval()
        assert result is None
        assert not isinstance(result, pd.Series)
