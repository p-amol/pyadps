"""
Tests for the percent-good threshold advisor in signal_quality.py.

Covers:
- _load_adcp_coefficients
- _extract_frequency
- _extract_fl_params
- compute_percent_good_threshold
- StdDevResult dataclass
- ProcessedDataset.get_percent_good_threshold
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pyadps.processing.signal_quality import (
    StdDevResult,
    _extract_fl_params,
    _extract_frequency,
    _load_adcp_coefficients,
    compute_percent_good_threshold,
)


# ============================================================================
# FIXTURES
# ============================================================================


def _make_fl_dataset(
    frequency="300 kHz",
    depth_cell_length_cm=100,
    num_cells=30,
    pings_per_ensemble=100,
    n_ens=5,
    decoded=True,
):
    """Build a minimal Fixed Leader dataset."""
    data_vars = {
        "depth_cell_length": (
            "ensemble",
            np.full(n_ens, depth_cell_length_cm, dtype=np.uint16),
        ),
        "num_cells": (
            "ensemble",
            np.full(n_ens, num_cells, dtype=np.uint8),
        ),
        "pings_per_ensemble": (
            "ensemble",
            np.full(n_ens, pings_per_ensemble, dtype=np.uint16),
        ),
    }
    if decoded:
        data_vars["frequency"] = (
            "ensemble",
            np.array([frequency] * n_ens, dtype=object),
        )
    return xr.Dataset(data_vars, coords={"ensemble": np.arange(n_ens)})


def _syscode_for_freq(freq_kHz: int) -> int:
    """Return a system_configuration_code with the correct frequency bits.

    bits[13:16] in format(syscode,"016b") encodes frequency:
      75→"000", 150→"001", 300→"010", 600→"011", 1200→"100", 2400→"101"
    Those are bits 2-0 of the integer.
    """
    mapping = {75: 0, 150: 1, 300: 2, 600: 3, 1200: 4, 2400: 5}
    return mapping[freq_kHz]


@pytest.fixture
def fl_300khz_1m():
    """300 kHz, 1.0 m bins, 30 cells, 100 pings — standard test dataset."""
    return _make_fl_dataset(
        frequency="300 kHz",
        depth_cell_length_cm=100,
        num_cells=30,
        pings_per_ensemble=100,
    )


@pytest.fixture
def fl_600khz_05m():
    """600 kHz, 0.5 m bins, 20 cells, 100 pings."""
    return _make_fl_dataset(
        frequency="600 kHz",
        depth_cell_length_cm=50,
        num_cells=20,
        pings_per_ensemble=100,
    )


@pytest.fixture
def fl_no_decoded():
    """Dataset with system_configuration_code instead of decoded frequency (300 kHz)."""
    syscode = _syscode_for_freq(300)
    return xr.Dataset(
        {
            "system_configuration_code": (
                "ensemble",
                np.full(5, syscode, dtype=np.uint16),
            ),
            "depth_cell_length": (
                "ensemble",
                np.full(5, 100, dtype=np.uint16),
            ),
            "num_cells": ("ensemble", np.full(5, 30, dtype=np.uint8)),
            "pings_per_ensemble": ("ensemble", np.full(5, 100, dtype=np.uint16)),
        }
    )


# ============================================================================
# TESTS: _load_adcp_coefficients
# ============================================================================


class TestLoadAdcpCoefficients:
    def test_loads_from_package(self):
        coeffs = _load_adcp_coefficients()
        assert isinstance(coeffs, dict)
        assert len(coeffs) > 0

    def test_package_contains_expected_frequencies(self):
        coeffs = _load_adcp_coefficients()
        expected = {75, 150, 300, 600, 1200, 2400}
        assert {int(k) for k in coeffs} == expected

    def test_each_entry_has_abc_keys(self):
        coeffs = _load_adcp_coefficients()
        for freq, bins in coeffs.items():
            for bsize, entry in bins.items():
                assert "a" in entry, f"missing 'a' for freq={freq}, bin={bsize}"
                assert "b" in entry, f"missing 'b' for freq={freq}, bin={bsize}"
                assert "c" in entry, f"missing 'c' for freq={freq}, bin={bsize}"

    def test_loads_from_custom_path(self, tmp_path):
        data = {"300": {"1.0": {"a": 0.1, "b": 0.05, "c": 5.0, "r_squared": 0.99}}}
        p = tmp_path / "custom_coeffs.json"
        p.write_text(json.dumps(data))
        coeffs = _load_adcp_coefficients(str(p))
        assert "300" in coeffs

    def test_raises_for_missing_custom_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            _load_adcp_coefficients(str(tmp_path / "nonexistent.json"))


# ============================================================================
# TESTS: _extract_frequency
# ============================================================================


class TestExtractFrequency:
    def test_decoded_field_300khz(self):
        ds = _make_fl_dataset(frequency="300 kHz")
        assert _extract_frequency(ds) == 300

    def test_decoded_field_600khz(self):
        ds = _make_fl_dataset(frequency="600 kHz")
        assert _extract_frequency(ds) == 600

    def test_decoded_field_75khz(self):
        ds = _make_fl_dataset(frequency="75 kHz")
        assert _extract_frequency(ds) == 75

    def test_decoded_field_1200khz(self):
        ds = _make_fl_dataset(frequency="1200 kHz")
        assert _extract_frequency(ds) == 1200

    def test_decoded_field_2400khz(self):
        ds = _make_fl_dataset(frequency="2400 kHz")
        assert _extract_frequency(ds) == 2400

    def test_fallback_to_syscode_300khz(self, fl_no_decoded):
        assert _extract_frequency(fl_no_decoded) == 300

    def test_fallback_emits_warning(self, fl_no_decoded, caplog):
        import logging

        with caplog.at_level(
            logging.WARNING, logger="pyadps.processing.signal_quality"
        ):
            _extract_frequency(fl_no_decoded)
        assert any("system_configuration_code" in m for m in caplog.messages)

    @pytest.mark.parametrize("freq_kHz", [75, 150, 300, 600, 1200, 2400])
    def test_fallback_all_valid_frequencies(self, freq_kHz):
        syscode = _syscode_for_freq(freq_kHz)
        ds = xr.Dataset(
            {
                "system_configuration_code": (
                    "ensemble",
                    np.full(3, syscode, dtype=np.uint16),
                )
            }
        )
        assert _extract_frequency(ds) == freq_kHz

    def test_raises_when_both_fields_absent(self):
        ds = xr.Dataset(
            {"depth_cell_length": ("ensemble", np.array([100], dtype=np.uint16))}
        )
        with pytest.raises(ValueError, match="absent"):
            _extract_frequency(ds)

    def test_raises_for_unknown_bit_code(self):
        # bit code "111" (=7) is not in _FREQ_BIT_MAP
        syscode = 0b111  # bits 0-2 = 111
        ds = xr.Dataset(
            {
                "system_configuration_code": (
                    "ensemble",
                    np.full(3, syscode, dtype=np.uint16),
                )
            }
        )
        with pytest.raises(ValueError, match="Unknown frequency bit-code"):
            _extract_frequency(ds)

    def test_uses_most_common_value(self):
        # Majority 300 kHz (code=2), minority 600 kHz (code=3)
        freqs = np.array(
            ["300 kHz", "300 kHz", "300 kHz", "600 kHz"], dtype=object
        )
        ds = xr.Dataset({"frequency": ("ensemble", freqs)})
        assert _extract_frequency(ds) == 300


# ============================================================================
# TESTS: _extract_fl_params
# ============================================================================


class TestExtractFlParams:
    def test_returns_all_keys(self, fl_300khz_1m):
        params = _extract_fl_params(fl_300khz_1m)
        assert set(params) == {
            "frequency",
            "bin_size",
            "num_cells",
            "pings_per_ensemble",
            "depth_range",
        }

    def test_frequency_correct(self, fl_300khz_1m):
        assert _extract_fl_params(fl_300khz_1m)["frequency"] == 300

    def test_bin_size_cm_to_m_conversion(self, fl_300khz_1m):
        # depth_cell_length=100 cm → bin_size=1.0 m
        assert _extract_fl_params(fl_300khz_1m)["bin_size"] == pytest.approx(1.0)

    def test_bin_size_50cm(self):
        ds = _make_fl_dataset(depth_cell_length_cm=50)
        assert _extract_fl_params(ds)["bin_size"] == pytest.approx(0.5)

    def test_num_cells(self, fl_300khz_1m):
        assert _extract_fl_params(fl_300khz_1m)["num_cells"] == 30

    def test_pings_per_ensemble(self, fl_300khz_1m):
        assert _extract_fl_params(fl_300khz_1m)["pings_per_ensemble"] == 100

    def test_depth_range_equals_bin_times_cells(self, fl_300khz_1m):
        params = _extract_fl_params(fl_300khz_1m)
        assert params["depth_range"] == pytest.approx(
            params["bin_size"] * params["num_cells"]
        )

    def test_depth_range_value(self, fl_300khz_1m):
        # 1.0 m * 30 cells = 30.0 m
        assert _extract_fl_params(fl_300khz_1m)["depth_range"] == pytest.approx(30.0)

    def test_most_common_used_for_varying_values(self):
        # 4 out of 5 ensembles have 30 cells; one has 25
        cells = np.array([30, 30, 25, 30, 30], dtype=np.uint8)
        ds = _make_fl_dataset(num_cells=30)
        ds["num_cells"] = ("ensemble", cells)
        assert _extract_fl_params(ds)["num_cells"] == 30

    def test_raises_for_missing_field(self, fl_300khz_1m):
        ds = fl_300khz_1m.drop_vars("num_cells")
        with pytest.raises(KeyError, match="num_cells"):
            _extract_fl_params(ds)

    def test_raises_for_missing_depth_cell_length(self, fl_300khz_1m):
        ds = fl_300khz_1m.drop_vars("depth_cell_length")
        with pytest.raises(KeyError, match="depth_cell_length"):
            _extract_fl_params(ds)

    def test_raises_for_missing_pings(self, fl_300khz_1m):
        ds = fl_300khz_1m.drop_vars("pings_per_ensemble")
        with pytest.raises(KeyError, match="pings_per_ensemble"):
            _extract_fl_params(ds)


# ============================================================================
# TESTS: compute_percent_good_threshold — normal cases
# ============================================================================

# Pre-computed reference values (300 kHz, 1.0 m bin, 30 m range):
#   a=0.03155, b=0.051195, c=7.224023
#   sigma_single = 0.03155 * exp(0.051195*30) + 7.224023 ≈ 7.3706 cm/s
_REF_SIGMA_SINGLE = 7.370581
_REF_SIGMA_ENS_100 = _REF_SIGMA_SINGLE / math.sqrt(100)


class TestComputePercentGoodThreshold:
    def test_returns_std_dev_result(self, fl_300khz_1m):
        result = compute_percent_good_threshold(fl_300khz_1m, desired_std=1.0)
        assert isinstance(result, StdDevResult)

    def test_single_ping_std_value(self, fl_300khz_1m):
        result = compute_percent_good_threshold(fl_300khz_1m, desired_std=1.0)
        assert result.single_ping_std == pytest.approx(_REF_SIGMA_SINGLE, rel=1e-4)

    def test_ensemble_std_equals_single_over_sqrt_pings(self, fl_300khz_1m):
        result = compute_percent_good_threshold(fl_300khz_1m, desired_std=1.0)
        expected = result.single_ping_std / math.sqrt(result.pings_per_ensemble)
        assert result.ensemble_std == pytest.approx(expected, rel=1e-9)

    def test_achievable_case(self, fl_300khz_1m):
        # desired_std=1.0, sigma_single≈7.37 → n_valid=55, 55<=100 → achievable
        result = compute_percent_good_threshold(fl_300khz_1m, desired_std=1.0)
        assert result.achievable is True
        assert result.valid_pings_required == 55
        assert result.percent_good_cutoff == pytest.approx(55.0)

    def test_unachievable_case(self, fl_300khz_1m):
        # desired_std=0.5 → n_valid=218, 218>100 → not achievable
        result = compute_percent_good_threshold(fl_300khz_1m, desired_std=0.5)
        assert result.achievable is False
        assert result.valid_pings_required == 100  # clamped to n_total
        assert result.percent_good_cutoff == pytest.approx(100.0)

    def test_unachievable_emits_warning(self, fl_300khz_1m, caplog):
        import logging

        with caplog.at_level(
            logging.WARNING, logger="pyadps.processing.signal_quality"
        ):
            compute_percent_good_threshold(fl_300khz_1m, desired_std=0.5)
        assert any("valid pings" in m for m in caplog.messages)

    def test_achievable_boundary(self, fl_300khz_1m):
        # desired_std just above ensemble_std → just achievable
        slightly_above = _REF_SIGMA_ENS_100 + 0.01
        result = compute_percent_good_threshold(fl_300khz_1m, desired_std=slightly_above)
        assert result.achievable is True

    def test_output_frequency(self, fl_300khz_1m):
        result = compute_percent_good_threshold(fl_300khz_1m, desired_std=1.0)
        assert result.frequency == 300

    def test_output_bin_size(self, fl_300khz_1m):
        result = compute_percent_good_threshold(fl_300khz_1m, desired_std=1.0)
        assert result.bin_size == pytest.approx(1.0)

    def test_output_depth_range(self, fl_300khz_1m):
        result = compute_percent_good_threshold(fl_300khz_1m, desired_std=1.0)
        assert result.depth_range == pytest.approx(30.0)

    def test_output_pings_per_ensemble(self, fl_300khz_1m):
        result = compute_percent_good_threshold(fl_300khz_1m, desired_std=1.0)
        assert result.pings_per_ensemble == 100

    def test_output_desired_std(self, fl_300khz_1m):
        result = compute_percent_good_threshold(fl_300khz_1m, desired_std=1.0)
        assert result.desired_std == pytest.approx(1.0)

    def test_600khz_05m_bin(self, fl_600khz_05m):
        # 600 kHz, 0.5 m bin, depth_range=10 m, 100 pings, desired=1.0
        # sigma_single ≈ 7.4485, n_valid=56, pg=56%
        result = compute_percent_good_threshold(fl_600khz_05m, desired_std=1.0)
        assert result.frequency == 600
        assert result.bin_size == pytest.approx(0.5)
        assert result.depth_range == pytest.approx(10.0)
        assert result.valid_pings_required == 56
        assert result.percent_good_cutoff == pytest.approx(56.0)
        assert result.achievable is True

    def test_percent_good_range(self, fl_300khz_1m):
        result = compute_percent_good_threshold(fl_300khz_1m, desired_std=1.0)
        assert 0.0 <= result.percent_good_cutoff <= 100.0


# ============================================================================
# TESTS: compute_percent_good_threshold — overrides
# ============================================================================


class TestComputeOverrides:
    def test_frequency_override(self, fl_300khz_1m):
        # Override to 600 kHz (which has different coefficients)
        r300 = compute_percent_good_threshold(fl_300khz_1m, desired_std=1.0)
        r600 = compute_percent_good_threshold(
            fl_300khz_1m, desired_std=1.0, frequency=600
        )
        assert r600.frequency == 600
        assert r600.single_ping_std != pytest.approx(r300.single_ping_std, rel=0.01)

    def test_bin_size_override(self, fl_300khz_1m):
        result = compute_percent_good_threshold(
            fl_300khz_1m, desired_std=1.0, bin_size=0.5
        )
        assert result.bin_size == pytest.approx(0.5)

    def test_depth_range_override_shorter(self, fl_300khz_1m):
        # Shorter range → lower single_ping_std (curve increases with depth)
        r_full = compute_percent_good_threshold(fl_300khz_1m, desired_std=1.0)
        r_short = compute_percent_good_threshold(
            fl_300khz_1m, desired_std=1.0, depth_range=10.0
        )
        assert r_short.depth_range == pytest.approx(10.0)
        assert r_short.single_ping_std < r_full.single_ping_std

    def test_depth_range_override_longer(self, fl_300khz_1m):
        r_full = compute_percent_good_threshold(fl_300khz_1m, desired_std=1.0)
        r_long = compute_percent_good_threshold(
            fl_300khz_1m, desired_std=1.0, depth_range=60.0
        )
        assert r_long.single_ping_std > r_full.single_ping_std

    def test_n_pings_override_makes_achievable(self, fl_300khz_1m):
        # With 40 pings, desired_std=1.0 is unachievable (n_valid=55 > 40)
        r40 = compute_percent_good_threshold(
            fl_300khz_1m, desired_std=1.0, n_pings=40
        )
        assert r40.achievable is False

        # With 200 pings it becomes achievable
        r200 = compute_percent_good_threshold(
            fl_300khz_1m, desired_std=1.0, n_pings=200
        )
        assert r200.achievable is True
        assert r200.pings_per_ensemble == 200

    def test_all_overrides_together(self, fl_300khz_1m):
        result = compute_percent_good_threshold(
            fl_300khz_1m,
            desired_std=1.0,
            frequency=600,
            bin_size=0.5,
            depth_range=10.0,
            n_pings=80,
        )
        assert result.frequency == 600
        assert result.bin_size == pytest.approx(0.5)
        assert result.depth_range == pytest.approx(10.0)
        assert result.pings_per_ensemble == 80

    def test_fallback_dataset_uses_most_common(self):
        # Ensure override does not affect most-common extraction of other fields
        ds = _make_fl_dataset(
            frequency="300 kHz",
            depth_cell_length_cm=100,
            num_cells=30,
            pings_per_ensemble=100,
        )
        result = compute_percent_good_threshold(ds, desired_std=1.0, depth_range=15.0)
        assert result.depth_range == pytest.approx(15.0)
        assert result.bin_size == pytest.approx(1.0)  # still from dataset


# ============================================================================
# TESTS: compute_percent_good_threshold — error cases
# ============================================================================


class TestComputeErrors:
    def test_raises_for_unsupported_frequency(self, fl_300khz_1m):
        with pytest.raises(ValueError, match="38 kHz"):
            compute_percent_good_threshold(
                fl_300khz_1m, desired_std=1.0, frequency=38
            )

    def test_error_message_lists_valid_frequencies(self, fl_300khz_1m):
        with pytest.raises(ValueError) as exc_info:
            compute_percent_good_threshold(
                fl_300khz_1m, desired_std=1.0, frequency=400
            )
        msg = str(exc_info.value)
        for freq in [75, 150, 300, 600, 1200, 2400]:
            assert str(freq) in msg

    def test_raises_for_unsupported_bin_size(self, fl_300khz_1m):
        with pytest.raises(ValueError, match="Bin size 3.0"):
            compute_percent_good_threshold(
                fl_300khz_1m, desired_std=1.0, bin_size=3.0
            )

    def test_error_message_lists_valid_bin_sizes_for_frequency(self, fl_300khz_1m):
        with pytest.raises(ValueError) as exc_info:
            compute_percent_good_threshold(
                fl_300khz_1m, desired_std=1.0, bin_size=3.0
            )
        msg = str(exc_info.value)
        # Valid bins for 300 kHz: 0.5, 1.0, 2.0, 4.0
        for b in [0.5, 1.0, 2.0, 4.0]:
            assert str(b) in msg

    def test_raises_for_non_positive_desired_std(self, fl_300khz_1m):
        with pytest.raises(ValueError, match="positive"):
            compute_percent_good_threshold(fl_300khz_1m, desired_std=0.0)

    def test_raises_for_negative_desired_std(self, fl_300khz_1m):
        with pytest.raises(ValueError, match="positive"):
            compute_percent_good_threshold(fl_300khz_1m, desired_std=-1.0)

    def test_raises_when_frequency_field_absent_and_no_syscode(self, fl_300khz_1m):
        ds = fl_300khz_1m.drop_vars("frequency")
        # No system_configuration_code either → should raise
        with pytest.raises(ValueError, match="absent"):
            compute_percent_good_threshold(ds, desired_std=1.0)

    def test_valid_bin_size_for_different_frequency(self):
        # 75 kHz valid bins: 2.0, 4.0, 8.0, 16.0 — try an invalid one
        ds = _make_fl_dataset(
            frequency="75 kHz",
            depth_cell_length_cm=50,  # 0.5 m — not valid for 75 kHz
            num_cells=20,
            pings_per_ensemble=50,
        )
        with pytest.raises(ValueError, match="Bin size 0.5"):
            compute_percent_good_threshold(ds, desired_std=1.0)

    def test_valid_bin_size_for_75khz_succeeds(self):
        ds = _make_fl_dataset(
            frequency="75 kHz",
            depth_cell_length_cm=800,  # 8.0 m — valid for 75 kHz
            num_cells=10,
            pings_per_ensemble=50,
        )
        result = compute_percent_good_threshold(ds, desired_std=5.0)
        assert result.frequency == 75
        assert result.bin_size == pytest.approx(8.0)


# ============================================================================
# TESTS: StdDevResult dataclass
# ============================================================================


class TestStdDevResult:
    @pytest.fixture
    def sample_result(self, fl_300khz_1m):
        return compute_percent_good_threshold(fl_300khz_1m, desired_std=1.0)

    def test_field_types(self, sample_result):
        assert isinstance(sample_result.frequency, int)
        assert isinstance(sample_result.bin_size, float)
        assert isinstance(sample_result.depth_range, float)
        assert isinstance(sample_result.pings_per_ensemble, int)
        assert isinstance(sample_result.single_ping_std, float)
        assert isinstance(sample_result.ensemble_std, float)
        assert isinstance(sample_result.desired_std, float)
        assert isinstance(sample_result.valid_pings_required, int)
        assert isinstance(sample_result.percent_good_cutoff, float)
        assert isinstance(sample_result.achievable, bool)

    def test_str_contains_frequency(self, sample_result):
        assert "300" in str(sample_result)

    def test_str_contains_percent_good(self, sample_result):
        assert "%" in str(sample_result)

    def test_str_achievable_shows_status(self):
        ds = _make_fl_dataset()
        r_ok = compute_percent_good_threshold(ds, desired_std=1.0)
        r_bad = compute_percent_good_threshold(ds, desired_std=0.1)
        assert "achievable" in str(r_ok)
        assert "NOT achievable" in str(r_bad)

    def test_valid_pings_never_exceeds_total(self, fl_300khz_1m):
        result = compute_percent_good_threshold(fl_300khz_1m, desired_std=0.01)
        assert result.valid_pings_required <= result.pings_per_ensemble

    def test_percent_good_always_between_0_and_100(self, fl_300khz_1m):
        for d in [0.01, 0.5, 1.0, 5.0, 10.0]:
            result = compute_percent_good_threshold(fl_300khz_1m, desired_std=d)
            assert 0.0 <= result.percent_good_cutoff <= 100.0

    def test_larger_desired_std_gives_lower_pg_cutoff(self, fl_300khz_1m):
        r_tight = compute_percent_good_threshold(fl_300khz_1m, desired_std=1.0)
        r_loose = compute_percent_good_threshold(fl_300khz_1m, desired_std=2.0)
        assert r_tight.percent_good_cutoff >= r_loose.percent_good_cutoff


# ============================================================================
# TESTS: ProcessedDataset.get_percent_good_threshold
# ============================================================================


class TestProcessedDatasetIntegration:
    @pytest.fixture
    def proc_dataset(self):
        """ProcessedDataset built from a full-shape dataset that includes FL fields."""
        from pyadps.processing.core import ProcessedDataset

        n_time, n_cell, n_beam = 10, 30, 4
        times = pd.date_range("2024-01-01", periods=n_time, freq="h")

        ds = xr.Dataset(
            {
                "velocity": (
                    ("beam", "cell", "time"),
                    np.random.randint(-300, 300, (n_beam, n_cell, n_time)),
                ),
                "correlation": (
                    ("beam", "cell", "time"),
                    np.full((n_beam, n_cell, n_time), 100, dtype=np.int16),
                ),
                "echo_intensity": (
                    ("beam", "cell", "time"),
                    np.full((n_beam, n_cell, n_time), 80, dtype=np.int16),
                ),
                "percent_good": (
                    ("beam", "cell", "time"),
                    np.full((n_beam, n_cell, n_time), 75, dtype=np.int16),
                ),
                # Fixed Leader fields (broadcast along time dimension)
                "frequency": (
                    "time",
                    np.array(["300 kHz"] * n_time, dtype=object),
                ),
                "depth_cell_length": (
                    "time",
                    np.full(n_time, 100, dtype=np.uint16),
                ),
                "num_cells": (
                    "time",
                    np.full(n_time, n_cell, dtype=np.uint8),
                ),
                "pings_per_ensemble": (
                    "time",
                    np.full(n_time, 100, dtype=np.uint16),
                ),
            },
            coords={
                "time": times,
                "cell": np.arange(n_cell),
                "beam": np.arange(n_beam),
            },
        )
        return ProcessedDataset(ds)

    def test_method_exists(self, proc_dataset):
        assert hasattr(proc_dataset, "get_percent_good_threshold")

    def test_returns_std_dev_result(self, proc_dataset):
        result = proc_dataset.get_percent_good_threshold(desired_std=1.0)
        assert isinstance(result, StdDevResult)

    def test_result_frequency_matches_dataset(self, proc_dataset):
        result = proc_dataset.get_percent_good_threshold(desired_std=1.0)
        assert result.frequency == 300

    def test_overrides_passed_through(self, proc_dataset):
        result = proc_dataset.get_percent_good_threshold(
            desired_std=1.0, n_pings=50
        )
        assert result.pings_per_ensemble == 50

    def test_invalid_frequency_raises(self, proc_dataset):
        with pytest.raises(ValueError, match="no exponential fit"):
            proc_dataset.get_percent_good_threshold(desired_std=1.0, frequency=38)

    def test_result_usable_with_apply_signal_quality(self, proc_dataset):
        result = proc_dataset.get_percent_good_threshold(desired_std=1.0)
        # Should not raise — the cutoff is a valid percent_good threshold
        proc_dataset.apply_signal_quality(percent_good=result.percent_good_cutoff)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
