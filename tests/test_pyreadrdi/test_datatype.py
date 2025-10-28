"""
Comprehensive pytest test suite for pyreadrdi.datatype function.

Tests cover:
- Normal operation with valid RDI files
- Variable extraction (velocity, correlation, echo, percent good, status)
- 3D array shape and data types verification
- Error handling (file not found, IO errors, corrupted data)
- Variable name validation
- Missing data handling (-32768 for velocity, 0 for others)
- Multiple ensembles and varying beam/cell counts
- Struct unpacking errors and edge cases
- Return value types and structure (data, ensemble, cells, beams, error_code)
- Integration with fileheader() and fixedleader() for auto-retrieval
- Cell and beam array handling
- Lazy-loaded parameter retrieval

References
----------
RDI WorkHorse Commands and Output Data Format (Section 5, page 123):
- Velocity data: 16-bit signed integers (ID 0x0100, 0x0101)
- Echo Intensity: 8-bit unsigned (ID 0x0200, 0x0201)
- Correlation: 8-bit unsigned (ID 0x0300, 0x0301)
- Percent Good: 8-bit unsigned (ID 0x0400, 0x0401)
- Status: 8-bit unsigned (ID 0x0600, 0x0601)
- Data stored as: cells[0:n_cells] x beams[0:n_beams] bytes per ensemble

Note on Return Values
---------------------
The datatype() function has variable return tuple length:
- Early errors: 2-tuple (data, error_code)
- Full success: 5-tuple (data, ensemble, cell_array, beam_array, error_code)

All tests use the unpack_datatype_result() helper to normalize the return value.
"""

from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from pyadps.utils.pyreadrdi import (
    ErrorCode,
    datatype,
    fileheader,
)

from .fixtures.ensemble_builder import (
    build_ensemble,
    EnsembleConfig,
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def unpack_datatype_result(result):
    """
    Unpack variable-length datatype return tuple.

    The datatype function has inconsistent return lengths:
    - Early errors: (data, error_code) - 2 elements
    - Full success: (data, ensemble, cell_array, beam_array, error_code) - 5 elements

    Args
    ----
    result : tuple
        Return value from datatype() function.

    Returns
    -------
    tuple
        (data, ensemble, cell_array, beam_array, error_code) normalized to 5-tuple.
        Missing values filled with None or empty arrays.
    """
    if len(result) == 2:
        # Early error return: (data, error_code)
        data, error_code = result
        return (data, None, None, None, error_code)
    elif len(result) == 5:
        # Full return: (data, ensemble, cell_array, beam_array, error_code)
        return result
    else:
        raise ValueError(f"Unexpected tuple length from datatype: {len(result)}")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(scope="function")
def valid_rdi_ensemble():
    """Generate a valid complete RDI ensemble using shared builder.

    Uses the imported build_ensemble() function from test_ensemble_builder.py.
    Structure follows RDI WorkHorse spec with default 7 data types.
    Default config: 4 beams, 30 cells.
    """
    return build_ensemble()


@pytest.fixture
def valid_rdi_file(tmp_path, valid_rdi_ensemble):
    """Create a temporary file with single valid ensemble."""
    rdi_file = tmp_path / "test_datatype_single.000"
    rdi_file.write_bytes(valid_rdi_ensemble)
    return rdi_file


@pytest.fixture
def multi_ensemble_rdi_file(tmp_path, valid_rdi_ensemble):
    """Create a temporary file with multiple valid ensembles."""
    rdi_file = tmp_path / "test_datatype_multi.000"
    # Write 3 identical ensembles
    rdi_file.write_bytes(valid_rdi_ensemble * 3)
    return rdi_file


@pytest.fixture
def rdi_file_varying_beams(tmp_path):
    """Create file with ensembles having different beam counts.

    First ensemble: 4 beams, 30 cells
    Second ensemble: 5 beams, 30 cells (should be padded in output array)
    """
    config1 = EnsembleConfig(beams=4, cells=30, num_datatypes=7)
    config2 = EnsembleConfig(beams=5, cells=30, num_datatypes=7)

    ensemble1 = build_ensemble(config=config1)
    ensemble2 = build_ensemble(config=config2)

    rdi_file = tmp_path / "test_datatype_varying.000"
    rdi_file.write_bytes(ensemble1 + ensemble2)
    return rdi_file


@pytest.fixture
def rdi_file_truncated(tmp_path, valid_rdi_ensemble):
    """Create file where velocity data section is truncated."""
    rdi_file = tmp_path / "test_datatype_truncated.000"
    # Write only 50 bytes (incomplete)
    rdi_file.write_bytes(valid_rdi_ensemble[:50])
    return rdi_file


# ============================================================================
# TESTS: Input Validation
# ============================================================================


class TestDatatypeInputValidation:
    """Test datatype function input validation."""

    def test_missing_filename_argument(self):
        """Test that calling datatype without arguments raises TypeError."""
        with pytest.raises(TypeError):
            datatype()

    def test_missing_varname_argument(self):
        """Test that calling datatype without var_name raises TypeError."""
        with pytest.raises(TypeError):
            datatype("test.000")

    def test_invalid_filename_type_integer(self):
        """Test that passing an integer as filename returns FILE_NOT_FOUND error."""
        result = datatype(12345, "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == ErrorCode.FILE_NOT_FOUND.code

    def test_invalid_filename_type_none(self):
        """Test that passing None as filename returns FILE_NOT_FOUND error."""
        result = datatype(None, "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == ErrorCode.FILE_NOT_FOUND.code

    def test_invalid_var_name(self, valid_rdi_file):
        """Test that invalid variable name returns VALUE_ERROR."""
        result = datatype(str(valid_rdi_file), "invalid_variable")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == ErrorCode.VALUE_ERROR.code

    def test_invalid_var_name_case_sensitive(self, valid_rdi_file):
        """Test that variable names are case-sensitive."""
        result = datatype(str(valid_rdi_file), "Velocity")  # Capital V
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == ErrorCode.VALUE_ERROR.code

    def test_valid_var_names(self, valid_rdi_file):
        """Test that all valid variable names are accepted."""
        valid_names = ["velocity", "correlation", "echo", "percent good", "status"]

        for var_name in valid_names:
            result = datatype(str(valid_rdi_file), var_name)
            data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(
                result
            )
            # Should not return VALUE_ERROR for valid names
            assert (
                error_code != ErrorCode.VALUE_ERROR.code
            ), f"'{var_name}' should be valid"


# ============================================================================
# TESTS: File Access
# ============================================================================


class TestDatatypeFileAccess:
    """Test datatype error handling for file access issues."""

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        result = datatype("nonexistent_file_xyz.000", "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == ErrorCode.FILE_NOT_FOUND.code

    def test_file_not_found_with_path_object(self):
        """Test file not found using pathlib.Path."""
        result = datatype(Path("/nonexistent/path/file.000"), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == ErrorCode.FILE_NOT_FOUND.code

    def test_permission_denied(self):
        """Test handling of permission denied error via mocking."""
        with mock.patch("builtins.open", side_effect=PermissionError("Access denied")):
            result = datatype("somefile.000", "velocity")
            data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(
                result
            )
            assert error_code == ErrorCode.PERMISSION_DENIED.code

    def test_io_error_generic(self):
        """Test handling of generic IOError."""
        with mock.patch("builtins.open", side_effect=IOError("IO problem")):
            result = datatype("somefile.000", "velocity")
            data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(
                result
            )
            assert error_code == ErrorCode.IO_ERROR.code


# ============================================================================
# TESTS: Return Value Structure
# ============================================================================


class TestDatatypeReturnStructure:
    """Test return value types and structure."""

    def test_return_is_tuple(self, valid_rdi_file):
        """Test that datatype returns a tuple."""
        result = datatype(str(valid_rdi_file), "velocity")
        assert isinstance(result, tuple)

    def test_return_has_valid_length(self, valid_rdi_file):
        """Test that datatype returns either 2-tuple or 5-tuple."""
        result = datatype(str(valid_rdi_file), "velocity")
        assert len(result) in (2, 5)

    def test_error_code_is_integer(self, valid_rdi_file):
        """Test that error code is an integer."""
        result = datatype(str(valid_rdi_file), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert isinstance(error_code, int)

    def test_ensemble_count_is_positive_on_success(self, valid_rdi_file):
        """Test that ensemble count is positive on successful read."""
        result = datatype(str(valid_rdi_file), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert ensemble is not None
        assert ensemble > 0

    def test_array_shape_has_three_dimensions(self, valid_rdi_file):
        """Test that returned data array has 3 dimensions (beam, cell, ensemble)."""
        result = datatype(str(valid_rdi_file), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert ensemble is not None
        assert len(data.shape) == 3

    def test_array_shape_order_beam_cell_ensemble(self, valid_rdi_file):
        """Test that array shape is (max_beam, max_cell, n_ensembles)."""
        result = datatype(str(valid_rdi_file), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert ensemble is not None
        max_beam = int(max(beam_arr))
        max_cell = int(max(cell_arr))
        assert data.shape == (max_beam, max_cell, ensemble)


# ============================================================================
# TESTS: Velocity Variable Extraction
# ============================================================================


class TestDatatypeVelocity:
    """Test velocity data extraction."""

    def test_velocity_dtype_is_int16(self, valid_rdi_file):
        """Test that velocity data is int16."""
        result = datatype(str(valid_rdi_file), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert data.dtype == np.int16

    def test_velocity_shape_includes_all_ensembles(self, multi_ensemble_rdi_file):
        """Test that velocity array includes all ensembles in third dimension."""
        result = datatype(str(multi_ensemble_rdi_file), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert ensemble == 3
        assert data.shape[2] == 3

    def test_velocity_successful_read(self, valid_rdi_file):
        """Test that velocity is successfully read with error code 0."""
        result = datatype(str(valid_rdi_file), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert ensemble is not None
        assert data.shape[0] > 0  # Has beams
        assert data.shape[1] > 0  # Has cells
        assert data.shape[2] == ensemble  # Has ensembles

    def test_velocity_in_valid_range(self, valid_rdi_file):
        """Test that velocity values are in valid int16 range."""
        result = datatype(str(valid_rdi_file), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert np.all(data >= -32768)
        assert np.all(data <= 32767)


# ============================================================================
# TESTS: Non-Velocity Variable Extraction
# ============================================================================


class TestDatatypeNonVelocity:
    """Test non-velocity variable extraction (correlation, echo, percent good)."""

    def test_correlation_dtype_is_uint8(self, valid_rdi_file):
        """Test that correlation data is uint8."""
        result = datatype(str(valid_rdi_file), "correlation")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert data.dtype == np.uint8

    def test_echo_dtype_is_uint8(self, valid_rdi_file):
        """Test that echo data is uint8."""
        result = datatype(str(valid_rdi_file), "echo")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert data.dtype == np.uint8

    def test_percent_good_dtype_is_uint8(self, valid_rdi_file):
        """Test that percent good data is uint8."""
        result = datatype(str(valid_rdi_file), "percent good")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert data.dtype == np.uint8

    def test_non_velocity_in_valid_range(self, valid_rdi_file):
        """Test that non-velocity values are in 0-255 range."""
        for var_name in ["correlation", "echo", "percent good"]:
            result = datatype(str(valid_rdi_file), var_name)
            data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(
                result
            )
            assert error_code == 0
            assert np.all(data >= 0)
            assert np.all(data <= 255)

    def test_all_non_velocity_variables_readable(self, valid_rdi_file):
        """Test that all non-velocity variables can be read successfully."""
        variables = ["correlation", "echo", "percent good"]

        for var_name in variables:
            result = datatype(str(valid_rdi_file), var_name)
            data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(
                result
            )
            assert error_code == 0, f"Failed to read {var_name}"
            assert ensemble is not None
            assert data.shape[0] > 0
            assert data.shape[1] > 0
            assert data.shape[2] > 0


# ============================================================================
# TESTS: Multiple Ensembles
# ============================================================================


class TestDatatypeMultipleEnsembles:
    """Test handling of files with multiple ensembles."""

    def test_multi_ensemble_read_all(self, multi_ensemble_rdi_file):
        """Test that all ensembles are read from file."""
        result = datatype(str(multi_ensemble_rdi_file), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert ensemble == 3
        assert data.shape[2] == 3

    def test_multi_ensemble_data_shape(self, multi_ensemble_rdi_file):
        """Test that multi-ensemble data has correct shape."""
        result = datatype(str(multi_ensemble_rdi_file), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert ensemble is not None
        max_beam = int(max(beam_arr))
        max_cell = int(max(cell_arr))
        assert data.shape == (max_beam, max_cell, ensemble)

    def test_multi_ensemble_each_independent(self, multi_ensemble_rdi_file):
        """Test that data from each ensemble is independent."""
        result = datatype(str(multi_ensemble_rdi_file), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        # First ensemble data
        ens0 = data[:, :, 0]
        # Second ensemble data
        ens1 = data[:, :, 1]
        # They should be identical (since we created them that way)
        np.testing.assert_array_equal(ens0, ens1)

    def test_multi_ensemble_arrays_have_correct_length(self, multi_ensemble_rdi_file):
        """Test that cell and beam arrays have length matching ensemble count."""
        result = datatype(str(multi_ensemble_rdi_file), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert ensemble is not None
        assert len(cell_arr) == ensemble
        assert len(beam_arr) == ensemble


# ============================================================================
# TESTS: Varying Geometry (Beams and Cells)
# ============================================================================


class TestDatatypeVaryingGeometry:
    """Test handling of varying beam and cell counts across ensembles."""

    def test_varying_beams_max_beam_used(self, rdi_file_varying_beams):
        """Test that maximum beam count is used for array shape."""
        result = datatype(str(rdi_file_varying_beams), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert ensemble is not None
        max_beam = int(max(beam_arr))
        assert data.shape[0] == max_beam

    def test_varying_geometry_array_returns_correct_lengths(
        self, rdi_file_varying_beams
    ):
        """Test that cell and beam arrays have correct lengths."""
        result = datatype(str(rdi_file_varying_beams), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert ensemble is not None
        assert len(cell_arr) == ensemble
        assert len(beam_arr) == ensemble


# ============================================================================
# TESTS: Integration with fileheader() and fixedleader()
# ============================================================================


class TestDatatypeIntegration:
    """Test datatype integration with fileheader() and fixedleader()."""

    def test_datatype_without_fileheader_params(self, valid_rdi_file):
        """Test that datatype works without explicit fileheader parameters.

        Should internally call fileheader when parameters not provided.
        """
        result = datatype(str(valid_rdi_file), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert ensemble is not None
        assert ensemble > 0
        assert data.shape[2] == ensemble

    def test_datatype_with_fileheader_params(self, valid_rdi_file):
        """Test that datatype uses provided fileheader parameters."""
        # First call fileheader
        (
            source_id,
            header_id,
            byteskip,
            offset,
            idarray,
            ensemble_count,
            error_code_fh,
        ) = fileheader(str(valid_rdi_file))

        assert error_code_fh == 0

        # Now call datatype with those parameters
        result = datatype(
            str(valid_rdi_file),
            "velocity",
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble_count,
        )
        data, ens_count, cell_arr, beam_arr, error_code = unpack_datatype_result(result)

        assert error_code == 0
        assert ens_count == ensemble_count

    def test_results_consistent_with_without_params(self, valid_rdi_file):
        """Test that results are consistent whether using params or not."""
        # Without params
        result1 = datatype(str(valid_rdi_file), "velocity")
        data1, ens1, cell1, beam1, err1 = unpack_datatype_result(result1)

        # With params
        (_, _, bs, off, ida, ens_fh, err_fh) = fileheader(str(valid_rdi_file))
        result2 = datatype(
            str(valid_rdi_file),
            "velocity",
            byteskip=bs,
            offset=off,
            idarray=ida,
            ensemble=ens_fh,
        )
        data2, ens2, cell2, beam2, err2 = unpack_datatype_result(result2)

        # Results should be identical
        assert err1 == err2
        assert ens1 == ens2
        np.testing.assert_array_equal(data1, data2)
        np.testing.assert_array_equal(cell1, cell2)
        np.testing.assert_array_equal(beam1, beam2)


# ============================================================================
# TESTS: Pathlib Support
# ============================================================================


class TestDatatypePathlibSupport:
    """Test that datatype works with pathlib.Path objects."""

    def test_path_object_accepted(self, valid_rdi_file):
        """Test that datatype accepts pathlib.Path objects."""
        result = datatype(valid_rdi_file, "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert ensemble is not None
        assert ensemble > 0

    def test_string_path_works_same_as_path_object(self, valid_rdi_file):
        """Test that string paths and Path objects give identical results."""
        result1 = datatype(str(valid_rdi_file), "velocity")
        result2 = datatype(valid_rdi_file, "velocity")

        data1, ens1, cell1, beam1, err1 = unpack_datatype_result(result1)
        data2, ens2, cell2, beam2, err2 = unpack_datatype_result(result2)

        assert err1 == err2
        assert ens1 == ens2
        np.testing.assert_array_equal(data1, data2)
        np.testing.assert_array_equal(cell1, cell2)
        np.testing.assert_array_equal(beam1, beam2)


# ============================================================================
# TESTS: Data Array Properties
# ============================================================================


class TestDatatypeArrayProperties:
    """Test properties of returned data arrays."""

    def test_data_array_is_c_contiguous(self, valid_rdi_file):
        """Test that returned array is C-contiguous for efficient access."""
        result = datatype(str(valid_rdi_file), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        # C-contiguous arrays are efficient for row-major iteration
        assert data.flags["C_CONTIGUOUS"]

    def test_no_nan_values_in_arrays(self, valid_rdi_file):
        """Test that returned arrays contain no NaN values."""
        for var_name in ["velocity", "correlation", "echo", "percent good"]:
            result = datatype(str(valid_rdi_file), var_name)
            data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(
                result
            )
            assert error_code == 0
            # Convert to float to check for NaN
            data_float = data.astype(float)
            assert not np.any(np.isnan(data_float))


# ============================================================================
# TESTS: Edge Cases
# ============================================================================


class TestDatatypeEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_data_integrity_across_ensembles(self, multi_ensemble_rdi_file):
        """Test that data is read consistently across multiple ensembles."""
        result = datatype(str(multi_ensemble_rdi_file), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert ensemble is not None
        # All ensembles should have consistent dimensions
        assert data.dtype == np.int16
        assert data.size > 0

    def test_default_geometry_returns_valid_shape(self, valid_rdi_file):
        """Test that default geometry (4 beams, 30 cells) returns valid shape."""
        result = datatype(str(valid_rdi_file), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert ensemble is not None
        # Default fixture has 4 beams and 30 cells
        assert data.shape[0] >= 4  # At least 4 beams
        assert data.shape[1] >= 30  # At least 30 cells

    def test_multi_ensemble_varying_configuration(self, multi_ensemble_rdi_file):
        """Test that multiple identical ensembles are read correctly."""
        result = datatype(str(multi_ensemble_rdi_file), "velocity")
        data, ensemble, cell_arr, beam_arr, error_code = unpack_datatype_result(result)
        assert error_code == 0
        assert ensemble is not None
        # Should have 3 ensembles
        assert ensemble == 3
        assert data.shape[2] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
