"""
Test suite for multifile.py - ADCP multi-file processing module.

This module provides tests for:
- ADCPFileConfig dataclass
- FileValidationResult and CombineResult dataclasses
- validate_adcp_file() function
- read_adcp_file() function
- find_adcp_files() function
- combine_adcp_files() function
- combine_file_list() function
- CLI interface

Run with: pytest test_multifile.py -v
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# -----------------------------------------------------------------------------
# IMPORT CONFIGURATION
# -----------------------------------------------------------------------------

try:
    from pyadps.processing.multifile import (
        ADCPFileConfig,
        FileValidationResult,
        CombineResult,
        validate_adcp_file,
        read_adcp_file,
        find_adcp_files,
        combine_adcp_files,
        combine_file_list,
        setup_logging,
        main,
    )

    PATCH_PREFIX = "pyadps.processing.multifile"
except ImportError:
    from multifile import (
        ADCPFileConfig,
        FileValidationResult,
        CombineResult,
        validate_adcp_file,
        read_adcp_file,
        find_adcp_files,
        combine_adcp_files,
        combine_file_list,
        setup_logging,
        main,
    )

    PATCH_PREFIX = "multifile"


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------


@pytest.fixture
def default_config():
    """Create default ADCPFileConfig."""
    return ADCPFileConfig()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def valid_adcp_header():
    """Create valid ADCP file header bytes."""
    # Header signature (0x7f7f) + ensemble size (100 bytes as little-endian)
    # Ensemble size field is at offset 2, length 2
    # Total ensemble size = 100 + 2 (header_size_adjustment) = 102
    header = b"\x7f\x7f"  # Header signature
    ensemble_size = (100).to_bytes(2, byteorder="little")  # 100 bytes
    return header + ensemble_size


@pytest.fixture
def valid_adcp_file(temp_dir, valid_adcp_header):
    """Create a valid ADCP binary file."""
    # Create file with 5 complete ensembles
    ensemble_size = 102  # 100 + 2 adjustment

    # Build file content
    content = bytearray()
    for i in range(5):
        # Each ensemble starts with header + size, then padding
        ensemble = bytearray(valid_adcp_header)
        ensemble.extend(b"\x00" * (ensemble_size - len(valid_adcp_header)))
        content.extend(ensemble)

    filepath = temp_dir / "valid.000"
    filepath.write_bytes(bytes(content))
    return filepath


@pytest.fixture
def truncated_adcp_file(temp_dir, valid_adcp_header):
    """Create a truncated ADCP file (incomplete last ensemble)."""
    ensemble_size = 102

    content = bytearray()
    # 3 complete ensembles
    for i in range(3):
        ensemble = bytearray(valid_adcp_header)
        ensemble.extend(b"\x00" * (ensemble_size - len(valid_adcp_header)))
        content.extend(ensemble)

    # Add partial ensemble (50 bytes instead of 102)
    content.extend(valid_adcp_header)
    content.extend(b"\x00" * 46)  # Total 50 bytes

    filepath = temp_dir / "truncated.000"
    filepath.write_bytes(bytes(content))
    return filepath


@pytest.fixture
def invalid_header_file(temp_dir):
    """Create a file with invalid ADCP header."""
    content = b"\x00\x00\x00\x00" + b"\x00" * 100
    filepath = temp_dir / "invalid.000"
    filepath.write_bytes(content)
    return filepath


@pytest.fixture
def offset_header_file(temp_dir, valid_adcp_header):
    """Create a file with valid header at non-zero offset.

    Simulates a file with garbage data before the actual ADCP data.
    """
    ensemble_size = 102

    # Garbage data before header (doesn't contain \x7f\x7f)
    garbage = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09" * 4  # 40 bytes

    # Build valid ensembles
    content = bytearray(garbage)
    for _ in range(3):
        ensemble = bytearray(valid_adcp_header)
        ensemble.extend(b"\x00" * (ensemble_size - len(valid_adcp_header)))
        content.extend(ensemble)

    filepath = temp_dir / "offset.000"
    filepath.write_bytes(bytes(content))
    return filepath


@pytest.fixture
def empty_file(temp_dir):
    """Create an empty file."""
    filepath = temp_dir / "empty.000"
    filepath.write_bytes(b"")
    return filepath


@pytest.fixture
def multiple_adcp_files(temp_dir, valid_adcp_header):
    """Create multiple valid ADCP files in a directory."""
    ensemble_size = 102
    files = []

    for i in range(3):
        content = bytearray()
        # Each file has different number of ensembles
        for j in range(i + 2):  # 2, 3, 4 ensembles
            ensemble = bytearray(valid_adcp_header)
            ensemble.extend(b"\x00" * (ensemble_size - len(valid_adcp_header)))
            content.extend(ensemble)

        filepath = temp_dir / f"data{i:03d}.000"
        filepath.write_bytes(bytes(content))
        files.append(filepath)

    return files


@pytest.fixture
def mismatched_ensemble_files(temp_dir):
    """Create files with different ensemble sizes (mismatched configuration)."""
    files = []

    # File 1: ensemble size 102 (100 + 2)
    header1 = b"\x7f\x7f" + (100).to_bytes(2, byteorder="little")
    content1 = bytearray()
    for _ in range(3):
        ensemble = bytearray(header1)
        ensemble.extend(b"\x00" * (102 - len(header1)))
        content1.extend(ensemble)
    filepath1 = temp_dir / "file1.000"
    filepath1.write_bytes(bytes(content1))
    files.append(filepath1)

    # File 2: ensemble size 202 (200 + 2) - DIFFERENT!
    header2 = b"\x7f\x7f" + (200).to_bytes(2, byteorder="little")
    content2 = bytearray()
    for _ in range(3):
        ensemble = bytearray(header2)
        ensemble.extend(b"\x00" * (202 - len(header2)))
        content2.extend(ensemble)
    filepath2 = temp_dir / "file2.000"
    filepath2.write_bytes(bytes(content2))
    files.append(filepath2)

    # File 3: ensemble size 102 (same as file 1)
    content3 = bytearray()
    for _ in range(3):
        ensemble = bytearray(header1)
        ensemble.extend(b"\x00" * (102 - len(header1)))
        content3.extend(ensemble)
    filepath3 = temp_dir / "file3.000"
    filepath3.write_bytes(bytes(content3))
    files.append(filepath3)

    return files


# -----------------------------------------------------------------------------
# ADCPFILECONFIG TESTS
# -----------------------------------------------------------------------------


class TestADCPFileConfig:
    """Tests for ADCPFileConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ADCPFileConfig()
        assert config.file_extension == "*.000"
        assert config.header_signature == b"\x7f\x7f"
        assert config.ensemble_size_offset == 2
        assert config.ensemble_size_length == 2
        assert config.header_size_adjustment == 2
        assert config.max_header_search == 10000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ADCPFileConfig(
            file_extension="*.dat",
            header_size_adjustment=4,
            max_header_search=5000,
        )
        assert config.file_extension == "*.dat"
        assert config.header_size_adjustment == 4
        assert config.max_header_search == 5000
        # Other values should remain default
        assert config.header_signature == b"\x7f\x7f"


# -----------------------------------------------------------------------------
# DATACLASS TESTS
# -----------------------------------------------------------------------------


class TestFileValidationResult:
    """Tests for FileValidationResult dataclass."""

    def test_default_values(self):
        """Test default values indicate invalid file."""
        result = FileValidationResult()
        assert result.is_valid is False
        assert result.header_offset == 0
        assert result.ensemble_size == 0
        assert result.valid_ensembles == 0
        assert result.is_truncated is False
        assert result.error_message == ""

    def test_valid_result(self):
        """Test creating a valid result."""
        result = FileValidationResult(
            is_valid=True,
            header_offset=0,
            ensemble_size=102,
            valid_ensembles=10,
            total_ensembles=10,
        )
        assert result.is_valid is True
        assert result.valid_ensembles == 10


class TestCombineResult:
    """Tests for CombineResult dataclass."""

    def test_default_values(self):
        """Test default values indicate failure."""
        result = CombineResult()
        assert result.success is False
        assert result.files_processed == 0
        assert result.files_total == 0
        assert result.total_bytes == 0
        assert result.skipped_files == []

    def test_successful_result(self):
        """Test creating a successful result."""
        result = CombineResult(
            success=True,
            files_processed=5,
            files_total=5,
            total_bytes=1000,
            total_ensembles=50,
            output_path=Path("output.000"),
        )
        assert result.success is True
        assert result.files_processed == 5


# -----------------------------------------------------------------------------
# VALIDATE_ADCP_FILE TESTS
# -----------------------------------------------------------------------------


class TestValidateAdcpFile:
    """Tests for validate_adcp_file function."""

    def test_valid_file(self, valid_adcp_file, default_config):
        """Test validation of valid ADCP file."""
        result = validate_adcp_file(valid_adcp_file, default_config)
        assert result.is_valid is True
        assert result.header_offset == 0
        assert result.ensemble_size == 102
        assert result.valid_ensembles == 5
        assert result.is_truncated is False

    def test_truncated_file(self, truncated_adcp_file, default_config):
        """Test validation of truncated file."""
        result = validate_adcp_file(truncated_adcp_file, default_config)
        assert result.is_valid is True
        assert result.valid_ensembles == 3
        assert result.is_truncated is True

    def test_invalid_header(self, invalid_header_file, default_config):
        """Test validation of file with invalid header."""
        result = validate_adcp_file(invalid_header_file, default_config)
        assert result.is_valid is False
        assert "No valid ADCP header" in result.error_message

    def test_empty_file(self, empty_file, default_config):
        """Test validation of empty file."""
        result = validate_adcp_file(empty_file, default_config)
        assert result.is_valid is False
        assert "empty" in result.error_message.lower()

    def test_nonexistent_file(self, default_config):
        """Test validation of nonexistent file."""
        result = validate_adcp_file("/nonexistent/file.000", default_config)
        assert result.is_valid is False
        assert "not found" in result.error_message.lower()

    def test_directory_path(self, temp_dir, default_config):
        """Test validation when path is a directory."""
        result = validate_adcp_file(temp_dir, default_config)
        assert result.is_valid is False
        assert "not a file" in result.error_message.lower()

    def test_offset_header(self, offset_header_file, default_config):
        """Test validation of file with header at non-zero offset."""
        result = validate_adcp_file(offset_header_file, default_config)
        assert result.is_valid is True
        assert result.header_offset > 0

    def test_default_config(self, valid_adcp_file):
        """Test validation with default config (None)."""
        result = validate_adcp_file(valid_adcp_file)
        assert result.is_valid is True


# -----------------------------------------------------------------------------
# READ_ADCP_FILE TESTS
# -----------------------------------------------------------------------------


class TestReadAdcpFile:
    """Tests for read_adcp_file function."""

    def test_read_valid_file(self, valid_adcp_file, default_config):
        """Test reading valid ADCP file."""
        data = read_adcp_file(valid_adcp_file, default_config)
        assert len(data) == 5 * 102  # 5 ensembles * 102 bytes

    def test_read_truncated_file(self, truncated_adcp_file, default_config):
        """Test reading truncated file returns only complete ensembles."""
        data = read_adcp_file(truncated_adcp_file, default_config)
        assert len(data) == 3 * 102  # Only 3 complete ensembles

    def test_read_invalid_file_raises(self, invalid_header_file, default_config):
        """Test reading invalid file raises ValueError."""
        with pytest.raises(ValueError, match="Invalid ADCP file"):
            read_adcp_file(invalid_header_file, default_config)

    def test_read_nonexistent_file_raises(self, default_config):
        """Test reading nonexistent file raises error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            read_adcp_file("/nonexistent/file.000", default_config)

    def test_read_without_validation(self, valid_adcp_file, default_config):
        """Test reading file without validation."""
        data = read_adcp_file(valid_adcp_file, default_config, validate=False)
        assert len(data) > 0

    def test_read_default_config(self, valid_adcp_file):
        """Test reading with default config (None)."""
        data = read_adcp_file(valid_adcp_file)
        assert len(data) > 0


# -----------------------------------------------------------------------------
# FIND_ADCP_FILES TESTS
# -----------------------------------------------------------------------------


class TestFindAdcpFiles:
    """Tests for find_adcp_files function."""

    def test_find_files(self, multiple_adcp_files, default_config):
        """Test finding ADCP files in directory."""
        folder = multiple_adcp_files[0].parent
        files = find_adcp_files(folder, default_config)
        assert len(files) == 3
        assert all(f.suffix == ".000" for f in files)

    def test_find_files_sorted(self, multiple_adcp_files, default_config):
        """Test that found files are sorted."""
        folder = multiple_adcp_files[0].parent
        files = find_adcp_files(folder, default_config)
        assert files == sorted(files)

    def test_find_no_files(self, temp_dir, default_config):
        """Test finding files in empty directory."""
        files = find_adcp_files(temp_dir, default_config)
        assert files == []

    def test_find_nonexistent_folder(self, default_config):
        """Test finding files in nonexistent folder."""
        with pytest.raises(FileNotFoundError):
            find_adcp_files("/nonexistent/folder", default_config)

    def test_find_not_directory(self, valid_adcp_file, default_config):
        """Test finding files when path is not a directory."""
        with pytest.raises(NotADirectoryError):
            find_adcp_files(valid_adcp_file, default_config)

    def test_find_custom_extension(self, temp_dir):
        """Test finding files with custom extension."""
        # Create files with different extension
        (temp_dir / "data1.dat").write_bytes(b"test")
        (temp_dir / "data2.dat").write_bytes(b"test")
        (temp_dir / "data3.000").write_bytes(b"test")

        config = ADCPFileConfig(file_extension="*.dat")
        files = find_adcp_files(temp_dir, config)
        assert len(files) == 2
        assert all(f.suffix == ".dat" for f in files)

    def test_find_default_config(self, multiple_adcp_files):
        """Test finding files with default config (None)."""
        folder = multiple_adcp_files[0].parent
        files = find_adcp_files(folder)
        assert len(files) == 3


# -----------------------------------------------------------------------------
# COMBINE_ADCP_FILES TESTS
# -----------------------------------------------------------------------------


class TestCombineAdcpFiles:
    """Tests for combine_adcp_files function."""

    def test_combine_success(self, multiple_adcp_files, temp_dir, default_config):
        """Test successful file combination."""
        folder = multiple_adcp_files[0].parent
        output = temp_dir / "combined.000"

        result = combine_adcp_files(folder, output, default_config)

        assert result.success is True
        assert result.files_processed == 3
        assert result.files_total == 3
        assert result.output_path == output
        assert result.total_bytes > 0
        assert result.total_ensembles == 2 + 3 + 4  # 9 total ensembles
        assert output.exists()

    def test_combine_creates_output_dir(
        self, multiple_adcp_files, temp_dir, default_config
    ):
        """Test that output directory is created if needed."""
        folder = multiple_adcp_files[0].parent
        output = temp_dir / "subdir" / "nested" / "combined.000"

        result = combine_adcp_files(folder, output, default_config)

        assert result.success is True
        assert output.exists()

    def test_combine_empty_folder(self, temp_dir, default_config):
        """Test combining from empty folder."""
        output = temp_dir / "combined.000"

        result = combine_adcp_files(temp_dir, output, default_config)

        assert result.success is False
        assert "No ADCP files found" in result.error_message

    def test_combine_nonexistent_folder(self, temp_dir, default_config):
        """Test combining from nonexistent folder."""
        output = temp_dir / "combined.000"

        result = combine_adcp_files("/nonexistent/folder", output, default_config)

        assert result.success is False
        assert "not found" in result.error_message.lower()

    def test_combine_skip_invalid(self, multiple_adcp_files, temp_dir, default_config):
        """Test skipping invalid files."""
        folder = multiple_adcp_files[0].parent

        # Create an invalid file
        invalid_file = folder / "invalid.000"
        invalid_file.write_bytes(b"\x00\x00\x00\x00" * 100)

        output = temp_dir / "combined.000"
        result = combine_adcp_files(folder, output, default_config, skip_invalid=True)

        assert result.success is True
        assert result.files_processed == 3  # Only valid files
        assert "invalid.000" in result.skipped_files

    def test_combine_strict_mode(self, multiple_adcp_files, temp_dir, default_config):
        """Test strict mode stops on invalid file."""
        folder = multiple_adcp_files[0].parent

        # Create an invalid file that sorts first
        invalid_file = folder / "aaa_invalid.000"
        invalid_file.write_bytes(b"\x00\x00\x00\x00" * 100)

        output = temp_dir / "combined.000"
        result = combine_adcp_files(folder, output, default_config, skip_invalid=False)

        assert result.success is False
        assert "Invalid file" in result.error_message

    def test_combine_output_content(
        self, multiple_adcp_files, temp_dir, default_config
    ):
        """Test that combined output contains correct data."""
        folder = multiple_adcp_files[0].parent
        output = temp_dir / "combined.000"

        result = combine_adcp_files(folder, output, default_config)

        # Verify output size matches sum of valid ensembles
        expected_bytes = (2 + 3 + 4) * 102  # 9 ensembles * 102 bytes
        assert result.total_bytes == expected_bytes
        assert output.stat().st_size == expected_bytes

    def test_combine_ensemble_size_mismatch_skip(
        self, mismatched_ensemble_files, temp_dir, default_config
    ):
        """Test that mismatched ensemble sizes are skipped by default."""
        folder = mismatched_ensemble_files[0].parent
        output = temp_dir / "combined.000"

        result = combine_adcp_files(folder, output, default_config, skip_invalid=True)

        assert result.success is True
        assert result.files_processed == 2  # file1 and file3 (same size)
        assert len(result.skipped_files) == 1  # file2 (different size)
        assert "file2.000" in result.skipped_files

    def test_combine_ensemble_size_mismatch_strict(
        self, mismatched_ensemble_files, temp_dir, default_config
    ):
        """Test that mismatched ensemble sizes cause error in strict mode."""
        folder = mismatched_ensemble_files[0].parent
        output = temp_dir / "combined.000"

        result = combine_adcp_files(
            folder,
            output,
            default_config,
            skip_invalid=False,
            require_matching_ensemble_size=True,
        )

        assert result.success is False
        assert "mismatch" in result.error_message.lower()

    def test_combine_no_size_check(
        self, mismatched_ensemble_files, temp_dir, default_config
    ):
        """Test combining with ensemble size check disabled."""
        folder = mismatched_ensemble_files[0].parent
        output = temp_dir / "combined.000"

        result = combine_adcp_files(
            folder, output, default_config, require_matching_ensemble_size=False
        )

        assert result.success is True
        assert result.files_processed == 3  # All files processed
        assert len(result.skipped_files) == 0


# -----------------------------------------------------------------------------
# COMBINE_FILE_LIST TESTS
# -----------------------------------------------------------------------------


class TestCombineFileList:
    """Tests for combine_file_list function."""

    def test_combine_list_success(self, multiple_adcp_files, temp_dir, default_config):
        """Test successful combination of file list."""
        output = temp_dir / "combined.000"

        result = combine_file_list(multiple_adcp_files, output, default_config)

        assert result.success is True
        assert result.files_processed == 3
        assert output.exists()

    def test_combine_list_partial(self, multiple_adcp_files, temp_dir, default_config):
        """Test combining subset of files."""
        output = temp_dir / "combined.000"

        # Only first two files
        result = combine_file_list(multiple_adcp_files[:2], output, default_config)

        assert result.success is True
        assert result.files_processed == 2
        assert result.total_ensembles == 2 + 3  # 5 ensembles

    def test_combine_list_empty(self, temp_dir, default_config):
        """Test combining empty file list."""
        output = temp_dir / "combined.000"

        result = combine_file_list([], output, default_config)

        assert result.success is False
        assert "No files provided" in result.error_message

    def test_combine_list_with_invalid(
        self, multiple_adcp_files, temp_dir, default_config
    ):
        """Test combining list with invalid file."""
        # Add invalid file path
        files = list(multiple_adcp_files) + [Path("/nonexistent/file.000")]
        output = temp_dir / "combined.000"

        result = combine_file_list(files, output, default_config, skip_invalid=True)

        assert result.success is True
        assert result.files_processed == 3
        assert len(result.skipped_files) == 1

    def test_combine_list_string_paths(
        self, multiple_adcp_files, temp_dir, default_config
    ):
        """Test combining with string paths instead of Path objects."""
        output = temp_dir / "combined.000"
        string_paths = [str(f) for f in multiple_adcp_files]

        result = combine_file_list(string_paths, output, default_config)

        assert result.success is True
        assert result.files_processed == 3

    def test_combine_list_ensemble_size_mismatch(
        self, mismatched_ensemble_files, temp_dir, default_config
    ):
        """Test file list with mismatched ensemble sizes."""
        output = temp_dir / "combined.000"

        result = combine_file_list(
            mismatched_ensemble_files, output, default_config, skip_invalid=True
        )

        assert result.success is True
        assert result.files_processed == 2  # file1 and file3
        assert len(result.skipped_files) == 1  # file2

    def test_combine_list_no_size_check(
        self, mismatched_ensemble_files, temp_dir, default_config
    ):
        """Test file list with size check disabled."""
        output = temp_dir / "combined.000"

        result = combine_file_list(
            mismatched_ensemble_files,
            output,
            default_config,
            require_matching_ensemble_size=False,
        )

        assert result.success is True
        assert result.files_processed == 3


# -----------------------------------------------------------------------------
# SETUP_LOGGING TESTS
# -----------------------------------------------------------------------------


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_warning_level(self):
        """Test setting WARNING level."""
        setup_logging(0)
        # No assertion needed, just verify no error

    def test_setup_info_level(self):
        """Test setting INFO level."""
        setup_logging(1)

    def test_setup_debug_level(self):
        """Test setting DEBUG level."""
        setup_logging(2)

    def test_setup_higher_verbosity(self):
        """Test higher verbosity still works."""
        setup_logging(5)  # Should still set DEBUG


# -----------------------------------------------------------------------------
# CLI TESTS
# -----------------------------------------------------------------------------


class TestCLI:
    """Tests for CLI main function."""

    def test_cli_success(self, multiple_adcp_files, temp_dir):
        """Test CLI with valid arguments."""
        folder = multiple_adcp_files[0].parent
        output = temp_dir / "cli_output.000"

        with patch("sys.argv", ["multifile", str(folder), "-o", str(output)]):
            exit_code = main()

        assert exit_code == 0
        assert output.exists()

    def test_cli_verbose(self, multiple_adcp_files, temp_dir):
        """Test CLI with verbose flag."""
        folder = multiple_adcp_files[0].parent
        output = temp_dir / "cli_output.000"

        with patch("sys.argv", ["multifile", str(folder), "-o", str(output), "-v"]):
            exit_code = main()

        assert exit_code == 0

    def test_cli_very_verbose(self, multiple_adcp_files, temp_dir):
        """Test CLI with double verbose flag."""
        folder = multiple_adcp_files[0].parent
        output = temp_dir / "cli_output.000"

        with patch("sys.argv", ["multifile", str(folder), "-o", str(output), "-vv"]):
            exit_code = main()

        assert exit_code == 0

    def test_cli_strict_mode(self, multiple_adcp_files, temp_dir):
        """Test CLI with strict flag."""
        folder = multiple_adcp_files[0].parent
        output = temp_dir / "cli_output.000"

        with patch(
            "sys.argv", ["multifile", str(folder), "-o", str(output), "--strict"]
        ):
            exit_code = main()

        assert exit_code == 0

    def test_cli_custom_extension(self, temp_dir):
        """Test CLI with custom extension."""
        # Create .dat files
        for i in range(2):
            (temp_dir / f"data{i}.dat").write_bytes(
                b"\x7f\x7f" + (100).to_bytes(2, "little") + b"\x00" * 98
            )

        output = temp_dir / "cli_output.dat"

        with patch(
            "sys.argv",
            ["multifile", str(temp_dir), "-o", str(output), "--extension", "*.dat"],
        ):
            exit_code = main()

        # May fail due to invalid file content, but should not crash
        assert exit_code in (0, 1)

    def test_cli_empty_folder(self, temp_dir):
        """Test CLI with empty folder."""
        output = temp_dir / "cli_output.000"

        with patch("sys.argv", ["multifile", str(temp_dir), "-o", str(output)]):
            exit_code = main()

        assert exit_code == 1  # Should fail

    def test_cli_nonexistent_folder(self, temp_dir):
        """Test CLI with nonexistent folder."""
        output = temp_dir / "cli_output.000"

        with patch("sys.argv", ["multifile", "/nonexistent/folder", "-o", str(output)]):
            exit_code = main()

        assert exit_code == 1


# -----------------------------------------------------------------------------
# EDGE CASES AND INTEGRATION TESTS
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and integration scenarios."""

    def test_very_small_file(self, temp_dir, default_config):
        """Test handling of file smaller than header."""
        small_file = temp_dir / "small.000"
        small_file.write_bytes(b"\x7f\x7f")  # Only header signature, no size

        result = validate_adcp_file(small_file, default_config)
        assert result.is_valid is False

    def test_single_ensemble_file(self, temp_dir, valid_adcp_header, default_config):
        """Test file with single ensemble."""
        ensemble_size = 102
        content = bytearray(valid_adcp_header)
        content.extend(b"\x00" * (ensemble_size - len(valid_adcp_header)))

        filepath = temp_dir / "single.000"
        filepath.write_bytes(bytes(content))

        result = validate_adcp_file(filepath, default_config)
        assert result.is_valid is True
        assert result.valid_ensembles == 1

    def test_large_ensemble_count(self, temp_dir, valid_adcp_header, default_config):
        """Test file with many ensembles."""
        ensemble_size = 102
        num_ensembles = 100

        content = bytearray()
        for _ in range(num_ensembles):
            ensemble = bytearray(valid_adcp_header)
            ensemble.extend(b"\x00" * (ensemble_size - len(valid_adcp_header)))
            content.extend(ensemble)

        filepath = temp_dir / "large.000"
        filepath.write_bytes(bytes(content))

        result = validate_adcp_file(filepath, default_config)
        assert result.is_valid is True
        assert result.valid_ensembles == num_ensembles

    def test_combine_single_file(self, valid_adcp_file, temp_dir, default_config):
        """Test combining single file."""
        output = temp_dir / "combined.000"

        result = combine_file_list([valid_adcp_file], output, default_config)

        assert result.success is True
        assert result.files_processed == 1

    def test_path_with_spaces(self, temp_dir, valid_adcp_header):
        """Test handling paths with spaces."""
        # Create directory with space in name
        space_dir = temp_dir / "folder with spaces"
        space_dir.mkdir()

        # Create valid file
        ensemble_size = 102
        content = bytearray(valid_adcp_header)
        content.extend(b"\x00" * (ensemble_size - len(valid_adcp_header)))

        filepath = space_dir / "data with spaces.000"
        filepath.write_bytes(bytes(content) * 3)

        result = validate_adcp_file(filepath)
        assert result.is_valid is True

    def test_unicode_path(self, temp_dir, valid_adcp_header):
        """Test handling paths with unicode characters."""
        # Create directory with unicode name
        unicode_dir = temp_dir / "donnÃ©es_ocÃ©an"
        unicode_dir.mkdir()

        ensemble_size = 102
        content = bytearray(valid_adcp_header)
        content.extend(b"\x00" * (ensemble_size - len(valid_adcp_header)))

        filepath = unicode_dir / "donnÃ©es.000"
        filepath.write_bytes(bytes(content) * 3)

        result = validate_adcp_file(filepath)
        assert result.is_valid is True


# =============================================================================
# HELPERS
# =============================================================================

_HEADER = b"\x7f\x7f"
_SIZE_FIELD_100 = (100).to_bytes(2, byteorder="little")  # little-endian 100
_ENSEMBLE_SIZE = 102  # 100 + header_size_adjustment(2)


def _valid_ensemble(n=1):
    """Return n complete valid ensembles (102 bytes each)."""
    single = _HEADER + _SIZE_FIELD_100 + b"\x00" * (_ENSEMBLE_SIZE - 4)
    return single * n


def _invalid_bytes():
    """Return bytes with no valid ADCP header."""
    return b"\x00" * 100


# =============================================================================
# VALIDATE_ADCP_FILE COVERAGE GAPS
# =============================================================================


class TestValidateAdcpFileCoverageGaps:
    """Cover lines 157-159 and 192-193 in validate_adcp_file()."""

    # -------------------------------------------------------------------------
    # Lines 157-159: IOError during open()
    # -------------------------------------------------------------------------

    def test_ioerror_reading_file_captured_in_result(self, temp_dir, default_config):
        """IOError raised by open() is caught; error_message set (lines 157-159)."""
        f = temp_dir / "ioerror.000"
        f.write_bytes(_HEADER + b"\x00" * 20)  # non-empty, passes size check

        with patch("builtins.open", side_effect=IOError("permission denied")):
            result = validate_adcp_file(f, default_config)

        assert result.is_valid is False
        assert "Error reading file" in result.error_message
        assert "permission denied" in result.error_message

    def test_ioerror_reading_file_does_not_propagate(self, temp_dir, default_config):
        """IOError is swallowed; no exception reaches the caller."""
        f = temp_dir / "ioerror2.000"
        f.write_bytes(_HEADER + b"\x00" * 20)

        with patch("builtins.open", side_effect=IOError("disk error")):
            result = validate_adcp_file(f, default_config)  # must not raise

        assert isinstance(result, FileValidationResult)

    # -------------------------------------------------------------------------
    # Lines 192-193: ensemble_size <= 0
    # -------------------------------------------------------------------------

    def test_zero_ensemble_size_triggers_invalid(self, temp_dir):
        """Size field=0 with adjustment=0 gives ensemble_size=0 (lines 192-193)."""
        f = temp_dir / "zero_size.000"
        # Size field at offset 2-3 is zero; adjustment=0 → total=0
        f.write_bytes(b"\x7f\x7f" + b"\x00\x00" + b"\x00" * 100)

        config = ADCPFileConfig(header_size_adjustment=0)
        result = validate_adcp_file(f, config)

        assert result.is_valid is False
        assert "Invalid ensemble size" in result.error_message
        assert "0" in result.error_message

    def test_negative_ensemble_size_triggers_invalid(self, temp_dir):
        """Negative header_size_adjustment can produce ensemble_size < 0."""
        f = temp_dir / "neg_size.000"
        f.write_bytes(b"\x7f\x7f" + b"\x00\x00" + b"\x00" * 100)

        config = ADCPFileConfig(header_size_adjustment=-10)
        result = validate_adcp_file(f, config)

        assert result.is_valid is False
        assert "Invalid ensemble size" in result.error_message


# =============================================================================
# COMBINE_ADCP_FILES COVERAGE GAPS
# =============================================================================


class TestCombineAdcpFilesCoverageGaps:
    """Cover exception paths and IOError in combine_adcp_files()."""

    # -------------------------------------------------------------------------
    # Lines 478-481: Exception → skip_invalid=True → file added to skipped_files
    # -------------------------------------------------------------------------

    def test_exception_in_loop_skip_true_adds_to_skipped(
        self, valid_adcp_file, temp_dir, default_config
    ):
        """RuntimeError inside per-file loop → file skipped when skip_invalid=True."""
        folder = valid_adcp_file.parent
        output = temp_dir / "out.000"

        with patch(
            f"{PATCH_PREFIX}.validate_adcp_file",
            side_effect=RuntimeError("corrupt"),
        ):
            result = combine_adcp_files(
                folder, output, default_config, skip_invalid=True
            )

        assert result.success is False
        assert valid_adcp_file.name in result.skipped_files

    # -------------------------------------------------------------------------
    # Lines 482-484: Exception → skip_invalid=False → immediate error return
    # -------------------------------------------------------------------------

    def test_exception_in_loop_skip_false_returns_error(
        self, valid_adcp_file, temp_dir, default_config
    ):
        """RuntimeError inside per-file loop → error set immediately when skip_invalid=False."""
        folder = valid_adcp_file.parent
        output = temp_dir / "out.000"

        with patch(
            f"{PATCH_PREFIX}.validate_adcp_file",
            side_effect=RuntimeError("corrupt"),
        ):
            result = combine_adcp_files(
                folder, output, default_config, skip_invalid=False
            )

        assert result.success is False
        assert "Error processing" in result.error_message
        assert valid_adcp_file.name in result.error_message

    # -------------------------------------------------------------------------
    # Lines 488-489: combined_data empty after all exceptions (skip_invalid=True)
    # -------------------------------------------------------------------------

    def test_no_valid_data_error_when_all_files_raise(
        self, valid_adcp_file, temp_dir, default_config
    ):
        """All files skip due to exception → 'No valid data extracted' (lines 488-489)."""
        folder = valid_adcp_file.parent
        output = temp_dir / "out.000"

        with patch(
            f"{PATCH_PREFIX}.validate_adcp_file",
            side_effect=RuntimeError("all bad"),
        ):
            result = combine_adcp_files(
                folder, output, default_config, skip_invalid=True
            )

        assert result.success is False
        assert "No valid data extracted" in result.error_message

    # -------------------------------------------------------------------------
    # Lines 510-512: IOError writing the output file
    # -------------------------------------------------------------------------

    def test_ioerror_on_write_captured(self, valid_adcp_file, temp_dir, default_config):
        """IOError when writing output is caught; error_message set (lines 510-512)."""
        folder = valid_adcp_file.parent
        output = temp_dir / "out.000"
        real_open = open

        def selective_open(path, mode="r", **kw):
            if mode == "wb":
                raise IOError("disk full")
            return real_open(path, mode, **kw)

        with patch("builtins.open", side_effect=selective_open):
            result = combine_adcp_files(folder, output, default_config)

        assert result.success is False
        assert "Error writing output file" in result.error_message
        assert "disk full" in result.error_message


# =============================================================================
# COMBINE_FILE_LIST COVERAGE GAPS
# =============================================================================


class TestCombineFileListCoverageGaps:
    """Cover error paths in combine_file_list()."""

    # -------------------------------------------------------------------------
    # Lines 577-580: invalid file + skip_invalid=False
    # -------------------------------------------------------------------------

    def test_invalid_file_strict_mode_sets_error(self, temp_dir, default_config):
        """Invalid file → error set immediately when skip_invalid=False (lines 577-580)."""
        bad = temp_dir / "bad.000"
        bad.write_bytes(_invalid_bytes())
        output = temp_dir / "out.000"

        result = combine_file_list([bad], output, default_config, skip_invalid=False)

        assert result.success is False
        assert "Invalid file" in result.error_message
        assert "bad.000" in result.error_message

    def test_invalid_file_strict_error_contains_validation_reason(
        self, temp_dir, default_config
    ):
        """Error message includes the validation failure detail."""
        bad = temp_dir / "bad.000"
        bad.write_bytes(_invalid_bytes())
        output = temp_dir / "out.000"

        result = combine_file_list([bad], output, default_config, skip_invalid=False)

        assert "No valid ADCP header" in result.error_message

    # -------------------------------------------------------------------------
    # Lines 599-600: ensemble size mismatch + skip_invalid=False
    # -------------------------------------------------------------------------

    def test_ensemble_size_mismatch_strict_sets_error(
        self, mismatched_ensemble_files, temp_dir, default_config
    ):
        """Ensemble size mismatch → error set immediately when skip_invalid=False (lines 599-600)."""
        output = temp_dir / "out.000"

        result = combine_file_list(
            mismatched_ensemble_files,
            output,
            default_config,
            skip_invalid=False,
            require_matching_ensemble_size=True,
        )

        assert result.success is False
        # The mismatch error message is written directly (not via validation)
        assert result.error_message
        assert (
            "mismatch" in result.error_message.lower()
            or "expected" in result.error_message
        )

    # -------------------------------------------------------------------------
    # Lines 615-618: Exception → skip_invalid=True → file skipped
    # -------------------------------------------------------------------------

    def test_exception_in_list_loop_skip_true_adds_to_skipped(
        self, valid_adcp_file, temp_dir, default_config
    ):
        """RuntimeError in loop → file skipped when skip_invalid=True (lines 615-618)."""
        output = temp_dir / "out.000"

        with patch(
            f"{PATCH_PREFIX}.validate_adcp_file",
            side_effect=RuntimeError("read error"),
        ):
            result = combine_file_list(
                [valid_adcp_file], output, default_config, skip_invalid=True
            )

        assert result.success is False
        assert valid_adcp_file.name in result.skipped_files

    # -------------------------------------------------------------------------
    # Lines 619-621: Exception → skip_invalid=False → immediate error return
    # -------------------------------------------------------------------------

    def test_exception_in_list_loop_skip_false_returns_error(
        self, valid_adcp_file, temp_dir, default_config
    ):
        """RuntimeError in loop → error set immediately when skip_invalid=False (lines 619-621)."""
        output = temp_dir / "out.000"

        with patch(
            f"{PATCH_PREFIX}.validate_adcp_file",
            side_effect=RuntimeError("read error"),
        ):
            result = combine_file_list(
                [valid_adcp_file], output, default_config, skip_invalid=False
            )

        assert result.success is False
        assert "Error processing" in result.error_message

    # -------------------------------------------------------------------------
    # Lines 625-626: combined_data empty after all files skip/fail
    # -------------------------------------------------------------------------

    def test_no_valid_data_error_in_file_list_when_all_raise(
        self, valid_adcp_file, temp_dir, default_config
    ):
        """All files raise → 'No valid data extracted' (lines 625-626)."""
        output = temp_dir / "out.000"

        with patch(
            f"{PATCH_PREFIX}.validate_adcp_file",
            side_effect=RuntimeError("all bad"),
        ):
            result = combine_file_list(
                [valid_adcp_file], output, default_config, skip_invalid=True
            )

        assert result.success is False
        assert "No valid data extracted" in result.error_message

    # -------------------------------------------------------------------------
    # Lines 646-648: IOError writing output in combine_file_list
    # -------------------------------------------------------------------------

    def test_ioerror_on_write_in_file_list_captured(
        self, valid_adcp_file, temp_dir, default_config
    ):
        """IOError writing output caught; error_message set (lines 646-648)."""
        output = temp_dir / "out.000"
        real_open = open

        def selective_open(path, mode="r", **kw):
            if mode == "wb":
                raise IOError("no space left")
            return real_open(path, mode, **kw)

        with patch("builtins.open", side_effect=selective_open):
            result = combine_file_list([valid_adcp_file], output, default_config)

        assert result.success is False
        assert "Error writing output file" in result.error_message
        assert "no space left" in result.error_message


# =============================================================================
# CLI COVERAGE GAPS
# =============================================================================


class TestCLICoverageGaps:
    """Cover lines 781-783, 789-791, and 799 in main() and __main__."""

    # -------------------------------------------------------------------------
    # Lines 781-783: success with skipped files → skipped list printed
    # -------------------------------------------------------------------------

    def test_success_with_skipped_files_prints_list(self, temp_dir, capsys):
        """Skipped file names printed on success (lines 781-783)."""
        folder = temp_dir / "mixed"
        folder.mkdir()
        output = temp_dir / "out.000"

        # One valid file; one invalid file that sorts before it
        (folder / "good.000").write_bytes(_valid_ensemble(1))
        (folder / "aaa_bad.000").write_bytes(_invalid_bytes())

        with patch("sys.argv", ["multifile", str(folder), "-o", str(output)]):
            code = main()

        assert code == 0
        stdout = capsys.readouterr().out
        assert "Skipped" in stdout
        assert "aaa_bad.000" in stdout

    def test_success_with_skipped_files_shows_count(self, temp_dir, capsys):
        """Skipped count appears in success output."""
        folder = temp_dir / "mixed2"
        folder.mkdir()
        output = temp_dir / "out2.000"

        (folder / "good.000").write_bytes(_valid_ensemble(1))
        (folder / "aaa_bad.000").write_bytes(_invalid_bytes())

        with patch("sys.argv", ["multifile", str(folder), "-o", str(output)]):
            main()

        stdout = capsys.readouterr().out
        assert "1" in stdout  # at least the count digit appears

    # -------------------------------------------------------------------------
    # Lines 789-791: failure with skipped files → skipped list printed
    # -------------------------------------------------------------------------

    def test_failure_with_skipped_files_prints_list(self, temp_dir, capsys):
        """Skipped file names printed under failure block (lines 789-791)."""
        folder = temp_dir / "allbad"
        folder.mkdir()
        output = temp_dir / "out_bad.000"

        # Two invalid files: both skipped → no data → failure, skipped_files non-empty
        (folder / "bad1.000").write_bytes(_invalid_bytes())
        (folder / "bad2.000").write_bytes(_invalid_bytes())

        with patch("sys.argv", ["multifile", str(folder), "-o", str(output)]):
            code = main()

        assert code == 1
        stdout = capsys.readouterr().out
        assert "Failed to combine files" in stdout
        assert "Skipped files" in stdout
        assert "bad1.000" in stdout
        assert "bad2.000" in stdout

    def test_failure_error_message_printed(self, temp_dir, capsys):
        """Error message is shown in failure output."""
        folder = temp_dir / "allbad2"
        folder.mkdir()
        output = temp_dir / "out_bad2.000"
        (folder / "bad.000").write_bytes(_invalid_bytes())

        with patch("sys.argv", ["multifile", str(folder), "-o", str(output)]):
            code = main()

        assert code == 1
        stdout = capsys.readouterr().out
        assert "Error:" in stdout

    # -------------------------------------------------------------------------
    # Line 799: exit(main()) via __main__
    # -------------------------------------------------------------------------

    def test_main_block_success_exits_zero(self, temp_dir):
        """Running multifile.py as __main__ exits 0 on success (line 799)."""
        import inspect

        try:
            import pyadps.processing.multifile as _mf
        except ImportError:
            import multifile as _mf
        mf_path = inspect.getfile(_mf)

        folder = temp_dir / "main_ok"
        folder.mkdir()
        (folder / "data.000").write_bytes(_valid_ensemble(1))
        out = temp_dir / "main_out.000"

        proc = subprocess.run(
            [sys.executable, mf_path, str(folder), "-o", str(out)],
            capture_output=True,
        )
        assert proc.returncode == 0

    def test_main_block_failure_exits_one(self, temp_dir):
        """Running multifile.py as __main__ exits 1 on failure (line 799)."""
        import inspect

        try:
            import pyadps.processing.multifile as _mf
        except ImportError:
            import multifile as _mf
        mf_path = inspect.getfile(_mf)

        folder = temp_dir / "main_fail"
        folder.mkdir()  # empty → no files → failure
        out = temp_dir / "main_out_fail.000"

        proc = subprocess.run(
            [sys.executable, mf_path, str(folder), "-o", str(out)],
            capture_output=True,
        )
        assert proc.returncode == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
