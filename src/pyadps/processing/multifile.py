# pyadps/processing/multifile.py
"""
ADCP Multi-File Processing Module.

This module provides functionality for combining multiple ADCP binary files
into a single file. This is useful when deployments are split across multiple
files due to instrument memory limitations or data retrieval schedules.

Key Features:
- Validate ADCP binary file headers
- Handle corrupted or partial files gracefully
- Combine multiple files into a single binary file
- CLI interface for batch processing

Example usage:
    # Programmatic usage
    from pyadps.processing.multifile import combine_adcp_files

    success = combine_adcp_files(
        folder_path='raw_data/',
        output_file='combined.000'
    )

    # CLI usage
    python -m pyadps.processing.multifile raw_data/ -o combined.000 -v
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class ADCPFileConfig:
    """
    Configuration for ADCP binary file processing.

    Attributes
    ----------
    file_extension : str
        Glob pattern for ADCP files (default: "*.000").
    header_signature : bytes
        Two-byte header signature for ADCP files (0x7f7f).
    ensemble_size_offset : int
        Byte offset to ensemble size field in header.
    ensemble_size_length : int
        Length of ensemble size field in bytes.
    header_size_adjustment : int
        Adjustment to add to ensemble size from header.
    max_header_search : int
        Maximum bytes to search for header signature.
    """

    file_extension: str = "*.000"
    header_signature: bytes = b"\x7f\x7f"
    ensemble_size_offset: int = 2
    ensemble_size_length: int = 2
    header_size_adjustment: int = 2
    max_header_search: int = 10000  # Don't search entire file for header


# =============================================================================
# RESULT DATACLASSES
# =============================================================================


@dataclass
class FileValidationResult:
    """
    Result of validating an ADCP file.

    Attributes
    ----------
    is_valid : bool
        Whether the file passed validation.
    header_offset : int
        Byte offset where valid header was found (0 if at start).
    ensemble_size : int
        Size of each ensemble in bytes.
    valid_ensembles : int
        Number of complete, valid ensembles in file.
    total_ensembles : int
        Total ensembles (including partial).
    is_truncated : bool
        Whether file appears to be truncated.
    error_message : str
        Error message if validation failed.
    """

    is_valid: bool = False
    header_offset: int = 0
    ensemble_size: int = 0
    valid_ensembles: int = 0
    total_ensembles: int = 0
    is_truncated: bool = False
    error_message: str = ""


def validate_adcp_file(
    filepath: Union[str, Path],
    config: Optional[ADCPFileConfig] = None,
) -> FileValidationResult:
    """
    Validate an ADCP binary file.

    Parameters
    ----------
    filepath : str or Path
        Path to ADCP binary file.
    config : ADCPFileConfig, optional
        Configuration for file validation.

    Returns
    -------
    FileValidationResult
        Validation results including header offset, ensemble count, etc.

    Examples
    --------
    >>> result = validate_adcp_file('data.000')
    >>> if result.is_valid:
    ...     print(f"File has {result.valid_ensembles} ensembles")
    """
    config = config or ADCPFileConfig()
    filepath = Path(filepath)
    result = FileValidationResult()

    # Check file exists and is readable
    if not filepath.exists():
        result.error_message = f"File not found: {filepath}"
        return result

    if not filepath.is_file():
        result.error_message = f"Path is not a file: {filepath}"
        return result

    file_size = filepath.stat().st_size
    if file_size == 0:
        result.error_message = f"File is empty: {filepath}"
        return result

    # Read file and validate header
    try:
        with open(filepath, "rb") as f:
            data = f.read()
    except IOError as e:
        result.error_message = f"Error reading file: {e}"
        return result

    # Check for valid header at start
    if data.startswith(config.header_signature):
        result.header_offset = 0
    else:
        # Search for header signature within max_header_search bytes
        search_limit = min(len(data), config.max_header_search)
        header_index = data[:search_limit].find(config.header_signature)
        if header_index == -1:
            result.error_message = "No valid ADCP header found in file"
            return result
        result.header_offset = header_index
        logger.warning(
            f"Header found at byte {header_index} in {filepath.name}, "
            "data before header will be skipped"
        )

    # Calculate ensemble size from header
    valid_data = data[result.header_offset :]
    offset = config.ensemble_size_offset
    length = config.ensemble_size_length

    if len(valid_data) < offset + length:
        result.error_message = "File too small to contain valid header"
        return result

    result.ensemble_size = (
        int.from_bytes(valid_data[offset : offset + length], byteorder="little")
        + config.header_size_adjustment
    )

    if result.ensemble_size <= 0:
        result.error_message = f"Invalid ensemble size: {result.ensemble_size}"
        return result

    # Calculate number of valid ensembles
    valid_data_size = len(valid_data)
    result.total_ensembles = (
        valid_data_size + result.ensemble_size - 1
    ) // result.ensemble_size
    result.valid_ensembles = valid_data_size // result.ensemble_size

    if valid_data_size % result.ensemble_size != 0:
        result.is_truncated = True
        logger.warning(
            f"File {filepath.name} appears truncated: "
            f"{result.valid_ensembles}/{result.total_ensembles} complete ensembles"
        )

    result.is_valid = True
    return result


# =============================================================================
# FILE PROCESSING
# =============================================================================


def read_adcp_file(
    filepath: Union[str, Path],
    config: Optional[ADCPFileConfig] = None,
    validate: bool = True,
) -> bytes:
    """
    Read and extract valid data from an ADCP binary file.

    Parameters
    ----------
    filepath : str or Path
        Path to ADCP binary file.
    config : ADCPFileConfig, optional
        Configuration for file processing.
    validate : bool, default True
        Whether to validate file before reading.

    Returns
    -------
    bytes
        Valid ADCP data (complete ensembles only).
        Returns empty bytes if file is invalid.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If file has no valid ADCP header (when validate=True).

    Examples
    --------
    >>> data = read_adcp_file('data.000')
    >>> print(f"Read {len(data)} bytes")
    """
    config = config or ADCPFileConfig()
    filepath = Path(filepath)

    if validate:
        validation = validate_adcp_file(filepath, config)
        if not validation.is_valid:
            raise ValueError(f"Invalid ADCP file: {validation.error_message}")

        # Read only valid data
        with open(filepath, "rb") as f:
            f.seek(validation.header_offset)
            valid_bytes = validation.valid_ensembles * validation.ensemble_size
            return f.read(valid_bytes)
    else:
        # Read entire file without validation
        with open(filepath, "rb") as f:
            return f.read()


def find_adcp_files(
    folder_path: Union[str, Path],
    config: Optional[ADCPFileConfig] = None,
) -> List[Path]:
    """
    Find all ADCP files in a folder.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing ADCP files.
    config : ADCPFileConfig, optional
        Configuration with file extension pattern.

    Returns
    -------
    List[Path]
        Sorted list of ADCP file paths.

    Raises
    ------
    FileNotFoundError
        If folder does not exist.
    NotADirectoryError
        If path is not a directory.

    Examples
    --------
    >>> files = find_adcp_files('raw_data/')
    >>> print(f"Found {len(files)} ADCP files")
    """
    config = config or ADCPFileConfig()
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not folder_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")

    files = sorted(folder_path.glob(config.file_extension))

    if not files:
        logger.warning(f"No {config.file_extension} files found in {folder_path}")
    else:
        logger.info(f"Found {len(files)} ADCP files in {folder_path}")

    return files


# =============================================================================
# FILE COMBINING
# =============================================================================


@dataclass
class CombineResult:
    """
    Result of combining multiple ADCP files.

    Attributes
    ----------
    success : bool
        Whether combination was successful.
    files_processed : int
        Number of files successfully processed.
    files_total : int
        Total number of files attempted.
    total_bytes : int
        Total bytes in combined output.
    total_ensembles : int
        Total ensembles in combined output.
    output_path : Path
        Path to output file (if written).
    skipped_files : List[str]
        Names of files that were skipped due to errors.
    error_message : str
        Error message if combination failed.
    """

    success: bool = False
    files_processed: int = 0
    files_total: int = 0
    total_bytes: int = 0
    total_ensembles: int = 0
    output_path: Optional[Path] = None
    skipped_files: List[str] = field(default_factory=list)
    error_message: str = ""


def combine_adcp_files(
    folder_path: Union[str, Path],
    output_file: Union[str, Path],
    config: Optional[ADCPFileConfig] = None,
    skip_invalid: bool = True,
    require_matching_ensemble_size: bool = True,
) -> CombineResult:
    """
    Combine multiple ADCP files from a folder into a single file.

    Parameters
    ----------
    folder_path : str or Path
        Path to folder containing ADCP files.
    output_file : str or Path
        Path for combined output file.
    config : ADCPFileConfig, optional
        Configuration for file processing.
    skip_invalid : bool, default True
        If True, skip invalid files and continue. If False, stop on first error.
    require_matching_ensemble_size : bool, default True
        If True, all files must have the same ensemble size as the first valid file.
        Files with mismatched ensemble sizes are skipped (or cause error if skip_invalid=False).

    Returns
    -------
    CombineResult
        Results including success status, file counts, and any errors.

    Examples
    --------
    >>> result = combine_adcp_files('raw_data/', 'combined.000')
    >>> if result.success:
    ...     print(f"Combined {result.files_processed} files")
    ...     print(f"Output: {result.output_path} ({result.total_bytes} bytes)")

    >>> # With error handling
    >>> result = combine_adcp_files('raw_data/', 'combined.000', skip_invalid=False)
    >>> if not result.success:
    ...     print(f"Error: {result.error_message}")
    """
    config = config or ADCPFileConfig()
    result = CombineResult()

    # Find files
    try:
        files = find_adcp_files(folder_path, config)
    except (FileNotFoundError, NotADirectoryError) as e:
        result.error_message = str(e)
        return result

    result.files_total = len(files)

    if not files:
        result.error_message = f"No ADCP files found in {folder_path}"
        return result

    # Process each file
    combined_data = bytearray()
    total_ensembles = 0
    reference_ensemble_size = None  # Ensemble size from first valid file

    for filepath in files:
        try:
            validation = validate_adcp_file(filepath, config)

            if not validation.is_valid:
                if skip_invalid:
                    logger.warning(
                        f"Skipping {filepath.name}: {validation.error_message}"
                    )
                    result.skipped_files.append(filepath.name)
                    continue
                else:
                    result.error_message = (
                        f"Invalid file {filepath.name}: {validation.error_message}"
                    )
                    return result

            # Check ensemble size consistency
            if require_matching_ensemble_size:
                if reference_ensemble_size is None:
                    # First valid file sets the reference
                    reference_ensemble_size = validation.ensemble_size
                    logger.info(
                        f"Reference ensemble size: {reference_ensemble_size} bytes (from {filepath.name})"
                    )
                elif validation.ensemble_size != reference_ensemble_size:
                    msg = (
                        f"Ensemble size mismatch in {filepath.name}: "
                        f"expected {reference_ensemble_size}, got {validation.ensemble_size}"
                    )
                    if skip_invalid:
                        logger.warning(f"Skipping {filepath.name}: {msg}")
                        result.skipped_files.append(filepath.name)
                        continue
                    else:
                        result.error_message = msg
                        return result

            # Read valid data
            data = read_adcp_file(filepath, config, validate=False)

            # Extract only complete ensembles
            valid_bytes = validation.valid_ensembles * validation.ensemble_size
            combined_data.extend(
                data[validation.header_offset : validation.header_offset + valid_bytes]
            )

            total_ensembles += validation.valid_ensembles
            result.files_processed += 1

            logger.info(
                f"Processed {filepath.name}: {validation.valid_ensembles} ensembles"
            )

        except Exception as e:
            if skip_invalid:
                logger.error(f"Error processing {filepath.name}: {e}")
                result.skipped_files.append(filepath.name)
            else:
                result.error_message = f"Error processing {filepath.name}: {e}"
                return result

    # Check if we have any data
    if not combined_data:
        result.error_message = "No valid data extracted from files"
        return result

    # Write output file
    output_path = Path(output_file)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(combined_data)

        result.output_path = output_path
        result.total_bytes = len(combined_data)
        result.total_ensembles = total_ensembles
        result.success = True

        logger.info(
            f"Successfully combined {result.files_processed}/{result.files_total} files "
            f"({result.total_ensembles} ensembles, {result.total_bytes:,} bytes) "
            f"to {output_path}"
        )

    except IOError as e:
        result.error_message = f"Error writing output file: {e}"
        return result

    return result


def combine_file_list(
    files: List[Union[str, Path]],
    output_file: Union[str, Path],
    config: Optional[ADCPFileConfig] = None,
    skip_invalid: bool = True,
    require_matching_ensemble_size: bool = True,
) -> CombineResult:
    """
    Combine a specific list of ADCP files into a single file.

    Parameters
    ----------
    files : List[str or Path]
        List of ADCP file paths to combine.
    output_file : str or Path
        Path for combined output file.
    config : ADCPFileConfig, optional
        Configuration for file processing.
    skip_invalid : bool, default True
        If True, skip invalid files and continue.
    require_matching_ensemble_size : bool, default True
        If True, all files must have the same ensemble size as the first valid file.

    Returns
    -------
    CombineResult
        Results including success status, file counts, and any errors.

    Examples
    --------
    >>> files = ['data001.000', 'data002.000', 'data003.000']
    >>> result = combine_file_list(files, 'combined.000')
    """
    config = config or ADCPFileConfig()
    result = CombineResult()
    result.files_total = len(files)

    if not files:
        result.error_message = "No files provided"
        return result

    # Process each file
    combined_data = bytearray()
    total_ensembles = 0
    reference_ensemble_size = None

    for filepath in files:
        filepath = Path(filepath)

        try:
            validation = validate_adcp_file(filepath, config)

            if not validation.is_valid:
                if skip_invalid:
                    logger.warning(
                        f"Skipping {filepath.name}: {validation.error_message}"
                    )
                    result.skipped_files.append(filepath.name)
                    continue
                else:
                    result.error_message = (
                        f"Invalid file {filepath.name}: {validation.error_message}"
                    )
                    return result

            # Check ensemble size consistency
            if require_matching_ensemble_size:
                if reference_ensemble_size is None:
                    reference_ensemble_size = validation.ensemble_size
                    logger.info(
                        f"Reference ensemble size: {reference_ensemble_size} bytes (from {filepath.name})"
                    )
                elif validation.ensemble_size != reference_ensemble_size:
                    msg = (
                        f"Ensemble size mismatch in {filepath.name}: "
                        f"expected {reference_ensemble_size}, got {validation.ensemble_size}"
                    )
                    if skip_invalid:
                        logger.warning(f"Skipping {filepath.name}: {msg}")
                        result.skipped_files.append(filepath.name)
                        continue
                    else:
                        result.error_message = msg
                        return result

            # Read and append valid data
            with open(filepath, "rb") as f:
                f.seek(validation.header_offset)
                valid_bytes = validation.valid_ensembles * validation.ensemble_size
                combined_data.extend(f.read(valid_bytes))

            total_ensembles += validation.valid_ensembles
            result.files_processed += 1

            logger.info(
                f"Processed {filepath.name}: {validation.valid_ensembles} ensembles"
            )

        except Exception as e:
            if skip_invalid:
                logger.error(f"Error processing {filepath.name}: {e}")
                result.skipped_files.append(filepath.name)
            else:
                result.error_message = f"Error processing {filepath.name}: {e}"
                return result

    # Check if we have any data
    if not combined_data:
        result.error_message = "No valid data extracted from files"
        return result

    # Write output file
    output_path = Path(output_file)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(combined_data)

        result.output_path = output_path
        result.total_bytes = len(combined_data)
        result.total_ensembles = total_ensembles
        result.success = True

        logger.info(
            f"Successfully combined {result.files_processed}/{result.files_total} files "
            f"to {output_path}"
        )

    except IOError as e:
        result.error_message = f"Error writing output file: {e}"
        return result

    return result


# =============================================================================
# CLI INTERFACE
# =============================================================================


def setup_logging(verbosity: int) -> None:
    """
    Configure logging based on verbosity level.

    Parameters
    ----------
    verbosity : int
        Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG).
    """
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """
    Main entry point for CLI usage.

    Examples
    --------
    # Combine files in a folder
    python -m pyadps.processing.multifile raw_data/ -o combined.000

    # With verbose output
    python -m pyadps.processing.multifile raw_data/ -o combined.000 -v

    # With debug output
    python -m pyadps.processing.multifile raw_data/ -o combined.000 -vv
    """
    parser = argparse.ArgumentParser(
        description="Combine multiple ADCP binary files into a single file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s raw_data/ -o combined.000
  %(prog)s raw_data/ -o combined.000 -v
  %(prog)s raw_data/ -o combined.000 -vv --strict
        """,
    )

    parser.add_argument(
        "folder",
        type=str,
        help="Path to folder containing ADCP files (*.000).",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="combined.000",
        help="Output filename for combined data (default: combined.000).",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG).",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on first invalid file instead of skipping.",
    )

    parser.add_argument(
        "--no-size-check",
        action="store_true",
        help="Disable ensemble size consistency check between files.",
    )

    parser.add_argument(
        "--extension",
        type=str,
        default="*.000",
        help="File extension pattern (default: *.000).",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Create config with custom extension if provided
    config = ADCPFileConfig(file_extension=args.extension)

    # Run combination
    print(f"Combining ADCP files from: {args.folder}")
    print(f"Output file: {args.output}")
    print("-" * 50)

    result = combine_adcp_files(
        folder_path=args.folder,
        output_file=args.output,
        config=config,
        skip_invalid=not args.strict,
        require_matching_ensemble_size=not args.no_size_check,
    )

    # Print results
    print("-" * 50)

    if result.success:
        print(
            f"  Successfully combined {result.files_processed}/{result.files_total} files"
        )
        print(f"  Total ensembles: {result.total_ensembles:,}")
        print(f"  Total bytes: {result.total_bytes:,}")
        print(f"  Output: {result.output_path}")

        if result.skipped_files:
            print(f"\n Skipped {len(result.skipped_files)} files:")
            for name in result.skipped_files:
                print(f"    - {name}")
    else:
        print("Failed to combine files")
        print(f"Error: {result.error_message}")

        if result.skipped_files:
            print("\n  Skipped files:")
            for name in result.skipped_files:
                print(f"    - {name}")

        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    exit(main())
