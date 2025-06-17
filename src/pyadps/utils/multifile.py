"""
ADCP (Acoustic Doppler Current Profiler) File Processor
A clean, maintainable implementation for processing and combining ADCP binary files.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Union

# Import from our separate logging module
from .logging_utils import LogLevel, get_console_logger


@dataclass
class ADCPConfig:
    """Configuration for ADCP file processing"""

    file_extension: str = "*.000"
    header_signature: bytes = b"\x7f\x7f"
    header_signature_ext: bytes = b"\x7f\x7f\xf0\x02"
    ensemble_size_offset: int = 2
    ensemble_size_length: int = 2
    header_size_adjustment: int = 2
    chunk_size: int = 8192  # For large file processing


class ADCPError(Exception):
    """Base exception for ADCP processing errors"""

    pass


class InvalidHeaderError(ADCPError):
    """Raised when ADCP file has invalid header"""

    pass


class CorruptedFileError(ADCPError):
    """Raised when ADCP file is corrupted"""

    pass


class ADCPFileValidator:
    """Validates ADCP files and headers"""

    def __init__(self, config: ADCPConfig, logger_name: str = "adcp_validator"):
        self.config = config
        self.logger = get_console_logger(logger_name, LogLevel.INFO)

    def find_header_start(self, data: bytes) -> int:
        """Find the first occurrence of the extended header signature"""
        return data.find(self.config.header_signature_ext)

    def validate_file_path(self, filepath: Path) -> None:
        """Validate file path exists and is accessible"""
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} does not exist")
        if not filepath.is_file():
            raise ValueError(f"Path {filepath} is not a file")
        if filepath.stat().st_size == 0:
            raise ValueError(f"File {filepath} is empty")

    def has_valid_header(self, data: bytes) -> bool:
        """Check if data starts with valid ADCP header"""
        return data.startswith(self.config.header_signature)


class ADCPFileProcessor:
    """Processes individual ADCP files"""

    def __init__(self, config: ADCPConfig = None, logger_name: str = "adcp_processor"):
        self.config = config or ADCPConfig()
        self.validator = ADCPFileValidator(self.config, f"{logger_name}_validator")
        self.logger = get_console_logger(logger_name, LogLevel.INFO)

    def _calculate_ensemble_size(self, data: bytes) -> int:
        """Calculate size of single ensemble from header"""
        offset = self.config.ensemble_size_offset
        length = self.config.ensemble_size_length
        return (
            int.from_bytes(data[offset : offset + length], byteorder="little")
            + self.config.header_size_adjustment
        )

    def _validate_file_integrity(
        self, filepath: Path, data: bytes, ensemble_size: int
    ) -> int:
        """Validate file integrity and return number of valid ensembles"""
        file_size = filepath.stat().st_size
        if file_size % ensemble_size != 0:
            valid_ensembles = file_size // ensemble_size
            self.logger.warning(
                f"File {filepath.name} is corrupted. "
                f"Valid ensembles: {valid_ensembles}/{valid_ensembles + 1}"
            )
            return valid_ensembles
        return file_size // ensemble_size

    def process_file(self, filepath: Union[str, Path]) -> bytes:
        """Process a single ADCP file and return valid data"""
        filepath = Path(filepath)
        try:
            self.validator.validate_file_path(filepath)

            with open(filepath, "rb") as f:
                data = f.read()

            header_index = 0
            # Check if file starts with valid header
            if not self.validator.has_valid_header(data):
                header_index = self.validator.find_header_start(data)
                if header_index == -1:
                    raise InvalidHeaderError(
                        f"File {filepath.name} contains no valid ADCP header"
                    )
                self.logger.warning(
                    f"File {filepath.name} header found at byte {header_index}. "
                    "Truncating invalid data before header."
                )
            else:
                self.logger.info(f"Valid ADCP file: {filepath.name}")

            # Calculate ensemble size and validate file integrity
            ensemble_size = self._calculate_ensemble_size(data[header_index:])
            valid_ensembles = self._validate_file_integrity(
                filepath, data, ensemble_size
            )

            # Return only valid data
            end_index = header_index + (valid_ensembles * ensemble_size)
            return data[header_index:end_index]

        except (InvalidHeaderError, FileNotFoundError, ValueError) as e:
            self.logger.error(f"Error processing {filepath.name}: {e}")
            return b""
        except Exception as e:
            self.logger.error(f"Unexpected error processing {filepath.name}: {e}")
            return b""


class ADCPBinFileCombiner:
    """Combines or joins multiple ADCP files"""

    def __init__(self, config: ADCPConfig = None, logger_name: str = "adcp_combiner"):
        self.config = config or ADCPConfig()
        self.processor = ADCPFileProcessor(self.config, f"{logger_name}_processor")
        self.logger = get_console_logger(logger_name, LogLevel.INFO)

    def get_adcp_files(self, folder_path: Union[str, Path]) -> List[Path]:
        """Get all ADCP files from folder"""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder {folder_path} does not exist")
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Path {folder_path} is not a directory")

        files = sorted(folder_path.glob(self.config.file_extension))
        if not files:
            self.logger.error(
                f"No {self.config.file_extension} files found in {folder_path}"
            )
        return files

    def combine_files(self, files: List[Union[str, Path]]) -> bytearray:
        """Combine multiple ADCP files into single bytearray"""
        if not files:
            self.logger.warning("No files provided for combination")
            return bytearray()

        combined_data = bytearray()
        processed_count = 0

        for file_path in files:
            valid_data = self.processor.process_file(file_path)
            if valid_data:
                combined_data.extend(valid_data)
                processed_count += 1

        self.logger.info(f"Successfully combined {processed_count}/{len(files)} files")
        return combined_data

    def combine_folder(
        self, folder_path: Union[str, Path], output_file: Union[str, Path]
    ) -> bool:
        """Combine all ADCP files from folder and write to output file"""
        try:
            files = self.get_adcp_files(folder_path)
            if not files:
                self.logger.error("No valid files found to combine")
                return False

            combined_data = self.combine_files(files)
            if not combined_data:
                self.logger.error("No valid data to write")
                return False

            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                f.write(combined_data)

            self.logger.info(
                # f"Successfully combined {len(files)} files. "
                f"Output written to: {output_path} ({len(combined_data)} bytes)"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error combining folder {folder_path}: {e}")
            return False


def main():
    """Main entry point for CLI usage"""
    try:
        folder = input("Enter folder containing ADCP files (*.000): ").strip()
        if not folder:
            print("No folder specified. Exiting.")
            return

        output = input("Enter output filename (default: merged_000.000): ").strip()
        if not output:
            output = "merged_000.000"

        # Create combiner with custom logger configuration
        combiner = ADCPBinFileCombiner(logger_name="adcp_main")
        success = combiner.combine_folder(folder, output)

        if success:
            print(f"✅ Files successfully combined to {output}")
        else:
            print("❌ Failed to combine files. Check logs for details.")

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
