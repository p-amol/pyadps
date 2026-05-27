# multifile Module

Tools for combining multiple ADCP binary files.

## Overview

The `multifile` module provides tools for combining multiple ADCP binary files 
into a single file. This is essential when deployments are split across multiple 
files due to instrument memory limitations, data retrieval schedules, or file 
size constraints.

**Key Features:**

- Validate ADCP binary file headers and structure
- Handle corrupted or truncated files gracefully
- Ensure ensemble size consistency across files
- Combine files with detailed progress reporting
- CLI interface for batch processing

## Quick Start

### Python API

```python
from pyadps.processing.multifile import combine_adcp_files

# Combine all .000 files in a folder
result = combine_adcp_files('raw_data/', 'combined.000')

if result.success:
    print(f"Combined {result.files_processed} files")
    print(f"Total ensembles: {result.total_ensembles}")
    print(f"Output: {result.output_path}")
```

### Command Line

```bash
# Basic usage
python -m pyadps.processing.multifile raw_data/ -o combined.000

# With verbose output
python -m pyadps.processing.multifile raw_data/ -o combined.000 -v

# Stop on first error (strict mode)
python -m pyadps.processing.multifile raw_data/ -o combined.000 --strict
```

---

## Core Concepts

### ADCP Binary File Structure

ADCP binary files consist of sequential **ensembles** (also called pings):

```
┌──────────────────────────────────────────────────────────────┐
│ File Structure                                                │
├──────────────────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│ │ Ensemble 1  │ │ Ensemble 2  │ │ Ensemble 3  │ ...          │
│ │             │ │             │ │             │              │
│ │ Header:     │ │ Header:     │ │ Header:     │              │
│ │ 0x7F 0x7F   │ │ 0x7F 0x7F   │ │ 0x7F 0x7F   │              │
│ │ Size bytes  │ │ Size bytes  │ │ Size bytes  │              │
│ │ Data...     │ │ Data...     │ │ Data...     │              │
│ └─────────────┘ └─────────────┘ └─────────────┘              │
└──────────────────────────────────────────────────────────────┘
```

Each ensemble:

- Starts with header signature `0x7F 0x7F`
- Contains size field at byte offset 2
- Has fixed size determined by instrument configuration

### Ensemble Size Consistency

Files from the same deployment **must have the same ensemble size**. Different 
ensemble sizes indicate:

- Different instrument configurations
- Different deployments
- File corruption

By default, `multifile` validates that all files have matching ensemble sizes.

---

## Function Reference

### combine_adcp_files()

Combine all ADCP files from a folder.

```{function} combine_adcp_files(folder_path, output_file, config=None, skip_invalid=True, require_matching_ensemble_size=True)
Combine ADCP files from a folder.

:param folder_path: Folder containing ADCP files
:type folder_path: str or Path
:param output_file: Output file path
:type output_file: str or Path
:param config: Custom configuration
:type config: ADCPFileConfig, optional
:param skip_invalid: Skip invalid files instead of stopping
:type skip_invalid: bool
:param require_matching_ensemble_size: Validate ensemble size consistency
:type require_matching_ensemble_size: bool
:returns: Result object with processing details
:rtype: CombineResult
```

**Returns CombineResult with:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Whether combination succeeded |
| `files_processed` | int | Number of files successfully processed |
| `files_total` | int | Total number of files attempted |
| `total_bytes` | int | Total bytes in output file |
| `total_ensembles` | int | Total ensembles in output |
| `output_path` | Path | Path to output file |
| `skipped_files` | List[str] | Names of skipped files |
| `error_message` | str | Error message if failed |

---

### combine_file_list()

Combine a specific list of files (not a folder).

```{function} combine_file_list(files, output_file, config=None, skip_invalid=True, require_matching_ensemble_size=True)
Combine a specific list of ADCP files.

:param files: List of file paths to combine
:type files: List[str or Path]
:param output_file: Output file path
:type output_file: str or Path
:param config: Custom configuration
:type config: ADCPFileConfig, optional
:param skip_invalid: Skip invalid files instead of stopping
:type skip_invalid: bool
:param require_matching_ensemble_size: Validate ensemble size consistency
:type require_matching_ensemble_size: bool
:returns: Result object with processing details
:rtype: CombineResult
```

**Example:**

```python
from pyadps.processing.multifile import combine_file_list

# Combine specific files in order
files = [
    'data/deploy_001.000',
    'data/deploy_002.000',
    'data/deploy_003.000',
]

result = combine_file_list(files, 'combined.000')
```

---

### validate_adcp_file()

Validate a single ADCP file without combining.

```{function} validate_adcp_file(filepath, config=None)
Validate a single ADCP file.

:param filepath: Path to ADCP file
:type filepath: str or Path
:param config: Custom configuration
:type config: ADCPFileConfig, optional
:returns: Validation result
:rtype: FileValidationResult
```

**Returns FileValidationResult with:**

| Field | Type | Description |
|-------|------|-------------|
| `is_valid` | bool | Whether file is valid |
| `header_offset` | int | Byte offset where header was found |
| `ensemble_size` | int | Size of each ensemble in bytes |
| `valid_ensembles` | int | Number of complete ensembles |
| `total_ensembles` | int | Total ensembles (including partial) |
| `is_truncated` | bool | Whether file appears truncated |
| `error_message` | str | Error message if invalid |

**Example:**

```python
from pyadps.processing.multifile import validate_adcp_file

result = validate_adcp_file('data.000')

if result.is_valid:
    print(f"File has {result.valid_ensembles} complete ensembles")
    print(f"Ensemble size: {result.ensemble_size} bytes")
    if result.is_truncated:
        print("Warning: File is truncated")
else:
    print(f"Invalid file: {result.error_message}")
```

---

## Complete Workflow Examples

### Basic Combination

```python
from pyadps.processing.multifile import combine_adcp_files

# Combine all files from deployment folder
result = combine_adcp_files('deployment_2024/', 'deployment_2024_combined.000')

if result.success:
    print(f"Successfully combined {result.files_processed} files")
    print(f"Total ensembles: {result.total_ensembles}")
    print(f"Output size: {result.total_bytes / 1e6:.2f} MB")
else:
    print(f"Combination failed: {result.error_message}")
```

### With Validation

```python
from pyadps.processing.multifile import validate_adcp_file, combine_adcp_files
from pathlib import Path

# Validate all files first
input_dir = Path('raw_data/')
valid_files = []

for file in sorted(input_dir.glob('*.000')):
    validation = validate_adcp_file(file)
    if validation.is_valid:
        valid_files.append(file)
        print(f"✓ {file.name}: {validation.valid_ensembles} ensembles")
    else:
        print(f"✗ {file.name}: {validation.error_message}")

print(f"\n{len(valid_files)} valid files found")

# Combine valid files
if valid_files:
    result = combine_adcp_files(input_dir, 'combined.000')
    print(f"Combined output: {result.output_path}")
```

### Handling Truncated Files

```python
from pyadps.processing.multifile import combine_adcp_files

# Skip invalid/truncated files automatically
result = combine_adcp_files(
    'raw_data/',
    'combined.000',
    skip_invalid=True  # Default behavior
)

if result.skipped_files:
    print(f"Warning: Skipped {len(result.skipped_files)} files:")
    for f in result.skipped_files:
        print(f"  - {f}")
```

### Strict Mode (Stop on Error)

```python
from pyadps.processing.multifile import combine_adcp_files

# Stop immediately if any file is invalid
try:
    result = combine_adcp_files(
        'raw_data/',
        'combined.000',
        skip_invalid=False  # Strict mode
    )
except ValueError as e:
    print(f"Processing stopped: {e}")
```

---

## Command Line Interface

### Basic Usage

```bash
python -m pyadps.processing.multifile INPUT_DIR -o OUTPUT_FILE
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output file path (required) |
| `--verbose` | `-v` | Enable verbose output |
| `--strict` | | Stop on first error |
| `--pattern` | `-p` | File pattern (default: *.000) |

### Examples

```bash
# Basic combination
python -m pyadps.processing.multifile raw_data/ -o combined.000

# Verbose output
python -m pyadps.processing.multifile raw_data/ -o combined.000 -v

# Strict mode
python -m pyadps.processing.multifile raw_data/ -o combined.000 --strict

# Custom file pattern
python -m pyadps.processing.multifile raw_data/ -o combined.000 -p "*.pd0"
```

---

## Best Practices

1. **Validate before combining**: Run validation on all files first to identify 
   problems before starting the combination.

2. **Maintain file order**: Files are combined in alphabetical order by default. 
   Use `combine_file_list()` if you need a specific order.

3. **Check for truncation**: Truncated files at the end of a deployment are 
   common — the module handles these gracefully.

4. **Verify ensemble consistency**: Ensure all files have the same ensemble size 
   before combining.

5. **Keep originals**: Always keep the original files until you've verified the 
   combined output.

---

## Troubleshooting

### Ensemble size mismatch error

- Files may be from different deployments
- One file may be corrupted
- Use `validate_adcp_file()` to check individual files

### No files found

- Check the file extension pattern (default: `*.000`)
- Verify the directory path is correct
- Use the `-p` option to specify a different pattern

### Output file already exists

- The module will overwrite existing files
- Rename or move existing files before running

### Memory issues with large files

- Process files in smaller batches
- Use `combine_file_list()` with subsets of files

---

## See Also

- {doc}`autoprocess` — Automated processing
- {doc}`/io/pd0_parser` — Low-level binary parsing
