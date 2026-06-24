# multifile

Provides tools for combining multiple ADCP binary files from a single deployment
into one file before processing. Files are appended in the order they are found
(alphabetical by default).

## Functions

| Function | Description |
|----------|-------------|
| `combine_adcp_files(folder_path, output_file, ...)` | Combine all ADCP files found in a folder |
| `combine_file_list(files, output_file, ...)` | Combine a specific ordered list of files |
| `validate_adcp_file(filepath)` | Check a single file for structural validity |

## Usage

```python
from pyadps.processing.multifile import combine_adcp_files, combine_file_list

# Combine all *.000 files in a folder (alphabetical order)
result = combine_adcp_files('raw_data/', 'combined.000')

if result.success:
    print(f"Combined {result.files_processed} files, {result.total_ensembles} ensembles")
else:
    print(result.error_message)

# Combine a specific ordered list
result = combine_file_list(
    ['data/deploy_001.000', 'data/deploy_002.000', 'data/deploy_003.000'],
    'combined.000'
)
```

### Validate before combining

```python
from pyadps.processing.multifile import validate_adcp_file
from pathlib import Path

for f in sorted(Path('raw_data/').glob('*.000')):
    v = validate_adcp_file(f)
    status = 'OK' if v.is_valid else v.error_message
    truncated = ' (truncated)' if v.is_truncated else ''
    print(f"{f.name}: {status}{truncated}")
```

### Command line

```bash
python -m pyadps.processing.multifile raw_data/ -o combined.000
python -m pyadps.processing.multifile raw_data/ -o combined.000 --strict  # stop on first error
python -m pyadps.processing.multifile raw_data/ -o combined.000 -p "*.pd0"  # custom extension
```

## Return Objects

**`CombineResult`** (returned by `combine_adcp_files` and `combine_file_list`):

| Field | Description |
|-------|-------------|
| `success` | Whether the combination completed |
| `files_processed` | Number of files successfully included |
| `total_ensembles` | Total ensembles in output |
| `output_path` | Path to the output file |
| `skipped_files` | List of filenames that were skipped |
| `error_message` | Populated if `success` is `False` |

**`FileValidationResult`** (returned by `validate_adcp_file`):

| Field | Description |
|-------|-------------|
| `is_valid` | Whether the file has a valid ADCP structure |
| `valid_ensembles` | Number of complete ensembles found |
| `ensemble_size` | Bytes per ensemble |
| `is_truncated` | Last ensemble is incomplete |
| `error_message` | Populated if `is_valid` is `False` |

## Non-Obvious Behaviors

**Ensemble size must match across all files.** All files in a deployment should
have the same ensemble size (bytes per ensemble). A mismatch means different
instrument configurations or a corrupt file — combining them would produce
unreadable output. The check is enabled by default; pass
`require_matching_ensemble_size=False` only if you are certain the mismatch is
benign.

**Truncated files are skipped by default** (`skip_invalid=True`). A truncated
file at the end of a deployment is common when the instrument was recovered
mid-ensemble. The partial last ensemble is dropped and the rest of the file is
included.

**File order matters.** Use `combine_file_list()` with an explicitly ordered
list if alphabetical sorting does not match chronological order.

## See Also

- {doc}`autoprocess` — Run the full pipeline on a combined file
- {doc}`core` — `ProcessedDataset` for step-by-step processing
