# autoprocess

Provides `autoprocess()` — a single-function entry point that runs the complete
processing pipeline from a config file or `ProcessingConfig` object. This is the
function the web app's Write File page calls internally.

## Function

```python
from pyadps.processing.autoprocess import autoprocess

result = autoprocess(config_file_or_object, **options)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config_file_or_object` | — | Path to `config.ini` or a `ProcessingConfig` object |
| `binary_file_path` | `None` | ADCP binary file path; overrides the path stored in config |
| `save_netcdf` | `False` | Save processed dataset to NetCDF |
| `save_velocity_only` | `False` | Save only u/v/w velocity components instead of full dataset |
| `velocity_units` | `'cm/s'` | Output units: `'mm/s'`, `'cm/s'`, or `'m/s'` |
| `output_dir` | `None` | Output directory; defaults to same directory as input file |
| `output_filename` | `None` | Output filename; auto-generated if not specified |
| `ensure_depth_ascending` | `True` | Reorder depth dimension to be ascending in output |
| `print_summary` | `True` | Print processing summary to console |

Returns the processed `xarray.Dataset`.

## Usage

```python
from pyadps.processing.autoprocess import autoprocess

# Run from a saved config file
result = autoprocess('config.ini')

# Save full NetCDF output
result = autoprocess('config.ini', save_netcdf=True, output_dir='processed/')

# Save velocity-only output in SI units
result = autoprocess('config.ini', save_netcdf=True,
                     save_velocity_only=True, velocity_units='m/s')
```

## Programmatic Use

Pass a `ProcessingConfig` object instead of a file path when building scripts
that don't rely on a saved `config.ini`:

```python
from pyadps.processing.autoprocess import autoprocess
from pyadps.processing.config import ProcessingConfig

config = ProcessingConfig()
config.isQCTest = True
config.correlation_threshold = 64
config.echo_intensity_threshold = 40

result = autoprocess(
    config,
    binary_file_path='data/deployment.000',
    save_netcdf=True,
    output_dir='results/',
    print_summary=False,
)
```

## Non-Obvious Behaviors

**`binary_file_path` overrides the config.** If you supply it, the file path
stored inside the config is ignored — useful for re-using one config across
multiple deployments.

**Auto-generated filenames** follow the pattern `<input_stem>_processed.nc`
(full dataset) or `<input_stem>_velocity.nc` (velocity-only).

## See Also

- {doc}`config` — `ProcessingConfig` reference and INI file format
- {doc}`core` — `ProcessedDataset` for step-by-step control
- {doc}`multifile` — Combining multiple deployment files before processing
