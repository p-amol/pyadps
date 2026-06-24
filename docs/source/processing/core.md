# core

The `ProcessedDataset` class orchestrates the six-step ADCP processing pipeline.
It maintains an immutable copy of the original dataset and delegates each step
to a dedicated Runner class, so the original data is never modified.

## ProcessedDataset

```python
import pyadps
from pyadps.processing import ProcessedDataset

ds = pyadps.read('deployment.000')
proc = ProcessedDataset(ds)
```

### Pipeline Methods

All pipeline methods return `self`, enabling method chaining.

| Method | Description |
|--------|-------------|
| `apply_time_axis()` | Snap drifted timestamps; fill time gaps |
| `apply_sensor_health()` | Roll/pitch checks; sound speed correction |
| `apply_signal_quality()` | Correlation, echo, error velocity, percent good, false target |
| `apply_profile_operation()` | Trim ends, side-lobe cut, manual cut, regrid |
| `apply_velocity_check()` | Thresholds, magnetic correction, despike, flatline |
| `apply_attributes(attributes)` | Add custom metadata to the dataset |
| `apply_config(config)` | Apply all steps from a `config.ini` file or `ProcessingConfig` object |
| `finalize()` | Return the processed `xarray.Dataset` with metadata |

### Output Methods

| Method | Description |
|--------|-------------|
| `to_netcdf(filepath)` | Finalise and save the full dataset |
| `velocity_to_netcdf(filepath, units='cm/s')` | Save u/v/w velocity components only |
| `save_netcdf(config, ...)` | Finalise and save using paths from a config |
| `export_config(filepath)` | Save all applied settings to `config.ini` |
| `export_config_string()` | Return settings as an INI-formatted string |

### Utility Methods

| Method | Description |
|--------|-------------|
| `reset()` | Restore working dataset to the original state |
| `get_current_stats()` | Current mask statistics (total, valid, masked counts) |
| `print_summary()` | Print a formatted QC pipeline summary |

### Class Methods

| Method | Description |
|--------|-------------|
| `from_file(config, binary_file_path)` | Read a binary file and return an initialised `ProcessedDataset` |
| `from_ini(filepath)` | Shorthand for `from_file` using an INI path |

## Chained Workflow

```python
result = (
    ProcessedDataset(ds)
    .apply_time_axis(snap=True, snap_freq='h')
    .apply_sensor_health(roll=True, roll_threshold=15.0)
    .apply_signal_quality(correlation=64, echo_intensity=40)
    .apply_profile_operation(cut_bins_side_lobe=True, water_depth=50.0)
    .apply_velocity_check(cutoff_u=2500, cutoff_v=2500, cutoff_w=500)
    .finalize()
)

result.to_netcdf('processed.nc')
```

## Config-Based Workflow

```python
# Apply all steps from a saved config
proc = ProcessedDataset(ds)
proc.apply_config('config.ini').finalize()

# Or read file and config in one step
proc = ProcessedDataset.from_ini('config.ini')
proc.apply_config(proc.config)
proc.export_config('run_record.ini')
```

## CutRegion

For manual bin cutting, `CutRegion` defines a rectangular region by cell
and ensemble index bounds. `None` means no limit on that dimension.

```python
from pyadps.processing.core import CutRegion

proc.apply_profile_operation(
    cut_bins_manual=[
        CutRegion(min_cell=0, max_cell=3),            # First 4 cells, all ensembles
        CutRegion(min_ensemble=0, max_ensemble=50),   # First 50 ensembles, all cells
    ]
)
```

## See Also

- {doc}`sensor_health` — SensorHealthRunner details
- {doc}`signal_quality` — SignalQualityRunner details
- {doc}`profile_operation` — ProfileOperationRunner details
- {doc}`velocity_check` — VelocityCheckRunner details
- {doc}`config` — ProcessingConfig reference
