# Processing Module

The `pyadps.processing` module provides a six-step quality control pipeline
for ADCP data. The primary interface is the `ProcessedDataset` class, which
orchestrates the full pipeline while keeping the original dataset untouched.

## Processing Workflow

Steps should be applied in this order. Any step can be skipped.

| Step | Method | Purpose |
|------|--------|---------|
| 1 | `apply_time_axis()` | Snap drifted timestamps; fill time gaps |
| 2 | `apply_sensor_health()` | Check tilt, correct sound speed |
| 3 | `apply_signal_quality()` | Correlation, echo, error velocity, percent good |
| 4 | `apply_profile_operation()` | Trim ends, cut bins, regrid |
| 5 | `apply_velocity_check()` | Thresholds, magnetic correction, despike |
| 6 | `finalize()` | Return processed `xarray.Dataset` with metadata |

```{note}
Profile operations (Step 4) must come **after** signal quality checks (Step 3).
Regridding changes the cell structure and invalidates cell-based masks from
earlier steps.
```

## Quick Example

```python
import pyadps
from pyadps.processing import ProcessedDataset

ds = pyadps.read('deployment.000')

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

## Module Overview

### Core

| Module | Description |
|--------|-------------|
| `core` | `ProcessedDataset` — orchestrates the full pipeline |
| `config` | `ProcessingConfig` — load/save processing settings as `config.ini` |

### Runner Classes

Each processing step delegates to a dedicated Runner class. You can access
them directly via `ProcessedDataset` for advanced control, or use them
standalone.

| Module | Runner Class | Handles |
|--------|-------------|---------|
| `sensor_health` | `SensorHealthRunner` | Roll, pitch, sound speed correction |
| `signal_quality` | `SignalQualityRunner` | Correlation, echo, error velocity, percent good, false target |
| `profile_operation` | `ProfileOperationRunner` | Trim, side-lobe cut, manual cut, regrid |
| `velocity_check` | `VelocityCheckRunner` | Thresholds, magnetic declination, despike, flatline |
| `time_axis` | — | Time snapping and gap filling utilities |

### Automation and Batch Processing

| Module | Key Function | Description |
|--------|-------------|-------------|
| `autoprocess` | `autoprocess()` | Reprocess a file using a saved `config.ini` |
| `multifile` | `combine_file_list()` | Merge multiple binary files into one |

### Utilities

| Module | Description |
|--------|-------------|
| `utility` | `QCCheckStats`, `DataModificationStats`, `QCPipelineReport` — statistics dataclasses |

## Config-Based Processing

Processing settings can be saved and replayed via `config.ini`:

```python
proc = ProcessedDataset(ds)
proc.apply_signal_quality(correlation=64, echo_intensity=40)

# Export settings after processing
proc.export_config('config.ini')

# Replay later
from pyadps.processing.autoprocess import autoprocess
result = autoprocess('config.ini', binary_file_path='deployment.000')
```

## Contents

```{toctree}
:maxdepth: 1

core
sensor_health
signal_quality
profile_operation
velocity_check
time_axis
autoprocess
multifile
config
utility
```

## See Also

- {doc}`/io/index` — Loading ADCP data
- {doc}`/api/index` — Full API reference
