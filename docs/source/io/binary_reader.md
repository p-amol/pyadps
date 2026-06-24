# binary_reader

High-level reader for RDI ADCP binary files. Returns `xarray.Dataset` objects
ready for analysis.

This is the primary entry point for loading ADCP data. It wraps {doc}`pd0_parser`
and automatically handles metadata extraction, coordinate assignment, and accessor
registration.

## Functions

| Function | Description |
|----------|-------------|
| `pyadps.read()` | Load complete dataset — primary entry point |
| `pyadps.read_header()` | File structure and data type mapping |
| `pyadps.read_fixed_leader()` | Static instrument configuration (36 raw + 25 decoded fields) |
| `pyadps.read_variable_leader()` | Per-ensemble sensor data (48 raw + 46 decoded + 2 composite fields) |
| `pyadps.read_velocity()` | Velocity array — shape `(beam, cell, time)`, units mm/s |
| `pyadps.read_correlation()` | Correlation array — shape `(beam, cell, time)`, range 0–255 |
| `pyadps.read_echo_intensity()` | Echo intensity array — range 0–255 (multiply by 0.45 for dB) |
| `pyadps.read_percent_good()` | Percent good array — range 0–100 |
| `pyadps.read_status()` | Status diagnostic bit flags |

## Usage

```python
import pyadps

# Load the complete dataset
ds = pyadps.read('deployment.000')

# Load specific components only — useful for large files
ds = pyadps.read('deployment.000', data_types=['Velocity', 'Correlation'])
```

**`data_types` options:** `'FixedLeader'`, `'VariableLeader'`, `'Velocity'`,
`'Correlation'`, `'Echo'`, `'PercentGood'`, `'Status'`.

`VariableLeader` is always included because it provides the time coordinate.

## Key Defaults

| Parameter | Default | Effect |
|-----------|---------|--------|
| `missing_as_nan` | `True` | RDI missing value (−32768) replaced with `NaN` |
| `include_mask` | `True` | A 3D QC mask variable is created from missing values |
| `include_decoded` | `True` | Decoded fields (frequency, heading in degrees, etc.) are included |
| `use_time_as_primary_dim` | `True` | Time is the first dimension |
| `use_depth_as_primary_dim` | `False` | Cell index used; set `True` for physical depth in metres |

## Time Utilities

Two time axis utilities are available for correcting irregular timestamps.
These are most conveniently accessed through `ProcessedDataset.apply_time_axis()`
in the processing module, but can also be called directly:

```python
from pyadps.processing.time_axis import snap_time_axis, fill_time_gaps

# Snap drifted timestamps to the nearest hour
ds_snapped, success, msg = snap_time_axis(ds, freq='h', tolerance='5min')

# Fill missing ensembles so the time axis is uniform
ds_filled = fill_time_gaps(ds_snapped, method='auto')
```

## See Also

- {doc}`pd0_parser` — Low-level binary parsing
- {doc}`accessors` — Domain-specific accessor methods
- {doc}`/processing/index` — Quality control pipeline
