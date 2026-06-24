# time_axis

Provides two functions for correcting irregular ADCP timestamps. These are most
conveniently called via `ProcessedDataset.apply_time_axis()`, but can also be
used directly.

## Functions

| Function | Description |
|----------|-------------|
| `snap_time_axis(ds, freq, tolerance, target_minute)` | Round timestamps to the nearest regular interval |
| `fill_time_gaps(ds, method, forward_fill_fixed_leader, forward_fill_variable_leader)` | Reindex to a regular grid, inserting NaN ensembles for gaps |

## Usage

```python
import pyadps
from pyadps.processing.time_axis import snap_time_axis, fill_time_gaps

ds = pyadps.read('deployment.000')

# Snap drifted timestamps to the nearest hour
ds_snapped, success, msg = snap_time_axis(ds, freq='h', tolerance='5min')

# Fill missing ensembles so the time axis is uniform
ds_filled = fill_time_gaps(ds_snapped, method='auto')
```

Via `ProcessedDataset`:

```python
from pyadps.processing import ProcessedDataset

proc = ProcessedDataset(ds)
proc.apply_time_axis(snap=True, snap_freq='h', snap_tolerance='5min',
                     fill_gaps=True, fill_method='auto')
```

## Non-Obvious Behaviors

**`snap_time_axis()` returns a tuple `(dataset, success, message)`.** If any
timestamp requires a correction larger than `tolerance`, the function aborts and
returns `(None, False, message)` rather than silently over-correcting.

```python
ds_snapped, success, msg = snap_time_axis(ds, freq='h', tolerance='5min')
if not success:
    print(f"Snapping aborted: {msg}")
```

**`fill_time_gaps()` forward-fills leader variables.** Fixed and variable leader
fields (configuration metadata, sensor readings) are propagated from the last
known ensemble into gap-filled slots. Velocity, correlation, and echo data are
filled with RDI missing value codes.

**`method='auto'`** in `fill_time_gaps()` detects the interval from the median
spacing of existing timestamps — no need to specify frequency explicitly if the
data has a consistent sampling rate.

## See Also

- {doc}`core` — `ProcessedDataset.apply_time_axis()` for the high-level interface
- {doc}`/io/accessors` — `ds.variable_leader.get_time_interval()` and `is_time_regular()` for diagnosing the time axis before correction
