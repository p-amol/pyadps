# time_axis Module

Time axis handling and regularization for ADCP data.

## Overview

The `time_axis` module provides functions for correcting and regularizing ADCP 
time coordinates. It handles common issues like clock drift, irregular sampling, 
and time gaps.

## Quick Start

```python
from pyadps.io.binary_reader import snap_time_axis, fill_time_gaps
import pyadps

# Load data
ds = pyadps.read('deployment.000')

# Snap to regular hourly intervals
ds_snapped, success, message = snap_time_axis(ds, freq='h', tolerance='5min')

# Fill any gaps with NaN
ds_regular = fill_time_gaps(ds_snapped, method='h')
```

---

## Functions

### snap_time_axis()

Snap/round time coordinates to regular intervals.

```{function} snap_time_axis(ds, freq='h', tolerance='5min', target_minute=0)
Snap time coordinates to regular intervals.

:param ds: Dataset with time coordinate
:type ds: xarray.Dataset
:param freq: Target frequency ('h', 'min', '30min', etc.)
:type freq: str
:param tolerance: Maximum allowed correction
:type tolerance: str
:param target_minute: Target minute within hour (0-59)
:type target_minute: int
:returns: Tuple of (snapped_dataset, success, message)
:rtype: tuple
```

**Example:**

```python
# Snap to hourly
ds_snapped, success, msg = snap_time_axis(ds, freq='h', tolerance='5min')

if success:
    print("Time axis snapped successfully")
else:
    print(f"Snapping failed: {msg}")
```

---

### fill_time_gaps()

Fill missing timestamps in time series with NaN data.

```{function} fill_time_gaps(ds, method='h')
Fill gaps in time series.

:param ds: Dataset with time coordinate
:type ds: xarray.Dataset
:param method: Fill frequency ('h', 'min', '30min', etc.)
:type method: str
:returns: Dataset with filled gaps
:rtype: xarray.Dataset
```

**Example:**

```python
ds_filled = fill_time_gaps(ds, method='h')
```

---

## Common Workflows

### Regularize Time Series

```python
import pyadps
from pyadps.io.binary_reader import snap_time_axis, fill_time_gaps

# Load data
ds = pyadps.read('deployment.000')

# Check time regularity
interval = ds.variable_leader.get_time_interval()
print(f"Modal interval: {interval}")

# Snap and fill
ds_snapped, success, _ = snap_time_axis(ds, freq='h', tolerance='5min')
ds_regular = fill_time_gaps(ds_snapped, method='h')

# Export
ds_regular.to_netcdf('deployment_regular.nc')
```

---

## See Also

- {doc}`core` — ProcessedDataset time axis step
- {doc}`/io/binary_reader` — Time utilities in binary_reader
