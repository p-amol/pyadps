# profile_operation Module

Profile-level operations for ADCP data processing.

## Overview

The `profile_operation` module provides the `ProfileOperationRunner` class for 
performing profile-level operations on ADCP data. It handles mask-based operations 
(trimming, bin cutting) and dataset modifications (regridding) while tracking 
statistics and processing history.

```{warning}
**Critical Sequencing Rule:** Quality control checks (signal quality, velocity 
checks) must be performed BEFORE regridding, as regridding changes the dataset 
structure from cells to depths, making cell-based masks incompatible.
```

## Quick Start

```python
from pyadps.io import read
from pyadps.processing.profile_operation import ProfileOperationRunner

# Read ADCP data
ds = read("your_file.000")

# Apply profile operations with method chaining
runner = ProfileOperationRunner(ds)
ds_processed = (
    runner
    .trim_ensembles(start=10, end=5)
    .cut_bins_side_lobe(extra_cells=2)
    .cut_bins_manual(min_cell=0, max_cell=2)
    .regrid(method='linear')
    .finalize()
)

# View processing statistics
runner.print_statistics()
```

## Installation

```python
from pyadps.processing.profile_operation import ProfileOperationRunner
```

---

## Key Concepts

### Operation Types

Profile operations fall into two categories:

| Type | Operations | Effect |
|------|------------|--------|
| **Mask-based** | `trim_ensembles`, `cut_bins_side_lobe`, `cut_bins_manual` | Updates mask only, preserves data structure |
| **Dataset modification** | `regrid` | Transforms dataset structure (cells → depth levels) |

### Processing Order

The recommended processing order is:

1. **Time axis correction** (time_axis module)
2. **Sensor health checks** (sensor_health module)
3. **Signal quality checks** (signal_quality module) ← **Must be BEFORE regridding**
4. **Profile operations** (this module):
   - Trim ensembles
   - Cut bins (side lobe, manual)
   - Regrid ← **Final step**
5. **Velocity checks** (velocity_check module) — if on original cells
6. **Output** (to NetCDF)

### Mask Convention

The module uses a binary mask where:

- **0 = Valid data** (included in analysis)
- **1 = Invalid/flagged data** (excluded from analysis)

### Immutability

The original dataset is never modified:

- `runner.original` — Immutable copy of input dataset
- `runner.dataset` — Working copy that accumulates changes

### Missing Value Convention

After regridding, all invalid/masked data is represented as `np.nan` for 
consistency across all variables.

---

## Available Operations

### trim_ensembles()

Mask ensembles (time steps) at the beginning and/or end of deployment. Used 
to remove data during instrument deployment and recovery.

```{function} trim_ensembles(start=None, end=None)
Mask ensembles at start and/or end of deployment.

:param start: Number of ensembles to mask from start
:type start: int, optional
:param end: Number of ensembles to mask from end
:type end: int, optional
:returns: self (for method chaining)
:rtype: ProfileOperationRunner
```

**Flagging behavior:** Marks all beams and cells as invalid for the specified ensembles.

**Example:**

```python
# Mask deployment period (first 100 ensembles)
runner.trim_ensembles(start=100)

# Mask recovery period (last 50 ensembles)
runner.trim_ensembles(end=50)

# Mask both
runner.trim_ensembles(start=100, end=50)
```

---

### cut_bins_side_lobe()

Mask cells contaminated by acoustic side-lobe interference from the surface 
(upward-looking) or bottom (downward-looking).

```{function} cut_bins_side_lobe(orientation=None, water_depth=None, extra_cells=1)
Remove side-lobe contaminated cells.

:param orientation: Beam direction ('up' or 'down'). Auto-detected if None.
:type orientation: str, optional
:param water_depth: Water column depth in meters (required for downward ADCP)
:type water_depth: float, optional
:param extra_cells: Additional cells to mask beyond calculated limit
:type extra_cells: int
:returns: self (for method chaining)
:rtype: ProfileOperationRunner
```

**Flagging behavior:** For each ensemble, calculates the valid cell range based 
on transducer depth and beam angle, then masks cells beyond that range.

**Example:**

```python
# Upward-looking ADCP (surface interference)
runner.cut_bins_side_lobe(orientation='up', extra_cells=2)

# Downward-looking ADCP (bottom interference)
runner.cut_bins_side_lobe(orientation='down', water_depth=100.0, extra_cells=2)

# Auto-detect orientation from dataset
runner.cut_bins_side_lobe(extra_cells=1)
```

---

### cut_bins_manual()

Mask a rectangular region defined by cell and ensemble ranges.

```{function} cut_bins_manual(min_cell=0, max_cell=None, min_ensemble=0, max_ensemble=None)
Manually mask a region of cells and ensembles.

:param min_cell: Minimum cell index (inclusive)
:type min_cell: int
:param max_cell: Maximum cell index (exclusive)
:type max_cell: int, optional
:param min_ensemble: Minimum ensemble index (inclusive)
:type min_ensemble: int
:param max_ensemble: Maximum ensemble index (exclusive)
:type max_ensemble: int, optional
:returns: self (for method chaining)
:rtype: ProfileOperationRunner
```

**Flagging behavior:** Marks all beams as invalid within the specified rectangular region.

**Example:**

```python
# Mask first 5 cells (near-transducer interference)
runner.cut_bins_manual(min_cell=0, max_cell=5)

# Mask cells 20-40 during ensembles 1000-2000
runner.cut_bins_manual(min_cell=20, max_cell=40, min_ensemble=1000, max_ensemble=2000)
```

---

### regrid()

Transform data from irregular instrument cells to a regular depth grid.

```{function} regrid(method='linear', depth_min=None, depth_max=None, depth_step=None)
Regrid data to regular depth levels.

:param method: Interpolation method ('linear' or 'nearest')
:type method: str
:param depth_min: Minimum depth for output grid (meters)
:type depth_min: float, optional
:param depth_max: Maximum depth for output grid (meters)
:type depth_max: float, optional
:param depth_step: Depth step for output grid (meters)
:type depth_step: float, optional
:returns: self (for method chaining)
:rtype: ProfileOperationRunner
```

```{warning}
After regridding:
- Dataset structure changes from `(beam, cell, time)` to `(beam, depth, time)`
- Mask-based operations can no longer be applied
- All masked data is represented as `np.nan`
```

**Example:**

```python
# Default regridding (auto-detect depth range)
runner.regrid(method='linear')

# Custom depth grid
runner.regrid(method='linear', depth_min=5.0, depth_max=100.0, depth_step=2.0)

# Nearest-neighbor interpolation (preserves original values)
runner.regrid(method='nearest')
```

---

## Pipeline Methods

### apply_pipeline()

Apply multiple operations in one call using a configuration dictionary:

```{function} apply_pipeline(operations=None, order=None)
Apply multiple profile operations from configuration.

:param operations: Dictionary of operation names to parameter dictionaries
:type operations: dict, optional
:param order: List specifying operation order
:type order: list, optional
:returns: self (for method chaining)
:rtype: ProfileOperationRunner
```

**Example:**

```python
runner.apply_pipeline(
    operations={
        "trim_ensembles": {"start": 100, "end": 50},
        "cut_bins_side_lobe": {"extra_cells": 2},
        "regrid": {"method": "linear"},
    },
    order=["trim_ensembles", "cut_bins_side_lobe", "regrid"]
)
```

### reset()

Reset to original state:

```python
runner.reset()
```

### finalize()

Complete processing and return the dataset:

```python
ds_processed = runner.finalize()
```

---

## Statistics and Reporting

### get_statistics()

Get operation statistics as a dictionary:

```python
stats = runner.get_statistics()
trim_stats = stats["Trim Ensembles"]
print(f"Ensembles trimmed: {trim_stats.cells_newly_masked}")
```

### print_statistics()

Print a formatted statistics table:

```python
runner.print_statistics()
```

### export_statistics_dict()

Export all statistics as a JSON-serializable dictionary:

```python
import json

stats_dict = runner.export_statistics_dict()
with open("profile_stats.json", "w") as f:
    json.dump(stats_dict, f, indent=2)
```

---

## Complete Workflow Examples

### Basic Profile Operations

```python
from pyadps.io import read
from pyadps.processing.profile_operation import ProfileOperationRunner

# Read data
ds = read("mooring_adcp.000")

# Apply profile operations
runner = ProfileOperationRunner(ds)
ds_processed = (
    runner
    .trim_ensembles(start=100, end=50)
    .cut_bins_side_lobe(extra_cells=2)
    .finalize()
)

runner.print_statistics()
ds_processed.to_netcdf("mooring_trimmed.nc")
```

### Full Pipeline with Regridding

```python
from pyadps.io import read
from pyadps.processing.signal_quality import SignalQualityRunner
from pyadps.processing.profile_operation import ProfileOperationRunner

# Read data
ds = read("deployment.000")

# Step 1: Apply signal quality QC FIRST
sq_runner = SignalQualityRunner(ds)
ds_qc = (
    sq_runner
    .correlation(cutoff=64)
    .echo_intensity(cutoff=40)
    .finalize()
)

# Step 2: Apply profile operations including regridding
po_runner = ProfileOperationRunner(ds_qc)
ds_final = (
    po_runner
    .trim_ensembles(start=100, end=50)
    .cut_bins_side_lobe(extra_cells=2)
    .regrid(method='linear')  # Must be last!
    .finalize()
)

# Save results
ds_final.to_netcdf("deployment_processed.nc")
```

### Upward vs Downward Looking ADCP

```python
# Upward-looking ADCP (deployed on bottom, looking up)
runner = ProfileOperationRunner(ds)
ds_processed = (
    runner
    .cut_bins_side_lobe(orientation='up', extra_cells=2)
    .finalize()
)

# Downward-looking ADCP (deployed at surface, looking down)
runner = ProfileOperationRunner(ds)
ds_processed = (
    runner
    .cut_bins_side_lobe(orientation='down', water_depth=150.0, extra_cells=2)
    .finalize()
)
```

---

## Best Practices

1. **Apply QC before regridding**: Signal quality and velocity checks must be 
   done before regridding changes the coordinate system.

2. **Regrid as the final step**: Once regridded, cell-based operations are no 
   longer possible.

3. **Include extra_cells margin**: Always include at least 1-2 extra cells when 
   removing side-lobe contamination.

4. **Check orientation**: Verify the ADCP orientation is correctly detected or 
   specify it explicitly.

5. **Verify trim ranges**: Before finalizing, check that trim values don't remove 
   too much valid data.

---

## Troubleshooting

### Regridding produces all NaN

- Check that there is valid data after masking operations
- Verify depth coordinates are reasonable
- Check for missing transducer depth or beam angle values

### Side lobe calculation seems wrong

- Verify `orientation` is set correctly
- For downward ADCPs, ensure `water_depth` is provided
- Check that beam angle is available in the dataset

### Trim indices out of range

- Ensemble indices are 0-based
- Verify the total number of ensembles in your dataset

---

## See Also

- {doc}`signal_quality` — Signal quality assessment (must be done before regridding)
- {doc}`velocity_check` — Velocity validation
- {doc}`core` — ProcessedDataset orchestrator
