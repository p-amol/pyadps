# profile_operation

Provides `ProfileOperationRunner` — the class that `ProcessedDataset.apply_profile_operation()`
delegates to. Use it directly for finer control over trimming, bin-cutting, and regridding.

## ProfileOperationRunner

```python
import pyadps
from pyadps.processing.profile_operation import ProfileOperationRunner

ds = pyadps.read('deployment.000')
runner = ProfileOperationRunner(ds)
```

### Operation Methods

All methods return `self` for chaining.

| Method | Description |
|--------|-------------|
| `trim_ensembles(start, end)` | Mask ensembles at the start and/or end of deployment |
| `cut_bins_side_lobe(orientation, water_depth, extra_cells=1)` | Mask cells contaminated by acoustic side-lobe interference |
| `cut_bins_manual(min_cell, max_cell, min_ensemble, max_ensemble)` | Mask a rectangular region by cell and ensemble index |
| `regrid(method='linear', depth_min, depth_max, depth_step)` | Interpolate from cell coordinates to a regular depth grid |

### Control and Output Methods

| Method | Description |
|--------|-------------|
| `reset()` | Restore working dataset to original state |
| `finalize()` | Return processed `xarray.Dataset` |
| `apply_pipeline(operations, order)` | Apply multiple operations from a configuration dict |

### Statistics Methods

| Method | Description |
|--------|-------------|
| `get_statistics()` | Dict of `QCCheckStats` keyed by operation name |
| `get_modifications()` | Dict of `DataModificationStats` keyed by variable name |
| `print_statistics()` | Print formatted operation summary table |
| `get_pipeline_report()` | Return `QCPipelineReport` for `ProcessedDataset` |

## Usage

```python
runner = ProfileOperationRunner(ds)
ds_processed = (runner
    .trim_ensembles(start=100, end=50)
    .cut_bins_side_lobe(extra_cells=2)
    .regrid(method='linear')
    .finalize())

runner.print_statistics()
```

## Non-Obvious Behaviors

**QC must be done before regridding.** Signal quality and velocity checks operate
on the cell-based coordinate system. Once `regrid()` converts the dataset from
`(beam, cell, time)` to `(beam, depth, time)`, mask-based operations can no longer
be applied. Always call `regrid()` last.

**Side-lobe cut for downward-looking ADCPs** requires `water_depth` to determine
the valid range. Upward-looking ADCPs auto-detect the range from the surface reflection.

```python
# Upward-looking: auto-detects orientation
runner.cut_bins_side_lobe(extra_cells=2)

# Downward-looking: supply water column depth in metres
runner.cut_bins_side_lobe(orientation='down', water_depth=150.0, extra_cells=2)
```

**Config-based use.** When running from a `config.ini`, `apply_pipeline()` applies
all operations in the specified order:

```python
runner.apply_pipeline(
    operations={
        'trim_ensembles': {'start': 100, 'end': 50},
        'cut_bins_side_lobe': {'extra_cells': 2},
        'regrid': {'method': 'linear'},
    },
    order=['trim_ensembles', 'cut_bins_side_lobe', 'regrid']
)
```

## See Also

- {doc}`core` — `ProcessedDataset.apply_profile_operation()` for the high-level interface
- {doc}`signal_quality` — Must be applied before regridding
- {doc}`velocity_check` — Applied after profile operations on non-regridded data
- {doc}`utility` — `QCCheckStats` reference
