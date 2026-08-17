# velocity_check

Provides `VelocityCheckRunner` — the class that `ProcessedDataset.apply_velocity_check()`
delegates to. Use it directly for finer control over individual checks or to inspect
per-check statistics.

## VelocityCheckRunner

```python
import pyadps
from pyadps.processing.velocity_check import VelocityCheckRunner

ds = pyadps.read('deployment.000')
runner = VelocityCheckRunner(ds)
```

### QC Check Methods

All methods return `self` for chaining.

| Method | Default | Description |
|--------|---------|-------------|
| `magnetic_correction(declination, lat, lon, year, use_api)` | — | Rotate U/V to correct for magnetic declination |
| `trim_depths(depths, apply_to_all_variables=False)` | — | Manually mask specific depth bins across every ensemble (requires a regridded dataset) |
| `threshold(cutoff_u=2500, cutoff_v=2500, cutoff_w=500)` | mm/s | Flag cells exceeding per-component velocity limits |
| `despike(kernel_size=7, cutoff=6.0)` | σ=6 | Flag transient spikes using a median-filter approach |
| `flatline(kernel_size=4, cutoff=1.0)` | 1 mm/s | Flag constant-value segments (frozen sensor) |

`ProcessedDataset.apply_velocity_check()` applies checks in this order:
magnetic correction, depth trim, threshold, despike, flatline - matching
the Velocity Processing page's tab order.

### Control and Output Methods

| Method | Description |
|--------|-------------|
| `reset()` | Restore working dataset to original state |
| `finalize()` | Return processed `xarray.Dataset` |
| `apply_pipeline(checks, order)` | Apply multiple checks from a configuration dict |

### Statistics Methods

| Method | Description |
|--------|-------------|
| `get_statistics()` | Dict of `QCCheckStats` keyed by check name |
| `get_modifications()` | Dict of `DataModificationStats` for data corrections |
| `print_statistics()` | Print formatted QC summary table |
| `get_pipeline_report()` | Return `QCPipelineReport` for `ProcessedDataset` |

## Usage

```python
runner = VelocityCheckRunner(ds)
ds_qc = (runner
    .magnetic_correction(declination=-5.0)
    .trim_depths(depths=[12.0])
    .threshold(cutoff_u=2500, cutoff_v=2500, cutoff_w=500)
    .despike(kernel_size=7, cutoff=6.0)
    .flatline(kernel_size=4, cutoff=1.0)
    .finalize())

runner.print_statistics()
```

## Non-Obvious Behaviors

**`magnetic_correction()` modifies data values, not the mask.** It rotates U and V
to true geographic coordinates. Apply it before any QC checks so that thresholds
apply to geographically-correct velocities.

**Declination sources.** Provide `declination` directly, or supply `lat`, `lon`,
and `year` to calculate it from local COF files. Set `use_api=True` for NOAA online
lookup.

```python
# Known declination
runner.magnetic_correction(declination=-5.0)

# Calculate from position (requires pygeomag)
runner.magnetic_correction(lat=40.0, lon=-74.0, year=2023.5)
```

**W threshold is asymmetric by design.** Vertical velocities are physically 5–10×
smaller than horizontal, so the default `cutoff_w=500` mm/s is already more
restrictive than the horizontal defaults of 2500 mm/s.

**Beam 3 is a combined flag.** After every check, beam 3 in the mask is set to the
OR of beams 0, 1, and 2 — a cell is flagged in beam 3 if any velocity component fails.

**`trim_depths()` requires a regridded dataset.** It operates on the `depth`
coordinate produced by `regrid()` (see {doc}`profile_operation`), not the
native `cell` index — it raises `ValueError` if the dataset hasn't been
regridded yet. It exists because `cut_bins_side_lobe()`'s geometric cutoff
is computed from `transducer_depth`, which drifts over a deployment, so a
boundary depth bin can end up only *partially* masked even when the
underlying contamination (e.g. surface backscatter) is present throughout —
this lets you manually finish the job after visually confirming which
depths are actually affected.

**`trim_depths()` masks velocity only by default**, matching every other
check in this module — `apply_to_all_variables=True` also masks
echo_intensity/correlation/percent good at the selected depths, appropriate
only once you've confirmed the raw diagnostic itself is contaminated, not
just assumed it.

```python
runner.trim_depths(depths=[12.0], apply_to_all_variables=False)
```

## See Also

- {doc}`core` — `ProcessedDataset.apply_velocity_check()` for the high-level interface
- {doc}`profile_operation` — Applied before velocity check when regridding
- {doc}`utility` — `QCCheckStats` reference
