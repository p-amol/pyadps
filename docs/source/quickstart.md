# Quick Start

Get up and running with pyadps in minutes.

## Loading Data

### Default Read

The simplest way to load an ADCP binary file. The result is an
[xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html)
containing all variables (velocity, correlation, echo intensity, percent good,
fixed leader, and variable leader data).

```python
import pyadps

ds = pyadps.read('deployment.000')
print(ds)
```

### Plot and Save Raw Data

```python
# Plot eastward velocity as a depth–time section
ds['velocity'].sel(beam=0).plot()

# Save the raw dataset to NetCDF
ds.to_netcdf('raw_output.nc')
```

---

## Processing Data

Processing is done through the `ProcessedDataset` class, which runs a
six-step quality control pipeline. Steps can be chained and any step can
be omitted. The original dataset is never modified.

```python
from pyadps.processing import ProcessedDataset

proc = ProcessedDataset(ds)
```

### Step 1 — Time Axis Correction

Snap irregular timestamps to a regular grid and fill any time gaps.

```python
proc.apply_time_axis(snap=True, snap_freq='h')
```

### Step 2 — Sensor Health

Flag ensembles where the instrument tilt exceeds acceptable limits.

```python
proc.apply_sensor_health(roll=True, roll_threshold=15.0,
                         pitch=True, pitch_threshold=15.0)
```

### Step 3 — Signal Quality

Mask low-quality data using correlation, echo intensity, error velocity,
and percent-good thresholds.

```python
proc.apply_signal_quality(correlation=64, echo_intensity=40,
                          error_velocity=2000, percent_good=25)
```

### Step 4 — Profile Operation

Remove side-lobe contaminated bins near the sea surface and trim
deployment/recovery periods.

```python
proc.apply_profile_operation(cut_bins_side_lobe=True, water_depth=50.0,
                             trim_start=10, trim_end=10)
```

### Step 5 — Velocity Check

Remove physically unrealistic velocities and apply magnetic declination
correction.

```python
proc.apply_velocity_check(cutoff_u=2500, cutoff_v=2500, cutoff_w=500,
                          magnetic_correction=True, declination=-1.5)
```

### Step 6 — Finalize and Save

`finalize()` returns a standard `xarray.Dataset` with processing metadata
embedded in the global attributes.

```python
result = proc.finalize()

# Save full processed dataset
result.to_netcdf('processed.nc')

# Save only velocity components (u, v, w) in cm/s
proc.velocity_to_netcdf('velocity.nc', units='cm/s')

# Export the processing configuration for reproducibility
proc.export_config('config.ini')
```

### Method Chaining

All steps support fluent chaining for a compact workflow:

```python
result = (
    ProcessedDataset(ds)
    .apply_time_axis(snap=True, snap_freq='h')
    .apply_sensor_health(roll=True, roll_threshold=15.0)
    .apply_signal_quality(correlation=64, echo_intensity=40)
    .apply_profile_operation(cut_bins_side_lobe=True, regrid=True)
    .apply_velocity_check(cutoff_u=2500, cutoff_v=2500, cutoff_w=500)
    .finalize()
)

result.to_netcdf('processed.nc')
```

---

## Add-On Modules

### Auto Processing

Re-run a processing workflow from a saved `config.ini` file — useful for
batch reprocessing with adjusted thresholds.

```python
from pyadps.processing.autoprocess import autoprocess

result = autoprocess(
    config_file_or_object='config.ini',
    binary_file_path='deployment.000',
    save_netcdf=True,
)
```

The same workflow is available from the command line via the `run-auto`
script, installed alongside `pyadps` — no Python needed:

```bash
run-auto config.ini --binary deployment.000
```

By default this writes `deployment_processed.nc` next to the binary file.

| Flag | Description |
|------|-------------|
| `-b`, `--binary` | Path to the ADCP binary file (defaults to the path recorded in the config) |
| `-o`, `--output-dir` | Directory for the output NetCDF file |
| `--output-filename` | Output filename (defaults to `<input>_processed.nc`) |
| `--velocity-only` | Save only the velocity components (`u`, `v`, `w`) |
| `--velocity-units` | Units for `--velocity-only` output: `mm/s`, `cm/s`, or `m/s` |
| `--no-depth-ascending` | Skip forcing ascending depth order in the output |
| `-q`, `--quiet` | Suppress the processing summary |

Run `run-auto --help` to see this from the terminal.

### Binary File Combiner

Combine multiple sequential ADCP binary files into a single file.

```python
from pathlib import Path
from pyadps.processing.multifile import combine_file_list

files = [Path('deploy_000.000'), Path('deploy_001.000'), Path('deploy_002.000')]

result = combine_file_list(files, output_file=Path('merged.000'))
print(f"Combined {result.total_ensembles} ensembles from {result.files_processed} files")
```

---

## Next Steps

- {doc}`webapp/index` — Interactive processing via the web interface
- {doc}`io/index` — Full I/O module reference
- {doc}`processing/index` — Detailed processing pipeline documentation
- {doc}`tutorials/index` — Step-by-step tutorials
