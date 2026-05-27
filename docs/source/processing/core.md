# core Module

Central processing functions and the ProcessedDataset orchestrator class.

## Overview

The `core` module provides the main orchestration layer for ADCP data processing. 
The `ProcessedDataset` class coordinates the complete six-step processing pipeline, 
delegating to specialized Runner classes for each step while maintaining a unified 
interface and collecting statistics from all operations.

## Architecture

```
ProcessedDataset (Orchestrator)
    │
    ├── apply_time_axis()      → time_axis.py functions
    ├── apply_sensor_health()  → SensorHealthRunner
    ├── apply_signal_quality() → SignalQualityRunner
    ├── apply_profile_operation() → ProfileOperationRunner
    ├── apply_velocity_check() → VelocityCheckRunner
    └── finalize()             → Returns processed xr.Dataset
```

## Quick Start

```python
import pyadps
from pyadps.processing import ProcessedDataset

# Read raw ADCP data
ds = pyadps.read("deployment.000")

# Create processor and apply pipeline
proc = ProcessedDataset(ds)

result = (
    proc
    .apply_time_axis(snap=True, snap_freq="h")
    .apply_sensor_health(roll=True, roll_threshold=15.0)
    .apply_signal_quality(correlation=64, echo_intensity=40)
    .apply_profile_operation(cut_bins_side_lobe=True, regrid=True)
    .apply_velocity_check(cutoff_u=2500, cutoff_v=2500, cutoff_w=500)
    .finalize()
)

# Save results
result.to_netcdf("deployment_processed.nc")

# View summary
proc.print_summary()
```

---

## ProcessedDataset Class

### Constructor

```{function} ProcessedDataset(dataset)
Create a new ProcessedDataset orchestrator.

:param dataset: Raw ADCP dataset from pyadps.read()
:type dataset: xarray.Dataset
```

**Example:**

```python
ds = pyadps.read("file.000")
proc = ProcessedDataset(ds)
```

---

## Processing Steps

### Step 1: Time Axis Correction

Corrects irregular timestamps and fills time gaps.

```{function} apply_time_axis(snap=False, snap_freq='h', snap_tolerance='5min', snap_target_minute=None, fill_gaps=False, fill_method='auto')
Apply time axis corrections.

:param snap: Enable snapping to regular intervals
:type snap: bool
:param snap_freq: Target frequency ('h', 'D', '30min', etc.)
:type snap_freq: str
:param snap_tolerance: Maximum allowed correction
:type snap_tolerance: str
:param snap_target_minute: Snap to specific minute (0-59)
:type snap_target_minute: int, optional
:param fill_gaps: Enable gap filling
:type fill_gaps: bool
:param fill_method: Gap fill method ('auto', 'h', 'D', etc.)
:type fill_method: str
:returns: self (for method chaining)
:rtype: ProcessedDataset
```

**Example:**

```python
# Snap to hourly, allow 5-minute corrections
proc.apply_time_axis(snap=True, snap_freq="h", snap_tolerance="5min")

# Fill gaps with auto-detected frequency
proc.apply_time_axis(fill_gaps=True, fill_method="auto")
```

---

### Step 2: Sensor Health Check

Validates environmental sensors and optionally corrects sound speed.

```{function} apply_sensor_health(roll=False, roll_threshold=15.0, pitch=False, pitch_threshold=15.0, correct_sound_speed=False, temperature=None, salinity=None, depth=None)
Apply sensor health checks.

:param roll: Enable roll check
:type roll: bool
:param roll_threshold: Maximum acceptable roll (degrees)
:type roll_threshold: float
:param pitch: Enable pitch check
:type pitch: bool
:param pitch_threshold: Maximum acceptable pitch (degrees)
:type pitch_threshold: float
:param correct_sound_speed: Enable sound speed correction
:type correct_sound_speed: bool
:param temperature: Fixed temperature (°C), or None to use dataset
:type temperature: float, optional
:param salinity: Fixed salinity (PSU), or None to use dataset
:type salinity: float, optional
:param depth: Fixed depth (m), or None to use dataset
:type depth: float, optional
:returns: self (for method chaining)
:rtype: ProcessedDataset
```

**Example:**

```python
# Check tilt sensors
proc.apply_sensor_health(
    roll=True, roll_threshold=15.0,
    pitch=True, pitch_threshold=15.0
)

# Apply sound speed correction with CTD data
proc.apply_sensor_health(
    correct_sound_speed=True,
    temperature=12.5,
    salinity=34.8,
)
```

---

### Step 3: Signal Quality Check

Applies RDI-standard quality control checks based on acoustic signal properties.

```{function} apply_signal_quality(correlation=None, echo_intensity=None, error_velocity=None, percent_good=None, false_target=None, threebeam=False, beam_ignore=None)
Apply signal quality checks.

:param correlation: Minimum correlation (0-255), or None to skip
:type correlation: int, optional
:param echo_intensity: Minimum echo intensity (0-255), or None to skip
:type echo_intensity: int, optional
:param error_velocity: Maximum error velocity (mm/s), or None to skip
:type error_velocity: int, optional
:param percent_good: Minimum percent good (0-100), or None to skip
:type percent_good: int, optional
:param false_target: Maximum echo difference (0-255), or None to skip
:type false_target: int, optional
:param threebeam: Enable 3-beam solutions
:type threebeam: bool
:param beam_ignore: Beam to ignore (0-3) in 3-beam mode
:type beam_ignore: int, optional
:returns: self (for method chaining)
:rtype: ProcessedDataset
```

**Typical Thresholds:**

| Check | Typical Value | Description |
|-------|---------------|-------------|
| correlation | 64 | RDI default minimum |
| echo_intensity | 40 | Minimum signal strength |
| error_velocity | 2000 | Max error in mm/s |
| percent_good | 50 | Min % good pings |
| false_target | 50 | Max echo difference |

**Example:**

```python
# Standard QC
proc.apply_signal_quality(
    correlation=64,
    echo_intensity=40,
    percent_good=50
)
```

---

### Step 4: Profile Operations

Applies spatial operations including bin trimming and regridding.

```{function} apply_profile_operation(trim_start=None, trim_end=None, cut_bins_side_lobe=False, side_lobe_extra_cells=1, cut_bins_manual=None, regrid=False, regrid_method='linear')
Apply profile operations.

:param trim_start: Number of ensembles to mask from start
:type trim_start: int, optional
:param trim_end: Number of ensembles to mask from end
:type trim_end: int, optional
:param cut_bins_side_lobe: Enable side lobe contamination removal
:type cut_bins_side_lobe: bool
:param side_lobe_extra_cells: Additional cells to mask beyond calculation
:type side_lobe_extra_cells: int
:param cut_bins_manual: Manual bin cutting parameters dict
:type cut_bins_manual: dict, optional
:param regrid: Enable regridding to regular depth levels
:type regrid: bool
:param regrid_method: Interpolation method ('linear', 'nearest')
:type regrid_method: str
:returns: self (for method chaining)
:rtype: ProcessedDataset
```

**Example:**

```python
# Remove deployment/recovery and side lobe contamination
proc.apply_profile_operation(
    trim_start=100,
    trim_end=50,
    cut_bins_side_lobe=True,
    side_lobe_extra_cells=2
)
```

---

### Step 5: Velocity Check

Validates velocity data and applies corrections.

```{function} apply_velocity_check(magnetic_declination=None, cutoff_u=2500, cutoff_v=2500, cutoff_w=500, despike=True, despike_kernel=13, despike_cutoff=3.0, flatline=True, flatline_kernel=4, flatline_cutoff=1.0)
Apply velocity checks and corrections.

:param magnetic_declination: Magnetic declination in degrees (+ = east)
:type magnetic_declination: float, optional
:param cutoff_u: U (East) threshold in mm/s
:type cutoff_u: float
:param cutoff_v: V (North) threshold in mm/s
:type cutoff_v: float
:param cutoff_w: W (Vertical) threshold in mm/s
:type cutoff_w: float
:param despike: Enable despike filter
:type despike: bool
:param despike_kernel: Despike window size (must be odd)
:type despike_kernel: int
:param despike_cutoff: Despike threshold in standard deviations
:type despike_cutoff: float
:param flatline: Enable flatline detection
:type flatline: bool
:param flatline_kernel: Minimum consecutive points for flatline
:type flatline_kernel: int
:param flatline_cutoff: Maximum variation for flatline (mm/s)
:type flatline_cutoff: float
:returns: self (for method chaining)
:rtype: ProcessedDataset
```

**Example:**

```python
proc.apply_velocity_check(
    magnetic_declination=-5.0,
    cutoff_u=2500,
    cutoff_v=2500,
    cutoff_w=500,
    despike=True,
    despike_cutoff=3.0
)
```

---

### Finalize

Complete processing and return the final dataset.

```{function} finalize()
Finalize processing and return processed dataset.

:returns: Processed dataset with all QC flags applied
:rtype: xarray.Dataset
```

**Example:**

```python
result = proc.finalize()
result.to_netcdf("output.nc")
```

---

## Configuration-Based Processing

Load processing parameters from an INI configuration file:

```python
proc = ProcessedDataset(ds)
result = proc.apply_config("config.ini").finalize()
result.to_netcdf("output.nc")
```

---

## Reporting and Statistics

### print_summary()

Print a formatted summary of all processing steps:

```python
proc.print_summary()
```

### get_summary()

Get the summary as a string:

```python
summary = proc.get_summary()
with open("report.txt", "w") as f:
    f.write(summary)
```

### Accessing Individual Runner Reports

```python
# Access sensor health statistics
sensor_report = proc.sensor_health_runner.get_pipeline_report()

# Access signal quality statistics
signal_report = proc.signal_quality_runner.get_pipeline_report()

# Access velocity check statistics
velocity_report = proc.velocity_check_runner.get_pipeline_report()
```

---

## Complete Workflow Example

```python
"""Complete ADCP processing workflow."""
import pyadps
from pyadps.processing import ProcessedDataset

# Load raw data
ds = pyadps.read("mooring_deployment.000")

# Create processor
proc = ProcessedDataset(ds)

# Apply full processing pipeline
result = (
    proc
    # Step 1: Time axis
    .apply_time_axis(
        snap=True,
        snap_freq="h",
        snap_tolerance="5min"
    )
    # Step 2: Sensor health
    .apply_sensor_health(
        roll=True, roll_threshold=15.0,
        pitch=True, pitch_threshold=15.0,
        correct_sound_speed=True
    )
    # Step 3: Signal quality
    .apply_signal_quality(
        correlation=64,
        echo_intensity=40,
        error_velocity=2000,
        percent_good=50
    )
    # Step 4: Profile operations
    .apply_profile_operation(
        trim_start=100,
        trim_end=50,
        cut_bins_side_lobe=True,
        regrid=True
    )
    # Step 5: Velocity check
    .apply_velocity_check(
        magnetic_declination=-5.0,
        cutoff_u=2500,
        cutoff_v=2500,
        cutoff_w=500,
        despike=True
    )
    .finalize()
)

# Print processing summary
proc.print_summary()

# Save results
result.to_netcdf("mooring_processed.nc")

print(f"Processing complete. Valid data: {proc.final_valid_pct:.1f}%")
```

---

## Best Practices

1. **Follow the recommended order** — The six-step pipeline is designed to be applied in sequence for optimal results.

2. **Apply signal quality before regridding** — Cell-based QC checks must be done before regridding changes the coordinate system.

3. **Use reset() for experimentation** — Test different parameters without reloading data.

4. **Save statistics** — Export processing reports for reproducibility and documentation.

5. **Check intermediate results** — Use `print_summary()` after each step to monitor data quality.

---

## See Also

- {doc}`sensor_health` — Sensor health checks
- {doc}`signal_quality` — Signal quality assessment
- {doc}`velocity_check` — Velocity validation
- {doc}`profile_operation` — Profile operations
- {doc}`autoprocess` — Automated processing
