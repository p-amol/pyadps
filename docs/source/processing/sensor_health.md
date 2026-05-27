# sensor_health Module

Sensor health checks for ADCP data quality control.

## Overview

The `sensor_health` module provides the `SensorHealthRunner` class for performing 
sensor health checks on ADCP data. It validates tilt sensors (roll, pitch), 
environmental measurements (temperature, salinity), and optionally corrects 
sound speed calculations.

## Quick Start

```python
from pyadps.io import read
from pyadps.processing.sensor_health import SensorHealthRunner

# Read ADCP data
ds = read('your_file.000')

# Create runner and apply checks
runner = SensorHealthRunner(ds)
result = (runner
    .roll_check(threshold=20.0)
    .pitch_check(threshold=15.0)
    .finalize())

# View processing statistics
runner.print_statistics()
```

## Installation

```python
from pyadps.processing.sensor_health import (
    SensorHealthRunner,
    roll_check,
    pitch_check,
    correct_sound_speed,
)
```

---

## Key Concepts

### Mask Convention

The mask follows the numpy/xarray convention:

- **0 = valid data** (good quality)
- **1 = invalid/flagged data** (should be excluded)

### Immutability

The `SensorHealthRunner` preserves your original data:

- `runner.original` — Always contains the unmodified input dataset
- `runner.dataset` — Current working copy (updated after each operation)

### Method Chaining

All processing methods return `self`, enabling fluent chaining:

```python
result = (runner
    .replace_data(ctd_temp, "temperature")
    .replace_data(ctd_sal, "salinity")
    .correct_sound_speed()
    .roll_check(threshold=20.0)
    .pitch_check(threshold=15.0)
    .finalize())
```

---

## Available Methods

### Data Replacement

#### replace_data()

Replace a variable in the dataset with external data (e.g., CTD measurements).

```{function} replace_data(data, variable_name)
Replace a dataset variable with external data.

:param data: New data array (in physical units)
:type data: numpy.ndarray
:param variable_name: Name of variable to replace
:type variable_name: str
:returns: self (for method chaining)
:rtype: SensorHealthRunner
```

**Example:**

```python
# Replace temperature with CTD data (in °C)
ctd_temperature = np.array([15.2, 15.3, 15.1, ...])
runner.replace_data(ctd_temperature, "temperature")
```

```{note}
The function automatically applies the variable's `scale_factor` to convert 
physical units to RDI internal format. For example, 15.0°C is stored as 1500 
(with scale_factor=0.01).
```

---

### Sound Speed Correction

#### correct_sound_speed()

Calculate corrected sound speed from temperature, salinity, and depth using 
the Urick (1983) formula.

```{function} correct_sound_speed(correct_velocity=True, horizontal_only=True)
Calculate and apply sound speed correction.

:param correct_velocity: Also correct velocity using sound speed ratio
:type correct_velocity: bool
:param horizontal_only: Correct only U and V components
:type horizontal_only: bool
:returns: self (for method chaining)
:rtype: SensorHealthRunner
```

**Example:**

```python
# Full sound speed correction with velocity adjustment
runner.correct_sound_speed()

# Sound speed only, no velocity correction
runner.correct_sound_speed(correct_velocity=False)

# Correct all velocity components including W
runner.correct_sound_speed(horizontal_only=False)
```

---

### Sensor Checks

#### roll_check()

Flag ensembles where roll exceeds the threshold.

```{function} roll_check(threshold=15.0)
Flag ensembles with excessive roll.

:param threshold: Maximum acceptable roll in degrees (absolute value)
:type threshold: float
:returns: self (for method chaining)
:rtype: SensorHealthRunner
```

**Example:**

```python
# Standard roll check
runner.roll_check()

# Stricter threshold for sensitive deployments
runner.roll_check(threshold=10.0)
```

---

#### pitch_check()

Flag ensembles where pitch exceeds the threshold.

```{function} pitch_check(threshold=15.0)
Flag ensembles with excessive pitch.

:param threshold: Maximum acceptable pitch in degrees (absolute value)
:type threshold: float
:returns: self (for method chaining)
:rtype: SensorHealthRunner
```

**Example:**

```python
runner.pitch_check(threshold=20.0)
```

---

### Control Methods

#### reset()

Reset the dataset to its original state and clear all history.

```python
# Experiment with different thresholds
runner.roll_check(threshold=20.0)
print(f"Flagged: {runner.statistics[-1].cells_newly_masked}")

runner.reset()  # Go back to original

runner.roll_check(threshold=10.0)
print(f"Flagged: {runner.statistics[-1].cells_newly_masked}")
```

#### finalize()

Finalize processing and return the processed dataset with metadata.

```python
result = runner.finalize()
# result is an xr.Dataset with processing history in attributes
```

---

## Statistics and Reporting

### get_statistics()

Get a dictionary of QC check statistics keyed by check name.

```python
stats = runner.get_statistics()
roll_stats = stats["Roll Check"]
print(f"Newly masked: {roll_stats.cells_newly_masked}")
print(f"Valid data: {roll_stats.valid_pct:.1f}%")
```

### get_modifications()

Get a dictionary of data modification statistics keyed by variable name.

```python
mods = runner.get_modifications()
if "temperature" in mods:
    temp_mod = mods["temperature"][0]
    print(f"Mean change: {temp_mod.mean_change:.2f}")
```

### print_statistics()

Print a formatted statistics table to the console.

```python
runner.print_statistics()
```

**Output:**

```
====================================================================================================
SENSOR HEALTH PROCESSING STATISTICS
====================================================================================================
Baseline masked: 0 (0.00%)
Total cells: 4,000
----------------------------------------------------------------------------------------------------
DATA MODIFICATIONS:
Operation                 | Variable             |   Original Mean |   Modified Mean |       Change
----------------------------------------------------------------------------------------------------
correct_sound_speed       | sound_speed          |        1500.00 |        1485.32 |      -14.68
----------------------------------------------------------------------------------------------------
QC CHECKS:
Check                     |    Threshold |   Pre-Masked |       Impact |   Cumulative |      Valid
----------------------------------------------------------------------------------------------------
Roll Check                |         20.0 |        0.00% |        5.00% |        5.00% |     95.00%
Pitch Check               |         15.0 |        5.00% |        2.00% |        7.00% |     93.00%
----------------------------------------------------------------------------------------------------
FINAL: 3,720 valid cells (93.00%) | Sensor health impact: +7.00%
====================================================================================================
```

### get_report()

Get a formatted report as a string (useful for saving to file).

```python
report = runner.get_report()
with open("processing_report.txt", "w") as f:
    f.write(report)
```

### export_statistics_dict()

Export all statistics as a JSON-serializable dictionary.

```python
import json

export = runner.export_statistics_dict()
with open("statistics.json", "w") as f:
    json.dump(export, f, indent=2)
```

---

## Complete Workflow Examples

### Example 1: Basic Quality Control

```python
from pyadps.io import read
from pyadps.processing.sensor_health import SensorHealthRunner

# Load data
ds = read('mooring_adcp.000')

# Apply standard checks
runner = SensorHealthRunner(ds)
result = (runner
    .roll_check(threshold=15.0)
    .pitch_check(threshold=15.0)
    .finalize())

# View results
runner.print_statistics()
print(f"\nValid data remaining: {runner.get_statistics()['Pitch Check'].valid_pct:.1f}%")
```

### Example 2: With CTD Data Replacement

```python
import numpy as np
from pyadps.io import read
from pyadps.processing.sensor_health import SensorHealthRunner

# Load ADCP data
ds = read('deployment.000')

# Load CTD data (in physical units)
ctd_temp = np.loadtxt('ctd_temperature.csv')  # in °C
ctd_sal = np.loadtxt('ctd_salinity.csv')      # in PSU

# Process with CTD corrections
runner = SensorHealthRunner(ds)
result = (runner
    .replace_data(ctd_temp, "temperature")
    .replace_data(ctd_sal, "salinity")
    .correct_sound_speed()
    .roll_check(threshold=20.0)
    .pitch_check(threshold=15.0)
    .finalize())

# Check modifications
mods = runner.get_modifications()
print(f"Sound speed change: {mods['sound_speed'][0].mean_change:.2f} m/s")
```

### Example 3: Parameter Experimentation

```python
from pyadps.io import read
from pyadps.processing.sensor_health import SensorHealthRunner

ds = read('test_data.000')
runner = SensorHealthRunner(ds)

# Try different roll thresholds
for threshold in [10, 15, 20, 25]:
    runner.reset()  # Start fresh
    runner.roll_check(threshold=threshold)
    
    stats = runner.statistics[-1]
    print(f"Roll threshold {threshold}°: {stats.newly_masked_pct:.1f}% flagged")
```

---

## Statistics Classes Reference

### QCCheckStats

Statistics for a single QC check.

| Property | Type | Description |
|----------|------|-------------|
| `check_name` | str | Name of the check |
| `threshold` | float/tuple/None | Threshold value used |
| `cells_pre_masked` | int | Cells masked before this check |
| `cells_newly_masked` | int | Cells newly masked by this check |
| `cells_total_masked` | int | Total cells masked after this check |
| `total_cells` | int | Total cells in the mask |
| `pre_masked_pct` | float | Percentage pre-masked |
| `newly_masked_pct` | float | Percentage newly masked (impact) |
| `total_masked_pct` | float | Percentage total masked (cumulative) |
| `valid_cells` | int | Number of valid cells remaining |
| `valid_pct` | float | Percentage of valid cells |

### DataModificationStats

Statistics for a data modification operation.

| Property | Type | Description |
|----------|------|-------------|
| `operation` | str | Operation name |
| `variable_name` | str | Variable that was modified |
| `original_stats` | dict | Original data statistics (min, max, mean, std) |
| `modified_stats` | dict | Modified data statistics |
| `mean_change` | float | Change in mean value |
| `mean_change_pct` | float | Percentage change in mean |

---

## Best Practices

1. **Always check statistics** after processing to understand data quality impact
2. **Use `reset()`** for parameter experimentation without reloading data
3. **Save statistics** alongside processed data for reproducibility
4. **Replace data before sound speed correction** when using external CTD data
5. **Apply roll/pitch checks** before other QC checks as they flag entire ensembles

---

## Troubleshooting

**Q: My velocity data looks wrong after sound speed correction.**

A: Check if `horizontal_only=True` is appropriate for your analysis. For some 
applications, you may need to correct all velocity components with 
`horizontal_only=False`.

**Q: Scale factor warning when using replace_data.**

A: If the variable doesn't have a `scale_factor` attribute, ensure your input 
data is already in RDI internal format, or set `apply_scale_factor=False`.

**Q: Statistics show 0% masked but I know there should be flagged data.**

A: Verify that your threshold is set correctly. Also check if the input dataset 
already has missing values that aren't being counted.

---

## See Also

- {doc}`signal_quality` — Signal quality assessment
- {doc}`velocity_check` — Velocity validation
- {doc}`core` — ProcessedDataset orchestrator
