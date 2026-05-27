# velocity_check Module

Velocity quality control and correction for ADCP data.

## Overview

The `velocity_check` module provides the `VelocityCheckRunner` class for performing 
velocity quality control (QC) checks and corrections on ADCP data. It validates 
measured current velocities and identifies unreliable data caused by instrument 
tilt, acoustic interference, sensor failures, or physical impossibilities.

## Quick Start

```python
from pyadps.io import read
from pyadps.processing.velocity_check import VelocityCheckRunner

# Read ADCP data
ds = read("your_file.000")

# Apply QC checks with method chaining
runner = VelocityCheckRunner(ds)
ds_qc = (
    runner
    .magnetic_correction(declination=-5.0)
    .threshold(cutoff_u=2500, cutoff_v=2500, cutoff_w=500)
    .despike(kernel_size=13, cutoff=3.0)
    .flatline(kernel_size=4, cutoff=1.0)
    .finalize()
)

# View processing statistics
runner.print_statistics()
```

## Installation

```python
from pyadps.processing.velocity_check import VelocityCheckRunner
```

---

## Key Concepts

### Mask Convention

The module uses a binary mask where:

- **0 = Valid data** (passes QC)
- **1 = Invalid/flagged data** (fails QC)

### Per-Component Thresholds

Velocity threshold checks support different thresholds for each component:

| Component | Description | Default | Rationale |
|-----------|-------------|---------|-----------|
| U (East) | Horizontal | 2500 mm/s | Strong tidal/wind-driven currents |
| V (North) | Horizontal | 2500 mm/s | Strong tidal/wind-driven currents |
| W (Vertical) | Vertical | 500 mm/s | Vertical velocities are 5-10x weaker |

### Combined Mask (Beam 3)

All velocity QC checks update a combined mask:

- **Beams 0-2**: Individual velocity component flags (U, V, W)
- **Beam 3**: Combined flag = OR(U, V, W) — flagged if ANY component fails

### Immutability

The original dataset is never modified:

- `runner.original` — Immutable copy of input dataset
- `runner.dataset` — Working copy that accumulates QC flags

### Method Chaining

All QC methods return `self`, enabling fluent method chaining:

```python
ds_qc = runner.threshold().despike().flatline().finalize()
```

---

## Available Operations

### magnetic_correction()

Rotates U and V velocity components to correct for magnetic declination. 
This **modifies data values**, not the mask.

```{function} magnetic_correction(declination=None, use_api=False, lat=None, lon=None, year=None)
Apply magnetic declination correction.

:param declination: Declination in degrees (+ = east)
:type declination: float, optional
:param use_api: Use NOAA API instead of local COF files
:type use_api: bool
:param lat: Latitude for calculation
:type lat: float, optional
:param lon: Longitude for calculation
:type lon: float, optional
:param year: Year for calculation (decimal)
:type year: float, optional
:returns: self (for method chaining)
:rtype: VelocityCheckRunner
```

**Declination sources:**

- Explicit value: Provide `declination` parameter directly
- Local calculation: Uses pygeomag with WMM COF files
- NOAA API: Set `use_api=True` for online lookup

**Example:**

```python
# With known declination
runner.magnetic_correction(declination=-5.0)

# Calculate from location (uses local COF files)
runner.magnetic_correction(lat=40.0, lon=-74.0, year=2023.5)

# Calculate using NOAA API
runner.magnetic_correction(lat=40.0, lon=-74.0, year=2023.5, use_api=True)
```

**Rotation formulas:**

- U_new = U × cos(dec) + V × sin(dec)
- V_new = -U × sin(dec) + V × cos(dec)

---

### threshold()

Flags cells where velocity magnitude exceeds component-specific thresholds.

```{function} threshold(cutoff_u=2500.0, cutoff_v=2500.0, cutoff_w=500.0)
Apply velocity threshold check.

:param cutoff_u: U (East) threshold in mm/s
:type cutoff_u: float
:param cutoff_v: V (North) threshold in mm/s
:type cutoff_v: float
:param cutoff_w: W (Vertical) threshold in mm/s
:type cutoff_w: float
:returns: self (for method chaining)
:rtype: VelocityCheckRunner
```

**Flagging behavior:**

- Flags beam 0 where `|U| > cutoff_u`
- Flags beam 1 where `|V| > cutoff_v`
- Flags beam 2 where `|W| > cutoff_w`
- Updates beam 3 (combined) as OR of beams 0, 1, 2

**Example:**

```python
# Standard thresholds
runner.threshold(cutoff_u=2500, cutoff_v=2500, cutoff_w=500)

# Stricter for low-energy environment
runner.threshold(cutoff_u=1000, cutoff_v=1000, cutoff_w=200)

# More permissive for high-energy environment
runner.threshold(cutoff_u=5000, cutoff_v=5000, cutoff_w=1000)
```

---

### despike()

Identifies and flags anomalous spikes using a median filter approach.

```{function} despike(kernel_size=13, cutoff=3.0)
Apply despike filter.

:param kernel_size: Window size for median filter (must be odd)
:type kernel_size: int
:param cutoff: Number of standard deviations for spike detection
:type cutoff: float
:returns: self (for method chaining)
:rtype: VelocityCheckRunner
```

**Algorithm:**

1. Apply median filter to each time series
2. Compute absolute difference from filtered values
3. Calculate standard deviation of differences
4. Flag points where `|velocity - median| > cutoff × std_dev`

**Example:**

```python
# Standard despike
runner.despike(kernel_size=13, cutoff=3.0)

# More aggressive spike removal
runner.despike(kernel_size=7, cutoff=2.0)

# Less aggressive (only obvious spikes)
runner.despike(kernel_size=21, cutoff=4.0)
```

```{note}
The same cutoff multiplier works for all components because the algorithm is 
self-normalizing — each component's threshold scales with its own variability.
```

---

### flatline()

Detects and flags constant velocity values over time, which typically indicate 
a frozen sensor or data transmission error.

```{function} flatline(kernel_size=4, cutoff=1.0)
Apply flatline detection.

:param kernel_size: Minimum consecutive points to flag as flatline
:type kernel_size: int
:param cutoff: Maximum variation to consider "constant" (mm/s)
:type cutoff: float
:returns: self (for method chaining)
:rtype: VelocityCheckRunner
```

**Algorithm:**

1. Compute consecutive differences in velocity
2. Identify segments where all differences ≤ cutoff
3. Flag segments with length ≥ kernel_size

**Example:**

```python
# Standard flatline detection
runner.flatline(kernel_size=4, cutoff=1.0)

# Detect longer flatlines only
runner.flatline(kernel_size=10, cutoff=1.0)

# More sensitive (detect near-constant values)
runner.flatline(kernel_size=4, cutoff=5.0)
```

---

## Pipeline Methods

### apply_pipeline()

Apply multiple checks in one call using a configuration dictionary:

```{function} apply_pipeline(checks=None, order=None)
Apply multiple velocity checks from configuration.

:param checks: Dictionary of check names to parameter dictionaries
:type checks: dict, optional
:param order: List specifying check order
:type order: list, optional
:returns: self (for method chaining)
:rtype: VelocityCheckRunner
```

**Example:**

```python
runner.apply_pipeline(
    checks={
        "threshold": {"cutoff_u": 2500, "cutoff_v": 2500, "cutoff_w": 500},
        "despike": {"kernel_size": 13, "cutoff": 3.0},
        "flatline": {"kernel_size": 4, "cutoff": 1.0},
    },
    order=["threshold", "despike", "flatline"]
)
```

### reset()

Reset to original state, clearing all QC flags and history:

```python
runner.reset()
```

### finalize()

Complete processing and return the dataset with history attributes:

```python
ds_qc = runner.finalize()
```

---

## Statistics and Reporting

### get_statistics()

Get QC statistics as a dictionary keyed by check name:

```python
stats = runner.get_statistics()
threshold_stats = stats["Velocity Threshold"]
print(f"Threshold impact: {threshold_stats.newly_masked_pct:.2f}%")
```

### get_modifications()

Get data modification statistics (e.g., magnetic correction):

```python
mods = runner.get_modifications()
if "velocity" in mods:
    print(f"Velocity modified by magnetic correction")
```

### print_statistics()

Print a formatted statistics table:

```python
runner.print_statistics()
```

**Output:**

```
====================================================================================================
VELOCITY CHECK PROCESSING STATISTICS
====================================================================================================
Baseline masked: 0 (0.00%)
Total cells: 4,000
----------------------------------------------------------------------------------------------------
DATA MODIFICATIONS:
Operation                 | Variable             |   Original Mean |   Modified Mean |       Change
----------------------------------------------------------------------------------------------------
magnetic_correction       | velocity             |         125.32 |         128.15 |       +2.83
----------------------------------------------------------------------------------------------------
QC CHECKS:
Check                     |    Threshold |   Pre-Masked |       Impact |   Cumulative |      Valid
----------------------------------------------------------------------------------------------------
Velocity Threshold        | U:2500 V:2500 W:500 |  0.00% |        3.50% |        3.50% |     96.50%
Despike                   | k=13, σ=3.0  |        3.50% |        1.25% |        4.75% |     95.25%
Flatline                  | k=4, c=1.0   |        4.75% |        0.50% |        5.25% |     94.75%
----------------------------------------------------------------------------------------------------
FINAL: 3,790 valid cells (94.75%) | Velocity check impact: +5.25%
====================================================================================================
```

### export_statistics_dict()

Export all statistics as a JSON-serializable dictionary:

```python
import json

stats_dict = runner.export_statistics_dict()
with open("velocity_qc_statistics.json", "w") as f:
    json.dump(stats_dict, f, indent=2)
```

---

## Complete Workflow Examples

### Basic Velocity QC

```python
from pyadps.io import read
from pyadps.processing.velocity_check import VelocityCheckRunner

# Read data
ds = read("mooring_adcp.000")

# Apply standard QC checks
runner = VelocityCheckRunner(ds)
ds_qc = (
    runner
    .threshold(cutoff_u=2500, cutoff_v=2500, cutoff_w=500)
    .despike(kernel_size=13, cutoff=3.0)
    .flatline(kernel_size=4, cutoff=1.0)
    .finalize()
)

runner.print_statistics()
ds_qc.to_netcdf("mooring_velocity_qc.nc")
```

### With Magnetic Declination Correction

```python
runner = VelocityCheckRunner(ds)
ds_qc = (
    runner
    .magnetic_correction(declination=-5.0)  # Apply first!
    .threshold(cutoff_u=2500, cutoff_v=2500, cutoff_w=500)
    .despike(kernel_size=13, cutoff=3.0)
    .finalize()
)
```

### Parameter Sensitivity Analysis

```python
runner = VelocityCheckRunner(ds)

print("W Threshold Sensitivity Analysis:")
for cutoff_w in [200, 300, 500, 750, 1000]:
    runner.reset()
    runner.threshold(cutoff_w=cutoff_w)
    stats = runner.statistics[-1]
    print(f"  cutoff_w={cutoff_w}: {stats.valid_pct:.1f}% valid")
```

---

## Default Thresholds Reference

| Check | Parameter | Default | Valid Range | Units |
|-------|-----------|---------|-------------|-------|
| Threshold | cutoff_u | 2500 | 0-10000 | mm/s |
| Threshold | cutoff_v | 2500 | 0-10000 | mm/s |
| Threshold | cutoff_w | 500 | 0-5000 | mm/s |
| Despike | kernel_size | 13 | 3-51 (odd) | samples |
| Despike | cutoff | 3.0 | 0.5-10 | σ (std dev) |
| Flatline | kernel_size | 4 | 2-100 | samples |
| Flatline | cutoff | 1.0 | 0-100 | mm/s |

---

## Best Practices

1. **Apply magnetic declination first**: If needed, correct for declination before 
   QC checks so that U truly represents East and V truly represents North.

2. **Use appropriate W threshold**: Vertical velocities are typically 5-10x smaller 
   than horizontal. The default 500 mm/s is appropriate for most open-ocean deployments.

3. **Order of checks matters**:
   - Threshold check first (removes physically impossible values)
   - Despike second (identifies transient anomalies)
   - Flatline last (detects sensor failures)

4. **Consider environment**: High-energy environments (tidal channels, river mouths) 
   may need higher thresholds; low-energy environments (deep ocean) may need lower.

5. **Use `reset()` for experimentation**: Test different thresholds without reloading data.

---

## Troubleshooting

### High flagging rates

If more than 20-30% of data is flagged:

1. Check if thresholds are appropriate for your environment
2. Review per-component impacts — is one component dominating?
3. Consider relaxing W threshold if vertical velocities are being over-flagged

### Magnetic declination issues

**Q: I get an error about missing lat/lon/year**

A: Either provide `declination` directly, or ensure your dataset has 
`latitude`/`longitude` attributes and a `time` coordinate.

**Q: pygeomag not found**

A: Install pygeomag (`pip install pygeomag`) or use the NOAA API:
```python
runner.magnetic_correction(lat=40.0, lon=-74.0, year=2023.5, use_api=True)
```

### Despike issues

- **Too many spikes detected**: Increase `cutoff` (e.g., 4.0) or `kernel_size`
- **Too few spikes detected**: Decrease `cutoff` (e.g., 2.0) or `kernel_size`

### Flatline issues

- Increase `cutoff` tolerance if sensor has slight noise
- Decrease `kernel_size` if flatlines are short duration

---

## See Also

- {doc}`sensor_health` — Sensor health checks
- {doc}`signal_quality` — Signal quality assessment
- {doc}`core` — ProcessedDataset orchestrator
