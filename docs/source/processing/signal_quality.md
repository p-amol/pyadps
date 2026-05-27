# signal_quality Module

Signal quality control checks for ADCP data.

## Overview

The `signal_quality` module provides the `SignalQualityRunner` class for performing 
signal quality control (QC) checks on ADCP data. It validates acoustic signal 
measurements to identify unreliable data caused by poor acoustic conditions, 
interference, or instrument issues.

## Quick Start

```python
from pyadps.io import read
from pyadps.processing.signal_quality import SignalQualityRunner

# Read ADCP data
ds = read("your_file.000")

# Apply QC checks with method chaining
runner = SignalQualityRunner(ds)
ds_qc = (
    runner
    .correlation(cutoff=64)
    .echo_intensity(cutoff=40)
    .error_velocity(cutoff=2000)
    .percent_good(cutoff=50)
    .false_target(cutoff=50)
    .finalize()
)

# View processing statistics
runner.print_statistics()
```

## Installation

```python
from pyadps.processing.signal_quality import SignalQualityRunner
```

---

## Key Concepts

### Mask Convention

The module uses a binary mask where:

- **0 = Valid data** (passes QC)
- **1 = Invalid/flagged data** (fails QC)

### Beam-Specific Masking

Signal quality checks use a 3D mask with shape `(beam, cell, time)`:

- **Beams 0-2**: Individual beam quality flags (for correlation and echo intensity)
- **Beam 3**: Combined signal quality flag (for error velocity, percent good, false target)

### Immutability

The original dataset is never modified:

- `runner.original` — Immutable copy of input dataset
- `runner.dataset` — Working copy that accumulates QC flags

### Method Chaining

All QC methods return `self`, enabling fluent method chaining:

```python
ds_qc = runner.correlation().echo_intensity().error_velocity().finalize()
```

---

## Available QC Checks

### correlation()

Flags cells where correlation count is below the threshold. Low correlation 
indicates poor signal quality.

```{function} correlation(cutoff=64, threebeam=False, beam_ignore=None)
Apply correlation threshold check.

:param cutoff: Minimum acceptable correlation (0-255)
:type cutoff: int
:param threebeam: Enable three-beam mode
:type threebeam: bool
:param beam_ignore: Beam to ignore in three-beam mode (0-3)
:type beam_ignore: int, optional
:returns: self (for method chaining)
:rtype: SignalQualityRunner
```

**Default threshold**: 64 (RDI recommendation)

**Flagging behavior**: Flags each beam independently where `correlation < cutoff`

**Example:**

```python
# Standard correlation check
runner.correlation(cutoff=64)

# More strict correlation check
runner.correlation(cutoff=100)

# Three-beam mode (ignore beam 2 if known bad)
runner.correlation(cutoff=64, threebeam=True, beam_ignore=2)
```

---

### echo_intensity()

Flags cells where echo intensity (signal strength) is below the threshold.

```{function} echo_intensity(cutoff=40, threebeam=False, beam_ignore=None)
Apply echo intensity threshold check.

:param cutoff: Minimum acceptable echo intensity (0-255)
:type cutoff: int
:param threebeam: Enable three-beam mode
:type threebeam: bool
:param beam_ignore: Beam to ignore in three-beam mode (0-3)
:type beam_ignore: int, optional
:returns: self (for method chaining)
:rtype: SignalQualityRunner
```

**Default threshold**: 40 counts

**Flagging behavior**: Flags each beam independently where `echo_intensity < cutoff`

**Example:**

```python
# Standard echo intensity check
runner.echo_intensity(cutoff=40)

# Higher threshold for cleaner data
runner.echo_intensity(cutoff=60)
```

---

### error_velocity()

Flags cells where the error velocity exceeds the threshold. High error velocity 
indicates inconsistent measurements between beam pairs.

```{function} error_velocity(cutoff=2000)
Apply error velocity threshold check.

:param cutoff: Maximum acceptable error velocity (mm/s)
:type cutoff: int
:returns: self (for method chaining)
:rtype: SignalQualityRunner
```

**Default threshold**: 2000 mm/s

**Flagging behavior**: Flags beam 3 (combined mask) where `|error_velocity| > cutoff`

**Example:**

```python
# Standard error velocity check
runner.error_velocity(cutoff=2000)

# Stricter threshold
runner.error_velocity(cutoff=1000)
```

---

### percent_good()

Flags cells where the percent good value is below the threshold.

```{function} percent_good(cutoff=50, method='max')
Apply percent good threshold check.

:param cutoff: Minimum acceptable percent good (0-100)
:type cutoff: int
:param method: Method to combine beams ('max', 'min', or 'mean')
:type method: str
:returns: self (for method chaining)
:rtype: SignalQualityRunner
```

**Default threshold**: 50%

**Methods:**

- `"max"` (default): Uses maximum across beams (most permissive)
- `"min"`: Uses minimum across beams (most strict)
- `"mean"`: Uses average across beams

**Flagging behavior**: Flags beam 3 (combined mask) where combined percent good < cutoff

**Example:**

```python
# Standard percent good check (use max across beams)
runner.percent_good(cutoff=50, method="max")

# Strict check (all beams must have good data)
runner.percent_good(cutoff=50, method="min")
```

---

### false_target()

Flags cells where the difference between maximum and minimum echo intensity 
across beams exceeds the threshold. Large differences indicate potential 
false targets (fish, debris, etc.).

```{function} false_target(cutoff=50)
Apply false target detection.

:param cutoff: Maximum acceptable echo intensity difference (0-255)
:type cutoff: int
:returns: self (for method chaining)
:rtype: SignalQualityRunner
```

**Default threshold**: 50 counts

**Flagging behavior**: Flags beam 3 (combined mask) where `(echo_max - echo_min) > cutoff`

**Example:**

```python
# Standard false target detection
runner.false_target(cutoff=50)

# More sensitive detection
runner.false_target(cutoff=30)
```

---

## Pipeline Methods

### apply_pipeline()

Apply multiple checks in one call using a configuration dictionary:

```{function} apply_pipeline(checks=None, order=None)
Apply multiple QC checks from configuration.

:param checks: Dictionary of check names to threshold values
:type checks: dict, optional
:param order: List specifying check order
:type order: list, optional
:returns: self (for method chaining)
:rtype: SignalQualityRunner
```

**Example:**

```python
runner.apply_pipeline(
    checks={
        "correlation": 64,
        "echo_intensity": 40,
        "error_velocity": 2000,
        "percent_good": 50,
        "false_target": 50,
    },
    order=["correlation", "echo_intensity", "error_velocity", "percent_good", "false_target"]
)
```

**Default behavior** (no arguments): Applies all checks with default thresholds.

```python
# Apply all default checks
runner.apply_pipeline()
```

---

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
corr_stats = stats["Correlation"]
print(f"Correlation impact: {corr_stats.newly_masked_pct:.2f}%")
```

### print_statistics()

Print a formatted statistics table:

```python
runner.print_statistics()
```

**Output:**

```
====================================================================================================
SIGNAL QUALITY PROCESSING STATISTICS
====================================================================================================
Baseline masked: 0 (0.00%)
Total cells: 4,000
----------------------------------------------------------------------------------------------------
QC CHECKS:
Check                     |    Threshold |   Pre-Masked |       Impact |   Cumulative |      Valid
----------------------------------------------------------------------------------------------------
Correlation               |           64 |        0.00% |        5.25% |        5.25% |     94.75%
Echo Intensity            |           40 |        5.25% |        2.10% |        7.35% |     92.65%
Error Velocity            |         2000 |        7.35% |        1.50% |        8.85% |     91.15%
Percent Good              |           50 |        8.85% |        3.25% |       12.10% |     87.90%
False Target              |           50 |       12.10% |        0.75% |       12.85% |     87.15%
----------------------------------------------------------------------------------------------------
FINAL: 3,486 valid cells (87.15%) | Signal quality impact: +12.85%
====================================================================================================
```

### export_statistics_dict()

Export all statistics as a JSON-serializable dictionary:

```python
import json

stats_dict = runner.export_statistics_dict()
with open("qc_statistics.json", "w") as f:
    json.dump(stats_dict, f, indent=2)
```

---

## Complete Workflow Examples

### Basic Quality Control

```python
from pyadps.io import read
from pyadps.processing.signal_quality import SignalQualityRunner

# Read data
ds = read("mooring_adcp.000")

# Apply standard QC checks
runner = SignalQualityRunner(ds)
ds_qc = (
    runner
    .correlation(cutoff=64)
    .echo_intensity(cutoff=40)
    .error_velocity(cutoff=2000)
    .percent_good(cutoff=50)
    .finalize()
)

# Print results
runner.print_statistics()

# Save to NetCDF
ds_qc.to_netcdf("mooring_adcp_qc.nc")
```

### Parameter Experimentation

```python
runner = SignalQualityRunner(ds)

# Test different correlation thresholds
for cutoff in [50, 64, 80, 100]:
    runner.reset()
    runner.correlation(cutoff=cutoff)
    stats = runner.statistics[-1]
    print(f"Correlation cutoff={cutoff}: {stats.valid_pct:.1f}% valid")
```

### Three-Beam Mode

When one beam is known to be problematic:

```python
runner = SignalQualityRunner(ds)
ds_qc = (
    runner
    .correlation(cutoff=64, threebeam=True, beam_ignore=2)
    .echo_intensity(cutoff=40, threebeam=True, beam_ignore=2)
    .error_velocity(cutoff=2000)
    .finalize()
)
```

---

## Default Thresholds Reference

| Check | Default Value | Valid Range | Units |
|-------|--------------|-------------|-------|
| Correlation | 64 | 0-255 | counts |
| Echo Intensity | 40 | 0-255 | counts |
| Error Velocity | 2000 | 0-5000 | mm/s |
| Percent Good | 50 | 0-100 | % |
| False Target | 50 | 0-255 | counts |

---

## Best Practices

1. **Apply checks in order of importance**: Start with correlation, then echo 
   intensity, then combined checks (error velocity, percent good, false target).

2. **Use default thresholds as starting point**: RDI-recommended defaults work 
   well for most deployments.

3. **Examine statistics before finalizing**: Use `print_statistics()` to verify 
   reasonable flagging rates.

4. **Use three-beam mode cautiously**: Only ignore a beam if you have confirmed 
   hardware issues.

5. **Export statistics for documentation**: Use `export_statistics_dict()` to 
   save QC metrics for reports.

---

## Troubleshooting

### High flagging rates

If more than 30-40% of data is flagged:

1. Check raw data quality — there may be instrument issues
2. Review individual check impacts using `get_statistics()`
3. Consider relaxing thresholds if scientifically justified
4. Use `reset()` to experiment with different thresholds

### Missing variables

If a check logs "data not found", verify your dataset contains:

- `correlation`
- `echo_intensity` or `echo`
- `velocity` (with beam dimension)
- `percent_good` or `pg`

### Unexpected mask behavior

Remember:

- Correlation and echo intensity flag individual beams (0-3)
- Error velocity, percent good, and false target flag beam 3 only
- Masks accumulate — each check adds to previous flagging

---

## See Also

- {doc}`sensor_health` — Sensor health checks
- {doc}`velocity_check` — Velocity validation
- {doc}`core` — ProcessedDataset orchestrator
