# Processing Module

The `pyadps.processing` module provides quality control and data processing tools.

## Overview

The processing module contains specialized "runner" classes that perform 
quality control tests and data operations on ADCP data. Each runner is designed 
to handle a specific aspect of ADCP data validation and processing.

## Processing Workflow

A typical ADCP processing workflow:

1. **Read data** → `pyadps.read()`
2. **Check sensor health** → `SensorHealthRunner`
3. **Assess signal quality** → `SignalQualityRunner`
4. **Validate velocities** → `VelocityCheckRunner`
5. **Apply profile operations** → `ProfileOperationRunner`
6. **Export results** → `ProcessedDataset`

## Module Overview

### Quality Control Runners

| Module | Description |
|--------|-------------|
| {doc}`sensor_health` | Tilt, temperature, voltage checks |
| {doc}`signal_quality` | Correlation, echo intensity, percent good |
| {doc}`velocity_check` | Error velocity, threshold tests |

### Data Operations

| Module | Description |
|--------|-------------|
| {doc}`core` | Central processing functions and ProcessedDataset |
| {doc}`profile_operation` | Bin operations, trimming, masking |
| {doc}`time_axis` | Time regularization, gap handling |

### Automation

| Module | Description |
|--------|-------------|
| {doc}`autoprocess` | Automated processing pipeline |
| {doc}`multifile` | Batch processing multiple files |
| {doc}`config` | Processing configuration |

### Utilities

| Module | Description |
|--------|-------------|
| {doc}`plotgen` | Diagnostic plots |
| {doc}`utility` | Helper functions |

## Quick Example

```python
import pyadps
import pyadps.accessors
from pyadps.processing import SensorHealthRunner, SignalQualityRunner

# Load data
ds = pyadps.read('deployment.000')

# Run sensor health checks
sensor_runner = SensorHealthRunner(ds)
sensor_runner.run_all_tests()
sensor_results = sensor_runner.get_results()

# Run signal quality checks
signal_runner = SignalQualityRunner(ds)
signal_runner.run_all_tests()
signal_results = signal_runner.get_results()

# Combine results
from pyadps.processing import ProcessedDataset
processed = ProcessedDataset(ds, [sensor_results, signal_results])
processed.to_netcdf('qc_processed.nc')
```

## Contents

```{toctree}
:maxdepth: 2

core
sensor_health
signal_quality
velocity_check
profile_operation
time_axis
autoprocess
multifile
utility
config
```

## See Also

- {doc}`/io/index` — Data loading functions
- {doc}`/tutorials/index` — Step-by-step tutorials
