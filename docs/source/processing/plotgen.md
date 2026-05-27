# plotgen Module

Diagnostic plot generation for ADCP data.

## Overview

The `plotgen` module provides functions for creating diagnostic plots of ADCP 
data. It supports visualization of velocity profiles, time series, quality 
control results, and other diagnostic information.

## Quick Start

```python
from pyadps.processing.plotgen import plot_velocity_profile, plot_time_series
import pyadps

# Load data
ds = pyadps.read('deployment.000')

# Create velocity profile plot
fig = plot_velocity_profile(ds, ensemble=100)

# Create time series plot
fig = plot_time_series(ds, cell=10)
```

---

## Available Plots

### Velocity Profiles

- `plot_velocity_profile()` — Single ensemble velocity profile
- `plot_velocity_section()` — Time-depth velocity section

### Time Series

- `plot_time_series()` — Single cell time series
- `plot_stick_plot()` — Vector stick plot

### Quality Control

- `plot_mask()` — QC mask visualization
- `plot_statistics()` — QC statistics summary

### Diagnostics

- `plot_correlation()` — Correlation patterns
- `plot_echo_intensity()` — Echo intensity patterns

---

## See Also

- {doc}`core` — ProcessedDataset for data preparation
- {doc}`/io/binary_reader` — Data loading
