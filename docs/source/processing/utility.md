# utility Module

Utility functions for ADCP data processing.

## Overview

The `utility` module provides helper functions used throughout the pyadps 
processing pipeline. These include coordinate transformations, unit conversions, 
and data manipulation utilities.

## Quick Start

```python
from pyadps.processing.utility import transform_coordinates, convert_units

# Transform from beam to Earth coordinates
velocity_earth = transform_coordinates(velocity_beam, heading, pitch, roll)

# Convert velocity units
velocity_ms = convert_units(velocity_mm, from_unit='mm/s', to_unit='m/s')
```

---

## Available Functions

### Coordinate Transformations

- `transform_beam_to_instrument()` — Beam to instrument coordinates
- `transform_instrument_to_ship()` — Instrument to ship coordinates
- `transform_ship_to_earth()` — Ship to Earth coordinates
- `transform_coordinates()` — Complete transformation pipeline

### Unit Conversions

- `convert_velocity_units()` — Convert velocity units
- `convert_depth_units()` — Convert depth units

### Data Utilities

- `calculate_depth_array()` — Calculate depth from cell configuration
- `mask_to_nan()` — Convert mask to NaN values
- `nan_to_mask()` — Convert NaN values to mask

---

## See Also

- {doc}`core` — ProcessedDataset orchestrator
- {doc}`/io/binary_reader` — Data loading
