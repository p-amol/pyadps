# I/O Module

The `pyadps.io` module handles reading and parsing RDI ADCP binary files in PD0 format.

## Overview

The I/O module provides a three-layer architecture for reading ADCP data:

| Layer | Module | Purpose |
|-------|--------|---------|
| 1 | `pd0_parser` | Low-level binary parsing (bytes → NumPy arrays) |
| 2 | `binary_reader` | xarray.Dataset conversion with metadata |
| 3 | `accessors` | Domain-specific analysis methods |

This layered design allows users to choose the appropriate abstraction level for their needs:

- **Most users** should use `binary_reader` for its convenient xarray integration
- **Advanced users** can use `pd0_parser` for direct binary access
- **All users** benefit from `accessors` for domain-specific operations

## Module Overview

| Module | Description |
|--------|-------------|
| {doc}`pd0_parser` | Low-level PD0 binary format parser returning NumPy arrays |
| {doc}`binary_reader` | High-level file reader returning xarray.Dataset objects |
| {doc}`accessors` | xarray accessors for domain-specific analysis methods |

## Typical Usage

For most users, start with `binary_reader`:

```python
import pyadps
import pyadps.accessors  # Register domain-specific accessor methods

# Load complete ADCP dataset
ds = pyadps.read('deployment.000')

# Access velocity data
velocity = ds['velocity']
print(f"Shape: {velocity.shape}")  # (beam, cell, time)

# Use xarray features directly
ds['velocity'].plot()
ds.to_netcdf('output.nc')

# Access domain-specific methods via accessors
config = ds.fixed_leader.system_configuration()
ds.variable_leader.summary()
```

For advanced users needing direct binary access, use `pd0_parser`:

```python
from pyadps.io import pd0_parser

# Parse file header
dt, byte, byteskip, offset, idarray, n_ens, err = pd0_parser.fileheader("file.000")

# Read velocity data as NumPy array
vel, n_ens, cells, beams, err = pd0_parser.datatype("file.000", "velocity")
```

## Data Flow

```
┌─────────────────┐
│  RDI PD0 File   │
│   (binary)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   pd0_parser    │  Layer 1: Binary → NumPy
│  (low-level)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  binary_reader  │  Layer 2: NumPy → xarray.Dataset
│  (high-level)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   accessors     │  Layer 3: Domain-specific methods
│  (analysis)     │
└─────────────────┘
```

## Quick Reference

### Loading Data

```python
import pyadps
import pyadps.accessors

# Complete dataset (most common)
ds = pyadps.read('file.000')

# Specific components only
ds = pyadps.read('file.000', data_types=['Velocity', 'Correlation'])

# Individual readers
ds_header = pyadps.read_header('file.000')
ds_fl = pyadps.read_fixed_leader('file.000')
ds_vl = pyadps.read_variable_leader('file.000')
ds_vel = pyadps.read_velocity('file.000')
```

### Accessor Methods

```python
# Header operations
ds.header.check_file()
ds.header.summary()

# Fixed Leader operations
ds.fixed_leader.system_configuration()
ds.fixed_leader.validate()

# Variable Leader operations
ds.variable_leader.ensemble_continuity_check()
ds.variable_leader.bit_result_summary()
```

## Contents

```{toctree}
:maxdepth: 2

pd0_parser
binary_reader
accessors
```

## See Also

- {doc}`/processing/index` — Quality control and data processing
- {doc}`/tutorials/index` — Step-by-step tutorials
