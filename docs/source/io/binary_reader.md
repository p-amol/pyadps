# binary_reader Module

High-level data loading module for RDI ADCP files with xarray integration.

## Overview

`binary_reader.py` is the primary data loading module in pyadps. It provides 
functions to read Teledyne RDI Acoustic Doppler Current Profiler (ADCP) binary 
files (PD0 format) and convert them into `xarray.Dataset` objects for modern 
scientific Python workflows.

This module builds on {doc}`pd0_parser` to provide:

- Automatic metadata extraction and CF Convention attributes
- xarray.Dataset output with proper dimensions and coordinates
- Decoded fields with human-readable values
- Integration with {doc}`accessors` for domain-specific analysis

## Installation

```python
import pyadps
import pyadps.accessors  # Register domain-specific accessor methods

# Load complete ADCP dataset
ds = pyadps.read('deployment.000')
```

## Quick Start

```python
import pyadps
import pyadps.accessors  # Register accessor methods

# Load complete ADCP dataset
ds = pyadps.read('deployment.000')

# Access velocity data
velocity = ds['velocity']
print(velocity.shape)  # (beam, cell, ensemble) or (beam, cell, time)

# Use xarray features directly
ds['velocity'].plot()
ds.mean(dim='time')
ds.to_netcdf('output.nc')

# Access domain-specific methods via accessors
config = ds.fixed_leader.system_configuration()
ds.variable_leader.summary()
```

---

## Module Architecture

The module follows a three-layer processing pipeline:

| Layer | Module | Purpose |
|-------|--------|---------|
| 1 | `pd0_parser.py` | Low-level binary parsing (bytes → numpy arrays) |
| 2 | `binary_reader.py` | xarray.Dataset conversion with metadata |
| 3 | `accessors.py` | Domain-specific analysis methods |

---

## Primary Functions

### read() — Main Entry Point

The primary function for loading complete ADCP datasets.

```{function} read(adcp_file, include_decoded=True, data_types=None, use_time_as_primary_dim=True, use_depth_as_primary_dim=False, include_header=False)
Load complete ADCP dataset from PD0 file.

:param adcp_file: Path to ADCP binary file (PD0 format)
:type adcp_file: str or Path
:param include_decoded: Include computed/decoded fields (default: True)
:type include_decoded: bool
:param data_types: Specific data types to load (default: all)
:type data_types: list, optional
:param use_time_as_primary_dim: Use time as primary dimension (default: True)
:type use_time_as_primary_dim: bool
:param use_depth_as_primary_dim: Use depth instead of cell index (default: False)
:type use_depth_as_primary_dim: bool
:param include_header: Include raw header metadata (default: False)
:type include_header: bool
:returns: Dataset with all requested ADCP data
:rtype: xarray.Dataset
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adcp_file` | str or Path | — | Path to ADCP binary file (PD0 format) |
| `include_decoded` | bool | True | Include computed/decoded fields |
| `data_types` | list | None | Specific data types to load |
| `use_time_as_primary_dim` | bool | True | Use time as primary dimension |
| `use_depth_as_primary_dim` | bool | False | Use depth instead of cell index |
| `include_header` | bool | False | Include raw header metadata |

**Example: Basic Usage**

```python
import pyadps

# Load everything
ds = pyadps.read('file.000')
print(ds)
```

**Example: Selective Loading**

```python
# Load only velocity and correlation data
ds = pyadps.read('file.000', data_types=['Velocity', 'Correlation'])

# Load only configuration (no measurement arrays)
ds = pyadps.read('file.000', data_types=['FixedLeader', 'VariableLeader'])

# Load raw fields only (faster, smaller dataset)
ds = pyadps.read('file.000', include_decoded=False)
```

**Available `data_types` Options:**

- Components: `'FixedLeader'`, `'VariableLeader'`
- Data arrays: `'Velocity'`, `'Correlation'`, `'Echo'`, `'PercentGood'`, `'Status'`

```{note}
VariableLeader is always read by default because it provides the time coordinate. 
Exclude it only if you explicitly don't need timestamp information.
```

---

### read_header() — File Header Metadata

Reads ADCP file header information including byte structure and data type mappings.

```{function} read_header(adcp_file)
Read ADCP file header metadata.

:param adcp_file: Path to ADCP binary file
:type adcp_file: str or Path
:returns: Dataset with header metadata
:rtype: xarray.Dataset
```

**Variables Included:**

| Variable | Description |
|----------|-------------|
| `data_type_array` | Data type identifiers |
| `byte_skip` | Byte offsets for ensembles |
| `address_offset` | Address offsets within ensembles |
| `data_id` | Data type IDs |

**Accessor Methods:**

```python
ds_header = pyadps.read_header('file.000')

# Check file integrity
ds_header.header.check_file()
ds_header.header.print_check_file()

# List available data types
ds_header.header.data_types(ens=0)
ds_header.header.get_available_data_types()

# Generate summary
print(ds_header.header.summary())

# Comprehensive validation
ds_header.header.validate()
```

---

### read_fixed_leader() — Configuration Data

Reads ADCP instrument configuration (36 raw fields + 25 decoded fields).

```{function} read_fixed_leader(adcp_file, include_decoded=True, json_file_path=None)
Read Fixed Leader configuration data.

:param adcp_file: Path to ADCP binary file
:type adcp_file: str or Path
:param include_decoded: Include decoded fields (default: True)
:type include_decoded: bool
:param json_file_path: Custom metadata file (optional)
:type json_file_path: str or Path, optional
:returns: Dataset with configuration variables
:rtype: xarray.Dataset
```

**Key Variables:**

| Variable | Description |
|----------|-------------|
| `system_configuration_code` | Raw system config (16-bit) |
| `frequency` | Decoded frequency (e.g., "300 kHz") |
| `beam_pattern` | Beam configuration |
| `number_of_cells` | Depth cell count |
| `pings_per_ensemble` | Averaging count |
| `depth_cell_length` | Cell size (cm) |
| `blank_after_transmit` | Blanking distance (cm) |
| `profiling_mode` | Operating mode |

**Accessor Methods:**

```python
ds_fl = pyadps.read_fixed_leader('file.000')

# System configuration breakdown
config = ds_fl.fixed_leader.system_configuration()
print(config['Frequency'])
print(config['Beam Pattern'])

# Full summary with all fields
ds_fl.fixed_leader.summary()

# Validation checks
ds_fl.fixed_leader.validate()
```

---

### read_variable_leader() — Sensor Data

Reads per-ensemble sensor measurements (48 raw + 46 decoded + 2 composite fields).

```{function} read_variable_leader(adcp_file, include_decoded=True, use_time_dim=False, frequency=None)
Read Variable Leader sensor data.

:param adcp_file: Path to ADCP binary file
:type adcp_file: str or Path
:param include_decoded: Include decoded fields (default: True)
:type include_decoded: bool
:param use_time_dim: Use time as dimension (default: False)
:type use_time_dim: bool
:param frequency: ADCP frequency for ADC scaling (auto-detect)
:type frequency: int, optional
:returns: Dataset with sensor variables
:rtype: xarray.Dataset
```

**Key Variables:**

| Category | Variables |
|----------|-----------|
| Timestamps | `rtc_year`, `rtc_month`, `rtc_day`, `rtc_hour`, `rtc_minute`, `rtc_second` |
| Motion | `heading`, `pitch`, `roll`, `heading_degrees`, `pitch_degrees`, `roll_degrees` |
| Environmental | `temperature`, `pressure`, `salinity`, `depth` |
| Diagnostics | `bit_result`, `speed_of_sound`, `error_status_word_1` through `_4` |
| ADC | `adc_channel_0` through `adc_channel_7`, `xmit_voltage`, `xmit_current`, `ambient_temperature` |

**Accessor Methods:**

```python
ds_vl = pyadps.read_variable_leader('file.000')

# Comprehensive summary
ds_vl.variable_leader.summary()

# Check ensemble continuity
continuity = ds_vl.variable_leader.ensemble_continuity_check()

# BIT test results
bit_summary = ds_vl.variable_leader.bit_result_summary()

# Error status analysis
esw_summary = ds_vl.variable_leader.error_status_word_summary()

# Motion sensor checks
motion = ds_vl.variable_leader.check_motion_sensors()

# Timestamp validation
ts_valid = ds_vl.variable_leader.validate_timestamps()

# Time interval analysis
interval = ds_vl.variable_leader.get_time_interval()
freq = ds_vl.variable_leader.get_time_interval_frequency()
```

---

### Data Type Readers

Individual functions for reading specific measurement arrays.

#### read_velocity()

```{function} read_velocity(adcp_file, cell=0, beam=0)
Read velocity data from ADCP file.

:param adcp_file: Path to ADCP binary file
:type adcp_file: str or Path
:param cell: Cell number (0 for all cells)
:type cell: int
:param beam: Beam number (0 for all beams)
:type beam: int
:returns: Dataset with velocity variable
:rtype: xarray.Dataset
```

- Shape: `(beam, cell, ensemble)`
- Units: mm/s (signed 16-bit integer)
- Missing value: -32768

#### read_correlation()

```{function} read_correlation(adcp_file)
Read correlation data from ADCP file.

:param adcp_file: Path to ADCP binary file
:type adcp_file: str or Path
:returns: Dataset with correlation variable
:rtype: xarray.Dataset
```

- Shape: `(beam, cell, ensemble)`
- Range: 0-255 (unsigned 8-bit)
- Higher values = better data quality

#### read_echo_intensity()

```{function} read_echo_intensity(adcp_file)
Read echo intensity data from ADCP file.

:param adcp_file: Path to ADCP binary file
:type adcp_file: str or Path
:returns: Dataset with echo_intensity variable
:rtype: xarray.Dataset
```

- Shape: `(beam, cell, ensemble)`
- Range: 0-255 (counts)
- Multiply by 0.45 to convert to dB

#### read_percent_good()

```{function} read_percent_good(adcp_file)
Read percent good data from ADCP file.

:param adcp_file: Path to ADCP binary file
:type adcp_file: str or Path
:returns: Dataset with percent_good variable
:rtype: xarray.Dataset
```

- Shape: `(beam, cell, ensemble)`
- Range: 0-100 (percentage)
- Higher values = more valid pings

#### read_status()

```{function} read_status(adcp_file)
Read status data from ADCP file.

:param adcp_file: Path to ADCP binary file
:type adcp_file: str or Path
:returns: Dataset with status variable
:rtype: xarray.Dataset
```

- Shape: `(beam, cell, ensemble)`
- Diagnostic bit flags

---

## Dataset Structure

### Dimensions

| Dimension | Description |
|-----------|-------------|
| `ensemble` | Time index (0 to N-1) |
| `cell` | Depth bin index (0 to M-1) |
| `beam` | Acoustic beam number (typically 0-3) |
| `data_type` | Header data type configurations |

### Coordinates

| Coordinate | Type | Description |
|------------|------|-------------|
| `time` | datetime64[ns] | Timestamp from RTC fields |
| `depth` | float32 | Distance from transducer (meters) |

### Dataset Attributes

```python
ds.attrs['filename']          # Original filename
ds.attrs['total_ensembles']   # Number of ensembles
ds.attrs['file_size_bytes']   # File size
ds.attrs['pyadps_version']    # "1.0.0"
ds.attrs['adcp_data_format']  # "PD0"
ds.attrs['components']        # Loaded components dictionary
```

---

## Accessor Pattern

Domain-specific methods are accessed through namespaced accessors:

```python
import pyadps.accessors  # Required!

# Header operations
ds.header.data_types()
ds.header.check_file()
ds.header.summary()

# Fixed Leader operations
ds.fixed_leader.system_configuration()
ds.fixed_leader.summary()
ds.fixed_leader.validate()

# Variable Leader operations
ds.variable_leader.summary()
ds.variable_leader.ensemble_continuity_check()
ds.variable_leader.bit_result_summary()
ds.variable_leader.get_time_interval()
```

```{warning}
Import `pyadps.accessors` to register accessor methods. Without this import,
methods like `ds.header.*`, `ds.fixed_leader.*`, etc. will not be available.
```

---

## Time Series Utilities

### snap_time_axis()

Snap/round time coordinates to regular intervals.

```{function} snap_time_axis(ds, freq='h', tolerance='5min', target_minute=0)
Snap time coordinates to regular intervals.

:param ds: Dataset with time coordinate
:type ds: xarray.Dataset
:param freq: Target frequency ('h', 'min', '30min', etc.)
:type freq: str
:param tolerance: Maximum allowed correction
:type tolerance: str
:param target_minute: Target minute within hour (optional)
:type target_minute: int
:returns: Tuple of (snapped_dataset, success, message)
:rtype: tuple
```

**Example:**

```python
from pyadps.io.binary_reader import snap_time_axis

ds_snapped, success, message = snap_time_axis(
    ds,
    freq='h',              # Target frequency
    tolerance='5min',      # Maximum allowed correction
    target_minute=0        # Target minute within hour
)

if success:
    print("Time axis snapped successfully")
else:
    print(f"Snapping failed: {message}")
```

### fill_time_gaps()

Fill missing timestamps in time series with NaN data.

```{function} fill_time_gaps(ds, method='h')
Fill missing timestamps with NaN data.

:param ds: Dataset with time coordinate
:type ds: xarray.Dataset
:param method: Fill frequency ('h', 'min', '30min', etc.)
:type method: str
:returns: Dataset with filled gaps
:rtype: xarray.Dataset
```

**Example:**

```python
from pyadps.io.binary_reader import fill_time_gaps

ds_filled = fill_time_gaps(
    ds,
    method='h'  # Fill frequency
)
```

---

## Common Workflows

### Basic Data Loading and Export

```python
import pyadps
import pyadps.accessors

# Load data
ds = pyadps.read('deployment.000')

# Quick inspection
print(ds)
print(ds['velocity'].shape)

# Export to NetCDF
ds.to_netcdf('processed_data.nc')
```

### Performance-Optimized Loading

```python
# Pre-fetch header for multiple component reads
ds_header = pyadps.read_header('file.000')

# Extract parameters once
byteskip = ds_header.byte_skip.values
offset = ds_header.address_offset.values
idarray = ds_header.data_id.values
n_ensembles = ds_header.attrs['total_ensembles']

# Read components with pre-fetched parameters (faster)
ds_fl = pyadps.read_fixed_leader(
    'file.000',
    byteskip=byteskip,
    offset=offset,
    idarray=idarray,
    ensemble=n_ensembles
)

ds_vl = pyadps.read_variable_leader(
    'file.000',
    byteskip=byteskip,
    offset=offset,
    idarray=idarray,
    ensemble=n_ensembles
)
```

### Quality Control Workflow

```python
import pyadps
import pyadps.accessors

ds = pyadps.read('file.000')

# Check file integrity
ds.header.check_file()

# Validate configuration
ds.fixed_leader.validate()

# Check data quality
ds.variable_leader.summary()

# Examine BIT test results
bit_summary = ds.variable_leader.bit_result_summary()
if not bit_summary['all_passed']:
    print(f"BIT errors in {bit_summary['error_count']} ensembles")

# Check for time gaps
continuity = ds.variable_leader.ensemble_continuity_check()
if not continuity['is_continuous']:
    print(f"Found {continuity['gap_count']} gaps in ensemble numbers")
```

### Time Series Processing

```python
from pyadps.io.binary_reader import snap_time_axis, fill_time_gaps

ds = pyadps.read('file.000')

# Check time regularity
interval = ds.variable_leader.get_time_interval()
freq_dist = ds.variable_leader.get_time_interval_frequency()
print(f"Modal interval: {interval}")
print(f"Interval distribution: {freq_dist}")

# Snap time to regular grid
ds_snapped, success, msg = snap_time_axis(ds, freq='h', tolerance='30s')

if success:
    # Fill any remaining gaps
    ds_regular = fill_time_gaps(ds_snapped, method='h')
else:
    # If snapping failed, just fill gaps
    ds_regular = fill_time_gaps(ds, method='h')
```

### Selective Data Loading

```python
# Load only what you need for faster processing

# Just velocities (VariableLeader included for time)
ds = pyadps.read('file.000', data_types=['Velocity'])

# Configuration only
ds = pyadps.read('file.000', data_types=['FixedLeader', 'VariableLeader'])

# Raw fields only (no decoded computations)
ds = pyadps.read('file.000', include_decoded=False)

# Multiple data types
ds = pyadps.read('file.000', data_types=['Velocity', 'Correlation', 'Echo'])
```

---

## Error Handling

The module provides informative error messages for common issues:

```python
try:
    ds = pyadps.read('file.000')
except FileNotFoundError:
    print("ADCP file not found")
except ValueError as e:
    print(f"Invalid file format: {e}")
except KeyError as e:
    print(f"Missing required field: {e}")
```

### Error Codes

The module uses `pd0_parser.ErrorCode` for binary parsing errors:

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | File not found |
| 2 | Permission denied |
| 3 | I/O error |
| 4 | Out of memory |

Access error messages via:

```python
ds.attrs['error_message']
```

---

## CF Convention Compliance

All variables include CF Convention attributes:

```python
# Variable attributes
ds['velocity'].attrs['long_name']       # Human-readable name
ds['velocity'].attrs['units']           # UDUNITS2 format
ds['velocity'].attrs['valid_min']       # Valid range minimum
ds['velocity'].attrs['valid_max']       # Valid range maximum
ds['velocity'].attrs['_FillValue']      # Missing data indicator
ds['velocity'].attrs['scale_factor']    # Scaling factor
ds['velocity'].attrs['description']     # Detailed description
```

---

## Migration from v0.4.0

### Before (v0.4.0)

```python
import pyadps

ds = pyadps.ReadFile('file.000')
velocity = ds.velocity.data
time = ds.time
```

### After (v1.0.0)

```python
import pyadps
import pyadps.accessors

ds = pyadps.read('file.000')
velocity = ds['velocity'].values  # or ds.velocity.values
time = ds['time'].values
```

### Key Differences

| Feature | v0.4.0 | v1.0.0 |
|---------|--------|--------|
| Entry point | `pyadps.ReadFile()` | `pyadps.read()` |
| Data structure | Custom class | xarray.Dataset |
| Data access | `ds.velocity.data` | `ds['velocity'].values` |
| Custom methods | Direct attributes | Accessor namespaces |
| NetCDF export | Manual | `ds.to_netcdf()` |
| Plotting | Manual | `ds['var'].plot()` |

---

## Dependencies

- **xarray**: Core data structure
- **numpy**: Numerical arrays
- **pandas**: Time series handling
- **pd0_parser**: Binary parsing (internal module)

---

## See Also

- {doc}`pd0_parser` — Low-level binary parsing
- {doc}`accessors` — Accessor class implementations
- `fixed_leader_meta.json` — Fixed Leader field definitions
- `variable_leader_meta.json` — Variable Leader field definitions

---

## Version Information

- **Module version:** 1.0.0
- **Supported formats:** PD0 (RDI WorkHorse)
- **Python:** 3.9+
