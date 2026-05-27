# pd0_parser Module

Low-level Python module for parsing RDI ADCP binary files in PD0 format.

## Overview

`pd0_parser.py` provides low-level functions to parse RDI (RD Instruments) 
Acoustic Doppler Current Profiler (ADCP) binary files. It supports Workhorse, 
Ocean Surveyor, and DVS ADCPs and returns data as NumPy arrays for integration 
with the scientific Python ecosystem.

This module is the foundation layer of pyadps I/O operations. For most users, 
the higher-level {doc}`binary_reader` module provides a more convenient interface 
with xarray integration.

## Installation

The module requires Python 3.9+ and NumPy:

```python
import numpy as np
from pyadps.io.pd0_parser import fileheader, fixedleader, variableleader, datatype, ErrorCode
```

## Quick Start

```python
# Read complete velocity data from an RDI file
from pyadps.io import pd0_parser

velocity, n_ensembles, cells, beams, error = pd0_parser.datatype("path/to/file.000", "velocity")

if error == 0:
    print(f"Successfully read {n_ensembles} ensembles")
    print(f"Data shape: {velocity.shape}")  # (beams, cells, ensembles)
```

## Core Functions

### fileheader()

Parses the file header and builds arrays mapping ensemble locations and data types. 
This is typically the first function called and its output can be passed to other 
functions for efficiency.

```{function} fileheader(rdi_file)
Parse RDI file header to extract ensemble metadata.

:param rdi_file: Path to RDI binary file in PD0 format
:type rdi_file: str or Path
:returns: Tuple of (datatype, byte, byteskip, offset, idarray, ensemble, error_code)
:rtype: tuple
```

**Return Values:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `datatype` | `np.ndarray` | Number of data types per ensemble |
| 1 | `byte` | `np.ndarray` | Ensemble size in bytes (excluding checksum) |
| 2 | `byteskip` | `np.ndarray` | File offset to each ensemble start |
| 3 | `offset` | `np.ndarray` | Address offsets for data types within each ensemble |
| 4 | `idarray` | `np.ndarray` | Data type IDs for each ensemble |
| 5 | `ensemble` | `int` | Number of valid ensembles parsed |
| 6 | `error_code` | `int` | 0 on success, non-zero on error |

**Example:**

```python
dt, byte, byteskip, offset, idarray, n_ens, err = pd0_parser.fileheader("deployment.000")

if err == 0:
    print(f"File contains {n_ens} ensembles")
    print(f"Ensemble size: {byte[0]} bytes")
    print(f"Data types per ensemble: {dt[0]}")
```

---

### fixedleader()

Extracts Fixed Leader data containing system configuration, hardware information, 
and deployment settings that remain constant throughout a file.

```{function} fixedleader(rdi_file, byteskip=None, offset=None, idarray=None, ensemble=0)
Extract Fixed Leader data from RDI file.

:param rdi_file: Path to RDI binary file
:type rdi_file: str or Path
:param byteskip: File offsets from fileheader(). Auto-fetched if None.
:type byteskip: np.ndarray, optional
:param offset: Address offsets from fileheader(). Auto-fetched if None.
:type offset: np.ndarray, optional
:param idarray: Data type IDs from fileheader(). Auto-fetched if None.
:type idarray: np.ndarray, optional
:param ensemble: Number of ensembles. Auto-fetched if 0.
:type ensemble: int, optional
:returns: Tuple of (data, ensemble, error_code)
:rtype: tuple
```

**Return Values:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `data` | `np.ndarray` | Shape (36, n_ensembles) array of Fixed Leader fields |
| 1 | `ensemble` | `int` | Number of ensembles |
| 2 | `error_code` | `int` | 0 on success, non-zero on error |

**Fixed Leader Field Indices:**

| Index | Field | Units/Description |
|-------|-------|-------------------|
| 0 | Fixed Leader ID | 0x0000 or 0x0001 |
| 1 | CPU Firmware Version | — |
| 2 | CPU Firmware Revision | — |
| 3 | System Configuration | Bit field (frequency, beam pattern, etc.) |
| 4 | Real/Sim Flag | 0 = real data |
| 5 | Lag Length | — |
| 6 | Number of Beams | Typically 4 |
| 7 | Number of Cells | Depth bins (WN command) |
| 8 | Pings per Ensemble | WP command |
| 9 | Depth Cell Length | cm (WS command) |
| 10 | Blank After Transmit | cm (WF command) |
| 11 | Profiling Mode | WM command |
| 12 | Low Correlation Threshold | WC command |
| 13 | Number of Code Repetitions | — |
| 14 | Percent Good Minimum | WG command |
| 15 | Error Velocity Threshold | mm/s (WE command) |
| 16 | Time Per Ping - Minutes | TP command |
| 17 | Time Per Ping - Seconds | TP command |
| 18 | Time Per Ping - Hundredths | TP command |
| 19 | Coordinate Transform | EX command |
| 20 | Heading Alignment | 0.01° (EA command) |
| 21 | Heading Bias | 0.01° (EB command) |
| 22 | Sensor Source | EZ command |
| 23 | Sensors Available | Bit field |
| 24 | Bin 1 Distance | cm |
| 25 | Transmit Pulse Length | cm (WT command) |
| 26 | Reference Layer Average | WL command |
| 27 | False Target Threshold | WA command |
| 28 | Spare | — |
| 29 | Transmit Lag Distance | cm |
| 30 | CPU Board Serial Number | Big-endian 8-byte |
| 31 | System Bandwidth | WB command |
| 32 | System Power | CQ command |
| 33 | Spare | — |
| 34 | Instrument Serial Number | — |
| 35 | Beam Angle | degrees |

**Example:**

```python
fl_data, n_ens, err = pd0_parser.fixedleader("deployment.000")

if err == 0:
    n_beams = fl_data[6, 0]   # Number of beams
    n_cells = fl_data[7, 0]   # Number of depth cells
    cell_size = fl_data[9, 0] # Cell size in cm
    
    print(f"Configuration: {n_beams} beams, {n_cells} cells @ {cell_size} cm")
```

---

### variableleader()

Extracts Variable Leader data containing time-varying measurements: timestamps, 
heading, pitch, roll, temperature, pressure, and sensor readings.

```{function} variableleader(rdi_file, byteskip=None, offset=None, idarray=None, ensemble=0)
Extract Variable Leader data from RDI file.

:param rdi_file: Path to RDI binary file
:type rdi_file: str or Path
:param byteskip: File offsets from fileheader(). Auto-fetched if None.
:type byteskip: np.ndarray, optional
:param offset: Address offsets from fileheader(). Auto-fetched if None.
:type offset: np.ndarray, optional
:param idarray: Data type IDs from fileheader(). Auto-fetched if None.
:type idarray: np.ndarray, optional
:param ensemble: Number of ensembles. Auto-fetched if 0.
:type ensemble: int, optional
:returns: Tuple of (data, ensemble, error_code)
:rtype: tuple
```

**Return Values:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `data` | `np.ndarray` | Shape (48, n_ensembles) array of Variable Leader fields |
| 1 | `ensemble` | `int` | Number of ensembles |
| 2 | `error_code` | `int` | 0 on success, non-zero on error |

**Variable Leader Field Indices:**

| Index | Field | Units/Scale |
|-------|-------|-------------|
| 0 | Variable Leader ID | 0x0080 or 0x0081 |
| 1 | Ensemble Number (LSW) | — |
| 2 | Year | RTC year (0-99) |
| 3 | Month | 1-12 |
| 4 | Day | 1-31 |
| 5 | Hour | 0-23 |
| 6 | Minute | 0-59 |
| 7 | Second | 0-59 |
| 8 | Hundredths | 0-99 |
| 9 | Ensemble Number MSB | — |
| 10 | BIT Result | Built-in test status |
| 11 | Sound Speed | m/s |
| 12 | Transducer Depth | dm (0.1 m) |
| 13 | Heading | 0.01° |
| 14 | Pitch | 0.01° (signed) |
| 15 | Roll | 0.01° (signed) |
| 16 | Temperature | 0.01°C |
| 17 | Salinity | ppt (signed) |
| 18-20 | MPT (Min, Sec, Hundredths) | Minimum pre-ping time |
| 21 | Heading Std Dev | 1° |
| 22 | Pitch Std Dev | 0.1° |
| 23 | Roll Std Dev | 0.1° |
| 24-31 | ADC Channels 0-7 | Raw ADC counts |
| 32-35 | Error Status Word | 4 bytes |
| 36 | Reserved | — |
| 37 | Pressure | deca-pascals |
| 38 | Pressure Variance | deca-pascals |
| 39 | Spare | — |
| 40-47 | Y2K Time | Century, Year, Month, Day, Hour, Min, Sec, Hundredths |

**Example:**

```python
vl_data, n_ens, err = pd0_parser.variableleader("deployment.000")

if err == 0:
    # Extract time series
    heading = vl_data[13, :] * 0.01    # Convert to degrees
    pitch = vl_data[14, :] * 0.01      # Convert to degrees
    roll = vl_data[15, :] * 0.01       # Convert to degrees
    temperature = vl_data[16, :] * 0.01  # Convert to °C
    
    print(f"Mean heading: {heading.mean():.1f}°")
    print(f"Temperature range: {temperature.min():.1f} - {temperature.max():.1f}°C")
```

---

### datatype()

Extracts 3D profile data arrays for velocity, correlation, echo intensity, 
percent good, or status data.

```{function} datatype(filename, var_name, cell=0, beam=0, byteskip=None, offset=None, idarray=None, ensemble=0)
Extract 3D profile data from RDI file.

:param filename: Path to RDI binary file
:type filename: str or Path
:param var_name: Data type name ('velocity', 'correlation', 'echo', 'percent good', 'status')
:type var_name: str
:param cell: Cell counts array. Auto-fetched if 0.
:type cell: int or np.ndarray, optional
:param beam: Beam counts array. Auto-fetched if 0.
:type beam: int or np.ndarray, optional
:param byteskip: File offsets from fileheader(). Auto-fetched if None.
:type byteskip: np.ndarray, optional
:param offset: Address offsets from fileheader(). Auto-fetched if None.
:type offset: np.ndarray, optional
:param idarray: Data type IDs from fileheader(). Auto-fetched if None.
:type idarray: np.ndarray, optional
:param ensemble: Number of ensembles. Auto-fetched if 0.
:type ensemble: int, optional
:returns: Tuple of (data, ensemble, cell_array, beam_array, error_code)
:rtype: tuple
```

**Return Values:**

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | `data` | `np.ndarray` | Shape (beams, cells, ensembles) |
| 1 | `ensemble` | `int` | Number of ensembles |
| 2 | `cell_array` | `np.ndarray` | Cell counts per ensemble |
| 3 | `beam_array` | `np.ndarray` | Beam counts per ensemble |
| 4 | `error_code` | `int` | 0 on success, non-zero on error |

**Data Types and Characteristics:**

| Variable | Data ID | dtype | Missing Value | Units |
|----------|---------|-------|---------------|-------|
| `velocity` | 0x0100/0x0101 | `int16` | -32768 | mm/s |
| `correlation` | 0x0200/0x0201 | `uint8` | 0 | counts (0-255) |
| `echo` | 0x0300/0x0301 | `uint8` | 0 | counts (0-255) |
| `percent good` | 0x0400/0x0401 | `uint8` | 0 | percent (0-100) |
| `status` | 0x0500/0x0501 | `uint8` | 0 | bit flags |

**Example:**

```python
# Read velocity data
vel, n_ens, cells, beams, err = pd0_parser.datatype("deployment.000", "velocity")

if err == 0:
    # vel shape: (4, 30, 1000) for 4 beams, 30 cells, 1000 ensembles
    
    # Mask invalid values
    vel_masked = np.ma.masked_equal(vel, -32768)
    
    # Time series at beam 0, cell 10
    v_timeseries = vel_masked[0, 10, :]
    
    # Depth profile at ensemble 500
    v_profile = vel_masked[:, :, 500]
    
    print(f"Mean velocity (beam 0, cell 10): {v_timeseries.mean():.1f} mm/s")
```

---

## Error Handling

All functions return an error code as part of their return tuple. Use the 
`ErrorCode` enum to interpret results:

```python
from pyadps.io.pd0_parser import ErrorCode

# Check error codes
if error == ErrorCode.SUCCESS.code:  # 0
    print("Success!")
elif error == ErrorCode.FILE_NOT_FOUND.code:  # 1
    print("File not found")
elif error == ErrorCode.CHECKSUM_ERROR.code:  # 10
    print("Checksum verification failed - data may be corrupted")
```

**Error Code Reference:**

| Code | Name | Description |
|------|------|-------------|
| 0 | `SUCCESS` | Operation completed successfully |
| 1 | `FILE_NOT_FOUND` | File does not exist |
| 2 | `PERMISSION_DENIED` | Access denied |
| 3 | `IO_ERROR` | File open/read failed |
| 4 | `OUT_OF_MEMORY` | Insufficient memory |
| 5 | `WRONG_RDIFILE_TYPE` | Not a valid RDI PD0 file |
| 6 | `ID_NOT_FOUND` | Data type ID not found in ensemble |
| 7 | `DATATYPE_MISMATCH` | Inconsistent data types between ensembles |
| 8 | `FILE_CORRUPTED` | Invalid file structure or truncated data |
| 9 | `VALUE_ERROR` | Invalid argument provided |
| 10 | `CHECKSUM_ERROR` | Ensemble checksum verification failed |
| 99 | `UNKNOWN_ERROR` | Unexpected error |

**Getting Error Messages:**

```python
message = ErrorCode.get_message(error)
print(message)  # e.g., "Error: File not found."
```

---

## Performance Optimization

For processing multiple data types from the same file, call `fileheader()` once 
and pass results to other functions:

```python
from pyadps.io import pd0_parser

# Efficient: call fileheader() once
dt, byte, byteskip, offset, idarray, n_ens, err = pd0_parser.fileheader("large_file.000")

# Reuse header data for all subsequent calls
fl_data, _, _ = pd0_parser.fixedleader("large_file.000", byteskip, offset, idarray, n_ens)
vl_data, _, _ = pd0_parser.variableleader("large_file.000", byteskip, offset, idarray, n_ens)
vel, _, _, _, _ = pd0_parser.datatype("large_file.000", "velocity", 
                           byteskip=byteskip, offset=offset, 
                           idarray=idarray, ensemble=n_ens)
echo, _, _, _, _ = pd0_parser.datatype("large_file.000", "echo",
                            byteskip=byteskip, offset=offset, 
                            idarray=idarray, ensemble=n_ens)
```

---

## Complete Example

```python
"""Complete workflow for reading an RDI ADCP file."""
import numpy as np
from pyadps.io import pd0_parser

# File path
rdi_file = "mooring_deployment.000"

# Step 1: Parse file header
dt, byte, byteskip, offset, idarray, n_ens, err = pd0_parser.fileheader(rdi_file)

if err != 0:
    print(f"Error: {pd0_parser.ErrorCode.get_message(err)}")
    exit(1)

print(f"File contains {n_ens} valid ensembles")

# Step 2: Get configuration from Fixed Leader
fl_data, _, _ = pd0_parser.fixedleader(rdi_file, byteskip, offset, idarray, n_ens)
n_beams = int(fl_data[6, 0])
n_cells = int(fl_data[7, 0])
cell_size_cm = int(fl_data[9, 0])
bin1_dist_cm = int(fl_data[24, 0])

print(f"Configuration: {n_beams} beams, {n_cells} cells")
print(f"Cell size: {cell_size_cm} cm, Bin 1 distance: {bin1_dist_cm} cm")

# Step 3: Get timestamps and motion from Variable Leader
vl_data, _, _ = pd0_parser.variableleader(rdi_file, byteskip, offset, idarray, n_ens)
heading = vl_data[13, :] * 0.01
pitch = vl_data[14, :] * 0.01
roll = vl_data[15, :] * 0.01
temperature = vl_data[16, :] * 0.01

print(f"Temperature: {temperature.mean():.2f} ± {temperature.std():.2f} °C")

# Step 4: Extract velocity data
vel, _, cells, beams, _ = pd0_parser.datatype(rdi_file, "velocity",
                                   byteskip=byteskip, offset=offset,
                                   idarray=idarray, ensemble=n_ens)

# Mask invalid values
vel_masked = np.ma.masked_equal(vel, -32768)

# Calculate depth array
depths_m = (bin1_dist_cm + np.arange(n_cells) * cell_size_cm) / 100.0

print(f"\nDepth range: {depths_m[0]:.1f} - {depths_m[-1]:.1f} m")
print(f"Velocity data shape: {vel_masked.shape}")
print(f"Valid data percentage: {(~vel_masked.mask).mean() * 100:.1f}%")

# Step 5: Extract quality metrics
corr, _, _, _, _ = pd0_parser.datatype(rdi_file, "correlation",
                            byteskip=byteskip, offset=offset,
                            idarray=idarray, ensemble=n_ens)
echo_int, _, _, _, _ = pd0_parser.datatype(rdi_file, "echo",
                                byteskip=byteskip, offset=offset,
                                idarray=idarray, ensemble=n_ens)

print(f"Mean correlation: {corr.mean():.1f}")
print(f"Mean echo intensity: {echo_int.mean():.1f}")
```

---

## Logging

The module uses Python's `logging` module. Configure logging to see detailed 
information:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("pyadps.io.pd0_parser")
logger.setLevel(logging.DEBUG)

# Now function calls will produce detailed log messages
```

---

## Technical Notes

### Checksum Verification

Per RDI specification Section 7.2, checksums are verified for every ensemble. 
If verification fails, parsing stops and accumulated valid data is returned. 
The checksum is calculated by summing all bytes in the ensemble (excluding the 
2-byte checksum) and comparing the lower 16 bits.

### Byte Order

All RDI data is little-endian except for the CPU Board Serial Number which is 
big-endian.

### Data IDs

The module recognizes these standard RDI data type IDs:

| ID (hex) | ID (decimal) | Description |
|----------|--------------|-------------|
| 0x7F7F | 32639 | Header |
| 0x0000, 0x0001 | 0, 1 | Fixed Leader |
| 0x0080, 0x0081 | 128, 129 | Variable Leader |
| 0x0100, 0x0101 | 256, 257 | Velocity |
| 0x0200, 0x0201 | 512, 513 | Correlation |
| 0x0300, 0x0301 | 768, 769 | Echo Intensity |
| 0x0400, 0x0401 | 1024, 1025 | Percent Good |
| 0x0500, 0x0501 | 1280, 1281 | Status |

### File Truncation Handling

If a file is truncated or corrupted mid-ensemble, the module returns all 
successfully parsed ensembles up to the point of corruption with an appropriate 
error code.

---

## See Also

- {doc}`binary_reader` — Higher-level wrapper with xarray integration
- {doc}`accessors` — Domain-specific accessor methods
- [RDI WorkHorse Commands and Output Data Format](https://www.teledynemarine.com/) specification (P/N 957-6156-00)
