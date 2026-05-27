# accessors Module

xarray accessors providing domain-specific methods for ADCP data analysis.

## Overview

`accessors.py` extends xarray.Dataset objects with ADCP-specific functionality using 
the xarray accessor pattern. When you import `pyadps.accessors`, it registers 
custom namespaces that attach domain-specific methods to any Dataset created by 
the `binary_reader` module.

This design provides:

- Full xarray compatibility (all standard methods work normally)
- No naming conflicts with xarray's API
- Organized, discoverable methods under component-specific namespaces
- Lazy evaluation for performance

## Installation

Import the accessors module to register the accessor namespaces:

```python
import pyadps
import pyadps.accessors  # Required to register accessors

# Now accessor methods are available on any pyadps Dataset
ds = pyadps.read('file.000')
ds.header.summary()           # Header accessor
ds.fixed_leader.validate()    # Fixed Leader accessor
ds.variable_leader.summary()  # Variable Leader accessor
```

```{warning}
You must import `pyadps.accessors` before using accessor methods. Without this 
import, methods like `ds.header.summary()` will raise an `AttributeError`.
```

## Quick Start

```python
import pyadps
import pyadps.accessors

# Load data
ds = pyadps.read('deployment.000')

# Check file integrity
ds.header.check_file()

# View system configuration
config = ds.fixed_leader.system_configuration()
print(config['Frequency'])

# Check data quality
ds.variable_leader.summary()

# Validate timestamps
ts_valid = ds.variable_leader.validate_timestamps()
```

## Available Accessors

| Accessor | Namespace | Description |
|----------|-----------|-------------|
| `HeaderAccessor` | `ds.header.*` | File structure and integrity checks |
| `FixedLeaderAccessor` | `ds.fixed_leader.*` | System configuration and validation |
| `VariableLeaderAccessor` | `ds.variable_leader.*` | Sensor data analysis and diagnostics |

---

## HeaderAccessor

Access via: `ds.header.*`

The Header accessor provides methods for file structure validation and integrity 
checking. Use it to verify that an ADCP file was read correctly and has consistent 
structure across all ensembles.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `ensemble` | `int` | Total number of ensembles |
| `data_type` | `int` | Number of data types per ensemble |
| `error_message` | `str` | Error message from file reading |
| `error_code` | `int` | Numeric error code (0 = success) |
| `file_size_bytes` | `int` | Actual file size on disk |
| `calculated_size_bytes` | `int` | Expected file size from structure |
| `file_size_match` | `bool` | True if actual matches expected |

### Methods

#### check_file()

Performs comprehensive file integrity checks.

```{function} check_file()
Check file structure validity and uniformity.

:returns: Dictionary with check results
:rtype: dict
```

**Return Dictionary Keys:**

| Key | Type | Description |
|-----|------|-------------|
| `File Size (MB)` | `float` | File size in megabytes |
| `File Size Match` | `bool` | Actual vs calculated size match |
| `Byte Uniformity` | `bool` | All ensembles same byte count |
| `Data Type Uniformity` | `bool` | All ensembles same data types |
| `Byte Skip Uniformity` | `bool` | Consistent byte offsets |
| `Address Offset Uniformity` | `bool` | Consistent address offsets |
| `Data ID Uniformity` | `bool` | Consistent data type IDs |

**Example:**

```python
ds = pyadps.read_header('file.000')
check = ds.header.check_file()

if check['File Size Match'] and check['Byte Uniformity']:
    print("File structure is valid")
else:
    print("WARNING: File may be corrupted")
```

---

#### get_available_data_types()

Lists data types present in the file.

```{function} get_available_data_types(ens=0)
Get list of data types in a specific ensemble.

:param ens: Ensemble index (default: 0)
:type ens: int
:returns: List of data type names
:rtype: list[str]
```

**Example:**

```python
data_types = ds.header.get_available_data_types(ens=0)
print(data_types)
# ['Fixed Leader', 'Variable Leader', 'Velocity', 'Correlation', 'Echo', 'Percent Good']
```

---

#### has_data_type()

Check if a specific data type is present.

```{function} has_data_type(data_type_name)
Check if a data type exists in the file.

:param data_type_name: Name of data type to check
:type data_type_name: str
:returns: True if present
:rtype: bool
```

**Valid Data Type Names:**

- `'Fixed Leader'`
- `'Variable Leader'`
- `'Velocity'`
- `'Correlation'`
- `'Echo'`
- `'Percent Good'`
- `'Status'`
- `'Bottom Track'`

**Example:**

```python
if ds.header.has_data_type('Bottom Track'):
    print("Bottom tracking data available")
```

---

#### get_ensemble_info()

Get detailed information about a specific ensemble.

```{function} get_ensemble_info(ensemble)
Get metadata for a specific ensemble.

:param ensemble: Ensemble index
:type ensemble: int
:returns: Dictionary with ensemble metadata
:rtype: dict
```

**Example:**

```python
info = ds.header.get_ensemble_info(0)
print(f"Ensemble size: {info['byte_size']} bytes")
print(f"Data types: {info['data_types']}")
```

---

#### validate()

Comprehensive validation with detailed reporting.

```{function} validate()
Validate header structure and return detailed report.

:returns: Dictionary with validation results
:rtype: dict
```

**Example:**

```python
report = ds.header.validate()
if report['valid']:
    print("Header validation passed")
else:
    for issue in report['issues']:
        print(f"Issue: {issue}")
```

---

#### summary()

Print comprehensive header summary.

```{function} summary()
Print formatted header summary to stdout.
```

**Example:**

```python
ds.header.summary()
```

**Output:**

```
======================================================================
                        FILE HEADER SUMMARY
======================================================================
Source File    : /path/to/deployment.000
File Size      : 125.34 MB
Total Ensembles: 10000

======================================================================
                    INTEGRITY CHECK (CRITICAL)
======================================================================
File Size Match             : PASS
Byte Uniformity             : PASS
Datatype Uniformity         : PASS
Byte Skip Uniformity        : PASS
Address Offset Uniformity   : PASS
Data ID Uniformity          : PASS
Overall Status              : HEALTHY

======================================================================
                    DATA TYPES (Ensemble 0)
======================================================================
  - Fixed Leader
  - Variable Leader
  - Velocity
  - Correlation
  - Echo
  - Percent Good

======================================================================
```

---

## FixedLeaderAccessor

Access via: `ds.fixed_leader.*`

The Fixed Leader accessor provides methods for interpreting ADCP system configuration 
data. Fixed Leader data contains settings that typically remain constant throughout 
a deployment.

### Methods

#### system_configuration()

Decode the system configuration bit field.

```{function} system_configuration(ens=-1)
Get human-readable system configuration.

:param ens: Ensemble index (-1 for first ensemble)
:type ens: int
:returns: Dictionary of configuration settings
:rtype: dict
```

**Return Dictionary Keys:**

| Key | Example Value | Description |
|-----|---------------|-------------|
| `Frequency` | `"300 kHz"` | Acoustic frequency |
| `Beam Pattern` | `"Concave"` | Beam geometry |
| `Sensor Configuration` | `"Configuration #1"` | Sensor setup |
| `XDCR HD` | `"Not attached"` | Transducer head status |
| `Beam Direction` | `"Downward"` | Looking direction |
| `Beam Angle` | `"20°"` | Beam angle from vertical |
| `Janus Configuration` | `"Type 2"` | Janus config type |

**Example:**

```python
ds = pyadps.read_fixed_leader('file.000')
config = ds.fixed_leader.system_configuration()

print(f"Frequency: {config['Frequency']}")
print(f"Beam Pattern: {config['Beam Pattern']}")
print(f"Direction: {config['Beam Direction']}")
```

---

#### coordinate_transformation()

Get coordinate transformation settings.

```{function} coordinate_transformation(ens=0)
Get coordinate transformation configuration.

:param ens: Ensemble index
:type ens: int
:returns: Dictionary of transformation settings
:rtype: dict
```

**Return Dictionary Keys:**

| Key | Type | Description |
|-----|------|-------------|
| `Coordinates` | `str` | Coordinate system (Beam, Instrument, Ship, Earth) |
| `Tilt Correction` | `bool` | Tilt sensors used in transformation |
| `Three-Beam Solution` | `bool` | Three-beam solutions allowed |
| `Bin Mapping` | `bool` | Bin mapping used |

**Example:**

```python
transform = ds.fixed_leader.coordinate_transformation()

print(f"Coordinate System: {transform['Coordinates']}")
if transform['Tilt Correction']:
    print("Tilt correction is enabled")
```

---

#### sensor_info()

Get sensor availability or source selection.

```{function} sensor_info(ens=0, field='source')
Get sensor configuration information.

:param ens: Ensemble index
:type ens: int
:param field: 'source' for selection, 'avail' for availability
:type field: str
:returns: Dictionary of sensor status
:rtype: dict
```

**Example:**

```python
# Check sensor sources
sources = ds.fixed_leader.sensor_info(field='source')
print(f"Heading source selected: {sources.get('Heading', False)}")

# Check sensor availability
available = ds.fixed_leader.sensor_info(field='avail')
print(f"Pressure sensor available: {available.get('Pressure', False)}")
```

---

#### is_uniform()

Check if Fixed Leader fields are uniform across ensembles.

```{function} is_uniform()
Check uniformity of each field across all ensembles.

:returns: Dictionary mapping field names to uniformity status
:rtype: dict[str, bool]
```

**Example:**

```python
uniformity = ds.fixed_leader.is_uniform()

for field, is_uniform in uniformity.items():
    status = "UNIFORM" if is_uniform else "VARIES"
    print(f"{field}: {status}")
```

---

#### validate()

Validate Fixed Leader data integrity.

```{function} validate()
Validate Fixed Leader data and return detailed report.

:returns: Dictionary with validation results
:rtype: dict
```

**Return Dictionary Keys:**

| Key | Type | Description |
|-----|------|-------------|
| `valid` | `bool` | Overall validation status |
| `issues` | `list` | Critical issues found |
| `warnings` | `list` | Non-critical warnings |

**Example:**

```python
report = ds.fixed_leader.validate()

if not report['valid']:
    print("Validation failed!")
    for issue in report['issues']:
        print(f"  - {issue}")
```

---

#### summary()

Print comprehensive Fixed Leader summary.

```{function} summary()
Print formatted Fixed Leader summary to stdout.
```

**Example:**

```python
ds.fixed_leader.summary()
```

---

## VariableLeaderAccessor

Access via: `ds.variable_leader.*`

The Variable Leader accessor provides methods for analyzing time-varying sensor 
data including timestamps, motion sensors, diagnostics, and environmental 
measurements.

### Methods

#### ensemble_continuity_check()

Check for gaps in ensemble numbering.

```{function} ensemble_continuity_check()
Check ensemble number continuity.

:returns: Dictionary with continuity analysis
:rtype: dict
```

**Return Dictionary Keys:**

| Key | Type | Description |
|-----|------|-------------|
| `is_continuous` | `bool` | True if no gaps |
| `gap_count` | `int` | Number of gaps found |
| `gap_locations` | `list` | Indices where gaps occur |
| `expected_count` | `int` | Expected ensemble count |
| `actual_count` | `int` | Actual ensemble count |

**Example:**

```python
ds = pyadps.read_variable_leader('file.000')
continuity = ds.variable_leader.ensemble_continuity_check()

if not continuity['is_continuous']:
    print(f"Found {continuity['gap_count']} gaps in data")
    print(f"Gap locations: {continuity['gap_locations'][:5]}...")
```

---

#### ensemble_rollover_count()

Count ensemble number rollovers (16-bit counter wraparound).

```{function} ensemble_rollover_count()
Count ensemble counter rollovers.

:returns: Number of rollovers detected
:rtype: int
```

**Example:**

```python
rollovers = ds.variable_leader.ensemble_rollover_count()
print(f"Ensemble counter rolled over {rollovers} times")
```

---

#### bit_result_summary()

Summarize Built-In Test (BIT) results.

```{function} bit_result_summary(include_decoded=True)
Analyze BIT test results across all ensembles.

:param include_decoded: Include detailed bit decoding
:type include_decoded: bool
:returns: Dictionary with BIT analysis
:rtype: dict
```

**Return Dictionary Keys:**

| Key | Type | Description |
|-----|------|-------------|
| `all_passed` | `bool` | True if all BIT tests passed |
| `error_count` | `int` | Number of ensembles with errors |
| `error_ensembles` | `list` | Indices of failed ensembles |
| `bit_flags` | `dict` | Decoded bit flag summary (if include_decoded) |

**BIT Flag Meanings:**

| Bit | Name | Meaning if Set |
|-----|------|----------------|
| 0 | DEMOD | Demodulator error |
| 1 | TIMING | Timing card error |
| 2 | DSP | DSP error |
| 3 | TEMP | Temperature sensor error |
| 4 | TILT | Tilt sensor error |
| 5 | VOLT | Voltage error |
| 6 | RCVR | Receiver error |
| 7 | XMTR | Transmitter error |

**Example:**

```python
bit_summary = ds.variable_leader.bit_result_summary()

if not bit_summary['all_passed']:
    print(f"BIT errors in {bit_summary['error_count']} ensembles")
    
    # Check specific flags
    if bit_summary.get('bit_flags', {}).get('TILT', 0) > 0:
        print("WARNING: Tilt sensor errors detected")
```

---

#### error_status_word_summary()

Summarize Error Status Word (ESW) occurrences.

```{function} error_status_word_summary(include_decoded=True)
Analyze ESW across all ensembles.

:param include_decoded: Include detailed bit decoding
:type include_decoded: bool
:returns: Dictionary with ESW analysis
:rtype: dict
```

**Example:**

```python
esw_summary = ds.variable_leader.error_status_word_summary()

print(f"Ensembles with errors: {esw_summary['error_count']}")
for flag_name, count in esw_summary.get('esw_flags', {}).items():
    if count > 0:
        print(f"  {flag_name}: {count} occurrences")
```

---

#### validate_timestamps()

Validate RTC timestamp fields.

```{function} validate_timestamps()
Validate timestamp field plausibility.

:returns: Dictionary with validation results
:rtype: dict
```

**Return Dictionary Keys:**

| Key | Type | Description |
|-----|------|-------------|
| `valid` | `bool` | Overall timestamp validity |
| `issues` | `list` | Specific issues found |
| `time_range` | `tuple` | (start_time, end_time) |
| `duration` | `timedelta` | Total deployment duration |

**Example:**

```python
ts_valid = ds.variable_leader.validate_timestamps()

if ts_valid['valid']:
    print(f"Time range: {ts_valid['time_range'][0]} to {ts_valid['time_range'][1]}")
    print(f"Duration: {ts_valid['duration']}")
else:
    for issue in ts_valid['issues']:
        print(f"Timestamp issue: {issue}")
```

---

#### get_time_interval()

Get modal time interval between ensembles.

```{function} get_time_interval()
Calculate the most common time interval.

:returns: Modal time interval
:rtype: pandas.Timedelta
```

**Example:**

```python
interval = ds.variable_leader.get_time_interval()
print(f"Sampling interval: {interval}")  # e.g., "0 days 01:00:00"
```

---

#### get_time_interval_frequency()

Get distribution of time intervals.

```{function} get_time_interval_frequency()
Get frequency distribution of time intervals.

:returns: Counter of interval occurrences
:rtype: collections.Counter
```

**Example:**

```python
freq_dist = ds.variable_leader.get_time_interval_frequency()

print("Time interval distribution:")
for interval, count in freq_dist.most_common(5):
    print(f"  {interval}: {count} occurrences")
```

---

#### check_motion_sensors()

Analyze motion sensor readings.

```{function} check_motion_sensors()
Check heading, pitch, and roll statistics.

:returns: Dictionary with motion sensor analysis
:rtype: dict
```

**Example:**

```python
motion = ds.variable_leader.check_motion_sensors()

print(f"Heading: {motion['heading_mean']:.1f}° ± {motion['heading_std']:.1f}°")
print(f"Pitch: {motion['pitch_mean']:.2f}° ± {motion['pitch_std']:.2f}°")
print(f"Roll: {motion['roll_mean']:.2f}° ± {motion['roll_std']:.2f}°")
```

---

#### summary()

Print comprehensive Variable Leader summary.

```{function} summary()
Print formatted Variable Leader summary to stdout.
```

**Example:**

```python
ds.variable_leader.summary()
```

---

## Complete Workflow Example

```python
"""Complete data quality assessment using accessors."""
import pyadps
import pyadps.accessors

# Load complete dataset
ds = pyadps.read('deployment.000')

# Step 1: Check file integrity
print("=== File Integrity ===")
check = ds.header.check_file()
if check['File Size Match']:
    print("✓ File structure valid")
else:
    print("✗ File may be corrupted")

# Step 2: Check configuration
print("\n=== System Configuration ===")
config = ds.fixed_leader.system_configuration()
print(f"Frequency: {config['Frequency']}")
print(f"Direction: {config['Beam Direction']}")

transform = ds.fixed_leader.coordinate_transformation()
print(f"Coordinates: {transform['Coordinates']}")

# Step 3: Validate configuration uniformity
validation = ds.fixed_leader.validate()
if validation['valid']:
    print("✓ Configuration valid")
else:
    for issue in validation['issues']:
        print(f"✗ {issue}")

# Step 4: Check data continuity
print("\n=== Data Continuity ===")
continuity = ds.variable_leader.ensemble_continuity_check()
if continuity['is_continuous']:
    print("✓ No gaps in ensemble numbers")
else:
    print(f"✗ Found {continuity['gap_count']} gaps")

# Step 5: Check diagnostics
print("\n=== Diagnostics ===")
bit_summary = ds.variable_leader.bit_result_summary()
if bit_summary['all_passed']:
    print("✓ All BIT tests passed")
else:
    print(f"✗ BIT errors in {bit_summary['error_count']} ensembles")

# Step 6: Validate timestamps
print("\n=== Timestamps ===")
ts_valid = ds.variable_leader.validate_timestamps()
if ts_valid['valid']:
    print(f"✓ Duration: {ts_valid['duration']}")
else:
    for issue in ts_valid['issues']:
        print(f"✗ {issue}")

# Step 7: Print full summaries
print("\n" + "="*70)
ds.header.summary()
ds.fixed_leader.summary()
ds.variable_leader.summary()
```

---

## Design Pattern

The accessor pattern used by pyadps follows xarray's extension mechanism:

```python
import xarray as xr

@xr.register_dataset_accessor("header")
class HeaderAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
    
    def check_file(self):
        # Access underlying Dataset via self._obj
        ...
```

This pattern provides several benefits:

1. **No subclassing** — Works with standard xarray.Dataset
2. **Lazy initialization** — Accessors only created when accessed
3. **Namespace isolation** — Methods grouped logically
4. **Full compatibility** — All xarray methods still work

---

## See Also

- {doc}`binary_reader` — Data loading functions that create accessor-compatible Datasets
- {doc}`pd0_parser` — Low-level binary parsing (accessors not available)
- [xarray Accessors Documentation](https://docs.xarray.dev/en/stable/extending.html#extending-with-accessors)
