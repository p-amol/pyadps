# autoprocess Module

Unified entry point for automated ADCP data processing.

## Overview

The `autoprocess` module provides a single-function interface for complete ADCP 
data processing. It coordinates the full pipeline from reading binary files 
through quality control to final output, supporting both configuration files 
and programmatic usage.

**Key Features:**

- Single function interface for complete processing workflow
- Supports both config files (Streamlit UI) and programmatic usage
- Automatic output file generation
- Velocity-only export option with unit conversion
- Configurable depth ordering for output

## Quick Start

### Basic Usage

```python
from pyadps.processing.autoprocess import autoprocess

# Process using a config file
result = autoprocess('deployment_config.ini')

# Process with explicit binary file path
result = autoprocess('config.ini', binary_file_path='data/mooring_2024.000')
```

### Save Output to NetCDF

```python
# Save full processed dataset
result = autoprocess('config.ini', save_netcdf=True)

# Save to custom location
result = autoprocess(
    'config.ini',
    save_netcdf=True,
    output_dir='processed/',
    output_filename='deployment_2024_qc.nc'
)
```

---

## Function Reference

### autoprocess()

```{function} autoprocess(config_file_or_object, binary_file_path=None, save_netcdf=False, save_velocity_only=False, output_dir=None, output_filename=None, velocity_units='cm/s', ensure_depth_ascending=True, print_summary=True)
Process ADCP data using configuration file or object.

:param config_file_or_object: Path to .ini config file OR ProcessingConfig object
:type config_file_or_object: str, Path, or ProcessingConfig
:param binary_file_path: Path to ADCP binary file (overrides config)
:type binary_file_path: str or Path, optional
:param save_netcdf: Save processed dataset to NetCDF file
:type save_netcdf: bool
:param save_velocity_only: Save only velocity components (u, v, w)
:type save_velocity_only: bool
:param output_dir: Output directory (default: same as input)
:type output_dir: str or Path, optional
:param output_filename: Output filename (default: auto-generated)
:type output_filename: str, optional
:param velocity_units: Units for velocity output ('mm/s', 'cm/s', 'm/s')
:type velocity_units: str
:param ensure_depth_ascending: Ensure depth increases with index
:type ensure_depth_ascending: bool
:param print_summary: Print processing summary to console
:type print_summary: bool
:returns: Processed dataset with QC mask applied
:rtype: xarray.Dataset
```

---

## Usage Patterns

### Pattern 1: Config File Workflow (Streamlit)

When using the Streamlit UI, a config.ini file is generated with all processing 
parameters:

```python
# Config file contains input path and all QC parameters
result = autoprocess('config.ini')
```

The config file specifies:

- Input file location
- Time axis corrections
- Sensor health thresholds
- Signal quality parameters
- Profile operations
- Velocity check settings

### Pattern 2: Programmatic Workflow

For scripting and automation:

```python
from pyadps.processing.autoprocess import autoprocess
from pyadps.processing.config import ProcessingConfig

# Create config programmatically
config = ProcessingConfig()
config.isQCTest = True
config.correlation_threshold = 64
config.echo_intensity_threshold = 30

# Process with explicit binary path
result = autoprocess(
    config,
    binary_file_path='data/deployment.000',
    save_netcdf=True,
    output_dir='results/'
)
```

### Pattern 3: Velocity-Only Export

For applications that only need current velocity data:

```python
result = autoprocess(
    'config.ini',
    save_netcdf=True,
    save_velocity_only=True,
    velocity_units='m/s'  # SI units
)
```

This creates a streamlined NetCDF with:

- `u` — Eastward velocity (2D: depth × time)
- `v` — Northward velocity (2D: depth × time)
- `w` — Vertical velocity (2D: depth × time)
- `time` — Time coordinate
- `depth` — Depth coordinate

### Pattern 4: Silent Processing

For batch scripts where console output is not needed:

```python
result = autoprocess(
    'config.ini',
    print_summary=False
)
```

---

## Output Options

### Full Dataset Export

Default output includes all variables and the QC mask:

```python
result = autoprocess('config.ini', save_netcdf=True)
# Output: input_filename_processed.nc
```

### Velocity-Only Export

Streamlined output with only velocity components:

```python
result = autoprocess(
    'config.ini',
    save_netcdf=True,
    save_velocity_only=True
)
# Output: input_filename_velocity.nc
```

### Velocity Unit Options

| Unit | Description | Use Case |
|------|-------------|----------|
| `"mm/s"` | Millimeters per second | Original instrument units |
| `"cm/s"` | Centimeters per second | Common oceanographic convention |
| `"m/s"` | Meters per second | SI units, numerical modeling |

```python
# Export in SI units
result = autoprocess(
    'config.ini',
    save_netcdf=True,
    save_velocity_only=True,
    velocity_units='m/s'
)
```

---

## Batch Processing Example

```python
from pathlib import Path
from pyadps.processing.autoprocess import autoprocess
from pyadps.processing.config import ProcessingConfig

# Create shared config
config = ProcessingConfig()
config.isQCTest = True
config.correlation_threshold = 64
config.echo_intensity_threshold = 40

# Process all files in directory
input_dir = Path('raw_data/')
output_dir = Path('processed/')
output_dir.mkdir(exist_ok=True)

for file in input_dir.glob('*.000'):
    print(f"Processing {file.name}...")
    
    result = autoprocess(
        config,
        binary_file_path=file,
        save_netcdf=True,
        output_dir=output_dir,
        velocity_units='cm/s',
        print_summary=False
    )
    
    print(f"  Saved: {output_dir / file.stem}_processed.nc")
```

---

## Configuration File Format

The configuration file uses INI format:

```ini
[General]
input_file = /path/to/deployment.000

[TimeAxis]
snap = true
snap_freq = h
snap_tolerance = 5min

[SensorHealth]
roll_check = true
roll_threshold = 15.0
pitch_check = true
pitch_threshold = 15.0

[SignalQuality]
correlation = 64
echo_intensity = 40
error_velocity = 2000
percent_good = 50

[ProfileOperation]
trim_start = 100
trim_end = 50
cut_bins_side_lobe = true
regrid = true

[VelocityCheck]
cutoff_u = 2500
cutoff_v = 2500
cutoff_w = 500
despike = true
```

---

## Error Handling

```python
try:
    result = autoprocess('config.ini')
except FileNotFoundError:
    print("Config file or ADCP file not found")
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Processing error: {e}")
```

---

## See Also

- {doc}`core` — ProcessedDataset orchestrator
- {doc}`config` — Configuration options
- {doc}`multifile` — Multi-file processing
