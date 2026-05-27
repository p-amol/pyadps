# config Module

Configuration management for ADCP processing.

## Overview

The `config` module provides the `ProcessingConfig` class for managing processing 
parameters. It supports loading from INI files (for Streamlit UI integration) 
and programmatic configuration.

## Quick Start

```python
from pyadps.processing.config import ProcessingConfig

# Load from INI file
config = ProcessingConfig.from_file('processing.ini')

# Create programmatically
config = ProcessingConfig()
config.correlation_threshold = 64
config.echo_intensity_threshold = 40
config.roll_threshold = 15.0
```

---

## ProcessingConfig Class

### Constructor

```python
config = ProcessingConfig()
```

### Loading from File

```python
config = ProcessingConfig.from_file('config.ini')
```

### Saving to File

```python
config.to_file('config.ini')
```

---

## Configuration Parameters

### Signal Quality

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `correlation_threshold` | int | 64 | Minimum correlation |
| `echo_intensity_threshold` | int | 40 | Minimum echo intensity |
| `error_velocity_threshold` | int | 2000 | Maximum error velocity (mm/s) |
| `percent_good_threshold` | int | 50 | Minimum percent good |
| `false_target_threshold` | int | 50 | Maximum echo difference |

### Sensor Health

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `roll_threshold` | float | 15.0 | Maximum roll (degrees) |
| `pitch_threshold` | float | 15.0 | Maximum pitch (degrees) |
| `correct_sound_speed` | bool | False | Enable sound speed correction |

### Velocity Check

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `velocity_threshold_u` | float | 2500 | U threshold (mm/s) |
| `velocity_threshold_v` | float | 2500 | V threshold (mm/s) |
| `velocity_threshold_w` | float | 500 | W threshold (mm/s) |
| `despike_enabled` | bool | True | Enable despike filter |
| `despike_cutoff` | float | 3.0 | Despike threshold (σ) |

### Profile Operations

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trim_start` | int | 0 | Ensembles to trim from start |
| `trim_end` | int | 0 | Ensembles to trim from end |
| `cut_side_lobe` | bool | False | Enable side lobe removal |
| `regrid` | bool | False | Enable regridding |

---

## INI File Format

```ini
[General]
input_file = /path/to/data.000

[SignalQuality]
correlation = 64
echo_intensity = 40
error_velocity = 2000
percent_good = 50

[SensorHealth]
roll_check = true
roll_threshold = 15.0
pitch_check = true
pitch_threshold = 15.0

[VelocityCheck]
cutoff_u = 2500
cutoff_v = 2500
cutoff_w = 500
despike = true
despike_cutoff = 3.0

[ProfileOperation]
trim_start = 100
trim_end = 50
cut_bins_side_lobe = true
regrid = true
```

---

## See Also

- {doc}`autoprocess` — Automated processing using config
- {doc}`core` — ProcessedDataset orchestrator
