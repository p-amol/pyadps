# pyadps Documentation

**Professional Python package for RDI ADCP data processing**

pyadps provides tools for reading, quality controlling, and processing
Acoustic Doppler Current Profiler (ADCP) data — designed for Teledyne RDI
ADCPs recording in the PD0 binary format. PD0 files from other RDI models,
such as Ocean Surveyor and DVS, can also be read; take extra care when
processing that data, since the pipeline's defaults were tuned against
Workhorse deployments.

## Background

`pyadps` was built to process PD0 files from moored ADCP deployments that
lack navigation (GPS) data. It was developed primarily for the COSINE and
ECO-IOD mooring programs in the north Indian Ocean, which together span
over 600 ADCP deployments — a scale that made it worth standardizing and
documenting the processing steps rather than repeating ad hoc scripts for
each dataset. The processing pipeline is designed to encourage a close
look at the quality-control results before the velocity output is treated
as final. The package currently processes only data recorded in Earth
coordinates; Beam-coordinate support and the associated coordinate
transformation are planned for a future release.

This version of the package was developed with extensive use of Claude
(Anthropic) as a coding assistant.

## Features

- Read RDI binary files (PD0 format) with xarray integration
- Robust reading of corrupted or truncated binary files, with per-ensemble
  checksum verification and partial-data recovery
- Six-step quality control pipeline: time axis correction, sensor health,
  signal quality, profile operations, and velocity checks
- Interactive web interface (Streamlit) — no Python knowledge required
- Batch processing and multi-file combining
- CF Convention compliant output
- Reproducible processing via `config.ini` export
- Extensively tested: 4,200+ automated tests (`pytest`), 98% coverage

## Quick Example

```python
import pyadps

# Read an RDI file
ds = pyadps.read("deployment.000")

# Check system configuration
config = ds.fixed_leader.system_configuration()
print(f"Frequency: {config['Frequency']}")

# Access velocity data
velocity = ds['velocity']
print(f"Shape: {velocity.shape}")

# Export to NetCDF
ds.to_netcdf('output.nc')
```

## Processing Pipeline

```{graphviz} ../../src/pyadps/pipeline_flowchart.dot
```

## Package Structure

pyadps is organized into three main components:

```{toctree}
:maxdepth: 1

installation
quickstart
webapp/index
tutorials/index
```

### Module Reference

```{toctree}
:maxdepth: 2

io/index
processing/index
```

## Additional Resources

```{toctree}
:maxdepth: 1

reference/changelog
```

## Getting Started

1. **Install pyadps**: See {doc}`installation` for instructions
2. **Read the Quick Start**: {doc}`quickstart` provides a 5-minute introduction
3. **Try the Web App**: {doc}`webapp/index` covers the interactive interface
4. **Go deeper**: {doc}`io/index` and {doc}`processing/index` cover the module details

## Module Quick Reference

### I/O Functions

| Function | Description |
|----------|-------------|
| `pyadps.read()` | Load complete ADCP dataset |
| `pyadps.read_header()` | Read file header metadata |
| `pyadps.read_fixed_leader()` | Read configuration data |
| `pyadps.read_variable_leader()` | Read sensor data |
| `pyadps.read_velocity()` | Read velocity arrays |

### Processing Classes

| Class | Description |
|-------|-------------|
| `SensorHealthRunner` | Check sensor diagnostics |
| `SignalQualityRunner` | Assess signal quality |
| `VelocityCheckRunner` | Validate velocities |
| `ProfileOperationRunner` | Profile operations |
| `ProcessedDataset` | Processed data container |

### Accessor Methods

| Accessor | Namespace | Key Methods |
|----------|-----------|-------------|
| Header | `ds.header.*` | `check_file()`, `summary()` |
| Fixed Leader | `ds.fixed_leader.*` | `system_configuration()`, `validate()` |
| Variable Leader | `ds.variable_leader.*` | `bit_result_summary()`, `validate_timestamps()` |

## Support

- **Documentation**: You're reading it!
- **Issue Tracker**: [GitHub Issues](https://github.com/p-amol/pyadps/issues)
- **Source Code**: [GitHub Repository](https://github.com/p-amol/pyadps)

## License

pyadps is released under the MIT License.
