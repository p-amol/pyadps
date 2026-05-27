# pyadps Documentation

**Professional Python package for RDI ADCP data processing**

pyadps provides tools for reading, quality controlling, and processing 
Acoustic Doppler Current Profiler (ADCP) data from RDI instruments.

## Features

- Read RDI binary files (PD0 format) with xarray integration
- Comprehensive quality control tests
- Sensor health diagnostics
- Velocity validation
- Interactive web interface (Streamlit)
- Batch processing support
- CF Convention compliant output

## Quick Example

```python
import pyadps
import pyadps.accessors  # Register domain-specific methods

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

## Package Structure

pyadps is organized into three main components:

### I/O Module

Read and parse RDI ADCP binary files:

```{toctree}
:maxdepth: 2

io/index
```

### Processing Module

Quality control and data processing tools:

```{toctree}
:maxdepth: 2

processing/index
```

### Web Application

Interactive Streamlit interface:

```{toctree}
:maxdepth: 1

webapp/index
```

## Additional Resources

```{toctree}
:maxdepth: 1

installation
quickstart
tutorials/index
api/index
reference/changelog
```

## Getting Started

1. **Install pyadps**: See {doc}`installation` for instructions
2. **Read the Quick Start**: {doc}`quickstart` provides a 5-minute introduction
3. **Explore the I/O module**: {doc}`io/index` covers data loading
4. **Learn quality control**: {doc}`processing/index` covers QC workflows

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
