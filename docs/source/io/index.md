# I/O Module

The `pyadps.io` module reads RDI ADCP binary files (PD0 format) and converts
them into [xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html)
objects ready for analysis and processing.

## Architecture

The module is organised in three layers:

| Layer | Module | Purpose |
|-------|--------|---------|
| 1 | `pd0_parser` | Low-level binary parsing — bytes to NumPy arrays |
| 2 | `binary_reader` | High-level reader — NumPy arrays to xarray.Dataset |
| 3 | `accessors` | Domain-specific methods attached to the Dataset |

Most users only need the `binary_reader` layer via `pyadps.read()`. The
`pd0_parser` layer is for advanced users who need direct binary access.

## Reading a File

```python
import pyadps

# Load the complete dataset
ds = pyadps.read('deployment.000')
print(ds)
```

The returned object is a standard `xarray.Dataset` containing velocity,
correlation, echo intensity, percent good, fixed leader, and variable leader
data. All xarray operations (indexing, plotting, NetCDF export) work directly.

### Loading Specific Components

Use `data_types` to load only the variables you need — useful for large files:

```python
ds = pyadps.read('deployment.000', data_types=['Velocity', 'Correlation'])
```

Available options: `'FixedLeader'`, `'VariableLeader'`, `'Velocity'`,
`'Correlation'`, `'Echo'`, `'PercentGood'`, `'Status'`.

### Key Defaults

| Parameter | Default | Effect |
|-----------|---------|--------|
| `missing_as_nan` | `True` | RDI missing value (−32768) replaced with `NaN` |
| `include_mask` | `True` | A 3D QC mask variable is created from missing values |
| `use_time_as_primary_dim` | `True` | Time is the first dimension |
| `use_depth_as_primary_dim` | `False` | Cell index used instead of physical depth |

## Available Functions

| Function | Description |
|----------|-------------|
| `pyadps.read()` | Load complete dataset — primary entry point |
| `pyadps.read_header()` | Read file header and structure metadata |
| `pyadps.read_fixed_leader()` | Read static instrument configuration |
| `pyadps.read_variable_leader()` | Read per-ensemble sensor data |
| `pyadps.read_velocity()` | Read velocity array (beam × cell × time) in mm/s |
| `pyadps.read_correlation()` | Read correlation array (0–255) |
| `pyadps.read_echo_intensity()` | Read echo intensity array (0–255, ×0.45 for dB) |
| `pyadps.read_percent_good()` | Read percent good array (0–100) |
| `pyadps.read_status()` | Read status diagnostic bit flags |

## Accessor Methods

Accessor methods are automatically registered when you `import pyadps` — no
additional import is needed. They are available on any Dataset returned by the
read functions.

### Header (`ds.header.*`)

| Method | Description |
|--------|-------------|
| `check_file()` | Verify file integrity (size, byte uniformity) |
| `get_available_data_types()` | List data types present in the file |
| `get_ensemble_info(ensemble)` | Metadata for a specific ensemble |
| `validate()` | Full validation report |
| `summary()` | Print formatted header summary |

### Fixed Leader (`ds.fixed_leader.*`)

| Method | Description |
|--------|-------------|
| `system_configuration()` | Frequency, beam pattern, orientation |
| `coordinate_transformation()` | Coordinate system and transformation settings |
| `sensor_info()` | Available and active sensors |
| `is_uniform()` | Check if configuration is consistent across ensembles |
| `validate()` | Full validation report |
| `summary()` | Print formatted configuration summary |

### Variable Leader (`ds.variable_leader.*`)

| Method | Description |
|--------|-------------|
| `validate_timestamps()` | Check time coordinate validity |
| `get_time_interval()` | Most common ensemble interval |
| `is_time_regular()` | Check if time axis is uniform |
| `get_time_component_frequency()` | Frequency distribution of time components |
| `ensemble_continuity_check()` | Detect gaps in ensemble numbering |
| `bit_result_summary()` | Decode Built-In Test result bits |
| `error_status_word_summary()` | Decode all four Error Status Words |
| `summary()` | Print formatted variable leader summary |

## Low-Level Access

For direct binary access without the xarray layer:

```python
from pyadps.io import pd0_parser

# Parse file structure
dt, byte, byteskip, offset, idarray, n_ens, err = pd0_parser.fileheader('deployment.000')

# Read velocity as NumPy array
data, n_ens, cell_array, beam_array, err = pd0_parser.datatype('deployment.000', 'velocity')
```

See {doc}`pd0_parser` for full details.

## Contents

```{toctree}
:maxdepth: 1

pd0_parser
binary_reader
accessors
```

## See Also

- {doc}`/processing/index` — Quality control and processing pipeline
