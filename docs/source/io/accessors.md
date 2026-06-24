# accessors

Domain-specific methods attached to `xarray.Dataset` objects via the xarray
accessor pattern. Accessors are automatically registered when you
`import pyadps` — no additional import is needed.

Three accessor namespaces are available on any Dataset returned by the
read functions:

| Namespace | Description |
|-----------|-------------|
| `ds.header.*` | File structure and integrity checks |
| `ds.fixed_leader.*` | System configuration and validation |
| `ds.variable_leader.*` | Sensor data analysis and diagnostics |

## Quick Example

```python
import pyadps

ds = pyadps.read('deployment.000')

# File integrity
ds.header.check_file()

# System configuration
config = ds.fixed_leader.system_configuration()
print(config['Frequency'])

# Timestamp validation
ds.variable_leader.validate_timestamps()
```

---

## ds.header

| Method | Description |
|--------|-------------|
| `check_file()` | Verify file size, byte uniformity, and data type consistency |
| `get_available_data_types(ens=0)` | List data types present in the file |
| `has_data_type(name)` | Check whether a specific data type exists |
| `get_ensemble_info(ensemble)` | Byte size and data types for a specific ensemble |
| `validate()` | Full validation report |
| `summary()` | Print formatted header summary |

---

## ds.fixed_leader

| Method | Description |
|--------|-------------|
| `system_configuration(ens=-1)` | Frequency, beam pattern, orientation, beam angle |
| `coordinate_transformation(ens=0)` | Coordinate system, tilt correction, bin mapping |
| `sensor_info(ens=0, field='source')` | Sensor availability and source selection |
| `is_uniform()` | Check each field is consistent across all ensembles |
| `validate()` | Full validation report |
| `summary()` | Print formatted configuration summary |

---

## ds.variable_leader

| Method | Description |
|--------|-------------|
| `validate_timestamps()` | Check timestamp plausibility and return time range |
| `get_time_interval()` | Modal time interval between ensembles |
| `is_time_regular(tolerance_s=1.0)` | Check whether the time axis is uniform |
| `get_time_interval_frequency()` | Distribution of all observed intervals |
| `get_time_component_frequency(component)` | Frequency count for a time component (e.g. `'minute'`) |
| `ensemble_continuity_check()` | Detect gaps in ensemble numbering |
| `ensemble_rollover_count()` | Count 16-bit ensemble counter wraparounds |
| `bit_result_summary()` | Decode Built-In Test result bits across all ensembles |
| `error_status_word_summary()` | Decode all four Error Status Words |
| `summary()` | Print formatted variable leader summary |

---

## See Also

- {doc}`binary_reader` — Functions that return accessor-enabled Datasets
- {doc}`pd0_parser` — Low-level access (no accessors available)
