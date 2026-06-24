# sensor_health

Provides `SensorHealthRunner` — the class that `ProcessedDataset.apply_sensor_health()`
delegates to. Use it directly when you need finer control over individual checks
or want to access per-step statistics.

## SensorHealthRunner

```python
import pyadps
from pyadps.processing.sensor_health import SensorHealthRunner

ds = pyadps.read('deployment.000')
runner = SensorHealthRunner(ds)
```

### Methods

| Method | Description |
|--------|-------------|
| `replace_data(data, variable_name)` | Replace a dataset variable with external data (e.g. CTD) |
| `correct_sound_speed(correct_velocity=True, horizontal_only=True)` | Recalculate sound speed and optionally correct velocity |
| `roll_check(threshold=15.0)` | Flag ensembles where roll exceeds threshold (degrees) |
| `pitch_check(threshold=15.0)` | Flag ensembles where pitch exceeds threshold (degrees) |
| `reset()` | Restore working dataset to original state |
| `finalize()` | Return processed `xarray.Dataset` |

All methods return `self` for chaining.

### Statistics Methods

| Method | Description |
|--------|-------------|
| `get_statistics()` | Dict of `QCCheckStats` keyed by check name |
| `get_modifications()` | Dict of `DataModificationStats` keyed by variable name |
| `print_statistics()` | Print formatted QC summary table |
| `get_report()` | Return summary as a string |
| `get_pipeline_report()` | Return `QCPipelineReport` for use by `ProcessedDataset` |

## Usage

```python
import numpy as np

runner = SensorHealthRunner(ds)

# Replace ADCP temperature with CTD data, then correct sound speed
ctd_temp = np.loadtxt('ctd_temperature.csv')  # °C
ctd_sal  = np.loadtxt('ctd_salinity.csv')     # PSU

result = (runner
    .replace_data(ctd_temp, 'temperature')
    .replace_data(ctd_sal, 'salinity')
    .correct_sound_speed()
    .roll_check(threshold=15.0)
    .pitch_check(threshold=15.0)
    .finalize())

runner.print_statistics()
```

```{note}
`replace_data()` automatically applies the variable's `scale_factor` to convert
physical units to RDI internal format (e.g. 15.0 °C → 1500 internally).
```

## See Also

- {doc}`core` — `ProcessedDataset.apply_sensor_health()` for the high-level interface
- {doc}`utility` — `QCCheckStats` and `DataModificationStats` reference
