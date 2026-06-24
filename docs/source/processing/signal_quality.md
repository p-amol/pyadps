# signal_quality

Provides `SignalQualityRunner` — the class that `ProcessedDataset.apply_signal_quality()`
delegates to. Use it directly for finer control or to inspect per-check statistics.

## SignalQualityRunner

```python
import pyadps
from pyadps.processing.signal_quality import SignalQualityRunner

ds = pyadps.read('deployment.000')
runner = SignalQualityRunner(ds)
```

### QC Check Methods

All methods return `self` for chaining. Pass `threebeam=True, beam_ignore=N`
to any beam-level check to ignore a known-bad beam.

| Method | Default cutoff | Description |
|--------|---------------|-------------|
| `correlation(cutoff=64)` | 64 counts | Flag cells below minimum correlation |
| `echo_intensity(cutoff=40)` | 40 counts | Flag cells below minimum echo intensity |
| `error_velocity(cutoff=2000)` | 2000 mm/s | Flag cells above maximum error velocity |
| `percent_good(cutoff=50)` | 50 % | Flag cells below minimum percent good |
| `false_target(cutoff=50)` | 50 counts | Flag cells with large cross-beam echo differences |

### Control and Output Methods

| Method | Description |
|--------|-------------|
| `reset()` | Restore working dataset to original state |
| `finalize()` | Return processed `xarray.Dataset` |
| `get_statistics()` | Dict of `QCCheckStats` keyed by check name |
| `print_statistics()` | Print formatted QC summary table |
| `get_pipeline_report()` | Return `QCPipelineReport` for `ProcessedDataset` |

## Usage

```python
result = (runner
    .correlation(cutoff=64)
    .echo_intensity(cutoff=40)
    .error_velocity(cutoff=2000)
    .percent_good(cutoff=50)
    .false_target(cutoff=50)
    .finalize())

runner.print_statistics()
```

### Three-Beam Mode

When one beam has a known hardware problem, enable three-beam mode to exclude
it from correlation and echo checks:

```python
result = (runner
    .correlation(cutoff=64, threebeam=True, beam_ignore=2)
    .echo_intensity(cutoff=40, threebeam=True, beam_ignore=2)
    .error_velocity(cutoff=2000)
    .finalize())
```

## Percent Good Threshold Advisor

`ProcessedDataset` exposes a helper that recommends a percent-good cutoff
for a target current precision, using ADCP noise curves:

```python
from pyadps.processing import ProcessedDataset

proc = ProcessedDataset(ds)
result = proc.get_percent_good_threshold(desired_std=1.0)  # target: 1 cm/s std dev
print(result)

# Apply the recommended cutoff
proc.apply_signal_quality(percent_good=result.percent_good_cutoff)
```

## See Also

- {doc}`core` — `ProcessedDataset.apply_signal_quality()` for the high-level interface
- {doc}`utility` — `QCCheckStats` reference
