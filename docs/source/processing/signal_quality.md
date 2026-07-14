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

All methods return `self` for chaining. Pass `beam_ignore=N` to `correlation`,
`echo_intensity`, or `false_target` to exclude a known-bad beam. Only
`percent_good` accepts `threebeam` (see below); it has no `beam_ignore`.

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

### Beam to Ignore vs. Three-Beam Mode

These are two distinct mechanisms, not variants of one setting:

- **`beam_ignore`**: the user names a specific beam known to be bad (from
  correlation, echo intensity, or percent-good diagnostics), and the check
  excludes it unconditionally. Supported by `correlation`, `echo_intensity`,
  and `false_target`.
- **`threebeam`**: changes which percent-good components are summed
  (`percent_good` only) — it does not identify *which* beam is bad, it reads
  a fact the instrument already recorded (PG1/PG4 3-beam/4-beam solution
  counts).

When one beam has a known hardware problem, use `beam_ignore` to exclude it:

```python
result = (runner
    .correlation(cutoff=64, beam_ignore=2)
    .echo_intensity(cutoff=40, beam_ignore=2)
    .false_target(cutoff=50, beam_ignore=2)
    .error_velocity(cutoff=2000)
    .finalize())
```

`correlation_check`, `echo_intensity_check`, and `false_target_detection`
intentionally have no `threebeam` parameter — auto-detecting a bad beam
without being told which one isn't reliably possible on post-collection data;
see "Future Work" below for what it would take. `percent_good` is the only
check where `threebeam` applies, because PG1/PG4 are literal counts the
instrument already recorded, not an inference.

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

## Future Work

**Automatic three-beam detection (not implemented).** The RDI instrument's own
real-time three-beam solution *detects* an anomalous beam per ping and
reconstructs its velocity from the other three, assuming no vertical shear,
before writing the ensemble. A post-collection equivalent — auto-detecting
the bad beam rather than requiring `beam_ignore` to name it — depends on which
quantity is involved:

- **Velocity** (`correlation`, indirectly): once transformed to Earth
  coordinates, each component (East/North/Up/Error) is already a fixed linear
  combination of all 4 beams — a single beam's contribution can no longer be
  isolated or dropped after the fact. This stays solvable only in raw,
  single-ping **beam-coordinate** data, which pyadps does not currently read
  or process.
- **Echo intensity** (`echo_intensity`, `false_target`): this data stays
  beam-indexed regardless of coordinate transform, so the coordinate problem
  above doesn't apply. The blocker here is **single-ping resolution**:
  detecting a genuine per-ping anomaly (e.g. a fish transiting one beam's
  path) requires looking at individual pings, since ensemble-averaging blends
  any such transient into the mean before this function ever sees it.

`percent_good`'s PG1/PG4 counts remain the correct source for "how often did
a 3-beam solution happen" on ensemble-averaged data, since they are the
instrument's own recorded tally rather than a reconstruction attempted after
the fact.

If beam-coordinate, single-ping support is added to pyadps in the future,
genuine auto-detection would belong in a new function scoped to that input —
not in `correlation_check`, `echo_intensity_check`, or
`false_target_detection`, which operate on post-collection data.

## See Also

- {doc}`core` — `ProcessedDataset.apply_signal_quality()` for the high-level interface
- {doc}`utility` — `QCCheckStats` reference
