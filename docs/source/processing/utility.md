# utility

Shared dataclasses returned by all Runner classes when you call `get_statistics()`
or `get_modifications()`. Import them only if you need to type-hint or inspect
these objects programmatically.

```python
from pyadps.processing.utility import QCCheckStats, DataModificationStats, QCPipelineReport
```

## QCCheckStats

Returned in the dict from `runner.get_statistics()`, keyed by check name.

| Attribute | Type | Description |
|-----------|------|-------------|
| `check_name` | str | Name of the check (e.g. `'Correlation Check'`) |
| `threshold` | float / list / tuple | Threshold value(s) used |
| `cells_pre_masked` | int | Cells already masked before this check |
| `cells_newly_masked` | int | Cells flagged by this check |
| `cells_total_masked` | int | Cumulative masked cells after this check |
| `total_cells` | int | Total cells in the dataset |
| `pre_masked_pct` | float | Percent masked before this check |
| `newly_masked_pct` | float | Percent flagged by this check (impact) |
| `total_masked_pct` | float | Cumulative percent masked |
| `valid_cells` | int | Cells still valid after this check |
| `valid_pct` | float | Percent still valid |
| `to_dict()` | dict | JSON-serializable representation |

```python
stats = runner.get_statistics()
for name, s in stats.items():
    print(f"{name}: {s.newly_masked_pct:.1f}% flagged, {s.valid_pct:.1f}% valid")
```

## DataModificationStats

Returned in the dict from `runner.get_modifications()`, keyed by variable name.

| Attribute | Type | Description |
|-----------|------|-------------|
| `operation` | str | Operation name (e.g. `'replace_data'`, `'correct_sound_speed'`) |
| `variable_name` | str | Dataset variable that was modified |
| `original_stats` | dict | `min`, `max`, `mean`, `std` of data before modification |
| `modified_stats` | dict | `min`, `max`, `mean`, `std` of data after modification |
| `mean_change` | float \| None | Absolute change in mean |
| `mean_change_pct` | float \| None | Percentage change in mean |
| `to_dict()` | dict | JSON-serializable representation |

## QCPipelineReport

Returned by `runner.get_pipeline_report()`. Aggregates all checks and
modifications for one processing stage. Used internally by `ProcessedDataset`
to build the full-pipeline summary.

| Attribute | Type | Description |
|-----------|------|-------------|
| `module_name` | str | Processing stage name |
| `baseline_masked` | int | Cells masked before any checks in this stage |
| `total_cells` | int | Total cells in the dataset |
| `checks` | list[QCCheckStats] | Ordered list of per-check statistics |
| `modifications` | list[DataModificationStats] | Data modification records |
| `final_valid_pct` | float | Percent valid after all checks in this stage |
| `pipeline_impact_pct` | float | Additional masking caused by this stage |
| `to_dict()` | dict | JSON-serializable representation |

## See Also

- {doc}`sensor_health` — `get_statistics()` and `get_modifications()`
- {doc}`signal_quality` — `get_statistics()`
- {doc}`profile_operation` — `get_statistics()`
- {doc}`velocity_check` — `get_statistics()` and `get_modifications()`
- {doc}`core` — `ProcessedDataset.print_summary()`
