# autoprocess

Provides `autoprocess()` — a single-function entry point that runs the complete
processing pipeline from a config file or `ProcessingConfig` object. This is the
function the web app's Add-Ons → Auto Processing tool, and the `pyadps-auto` CLI,
call internally to reprocess a file from a saved `config.ini`.

## Function

```python
from pyadps.processing.autoprocess import autoprocess

result = autoprocess(config_file_or_object, **options)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `config_file_or_object` | — | Path to `config.ini` or a `ProcessingConfig` object |
| `binary_file_path` | `None` | ADCP binary file path; overrides the path stored in config |
| `save_netcdf` | `False` | Save output to NetCDF. If `False`, nothing is written to disk — only the in-memory `xarray.Dataset` is returned |
| `save_velocity_only` | `False` | Shorthand for `include_velocity=True` with every other `include_*` `False`. Kept for backward compatibility; prefer `include_*` below for anything more specific |
| `include_velocity`, `include_echo`, `include_correlation`, `include_percent_good`, `include_mask` | `None` | Which components to save — the same choice as the Write File page's "Select Data Components to Export". `None` falls back to the config's stored `[ExportOptions]` (what was last exported via that page), if present; otherwise the entire dataset is saved (the original behavior of this function) |
| `apply_mask` | `None` | If `True`, masked cells become `NaN` in Velocity only — never in Echo Intensity/Correlation/Percent Good (see `get_export_dataset`). `None` falls back to the config's stored choice, or `True` |
| `save_raw_netcdf` | `None` | If `True` (and `save_netcdf` is `True`), also save the raw (unprocessed) dataset as NetCDF alongside the processed output. `None` falls back to whether a raw NetCDF download actually happened for this config.ini (`[RawExport]`); otherwise `False` |
| `raw_include_fixed_leader`, `raw_include_variable_leader`, `raw_include_velocity`, `raw_include_echo`, `raw_include_correlation`, `raw_include_percent_good` | `None` | Which raw components to include when `save_raw_netcdf` is `True` — the same choice as the Download Raw File page's component checkboxes (checking all six there is its "Entire Data Set" option). `None` falls back to the config's stored `[RawExport]` selection, if present, or `True` (entire dataset) otherwise |
| `output_dir` | `None` | Output directory; defaults to same directory as input file |
| `output_filename` | `None` | Output filename; auto-generated if not specified (see filename patterns below) |
| `velocity_units` | `None` | Output units: `'mm/s'`, `'cm/s'`, or `'m/s'`. `None` falls back to the config's stored choice, or `'cm/s'` |
| `ensure_depth_ascending` | `True` | Reorder depth dimension to be ascending in output |
| `print_summary` | `True` | Print processing summary to console |

Returns the processed `xarray.Dataset`.

```{warning}
This function only ever produces **NetCDF** output — there is no CSV
option. CSV export is only available interactively on the Write File
page; `config.ini` doesn't record the output format, only which
components were selected.
```

## Usage

```python
from pyadps.processing.autoprocess import autoprocess

# Run from a saved config file (nothing written to disk without save_netcdf=True)
result = autoprocess('config.ini')

# Save full NetCDF output
result = autoprocess('config.ini', save_netcdf=True, output_dir='processed/')

# Save velocity-only output in SI units
result = autoprocess('config.ini', save_netcdf=True,
                     save_velocity_only=True, velocity_units='m/s')

# Save velocity + echo intensity, with the QC mask included as its own variable
result = autoprocess('config.ini', save_netcdf=True,
                     include_velocity=True, include_echo=True, include_mask=True)

# Also save the entire raw dataset alongside the processed output
result = autoprocess('config.ini', save_netcdf=True, save_raw_netcdf=True)

# Save only Velocity + Correlation from the raw dataset
result = autoprocess(
    'config.ini', save_netcdf=True, save_raw_netcdf=True,
    raw_include_velocity=True, raw_include_correlation=True,
)
```

## Programmatic Use

Pass a `ProcessingConfig` object instead of a file path when building scripts
that don't rely on a saved `config.ini`:

```python
from pyadps.processing.autoprocess import autoprocess
from pyadps.processing.config import ProcessingConfig

config = ProcessingConfig()
config.isQCTest = True
config.correlation_threshold = 64
config.echo_intensity_threshold = 40

result = autoprocess(
    config,
    binary_file_path='data/deployment.000',
    save_netcdf=True,
    output_dir='results/',
    print_summary=False,
)
```

## Command-Line Use

The `pyadps-auto` CLI wraps this function for non-interactive reprocessing
(`save_netcdf=True` is always implied):

```bash
pyadps-auto config.ini --include-echo --include-mask --output-dir processed/
pyadps-auto config.ini --velocity-only --velocity-units m/s
pyadps-auto config.ini --save-raw-netcdf
pyadps-auto config.ini --save-raw-netcdf --raw-include-velocity --raw-include-correlation
```

Run `pyadps-auto --help` for the full flag list — it mirrors the `include_*`,
`apply_mask`, `save_raw_netcdf`, `raw_include_*`, and `velocity_units`
parameters above. Note the `--raw-include-*` flags can only turn a
component *on*; to select a specific subset that excludes some component
from the config's stored default, edit `[RawExport]` in the config.ini
directly instead.

## Non-Obvious Behaviors

**`binary_file_path` overrides the config.** If you supply it, the file path
stored inside the config is ignored — useful for re-using one config across
multiple deployments.

**Component selection auto-applies from the config, if present.** A
`config.ini` saved after exporting Velocity + Echo Intensity on the Write
File page will, by default, reproduce that same combination here — no
`include_*` arguments needed. This only kicks in when the config.ini
actually has an `[ExportOptions]` section (i.e. it was saved after a real
export); older configs, or ones where nothing was ever exported, fall
through to the original full-dataset behavior. Any explicit `include_*`/
`apply_mask`/`velocity_units`/`save_raw_netcdf`/`raw_include_*` argument
always overrides whatever the config stores.

**Raw component selection works the same way, independently.** A
`config.ini` saved after downloading, say, just Velocity + Correlation as
raw NetCDF on the Download Raw File page will reproduce that exact subset
when `save_raw_netcdf` ends up `True` (`[RawExport]` section) — not the
entire raw dataset. Only an actual NetCDF download on that page sets this;
a CSV download, or no download at all, leaves it unset and `save_raw_netcdf`
defaults to `False` as before.

**Auto-generated filenames** depend on which components ended up included:

| What's included | Filename |
|---|---|
| Entire dataset (default) | `<input_stem>_processed.nc` |
| Velocity only | `<input_stem>_velocity.nc` |
| Any other combination | `<input_stem>_export.nc` |
| Raw dataset (`save_raw_netcdf=True`) | `<input_stem>_RAW_DATA.nc`, written alongside whichever of the above applies |

## See Also

- {doc}`config` — `ProcessingConfig` reference and INI file format
- {doc}`core` — `ProcessedDataset` for step-by-step control
- {doc}`multifile` — Combining multiple deployment files before processing
