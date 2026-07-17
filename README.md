# pyadps

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/pyadps.svg)](https://pypi.org/project/pyadps/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://pyadps.readthedocs.io)

`pyadps` is a Python package for processing moored Acoustic Doppler Current Profiler (ADCP) data. It provides data reading, quality control, NetCDF export, and an interactive web interface — designed for Teledyne RDI ADCPs recording in the PD0 binary format. PD0 files from other RDI models, such as Ocean Surveyor and DVS, can also be read; take extra care when processing that data, since the pipeline's defaults were tuned against Workhorse deployments.

- **Documentation:** <https://pyadps.readthedocs.io>
- **Source code:** <https://github.com/p-amol/pyadps>
- **Bug reports:** <https://github.com/p-amol/pyadps/issues>
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)

## Background

`pyadps` was built to process Teledyne RDI PD0 files from moored ADCP deployments that lack navigation (GPS) data. It was developed primarily for the COSINE and ECO-IOD mooring programs in the north Indian Ocean, which together span over 600 ADCP deployments — a scale that made it worth standardizing and documenting the processing steps rather than repeating ad hoc scripts for each dataset. The accompanying web interface lets users with limited programming experience run the same pipeline, while the processing workflow still requires a close look at the quality-control results before the velocity output is treated as final. The package currently processes only data recorded in Earth coordinates; Beam-coordinate support and the associated coordinate transformation are planned for a future release.

## Features

- Read RDI binary files (PD0 format) as `xarray.Dataset`
- Robust reading of corrupted or truncated binary files, with per-ensemble checksum verification and partial-data recovery
- Six-step quality control pipeline (time axis, sensor health, signal quality, profile operations, velocity checks)
- Interactive web interface (Streamlit) — no Python knowledge required
- Batch processing and multi-file combining
- CF Convention compliant NetCDF output
- Reproducible processing via `config.ini` export
- Extensively tested: 4,200+ automated tests (`pytest`), 98% coverage, spanning the I/O layer, processing pipeline, and every Streamlit page

## Installation

Requires **Python 3.12**. Install in a dedicated virtual environment.

### Using `venv`

```bash
python3.12 -m venv pyadps-env
source pyadps-env/bin/activate   # Windows: pyadps-env\Scripts\activate
pip install pyadps
```

### Using `conda`

```bash
conda create -n pyadps-env python=3.12
conda activate pyadps-env
conda install pip
pip install pyadps
```

### From Source

```bash
git clone https://github.com/p-amol/pyadps.git
cd pyadps
pip install -e .
```

## Quick Start

### Web Interface

The easiest way to get started — no Python required beyond installation:

```bash
pyadps-gui
```

This launches a Streamlit app that guides you through each processing step.

### Python API

```python
import pyadps

# Load an RDI binary file as an xarray.Dataset
ds = pyadps.read('deployment.000')

# Inspect system configuration
config = ds.fixed_leader.system_configuration()
print(f"Frequency: {config['Frequency']}")

# Save raw data to NetCDF
ds.to_netcdf('raw_output.nc')
```

### Processing Pipeline

```python
from pyadps.processing import ProcessedDataset

proc = ProcessedDataset(ds)
result = (
    proc
    .apply_time_axis(snap=True, snap_freq='h')
    .apply_sensor_health(roll=True, roll_threshold=15.0)
    .apply_signal_quality(correlation=64, echo_intensity=40,
                          error_velocity=2000, percent_good=25)
    .apply_profile_operation(cut_bins_side_lobe=True, water_depth=50.0,
                             trim_start=10, trim_end=10)
    .apply_velocity_check(cutoff_u=2500, cutoff_v=2500, cutoff_w=500,
                          magnetic_correction=True, declination=-1.5)
    .finalize()
)

# Save full dataset or velocity-only output
result.to_netcdf('processed.nc')
proc.velocity_to_netcdf('velocity.nc', units='cm/s')

# Export settings for reproducibility
proc.export_config('config.ini')
```

### Auto Processing

Re-run a processing workflow from a saved `config.ini` file — useful for
batch reprocessing with adjusted thresholds.

```python
from pyadps.processing.autoprocess import autoprocess

result = autoprocess(
    config_file_or_object='config.ini',
    binary_file_path='deployment.000',
    save_netcdf=True,
)
```

The same workflow is available from the command line via the `pyadps-auto`
script, installed alongside `pyadps`:

```bash
pyadps-auto config.ini --binary deployment.000
```

By default this writes `deployment_processed.nc` next to the binary file. Add
`--velocity-only` to export just the velocity components, or `-o` to choose an
output directory. Run `pyadps-auto --help` for the full list of options.

### Binary File Combiner

Combine multiple sequential ADCP binary files — e.g. a deployment split across
files due to instrument memory limits — into a single file.

```python
from pathlib import Path
from pyadps.processing.multifile import combine_file_list

files = [Path('deploy_000.000'), Path('deploy_001.000'), Path('deploy_002.000')]

result = combine_file_list(files, output_file=Path('merged.000'))
print(f"Combined {result.total_ensembles} ensembles from {result.files_processed} files")
```

Or from the command line via the `pyadps-cat` script, pointed at a folder of
files instead of an explicit list:

```bash
pyadps-cat raw_data/ -o combined.000
```

By default this matches `*.000` files, skips any invalid ones, and requires
matching ensemble sizes across files. Run `pyadps-cat --help` for the full list
of options.

For the complete guide see the [documentation](https://pyadps.readthedocs.io).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Note

*This version of the package was developed with extensive use of Claude (Anthropic) as a coding assistant.*
