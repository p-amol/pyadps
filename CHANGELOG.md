# Changelog

All notable changes to `pyadps` are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and version numbers follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `autoprocess()` and the Add-Ons "Auto Processing" page can now reproduce the exact
  component selection last used on the Write File page (Velocity/Echo Intensity/
  Correlation/Percent Good/QC Mask, mask application, velocity units) via a new
  `[ExportOptions]` section in `config.ini`, instead of only supporting "entire
  dataset" or "velocity only"
- `autoprocess()`/Add-Ons can also regenerate the raw (unprocessed) NetCDF alongside
  the processed output (`save_raw_netcdf`), reproducing the Download Raw File page's
  component picker (fixed leader, variable leader, velocity, echo intensity,
  correlation, percent good) via a new `[RawExport]` section in `config.ini`
- QC Mask is now a selectable export component in its own right (`include_mask` /
  the "QC Mask" checkbox on the Write File page), so masked cells can be inspected
  without forcing the mask onto Velocity
- `config.ini`'s `[FileSettings]` records the `pyadps_version` that processed the
  file, refreshed on every read and reprocess
- New CLI flags on `pyadps-auto` mirroring all of the above
  (`--include-echo`/`--include-correlation`/`--include-percent-good`/
  `--include-mask`, `--save-raw-netcdf`, `--raw-include-*`)

### Fixed

- Exporting multiple components together (e.g. Velocity + Echo Intensity) still
  triggered Ferret's "Unspecified or unsupported ordering of axes" warning after the
  1.0.2 fix, and Signal Quality processing could silently wipe coordinate attributes
  due to an xarray in-place mutation. NetCDF exports now carry correct CF axis
  metadata and raw-file provenance (`filename`, `adcp_data_format`) regardless of
  which components are included
- The QC mask was being applied to Echo Intensity, Correlation, and Percent Good on
  export, silently replacing valid per-beam diagnostic readings with `NaN` whenever
  Velocity failed QC for an unrelated reason. The mask is now applied to Velocity
  only; it can still be exported as its own variable (see "Added")
- **Several Signal Quality checks did not actually mask the data they claimed to
  flag.** `error_velocity_check`, `percent_good_check`, and `false_target_detection`
  wrote their flags only to the mask's beam-index-3 "combined" summary slot, but the
  NetCDF/CSV export path only ever reads mask beam-indices 0-2 to mask Velocity's
  U/V/W components — so these three checks correctly reported flagged cells in the
  QC statistics, but had **no effect on the actual exported/masked velocity data**.
  Separately, `correlation_check` masked only the single beam-position that failed,
  rather than the whole depth cell. All four now collapse a failing check to
  whole-cell masking across all four beam positions, matching
  `echo_intensity_check`'s existing behavior and the underlying physics: once
  beam data is combined into Earth-coordinate U/V/W, a single bad beam invalidates
  the whole cell's solution, not just one beam-position of it
- Raw NetCDF downloads from the Download Raw File page recorded `isRawExportOptions`
  as permanently `True` once set, even after a later download selected a different,
  partial component subset — `config.ini` now always reflects the most recent
  download
- The Read File page footer showed a hardcoded, stale `"pyadps v1.0.0"` string
  instead of the installed version
- `pyadps_version` in `config.ini` stayed frozen at the original read's version after
  reprocessing with a different pyadps install

## [1.0.2] - 2026-08-10

### Added

- Citation section (README and docs front page) referencing the package's Zenodo DOI
- Documented `pyadps.load_example()` in the README and Quick Start guide, so users can
  try the package on a bundled demo dataset without a binary ADCP file

### Fixed

- NetCDF exports that included a beam-dimensioned variable (Echo Intensity,
  Correlation, Percent Good) alongside Velocity carried incorrect CF axis metadata:
  the `beam` coordinate was tagged `axis: "X"` (a real spatial axis), and `cell`
  duplicated `depth`'s `axis: "Z"` role with the opposite `positive` direction. Ferret
  would warn "Unspecified or unsupported ordering of axes" and show inconsistent axis
  directions across grids in the same file. `beam` now uses Ferret's `axis: "E"`
  (ensemble) code, the correct label for a categorical, non-spatial dimension; `cell`
  no longer claims the Z axis that `depth` already owns. Reading these files now
  requires Ferret v6.8 or later (2013+), noted on the Write File and Download Raw
  File documentation pages.

## [1.0.1] - 2026-07-17

### Fixed

- **Critical:** the Streamlit app would crash (server segfault, not a normal
  Python exception) on the first file upload for anyone installing via `pip
  install pyadps`. `pyarrow` was never a direct dependency — it came in
  transitively through `streamlit`'s unbounded `pyarrow >= 7.0` constraint,
  so a fresh install resolved the newest release (25.0.0), which is
  incompatible with the pinned `numpy~=1.26.4` inside `pyarrow`'s
  pandas-to-Arrow conversion (used internally by `st.dataframe()`). Fixed by
  declaring `pyarrow = "~=17.0"` explicitly. Existing `dev`-branch checkouts
  with an already-installed compatible `pyarrow` were never affected, which
  is why this didn't surface in local testing.

## [1.0.0] - 2026-07-17

### Added

- New `binary_reader` module with `xarray.Dataset` output
- xarray accessors for domain-specific operations (`ds.fixed_leader`, `ds.variable_leader`, ...)
- CF Convention compliant variable attributes
- Comprehensive user documentation (Read the Docs)
- Selectable export components (velocity, echo intensity, correlation, percent good) on
  the Write File page, replacing the earlier velocity-only/full-dataset choice
- Short (u, v, w) velocity variable naming option alongside the CF-style long names
- Processing pipeline flowchart on the Streamlit home page and in the docs, showing
  the six-step pipeline plus the decision logic within Time Diagnostics, Sensor
  Health, Signal Quality, and Profile Operations
- "Background" section (motivation, deployment context, current coordinate-system
  limitation) on the README, Streamlit home page, and docs front page
- "Processing Guidelines" section on the Web Application docs page, with
  per-page tips on what to check at each step

### Changed

- Replaced the `ReadFile` class with a function-based API (`pyadps.read()`)
- Migrated from custom data structures to `xarray.Dataset` throughout
- Improved error handling with an `ErrorCode` enum
- Three-beam mode in the Signal Quality QC pipeline is now scoped to the Percent Good
  check only; it previously also appeared on Correlation, Echo Intensity, and False
  Target, where it was either non-functional or not reliably grounded on
  ensemble-averaged data
- Renamed the installed console scripts for a consistent, collision-resistant
  naming scheme: `run-pyadps` → `pyadps-gui`, `run-auto` → `pyadps-auto`,
  `run-cat` → `pyadps-cat` (new; combines multiple ADCP binary files)

### Fixed

- Documentation examples calling `ds.header.*` right after a plain `pyadps.read()`
  call, which silently misreports file status since `include_header` defaults to
  `False` (`installation.md`, `io/accessors.md`, `io/index.md`)
- Sample `config.ini` in `processing/config.md` used incorrect key names in the
  `[QCTest]`, `[ProfileTest]`, and `[VelocityTest]` sections that don't match
  `ProcessingConfig.from_ini()` — a hand-edited file following the old example
  would have had those settings silently ignored
- `ProfileOperationRunner.regrid()` documentation listed nonexistent parameters
  and the wrong default interpolation method
- `pd0_parser` error code 5 documented under the wrong name
  (`WRONG_RDIFILE_TYPE` instead of `WRONG_ADCPFILE_TYPE`)
- Write File page documentation describing the pre-redesign Export Data tab
- `pyadps-auto` console script pointed at a module that no longer existed and
  crashed immediately on use; it now has a working CLI
- Time Diagnostics' "Fill Time Gaps" (forward-fill) could silently leave gaps
  as `NaN` instead of filling them when the optional `bottleneck` package
  wasn't installed; it's now a declared dependency

### Deprecated

- `pyadps.ReadFile()` — use `pyadps.read()` instead

### Migration Guide

See the [binary_reader documentation](https://pyadps.readthedocs.io/en/latest/io/binary_reader.html)
for migration instructions from v0.3.3.

## [0.3.3]

- Initial public release
- Basic RDI file reading with the `ReadFile` class
- Streamlit web interface

## Version Numbering

pyadps follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)
