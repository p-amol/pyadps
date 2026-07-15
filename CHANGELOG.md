# Changelog

All notable changes to `pyadps` are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and version numbers follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- New `binary_reader` module with `xarray.Dataset` output
- xarray accessors for domain-specific operations (`ds.fixed_leader`, `ds.variable_leader`, ...)
- CF Convention compliant variable attributes
- Comprehensive user documentation (Read the Docs)
- Selectable export components (velocity, echo intensity, correlation, percent good) on
  the Write File page, replacing the earlier velocity-only/full-dataset choice
- Short (u, v, w) velocity variable naming option alongside the CF-style long names

### Changed

- Replaced the `ReadFile` class with a function-based API (`pyadps.read()`)
- Migrated from custom data structures to `xarray.Dataset` throughout
- Improved error handling with an `ErrorCode` enum
- Three-beam mode in the Signal Quality QC pipeline is now scoped to the Percent Good
  check only; it previously also appeared on Correlation, Echo Intensity, and False
  Target, where it was either non-functional or not reliably grounded on
  ensemble-averaged data

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
