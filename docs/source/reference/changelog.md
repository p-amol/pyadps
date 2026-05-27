# Changelog

All notable changes to pyadps will be documented in this file.

## [1.0.0] - 2024-XX-XX

### Added

- New `binary_reader` module with xarray.Dataset output
- xarray accessors for domain-specific operations
- CF Convention compliant variable attributes
- Comprehensive user documentation

### Changed

- Replaced `ReadFile` class with function-based API
- Migrated from custom data structures to xarray.Dataset
- Improved error handling with ErrorCode enum

### Deprecated

- `pyadps.ReadFile()` — Use `pyadps.read()` instead

### Migration Guide

See {doc}`/io/binary_reader` for migration instructions from v0.4.0.

## [0.4.0] - Previous Version

- Initial public release
- Basic RDI file reading with `ReadFile` class
- Streamlit web interface

## Version Numbering

pyadps follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)
