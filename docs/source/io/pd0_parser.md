# pd0_parser

Low-level binary parser for RDI ADCP files in PD0 format. Returns raw data
as NumPy arrays.

```{note}
Most users should use {doc}`binary_reader` instead, which wraps this module
and returns convenient `xarray.Dataset` objects. Use `pd0_parser` only if
you need direct binary access or are building a custom reader.
```

## Functions

| Function | Description |
|----------|-------------|
| `fileheader()` | Parse file structure — ensemble locations and data type offsets |
| `fixedleader()` | Read static instrument configuration (36 fields × n ensembles) |
| `variableleader()` | Read per-ensemble sensor data (48 fields × n ensembles) |
| `datatype()` | Read 3D profile data — velocity, correlation, echo, percent good, status |

## Usage

Call `fileheader()` once and pass its output to all other functions — this
avoids re-scanning the file for each data type.

```python
from pyadps.io import pd0_parser

# Parse file structure once
dt, byte, byteskip, offset, idarray, n_ens, err = pd0_parser.fileheader('deployment.000')

# Reuse for all subsequent reads
fl, _, _          = pd0_parser.fixedleader('deployment.000', byteskip, offset, idarray, n_ens)
vl, _, _          = pd0_parser.variableleader('deployment.000', byteskip, offset, idarray, n_ens)
vel, _, _, _, _   = pd0_parser.datatype('deployment.000', 'velocity',
                        byteskip=byteskip, offset=offset, idarray=idarray, ensemble=n_ens)
```

**`datatype()` variable names:** `'velocity'`, `'correlation'`, `'echo'`,
`'percent good'`, `'status'`.

**Data shapes:**
- `fixedleader()` → `(36, n_ensembles)`
- `variableleader()` → `(48, n_ensembles)`
- `datatype()` → `(beams, cells, n_ensembles)`

Velocity is in mm/s with −32768 as the missing value sentinel. All other
functions return raw integer counts.

## Error Handling

Every function returns an error code as the last element of its return tuple.

```python
data, n_ens, err = pd0_parser.fixedleader('deployment.000')

if err != 0:
    print(pd0_parser.ErrorCode.get_message(err))
```

| Code | Name | Description |
|------|------|-------------|
| 0 | `SUCCESS` | Completed successfully |
| 1 | `FILE_NOT_FOUND` | File does not exist |
| 2 | `PERMISSION_DENIED` | Access denied |
| 3 | `IO_ERROR` | File open or read failed |
| 4 | `OUT_OF_MEMORY` | Insufficient memory |
| 5 | `WRONG_RDIFILE_TYPE` | Not a valid RDI PD0 file |
| 6 | `ID_NOT_FOUND` | Data type ID not found in ensemble |
| 7 | `DATATYPE_MISMATCH` | Inconsistent data types across ensembles |
| 8 | `FILE_CORRUPTED` | Invalid structure or truncated data |
| 9 | `VALUE_ERROR` | Invalid argument |
| 10 | `CHECKSUM_ERROR` | Ensemble checksum failed |
| 99 | `UNKNOWN_ERROR` | Unexpected error |

If a file is truncated mid-ensemble, all successfully parsed ensembles up
to the point of corruption are returned along with `CHECKSUM_ERROR`.

## See Also

- {doc}`binary_reader` — High-level reader returning `xarray.Dataset`
- {doc}`accessors` — Domain-specific methods on the Dataset
