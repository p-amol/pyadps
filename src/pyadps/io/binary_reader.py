#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Header Reader for pyadps - ACCESSOR-BASED ARCHITECTURE

This module provides functions to read ADCP file headers into xarray.Dataset
objects with accessor-based extensions for domain-specific operations.

Architecture:
    Data Structure: xarray.Dataset
    Custom Methods: Via @xr.register_dataset_accessor (in accessors.py)
    Access Pattern: ds.header.method_name()
    Backward Compat: FileHeaderLegacy kept for v0.4.0

The Header dataset contains ensemble-level metadata including:
- File structure information (byte counts, offsets)
- Data type mappings and IDs
- File integrity information

Usage:
    >>> import pyadps
    >>> import pyadps.accessors  # Register accessors
    >>> ds = pyadps.Header('test.000')
    >>> ds.header.data_types(ens=0)
    >>> ds.header.check_file()

Author: pyadps development team
License: MIT
"""

import json

import logging
from pathlib import Path
from typing import Dict, Optional, Union, Any

import numpy as np
import pandas as pd
from pandas._typing import ArrayLike
import xarray as xr

import importlib.resources as pkg_resources
from pyadps.io import pd0_parser


logger = logging.getLogger(__name__)

# ============================================================================
# MODULE CONSTANTS
# ============================================================================

# Version: Extract from package metadata (pyproject.toml) when installed
try:
    from importlib.metadata import version as get_version

    PYADPS_VERSION = get_version("pyadps")
except Exception:  # pragma: no cover
    # Fallback for development or if package not installed
    PYADPS_VERSION = "0.0.0.dev0"  # pragma: no cover

# Data format identifier for RDI PD0 binary format
ADCP_DATA_FORMAT = "PD0"

# Type aliases for clarity
FilePathType = Union[str, Path]


# ============================================================================
# NEW IMPLEMENTATION (xarray.Dataset based with Accessors)
# ============================================================================


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

# Missing value constant for velocity data (16-bit signed integer)
VELOCITY_MISSING_VALUE = -32768

# Data ID to Name mapping (used by both accessor and read)
_DATA_ID_NAME_MAP = {
    0: "Fixed Leader",
    1: "Fixed Leader",
    128: "Variable Leader",
    129: "Variable Leader",
    256: "Velocity",
    257: "Velocity",
    512: "Correlation",
    513: "Correlation",
    768: "Echo",
    769: "Echo",
    1024: "Percent Good",
    1025: "Percent Good",
    1280: "Status",
    1536: "Bottom Track",
}


def _get_available_data_types(ds_header: xr.Dataset, ens: int = 0) -> list:
    """
    Extract available data types from header dataset.

    This is a helper function used by both the read() function and the
    Header accessor to identify which data types are present in the file.

    Parameters
    ----------
    ds_header : xr.Dataset
        Header dataset (from read_header)
    ens : int, optional
        Ensemble index (0-based), by default 0 (first ensemble)

    Returns
    -------
    list
        List of data type names (e.g., ['Fixed Leader', 'Variable Leader', 'Velocity'])

    Notes
    -----
    - Used internally by read() and Header accessor
    - Extracts data IDs from header and maps to human-readable names
    - Stops at ID=0 (end marker)
    - Avoids duplicates
    """
    available_datatypes: list[str] = []

    if "data_id" not in ds_header.data_vars:
        logger.warning(f"No 'data_id' variable found for ensemble {ens}")
        return available_datatypes

    try:
        # Extract data IDs for specified ensemble
        if "ensemble" in ds_header.dims:
            data_id_array = ds_header["data_id"].isel(ensemble=ens).values
        else:
            data_id_array = ds_header["data_id"].isel(time=ens).values

        for data_id in data_id_array:
            data_id_int = int(data_id)

            # Map ID to name
            if data_id_int in _DATA_ID_NAME_MAP:
                name = _DATA_ID_NAME_MAP[data_id_int]
                if name not in available_datatypes:  # Avoid duplicates
                    available_datatypes.append(name)
            else:
                logger.warning(f"Unknown data ID: {data_id_int}")
                available_datatypes.append(f"Unknown (ID: {data_id_int})")

    except (KeyError, IndexError) as e:
        logger.warning(f"Error extracting data types: {e}")
        return []

    return available_datatypes


def read_header(adcp_file: FilePathType) -> xr.Dataset:
    """
    Load ADCP file header metadata into xarray.Dataset.

    This is the primary function for reading ADCP file headers in pyadps v1.0.0.
    Returns a plain xarray.Dataset with custom methods available via the
    .header accessor (registered in accessors.py).

    The file size is computed at read time and stored as an immutable attribute,
    eliminating the need to store the file path and preventing issues with
    post-load file modifications.

    Parameters
    ----------
    adcp_file : str or Path
        Path to ADCP binary file in PD0 format.

    Returns
    -------
    xr.Dataset
        xarray Dataset with ADCP file header metadata. Access custom
        methods via .header namespace:

        - ds.header.data_types()
        - ds.header.check_file()
        - ds.header.print_check_file()
        - ds.header.summary()
        - ds.header.validate()
        - ds.header.get_available_data_types()

    Raises
    ------
    FileNotFoundError
        If adcp_file does not exist.
    ValueError
        If file is not a valid ADCP file.

    Examples
    --------
    >>> import pyadps
    >>> import pyadps.accessors  # Register accessor
    >>> ds = pyadps.read_header('test.000')
    >>> print(ds)  # Standard xarray.Dataset
    >>> ds.header.check_file()  # Custom method via accessor
    >>> ds.header.print_check_file()
    >>> print(f"File size: {ds.attrs['file_size_bytes']} bytes")

    Notes
    -----
    Custom methods are accessed via the .header accessor, which is
    automatically registered when pyadps.accessors is imported. This
    maintains clean separation between xarray methods and domain-specific
    ADCP operations.

    The file_size_bytes attribute is computed at read time using
    Path.stat().st_size, making it immutable after dataset creation.
    If the file cannot be read, file_size_bytes is set to -1.

    See Also
    --------
    FileHeaderLegacy : v0.4.0 interface (deprecated)
    Header : Convenience wrapper class
    """

    filename = str(adcp_file)

    # Parse file using pd0_parser
    (
        datatype,
        bytes_array,
        byteskip,
        address_offset,
        dataid,
        ensemble,
        error_code,
    ) = pd0_parser.fileheader(filename)

    # Get error message
    error_message = pd0_parser.ErrorCode.get_message(error_code)

    if error_code in (
        1,
        2,
        3,
        4,
    ):  # FILE_NOT_FOUND, PERMISSION_DENIED, IO_ERROR, OUT_OF_MEMORY
        logger.error(f"Critical error reading file header: {error_message}")
        raise FileNotFoundError(f"Cannot read ADCP file '{filename}': {error_message}")

    # Determine number of datatypes (should be same for all ensembles if uniform)
    num_datatypes = int(datatype[0]) if len(datatype) > 0 else 0

    # ========================================================================
    # Compute file size at read time (not lazy)
    # ========================================================================
    file_size_bytes = -1  # Default value for error cases
    try:
        file_size_bytes = int(Path(adcp_file).stat().st_size)
    except (OSError, IOError) as e:
        logger.warning(f"Could not read file size for {adcp_file}: {e}")

    # ========================================================================
    # Build xarray.Dataset
    # ========================================================================

    # Coordinates
    coords = {
        "ensemble": np.arange(ensemble) + 1,
        "data_type": np.arange(num_datatypes) + 1,
    }

    # Data variables
    data_vars = {
        "data_type_array": ("ensemble", datatype.astype(np.int16)),
        "byte": ("ensemble", bytes_array.astype(np.int32)),
        "byte_skip": ("ensemble", byteskip.astype(np.int32)),
        "address_offset": (
            ("ensemble", "data_type"),
            address_offset.astype(np.int16),
        ),
        "data_id": (
            ("ensemble", "data_type"),
            dataid.astype(np.int16),
        ),
    }

    # Attributes (metadata)
    # NOTE: The following are now computed as accessor properties:
    # - filename
    # - total_ensembles
    # - num_data_types (number of data types)
    # - error_message (derived from error_message)
    # - calculated_size_bytes (computed from bytes data_var)
    # - file_size_match (comparison of computed sizes)
    # Access these via: ds.header.property_name
    attrs = {
        "filename": Path(adcp_file).name,  # Store just the filename, not the full path
        "total_ensembles": int(ensemble),
        "num_data_types": int(num_datatypes),
        "error_message": str(error_message),
        "file_size_bytes": int(file_size_bytes),
        "adcp_data_format": ADCP_DATA_FORMAT,
        "pyadps_component": "Header",
        "pyadps_version": PYADPS_VERSION,
    }

    # Create and return xarray.Dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=attrs,
    )

    return ds


def read_fixed_leader(
    adcp_file: FilePathType,
    byteskip: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    idarray: Optional[np.ndarray] = None,
    ensemble: int = 0,
    json_file_path: Optional[str] = None,
    include_decoded: bool = True,
) -> xr.Dataset:
    """
    Load ADCP Fixed Leader data into xarray.Dataset with raw and optionally decoded variables.

    Fixed Leader data contains non-dynamic ADCP configuration and hardware
    information that generally remains constant throughout a file. This function
    reads the raw Fixed Leader fields (36 fields from PD0 format) and creates
    individual xarray DataArrays for each field with full NetCDF attributes
    from fixed_leader_meta.json metadata.

    If include_decoded=True (default), computes 25 additional decoded fields
    derived from bit-extraction of raw fields. These include frequency, beam pattern,
    sensor availability flags, and other configuration information.

    The function returns a plain xarray.Dataset with custom methods available
    via the .fixed_leader accessor (registered in accessors.py).

    Parameters
    ----------
    adcp_file : str or Path
        Path to ADCP binary file in PD0 format.
    byteskip : np.ndarray, optional
        From read_header(). If None, auto-fetch from fileheader.
        Array of bytes to skip for each ensemble.
    offset : np.ndarray, optional
        From read_header(). If None, auto-fetch from fileheader.
        Array of offsets for each ensemble.
    idarray : np.ndarray, optional
        From read_header(). If None, auto-fetch from fileheader.
        Array of data type IDs for each ensemble.
    ensemble : int, optional
        Number of ensembles. If 0, auto-fetch from fileheader. Default is 0.
    json_file_path : str, optional
        Path to custom metadata JSON file. If None, uses default package location
        (pyadps.io.metadata.fixed_leader_meta.json).
        Can be absolute or relative filesystem path.

        Examples:
            None                                    -> Use package default
            "/etc/pyadps/fixed_leader_meta.json"  -> Absolute path
            "./config/fixed_leader_meta.json"     -> Relative path

    include_decoded : bool, optional
        If True (default), compute and include 25 decoded fields derived from
        bit-extraction of raw fields. These include frequency, beam pattern,
        sensor availability flags, and other configuration information.
        If False, return only the 36 raw fields.
        Default is True.

    Returns
    -------
    xr.Dataset
        xarray Dataset with Fixed Leader variables. Access custom methods via
        .fixed_leader namespace:

        - If include_decoded=True: 61 variables (36 raw + 25 decoded)
        - If include_decoded=False: 36 variables (raw only)

        Custom accessor methods:
        - ds.fixed_leader.field(ens=0)
        - ds.fixed_leader.system_configuration(ens=-1)
        - ds.fixed_leader.ex_coord_trans(ens=0)
        - ds.fixed_leader.ez_sensor(ens=0, field='source')
        - ds.fixed_leader.is_uniform
        - ds.fixed_leader.get_field(field_name, ens=None)
        - ds.fixed_leader.to_dict(ens=0)
        - ds.fixed_leader.validate()
        - ds.fixed_leader.get_raw_fields()
        - ds.fixed_leader.get_decoded_fields()

    Raises
    ------
    FileNotFoundError
        If adcp_file does not exist or metadata file cannot be found.
    ValueError
        If file is not a valid ADCP file or data cannot be read.
    json.JSONDecodeError
        If metadata file is malformed.

    Examples
    --------
    >>> import pyadps
    >>> import pyadps.accessors  # Register accessor
    >>> ds = pyadps.read_fixed_leader('test.000')
    >>> print(ds)  # Standard xarray.Dataset with 61 variables (36 raw + 25 decoded)
    >>> n_beams = ds['num_beams'].values
    >>> config = ds.fixed_leader.system_configuration()  # Via accessor
    >>> print(f"Frequency: {config['Frequency']}")

    Load with decoded fields (default):

    >>> ds = pyadps.read_fixed_leader('test.000')
    >>> print(ds['frequency'].values)  # Decoded field: ['300 kHz']
    >>> print(len(ds.data_vars))  # 61 variables

    Load raw fields only:

    >>> ds_raw = pyadps.read_fixed_leader('test.000', include_decoded=False)
    >>> print(len(ds_raw.data_vars))  # 36 variables

    Using custom metadata file:

    >>> ds = pyadps.read_fixed_leader('test.000',
    ...                               json_file_path='/etc/pyadps/fixed_leader_meta.json')

    Notes
    -----
    Each of the 36 Fixed Leader fields is stored as a separate xarray DataArray
    with dimensions (ensemble,). Variable names and attributes come from
    fixed_leader_meta.json raw_fields section to ensure CF Convention compliance.

    Decoded fields are computed by bit-extraction from raw fields. For example,
    'frequency' is extracted from bits 15-13 of 'system_configuration_code' and
    mapped to human-readable values like '300 kHz'.

    If byteskip, offset, idarray, or ensemble are not provided, this function
    will call read_header() to fetch them automatically. This is convenient but
    may be inefficient if reading multiple components from the same file.

    For better performance with large files, pre-fetch header information:

    >>> header_ds = pyadps.read_header('test.000')
    >>> fl_ds = pyadps.read_fixed_leader('test.000',
    ...                                   byteskip=header_ds.byte_skip.values,
    ...                                   offset=header_ds.address_offset.values,
    ...                                   idarray=header_ds.data_id.values,
    ...                                   ensemble=header_ds.attrs['total_ensembles'])

    For custom metadata file location:

    >>> fl_ds = pyadps.read_fixed_leader('test.000',
    ...                                   json_file_path='/custom/path/fixed_leader_meta.json')

    See Also
    --------
    read_header : Load file header metadata
    FixedLeaderAccessor : Custom methods for Fixed Leader data
    """

    filename = str(adcp_file)

    # If optional parameters not provided, fetch from fileheader
    if byteskip is None or offset is None or idarray is None or ensemble == 0:
        logger.debug(f"Auto-fetching header info from {filename}")
        header_ds = read_header(filename)
        if byteskip is None:
            byteskip = header_ds.byte_skip.values
        if offset is None:
            offset = header_ds.address_offset.values
        if idarray is None:
            idarray = header_ds.data_id.values
        if ensemble == 0:
            ensemble = header_ds.attrs["total_ensembles"]

    # Extract Fixed Leader data using pd0_parser
    fl_data, ensembles, error_code = pd0_parser.fixedleader(
        filename,
        byteskip=byteskip,
        offset=offset,
        idarray=idarray,
        ensemble=ensemble,
    )

    # Get error message
    error_message = pd0_parser.ErrorCode.get_message(error_code)

    # ========================================================================
    # Load metadata from fixed_leader_meta.json
    # ========================================================================
    logger.debug(f"Loading metadata with json_file_path={json_file_path}")
    metadata = _load_fixed_leader_metadata(json_file_path=json_file_path)

    # ========================================================================
    # Build xarray.Dataset with Fixed Leader data
    # ========================================================================

    # fl_data shape: (36, n_ensembles) where 36 is the number of fixed leader fields
    # We need to transpose to (n_ensembles, 36) for proper xarray dimensions

    n_fields = fl_data.shape[0]  # Should be 36
    n_ensembles = fl_data.shape[1] if len(fl_data.shape) > 1 else 1

    if len(fl_data.shape) == 1:
        # Single ensemble case - reshape
        fl_data = fl_data.reshape(-1, 1)

    # Coordinates
    coords = {
        "ensemble": np.arange(n_ensembles),
    }

    # Data variables - Create 36 individual variables from raw fields metadata
    data_vars = {}

    for field_idx, field_meta in metadata.items():
        if field_meta.get("is_raw", False):
            var_name = field_meta["name"]
            field_index = field_meta["index"]

            # Extract data for this field (row from fl_data array)
            var_data = fl_data[field_index, :]

            # Create data variable with ensemble dimension
            data_vars[var_name] = ("ensemble", var_data.astype(np.int64))

            # Store metadata as attributes for this variable
            # Will be attached after dataset creation

    # Create base xarray.Dataset
    # Update num_fields based on include_decoded parameter
    total_variables = n_fields + (25 if include_decoded else 0)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "filename": Path(adcp_file).name,
            "total_ensembles": int(n_ensembles),
            "num_fields": int(total_variables),
            "error_message": str(error_message),
            "pyadps_component": "FixedLeader",
            "pyadps_version": PYADPS_VERSION,
            "adcp_data_format": ADCP_DATA_FORMAT,
        },
    )

    # ========================================================================
    # Apply variable-level attributes from metadata
    # ========================================================================
    for field_idx, field_meta in metadata.items():
        if field_meta.get("is_raw", False):
            var_name = field_meta["name"]
            if var_name in ds.data_vars:
                # Build attributes dict for this variable
                var_attrs = _build_variable_attributes(field_meta)
                ds[var_name].attrs.update(var_attrs)

    # ========================================================================
    # Add decoded fields if requested
    # ========================================================================
    if include_decoded:
        ds = _add_decoded_fields(ds, metadata)

    return ds


# ============================================================================
# HELPER FUNCTIONS FOR FIXED LEADER XARRAY CONVERSION
# ============================================================================


def _extract_decoded_field(
    ds: xr.Dataset,
    decoded_meta: Dict,
    bit_info: Dict,
) -> Union[np.ndarray, None]:
    """
    Extract single decoded field via binary string slicing and mapping.

    This function uses the a bit extraction, which converts integers to
    binary strings and extracts bit ranges using direct string slicing.

    The bit specification in metadata uses string indices directly:
    - "15-13" means binary_string[13:16]  (indices 13, 14, 15)
    - "12" means binary_string[12]         (index 12)
    - "11-10" means binary_string[10:12]   (indices 10, 11)

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the source field (raw field).
    decoded_meta : dict
        Metadata dict for the decoded field.
    bit_info : dict
        Bit extraction information with keys:
        - "source_field" : str - name of raw field to extract from
        - "bits" : str - bit range using string indices (e.g., "15-13")
        - "bit" : str/int - single index (e.g., "12")
        - "mapping" : dict - optional mapping of extracted values to strings

    Returns
    -------
    np.ndarray or None
        Extracted and optionally mapped values. Returns None if source field
        not found or extraction fails.

    Examples
    --------
    >>> bit_info = {
    ...     "source_field": "system_configuration_code",
    ...     "bits": "15-13",
    ...     "mapping": {"0": "75 kHz", "1": "150 kHz", "2": "300 kHz"}
    ... }
    >>> decoded_values = _extract_decoded_field(ds, decoded_meta, bit_info)
    >>> print(decoded_values)
    array(['300 kHz', '300 kHz', ...], dtype=object)
    """
    source_field = bit_info.get("source_field")
    # Handle both "bits" (range) and "bit" (single) keys in metadata
    bits_spec = bit_info.get("bits") or bit_info.get("bit")
    mapping = bit_info.get("mapping", {})

    # Validate source field exists
    if source_field not in ds.data_vars:
        logger.warning(
            f"Source field '{source_field}' not found in dataset. "
            f"Cannot extract decoded field."
        )
        return None

    # Validate bits specification
    if bits_spec is None:
        logger.warning(
            "No 'bits' or 'bit' specification found in bit_extraction. "
            "Cannot extract decoded field."
        )
        return None

    # Get raw values
    raw_values = ds[source_field].values

    # Determine bit width (usually 16 for system config codes, 8 for sensor codes)
    # Assume 16 bits for codes > 255, otherwise 8 bits
    sample_value = int(raw_values.flat[0]) if raw_values.size > 0 else 0
    bit_width = 16 if sample_value > 255 else 8

    # Parse bit specification
    # Metadata format uses string indices: "15-13" means indices 13:16
    # This maps to: convert "15-13" -> string[13:16]
    if "-" in str(bits_spec):
        # Range specification: "15-13" means indices 13:16
        # The format is: "higher_index-lower_index" where we extract [lower_index:higher_index+1]
        parts = str(bits_spec).split("-")
        higher_idx = int(parts[0])
        lower_idx = int(parts[1])
        start_idx = lower_idx
        end_idx = higher_idx + 1
    else:
        # Single index specification: "12" means index [12:13]
        idx = int(bits_spec)
        start_idx = idx
        end_idx = idx + 1

    # Extract bits using binary string representation (matching FileHeader method)
    extracted_ints = []
    for val in raw_values:
        binary_str = format(int(val), f"0{bit_width}b")
        bit_substring = binary_str[start_idx:end_idx]
        extracted_ints.append(int(bit_substring, 2))

    # Map to readable values if mapping provided
    if mapping:
        mapped = []
        # NEW v1.2: Get dtype from metadata to determine output type
        dtype = decoded_meta.get("dtype", "string")

        for val in extracted_ints:
            str_val = str(int(val))
            mapped_val = mapping.get(str_val, f"Unknown({int(val)})")

            # NEW v1.2: Handle both integer and string encoding
            if dtype == "int64":
                # Convert mapped value to integer for v1.2 integer-encoded fields
                try:
                    mapped.append(int(mapped_val))
                except (ValueError, TypeError):
                    # Fallback for unknown values
                    mapped.append(0)
            else:
                # Keep as string for backward compatibility with string-encoded fields
                mapped.append(mapped_val)

        # NEW v1.2: Return appropriate dtype based on metadata
        if dtype == "int64":
            return np.array(mapped, dtype=np.int64)
        else:
            return np.array(mapped, dtype=object)

    return np.array(extracted_ints, dtype=np.int64)


def _add_decoded_fields(
    ds: xr.Dataset,
    metadata: Dict[int, Dict],
) -> xr.Dataset:
    """
    Compute and add 25 decoded fields to Fixed Leader dataset.

    Decoded fields are derived from bit-extraction of raw fields like:
    - system_configuration_code â†’ frequency, beam_pattern, etc.
    - sensor_available_code â†’ temp_available, compass_available, etc.
    - coordinate_transformation_code â†’ earth_coordinates, bin_mapping, etc.

    Each decoded field is added as a separate xarray DataArray with
    appropriate variable attributes from the metadata.

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset with 36 raw Fixed Leader fields.
    metadata : dict[int, dict]
        Dictionary mapping field index (0-35) to field metadata.
        Should include both raw_fields and decoded_fields sections
        from fixed_leader_meta.json.

    Returns
    -------
    xr.Dataset
        Modified dataset with 25 decoded fields added. Dataset now has
        61 total variables (36 raw + 25 decoded).

    Notes
    -----
    Decoded fields share the 'ensemble' dimension with raw fields.
    Each decoded field gets full metadata attributes via
    _build_variable_attributes().

    If a decoded field fails to be computed (e.g., source field not found),
    it is skipped with a warning logged.

    See Also
    --------
    _extract_decoded_field : Extract single decoded field via bit manipulation
    _build_variable_attributes : Build NetCDF attributes from metadata
    """
    # Load full metadata including decoded_fields section
    # Note: metadata param contains only raw_fields dict,
    # so we need to load full metadata to access decoded_fields
    json_file_path = None  # Use default
    full_metadata_json = _load_full_metadata_json(json_file_path)
    decoded_fields_meta = full_metadata_json.get("decoded_fields", {})

    if not decoded_fields_meta:
        logger.warning(
            "No decoded_fields section found in metadata. Skipping decoded fields."
        )
        return ds

    # Iterate over decoded fields and add each to dataset
    for field_key, decoded_meta in decoded_fields_meta.items():
        # Skip if not marked as decoded
        if not decoded_meta.get("is_decoded", False):
            continue

        var_name = decoded_meta.get("name")
        bit_info = decoded_meta.get("bit_extraction", {})

        if not var_name:
            logger.warning(
                f"Decoded field {field_key} has no 'name' attribute. Skipping."
            )
            continue

        if not bit_info:
            logger.warning(
                f"Decoded field '{var_name}' has no 'bit_extraction' info. Skipping."
            )
            continue

        try:
            # Extract the decoded values
            decoded_value = _extract_decoded_field(ds, decoded_meta, bit_info)

            if decoded_value is None:
                logger.debug(f"Could not extract decoded field '{var_name}'. Skipping.")
                continue

            # Add to dataset with ensemble dimension
            # NEW v1.2: Use dtype returned from _extract_decoded_field (already correct)
            if isinstance(decoded_value, np.ndarray):
                ds[var_name] = ("ensemble", decoded_value)
            else:
                ds[var_name] = ("ensemble", np.array(decoded_value, dtype=object))

            # Add metadata attributes
            var_attrs = _build_variable_attributes(decoded_meta)
            ds[var_name].attrs.update(var_attrs)

            logger.debug(f"Successfully added decoded field: {var_name}")

        except Exception as e:
            logger.error(f"Failed to compute decoded field '{var_name}': {e}")
            continue

    return ds


def _load_full_metadata_json(json_file_path: Optional[str] = None) -> Dict:
    """
    Load complete metadata JSON including both raw_fields and decoded_fields.

    Helper function to load the full metadata JSON structure (not just raw fields).

    Parameters
    ----------
    json_file_path : str, optional
        Path to custom metadata JSON file. If None, uses default package location.

    Returns
    -------
    dict
        Full metadata dictionary with all sections (raw_fields, decoded_fields, etc.)

    Raises
    ------
    FileNotFoundError
        If metadata file cannot be found.
    json.JSONDecodeError
        If metadata file is malformed.
    """
    default_filename = "fixed_leader_meta.json"
    default_package = "pyadps.io.metadata"

    # Strategy 1: Load from custom file path if provided
    if json_file_path is not None:
        logger.debug(f"Loading full metadata from custom path: {json_file_path}")
        custom_path = Path(json_file_path)

        if not custom_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at: {custom_path.resolve()}"
            )

        if not custom_path.is_file():
            raise ValueError(f"Path is not a file: {custom_path.resolve()}")

        try:
            with open(custom_path, "r", encoding="utf-8") as f:
                full_metadata = json.load(f)
            logger.debug(f"Loaded full metadata from: {custom_path.resolve()}")
            return full_metadata
        except json.JSONDecodeError as e:
            logger.error(f"Metadata file is malformed at {custom_path}: {e}")
            raise
        except IOError as e:
            raise FileNotFoundError(
                f"Cannot read metadata file: {custom_path.resolve()}\nError: {e}"
            )

    # Strategy 2: Use importlib.resources (package default)
    logger.debug(
        f"Loading full {default_filename} via importlib.resources "
        f"from {default_package}..."
    )

    try:
        files = pkg_resources.files(default_package)
        metadata_file = files / default_filename

        if metadata_file.is_file():
            logger.debug(f"Found {default_filename} in package {default_package}")
            content = metadata_file.read_text(encoding="utf-8")
            full_metadata = json.loads(content)
            return full_metadata
        else:
            raise FileNotFoundError(
                f"Resource {default_filename} not found in package {default_package}"
            )

    except json.JSONDecodeError as e:
        logger.error(f"Package metadata file is malformed: {e}")
        raise
    except (FileNotFoundError, ModuleNotFoundError) as e:
        error_msg = (
            f"Could not find '{default_filename}'.\n\n"
            f"Attempted locations:\n"
            f"  1. Package: {default_package}.{default_filename}\n\n"
            f"Solutions:\n"
            f"  â€¢ Provide explicit path:\n"
            f"    _load_full_metadata_json('/path/to/{default_filename}')\n"
            f"  â€¢ Ensure pyadps is properly installed with metadata files"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg) from e


def _load_fixed_leader_metadata(
    json_file_path: Optional[str] = None,
) -> Dict[int, Dict]:
    """
    Load Fixed Leader metadata from fixed_leader_meta.json.

    Uses importlib.resources for reliable package resource access, with
    optional support for custom file paths.

    Parameters
    ----------
    json_file_path : str, optional
        Path to custom metadata JSON file. If None, uses default package location.
        Can be absolute or relative filesystem path.

        Examples:
            None                                    -> Use package default
            "/etc/pyadps/fixed_leader_meta.json"  -> Absolute path
            "./config/fixed_leader_meta.json"     -> Relative path

    Returns
    -------
    dict
        Dictionary mapping field index to field metadata dict.
        Keys are integer indices (0, 1, ..., 35).

    Raises
    ------
    FileNotFoundError
        If metadata file cannot be found at specified or default location.
    json.JSONDecodeError
        If metadata file is malformed.
    ValueError
        If json_file_path is not a valid file.

    Examples
    --------
    >>> # Use default package location
    >>> metadata = _load_fixed_leader_metadata()

    >>> # Load from custom absolute path
    >>> metadata = _load_fixed_leader_metadata('/path/to/fixed_leader_meta.json')

    >>> # Load from custom relative path
    >>> metadata = _load_fixed_leader_metadata('./data/fixed_leader_meta.json')
    """

    default_filename = "fixed_leader_meta.json"
    default_package = "pyadps.io.metadata"

    # Strategy 1: Load from custom file path if provided
    if json_file_path is not None:
        logger.debug(f"Loading metadata from custom path: {json_file_path}")
        custom_path = Path(json_file_path)

        if not custom_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at: {custom_path.resolve()}"
            )

        if not custom_path.is_file():
            raise ValueError(f"Path is not a file: {custom_path.resolve()}")

        try:
            with open(custom_path, "r", encoding="utf-8") as f:
                full_metadata = json.load(f)
            logger.debug(f"Loaded metadata from: {custom_path.resolve()}")
            return _extract_and_sort_raw_fields(full_metadata)
        except json.JSONDecodeError as e:
            logger.error(f"Metadata file is malformed at {custom_path}: {e}")
            raise
        except IOError as e:
            raise FileNotFoundError(
                f"Cannot read metadata file: {custom_path.resolve()}\nError: {e}"
            )

    # Strategy 2: Use importlib.resources (package default)
    logger.debug(
        f"Loading {default_filename} via importlib.resources "
        f"from {default_package}..."
    )

    try:
        # Use files() API (Python 3.12+)
        files = pkg_resources.files(default_package)
        metadata_file = files / default_filename

        if metadata_file.is_file():
            logger.debug(f"Found {default_filename} in package {default_package}")
            content = metadata_file.read_text(encoding="utf-8")
            full_metadata = json.loads(content)
            return _extract_and_sort_raw_fields(full_metadata)
        else:
            raise FileNotFoundError(
                f"Resource {default_filename} not found in package {default_package}"
            )

    except json.JSONDecodeError as e:
        logger.error(f"Package metadata file is malformed: {e}")
        raise
    except (FileNotFoundError, ModuleNotFoundError) as e:
        error_msg = (
            f"Could not find '{default_filename}'.\n\n"
            f"Attempted locations:\n"
            f"  1. Package: {default_package}.{default_filename}\n\n"
            f"Solutions:\n"
            f"  Provide explicit path:\n"
            f"    _load_fixed_leader_metadata('/path/to/{default_filename}')\n"
            f"  Ensure pyadps is properly installed with metadata files"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg) from e


def _extract_and_sort_raw_fields(full_metadata: Dict) -> Dict[int, Dict]:
    """
    Extract and sort raw fields from loaded metadata.

    Helper function to standardize metadata extraction logic.

    Parameters
    ----------
    full_metadata : dict
        Full metadata dict loaded from JSON

    Returns
    -------
    dict
        Dictionary mapping field index (int) to field metadata dict
    """
    raw_fields = full_metadata.get("raw_fields", {})

    if not raw_fields:
        raise ValueError(
            "flmeta.json does not contain 'raw_fields' section. "
            "Ensure metadata file is properly formatted."
        )

    # Convert keys to sorted order for consistent field indexing
    # Keys should be "0", "1", ..., "35"
    metadata_dict = {}
    try:
        for key in sorted(raw_fields.keys(), key=lambda x: int(x)):
            metadata_dict[int(key)] = raw_fields[key]
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid field keys in flmeta.json. Keys must be numeric strings. "
            f"Error: {e}"
        )

    if len(metadata_dict) != 36:
        logger.warning(
            f"Expected 36 raw fields, found {len(metadata_dict)}. "
            f"Some fields may be missing from flmeta.json."
        )

    return metadata_dict


# ============================================================================
# VARIABLE LEADER FUNCTIONS FOR FILEREADER.PY
# ============================================================================


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _load_variable_leader_metadata(
    json_file_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load Variable Leader field metadata from JSON file.

    Loads raw_fields, composite_fields, and decoded_fields sections from metadata.

    Parameters
    ----------
    json_file_path : str, optional
        Path to custom metadata JSON file. If None, uses package default.

    Returns
    -------
    dict
        Metadata with keys: 'raw_fields', 'composite_fields', 'decoded_fields'
    """
    try:
        if json_file_path is not None:
            metadata_path = Path(json_file_path)
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {json_file_path}")
            logger.debug(f"Loading metadata from {metadata_path}")
            with open(metadata_path, "r") as f:
                metadata_full = json.load(f)
        else:
            # Use package default via importlib.resources
            try:
                files = pkg_resources.files("pyadps.io.metadata")
                metadata_text = files.joinpath("variable_leader_meta.json").read_text()
            except (AttributeError, TypeError, ModuleNotFoundError):
                metadata_text = pkg_resources.read_text(
                    "pyadps.io.metadata", "variable_leader_meta.json"
                )
            metadata_full = json.loads(metadata_text)
            logger.debug("Loaded metadata from package default location")

    except json.JSONDecodeError as e:
        logger.error(f"Metadata JSON is malformed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load Variable Leader metadata: {e}")
        raise

    # Extract all field sections
    return {
        "raw_fields": metadata_full.get("raw_fields", {}),
        "composite_fields": metadata_full.get("composite_fields", {}),
        "decoded_fields": metadata_full.get("decoded_fields", {}),
        "metadata": metadata_full,
    }


def _build_variable_attributes(field_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build xarray variable attributes from field metadata.

    Ensures CF Convention compliance.

    Parameters
    ----------
    field_meta : dict
        Field metadata dictionary

    Returns
    -------
    dict
        Attributes dictionary for xarray.DataArray.attrs
    """
    attrs = {}

    # CF Convention fields
    if "long_name" in field_meta:
        attrs["long_name"] = field_meta["long_name"]

    # Units (UDUNITS2 format)
    if "unit" in field_meta:
        unit_val = field_meta["unit"]
        if unit_val == "1":
            attrs["units"] = "dimensionless"
        else:
            attrs["units"] = unit_val

    # Valid range
    if "valid_min" in field_meta and field_meta["valid_min"] is not None:
        attrs["valid_min"] = field_meta["valid_min"]
    if "valid_max" in field_meta and field_meta["valid_max"] is not None:
        attrs["valid_max"] = field_meta["valid_max"]

    # Scaling factors (if not default)
    if field_meta.get("scale_factor", 1.0) != 1.0:
        attrs["scale_factor"] = field_meta["scale_factor"]
    if field_meta.get("add_offset", 0.0) != 0.0:
        attrs["add_offset"] = field_meta["add_offset"]

    # Descriptive metadata
    if "description" in field_meta:
        attrs["description"] = field_meta["description"]
    if "source" in field_meta:
        attrs["source"] = field_meta["source"]
    if "comments" in field_meta:
        attrs["comments"] = field_meta["comments"]

    # Classification flags
    attrs["is_raw"] = str(field_meta.get("is_raw", False)).lower()
    attrs["is_decoded"] = str(field_meta.get("is_decoded", False)).lower()
    attrs["ensemble_varying"] = str(field_meta.get("ensemble_varying", False)).lower()

    return attrs


def _compute_composite_timestamp(
    ds: xr.Dataset, timestamp_type: str = "rtc"
) -> np.ndarray:
    """
    Compute composite timestamp from RTC fields.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with raw Variable Leader data
    timestamp_type : str
        'rtc' for RTC timestamp or 'y2k' for Y2K timestamp

    Returns
    -------
    np.ndarray
        datetime64[ns] array of timestamps
    """
    try:
        if timestamp_type == "rtc":
            year = ds["rtc_year"].values.astype(np.int32) + 2000
            month = ds["rtc_month"].values.astype(np.int32)
            day = ds["rtc_day"].values.astype(np.int32)
            hour = ds["rtc_hour"].values.astype(np.int32)
            minute = ds["rtc_minute"].values.astype(np.int32)
            second = ds["rtc_second"].values.astype(np.int32)
            hundredth = ds["rtc_hundredth"].values.astype(np.int32)
        elif timestamp_type == "y2k":
            year = ds["y2k_century"].values.astype(np.int32) * 100 + ds[
                "y2k_year"
            ].values.astype(np.int32)
            month = ds["y2k_month"].values.astype(np.int32)
            day = ds["y2k_day"].values.astype(np.int32)
            hour = ds["y2k_hour"].values.astype(np.int32)
            minute = ds["y2k_minute"].values.astype(np.int32)
            second = ds["y2k_second"].values.astype(np.int32)
            hundredth = ds["y2k_hundredth"].values.astype(np.int32)
        else:
            raise ValueError(f"Unknown timestamp_type: {timestamp_type}")

        n_ens = len(year)
        datetimes = np.empty(n_ens, dtype="datetime64[ns]")

        for i in range(n_ens):
            microsecond = int(hundredth[i] * 10000)
            dt_str = (
                f"{year[i]:04d}-{month[i]:02d}-{day[i]:02d}T"
                f"{hour[i]:02d}:{minute[i]:02d}:{second[i]:02d}."
                f"{microsecond:06d}"
            )
            try:
                datetimes[i] = np.datetime64(dt_str, "ns")
            except Exception as ex:
                logger.warning(
                    f"Invalid datetime at ensemble {i}: {dt_str}. Error: {ex}"
                )
                datetimes[i] = np.datetime64("NaT")

        return datetimes

    except KeyError as e:
        raise KeyError(f"Missing required timestamp field: {e}")


def _compute_motion_sensors(ds: xr.Dataset) -> Dict[str, np.ndarray]:
    """
    Compute motion sensor values in degrees.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with raw Variable Leader data

    Returns
    -------
    dict
        Dictionary with 'heading_degrees', 'pitch_degrees', 'roll_degrees'
    """
    try:
        heading = ds["heading"].values.astype(np.float64) * 0.01
        pitch = ds["pitch"].values.astype(np.float64) * 0.01
        roll = ds["roll"].values.astype(np.float64) * 0.01

        return {
            "heading_degrees": heading,
            "pitch_degrees": pitch,
            "roll_degrees": roll,
        }
    except KeyError as e:
        raise KeyError(f"Motion sensor field not found: {e}")


def _determine_adcp_frequency(
    adcp_file: FilePathType,
    frequency: Optional[int] = None,
    byteskip: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    idarray: Optional[np.ndarray] = None,
    ensemble: int = 0,
) -> int:
    """
    Determine ADCP frequency for ADC channel scaling.

    Parameters
    ----------
    adcp_file : str or Path
        Path to ADCP binary file.
    frequency : int, optional
        User-provided frequency (unit = kHz). If provided, validated and used directly.
        Valid: 75, 150, 300, 600, 1200, 2400
    byteskip, offset, idarray, ensemble : optional
        Parameters for read_fixed_leader() if frequency needs auto-detection.

    Returns
    -------
    tuple[str, str]
        (frequency, source) where source is "user-provided" or "auto-detected"

    Raises
    ------
    ValueError
        If frequency cannot be determined or is invalid.
    """

    VALID_FREQUENCIES = [
        75,
        150,
        300,
        600,
        1200,
        2400,
    ]

    logger.debug(f"Determining frequency: input={frequency}")

    # Case 1: User provided explicit frequency
    if frequency is not None:
        if frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f"Invalid frequency: '{frequency}'. "
                f"Valid options: {VALID_FREQUENCIES}"
            )
        logger.info(f"Using user-provided frequency: {frequency}")
        return frequency

    # Case 2: Auto-read from FixedLeader
    logger.debug("Attempting to auto-read frequency from FixedLeader...")
    try:
        fl_ds = read_fixed_leader(
            adcp_file,
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble,
            json_file_path=None,
            include_decoded=True,
        )

        if "frequency" not in fl_ds.data_vars:
            raise KeyError("'frequency' field not found in FixedLeader")

        frequency = fl_ds["frequency"].values[0]

        if frequency not in VALID_FREQUENCIES:
            raise ValueError(
                f"Invalid frequency from FixedLeader: '{frequency}'. "
                f"Valid options: {VALID_FREQUENCIES}"
            )

        logger.info(f"Auto-detected frequency from FixedLeader: {frequency}")
        return frequency

    except Exception as e:
        raise ValueError(
            f"Could not auto-read frequency from FixedLeader. "
            f"Error: {type(e).__name__}: {e}. "
            f"Please provide frequency explicitly: "
            f"read_variable_leader(..., frequency='300 kHz')"
        ) from e


def _compute_adc_channels(
    ds: xr.Dataset, frequency: int, offset: float = -0.20
) -> Dict[str, np.ndarray]:
    """
    Compute physical ADC channel values.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with raw Variable Leader data
    frequency : int, optional
        User-provided frequency (unit = kHz). If provided, validated and used directly.
        Valid: 75, 150, 300, 600, 1200, 2400
    offset : float
        Temperature offset coefficient

    Returns
    -------
    dict
        Dictionary with 'xmit_voltage', 'xmit_current', 'ambient_temperature'
    """

    scale_list = {
        75: [2092719, 43838],
        150: [592157, 11451],
        300: [592157, 11451],
        600: [380667, 11451],
        1200: [253765, 11451],
        2400: [253765, 11451],
    }

    scale_factor = scale_list.get(frequency, [592157, 11451])

    try:
        adc0 = ds["adc_channel_0"].values.astype(np.float64)
        adc1 = ds["adc_channel_1"].values.astype(np.float64)
        adc2 = ds["adc_channel_2"].values.astype(np.float64)

        xmit_voltage = adc1 * (scale_factor[0] / 1000000.0)
        xmit_current = adc0 * (scale_factor[1] / 1000000.0)

        # Temperature polynomial (RDI standard)
        # NOTE: ADC channel 2 is stored as an 8-bit value (0-255) in the PD0 file format,
        # but the RDI polynomial expects 16-bit raw ADC count values (0-65535).
        # Scale the 8-bit ADC to equivalent 16-bit range.
        # Temperature polynomial (RDI standard)
        #
        adc2_16bit = adc2 * (65535.0 / 255.0)
        a0 = 9.82697464e1
        a1 = -5.86074151382e-3
        a2 = 1.60433886495e-7
        a3 = -2.32924716883e-12
        # ambient_temp = offset + ((a3 * adc2 + a2) * adc2 + a1) * adc2 + a0
        ambient_temp = (
            offset + ((a3 * adc2_16bit + a2) * adc2_16bit + a1) * adc2_16bit + a0
        )

        return {
            "xmit_voltage": xmit_voltage,
            "xmit_current": xmit_current,
            "ambient_temperature": ambient_temp,
        }
    except KeyError as e:
        raise KeyError(f"ADC channel field not found: {e}")


def _decode_bit_result(ds: xr.Dataset) -> Dict[str, np.ndarray]:
    """
    Decode BIT result bit fields.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with raw Variable Leader data

    Returns
    -------
    dict
        Dictionary with individual bit fields (8 fields)
    """
    bit_names = [
        "bit_reserved_1",
        "bit_reserved_2",
        "bit_reserved_3",
        "bit_demod_1_error",
        "bit_demod_0_error",
        "bit_reserved_4",
        "bit_timing_card_error",
        "bit_reserved_5",
    ]

    try:
        bit_data = ds["bit_result"].values.astype(np.uint16)
        result = {}

        for bit_pos, bit_name in enumerate(bit_names):
            # Extract bit at position bit_pos
            result[bit_name] = ((bit_data >> bit_pos) & 1).astype(np.uint8)

        return result
    except KeyError as e:
        raise KeyError(f"BIT result field not found: {e}")


def _decode_error_status_word(ds: xr.Dataset, esw_num: int) -> Dict[str, np.ndarray]:
    """
    Decode Error Status Word bit fields.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with raw Variable Leader data
    esw_num : int
        Error Status Word number (1-4)

    Returns
    -------
    dict
        Dictionary with 8 individual bit fields
    """
    esw_names = {
        1: [
            "esw1_bus_error_exception",
            "esw1_address_error_exception",
            "esw1_illegal_instruction_exception",
            "esw1_zero_divide_exception",
            "esw1_emulator_exception",
            "esw1_unassigned_exception",
            "esw1_watchdog_restart",
            "esw1_battery_saver_power",
        ],
        2: [
            "esw2_pinging",
            "esw2_not_used_1",
            "esw2_not_used_2",
            "esw2_not_used_3",
            "esw2_not_used_4",
            "esw2_not_used_5",
            "esw2_cold_wakeup_occurred",
            "esw2_unknown_wakeup_occurred",
        ],
        3: [
            "esw3_clock_read_error",
            "esw3_unexpected_alarm",
            "esw3_clock_jump_forward",
            "esw3_clock_jump_backward",
            "esw3_not_used_6",
            "esw3_not_used_7",
            "esw3_not_used_8",
            "esw3_not_used_9",
        ],
        4: [
            "esw4_not_used_10",
            "esw4_not_used_11",
            "esw4_not_used_12",
            "esw4_power_fail_unrecorded",
            "esw4_spurious_level_4_intr_dsp",
            "esw4_spurious_level_5_intr_uart",
            "esw4_spurious_level_6_intr_clock",
            "esw4_level_7_interrupt",
        ],
    }

    field_name = f"error_status_word_{esw_num}"

    try:
        esw_data = ds[field_name].values.astype(np.uint16)
        bit_names = esw_names[esw_num]
        result = {}

        # ESW values are only 8-bit per section, mask to 8 bits
        esw_data = (esw_data & 0xFF).astype(np.uint16)

        for bit_pos, bit_name in enumerate(bit_names):
            # Extract bit at position bit_pos
            result[bit_name] = ((esw_data >> bit_pos) & 1).astype(np.uint8)

        return result
    except KeyError as e:
        raise KeyError(f"Error Status Word {esw_num} field not found: {e}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def read_variable_leader(
    adcp_file: "FilePathType",
    byteskip: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    idarray: Optional[np.ndarray] = None,
    ensemble: int = 0,
    json_file_path: Optional[str] = None,
    include_decoded: bool = True,
    use_time_dim: bool = False,
    frequency: Optional[int] = None,
) -> xr.Dataset:
    """
    Load ADCP Variable Leader data into xarray.Dataset with raw, composite, and decoded variables.

    This function reads the 48 raw Variable Leader fields from PD0 format and optionally
    computes composite timestamps and 46 decoded fields. All variables are stored as
    individual xarray DataArrays with full CF Convention metadata.

    **Total Variables:**
    - If include_decoded=False: 48 raw fields
    - If include_decoded=True: 48 raw + 2 composite + 46 decoded = 96 variables

    **Decoded Fields (when include_decoded=True):**
    - Motion sensors (3): heading_degrees, pitch_degrees, roll_degrees (in degrees)
    - Composite timestamp (1): rtc_datetime (datetime64[ns])
    - ADC physical values (3): xmit_voltage, xmit_current, ambient_temperature
    - BIT result bits (8): bit_reserved_1-5, bit_demod_0/1_error, bit_timing_card_error
    - Error Status Word bits (32): esw1/2/3/4 with 8 bit fields each

    Parameters
    ----------
    adcp_file : str or Path
        Path to ADCP binary file in PD0 format.
    byteskip : np.ndarray, optional
        From read_header(). If None, auto-fetch from fileheader.
    offset : np.ndarray, optional
        From read_header(). If None, auto-fetch from fileheader.
    idarray : np.ndarray, optional
        From read_header(). If None, auto-fetch from fileheader.
    ensemble : int, optional
        Number of ensembles. If 0, auto-fetch from fileheader. Default is 0.
    json_file_path : str, optional
        Path to custom metadata JSON file. If None, uses package default.
    include_decoded : bool, optional
        If True (default), compute and include 46 decoded fields.
        If False, return only 48 raw fields (faster).
    use_time_dim : bool, optional
        If True, use time coordinate as dimension instead of ensemble.
        Default is False. Requires include_decoded=True if timestamps needed.
    frequency : int, optional
        ADCP frequency (unit = kHz) for ADC channel scaling. Valid options are:
        75, 150, 300, 600, 1200, 2400. If None, attempts to read from the
        read_fixed_leader component. This is the recommended approach
        for most users as it reads the actual ADCP frequency from the file.
        Default is None (auto-detects from the fixed leader data of the ADCP binary file).

    Returns
    -------
    xr.Dataset
        xarray Dataset with Variable Leader variables.

        Coordinates:
        - ensemble: Ensemble index (0 to n_ensembles-1)
        - time (optional): Composite timestamp if use_time_dim=True and include_decoded=True

        Variables:
        - 48 raw fields (always)
        - 2 composite timestamp fields (if include_decoded=True)
        - 46 decoded fields (if include_decoded=True)

        Attributes (dataset-level):
        - filename: Input file name
        - total_ensembles: Number of ensembles
        - num_fields: Total number of variables
        - error_message: Status from file read
        - pyadps_component: "VariableLeader"
        - pyadps_version: Package version (from pyproject.toml)
        - adcp_data_format: "PD0"

    Raises
    ------
    FileNotFoundError
        If adcp_file does not exist or metadata file cannot be found.
    ValueError
        If file is not a valid ADCP file or data cannot be read.
    json.JSONDecodeError
        If metadata file is malformed.

    Examples
    --------
    Basic usage (all variables with decoded fields):

    >>> import pyadps
    >>> vl_ds = pyadps.read_variable_leader('test.000')
    >>> print(len(vl_ds.data_vars))  # 96 variables
    >>> heading = vl_ds['heading_degrees']  # In degrees, not raw units
    >>> temp = vl_ds['ambient_temperature']  # In Â°C

    Raw fields only (faster):

    >>> vl_ds = pyadps.read_variable_leader('test.000', include_decoded=False)
    >>> print(len(vl_ds.data_vars))  # 48 variables

    With time dimension for plotting:

    >>> vl_ds = pyadps.read_variable_leader('test.000', use_time_dim=True)
    >>> print(vl_ds.dims)  # time dimension available
    >>> vl_ds.heading_degrees.plot()  # xarray knows how to handle time axis

    Notes
    -----
    Variable Leader data contains dynamic ADCP measurements varying per ensemble:
    - Timestamps (RTC or Y2K fields)
    - Motion sensors (heading, pitch, roll)
    - Environmental data (temperature, salinity, pressure, depth)
    - Diagnostic data (ADC channels, error status, BIT results)
    - Water properties (speed of sound)

    When include_decoded=True (default), computed fields are:
    - Angles converted from 0.01Â° units to degrees
    - ADC channels scaled to physical units (V, A, Â°C)
    - Timestamps composed into datetime64[ns]
    - Bit fields extracted and decoded (0 or 1 values)

    For basic users using ReadFile class, this function is called internally
    with include_decoded=True and use_time_dim=True by default.

    For advanced users using isolated functions, see:
    - read_header() for file metadata
    - read_fixed_leader() for configuration data
    - Time alignment helper in filereader module
    """

    filename = str(adcp_file)

    # ====================================================================
    # Auto-fetch header parameters if not provided
    # ====================================================================
    if byteskip is None or offset is None or idarray is None or ensemble == 0:
        logger.debug(f"Auto-fetching header info from {filename}")
        header_ds = read_header(filename)
        if byteskip is None:
            byteskip = header_ds.byte_skip.values
        if offset is None:
            offset = header_ds.address_offset.values
        if idarray is None:
            idarray = header_ds.data_id.values
        if ensemble == 0:
            ensemble = header_ds.attrs["total_ensembles"]

    # ====================================================================
    # Extract Variable Leader data using pd0_parser
    # ====================================================================
    logger.debug(f"Extracting Variable Leader data from {filename}")
    vl_data, ensembles, error_code = pd0_parser.variableleader(
        filename,
        byteskip=byteskip,
        offset=offset,
        idarray=idarray,
        ensemble=ensemble,
    )

    error_message = pd0_parser.ErrorCode.get_message(error_code)
    if error_code != 0:
        logger.warning(f"Variable Leader read error: {error_message}")

    # ====================================================================
    # Load metadata
    # ====================================================================
    logger.debug(f"Loading metadata with json_file_path={json_file_path}")
    metadata_pkg = _load_variable_leader_metadata(json_file_path=json_file_path)
    raw_fields = metadata_pkg["raw_fields"]
    composite_fields = metadata_pkg["composite_fields"]
    decoded_fields = metadata_pkg["decoded_fields"]

    # ====================================================================
    # Build xarray.Dataset with raw fields
    # ====================================================================

    n_fields = vl_data.shape[0]  # Should be 48
    n_ensembles = vl_data.shape[1] if len(vl_data.shape) > 1 else 1

    if len(vl_data.shape) == 1:
        vl_data = vl_data.reshape(-1, 1)

    logger.debug(f"Variable Leader data shape: {vl_data.shape}")

    # Coordinates
    coords = {
        "ensemble": np.arange(n_ensembles),
    }

    # Data variables - raw fields only
    data_vars = {}

    for field_idx, field_meta in raw_fields.items():
        if field_meta.get("is_raw", False):
            var_name = field_meta["name"]
            field_index = field_meta["index"]
            var_data = vl_data[field_index, :]
            data_vars[var_name] = ("ensemble", var_data.astype(np.int64))

    # Create base dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "filename": Path(adcp_file).name,
            "total_ensembles": int(n_ensembles),
            "num_fields": int(n_fields),
            "error_message": str(error_message),
            "pyadps_component": "VariableLeader",
            "pyadps_version": PYADPS_VERSION,
            "adcp_data_format": ADCP_DATA_FORMAT,
        },
    )

    # Attributes for coordinate variables
    ensemble_attrs = {
        "axis": "T",  # T for Time
        "long_name": "Ensemble number",
        "standard_name": "time_counter",  # If using time
    }

    # Apply coordinate attributes
    ds.coords["ensemble"].attrs.update(ensemble_attrs)

    # ====================================================================
    # Apply variable-level attributes for raw fields
    # ====================================================================
    for field_idx, field_meta in raw_fields.items():
        if field_meta.get("is_raw", False):
            var_name = field_meta["name"]
            if var_name in ds.data_vars:
                var_attrs = _build_variable_attributes(field_meta)
                ds[var_name].attrs.update(var_attrs)

    # ====================================================================
    # Add composite timestamp fields
    # ====================================================================
    if include_decoded or use_time_dim:
        logger.debug("Computing composite timestamp fields")

        # RTC timestamp (always compute if needed)
        rtc_times = _compute_composite_timestamp(ds, timestamp_type="rtc")
        ds["rtc_datetime"] = ("ensemble", rtc_times)
        ds["rtc_datetime"].attrs.update(
            _build_variable_attributes(composite_fields.get("rtc_datetime", {}))
        )

        # Y2K timestamp (if Y2K fields exist)
        if "y2k_year" in ds.data_vars:
            y2k_times = _compute_composite_timestamp(ds, timestamp_type="y2k")
            ds["y2k_datetime"] = ("ensemble", y2k_times)
            ds["y2k_datetime"].attrs.update(
                _build_variable_attributes(composite_fields.get("y2k_datetime", {}))
            )

    # ====================================================================
    # Add decoded fields
    # ====================================================================
    if include_decoded:
        logger.debug("Computing decoded fields")

        # Motion sensors (convert to degrees)
        try:
            motion = _compute_motion_sensors(ds)
            # Build a reverse lookup: field_name -> metadata
            decoded_by_name = {
                meta.get("name"): meta
                for meta in decoded_fields.values()
                if meta.get("name")
            }
            for field_name, data in motion.items():
                ds[field_name] = ("ensemble", data)
                if field_name in decoded_by_name:
                    ds[field_name].attrs.update(
                        _build_variable_attributes(decoded_by_name[field_name])
                    )
        except KeyError as e:
            logger.warning(f"Could not compute motion sensors: {e}")

        # ADC channels (compute physical values)
        # Try to get frequency from FixedLeader if available
        # Determine ADCP frequency
        frequency = _determine_adcp_frequency(
            adcp_file,
            frequency=frequency,
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble,
        )
        logger.debug(f"Using frequency: {frequency}")

        try:
            adc = _compute_adc_channels(ds, frequency=frequency)
            for field_name, data in adc.items():
                ds[field_name] = ("ensemble", data)
                for idx, meta in decoded_fields.items():
                    if meta.get("name") == field_name:
                        ds[field_name].attrs.update(_build_variable_attributes(meta))
                        break
        except KeyError as e:
            logger.warning(f"Could not compute ADC channels: {e}")

        # BIT result bits
        try:
            bit_bits = _decode_bit_result(ds)
            for field_name, data in bit_bits.items():
                ds[field_name] = ("ensemble", data)
                for idx, meta in decoded_fields.items():
                    if meta.get("name") == field_name:
                        ds[field_name].attrs.update(_build_variable_attributes(meta))
                        break
        except KeyError as e:
            logger.warning(f"Could not decode BIT result: {e}")

        # Error Status Words (1-4)
        for esw_num in [1, 2, 3, 4]:
            try:
                esw_bits = _decode_error_status_word(ds, esw_num)
                for field_name, data in esw_bits.items():
                    ds[field_name] = ("ensemble", data)
                    for idx, meta in decoded_fields.items():
                        if meta.get("name") == field_name:
                            ds[field_name].attrs.update(
                                _build_variable_attributes(meta)
                            )
                            break
            except KeyError as e:
                logger.warning(f"Could not decode ESW{esw_num}: {e}")

    # ====================================================================
    # Add time coordinate if requested
    # ====================================================================
    if use_time_dim:
        # rtc_datetime is always present here because line 1785 ensures it's created
        # whenever use_time_dim=True (condition: include_decoded or use_time_dim)
        ds = ds.assign_coords(time=("ensemble", ds["rtc_datetime"].values))

        # Swap dimension
        ds = ds.swap_dims({"ensemble": "time"})

    # ====================================================================
    # Log and return
    # ====================================================================
    num_vars = len(ds.data_vars)
    logger.info(
        f"Successfully loaded Variable Leader data from {filename}: "
        f"{n_ensembles} ensembles Ã— {num_vars} variables"
    )

    return ds


# ============================================================================
# VELOCITY DATA READER
# ============================================================================


def read_velocity(
    adcp_file: FilePathType,
    cell: int = 0,
    beam: int = 0,
    byteskip: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    idarray: Optional[np.ndarray] = None,
    ensemble: int = 0,
    missing_as_nan: bool = True,
) -> xr.Dataset:
    """
    Load ADCP velocity data into xarray.Dataset.

    This function reads velocity data from RDI ADCP binary files and returns
    it as an xarray.Dataset with proper dimensions (beam, cell, ensemble) and
    CF Convention-compliant attributes.

    The velocity data represents the water current velocity measured at each
    cell and beam combination within the ADCP. Velocity is typically measured
    in millimeters per second (mm/s).

    Parameters
    ----------
    adcp_file : str or Path
        Path to ADCP binary file in PD0 format.
    cell : int, optional
        Cell number to extract data from. Default is 0 (all cells).
        If 0, all cells are included; otherwise, only the specified cell.
    beam : int, optional
        Beam number to extract data from. Default is 0 (all beams).
        If 0, all beams are included; otherwise, only the specified beam.
    byteskip : np.ndarray, optional
        From read_header(). If None, auto-fetch from fileheader.
        Array of bytes to skip for each ensemble.
    offset : np.ndarray, optional
        From read_header(). If None, auto-fetch from fileheader.
        Array of offsets for each ensemble.
    idarray : np.ndarray, optional
        From read_header(). If None, auto-fetch from fileheader.
        Array of data type IDs for each ensemble.
    ensemble : int, optional
        Number of ensembles. If 0, auto-fetch from fileheader. Default is 0.
    missing_as_nan : bool, optional
        If True (default), replace the RDI missing value sentinel (-32768) with
        np.nan and store velocity as float32. Xarray aggregations (.mean(), .std(),
        etc.) then skip missing cells automatically.
        If False, retain the raw int16 data with -32768 as the sentinel value
        (legacy behaviour). Use False only when memory is critical or you need to
        preserve the exact binary representation.

    Returns
    -------
    xr.Dataset
        xarray Dataset with velocity data. Variables include:
        - velocity : (beam, cell, ensemble) float32 or int16
            Water current velocity in mm/s.
            float32 with NaN when missing_as_nan=True (default).
            int16 with -32768 sentinel when missing_as_nan=False.

        Dataset attributes:
        - filename : str
            Name of the source file
        - total_ensembles : int
            Total number of ensembles in dataset
        - num_cells : int
            Number of cells in the data
        - num_beams : int
            Number of beams in the data
        - error_message : str
            Error/status message from file reading
        - pyadps_component : str
            "Velocity"
        - pyadps_version : str
            Package version (from pyproject.toml)
        - adcp_data_format : str
            "PD0"

    Raises
    ------
    FileNotFoundError
        If adcp_file does not exist.
    ValueError
        If file is not a valid ADCP file or data cannot be read.

    Examples
    --------
    >>> import pyadps
    >>> ds = pyadps.read_velocity('test.000')
    >>> print(ds)  # xarray.Dataset with velocity data
    >>> velocities = ds['velocity'].values  # (beams, cells, ensembles)
    >>> print(f"Shape: {ds['velocity'].shape}")

    Extract specific cell and beam:

    >>> ds_cell0 = pyadps.read_velocity('test.000', cell=1, beam=1)
    >>> print(ds_cell0['velocity'].values)

    For better performance with large files, pre-fetch header information:

    >>> header_ds = pyadps.read_header('test.000')
    >>> vel_ds = pyadps.read_velocity(
    ...     'test.000',
    ...     byteskip=header_ds.byte_skip.values,
    ...     offset=header_ds.address_offset.values,
    ...     idarray=header_ds.data_id.values,
    ...     ensemble=header_ds.attrs['total_ensembles']
    ... )

    Notes
    -----
    Velocity data extracted using pd0_parser.datatype() with datatype="velocity".
    By default (missing_as_nan=True) missing values (-32768) are replaced with
    np.nan and the array is stored as float32. Set missing_as_nan=False to retain
    the raw int16 data with -32768 as the sentinel.

    The velocity data has dimensions (beam, cell, ensemble) where:
    - cell: Depth cells (typically 0 to ~100+)
    - beam: Beams in the ADCP (typically 0 to 3 for 4-beam systems)
    - ensemble: Time steps (recordings)

    Units are millimeters per second (mm/s) per CF Convention.

    See Also
    --------
    read_correlation : Load correlation magnitude data
    read_echo : Load echo intensity data
    read_percent_good : Load percent good data
    read_header : Load file header metadata
    """

    filename = str(adcp_file)

    # If optional parameters not provided, fetch from fileheader
    if byteskip is None or offset is None or idarray is None or ensemble == 0:
        logger.debug(f"Auto-fetching header info from {filename}")
        header_ds = read_header(filename)
        if byteskip is None:
            byteskip = header_ds.byte_skip.values
        if offset is None:
            offset = header_ds.address_offset.values
        if idarray is None:
            idarray = header_ds.data_id.values
        if ensemble == 0:
            ensemble = header_ds.attrs["total_ensembles"]

    # Extract velocity data using pd0_parser
    logger.debug(f"Extracting velocity data: cell={cell}, beam={beam}")
    data, ens, cells, beams, error_code = pd0_parser.datatype(
        filename,
        "velocity",
        cell=cell,
        beam=beam,
        byteskip=byteskip,
        offset=offset,
        idarray=idarray,
        ensemble=ensemble,
    )

    # Get error message
    error_message = pd0_parser.ErrorCode.get_message(error_code)

    # ========================================================================
    # Determine data structure
    # ========================================================================
    # data shape: (num_cells, num_beams, num_ensembles) for full data
    # or (num_ensembles,) for single cell/beam extraction

    if len(data.shape) == 3:
        # Full 3D data: (beams, cells, ensembles)
        n_beams = data.shape[0]
        n_cells = data.shape[1]
        n_ensembles = data.shape[2]
    else:
        raise ValueError(
            f"Unexpected data shape from pd0_parser.datatype(): {data.shape}. "
            f"Expected 3D (full data), "
            f"got {len(data.shape)}D. This indicates a bug in reading binary data."
        )

    # ========================================================================
    # Build xarray.Dataset
    # ========================================================================

    # Coordinates
    coords = {
        "beam": np.arange(n_beams),
        "cell": np.arange(n_cells),
        "ensemble": np.arange(n_ensembles),
    }

    # Data variables — dtype and missing-value handling depends on missing_as_nan
    if missing_as_nan:
        vel_data = data.astype(np.float32)
        vel_data[vel_data <= VELOCITY_MISSING_VALUE] = np.nan
    else:
        vel_data = data.astype(np.int16)

    data_vars = {
        "velocity": (("beam", "cell", "ensemble"), vel_data),
    }

    # Attributes for coordinate variables
    ensemble_attrs = {
        "axis": "T",  # T for Time
        "long_name": "Ensemble number",
        "standard_name": "time_counter",  # If using time
    }
    cell_attrs = {
        "axis": "Z",  # Z for vertical/depth
        "long_name": "Depth cell number",
        "positive": "up",  # Positive upward
    }
    beam_attrs = {
        "axis": "X",  # X for cross-beam
        "long_name": "Beam number",
    }
    # Attributes for velocity variable (CF Convention)
    if missing_as_nan:
        velocity_attrs = {
            "long_name": "Water current velocity",
            "units": "mm s-1",
            "valid_min": -32767,
            "valid_max": 32767,
            "scale_factor": 1.0,
            "add_offset": 0.0,
            "description": "Velocity magnitude measured by ADCP beams",
            "source": "RDI WorkHorse ADCP",
            "comments": "Negative values indicate flow direction opposite to beam direction",
            "missing_value_handling": "nan",
        }
    else:
        velocity_attrs = {
            "long_name": "Water current velocity",
            "units": "mm s-1",
            "valid_min": -32768,
            "valid_max": 32767,
            "missing_value": -32768,
            "_FillValue": -32768,
            "scale_factor": 1.0,
            "add_offset": 0.0,
            "description": "Velocity magnitude measured by ADCP beams",
            "source": "RDI WorkHorse ADCP",
            "comments": "Negative values indicate flow direction opposite to beam direction",
            "missing_value_handling": "sentinel",
        }

    # Dataset-level attributes
    ds_attrs = {
        "filename": Path(adcp_file).name,
        "total_ensembles": int(n_ensembles),
        "num_cells": int(n_cells),
        "num_beams": int(n_beams),
        "error_message": str(error_message),
        "pyadps_component": "Velocity",
        "pyadps_version": PYADPS_VERSION,
        "adcp_data_format": ADCP_DATA_FORMAT,
    }

    # Create xarray.Dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=ds_attrs,
    )

    # Apply attributes
    ds.coords["ensemble"].attrs.update(ensemble_attrs)
    ds.coords["cell"].attrs.update(cell_attrs)
    ds.coords["beam"].attrs.update(beam_attrs)
    ds["velocity"].attrs.update(velocity_attrs)

    # Log and return
    logger.info(
        f"Successfully loaded velocity data from {filename}: "
        f"{n_cells} cells Ã— {n_beams} beams Ã— {n_ensembles} ensembles"
    )

    return ds


def read_correlation(
    adcp_file: FilePathType,
    cell: int = 0,
    beam: int = 0,
    byteskip: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    idarray: Optional[np.ndarray] = None,
    ensemble: int = 0,
) -> xr.Dataset:
    """
    Load ADCP correlation magnitude data into xarray.Dataset.

    Correlation is a quality indicator for velocity measurement - higher values
    indicate better data quality (0-255 scale).
    """
    filename = str(adcp_file)

    if byteskip is None or offset is None or idarray is None or ensemble == 0:
        logger.debug(f"Auto-fetching header info from {filename}")
        header_ds = read_header(filename)
        if byteskip is None:
            byteskip = header_ds.byte_skip.values
        if offset is None:
            offset = header_ds.address_offset.values
        if idarray is None:
            idarray = header_ds.data_id.values
        if ensemble == 0:
            ensemble = header_ds.attrs["total_ensembles"]

    logger.debug(f"Extracting correlation data: cell={cell}, beam={beam}")
    data, ens, cells, beams, error_code = pd0_parser.datatype(
        filename,
        "correlation",
        cell=cell,
        beam=beam,
        byteskip=byteskip,
        offset=offset,
        idarray=idarray,
        ensemble=ensemble,
    )

    error_message = pd0_parser.ErrorCode.get_message(error_code)

    if len(data.shape) == 3:
        n_beams, n_cells, n_ensembles = data.shape[0], data.shape[1], data.shape[2]
    else:
        raise ValueError(
            f"Unexpected data shape from pd0_parser.datatype(): {data.shape}. "
            f"Expected 3D (full data), "
            f"got {len(data.shape)}D. This indicates a bug in reading binary data."
        )

    coords = {
        "beam": np.arange(n_beams),
        "cell": np.arange(n_cells),
        "ensemble": np.arange(n_ensembles),
    }
    data_vars = {"correlation": (("beam", "cell", "ensemble"), data.astype(np.uint8))}
    # Attributes for coordinate variables
    ensemble_attrs = {
        "axis": "T",  # T for Time
        "long_name": "Ensemble number",
        "standard_name": "time_counter",  # If using time
    }
    cell_attrs = {
        "axis": "Z",  # Z for vertical/depth
        "long_name": "Depth cell number",
        "positive": "up",  # Positive upward
    }
    beam_attrs = {
        "axis": "X",  # X for cross-beam
        "long_name": "Beam number",
    }
    correlation_attrs = {
        "long_name": "Correlation Magnitude",
        "units": "dimensionless",
        "valid_min": 0,
        "valid_max": 255,
        "description": "Correlation magnitude of ADCP velocity estimate",
        "comments": "Higher values indicate better velocity data quality",
    }
    ds_attrs = {
        "filename": Path(adcp_file).name,
        "total_ensembles": int(n_ensembles),
        "num_cells": int(n_cells),
        "num_beams": int(n_beams),
        "error_message": str(error_message),
        "pyadps_component": "Correlation",
        "pyadps_version": PYADPS_VERSION,
        "adcp_data_format": ADCP_DATA_FORMAT,
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds_attrs)
    # Apply attributes
    ds.coords["ensemble"].attrs.update(ensemble_attrs)
    ds.coords["cell"].attrs.update(cell_attrs)
    ds.coords["beam"].attrs.update(beam_attrs)
    ds["correlation"].attrs.update(correlation_attrs)

    logger.info(
        f"Successfully loaded correlation data: {n_cells}Ã—{n_beams}Ã—{n_ensembles}"
    )
    return ds


def read_echo_intensity(
    adcp_file: FilePathType,
    cell: int = 0,
    beam: int = 0,
    byteskip: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    idarray: Optional[np.ndarray] = None,
    ensemble: int = 0,
) -> xr.Dataset:
    """
    Load ADCP echo intensity data into xarray.Dataset.

    Echo intensity (0-255) represents acoustic backscatter strength. Multiply
    by scale_factor (0.45) to convert to dB.
    """
    filename = str(adcp_file)

    if byteskip is None or offset is None or idarray is None or ensemble == 0:
        header_ds = read_header(filename)
        if byteskip is None:
            byteskip = header_ds.byte_skip.values
        if offset is None:
            offset = header_ds.address_offset.values
        if idarray is None:
            idarray = header_ds.data_id.values
        if ensemble == 0:
            ensemble = header_ds.attrs["total_ensembles"]

    data, ens, cells, beams, error_code = pd0_parser.datatype(
        filename,
        "echo",
        cell=cell,
        beam=beam,
        byteskip=byteskip,
        offset=offset,
        idarray=idarray,
        ensemble=ensemble,
    )

    error_message = pd0_parser.ErrorCode.get_message(error_code)

    if len(data.shape) == 3:
        n_beams, n_cells, n_ensembles = data.shape[0], data.shape[1], data.shape[2]
    else:
        raise ValueError(
            f"Unexpected data shape from pd0_parser.datatype(): {data.shape}. "
            f"Expected 3D (full data), "
            f"got {len(data.shape)}D. This indicates a bug in reading binary data."
        )

    coords = {
        "beam": np.arange(n_beams),
        "cell": np.arange(n_cells),
        "ensemble": np.arange(n_ensembles),
    }
    data_vars = {
        "echo_intensity": (("beam", "cell", "ensemble"), data.astype(np.uint8))
    }
    # Attributes for coordinate variables
    ensemble_attrs = {
        "axis": "T",  # T for Time
        "long_name": "Ensemble number",
        "standard_name": "time_counter",  # If using time
    }
    cell_attrs = {
        "axis": "Z",  # Z for vertical/depth
        "long_name": "Depth cell number",
        "positive": "up",  # Positive upward
    }
    beam_attrs = {
        "axis": "X",  # X for cross-beam
        "long_name": "Beam number",
    }
    echo_attrs = {
        "long_name": "Echo Intensity",
        "units": "counts",
        "valid_min": 0,
        "valid_max": 255,
        "scale_factor": 0.45,
        "description": "Acoustic signal strength (backscatter) from water particles",
        "comments": "Multiply by scale_factor (0.45) to convert to dB",
    }
    ds_attrs = {
        "filename": Path(adcp_file).name,
        "total_ensembles": int(n_ensembles),
        "num_cells": int(n_cells),
        "num_beams": int(n_beams),
        "error_message": str(error_message),
        "pyadps_component": "Echo",
        "pyadps_version": PYADPS_VERSION,
        "adcp_data_format": ADCP_DATA_FORMAT,
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds_attrs)
    # Apply attributes
    ds.coords["ensemble"].attrs.update(ensemble_attrs)
    ds.coords["cell"].attrs.update(cell_attrs)
    ds.coords["beam"].attrs.update(beam_attrs)
    ds["echo_intensity"].attrs.update(echo_attrs)

    logger.info(
        f"Successfully loaded echo intensity data: {n_cells}Ã—{n_beams}Ã—{n_ensembles}"
    )
    return ds


def read_percent_good(
    adcp_file: FilePathType,
    cell: int = 0,
    beam: int = 0,
    byteskip: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    idarray: Optional[np.ndarray] = None,
    ensemble: int = 0,
) -> xr.Dataset:
    """
    Load ADCP percent good data into xarray.Dataset.

    Percent good (0-100) indicates the percentage of valid velocity data.
    Higher values indicate better data quality.
    """
    filename = str(adcp_file)

    if byteskip is None or offset is None or idarray is None or ensemble == 0:
        header_ds = read_header(filename)
        if byteskip is None:
            byteskip = header_ds.byte_skip.values
        if offset is None:
            offset = header_ds.address_offset.values
        if idarray is None:
            idarray = header_ds.data_id.values
        if ensemble == 0:
            ensemble = header_ds.attrs["total_ensembles"]

    data, ens, cells, beams, error_code = pd0_parser.datatype(
        filename,
        "percent good",
        cell=cell,
        beam=beam,
        byteskip=byteskip,
        offset=offset,
        idarray=idarray,
        ensemble=ensemble,
    )

    error_message = pd0_parser.ErrorCode.get_message(error_code)

    if len(data.shape) == 3:
        n_beams, n_cells, n_ensembles = data.shape[0], data.shape[1], data.shape[2]
    else:
        raise ValueError(
            f"Unexpected data shape from pd0_parser.datatype(): {data.shape}. "
            f"Expected 3D (full data), "
            f"got {len(data.shape)}D. This indicates a bug in reading binary data."
        )

    coords = {
        "beam": np.arange(n_beams),
        "cell": np.arange(n_cells),
        "ensemble": np.arange(n_ensembles),
    }
    data_vars = {"percent_good": (("beam", "cell", "ensemble"), data.astype(np.uint8))}
    # Attributes for coordinate variables
    ensemble_attrs = {
        "axis": "T",  # T for Time
        "long_name": "Ensemble number",
        "standard_name": "time_counter",  # If using time
    }
    cell_attrs = {
        "axis": "Z",  # Z for vertical/depth
        "long_name": "Depth cell number",
        "positive": "up",  # Positive upward
    }
    beam_attrs = {
        "axis": "X",  # X for cross-beam
        "long_name": "Beam number",
    }
    pg_attrs = {
        "long_name": "Percent Good",
        "units": "percent",
        "valid_min": 0,
        "valid_max": 100,
        "description": "Percentage of valid velocity data in the ensemble",
        "comments": "Quality indicator. Higher values indicate better data quality",
    }
    ds_attrs = {
        "filename": Path(adcp_file).name,
        "total_ensembles": int(n_ensembles),
        "num_cells": int(n_cells),
        "num_beams": int(n_beams),
        "error_message": str(error_message),
        "pyadps_component": "PercentGood",
        "pyadps_version": PYADPS_VERSION,
        "adcp_data_format": ADCP_DATA_FORMAT,
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds_attrs)
    # Apply attributes
    ds.coords["ensemble"].attrs.update(ensemble_attrs)
    ds.coords["cell"].attrs.update(cell_attrs)
    ds.coords["beam"].attrs.update(beam_attrs)
    ds["percent_good"].attrs.update(pg_attrs)

    logger.info(
        f"Successfully loaded percent good data: {n_cells}Ã—{n_beams}Ã—{n_ensembles}"
    )
    return ds


def read_status(
    adcp_file: FilePathType,
    cell: int = 0,
    beam: int = 0,
    byteskip: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    idarray: Optional[np.ndarray] = None,
    ensemble: int = 0,
) -> xr.Dataset:
    """
    Load ADCP status data into xarray.Dataset.

    Status contains diagnostic and quality flags for each measurement cell.
    """
    filename = str(adcp_file)

    if byteskip is None or offset is None or idarray is None or ensemble == 0:
        header_ds = read_header(filename)
        if byteskip is None:
            byteskip = header_ds.byte_skip.values
        if offset is None:
            offset = header_ds.address_offset.values
        if idarray is None:
            idarray = header_ds.data_id.values
        if ensemble == 0:
            ensemble = header_ds.attrs["total_ensembles"]

    data, ens, cells, beams, error_code = pd0_parser.datatype(
        filename,
        "status",
        cell=cell,
        beam=beam,
        byteskip=byteskip,
        offset=offset,
        idarray=idarray,
        ensemble=ensemble,
    )

    error_message = pd0_parser.ErrorCode.get_message(error_code)

    if len(data.shape) == 3:
        n_beams, n_cells, n_ensembles = data.shape[0], data.shape[1], data.shape[2]
    else:
        raise ValueError(
            f"Unexpected data shape from pd0_parser.datatype(): {data.shape}. "
            f"Expected 3D (full data), "
            f"got {len(data.shape)}D. This indicates a bug in reading binary data."
        )

    coords = {
        "beam": np.arange(n_beams),
        "cell": np.arange(n_cells),
        "ensemble": np.arange(n_ensembles),
    }
    data_vars = {"status": (("beam", "cell", "ensemble"), data.astype(np.uint8))}
    # Attributes for coordinate variables
    ensemble_attrs = {
        "axis": "T",  # T for Time
        "long_name": "Ensemble number",
        "standard_name": "time_counter",  # If using time
    }
    cell_attrs = {
        "axis": "Z",  # Z for vertical/depth
        "long_name": "Depth cell number",
        "positive": "up",  # Positive upward
    }
    beam_attrs = {
        "axis": "X",  # X for cross-beam
        "long_name": "Beam number",
    }
    status_attrs = {
        "long_name": "Status Code",
        "units": "dimensionless",
        "valid_min": 0,
        "valid_max": 255,
        "description": "Status flags encoding diagnostic information",
        "comments": "Bit flags indicate various diagnostic and quality conditions",
    }
    ds_attrs = {
        "filename": Path(adcp_file).name,
        "total_ensembles": int(n_ensembles),
        "num_cells": int(n_cells),
        "num_beams": int(n_beams),
        "error_message": str(error_message),
        "pyadps_component": "Status",
        "pyadps_version": PYADPS_VERSION,
        "adcp_data_format": ADCP_DATA_FORMAT,
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=ds_attrs)
    # Apply attributes
    ds.coords["ensemble"].attrs.update(ensemble_attrs)
    ds.coords["cell"].attrs.update(cell_attrs)
    ds.coords["beam"].attrs.update(beam_attrs)
    ds["status"].attrs.update(status_attrs)

    logger.info(f"Successfully loaded status data: {n_cells}Ã—{n_beams}Ã—{n_ensembles}")
    return ds


# ============================================================================
# ORCHESTRATION FUNCTION: read (v1.0.0 PRIMARY ENTRY POINT)
# ============================================================================


def _create_velocity_mask(ds_velocity: xr.Dataset) -> xr.Dataset:
    """
    Create a 3D baseline mask from velocity data based on missing values.

    This function generates a quality mask where values are marked as invalid (1)
    or valid (0) based on the RDI missing value code (-32768) in the velocity data.

    The mask has 4 beams corresponding to U, V, W velocity components and a
    combined signal quality mask:
    - mask[0, :, :] = U velocity failures
    - mask[1, :, :] = V velocity failures
    - mask[2, :, :] = W velocity failures
    - mask[3, :, :] = Combined mask (OR of U, V, W)

    The 4th beam is NOT based on error velocity (beam 3 of input data).
    Instead, it represents the combined mask of the first three velocity
    components, marking a cell as invalid if ANY of u, v, or w is missing.
    This follows the principle that error velocity is a diagnostic value,
    not a physical velocity component.

    Parameters
    ----------
    ds_velocity : xr.Dataset
        Velocity dataset from read_velocity() with:
        - velocity : DataArray with dims (beam, cell, ensemble)
          Shape is typically (4, n_cells, n_ensembles) where:
          - beam 0 = U (east-west) velocity
          - beam 1 = V (north-south) velocity
          - beam 2 = W (vertical) velocity
          - beam 3 = Error velocity (diagnostic)

    Returns
    -------
    xr.Dataset
        Dataset containing 3D mask variable with dims (beam, cell, ensemble):
        - 1 = invalid data (missing value detected)
        - 0 = valid data
        - beam dimension represents [u_mask, v_mask, w_mask, signal_quality]

        Variables:
        - mask : (beam, cell, ensemble) int8

        Variable attributes include:
        - description: Explanation of the mask
        - missing_value_code: The value used to identify missing data
        - beam_0: "U velocity component mask"
        - beam_1: "V velocity component mask"
        - beam_2: "W velocity component mask"
        - beam_3: "Combined signal quality mask (U OR V OR W)"

    Notes
    -----
    The baseline mask only identifies data flagged as missing by the ADCP
    instrument. Additional quality control checks (correlation, echo intensity,
    percent good, etc.) should be applied separately during processing.

    The 4th beam of the mask (signal quality) is the logical OR of the first
    three components. This means if ANY physical velocity component is missing
    for a given cell and ensemble, the signal quality mask will also be 1.

    Examples
    --------
    >>> ds_vel = read_velocity('file.000')
    >>> ds_mask = _create_velocity_mask(ds_vel)
    >>> print(ds_mask['mask'].shape)  # (4, n_cells, n_ensembles)
    >>> # Count invalid U velocities
    >>> u_invalid_count = (ds_mask['mask'].sel(beam=0) == 1).sum().item()
    >>> # Find cells where any velocity is missing
    >>> any_missing = (ds_mask['mask'].sel(beam=3) == 1)

    See Also
    --------
    read_velocity : Load velocity data from ADCP file
    read : Main function that calls this helper
    """
    velocity = ds_velocity["velocity"]

    # Get dimensions
    n_beams = velocity.sizes["beam"]
    n_cells = velocity.sizes["cell"]
    n_ensembles = velocity.sizes["ensemble"]

    # Validate expected shape
    if n_beams < 4:
        logger.warning(
            f"Expected 4 beams in velocity data, got {n_beams}. "
            "Mask will be created for available beams only."
        )

    # Create mask array (same shape as velocity)
    # 1 = invalid (missing), 0 = valid
    mask_data = np.zeros((4, n_cells, n_ensembles), dtype=np.int8)

    # Extract velocity values
    vel_values = velocity.values  # Shape: (beam, cell, ensemble)

    # Create masks for U, V, W components (beams 0, 1, 2)
    # Do NOT mask based on error velocity (beam 3)
    # Use NaN detection for float arrays, sentinel comparison for int arrays
    is_nan_mode = np.issubdtype(vel_values.dtype, np.floating)
    for beam_idx in range(min(3, n_beams)):
        if is_nan_mode:
            mask_data[beam_idx, :, :] = np.isnan(
                vel_values[beam_idx, :, :]
            ).astype(np.int8)
        else:
            mask_data[beam_idx, :, :] = (
                vel_values[beam_idx, :, :] <= VELOCITY_MISSING_VALUE
            ).astype(np.int8)

    # Create combined signal quality mask (4th beam)
    # This is the logical OR of U, V, W masks
    # If ANY of u, v, w is missing, mark as invalid
    mask_data[3, :, :] = (
        mask_data[0, :, :] | mask_data[1, :, :] | mask_data[2, :, :]
    ).astype(np.int8)

    # Coordinates
    coords = {
        "beam": np.arange(4),
        "cell": ds_velocity.coords["cell"].values,
        "ensemble": ds_velocity.coords["ensemble"].values,
    }

    # Data variables
    data_vars = {
        "mask": (("beam", "cell", "ensemble"), mask_data),
    }

    # Mask variable attributes
    mask_attrs = {
        "long_name": "Velocity quality mask",
        "description": (
            "Baseline quality mask based on missing values in velocity data. "
            "1 = invalid (missing), 0 = valid."
        ),
        "units": "dimensionless",
        "missing_value_code": VELOCITY_MISSING_VALUE,
        "flag_values": "0, 1",
        "flag_meanings": "valid invalid",
        "beam_0": "U velocity component mask",
        "beam_1": "V velocity component mask",
        "beam_2": "W velocity component mask",
        "beam_3": "Combined signal quality mask (U OR V OR W)",
        "comment": (
            "The 4th beam (beam_3) represents combined signal quality, not error "
            "velocity. It is the logical OR of U, V, W masks - marking a cell as "
            "invalid if ANY physical velocity component is missing."
        ),
    }

    # Dataset-level attributes
    ds_attrs = {
        "pyadps_component": "Mask",
        "pyadps_version": PYADPS_VERSION,
    }

    # Create xarray.Dataset
    ds_mask = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=ds_attrs,
    )

    # Apply variable attributes
    ds_mask["mask"].attrs.update(mask_attrs)

    # Log statistics
    u_invalid = int((mask_data[0, :, :] == 1).sum())
    v_invalid = int((mask_data[1, :, :] == 1).sum())
    w_invalid = int((mask_data[2, :, :] == 1).sum())
    combined_invalid = int((mask_data[3, :, :] == 1).sum())
    total_cells = n_cells * n_ensembles

    logger.debug(
        f"Created velocity mask: shape={ds_mask['mask'].shape}, "
        f"U invalid={u_invalid} ({100*u_invalid/total_cells:.2f}%), "
        f"V invalid={v_invalid} ({100*v_invalid/total_cells:.2f}%), "
        f"W invalid={w_invalid} ({100*w_invalid/total_cells:.2f}%), "
        f"combined invalid={combined_invalid} ({100*combined_invalid/total_cells:.2f}%)"
    )

    return ds_mask


def _compute_time_coordinate(ds_vl: xr.Dataset) -> ArrayLike:
    """
    Compute datetime64 array from Variable Leader RTC (Real-Time Clock) fields.

    Constructs a complete datetime64 array using:
    - RTC Year, Month, Day, Hour, Minute, Second from Variable Leader

    Parameters
    ----------
    ds_vl : xr.Dataset
        Variable Leader dataset with RTC fields.
        Expected to contain: rtc_year, rtc_month, rtc_day, rtc_hour,
        rtc_minute, rtc_second

    Returns
    -------
    np.ndarray
        datetime64[ns] array with length = number of ensembles

    Raises
    ------
    KeyError
        If required RTC fields are missing from Variable Leader dataset.
    ValueError
        If RTC values are invalid (e.g., invalid month/day combinations).
    """
    try:
        year = ds_vl["rtc_year"].values + 2000
        month = ds_vl["rtc_month"].values
        day = ds_vl["rtc_day"].values
        hour = ds_vl["rtc_hour"].values
        minute = ds_vl["rtc_minute"].values
        second = ds_vl["rtc_second"].values

        date_df = pd.DataFrame(
            {
                "year": year,
                "month": month,
                "day": day,
                "hour": hour,
                "minute": minute,
                "second": second,
            }
        )

        time_array = pd.to_datetime(date_df).values

        logger.debug(
            f"Computed time coordinate: {len(time_array)} timestamps from {time_array[0]} to {time_array[-1]}"
        )
        return time_array

    except KeyError as e:
        raise KeyError(f"Missing required RTC field in Variable Leader: {e}") from e
    except ValueError as e:
        raise ValueError(f"Invalid RTC values detected: {e}") from e


def _compute_depth_coordinate(
    ds_fl: xr.Dataset, ds_vl: xr.Dataset, use_fl_accessor: bool = True
) -> np.ndarray:
    """
    Compute depth array from Fixed Leader bin configuration and Variable Leader depth.

    Calculates the depth coordinate (in meters) for each measurement cell using:
    - num_cells: Number of measurement cells (from Fixed Leader)
    - bin_1_distance: Distance to first bin (from Fixed Leader)
    - depth_cell_length: Length of each depth cell (from Fixed Leader)
    - beam_direction: Direction of beams (from Fixed Leader, decoded)
    - transducer_depth: Mean transducer depth (from Variable Leader)

    Parameters
    ----------
    ds_fl : xr.Dataset
        Fixed Leader dataset with bin configuration fields.
    ds_vl : xr.Dataset
        Variable Leader dataset with transducer depth.
    use_fl_accessor : bool, default True
        If True, uses fixed_leader accessor for system_configuration().
        If False, uses raw field values directly.

    Returns
    -------
    np.ndarray
        float32 array of depths in meters, shape (num_cells,)

    Raises
    ------
    KeyError
        If required fields are missing.
    ValueError
        If computed depth values are invalid.
    """
    try:
        # Read required configuration from Fixed Leader
        num_cells = int(ds_fl["num_cells"].values[0])
        bin_1_dist = float(ds_fl["bin_1_distance"].values[0]) / 100.0  # cm Ã¢â€ â€™ m
        depth_cell_len = (
            float(ds_fl["depth_cell_length"].values[0]) / 100.0
        )  # cm Ã¢â€ â€™ m

        # Get beam direction
        if use_fl_accessor and hasattr(ds_fl, "fixed_leader"):
            beam_direction = ds_fl.fixed_leader.system_configuration()["Beam Direction"]
        else:
            # Fallback: read from decoded beam_direction field if available
            if "beam_direction" in ds_fl.data_vars:
                beam_dir_code = int(ds_fl["beam_direction"].values[0])
                beam_direction = "up" if beam_dir_code == 0 else "down"
            else:
                logger.warning("Could not determine beam direction; assuming downward")
                beam_direction = "down"

        # Read mean transducer depth from Variable Leader (in decimeters, dm)
        transducer_depth_dm = ds_vl["transducer_depth"].values
        mean_depth = float(np.mean(transducer_depth_dm)) / 10.0  # dm Ã¢â€ â€™ m
        mean_depth = np.trunc(mean_depth)  # Truncate to integer meters

        # Direction sign: up = -1, down = +1
        direction_sign = -1 if beam_direction.lower() == "up" else 1

        # Compute first depth (to center of first bin)
        first_depth = mean_depth + direction_sign * bin_1_dist

        # Compute all depths for each cell
        depths = np.linspace(
            first_depth,
            first_depth + direction_sign * (num_cells - 1) * depth_cell_len,
            num_cells,
            dtype=np.float32,
        )

        logger.debug(
            f"Computed depth coordinate: {num_cells} cells from {depths[0]:.2f}m "
            f"to {depths[-1]:.2f}m (direction: {beam_direction})"
        )
        return depths

    except KeyError as e:
        raise KeyError(f"Missing required field for depth computation: {e}") from e
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid values for depth computation: {e}") from e


def _merge_datasets(
    ds_header: xr.Dataset,
    ds_fl: Optional[xr.Dataset] = None,
    ds_vl: Optional[xr.Dataset] = None,
    ds_velocity: Optional[xr.Dataset] = None,
    ds_correlation: Optional[xr.Dataset] = None,
    ds_echo: Optional[xr.Dataset] = None,
    ds_percent_good: Optional[xr.Dataset] = None,
    ds_status: Optional[xr.Dataset] = None,
    time_coord: Optional[ArrayLike] = None,
    depth_coord: Optional[np.ndarray] = None,
    ds_mask: Optional[xr.Dataset] = None,
    include_header: bool = False,
) -> xr.Dataset:
    """
    Merge individual component datasets into a single comprehensive xarray.Dataset.

    This function coordinates the merging of Header, FixedLeader, VariableLeader,
    and optional data type datasets (Velocity, Correlation, Echo, PercentGood, Status)
    into a single xarray.Dataset with proper coordinate alignment.

    Merging strategy:
    1. Start with Header dataset
    2. Merge Fixed/Variable Leader variables
    3. Merge optional data type datasets
    4. Add computed coordinates (time, depth)
    5. Add velocity mask if provided
    6. Consolidate attributes into dataset-level metadata

    Parameters
    ----------
    ds_header : xr.Dataset
        Header dataset (required, contains metadata structure).
    ds_fl : xr.Dataset, optional
        Fixed Leader dataset with configuration variables.
    ds_vl : xr.Dataset, optional
        Variable Leader dataset with sensor data and timestamps.
    ds_velocity : xr.Dataset, optional
        Velocity data dataset.
    ds_correlation : xr.Dataset, optional
        Correlation data dataset.
    ds_echo : xr.Dataset, optional
        Echo intensity data dataset.
    ds_percent_good : xr.Dataset, optional
        Percent good data dataset.
    ds_status : xr.Dataset, optional
        Status data dataset.
    time_coord : np.ndarray, optional
        Precomputed time coordinate (datetime64). If None, not added.
    depth_coord : np.ndarray, optional
        Precomputed depth coordinate (float32 meters). If None, not added.
    ds_mask : xr.Dataset, optional
        Velocity mask dataset from _create_velocity_mask(). If None, not added.
        Contains 'mask' variable with shape (beam, cell, ensemble), values 0=valid, 1=invalid.
    include_header : bool, default False
        If True, include header data variables in the merged dataset.
        If False, header variables are excluded (only attributes are kept).

    Returns
    -------
    xr.Dataset
        Merged dataset with all available variables, coordinates, and
        consolidated attributes.

    Raises
    ------
    ValueError
        If datasets have incompatible dimensions or coordinate conflicts.
    """
    # Start with header as base
    ds_merged = ds_header.copy(deep=True)

    # If include_header is False, drop header data variables but keep attributes
    if not include_header:
        header_vars_to_drop = [
            "data_type_array",
            "byte",
            "byte_skip",
            "address_offset",
            "data_id",
        ]
        vars_to_drop = [v for v in header_vars_to_drop if v in ds_merged.data_vars]
        if vars_to_drop:
            logger.debug(
                f"Excluding header data variables: {vars_to_drop} (include_header=False)"
            )
            ds_merged = ds_merged.drop_vars(vars_to_drop)

        # Also drop data_type coordinate since it only relates to header
        if "data_type" in ds_merged.coords:
            logger.debug("Dropping data_type coordinate (include_header=False)")
            ds_merged = ds_merged.drop_vars("data_type")

    component_list = {"fixed_leader": [], "variable_leader": []}
    # Merge Fixed Leader if provided
    if ds_fl is not None:
        logger.debug(f"Merging Fixed Leader: {len(ds_fl.data_vars)} variables")
        ds_merged = xr.merge(
            [ds_merged, ds_fl.drop_vars(["ensemble", "data_type"], errors="ignore")],
            compat="no_conflicts",
        )

    # Merge Variable Leader if provided
    if ds_vl is not None:
        logger.debug(f"Merging Variable Leader: {len(ds_vl.data_vars)} variables")
        ds_merged = xr.merge(
            [ds_merged, ds_vl.drop_vars(["ensemble"], errors="ignore")],
            compat="no_conflicts",
        )

    # Merge data type datasets
    datasets_to_merge = [
        (ds_velocity, "Velocity"),
        (ds_correlation, "Correlation"),
        (ds_echo, "Echo"),
        (ds_percent_good, "PercentGood"),
        (ds_status, "Status"),
    ]

    for ds, name in datasets_to_merge:
        if ds is not None:
            logger.debug(f"Merging {name}: {len(ds.data_vars)} variables")
            ds_merged = xr.merge(
                [ds_merged, ds.drop_vars(["ensemble"], errors="ignore")],
                compat="no_conflicts",
            )

    # Add velocity mask if provided
    if ds_mask is not None:
        logger.debug(f"Merging Mask: {len(ds_mask.data_vars)} variables")
        ds_merged = xr.merge(
            [ds_merged, ds_mask.drop_vars(["ensemble"], errors="ignore")],
            compat="no_conflicts",
        )

    # Add time coordinate if provided
    if time_coord is not None:
        logger.debug(f"Adding time coordinate: {len(time_coord)} timestamps")
        # Create time as a proper coordinate
        ds_merged = ds_merged.assign_coords(time=("ensemble", time_coord))
        ds_merged["time"].attrs.update(
            {
                "standard_name": "time",
                "long_name": "Time (from RTC)",
                "axis": "T",
            }
        )

    # Add depth coordinate if provided
    if depth_coord is not None:
        logger.debug(f"Adding depth coordinate: {len(depth_coord)} cells")
        # Create depth as a proper coordinate
        ds_merged = ds_merged.assign_coords(depth=("cell", depth_coord))
        ds_merged["depth"].attrs.update(
            {
                "standard_name": "depth",
                "long_name": "Distance from mean transducer depth",
                "units": "m",
                "axis": "Z",
                "positive": "down",
            }
        )

    # Consolidate component attributes into merged dataset
    # Store which components were included
    components = {
        "header": True,
        "fixed_leader": ds_fl is not None,
        "variable_leader": ds_vl is not None,
        "velocity": ds_velocity is not None,
        "correlation": ds_correlation is not None,
        "echo": ds_echo is not None,
        "percent_good": ds_percent_good is not None,
        "status": ds_status is not None,
        "mask": ds_mask is not None,
    }

    ds_merged.attrs["components"] = str(components)
    ds_merged.attrs["pyadps_component"] = "Complete"
    if ds_fl is not None:
        ds_merged.attrs["fixed_leader_variables"] = list(ds_fl.drop_vars(["ensemble"]))
    if ds_vl is not None:
        ds_merged.attrs["variable_leader_variables"] = list(
            ds_vl.drop_vars(["ensemble"])
        )

    logger.info(
        f"Successfully merged ADCP dataset: {len(ds_merged.data_vars)} variables, "
        f"{len(ds_merged.dims)} dimensions"
    )
    return ds_merged


def read(
    adcp_file: FilePathType,
    include_decoded: bool = True,
    data_types: Optional[list] = None,
    use_time_as_primary_dim: bool = True,
    use_depth_as_primary_dim: bool = False,
    include_header: bool = False,
    include_mask: bool = True,
    missing_as_nan: bool = True,
) -> xr.Dataset:
    """
    Load complete ADCP dataset from RDI binary file into xarray.Dataset.

    This is the primary v1.0.0 interface that replaces the legacy ReadFile class.
    It orchestrates the reading of all ADCP file components (Header, FixedLeader,
    VariableLeader, and data types like Velocity, Correlation, Echo, etc.) and
    merges them into a single comprehensive xarray.Dataset with proper coordinates
    and metadata.

    The returned dataset is compatible with all xarray operations and can be
    accessed via domain-specific accessor methods (registered in accessors.py)
    for backward compatibility.

    Parameters
    ----------
    adcp_file : str or Path
        Path to ADCP binary file in PD0 format.
    include_decoded : bool, default True
        Include decoded variables from Fixed/Variable Leaders.
        Decoded variables include interpreted fields like frequency, beam pattern,
        motion sensors, ADC channels, and error status word fields.
    data_types : list of str, optional
        Specific components and data types to include. If None, includes all available.

        Component options: 'FixedLeader', 'VariableLeader'
        Data type options: 'Velocity', 'Correlation', 'Echo', 'PercentGood', 'Status'

        Note: VariableLeader is ALWAYS read by default (provides time coordinate).
        Exclude it only if you explicitly don't need time information.

        Examples:
        - None (default) - All components and data types
        - ['FixedLeader', 'VariableLeader', 'Velocity'] - Config + sensor data + velocities
        - ['Velocity', 'Correlation', 'Echo', 'Percent Good] - VariableLeader + specific measurements (VL always included)
        - ['FixedLeader'] - FixedLeader + VariableLeader (VL always included for time)
        - ['VariableLeader'] - Only configuration (no measurement data types)
    use_time_as_primary_dim : bool, default True
        Use time as the primary (first) dimension instead of ensemble.
        If True, data variables are arranged with time as first dimension.
        If False, ensemble is the primary dimension (legacy behavior).
        This affects the layout of multi-dimensional variables like velocity.
    use_depth_as_primary_dim : bool, default False
        Use depth as the dimension instead of cell.
        If True, data variables are arranged with cell.
        If False, ensemble is the primary dimension.
        This affects the layout of multi-dimensional variables like velocity.
        WARNING: Mean transducer depth is used to compute depth of each cell. This option
        is not recommended if the ADCP has large vertical oscillations.
    include_header : bool, default False
        Include header metadata variables in the returned dataset.
        If False (default), header data is not included in the output.
        If True, includes header variables: data_type_array, byte, byte_skip,
        address_offset, data_id. Users can call read_header() separately if needed.
    include_mask : bool, default True
        Include the velocity quality mask in the returned dataset.
        If True (default), creates a 3D mask variable 'mask' based on
        missing values (-32768) in the velocity data. The mask has dimensions
        (beam, cell, ensemble) where:
        - beam 0: U velocity component mask
        - beam 1: V velocity component mask
        - beam 2: W velocity component mask
        - beam 3: Combined signal quality mask (U OR V OR W)
        If False, no mask is created (useful for faster loading when mask not needed).
        Note: Mask is only created if Velocity data is loaded.
    missing_as_nan : bool, default True
        If True (default), replace the RDI missing value sentinel (-32768) with
        np.nan and store velocity as float32. Xarray aggregations skip NaN cells
        automatically, making this the safe default for analysis.
        If False, retain raw int16 velocity data with -32768 as the sentinel.
        Passed through to read_velocity().

    Returns
    -------
    xr.Dataset
        Complete ADCP dataset with:

        Dimensions:
        - ensemble: N ensembles (time dimension)
        - cell: M measurement cells (depth dimension)
        - beam: K beams (typically 4)
        - data_type: L data type configurations

        Coordinates:
        - time: datetime64[ns] from RTC fields (if include_decoded=True)
        - depth: float32 from bin configuration (if include_decoded=True)

        Variables:
        - Header metadata (byte counts, data IDs, offsets) - if include_header=True
        - Fixed Leader configuration (36 raw + 25 decoded if include_decoded) - if FixedLeader in data_types
        - Variable Leader measurements (ALWAYS included, provides time coordinate)
        - Data type arrays (Velocity, Correlation, Echo, PercentGood, Status) - selected by data_types
        - mask : (beam, cell, ensemble) int8
            3D quality mask based on velocity missing values. Included if Velocity is loaded
            and include_mask=True. Values: 0=valid, 1=invalid.
            Beam dimensions represent:
            - beam 0: U velocity component mask
            - beam 1: V velocity component mask
            - beam 2: W velocity component mask
            - beam 3: Combined signal quality mask (U OR V OR W)
            The 4th beam is NOT the error velocity mask - it represents combined
            signal quality where ANY of u, v, w being missing marks the cell invalid.

        Note: VariableLeader is always read to provide the time coordinate unless explicitly excluded.
        If data_types=None, all available components and data types are included.
        If data_types specifies certain items, VariableLeader is always added (e.g., data_types=['Velocity'] loads VariableLeader + Velocity)

        Attributes (dataset-level):
        - filename: Original RDI filename
        - total_ensembles: Number of ensembles
        - file_size_bytes: File size in bytes
        - error_message: Any errors encountered during read
        - pyadps_version: Package version (from pyproject.toml)
        - adcp_data_format: "PD0"
        - components: Dictionary of which components were included

    Raises
    ------
    FileNotFoundError
        If adcp_file does not exist.
    ValueError
        If file is not a valid ADCP file.
    KeyError
        If required fields are missing for coordinate computation.

    Examples
    --------
    Basic usage - read all components:

    >>> import pyadps
    >>> import pyadps.accessors  # Register accessors
    >>> ds = pyadps.read('file.000')
    >>> print(ds)
    >>> print(ds['velocity'].shape)  # (beam, cell, ensemble)
    >>> print(ds['time'].values[:5])

    Access data using native xarray:

    >>> velocity = ds['velocity'].values
    >>> time = ds['time'].values
    >>> depth = ds['depth'].values
    >>> pressure = ds['pressure'].values

    Use xarray features:

    >>> ds['velocity'].isel(beam=1).plot()
    >>> mean_velocity = ds['velocity'].mean(dim='ensemble')
    >>> ds.to_netcdf('output.nc')

    Access velocity mask for quality control:

    >>> mask = ds['mask']
    >>> # Count invalid U velocity measurements
    >>> u_invalid = (mask.sel(beam=0) == 1).sum().item()
    >>> # Find cells where any velocity component is missing
    >>> any_missing = (mask.sel(beam=3) == 1)
    >>> # Apply mask to velocity data
    >>> velocity_valid = ds['velocity'].where(mask == 0)

    Access metadata via accessors:

    >>> config = ds.fixed_leader.system_configuration()
    >>> freq = config['Frequency']
    >>> heading = ds['heading'].values

    Load only specific data types:

    >>> ds = pyadps.read('file.000', data_types=['Velocity', 'Correlation'])
    >>> # Result: VariableLeader + Velocity + Correlation (VL always included for time)

    Load only configuration (VariableLeader always included for time):

    >>> ds = pyadps.read('file.000', data_types=['FixedLeader', 'VariableLeader'])

    Load FixedLeader + Velocity (VariableLeader automatically included):

    >>> ds = pyadps.read('file.000', data_types=['FixedLeader', 'Velocity'])
    >>> # Result: FixedLeader + VariableLeader + Velocity (VL auto-included)

    Load without decoded fields (faster, smaller dataset):

    >>> ds = pyadps.read('file.000', include_decoded=False)

    Include header metadata if needed for file validation:

    >>> ds = pyadps.read('file.000', include_header=True)
    >>> print(ds['byte_skip'].values)  # File byte structure info

    Load without velocity mask (faster loading when mask not needed):

    >>> ds = pyadps.read('file.000', include_mask=False)
    >>> # Dataset will not include 'mask' variable

    See Also
    --------
    read_header : Read just the file header
    read_fixed_leader : Read just Fixed Leader data
    read_variable_leader : Read just Variable Leader data
    read_velocity : Read just velocity data
    """
    filename = str(adcp_file)
    logger.info(f"Starting ADCP read: {filename}")

    # ========================================================================
    # Step 1: Read Header - Required to get file structure
    # ========================================================================
    logger.debug("Reading file header...")
    ds_header = read_header(filename)

    byteskip = ds_header["byte_skip"].values
    offset = ds_header["address_offset"].values
    idarray = ds_header["data_id"].values
    ensemble_count = int(ds_header.attrs["total_ensembles"])
    available_datatypes = _get_available_data_types(ds_header, ens=0)

    logger.info(
        f"File structure: {ensemble_count} ensembles, "
        f"data types: {available_datatypes}"
    )

    # ========================================================================
    # Step 2: Read Fixed Leader (conditional)
    # ========================================================================
    ds_fl = None
    if data_types is None or "FixedLeader" in data_types:
        logger.debug("Reading Fixed Leader...")
        try:
            ds_fl = read_fixed_leader(
                filename,
                byteskip=byteskip,
                offset=offset,
                idarray=idarray,
                ensemble=ensemble_count,
                include_decoded=include_decoded,
            )
        except Exception as e:
            logger.warning(f"Could not read Fixed Leader data: {e}")
    else:
        logger.debug("Skipping Fixed Leader (not in data_types)")

    # ========================================================================
    # Step 3: Read Variable Leader (ALWAYS - provides time coordinate)
    # ========================================================================
    # VariableLeader is ALWAYS read regardless of data_types parameter
    # It provides the time coordinate, which is essential for ADCP analysis
    ds_vl = None
    logger.debug("Reading Variable Leader (always included for time coordinate)...")
    try:
        ds_vl = read_variable_leader(
            filename,
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble_count,
            include_decoded=include_decoded,
        )
    except Exception as e:
        logger.warning(f"Could not read Variable Leader data: {e}")

    # ========================================================================
    # Step 4: Compute Derived Coordinates
    # ========================================================================
    time_coord = None
    depth_coord = None

    if ds_vl is not None:
        logger.debug("Computing time coordinate...")
        try:
            time_coord = _compute_time_coordinate(ds_vl)
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Could not compute time coordinate: {e}")
    else:
        logger.debug("Skipping time coordinate (Variable Leader not loaded)")

    if ds_fl is not None and ds_vl is not None:
        logger.debug("Computing depth coordinate...")
        try:
            depth_coord = _compute_depth_coordinate(ds_fl, ds_vl, use_fl_accessor=True)
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Could not compute depth coordinate: {e}")
    else:
        logger.debug(
            "Skipping depth coordinate (FixedLeader or VariableLeader not loaded)"
        )

    # ========================================================================
    # Step 5: Read Optional Data Types
    # ========================================================================
    ds_velocity = None
    ds_correlation = None
    ds_echo = None
    ds_percent_good = None
    ds_status = None

    # Determine which data types to read
    types_to_read = data_types if data_types else available_datatypes

    if "Velocity" in types_to_read and "Velocity" in available_datatypes:
        logger.debug("Reading Velocity data...")
        try:
            ds_velocity = read_velocity(
                filename,
                byteskip=byteskip,
                offset=offset,
                idarray=idarray,
                ensemble=ensemble_count,
                missing_as_nan=missing_as_nan,
            )
        except Exception as e:
            logger.warning(f"Could not read Velocity data: {e}")

    if "Correlation" in types_to_read and "Correlation" in available_datatypes:
        logger.debug("Reading Correlation data...")
        try:
            ds_correlation = read_correlation(
                filename,
                byteskip=byteskip,
                offset=offset,
                idarray=idarray,
                ensemble=ensemble_count,
            )
        except Exception as e:
            logger.warning(f"Could not read Correlation data: {e}")

    if "Echo" in types_to_read and "Echo" in available_datatypes:
        logger.debug("Reading Echo Intensity data...")
        try:
            ds_echo = read_echo_intensity(
                filename,
                byteskip=byteskip,
                offset=offset,
                idarray=idarray,
                ensemble=ensemble_count,
            )
        except Exception as e:
            logger.warning(f"Could not read Echo Intensity data: {e}")

    if "Percent Good" in types_to_read and "Percent Good" in available_datatypes:
        logger.debug("Reading Percent Good data...")
        try:
            ds_percent_good = read_percent_good(
                filename,
                byteskip=byteskip,
                offset=offset,
                idarray=idarray,
                ensemble=ensemble_count,
            )
        except Exception as e:
            logger.warning(f"Could not read Percent Good data: {e}")

    if "Status" in types_to_read and "Status" in available_datatypes:
        logger.debug("Reading Status data...")
        try:
            ds_status = read_status(
                filename,
                byteskip=byteskip,
                offset=offset,
                idarray=idarray,
                ensemble=ensemble_count,
            )
        except Exception as e:
            logger.warning(f"Could not read Status data: {e}")

    # ========================================================================
    # Step 5b: Create Velocity Mask (if velocity data was loaded and mask requested)
    # ========================================================================
    ds_mask = None
    if include_mask and ds_velocity is not None:
        logger.debug("Creating velocity mask from missing values...")
        try:
            ds_mask = _create_velocity_mask(ds_velocity)
        except Exception as e:
            logger.warning(f"Could not create velocity mask: {e}")
    elif not include_mask:
        logger.debug("Skipping velocity mask creation (include_mask=False)")

    # ========================================================================
    # Step 6: Pre-Merge Ensemble Consistency Check
    # ========================================================================
    # Check ensemble consistency BEFORE merging to prevent merge errors
    logger.debug("Checking ensemble consistency before merge...")

    if data_types is not None and "VariableLeader" not in data_types:
        ds_vl = None
    # Collect all datasets to check
    datasets_to_check = {
        "header": ds_header,
        "fixed_leader": ds_fl,
        "variable_leader": ds_vl,
        "velocity": ds_velocity,
        "mask": ds_mask,
        "correlation": ds_correlation,
        "echo": ds_echo,
        "percent_good": ds_percent_good,
        "status": ds_status,
    }

    # Find ensemble sizes for each dataset
    ensemble_sizes = {}
    for name, ds_component in datasets_to_check.items():
        if ds_component is not None and "ensemble" in ds_component.dims:
            ensemble_sizes[name] = ds_component.sizes["ensemble"]

    # Check if all have same size
    if ensemble_sizes and len(set(ensemble_sizes.values())) > 1:
        logger.warning(
            f"Ensemble count inconsistency detected before merge: {ensemble_sizes}"
        )
        min_ensemble = min(ensemble_sizes.values())
        logger.info(f"Truncating all datasets to {min_ensemble} ensembles before merge")

        # Truncate all datasets to minimum ensemble count
        if ds_fl is not None and "ensemble" in ds_fl.dims:
            ds_fl = ds_fl.isel(ensemble=slice(0, min_ensemble))
        if ds_vl is not None and "ensemble" in ds_vl.dims:
            ds_vl = ds_vl.isel(ensemble=slice(0, min_ensemble))
        if ds_velocity is not None and "ensemble" in ds_velocity.dims:
            ds_velocity = ds_velocity.isel(ensemble=slice(0, min_ensemble))
        if ds_mask is not None and "ensemble" in ds_mask.dims:
            ds_mask = ds_mask.isel(ensemble=slice(0, min_ensemble))
        if ds_correlation is not None and "ensemble" in ds_correlation.dims:
            ds_correlation = ds_correlation.isel(ensemble=slice(0, min_ensemble))
        if ds_echo is not None and "ensemble" in ds_echo.dims:
            ds_echo = ds_echo.isel(ensemble=slice(0, min_ensemble))
        if ds_percent_good is not None and "ensemble" in ds_percent_good.dims:
            ds_percent_good = ds_percent_good.isel(ensemble=slice(0, min_ensemble))
        if ds_status is not None and "ensemble" in ds_status.dims:
            ds_status = ds_status.isel(ensemble=slice(0, min_ensemble))

    # ========================================================================
    # Step 7: Merge All Datasets
    # ========================================================================
    logger.debug("Merging datasets...")
    ds = _merge_datasets(
        ds_header,
        ds_fl=ds_fl,
        ds_vl=ds_vl,
        ds_velocity=ds_velocity,
        ds_correlation=ds_correlation,
        ds_echo=ds_echo,
        ds_percent_good=ds_percent_good,
        ds_status=ds_status,
        time_coord=time_coord,
        depth_coord=depth_coord,
        ds_mask=ds_mask,
        include_header=include_header,
    )

    # ========================================================================
    # Step 8: Validate Dataset Content
    # ========================================================================
    # Warn if dataset has very few variables (likely an error in data_types filtering)
    if len(ds.data_vars) == 0:
        logger.warning(
            "Resulting dataset has no data variables. Check that data_types "
            "parameter includes valid component names."
        )

    # ========================================================================
    # Step 9: Transpose Dimensions (if use_time_as_primary_dim=True)
    # ========================================================================
    if use_time_as_primary_dim and "time" in ds.coords:
        logger.debug("Transposing dimensions to use time as primary axis...")
        ds = ds.swap_dims({"ensemble": "time"})
        # ds = ds.drop_vars("ensemble")
        logger.debug("Dimension transposition complete")

    if use_depth_as_primary_dim and "depth" in ds.coords:
        logger.debug("Transposing dimensions to use depth as primary axis...")
        ds = ds.swap_dims({"cell": "depth"})
        ds = ds.drop_vars("cell")
    else:
        if "depth" in ds.coords:
            pass
            # ds = ds.drop_vars("depth")

    logger.info(
        f"Successfully loaded ADCP file: {len(ds.data_vars)} variables, "
        f"{dict(ds.sizes)}"
    )
    return ds
