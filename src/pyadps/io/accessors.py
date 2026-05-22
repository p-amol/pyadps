#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xarray Accessors for ADCP Data Processing (Vendor-Neutral)

This module provides xarray accessors that attach domain-specific methods
to xarray.Dataset objects created by pyadps functions. Accessors are registered
with @xr.register_dataset_accessor() decorator and provide methods under
custom namespaces (e.g., ds.header.*, ds.fixed_leader.*, etc.).

Design Pattern:
- Accessors extend xarray.Dataset functionality without subclassing
- All custom methods available under component-specific namespace
- No method naming conflicts with xarray API
- Full xarray compatibility maintained
- Vendor-neutral terminology from the start (for future multi-vendor support)

Reference: https://docs.xarray.dev/en/stable/extending.html#extending-with-accessors

Accessor Registration:
- @xr.register_dataset_accessor("header")          -> Header data operations
- @xr.register_dataset_accessor("fixed_leader")    -> FixedLeader data operations
- @xr.register_dataset_accessor("variable_leader") -> VariableLeader data operations
- @xr.register_dataset_accessor("velocity")        -> Velocity data operations
- @xr.register_dataset_accessor("correlation")     -> Correlation data operations
- @xr.register_dataset_accessor("echo")            -> Echo intensity data operations
- @xr.register_dataset_accessor("percent_good")    -> Percent good data operations
- @xr.register_dataset_accessor("status")          -> Status data operations

Author: pyadps development team
License: MIT
Version: 1.0.0
"""

from collections import Counter
from typing import Dict, List, Optional
import logging
import xarray as xr
import numpy as np
import pandas as pd


from pyadps.io import pd0_parser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def _check_equal(array: np.ndarray) -> bool:
    """
    Check if all elements in array are equal.

    This utility function is used by accessor methods to validate
    uniformity across ensembles (e.g., byte counts, data types).

    Parameters
    ----------
    array : np.ndarray
        Input array to check for uniformity.

    Returns
    -------
    bool
        True if all elements are equal or array is empty, False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> _check_equal(np.array([5, 5, 5]))
    True
    >>> _check_equal(np.array([5, 5, 6]))
    False
    >>> _check_equal(np.array([]))
    True
    """
    if len(array) == 0:
        return True
    return bool(np.all(array == array[0]))


def _check_2d_rows_uniform(array: np.ndarray, array_name: str = "array") -> bool:
    """
    Check if all rows in a 2D array are uniform across columns.

    For a 2D array (rows x columns), this function verifies that each row
    contains identical values across all columns.

    Parameters
    ----------
    array : np.ndarray
        2D array of shape (num_rows, num_columns).
    array_name : str, optional
        Name of array for logging purposes, by default "array".

    Returns
    -------
    bool
        True if each row is uniform across columns, False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> address_offset = np.array([
    ...     [18, 18, 18, 18],
    ...     [77, 77, 77, 77],
    ...     [142, 142, 142, 142]
    ... ])
    >>> _check_2d_rows_uniform(address_offset, "address_offset")
    True

    >>> data_id = np.array([
    ...     [0, 0, 0, 0],
    ...     [128, 128, 128, 128],
    ...     [256, 256, 256, 256]
    ... ])
    >>> _check_2d_rows_uniform(data_id, "data_id")
    True
    """
    if array.size == 0:
        return True

    # Ensure 2D array
    if array.ndim != 2:
        logger.warning(f"{array_name} should be 2D, got {array.ndim}D")
        return False

    # Check each row for uniformity across columns
    for row_idx, row in enumerate(array.T):
        if not _check_equal(row):
            logger.debug(f"{array_name} row {row_idx} is not uniform: {row}")
            return False

    return True


def _check_byte_skip_uniformity(byte_skip: np.ndarray, num_ensembles: int) -> bool:
    """
    Check if byte_skip values are uniform after dividing by ensemble indices.

    This utility function validates that byte_skip follows the expected pattern:
    byte_skip[i] = byte_skip[0] * (i + 1) for all ensembles.

    Parameters
    ----------
    byte_skip : np.ndarray
        1D array of byte_skip values (one per ensemble).
    num_ensembles : int
        Total number of ensembles.

    Returns
    -------
    bool
        True if byte_skip values follow the expected uniform pattern, False otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> byte_skip = np.array([1134, 2268, 3402, 4536])
    >>> _check_byte_skip_uniformity(byte_skip, 4)
    True
    >>> byte_skip_bad = np.array([1134, 2268, 3400, 4536])
    >>> _check_byte_skip_uniformity(byte_skip_bad, 4)
    False
    """
    if len(byte_skip) == 0:
        return True

    # Compute verification array: byte_skip / (ensemble + 1)
    ensemble_indices = np.arange(num_ensembles, dtype=np.float64)
    verification_array = byte_skip.astype(np.float64) / (ensemble_indices + 1)

    # Check if all values in verification array are equal (within floating point tolerance)
    return bool(np.allclose(verification_array, verification_array[0], rtol=1e-9))


# ============================================================================
# HEADER ACCESSOR
# ============================================================================


@xr.register_dataset_accessor("header")
class HeaderAccessor:
    """
    xarray accessor for ADCP Header data operations.

    Header data contains ensemble-level metadata including file structure
    information, data type mappings, and file integrity information.

    Access via: ds.header.*

    Attributes
    ----------
    _obj : xr.Dataset
        The underlying xarray.Dataset object.

    Methods
    -------
    data_types(ens=0)
        Lists out the data types for any one ensemble (default = 0)
    check_file()
        Checks if file structure is valid and uniform
    print_check_file()
        Prints formatted file check results
    get_ensemble_info(ensemble)
        Get detailed information about a specific ensemble
    has_data_type(data_type_name)
        Check if a specific data type is present in the file
    get_available_datatypes()
        Get all unique data types across all ensembles
    validate()
        Comprehensive validation of header structure
    summary()
        Generate human-readable summary string (non-printing version)

    Examples
    --------
    >>> import pyadps
    >>> import pyadps.accessors  # Register accessor
    >>> ds = pyadps.read_header('test.000')
    >>> ds.header.data_types(ens=0)
    >>> ds.header.check_file()
    >>> ds.header.print_check_file()
    >>> print(ds.header.summary())

    Notes
    -----
    This accessor is specifically for Header datasets. For other ADCP
    data types, use their respective accessors (fixed_leader, variable_leader, etc.).
    """

    # Data ID to name mapping
    _ID_NAME_MAP = {
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

    def __init__(self, xarray_obj: xr.Dataset):
        """Initialize the accessor with validation and cached properties."""
        self._obj = xarray_obj
        self._validate_dataset()

        # Cache for computed properties
        self._ensemble = None
        self._data_type_index = None
        self._error_code = None
        self._calculated_size_bytes = None

    def _validate_dataset(self) -> None:
        """
        Validate that dataset has required attributes.

        Raises
        ------
        ValueError
            If dataset lacks required Header attributes.
        """
        required_attrs = ["filename", "total_ensembles", "pyadps_component"]
        missing = [attr for attr in required_attrs if attr not in self._obj.attrs]

        if missing:
            raise ValueError(
                f"Dataset missing required attributes: {missing}. "
                f"This may not be a valid Header dataset."
            )

        # Verify this is actually a Header dataset
        if self._obj.attrs.get("pyadps_component") != "Header":
            logger.warning(
                "This accessor is optimized for Header datasets. "
                "Some methods may not work as expected."
            )

    # ========================================================================
    # COMPUTED PROPERTIES (Lazy-Evaluated)
    # ========================================================================

    @property
    def ensemble(self) -> int:
        """
        Get actual file size in bytes from the file system.

        Returns
        -------
        int
            Total number of ensembles in each data

        Notes
        -----
        If the each data has different ensemble, returns the minimum common ensemble

        Examples
        --------
        >>> ds = pyadps.read_header('test.000')
        >>> print(ds.header.ensembles)
        501
        """
        return self._obj.attrs.get("total_ensembles", 0)

    @property
    def data_type(self) -> int:
        """
        Get actual file size in bytes from the file system.

        Returns
        -------
        int
            Total number of ensembles in each data

        Notes
        -----
        If the each data has different ensemble, returns the minimum common ensemble

        Examples
        --------
        >>> ds = pyadps.read_header('test.000')
        >>> print(ds.header.ensembles)
        501
        """
        return self._obj.attrs.get("num_data_types", 0)

    @property
    def error_message(self) -> str:
        """
        Get error message from Dataset attributes.

        Returns
        -------
        str
            Error message describing any issues encountered during file reading.

        Examples
        --------
        >>> ds = pyadps.read_header('test.000')
        >>> print(ds.header.error_message)
        'Success'
        """
        return self._obj.attrs.get("error_message", "Unknown")

    @property
    def error_code(self) -> int:
        """
        Get error code by reverse-lookup from error message.

        The error code is not stored in the Dataset to avoid redundancy.
        Instead, it is derived from the error_message using the ErrorCode
        enum from pd0_parser. The result is cached for performance.

        Returns
        -------
        int
            Error code (0=Success, 1=File not found, 5=Wrong file type, etc.)

        Notes
        -----
        The error code is cached after first access. To refresh the code
        (e.g., after external file changes), call clear_property_cache().

        Examples
        --------
        >>> ds = pyadps.read_header('test.000')
        >>> print(ds.header.error_code)
        0
        >>> if ds.header.error_code != 0:
        ...     print(f"Error: {ds.header.error_message}")
        """
        if self._error_code is None:
            self._error_code = pd0_parser.ErrorCode.get_code(self.error_message)
        return self._error_code

    @property
    def file_size_bytes(self) -> int:
        """
        Get file size in bytes computed at read time.

        The file size is computed during dataset creation using Path.stat().st_size
        and stored as an immutable attribute. This avoids the need to store the
        file path and ensures the file size is captured at the exact moment of
        data loading.

        Returns
        -------
        int
            File size in bytes, or -1 if file could not be read at load time.

        Notes
        -----
        Unlike previous versions, file size is immutable after dataset creation.
        This means external file changes after loading will not be reflected.
        If you need to refresh file size information, create a new dataset by
        calling read_header() again.

        Examples
        --------
        >>> ds = pyadps.read_header('test.000')
        >>> size_mb = ds.header.file_size_bytes / (1024 * 1024)
        >>> print(f"File size: {size_mb:.2f} MB")
        """
        return self._obj.attrs.get("file_size_bytes", -1)

    @property
    def calculated_size_bytes(self) -> int:
        """
        Calculate expected file size from ensemble data.

        Computes the expected file size as: sum(byte) + 2*num_ensembles
        The +2 per ensemble accounts for the checksum bytes at the end of each
        ensemble. Result is cached for performance.

        Returns
        -------
        int
            Calculated file size in bytes, or -1 if data unavailable.

        Notes
        -----
        This calculation is based on the "byte" data variable in the Dataset.
        It reflects the expected size from the ensemble structure.

        Examples
        --------
        >>> ds = pyadps.read_header('test.000')
        >>> calc_size = ds.header.calculated_size_bytes
        >>> actual_size = ds.header.file_size_bytes
        >>> if calc_size == actual_size:
        ...     print("File structure is valid")
        """
        if self._calculated_size_bytes is None:
            if "byte" in self._obj.data_vars:
                try:
                    self._calculated_size_bytes = int(
                        self._obj.byte.sum() + 2 * len(self._obj.ensemble)
                    )
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Could not calculate file size: {e}")
                    self._calculated_size_bytes = -1
            else:
                self._calculated_size_bytes = -1
        return self._calculated_size_bytes

    @property
    def file_size_match(self) -> bool:
        """
        Check if system file size matches calculated size.

        Compares the actual file size (from disk) with the calculated size
        (from ensemble structure). Returns True only if both are positive
        and equal.

        Returns
        -------
        bool
            True if file sizes match and are valid, False otherwise.

        Examples
        --------
        >>> ds = pyadps.read_header('test.000')
        >>> if ds.header.file_size_match:
        ...     print("File structure is valid")
        >>> else:
        ...     print("WARNING: File size mismatch detected")
        """
        sys_size = self.file_size_bytes
        calc_size = self.calculated_size_bytes
        return sys_size == calc_size and sys_size > 0

    # ========================================================================
    # PUBLIC METHODS
    # ========================================================================

    def get_available_data_types(self, ens: int = 0) -> List[str]:
        """
        Get list of data types present in a specific ensemble.

        RDI/ADCP files can contain multiple data types per ensemble, each
        identified by a unique ID. This method maps IDs to human-readable
        names. Some IDs have dual-mode variants (e.g., Broadband vs Narrowband).

        This method uses the shared helper function _get_available_data_types()
        from filereader module to ensure consistency.

        Parameters
        ----------
        ens : int, optional
            Ensemble index (0-based), by default 0 (first ensemble).

        Returns
        -------
        list of str
            Data type names for the specified ensemble.

        Raises
        ------
        IndexError
            If ens is out of valid range.

        Examples
        --------
        >>> ds = pyadps.Header('test.000')
        >>> dtypes_0 = ds.header.data_types(ens=0)
        >>> print(dtypes_0)
        ['Fixed Leader', 'Variable Leader', 'Velocity', 'Echo']

        >>> dtypes_5 = ds.header.data_types(ens=5)
        """
        from pyadps.io.binary_reader import _get_available_data_types

        total_ensembles = self._obj.attrs.get("total_ensembles", 0)

        if ens < 0 or ens >= total_ensembles:
            raise IndexError(
                f"Ensemble index {ens} out of range [0, {total_ensembles - 1}]"
            )

        # Use shared helper function
        return _get_available_data_types(self._obj, ens=ens)

    def check_file(self) -> Dict:
        """
        Perform integrity checks on the file.

        Validates:
        1. System file size matches calculated file size
        2. Uniformity of bytes across ensembles
        3. Uniformity of data types across ensembles
        4. Uniformity of byte_skip (after normalization by ensemble)
        5. Uniformity of address_offset across ensembles
        6. Uniformity of data_id across ensembles

        Returns
        -------
        dict
            Dictionary containing check results:

            - 'System File Size (B)': int
              Actual file size on disk in bytes.
            - 'Calculated File Size (B)': int
              Expected file size based on ensemble structure.
            - 'File Size (MB)': float
              File size in megabytes.
            - 'File Size Match': bool
              True if system and calculated sizes match.
            - 'Byte Uniformity': bool
              True if all ensembles have same byte count.
            - 'Data Type Uniformity': bool
              True if all ensembles have same number of data types.
            - 'Byte Skip Uniformity': bool
              True if byte_skip / (ensemble + 1) values are uniform.
            - 'Address Offset Uniformity': bool
              True if address_offset is uniform across ensembles per data type.
            - 'Data ID Uniformity': bool
              True if data_id values are uniform across ensembles per data type.

        Examples
        --------
        >>> ds = pyadps.Header('test.000')
        >>> check = ds.header.check_file()
        >>> for key, value in check.items():
        ...     print(f"{key}: {value}")

        >>> if check['File Size Match']:
        ...     print("File structure is valid")
        """
        # Use the new accessor properties
        sys_file_size = self.file_size_bytes
        cal_file_size = self.calculated_size_bytes
        num_ensembles = self._obj.attrs.get("total_ensembles", 0)

        # Check uniformity using utility functions
        bytes_values = self._obj.byte.values if "byte" in self._obj.data_vars else []
        datatypes_values = (
            self._obj.data_type_array.values
            if "data_type_array" in self._obj.data_vars
            else []
        )
        byte_skip_values = (
            self._obj.byte_skip.values if "byte_skip" in self._obj.data_vars else []
        )
        address_offset_values = (
            self._obj.address_offset.values
            if "address_offset" in self._obj.data_vars
            else np.array([])
        )
        data_id_values = (
            self._obj.data_id.values
            if "data_id" in self._obj.data_vars
            else np.array([])
        )

        byte_uniformity = _check_equal(np.array(bytes_values))
        datatype_uniformity = _check_equal(np.array(datatypes_values))
        byte_skip_uniformity = _check_byte_skip_uniformity(
            np.array(byte_skip_values, dtype=np.int64), num_ensembles
        )
        address_offset_uniformity = _check_2d_rows_uniform(
            np.asarray(address_offset_values), "address_offset"
        )
        data_id_uniformity = _check_2d_rows_uniform(
            np.asarray(data_id_values), "data_id"
        )

        check_dict = {
            "System File Size (B)": sys_file_size,
            "Calculated File Size (B)": cal_file_size,
            "File Size (MB)": cal_file_size / 1048576 if cal_file_size > 0 else 0,
            "File Size Match": self.file_size_match,
            "Byte Uniformity": byte_uniformity,
            "Data Type Uniformity": datatype_uniformity,
            "Byte Skip Uniformity": byte_skip_uniformity,
            "Address Offset Uniformity": address_offset_uniformity,
            "Data ID Uniformity": data_id_uniformity,
        }
        return check_dict

    def get_ensemble_info(self, ensemble: int) -> Dict:
        """
        Get detailed information about a specific ensemble.

        Returns comprehensive details about ensemble structure including
        byte sizes, offsets, data types, and IDs.

        Parameters
        ----------
        ensemble : int
            Ensemble number to query (0-indexed)

        Returns
        -------
        dict
            Dictionary with ensemble details:

            - 'ensemble_number': int
              The ensemble index queried
            - 'byte_size': int
              Number of bytes in this ensemble (excluding checksum)
            - 'byte_skip': int
              File offset to this ensemble
            - 'num_datatypes': int
              Number of data types in this ensemble
            - 'data_types': list of str
              Names of data types present
            - 'data_ids': list of int
              Numeric IDs of data types
            - 'address_offsets': list of int
              Address offsets within ensemble for each data type

        Raises
        ------
        IndexError
            If ensemble is out of valid range.

        Examples
        --------
        >>> ds = pyadps.Header('test.000')
        >>> info = ds.header.get_ensemble_info(0)
        >>> print(f"Ensemble size: {info['byte_size']} bytes")
        >>> print(f"Data types: {info['data_types']}")
        >>> print(f"Contains Velocity: {'Velocity' in info['data_types']}")
        """
        total_ensembles = self._obj.attrs.get("total_ensembles", 0)

        if ensemble < 0 or ensemble >= total_ensembles:
            raise IndexError(
                f"Ensemble {ensemble} out of range [0, {total_ensembles - 1}]"
            )

        ens_data = self._obj.isel(ensemble=ensemble)

        # Get valid data IDs (non-zero)
        data_ids = ens_data.data_id.values if "data_id" in ens_data.data_vars else []
        valid_mask = data_ids != 0
        valid_ids = data_ids[valid_mask].tolist() if len(data_ids) > 0 else []

        # Get corresponding offsets
        offsets = (
            ens_data.address_offset.values
            if "address_offset" in ens_data.data_vars
            else []
        )
        valid_offsets = offsets[valid_mask].tolist() if len(offsets) > 0 else []

        return {
            "ensemble_number": int(ensemble),
            "byte_size": int(ens_data.byte.values)
            if "byte" in ens_data.data_vars
            else 0,
            "byte_skip": int(ens_data.byte_skip.values)
            if "byte_skip" in ens_data.data_vars
            else 0,
            "num_datatypes": int(ens_data.data_type_array.values)
            if "data_type_array" in ens_data.data_vars
            else 0,
            "data_types": self.get_available_data_types(ensemble),
            "data_ids": valid_ids,
            "address_offsets": valid_offsets,
        }

    def has_data_type(self, data_type_name: str) -> bool:
        """
        Check if a specific data type is present in the file.

        This checks the first ensemble (which is typically representative
        of the entire file if data types are uniform).

        Parameters
        ----------
        data_type_name : str
            Name of data type to check for. Valid names:
            'Fixed Leader', 'Variable Leader', 'Velocity',
            'Correlation', 'Echo', 'Percent Good', 'Status',
            'Bottom Track'

        Returns
        -------
        bool
            True if data type is present in at least the first ensemble

        Examples
        --------
        >>> ds = pyadps.Header('test.000')
        >>> has_velocity = ds.header.has_data_type('Velocity')
        >>> print(f"Has velocity data: {has_velocity}")

        >>> if ds.header.has_data_type('Correlation'):
        ...     print("Correlation data available")
        """
        try:
            data_types = self.get_available_data_types(0)
            return data_type_name in data_types
        except (IndexError, KeyError) as e:
            logger.warning(f"Error checking for data type '{data_type_name}': {e}")
            return False

    def summary(self) -> None:
        """
        Print comprehensive header summary with all details.

        Displays:
        - File information (path, size, ensemble count)
        - File integrity check results with status indicators
        - Data types present in first ensemble
        - Available data types across all ensembles
        - Overall file health

        Professional formatted output suitable for user reporting.

        Examples
        --------
        >>> ds = pyadps.read_header('test.000')
        >>> ds.header.summary()

        Output:
        ==================== FILE HEADER SUMMARY ====================
        Source File    : /path/to/deployment.000
        File Size      : 125.34 MB
        Total Ensembles: 10000

        ==================== INTEGRITY CHECK ====================
        File Size Match             :  PASS
        Byte Uniformity             :  PASS
        Datatype Uniformity         :  PASS
        Byte Skip Uniformity        :  PASS
        Address Offset Uniformity   :  PASS
        Data ID  Uniformity         :  PASS
        Overall Status              :  HEALTHY

        ==================== DATA TYPES (Ensemble 0) ====================
        - Fixed Leader
        - Variable Leader
        - Velocity
        - Correlation
        - Echo
        - Percent Good

        ============================================================
        """
        check = self.check_file()
        data_types = self.get_available_data_types(0)

        # Determine overall status based on all checks
        all_critical_pass = (
            check["File Size Match"]
            and check["Byte Uniformity"]
            and check["Data Type Uniformity"]
        )
        all_uniformity_pass = (
            check["Byte Skip Uniformity"]
            and check["Address Offset Uniformity"]
            and check["Data ID Uniformity"]
        )

        if not all_critical_pass:
            status = "error"
        elif not all_uniformity_pass:
            status = "warning"
        else:
            status = "healthy"

        print("=" * 70)
        print("FILE HEADER SUMMARY".center(70))
        print("=" * 70)

        # File information
        print(f"Source File    : {self._obj.attrs.get('filename', 'N/A')}")
        print(f"File Size      : {check['File Size (MB)']:.2f} MB")
        print(f"Total Ensembles: {self._obj.attrs.get('total_ensembles', 'N/A')}")
        print()

        # Integrity check - Critical checks
        print("=" * 70)
        print("INTEGRITY CHECK (CRITICAL)".center(70))
        print("=" * 70)

        # Status symbols for critical checks
        check_symbol = "PASS" if check["File Size Match"] else "FAIL"
        print(f"File Size Match             : {check_symbol}")

        check_symbol = "PASS" if check["Byte Uniformity"] else "FAIL"
        print(f"Byte Uniformity             : {check_symbol}")

        check_symbol = "PASS" if check["Data Type Uniformity"] else "FAIL"
        print(f"Datatype Uniformity         : {check_symbol}")

        check_symbol = "PASS" if check["Byte Skip Uniformity"] else "FAIL"
        print(f"Byte Skip Uniformity        : {check_symbol}")

        check_symbol = "PASS" if check["Address Offset Uniformity"] else "FAIL"
        print(f"Address Offset Uniformity   : {check_symbol}")

        check_symbol = "PASS" if check["Data ID Uniformity"] else "FAIL"
        print(f"Data ID Uniformity          : {check_symbol}")

        print(f"Overall Status              : {status.upper()}")
        print()

        # Data types in first ensemble
        print("=" * 70)
        print("DATA TYPES (Ensemble 0)".center(70))
        print("=" * 70)
        for dt in data_types:
            print(f"  - {dt}")
        print()

        print("=" * 70)


# ============================================================================
# FIXED LEADER ACCESSOR (Placeholder)
# ============================================================================


@xr.register_dataset_accessor("fixed_leader")
class FixedLeaderAccessor:
    """
    xarray accessor for ADCP FixedLeader data operations.

    FixedLeader data contains static configuration information that
    typically doesn't change during a deployment. This accessor provides
    methods for extracting, validating, and interpreting Fixed Leader data.

    Access via: ds.fixed_leader.*

    Methods
    -------
    field(ens=0)
        Extract all Fixed Leader fields for a single ensemble
    is_uniform()
        Check uniformity of each field across ensembles
    is_uniform_all (property)
        Check if all fields are uniform
    system_configuration(ens=-1)
        Get human-readable system configuration (backward compat)
    coordinate_transformation(ens=0)
        Get coordinate transformation settings
    sensor_info(ens=0, field='source')
        Get sensor availability or source selection
    validate()
        Validate Fixed Leader data integrity

    Examples
    --------
    >>> import pyadps
    >>> import pyadps.accessors
    >>> ds = pyadps.read_fixed_leader('test.000')
    >>> config = ds.fixed_leader.system_configuration(ens=0)
    >>> uniformity = ds.fixed_leader.is_uniform()
    >>> sensors = ds.fixed_leader.sensor_info(ens=0, field='source')
    """

    # Class-level cache for Fixed Leader field names
    _FIXED_LEADER_FIELDS_CACHE = None

    def __init__(self, xarray_obj: xr.Dataset):
        """Initialize the accessor with validation."""
        self._obj = xarray_obj
        self._fl_fields = None
        self._validate_dataset()

    @classmethod
    def _load_fl_fields_from_metadata(cls) -> set:
        """
        Load FixedLeader field names from fixed_leader_meta.json.

        Returns field names from both raw_fields and decoded_fields sections
        to support all FixedLeader variables (with or without decoded fields).

        Returns
        -------
        set
            Set of FixedLeader field names (raw + decoded)
        """
        if cls._FIXED_LEADER_FIELDS_CACHE is not None:
            return cls._FIXED_LEADER_FIELDS_CACHE

        fl_fields = set()

        try:
            # Try to import from pyadps utilities first
            from pyadps.io import binary_reader

            metadata = binary_reader._load_fixed_leader_metadata()

            for field_idx, field_data in metadata.items():
                if "name" in field_data:
                    fl_fields.add(field_data["name"])

            logger.debug(
                f"Loaded {len(fl_fields)} FixedLeader raw fields from metadata"
            )

        except Exception as e:
            logger.warning(f"Error loading FixedLeader fields from metadata: {e}")
            fl_fields = set()

        cls._FIXED_LEADER_FIELDS_CACHE = fl_fields
        return fl_fields

    def _get_fl_fields(self) -> List[str]:
        """
        Identify which fields in the dataset belong to FixedLeader.

        This method intelligently identifies FL fields based on:
        1. If pyadps_component == "FixedLeader": all variables are FL fields
        2. If merged dataset: only returns fields in metadata
        3. Supports both raw and decoded fields

        Returns
        -------
        list
            List of variable names that belong to FixedLeader component
        """
        if self._fl_fields is not None:
            return self._fl_fields

        # Case 1: Standalone FixedLeader dataset
        if self._obj.attrs.get("pyadps_component") == "FixedLeader":
            self._fl_fields = list(self._obj.data_vars)

        else:
            # Case 2: Merged dataset - check if FixedLeader was included
            components_str = self._obj.attrs.get("components", "")

            # Check if fixed_leader was included in the merge
            has_fixed_leader = (
                "fixed_leader" in components_str and "True" in components_str
            )

            if has_fixed_leader:
                # FixedLeader was included - load metadata and filter
                known_fl_fields = self._load_fl_fields_from_metadata()

                # Only return fields that exist in the dataset
                self._fl_fields = [
                    var for var in self._obj.data_vars if var in known_fl_fields
                ]
            else:
                # FixedLeader was not included in this merged dataset
                self._fl_fields = []

        return self._fl_fields

    def _validate_dataset(self) -> None:
        """Validate that dataset is a FixedLeader dataset."""
        fl_fields = self._get_fl_fields()
        if not fl_fields:
            logger.warning(
                "This accessor is optimized for FixedLeader datasets. "
                "Some methods may not work as expected."
            )

    def field(self, ens: int = 0) -> Dict[str, np.ndarray]:
        """
        Extract all Fixed Leader fields for a single ensemble.

        Parameters
        ----------
        ens : int, optional
            Ensemble number (0-indexed), by default 0 (first ensemble)

        Returns
        -------
        dict
            Dictionary mapping variable names to their values for the specified ensemble.

        Examples
        --------
        >>> ds = pyadps.read_fixed_leader('test.000')
        >>> ens_0 = ds.fixed_leader.field(ens=0)
        >>> print(ens_0['system_configuration_code'])
        """
        result = {}
        fl_fields = self._get_fl_fields()

        for var_name in fl_fields:
            if var_name not in self._obj.data_vars:
                continue
            result[var_name] = self._obj[var_name].values[ens]

        return result

    def is_uniform(self) -> Dict[str, bool]:
        """
        Check if each Fixed Leader field is uniform across all ensembles.

        Fixed Leader data should be constant for a deployment. This method
        verifies that each field has the same value across all ensembles.

        Returns
        -------
        dict
            Dictionary mapping variable names to uniformity (bool).
            True if all values identical, False if they vary.

        Examples
        --------
        >>> ds = pyadps.read_fixed_leader('test.000')
        >>> uniformity = ds.fixed_leader.is_uniform()
        >>> for field, is_unif in uniformity.items():
        ...     if not is_unif:
        ...         print(f"WARNING: {field} varies across ensembles!")
        """
        result = {}
        fl_fields = self._get_fl_fields()

        for var_name in fl_fields:
            if var_name not in self._obj.data_vars:
                continue

            values = self._obj[var_name].values
            # Check if all values are equal
            if len(values) == 0:
                result[var_name] = True
            else:
                # Handle different data types (string, numeric, etc.)
                try:
                    result[var_name] = bool(np.all(values == values[0]))
                except (TypeError, ValueError):
                    # For non-comparable types (e.g., object arrays)
                    result[var_name] = all(v == values[0] for v in values)

        return result

    @property
    def is_uniform_all(self) -> bool:
        """
        Check if ALL Fixed Leader fields are uniform (convenience property).

        Returns
        -------
        bool
            True if every field is uniform, False if any field varies.

        Examples
        --------
        >>> ds = pyadps.read_fixed_leader('test.000')
        >>> if ds.fixed_leader.is_uniform_all:
        ...     print("All FL fields are uniform (expected)")
        """
        uniformity = self.is_uniform()
        return all(uniformity.values()) if uniformity else True

    def system_configuration(self, ens: int = -1) -> Dict[str, str]:
        """
        Extract system configuration as human-readable dictionary.

        This method provides backward compatibility with v0.4.0. It decodes
        the system configuration code bits into human-readable format.

        Parameters
        ----------
        ens : int, optional
            Ensemble number (0-indexed). Use -1 (default) to get the most
            common configuration across all ensembles (useful if data varies).

        Returns
        -------
        dict
            Dictionary with keys: 'Frequency', 'Beam Pattern', 'Sensor Configuration',
            'XDCR HD', 'Beam Direction', 'Beam Angle', 'Janus Configuration'

        Raises
        ------
        ValueError
            If ens < -1 (invalid ensemble number)

        Examples
        --------
        >>> ds = pyadps.read_fixed_leader('test.000')
        >>> config = ds.fixed_leader.system_configuration(ens=0)
        >>> print(f"Frequency: {config['Frequency']}")
        Frequency: 300 kHz
        """
        if "system_configuration_code" not in self._obj.data_vars:
            raise ValueError(
                "system_configuration_code not found in dataset. "
                "Ensure include_decoded=True or raw field is available."
            )

        # Get system config code
        syscode = self._obj["system_configuration_code"].values

        if ens == -1:
            # Get most common value
            most_common_syscode = Counter(syscode).most_common()[0][0]
            binary_bits = format(int(most_common_syscode), "016b")
        elif ens >= 0:
            binary_bits = format(int(syscode[ens]), "016b")
        else:
            raise ValueError("Ensemble number should be greater than or equal to -1")

        # Lookup tables for decoding
        freq_code = {
            "000": "75 kHz",
            "001": "150 kHz",
            "010": "300 kHz",
            "011": "600 kHz",
            "100": "1200 kHz",
            "101": "2400 kHz",
            "110": "38 kHz",
        }

        beam_code = {"0": "Concave", "1": "Convex"}

        sensor_code = {
            "00": "#1",
            "01": "#2",
            "10": "#3",
            "11": "Sensor configuration not found",
        }

        xdcr_code = {"0": "Not attached", "1": "Attached"}

        dir_code = {"0": "Down", "1": "Up"}

        angle_code = {
            "0000": "15",
            "0001": "20",
            "0010": "30",
            "0011": "Other beam angle",
            "0111": "25",
            "1100": "45",
        }

        janus_code = {
            "0100": "4 Beam",
            "0101": "5 Beam CFIG DEMOD",
            "1111": "5 Beam CFIG 2 DEMOD",
        }

        # Extract and decode bits
        sys_cfg = {}

        # Bits 13-15: Frequency (3 bits)
        bit_group = binary_bits[13:16]
        sys_cfg["Frequency"] = freq_code.get(bit_group, "Frequency not found")

        # Bit 12: Beam Pattern (1 bit)
        bit_group = binary_bits[12]
        sys_cfg["Beam Pattern"] = beam_code.get(bit_group)

        # Bits 10-11: Sensor Configuration (2 bits)
        bit_group = binary_bits[10:12]
        sys_cfg["Sensor Configuration"] = sensor_code.get(bit_group)

        # Bit 9: XDCR HD (1 bit)
        bit_group = binary_bits[9]
        sys_cfg["XDCR HD"] = xdcr_code.get(bit_group)

        # Bit 8: Beam Direction (1 bit)
        bit_group = binary_bits[8]
        sys_cfg["Beam Direction"] = dir_code.get(bit_group)

        # Bits 4-7: Beam Angle (4 bits)
        bit_group = binary_bits[4:8]
        sys_cfg["Beam Angle"] = angle_code.get(bit_group, "Angle not found")

        # Bits 0-3: Janus Configuration (4 bits)
        bit_group = binary_bits[0:4]
        sys_cfg["Janus Configuration"] = janus_code.get(
            bit_group, "Janus cfg. not found"
        )

        return sys_cfg

    def coordinate_transformation(self, ens: int = 0) -> Dict[str, object]:
        """
        Extract coordinate transformation configuration from Fixed Leader data.

        This method decodes the coordinate transform code bits into
        human-readable format.

        Parameters
        ----------
        ens : int, optional
            Ensemble number (0-indexed), by default 0 (first ensemble)

        Returns
        -------
        dict
            Dictionary with keys: 'Coordinates', 'Tilt Correction',
            'Three-Beam Solution', 'Bin Mapping'
            Values are strings for Coordinates, booleans for others.

        Raises
        ------
        ValueError
            If ensemble number is out of range
        IndexError
            If coordinate_transformation_code not in dataset

        Examples
        --------
        >>> ds = pyadps.read_fixed_leader('test.000')
        >>> transform = ds.fixed_leader.coordinate_transformation(ens=0)
        >>> print(f"Coordinates: {transform['Coordinates']}")
        Coordinates: Earth Coordinates
        """
        if "coordinate_transformation_code" not in self._obj.data_vars:
            raise ValueError(
                "coordinate_transformation_code not found in dataset. "
                "Ensure raw field is available."
            )

        # Get transform code value and convert to binary
        transform_code = self._obj["coordinate_transformation_code"].values[ens]
        bit_group = format(int(transform_code), "08b")

        # Lookup tables for decoding
        trans_code = {
            "00": "Beam Coordinates",
            "01": "Instrument Coordinates",
            "10": "Ship Coordinates",
            "11": "Earth Coordinates",
        }

        bool_code = {"1": True, "0": False}

        transform = {}
        transform["Coordinates"] = trans_code.get(bit_group[3:5])
        transform["Tilt Correction"] = bool_code.get(bit_group[5])
        transform["Three-Beam Solution"] = bool_code.get(bit_group[6])
        transform["Bin Mapping"] = bool_code.get(bit_group[7])

        return transform

    def sensor_info(self, ens: int = 0, field: str = "source") -> Dict[str, bool]:
        """
        Extract sensor availability or source selection from Fixed Leader data.

        This method decodes sensor code bits into human-readable format.
        Can extract either sensor source selection or sensor availability.

        Parameters
        ----------
        ens : int, optional
            Ensemble number (0-indexed), by default 0 (first ensemble)
        field : str, optional
            Field to extract - 'source' or 'avail', by default "source"
            - 'source': Which sensors are selected as input sources
            - 'avail': Which sensors are available

        Returns
        -------
        dict
            Dictionary mapping sensor names to availability/selection (bool).
            Keys: 'Sound Speed', 'Depth Sensor', 'Heading Sensor',
            'Pitch Sensor', 'Roll Sensor', 'Conductivity Sensor',
            'Temperature Sensor'

        Raises
        ------
        ValueError
            If field is not 'source' or 'avail'
        ValueError
            If required sensor code field not in dataset

        Examples
        --------
        >>> ds = pyadps.read_fixed_leader('test.000')
        >>> sensors = ds.fixed_leader.sensor_info(ens=0, field='source')
        >>> print(f"Sound Speed: {sensors['Sound Speed']}")
        Sound Speed: True

        >>> availability = ds.fixed_leader.sensor_info(ens=0, field='avail')
        """
        if field == "source":
            field_name = "sensor_source_code"
        elif field == "avail":
            field_name = "sensor_available_code"
        else:
            raise ValueError(f"field must be 'source' or 'avail', got '{field}'")

        if field_name not in self._obj.data_vars:
            raise ValueError(
                f"{field_name} not found in dataset. "
                f"Ensure raw fields are available."
            )

        # Get sensor code value and convert to binary
        sensor_code_val = self._obj[field_name].values[ens]
        bit_group = format(int(sensor_code_val), "08b")

        # Boolean lookup table
        bool_code = {"1": True, "0": False}

        sensor = {}
        sensor["Sound Speed"] = bool_code.get(bit_group[1])
        sensor["Depth Sensor"] = bool_code.get(bit_group[2])
        sensor["Heading Sensor"] = bool_code.get(bit_group[3])
        sensor["Pitch Sensor"] = bool_code.get(bit_group[4])
        sensor["Roll Sensor"] = bool_code.get(bit_group[5])
        sensor["Conductivity Sensor"] = bool_code.get(bit_group[6])
        sensor["Temperature Sensor"] = bool_code.get(bit_group[7])

        return sensor

    def validate(self) -> Dict[str, any]:
        """
        Validate Fixed Leader data integrity.

        Performs comprehensive validation checks including:
        - All required fields are present
        - Field dimensions are consistent
        - Uniformity of data across ensembles (warning if not uniform)
        - Data type validity

        Returns
        -------
        dict
            Validation report with keys:
            - 'valid': bool, True if all checks pass
            - 'issues': list of strings describing any issues found
            - 'warnings': list of strings for non-critical issues

        Examples
        --------
        >>> ds = pyadps.read_fixed_leader('test.000')
        >>> report = ds.fixed_leader.validate()
        >>> if report['valid']:
        ...     print("Data is valid!")
        >>> else:
        ...     for issue in report['issues']:
        ...         print(f"ERROR: {issue}")
        """
        report = {"valid": True, "issues": [], "warnings": []}

        # Get only Fixed Leader fields from the dataset
        # This ensures we only validate FL fields, not other components (Variable Leader, Velocity, etc.)
        fl_fields = self._get_fl_fields()

        if not fl_fields:
            report["valid"] = False
            report["issues"].append("No Fixed Leader fields found in dataset")
            return report

        # Check minimum required fields (must be in the FL fields list)
        required_fields = [
            "system_configuration_code",
            "coordinate_transformation_code",
        ]

        missing_fields = [f for f in required_fields if f not in self._obj.data_vars]
        if missing_fields:
            report["valid"] = False
            report["issues"].append(
                f"Missing required Fixed Leader fields: {', '.join(missing_fields)}"
            )

        # Check dimension consistency - only for Fixed Leader fields
        if fl_fields:
            # Get the first FL field to use as reference
            first_fl_field = None
            first_fl_dims = None

            for var_name in fl_fields:
                if var_name in self._obj.data_vars:
                    if first_fl_field is None:
                        first_fl_field = var_name
                        first_fl_dims = self._obj[var_name].dims
                    else:
                        # Compare with reference
                        var_dims = self._obj[var_name].dims
                        if var_dims != first_fl_dims:
                            report["valid"] = False
                            report["issues"].append(
                                f"Dimension mismatch in Fixed Leader: {var_name} has dims {var_dims}, "
                                f"expected {first_fl_dims}"
                            )

        # Check uniformity and warn if not uniform
        # is_uniform() already uses _get_fl_fields() internally, so it only checks FL fields
        uniformity = self.is_uniform()
        non_uniform_fields = [k for k, v in uniformity.items() if not v]
        if non_uniform_fields:
            report["warnings"].append(
                f"Non-uniform Fixed Leader fields (should be constant): {', '.join(non_uniform_fields)}"
            )

        return report

    def summary(self) -> None:
        """
        Print comprehensive Fixed Leader summary with all details.

        Displays:
        - System configuration (decoded fields)
        - Coordinate transformation settings
        - Sensor availability and source selection
        - Uniformity status for key fields
        - Data integrity check results
        - Overall file health

        Professional formatted output suitable for user reporting.

        Examples
        --------
        >>> ds = pyadps.read_fixed_leader('test.000')
        >>> ds.fixed_leader.summary()

        Output:
        ==================== FIXED LEADER SUMMARY ====================
        System Configuration
          Frequency         : 300 kHz
          Beam Pattern      : Concave
          Sensor Config     : Configuration #1
          XDCR HD           : Not attached
          Beam Direction    : Downward
          Beam Angle        : 20°
          Janus Config      : Type 2

        Coordinate Transformation
          Coordinates       : Earth Coordinates
          Tilt Correction   : Enabled
          Three-Beam Soln   : Disabled
          Bin Mapping       : Disabled

        Sensor Configuration
          Temperature       : Available
          Depth             : Available
          Heading           : Available
          Pitch             : Available
          Roll              : Available
          Conductivity      : Not Available

        Data Uniformity Across Ensembles
          System Config     : UNIFORM
          Coordinate Trans  : UNIFORM

        Data Integrity
          Required fields   : ✓ PASS
          All uniform       : ✓ PASS
        ===============================================================
        """
        # Get all the information we need
        try:
            system_config = self.system_configuration(ens=-1)
        except (ValueError, IndexError):
            system_config = None

        try:
            coord_transform = self.coordinate_transformation(ens=-1)
        except (ValueError, IndexError):
            coord_transform = None

        try:
            sensor_source = self.sensor_info(ens=-1, field="source")
        except (ValueError, IndexError):
            sensor_source = None

        try:
            sensor_avail = self.sensor_info(ens=-1, field="avail")
        except (ValueError, IndexError):
            sensor_avail = None

        uniformity = self.is_uniform()
        validation = self.validate()

        # Print header
        print("=" * 70)
        print("FIXED LEADER SUMMARY".center(70))
        print("=" * 70)
        print()

        # System Configuration
        if system_config:
            print("System Configuration (Mode)")
            print(f"  Frequency         : {system_config.get('Frequency', 'N/A')}")
            print(f"  Beam Pattern      : {system_config.get('Beam Pattern', 'N/A')}")
            print(
                f"  Sensor Config     : {system_config.get('Sensor Configuration', 'N/A')}"
            )
            print(f"  XDCR HD           : {system_config.get('XDCR HD', 'N/A')}")
            print(f"  Beam Direction    : {system_config.get('Beam Direction', 'N/A')}")
            print(f"  Beam Angle        : {system_config.get('Beam Angle', 'N/A')}")
            print(
                f"  Janus Config      : {system_config.get('Janus Configuration', 'N/A')}"
            )
            print()

        # Coordinate Transformation
        if coord_transform:
            print("Coordinate Transformation (Mode)")
            print(f"  Coordinates       : {coord_transform.get('Coordinates', 'N/A')}")
            tilt_str = (
                "Enabled"
                if coord_transform.get("Tilt Correction", False)
                else "Disabled"
            )
            print(f"  Tilt Correction   : {tilt_str}")
            three_beam_str = (
                "Enabled"
                if coord_transform.get("Three-Beam Solution", False)
                else "Disabled"
            )
            print(f"  Three-Beam Soln   : {three_beam_str}")
            bin_map_str = (
                "Enabled" if coord_transform.get("Bin Mapping", False) else "Disabled"
            )
            print(f"  Bin Mapping       : {bin_map_str}")
            print()

        # Sensor Configuration
        if sensor_source or sensor_avail:
            print("Sensor Configuration (First Ensemble)")
            if sensor_source:
                print("  Sensor Sources:")
                for sensor, active in sensor_source.items():
                    status = "Selected" if active else "Not selected"
                    print(f"    {sensor:<20}: {status}")
            if sensor_avail:
                print("  Sensor Availability:")
                for sensor, avail in sensor_avail.items():
                    status = "Available" if avail else "Not available"
                    print(f"    {sensor:<20}: {status}")
            print()

        # Data Uniformity
        if uniformity:
            print("Data Uniformity Across Ensembles")
            for field_name, is_uniform in uniformity.items():
                status = "UNIFORM" if is_uniform else "VARIES"
                print(f"  {field_name:<25}: {status}")
            print()

        # Data Integrity
        print("Data Integrity")
        all_pass = len(validation["issues"]) == 0
        check_symbol = "PASS" if all_pass else "FAIL"
        print(f"  Required fields   : {check_symbol}")

        all_uniform = all(uniformity.values()) if uniformity else False
        check_symbol = "PASS" if all_uniform else "WARNING"
        print(f"  Field uniformity  : {check_symbol}")

        # Overall status
        print()
        if validation["valid"] and all_uniform:
            overall_status = "HEALTHY"
        elif validation["valid"]:
            overall_status = "WARNING"
        else:
            overall_status = "ERROR"
        print(f"  Overall Status    : {overall_status}")

        # Print issues and warnings if any
        if validation["issues"]:
            print()
            print("Issues:")
            for issue in validation["issues"]:
                print(f"  {issue}")

        if validation["warnings"]:
            print()
            print("Warnings:")
            for warning in validation["warnings"]:
                print(f"   {warning}")

        print()
        print("=" * 70)


# ============================================================================
# VARIABLE LEADER ACCESSOR FOR ACCESSORS.PY
# ============================================================================
# This provides domain-specific methods for Variable Leader data analysis
# including timestamp composition, motion sensor queries, diagnostic decoding,
# and environmental data retrieval.
# ============================================================================


@xr.register_dataset_accessor("variable_leader")
class VariableLeaderAccessor:
    """
    xarray accessor for ADCP VariableLeader data operations.

    VariableLeader data contains dynamic measurements that change with each
    ensemble (time-varying sensor data). This accessor provides methods for
    validating, analyzing, and reporting on Variable Leader data including
    timestamps, diagnostic information, and ensemble continuity.

    Access via: ds.variable_leader.*

    Methods
    -------
    ensemble_rollover_count()
        Detect and count ensemble number rollovers
    ensemble_continuity_check()
        Check for gaps or discontinuities in ensemble numbering
    bit_result_summary(include_decoded=True)
        Summarize Built-In Test (BIT) results with 8 detailed checks
    error_status_word_summary(include_decoded=True)
        Summarize Error Status Word (ESW) occurrences with 32 detailed checks
    validate_timestamps()
        Validate RTC timestamp fields for plausibility
    summary()
        Print comprehensive Variable Leader summary

    Examples
    --------
    >>> import pyadps
    >>> import pyadps.accessors
    >>> ds = pyadps.read_variable_leader('test.000')
    >>> continuity = ds.variable_leader.ensemble_continuity_check()
    >>> bit_check = ds.variable_leader.bit_result_summary()
    >>> esw_check = ds.variable_leader.error_status_word_summary()
    >>> ds.variable_leader.summary()
    """

    def __init__(self, xarray_obj: xr.Dataset):
        """Initialize the accessor with validation."""
        self._obj = xarray_obj
        self._validate_dataset()

    def _validate_dataset(self) -> None:
        """Validate that dataset is a VariableLeader dataset."""
        if self._obj.attrs.get("pyadps_component") != "VariableLeader":
            logger.warning(
                "This accessor is optimized for VariableLeader datasets. "
                "Some methods may not work as expected."
            )

    def ensemble_rollover_count(self) -> int:
        """
        Detect and count ensemble number rollovers.

        The ensemble counter is 16-bit (0-65535), with MSB tracking the
        full 24-bit counter (0-16,777,215). Count rollovers as an indicator
        of data continuity and potential file size.

        Returns
        -------
        int
            Number of ensemble rollovers detected

        Notes
        -----
        Full ensemble number = ensemble_msb * 65536 + ensemble_number
        """
        ensemble_num = self._obj["ensemble_number"].values
        ensemble_msb = self._obj["ensemble_msb"].values

        # Full ensemble number = MSB * 65536 + LSB
        full_ensemble = ensemble_msb * 65536 + ensemble_num

        # Count discontinuities (rollovers or gaps)
        diff = np.diff(full_ensemble)
        rollovers = np.sum(diff < -1000)  # Large negative jump = rollover

        return int(rollovers)

    def ensemble_continuity_check(self) -> Dict:
        """
        Check for gaps or discontinuities in ensemble numbering.

        Validates that ensemble numbers are sequential and identifies any
        gaps or jumps that might indicate dropped data or data corruption.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'is_continuous': bool (True if no gaps detected)
            - 'gap_count': int (number of detected gaps)
            - 'gap_locations': list of int (ensemble indices where gaps occur)
            - 'gap_sizes': list of int (ensemble counts skipped at each gap)

        Examples
        --------
        >>> continuity = ds.variable_leader.ensemble_continuity_check()
        >>> if continuity['is_continuous']:
        ...     print("Data is continuous")
        ... else:
        ...     print(f"Found {continuity['gap_count']} gaps")
        """
        ensemble_num = self._obj["ensemble_number"].values
        ensemble_msb = self._obj["ensemble_msb"].values
        full_ensemble = ensemble_msb * 65536 + ensemble_num

        diff = np.diff(full_ensemble)
        expected_diff = np.ones_like(diff)

        gaps = np.where(diff != expected_diff)[0]

        return {
            "is_continuous": len(gaps) == 0,
            "gap_count": len(gaps),
            "gap_locations": [int(i) for i in gaps],
            "gap_sizes": [int(diff[i] - 1) for i in gaps],
        }

    def bit_result_summary(self, include_decoded: bool = True) -> Dict:
        """
        Summarize Built-In Test (BIT) results with 8 detailed bit-level checks.

        Provides 8 detailed checks corresponding to each bit in the 16-bit BIT
        result field. Expected: all zeros (successful tests). Non-zero values
        indicate hardware or firmware errors during instrument operation.

        Parameters
        ----------
        include_decoded : bool, optional
            If True (default), include analysis of decoded bit fields for more
            detailed error interpretation. When True, uses decoded binary fields
            like 'bit_demod_0_error' for precise bit analysis.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'all_passed': bool (True if all BIT results == 0)
            - 'error_count': int (number of ensembles with errors)
            - 'unique_error_codes': list (distinct error values seen)
            - 'bit_checks': dict with 8 checks:
              - 'demod_0_error': dict with error_count only
              - 'demod_1_error': dict with error_count only
              - 'timing_card_error': dict with error_count only
              - 'reserved_1_error': dict with error_count only
              - 'reserved_2_error': dict with error_count only
              - 'reserved_3_error': dict with error_count only
              - 'reserved_4_error': dict with error_count only
              - 'reserved_5_error': dict with error_count only

        Notes
        -----
        BIT results of 0 indicate successful tests. Each of the 8 bits represents
        a specific hardware test. The 'include_decoded' parameter controls whether
        to use decoded bit fields (more accurate) or bit extraction (fallback).

        Examples
        --------
        >>> bit_status = ds.variable_leader.bit_result_summary()
        >>> if not bit_status['all_passed']:
        ...     for bit_name, check in bit_status['bit_checks'].items():
        ...         if check['error_count'] > 0:
        ...             print(f"{bit_name}: {check['error_count']} errors")
        """
        bit_result = self._obj["bit_result"].values

        errors = bit_result != 0
        unique_errors = (
            np.unique(bit_result[errors]).tolist()
            if len(bit_result[errors]) > 0
            else []
        )

        # Detailed bit checks (8 total)
        bit_checks = {}

        # Bit 0: Demod 0 Error
        if include_decoded and "bit_demod_0_error" in self._obj.data_vars:
            bit_values = self._obj["bit_demod_0_error"].values
        else:
            bit_values = (bit_result & 0x0001) != 0
        bit_checks["demod_0_error"] = {
            "error_count": int(np.sum(bit_values)),
        }

        # Bit 1: Demod 1 Error
        if include_decoded and "bit_demod_1_error" in self._obj.data_vars:
            bit_values = self._obj["bit_demod_1_error"].values
        else:
            bit_values = (bit_result & 0x0002) != 0
        bit_checks["demod_1_error"] = {
            "error_count": int(np.sum(bit_values)),
        }

        # Bit 2: Timing Card Error
        if include_decoded and "bit_timing_card_error" in self._obj.data_vars:
            bit_values = self._obj["bit_timing_card_error"].values
        else:
            bit_values = (bit_result & 0x0004) != 0
        bit_checks["timing_card_error"] = {
            "error_count": int(np.sum(bit_values)),
        }

        # Bits 3-7: Reserved
        for bit_pos, bit_name in [
            (3, "reserved_1"),
            (4, "reserved_2"),
            (5, "reserved_3"),
            (6, "reserved_4"),
            (7, "reserved_5"),
        ]:
            decoded_name = f"bit_{bit_name}_error"
            if include_decoded and decoded_name in self._obj.data_vars:
                bit_values = self._obj[decoded_name].values
            else:
                bit_values = (bit_result & (1 << bit_pos)) != 0
            bit_checks[f"{bit_name}_error"] = {
                "error_count": int(np.sum(bit_values)),
            }

        return {
            "all_passed": np.all(bit_result == 0),
            "error_count": int(np.sum(errors)),
            "unique_error_codes": unique_errors,
            "bit_checks": bit_checks,
        }

    def error_status_word_summary(self, include_decoded: bool = True) -> Dict:
        """
        Summarize Error Status Word (ESW) occurrences with 32 detailed bit checks.

        Analyzes all 4 Error Status Words (32 bits total) with detailed per-bit
        checking. Expected: all zeros (no errors/events). Non-zero bits indicate
        events such as clock adjustments, wakeup events, or power anomalies.

        Parameters
        ----------
        include_decoded : bool, optional
            If True (default), include analysis of decoded bit fields for precise
            event interpretation. When True, uses decoded fields like
            'esw1_clock_read_error' for detailed bit analysis.

        Returns
        -------
        dict
            Dictionary organized by ESW (ESW1, ESW2, ESW3, ESW4).
            Each ESW contains:
            - 'all_zeros': bool (True if no events in this ESW)
            - 'total_events': int (total ensembles with events)
            - 'bit_checks': dict with up to 8 bit checks per ESW:
              - Each bit has 'event_count' only (no error_indices)

        ESW1 Bits (8 checks):
          - address_error_exception
          - illegal_instruction_exception
          - emulator_exception
          - bus_error_exception
          - watchdog_restart
          - zero_divide_exception
          - unassigned_exception
          - battery_saver_power

        ESW2 Bits (8 checks):
          - pinging
          - cold_wakeup_occurred
          - unknown_wakeup_occurred
          - (and 5 reserved/unused bits)

        ESW3 Bits (8 checks):
          - clock_read_error
          - unexpected_alarm
          - clock_jump_forward
          - clock_jump_backward
          - (and 4 reserved/unused bits)

        ESW4 Bits (8 checks):
          - power_fail_unrecorded
          - spurious_level_4_intr_dsp
          - spurious_level_5_intr_uart
          - spurious_level_6_intr_clock
          - level_7_interrupt
          - (and 3 reserved/unused bits)

        Notes
        -----
        The 'include_decoded' parameter controls whether to use decoded bit fields
        (more accurate and human-readable) or bit extraction (fallback method).

        Examples
        --------
        >>> esw_status = ds.variable_leader.error_status_word_summary()
        >>> for esw_label, esw_data in esw_status.items():
        ...     if not esw_data['all_zeros']:
        ...         print(f"{esw_label}: {esw_data['total_events']} events")
        ...         for bit_name, bit_info in esw_data['bit_checks'].items():
        ...             if bit_info['event_count'] > 0:
        ...                 print(f"  {bit_name}: {bit_info['event_count']}")
        """
        # ESW field mappings with bit name patterns
        esw_configs = {
            "ESW1": {
                "field": "error_status_word_1",
                "bits": [
                    (0, "address_error_exception", "esw1_address_error_exception"),
                    (
                        1,
                        "illegal_instruction_exception",
                        "esw1_illegal_instruction_exception",
                    ),
                    (2, "emulator_exception", "esw1_emulator_exception"),
                    (3, "bus_error_exception", "esw1_bus_error_exception"),
                    (4, "watchdog_restart", "esw1_watchdog_restart"),
                    (5, "zero_divide_exception", "esw1_zero_divide_exception"),
                    (6, "unassigned_exception", "esw1_unassigned_exception"),
                    (7, "battery_saver_power", "esw1_battery_saver_power"),
                ],
            },
            "ESW2": {
                "field": "error_status_word_2",
                "bits": [
                    (0, "pinging", "esw2_pinging"),
                    (1, "cold_wakeup_occurred", "esw2_cold_wakeup_occurred"),
                    (2, "unknown_wakeup_occurred", "esw2_unknown_wakeup_occurred"),
                    (3, "not_used_1", "esw2_not_used_1"),
                    (4, "not_used_2", "esw2_not_used_2"),
                    (5, "not_used_3", "esw2_not_used_3"),
                    (6, "not_used_4", "esw2_not_used_4"),
                    (7, "not_used_5", "esw2_not_used_5"),
                ],
            },
            "ESW3": {
                "field": "error_status_word_3",
                "bits": [
                    (0, "clock_read_error", "esw3_clock_read_error"),
                    (1, "unexpected_alarm", "esw3_unexpected_alarm"),
                    (2, "clock_jump_forward", "esw3_clock_jump_forward"),
                    (3, "clock_jump_backward", "esw3_clock_jump_backward"),
                    (4, "not_used_6", "esw3_not_used_6"),
                    (5, "not_used_7", "esw3_not_used_7"),
                    (6, "not_used_8", "esw3_not_used_8"),
                    (7, "not_used_9", "esw3_not_used_9"),
                ],
            },
            "ESW4": {
                "field": "error_status_word_4",
                "bits": [
                    (0, "not_used_10", "esw4_not_used_10"),
                    (1, "not_used_11", "esw4_not_used_11"),
                    (2, "not_used_12", "esw4_not_used_12"),
                    (3, "power_fail_unrecorded", "esw4_power_fail_unrecorded"),
                    (4, "spurious_level_4_intr_dsp", "esw4_spurious_level_4_intr_dsp"),
                    (
                        5,
                        "spurious_level_5_intr_uart",
                        "esw4_spurious_level_5_intr_uart",
                    ),
                    (
                        6,
                        "spurious_level_6_intr_clock",
                        "esw4_spurious_level_6_intr_clock",
                    ),
                    (7, "level_7_interrupt", "esw4_level_7_interrupt"),
                ],
            },
        }

        summary = {}

        for esw_label, config in esw_configs.items():
            field_name = config["field"]

            if field_name not in self._obj.data_vars:
                continue

            esw_data = self._obj[field_name].values
            total_events = np.sum(esw_data != 0)

            bit_checks = {}

            # Analyze each of 8 bits
            for bit_pos, bit_name, decoded_field_name in config["bits"]:
                # Try to use decoded field if available and requested
                if include_decoded and decoded_field_name in self._obj.data_vars:
                    bit_values = self._obj[decoded_field_name].values
                else:
                    # Fall back to bit extraction
                    bit_values = (esw_data & (1 << bit_pos)) != 0

                event_count = int(np.sum(bit_values))

                bit_checks[bit_name] = {
                    "event_count": event_count,
                }

            summary[esw_label] = {
                "all_zeros": np.all(esw_data == 0),
                "total_events": int(total_events),
                "bit_checks": bit_checks,
            }

        return summary

    def validate_timestamps(self) -> Dict:
        """
        Validate RTC timestamp fields for plausibility.

        Checks that timestamp components fall within valid ranges:
        - Month: 1-12
        - Day: 1-31
        - Hour: 0-23
        - Minute: 0-59
        - Second: 0-59

        Returns
        -------
        dict
            Dictionary with validation results:
            - 'valid': bool (True if all timestamps are plausible)
            - 'issues': list of str (descriptions of any problems found)

        Notes
        -----
        This performs range checking only. It does not validate exact day
        counts per month or leap year considerations. For rigorous timestamp
        validation, use external datetime libraries after composing full
        datetime objects.
        """
        issues = []

        # Check month validity
        month = self._obj["rtc_month"].values
        if np.any((month < 1) | (month > 12)):
            invalid_months = np.unique(month[(month < 1) | (month > 12)])
            issues.append(f"Invalid months: {invalid_months.tolist()}")

        # Check day validity (simplified)
        day = self._obj["rtc_day"].values
        if np.any((day < 1) | (day > 31)):
            invalid_days = np.unique(day[(day < 1) | (day > 31)])
            issues.append(f"Invalid days: {invalid_days.tolist()}")

        # Check hour validity
        hour = self._obj["rtc_hour"].values
        if np.any((hour < 0) | (hour > 23)):
            invalid_hours = np.unique(hour[(hour < 0) | (hour > 23)])
            issues.append(f"Invalid hours: {invalid_hours.tolist()}")

        # Check minute validity
        minute = self._obj["rtc_minute"].values
        if np.any((minute < 0) | (minute > 59)):
            invalid_minutes = np.unique(minute[(minute < 0) | (minute > 59)])
            issues.append(f"Invalid minutes: {invalid_minutes.tolist()}")

        # Check second validity
        second = self._obj["rtc_second"].values
        if np.any((second < 0) | (second > 59)):
            invalid_seconds = np.unique(second[(second < 0) | (second > 59)])
            issues.append(f"Invalid seconds: {invalid_seconds.tolist()}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
        }

    def is_time_regular(self, tolerance_s: float = 1.0) -> bool:
        """
        Check if time axis is regular (constant intervals between ensembles).

        Compares the common time interval with all observed intervals. If all
        intervals are within the specified tolerance of the most common interval,
        the time axis is considered regular.

        Parameters
        ----------
        tolerance_s : float, default 1.0
            Maximum allowed deviation from the most common interval, in seconds.
            Time axis is regular if max deviation <= tolerance_s.

        Returns
        -------
        bool
            True if time intervals are regular within tolerance, False otherwise.

        Examples
        --------
        >>> ds = pyadps.read('file.000')
        >>> is_regular = ds.variable_leader.is_time_regular()
        >>> if is_regular:
        ...     print("Time axis is regular")
        ... else:
        ...     print("Time axis has irregular intervals")

        >>> # Check with stricter tolerance (0.1 seconds)
        >>> is_regular = ds.variable_leader.is_time_regular(tolerance_s=0.1)
        """
        if len(self._obj.time) < 2:
            return False

        # Calculate all time differences
        time_series = pd.Series(self._obj.time.values)
        diffs = time_series.diff().dropna()

        if diffs.empty:
            return False

        # Find the most common interval
        common_interval = diffs.mode()[0]

        # Check maximum deviation from common interval
        max_deviation = (diffs - common_interval).abs().max().total_seconds()

        return max_deviation <= tolerance_s

    def get_time_interval(self) -> Optional[pd.Timedelta]:
        """
        Get the most common time interval between consecutive ensembles.

        Calculates the time differences between all consecutive ensembles
        and returns the most frequently occurring interval. This is useful
        for understanding the sampling rate of the ADCP.

        Returns
        -------
        pd.Timedelta
            The most common time interval between ensembles.
            Returns None if less than 2 time points or all times are identical.

        Examples
        --------
        >>> ds = pyadps.read('file.000')
        >>> interval = ds.variable_leader.get_time_interval()
        >>> print(f"Sampling interval: {interval}")
        Sampling interval: 0 days 00:01:00

        >>> # Check if hourly sampling
        >>> if interval == pd.Timedelta(hours=1):
        ...     print("Hourly sampling detected")
        """
        if len(self._obj.time) < 2:
            return None

        time_series = pd.Series(self._obj.time.values)
        diffs = time_series.diff().dropna()

        if diffs.empty:
            return None

        # Return the most common interval
        common_interval = diffs.mode()
        if len(common_interval) > 0:
            return common_interval.iloc[0]
        return None

    def get_time_interval_frequency(self) -> Dict[str, int]:
        """
        Get frequency distribution of all time intervals between ensembles.

        Calculates time differences between consecutive ensembles and counts
        how often each interval occurs. Useful for diagnosing irregular time
        spacing or detecting sampling rate changes.

        Returns
        -------
        dict
            Mapping of time interval strings (HH:MM:SS format) to frequency counts.
            Example: {'00:00:01': 120, '00:00:02': 5} means 120 ensembles have
            1-second intervals and 5 have 2-second intervals.

        Examples
        --------
        >>> ds = pyadps.read('file.000')
        >>> freq = ds.variable_leader.get_time_interval_frequency()
        >>> print(freq)
        {'00:01:00': 120}  # 120 ensembles with 1-minute intervals

        >>> # Detect irregular intervals
        >>> if len(freq) > 1:
        ...     print(f"Warning: {len(freq)} different intervals detected")
        ...     for interval, count in sorted(freq.items()):
        ...         print(f"  {interval}: {count} occurrences")
        """
        if len(self._obj.time) < 2:
            return {}

        time_series = pd.Series(self._obj.time.values)
        diffs = time_series.diff().dropna()

        if diffs.empty:
            return {}

        # Convert timedeltas to string format (HH:MM:SS) for display
        # Count frequencies
        freq_counts = diffs.value_counts().sort_index()

        # Convert to dictionary with string keys
        result = {}
        for td, count in freq_counts.items():
            # Format timedelta as HH:MM:SS
            td_cast = pd.Timedelta(td)
            total_seconds = int(td_cast.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            result[time_str] = int(count)

        return result

    def get_time_component_frequency(self, component: str = "minute") -> Dict[int, int]:
        """
        Calculate the frequency of a time component (hour, minute, or second).

        Extracts a specific component from each timestamp and counts how many
        ensembles occur at each value. Useful for detecting sampling patterns,
        such as whether measurements occur at regular minute marks.

        Parameters
        ----------
        component : {'hour', 'minute', 'second'}, default 'minute'
            The time component to analyze.
            - 'hour': Extract hour of day (0-23)
            - 'minute': Extract minute of hour (0-59)
            - 'second': Extract second of minute (0-59)

        Returns
        -------
        dict
            Mapping of component value to frequency (count).
            Example for component='minute': {0: 120, 30: 5} means 120 ensembles
            at minute 0 and 5 ensembles at minute 30.

        Raises
        ------
        ValueError
            If component is not one of {'hour', 'minute', 'second'}.

        Examples
        --------
        >>> ds = pyadps.read('file.000')
        >>> freq = ds.variable_leader.get_time_component_frequency('minute')
        >>> print(freq)
        {0: 120}  # All 120 ensembles at minute 0 of the hour

        >>> # Check for regular hourly sampling
        >>> second_freq = ds.variable_leader.get_time_component_frequency('second')
        >>> if second_freq == {0: len(ds.time)}:
        ...     print("Measurements occur at top of each second")

        >>> # Detect non-uniform minute distribution
        >>> minute_freq = ds.variable_leader.get_time_component_frequency('minute')
        >>> if len(minute_freq) == 1:
        ...     print(f"All ensembles at minute {list(minute_freq.keys())[0]}")
        ... else:
        ...     print(f"Ensembles spread across {len(minute_freq)} different minutes")
        """
        valid_components = {"hour", "minute", "second"}
        if component not in valid_components:
            raise ValueError(
                f"component must be one of {valid_components}, got '{component}'"
            )

        # Convert time coordinate to DatetimeIndex for easy component extraction
        time_index = pd.DatetimeIndex(self._obj.time.values)

        # Extract the specified component and count frequencies
        if component == "hour":
            freq = time_index.hour.value_counts().sort_index()
        elif component == "minute":
            freq = time_index.minute.value_counts().sort_index()
        else:  # second
            freq = time_index.second.value_counts().sort_index()

        # Return as dictionary for consistency with v0.4.0
        return freq.to_dict()

    def summary(self) -> None:
        """
        Print comprehensive Variable Leader summary with all details.

        Displays:
        - Ensemble continuity and rollover information
        - BIT test results with 8 detailed bit-level checks
        - Error Status Word events with 32 detailed bit-level checks
        - Timestamp validity

        Professional formatted output suitable for user reporting.

        Examples
        --------
        >>> ds = pyadps.read_variable_leader('test.000')
        >>> ds.variable_leader.summary()
        """
        report = "=" * 70 + "\n"
        report += "VARIABLE LEADER DATA QUALITY REPORT\n"
        report += "=" * 70 + "\n\n"

        # Continuity
        cont = self.ensemble_continuity_check()
        report += "ENSEMBLE CONTINUITY:\n"
        report += f"  Continuous: {cont['is_continuous']}\n"
        if cont["gap_count"] > 0:
            report += f"  Gaps: {cont['gap_count']}\n"
            gap_locs = cont["gap_locations"][:10]
            report += f"  Gap locations: {gap_locs}"
            if len(cont["gap_locations"]) > 10:
                report += "..."
            report += "\n"
        report += "\n"

        # BIT Results with 8 detailed checks
        bit = self.bit_result_summary()
        report += "BIT TEST RESULTS (8 Detailed Checks):\n"
        report += f"  All Passed: {bit['all_passed']}\n"
        report += f"  Total Errors: {bit['error_count']} ensembles\n"
        if bit["unique_error_codes"]:
            report += f"  Error Codes: {bit['unique_error_codes']}\n"
        report += "\n  Bit-Level Details:\n"
        for bit_name, bit_check in sorted(bit["bit_checks"].items()):
            report += f"    {bit_name}: {bit_check['error_count']} errors\n"
        report += "\n"

        # Error Status Words with 32 detailed checks (8 bits x 4 words)
        esw = self.error_status_word_summary()
        report += "ERROR STATUS WORDS (32 Detailed Checks):\n"
        for esw_label in ["ESW1", "ESW2", "ESW3", "ESW4"]:
            if esw_label not in esw:
                continue
            esw_data = esw[esw_label]
            report += f"\n  {esw_label} - Total Events: {esw_data['total_events']}\n"
            report += f"    All Zeros: {esw_data['all_zeros']}\n"
            if not esw_data["all_zeros"]:
                report += "    Bit-Level Details:\n"
                for bit_name, bit_check in sorted(esw_data["bit_checks"].items()):
                    if bit_check["event_count"] > 0:
                        report += (
                            f"      {bit_name}: {bit_check['event_count']} events\n"
                        )
        report += "\n"

        # Time Analysis
        report += "TIME ANALYSIS:\n"

        # Timestamp Validation
        ts_valid = self.validate_timestamps()
        report += f"  Timestamps Valid: {ts_valid['valid']}\n"
        if ts_valid["issues"]:
            for issue in ts_valid["issues"]:
                report += f"    Issue: {issue}\n"

        # Temporal Patterns
        is_regular = self.is_time_regular()
        report += f"  Regular Intervals: {is_regular}\n"
        time_interval = self.get_time_interval()
        if time_interval is not None:
            report += f"  Most Common Interval: {time_interval}\n"
        else:
            report += "  Most Common Interval: N/A (insufficient data)\n"
        report += "\n"

        report += "=" * 70 + "\n"

        print(report)


# ============================================================================
# END OF VARIABLE LEADER ACCESSOR
# ============================================================================
