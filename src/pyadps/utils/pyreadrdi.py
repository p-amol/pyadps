"""
pyreadrdi.py

Module Overview
---------------
This module provides functionalities to read and parse RDI ADCP files.
It includes functions for reading file headers, fixed and variable leaders,
and data types like velocity, correlation, echo intensity, and percent good.
Currently reads only PD0 format.

Modules
-------
- fileheader: Function to read and parse the file header information.
- fixedleader: Function to read and parse the fixed leader section of an RDI file.
- variableleader: Function to read and parse the variable leader section of an RDI file.
- datatype: Function to read and parse 3D data types.
- ErrorCode: Enum class to define and manage error codes for file operations.

Creation Date
--------------
2024-09-01

Last Modified Date
--------------
2025-10-21

Version
-------
0.3.0

Author
------
[P. Amol] <prakashamol@gmail.com>

License
-------
This module is licensed under the MIT License. See LICENSE file for details.

Dependencies
------------
- numpy: Required for handling array operations.
- struct: Required for unpacking binary data.
- io: Provides file handling capabilities, including file-like object support.
- enum: Provides support for creating enumerations, used for defining error codes.

Usage
-----
To use this module, import the necessary functions as follows:

>>> from readrdi import fileheader, fixedleader, variableleader, datatype

Examples
--------
>>> header_data = fileheader('example.rdi')
>>> fixed_data, ensemble, error_code = fixedleader('example.rdi')
>>> var_data = variableleader('example.rdi')
>>> vel_data = datatype('example.rdi', "velocity")
>>> vel_data = datatype('example.rdi', "echo", beam=4, cell=20)

Other add-on functions and classes inlcude bcolors, safe_open, and ErrorCode.
Examples (add-on)
-------------------
>>> error = ErrorCode.FILE_NOT_FOUND
"""

import io
import os
import sys
from enum import Enum
from pathlib import Path
from struct import error as StructError
from struct import unpack
from typing import BinaryIO, Literal, NamedTuple, Optional, Tuple, Union, cast

import numpy as np

# Type aliases for clarity
FilePathType = Union[str, Path]
SafeOpenReturn = Tuple[Optional[BinaryIO], "ErrorCode"]
SafeReadReturn = Tuple[Optional[bytes], "ErrorCode"]
FileHeaderReturn = Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int
]
LeaderReturn = Tuple[np.ndarray, int, int]
DataTypeReturn = Union[
    Tuple[np.ndarray, int],
    Tuple[np.ndarray, int, np.ndarray, np.ndarray, int],
]


class bcolors:
    """
    ANSI terminal color codes for console output styling.

    This class provides color codes and formatting options for ANSI-compatible
    terminals. Use these codes to colorize text by prepending the code and
    appending `ENDC` to reset formatting.

    Attributes
    ----------
    HEADER : str
        Magenta text, typically used for headers.
    OKBLUE : str
        Blue text for general information.
    OKCYAN : str
        Cyan text for informational messages.
    OKGREEN : str
        Green text for success messages.
    WARNING : str
        Yellow text for warnings.
    FAIL : str
        Red text for errors or failures.
    ENDC : str
        Reset code to default color and formatting.
    BOLD : str
        Bold text formatting.
    UNDERLINE : str
        Underlined text formatting.

    Examples
    --------
    >>> print(f"{bcolors.OKGREEN}Success{bcolors.ENDC}")
    >>> print(f"{bcolors.FAIL}Error{bcolors.ENDC}")

    Notes
    -----
    Not all terminals support ANSI escape sequences. Appearance varies by
    terminal emulator. Test before relying on color in production scripts.
    """

    HEADER: str = "\033[95m"
    OKBLUE: str = "\033[94m"
    OKCYAN: str = "\033[96m"
    OKGREEN: str = "\033[92m"
    WARNING: str = "\033[93m"
    FAIL: str = "\033[91m"
    ENDC: str = "\033[0m"
    BOLD: str = "\033[1m"
    UNDERLINE: str = "\033[4m"


class ErrorCode(Enum):
    """
    Standardized error codes and messages for RDI file operations.

    This enum provides consistent error reporting throughout the module,
    replacing numeric error codes with semantic meaning.

    Attributes
    ----------
    SUCCESS : tuple
        Operation completed successfully (code 0).
    FILE_NOT_FOUND : tuple
        File does not exist (code 1).
    PERMISSION_DENIED : tuple
        Access denied (code 2).
    IO_ERROR : tuple
        File open failed (code 3).
    OUT_OF_MEMORY : tuple
        Insufficient memory (code 4).
    WRONG_RDIFILE_TYPE : tuple
        File type not recognized as RDI (code 5).
    ID_NOT_FOUND : tuple
        Data type ID not found (code 6).
    FILE_CORRUPTED : tuple
        File structure invalid (code 8).
    DATATYPE_MISMATCH : tuple
        Data type inconsistent with previous ensemble (code 7).
    VALUE_ERROR : tuple
        Invalid argument provided (code 9).
    UNKNOWN_ERROR : tuple
        Unspecified or unexpected error (code 99).

    Methods
    -------
    get_message(code: int) -> str
        Retrieve the message corresponding to an error code.
    """

    SUCCESS = (0, "Success")
    FILE_NOT_FOUND = (1, "Error: File not found.")
    PERMISSION_DENIED = (2, "Error: Permission denied.")
    IO_ERROR = (3, "IO Error: Unable to open file.")
    OUT_OF_MEMORY = (4, "Error: Out of memory.")
    WRONG_RDIFILE_TYPE = (5, "Error: Wrong RDI File Type.")
    ID_NOT_FOUND = (6, "Error: Data type ID not found.")
    DATATYPE_MISMATCH = (7, "Warning: Data type mismatch.")
    FILE_CORRUPTED = (8, "Warning: File Corrupted.")
    VALUE_ERROR = (9, "Value Error for incorrect argument.")
    UNKNOWN_ERROR = (99, "Unknown error.")

    def __init__(self, code: int, message: str) -> None:
        """
        Initialize ErrorCode enum member.

        Parameters
        ----------
        code : int
            The error code number.
        message : str
            The descriptive error message.
        """
        self.code: int = code
        self.message: str = message

    @classmethod
    def get_message(cls, code: int) -> str:
        """
        Retrieve the descriptive message corresponding to a given error code.

        Parameters
        ----------
        code : int
            The error code for which the message is to be retrieved.

        Returns
        -------
        str
            The descriptive message associated with the provided error code.
            Returns "Error: Invalid error code." if the code is not valid.
        """
        for error in cls:
            if error.code == code:
                return error.message
        return "Error: Invalid error code."


def safe_open(filename: FilePathType, mode: str = "rb") -> SafeOpenReturn:
    """
    Safely open a binary file with exception handling.

    Attempts to open a file and handles common exceptions gracefully,
    returning both the file object and an error code.

    Args
    ----
    filename : str or Path
        Path to the file to open.
    mode : str, optional
        File mode. Defaults to "rb" (read binary).

    Returns
    -------
    tuple[BinaryIO | None, ErrorCode]
        File object (None on error) and ErrorCode enum.

    Raises
    ------
    No exceptions raised. All errors are caught and returned as ErrorCode.

    Examples
    --------
    >>> f, error = safe_open("data.bin")
    >>> if error == ErrorCode.SUCCESS:
    ...     data = f.read()
    ...     f.close()
    """
    try:
        filename_str: str = os.path.abspath(str(filename))
        file: BinaryIO = cast(BinaryIO, open(filename_str, mode))
        return (file, ErrorCode.SUCCESS)
    except FileNotFoundError as e:
        print(f"FileNotFoundError: The file '{filename}' was not found: {e}")
        return (None, ErrorCode.FILE_NOT_FOUND)
    except PermissionError as e:
        print(f"PermissionError: Permission denied for '{filename}': {e}")
        return (None, ErrorCode.PERMISSION_DENIED)
    except IOError as e:
        print(f"IOError: An error occurred trying to open '{filename}': {e}")
        return (None, ErrorCode.IO_ERROR)
    except MemoryError as e:
        print(f"MemoryError: Out of memory '{filename}':{e}")
        return (None, ErrorCode.OUT_OF_MEMORY)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return (None, ErrorCode.UNKNOWN_ERROR)


def safe_read(bfile: BinaryIO, num_bytes: int) -> SafeReadReturn:
    """
    Read specified bytes from a binary file with validation.

    Reads exactly `num_bytes` from the file. Returns error if fewer bytes
    are available (EOF reached unexpectedly).

    Args
    ----
    bfile : BinaryIO
        Open binary file object.
    num_bytes : int
        Number of bytes to read.

    Returns
    -------
    tuple[bytes | None, ErrorCode]
        Bytes read (None on error) and ErrorCode enum.

    Examples
    --------
    >>> f, _ = safe_open("data.bin")
    >>> data, error = safe_read(f, 100)
    >>> if error == ErrorCode.SUCCESS:
    ...     print(len(data))  # Output: 100
    """

    try:
        readbytes: bytes = bfile.read(num_bytes)

        if len(readbytes) != num_bytes:
            print(f"Unexpected end of file: fewer than {num_bytes} bytes were read.")
            return (None, ErrorCode.FILE_CORRUPTED)
        else:
            return (readbytes, ErrorCode.SUCCESS)

    except (IOError, OSError) as e:
        print(f"File read error: {e}")
        return (None, ErrorCode.IO_ERROR)
    except ValueError as e:
        print(f"Value error: {e}")
        return (None, ErrorCode.VALUE_ERROR)


def fileheader(rdi_file: FilePathType) -> FileHeaderReturn:
    """
    Parse RDI file header to extract ensemble metadata.

    Reads the file header and builds arrays mapping ensemble number to file
    locations and data type information. Required as first step before
    reading Fixed/Variable Leaders or data types.

    Args
    ----
    rdi_file : str or Path
        Path to RDI binary file in PD0 format.

    Returns
    -------
    tuple
        (datatype, byte, byteskip, address_offset, dataid, ensemble, error_code)

        - datatype : np.ndarray (int16, shape (n_ensembles,))
            Number of data type records in each ensemble.
        - byte : np.ndarray (int16, shape (n_ensembles,))
            Byte count for each ensemble.
        - byteskip : np.ndarray (int32, shape (n_ensembles,))
            File offset (bytes from start) to each ensemble.
        - address_offset : np.ndarray (int, shape (n_ensembles, n_types))
            Byte offset within ensemble for each data type.
        - dataid : np.ndarray (int, shape (n_ensembles, n_types))
            Data type ID for each position in ensemble.
        - ensemble : int
            Number of ensembles successfully parsed.
        - error_code : int
            0 on success, non-zero ErrorCode.code value on error.

    Raises
    ------
    No exceptions. All errors returned via error_code.

    Notes
    -----
    - Mandatory checksum verification (RDI spec Section 7.2).
    - File may be truncated; returns partial data with appropriate error_code.
    - Data type IDs: 0-1 (Fixed Leader), 128-129 (Variable Leader),
      256-257 (Velocity), 512-513 (Correlation), 768-769 (Echo Intensity),
      1024-1025 (Percent Good), 1280-1281 (Status).

    Examples
    --------
    >>> dt, byte, skip, offset, ids, n_ens, err = fileheader("test.000")
    >>> if err == 0:
    ...     print(f"Parsed {n_ens} ensembles")
    """

    filename: str = str(rdi_file)
    headerid: np.ndarray = np.array([], dtype="int8")
    sourceid: np.ndarray = np.array([], dtype="int8")
    byte: np.ndarray = np.array([], dtype="int16")
    spare: np.ndarray = np.array([], dtype="int8")
    datatype: np.ndarray = np.array([], dtype="int16")
    address_offset: list[tuple[int, ...]] = []
    ensemble: int = 0
    error_code: int = 0
    dataid: list[list[int]] = []
    byteskip: np.ndarray = np.array([], dtype="int32")

    bfile: Optional[BinaryIO]
    bfile, error = safe_open(filename, mode="rb")
    if bfile is None:
        error_code = error.code
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            0,
            error_code,
        )

    bfile.seek(0, 0)
    bskip: int = 0
    i: int = 0
    hid: list[int] = [0] * 5

    try:
        while byt := bfile.read(6):
            hid[0], hid[1], hid[2], hid[3], hid[4] = unpack("<BBHBB", byt)
            headerid = np.append(headerid, np.int8(hid[0]))
            sourceid = np.append(sourceid, np.int16(hid[1]))
            byte = np.append(byte, np.int16(hid[2]))
            spare = np.append(spare, np.int16(hid[3]))
            datatype = np.append(datatype, np.int16(hid[4]))

            # Read data bytes based on data type count
            dbyte: Optional[bytes]
            dbyte, error = safe_read(bfile, 2 * datatype[i])
            if dbyte is None:
                if i == 0:
                    error_code = error.code
                    return (
                        np.array([]),
                        np.array([]),
                        np.array([]),
                        np.array([]),
                        np.array([]),
                        0,
                        error_code,
                    )
                else:
                    break

            # Check for id and datatype errors
            if i == 0:
                if headerid[0] != 127 or sourceid[0] != 127:
                    error = ErrorCode.WRONG_RDIFILE_TYPE
                    print(bcolors.FAIL + error.message + bcolors.ENDC)
                    error_code = error.code
                    return (
                        np.array([]),
                        np.array([]),
                        np.array([]),
                        np.array([]),
                        np.array([]),
                        0,
                        error_code,
                    )
            else:
                if headerid[i] != 127 or sourceid[i] != 127:
                    error = ErrorCode.ID_NOT_FOUND
                    print(bcolors.FAIL + error.message)
                    print(f"Ensembles reset to {i}" + bcolors.ENDC)
                    break

                if datatype[i] != datatype[i - 1]:
                    error = ErrorCode.DATATYPE_MISMATCH
                    print(bcolors.FAIL + error.message)
                    print(f"Data Types for ensemble {i} is {datatype[i - 1]}.")
                    print(f"Data Types for ensemble {i + 1} is {datatype[i]}.")
                    print(f"Ensembles reset to {i}" + bcolors.ENDC)
                    break

            try:
                data: tuple[int, ...] = unpack("H" * datatype[i], dbyte)
                address_offset.append(data)
            except Exception:
                error = ErrorCode.FILE_CORRUPTED
                error_code = error.code
                return (
                    np.array([]),
                    np.array([]),
                    np.array([]),
                    np.array([]),
                    np.array([]),
                    0,
                    error_code,
                )

            skip_array: list[int] = [0] * datatype[i]
            for dtype in range(datatype[i]):
                bseek: int = int(bskip) + int(address_offset[i][dtype])
                bfile.seek(bseek, 0)
                readbyte: bytes = bfile.read(2)
                skip_array[dtype] = int.from_bytes(
                    readbyte, byteorder="little", signed=False
                )

            dataid.append(skip_array)
            # bytekip is the number of bytes to skip to reach
            # an ensemble from beginning of file.
            # ?? Should byteskip be from current position ??
            bskip = int(bskip) + int(byte[i]) + 2
            bfile.seek(bskip, 0)
            byteskip = np.append(byteskip, np.int32(bskip))
            i += 1

    except (ValueError, StructError, OverflowError) as e:
        # except:
        print(bcolors.WARNING + "WARNING: The file is broken.")
        print(
            f"Function `fileheader` unable to extract data for ensemble {i + 1}. Total ensembles reset to {i}."
        )
        print(bcolors.UNDERLINE + "Details from struct function" + bcolors.ENDC)
        print(f"  Error Type: {type(e).__name__}")
        print(f"  Error Details: {e}")
        error = ErrorCode.FILE_CORRUPTED
        ensemble = i

    ensemble = i
    bfile.close()
    address_offset_array: np.ndarray = np.array(address_offset)
    dataid_array: np.ndarray = np.array(dataid)
    datatype = datatype[0:ensemble]
    byte = byte[0:ensemble]
    byteskip = byteskip[0:ensemble]
    error_code = error.code

    return (
        datatype,
        byte,
        byteskip,
        address_offset_array,
        dataid_array,
        ensemble,
        error_code,
    )


def fixedleader(
    rdi_file: FilePathType,
    byteskip: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    idarray: Optional[np.ndarray] = None,
    ensemble: int = 0,
) -> LeaderReturn:
    """
    Extract Fixed Leader data from RDI file.

    Reads Fixed Leader section (ID 0-1) containing system configuration,
    serial numbers, and sensor information. Parameters byteskip, offset,
    idarray can be provided from fileheader() for efficiency; otherwise
    fileheader() is called internally.

    Args
    ----
    rdi_file : str or Path
        Path to RDI binary file.
    byteskip : np.ndarray, optional
        Array of file offsets from fileheader(). If None, fileheader()
        is called internally.
    offset : np.ndarray, optional
        Address offsets from fileheader(). If None, fileheader() is
        called internally.
    idarray : np.ndarray, optional
        Data type IDs from fileheader(). If None, fileheader() is
        called internally.
    ensemble : int, optional
        Number of ensembles from fileheader(). If 0, fileheader() is
        called internally.

    Returns
    -------
    tuple
        (data, ensemble, error_code)

        - data : np.ndarray (int64, shape (36, n_ensembles))
            Fixed Leader fields: ID, CPU version, system config, beam/cell
            counts, pings per ensemble, sensor info, serial numbers, etc.
        - ensemble : int
            Number of ensembles successfully parsed (may be less than input).
        - error_code : int
            0 on success, non-zero on error.

    Notes
    -----
    - Handles missing serial numbers from old firmware (replaced with flag 0).
    - If called without optional parameters, it automatically calls fileheader().

    Examples
    --------
    >>> fl_data, n_ens, err = fixedleader("test.000")
    >>> if err == 0:
    ...     n_beams = fl_data[6, 0]  # Beam count from first ensemble
    ...     n_cells = fl_data[7, 0]  # Cell count from first ensemble
    """

    filename: str = str(rdi_file)
    error_code: int = 0
    error: ErrorCode = ErrorCode.SUCCESS

    if (
        not all((isinstance(v, np.ndarray) for v in (byteskip, offset, idarray)))
        or ensemble == 0
    ):
        _, _, byteskip, offset, idarray, ensemble, error_code = fileheader(filename)

    # Type narrowing: Cast to ndarray after initialization
    byteskip = cast(np.ndarray, byteskip)
    offset = cast(np.ndarray, offset)
    idarray = cast(np.ndarray, idarray)

    fid: np.ndarray = np.zeros((36, ensemble), dtype="int64")

    bfile: Optional[BinaryIO]
    bfile, error = safe_open(filename, "rb")
    if bfile is None:
        return (fid, ensemble, error.code)
    if error.code == 0 and error_code != 0:
        error.code = error_code

    # Note: When processing data from ADCPs with older firmware,
    # the instrument serial number may be missing. As a result,
    # garbage value is recorded, which sometimes is too large for a standard 64-bit integer.
    # The following variables are defined to replace garbage value with a missing value.
    # Flag to track if a missing serial number is detected
    is_serial_missing: bool = False
    # Define the maximum value for a standard signed int64
    INT64_MAX: int = 2**63 - 1
    # Define a missing value flag (0 is a safe unsigned integer choice)
    MISSING_VALUE_FLAG: int = 0

    bfile.seek(0, 0)
    for i in range(ensemble):
        fbyteskip: Optional[int] = None
        for count, item in enumerate(idarray[i]):
            if item in (0, 1):
                fbyteskip = offset[i][count]
        if fbyteskip is None:
            error = ErrorCode.ID_NOT_FOUND
            ensemble = i
            print(bcolors.WARNING + error.message)
            print(f"Total ensembles reset to {i}." + bcolors.ENDC)
            break
        else:  # inserted
            try:
                bfile.seek(fbyteskip, 1)
                bdata: bytes = bfile.read(59)
                # Fixed Leader ID, CPU Version no. & Revision no.
                (fid[0][i], fid[1][i], fid[2][i]) = unpack("<HBB", bdata[0:4])
                if fid[0][i] not in (0, 1):
                    error = ErrorCode.ID_NOT_FOUND
                    ensemble = i
                    print(bcolors.WARNING + error.message)
                    print(f"Total ensembles reset to {i}." + bcolors.ENDC)
                    break
                # System configuration & Real/Slim flag
                (fid[3][i], fid[4][i]) = unpack("<HB", bdata[4:7])
                # Lag Length, number of beams & Number of cells
                (fid[5][i], fid[6][i], fid[7][i]) = unpack("<BBB", bdata[7:10])
                # Pings per Ensemble, Depth cell length & Blank after transmit
                (fid[8][i], fid[9][i], fid[10][i]) = unpack("<HHH", bdata[10:16])
                # Signal Processing mode, Low correlation threshold & No. of
                # code repetition
                (fid[11][i], fid[12][i], fid[13][i]) = unpack("<BBB", bdata[16:19])
                # Percent good minimum & Error velocity threshold
                (fid[14][i], fid[15][i]) = unpack("<BH", bdata[19:22])
                # Time between ping groups (TP command)
                # Minute, Second, Hundredth
                (fid[16][i], fid[17][i], fid[18][i]) = unpack("<BBB", bdata[22:25])
                # Coordinate transform, Heading alignment & Heading bias
                (fid[19][i], fid[20][i], fid[21][i]) = unpack("<BHH", bdata[25:30])
                # Sensor source & Sensor available
                (fid[22][i], fid[23][i]) = unpack("<BB", bdata[30:32])
                # Bin 1 distance, Transmit pulse length & Reference layer ave
                (fid[24][i], fid[25][i], fid[26][i]) = unpack("<HHH", bdata[32:38])
                # False target threshold, Spare & Transmit lag distance
                (fid[27][i], fid[28][i], fid[29][i]) = unpack("<BBH", bdata[38:42])
                # CPU board serial number (Big Endian)
                try:
                    (fid[30][i]) = unpack(">Q", bdata[42:50])[0]
                    # Check for overflow only once to set the flag
                    if not is_serial_missing and fid[30][i] > INT64_MAX:
                        print(
                            bcolors.WARNING
                            + "WARNING: Invalid serial number detected (old firmware). Flagging for replacement."
                            + "DETAILS: Value exceeds expected range."
                            + bcolors.ENDC
                        )
                        is_serial_missing = True
                except (ValueError, OverflowError) as e:
                    if not is_serial_missing:
                        print(
                            bcolors.WARNING
                            + "WARNING: Failed to read serial number (old firmware). Flagging for replacement. \n"
                            + f"DETAILS: {e}"
                            + bcolors.ENDC
                        )
                        is_serial_missing = True
                # System bandwidth, system power & Spare
                (fid[31][i], fid[32][i], fid[33][i]) = unpack("<HBB", bdata[50:54])
                # Instrument serial number & Beam angle
                (fid[34][i], fid[35][i]) = unpack("<LB", bdata[54:59])

                bfile.seek(byteskip[i], 0)

            except (ValueError, StructError, OverflowError) as e:
                print(bcolors.WARNING + "WARNING: The file is broken.")
                print(
                    f"Function `fixedleader` unable to extract data for ensemble {i + 1}. Total ensembles reset to {i}."
                )
                print(bcolors.UNDERLINE + "Details from struct function" + bcolors.ENDC)
                print(f"  Error Type: {type(e).__name__}")
                print(f"  Error Details: {e}")
                error = ErrorCode.FILE_CORRUPTED
                ensemble = i

            except (OSError, io.UnsupportedOperation) as e:
                print(bcolors.WARNING + "WARNING: The file is broken.")
                print(
                    f"Function `fixedleader` unable to extract data for ensemble {i + 1}. Total ensembles reset to {i}."
                )
                print(f"File seeking error at iteration {i}: {e}" + bcolors.ENDC)
                error = ErrorCode.FILE_CORRUPTED
                ensemble = i

    bfile.close()
    error_code = error.code

    if is_serial_missing:
        print(
            bcolors.OKBLUE
            + "INFO: Replacing entire serial number array with missing value flag."
            + bcolors.ENDC
        )
        # If Serial No. is missing, flag all data after Serial No.
        fid[30, :] = MISSING_VALUE_FLAG  # Serial No.
        fid[31, :] = MISSING_VALUE_FLAG  # System Bandwidth
        fid[32, :] = MISSING_VALUE_FLAG  # System Power
        fid[33, :] = MISSING_VALUE_FLAG  # Spare 2
        fid[34, :] = MISSING_VALUE_FLAG  # Instrument No
        fid[35, :] = MISSING_VALUE_FLAG  # Beam Angle

    data: np.ndarray = fid[:, :ensemble]
    return (data, ensemble, error_code)


def variableleader(
    rdi_file: FilePathType,
    byteskip: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    idarray: Optional[np.ndarray] = None,
    ensemble: int = 0,
) -> LeaderReturn:
    """
    Extract Variable Leader data from RDI file.

    Reads Variable Leader section (ID 128-129) containing time, motion
    sensors (heading, pitch, roll), depth, temperature, and status for
    each ensemble.

    Args
    ----
    rdi_file : str or Path
        Path to RDI binary file.
    byteskip : np.ndarray, optional
        Array of file offsets from fileheader(). Auto-fetched if None.
    offset : np.ndarray, optional
        Address offsets from fileheader(). Auto-fetched if None.
    idarray : np.ndarray, optional
        Data type IDs from fileheader(). Auto-fetched if None.
    ensemble : int, optional
        Number of ensembles from fileheader(). Auto-fetched if 0.

    Returns
    -------
    tuple
        (data, ensemble, error_code)

        - data : np.ndarray (int32, shape (48, n_ensembles))
            Variable Leader fields: ID, RTC (year/month/day/time),
            heading, pitch, roll, temperature, salinity, pressure,
            motion sensor std devs, ADC readings, error status, etc.
        - ensemble : int
            Number of ensembles parsed.
        - error_code : int
            0 on success.

    Examples
    --------
    >>> vl_data, n_ens, err = variableleader("test.000")
    >>> if err == 0:
    ...     year = vl_data[2, 0]    # Year
    ...     month = vl_data[3, 0]   # Month
    ...     heading = vl_data[13, 0]  # Heading (0.01° units)
    """

    filename: str = str(rdi_file)
    error_code: int = 0
    error: ErrorCode = ErrorCode.SUCCESS

    if (
        not all((isinstance(v, np.ndarray) for v in (byteskip, offset, idarray)))
        or ensemble == 0
    ):
        _, _, byteskip, offset, idarray, ensemble, error_code = fileheader(filename)

    # Type narrowing: Cast to ndarray after initialization
    byteskip = cast(np.ndarray, byteskip)
    offset = cast(np.ndarray, offset)
    idarray = cast(np.ndarray, idarray)

    vid: np.ndarray = np.zeros((48, ensemble), dtype="int32")

    bfile: Optional[BinaryIO]
    bfile, error = safe_open(filename, "rb")
    if bfile is None:
        return (vid, ensemble, error.code)

    if error.code == 0 and error_code != 0:
        error.code = error_code

    bfile.seek(0, 0)
    for i in range(ensemble):
        fbyteskip: Optional[int] = None
        for count, item in enumerate(idarray[i]):
            if item in (128, 129):
                fbyteskip = offset[i][count]

        if fbyteskip is None:
            error = ErrorCode.ID_NOT_FOUND
            ensemble = i
            print(bcolors.WARNING + error.message)
            print(f"Total ensembles reset to {i}." + bcolors.ENDC)
            break
        else:
            try:
                bfile.seek(fbyteskip, 1)
                bdata: bytes = bfile.read(65)
                vid[0][i], vid[1][i] = unpack("<HH", bdata[0:4])
                if vid[0][i] not in (128, 129):
                    error = ErrorCode.ID_NOT_FOUND
                    ensemble = i
                    print(bcolors.WARNING + error.message)
                    print(f"Total ensembles reset to {i}." + bcolors.ENDC)
                    break
                # Extract WorkHorse ADCPâ€™s real-time clock (RTC)
                # Year, Month, Day, Hour, Minute, Second & Hundredth
                (
                    vid[2][i],
                    vid[3][i],
                    vid[4][i],
                    vid[5][i],
                    vid[6][i],
                    vid[7][i],
                    vid[8][i],
                ) = unpack("<BBBBBBB", bdata[4:11])
                # Extract Ensemble # MSB & BIT Result
                (vid[9][i], vid[10][i]) = unpack("<BH", bdata[11:14])
                # Extract sensor variables (directly or derived):
                # Sound Speed, Transducer Depth, Heading,
                # Pitch, Roll, Temperature & Salinity
                (
                    vid[11][i],
                    vid[12][i],
                    vid[13][i],
                    vid[14][i],
                    vid[15][i],
                    vid[16][i],
                    vid[17][i],
                ) = unpack("<HHHhhHh", bdata[14:28])
                # Extract [M]inimum Pre-[P]ing Wait [T]ime between ping groups
                # MPT minutes, MPT seconds & MPT hundredth
                (vid[18][i], vid[19][i], vid[20][i]) = unpack("<BBB", bdata[28:31])
                # Extract standard deviation of motion sensors:
                # Heading, Pitch, & Roll
                (vid[21][i], vid[22][i], vid[23][i]) = unpack("<BBB", bdata[31:34])
                # Extract ADC Channels (8)
                (
                    vid[24][i],
                    vid[25][i],
                    vid[26][i],
                    vid[27][i],
                    vid[28][i],
                    vid[29][i],
                    vid[30][i],
                    vid[31][i],
                ) = unpack("<BBBBBBBB", bdata[34:42])
                # Extract error status word (4)
                (vid[32][i], vid[33][i], vid[34][i], vid[35][i]) = unpack(
                    "<BBBB", bdata[42:46]
                )
                # Extract Reserved, Pressure, Pressure Variance & Spare
                (vid[36][i], vid[37][i], vid[38][i], vid[39][i]) = unpack(
                    "<HiiB", bdata[46:57]
                )
                # Extract Y2K time
                # Century, Year, Month, Day, Hour, Minute, Second, Hundredth
                (
                    vid[40][i],
                    vid[41][i],
                    vid[42][i],
                    vid[43][i],
                    vid[44][i],
                    vid[45][i],
                    vid[46][i],
                    vid[47][i],
                ) = unpack("<BBBBBBBB", bdata[57:65])

                bfile.seek(byteskip[i], 0)

            except (ValueError, StructError, OverflowError) as e:
                print(bcolors.WARNING + "WARNING: The file is broken.")
                print(
                    f"Function `variableleader` unable to extract data for ensemble {i + 1}. Total ensembles reset to {i}."
                )
                print(bcolors.UNDERLINE + "Details from struct function" + bcolors.ENDC)
                print(f"  Error Type: {type(e).__name__}")
                print(f"  Error Details: {e}")
                error = ErrorCode.FILE_CORRUPTED
                ensemble = i

            except (OSError, io.UnsupportedOperation) as e:
                print(bcolors.WARNING + "WARNING: The file is broken.")
                print(
                    f"Function `variableleader` unable to extract data for ensemble {i + 1}. Total ensembles reset to {i}."
                )
                print(f"File seeking error at iteration {i}: {e}" + bcolors.ENDC)
                error = ErrorCode.FILE_CORRUPTED
                ensemble = i

    bfile.close()
    error_code = error.code
    data: np.ndarray = vid[:, :ensemble]
    return (data, ensemble, error_code)


def datatype(
    filename: FilePathType,
    var_name: Literal["velocity", "correlation", "echo", "percent good", "status"],
    cell: Union[int, np.ndarray] = 0,
    beam: Union[int, np.ndarray] = 0,
    byteskip: Optional[np.ndarray] = None,
    offset: Optional[np.ndarray] = None,
    idarray: Optional[np.ndarray] = None,
    ensemble: int = 0,
) -> DataTypeReturn:
    """
    Extract 3D data arrays (velocity, correlation, echo, etc.) from RDI file.

    Reads ensemble data for variables that vary by beam and cell:
    velocity (16-bit), correlation, echo intensity, percent good, and
    status (all 8-bit).

    Args
    ----
    filename : str or Path
        Path to RDI binary file.
    var_name : str
        Variable to extract. One of: 'velocity', 'correlation', 'echo',
        'percent good', 'status'.
    cell : int or np.ndarray, optional
        Cell counts per ensemble. If int/0, fetched from fixedleader().
    beam : int or np.ndarray, optional
        Beam counts per ensemble. If int/0, fetched from fixedleader().
    byteskip : np.ndarray, optional
        File offsets from fileheader(). Auto-fetched if None.
    offset : np.ndarray, optional
        Address offsets from fileheader(). Auto-fetched if None.
    idarray : np.ndarray, optional
        Data type IDs from fileheader(). Auto-fetched if None.
    ensemble : int, optional
        Number of ensembles. Auto-fetched if 0.

    Returns
    -------
    tuple
        (data, ensemble, cell_array, beam_array, error_code)

        - data : np.ndarray (shape (max_beam, max_cell, n_ensembles))
            Data values. Velocity is int16 (-32768 = missing), others uint8.
            Dimensions padded to max across all ensembles.
        - ensemble : int
            Ensembles parsed (may be less than input if corrupted).
        - cell_array : np.ndarray (int, shape (n_ensembles,))
            Cell count for each ensemble.
        - beam_array : np.ndarray (int, shape (n_ensembles,))
            Beam count for each ensemble.
        - error_code : int
            0 on success.

    Raises
    ------
    No exceptions. Errors returned via error_code.

    Examples
    --------
    >>> vel, n_ens, cells, beams, err = datatype("test.000", "velocity")
    >>> if err == 0:
    ...     print(f"Shape: {vel.shape}")  # (n_beams, n_cells, n_ensembles)
    ...     v_beam0_cell0 = vel[0, 0, :]  # Velocity time series
    """

    varid: dict[str, tuple[int, ...]] = {
        "velocity": (256, 257),
        "correlation": (512, 513),
        "echo": (768, 769),
        "percent good": (1024, 1025),
        "status": (1280, 1281),
    }
    error_code: int = 0

    # Check for optional arguments.
    # These arguments are outputs of fileheader function.
    # Makes the code faster if the fileheader function is already executed.
    if (
        not all((isinstance(v, np.ndarray) for v in (byteskip, offset, idarray)))
        or ensemble == 0
    ):
        _, _, byteskip, offset, idarray, ensemble, error_code = fileheader(filename)
        if error_code > 0 and error_code < 6:
            return (np.array([]), error_code)

    # Type narrowing: Cast to ndarray after initialization
    byteskip = cast(np.ndarray, byteskip)
    offset = cast(np.ndarray, offset)
    idarray = cast(np.ndarray, idarray)

    # Type narrowing: Extract beam and cell values
    cell_array: np.ndarray
    beam_array: np.ndarray

    # These arguments are outputs of fixedleader function.
    # Makes the code faster if the fixedheader function is already executed.
    if isinstance(cell, (np.integer, int)) or isinstance(beam, (np.integer, int)):
        flead: np.ndarray
        flead, ensemble, fl_error_code = fixedleader(
            filename,
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble,
        )
        cell_array = flead[7][:]
        beam_array = flead[6][:]
        if fl_error_code != 0:
            error_code = fl_error_code
    else:
        cell_array = cell
        beam_array = beam

    # Extract max values for array shape
    max_beam: int = int(max(beam_array))
    max_cell: int = int(max(cell_array))

    # Velocity is 16 bits and all others are 8 bits.
    # Create empty array for the chosen variable name.
    if var_name == "velocity":
        bitint: int = 2
        inttype: Literal["int16", "uint8"] = "int16"
        var_array: np.ndarray = np.full(
            (max_beam, max_cell, ensemble), -32768, dtype=inttype
        )
    else:  # inserted
        bitint = 1
        inttype = "uint8"
        var_array = np.zeros((max_beam, max_cell, ensemble), dtype=inttype)

    # Read the file in safe mode.
    bfile: Optional[BinaryIO]
    bfile, error = safe_open(filename, "rb")
    if bfile is None:
        return (var_array, error.code)

    if error.code == 0 and error_code != 0:
        error.code = error_code

    bfile.seek(0, 0)
    vid_tuple: Optional[tuple[int, ...]] = varid.get(var_name)

    # Print error if the variable id is not found.
    if not vid_tuple:
        print(
            bcolors.FAIL
            + "ValueError: Invalid variable name. List of permissible variable names: 'velocity', 'correlation', 'echo', 'percent good', 'status'"
            + bcolors.ENDC
        )
        error = ErrorCode.VALUE_ERROR
        return (var_array, error.code)

    # Checks if variable id is found in address offset
    fbyteskip: Optional[list[int]] = None
    for count, item in enumerate(idarray[0][:]):
        if item in vid_tuple:
            fbyteskip = []
            for i in range(ensemble):
                fbyteskip.append(int(offset[i][count]))
            break

    if fbyteskip is None:
        print(
            bcolors.FAIL
            + "ERROR: Variable ID not found in address offset."
            + bcolors.ENDC
        )
        error = ErrorCode.ID_NOT_FOUND
        return (var_array, error.code)

    # READ DATA
    ensemble_idx: int = 0
    try:
        for ensemble_idx in range(ensemble):
            total_bytes: int = (
                beam_array[ensemble_idx] * cell_array[ensemble_idx] * bitint
            )
            bfile.seek(fbyteskip[ensemble_idx], 1)
            bdata: bytes = bfile.read(total_bytes)
            velocity_block: np.ndarray = np.frombuffer(bdata, dtype=inttype)
            var_array[
                : beam_array[ensemble_idx], : cell_array[ensemble_idx], ensemble_idx
            ] = velocity_block.reshape(
                (beam_array[ensemble_idx], cell_array[ensemble_idx])
            )
            bfile.seek(byteskip[ensemble_idx], 0)
        bfile.close()

    except (ValueError, StructError, OverflowError) as e:
        print(bcolors.WARNING + "WARNING: The file is broken.")
        print(
            f"Function `datatype` unable to extract {var_name} for ensemble {ensemble_idx + 1}. Total ensembles reset to {ensemble_idx}."
        )
        print(bcolors.UNDERLINE + "Details from struct function" + bcolors.ENDC)
        print(f"  Error Type: {type(e).__name__}")
        print(f"  Error Details: {e}")
        error = ErrorCode.FILE_CORRUPTED
        ensemble = ensemble_idx

    data: np.ndarray = var_array[:, :, :ensemble]
    return (data, ensemble, cell_array, beam_array, error_code)
