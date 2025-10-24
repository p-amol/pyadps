"""
pyreadrdi.py - Professional RDI ADCP Binary File Reader

Module for reading and parsing RDI Acoustic Doppler Current Profiler (ADCP)
binary files in PD0 format. Supports Workhorse, Ocean Surveyor, and DVS ADCPs.
"""

import io
import logging
from enum import Enum
from pathlib import Path
from struct import error as StructError
from struct import unpack
from typing import BinaryIO, Literal, Optional, Tuple, Union, cast

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)

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


class ErrorCode(Enum):
    """
    Enumeration for error codes with associated messages.

    This enum provides standardized error reporting throughout the module.

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
    DATATYPE_MISMATCH : tuple
        Data type inconsistent with previous ensemble (code 7).
    FILE_CORRUPTED : tuple
        File structure invalid (code 8).
    VALUE_ERROR : tuple
        Invalid argument provided (code 9).
    UNKNOWN_ERROR : tuple
        Unspecified or unexpected error (code 99).
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
    CHECKSUM_ERROR = (10, "Error: Ensemble checksum verification failed.")
    UNKNOWN_ERROR = (99, "Unknown error.")

    def __init__(self, code: int, message: str) -> None:
        """Initialize ErrorCode enum member."""
        self.code: int = code
        self.message: str = message

    @classmethod
    def get_message(cls, code: int) -> str:
        """
        Retrieve the message corresponding to an error code.

        Args
        ----
        code : int
            Error code number.

        Returns
        -------
        str
            Message associated with code, or "Error: Invalid error code."
        """
        for error in cls:
            if error.code == code:
                return error.message
        return "Error: Invalid error code."


def safe_open(filename: FilePathType, mode: str = "rb") -> SafeOpenReturn:
    """
    Safely open a binary file with exception handling.

    Attempts to open a file and handles common exceptions gracefully,
    returning both the file object and an error code via logging.

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

    Examples
    --------
    >>> f, error = safe_open("data.bin")
    >>> if error == ErrorCode.SUCCESS:
    ...     data = f.read()
    ...     f.close()
    """
    try:
        filename_str: str = str(Path(filename).resolve())
        file: BinaryIO = cast(BinaryIO, open(filename_str, mode))
        return (file, ErrorCode.SUCCESS)
    except FileNotFoundError as e:
        logger.error(f"File not found: '{filename}' - {e}")
        return (None, ErrorCode.FILE_NOT_FOUND)
    except PermissionError as e:
        logger.error(f"Permission denied accessing '{filename}': {e}")
        return (None, ErrorCode.PERMISSION_DENIED)
    except IOError as e:
        logger.error(f"IO error opening '{filename}': {e}")
        return (None, ErrorCode.IO_ERROR)
    except MemoryError as e:
        logger.error(f"Out of memory while accessing '{filename}': {e}")
        return (None, ErrorCode.OUT_OF_MEMORY)
    except Exception as e:
        logger.exception(f"Unexpected error opening '{filename}': {e}")
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
            logger.warning(
                f"Unexpected end of file: read {len(readbytes)} bytes, "
                f"expected {num_bytes} bytes"
            )
            return (None, ErrorCode.FILE_CORRUPTED)
        return (readbytes, ErrorCode.SUCCESS)

    except (IOError, OSError) as e:
        logger.error(f"File read error: {e}")
        return (None, ErrorCode.IO_ERROR)
    except ValueError as e:
        logger.error(f"Value error during read: {e}")
        return (None, ErrorCode.VALUE_ERROR)


# ============================================================================
# CHECKSUM VERIFICATION HELPERS (Private - Internal Use Only)
# ============================================================================


def _calculate_checksum(data: bytes) -> int:
    """
    Calculate RDI ensemble checksum per RDI spec Section 7.2.

    The checksum is calculated by summing all bytes in the ensemble
    (excluding the 2-byte checksum itself). Only the 2-byte least
    significant digits are used.

    Args
    ----
    data : bytes
        Ensemble bytes excluding the checksum word (last 2 bytes).

    Returns
    -------
    int
        16-bit checksum (lower 16 bits of sum).

    Examples
    --------
    >>> data = b'\\x7f\\x7f...'
    >>> checksum = _calculate_checksum(data)
    >>> checksum & 0xFFFF
    12345
    """
    return sum(data) & 0xFFFF


def _verify_ensemble_checksum(
    bfile: BinaryIO,
    ensemble_start_pos: int,
    ensemble_size: int,
) -> Tuple[bool, int]:
    """
    Verify checksum for a single ensemble at specified file position.

    Per RDI spec Section 7.2:
    1. Ensemble size (excluding checksum) is at offset +2 in header
    2. Checksum is located at: ensemble_start + ensemble_size
    3. Calculate checksum on all bytes from ensemble_start to end of data
    4. Compare to stored checksum

    Args
    ----
    bfile : BinaryIO
        Open binary file object.
    ensemble_start_pos : int
        File position of ensemble start (header ID byte).
    ensemble_size : int
        Size of ensemble in bytes (from header byte +2, excludes checksum).

    Returns
    -------
    tuple[bool, int]
        (is_valid, error_code) where error_code is 0 if valid,
        CHECKSUM_ERROR if mismatch.

    Notes
    -----
    Logs warning if checksum mismatch detected.
    """
    try:
        # Read ensemble data (excluding checksum)
        bfile.seek(ensemble_start_pos, 0)
        ensemble_data: bytes = bfile.read(ensemble_size)

        if len(ensemble_data) != ensemble_size:
            logger.warning(
                f"Incomplete ensemble at offset {ensemble_start_pos}: "
                f"expected {ensemble_size} bytes, got {len(ensemble_data)}"
            )
            return (False, ErrorCode.FILE_CORRUPTED.code)

        # Calculate expected checksum
        calculated_checksum: int = _calculate_checksum(ensemble_data)

        # Read stored checksum (2 bytes, little-endian)
        bfile.seek(ensemble_start_pos + ensemble_size, 0)
        checksum_bytes: bytes = bfile.read(2)

        if len(checksum_bytes) != 2:
            logger.warning(
                f"Missing checksum at offset {ensemble_start_pos + ensemble_size}"
            )
            return (False, ErrorCode.FILE_CORRUPTED.code)

        stored_checksum: int = int.from_bytes(
            checksum_bytes, byteorder="little", signed=False
        )

        # Compare checksums
        if calculated_checksum != stored_checksum:
            logger.error(
                f"Checksum mismatch at ensemble offset {ensemble_start_pos}: "
                f"calculated {calculated_checksum:#06x}, "
                f"stored {stored_checksum:#06x}."
            )
            return (False, ErrorCode.CHECKSUM_ERROR.code)

        return (True, ErrorCode.SUCCESS.code)

    except (IOError, OSError, ValueError) as e:
        logger.error(f"Error verifying checksum at offset {ensemble_start_pos}: {e}")
        return (False, ErrorCode.IO_ERROR.code)


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
        where ensemble is the count of successfully parsed ensembles and
        error_code is 0 on success or non-zero ErrorCode.code on error.

    Notes
    -----
    - File may be truncated; returns partial data with appropriate error_code.
    - Checksum verification is performed per RDI spec Section 7.2.
    - If checksum fails, parsing stops and accumulated data is returned.

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
            ensemble_start_pos: int = bfile.tell() - 6
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
                break

            # Check for id and datatype errors
            if i == 0:
                if headerid[0] != 127 or sourceid[0] != 127:
                    error = ErrorCode.WRONG_RDIFILE_TYPE
                    logger.error(error.message)
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
                    logger.warning(f"{error.message} Ensembles truncated at {i}")
                    break

                if datatype[i] != datatype[i - 1]:
                    error = ErrorCode.DATATYPE_MISMATCH
                    logger.warning(
                        f"{error.message} Ensemble {i}: {datatype[i - 1]} "
                        f"vs Ensemble {i + 1}: {datatype[i]}. "
                        f"Ensembles truncated at {i}"
                    )
                    break

            try:
                data: tuple[int, ...] = unpack("H" * datatype[i], dbyte)
                address_offset.append(data)
            except Exception as e:
                error = ErrorCode.FILE_CORRUPTED
                logger.error(f"Failed to unpack data type offsets: {e}")
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
            bskip = int(bskip) + int(byte[i]) + 2
            bfile.seek(bskip, 0)
            byteskip = np.append(byteskip, np.int32(bskip))

            # CHECKSUM VERIFICATION (Last step - per RDI spec Section 7.2)
            # If checksum fails here, we stop and return accumulated data
            is_valid_checksum, checksum_error = _verify_ensemble_checksum(
                bfile,
                ensemble_start_pos,
                byte[i],
            )

            if not is_valid_checksum:
                error_msg: str = ErrorCode.get_message(checksum_error)
                logger.error(
                    f"Checksum verification failed at ensemble {i + 1} "
                    f"(file offset {ensemble_start_pos}): {error_msg}. "
                    f"Truncating to {i} valid ensembles."
                )
                error = ErrorCode(checksum_error, error_msg)
                break

            i += 1

    except (ValueError, StructError, OverflowError) as e:
        logger.warning(
            f"File parsing incomplete at ensemble {i + 1}. "
            f"Truncating to {i} ensembles. Error: {type(e).__name__} - {e}"
        )
        error = ErrorCode.FILE_CORRUPTED
        ensemble = i

    ensemble = i
    bfile.close()

    if i == 0:
        error = ErrorCode.FILE_CORRUPTED
        logger.error("No valid ensembles found in file")
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
    serial numbers, and sensor information. Parameters can be provided from
    fileheader() for efficiency; otherwise fileheader() is called internally.

    Args
    ----
    rdi_file : str or Path
        Path to RDI binary file.
    byteskip : np.ndarray, optional
        Array of file offsets from fileheader(). If None, fileheader()
        is called internally.
    offset : np.ndarray, optional
        Address offsets from fileheader(). If None, fileheader()
        is called internally.
    idarray : np.ndarray, optional
        Data type IDs from fileheader(). If None, fileheader()
        is called internally.
    ensemble : int, optional
        Number of ensembles from fileheader(). If 0, fileheader()
        is called internally.

    Returns
    -------
    tuple
        (data, ensemble, error_code) where data is a (36, n_ensembles)
        array of Fixed Leader fields.

    Examples
    --------
    >>> fl_data, n_ens, err = fixedleader("test.000")
    >>> if err == 0:
    ...     n_beams = fl_data[6, 0]
    ...     n_cells = fl_data[7, 0]
    """

    filename: str = str(rdi_file)
    error_code: int = 0
    error: ErrorCode = ErrorCode.SUCCESS

    if (
        not all((isinstance(v, np.ndarray) for v in (byteskip, offset, idarray)))
        or ensemble == 0
    ):
        _, _, byteskip, offset, idarray, ensemble, error_code = fileheader(filename)

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

    # Handle missing serial numbers from old firmware
    is_serial_missing: bool = False
    INT64_MAX: int = 2**63 - 1
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
            logger.warning(f"{error.message} Ensembles truncated at {i}")
            break
        else:
            try:
                bfile.seek(fbyteskip, 1)
                bdata: bytes = bfile.read(59)
                # Fixed Leader ID, CPU Version no. & Revision no.
                (fid[0][i], fid[1][i], fid[2][i]) = unpack("<HBB", bdata[0:4])
                if fid[0][i] not in (0, 1):
                    error = ErrorCode.ID_NOT_FOUND
                    ensemble = i
                    logger.warning(f"{error.message} Ensembles truncated at {i}")
                    break
                # System configuration & Real/Slim flag
                (fid[3][i], fid[4][i]) = unpack("<HB", bdata[4:7])
                # Lag Length, number of beams & Number of cells
                (fid[5][i], fid[6][i], fid[7][i]) = unpack("<BBB", bdata[7:10])
                # Pings per Ensemble, Depth cell length & Blank after transmit
                (fid[8][i], fid[9][i], fid[10][i]) = unpack("<HHH", bdata[10:16])
                # Signal Processing mode, Low correlation threshold & No. of code repetition
                (fid[11][i], fid[12][i], fid[13][i]) = unpack("<BBB", bdata[16:19])
                # Percent good minimum & Error velocity threshold
                (fid[14][i], fid[15][i]) = unpack("<BH", bdata[19:22])
                # Time between ping groups (TP command): Minute, Second, Hundredth
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
                    if not is_serial_missing and fid[30][i] > INT64_MAX:
                        logger.warning(
                            f"Invalid serial number at ensemble {i} "
                            f"(likely old firmware). Value exceeds int64 range. "
                            f"Replacing serial data with missing value flags."
                        )
                        is_serial_missing = True
                except (ValueError, OverflowError) as e:
                    if not is_serial_missing:
                        logger.warning(
                            f"Failed to read serial number at ensemble {i} "
                            f"(likely old firmware): {e}. "
                            f"Replacing serial data with missing value flags."
                        )
                        is_serial_missing = True
                # System bandwidth, system power & Spare
                (fid[31][i], fid[32][i], fid[33][i]) = unpack("<HBB", bdata[50:54])
                # Instrument serial number & Beam angle
                (fid[34][i], fid[35][i]) = unpack("<LB", bdata[54:59])

                bfile.seek(byteskip[i], 0)

            except (ValueError, StructError, OverflowError) as e:
                logger.error(
                    f"Failed to parse Fixed Leader at ensemble {i + 1}. "
                    f"Truncating to {i} ensembles. "
                    f"Error: {type(e).__name__} - {e}"
                )
                error = ErrorCode.FILE_CORRUPTED
                ensemble = i

            except (OSError, io.UnsupportedOperation) as e:
                logger.error(
                    f"File seeking error at ensemble {i + 1}: {e}. "
                    f"Truncating to {i} ensembles."
                )
                error = ErrorCode.FILE_CORRUPTED
                ensemble = i

    bfile.close()
    error_code = error.code

    if is_serial_missing:
        logger.info(
            "Replacing serial number fields with missing value flags "
            "due to old firmware incompatibility."
        )
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
        (data, ensemble, error_code) where data is a (48, n_ensembles)
        array of Variable Leader fields.

    Examples
    --------
    >>> vl_data, n_ens, err = variableleader("test.000")
    >>> if err == 0:
    ...     year = vl_data[2, 0]
    ...     heading = vl_data[13, 0]
    """

    filename: str = str(rdi_file)
    error_code: int = 0
    error: ErrorCode = ErrorCode.SUCCESS

    if (
        not all((isinstance(v, np.ndarray) for v in (byteskip, offset, idarray)))
        or ensemble == 0
    ):
        _, _, byteskip, offset, idarray, ensemble, error_code = fileheader(filename)

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
            logger.warning(f"{error.message} Ensembles truncated at {i}")
            break
        else:
            try:
                bfile.seek(fbyteskip, 1)
                bdata: bytes = bfile.read(65)
                vid[0][i], vid[1][i] = unpack("<HH", bdata[0:4])
                if vid[0][i] not in (128, 129):
                    error = ErrorCode.ID_NOT_FOUND
                    ensemble = i
                    logger.warning(f"{error.message} Ensembles truncated at {i}")
                    break
                # Extract WorkHorse ADCP's real-time clock (RTC)
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
                # Ensemble # MSB & BIT Result
                (vid[9][i], vid[10][i]) = unpack("<BH", bdata[11:14])
                # Sound Speed, Transducer Depth, Heading, Pitch, Roll, Temperature & Salinity
                (
                    vid[11][i],
                    vid[12][i],
                    vid[13][i],
                    vid[14][i],
                    vid[15][i],
                    vid[16][i],
                    vid[17][i],
                ) = unpack("<HHHhhHh", bdata[14:28])
                # MPT minutes, MPT seconds & MPT hundredth
                (vid[18][i], vid[19][i], vid[20][i]) = unpack("<BBB", bdata[28:31])
                # Heading, Pitch, & Roll standard deviation
                (vid[21][i], vid[22][i], vid[23][i]) = unpack("<BBB", bdata[31:34])
                # ADC Channels (8)
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
                # Error status word (4)
                (vid[32][i], vid[33][i], vid[34][i], vid[35][i]) = unpack(
                    "<BBBB", bdata[42:46]
                )
                # Reserved, Pressure, Pressure Variance & Spare
                (vid[36][i], vid[37][i], vid[38][i], vid[39][i]) = unpack(
                    "<HiiB", bdata[46:57]
                )
                # Y2K time: Century, Year, Month, Day, Hour, Minute, Second, Hundredth
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
                logger.error(
                    f"Failed to parse Variable Leader at ensemble {i + 1}. "
                    f"Truncating to {i} ensembles. "
                    f"Error: {type(e).__name__} - {e}"
                )
                error = ErrorCode.FILE_CORRUPTED
                ensemble = i

            except (OSError, io.UnsupportedOperation) as e:
                logger.error(
                    f"File seeking error at ensemble {i + 1}: {e}. "
                    f"Truncating to {i} ensembles."
                )
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
        Variable to extract: 'velocity', 'correlation', 'echo',
        'percent good', or 'status'.
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
        (data, ensemble, cell_array, beam_array, error_code) where data
        is shape (max_beam, max_cell, n_ensembles). Velocity is int16
        (-32768 = missing); others are uint8.

    Examples
    --------
    >>> vel, n_ens, cells, beams, err = datatype("test.000", "velocity")
    >>> if err == 0:
    ...     print(f"Shape: {vel.shape}")
    ...     v_beam0_cell0 = vel[0, 0, :]  # Time series at beam 0, cell 0
    """

    varid: dict[str, tuple[int, ...]] = {
        "velocity": (256, 257),
        "correlation": (512, 513),
        "echo": (768, 769),
        "percent good": (1024, 1025),
        "status": (1280, 1281),
    }
    error_code: int = 0

    # Check for optional arguments from fileheader
    if (
        not all((isinstance(v, np.ndarray) for v in (byteskip, offset, idarray)))
        or ensemble == 0
    ):
        _, _, byteskip, offset, idarray, ensemble, error_code = fileheader(filename)
        if error_code > 0 and error_code < 6:
            return (np.array([]), error_code)

    byteskip = cast(np.ndarray, byteskip)
    offset = cast(np.ndarray, offset)
    idarray = cast(np.ndarray, idarray)

    # Extract beam and cell values from fixedleader if needed
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

    # Velocity is 16 bits, others are 8 bits
    if var_name == "velocity":
        bitint: int = 2
        inttype: Literal["int16", "uint8"] = "int16"
        var_array: np.ndarray = np.full(
            (max_beam, max_cell, ensemble), -32768, dtype=inttype
        )
    else:
        bitint = 1
        inttype = "uint8"
        var_array = np.zeros((max_beam, max_cell, ensemble), dtype=inttype)

    # Read the file safely
    bfile: Optional[BinaryIO]
    bfile, error = safe_open(filename, "rb")
    if bfile is None:
        return (var_array, error.code)

    if error.code == 0 and error_code != 0:
        error.code = error_code

    bfile.seek(0, 0)
    vid_tuple: Optional[tuple[int, ...]] = varid.get(var_name)

    # Validate variable name
    if not vid_tuple:
        logger.error(
            f"Invalid variable name: '{var_name}'. "
            f"Must be one of: 'velocity', 'correlation', 'echo', "
            f"'percent good', 'status'"
        )
        error = ErrorCode.VALUE_ERROR
        return (var_array, error.code)

    # Find variable ID in address offset
    fbyteskip: Optional[list[int]] = None
    for count, item in enumerate(idarray[0][:]):
        if item in vid_tuple:
            fbyteskip = []
            for i in range(ensemble):
                fbyteskip.append(int(offset[i][count]))
            break

    if fbyteskip is None:
        logger.error(
            f"Variable ID {vid_tuple} not found in file structure. "
            f"This data type may not be present in the file."
        )
        error = ErrorCode.ID_NOT_FOUND
        return (var_array, error.code)

    # Read data from file
    ensemble_idx: int = 0
    try:
        for ensemble_idx in range(ensemble):
            bfile.seek(fbyteskip[ensemble_idx], 1)
            bdata: bytes = bfile.read(2)
            total_bytes: int = (
                beam_array[ensemble_idx] * cell_array[ensemble_idx] * bitint
            )
            bdata = bfile.read(total_bytes)

            velocity_block: np.ndarray = np.frombuffer(bdata, dtype=inttype).copy()
            var_array[
                : beam_array[ensemble_idx], : cell_array[ensemble_idx], ensemble_idx
            ] = velocity_block.reshape(
                (cell_array[ensemble_idx], beam_array[ensemble_idx])
            ).T
            bfile.seek(byteskip[ensemble_idx], 0)
        bfile.close()

    except (ValueError, StructError, OverflowError) as e:
        logger.error(
            f"Failed to parse {var_name} data at ensemble {ensemble_idx + 1}. "
            f"Truncating to {ensemble_idx} ensembles. "
            f"Error: {type(e).__name__} - {e}"
        )
        error = ErrorCode.FILE_CORRUPTED
        ensemble = ensemble_idx

    data: np.ndarray = var_array[:, :, :ensemble]
    return (data, ensemble, cell_array, beam_array, error_code)
