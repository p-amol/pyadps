"""
pyreadrdi.py

Module Overview
---------------
This module provides functionalities to read and parse RDI ADCP files. 
It includes functions for reading file headers, fixed and variable leaders, 
and data types like velocity, correlation, echo intensity, and percent good.
Currently reads only PD0 format. 

Modules
-------------------
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
2024-09-01

Version
-------
0.2.0

Author
------
[P. Amol] <your.email@example.com>

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
from struct import error as StructError
from struct import unpack

import numpy as np


class bcolors:
    """
    Terminal color codes for styling console output.

    This class provides a set of color codes and text formatting options for styling
    terminal or console output. The codes can be used to change the text color and style
    in a terminal that supports ANSI escape sequences.

    Attributes
    ----------
    HEADER : str
        Color code for magenta text, typically used for headers.
    OKBLUE : str
        Color code for blue text, typically used for general information.
    OKCYAN : str
        Color code for cyan text, used for informational messages.
    OKGREEN : str
        Color code for green text, typically used for success messages.
    WARNING : str
        Color code for yellow text, used for warnings.
    FAIL : str
        Color code for red text, used for errors or failures.
    ENDC : str
        Reset color code to default. Resets the color and formatting.
    BOLD : str
        Bold text formatting code. Makes text bold.
    UNDERLINE : str
        Underlined text formatting code. Underlines the text.

    Usage
    -----
    To use these color codes, prepend them to your string and append `bcolors.ENDC`
    to reset the formatting. For example:

    >>> print(f"{bcolors.OKGREEN}Success{bcolors.ENDC}")
    >>> print(f"{bcolors.WARNING}Warning: This is a warning.{bcolors.ENDC}")

    Examples
    --------
    >>> print(f"{bcolors.OKBLUE}This text is blue.{bcolors.ENDC}")
    >>> print(f"{bcolors.FAIL}This text is red and indicates an error.{bcolors.ENDC}")
    >>> print(f"{bcolors.BOLD}{bcolors.UNDERLINE}Bold and underlined text.{bcolors.ENDC}")

    Notes
    -----
    These color codes use ANSI escape sequences and may not be supported in all terminal
    environments. The appearance may vary depending on the terminal emulator used.
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    # HEADER = "[95m"
    # OKBLUE = "[94m"
    # OKCYAN = "[96m"
    # OKGREEN = "[92m"
    # WARNING = "[93m"
    # FAIL = "[91m"
    # ENDC = "[0m"
    # BOLD = "[1m"
    # UNDERLINE = "[4m"


class ErrorCode(Enum):
    """
    Enumeration for error codes with associated messages.
    This enum provides a set of error codes and their corresponding descriptive messages.
    It is used to standardize error reporting and handling within the application.
    Attributes
    ----------
    SUCCESS : tuple
        Represents a successful operation.
    FILE_NOT_FOUND : tuple
        Error code for when a file is not found.
    PERMISSION_DENIED : tuple
        Error code for when access to a resource is denied.
    IO_ERROR: tuple
        Error Code for when the file fails to open.
    OUT_OF_MEMORY : tuple
        Error code for when the system runs out of memory.
    WRONG_RDIFILE_TYPE : tuple
        Error code for when a file type is not supported by RDI or incorrect.
    ID_NOT_FOUND: tuple
        Error code for when RDI file is found but the data type ID does not match.
    FILE_CORRUPTED : tuple
        Error code for when a file is corrupted and cannot be read.
    DATATYPE_MISMATCH: tuple
        Error code for when the data type is not same as the previous ensemble.
    VALUE_ERROR: tuple
        Error code for incorrect argument.
    UNKNOWN_ERROR : tuple
        Error code for an unspecified or unknown error.

    Methods
    -------
    get_message(code)
        Retrieves the descriptive message corresponding to a given error code.

    Parameters
    ----------
    code : int
        The error code for which the message is to be retrieved.

    Returns
    -------
    str
        The descriptive message associated with the provided error code. If the code
        is not valid, returns \"Error: Invalid error code.\"
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

    def __init__(self, code, message):
        self.code = code
        self.message = message

    @classmethod
    def get_message(cls, code):
        for error in cls:
            if error.code == code:
                return error.message
        else:  # inserted
            return "Error: Invalid error code."


def safe_open(filename, mode="rb"):
    """
    Safely open a file, handling common file-related errors.

    This function attempts to open a file and handles exceptions that may occur,
    such as the file not being found or a lack of necessary permissions.
    It returns the file object if successful, or an appropriate error message.

    Parameters
    ----------
    filepath : str
        The path to the file that you want to open.
    mode : str, optional
        The mode in which to open the file (e.g., 'r' for reading, 'w' for writing).
        Defaults to 'r'.

    Returns
    -------
    file object or None
        If the file is successfully opened, the file object is returned.
        If an error occurs, None is returned and an error message is printed.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    PermissionError
        If the file cannot be opened due to insufficient permissions.
    IOError
        If an I/O error occurs during the opening of the file.
    MemoryError
        If required memory cannot be allocated by Python.
    Exception
        If an unexpected error occurs



    Examples
    --------
    >>> f = safe_open('existing_file.txt')
    >>> if f:
    ...     content = f.read()
    ...     f.close()

    >>> safe_open('nonexistent_file.txt')
    Error: File not found.

    >>> safe_open('/restricted_access_file.txt')
    Error: Permission denied.
    """
    try:
        filename = os.path.abspath(filename)
        file = open(filename, mode)
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


def safe_read(bfile, num_bytes):
    """
    Safely read a specified number of bytes from a binary file.

    This function attempts to read `num_bytes` from the provided binary file object.
    It includes error handling for I/O errors, unexpected end-of-file, and other potential issues.

    Parameters
    ----------
    bfile : file object
        A binary file object opened for reading.
    num_bytes : int
        The number of bytes to read from the file.

    Returns
    -------
    bytes or None
        The bytes read from the file, or None if an error occurred.

    Raises
    ------
    IOError
        If an I/O error occurs during the file read operation.
    OSError
        If an operating system-related error occurs.
    ValueError
        If fewer than `num_bytes` are read from the file, indicating an unexpected end of file.
    """
    try:
        readbytes = bfile.read(num_bytes)

        if len(readbytes) != num_bytes:
            print(f"Unexpected end of file: fewer than {num_bytes} bytes were read.")
            return (readbytes, ErrorCode.FILE_CORRUPTED)
        else:
            return (readbytes, ErrorCode.SUCCESS)

    except (IOError, OSError) as e:
        print(f"File read error: {e}")
        return (None, ErrorCode.IO_ERROR)
    except ValueError as e:
        print(f"Value error: {e}")
        return (None, ErrorCode.VALUE_ERROR)


def fileheader(rdi_file):
    """
    Parse the binary RDI ADCP file and extract header information.

    This function reads a binary file and extracts several fields from its header,
    returning them as numpy arrays and integers.

    Parameters
    ----------
    filename : str
        The path to the binary file to be read.

    Returns
    -------
    datatype : numpy.ndarray
        A 1D numpy array of type `int16` representing the data type field from the file header.
    byte : numpy.ndarray
        A 1D numpy array of type `int16` representing the byte information from the file header.
    byteskip : numpy.ndarray
        A 1D numpy array of type `int32` indicating how many bytes to skip in the file.
    address_offset : numpy.ndarray
        A 2D numpy array of type `int` representing the address offsets within the file.
    dataid : numpy.ndarray
        A 2D numpy array of type `int` representing the data IDs extracted from the file header.
    ensemble : int
        An integer representing the ensemble information from the file header.
    error_code : int
        An integer representing the error code, where 0 typically indicates success.

    Raises
    ------
    IOError
        If there is an issue opening or reading from the file.
    ValueError
        If the file does not contain the expected structure or the data cannot be parsed correctly.

    Notes
    -----
    This function assumes that the file is in a specific RDI binary format and may not work correctly
    if the file format differs.

    Examples
    --------
    >>> datatype, byte, byteskip, address_offset, dataid, ensemble, error_code = fileheader("data.bin")
    >>> if error_code == 0:
    ...     print("File header read successfully.")
    ... else:
    ...     print(f"Error code: {error_code}")
    """

    filename = rdi_file
    headerid = np.array([], dtype="int8")
    sourceid = np.array([], dtype="int8")
    byte = np.array([], dtype="int16")
    spare = np.array([], dtype="int8")
    datatype = np.array([], dtype="int16")
    address_offset = []
    ensemble = 0
    error_code = 0
    dataid = []
    byteskip = np.array([], dtype="int32")
    dummytuple = ([], [], [], [], [], ensemble, error_code)

    bfile, error = safe_open(filename, mode="rb")
    if bfile is None:
        error_code = error.code
        dummytuple = ([], [], [], [], [], ensemble, error_code)
        return dummytuple
    bfile.seek(0, 0)
    bskip = i = 0
    hid = [None] * 5
    while byt := bfile.read(6):
        hid[0], hid[1], hid[2], hid[3], hid[4] = unpack("<BBHBB", byt)
        headerid = np.append(headerid, np.int8(hid[0]))
        sourceid = np.append(sourceid, np.int16(hid[1]))
        byte = np.append(byte, np.int16(hid[2]))
        spare = np.append(spare, np.int16(hid[3]))
        datatype = np.append(datatype, np.int16(hid[4]))

        # dbyte = bfile.read(2 * datatype[i])
        dbyte, error = safe_read(bfile, 2 * datatype[i])
        if dbyte is None:
            if i == 0:
                error_code = error.code
                dummytuple = ([], [], [], [], [], ensemble, error_code)
                return dummytuple
            else:
                break

        # Check for id and datatype errors
        if i == 0:
            if headerid[0] != 127 or sourceid[0] != 127:
                error = ErrorCode.WRONG_RDIFILE_TYPE
                print(bcolors.FAIL + error.message + bcolors.ENDC)
                error_code = error.code
                return dummytuple
        else:
            if headerid[i] != 127 or sourceid[i] != 127:
                error = ErrorCode.ID_NOT_FOUND
                print(bcolors.FAIL + error.message + bcolors.ENDC)
                break

            if datatype[i] != datatype[i - 1]:
                error = ErrorCode.DATATYPE_MISMATCH
                print(bcolors.FAIL + error.message)
                print(f"Data Types for ensemble {i} is {datatype[i - 1]}.")
                print(f"Data Types for ensemble {i + 1} is {datatype[i]}.")
                print(f"Ensembles reset to {i}" + bcolors.ENDC)
                break

        data = unpack("H" * datatype[i], dbyte)
        address_offset.append(data)

        skip_array = [None] * datatype[i]
        for dtype in range(datatype[i]):
            bseek = int(bskip) + int(address_offset[i][dtype])
            bfile.seek(bseek, 0)
            readbyte = bfile.read(2)
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

    ensemble = i
    bfile.close()
    address_offset = np.array(address_offset)
    dataid = np.array(dataid)
    datatype = datatype[0:ensemble]
    byte = byte[0:ensemble]
    byteskip = byteskip[0:ensemble]
    error_code = error.code
    return (datatype, byte, byteskip, address_offset, dataid, ensemble, error_code)


def fixedleader(rdi_file, byteskip=None, offset=None, idarray=None, ensemble=0):
    """
    Parse the fixed leader data from binary RDI ADCP file.

    This function extracts the fixed leader section of an RDI file. It uses
    optional parameters that can be obtained from the `fileheader` function. The function
    returns data extracted from the file, the ensemble number, and an error code indicating
    the status of the operation.

    Parameters
    ----------
    rdi_file : str
        The path to the RDI binary file from which to read the fixed leader section.
    byteskip : numpy.ndarray, optional
        Number of bytes to skip before reading the fixed leader section. If not provided,
        defaults to None. Can be obtained from the `fileheader` function.
    offset : numpy.ndarray, optional
        Offset in bytes from the start of the file to the fixed leader section. If not provided,
        defaults to None. Can be obtained from the `fileheader` function.
    idarray : numpy.ndarray, optional
        An optional list of IDs to be processed. If not provided, defaults to None. Can be obtained
        from the `fileheader` function.
    ensemble : int, optional
        The ensemble number to be used or processed. If not provided, defaults to 0. Can be obtained
        from the `fileheader` function.

    Returns
    -------
    data : numpy.ndarray
        Extracted data from the fixed leader section of the file. The type of `data` depends on the
        implementation and file structure.
    ensemble : int
        The ensemble number processed or retrieved from the file.
    error_code : int
        An error code indicating the status of the operation.

    Raises
    ------
    FileNotFoundError
        If the RDI file cannot be found.
    PermissionError
        If the file cannot be accessed due to permission issues.
    ValueError
        If provided parameters are of incorrect type or value.

    Examples
    --------
    >>> data, ensemble, error_code = fixedleader('data.rdi', byteskip=10, offset=50)
    >>> print(data, ensemble, error_code)
    (data_from_file, 0, 0)

    >>> data, ensemble, error_code = fixedleader('data.rdi', idarray=[1, 2, 3])
    >>> print(data, ensemble, error_code)
    (data_from_file, 0, 0)
    """

    filename = rdi_file
    error_code = 0

    if (
        not all((isinstance(v, np.ndarray) for v in (byteskip, offset, idarray)))
        or ensemble == 0
    ):
        _, _, byteskip, offset, idarray, ensemble, error_code = fileheader(filename)

    fid = [[0] * ensemble for _ in range(36)]

    bfile, error = safe_open(filename, "rb")
    if bfile is None:
        return (fid, ensemble, error.code)
    if error.code == 0 and error_code != 0:
        error.code = error_code
        error.message = error.get_message(error.code)

    bfile.seek(0, 0)
    for i in range(ensemble):
        fbyteskip = None
        for count, item in enumerate(idarray[i]):
            if item in (0, 1):
                fbyteskip = offset[1][count]
        if fbyteskip == None:
            error = ErrorCode.ID_NOT_FOUND
            ensemble = i
            print(bcolors.WARNING + error.message)
            print(f"Total ensembles reset to {i}." + bcolors.ENDC)
            break
        else:  # inserted
            try:
                bfile.seek(fbyteskip, 1)
                bdata = bfile.read(59)
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
                (fid[30][i]) = unpack(">Q", bdata[42:50])[0]
                # (fid[30][i], fid[31][i])= struct.unpack('>II', packed_data)
                # fid[30][i] = int.from_bytes(bdata[42:50], byteorder="big", signed=False)
                # System bandwidth, system power & Spare
                (fid[31][i], fid[32][i], fid[33][i]) = unpack("<HBB", bdata[50:54])
                # Instrument serial number & Beam angle
                (fid[34][i], fid[35][i]) = unpack("<LB", bdata[54:59])

                bfile.seek(byteskip[i], 0)

            except (ValueError, StructError) as e:
                print(bcolors.WARNING + "WARNING: The file is broken.")
                print(
                    f"Function `fixedleader` unable to extract data for ensemble {i + 1}. Total ensembles reset to {i}."
                )
                print("Details from struct function:")
                print(f"An error occurred: {e}" + bcolors.ENDC)
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
    fid = np.array(fid)
    data = fid[:, :ensemble]
    return (data, ensemble, error_code)


def variableleader(rdi_file, byteskip=None, offset=None, idarray=None, ensemble=0):
    """
    Parse the variable leader data from binary RDI ADCP file.

    This function extracts the variable leader section of an RDI file. It uses
    optional parameters that can be obtained from the `fileheader` function. The function
    returns data extracted from the file, the ensemble number, and an error code indicating
    the status of the operation.

    Parameters
    ----------
    rdi_file : str
        The path to the RDI binary file from which to read the fixed leader section.
    byteskip : numpy.ndarray, optional
        Number of bytes to skip before reading the fixed leader section. If not provided,
        defaults to None. Can be obtained from the `fileheader` function.
    offset : numpy.ndarray, optional
        Offset in bytes from the start of the file to the fixed leader section. If not provided,
        defaults to None. Can be obtained from the `fileheader` function.
    idarray : numpy.ndarray, optional
        An optional list of IDs to be processed. If not provided, defaults to None. Can be obtained
        from the `fileheader` function.
    ensemble : int, optional
        The ensemble number to be used or processed. If not provided, defaults to 0. Can be obtained
        from the `fileheader` function.

    Returns
    -------
    data : numpy.ndarray
        Extracted data from the variable leader section of the file.
    ensemble : int
        The ensemble number processed or retrieved from the file.
    error_code : int
        An error code indicating the status of the operation.

    Raises
    ------
    FileNotFoundError
        If the RDI file cannot be found.
    PermissionError
        If the file cannot be accessed due to permission issues.
    ValueError

        If provided parameters are of incorrect type or value.
    Examples
    --------
    >>> data, ensemble, error_code = fixedleader('data.rdi', byteskip=10, offset=50)
    >>> print(data, ensemble, error_code)
    (data_from_file, 0, 0)

    >>> data, ensemble, error_code = fixedleader('data.rdi', idarray=[1, 2, 3])
    >>> print(data, ensemble, error_code)
    """

    filename = rdi_file
    error_code = 0
    if (
        not all((isinstance(v, np.ndarray) for v in (byteskip, offset, idarray)))
        or ensemble == 0
    ):
        _, _, byteskip, offset, idarray, ensemble, error_code = fileheader(filename)
    vid = [[0] * ensemble for _ in range(48)]
    bfile, error = safe_open(filename, "rb")
    if bfile is None:
        return (vid, ensemble, error.code)
    if error.code == 0 and error_code != 0:
        error.code = error_code
        error.message = error.get_message(error.code)
    bfile.seek(0, 0)
    for i in range(ensemble):
        fbyteskip = None
        for count, item in enumerate(idarray[i]):
            if item in (128, 129):
                fbyteskip = offset[1][count]
        if fbyteskip == None:
            error = ErrorCode.ID_NOT_FOUND
            ensemble = i
            print(bcolors.WARNING + error.message)
            print(f"Total ensembles reset to {i}." + bcolors.ENDC)
            break
        else:
            try:
                bfile.seek(fbyteskip, 1)
                bdata = bfile.read(65)
                vid[0][i], vid[1][i] = unpack("<HH", bdata[0:4])
                if vid[0][i] not in (128, 129):
                    error = ErrorCode.ID_NOT_FOUND
                    ensemble = i
                    print(bcolors.WARNING + error.message)
                    print(f"Total ensembles reset to {i}." + bcolors.ENDC)
                    break
                    sys.exit(f"Variable Leader not found for Ensemble {i}")
                # Extract WorkHorse ADCP’s real-time clock (RTC)
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

            except (ValueError, StructError) as e:
                print(bcolors.WARNING + "WARNING: The file is broken.")
                print(
                    f"Function `variableleader` unable to extract data for ensemble {i + 1}. Total ensembles reset to {i}."
                )
                print("Details from struct function:")
                print(f"An error occurred: {e}" + bcolors.ENDC)
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
    vid = np.array(vid, dtype="int32")
    data = vid[:, :ensemble]
    return (data, ensemble, error_code)


def datatype(
    filename,
    var_name,
    cell=0,
    beam=0,
    byteskip=None,
    offset=None,
    idarray=None,
    ensemble=0,
):
    """
    Parse 3D data from binary RDI ADCP file.

    This function extracts 3D data like velocity, echo intensity,
    correlation, percent good, and status from the binary RDI file.
    It uses optional parameters can be obtained from the
    `fileheader` function and `variableleader` functions. The function
    returns data of shape (beam, cell, ensemble). The number of beams,
    cells and ensembles along with error code are also returned.

    Parameters
    ----------
    filename : TYPE STRING
        RDI ADCP binary file. The function can currently extract Workhorse,
        Ocean Surveyor, and DVS files.

    var_name : TYPE STRING
        Extracts RDI variables that are functions of beam and cells.
        List of permissible variable names: 'velocity', 'correlation',
        'echo', 'percent good', 'status'

    Returns
    -------
    data : numpy.ndarray
        Returns a 3-D array of size (beam, cell, ensemble) for the var_name.
    beam: int
        Returns number of beams.
    cell: int
        Returns number of cells.
    ensemble: int
        Returns number of ensembles.

    """

    varid = dict()

    # Define file ids:
    varid = {
        "velocity": (256, 257),
        "correlation": (512, 513),
        "echo": (768, 769),
        "percent good": (1024, 1025),
        "status": (1280, 1281),
    }
    error_code = 0

    # Check for optional arguments.
    # -----------------------------
    # These arguments are outputs of fileheader function.
    # Makes the code faster if the fileheader function is already executed.
    if (
        not all((isinstance(v, np.ndarray) for v in (byteskip, offset, idarray)))
        or ensemble == 0
    ):
        _, _, byteskip, offset, idarray, ensemble, error_code = fileheader(filename)
        if error_code > 0 and error_code < 6:
            return ([], error_code)

    # These arguments are outputs of fixedleader function.
    # Makes the code faster if the fixedheader function is already executed.
    if cell == 0 or beam == 0:
        flead, ensemble, fl_error_code = fixedleader(
            filename,
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble,
        )
        cell = int(flead[7][0])
        beam = int(flead[6][0])
        if fl_error_code != 0:
            error_code = fl_error_code
    else:
        cell = int(cell)
        beam = int(beam)

    # Velocity is 16 bits and all others are 8 bits.
    # Create empty array for the chosen variable name.
    if var_name == "velocity":
        var_array = np.zeros((beam, cell, ensemble), dtype="int16")
        bitstr = "<h"
        bitint = 2
    else:  # inserted
        var_array = np.zeros((beam, cell, ensemble), dtype="uint8")
        bitstr = "<B"
        bitint = 1
    # -----------------------------

    # Read the file in safe mode.
    bfile, error = safe_open(filename, "rb")
    if bfile is None:
        return (var_array, ensemble, error.code)
    if error.code == 0 and error_code != 0:
        error.code = error_code
        error.message = error.get_message(error.code)

    bfile.seek(0, 0)
    vid = varid.get(var_name)
    # Print error if the variable id is not found.
    if not vid:
        print(
            bcolors.FAIL
            + "ValueError: Invalid variable name. List of permissible variable names: 'velocity', 'correlation', 'echo', 'percent good', 'status'"
            + bcolors.ENDC
        )
        error = ErrorCode.VALUE_ERROR
        return (var_array, error.code)

    # Checks if variable id is found in address offset
    fbyteskip = None
    for count, item in enumerate(idarray[0][:]):
        if item in vid:
            fbyteskip = offset[1][count]
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
    for i in range(ensemble):
        bfile.seek(fbyteskip, 1)
        bdata = bfile.read(2)
        for cno in range(cell):
            for bno in range(beam):
                bdata = bfile.read(bitint)
                varunpack = unpack(bitstr, bdata)
                var_array[bno][cno][i] = varunpack[0]
        bfile.seek(byteskip[i], 0)
    bfile.close()

    data = var_array
    return (data, ensemble, cell, beam, error_code)
