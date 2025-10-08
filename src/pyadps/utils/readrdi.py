#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RDI ADCP Binary File Reader
===========================
This module provides classes and functions to read and extract data from RDI Acoustic Doppler
Current Profiler (ADCP) binary files. The module supports Workhorse, Ocean Surveyor, and DVS ADCPs.
It allows for parsing of various data types such as Fixed Leader, Variable Leader, Velocity, Correlation,
Echo Intensity, Percent Good, and Status data.

Classes
-------
FileHeader
    Extracts metadata from the ADCP file header.
FixedLeader
    Handles the Fixed Leader Data that contains configuration and system settings.
VariableLeader
    Extracts and processes the dynamic Variable Leader Data.
Velocity
    Retrieves the velocity data from the ADCP file.
Correlation
    Retrieves correlation data between ADCP beams.
Echo
    Retrieves echo intensity data from the ADCP file.
PercentGood
    Extracts the percentage of valid velocity data.
Status
    Parses the status data from the ADCP.
ReadFile
    Manages the entire data extraction process and unifies all data types.

Functions
---------
check_equal(array)
    Utility function to check if all values in an array are equal.
error_code(code)
    Maps an error code to a human-readable message.

Creation Date
--------------
2024-09-01

Last Modified Date
--------------
2025-08-14

Version
-------
0.4.0

Author
------
[P. Amol] <your.email@example.com>

License
-------
This module is licensed under the MIT License. See LICENSE file for details.

Dependencies
------------
- numpy : Used for array handling and mathematical operations.
- pyreadrdi : Python interface for reading RDI ADCP binary files (supports variableleader, fixedleader, etc.).
- sys : System-specific parameters and functions.
- DotDict : Utility class to handle dictionary-like objects with dot access.

Install the dependencies using:
    pip install numpy

Usage
-----
Basic Example:
```python
from readrdi import ReadFile

# Initialize the ReadFile class
adcp_obj = ReadFile('path_to_your_rdi_file.000')
velocity_data = adcp_obj.velocity.data
pressure_data = adcp_obj.variableleader.pressure.data

# Individual data types can be accessed without reading the entire file
# An example to access fixed leader data

import readrdi as rd

fixed_leader_obj = rd.FixedLeader('path_to_your_rdi_file.000')
isfixedleader_uniform_dict = fixed_leader_obj.is_uniform()

# Access velocity data
velocity_data = adcp_file.velocity.data

# Check for warnings and errors
if adcp_obj.isWarning:
    print("Warning: Some errors were encountered during data extraction.")
else:
    print("Data extracted successfully.")
"""

import importlib.resources as pkg_resources
import json
import os
import sys

import numpy as np
import pandas as pd
from collections import Counter
from pyadps.utils import pyreadrdi
from pyadps.utils.pyreadrdi import bcolors


class DotDict:
    """
    A dictionary-like class that allows access to dictionary items as attributes.
    If initialized with a dictionary, it converts it into attributes.
    If initialized without a dictionary, it loads one from a JSON file or initializes an empty dictionary.

    Parameters
    ----------
    dictionary : dict, optional
        A dictionary to initialize the DotDict with. If not provided, a JSON file is loaded or an empty dictionary is created.
    json_file_path : str, optional
        Path to the JSON file to load the dictionary from, by default "data.json".
    """

    def __init__(self, dictionary=None, json_file_path="data.json"):
        if dictionary is None:
            if pkg_resources.is_resource("pyadps.utils.metadata", json_file_path):
                with pkg_resources.open_text(
                    "pyadps.utils.metadata", json_file_path
                ) as f:
                    dictionary = json.load(f)
            # if os.path.exists(json_file_path):
            #     with open(json_file_path, "r") as file:
            #         dictionary = json.load(file)
            else:
                dictionary = {}  # Initialize an empty dictionary if no JSON file is found
        self._initialize_from_dict(dictionary)

    def _initialize_from_dict(self, dictionary):
        """
        Recursively initializes DotDict attributes from a given dictionary.

        Parameters
        ----------
        dictionary : dict
            The dictionary to convert into attributes for the DotDict object.
        """

        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)
            setattr(self, key, value)

    def __getattr__(self, key):
        """
        Retrieves an attribute from the DotDict object by its key.

        Parameters
        ----------
        key : str
            The attribute key to retrieve.

        Returns
        -------
        value : any
            The value corresponding to the given key.
        """
        return self.__dict__.get(key)


def error_code(code):
    if code == 0:
        error_string = "Data type is healthy"
    elif code == 1:
        error_string = "End of file"
    elif code == 2:
        error_string = "File Corrupted (ID not recognized)"
    elif code == 3:
        error_string = "Wrong file type"
    elif code == 4:
        error_string = "Data type mismatch"
    else:
        error_string = "Unknown error"
    return error_string


def check_equal(lst):
    """
    Checks if all elements in the list are equal.

    Parameters
    ----------
    lst : list
        A list of elements to check.

    Returns
    -------
    bool
        True if all elements in the list are equal, False otherwise.
    """
    return np.all(np.array(lst) == lst[0])


class FileHeader:
    """
    A class to handle the extraction and management of file header
    data from an RDI ADCP binary data.

    The `FileHeader` class uses the `fileheader` function (imported
    from an external module) to extract various data sets from a
    binary file. These data sets are assigned to instance variables within the class. The class provides methods
    to perform operations on the extracted data, such as checking the file format and determining data types.
    FileHeader class can be used for getting Header information from an
    RDI ADCP File. The information can be accessed by the class instance
    called 'header', which is a list.


    Input:
    ----------
    rdi_file = TYPE STRING
        RDI ADCP binary file. The class can currently extract Workhorse,
        Ocean Surveyor, and DVS files.

    Attributes:
    -------------------
    filename =  TYPE CHARACTER
        Returns the input filename
    ensembles = TYPE INTEGER
        Total number of ensembles in the file
    header = DICTIONARY(STRING, LIST(INTEGER))
        KEYS: 'Header ID', 'Source ID', 'Bytes', 'Spare',
              'Data Types', 'Address Offset'

    Methods:
    --------
    datatypes(ens=0)
        Lists out the data types for any one ensemble (default = 0)
    check_file()
        Checks if
        1. system file size and calculated file size are same
        2. no. of data types and no. of bytes are same for all ensembles

    Example code to access the class
    --------------------------------
    myvar = FileHeader(rdi_file)
    myvar.header['Header Id']
    myvar.datatypes(ens=1)
    myvar.check_file()

    """

    def __init__(self, rdi_file):
        """
        Initializes the FileHeader object, extracts header data, and assigns it to instance variables.

        Parameters
        ----------
        rdi_file : str
            The RDI ADCP binary file to extract data from.
        """
        self.filename = rdi_file
        (
            self.datatypes,
            self.bytes,
            self.byteskip,
            self.address_offset,
            self.dataid,
            self.ensembles,
            self.error,
        ) = pyreadrdi.fileheader(rdi_file)
        self.warning = pyreadrdi.ErrorCode.get_message(self.error)

    def data_types(self, ens=0):
        """
        Finds the available data types for an ensemble.

        Parameters
        ----------
        ens : int, optional
            Ensemble number to get data types for, by default 0 or the first ensemble.

        Returns
        -------
        list
            A list of data type names corresponding to the ensemble.
        """

        data_id_array = self.dataid[ens]
        id_name_array = list()
        i = 0

        for data_id in data_id_array:
            # Checks dual mode IDs (BroadBand or NarrowBand)
            # The first ID is generally the default ID
            if data_id in (0, 1):
                id_name = "Fixed Leader"
            elif data_id in (128, 129):
                id_name = "Variable Leader"
            elif data_id in (256, 257):
                id_name = "Velocity"
            elif data_id in (512, 513):
                id_name = "Correlation"
            elif data_id in (768, 769):
                id_name = "Echo"
            elif data_id in (1024, 1025):
                id_name = "Percent Good"
            elif data_id == 1280:
                id_name = "Status"
            elif data_id == 1536:
                id_name = "Bottom Track"
            else:
                id_name = "ID not Found"

            id_name_array.append(id_name)
            i += 1

        return id_name_array

    def check_file(self):
        """
        Checks if the system file size matches the calculated file size and
        verifies uniformity across bytes and data types for all ensembles.

        Returns
        -------
        dict
            A dictionary containing file size and uniformity checks.
        """
        file_stats = os.stat(self.filename)
        sys_file_size = file_stats.st_size
        cal_file_size = sum((self.bytes).astype(int)) + 2 * len(self.bytes)

        check = dict()

        check["System File Size (B)"] = sys_file_size
        check["Calculated File Size (B)"] = cal_file_size
        check["File Size (MB)"] = cal_file_size / 1048576

        if sys_file_size != cal_file_size:
            check["File Size Match"] = False
        else:
            check["File Size Match"] = True

        check["Byte Uniformity"] = check_equal(self.bytes.tolist())
        check["Data Type Uniformity"] = check_equal(self.bytes.tolist())

        return check

    def print_check_file(self):
        """
        Prints a summary of the file size check results, including system file size, calculated file size,
        and warnings if discrepancies are found.

        Returns
        -------
        None
        """
        file_stats = os.stat(self.filename)
        sys_file_size = file_stats.st_size
        cal_file_size = sum(self.bytes) + 2 * len(self.bytes)

        print("---------------RDI FILE SIZE CHECK-------------------")
        print(f"System file size = {sys_file_size} B")
        print(f"Calculated file size = {cal_file_size} B")
        if sys_file_size != cal_file_size:
            print("WARNING: The file sizes do not match")
        else:
            print(
                "File size in MB (binary): % 8.2f MB\
                  \nFile sizes matches!"
                % (cal_file_size / 1048576)
            )
        print("-----------------------------------------------------")

        print(f"Total number of ensembles: {self.ensembles}")

        if check_equal(self.bytes.tolist()):
            print("No. of Bytes are same for all ensembles.")
        else:
            print("WARNING: No. of Bytes not equal for all ensembles.")

        if check_equal(self.datatypes.tolist()):
            print("No. of Data Types are same for all ensembles.")
        else:
            print("WARNING: No. of Data Types not equal for all ensembles.")

        return


# FIXED LEADER CODES #
class FixedLeader:
    """
    The class extracts Fixed Leader data from RDI File.

    Fixed Leader data are non-dynamic or constants. They
    contain hardware information and ADCP data that only
    change based on certain ADCP input commands. The data,
    generally, do not change within a file.
    """

    def __init__(
        self,
        rdi_file,
        byteskip=None,
        offset=None,
        idarray=None,
        ensemble=0,
    ):
        """
        Initializes the FixedLeader object, extracts Fixed Leader data, and stores it in a dictionary.
        The optional parameters can be obtained from FileHeader.

        Parameters
        ----------
        rdi_file : str
            The RDI ADCP binary file to extract data from.
        byteskip : int, optional
            Number of bytes to skip, by default None.
        offset : int, optional
            Offset value for data extraction, by default None.
        idarray : array-like, optional
            Array of IDs for data extraction, by default None.
        ensemble : int, optional
            Ensemble number to start extraction from, by default 0.
        """
        self.filename = rdi_file

        self.data, self.ensembles, self.error = pyreadrdi.fixedleader(
            self.filename,
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble,
        )
        self.warning = pyreadrdi.ErrorCode.get_message(self.error)

        # self.data = self.data.astype(np.uint64)
        self.fleader = dict()
        self._initialize_from_dict(DotDict(json_file_path="flmeta.json"))

    def _initialize_from_dict(self, dotdict):
        """
        Initializes the FixedLeader object from a DotDict.

        Parameters
        ----------
        dotdict : DotDict
            A DotDict object containing metadata for the Fixed Leader.
        """
        i = 1
        for key, value in dotdict.__dict__.items():
            setattr(self, key, value)
            data_slice = self.data[i]
            setattr(getattr(self, key), "data", data_slice)
            self.fleader[value.long_name] = data_slice
            i = i + 1

    def field(self, ens=0):
        """
        Returns Fixed Leader dictionary pairs for a single ensemble.

        Parameters
        ----------
        ens : int, optional
            Ensemble number to extract, by default 0 or the first ensemble.

        Returns
        -------
        dict
            A dictionary of Fixed Leader data for the specified ensemble.
        """

        return {key: value[ens] for key, value in self.fleader.items()}

    def is_uniform(self):
        """
        Checks whether Fixed Leader data fields are uniform across ensembles.

        Returns
        -------
        dict
            A dictionary indicating uniformity of each Fixed Leader data field.
        """
        output = dict()
        for key, value in self.fleader.items():
            output[key] = check_equal(value)
        return output

    def system_configuration(self, ens=0):
        """
        Extracts and interprets the system configuration from the Fixed Leader data.

        Parameters
        ----------
        ens : int, optional
            Ensemble number to extract system configuration for, by default 0.

        Returns
        -------
        dict
            A dictionary containing system configuration details.

        List of Returnable Keys
        -----------------------
            Returns values for the following keys
            No| Keys                   | Possible values
            ---------------------------------
            1 | "Frequency"            | ['75 kHz', '150 kHz', '300 kHz',
                                          '600 kHz', '1200 kHz', '2400 kHz',
                                          '38 kHz']
            2 | "Beam Pattern"         | ['Concave', 'Convex']
            3 | "Sensor Configuration" | ['#1', '#2', '#3']
            4 | "XDCR HD"              | ['Not Attached', 'Attached']
            5 | "Beam Direction"       | ['Up', 'Down']
            6 | "Beam Angle            | [15, 20, 30, 25, 45]
            7 | "Janus Configuration"  | ["4 Beam", "5 Beam CFIG DEMOD",
                                          "5 Beam CFIG 2 DEMOD"]
        """

        binary_bits = format(int(self.fleader["System Config Code"][ens]), "016b")
        # convert integer to binary format
        # In '016b': 0 adds extra zeros to the binary string
        #          : 16 is the total number of binary bits
        #          : b is used to convert integer to binary format
        #          : Add '#' to get python binary format ('#016b')
        sys_cfg = dict()

        freq_code = {
            "000": "75-kHz",
            "001": "150-kHz",
            "010": "300-kHz",
            "011": "600-kHz",
            "100": "1200-kHz",
            "101": "2400-kHz",
            "110": "38-kHz",
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

        bit_group = binary_bits[13:16]
        sys_cfg["Frequency"] = freq_code.get(bit_group, "Frequency not found")

        bit_group = binary_bits[12]
        sys_cfg["Beam Pattern"] = beam_code.get(bit_group)

        bit_group = binary_bits[10:12]
        sys_cfg["Sensor Configuration"] = sensor_code.get(bit_group)

        bit_group = binary_bits[9]
        sys_cfg["XDCR HD"] = xdcr_code.get(bit_group)

        bit_group = binary_bits[8]
        sys_cfg["Beam Direction"] = dir_code.get(bit_group)

        bit_group = binary_bits[4:8]
        sys_cfg["Beam Angle"] = angle_code.get(bit_group, "Angle not found")

        bit_group = binary_bits[0:4]
        sys_cfg["Janus Configuration"] = janus_code.get(
            bit_group, "Janus cfg. not found"
        )

        return sys_cfg

    def ex_coord_trans(self, ens=0):
        """
        Extracts the coordinate transformation configuration from the Fixed Leader data.

        Parameters
        ----------
        ens : int, optional
            Ensemble number to extract transformation configuration for, by default 0.

        Returns
        -------
        dict
            A dictionary of coordinate transformation details.
        """

        bit_group = format(int(self.fleader["Coord Transform Code"][ens]), "08b")
        transform = dict()

        trans_code = {
            "00": "Beam Coordinates",
            "01": "Instrument Coordinates",
            "10": "Ship Coordinates",
            "11": "Earth Coordinates",
        }

        bool_code = {"1": True, "0": False}

        transform["Coordinates"] = trans_code.get(bit_group[3:5])
        transform["Tilt Correction"] = bool_code.get(bit_group[5])
        transform["Three-Beam Solution"] = bool_code.get(bit_group[6])
        transform["Bin Mapping"] = bool_code.get(bit_group[7])

        return transform

    def ez_sensor(self, ens=0, field="source"):
        """
        Checks for available or selected sensors from the Fixed Leader.

        Parameters
        ----------
        ens : int, optional
            Ensemble number to extract sensor information for, by default 0.
        field : str, optional
            Sensor field to extract ('source' or 'avail'), by default "source".

        Returns
        -------
        dict
        A dictionary of sensor availability or source selection.

        """
        if field == "source":
            bit_group = format(int(self.fleader["Sensor Source Code"][ens]), "08b")
        elif field == "avail":
            bit_group = format(int(self.fleader["Sensor Avail Code"][ens]), "08b")
        else:
            sys.exit("ERROR (function ez_sensor): Enter valid argument.")

        sensor = dict()

        bool_code = {"1": True, "0": False}

        sensor["Sound Speed"] = bool_code.get(bit_group[1])
        sensor["Depth Sensor"] = bool_code.get(bit_group[2])
        sensor["Heading Sensor"] = bool_code.get(bit_group[3])
        sensor["Pitch Sensor"] = bool_code.get(bit_group[4])
        sensor["Roll Sensor"] = bool_code.get(bit_group[5])
        sensor["Conductivity Sensor"] = bool_code.get(bit_group[6])
        sensor["Temperature Sensor"] = bool_code.get(bit_group[7])

        return sensor


# VARIABLE LEADER CODES #
class VariableLeader:
    """
    The class extracts Variable Leader Data.

    Variable Leader data refers to the dynamic ADCP data
    (from clocks/sensors) that change with each ping. The
    WorkHorse ADCP always sends Variable Leader data as output
    data (LSBs first).

    Parameters
    ----------
    rdi_file : str
        RDI ADCP binary file. The class can currently extract Workhorse,
        Ocean Surveyor, and DVS files.
    """

    def __init__(
        self,
        rdi_file,
        byteskip=None,
        offset=None,
        idarray=None,
        ensemble=0,
    ):
        """
        Initializes the VariableLeader object and extracts data from the RDI ADCP binary file.

        Parameters
        ----------
        rdi_file : str
            The RDI ADCP binary file to extract data from.
        byteskip : int, optional
            Number of bytes to skip, by default None.
        offset : int, optional
            Offset value for data extraction, by default None.
        idarray : array-like, optional
            Array of IDs for data extraction, by default None.
        ensemble : int, optional
            Ensemble number to start extraction from, by default 0.
        """
        self.filename = rdi_file

        # Extraction starts here
        self.data, self.ensembles, self.error = pyreadrdi.variableleader(
            self.filename,
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble,
        )
        self.warning = pyreadrdi.ErrorCode.get_message(self.error)

        self.vleader = dict()
        self._initialize_from_dict(DotDict(json_file_path="vlmeta.json"))

    def _initialize_from_dict(self, dotdict):
        """
        Initializes the VariableLeader object attributes from a DotDict.

        Parameters
        ----------
        dotdict : DotDict
            A DotDict object containing metadata for the Variable Leader.
        """
        i = 1
        for key, value in dotdict.__dict__.items():
            setattr(self, key, value)
            data_slice = self.data[i]
            setattr(getattr(self, key), "data", data_slice)
            self.vleader[value.long_name] = data_slice
            i = i + 1

    def bitresult(self):
        """
        Extracts Bit Results from Variable Leader (Byte 13 & 14)
        This field is part of the WorkHorse ADCP’s Built-in Test function.
        A zero code indicates a successful BIT result.

        Note: Byte 14 used for future use.

        Returns
        -------
        dict
            A dictionary of test field results.

        """
        tfname = {
            "Reserved #1": "int16",
            "Reserved #2": "int16",
            "Reserved #3": "int16",
            "DEMOD 1 Error": "int16",
            "DEMOD 0 Error": "int16",
            "Reserved #4": "int16",
            "Timing Card Error": "int16",
            "Reserved #5": "int16",
        }

        test_field = dict()
        bit_array = self.vleader["Bit Result"]

        # The bit result is read as single 16 bits variable instead of
        # two 8-bits variable (Byte 13 & 14). The data is written in
        # little endian format. Therefore, the Byte 14 comes before Byte 13.

        for key, value in tfname.items():
            test_field[key] = np.array([], dtype=value)

        for item in bit_array:
            bit_group = format(int(item), "016b")
            bitpos = 8
            for key, value in tfname.items():
                bitappend = getattr(np, value)(bit_group[bitpos])
                test_field[key] = np.append(test_field[key], bitappend)
                bitpos += 1

        return test_field

    def adc_channel(self, offset=-0.20):
        """
        Extracts ADC Channel data and computes values for Xmit Voltage, Xmit Current,
        and Ambient Temperature using system configuration.

        Parameters
        ----------
        offset : float, optional
            Offset value for temperature calculation, by default -0.20.

        Returns
        -------
        dict
            A dictionary of channel data including Xmit Voltage, Xmit Current, and Ambient Temperature.
        """
        # -----------CODE INCOMPLETE-------------- #
        channel = dict()
        scale_list = {
            "75-kHz": [2092719, 43838],
            "150-kHz": [592157, 11451],
            "300-kHz": [592157, 11451],
            "600-kHz": [380667, 11451],
            "1200-kHz": [253765, 11451],
            "2400-kHz": [253765, 11451],
        }

        adc0 = self.vleader["ADC Channel 0"]
        adc1 = self.vleader["ADC Channel 1"]
        adc2 = self.vleader["ADC Channel 2"]

        fixclass = FixedLeader(self.filename).system_configuration()

        scale_factor = scale_list.get(fixclass["Frequency"])

        channel["Xmit Voltage"] = adc1 * (scale_factor[0] / 1000000)

        channel["Xmit Current"] = adc0 * (scale_factor[1] / 1000000)

        # Coefficients for temperature equation
        a0 = 9.82697464e1
        a1 = -5.86074151382e-3
        a2 = 1.60433886495e-7
        a3 = -2.32924716883e-12

        channel["Ambient Temperature"] = (
            offset + ((a3 * adc2 + a2) * adc2 + a1) * adc2 + a0
        )

        return channel

    def error_status_word(self, esw=1):
        bitset1 = (
            "Bus Error exception",
            "Address Error exception",
            "Zero Divide exception",
            "Emulator exception",
            "Unassigned exception",
            "Watchdog restart occurred",
            "Batter Saver Power",
        )

        bitset2 = (
            "Pinging",
            "Not Used 1",
            "Not Used 2",
            "Not Used 3",
            "Not Used 4",
            "Not Used 5",
            "Cold Wakeup occured",
            "Unknown Wakeup occured",
        )

        bitset3 = (
            "Clock Read error occured",
            "Unexpected alarm",
            "Clock jump forward",
            "Clock jump backward",
            "Not Used 6",
            "Not Used 7",
            "Not Used 8",
            "Not Used 9",
        )

        bitset4 = (
            "Not Used 10",
            "Not Used 11",
            "Not Used 12",
            "Power Fail Unrecorded",
            "Spurious level 4 intr DSP",
            "Spurious level 5 intr UART",
            "Spurious level 6 intr CLOCK",
            "Level 7 interrup occured",
        )

        if esw == 1:
            bitset = bitset1
            errorarray = self.vleader["Error Status Word 1"]
        elif esw == 2:
            bitset = bitset2
            errorarray = self.vleader["Error Status Word 2"]
        elif esw == 3:
            bitset = bitset3
            errorarray = self.vleader["Error Status Word 3"]
        else:
            bitset = bitset4
            errorarray = self.vleader["Error Status Word 4"]

        errorstatus = dict()
        # bitarray = np.zeros(32, dtype='str')

        for item in bitset:
            errorstatus[item] = np.array([])

        for data in errorarray:
            if data != -32768:
                byte_split = format(int(data), "08b")
                bitposition = 0
                for item in bitset:
                    errorstatus[item] = np.append(
                        errorstatus[item], int(byte_split[bitposition])
                    )
                    bitposition += 1
            else:
                for item in bitset:
                    errorstatus[item] = np.append(errorstatus[item], 0)

        return errorstatus


class Velocity:
    """
    The class extracts velocity data from RDI ADCP files.

    Parameters
    ----------
    filename : str
        The RDI ADCP binary file to extract data from.
    cell : int, optional
        Cell number to extract, by default 0.
    beam : int, optional
        Beam number to extract, by default 0.
    byteskip : int, optional
        Number of bytes to skip, by default None.
    offset : int, optional
        Offset value for data extraction, by default None.
    idarray : array-like, optional
        Array of IDs for data extraction, by default None.
    ensemble : int, optional
        Ensemble number to start extraction from, by default 0.
    """

    def __init__(
        self,
        filename,
        cell=0,
        beam=0,
        byteskip=None,
        offset=None,
        idarray=None,
        ensemble=0,
    ):
        self.filename = filename
        error = 0
        data, ens, cell, beam, error = pyreadrdi.datatype(
            self.filename,
            "velocity",
            cell=cell,
            beam=beam,
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble,
        )
        self.warning = pyreadrdi.ErrorCode.get_message(error)

        self.data = data
        self.error = error
        self.ensembles = ens
        self.cells = cell
        self.beams = beam

        self.unit = "mm/s"
        self.missing_value = "-32768"
        self.scale_factor = 1
        self.valid_min = -32768
        self.valid_max = 32768


class Correlation:
    """
    The class extracts correlation data from RDI ADCP files.

    Parameters
    ----------
    filename : str
        The RDI ADCP binary file to extract data from.
    cell : int, optional
        Cell number to extract, by default 0.
    beam : int, optional
        Beam number to extract, by default 0.
    byteskip : int, optional
        Number of bytes to skip, by default None.
    offset : int, optional
        Offset value for data extraction, by default None.
    idarray : array-like, optional
        Array of IDs for data extraction, by default None.
    ensemble : int, optional
        Ensemble number to start extraction from, by default 0.
    """

    def __init__(
        self,
        filename,
        cell=0,
        beam=0,
        byteskip=None,
        offset=None,
        idarray=None,
        ensemble=0,
    ):
        self.filename = filename
        error = 0
        data, ens, cell, beam, error = pyreadrdi.datatype(
            self.filename,
            "correlation",
            cell=cell,
            beam=beam,
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble,
        )
        self.warning = pyreadrdi.ErrorCode.get_message(error)

        self.data = data
        self.error = error
        self.ensembles = ens
        self.cells = cell
        self.beams = beam

        self.unit = ""
        self.scale_factor = 1
        self.valid_min = 0
        self.valid_max = 255
        self.long_name = "Correlation Magnitude"


class Echo:
    """
    The class extracts echo intensity data from RDI ADCP files.

    Parameters
    ----------
    filename : str
        The RDI ADCP binary file to extract data from.
    cell : int, optional
        Cell number to extract, by default 0.
    beam : int, optional
        Beam number to extract, by default 0.
    byteskip : int, optional
        Number of bytes to skip, by default None.
    offset : int, optional
        Offset value for data extraction, by default None.
    idarray : array-like, optional
        Array of IDs for data extraction, by default None.
    ensemble : int, optional
        Ensemble number to start extraction from, by default 0.
    """

    def __init__(
        self,
        filename,
        cell=0,
        beam=0,
        byteskip=None,
        offset=None,
        idarray=None,
        ensemble=0,
    ):
        self.filename = filename
        error = 0
        data, ens, cell, beam, error = pyreadrdi.datatype(
            self.filename,
            "echo",
            cell=cell,
            beam=beam,
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble,
        )
        self.warning = pyreadrdi.ErrorCode.get_message(error)

        self.data = data
        self.error = error
        self.ensembles = ens
        self.cells = cell
        self.beams = beam

        self.unit = "counts"
        self.scale_factor = "0.45"
        self.valid_min = 0
        self.valid_max = 255
        self.long_name = "Echo Intensity"


class PercentGood:
    """
    The class extracts Percent Good data from RDI ADCP files.

    Parameters
    ----------
    filename : str
        The RDI ADCP binary file to extract data from.
    cell : int, optional
        Cell number to extract, by default 0.
    beam : int, optional
        Beam number to extract, by default 0.
    byteskip : int, optional
        Number of bytes to skip, by default None.
    offset : int, optional
        Offset value for data extraction, by default None.
    idarray : array-like, optional
        Array of IDs for data extraction, by default None.
    ensemble : int, optional
        Ensemble number to start extraction from, by default 0.
    """

    def __init__(
        self,
        filename,
        cell=0,
        beam=0,
        byteskip=None,
        offset=None,
        idarray=None,
        ensemble=0,
    ):
        self.filename = filename
        error = 0
        data, ens, cell, beam, error = pyreadrdi.datatype(
            self.filename,
            "percent good",
            cell=cell,
            beam=beam,
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble,
        )
        self.warning = pyreadrdi.ErrorCode.get_message(error)

        self.data = data
        self.error = error
        self.ensembles = ens
        self.cells = cell
        self.beams = beam

        self.unit = "percent"
        self.valid_min = 0
        self.valid_max = 100
        self.long_name = "Percent Good"


class Status:
    """
    The class extracts Status data from RDI ADCP files.

    Parameters
    ----------
    filename : str
        The RDI ADCP binary file to extract data from.
    cell : int, optional
        Cell number to extract, by default 0.
    beam : int, optional
        Beam number to extract, by default 0.
    byteskip : int, optional
        Number of bytes to skip, by default None.
    offset : int, optional
        Offset value for data extraction, by default None.
    idarray : array-like, optional
        Array of IDs for data extraction, by default None.
    ensemble : int, optional
        Ensemble number to start extraction from, by default 0.
    """

    def __init__(
        self,
        filename,
        cell=0,
        beam=0,
        byteskip=None,
        offset=None,
        idarray=None,
        ensemble=0,
    ):
        self.filename = filename
        error = 0
        data, ens, cell, beam, error = pyreadrdi.datatype(
            self.filename,
            "status",
            cell=cell,
            beam=beam,
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble,
        )
        self.warning = pyreadrdi.ErrorCode.get_message(error)

        self.data = data
        self.error = error
        self.ensembles = ens
        self.cells = cell
        self.beams = beam

        self.unit = ""
        self.valid_min = 0
        self.valid_max = 1
        self.long_name = "Status Data Format"


class ReadFile:
    """
    Class to read and extract data from RDI ADCP binary files, organizing data types like
    Fixed Leader, Variable Leader, Velocity, and others.

    Parameters
    ----------
    filename : str
        The RDI ADCP binary file to be read.
    """

    def __init__(self, filename, is_fix_ensemble=True, fix_time=False):
        """
        Initializes the ReadFile object and extracts data from the RDI ADCP binary file.
        """
        self.fileheader = FileHeader(filename)
        datatype_array = self.fileheader.data_types()
        error_array = {"Fileheader": self.fileheader.error}
        warning_array = {"Fileheader": self.fileheader.warning}
        ensemble_array = {"Fileheader": self.fileheader.ensembles}

        byteskip = self.fileheader.byteskip
        offset = self.fileheader.address_offset
        idarray = self.fileheader.dataid
        ensemble = self.fileheader.ensembles

        self.fixedleader = FixedLeader(
            filename,
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble,
        )
        error_array["Fixed Leader"] = self.fixedleader.error
        warning_array["Fixed Leader"] = self.fixedleader.warning
        ensemble_array["Fixed Leader"] = self.fixedleader.ensembles
        cells = int(self.fixedleader.fleader["Cells"][0])
        beams = int(self.fixedleader.fleader["Beams"][0])
        ensemble = self.fixedleader.ensembles

        self.variableleader = VariableLeader(
            filename,
            byteskip=byteskip,
            offset=offset,
            idarray=idarray,
            ensemble=ensemble,
        )
        error_array["Variable Leader"] = self.variableleader.error
        warning_array["Variable Leader"] = self.variableleader.warning
        ensemble_array["Variable Leader"] = self.variableleader.ensembles
        ensemble = self.fixedleader.ensembles

        if "Velocity" in datatype_array:
            self.velocity = Velocity(
                filename,
                cell=int(cells),
                beam=int(beams),
                byteskip=byteskip,
                offset=offset,
                idarray=idarray,
                ensemble=ensemble,
            )
            error_array["Velocity"] = self.velocity.error
            warning_array["Velocity"] = self.velocity.warning
            ensemble_array["Velocity"] = self.velocity.ensembles

        if "Correlation" in datatype_array:
            self.correlation = Correlation(
                filename,
                cell=cells,
                beam=beams,
                byteskip=byteskip,
                offset=offset,
                idarray=idarray,
                ensemble=ensemble,
            )
            error_array["Correlation"] = self.correlation.error
            warning_array["Correlation"] = self.correlation.warning
            ensemble_array["Correlation"] = self.correlation.ensembles

        if "Echo" in datatype_array:
            self.echo = Echo(
                filename,
                cell=cells,
                beam=beams,
                byteskip=byteskip,
                offset=offset,
                idarray=idarray,
                ensemble=ensemble,
            )
            error_array["Echo"] = self.echo.error
            warning_array["Echo"] = self.echo.warning
            ensemble_array["Echo"] = self.echo.ensembles

        if "Percent Good" in datatype_array:
            self.percentgood = PercentGood(
                filename,
                cell=cells,
                beam=beams,
                byteskip=byteskip,
                offset=offset,
                idarray=idarray,
                ensemble=ensemble,
            )
            error_array["Percent Good"] = self.percentgood.error
            warning_array["Percent Good"] = self.percentgood.warning
            ensemble_array["Percent Good"] = self.percentgood.ensembles

        if "Status" in datatype_array:
            self.status = Status(
                filename,
                cell=cells,
                beam=beams,
                byteskip=byteskip,
                offset=offset,
                idarray=idarray,
                ensemble=ensemble,
            )
            error_array["Status"] = self.status.error
            warning_array["Status"] = self.status.warning
            ensemble_array["Status"] = self.status.ensembles

        # Add Time Axis
        year = self.variableleader.vleader["RTC Year"]
        month = self.variableleader.vleader["RTC Month"]
        day = self.variableleader.vleader["RTC Day"]
        hour = self.variableleader.vleader["RTC Hour"]
        minute = self.variableleader.vleader["RTC Minute"]
        second = self.variableleader.vleader["RTC Second"]
        year = year + 2000
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
        self.time = pd.to_datetime(date_df)
        self.isTimeRegular = self.is_time_regular()

        # Depth
        # Create a depth axis with mean depth in 'm'
        cell1 = self.fixedleader.field()["Cells"]
        bin1dist1 = self.fixedleader.field()["Bin 1 Dist"] / 100
        depth_cell_len1 = self.fixedleader.field()["Depth Cell Len"] / 100
        beam_direction1 = self.fixedleader.system_configuration()["Beam Direction"]
        mean_depth = np.mean(self.variableleader.vleader["Depth of Transducer"]) / 10
        mean_depth = np.trunc(mean_depth)
        if beam_direction1.lower() == "up":
            sgn = -1
        else:
            sgn = 1
        first_depth = mean_depth + sgn * bin1dist1
        last_depth = first_depth + sgn * cell1 * depth_cell_len1
        z = np.arange(first_depth, last_depth, sgn * depth_cell_len1)
        self.depth = z

        # Add all attributes/method/data from FixedLeader and VariableLeader
        self._copy_attributes_from_var()

        # Error Codes and Warnings
        self.error_codes = error_array
        self.warnings = warning_array
        self.ensemble_array = ensemble_array
        self.ensemble_value_array = np.array(list(self.ensemble_array.values()))

        self.isEnsembleEqual = check_equal(self.ensemble_value_array)
        self.isFixedEnsemble = False

        ec = np.array(list(self.error_codes.values()))

        if np.all(ec == 0):
            self.isWarning = False
        else:
            self.isWarning = True

        # Add additional attributes
        # Ensemble
        dtens = self.ensemble_value_array
        minens = np.min(dtens)
        self.ensembles = minens

        # Add attribute that lists all variables/functions
        self.list_vars = list(vars(self).keys())

        # By default, fix the time axis if requested
        if fix_time:
            self.fix_time_axis()

        # By default fix ensemble
        if is_fix_ensemble and not self.isEnsembleEqual:
            self.fixensemble()

    def _copy_attributes_from_var(self):
        for attr_name, attr_value in self.variableleader.__dict__.items():
            # Copy each attribute of var into self
            setattr(self, attr_name, attr_value)
        for attr_name, attr_value in self.fixedleader.__dict__.items():
            # Copy each attribute of var into self
            setattr(self, attr_name, attr_value)

    def __getattr__(self, name):
        # Delegate attribute/method access to self.var if not found in self
        if hasattr(self.variableleader, name):
            return getattr(self.variableleader, name)
        if hasattr(self.fixedleader, name):
            return getattr(self.fixedleader, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    # Add this method inside your ReadFile class

    def get_time_component_frequency(self, component="minute"):
        """
        Calculates the frequency of a specific time component (hour, minute, or second).

        This is a diagnostic tool to help identify irregularities in the time axis.
        For example, for ideal hourly data, the minute and second frequencies
        should show a single entry (0) with a count equal to the number of ensembles.

        Parameters
        ----------
        component : str, optional
            The time component to analyze. Must be one of 'hour', 'minute',
            or 'second', by default "minute".

        Returns
        -------
        pandas.Series
            A pandas Series with the time component values as the index and
            their frequency (count) as the values, sorted in descending order.
        """
        if component not in ["hour", "minute", "second"]:
            print(
                bcolors.FAIL
                + "Error: Component must be 'hour', 'minute', or 'second'."
                + bcolors.ENDC
            )
            return

        if component == "hour":
            return self.time.dt.hour.value_counts()
        elif component == "minute":
            return self.time.dt.minute.value_counts()
        else:  # second
            return self.time.dt.second.value_counts()

    # In readrdi.py, replace the snap_time_axis method inside the ReadFile class

    def snap_time_axis(self, freq="h", tolerance="5min", target_minute=None):
        """
        Rounds the time axis, returning a status and message for the UI.
        """
        if target_minute is not None:
            action_msg = f"snapping to nearest minute '{target_minute}'"
            try:
                offset = pd.to_timedelta(target_minute, unit="m")
                snapped_time = (self.time - offset).dt.round("h") + offset
            except Exception as e:
                return False, f"Error during rounding: {e}"
        else:
            action_msg = f"snapping to nearest frequency '{freq}'"
            try:
                snapped_time = self.time.dt.round(freq)
            except Exception as e:
                return False, f"Error during rounding: {e}"

        tolerance_delta = pd.to_timedelta(tolerance)

        # Safety Check 1: Tolerance
        max_diff = (self.time - snapped_time).abs().max()
        if max_diff > tolerance_delta:
            message = f"Operation Aborted: Maximum required correction is {max_diff}, which exceeds the tolerance of {tolerance}."
            return False, message

        # Safety Check 2: Duplicates
        if snapped_time.duplicated().any():
            message = "Operation Aborted: Snapping would create duplicate timestamps. The data may be too irregular or have large gaps."
            return False, message

        # Success Case
        self.time = snapped_time
        self.isTimeRegular = self.is_time_regular()
        success_message = f"Successfully snapped time axis. Maximum correction applied was {max_diff}."
        return True, success_message

    def is_time_regular(self, tolerance_s=1):
        """
        Checks if the time axis has a regular, equal interval.

        Args:
            tolerance_ns (int): Tolerance in nanoseconds to consider intervals equal.
                                Defaults to 1 second.

        Returns:
            bool: True if the time axis is regular, False otherwise.
        """
        if len(self.time) < 2:
            return False  # Not enough data to determine regularity

        diffs = pd.Series(self.time).diff().dropna()
        if diffs.empty:
            return False

        # Get the most common difference (mode)
        common_interval = diffs.mode()[0]

        # Check if all differences are approximately equal to the common_interval
        max_deviation_ns = (diffs - common_interval).abs().max().total_seconds()

        return max_deviation_ns <= tolerance_s

    def get_time_interval(self):
        """
        Calculates the most common time interval (frequency) in the data.

        Returns:
            pd.Timedelta or None: The detected modal interval, or None if insufficient data.
        """
        if len(self.time) < 2:
            return None

        diffs = pd.Series(self.time).diff().dropna()
        if diffs.empty:
            return None

        # Find the most common interval (mode)
        return diffs.mode()[0]

    def get_time_interval_frequency(self):
        if len(self.time) < 2:
            return None

        diffs = pd.Series(self.time).diff().dropna()
        if diffs.empty:
            return None
        all_diffs = diffs.tolist()
        frequency = Counter(all_diffs)
        # return dict(frequency)
        return {str(td): count for td, count in frequency.items()}

    def fill_time_axis(self):
        """
        Resamples all data to a regular time axis, filling gaps with missing values.
        Returns a status and message for the UI.
        """
        # 1. Determine the expected frequency
        frequency = self.get_time_interval()
        if frequency is None:
            return (
                False,
                "Operation Aborted: Could not determine a regular time interval from the data.",
            )

        # 2. Create the new, complete time index
        full_time_index = pd.date_range(
            start=self.time.min(), end=self.time.max(), freq=frequency
        )

        if len(full_time_index) == len(self.time) and self.isTimeRegular:
            return (
                False,
                "No action taken: The time axis is already regular and has no gaps.",
            )

        # --- Handle Fixed Leader Data ---
        fleader_df = pd.DataFrame(self.fixedleader.fleader, index=self.time)
        fleader_df_resampled = fleader_df.reindex(full_time_index)
        for key in fleader_df_resampled.columns:
            mode_value = fleader_df[key].mode().iloc[0]
            original_dtype = fleader_df[key].dtype
            fleader_df_resampled[key] = (
                fleader_df_resampled[key].fillna(mode_value).astype(original_dtype)
            )
        for key in self.fixedleader.fleader:
            self.fixedleader.fleader[key] = fleader_df_resampled[key].values

        # --- Handle Variable Leader Data ---
        vleader_df = pd.DataFrame(self.variableleader.vleader, index=self.time)
        vleader_df_resampled = vleader_df.reindex(full_time_index)
        for key, attrs in self.variableleader.vleader.items():
            missing_val = getattr(attrs, "missing_value", -32768)
            self.variableleader.vleader[key] = (
                vleader_df_resampled[key].fillna(missing_val).values.astype(np.int64)
            )

        # --- Handle 3D/4D data types ---
        missing_indices = np.where(full_time_index.isin(self.time) == False)[0]
        data_to_resize = {
            "velocity": -32768,
            "correlation": 255,
            "echo": 255,
            "percentgood": 255,
            "status": 0,
        }
        for name, missing_val in data_to_resize.items():
            if hasattr(self, name):
                obj = getattr(self, name)
                if np.issubdtype(obj.data.dtype, np.integer):
                    new_data = np.insert(
                        obj.data, missing_indices, missing_val, axis=-1
                    )
                    obj.data = new_data
                    obj.ensembles = new_data.shape[-1]

        # --- Update the main object's state ---
        self.time = pd.Series(full_time_index)
        self.ensembles = len(full_time_index)
        self.isTimeRegular = True

        message = f"Successfully filled time axis gaps. Total ensembles are now {self.ensembles}."
        return True, message

    def resize_fixedleader(self, newshape):
        for key in self.fixedleader.fleader:
            attr_name = key.lower().replace(" ", "_")
            attr_obj = getattr(self.fixedleader, attr_name)
            attr_obj.data = attr_obj.data[:newshape]

    def resize_variableleader(self, newshape):
        for key in self.variableleader.vleader:
            attr_name = key.lower().replace(" ", "_")
            attr_obj = getattr(self.variableleader, attr_name)
            attr_obj.data = attr_obj.data[:newshape]

    def fixensemble(self, min_cutoff=0):
        """
        Fixes the ensemble size across all data types in the file if they differ.

        Parameters
        ----------
        min_cutoff : int, optional
            Minimum number of ensembles to consider when fixing, by default 0.

        Returns
        -------
        None
        """
        datatype_array = self.fileheader.data_types()
        # Check if the number of ensembles in a data type
        # is less than min_cutoff.
        # Some data type can have zero ensembles
        dtens = self.ensemble_value_array
        new_array = dtens[dtens > min_cutoff]
        minens = np.min(new_array)

        if not self.isEnsembleEqual:
            self.fileheader.ensembles = minens
            self.fileheader.datatypes = self.fileheader.datatypes[:minens]
            self.fileheader.bytes = self.fileheader.bytes[:minens]
            self.fileheader.byteskip = self.fileheader.byteskip[:minens]
            self.fileheader.address_offset = self.fileheader.address_offset[:minens, :]
            self.fileheader.dataid = self.fileheader.dataid[:minens, :]
            if "Fixed Leader" in datatype_array:
                self.fixedleader.data = self.fixedleader.data[:, :minens]
                self.fixedleader.fleader = {
                    k: v[:minens] for k, v in self.fixedleader.fleader.items()
                }
                self.fixedleader.ensembles = minens
                self.resize_fixedleader(minens)
            if "Variable Leader" in datatype_array:
                self.variableleader.data = self.variableleader.data[:, :minens]
                self.variableleader.vleader = {
                    k: v[:minens] for k, v in self.variableleader.vleader.items()
                }
                self.variableleader.ensembles = minens
                self.resize_variableleader(minens)
            if "Velocity" in datatype_array:
                self.velocity.data = self.velocity.data[:, :, :minens]
                self.velocity.ensembles = minens
            if "Correlation" in datatype_array:
                self.correlation.data = self.correlation.data[:, :, :minens]
                self.correlation.ensembles = minens
            if "Echo" in datatype_array:
                self.echo.data = self.echo.data[:, :, :minens]
                self.echo.ensembles = minens
            if "Percent Good" in datatype_array:
                self.percentgood.data = self.percentgood.data[:, :, :minens]
                self.percentgood.ensembles = minens
            if "Status" in datatype_array:
                self.status.data = self.status.data[:, :, :minens]
                self.status.ensembles = minens

            self.time = self.time[:minens]
            print(
                bcolors.OKBLUE
                + f"Ensembles fixed to {minens}. All data types have same ensembles."
                + bcolors.ENDC
            )
        else:
            print(
                "WARNING: No response was initiated. All data types have same ensemble."
            )

        self.isFixedEnsemble = True
