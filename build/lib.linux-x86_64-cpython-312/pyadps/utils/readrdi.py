#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 17:54:35 2020
Testing 2 D array for fleader variable fid
@author: amol
"""

import os
import sys

# import fortreadrdi
import numpy as np

from pyadps.utils import fortreadrdi, pyreadrdi

# import pyreadrdi



# readbin = pyreadrdi


def error_code(code):
    """
    Return a human-readable error message based on the provided error code.

    Parameters
    ----------
    code : int
        The error code to interpret.

    Returns
    -------
    error_string : str
        A string description of the error corresponding to the given code.
    """
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
    Check if all elements in a list or numpy array are equal.

    Parameters
    ----------
    lst : list or numpy.ndarray
        The list or array to check for uniformity.

    Returns
    -------
    bool
        True if all elements are equal, False otherwise.
    """
    return np.all(np.array(lst) == lst[0])


class FileHeader:
    """
    Class to extract and check header information from an RDI ADCP binary file.

    This class provides methods to read the file header, identify available data types 
    in ensembles, and check the consistency of file size and data types across the file.

    Parameters
    ----------
    
    .. list-table:: Parameters
       :widths: 20 80
       :header-rows: 1

       * - Parameter
         - Description
       * - rdi_file : str
         - The path to the RDI ADCP binary file.
       * - run : {'python', 'fortran'}, optional
         - Specify whether to use the Python or Fortran implementation for reading the 
           file header. Default is 'python'.

    Attributes
    ----------
    
    .. list-table:: Attributes
       :widths: 20 80
       :header-rows: 1

       * - Attribute
         - Description
       * - filename : str
         - The input filename.
       * - ensembles : int
         - Total number of ensembles in the file.
       * - datatypes : list
         - Data types for each ensemble.
       * - bytes : list
         - Number of bytes for each ensemble.
       * - byteskip : list
         - Number of bytes to skip for each ensemble.
       * - address_offset : numpy.ndarray
         - Address offsets for each ensemble.
       * - dataid : numpy.ndarray
         - Data IDs for each ensemble.
    """

    def __init__(self, rdi_file, run="fortran"):
        self.filename = rdi_file
        if run == "fortran":
            (
                self.datatypes,
                self.bytes,
                self.byteskip,
                self.address_offset,
                self.dataid,
                self.ensembles,
                self.error,
            ) = fortreadrdi.fileheader(rdi_file)
            self.datatypes = self.datatypes[0 : self.ensembles]
            self.bytes = self.bytes[0 : self.ensembles]
            self.byteskip = self.byteskip[0 : self.ensembles]
            self.address_offset = self.address_offset[
                0 : self.ensembles, 0 : int(self.datatypes[0])
            ]
            self.dataid = self.dataid[0 : self.ensembles, 0 : int(self.datatypes[0])]
        elif run == "python":
            (
                self.datatypes,
                self.bytes,
                self.byteskip,
                self.address_offset,
                self.dataid,
                self.ensembles,
            ) = pyreadrdi.fileheader(rdi_file)

        else:
            sys.exit("Select python or fortran for run arg")

        self.warning = error_code(self.error)

    def data_types(self, ens=0):
        """
        Find and return the available data types for a given ensemble.

        Parameters
        ----------
        ens : int, optional
            Ensemble number. Default is 0.

        Returns
        -------
        id_name_array : list of str
            List of data type names for the given ensemble.
            
            
        Examples
        --------
        >>> rdi_file = 'path_to_rdi_file.adcp'
        >>> file_header = FileHeader(rdi_file)
        >>> data_types = file_header.data_types(ens=0)
        >>> print(data_types)
        ['Fixed Leader', 'Velocity', 'Correlation']
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
        Check if the file size and data types are consistent across the RDI file.

        Returns
        -------
        check : dict
            Dictionary containing the file size check results and data uniformity.
            
        Examples
        --------
        >>> rdi_file = 'path_to_rdi_file.adcp'
        >>> file_header = FileHeader(rdi_file)
        >>> check = file_header.check_file()
        >>> print(check)
        {
            'System File Size (B)': 10485760,
            'Calculated File Size (B)': 10485760,
            'File Size (MB)': 10.0,
            'File Size Match': True,
            'Byte Uniformity': True,
            'Data Type Uniformity': True
        }
        """
        file_stats = os.stat(self.filename)
        sys_file_size = file_stats.st_size
        cal_file_size = sum(self.bytes) + 2 * len(self.bytes)

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
        Print the results of the file size check and data uniformity.

        Returns
        -------
        None
        
        Examples
        --------
        >>> rdi_file = 'path_to_rdi_file.adcp'
        >>> file_header = FileHeader(rdi_file)
        >>> file_header.print_check_file()
        ---------------RDI FILE SIZE CHECK-------------------
        System file size = 10485760 B
        Calculated file size = 10485760 B
        File size in MB: 10.00 MB
        File sizes match!
        -----------------------------------------------------
        Total number of ensembles: 100
        Number of Bytes are consistent across all ensembles.
        Number of Data Types are consistent across all ensembles.
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

    def print_error(self):
        """
        Print the human-readable error message based on the error code.

        This method interprets the error code stored in `self.error` and prints
        the corresponding error message. It is useful for diagnosing issues with
        the RDI ADCP binary file.

        Returns
        -------
        None
        
        Examples
        --------
        >>> file_header = FileHeader('path_to_rdi_file.adcp')
        >>> file_header.print_error()
        'File healthy'  # Example output based on the error code
        """
        if self.error == 0:
            self.name_error = "File healthy"
        elif self.error == 1:
            self.name_error = "End of file"
        elif self.error == 2:
            self.name_error = "File corrupted"
        elif self.error == 3:
            self.name_error = "Wrong file type"
        elif self.error == 5:
            self.name_error = "Data type mismatch"
        else:
            self.name_error = "Unknown error"

        print(self.name_error)

        return


# FIXED LEADER CODES #


def flead_dict(fid, dim=2):
    """

    Parameters
    ----------
    fid : TYPE
        DESCRIPTION.

    Returns
    -------
    flead : TYPE
        DESCRIPTION.

    """
    fname = {
        "CPU Version": "int16",
        "CPU Revision": "int16",
        "System Config Code": "int16",
        "Real Flag": "int16",
        "Lag Length": "int16",
        "Beams": "int16",
        "Cells": "int16",
        "Pings": "int16",
        "Depth Cell Len": "int16",
        "Blank Transmit": "int16",
        "Signal Mode": "int16",
        "Correlation Thresh": "int16",
        "Code Reps": "int16",
        "Percent Good Min": "int16",
        "Error Velocity Thresh": "int16",
        "TP Minute": "int16",
        "TP Second": "int16",
        "TP Hundredth": "int16",
        "Coord Transform Code": "int16",
        "Head Alignment": "int16",
        "Head Bias": "int16",
        "Sensor Source Code": "int16",
        "Sensor Avail Code": "int16",
        "Bin 1 Dist": "int16",
        "Xmit Pulse Len": "int16",
        "Ref Layer Avg": "int16",
        "False Target Thresh": "int16",
        "Spare 1": "int16",
        "Transmit Lag Dist": "int16",
        "CPU Serial No": "int64",
        "System Bandwidth": "int16",
        "System Power": "int16",
        "Spare 2": "int16",
        "Instrument No": "int32",
        "Beam Angle": "int16",
    }

    flead = dict()
    counter = 1
    for key, value in fname.items():
        if dim == 2:
            flead[key] = getattr(np, value)(fid[:][counter])
        elif dim == 1:
            flead[key] = getattr(np, value)(fid[counter])
        else:
            print("ERROR: Higher dimensions not allowed")
            sys.exit()

        counter += 1

    return flead


class FixedLeader:
    """

    The class extracts Fixed Leader data. Fixed Leader data are non-dynamic
    or constants. They contain hardware information and ADCP data that only
    change based on certain ADCP Input commands. The data, generally, do not
    change within a file.

    Parameter:
    ----------
    rdi_file = TYPE STRING
        RDI ADCP binary file. The class can currently extract Workhorse,
        Ocean Surveyor, and DVS files.

    """

    def __init__(self, rdi_file, run="fortran"):
        self.filename = rdi_file

        if run == "fortran":
            self.data, self.ensembles, self.error = fortreadrdi.fixedleader(
                self.filename
            )
            self.data = self.data[0 : self.ensembles, 0:36].T
        elif run == "python":
            self.data, self.ensembles = pyreadrdi.fixedleader(self.filename)
        else:
            sys.exit("Select 'python' or 'fortran' for run arg")

        self.fleader = flead_dict(self.data)
        self.warning = error_code(self.error)

    def field(self, ens=0):
        """
        Returns Fixed Leader dictionary pairs for a single ensemble

        """

        f1 = np.array(self.data)
        return flead_dict(f1[:, ens], dim=1)

    def is_uniform(self):
        output = dict()
        for key, value in self.fleader.items():
            output[key] = check_equal(value)
        return output

    def system_configuration(self, ens=0):
        """

        Parameters
        ----------
        binary_code : TYPE INTEGER
            The parameter is the System Configuration field from the
            RDI Fixed Leader. (Integer converted converted from 2-bytes)
        Returns
        -------
        sys_cfg : TYPE DICTIONARY[STRING: STRING]
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

        binary_bits = format(self.fleader["System Config Code"][ens], "016b")
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

        Parameters
        ----------
        int_coord_trans : TYPE INTEGER
            DESCRIPTION.

        Returns
        -------
        co_tran : TYPE
            DESCRIPTION.

        """

        bit_group = format(self.fleader["Coord Transform Code"][ens], "08b")
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
        Checks for available sensors from FIXED LEADER.

        Parameters
        ----------
        ensemble : TYPE INTEGER, DEFAULT=0
            DESCRIPTION. Finds sensor source/source for a given ensemble.

        field: TYPE STRING
               OPTIONS:
               1) 'source' (DEFAULT):  Contains the selected source of
                environmental sensor data (EZ command).
               2) 'avail':  Reflects which sensors are available.
                            NOTE: As sound speed does not have a sensor, this
                            field is always 'False' for the option.
        Returns
        -------
        sensor : TYPE DICTIONARY[STRING: BOOLEAN]
            DESCRIPTION. Returns sensor name and boolean.

        """
        if field == "source":
            bit_group = format(self.fleader["Sensor Source Code"][ens], "08b")
        elif field == "avail":
            bit_group = format(self.fleader["Sensor Avail Code"][ens], "08b")
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
def vlead_dict(vid):
    """
    Returns a dictionary containing:
        1) Keys: All Variable Leader data names
        2) Item: Empty list

    Returns
    -------
    vlead

    """
    vname = {
        "RDI Ensemble": "int16",
        "RTC Year": "int16",
        "RTC Month": "int16",
        "RTC Day": "int16",
        "RTC Hour": "int16",
        "RTC Minute": "int16",
        "RTC Second": "int16",
        "RTC Hundredth": "int16",
        "Ensemble MSB": "int16",
        "Bit Result": "int16",
        "Speed of Sound": "int16",
        "Depth of Transducer": "int16",
        "Heading": "int32",
        "Pitch": "int16",
        "Roll": "int16",
        "Salinity": "int16",
        "Temperature": "int16",
        "MPT Minute": "int16",
        "MPT Second": "int16",
        "MPT Hundredth": "int16",
        "Hdg Std Dev": "int16",
        "Pitch Std Dev": "int16",
        "Roll Std Dev": "int16",
        "ADC Channel 0": "int16",
        "ADC Channel 1": "int16",
        "ADC Channel 2": "int16",
        "ADC Channel 3": "int16",
        "ADC Channel 4": "int16",
        "ADC Channel 5": "int16",
        "ADC Channel 6": "int16",
        "ADC Channel 7": "int16",
        "Error Status Word 1": "int16",
        "Error Status Word 2": "int16",
        "Error Status Word 3": "int16",
        "Error Status Word 4": "int16",
        "Reserved": "int16",
        "Pressure": "int32",
        "Pressure Variance": "int32",
        "Spare": "int16",
        "Y2K Century": "int16",
        "Y2K Year": "int16",
        "Y2K Month": "int16",
        "Y2K Day": "int16",
        "Y2K Hour": "int16",
        "Y2K Minute": "int16",
        "Y2K Second": "int16",
        "Y2K Hundredth": "int16",
    }

    vlead = dict()

    counter = 1
    for key, value in vname.items():
        vlead[key] = getattr(np, value)(vid[:][counter])
        counter += 1

    return vlead


class VariableLeader:
    """
    The class extracts Variable Leader Data.
    Variable Leader data refers to the dynamic ADCP data
    (from clocks/sensors) that change with each ping. The
    WorkHorse ADCP always sends Variable Leader data as output
    data (LSBs first).

    Parameter:
    ----------
    rdi_file = TYPE STRING
        RDI ADCP binary file. The class can currently extract Workhorse,
        Ocean Surveyor, and DVS files.

    """

    def __init__(self, rdi_file, run="fortran"):
        self.filename = rdi_file

        if run == "fortran":
            self.data, self.ensembles, self.error = fortreadrdi.variableleader(
                self.filename
            )
            self.data = self.data[0 : self.ensembles, 0:48].T
        elif run == "python":
            # Extraction starts here
            self.data, self.ensembles = pyreadrdi.variableleader(self.filename)
        else:
            sys.exit("Select 'python' or 'fortran' for run arg")

        self.vleader = vlead_dict(self.data)
        self.warning = error_code(self.error)

    def bit_result(self):
        """
        Extracts Bit Results from Variable Leader (Byte 13 & 14)
        This field is part of the WorkHorse ADCP’s Built-in Test function.
        A zero code indicates a successful BIT result.

        Note: Byte 14 used for future use.

        Returns
        -------
        test_field : TYPE DICTIONARY
            DESCRIPTION.

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
            bit_group = format(item, "016b")
            bitpos = 8
            for key, value in tfname.items():
                bitappend = getattr(np, value)(bit_group[bitpos])
                test_field[key] = np.append(test_field[key], bitappend)
                bitpos += 1

        return test_field

    def adc_channel(self, offset=-0.20):
        """
        Returns
        -------
        None.

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

        print(fixclass["Frequency"])

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


class Velocity:
    def __init__(self, filename, run="fortran"):
        error = 0
        if run == "fortran":
            data, ens, cell, beam, error = fortreadrdi.variables(filename, "velocity")
            data = data[0:beam, 0:cell, 0:ens]
        elif run == "python":
            data = pyreadrdi.variables(filename, "velocity")
            array_shape = np.shape(data)
            beam = array_shape[0]
            cell = array_shape[1]
            ens = array_shape[2]
        else:
            sys.exit("Select 'python' or 'fortran' for run arg")

        self.data = data
        self.error = error
        self.ensembles = ens
        self.cells = cell
        self.beams = beam

        # self.data, self.error = velocity(filename, run=run)
        self.filename = filename
        self.units = "mm/s"
        self.missing_value = "-32768"
        self.scale_factor = 1
        self.valid_min = -32768
        self.valid_max = 32768
        self.warning = error_code(self.error)


class Correlation:
    def __init__(self, filename, run="fortran"):
        error = 0
        if run == "fortran":
            data, ens, cell, beam, error = fortreadrdi.variables(
                filename, "correlation"
            )
            data = data[0:beam, 0:cell, 0:ens]
        elif run == "python":
            data = pyreadrdi.variables(filename, "correlation")
            array_shape = np.shape(data)
            beam = array_shape[0]
            cell = array_shape[1]
            ens = array_shape[2]
        else:
            sys.exit("Select 'python' or 'fortran' for run arg")

        self.data = data
        self.error = error
        self.ensembles = ens
        self.cells = cell
        self.beams = beam

        # self.data, self.error = correlation(filename, run=run)
        self.filename = filename
        self.units = ""
        self.scale_factor = 1
        self.valid_min = 0
        self.valid_max = 255
        self.long_name = "Correlation Magnitude"
        self.warning = error_code(self.error)


class Echo:
    def __init__(self, filename, run="fortran"):
        error = 0
        if run == "fortran":
            data, ens, cell, beam, error = fortreadrdi.variables(filename, "echo")
            data = data[0:beam, 0:cell, 0:ens]
        elif run == "python":
            data = pyreadrdi.variables(filename, "echo")
            array_shape = np.shape(data)
            beam = array_shape[0]
            cell = array_shape[1]
            ens = array_shape[2]
        else:
            sys.exit("Select 'python' or 'fortran' for run arg")

        self.data = data
        self.error = error
        self.ensembles = ens
        self.cells = cell
        self.beams = beam

        # self.data, self.error = echo(filename, run=run)
        self.filename = filename
        self.units = "counts"
        self.scale_factor = "0.45"
        self.valid_min = 0
        self.valid_max = 255
        self.long_name = "Echo Intensity"
        self.warning = error_code(self.error)


class PercentGood:
    def __init__(self, filename, run="fortran"):
        error = 0
        if run == "fortran":
            data, ens, cell, beam, error = fortreadrdi.variables(filename, "pgood")
            data = data[0:beam, 0:cell, 0:ens]
        elif run == "python":
            data = pyreadrdi.variables(filename, "percent good")
            array_shape = np.shape(data)
            beam = array_shape[0]
            cell = array_shape[1]
            ens = array_shape[2]
        else:
            sys.exit("Select 'python' or 'fortran' for run arg")

        self.data = data
        self.error = error
        self.ensembles = ens
        self.cells = cell
        self.beams = beam

        # self.data, self.error = percentgood(filename, run=run)
        self.filename = filename
        self.units = "percent"
        self.valid_min = 0
        self.valid_max = 100
        self.long_name = "Percent Good"
        self.warning = error_code(self.error)


class Status:
    def __init__(self, filename, run="fortran"):
        self.isFixedEnsemble = False
        error = 0
        if run == "fortran":
            data, ens, cell, beam = fortreadrdi.variables(filename, "status")
            data = data[0:beam, 0:cell, 0:ens]
        elif run == "python":
            data = pyreadrdi.variables(filename, "status")
            array_shape = np.shape(data)
            beam = array_shape[0]
            cell = array_shape[1]
            ens = array_shape[2]
        else:
            sys.exit("Select 'python' or 'fortran' for run arg")

        self.data = data
        self.error = error
        self.ensembles = ens
        self.cells = cell
        self.beams = beam

        # self.data, self.error = status(filename, run=run)
        self.filename = filename
        self.units = ""
        self.valid_min = 0
        self.valid_max = 1
        self.long_name = "Status Data Format"
        self.warning = error_code(self.error)


class ReadFile:
    def __init__(self, filename, run="fortran"):
        self.fileheader = FileHeader(filename, run=run)
        datatype_array = self.fileheader.data_types()
        error_array = {"Fileheader": self.fileheader.error}
        warning_array = {"Fileheader": self.fileheader.warning}
        ensemble_array = {"Fileheader": self.fileheader.ensembles}

        if "Fixed Leader" in datatype_array:
            self.fixedleader = FixedLeader(filename, run=run)
            error_array["Fixed Leader"] = self.fixedleader.error
            warning_array["Fixed Leader"] = self.fixedleader.warning
            ensemble_array["Fixed Leader"] = self.fixedleader.ensembles

        if "Variable Leader" in datatype_array:
            self.variableleader = VariableLeader(filename, run=run)
            error_array["Variable Leader"] = self.variableleader.error
            warning_array["Variable Leader"] = self.variableleader.warning
            ensemble_array["Variable Leader"] = self.variableleader.ensembles

        if "Velocity" in datatype_array:
            self.velocity = Velocity(filename, run=run)
            error_array["Velocity"] = self.velocity.error
            warning_array["Velocity"] = self.velocity.warning
            ensemble_array["Velocity"] = self.velocity.ensembles

        if "Correlation" in datatype_array:
            self.correlation = Correlation(filename, run=run)
            error_array["Correlation"] = self.correlation.error
            warning_array["Correlation"] = self.correlation.warning
            ensemble_array["Correlation"] = self.correlation.ensembles

        if "Echo" in datatype_array:
            self.echo = Echo(filename, run=run)
            error_array["Echo"] = self.echo.error
            warning_array["Echo"] = self.echo.warning
            ensemble_array["Echo"] = self.echo.ensembles

        if "Percent Good" in datatype_array:
            self.percentgood = PercentGood(filename, run=run)
            error_array["Percent Good"] = self.percentgood.error
            warning_array["Percent Good"] = self.percentgood.warning
            ensemble_array["Percent Good"] = self.percentgood.ensembles

        if "Status" in datatype_array:
            self.status = Status(filename, run=run)
            error_array["Status"] = self.status.error
            warning_array["Status"] = self.status.warning
            ensemble_array["Status"] = self.status.ensembles

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

    def fixensemble(self, min_cutoff=0):
        datatype_array = self.fileheader.data_types()
        # Check if the number of ensembles in a data type
        # is less than min_cutoff.
        # Some data type can have zero ensembles
        dtens = self.ensemble_value_array
        new_array = dtens[dtens > min_cutoff]
        minens = np.min(new_array)

        if not self.isEnsembleEqual:
            if "Fixed Leader" in datatype_array:
                self.fixedleader.data = self.fixedleader.data[:, :minens]
                self.fixedleader.ensembles = minens
            if "Variable Leader" in datatype_array:
                self.variableleader.data = self.variableleader.data[:, :minens]
                self.variableleader.ensembles = minens
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
            print(f"Ensembles fixed to {minens}. All data types have same ensembles.")
        else:
            print(
                "WARNING: No response was initiated. All data types have same ensemble."
            )

        self.isFixedEnsemble = True
