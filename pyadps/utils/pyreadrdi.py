#!/usr/bin/env python3

import sys
from struct import unpack

import numpy as np


def fileheader(rdi_file):
    filename = rdi_file
    bfile = open(filename, "rb")
    bfile.seek(0, 0)

    bskip = i = 0

    hid = [None] * 5

    # Empty list assigned to Header ID variables
    headerid = np.array([], dtype="int8")
    sourceid = np.array([], dtype="int8")
    byte = np.array([], dtype="int16")
    spare = np.array([], dtype="int8")
    datatype = np.array([], dtype="int16")
    address_offset = []

    # Variables required find and change the position of File Handle
    # dataid = List contains available data types in an ensemble
    # byteskip = Total bytes required to seek to an ensemble

    dataid = []
    byteskip = np.array([], dtype="int32")

    while byt := bfile.read(6):
        (hid[0], hid[1], hid[2], hid[3], hid[4]) = unpack("<BBHBB", byt)
        headerid = np.append(headerid, np.int8(hid[0]))
        sourceid = np.append(sourceid, np.int16(hid[1]))
        byte = np.append(byte, np.int16(hid[2]))
        spare = np.append(spare, np.int16(hid[3]))
        datatype = np.append(datatype, np.int16(hid[4]))
        # START FILE ERROR HANDLING CONDITIONS
        if i == 0:
            if headerid[0] != 127:
                sys.exit(
                    "\nERROR: File type not recognized. \
                          Enter a valid RDI file."
                )
        else:
            if headerid[i] != 127:
                sys.exit(
                    f"\nERROR: The file appears corrupted.\
                          Ensemble no. {i+1} does not have a valid RDI \
                          file ID"
                )

        if i == 0:
            if sourceid[0] != 127:
                sys.exit(
                    "\nERROR: RDI file detected but cannot identify \
                          data Source ID. Exiting program."
                )
        else:
            if sourceid[i] != 127:
                sys.exit(
                    f"\nERROR: The file appears corrupted.\
                          Ensemble no. {i+1} does not have a valid \
                          Source ID. Exiting program."
                )

        # Get the address offset for each data type
        dbyte = bfile.read(2 * datatype[i])
        data = unpack("H" * datatype[i], dbyte)
        address_offset.append(data)

        skip_array = [None] * datatype[i]
        for dtype in range(datatype[i]):
            bseek = bskip + address_offset[i][dtype]
            bfile.seek(bseek, 0)
            readbyte = bfile.read(2)
            skip_array[dtype] = int.from_bytes(
                readbyte, byteorder="little", signed="False"
            )

        dataid.append(skip_array)
        # bytekip is the number of bytes to skip to reach
        # an ensemble from beginning of file.
        # ?? Should byteskip be from current position ??
        bskip = bskip + byte[i] + 2
        bfile.seek(bskip, 0)
        byteskip = np.append(byteskip, np.int32(bskip))
        i += 1

    ensemble = i
    bfile.close()

    address_offset = np.array(address_offset)
    dataid = np.array(dataid)

    return datatype, byte, byteskip, address_offset, dataid, ensemble


# FIXED LEADER CODES #


def fixedleader(rdi_file):

    # Extraction starts here
    filename = rdi_file

    # Extract info from HeaderID class
    datatype, byte, byteskip, offset, idarray, ensemble = fileheader(filename)

    bfile = open(filename, "rb")
    bfile.seek(0, 0)

    # Create 2-D dictionary of size (ensemble * variables)
    fid = [[0] * ensemble for x in range(36)]
    # fid = np.empty((36, ens), dtype='str')

    for i in range(ensemble):
        fbyteskip = "NaN"
        # Check for the Variable Leader ID
        for count, item in enumerate(idarray[i]):
            if item in (0, 1):
                fbyteskip = offset[1][count]

        if fbyteskip == "NaN":
            sys.exit(f"Fixed Leader ID not found for Ensemble {i}")

        bfile.seek(fbyteskip, 1)

        # Read all the bytes from Fixed Leader #
        # CAUTION: Some files may not have 59 bytes!
        bdata = bfile.read(59)

        # Fixed Leader ID, CPU Version no. & Revision no.
        (fid[0][i], fid[1][i], fid[2][i]) = unpack("<HBB", bdata[0:4])
        if fid[0][i] not in (0, 1):
            sys.exit(f"Fixed Leader not found for Ensemble {i}")
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
        # fid[30][i] = unpack('>Q', bdata[42:50])
        fid[30][i] = int.from_bytes(bdata[42:50], byteorder="big", signed="False")
        # System bandwidth, system power & Spare
        (fid[31][i], fid[32][i], fid[33][i]) = unpack("<HBB", bdata[50:54])
        # Instrument serial number & Beam angle
        (fid[34][i], fid[35][i]) = unpack("<LB", bdata[54:59])

        bfile.seek(byteskip[i], 0)

    bfile.close()
    return fid, ensemble


def variableleader(rdi_file):
    filename = rdi_file

    # Extract info from HeaderID class
    datatype, byte, byteskip, offset, idarray, ensemble = fileheader(filename)

    # Create 2-D dictionary of size (ensemble * variables)
    vid = [[0] * ensemble for x in range(48)]

    # Extraction starts here

    bfile = open(filename, "rb")
    bfile.seek(0, 0)

    for i in range(ensemble):
        fbyteskip = "NaN"
        # Check for the Variable Leader ID
        for count, item in enumerate(idarray[i]):
            if item in (128, 129):
                fbyteskip = offset[1][count]

        if fbyteskip == "NaN":
            sys.exit(f"Variable Leader ID not found for Ensemble {i}")

        bfile.seek(fbyteskip, 1)

        # Read all the bytes from Fixed Leader
        # CAUTION: Some files may not have 65 bytes!
        bdata = bfile.read(65)

        # Start extracting Variable Leader ID & RDI Ensemble.
        (vid[0][i], vid[1][i]) = unpack("<HH", bdata[0:4])
        if vid[0][i] not in (128, 129):
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
        (vid[32][i], vid[33][i], vid[34][i], vid[35][i]) = unpack("<BBBB", bdata[42:46])
        # Extract Reserved, Pressure, Pressure Variance & Spare
        (vid[36][i], vid[37][i], vid[38][i], vid[39][i]) = unpack("<HiiB", bdata[46:57])
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
    bfile.close()
    return vid, ensemble


# DATA CODES (VELOCITY, CORRELATION, ECHO, PERCENT GOOD, & STATUS)
def variables(filename, var_name):

    varid = dict()

    # Define file ids:
    varid = {
        "velocity": (256, 257),
        "correlation": (512, 513),
        "echo": (768, 769),
        "percent good": (1024, 1025),
        "status": (1280, 1281),
    }

    # Get data from FileHeader

    datatype, byte, byteskip, offset, idarray, ensemble = fileheader(filename)

    # Get data from FixedLeader
    flead, ensembles = fixedleader(filename)
    cell = flead[7][:]
    beam = flead[6][:]

    bfile = open(filename, "rb")
    bfile.seek(0, 0)

    vid = varid.get(var_name)

    if not vid:
        sys.exit(
            """ERROR: Variable name not found.
                 List of permissible variable names: 'velocity',
                 'correlation', 'echo', 'percent good', 'status'"""
        )

    # Assign Data Types (16 bit for VELOCITY & 8 bit for others)
    if var_name == "velocity":
        var_array = np.zeros((max(beam), max(cell), ensemble), dtype="int16")
        # var_array = np.zeros((ensemble, max(cell), max(beam)), dtype="int16")
        bitstr = "<h"
        bitint = 2
    else:
        var_array = np.zeros((max(beam), max(cell), ensemble), dtype="int8")
        # var_array = np.zeros((ensemble, max(cell), max(beam)), dtype="uint8")

        bitstr = "<B"
        bitint = 1

    # Read the data
    for i in range(ensemble):
        # Check for the variable ID
        for count, item in enumerate(idarray[i]):
            if item in vid:
                fbyteskip = offset[1][count]
        try:
            fbyteskip
        except NameError:
            sys.exit(f"{var_name} ID not found for Ensemble {i}")

        # Seek to the variable ID & Read the ID
        bfile.seek(fbyteskip, 1)
        bdata = bfile.read(2)

        for cno in range(cell[i]):
            for bno in range(beam[i]):
                bdata = bfile.read(bitint)
                (varunpack) = unpack(bitstr, bdata)
                var_array[bno][cno][i] = varunpack[0]
                # var_array[i][cno][bno] = varunpack[0]

        bfile.seek(byteskip[i], 0)

    bfile.close()

    return var_array
