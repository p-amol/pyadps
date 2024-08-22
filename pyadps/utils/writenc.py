#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:56:44 2020

This module provides functions to create and manipulate NetCDF files from ADCP binary data. 
It includes functionality for adding global attributes, creating NetCDF dimensions and variables, 
and writing processed data.

@author: amol
"""

import time

import netCDF4 as nc4
import numpy as np
import pandas as pd
import streamlit as st
from netCDF4 import date2num

from pyadps.utils import readrdi as rd

# import readrdi as rd


def pd2nctime(time, t0="hours since 2000-01-01"):
    """
    Converts pandas datetime format to NetCDF datetime format.

    Parameters
    ----------
    time : pandas.DatetimeIndex
        The time data in pandas datetime format.
    t0 : str, optional
        The reference time for the NetCDF datetime format. Default is "hours since 2000-01-01".

    Returns
    -------
    numpy.ndarray
        The time data converted to NetCDF datetime format.
    
    Example
    -------
    >>> import pandas as pd
    >>> from pyadps.utils.writenc import pd2nctime
    >>> time_index = pd.date_range('2024-01-01', periods=10, freq='H')
    >>> pd2nctime(time_index)
    array([...])
    """
    dti = pd.DatetimeIndex(time)
    pydt = dti.to_pydatetime()
    nctime = date2num(pydt, t0)
    return nctime


def flead_ncatt(fl_obj, ncfile_id, ens=0):
    """
    Adds global attributes to the NetCDF file from the Fixed Leader object.

    Parameters
    ----------
    fl_obj : FixedLeader
        The FixedLeader object containing the attributes to add.
    ncfile_id : netCDF4.Dataset
        The NetCDF file object to which attributes are added.
    ens : int, optional
        The ensemble index to extract attributes for. Default is 0.

    Returns
    -------
    None

    Example
    -------
    >>> from pyadps.utils.readrdi import FixedLeader
    >>> from pyadps.utils.writenc import flead_ncatt
    >>> fl = FixedLeader('path/to/file')
    >>> ncfile = nc4.Dataset('path/to/output.nc', 'w')
    >>> flead_ncatt(fl, ncfile)
    """

    ncfile_id.history = "Created " + time.ctime(time.time())
    for key, value in fl_obj.fleader.items():
        format_key = key.replace(" ", "_")
        setattr(ncfile_id, format_key, format(value[ens], "d"))

    for key, value in fl_obj.system_configuration(ens).items():
        format_key = key.replace(" ", "_")
        setattr(ncfile_id, format_key, format(value))

    for key, value in fl_obj.ex_coord_trans(ens).items():
        format_key = key.replace(" ", "_")
        setattr(ncfile_id, format_key, format(value))

    for field in ["source", "avail"]:
        for key, value in fl_obj.ez_sensor(ens, field).items():
            format_key = key.replace(" ", "_")
            format_key = format_key + "_" + field.capitalize()
            setattr(ncfile_id, format_key, format(value))


def main(infile, outfile):

    """
    Creates a NetCDF file from an ADCP binary file. Stores 3-D data types like velocity, echo, 
    correlation, and percent good.

    Parameters
    ----------
    infile : str
        Path to the input ADCP binary file.
    outfile : str
        Path to the output NetCDF file.

    Returns
    -------
    None

    Example
    -------
    >>> from pyadps.utils.writenc import main
    >>> main('path/to/adcp_file.bin', 'path/to/output.nc')
    """

    outnc = nc4.Dataset(outfile, "w", format="NETCDF4")

    flead = rd.FixedLeader(infile)
    cell_list = flead.fleader["Cells"]
    beam_list = flead.fleader["Beams"]

    # Dimensions
    outnc.createDimension("ensemble", None)
    outnc.createDimension("cell", max(cell_list))
    outnc.createDimension("beam", max(beam_list))

    # Variables
    # Dimension Variables
    ensemble = outnc.createVariable("ensemble", "u4", ("ensemble",))
    ensemble.axis = "T"
    cell = outnc.createVariable("cell", "i2", ("cell",))
    cell.axis = "Z"
    beam = outnc.createVariable("beam", "i2", ("beam",))
    beam.axis = "X"

    # Variables

    # Data
    cell[:] = np.arange(1, max(cell_list) + 1, 1)
    beam[:] = np.arange(1, max(beam_list) + 1, 1)

    varlist = rd.FileHeader(infile).data_types(1)
    varlist.remove("Fixed Leader")
    varlist.remove("Variable Leader")

    varid = [0] * len(varlist)

    for i, item in enumerate(varlist):
        if item == "Velocity":
            varid[i] = outnc.createVariable(
                item, "i2", ("ensemble", "cell", "beam"), fill_value=-32768
            )
            varid[i].missing_value = -32768
            var = rd.variables(infile, item)

        else:
            # Unsigned integers might be assigned for future netcdf versions
            format_item = item.replace(" ", "_")  # For percent good
            varid[i] = outnc.createVariable(
                format_item, "i2", ("ensemble", "cell", "beam")
            )
            var = np.array(rd.variables(infile, item), dtype="int16")

        vshape = var.T.shape
        print(vshape)
        if i == 0:
            ensemble[:] = np.arange(1, vshape[0] + 1, 1)
        varid[i][0 : vshape[0], 0 : vshape[1], 0 : vshape[2]] = var.T

    # outnc.history = "Created " + time.ctime(time.time())
    flead_ncatt(flead, outnc)

    outnc.close()


def vlead_nc(infile, outfile):
    """
    Creates a NetCDF file containing Variable Leader data.

    Parameters
    ----------
    infile : str
        Path to the input ADCP binary file.
    outfile : str
        Path to the output NetCDF file.

    Returns
    -------
    None

    Example
    -------
    >>> from pyadps.utils.writenc import vlead_nc
    >>> vlead_nc('path/to/adcp_file.bin', 'path/to/output.nc')
    """
    outnc = nc4.Dataset(outfile, "w", format="NETCDF4")

    # Dimensions
    outnc.createDimension("ensemble", None)

    # Variables
    # Dimension Variables
    ensemble = outnc.createVariable("ensemble", "i4", ("ensemble",))
    ensemble.axis = "T"

    vlead = rd.VariableLeader(infile)
    vdict = vlead.vleader
    varid = [0] * len(vdict)

    i = 0

    for key, values in vdict.items():
        format_item = key.replace(" ", "_")
        varid[i] = outnc.createVariable(
            format_item, "i4", "ensemble", fill_value=-32768
        )
        var = values
        vshape = var.shape
        if i == 0:
            ensemble[:] = np.arange(1, vshape[0] + 1, 1)

        varid[i][0 : vshape[0]] = var
        i += 1

    outnc.close()


def finalnc(outfile, depth, time, data, t0="hours since 2000-01-01"):
    """
    Creates a processed NetCDF file with depth, time, and data variables.

    Parameters
    ----------
    outfile : str
        Path to the output NetCDF file.
    depth : numpy.ndarray
        Array containing the depth values (negative for depth).
    time : pandas.DatetimeIndex
        Time axis in pandas datetime format.
    data : numpy.ndarray
        3-D array containing the velocity data (beam, depth, time).
    t0 : str, optional
        The reference time for the NetCDF datetime format. Default is "hours since 2000-01-01".

    Returns
    -------
    None

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pyadps.utils.writenc import finalnc
    >>> depth = np.array([1.0, 2.0, 3.0])
    >>> time = pd.date_range('2024-01-01', periods=3, freq='H')
    >>> data = np.random.rand(3, 3, 3)
    >>> finalnc('path/to/output.nc', depth, time, data)
    """
    ncfile = nc4.Dataset(outfile, mode="w", format="NETCDF4")
    zsize = len(depth)
    tsize = len(time)
    ncfile.createDimension("depth", zsize)
    ncfile.createDimension("time", tsize)

    z = ncfile.createVariable("depth", np.float32, ("depth"))
    z.units = "m"
    z.long_name = "depth"

    t = ncfile.createVariable("time", np.float32, ("time"))
    t.units = t0
    t.long_name = "time"

    # Create 2D variables
    uvel = ncfile.createVariable("u", np.float32, ("time", "depth"), fill_value=-32768)
    uvel.units = "cm/s"
    uvel.long_name = "zonal_velocity"

    vvel = ncfile.createVariable("v", np.float32, ("time", "depth"), fill_value=-32768)
    vvel.units = "cm/s"
    vvel.long_name = "meridional_velocity"

    wvel = ncfile.createVariable("w", np.float32, ("time", "depth"), fill_value=-32768)
    wvel.units = "cm/s"
    wvel.long_name = "vertical_velocity"

    evel = ncfile.createVariable(
        "err", np.float32, ("time", "depth"), fill_value=-32768
    )
    evel.units = "cm/s"
    evel.long_name = "error_velocity"

    nctime = pd2nctime(time, t0)
    # write data
    z[:] = depth * -1
    t[:] = nctime
    uvel[:, :] = data[0, :, :].T
    vvel[:, :] = data[1, :, :].T
    wvel[:, :] = data[2, :, :].T
    evel[:, :] = data[3, :, :].T

    ncfile.close()
