#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:56:44 2020

@author: amol
"""

import time

import netCDF4 as nc4
import numpy as np
import pandas as pd
import streamlit as st
from netCDF4 import date2num

from pyadps.utils import readrdi as rd



def pd2nctime(time, t0="hours since 2000-01-01"):
    """
    Function to convert pandas datetime format to netcdf datetime format.
    """
    dti = pd.DatetimeIndex(time)
    pydt = dti.to_pydatetime()
    nctime = date2num(pydt, t0)
    return nctime


def flead_ncatt(fl_obj, ncfile_id, ens=0):
    """
    Adds global attributes to netcdf file. All variables from Fixed Leader
    are appended for a given ensemble.

        Parameters
        ----------
        fl_obj : TYPE, FixedLeader Object
            DESCRIPTION.
            ncfile_id : TYPE
            DESCRIPTION.
        ens : TYPE, INTEGER optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

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


def main(infile, outfile, attributes=None):
    """
    Main function to create netcdf file. Stores 3-D data types like
    velocity, echo, correlation, and percent good.

    Args:
        infile (string): Input file path including filename
        outfile (string): Output file path including filename

    Returns
    -------
    None.

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
            vel = getattr(rd, item)
            var = vel(infile).data
            # var = rd.variables(infile, item)

        else:
            # Unsigned integers might be assigned for future netcdf versions
            format_item = item.replace(" ", "")  # For percent good
            varid[i] = outnc.createVariable(
                format_item, "i2", ("ensemble", "cell", "beam")
            )
            datatype = getattr(rd, format_item)
            var = np.array(datatype(infile).data, dtype="int16")
            # var = np.array(rd.variables(infile, item), dtype="int16")

        vshape = var.T.shape
        if i == 0:
            ensemble[:] = np.arange(1, vshape[0] + 1, 1)
        varid[i][0 : vshape[0], 0 : vshape[1], 0 : vshape[2]] = var.T
        
    # Add global attributes if provided
    if attributes:
        for key, value in attributes.items():
            setattr(outnc, key, str(value))  # Convert to string to store in NetCDF metadata

    # outnc.history = "Created " + time.ctime(time.time())
    flead_ncatt(flead, outnc)
    

    outnc.close()


def vlead_nc(infile, outfile, attributes=None):
    """
    Function to create ncfile containing Variable Leader.

    Args:
        infile (string): Input file path including filename
        outfile (string): Output file path including filename
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
        
    # Add global attributes if provided
    if attributes:
        for key, value in attributes.items():
            setattr(outnc, key, str(value))  # Store attributes as strings

    outnc.close()


def finalnc(outfile, depth, time, data, t0="hours since 2000-01-01", attributes=None):
    """
    Function to create the processed NetCDF file.

    Args:
        outfile (string): Output file path
        depth (numpy array): Contains the depth values (negative for depth)
        time (pandas array): Time axis in Pandas datetime format
        data (numpy array): Velocity (beam, depth, time)
        t0 (string): Time unit and origin
    """
    ncfile = nc4.Dataset(outfile, mode="w", format="NETCDF4")
    # Check if depth is scalar or array
    if np.isscalar(depth):
        zsize = 1  # Handle scalar depth
    else:
        zsize = len(depth)  # Handle array depth
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
    
    # Add global attributes if provided
    if attributes:
        for key, value in attributes.items():
            setattr(ncfile, key, str(value))  # Store attributes as strings

    ncfile.close()
