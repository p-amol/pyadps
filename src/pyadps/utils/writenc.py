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


def rawnc(
    infile,
    outfile,
    time,
    axis_option=None,
    attributes=None,
    t0="hours since 2000-01-01",
):
    """
    rawnc is a function to create netcdf file. Stores 3-D data types like
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
    # Define the primary axis based on axis_option
    if axis_option == "ensemble":
        outnc.createDimension("ensemble", None)
        primary_axis = "ensemble"
        ensemble = outnc.createVariable("ensemble", "i4", ("ensemble",))
        ensemble.axis = "T"
    elif axis_option == "time":
        tsize = len(time)
        outnc.createDimension("time", tsize)
        primary_axis = "time"
        time_var = outnc.createVariable("time", "i4", ("time",))
        time_var.axis = "T"
        time_var.units = t0
        time_var.long_name = "time"

        # Convert time_data to numerical format
        nctime = pd2nctime(time, t0)
        time_var[:] = nctime

    else:
        raise ValueError(f"Invalid axis_option: {axis_option}.")

    outnc.createDimension("cell", max(cell_list))
    outnc.createDimension("beam", max(beam_list))

    # Variables
    # Dimension Variables
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
                item, "i2", (primary_axis, "cell", "beam"), fill_value=-32768
            )
            # varid[i].missing_value = -32768
            vel = getattr(rd, item)
            var = vel(infile).data
            # var = rd.variables(infile, item)

        else:
            # Unsigned integers might be assigned for future netcdf versions
            format_item = item.replace(" ", "")  # For percent good
            varid[i] = outnc.createVariable(
                format_item, "i2", (primary_axis, "cell", "beam")
            )
            datatype = getattr(rd, format_item)
            var = np.array(datatype(infile).data, dtype="int16")
            # var = np.array(rd.variables(infile, item), dtype="int16")

        vshape = var.T.shape
        if i == 0:
            if primary_axis == "ensemble":
                ensemble[:] = np.arange(1, vshape[0] + 1, 1)

        varid[i][0 : vshape[0], 0 : vshape[1], 0 : vshape[2]] = var.T

    # Add global attributes if provided
    if attributes:
        for key, value in attributes.items():
            setattr(
                outnc, key, str(value)
            )  # Convert to string to store in NetCDF metadata

    # outnc.history = "Created " + time.ctime(time.time())
    flead_ncatt(flead, outnc)

    outnc.close()

def flead_nc(
    infile,
    outfile,
    time,
    axis_option=None,
    attributes=None,
    t0="hours since 2000-01-01",
):
    """
    Function to create ncfile containing Variable Leader.

    Args:
        infile (string): Input file path including filename
        outfile (string): Output file path including filename
    """
    outnc = nc4.Dataset(outfile, "w", format="NETCDF4")

    # Dimensions
    # Define the primary axis based on axis_option
    if axis_option == "ensemble":
        outnc.createDimension("ensemble", None)
        primary_axis = "ensemble"
        ensemble = outnc.createVariable("ensemble", "i4", ("ensemble",))
        ensemble.axis = "T"
    elif axis_option == "time":
        tsize = len(time)
        outnc.createDimension("time", tsize)
        primary_axis = "time"
        time_var = outnc.createVariable("time", "i4", ("time",))
        time_var.axis = "T"
        time_var.units = t0
        time_var.long_name = "time"

        # Convert time_data to numerical format
        nctime = pd2nctime(time, t0)
        time_var[:] = nctime

    else:
        raise ValueError(f"Invalid axis_option: {axis_option}.")

    # Variables

    flead = rd.FixedLeader(infile)
    fdict = flead.fleader
    varid = [0] * len(fdict)

    i = 0

    for key, values in fdict.items():
        format_item = key.replace(" ", "_")
        varid[i] = outnc.createVariable(
            format_item, "i4", primary_axis, fill_value=-32768
        )
        var = values
        vshape = var.shape
        if i == 0:
            if primary_axis == "ensemble":
                ensemble[:] = np.arange(1, vshape[0] + 1, 1)

        varid[i][0 : vshape[0]] = var
        i += 1

    # Add global attributes if provided
    if attributes:
        for key, value in attributes.items():
            setattr(outnc, key, str(value))  # Store attributes as strings

    outnc.close()

def vlead_nc(
    infile,
    outfile,
    time,
    axis_option=None,
    attributes=None,
    t0="hours since 2000-01-01",
):
    """
    Function to create ncfile containing Variable Leader.

    Args:
        infile (string): Input file path including filename
        outfile (string): Output file path including filename
    """
    outnc = nc4.Dataset(outfile, "w", format="NETCDF4")

    # Dimensions
    # Define the primary axis based on axis_option
    if axis_option == "ensemble":
        outnc.createDimension("ensemble", None)
        primary_axis = "ensemble"
        ensemble = outnc.createVariable("ensemble", "i4", ("ensemble",))
        ensemble.axis = "T"
    elif axis_option == "time":
        tsize = len(time)
        outnc.createDimension("time", tsize)
        primary_axis = "time"
        time_var = outnc.createVariable("time", "i4", ("time",))
        time_var.axis = "T"
        time_var.units = t0
        time_var.long_name = "time"

        # Convert time_data to numerical format
        nctime = pd2nctime(time, t0)
        time_var[:] = nctime

    else:
        raise ValueError(f"Invalid axis_option: {axis_option}.")

    # Variables

    vlead = rd.VariableLeader(infile)
    vdict = vlead.vleader
    varid = [0] * len(vdict)

    i = 0

    for key, values in vdict.items():
        format_item = key.replace(" ", "_")
        varid[i] = outnc.createVariable(
            format_item, "i4", primary_axis, fill_value=-32768
        )
        var = values
        vshape = var.shape
        if i == 0:
            if primary_axis == "ensemble":
                ensemble[:] = np.arange(1, vshape[0] + 1, 1)

        varid[i][0 : vshape[0]] = var
        i += 1

    # Add global attributes if provided
    if attributes:
        for key, value in attributes.items():
            setattr(outnc, key, str(value))  # Store attributes as strings

    outnc.close()


def finalnc(
    outfile, depth, final_mask, final_echo, final_corr , final_pgood, time, data, t0="hours since 2000-01-01", attributes=None
):
    """
    Function to create the processed NetCDF file.

    Args:
        outfile (string): Output file path
        depth (numpy array): Contains the depth values (negative for depth)
        time (pandas array): Time axis in Pandas datetime format
        data (numpy array): Velocity (beam, depth, time)
        t0 (string): Time unit and origin
    """
    fill = -32768

    # Change velocity to cm/s
    data = data.astype(np.float64)
    data[data > fill] /= 10

    # Change depth to positive
    # depth = abs(depth)

    # Reverse the arrays if depth in descending order
    if np.all(depth[:-1] >= depth[1:]):
        depth = depth[::-1]
        data = data[:, ::-1, :]
        final_mask = final_mask[::-1, :]
        final_echo = final_echo[:,::-1, :]
        final_corr = final_corr[:,::-1, :]
        final_pgood = final_pgood[:,::-1, :]
    
    # adding mask to echo,correlation,pgood 

#    final_echo = final_echo * final_mask
#    final_corr = final_corr * final_mask
#    final_pgood = final_pgood * final_mask

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
    z.positive = "down"

    t = ncfile.createVariable("time", np.float32, ("time"))
    t.units = t0
    t.long_name = "time"

    # Create 2D variables
    uvel = ncfile.createVariable("u", np.float32, ("time", "depth"), fill_value=fill)
    uvel.units = "cm/s"
    uvel.long_name = "zonal_velocity"

    vvel = ncfile.createVariable("v", np.float32, ("time", "depth"), fill_value=fill)
    vvel.units = "cm/s"
    vvel.long_name = "meridional_velocity"

    wvel = ncfile.createVariable("w", np.float32, ("time", "depth"), fill_value=fill)
    wvel.units = "cm/s"
    wvel.long_name = "vertical_velocity"

    evel = ncfile.createVariable(
        "err", np.float32, ("time", "depth"), fill_value=-32768
    )
    evel.units = "cm/s"
    evel.long_name = "error_velocity"

    mvel = ncfile.createVariable("mask", np.float32, ("time", "depth"), fill_value=fill)
    mvel.long_name = "Velocity Mask (1: bad value, 0: good value)"

    AGC1 = ncfile.createVariable("AGC1", np.float32, ("time", "depth"), fill_value=-32768)
    AGC1.long_name = "Echo intensity (AGC) Beam1"

    AGC2 = ncfile.createVariable("AGC2", np.float32, ("time", "depth"), fill_value=-32768)
    AGC2.long_name = "Echo intensity (AGC) Beam1"

    AGC3 = ncfile.createVariable("AGC3", np.float32, ("time", "depth"), fill_value=-32768)
    AGC3.long_name = "Echo intensity (AGC) Beam1"

    AGC4 = ncfile.createVariable("AGC4", np.float32, ("time", "depth"), fill_value=-32768)
    AGC4.long_name = "Echo intensity (AGC) Beam1"

    corr1 = ncfile.createVariable("corr1", np.float32, ("time", "depth"), fill_value=-32768)
    corr1.long_name = "Beam 1 correlation"

    corr2 = ncfile.createVariable("corr2", np.float32, ("time", "depth"), fill_value=-32768)
    corr2.long_name = "Beam 2 correlation"

    corr3 = ncfile.createVariable("corr3", np.float32, ("time", "depth"), fill_value=-32768)
    corr3.long_name = "Beam 3 correlation"

    corr4 = ncfile.createVariable("corr4", np.float32, ("time", "depth"), fill_value=-32768)
    corr4.long_name = "Beam 4 correlation"  

    PGd1 = ncfile.createVariable("PGd1", np.float32, ("time", "depth"), fill_value=-32768)
    PGd1.long_name = "Percent Good Beam 1"

    PGd2 = ncfile.createVariable("PGd2", np.float32, ("time", "depth"), fill_value=-32768)
    PGd2.long_name = "Percent Good Beam 2"

    PGd3 = ncfile.createVariable("PGd3", np.float32, ("time", "depth"), fill_value=-32768)
    PGd3.long_name = "Percent Good Beam 3"

    PGd4 = ncfile.createVariable("PGd4", np.float32, ("time", "depth"), fill_value=-32768)
    PGd4.long_name = "Percent Good Beam 4"   

    nctime = pd2nctime(time, t0)
    # write data
    z[:] = depth
    t[:] = nctime
    uvel[:, :] = data[0, :, :].T
    vvel[:, :] = data[1, :, :].T
    wvel[:, :] = data[2, :, :].T
    evel[:, :] = data[3, :, :].T
    mvel[:, :] = final_mask.T
    AGC1[:, :] = final_echo[0, :, :].T
    AGC2[:, :] = final_echo[1, :, :].T
    AGC3[:, :] = final_echo[2, :, :].T
    AGC4[:, :] = final_echo[3, :, :].T
    corr1[:, :] = final_corr[0, :, :].T
    corr2[:, :] = final_corr[1, :, :].T
    corr3[:, :] = final_corr[2, :, :].T
    corr4[:, :] = final_corr[3, :, :].T
    PGd1[:, :] = final_pgood[0, :, :].T
    PGd2[:, :] = final_pgood[1, :, :].T
    PGd3[:, :] = final_pgood[2, :, :].T
    PGd4[:, :] = final_pgood[3, :, :].T

    # Add global attributes if provided
    if attributes:
        for key, value in attributes.items():
            setattr(ncfile, key, str(value))  # Store attributes as strings

    ncfile.mask_applied = "True"

    ncfile.close()
