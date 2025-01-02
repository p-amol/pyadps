import os
import tempfile

import configparser
import json
import streamlit as st
import numpy as np
import pandas as pd
import utils.writenc as wr
from utils import readrdi
from utils.profile_test import side_lobe_beam_angle,manual_cut_bins
from utils.profile_test import regrid2d, regrid3d
from utils.signal_quality import (
    default_mask,
    ev_check,
    false_target,
    pg_check,
    echo_check,
    correlation_check
)
from utils.velocity_test import (
    despike,
    flatline,
    velocity_cutoff,
    wmm2020api,
    velocity_modifier
)

@st.cache_data
def file_write(path,date, axis_option, attributes, add_attribute=True):
    tempdirname = tempfile.TemporaryDirectory(delete=False)
    st.session_state.rawfilename = tempdirname.name + "/rawfile.nc"
    
    if add_attribute:
        wr.rawnc(path, st.session_state.rawfilename, date, axis_option, attributes=attributes)
    else:
        wr.rawnc(path, st.session_state.rawfilename, date, axis_option)

@st.cache_data
def file_write_vlead(path, axis_option, date, attributes, add_attribute=True):
    tempvardirname = tempfile.TemporaryDirectory(delete=False)
    st.session_state.vleadfilename = tempvardirname.name + "/vlead.nc"
    
    if add_attribute:
        wr.vlead_nc(path, st.session_state.vleadfilename, date, axis_option, attributes=attributes)
    else:
        wr.vlead_nc(path, st.session_state.vleadfilename, date, axis_option)

@st.cache_data
def file_finalwrite(filename="processed_file.nc"):
    tempfinaldirname = tempfile.TemporaryDirectory(delete=False)
    outfilepath = tempfinaldirname.name + "/" + filename
    return outfilepath

@st.cache_data
def file_access(uploaded_file):
    """
    Function creates temporary directory to store the uploaded file.
    The path of the file is returned

    Args:
        uploaded_file (string): Name of the uploaded file

    Returns:
        path (string): Path of the uploaded file
    """
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return path

def display_config_as_json(config_file):
    config = configparser.ConfigParser()
    config.read_string(config_file.getvalue().decode("utf-8"))
    st.json({section: dict(config[section]) for section in config.sections()})

def main():
    st.title("ADCP Data Auto Processing Tool")
    st.write("Upload a binary input file and config.ini file for processing.")

    # File Upload Section
    uploaded_binary_file = st.file_uploader("Upload ADCP Binary File", type=["000", "bin"])
    uploaded_config_file = st.file_uploader("Upload Config File (config.ini)", type=["ini"])

    if uploaded_binary_file and uploaded_config_file:
        st.success("Files uploaded successfully!")

        # Display config.ini file content as JSON
        display_config_as_json(uploaded_config_file)

        # Process files
        process_files(uploaded_binary_file, uploaded_config_file)


def process_files(binary_file, config_file):
    # Load configuration
    config = configparser.ConfigParser()

    # Decode and parse the config file
    config_content = config_file.read().decode("utf-8")
    config.read_string(config_content)


#    config.read(config_file)

 #   config.read_string(config_file.read().decode("utf-8"))

    fpath = file_access(binary_file)


    # Read binary file
    st.write("Reading binary file. Please wait...")
    ds = readrdi.ReadFile(fpath)
    if not ds.isEnsembleEqual:
        ds.fixensemble()
    st.success("Binary file read successfully!")

    # Extract data
    header = ds.fileheader
    flobj = ds.fixedleader
    vlobj = ds.variableleader
    velocity = ds.velocity.data
    echo = ds.echo.data
    correlation = ds.correlation.data
    pgood = ds.percentgood.data
    ensembles = header.ensembles
    cells = flobj.field()["Cells"]
    fdata = flobj.fleader
    vdata = vlobj.vleader
  #  depth = ds.variableleader.depth_of_transducer

    # Initialize mask
    mask = default_mask(ds)

    # Debugging statement
    st.write(f"Mask initialized with shape: {mask.shape}")
    x = np.arange(0, ensembles, 1)
    y = np.arange(0, cells, 1)
    depth = None

    axis_option = config.get("DownloadOptions","axis_option")


    # QC Test
    isQCTest = config.getboolean("QCTest", "qc_test")

    if isQCTest:
        ct = config.getint("QCTest", "correlation")
        evt = config.getint("QCTest", "error_velocity")
        et = config.getint("QCTest", "echo_intensity")
        ft = config.getint("QCTest", "false_target")
        is3Beam = config.getboolean("QCTest", "three_beam")
        pgt = config.getint("QCTest", "percentage_good")
        orientation = config.get("QCTest", "orientation")

        mask = pg_check(ds, mask, pgt, threebeam=is3Beam)
        mask = correlation_check(ds, mask, ct)
        mask = echo_check(ds, mask, et)
        mask = ev_check(ds, mask, evt)
        mask = false_target(ds, mask, ft, threebeam=True)

    # Profile Test
    endpoints = None
    isProfileTest = config.getboolean("ProfileTest", "profile_test")
    if isProfileTest:
        isTrimEnds = config.getboolean("ProfileTest", "trim_ends")
        if isTrimEnds:
            start_index = config.getint("ProfileTest", "trim_ends_start_index")
            end_index = config.getint("ProfileTest", "trim_ends_end_index")
            if start_index > 0:
                mask[:, :start_index] = 1

            if end_index < x[-1]:
                mask[:, end_index:] = 1

            endpoints = np.array([start_index, end_index])

            st.write("Trim Ends complete.")

        isCutBins = config.getboolean("ProfileTest", "cut_bins")
        if isCutBins:
            water_column_depth = 0
            add_cells = config.getint("ProfileTest", "cut_bins_add_cells")
            if orientation == "down":
                water_column_depth = config.get("ProfileTest", "water_column_depth")
                water_column_depth = int(water_column_depth)
                mask = side_lobe_beam_angle(
                    ds,
                    mask,
                    orientation=orientation,
                    water_column_depth=water_column_depth,
                    extra_cells=add_cells
                    )
            else :
                mask = side_lobe_beam_angle(
                    ds,
                    mask,
                    orientation=orientation,
                    water_column_depth=water_column_depth,
                    extra_cells=add_cells
                    )

            st.write("Cutbins complete.")

        #Manual Cut Bins
        isManual_cutbins = config.getboolean("ProfileTest", "manual_cutbins")
        if isManual_cutbins:
            raw_bins = config.get("ProfileTest", "manual_cut_bins")
            bin_groups = raw_bins.split("]")  

            for group in bin_groups:
                if group.strip():  # Ignore empty parts
                    # Clean and split the values
                    clean_group = group.replace("[", "").strip()
                    values = list(map(int, clean_group.split(",")))
                    min_cell, max_cell, min_ensemble, max_ensemble = values
                    mask = manual_cut_bins(mask, min_cell, max_cell, min_ensemble, max_ensemble)

            st.write("Manual cut bins applied.")


        isRegrid = config.getboolean("ProfileTest", "regrid")
        if isRegrid:
            st.write("File regridding started. This will take a few seconds ...")

            regrid_option = config.get("ProfileTest", "regrid_option")
            interpolate = config.get("ProfileTest", "regrid_interpolation")
            boundary = 0
            if regrid_option == "Manual":
                boundary = config.get("ProfileTest","transducer_depth")
                z, velocity = regrid3d(ds,velocity, -32768,trimends=endpoints,orientation=orientation,method=interpolate,boundary_limit=boundary)
                z, echo = regrid3d(ds,echo, -32768,trimends=endpoints,orientation=orientation,method=interpolate,boundary_limit=boundary)
                z, correlation = regrid3d(ds,correlation, -32768,trimends=endpoints,orientation=orientation,method=interpolate,boundary_limit=boundary)
                z, pgood = regrid3d(ds,pgood, -32768,trimends=endpoints,orientation=orientation,method=interpolate,boundary_limit=boundary)
                z, mask = regrid2d(ds,mask, 1,trimends=endpoints,orientation=orientation,method=interpolate,boundary_limit=boundary)
                depth = z
            else :
                z, velocity = regrid3d(ds,velocity, -32768,trimends=endpoints,orientation=orientation,method=interpolate,boundary_limit=boundary)
                z, echo = regrid3d(ds,echo, -32768,trimends=endpoints,orientation=orientation,method=interpolate,boundary_limit=boundary)
                z, correlation = regrid3d(ds,correlation, -32768,trimends=endpoints,orientation=orientation,method=interpolate,boundary_limit=boundary)
                z, pgood = regrid3d(ds,pgood, -32768,trimends=endpoints,orientation=orientation,method=interpolate,boundary_limit=boundary)
                z, mask = regrid2d(ds,mask, 1,trimends=endpoints,orientation=orientation,method=interpolate,boundary_limit=boundary)
                depth = z

            st.write("Regrid Complete.")

        st.write("Profile Test complete.")

    isVelocityTest = config.getboolean("VelocityTest", "velocity_test")
    if isVelocityTest:
        isMagneticDeclination = config.getboolean(
            "VelocityTest", "magnetic_declination"
        )
        if isMagneticDeclination:
            maglat = config.getfloat("VelocityTest", "latitude")
            maglon = config.getfloat("VelocityTest", "longitude")
            magdep = config.getfloat("VelocityTest", "depth")
            magyear = config.getfloat("VelocityTest", "year")
            year = int(magyear)
      #      mag = config.getfloat("VelocityTest", "mag")


            mag = wmm2020api(
                 maglat, maglon, year
                )
            velocity = velocity_modifier(velocity,mag)
            st.write(f"Magnetic Declination applied. The value is {mag[0]} degrees.")
        isCutOff = config.getboolean("VelocityTest", "cutoff")
        if isCutOff:
            maxu = config.getint("VelocityTest", "max_zonal_velocity")
            maxv = config.getint("VelocityTest", "max_meridional_velocity")
            maxw = config.getint("VelocityTest", "max_vertical_velocity")
            mask = velocity_cutoff(velocity[0, :, :], mask, cutoff=maxu)
            mask = velocity_cutoff(velocity[1, :, :], mask, cutoff=maxv)
            mask = velocity_cutoff(velocity[2, :, :], mask, cutoff=maxw)
            st.write("Maximum velocity cutoff applied.")

        isDespike = config.getboolean("VelocityTest", "despike")
        if isDespike:
            despike_kernal = config.getint("VelocityTest", "despike_kernal_size")
            despike_cutoff = config.getint("VelocityTest", "despike_cutoff")

            mask = despike(
                velocity[0, :, :],
                mask,
                kernal_size=despike_kernal,
                cutoff=despike_cutoff,
            )
            mask = despike(
                velocity[1, :, :],
                mask,
                kernal_size=despike_kernal,
                cutoff=despike_cutoff,
            )
            st.write("Velocity data despiked.")

        isFlatline = config.getboolean("VelocityTest", "flatline")
        if isFlatline:
            despike_kernal = config.getint("VelocityTest", "flatline_kernal_size")
            despike_cutoff = config.getint("VelocityTest", "flatline_deviation")

            mask = flatline(
                velocity[0, :, :],
                mask,
                kernal_size=despike_kernal,
                cutoff=despike_cutoff,
            )
            mask = flatline(
                velocity[1, :, :],
                mask,
                kernal_size=despike_kernal,
                cutoff=despike_cutoff,
            )
            mask = flatline(
                velocity[2, :, :],
                mask,
                kernal_size=despike_kernal,
                cutoff=despike_cutoff,
            )
            
            st.write("Flatlines in velocity removed.")

        st.write("Velocity Test complete.")

    # Apply mask to velocity data
    isApplyMask = config.get("DownloadOptions", "apply_mask")
    if isApplyMask:
        velocity[:, mask == 1] = -32768
        st.write("Mask Applied.")

    # Create Depth axis if regrid not applied
    if depth is None:
        mean_depth = np.mean(vlobj.vleader["Depth of Transducer"]) / 10
        mean_depth = np.trunc(mean_depth)
        cells = flobj.field()["Cells"]
        cell_size = flobj.field()["Depth Cell Len"] / 100
        bin1dist = flobj.field()["Bin 1 Dist"] / 100
        max_depth = mean_depth - bin1dist
        min_depth = max_depth - cells * cell_size
        depth = np.arange(-1 * max_depth, -1 * min_depth, cell_size)

        st.write("WARNING: File not regrided. Depth axis created based on mean depth.")

    # Create Time axis
    year = vlobj.vleader["RTC Year"]
    month = vlobj.vleader["RTC Month"]
    day = vlobj.vleader["RTC Day"]
    hour = vlobj.vleader["RTC Hour"]
    minute = vlobj.vleader["RTC Minute"]
    second = vlobj.vleader["RTC Second"]

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

    date_raw = pd.to_datetime(date_df)
    date_vlead = pd.to_datetime(date_df)
    date_final = pd.to_datetime(date_df)

    st.write("Time axis created.")

    isAttributes = config.get("Optional", "attributes")
    if isAttributes:
        attributes = [att for att in config["Optional"]]
        attributes = dict(config["Optional"].items())
        del attributes["attributes"]
    else:
        attributes = None

    isWriteRawNC = config.get("DownloadOptions", "download_raw")
    isWriteVleadNC = config.get("DownloadOptions", "download_vlead")
    isWriteProcNC = config.get("DownloadOptions", "download_processed")
    
    if isWriteRawNC:
        filepath = config.get("FileSettings", "output_file_path")
        filename = config.get("FileSettings", "output_file_name_raw")
        output_file_path = os.path.join(filepath, filename)
        if isAttributes:
            wr.rawnc(fpath, output_file_path, date_raw, axis_option, attributes, isAttributes)

        st.write("Raw file written.")

    if isWriteVleadNC:
        filepath = config.get("FileSettings", "output_file_path")
        filename = config.get("FileSettings", "output_file_name_vlead")
        output_file_path = os.path.join(filepath, filename)
        if isAttributes:
            wr.vlead_nc(fpath, output_file_path, date_vlead, axis_option, attributes, isAttributes)

        st.write("Variable Leader file written.")


    st.session_state.processed_filename = file_finalwrite()
    st.session_state.final_depth = depth
    depth1 = np.trunc(st.session_state.final_depth)

    if isWriteProcNC:
        filepath = config.get("FileSettings", "output_file_path")
        filename = config.get("FileSettings", "output_file_name_processed")
        output_file_path = os.path.join(filepath, filename)
        if isAttributes:
            wr.finalnc(output_file_path, depth1,mask,date_final,velocity,attributes=attributes)
        else:
            wr.finalnc(output_file_path, depth1,mask,date_final,velocity)

        st.write("Processed file written.")
        
if __name__ == "__main__":
    main()
