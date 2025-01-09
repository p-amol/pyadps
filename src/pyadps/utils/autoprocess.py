import configparser
import os

import numpy as np
import pandas as pd
import pyadps.utils.writenc as wr
from pyadps.utils import readrdi
from pyadps.utils.profile_test import side_lobe_beam_angle, manual_cut_bins
from pyadps.utils.profile_test import regrid2d, regrid3d
from pyadps.utils.signal_quality import (
    default_mask,
    ev_check,
    false_target,
    pg_check,
    echo_check,
    correlation_check,
)
from pyadps.utils.velocity_test import (
    despike,
    flatline,
    velocity_cutoff,
    wmm2020api,
    velocity_modifier,
)


def main():
    # Get the config file
    try:
        filepath = input("Enter config file name: ")
        if os.path.exists(filepath):
            autoprocess(filepath)
        else:
            print("File not found!")
    except Exception as e:
        import traceback

        print("Error: Unable to process the data.")
        traceback.print_exc()


def autoprocess(config_file, binary_file_path=None):
    # Load configuration
    config = configparser.ConfigParser()

    # Decode and parse the config file
    # Check if config_file is a file-like object or a file path
    if hasattr(config_file, "read"):
        # If it's a file-like object, read its content
        config_content = config_file.read().decode("utf-8")
    else:
        # If it's a file path, open the file and read its content
        with open(config_file, "r", encoding="utf-8") as file:
            config_content = file.read()
    config.read_string(config_content)

    if not binary_file_path:
        input_file_name = config.get("FileSettings", "input_file_name")
        input_file_path = config.get("FileSettings", "input_file_path")
        full_input_file_path = os.path.join(input_file_path, input_file_name)
    else:
        full_input_file_path = binary_file_path

    print("File reading started. Please wait for a few seconds ...")
    ds = readrdi.ReadFile(full_input_file_path)
    print("File reading complete.")

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
    x = np.arange(0, ensembles, 1)
    y = np.arange(0, cells, 1)
    depth = None

    axis_option = config.get("DownloadOptions", "axis_option")

    # Sensor Test
    isSensorTest = config.getboolean("SensorTest", "sensor_test")
    isRollTest = config.getboolean("RollTest", "roll_test")
    # if isSensorTest:
    #     if isRollTest:

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

            print("Trim Ends complete.")

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
                    extra_cells=add_cells,
                )
            else:
                mask = side_lobe_beam_angle(
                    ds,
                    mask,
                    orientation=orientation,
                    water_column_depth=water_column_depth,
                    extra_cells=add_cells,
                )

            print("Cutbins complete.")

        # Manual Cut Bins
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
                    mask = manual_cut_bins(
                        mask, min_cell, max_cell, min_ensemble, max_ensemble
                    )

            print("Manual cut bins applied.")

        isRegrid = config.getboolean("ProfileTest", "regrid")
        if isRegrid:
            print("File regridding started. This will take a few seconds ...")

            regrid_option = config.get("ProfileTest", "regrid_option")
            interpolate = config.get("ProfileTest", "regrid_interpolation")
            boundary = 0
            if regrid_option == "Manual":
                boundary = config.get("ProfileTest", "transducer_depth")
                z, velocity = regrid3d(
                    ds,
                    velocity,
                    -32768,
                    trimends=endpoints,
                    orientation=orientation,
                    method=interpolate,
                    boundary_limit=boundary,
                )
                z, echo = regrid3d(
                    ds,
                    echo,
                    -32768,
                    trimends=endpoints,
                    orientation=orientation,
                    method=interpolate,
                    boundary_limit=boundary,
                )
                z, correlation = regrid3d(
                    ds,
                    correlation,
                    -32768,
                    trimends=endpoints,
                    orientation=orientation,
                    method=interpolate,
                    boundary_limit=boundary,
                )
                z, pgood = regrid3d(
                    ds,
                    pgood,
                    -32768,
                    trimends=endpoints,
                    orientation=orientation,
                    method=interpolate,
                    boundary_limit=boundary,
                )
                z, mask = regrid2d(
                    ds,
                    mask,
                    1,
                    trimends=endpoints,
                    orientation=orientation,
                    method=interpolate,
                    boundary_limit=boundary,
                )
                depth = z
            else:
                z, velocity = regrid3d(
                    ds,
                    velocity,
                    -32768,
                    trimends=endpoints,
                    orientation=orientation,
                    method=interpolate,
                    boundary_limit=boundary,
                )
                z, echo = regrid3d(
                    ds,
                    echo,
                    -32768,
                    trimends=endpoints,
                    orientation=orientation,
                    method=interpolate,
                    boundary_limit=boundary,
                )
                z, correlation = regrid3d(
                    ds,
                    correlation,
                    -32768,
                    trimends=endpoints,
                    orientation=orientation,
                    method=interpolate,
                    boundary_limit=boundary,
                )
                z, pgood = regrid3d(
                    ds,
                    pgood,
                    -32768,
                    trimends=endpoints,
                    orientation=orientation,
                    method=interpolate,
                    boundary_limit=boundary,
                )
                z, mask = regrid2d(
                    ds,
                    mask,
                    1,
                    trimends=endpoints,
                    orientation=orientation,
                    method=interpolate,
                    boundary_limit=boundary,
                )
                depth = z

            print("Regrid Complete.")

        print("Profile Test complete.")

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

            mag = wmm2020api(maglat, maglon, year)
            velocity = velocity_modifier(velocity, mag)
            print(f"Magnetic Declination applied. The value is {mag[0]} degrees.")
        isCutOff = config.getboolean("VelocityTest", "cutoff")
        if isCutOff:
            maxu = config.getint("VelocityTest", "max_zonal_velocity")
            maxv = config.getint("VelocityTest", "max_meridional_velocity")
            maxw = config.getint("VelocityTest", "max_vertical_velocity")
            mask = velocity_cutoff(velocity[0, :, :], mask, cutoff=maxu)
            mask = velocity_cutoff(velocity[1, :, :], mask, cutoff=maxv)
            mask = velocity_cutoff(velocity[2, :, :], mask, cutoff=maxw)
            print("Maximum velocity cutoff applied.")

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
            print("Velocity data despiked.")

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

            print("Flatlines in velocity removed.")

        print("Velocity Test complete.")

    # Apply mask to velocity data
    isApplyMask = config.get("DownloadOptions", "apply_mask")
    if isApplyMask:
        velocity[:, mask == 1] = -32768
        print("Mask Applied.")

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

        print("WARNING: File not regrided. Depth axis created based on mean depth.")

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

    print("Time axis created.")

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
            wr.rawnc(
                full_input_file_path,
                output_file_path,
                date_raw,
                axis_option,
                attributes,
                isAttributes,
            )

        print("Raw file written.")

    if isWriteVleadNC:
        filepath = config.get("FileSettings", "output_file_path")
        filename = config.get("FileSettings", "output_file_name_vlead")
        output_file_path = os.path.join(filepath, filename)
        if isAttributes:
            wr.vlead_nc(
                full_input_file_path,
                output_file_path,
                date_vlead,
                axis_option,
                attributes,
                isAttributes,
            )

            print("Vlead file written.")

    depth1 = depth

    if isWriteProcNC:
        filepath = config.get("FileSettings", "output_file_path")
        filename = config.get("FileSettings", "output_file_name_processed")
        full_file_path = os.path.join(filepath, filename)

        wr.finalnc(
            full_file_path,
            depth1,
            mask,
            date_final,
            velocity,
            attributes=attributes,  # Pass edited attributes
        )
        print("Processed file written.")


if __name__ == "__main__":
    main()
