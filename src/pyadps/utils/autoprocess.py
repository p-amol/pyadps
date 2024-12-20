import configparser
import os

import numpy as np
import pandas as pd
import pyadps.utils.writenc as wr
from pyadps.utils import readrdi
from pyadps.utils.profile_test import side_lobe_beam_angle
from pyadps.utils.profile_test import regrid2d, regrid3d
from pyadps.utils.signal_quality import (
    default_mask,
    ev_check,
    false_target,
    pg_check,
    qc_check,
)
from pyadps.utils.velocity_test import (
    despike,
    flatline,
    wmm2020api,
    velocity_modifier,
    velocity_cutoff,
)

def main():
    # Get the config file
    try:
        filepath = input("Enter config file name: ")
        if os.path.exists(filepath):
            autoprocess(filepath)
        else:
            print("File not found!")
    except:
        print("Error: Unable to process the data.")

def autoprocess(filepath):
    config = configparser.ConfigParser()
    config.read(filepath)
    input_file_name = config.get("FileSettings", "input_file_name")
    input_file_path = config.get("FileSettings", "input_file_path")

    full_input_file_path = os.path.join(input_file_path, input_file_name)

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

    mask = default_mask(flobj, velocity)
    print("Default Mask created.")
    x = np.arange(0, ensembles, 1)
    y = np.arange(0, cells, 1)
    depth = None

    # QC Test
    isQCTest = config.getboolean("QCTest", "qc_test")

    if isQCTest:
        ct = config.getint("QCTest", "correlation")
        evt = config.getint("QCTest", "error_velocity")
        et = config.getint("QCTest", "echo_intensity")
        ft = config.getint("QCTest", "false_target")
        is3Beam = config.getboolean("QCTest", "three_beam")
        pgt = config.getint("QCTest", "percentage_good")

        mask = pg_check(pgood, mask, pgt, threebeam=is3Beam)
        mask = qc_check(correlation, mask, ct)
        mask = qc_check(echo, mask, et)
        mask = ev_check(velocity[3, :, :], mask, evt)
        mask = false_target(echo, mask, ft, threebeam=True)
        print("QC Test complete.")

    endpoints = None
    isProfileTest = config.getboolean("ProfileTest", "profile_test")
    if isProfileTest:
        isTrimEnds = config.getboolean("ProfileTest", "trim_ends")
        if isTrimEnds:
            start_index = config.getint("ProfileTest", "trim_ends_start_index")
            end_index = config.getint("ProfileTest", "trim_ends_end_index")
            # if start_index < 0 or start_index > ensembles:

            if start_index > 0:
                mask[:, :start_index] = 1

            if end_index < x[-1]:
                mask[:, end_index:] = 1

            endpoints = np.array([start_index, end_index])

            print("Trim Ends complete.")

        isCutBins = config.getboolean("ProfileTest", "cut_bins")
        if isCutBins:
            add_cells = config.getint("ProfileTest", "cut_bins_add_cells")
            mask = side_lobe_beam_angle(flobj, vlobj, mask, extra_cells=add_cells)

            print("Cutbins complete.")

        isRegrid = config.getboolean("ProfileTest", "regrid")
        if isRegrid:
            print("File regridding started. This will take a few seconds ...")
            regrid_option = config.get("ProfileTest", "regrid_option")
            z, velocity = regrid3d(
                flobj,
                vlobj,
                velocity,
                -32768,
                trimends=endpoints,
            )
            z, echo = regrid3d(flobj, vlobj, echo, -32768, trimends=endpoints)
            z, correlation = regrid3d(
                flobj, vlobj, correlation, -32768, trimends=endpoints
            )
            z, pgood = regrid3d(flobj, vlobj, pgood, -32768, trimends=endpoints)
            z, mask = regrid2d(flobj, vlobj, mask, 1, trimends=endpoints)
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

            
            mag = wmm2020api(maglat, maglon, magyear)
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

    date = pd.to_datetime(date_df)

    print("Time axis created.")

    isWriteRawNC = config.get("DownloadOptions", "download_raw")
    isWriteProcNC = config.get("DownloadOptions", "download_processed")
    isAttributes = config.get("Optional", "attributes")

    if isAttributes:
        attributes = [att for att in config["Optional"]]
        attributes = dict(config["Optional"].items())
        del attributes["attributes"]
    else:
        attributes = None

    if isWriteRawNC:
        filepath = config.get("FileSettings", "output_file_path")
        filename = config.get("FileSettings", "output_file_name_raw")
        output_file_path = os.path.join(filepath, filename)
        if isAttributes:
            wr.rawnc(full_input_file_path, output_file_path, attributes=attributes)

        print("Raw file written.")

    if isWriteProcNC:
        filepath = config.get("FileSettings", "output_file_path")
        filename = config.get("FileSettings", "output_file_name_processed")
        full_file_path = os.path.join(filepath, filename)

        wr.finalnc(
            full_file_path,
            depth,
            date,
            velocity,
            attributes=attributes,  # Pass edited attributes
        )
        print("Processed file written.")


if __name__ == "__main__":
    main()
