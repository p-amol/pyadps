import configparser
import os

import numpy as np
import pandas as pd
import pyadps.utils.writenc as wr
from netCDF4 import date2num
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
    magdec,
    wmm2020api,
    velocity_modifier,
)
from pyadps.utils.sensor_health import sound_speed_correction, tilt_sensor_check


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
    roll = ds.variableleader.roll.data
    pitch = ds.variableleader.pitch.data
    sound = ds.variableleader.speed_of_sound.data
    depth = ds.variableleader.depth_of_transducer.data
    temperature = (
        ds.variableleader.temperature.data * ds.variableleader.temperature.scale
    )
    salinity = ds.variableleader.salinity.data * ds.variableleader.salinity.scale
    orientation = ds.fixedleader.system_configuration()["Beam Direction"]
    ensembles = header.ensembles
    cells = flobj.field()["Cells"]
    beams = flobj.field()["Beams"]
    cell_size = flobj.field()["Depth Cell Len"]
    bin1dist = flobj.field()["Bin 1 Dist"]
    beam_angle = int(flobj.system_configuration()["Beam Angle"])
    fdata = flobj.fleader
    vdata = vlobj.vleader
    #  depth = ds.variableleader.depth_of_transducer

    # Initialize mask
    mask = default_mask(ds)

    # Debugging statement
    x = np.arange(0, ensembles, 1)
    y = np.arange(0, cells, 1)

    axis_option = config.get("DownloadOptions", "axis_option")

    # Sensor Test
    isSensorTest = config.getboolean("SensorTest", "sensor_test")
    if isSensorTest:
        isDepthModified = config.getboolean("SensorTest", "is_depth_modified")
        if isDepthModified:
            depth_option = config.get("SensorTest", "depth_input_option")
            if depth_option == "Fixed Value":
                fixed_depth = config.getfloat("SensorTest", "fixed_depth")
                depth = np.full(ensembles, fixed_depth)
                depth *= 10
            elif depth_option == "File Upload":
                depth_file_path = config.get("SensorTest", "depth_file_path")
                df = pd.read_csv(depth_file_path)
                depth = np.squeeze(df)
                if len(depth) != ensembles:
                    print("""
                          Error: Uploaded file ensembles and 
                          actual ensembles mismatch
                          """)
                else:
                    print("Depth file uploaded.")

        isSalinityModified = config.getboolean("SensorTest", "is_salinity_modified")
        if isSalinityModified:
            salinity_option = config.get("SensorTest", "salinity_input_option")
            if salinity_option == "Fixed Value":
                fixed_salinity = config.getfloat("SensorTest", "fixed_salinity")
                salinity = np.full(ensembles, fixed_salinity)
                salinity *= 10
            elif salinity_option == "File Upload":
                salinity_file_path = config.get("SensorTest", "salinity_file_path")
                df = pd.read_csv(salinity_file_path)
                salinity = np.squeeze(df)
                if len(salinity) != ensembles:
                    print("""
                          Error: Uploaded file ensembles and 
                          actual ensembles mismatch
                          """)
                else:
                    print("Salinity file uploaded.")

        isTemperatureModified = config.getboolean(
            "SensorTest", "is_temperature_modified"
        )
        if isTemperatureModified:
            temperature_option = config.get("SensorTest", "temperature_input_option")
            if temperature_option == "Fixed Value":
                fixed_temperature = config.getfloat("SensorTest", "fixed_temperature")
                temperature = np.full(ensembles, fixed_temperature)
                temperature *= 10
            elif temperature_option == "File Upload":
                temperature_file_path = config.get(
                    "SensorTest", "temperature_file_path"
                )
                df = pd.read_csv(temperature_file_path)
                temperature = np.squeeze(df)
                if len(temperature) != ensembles:
                    print("""
                          Error: Uploaded file ensembles and 
                          actual ensembles mismatch
                          """)
                else:
                    print("Temperature file uploaded.")

        isRollCheck = config.getboolean("SensorTest", "roll_check")
        if isRollCheck:
            roll_cutoff = config.getint("SensorTest", "roll_cutoff")
            mask = tilt_sensor_check(roll, mask, cutoff=roll_cutoff)

        isPitchCheck = config.getboolean("SensorTest", "pitch_check")
        if isPitchCheck:
            pitch_cutoff = config.getint("SensorTest", "pitch_cutoff")
            mask = tilt_sensor_check(pitch, mask, cutoff=pitch_cutoff)

        isVelocityModified = config.getboolean("SensorTest", "velocity_modified")
        if isVelocityModified:
            velocity = sound_speed_correction(
                velocity, sound, temperature, salinity, depth
            )

    # QC Test
    isQCTest = config.getboolean("QCTest", "qc_test")
    if isQCTest:
        isQCCheck = config.get("QCTest", "qc_check")
        if isQCCheck:
            ct = config.getint("QCTest", "correlation")
            evt = config.getint("QCTest", "error_velocity")
            et = config.getint("QCTest", "echo_intensity")
            ft = config.getint("QCTest", "false_target")
            is3Beam = config.getboolean("QCTest", "three_beam")
            pgt = config.getint("QCTest", "percent_good")
            orientation = config.get("QCTest", "orientation")

            mask = pg_check(ds, mask, pgt, threebeam=is3Beam)
            mask = correlation_check(ds, mask, ct)
            mask = echo_check(ds, mask, et)
            mask = ev_check(ds, mask, evt)
            mask = false_target(ds, mask, ft, threebeam=True)

            print("QC Check Complete.")

        isBeamModified = config.getboolean("QCTest", "beam_modified")
        if isBeamModified:
            orientation = config.get("QCTest", "orientation")
            print("Beam orientation changed.")

    # Profile Test
    endpoints = None
    isProfileTest = config.getboolean("ProfileTest", "profile_test")
    if isProfileTest:
        isTrimEnds = config.getboolean("ProfileTest", "trim_ends_check")
        if isTrimEnds:
            start_index = config.getint("ProfileTest", "trim_start_ensemble")
            end_index = config.getint("ProfileTest", "trim_end_ensemble")
            if start_index > 0:
                mask[:, :start_index] = 1

            if end_index < x[-1]:
                mask[:, end_index:] = 1

            endpoints = np.array([start_index, end_index])

            print("Trim Ends complete.")

        isCutBins = config.getboolean("ProfileTest", "cutbins_sidelobe_check")
        if isCutBins:
            water_column_depth = config.getint("ProfileTest", "water_depth")
            extra_cells = config.getint("ProfileTest", "extra_cells")
            mask = side_lobe_beam_angle(
                depth,
                mask,
                orientation=orientation,
                water_column_depth=water_column_depth,
                extra_cells=extra_cells,
                cells=cells,
                cell_size=cell_size,
                bin1dist=bin1dist,
                beam_angle=beam_angle,
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

            # regrid_option = config.get("ProfileTest", "regrid_option")
            end_cell_option = config.get("ProfileTest", "end_cell_option")
            interpolate = config.get("ProfileTest", "interpolate")
            boundary = config.getint("ProfileTest", "boundary")
            z, velocity = regrid3d(
                depth,
                velocity,
                -32768,
                trimends=endpoints,
                orientation=orientation,
                end_cell_option=end_cell_option,
                method=interpolate,
                boundary_limit=boundary,
                cells=cells,
                cell_size=cell_size,
                bin1dist=bin1dist,
                beams=beams,
            )
            z, echo = regrid3d(
                depth,
                echo,
                -32768,
                trimends=endpoints,
                orientation=orientation,
                method=interpolate,
                boundary_limit=boundary,
                cells=cells,
                cell_size=cell_size,
                bin1dist=bin1dist,
                beams=beams,
            )
            z, correlation = regrid3d(
                depth,
                correlation,
                -32768,
                trimends=endpoints,
                orientation=orientation,
                method=interpolate,
                boundary_limit=boundary,
                cells=cells,
                cell_size=cell_size,
                bin1dist=bin1dist,
                beams=beams,
            )
            z, pgood = regrid3d(
                depth,
                pgood,
                -32768,
                trimends=endpoints,
                orientation=orientation,
                method=interpolate,
                boundary_limit=boundary,
                cells=cells,
                cell_size=cell_size,
                bin1dist=bin1dist,
                beams=beams,
            )
            z, mask = regrid2d(
                depth,
                mask,
                1,
                trimends=endpoints,
                orientation=orientation,
                method=interpolate,
                boundary_limit=boundary,
                cells=cells,
                cell_size=cell_size,
                bin1dist=bin1dist,
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
            magmethod = config.get("VelocityTest", "magnet_method")
            maglat = config.getfloat("VelocityTest", "magnet_latitude")
            maglon = config.getfloat("VelocityTest", "magnet_longitude")
            magdep = config.getfloat("VelocityTest", "magnet_depth")
            magyear = config.getfloat("VelocityTest", "magnet_year")
            year = int(magyear)
            #      mag = config.getfloat("VelocityTest", "mag")

            if magmethod == "pygeomag":
                mag = magdec(maglat, maglon, magdep, magyear)
            elif magmethod.lower() == "api":
                mag = wmm2020api(maglat, maglon, year)
            elif magmethod.lower() == "manual":
                mag = config.getint("VelocityTest", "magnet_user_input")
            else:
                mag = 0
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

    isAttributes = config.getboolean("DownloadOptions", "add_attributes_processed")
    if isAttributes:
        attributes = [att for att in config["DownloadOptions"]]
        attributes = dict(config["DownloadOptions"].items())
        del attributes["add_attributes_processed"]
    else:
        attributes = None

    isWriteRawNC = config.getboolean("DownloadOptions", "download_raw_netcdf")
    isWriteVleadNC = config.getboolean("DownloadOptions", "download_vlead_netcdf")
    isWriteProcNC = config.getboolean("DownloadOptions", "download_processed_netcdf")
    filepath = config.get("FileSettings", "output_file_path")

    print(isWriteRawNC)
    if isWriteRawNC:
        filename = config.get("FileSettings", "output_file_name_raw_netcdf")
        output_file_path = os.path.join(filepath, filename)
        print(date_raw.shape)
        wr.rawnc(
            full_input_file_path,
            output_file_path,
            date_raw,
            axis_option=axis_option,
            attributes=attributes,
        )

        print("Raw file written.")

    if isWriteVleadNC:
        filename = config.get("FileSettings", "output_file_name_vlead_netcdf")
        output_file_path = os.path.join(filepath, filename)
        wr.vlead_nc(
            full_input_file_path,
            output_file_path,
            date_vlead,
            axis_option=axis_option,
            attributes=attributes,
        )

        print("Vlead file written.")

    depth1 = depth

    if isWriteProcNC:
        filename = config.get("FileSettings", "output_file_name_processed_netcdf")
        output_file_path = os.path.join(filepath, filename)

        wr.finalnc(
            output_file_path,
            depth1,
            mask,
            date_final,
            velocity,
            attributes=attributes,  # Pass edited attributes
        )
        print("Processed file written.")


if __name__ == "__main__":
    main()
