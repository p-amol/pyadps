import os
import matplotlib.pyplot as plt
import numpy as np
import pyadps.utils.readrdi as rd
from pyadps.utils.plotgen import CutBins
from pyadps.utils.profile_test import side_lobe_beam_angle, trim_ends
from pyadps.utils.profile_test import regrid2d, regrid3d
from pyadps.utils.signal_quality import (
    default_mask,
    ev_check,
    false_target,
    pg_check,
    correlation_check,
    echo_check,
    qc_prompt,
)
from pyadps.utils.velocity_test import (
    despike,
    flatline,
    velocity_modifier,
    wmm2020api,
    velocity_cutoff,
)

import pyadps.utils.writenc as wr

plt.style.use("seaborn-v0_8-darkgrid")


def main():
    # Get the config file
    try:
        filepath = input("Enter file name: ")
        if os.path.exists(filepath):
            run_script(filepath)
        else:
            print("File not found!")
    except Exception as e:
        import traceback

        print("Error: Unable to process the data.")
        traceback.print_exc()


# filename = "./metadata/demo.000"


def run_script(filename):
    ds = rd.ReadFile(filename)
    fl = ds.fixedleader
    vl = ds.variableleader
    vel = ds.velocity.data
    echo = ds.echo.data
    cor = ds.correlation.data
    pgood = ds.percentgood.data
    time = ds.time
    beamdir = ds.fixedleader.system_configuration()["Beam Direction"]

    # Data pressure = vl.vleader["Pressure"]
    # beam_angle = int(fl.system_configuration()["Beam Angle"])
    # blank_size = fl.field()["Blank Transmit"]
    cell_size = fl.field()["Depth Cell Len"] / 100
    cells = fl.field()["Cells"]
    bin1dist = fl.field()["Bin 1 Dist"] / 100

    mean_depth = np.mean(ds.variableleader.depth_of_transducer.data) / 10
    mean_depth = np.trunc(mean_depth)
    if beamdir.lower() == "up":
        sgn = -1
    else:
        sgn = 1

    first_depth = mean_depth + sgn * bin1dist
    last_depth = first_depth + sgn * cells * cell_size
    z = np.arange(first_depth, last_depth, sgn * cell_size)

    # Original mask created from velocity
    mask = default_mask(ds)
    orig_mask = np.copy(mask)

    # Default threshold
    ct = fl.field()["Correlation Thresh"]
    et = 0
    pgt = fl.field()["Percent Good Min"]
    evt = fl.field()["Error Velocity Thresh"]
    ft = fl.field()["False Target Thresh"]

    # Get the threshold values
    ct = qc_prompt(ds, "Correlation Thresh")
    evt = qc_prompt(ds, "Error Velocity Thresh")
    pgt = qc_prompt(ds, "Percent Good Min")
    et = qc_prompt(ds, "Echo Intensity Thresh", echo)
    ft = qc_prompt(ds, "False Target Thresh")

    # Apply threshold
    # values, counts = np.unique(mask, return_counts=True)
    # print(values, counts, np.round(counts * 100 / np.sum(counts)))
    print("Applying Thresholds")
    mask = pg_check(ds, mask, pgt)
    mask = correlation_check(ds, mask, ct)
    mask = echo_check(ds, mask, et)
    mask = ev_check(ds, mask, evt)
    mask = false_target(ds, mask, ft, threebeam=True)

    ########## PROFILE TEST #########

    affirm = input("Would you like to trim the ends? [y/n]: ")
    if affirm.lower() == "y":
        mask = trim_ends(ds, mask)

    affirm = input("Would you remove the surface backscatter bins? [y/n]: ")
    if affirm.lower() == "y":
        mask = side_lobe_beam_angle(ds, mask)

    affirm = input("Would you like to manually select and mask data? [y/n]: ")
    if affirm.lower() == "y":
        manual = CutBins(echo[0, :, :], mask)
        plt.show()
        mask = manual.mask()

    affirm = input("Regrid the data based on pressure sensor? [y/n]:")
    if affirm.lower() == "y":
        z, vel = regrid3d(ds, vel, -32768)
        z, echo_reg = regrid3d(ds, echo, -32768)
        z, correlation_reg = regrid3d(ds, cor, -32768)
        z, percentgood_reg = regrid3d(ds, pgood, -32768)
        z, mask = regrid2d(ds, mask, -32768)

    ########## VELOCITY TEST ##########
    affirm = input("Apply correction for magnetic declination? [y/n]:")
    if affirm.lower() == "y":
        lat = input("Enter Latitude: ")
        lat = float(lat)

        lon = input("Enter Longitude: ")
        lon = float(lon)

        depth = input("Enter Depth (m): ")
        depth = float(depth)

        year = input("Year: ")
        year = int(year)

        mag = wmm2020api(lat, lon, year)
        vel = velocity_modifier(vel, mag)

    affirm = input("Apply velocity thresholds [y/n]: ")
    if affirm.lower() == "y":
        maxuvel = input("Enter maximum zonal velocity: ")
        maxuvel = float(maxuvel)

        maxvvel = input("Enter maximum meridional velocity: ")
        maxvvel = float(maxvvel)

        maxwvel = input("Enter maximum vertical velocity: ")
        maxwvel = float(maxwvel)
        mask = velocity_cutoff(vel[0, :, :], mask, cutoff=maxuvel)
        mask = velocity_cutoff(vel[1, :, :], mask, cutoff=maxvvel)
        mask = velocity_cutoff(vel[2, :, :], mask, cutoff=maxwvel)

    affirm = input("Despike the data? [y/n]: ")
    if affirm.lower() == "y":
        despike_kernel = input("Enter despike kernel size:")
        despike_kernel = int(despike_kernel)

        despike_cutoff = input("Enter despike cutoff (mm/s): ")
        despike_cutoff = float(despike_cutoff)

        mask = despike(
            vel[0, :, :], mask, kernel_size=despike_kernel, cutoff=despike_cutoff
        )
        mask = despike(
            vel[1, :, :], mask, kernel_size=despike_kernel, cutoff=despike_cutoff
        )

    affirm = input("Remove flatlines? [y/n]: ")
    if affirm.lower() == "y":
        flatline_kernel = input("Enter despike kernel size:")
        flatline_kernel = int(flatline_kernel)
        flatline_cutoff = input("Enter Flatline deviation: [y/n]")
        flatlineL_cutoff = int(flatline_cutoff)
        mask = flatline(
            vel[0, :, :], mask, kernel_size=flatline_kernel, cutoff=flatline_cutoff
        )
        mask = flatline(
            vel[1, :, :], mask, kernel_size=flatline_kernel, cutoff=flatline_cutoff
        )
        mask = flatline(
            vel[2, :, :], mask, kernel_size=flatline_kernel, cutoff=flatline_cutoff
        )
    apply_mask = input("Apply mask? [y/n]: ")
    if apply_mask.lower() == "y":
        for i in range(4):
            vel[i, :, :] = np.where(mask == 0, vel[i, :, :], -32768)

    outfilepath = input("Enter output file name (*nc): ")
    if os.path.exists(outfilepath):
        if os.path.isfile(outfilepath):
            print(f"The file already exists: {outfilepath}")
    else:
        wr.finalnc(outfilepath, z, mask, time, vel)


if __name__ == "__main__":
    main()
