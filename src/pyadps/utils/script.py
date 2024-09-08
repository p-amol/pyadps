import sys

import matplotlib.pyplot as plt
import numpy as np
import pyadps.utils.readrdi as rd
from pyadps.utils.cutbin import CutBins
from pyadps.utils.plotgen import plotmask, plotvar
from pyadps.utils.profile_test import side_lobe_beam_angle, trim_ends
from pyadps.utils.regrid import regrid2d, regrid3d
from pyadps.utils.signal_quality import (default_mask, ev_check, false_target,
                                         pg_check, qc_check, qc_prompt)
from pyadps.utils.velocity_test import (despike, flatline,
                                        magnetic_declination, velocity_cutoff)

plt.style.use("seaborn-v0_8-darkgrid")


# Read data
filename = "/home/amol/Desktop/BGS11000.000"
ds = rd.ReadFile(filename)
fl = ds.fixedleader
vl = ds.variableleader
vel = ds.velocity.data
echo = ds.echo.data
cor = ds.correlation.data
pgood = ds.percentgood.data

# Data pressure = vl.vleader["Pressure"]
beam_angle = int(fl.system_configuration()["Beam Angle"])
cell_size = fl.field()["Depth Cell Len"]
blank_size = fl.field()["Blank Transmit"]
cells = fl.field()["Cells"]

# sys.exit()

# Original mask created from velocity
mask = default_mask(fl, vel)
orig_mask = np.copy(mask)

# Default threshold
ct = fl.field()["Correlation Thresh"]
et = 0
pgt = fl.field()["Percent Good Min"]
evt = fl.field()["Error Velocity Thresh"]
ft = fl.field()["False Target Thresh"]

print(ct, et, pgt, evt, ft)

# Get the threshold values
ct = qc_prompt(fl, "Correlation Thresh")
evt = qc_prompt(fl, "Error Velocity Thresh")
pgt = qc_prompt(fl, "Percent Good Min")
et = qc_prompt(fl, "Echo Intensity Thresh", echo)
ft = qc_prompt(fl, "False Target Thresh")

# Apply threshold
values, counts = np.unique(mask, return_counts=True)
print(values, counts, np.round(counts * 100 / np.sum(counts)))
mask = pg_check(pgood, mask, pgt)
mask = qc_check(cor, mask, ct)
mask = qc_check(echo, mask, et)
mask = ev_check(vel[3, :, :], mask, evt)
mask = false_target(echo, mask, ft, threebeam=True)


########## PROFILE TEST #########

affirm = input("Would you like to trim the ends? [y/n]: ")
if affirm.lower() == "y":
    mask = trim_ends(vl, mask)

affirm = input("Would you remove the surface backscatter bins? [y/n]: ")
if affirm.lower() == "y":
    mask = side_lobe_beam_angle(fl, vl, mask)

affirm = input("Would you like to manually select and mask data?")
if affirm.lower() == "y":
    manual = CutBins(echo[0, :, :], mask)
    plt.show()
    mask = manual.mask()

affirm = input("Regrid the data based on pressure sensor? [y/n]:")
if affirm.lower() == "y":
    z, vel = regrid3d(fl, vl, vel, -32768)
    z, echo_reg = regrid3d(fl, vl, echo, -32768)
    z, correlation_reg = regrid3d(fl, vl, cor, -32768)
    z, percentgood_reg = regrid3d(fl, vl, pgood, -32768)
    z, mask = regrid2d(fl, vl, mask, -32768)

# affirm = input("Display original and revised mask files? [y/n]:")
# if affirm.lower() == "y":
#     plotmask(orig_mask, mask)


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

    velocity = magnetic_declination(vel, lat, lon, depth, year)

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
    despike_kernal = input("Enter despike kernal size:")
    despike_kernal = int(despike_kernal)

    despike_cutoff = input("Enter despike cutoff (mm/s): ")
    despike_cutoff = float(despike_cutoff)

    mask = despike(
        vel[0, :, :], mask, kernal_size=despike_kernal, cutoff=despike_cutoff
    )
    mask = despike(
        vel[1, :, :], mask, kernal_size=despike_kernal, cutoff=despike_cutoff
    )

affirm = input("Remove flatlines? [y/n]: ")
if affirm.lower() == "y":
    flatline_kernal = input("Enter despike kernal size:")
    flatline_kernal = int(flatline_kernal)
    flatline_cutoff = input("Enter Flatline deviation: [y/n]")
    flatlineL_cutoff = int(flatline_cutoff)
    mask = flatline(
        velocity[0, :, :], mask, kernal_size=flatline_kernal, cutoff=flatline_cutoff
    )
    mask = flatline(
        velocity[1, :, :], mask, kernal_size=flatline_kernal, cutoff=flatline_cutoff
    )
    mask = flatline(
        velocity[2, :, :], mask, kernal_size=flatline_kernal, cutoff=flatline_cutoff
    )
