import sys

import matplotlib.pyplot as plt
import numpy as np

import pyadps.utils.readrdi as rd
from pyadps.utils.cutbin import CutBins
from pyadps.utils.plotgen import plotmask, plotvar
from pyadps.utils.profile_test import side_lobe_beam_angle, trim_ends
from pyadps.utils.signal_quality import (ev_check, false_target, pg_check, qc_check,
                            qc_prompt, vel_mask)

plt.style.use("seaborn-v0_8-darkgrid")


# Read data
filename = "BGS11000.000"
fl = rd.FixedLeader(filename, run="fortran")
vl = rd.VariableLeader(filename, run="fortran")
vel = rd.velocity(filename, run="fortran")
echo = rd.echo(filename, run="fortran")
cor = rd.correlation(filename, run="fortran")
pgood = rd.percentgood(filename, run="fortran")

# Data pressure = vl.vleader["Pressure"]
beam_angle = int(fl.system_configuration()["Beam Angle"])
cell_size = fl.field()["Depth Cell Len"]
blank_size = fl.field()["Blank Transmit"]
cells = fl.field()["Cells"]

# sys.exit()

# Original mask created from velocity
mask = vel_mask(vel)
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
print(values, counts, np.round(counts[1] * 100 / np.sum(counts)))
mask = pg_check(pgood, mask, pgt)
mask = qc_check(cor, mask, ct)
mask = qc_check(echo, mask, et)
mask = ev_check(vel[3, :, :], mask, evt)
mask = false_target(echo, mask, ft, threebeam=True)


########## PROFILE TEST #########

mask = trim_ends(vl, mask)
mask = side_lobe_beam_angle(fl, vl, mask)

manual = CutBins(echo[0, :, :], mask)
plt.show()
mask = manual.mask()
plotmask(orig_mask, mask)
# plotvar(echo, "Echo Intensity", mask)
# plotvar(echo, "Echo Intensity")
#

########## VELOCITY TEST ##########
