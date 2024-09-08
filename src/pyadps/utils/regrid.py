import numpy as np
import scipy as sp

# import readrdi as rd


def regrid2d(
    flobj,
    vlobj,
    data,
    fill_value,
    minimum_depth="cell",
    trimends=None,
    method="nearest",
):
    depth = vlobj.vleader["Depth of Transducer"] / 10
    depth_interval = flobj.field()["Depth Cell Len"] / 100
    bins = flobj.field()["Cells"]
    ensembles = flobj.ensembles

    # Create a regular grid
    # Find the minimum depth.
    if minimum_depth == "surface":
        mindepth = depth_interval
    elif minimum_depth == "cell":
        if trimends is not None:
            dm = np.min(depth[trimends[0] : trimends[1]])
        else:
            dm = np.min(depth)
        mintransdepth = dm - bins * depth_interval
        mindepth = mintransdepth - mintransdepth % depth_interval
        mindepth = mindepth - depth_interval
        # If mindepth is above surface choose surface
        if mindepth < 0:
            mindepth = depth_interval
    else:
        mindepth = depth_interval

    maxbins = np.max(depth) // depth_interval + 1
    # print(np.max(depth), np.max(depth) % depth_interval)
    # if np.max(depth) % depth_interval > depth_interval / 2:
    #     maxbins = maxbins + 1

    maxdepth = maxbins * depth_interval
    z = np.arange(-1 * maxdepth, -1 * mindepth, depth_interval)
    regbins = len(z)

    # print(maxbins, bins, ensemble)
    data_regrid = np.zeros((regbins, ensembles))

    # Create original depth array
    for i, d in enumerate(depth):
        n = -1 * d + depth_interval * bins
        depth_bins = np.arange(-1 * d, n, depth_interval)
        f = sp.interpolate.interp1d(
            depth_bins,
            data[:, i],
            kind=method,
            fill_value=fill_value,
            bounds_error=False,
        )
        gridz = f(z)

        data_regrid[:, i] = gridz

    return z, data_regrid


def regrid3d(
    flobj,
    vlobj,
    data,
    fill_value,
    minimum_depth="cell",
    trimends=None,
    method="nearest",
):
    beams = flobj.field()["Beams"]
    z, data_dummy = regrid2d(
        flobj,
        vlobj,
        data[0, :, :],
        fill_value,
        minimum_depth=minimum_depth,
        trimends=trimends,
        method=method,
    )

    newshape = np.shape(data_dummy)
    data_regrid = np.zeros((beams, newshape[0], newshape[1]))
    data_regrid[0, :, :] = data_dummy

    for i in range(beams - 1):
        z, data_dummy = regrid2d(
            flobj,
            vlobj,
            data[i + 1, :, :],
            fill_value,
            minimum_depth=minimum_depth,
            trimends=trimends,
            method=method,
        )
        data_regrid[i + 1, :, :] = data_dummy

    return z, data_regrid


# # read data
# filename = "BGS11000.000"
# fl = rd.FixedLeader(filename, run="fortran")
# vl = rd.VariableLeader(filename, run="fortran")
# # echo = rd.echo(filename, run="fortran")
# vel = rd.velocity(filename, run="fortran")
# pressure = vl.vleader["Pressure"]
#
# shape = np.shape(vel[0, :, :])
# mask = np.zeros(shape)
# orig_mask = np.copy(mask)
#
# z, newvel = regrid2d(fl, vl, vel[0, :, :], fill_value=-32768)
# z, newmask = regrid(mask[:, :], pressure, depth_interval=4, fill_value=1)
# z, newvel3d = regrid3d(vel, pressure, depth_interval=4, fill_value=-32768)
