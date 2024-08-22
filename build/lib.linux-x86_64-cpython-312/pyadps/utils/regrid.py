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
    """
    Regrids 2D data from an irregular depth grid to a regular depth grid.

    Args:
        flobj: Object containing fixed leader information (e.g., depth cell length, number of cells).
        vlobj: Object containing variable leader information (e.g., transducer depth).
        data (numpy.ndarray): 2D array of data to be regridded (shape: depth x time).
        fill_value (float): Value used to fill in gaps in the data during interpolation.
        minimum_depth (str, optional): Specifies the minimum depth to start regridding. Can be 'cell' or 'surface'. Default is 'cell'.
        trimends (tuple, optional): A tuple to trim the data (start, end). Default is None.
        method (str, optional): Interpolation method to use. Default is 'nearest'.

    Returns:
        tuple: A tuple containing:
            - z (numpy.ndarray): Array of regridded depths.
            - data_regrid (numpy.ndarray): Regridded data array.

    Example:
    
        >>> import numpy as np
        >>> from regrid import regrid2d
        >>> flobj = ... # Initialize with appropriate fixed leader data
        >>> vlobj = ... # Initialize with appropriate variable leader data
        >>> data = np.random.rand(100, 200)  # Mock 2D data
        >>> fill_value = -32768
        >>> z, data_regrid = regrid2d(flobj, vlobj, data, fill_value)
        >>> print(z.shape, data_regrid.shape)
        (array of regridded depths, array of regridded data)
    """
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
    """
    Regrids 3D data from an irregular depth grid to a regular depth grid.

    Args:
        flobj: Object containing fixed leader information (e.g., depth cell length, number of cells).
        vlobj: Object containing variable leader information (e.g., transducer depth).
        data (numpy.ndarray): 3D array of data to be regridded (shape: beams x depth x time).
        fill_value (float): Value used to fill in gaps in the data during interpolation.
        minimum_depth (str, optional): Specifies the minimum depth to start regridding.Can be 'cell' or 'surface'. Default is 'cell'.
        trimends (tuple, optional): A tuple to trim the data (start, end). Default is None.
        method (str, optional): Interpolation method to use. Default is 'nearest'.

    Returns:
        tuple: A tuple containing:
            - z (numpy.ndarray): Array of regridded depths.
            - data_regrid (numpy.ndarray): Regridded data array.

    Example:
    
        >>> import numpy as np
        >>> from regrid import regrid3d
        >>> flobj = ... # Initialize with appropriate fixed leader data
        >>> vlobj = ... # Initialize with appropriate variable leader data
        >>> data = np.random.rand(4, 100, 200)  # Mock 3D data (4 beams)
        >>> fill_value = -32768
        >>> z, data_regrid = regrid3d(flobj, vlobj, data, fill_value)
        >>> print(z.shape, data_regrid.shape)
        (array of regridded depths, array of regridded 3D data)
    """
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
