import numpy as np
import scipy as sp
from pyadps.utils.readrdi import ReadFile
from .plotgen import PlotEnds


def trim_ends(ds, mask, transducer_depth=None, delta=20, method="Manual"):
    """
    Trim the ends of the data based on the provided method (e.g., manual selection)
    to remove invalid or irrelevant data points during deployment and recovery
    of moorings. This function modifies the mask to reflect the valid data range by marking
    invalid data points as `1` at the ends.

    Parameters
    ----------
    ds : pyadps.dataset
        The pyadps dataframe is loaded to obtain the data from the variable leader.
        This includes the depth of the transducer and other relevant information
        for trimming the data.
    mask : numpy.ndarray
        A mask array of the same shape as the data, where `1` indicates invalid data
        and `0` indicates valid data. The function modifies the mask based on the trimming
        process.
    method : str, optional
        The method used for trimming the data. Default is "Manual", which allows the user
        to manually select the start and end indices for trimming. Other methods can be
        added if required in the future.

    Returns
    -------
    numpy.ndarray
        The updated mask array with the ends trimmed. Data points at the beginning and end
        of the array marked as invalid (`1`) based on the trimming results.

    Notes
    -----
    - The function uses the transducer depth from the `vlobj` to determine the trimming boundaries.
    - When using the "Manual" method, the user is prompted with a plot that allows them to select
      the start and end indices for trimming. These indices are then used to adjust the mask.
    - The mask array is updated in-place to mark the trimmed areas as invalid.
    - The trimming process is based on the variable leader data and the selected method.

    Example
    -------
    >>> import pyadps
    >>> ds = pyadps.ReadFile("demo.000")
    >>> mask = pyadps.default_mask(ds)
    >>> updated_mask = trim_ends(ds, mask, method="Manual")
    """

    if isinstance(ds, ReadFile):
        vlobj = ds.variableleader
        transducer_depth = vlobj.vleader["Depth of Transducer"]
    elif isinstance(ds, np.ndarray) and ds.ndim == 1:
        transducer_depth = ds
    else:
        raise ValueError("Input must be a 1-D numpy array or a PyADPS instance")

    if method == "Manual":
        out = PlotEnds(transducer_depth, delta=delta)
        out.show()
        if out.start_ens > 0:
            mask[:, 0 : out.start_ens] = 1

        if out.end_ens < 0:
            mask[:, out.end_ens :] = 1

    return mask


def side_lobe_beam_angle(
    ds,
    mask,
    orientation="default",
    water_column_depth=0,
    extra_cells=2,
    cells=None,
    cell_size=None,
    bin1dist=None,
    beam_angle=None,
):
    """
    Mask the data contaminated due to surface/bottom backscatter based on the
    side lobe beam angle. This function can correct the orientation of the beam
    (upward or downward looking) and can account for deletion of additional cells.
    Water column depth is applicable for downward-looking ADCP.

    Parameters
    ----------
    ds : pyadps.dataset or np.ndarray
        The pyadps dataframe is loaded to obtain the data from the fixed and variable leader.
        This includes the depth of the transducer and other relevant information
        for trimming the data.

        If numpy.ndarray is loaded, the value should contain the transducer_depth.
        In such cases provide cells, cell_size, and bin 1 distance.
        Orientiation should be either 'up' or 'down' and not 'default'.

    mask : numpy.ndarray
        A mask array where invalid or false data points are marked. The mask is updated
        based on the calculated side lobe beam angles.
    orientation : str, optional
        The orientation of the beam. It can be 'default', 'up', or 'down'.
        Default orientation, set before deployment, is obtained from the file.
        If 'up' or 'down' is selected, the function will correct the orientation accordingly.
    water_column_depth : int, optional
        The depth of the water column. This value is used to adjust the beam angle
        calculations. Default is 0.
    extra_cells : int, optional
        The number of extra cells to consider when calculating the beam angle. Default is 2.
    cells: int, optional
        Number of cells
    cell_size: int, optional
        Cell size or depth cell length in cm
    beam_angle: int, optional
        Beam angle in degrees

    Returns
    -------
    numpy.ndarray
        The updated mask array with side lobe beam angle adjustments. The mask will
        reflect valid and invalid data points based on the calculated beam angles.

    Notes
    -----
    - The function uses the fixedleader and variableleader to retrieve the necessary information
      for the beam angle calculation. The `mask` array is updated based on these calculations.
    - The `orientation` parameter allows for adjustments to account for upward or downward
      looking ADCPs.
    - The `water_column_depth` permits detecting the bottom of the ocean for downward looking
      ADCP.
    - The `extra_cells` parameter adds additional cells in addition to those masked by beam angle
      calculation.

    Example
    -------
    >>> import pyadps
    >>> ds = pyadps.ReadFile("demo.000")
    >>> mask = pyadps.default_mask(ds)
    >>> updated_mask = side_lobe_beam_angle(ds, mask, orientation='down', water_column_depth=50, extra_cells=3)

    >>> transducer_depth = ds.variableleader.transducer_depth
    >>> cells = ds.fixedleader.cells
    >>> cell_size = ds.fixedleader.depth_cell_length
    >>> new_mask = side_lobe_beam_angle(transducer_depth, orientation='up', cells=cells, cell_size=cell_size, bin1dist=bindist)
    """
    if isinstance(ds, ReadFile) or ds.__class__.__name__ == "ReadFile":
        flobj = ds.fixedleader
        vlobj = ds.variableleader
        beam_angle = int(flobj.system_configuration()["Beam Angle"])
        cell_size = flobj.field()["Depth Cell Len"]
        bin1dist = flobj.field()["Bin 1 Dist"]
        cells = flobj.field()["Cells"]
        ensembles = flobj.ensembles
        transducer_depth = vlobj.vleader["Depth of Transducer"]
        if orientation.lower() == "default":
            orientation = flobj.system_configuration()["Beam Direction"]
    elif isinstance(ds, np.ndarray) and np.squeeze(ds).ndim == 1:
        transducer_depth = ds
        ensembles = np.size(ds)
    else:
        raise ValueError("Input must be a 1-D numpy array or a PyADPS instance")

    if orientation.lower() == "up":
        sgn = -1
        water_column_depth = 0
    elif orientation.lower() == "down":
        sgn = 1
    else:
        raise ValueError("Orientation should be either `up` or `down`")

    beam_angle = np.deg2rad(beam_angle)
    depth = transducer_depth / 10
    valid_depth = (water_column_depth - sgn * depth) * np.cos(
        beam_angle
    ) + sgn * bin1dist / 100
    valid_cells = np.trunc(valid_depth * 100 / cell_size) - extra_cells

    for i in range(ensembles):
        c = int(valid_cells[i])
        if cells > c:
            mask[c:, i] = 1

    return mask


def side_lobe_rssi_bump(echo, mask):
    pass


def manual_cut_bins(mask, min_cell, max_cell, min_ensemble, max_ensemble):
    """
    Apply manual bin cutting by selecting a specific range of cells and ensembles.

    Parameters:
        mask (numpy array): The mask array to modify.
        min_cell (int): The minimum cell index to mask.
        max_cell (int): The maximum cell index to mask.
        min_ensemble (int): The minimum ensemble index to mask.
        max_ensemble (int): The maximum ensemble index to mask.

    Returns:
        numpy array: The updated mask with selected areas masked.
    """
    # Ensure the indices are within valid range
    min_cell = max(0, min_cell)
    max_cell = min(mask.shape[0], max_cell)
    min_ensemble = max(0, min_ensemble)
    max_ensemble = min(mask.shape[1], max_ensemble)

    # Apply mask on the selected range
    mask[min_cell:max_cell, min_ensemble:max_ensemble] = 1

    return mask


def regrid2d(
    ds,
    data,
    fill_value,
    end_cell_option="cell",
    trimends=None,
    method="nearest",
    orientation="default",
    boundary_limit=0,
    cells=None,
    cell_size=None,
    bin1dist=None,
):
    """
    Regrids 2D data onto a new grid based on specified parameters.

    Parameters:
    -----------
    ds : pyadps.dataset or numpy.ndarray
        If pyadps dataframe is loaded, the data from the fixed and variable leader
        is automatically obtained. This includes the depth of the transducer and other relevant information
        for trimming the data.

        If numpy.ndarray is loaded, the value should contain the transducer_depth.
        In such cases provide cells, cell_size, and bin 1 distance.
        Orientiation should be either 'up' or 'down' and not 'default'.


    data : array-like
        The 2D data array to be regridded.

    fill_value : scalar
        The value used to fill missing or undefined grid points.

    end_cell_option : str or float, optional, default="cell"
        The depth of the last bin or boundary for the grid.
        Options include:
        - "cell" : Calculates the depth of the default last bin for the grid.
                   Truncates to surface for upward ADCP.
        - "surface": The data is gridded till the surface
        - "manual": User-defined depth for the grid.
                      Use boundary_limit option to provide the value.
        otherwise, a specific numerical depth value can be provided.

    trimends : tuple of floats, optional, default=None
        If provided, defines the ensemble range (start, end) for
        calculating the maximum/minimum transducer depth.
        Helps avoiding the deployment or retrieval data.
        E.g. (10, 3000)

    method : str, optional, default="nearest"
        The interpolation method to use for regridding based
        on scipy.interpolate.interp1d.
        Options include:
        - "nearest" : Nearest neighbor interpolation.
        - "linear" : Linear interpolation.
        - "cubic" : Cubic interpolation.

    orientation : str, optional, default="up"
        Defines the direction of the regridding for an upward/downward looking ADCP. Options include:
        - "up" : Regrid upwards (for upward-looking ADCP).
        - "down" : Regrid downwards (for downward-looking ADCP).

    boundary_limit : float, optional, default=0
        The limit for the boundary depth. This restricts the grid regridding to depths beyond the specified limit.

    cells: int, optional
        Number of cells

    cell_size: int, optional
        Cell size or depth cell length in cm

    bin1dist: int, optional
        Distance from the first bin in cm


    Returns:
    --------
    z: regridded depth
    regridded_data : array-like
        The regridded 2D data array, based on the specified method,
        orientation, and other parameters.

    Notes:
    ------
    - If `end_cell_option == boundary`, then `boundary_limit` is used to regrid the data.
    - This function allows for flexible regridding of 2D data to fit a new grid, supporting different interpolation methods.
    - The `boundary_limit` parameter helps restrict regridding to depths above or below a certain threshold.
    """

    if isinstance(ds, ReadFile) or ds.__class__.__name__ == "ReadFile":
        flobj = ds.fixedleader
        vlobj = ds.variableleader
        # Get values and convert to 'm'
        bin1dist = flobj.field()["Bin 1 Dist"] / 100
        transdepth = vlobj.vleader["Depth of Transducer"] / 10
        cell_size = flobj.field()["Depth Cell Len"] / 100
        cells = flobj.field()["Cells"]
        ensembles = flobj.ensembles
        if orientation.lower() == "default":
            orientation = flobj.system_configuration()["Beam Direction"]

    elif isinstance(ds, np.ndarray) and np.squeeze(ds).ndim == 1:
        transdepth = ds / 10
        ensembles = np.size(ds)

        if cells is None:
            raise ValueError("Input must include number of cells.")

        if cell_size is None:
            raise ValueError("Input must include cell size.")
        else:
            cell_size = cell_size / 100

        if bin1dist is None:
            raise ValueError("Input must include bin 1 distance.")
        else:
            bin1dist = bin1dist / 100

        if orientation.lower() != "up" and orientation.lower() != "down":
            raise ValueError("Orientation must be `up` or `down`.")
    else:
        raise ValueError("Input must be a 1-D numpy array or a PyADPS instance")

    if orientation.lower() == "up":
        sgn = -1
    else:
        sgn = 1

    # Create a regular grid

    # Find depth of first cell
    depth = transdepth + sgn * bin1dist

    # Find the maximum and minimum depth for first cell for upward
    # looking ADCP (minimum and maximum for downward looking)
    if trimends is not None:
        max_depth = abs(np.min(sgn * depth[trimends[0] : trimends[1]]))
        min_depth = abs(np.max(sgn * depth[trimends[0] : trimends[1]]))
    else:
        max_depth = abs(np.min(sgn * depth))
        min_depth = abs(np.max(sgn * depth))

    # FIRST CELL
    # Convert the first cell depth to the first regular grid depth
    depthfirstcell = max_depth - max_depth % cell_size

    # LAST CELL
    # Convert the last cell depth to last regular grid depth
    if end_cell_option.lower() == "surface":
        # Added one additional negative cell to accomodate 0 m.
        depthlastcell = sgn * cell_size
    elif end_cell_option.lower() == "cell":
        min_depth_regrid = min_depth - sgn * min_depth % cell_size
        depthlastcell = min_depth_regrid + sgn * (cells + 1) * cell_size
        # Check if this is required. Use 'surface' option
        if depthlastcell < 0:
            depthlastcell = sgn * cell_size
    elif end_cell_option.lower() == "manual":
        if sgn < 0 and boundary_limit > depthfirstcell:
            print(
                "ERROR: For upward looking ADCP, boundary limit should be less than transducer depth"
            )
            return
        if sgn > 0 and boundary_limit < depthfirstcell:
            print(
                "ERROR: For downward looking ADCP, boundary limit should be greater than transducer depth"
            )
            return
        # Set the last grid cell depth
        depthlastcell = boundary_limit
    else:
        print("ERROR: `end_cell_option` not recognized.")
        return

    # Negative used for upward and positive for downward.
    z = np.arange(sgn * depthfirstcell, sgn * depthlastcell, cell_size)
    regbins = len(z)

    regridded_data = np.zeros((regbins, ensembles))

    # Create original depth array
    for i, d in enumerate(depth):
        n = d + sgn * cell_size * cells
        # np.arange may include unexpected elements due to floating-point
        # precision issues at the stopping point. Changed to np.linspace.
        #
        # depth_bins = np.arange(sgn*d, sgn*n, cell_size)
        depth_bins = np.linspace(sgn * d, sgn * n, cells)
        f = sp.interpolate.interp1d(
            depth_bins,
            data[:, i],
            kind=method,
            fill_value=fill_value,
            bounds_error=False,
        )
        gridz = f(z)

        regridded_data[:, i] = gridz

    return abs(z), regridded_data


def regrid3d(
    ds,
    data,
    fill_value,
    end_cell_option="cell",
    trimends=None,
    method="nearest",
    orientation="up",
    boundary_limit=0,
    cells=None,
    cell_size=None,
    bin1dist=None,
    beams=None,
):
    """
    Regrids 3D data onto a new grid based on specified parameters.

    Parameters:
    -----------
    ds : pyadps.dataset
        The pyadps dataframe is loaded to obtain the data from the fixed and variable leader.
        This includes the depth of the transducer and other relevant information
        for trimming the data.

    data : array-like
        The 3D data array to be regridded, with dimensions
        typically representing time, depth, and another axis (e.g., ensembles).

    fill_value : scalar
        The value used to fill missing or undefined grid points.

    end_cell_option : str or float, optional, default="cell"
        The depth of the last bin or boundary for the grid.
        Options include:
        - "cell" : Calculates the depth of the default last bin for the grid.
                   Truncates to surface for upward ADCP.
        - "surface" : The data is gridded till the surface.
        - "manual" : User-defined depth for the grid.
                      Use boundary_limit option to provide the value.
        Otherwise, a specific numerical depth value can be provided.

    trimends : tuple of integer, optional, default=None
        If provided, defines the ensemble range (start, end) for
        calculating the maximum/minimum transducer depth.
        Helps avoiding the deployment or retrieval data.
        E.g., (10, 3000)

    method : str, optional, default="nearest"
        The interpolation method to use for regridding based
        on scipy.interpolate.interp1d.
        Options include:
        - "nearest" : Nearest neighbor interpolation.
        - "linear" : Linear interpolation.
        - "cubic" : Cubic interpolation.

    orientation : str, optional, default="up"
        Defines the direction of the regridding for an upward/downward looking ADCP. Options include:
        - "up" : Regrid upwards (for upward-looking ADCP).
        - "down" : Regrid downwards (for downward-looking ADCP).

    boundary_limit : float, optional, default=0
        The limit for the boundary depth. This restricts the grid regridding to depths beyond the specified limit.

    cells: int, optional
        Number of cells

    cell_size: int, optional
        Cell size or depth cell length in cm

    bin1dist: int, optional
        Distance from the first bin in cm

    beams: int, optional
        Number of beams



    Returns:
    --------
    z : array-like
        The regridded depth array.
    regridded_data : array-like
        The regridded 3D data array, based on the specified method,
        orientation, and other parameters.

    Notes:
    ------
    - If `end_cell_option == boundary`, then `boundary_limit` is used to regrid the data.
    - This function allows for flexible regridding of 3D data to fit a new grid, supporting different interpolation methods.
    - The `boundary_limit` parameter helps restrict regridding to depths above or below a certain threshold.
    - This function is an extension of 2D regridding to handle the time dimension or other additional axes in the data.
    """

    if isinstance(ds, ReadFile):
        flobj = ds.fixedleader
        beams = flobj.field()["Beams"]
    elif isinstance(ds, np.ndarray) and ds.ndim == 1:
        if beams is None:
            raise ValueError("Input must include number of beams.")
    else:
        raise ValueError("Input must be a 1-D numpy array or a PyADPS instance")

    z, data_dummy = regrid2d(
        ds,
        data[0, :, :],
        fill_value,
        end_cell_option=end_cell_option,
        trimends=trimends,
        method=method,
        orientation=orientation,
        boundary_limit=boundary_limit,
        cells=cells,
        cell_size=cell_size,
        bin1dist=bin1dist,
    )

    newshape = np.shape(data_dummy)
    regridded_data = np.zeros((beams, newshape[0], newshape[1]))
    regridded_data[0, :, :] = data_dummy

    for i in range(beams - 1):
        z, data_dummy = regrid2d(
            ds,
            data[i + 1, :, :],
            fill_value,
            end_cell_option=end_cell_option,
            trimends=trimends,
            method=method,
            orientation=orientation,
            boundary_limit=boundary_limit,
            cells=cells,
            cell_size=cell_size,
            bin1dist=bin1dist,
        )
        regridded_data[i + 1, :, :] = data_dummy

    return z, regridded_data
