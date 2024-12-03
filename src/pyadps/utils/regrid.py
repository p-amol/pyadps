import numpy as np
import scipy as sp

# import readrdi as rd


def regrid2d(
    flobj,
    vlobj,
    data,
    fill_value,
    end_bin_option="cell",
    trimends=None,
    method="nearest",
    orientation="default",
    boundary_limit=0,
):
    """
    Regrids 2D data onto a new grid based on specified parameters.

    Parameters:
    -----------
    flobj : object
        The fixed leader object that contains information 
        about the fixed leader data.
        
    vlobj : object
        The variable leader object that contains information 
        about the variable leader data.
        
    data : array-like
        The 2D data array to be regridded.
        
    fill_value : scalar
        The value used to fill missing or undefined grid points.
        
    end_bin_option : str or float, optional, default="cell"
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
        
    Returns:
    --------
    z: regridded depth
    regridded_data : array-like
        The regridded 2D data array, based on the specified method, 
        orientation, and other parameters.
    
    Notes:
    ------
    - If `end_bin_option == boundary`, then `boundary_limit` is used to regrid the data.
    - This function allows for flexible regridding of 2D data to fit a new grid, supporting different interpolation methods.
    - The `boundary_limit` parameter helps restrict regridding to depths above or below a certain threshold.
    """
   
    # Get values and convert to 'm'
    bin1dist = flobj.field()["Bin 1 Dist"] / 100
    transdepth = vlobj.vleader["Depth of Transducer"] / 10
    depth_interval = flobj.field()["Depth Cell Len"] / 100
    bins = flobj.field()["Cells"]
    ensembles = flobj.ensembles

    if orientation.lower() == "default":
        orientation = flobj.system_configuration()['Beam Direction']

    if orientation.lower() == "up":
        sgn = -1 
    else:
        sgn = 1

    # Create a regular grid

    # Find depth of first cell
    depth = transdepth + sgn*bin1dist

    # Find the maximum and minimum depth for first cell for upward 
    # looking ADCP (minimum and maximum for downward looking)
    if trimends is not None:
        max_depth = abs(np.min(sgn*depth[trimends[0] : trimends[1]]))
        min_depth = abs(np.max(sgn*depth[trimends[0] : trimends[1]]))
    else:
        max_depth = abs(np.min(sgn*depth))
        min_depth = abs(np.max(sgn*depth))

    # FIRST CELL
    # Convert the first cell depth to the first regular grid depth 
    depthfirstcell = max_depth - max_depth % depth_interval

    # LAST CELL
    # Convert the last cell depth to last regular grid depth 
    if end_bin_option.lower() == "surface":
        # Added one additional negative cell to accomodate 0 m.
        depthlastcell = sgn * depth_interval
    elif end_bin_option.lower() == "cell":
        min_depth_regrid = min_depth - sgn*min_depth % depth_interval
        depthlastcell = min_depth_regrid  + sgn* (bins+1) * depth_interval
        # Check if this is required. Use 'surface' option
        if depthlastcell < 0:
            depthlastcell = sgn*depth_interval 
    elif end_bin_option.lower() == "manual":
        if sgn < 0 and boundary_limit > depthfirstcell:
            print("ERROR: For upward looking ADCP, boundary limit should be less than transducer depth")
            return
        if sgn > 0 and boundary_limit < depthfirstcell:
            print("ERROR: For downward looking ADCP, boundary limit should be greater than transducer depth")
            return
        # Set the last grid cell depth
        depthlastcell = boundary_limit 
    else:
        print("ERROR: `end_bin_option` not recognized.")
        return
  
    # Negative used for upward and positive for downward.
    z = np.arange(sgn * depthfirstcell, sgn * depthlastcell, depth_interval)
    regbins = len(z)

    regridded_data = np.zeros((regbins, ensembles))

    # Create original depth array
    for i, d in enumerate(depth):
        n = d + sgn*depth_interval * bins
        # np.arange may include unexpected elements due to floating-point 
        # precision issues at the stopping point. Changed to np.linspace.
        #
        # depth_bins = np.arange(sgn*d, sgn*n, depth_interval)
        depth_bins = np.linspace(sgn*d, sgn*n, bins)
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
    flobj,
    vlobj,
    data,
    fill_value,
    end_bin_option="cell",
    trimends=None,
    method="nearest",
    orientation="up",
    boundary_limit=0,
):
    """
    Regrids 3D data onto a new grid based on specified parameters.

    Parameters:
    -----------
    flobj : object
        The fixed leader object that contains information 
        about the fixed leader data.
        
    vlobj : object
        The variable leader object that contains information 
        about the variable leader data.
        
    data : array-like
        The 3D data array to be regridded, with dimensions 
        typically representing time, depth, and another axis (e.g., ensembles).
        
    fill_value : scalar
        The value used to fill missing or undefined grid points.
        
    end_bin_option : str or float, optional, default="cell"
        The depth of the last bin or boundary for the grid.
        Options include:
        - "cell" : Calculates the depth of the default last bin for the grid.
                   Truncates to surface for upward ADCP.
        - "surface" : The data is gridded till the surface.  
        - "manual" : User-defined depth for the grid.
                      Use boundary_limit option to provide the value. 
        Otherwise, a specific numerical depth value can be provided.
        
    trimends : tuple of floats, optional, default=None
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
        
    Returns:
    --------
    z : array-like
        The regridded depth array.
    regridded_data : array-like
        The regridded 3D data array, based on the specified method, 
        orientation, and other parameters.
    
    Notes:
    ------
    - If `end_bin_option == boundary`, then `boundary_limit` is used to regrid the data.
    - This function allows for flexible regridding of 3D data to fit a new grid, supporting different interpolation methods.
    - The `boundary_limit` parameter helps restrict regridding to depths above or below a certain threshold.
    - This function is an extension of 2D regridding to handle the time dimension or other additional axes in the data.
    """

    beams = flobj.field()["Beams"]
    z, data_dummy = regrid2d(
        flobj,
        vlobj,
        data[0, :, :],
        fill_value,
        end_bin_option=end_bin_option,
        trimends=trimends,
        method=method,
        orientation=orientation,
        boundary_limit=boundary_limit,
    )

    newshape = np.shape(data_dummy)
    regridded_data = np.zeros((beams, newshape[0], newshape[1]))
    regridded_data[0, :, :] = data_dummy

    for i in range(beams - 1):
        z, data_dummy = regrid2d(
            flobj,
            vlobj,
            data[i + 1, :, :],
            fill_value,
            end_bin_option=end_bin_option,
            trimends=trimends,
            method=method,
            orientation=orientation,
            boundary_limit=boundary_limit,
        )
        regridded_data[i + 1, :, :] = data_dummy

    return z, regridded_data

