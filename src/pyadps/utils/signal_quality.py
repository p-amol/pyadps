import numpy as np
from pyadps.utils.plotgen import PlotNoise
from pyadps.utils.readrdi import ReadFile


def qc_check(var, mask, cutoff=0):
    """
    Perform a quality control check on the provided data and update the mask 
    based on a cutoff threshold. Values in `var` that are less than the cutoff 
    are marked as invalid in the mask.

    Parameters
    ----------
    var : numpy.ndarray
        The input array containing data to be checked against the cutoff.
    mask : numpy.ndarray
        An integer array of the same shape as `var`, where `1` indicates 
        invalid data and `0` indicates valid data.
    cutoff : int, optional
        The threshold value for quality control. Any value in `var` less than 
        or equal to this cutoff will be marked as invalid in the mask. Default is 0.

    Returns
    -------
    numpy.ndarray
        An updated integer mask array of the same shape as `var`, with `1` 
        indicating invalid data and `0` indicating valid data.

    Notes
    -----
    - The function modifies the `mask` by applying the cutoff condition. 
      Values in `var` that are less than or equal to the cutoff will be 
      marked as invalid (`1`), while all other values will remain valid (`0`).
    - Ensure that `var` and `mask` are compatible in shape for element-wise 
      operations.

    Example
    -------
    >>> import pyadps
    >>> ds = pyadps.Readfile('dummy.000')
    >>> var = ds.echo.data
    >>> mask = qc_check(var, mask, cutoff=40)
    """

    shape = np.shape(var)
    if len(shape) == 2:
        mask[var[:, :] < cutoff] = 1
    else:
        beam = shape[0]
        for i in range(beam):
            mask[var[i, :, :] < cutoff] = 1
    # values, counts = np.unique(mask, return_counts=True)
    # print(values, counts, np.round(counts[1] * 100 / np.sum(counts)))
    return mask


def correlation_check(ds, mask, cutoff=64):
    """
    Perform an correlation check on the provided variable and update the 
    mask to mark valid and invalid values based on a cutoff threshold.

    Parameters
    ----------
    ds : pyadps.dataset
        The input pyadps dataframe containing correlation data to be checked.
        Accepts 2-D or 3-D masks.
    mask : numpy.ndarray
        An integer array of the same shape as `var`, where `1` indicates invalid 
        data or masked data and `0` indicates valid data.
    cutoff : float, optional
        The threshold value for echo intensity. Any value in `ds.correlation.data` below 
        this cutoff will be considered invalid and marked as `1` in the mask. 
        Default is 64.

    Returns
    -------
    numpy.ndarray
        An updated integer mask array of the same shape as `var`, with `1` 
        indicating invalid or masked data (within the cutoff limit) and `0` indicating 
        valid.

    Notes
    -----
    - The function modifies the `mask` based on the cutoff condition. Valid 
      values in `var` retain their corresponding mask value as `0`, while 
      invalid values or previously masked elements are marked as `1`.
      operations.

    Example
    -------
    >>> import pyadps
    >>> ds = pyadps.Readfile('dummy.000')
    >>> outmask = correlation_check(ds, mask, cutoff=9999)
    """
    correlation = ds.correlation.data
    mask = qc_check(correlation, mask, cutoff=cutoff)
    return mask

def echo_check(ds, mask, cutoff=40):
    """
    Perform an echo intensity check on the provided variable and update the 
    mask to mark valid and invalid values based on a cutoff threshold.

    Parameters
    ----------
    ds : pyadps.dataset
        The input pyadps dataframe containing echo intensity data to be checked.
        Accepts 2-D or 3-D masks.
    mask : numpy.ndarray
        An integer array of the same shape as `var`, where `1` indicates invalid 
        data or masked data and `0` indicates valid data.
    cutoff : float, optional
        The threshold value for echo intensity. Any value in `ds.echo.data` below 
        this cutoff will be considered invalid and marked as `1` in the mask. 
        Default is 40.

    Returns
    -------
    numpy.ndarray
        An updated integer mask array of the same shape as `var`, with `1` 
        indicating invalid or masked data (within the cutoff limit) and `0` indicating 
        valid.

    Notes
    -----
    - The function modifies the `mask` based on the cutoff condition. Valid 
      values in `var` retain their corresponding mask value as `0`, while 
      invalid values or previously masked elements are marked as `1`.
    - Ensure that `var` and `mask` are compatible in shape for element-wise 
      operations.

    Example
    -------
    >>> import pyadps
    >>> ds = pyadps.Readfile('dummy.000')
    >>> outmask = echo_check(ds, mask, cutoff=9999)
    """

    echo = ds.echo.data
    mask = qc_check(echo, mask, cutoff=cutoff)
    return mask


def ev_check(ds, mask, cutoff=9999):
    """
    Perform an error velocity check on the provided variable and update the 
    mask to mark valid and invalid values based on a cutoff threshold.

    Parameters
    ----------
    ds : pyadps.dataset
        The input pyadps dataframe containing error velocity data to be checked.
    mask : numpy.ndarray
        An integer array of the same shape as `var`, where `1` indicates invalid 
        data or masked data and `0` indicates valid data.
    cutoff : float, optional
        The threshold value for error velocity. Any value in `var` exceeding 
        this cutoff will be considered invalid and marked as `0` in the mask. 
        Default is 9999.

    Returns
    -------
    numpy.ndarray
        An updated integer mask array of the same shape as `var`, with `1` 
        indicating invalid or masked data (within the cutoff limit) and `0` indicating 
        valid.

    Notes
    -----
    - The function modifies the `mask` based on the cutoff condition. Valid 
      values in `var` retain their corresponding mask value as `0`, while 
      invalid values or previously masked elements are marked as `1`.
    - Ensure that `var` and `mask` are compatible in shape for element-wise 
      operations.

    Example
    -------
    >>> import pyadps
    >>> ds = pyadps.Readfile('dummy.000')
    >>> outmask = ev_check(ds, mask, cutoff=9999)
    """

    var = ds.velocity.data[3, :, :]
    shape = np.shape(var)
    var = abs(var)
    if len(shape) == 2:
        mask[(var[:, :] >= cutoff) & (var[:, :] < 32768)] = 1
    else:
        beam = shape[2]
        for i in range(beam):
            mask[(var[i, :, :] >= cutoff) & (var[i, :, :] < 32768)] = 1
    return mask


def pg_check(ds, mask, cutoff=0, threebeam=True):
    """
    Perform a percent-good check on the provided data and update the mask 
    to mark valid and invalid values based on a cutoff threshold.

    Parameters
    ----------
    ds : pyadps.dataset 
        The input pyadps dataframe containing percent-good data, where values range from 
        0 to 100 (maximum percent good).
    mask : numpy.ndarray
        An integer array of the same shape as `pgood`, where `1` indicates 
        invalid data and `0` indicates valid data.
    cutoff : float, optional
        The threshold value for percent good. Any value in `pgood` greater than 
        or equal to this cutoff will be considered valid (marked as `0`), 
        while values not exceeding the cutoff are marked as invalid (`1`). 
        Default is 0.
    threebeam : bool, optional
        If `True`, sums up Percent Good 1 and Percent Good 4 for the check. 

    Returns
    -------
    numpy.ndarray
        An updated integer mask array of the same shape as `pgood`, with `1` 
        indicating invalid data and `0` indicating valid data.

    Notes
    -----
    - The function modifies the `mask` based on the cutoff condition. Valid 
      values in `pgood` are marked as `0`, while invalid values are marked 
      as `1` in the mask.
    - Ensure that `pgood` and `mask` are compatible in shape for element-wise 
      operations.
    - If `threebeam` is `True`, the logic may be adjusted to allow partial 
      validity based on specific criteria.

    Example
    -------
    >>> import pyadps
    >>> ds = pyadps.Readfile('dummy.000')
    >>> outmask = pg_check(ds, mask, cutoff=50, threebeam=True)
    """

    pgood = ds.percentgood.data
    if threebeam:
        pgood1 = pgood[0, :, :] + pgood[3, :, :]
    else:
        pgood1 = pgood[:, :, :]

    mask[pgood1[:, :] < cutoff] = 1
    return mask


def false_target(ds, mask, cutoff=255, threebeam=True):
    """
    Apply a false target detection algorithm based on echo intensity values. 
    This function identifies invalid or false targets in the data and updates 
    the mask accordingly based on a specified cutoff threshold.

    Parameters
    ----------
    ds : pyadps.dataset 
        The input pyadps dataframe containing echo intensity values, which are used to 
        detect false targets.
    mask : numpy.ndarray
        An integer array of the same shape as `echo`, where `1` indicates 
        invalid or false target data and `0` indicates valid data.
    cutoff : int, optional
        The threshold value for echo intensity. Any value in `echo` greater 
        than or equal to this cutoff will be considered a false target (invalid), 
        marked as `1` in the mask. Default is 255.
    threebeam : bool, optional
        If `True`, applies a relaxed check that considers data valid even 
        when only three beams report valid data. Default is `True`.

    Returns
    -------
    numpy.ndarray
        An updated integer mask array of the same shape as `echo`, with `1` 
        indicating false target or invalid data and `0` indicating valid data.

    Notes
    -----
    - The function modifies the `mask` by applying the cutoff condition. 
      Echo values greater than or equal to the cutoff are marked as false 
      targets (`1`), while values below the cutoff are considered valid (`0`).
    - If `threebeam` is `True`, a more lenient check may be applied to handle 
      data with fewer valid beams.
    - Ensure that `echo` and `mask` are compatible in shape for element-wise 
      operations.

    Example
    -------
    >>> import pyadps
    >>> ds = pyadps.Readfile('dummy.000')
    >>> mask = false_target(echo, mask, cutoff=255)
    """

    echo = ds.echo.data

    shape = np.shape(echo)
    for i in range(shape[1]):
        for j in range(shape[2]):
            x = np.sort(echo[:, i, j])
            if threebeam:
                if x[-1] - x[1] > cutoff:
                    mask[i, j] = 1
            else:
                if x[-1] - x[0] > cutoff:
                    mask[i, j] = 1

    values, counts = np.unique(mask, return_counts=True)
    # print(values, counts, np.round(counts[1] * 100 / np.sum(counts)))
    return mask


def default_mask(ds):
    """
    Create a default 2-D mask file based on the velocity data.
    This function generates a mask where values are marked as valid or invalid 
    based on the missing values from the velocity data.

    Parameters
    ----------
    ds : pyadps.dataset or numpy.ndarray
         A pyadps data frame is used to extract velocity and dimensions for the mask.
         If numpy.ndarray, enter the values for beams, cells and ensembles.

    Returns
    -------
    numpy.ndarray
        A mask array of the same shape as `velocity`, where `1` indicates invalid 
        data  and `0` indicates valid data.

    Notes
    -----
    - The function uses the velocity data along with the information from the 
      Fixed Leader object to determine which values are valid and which are invalid.

    Example
    -------
    >>> import pyadps
    >>> ds = pyadps.ReadFile('demo.000')
    >>> mask = pyadps.default_mask(ds)
    """
    if isinstance(ds, ReadFile) or ds.__class__.__name__ == "ReadFile":
        flobj = ds.fixedleader
        velocity = ds.velocity.data
        cells = flobj.field()["Cells"]
        beams = flobj.field()["Beams"]
        ensembles = flobj.ensembles
    elif isinstance(ds, np.ndarray) and ds.ndim == 3:
        velocity = ds
        beams = ds.shape[0]
        cells = ds.shape[1]
        ensembles = ds.shape[2] 
    else:
        raise ValueError("Input must be a 3-D numpy array or a PyADPS instance")

    mask = np.zeros((cells, ensembles))
    # Ignore mask for error velocity
    for i in range(beams - 1):
        mask[velocity[i, :, :] < -32767] = 1
    return mask


def qc_prompt(ds, name, data=None):
    """
    Prompt the user to confirm or adjust the quality control threshold for a specific 
    parameter based on predefined ranges. The function provides an interactive interface 
    for the user to adjust thresholds for various quality control criteria, with options 
    for certain thresholds like "Echo Intensity Thresh" to check the noise floor.

    Parameters
    ----------
    flobj : FixedLeader
        An instance of the FixedLeader class that holds metadata and configuration 
        data. The `flobj` is used to retrieve the current threshold values based on 
        the provided parameter name.
    name : str
        The name of the parameter for which the threshold is being adjusted. Examples 
        include "Echo Intensity Thresh", "Correlation Thresh", "Percent Good Min", etc.
    data : numpy.ndarray, optional
        The data associated with the threshold. This is required for parameters like 
        "Echo Intensity Thresh" where a noise floor check might be performed. Default is None.

    Returns
    -------
    int
        The updated threshold value, either the default or the new value entered by the user.

    Notes
    -----
    - The function will prompt the user to change the threshold for the given `name` parameter.
    - For certain parameters, the user may be asked if they would like to check the noise floor 
      (for example, for "Echo Intensity Thresh"). This triggers the display of a plot and lets 
      the user select a new threshold.
    - The function ensures that the new threshold is within the acceptable range for each parameter.
    - The default thresholds are provided if the user chooses not to change them.

    Example
    -------
    >>> import pyadps
    >>> ds = pyadps.ReadFile('demo.000')
    >>> name = "Echo Intensity Thresh"
    >>> threshold = qc_prompt(ds, name, data)
    The default threshold for echo intensity thresh is 0
    Would you like to change the threshold [y/n]: y
    Would you like to check the noise floor [y/n]: y
    Threshold changed to 50
    """

    flobj = ds.fixedleader
    cutoff = 0
    if name == "Echo Intensity Thresh":
        cutoff = 0
    else:
        cutoff = flobj.field()[name]

    if name in ["Echo Thresh", "Correlation Thresh", "False Target Thresh"]:
        var_range = [0, 255]
    elif name == "Percent Good Min":
        var_range = [0, 100]
    elif name == "Error Velocity Thresh":
        var_range = [0, 5000]
    else:
        var_range = [0, 255]

    print(f"The default threshold for {name.lower()} is {cutoff}")
    affirm = input("Would you like to change the threshold [y/n]: ")
    if affirm.lower() == "y":
        while True:
            if name == "Echo Intensity Thresh":
                affirm2 = input("Would you like to check the noise floor [y/n]: ")
                if affirm2.lower() == "y":
                    p = PlotNoise(data)
                    p.show()
                    cutoff = p.cutoff
                else:
                    cutoff = input(
                        f"Enter new {name} [{var_range[0]}-{var_range[1]}]: "
                    )
            else:
                cutoff = input(f"Enter new {name} [{var_range[0]}-{var_range[1]}]: ")

            cutoff = int(cutoff)
            try:
                if cutoff >= var_range[0] and int(cutoff) <= var_range[1]:
                    break
                else:
                    print(f"Enter an integer between {var_range[0]} and {var_range[1]}")
            except ValueError:
                print("Enter a valid number")

        print(f"Threshold changed to {cutoff}")

    else:
        print(f"Default threshold {cutoff} used.")
    # return int(ct)
    return cutoff
