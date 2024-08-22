import numpy as np


def qc_check(var, mask, cutoff=0):
    """
    Modify the mask array based on the cutoff criteria. Values in the `var` array
    that are less than the cutoff will be marked in the `mask` array.

    Args:
        var (numpy.ndarray): Array of data to check against the cutoff. Can be 2D (depth, time) or 3D (beam, depth, time).
        mask (numpy.ndarray): Mask array with the same shape as `var`. Values less than cutoff in `var` will set the corresponding `mask` entries to 1.
        cutoff (int): Threshold value. Default is 0. Values in `var` below this threshold will update the `mask`.

    Returns:
        numpy.ndarray: Modified `mask` array with updated values based on the cutoff.

    Example:
        >>> import numpy as np
        >>> from signal_quality import qc_check
        >>> var = np.array([[1, 2, 3], [4, 5, 6]])
        >>> mask = np.zeros_like(var)
        >>> qc_check(var, mask, cutoff=4)
        array([[1, 1, 1],
               [1, 1, 1]])
    """
    shape = np.shape(var)
    if len(shape) == 2:
        mask[var[:, :] < cutoff] = 1
    else:
        beam = shape[0]
        for i in range(beam):
            mask[var[i, :, :] < cutoff] = 1
    values, counts = np.unique(mask, return_counts=True)
    # print(values, counts, np.round(counts[1] * 100 / np.sum(counts)))
    return mask


cor_check = qc_check
echo_check = qc_check


def ev_check(var, mask, cutoff=9999):
    """
    Updates the mask array based on error velocity values that exceed the specified cutoff.

    Args:
        var (numpy.ndarray): Array of error velocities. Can be 2D (depth, time) or 3D (beam, depth, time).
        mask (numpy.ndarray): Mask array with the same shape as `var`. Entries in `mask` are updated if corresponding `var` values exceed the cutoff.
        cutoff (int): Threshold value for error velocities. Default is 9999.

    Returns:
        numpy.ndarray: Updated `mask` array with entries set to 1 where `var` exceeds the cutoff.

    Example:
    
        >>> import numpy as np
        >>> from signal_quality import ev_check
        >>> var = np.array([[10000, 20000], [30000, 40000]])
        >>> mask = np.zeros_like(var)
        >>> ev_check(var, mask, cutoff=20000)
        array([[0, 1],
               [1, 1]])
    """
    shape = np.shape(var)
    var = abs(var)
    if len(shape) == 2:
        mask[(var[:, :] >= cutoff) & (var[:, :] < 32768)] = 1
    else:
        beam = shape[2]
        for i in range(beam):
            mask[(var[i, :, :] >= cutoff) & (var[i, :, :] < 32768)] = 1
    values, counts = np.unique(mask, return_counts=True)
    # print(values, counts, np.round(counts[1] * 100 / np.sum(counts)))
    return mask


def pg_check(pgood, mask, cutoff=0, threebeam=True):
    """
    Updates the mask array based on the percent good values.

    Args:
        pgood (numpy.ndarray): Array of percent good values. Can be 2D (depth, time) or 3D (beam, depth, time).
        mask (numpy.ndarray): Mask array with the same shape as `pgood`. Entries in `mask` are updated if corresponding `pgood` values fall below the cutoff.
        cutoff (int): Threshold value for percent good. Default is 0.
        threebeam (bool): Whether to combine percent good values from multiple beams. Default is True.

    Returns:
        numpy.ndarray: Updated `mask` array with entries set to 1 where `pgood` is below the cutoff.

    Example:
    
        >>> import numpy as np
        >>> from signal_quality import pg_check
        >>> pgood = np.array([[90, 85], [80, 75]])
        >>> mask = np.zeros_like(pgood)
        >>> pg_check(pgood, mask, cutoff=80)
        array([[0, 0],
               [1, 1]])
    """
    if threebeam:
        pgood1 = pgood[0, :, :] + pgood[3, :, :]
    else:
        pgood1 = pgood[:, :, :]

    mask[pgood1[:, :] < cutoff] = 1
    values, counts = np.unique(mask, return_counts=True)
    # print(values, counts, np.round(counts[1] * 100 / np.sum(counts)))
    return mask


def false_target(echo, mask, cutoff=255, threebeam=True):
    """
    Identifies and masks false targets based on the difference in echo values.

    Args:
        echo (numpy.ndarray): Array of echo intensity values. Can be 3D (beam, depth, time).
        mask (numpy.ndarray): Mask array with the same shape as the 2D portion of `echo`. Entries in `mask` are updated if echo values indicate false targets.
        cutoff (int): Threshold for the maximum difference between sorted echo values. Default is 255.
        threebeam (bool): Whether to consider three beams for detecting false targets. Default is True.

    Returns:
        numpy.ndarray: Updated `mask` array with entries set to 1 where false targets are detected.

    Example:
    
        >>> import numpy as np
        >>> from signal_quality import false_target
        >>> echo = np.random.randint(0, 300, (3, 5, 5))
        >>> mask = np.zeros((5, 5))
        >>> false_target(echo, mask, cutoff=100)
        array([[1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]])
    """
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


def default_mask(flobj, velocity):
    """
    Creates a default mask array based on the given velocity data and FixedLeader object.

    Args:
        flobj: FixedLeader object providing field and ensemble information.
        velocity (numpy.ndarray): Array of velocity data with shape (beam, depth, time).

    Returns:
        numpy.ndarray: Default mask array based on velocity data. Mask values are set to 1 for out-of-range velocities.

    Example:
    
        >>> import numpy as np
        >>> from signal_quality import default_mask
        >>> flobj = FixedLeaderObject()  # Replace with actual FixedLeader object
        >>> velocity = np.random.rand(3, 10, 10) * 50000
        >>> mask = default_mask(flobj, velocity)
    """
    cells = flobj.field()["Cells"]
    beams = flobj.field()["Beams"]
    ensembles = flobj.ensembles
    mask = np.zeros((cells, ensembles))
    # Ignore mask for error velocity
    for i in range(beams - 1):
        mask[abs(velocity[i, :, :]) > 32766] = 1
    return mask


def qc_prompt(flobj, name, data=None):
    """
    Prompts the user to set or change the quality control (QC) threshold based on the given name.

    Args:
        flobj: FixedLeader object providing threshold information.
        name (str): Name of the QC parameter (e.g., "Echo Intensity Thresh").
        data: Optional data used for visual checks (e.g., noise floor plotting).

    Returns:
        int: The threshold value set by the user.

    Example:
    
        >>> from signal_quality import qc_prompt
        >>> flobj = FixedLeaderObject()  # Replace with actual FixedLeader object
        >>> qc_prompt(flobj, "Echo Intensity Thresh")
    """
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
