from itertools import groupby

import numpy as np
import scipy as sp
import wmm2020


def magnetic_declination(velocity, lat, lon, depth, year):
    """
    Corrects the horizontal velocities based on magnetic declination.

    This function uses the WMM2020 model to obtain the magnetic declination 
    and applies a rotation to the horizontal velocity components to correct 
    for magnetic declination.

    Args:
        velocity (numpy.ndarray): 3D array of velocities with shape (beam, depth, time).
        lat (float): Latitude in decimal degrees.
        lon (float): Longitude in decimal degrees.
        depth (float): Depth in meters.
        year (int): Year of the observation.

    Returns:
        Tuple:
            - numpy.ndarray: Rotated velocity array with shape (beam, depth, time).
            - float: Magnetic declination in degrees.

    Example:
        .. code-block:: python

            import numpy as np
            from velocity_test import magnetic_declination

            # Example velocity array
            velocity = np.random.rand(2, 100, 50) * 100

            # Apply magnetic declination correction
            corrected_velocity, declination = magnetic_declination(velocity, 34.05, -118.25, 500, 2024)

    """
    mag = wmm2020.wmm(lat, lon, depth, year)
    mag = np.deg2rad(mag.decl.data)
    velocity = np.where(velocity == -32768, np.nan, velocity)
    velocity[0, :, :] = velocity[0, :, :] * np.cos(mag) + velocity[1, :, :] * np.sin(
        mag
    )
    velocity[1, :, :] = -1 * velocity[0, :, :] * np.sin(mag) + velocity[
        1, :, :
    ] * np.cos(mag)
    velocity = np.where(velocity == np.nan, -32768, velocity)

    return velocity, np.rad2deg(mag)


def velocity_cutoff(velocity, mask, cutoff=250):
    """
    Masks all velocities above a specified cutoff value.

    This function applies a mask to the velocity data based on a cutoff value, 
    where velocities exceeding the cutoff are masked (set to 1).

    Args:
        velocity (numpy.ndarray): 2D array of velocities with shape (depth, time) in mm/s.
        mask (numpy.ndarray): 2D mask array with the same shape as velocity.
        cutoff (int): Cutoff value in cm/s. Velocities exceeding this value are masked.

    Returns:
        numpy.ndarray: Updated mask array with the same shape as input mask.

    Example:
        .. code-block:: python

            import numpy as np
            from velocity_test import velocity_cutoff

            # Example velocity and mask arrays
            velocity = np.random.rand(50, 100) * 500
            mask = np.zeros_like(velocity)

            # Apply velocity cutoff
            updated_mask = velocity_cutoff(velocity, mask, cutoff=25)
    """
    # Convert to mm/s
    cutoff = cutoff * 10
    mask[np.abs(velocity) > cutoff] = 1
    return mask


def despike(velocity, mask, kernal_size=13, cutoff=150):
    """
    Removes anomalous spikes in the velocity data using a median filter.

    This function applies a median filter to the velocity data to remove spikes. 
    Any value deviating from the filtered value by more than the cutoff is masked.

    Args:
        velocity (numpy.ndarray): 2D array of velocities with shape (depth, time) in mm/s.
        mask (numpy.ndarray): 2D mask array with the same shape as velocity.
        kernel_size (int): Size of the median filter kernel.
        cutoff (int): Threshold for detecting spikes. Values differing from the median by more than this are masked.

    Returns:
        numpy.ndarray: Updated mask array with the same shape as input mask.

    Example:
        .. code-block:: python

            import numpy as np
            from velocity_test import despike

            # Example velocity and mask arrays
            velocity = np.random.rand(50, 100) * 500
            mask = np.zeros_like(velocity)

            # Apply despiking
            cleaned_mask = despike(velocity, mask, kernel_size=15, cutoff=30)
    """
    cutoff = cutoff * 10
    velocity = np.where(velocity == -32768, np.nan, velocity)
    shape = np.shape(velocity)
    for j in range(shape[0]):
        filt = sp.signal.medfilt(velocity[j, :], kernal_size)
        diff = np.abs(velocity[j, :] - filt)
        mask[j, :] = np.where(diff < cutoff, mask[j, :], 1)
    return mask


def flatline(
    velocity,
    mask,
    kernal_size=4,
    cutoff=1,
):
    """
    Identifies and removes velocities that are constant over a period of time.

    This function checks for constant values in the velocity data over a specified 
    period and masks these constant values if they deviate less than the cutoff.

    Args:
        velocity (numpy.ndarray): 2D array of velocities with shape (depth, time).
        mask (numpy.ndarray): 2D mask array with the same shape as velocity.
        kernel_size (int): Number of time steps to check for constant values.
        cutoff (int): Permitted deviation in velocity to determine if a value is constant.

    Returns:
        numpy.ndarray: Updated mask array with the same shape as input mask.

    Example:
        .. code-block:: python

            import numpy as np
            from velocity_test import flatline

            # Example velocity and mask arrays
            velocity = np.random.rand(50, 100) * 500
            mask = np.zeros_like(velocity)

            # Apply flatline removal
            corrected_mask = flatline(velocity, mask, kernel_size=5, cutoff=2)
    """
    index = 0
    velocity = np.where(velocity == -32768, np.nan, velocity)
    shape = np.shape(velocity)
    dummymask = np.zeros(shape[1])
    for j in range(shape[0]):
        diff = np.diff(velocity[j, :])
        diff = np.insert(diff, 0, 0)
        dummymask[np.abs(diff) <= cutoff] = 1
        for k, g in groupby(dummymask):
            # subset_size = sum(1 for i in g)
            subset_size = len(list(g))
            if k == 1 and subset_size >= kernal_size:
                mask[j, index : index + subset_size] = 1
            index = index + subset_size
        dummymask = np.zeros(shape[1])
        index = 0

    return mask
