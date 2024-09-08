from itertools import groupby

import numpy as np
import scipy as sp
import wmm2020


def magnetic_declination(velocity, lat, lon, depth, year):
    """
    The function uses magnetic declination from wmm2020 to correct
    the horizontal velocities

    Args:
        velocity (numpy array): velocity(beam, depth, time)
        lat (parameter, float): Latitude in decimals
        lon (parameter, float): Longitude in decimals
        depth (parameter, float): depth in m
        year (parameter, integer): Year

    Returns:
        velocity (numpy array): Rotated velocity using magnetic declination
        mag: Magnetic declination (degrees)
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
    Masks all velocities above a cutoff. Note that
    velocity is a 2-D array.

    Args:
        velocity (numpy array, integer): Velocity(depth, time) in mm/s
        mask (numpy array, integer): Mask file
        cutoff (parameter, integer): Cutoff in cm/s

    Returns:
        mask
    """
    # Convert to mm/s
    cutoff = cutoff * 10
    mask[np.abs(velocity) > cutoff] = 1
    return mask


def despike(velocity, mask, kernal_size=13, cutoff=150):
    """
    Function to remove anomalous spikes in the data over a period of time.
    A median filter is used to despike the data.

    Args:
        velocity (numpy array, integer): Velocity(depth, time) in mm/s
        mask (numpy array, integer): Mask file
        kernal_size (paramater, integer): Number of ensembles over which the spike has to be checked
        cutoff (parameter, integer): [TODO:description]

    Returns:
        mask
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
    Function to check and remove velocities that are constant over a
    period of time.

    Args:
        velocity (numpy arrray, integer): Velocity (depth, time)
        mask (numpy  array, integer): Mask file
        kernal_size (parameter, integer): No. of ensembles over which flatline has to be detected
        cutoff (parameter, integer): Permitted deviation in velocity

    Returns:
        mask
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
