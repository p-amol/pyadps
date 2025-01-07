import numpy as np


def sound_speed_correction(
    velocity: np.ndarray,
    sound_speed: np.ndarray,
    temperature: np.ndarray,
    salinity: np.ndarray,
    depth: np.ndarray,
    horizontal: bool = True,
) -> np.ndarray:
    """
    Corrects velocity measurements for variations in sound speed.

    The function calculates the corrected sound speed based on temperature,
    salinity, and depth using empirical equations. It then adjusts the velocity
    measurements for the u and v components using the ratio of the original
    and corrected sound speeds, while leaving the w component unchanged by default.

    Parameter:
    ----------
        velocity (numpy.ndarray): 4D array of velocity measurements in m/s with
            components (u, v, w, error) along depth and time axes. Missing values
            should be represented by -32768.
        sound_speed (numpy.ndarray): 1D array of measured sound speed in m/s as a function of time.
        temperature (numpy.ndarray): 1D array of temperature in degrees Celsius as a function of time.
        salinity (numpy.ndarray): 1D array of salinity in PSU (Practical Salinity Units) as a function of time.
        depth (numpy.ndarray): 1D array of transducer depth in meters as a function of time.
        horizontal (bool): By default only horizontal velocities are corrected.

    Returns:
    --------
        numpy.ndarray: 3D array of corrected velocity measurements as 32-bit integers.
        The w component remains unchanged. Missing values (-32768) remain unchanged.

    Notes:
    ------
        The sound speed correction formula is derived empirically using the
        equation (Urick, 1983).
    """
    # Calculate corrected sound speed
    sound_speed_corrected = (
        1449.2
        + 4.6 * temperature
        - 0.055 * temperature**2
        + 0.00029 * temperature**3
        + (1.34 - 0.01 * temperature) * (salinity - 35)
        + 0.016 * depth
    )

    sound_speed = sound_speed
    sound_speed_corrected = sound_speed_corrected
    # Separate u, v, and w components
    u = velocity[0, :, :]
    v = velocity[1, :, :]
    w = velocity[2, :, :]
    e = velocity[3, :, :]

    # Correct u and v components
    u_corrected = np.where(
        u == -32768,
        -32768.0,
        u * sound_speed[np.newaxis, :] / sound_speed_corrected[np.newaxis, :],
    )

    v_corrected = np.where(
        v == -32768,
        -32768.0,
        v * sound_speed[np.newaxis, :] / sound_speed_corrected[np.newaxis, :],
    )

    if not horizontal:
        w_corrected = np.where(
            w == -32768,
            -32768.0,
            w * sound_speed[np.newaxis, :] / sound_speed_corrected[np.newaxis, :],
        )
    else:
        w_corrected = w

    # Combine corrected components back into a 4D array
    velocity_corrected = np.stack(
        [
            u_corrected.astype(np.int32),
            v_corrected.astype(np.int32),
            w_corrected.astype(np.int32),
            e.astype(np.int32),
        ],
        axis=0,
    )

    return velocity_corrected


def tilt_sensor_check(
    tilt: np.ndarray, mask: np.ndarray, cutoff: int = 15
) -> np.ndarray:
    """
    Updates the given 2D mask array based on the tilt sensor readings. If the tilt value in
    the 1D tilt array exceeds the specified cutoff, the corresponding values in the mask are
    set to 1.

    Parameters
    ----------
    tilt : np.ndarray
        A 1D array of tilt sensor readings.
    mask : np.ndarray
        A 2D array where the tilt values are checked against the cutoff.
    cutoff : int, optional
        The tilt value threshold. Default is 15. If a tilt value exceeds this threshold,
        the corresponding mask value is updated to 1.

    Returns
    -------
    np.ndarray
        A 2D array with updated mask values where tilt exceeds the cutoff.
    """
    tilt = tilt * 0.01
    updated_mask = np.where(tilt[:, np.newaxis] > cutoff, 1, mask.T)
    return updated_mask.T
