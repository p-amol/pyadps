import numpy as np
from pyadps.utils.plotgen import PlotNoise


def qc_check(var, mask, cutoff=0):
    """
    The module returns the modified mask file after checking the cutoff criteria.
    All values less than the cuttoff are masked.

    Args:
        var (numpy.ndarray):
        mask (numpy.ndarray): A mask file having same array size as var
        cutoff (int): Default cutoff is 0

    Returns:
        mask (numpy.ndarray): Modified mask file based on cutoff
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
    if threebeam:
        pgood1 = pgood[0, :, :] + pgood[3, :, :]
    else:
        pgood1 = pgood[:, :, :]

    mask[pgood1[:, :] < cutoff] = 1
    values, counts = np.unique(mask, return_counts=True)
    # print(values, counts, np.round(counts[1] * 100 / np.sum(counts)))
    return mask


def false_target(echo, mask, cutoff=255, threebeam=True):
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
    cells = flobj.field()["Cells"]
    beams = flobj.field()["Beams"]
    ensembles = flobj.ensembles
    mask = np.zeros((cells, ensembles))
    # Ignore mask for error velocity
    for i in range(beams - 1):
        mask[velocity[i, :, :] < -32767] = 1
    return mask


def qc_prompt(flobj, name, data=None):
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
