import numpy as np


def diam2vol(diam: float | np.ndarray) -> float | np.ndarray:
    """
    Compute bubble volume from diameter
    """
    # Make sure diameters are positives
    if isinstance(diam, np.ndarray):
        assert not np.any(diam, where=diam < 0)
    if isinstance(diam, float):
        assert diam > 0
    return (4 / 3) * np.pi * (diam / 2) ** 3


def vol2diam(vol: float | np.ndarray) -> float | np.ndarray:
    """
    Compute bubble diameter from volume
    """
    # Make sure volumes are positives
    if isinstance(vol, np.ndarray):
        assert not np.any(vol, where=vol < 0)
    if isinstance(vol, float):
        assert vol >= 0
    return 2 * np.pow(vol * (3 / 4) / np.pi, 1 / 3)


def check_vol_cons(diam_in: list[float], diam_out: list[float]) -> None:
    """
    Check that bubble volume is conserved
    """
    # Compute volume before bubble interaction
    vol_in = 0
    for diam in diam_in:
        vol_in += diam2vol(diam)
    # Compute volume after bubble interaction
    vol_out = 0
    for diam in diam_out:
        vol_out += diam2vol(diam)

    # Make sure volume mismatch is small
    if abs(vol_out - vol_in) > vol_in * 0.01:
        raise ValueError(
            f"Volume conservation error during coalescence of {diam_in} into {diam_out}"
        )


def check_conv(
    values: list[float] | np.ndarray,
    window_ave: int = 5,
    mean_threshold: float = 1e-6,
    std_threshold: float = 1e-6,
) -> bool:
    """
    Check if the input list of values has reached a statistically stationary state.
    """
    if isinstance(values, list):
        values = np.array(values)

    if len(values) < 2 * window_ave:
        return False

    # Compute window averages
    window_avgs = (
        np.convolve(np.array(values), np.ones(window_ave), "valid")
        / window_ave
    )

    # Compare means and standard deviation of the last two window averages
    mean_diff = abs(
        np.mean(window_avgs[-2:]) - np.mean(window_avgs[-window_ave:])
    )
    std_diff = abs(
        np.std(window_avgs[-2:]) - np.std(window_avgs[-window_ave:])
    )
    # print(window_avgs)
    return mean_diff < mean_threshold and std_diff < std_threshold


def get_bsd(x_pdf: np.ndarray, field: np.ndarray) -> np.ndarray:
    """
    Compute bubble size distribution from list of bubble size
    """

    n_x = len(x_pdf)
    weight = np.zeros(len(x_pdf))
    asum = np.zeros(len(x_pdf))
    bsum = np.zeros(len(x_pdf))
    inds = np.digitize(field, x_pdf)
    ind0 = np.argwhere(inds == 0)
    indlx = np.argwhere(inds == len(x_pdf))

    allind = np.array(range(len(inds)))
    safeind = np.setdiff1d(allind, np.union1d(ind0, indlx))
    indb0 = np.argwhere(field == x_pdf[0])
    safeind0 = np.intersect1d(indb0, ind0)
    indblx = np.argwhere(field == x_pdf[-1])
    safeindlx = np.intersect1d(indblx, indlx)

    a = abs(field[safeind] - x_pdf[inds[safeind] - 1])
    b = abs(field[safeind] - x_pdf[inds[safeind]])
    c = a + b
    a = a / c
    b = b / c
    for i in range(len(x_pdf)):
        asum[i] = np.sum(a[np.argwhere(inds[safeind] == i)])
        bsum[i] = np.sum(b[np.argwhere(inds[safeind] == i + 1)])
    weight = weight + asum + bsum
    weight[0] = weight[0] + len(safeind0)
    weight[-1] = weight[-1] + len(safeindlx)

    return weight / np.sum(weight)
