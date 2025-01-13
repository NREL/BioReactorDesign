import numpy as np


def diam2vol(diam: float | np.ndarray) -> float | np.ndarray:
    # Make sure diameters are positives
    if isinstance(diam, np.ndarray):
        assert not np.any(diam, where=diam < 0)
    if isinstance(diam, float):
        assert diam > 0
    return (4 / 3) * np.pi * (diam / 2) ** 3


def vol2diam(vol: float | np.ndarray) -> float | np.ndarray:
    # Make sure volumes are positives
    if isinstance(vol, np.ndarray):
        assert not np.any(vol, where=vol < 0)
    if isinstance(vol, float):
        assert vol > 0
    return 2 * np.pow(vol * (3 / 4) / np.pi, 1 / 3)


def check_vol_cons(diam_in: list[float], diam_out: list[float]) -> None:
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
