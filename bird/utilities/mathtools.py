import logging
import sys

import numpy as np

logger = logging.getLogger(__name__)


def conditional_average(
    x: np.ndarray, y: np.ndarray, nbins: int = 32
) -> tuple:
    """
    Compute a 1D conditional average of y with respect to x
    The conditional average is distributed to neighbors of the binned array when needed

    Parameters
    ----------
    x: np.ndarray
        1D array with respect to which conditional averaged is performed
    y : np.ndarray
        1D array conditioned
    nbins: int
        Number of bins through x

    Returns
    ----------
    x_cond: np.ndarray
        The binned array of values conditioned againsts
    y_cond: np.ndarray
        The conditional averages at each bin
    """
    # Check the shape of input arrays
    try:
        assert len(x.shape) <= 2
        assert len(y.shape) <= 2
        if len(x.shape) == 2:
            assert x.shape[1] == 1
        if len(y.shape) == 2:
            assert y.shape[1] == 1
    except AssertionError:
        error_msg = "conditional average of tensors is ambiguous"
        error_msg += f"\nx shape =  {x.shape}"
        error_msg += f"\ny shape =  {y.shape}"
        logger.error(error_msg)
        sys.exit()
    if len(x.shape) == 2:
        x = x[:, 0]
    if len(y.shape) == 2:
        y = y[:, 0]
    try:
        assert len(x) == len(y)
    except AssertionError:
        error_msg = "conditional average x and y have different dimension"
        error_msg += f"\ndim x =  {len(x)}"
        error_msg += f"\ndim y =  {len(y)}"
        logger.error(error_msg)
        sys.exit()

    # Bin conditional space
    mag = np.amax(x) - np.amin(x)
    x_bin = np.linspace(
        np.amin(x) - mag / (2 * nbins), np.amax(x) + mag / (2 * nbins), nbins
    )
    weight = np.zeros(nbins)
    weightVal = np.zeros(nbins)
    asum = np.zeros(nbins)
    bsum = np.zeros(nbins)
    avalsum = np.zeros(nbins)
    bvalsum = np.zeros(nbins)
    inds = np.digitize(x, x_bin)

    a = abs(y - x_bin[inds - 1])
    b = abs(y - x_bin[inds])
    c = a + b
    a = a / c
    b = b / c

    # Conditional average at each bin
    for i in range(nbins):
        asum[i] = np.sum(a[np.argwhere(inds == i)])
        bsum[i] = np.sum(b[np.argwhere(inds == i + 1)])
        avalsum[i] = np.sum(
            a[np.argwhere(inds == i)] * y[np.argwhere(inds == i)]
        )
        bvalsum[i] = np.sum(
            b[np.argwhere(inds == i + 1)] * y[np.argwhere(inds == i + 1)]
        )
    weight = asum + bsum
    weightVal = avalsum + bvalsum

    # Assemble output
    x_cond = x_bin
    y_cond = weightVal / (weight)

    return x_cond, y_cond
