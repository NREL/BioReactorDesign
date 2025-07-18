import logging

import numpy as np

logger = logging.getLogger(__name__)


def conditionalAverage(x, y, nbin):
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
        np.amin(x) - mag / (2 * nbin), np.amax(x) + mag / (2 * nbin), nbin
    )
    weight = np.zeros(nbin)
    weightVal = np.zeros(nbin)
    asum = np.zeros(nbin)
    bsum = np.zeros(nbin)
    avalsum = np.zeros(nbin)
    bvalsum = np.zeros(nbin)
    inds = np.digitize(x, x_bin)

    a = abs(y - x_bin[inds - 1])
    b = abs(y - x_bin[inds])
    c = a + b
    a = a / c
    b = b / c

    for i in range(nbin):
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

    return x_bin, weightVal / (weight)
