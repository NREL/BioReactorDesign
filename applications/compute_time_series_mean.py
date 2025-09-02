import logging

import numpy as np
from prettyPlot.plotting import plt, pretty_labels

from bird.postprocess.stats import calc_mean

logger = logging.getLogger(__name__)


def main():

    t = np.linspace(0, 2, 100)
    signal = np.cos(t * 2 * np.pi)

    mean_val, unc_val = calc_mean(signal, t)
    logger.info(f"Mean = {mean_val:.2g} +/- {unc_val:.2g}")

    t = np.linspace(0, 2, 10000)
    signal = np.cos(t * 2 * np.pi)
    mean_val, unc_val = calc_mean(signal, t)
    logger.info(f"Mean = {mean_val:.2g} +/- {unc_val:.2g}")


if __name__ == "__main__":
    main()
