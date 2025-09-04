import logging

import numpy as np

from bird.postprocess.stats import calc_mean

logger = logging.getLogger(__name__)


def main():

    # Sample with 100 points
    t = np.linspace(0, 2, 100)
    signal = np.cos(t * 2 * np.pi)

    mean_val, unc_val = calc_mean(signal, t)
    logger.info(f"Mean = {mean_val:.2g} +/- {unc_val:.2g}")

    # Oversample with 10000 points
    t = np.linspace(0, 2, 10000)
    signal = np.cos(t * 2 * np.pi)
    mean_val, unc_val = calc_mean(signal, t)
    logger.info(f"Mean = {mean_val:.2g} +/- {unc_val:.2g}")


if __name__ == "__main__":
    main()
