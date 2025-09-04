import logging
import os

import numpy as np
import pytest

from bird.postprocess.stats import calc_mean

logger = logging.getLogger(__name__)


def test_mean_stats_uniform():
    """
    Test for time series uniformly sampled over time
    """
    # Make artificial signal
    t = np.linspace(0, 2, 100)
    signal = np.cos(t * 2 * np.pi)

    mean_val, unc_val = calc_mean(signal)
    logger.info(f"Signal mean = {mean_val:.2g}+/-{unc_val:.2g}")

    mean_val2, unc_val2 = calc_mean(signal, t)
    logger.info(f"Signal mean and t = {mean_val2:.2g}+/-{unc_val2:.2g}")

    # Make sure equally spaced signal is recognized
    assert mean_val2 == mean_val
    assert unc_val2 == unc_val

    # Make artificial oversampled signal
    t = np.linspace(0, 2, 10000)
    signal = np.cos(t * 2 * np.pi)

    mean_val_oversampled, unc_val_oversampled = calc_mean(signal)

    logger.info(
        f"Signal mean oversampled = {mean_val_oversampled:.2g}+/-{unc_val_oversampled:.2g}"
    )

    # Make sure oversampling the signal does not artificially reduce uncertainty
    assert abs(unc_val_oversampled - unc_val) / unc_val < 0.1


def test_mean_stats_nonuniform():
    """
    Test for time series non-uniformly sampled over time
    """
    t = np.linspace(0, 2, 100)
    signal = np.cos(t * 2 * np.pi)

    mean_unif, unc_unif = calc_mean(signal)
    logger.info(f"Signal mean unif = {mean_unif:.2g}+/-{unc_unif:.2g}")

    pert_t = np.random.uniform(0, 1 / 100, 100)
    t = t + pert_t
    signal = np.cos(t * 2 * np.pi)
    mean_non_unif, unc_non_unif = calc_mean(signal, t)
    logger.info(
        f"Signal mean non-unif = {mean_non_unif:.2g}+/-{unc_non_unif:.2g}"
    )

    # Make sure non uniform signal is correctly treated
    assert abs(unc_non_unif - unc_unif) / unc_unif < 0.1


if __name__ == "__main__":
    test_mean_stats_uniform()
    test_mean_stats_nonuniform()
