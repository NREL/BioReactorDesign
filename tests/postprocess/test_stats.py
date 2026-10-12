import os

import numpy as np
import pytest

from bird import logger
from bird.postprocess.stats import calc_mean, steady_stat


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


def test_steady_stat():
    """
    Test the windowed (mean, 1-sigma) steady-state statistic
    """
    t = np.linspace(0, 10, 1000)
    signal = np.cos(t * 2 * np.pi)

    # explicit window: stat is calc_mean over the last 2 s, 95% -> 1-sigma
    mask = t >= t.max() - 2.0
    mean_ref, unc95_ref = calc_mean(signal[mask], t[mask])
    mean_val, sigma = steady_stat(signal, t, window=2.0)
    assert mean_val == pytest.approx(mean_ref)
    assert sigma == pytest.approx(unc95_ref / 1.96)

    # default window = 10% of the span (last 1 s here)
    default_mask = t >= t.max() - 0.1 * (t.max() - t.min())
    mean_def_ref, _ = calc_mean(signal[default_mask], t[default_mask])
    mean_def, _ = steady_stat(signal, t)
    assert mean_def == pytest.approx(mean_def_ref)

    # empty and all-nan series -> (nan, nan)
    mean_empty, sigma_empty = steady_stat([], [])
    assert np.isnan(mean_empty) and np.isnan(sigma_empty)
    mean_nan, sigma_nan = steady_stat([np.nan, np.nan], [0.0, 1.0])
    assert np.isnan(mean_nan) and np.isnan(sigma_nan)

    # a single sample in the window -> (value, 0.0)
    mean_one, sigma_one = steady_stat([1.0, 3.5], [0.0, 1.0], window=0.5)
    assert mean_one == 3.5 and sigma_one == 0.0
