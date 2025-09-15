import sys

import numpy as np

from bird import logger


def _check1D(signal: np.ndarray):
    """
    Check that array is 1D

    Parameters
    ----------
    signal: np.ndarray
        Signal data whose shape is checked
    """
    try:
        assert signal is not None
    except AssertionError:
        logger.error("Signal expected to be 1D but is None")
        raise ValueError
    try:
        assert np.sum(np.array(signal.shape) > 1) == 1
        signal = signal.flatten()
    except AssertionError:
        logger.error(f"Signal expected to be 1D, got shape {signal.shape}")
        raise ValueError


def _autocorr1D(time_series: np.ndarray) -> np.ndarray:
    """
    Compute normalized autocorrelation coefficient over different lag values

    Parameters
    ----------
    time_series: np.ndarray
        Time series of the signal to autocorrelated

    Returns
    ----------
    autocorr : np.ndarray
        Autocorrelation coefficient array over multiple lags
    """

    _check1D(time_series)
    time_series = time_series.flatten()
    time_series = time_series - np.mean(time_series)
    autocorr = np.correlate(time_series, time_series, mode="full")
    # Only keep the positive lags
    autocorr = autocorr[autocorr.size // 2 :]
    # Normalize
    autocorr /= autocorr[0]

    return autocorr


def _T0_fn(autocorr: np.ndarray) -> float:
    """
    Compute time between independent samples
    Following Trenberth, "Some Effects of Finite Sample Size and Persistence on Meteorological Statistics. Part I: Autocorrelations", 1984
    And Oliver et al., "Estimating uncertainties in statistics computed from direct numerical simulation", 2014

    Parameters
    ----------
    autocorr: np.ndarray
        Autocorrelation coefficient for different lag values

    Returns
    ----------
    T0 : float
        Time (in discretization step) for which samples can be considered independent.
    """
    _check1D(autocorr)
    autocorr = autocorr.flatten()

    N = len(autocorr)
    k = np.array(range(1, N))
    T0 = 1.0 + 2.0 * np.sum((1.0 - k / N) * autocorr[1:])

    # Check value of T0
    if T0 < 1.0 or T0 > N / 2:
        logger.warning(
            f"T0 value ({T0}) is suspicious, falling back to T0 = 1"
        )
        T0 = 1.0

    logger.debug(f"T0 = {T0}")

    return T0


def _to_equally_spaced(
    time_series: np.ndarray, time_values: np.ndarray | None
):
    """
    Transform to equally spaced time series

    Parameters
    ----------
    time_series: np.ndarray
        Time series of the signal
    time_values: np.ndarray|None
        The time values over which the time series is sampled.

    Returns
    ----------
    new_time_series, new_time_values: tuple
        Equally spaced version of time_series and time_values

    """
    if time_values is None:
        logger.debug(
            "Assuming time series sampled which equal spacing over time"
        )
        return time_series, time_values

    logger.debug("Making the time series equally spaced over time")
    _check1D(time_values)
    time_values = time_values.flatten()
    _check1D(time_series)
    time_series = time_series.flatten()

    min_time = time_values.min()
    max_time = time_values.max()
    dt = np.diff(time_values)
    min_diff = dt.min()
    max_diff = dt.max()
    try:
        assert min_diff > 0
    except AssertionError:
        logger.error(f"Time values are not ordered ({time_values})")
        raise ValueError

    # If already equally spaced, don't touch it
    if (dt.max() - dt.min()) < 1e-12 * (time_values.max() - time_values.min()):
        logger.debug("Time series already equally spaced")
        return time_series, time_values

    # Subsample, we could also oversample by dividing by min_diff
    n_val = int((max_time - min_time) / max_diff)
    new_time_values = np.linspace(min_time, max_time, n_val + 1)
    new_time_series = np.interp(new_time_values, time_values, time_series)
    logger.warning(
        f"New time series built with {n_val} pts instead of {len(time_series)} pts"
    )

    return new_time_series, new_time_values


def calc_mean(
    time_series: np.ndarray, time_values: np.ndarray | None = None
) -> tuple[float, float]:
    """
    Compute mean and the uncertainty about the mean, from a time-series

    Parameters
    ----------
    time_series: np.ndarray
        Time series of the signal
    time_values: np.ndarray | None
        The time values over which the time series is sampled.
        If None, the time values are assumed equally spaced.
        Otherwise, time_values is used to create a new equally spaced time_values

    Returns
    ----------
    mean_val: float
        Mean value of the time_series
    unc_val: float
        68% uncertainty (1 sigma) about the mean

    """

    time_series, time_values = _to_equally_spaced(time_series, time_values)

    autocorr = _autocorr1D(time_series)
    N = len(time_series)
    T0 = _T0_fn(autocorr)

    mean_val = np.mean(time_series)
    sigsq = np.var(time_series) * N / (N - T0)
    unc_val = np.sqrt(sigsq * T0 / N)

    return mean_val, unc_val
