import numpy as np
from flow import Bubbles, ZeroDFlow
from prettyPlot.plotting import *
from scipy.special import erf
from utils import diam2vol, vol2diam


def simple_binary_coalescence(bubbles: Bubbles, kwargs) -> None:
    coalescence_rate = kwargs["coalescence_rate"]  # s-1
    dt = kwargs["dt"]
    num_events = int(
        bubbles.coalescence_rate_factor
        * coalescence_rate
        * len(bubbles.diameters)
        * dt
    )
    if num_events < 1:
        bubbles.coalescence_rate_factor += 1
    for _ in range(num_events):
        # Make sure number of bubbles is at least 2
        if len(bubbles.diameters) < 2:
            break
        # Choose two random bubbles to coalesce
        id1, id2 = np.random.choice(len(bubbles.diameters), 2, replace=False)
        bubbles.coalesce(id_in=[id1, id2])


def simple_binary_breakup(bubbles: Bubbles, kwargs) -> None:
    breakup_rate = kwargs["breakup_rate"]  # s-1
    dt = kwargs["dt"]
    num_events = int(
        bubbles.breakup_rate_factor
        * breakup_rate
        * len(bubbles.diameters)
        * dt
    )
    if num_events < 1:
        bubbles.breakup_rate_factor += 1
    for _ in range(num_events):
        # Choose a bubble to break
        id1 = np.random.choice(len(bubbles.diameters))
        vol_in = diam2vol(bubbles.diameters[id1])
        # Break it in two bubbles of identical volume
        new_diameters = np.ones(2) * vol2diam(vol_in / 2)
        bubbles.breakup(id_break=id1, new_diam_list=list(new_diameters))


def resc_T(flow: ZeroDFlow) -> float:
    """
    T rescaling factor from Lehr's
    """
    return np.pow(flow.sigma / flow.rho, 2 / 5) * np.pow(flow.epsilon, -3 / 5)


def resc_L(flow: ZeroDFlow) -> float:
    """
    L rescaling factor from Lehr's
    """
    return np.pow(flow.sigma / flow.rho, 3 / 5) * np.pow(flow.epsilon, -2 / 5)


def lehr_breakup_freq(
    d: float | np.ndarray, flow: ZeroDFlow
) -> float | np.ndarray:
    """
    Returns frequency in s-1
    """
    ds = d / resc_L(flow)
    return 0.5 * np.pow(ds, 5 / 3) * np.exp(-np.sqrt(2) / ds**3) / T(flow)


def lehr_DSD(
    start_diam: float, end_diam: float | np.ndarray, flow: ZeroDFlow
) -> float | np.ndarray:
    """
    Daughter size distribution in Lehr's paper
    """
    ds = start_diam / resc_L(flow)
    dprimes = end_diam / resc_L(flow)
    prefactor = 6 / (np.pow(np.pi, 3 / 2) * dprimes**3)
    exp_term = np.exp(-9 / 4 * np.log(2 ** (2 / 5) * dprimes) ** 2)
    denom_term = 1 + erf(3 / 2 * np.log(2 ** (1 / 15) * ds))
    return prefactor * exp_term / denom_term / resc_L(flow) ** 3


if __name__ == "__main__":
    flow = ZeroDFlow()
    drange = np.linspace(1e-4, 1e-2, 100)
    # freq = breakup_freq(drange,flow)
    # plt.plot(drange, freq)
    # plt.show()

    pdf1 = DSD(1e-3, drange, flow)
    mean_d = np.sum(pdf1 * drange) / np.sum(pdf1)
    print(mean_d)
    plt.plot(drange, pdf1, color="b")
    plt.show()
