import numpy as np
from prettyPlot.plotting import *
from scipy.special import erf
from utils import check_vol_cons, diam2vol, vol2diam


class ZeroDFlow:
    def __init__(
        self,
        epsilon: float = 2,
        alpha: float = 0.25,
        U: float = 0,
        rho: float = 998,
        sigma: float = 0.0727,
    ):
        self.epsilon = epsilon  # turblent energy dissipation rate [m2 s-3]
        self.alpha = alpha  # Volume fraction of bubbles [-]
        self.U = U  # mean vel
        self.rho = rho  # liquid density [kg m-3]
        self.sigma = sigma  # surface tension [kg s-2]


class Bubbles:
    def __init__(
        self,
        diameters: np.ndarray | None = None,
        nbubbles: int | None = None,
        diam: float | None = None,
    ):
        if diameters is not None:
            self.diameters = diameters
        else:
            assert nbubbles is not None
            assert diam is not None
            self.diameters = np.ones(nbubbles) * diam
        self.mean_diameter = None
        self.update_mean()
        self.breakup_rate_factor = 1.0
        self.coalescence_rate_factor = 1.0

    def update_mean(self):
        self.mean_diameter = np.mean(self.diameters)

    def coalesce(self, id_in: list[int], new_diameter: float = None) -> None:

        check_vol_cons(
            diam_in=[self.diameters[id_b] for id_b in id_in],
            diam_out=[new_diameter],
        )

        # Update bubble diameters
        self.diameters = np.delete(self.diameters, [id_in])
        self.diameters = np.append(self.diameters, new_diameter)
        # Reset the coalescence rate factor
        self.coalescence_rate_factor = 1.0

    def breakup(self, id_break: int, new_diam_list: list[float]) -> None:
        check_vol_cons([self.diameters[id_break]], new_diam_list)
        # Update bubble diameters
        self.diameters = np.delete(self.diameters, id_break)
        self.diameters = np.append(self.diameters, new_diam_list)
        # Reset the breakup rate factor
        self.breakup_rate_factor = 1.0


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
