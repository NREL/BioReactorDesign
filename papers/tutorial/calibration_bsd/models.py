import numpy as np
from flow import Bubbles, ZeroDFlow
from prettyPlot.plotting import *
from scipy.special import erf
from utils import diam2vol, vol2diam


def simple_nary_coalescence(bubbles: Bubbles, kwargs: dict) -> None:
    """
    Coalescence of n_coal bubbles into a single bubble
    """
    coalescence_rate = kwargs["coalescence_rate"]  # s-1
    dt = kwargs["dt"]  # s
    n_coal = kwargs["n_coal"]
    max_coal_diam = kwargs["max_coal_diam"]
    num_events = int(
        bubbles.coalescence_rate_factor
        * coalescence_rate
        * len(bubbles.diameters)
        * dt
    )
    # If rate is too small to lead to any event,
    # increase the rate at the next timestep
    # to make sure the rate is correct over time
    if num_events < 1:
        bubbles.coalescence_rate_factor += 1
    else:
        bubbles.coalescence_rate_factor = 1.0

    for _ in range(num_events):
        # Make sure number of bubbles is at least n_coal
        if len(bubbles.diameters) < n_coal:
            break
        # Choose n_coal random bubbles to coalesce
        idcoal = np.random.choice(
            len(bubbles.diameters), n_coal, replace=False
        )
        # Compute the volume of the bubbles to coalesce
        vol_in = sum([diam2vol(bubbles.diameters[id_c]) for id_c in idcoal])
        # Compute the diameter of coalesced bubble
        new_diameter = vol2diam(vol_in)
        # Check that we are not creating too big of a bubble
        if new_diameter < max_coal_diam:
            bubbles.coalesce(id_in=idcoal, new_diameter=new_diameter)


def simple_nary_breakup(bubbles: Bubbles, kwargs: dict) -> None:
    """
    Breakup of a single bubble into n_break bubbles
    """
    breakup_rate = kwargs["breakup_rate"]  # s-1
    dt = kwargs["dt"]
    n_break = kwargs["n_break"]
    min_break_diam = kwargs["min_break_diam"]
    num_events = int(
        bubbles.breakup_rate_factor
        * breakup_rate
        * len(bubbles.diameters)
        * dt
    )
    # If rate is too small to lead to any event,
    # increase the rate at the next timestep
    # to make sure the rate is correct over time
    if num_events < 1:
        bubbles.breakup_rate_factor += 1
    else:
        bubbles.breakup_rate_factor = 1.0
    for _ in range(num_events):
        # Choose a bubble to break
        id1 = np.random.choice(len(bubbles.diameters))
        vol_in = diam2vol(bubbles.diameters[id1])
        # Break it in n_break bubbles of identical volume
        new_diameters = np.ones(n_break) * vol2diam(vol_in / n_break)
        # Check that we are not creating too small of bubbles
        if np.amin(new_diameters) > min_break_diam:
            bubbles.breakup(id_break=id1, new_diam_list=list(new_diameters))


def simple_normal_breakup(bubbles: Bubbles, kwargs: dict) -> None:
    """
    Breakup of a single bubble into bubbles sampled from Daughter size distribution, here a normal distribution
    """
    breakup_rate = kwargs["breakup_rate"]  # s-1
    dt = kwargs["dt"]
    n_break = kwargs["n_break"]
    min_break_diam = kwargs["min_break_diam"]
    mean_break_diam_fact = kwargs["mean_break_diam_fact"]
    std_break_diam_fact = kwargs["std_break_diam_fact"]
    vol_min = diam2vol(min_break_diam)

    num_events = int(
        bubbles.breakup_rate_factor
        * breakup_rate
        * len(bubbles.diameters)
        * dt
    )
    # If rate is too small to lead to any event,
    # increase the rate at the next timestep
    # to make sure the rate is correct over time
    if num_events < 1:
        bubbles.breakup_rate_factor += 1
    else:
        bubbles.breakup_rate_factor = 1.0
    for _ in range(num_events):
        # Choose a bubble to break
        id1 = np.random.choice(len(bubbles.diameters))
        vol_in = diam2vol(bubbles.diameters[id1])
        if vol_in > vol_min:
            # Break it in bubbles distributed as a normal dist
            diam_candidates = np.clip(
                np.random.normal(
                    bubbles.diameters[id1] * mean_break_diam_fact,
                    bubbles.diameters[id1] * std_break_diam_fact,
                    size=20,
                ),
                a_min=min_break_diam,
                a_max=vol2diam(vol_in - vol_min),
            )
            vol_out_all = diam2vol(diam_candidates)
            cumultative_vol = np.cumsum(vol_out_all)

            # find how many bubbles need to be created to satisfy volume conservation
            ind_lim = np.argwhere(cumultative_vol > (vol_in - vol_min))
            if abs(ind_lim[0][0]) < 1e-12:
                ind_lim[0][0] = 1
            # Add one last bubble to satisfy volume conservation
            diam_final = vol2diam(vol_in - cumultative_vol[ind_lim[0][0] - 1])
            new_diameters = np.hstack(
                (diam_candidates[: ind_lim[0][0]], diam_final)
            )
            # print(f"breakup in {len(new_diameters)} daughter bubbles")
            if np.amin(new_diameters) > min_break_diam:
                bubbles.breakup(
                    id_break=id1, new_diam_list=list(new_diameters)
                )


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
    from scipy.integrate import simpson

    flow = ZeroDFlow()
    drange = np.linspace(1e-4, 1e-2, 100)
    # freq = breakup_freq(drange,flow)
    # plt.plot(drange, freq)
    # plt.show()

    pdf1 = lehr_DSD(1e-3, drange, flow)
    # mean_d = np.sum(pdf1 * drange) / np.sum(pdf1)
    mean_d = simpson(pdf1 * drange, drange) / simpson(pdf1, drange)
    print(mean_d)
    plt.plot(drange, pdf1, color="b")
    plt.show()
