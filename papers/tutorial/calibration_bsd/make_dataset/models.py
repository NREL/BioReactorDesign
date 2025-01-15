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


