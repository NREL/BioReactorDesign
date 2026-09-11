import os

import numpy as np

from bird import logger
from bird.utilities.ofio import (
    read_bubble_diameter,
    read_cell_volumes,
    read_field,
    read_global_vars,
)

from ._cell_filter import _field_filter, _get_ind_liq, _weighted_average


def _read_liquid_density_field(
    case_folder: str,
    time_folder: str,
    n_cells: int | None,
    field_dict: dict,
) -> tuple[np.ndarray | float, dict]:
    """Liquid density field: thermo:rho.liquid, then rho.liquid; if neither is
    written, globalVars rho0MixLiq (then 1000)."""
    for rho_name in ("thermo:rho.liquid", "rho.liquid"):
        try:
            rho_liquid, field_dict = read_field(
                case_folder, time_folder, rho_name, n_cells, field_dict
            )
            return rho_liquid, field_dict
        except FileNotFoundError:
            continue
    try:
        rho0 = float(read_global_vars(case_folder).get("rho0MixLiq", 1000.0))
    except FileNotFoundError:
        rho0 = 1000.0
    logger.warning(
        f"No liquid density field in "
        f"{os.path.join(case_folder, time_folder)}, assuming {rho0} kg/m3"
    )
    return rho0, field_dict


def compute_gas_holdup(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    volume_time: str | None = None,
    field_dict: dict | None = None,
) -> tuple[float, dict]:
    r"""
    Calculate volume averaged gas hold up at a given time

    .. math::
       \frac{1}{V_{\rm liq, tot}} \int_{V_{\rm liq}} (1-\alpha_{\rm liq}) dV

    where:
      - :math:`V_{\rm liq, tot}` is the total volume of liquid in :math:`m^3`
      - :math:`\alpha_{\rm liq}` is the liquid phase volume fraction
      - :math:`V` is the volume of the cells where :math:`\alpha_{\rm liq}` is measured in :math:`m^3`

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    n_cells : int | None
        Number of cells in the domain.
        If None, it will deduced from the field reading
    volume_time : str | None
        Time folder to read to get the cell volumes.
        If None, finds volume time automatically
    field_dict : dict | None
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    gas_holdup: float
        Volume averaged gas holdup
    field_dict : dict
        Dictionary of fields read
    """

    if field_dict is None:
        field_dict = {}

    # Read relevant fields
    kwargs = {
        "case_folder": case_folder,
        "time_folder": time_folder,
        "n_cells": n_cells,
    }

    kwargs_vol = {
        "case_folder": case_folder,
        "time_folder": volume_time,
        "n_cells": n_cells,
    }

    alpha_liq, field_dict = read_field(
        field_name="alpha.liquid", field_dict=field_dict, **kwargs
    )
    ind_liq, field_dict = _get_ind_liq(field_dict=field_dict, **kwargs)
    cell_volume, field_dict = read_cell_volumes(
        field_dict=field_dict, **kwargs_vol
    )

    # Only compute over the pure liquid
    alpha_liq = _field_filter(alpha_liq, ind=ind_liq, field_type="scalar")
    cell_volume = _field_filter(cell_volume, ind=ind_liq, field_type="scalar")

    # Calculate
    gas_holdup = _weighted_average(1 - alpha_liq, cell_volume)

    return gas_holdup, field_dict


def compute_ave_bubble_diam(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    volume_time: str | None = None,
    field_dict: dict | None = None,
) -> tuple[float, dict]:
    r"""
    Calculate averaged bubble diameter over the liquid volume

    .. math::

       \frac{1}{V_{\rm liq, tot}} \int_{V_{\rm liq}} d_{\rm gas} dV

    where:
      - :math:`V_{\rm liq, tot}` is the toal volume of liquid in :math:`m^3`
      - :math:`d_{\rm gas}` is the bubble diameter in :math:`m`
      - :math:`V_{\rm liq}` is the volume of liquid where :math:`d_{\rm gas}` is measured in :math:`m^3`


    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    n_cells : int | None
        Number of cells in the domain.
        If None, it will deduced from the field reading
    volume_time : str | None
        Time folder to read to get the cell volumes.
        If None, finds volume time automatically
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    diam: float
        Volume averaged gas holdup
    field_dict : dict
        Dictionary of fields read
    """
    if field_dict is None:
        field_dict = {}

    # Read relevant fields
    kwargs = {
        "case_folder": case_folder,
        "time_folder": time_folder,
        "n_cells": n_cells,
    }
    kwargs_vol = {
        "case_folder": case_folder,
        "time_folder": volume_time,
        "n_cells": n_cells,
    }
    alpha_liq, field_dict = read_field(
        field_name="alpha.liquid", field_dict=field_dict, **kwargs
    )
    d_gas, field_dict = read_bubble_diameter(field_dict=field_dict, **kwargs)
    ind_liq, field_dict = _get_ind_liq(field_dict=field_dict, **kwargs)

    cell_volume, field_dict = read_cell_volumes(
        field_dict=field_dict, **kwargs_vol
    )

    # Only compute over the liquid
    alpha_liq = _field_filter(alpha_liq, ind=ind_liq, field_type="scalar")
    cell_volume = _field_filter(cell_volume, ind=ind_liq, field_type="scalar")
    d_gas = _field_filter(d_gas, ind=ind_liq, field_type="scalar")

    # Calculate
    diam = _weighted_average(d_gas, alpha_liq * cell_volume)

    return diam, field_dict


def interfacial_area(gas_holdup: float, bubble_diam: float) -> float:
    r"""Gas-liquid interfacial area per unit volume :math:`a = 6\,\epsilon / d`.

    Parameters
    ----------
    gas_holdup: float
        Gas holdup :math:`\epsilon` (dimensionless)
    bubble_diam: float
        Bubble diameter :math:`d`, in :math:`m`

    Returns
    ----------
    interfacial_area: float
        Interfacial area :math:`a`, in :math:`m^{-1}`
    """
    return 6.0 * gas_holdup / bubble_diam


def compute_ave_liquid_density(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    volume_time: str | None = None,
    field_dict: dict | None = None,
) -> tuple[float, dict]:
    r"""Volume-averaged liquid density over the liquid.

    Reads ``thermo:rho.liquid`` (then ``rho.liquid``); returns 1000 kg/m3 if
    neither field is written.

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of the time folder to analyze
    n_cells : int | None
        Number of cells in the domain.
        If None, it will deduced from the field reading
    volume_time : str | None
        Time folder to read to get the cell volumes.
        If None, finds volume time automatically
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    density: float
        Volume averaged liquid density, in :math:`kg.m^{-3}`
    field_dict : dict
        Dictionary of fields read
    """
    if field_dict is None:
        field_dict = {}

    ind_liq, field_dict = _get_ind_liq(
        case_folder, time_folder, n_cells=n_cells, field_dict=field_dict
    )
    rho_liquid, field_dict = _read_liquid_density_field(
        case_folder, time_folder, n_cells, field_dict
    )
    cell_volume, field_dict = read_cell_volumes(
        case_folder, volume_time, n_cells, field_dict
    )
    rho_liquid = _field_filter(rho_liquid, ind=ind_liq, field_type="scalar")
    cell_volume = _field_filter(cell_volume, ind=ind_liq, field_type="scalar")
    return _weighted_average(rho_liquid, cell_volume), field_dict


def compute_ave_liquid_velocity(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    volume_time: str | None = None,
    field_dict: dict | None = None,
) -> tuple[float, dict]:
    r"""Volume-averaged liquid velocity magnitude :math:`|U_{\rm liq}|` over the liquid.

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of the time folder to analyze
    n_cells : int | None
        Number of cells in the domain.
        If None, it will deduced from the field reading
    volume_time : str | None
        Time folder to read to get the cell volumes.
        If None, finds volume time automatically
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    velocity_magnitude: float
        Volume averaged liquid velocity magnitude, in :math:`m.s^{-1}`
    field_dict : dict
        Dictionary of fields read
    """
    if field_dict is None:
        field_dict = {}

    ind_liq, field_dict = _get_ind_liq(
        case_folder, time_folder, n_cells=n_cells, field_dict=field_dict
    )
    u_liquid, field_dict = read_field(
        case_folder, time_folder, "U.liquid", n_cells, field_dict
    )
    cell_volume, field_dict = read_cell_volumes(
        case_folder, volume_time, n_cells, field_dict
    )
    if np.ndim(u_liquid) == 1:  # uniform field, shape (3,)
        u_magnitude = float(np.linalg.norm(u_liquid))
    else:
        u_magnitude = np.linalg.norm(u_liquid, axis=1)
    u_magnitude = _field_filter(u_magnitude, ind=ind_liq, field_type="scalar")
    cell_volume = _field_filter(cell_volume, ind=ind_liq, field_type="scalar")
    return _weighted_average(u_magnitude, cell_volume), field_dict
