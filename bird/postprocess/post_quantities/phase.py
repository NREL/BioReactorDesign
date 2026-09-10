from bird.utilities.ofio import (
    read_bubble_diameter,
    read_cell_volumes,
    read_field,
)

from ._cell_filter import _field_filter, _get_ind_liq, _weighted_average


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

    :param gas_holdup: gas holdup :math:`\epsilon` [-]
    :param bubble_diam: bubble diameter :math:`d` [m]
    :return: interfacial area :math:`a` [1/m]
    """
    return 6.0 * gas_holdup / bubble_diam
