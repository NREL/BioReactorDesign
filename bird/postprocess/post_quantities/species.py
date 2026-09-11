from bird import logger
from bird.utilities.ofio import (
    read_cell_volumes,
    read_field,
    species_name_to_mw,
)

from ._cell_filter import _field_filter, _get_ind_liq, _weighted_average
from .phase import _read_liquid_density_field


def compute_ave_y_liq(
    case_folder: str,
    time_folder: str,
    species_name: str = "CO2",
    n_cells: int | None = None,
    volume_time: str | None = None,
    field_dict: dict | None = None,
) -> tuple[float, dict]:
    r"""
    Calculate liquid volume averaged mass fraction of a species at a given time

    .. math::
       \frac{1}{V_{\rm liq, tot}} \int_{V_{\rm liq}} Y dV_{\rm liq}

    where:
      - :math:`V_{\rm liq, tot}` is the toal volume of liquid
      - :math:`Y` is the species mass fraction
      - :math:`V_{\rm liq}` is the volume of liquid where :math:`Y` is measured


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
    species_name : str
        Name of the species
    field_dict : dict | None
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    liq_ave_y: float
        Liquid volume averaged mass fraction
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
    y_liq, field_dict = read_field(
        field_name=f"{species_name}.liquid", field_dict=field_dict, **kwargs
    )
    ind_liq, field_dict = _get_ind_liq(field_dict=field_dict, **kwargs)

    cell_volume, field_dict = read_cell_volumes(
        field_dict=field_dict, **kwargs_vol
    )

    # Only compute over the liquid
    alpha_liq = _field_filter(alpha_liq, ind=ind_liq, field_type="scalar")
    cell_volume = _field_filter(cell_volume, ind=ind_liq, field_type="scalar")
    y_liq = _field_filter(y_liq, ind=ind_liq, field_type="scalar")

    # Calculate
    liq_ave_y = _weighted_average(y_liq, alpha_liq * cell_volume)

    return liq_ave_y, field_dict


def compute_ave_conc_liq(
    case_folder: str,
    time_folder: str,
    species_name: str = "CO2",
    n_cells: int | None = None,
    volume_time: str | None = None,
    field_dict: dict | None = None,
) -> tuple[float, dict]:
    r"""
    Calculate liquid volume averaged concentration of a species at a given time

    .. math::
       \frac{1}{V_{\rm liq, tot}} \int_{V_{\rm liq}} \rho_{\rm liq} Y / W dV_{\rm liq}

    where:
      - :math:`V_{\rm liq, tot}` is the toal volume of liquid
      - :math:`\rho_{\rm liq}` is the liquid density
      - :math:`Y` is the species mass fraction
      - :math:`W` is the species molar mass
      - :math:`V_{\rm liq}` is the volume of liquid where :math:`Y` is measured

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of time folder to analyze
    species_name : str
        Name of the species
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
    conc_ave: float
        Liquid volume averaged species concentration
    field_dict : dict
        Dictionary of fields read
    """
    if field_dict is None:
        field_dict = {}

    mol_weight = species_name_to_mw(
        case_folder=case_folder, species_name=species_name
    )
    logger.debug(
        f"Computing concentration for {species_name} with molecular weight {mol_weight:.4g} kg/mol"
    )

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
    y_liq, field_dict = read_field(
        field_name=f"{species_name}.liquid", field_dict=field_dict, **kwargs
    )
    ind_liq, field_dict = _get_ind_liq(field_dict=field_dict, **kwargs)

    cell_volume, field_dict = read_cell_volumes(
        field_dict=field_dict, **kwargs_vol
    )
    rho_liq, field_dict = _read_liquid_density_field(
        case_folder, time_folder, n_cells, field_dict
    )

    # Only compute over the liquid
    alpha_liq = _field_filter(alpha_liq, ind=ind_liq, field_type="scalar")
    cell_volume = _field_filter(cell_volume, ind=ind_liq, field_type="scalar")
    y_liq = _field_filter(y_liq, ind=ind_liq, field_type="scalar")
    rho_liq = _field_filter(rho_liq, ind=ind_liq, field_type="scalar")

    conc_loc = rho_liq * y_liq / mol_weight

    conc_ave = _weighted_average(conc_loc, alpha_liq * cell_volume)

    return conc_ave, field_dict
