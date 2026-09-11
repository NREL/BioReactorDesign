import numpy as np

from bird.utilities.ofio import read_cell_volumes, read_field

from ._cell_filter import _field_filter, _get_ind_liq, _weighted_average
from .phase import _read_liquid_density_field

# Turbulent Prandtl (= turbulent Schmidt) for the nut/Prt fallback
TURBULENT_PRANDTL = 0.85


def compute_turbulent_diffusivity(
    case_folder: str,
    time_folder: str,
    n_cells: int | None = None,
    field_dict: dict | None = None,
) -> tuple[float, dict]:
    r"""Liquid-averaged turbulent mass diffusivity.

    .. math::
       D_{\rm turb} = \frac{\alpha_t^{\rm liq}}{\rho^{\rm liq}}

    averaged over the liquid [m^2/s]. ``alphat.liquid`` already carries the
    solver's turbulent Prandtl number; if it was not written, the field falls
    back to :math:`\nu_t^{\rm liq} / Pr_t` with :math:`Pr_t = 0.85`. The density
    is read from ``thermo:rho.liquid`` (then ``rho.liquid``), falling back to
    ``rho0MixLiq`` from ``globalVars`` and then 1000.

    Parameters
    ----------
    case_folder: str
        Path to case folder
    time_folder: str
        Name of the time folder to analyze
    n_cells : int | None
        Number of cells in the domain.
        If None, it will deduced from the field reading
    field_dict : dict
        Dictionary of fields used to avoid rereading the same fields to calculate different quantities

    Returns
    ----------
    turbulent_diffusivity: float
        Volume averaged turbulent mass diffusivity, in :math:`m^2.s^{-1}`
    field_dict : dict
        Dictionary of fields read
    """
    if field_dict is None:
        field_dict = {}

    ind_liq, field_dict = _get_ind_liq(
        case_folder, time_folder, n_cells=n_cells, field_dict=field_dict
    )
    cell_volume, field_dict = read_cell_volumes(
        case_folder, time_folder, n_cells, field_dict
    )

    rho_liquid, field_dict = _read_liquid_density_field(
        case_folder, time_folder, n_cells, field_dict
    )

    # Turbulent diffusivity field: alphat.liquid / rho, else nut.liquid / Prt
    try:
        alphat_liquid, field_dict = read_field(
            case_folder, time_folder, "alphat.liquid", n_cells, field_dict
        )
        diffusivity = np.asarray(alphat_liquid) / np.asarray(rho_liquid)
    except FileNotFoundError:
        nut_liquid, field_dict = read_field(
            case_folder, time_folder, "nut.liquid", n_cells, field_dict
        )
        diffusivity = np.asarray(nut_liquid) / TURBULENT_PRANDTL
    if diffusivity.ndim == 0:
        diffusivity = float(diffusivity)

    diffusivity = _field_filter(diffusivity, ind=ind_liq, field_type="scalar")
    cell_volume = _field_filter(cell_volume, ind=ind_liq, field_type="scalar")
    return _weighted_average(diffusivity, cell_volume), field_dict
